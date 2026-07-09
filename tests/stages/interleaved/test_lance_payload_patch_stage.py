# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved import lance_payload_patch_stage
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    CoordinatePlanIdentity,
    lance_coordinate_plan_schema,
    publish_coordinate_plan,
)
from nemo_curator.stages.interleaved.lance_payload_patch_stage import (
    LanceCoordinatePayloadPatchStage,
    _SampleStitcher,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_DOCUMENT_URI = "s3://document-bucket/documents.lance"
_IMAGE_URI = "s3://image-bucket/images.lance"
_FRAGMENT_ID = 7
_IMAGE_FRAGMENT_DIGEST = "f" * 64


class _FakeFragment:
    fragment_id = _FRAGMENT_ID
    physical_rows = 7
    num_deletions = 0
    metadata = SimpleNamespace(physical_rows=7, deletion_file=None)

    @staticmethod
    def deletion_file() -> None:
        return None


class _FakeScanner:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def to_batches(self) -> Iterator[pa.RecordBatch]:
        for start, length in ((0, 1), (1, 3), (4, 3)):
            yield from self._table.slice(start, length).to_batches()


class _FakeDocumentDataset:
    version = 1

    def __init__(self) -> None:
        self.schema = pa.schema(
            [
                pa.field("sample_id", pa.string(), nullable=False),
                pa.field("position", pa.int32(), nullable=False),
                pa.field("modality", pa.string(), nullable=False),
            ]
        )
        self._table = pa.Table.from_arrays(
            [
                pa.array(["a", "a", "b", "c", "c", "c", "d"], type=pa.string()),
                pa.array([0, 1, 0, 0, 1, 2, 0], type=pa.int32()),
                pa.array(["image", "text", "image", "image", "text", "image", "image"]),
            ],
            schema=self.schema,
        )
        self._fragment = _FakeFragment()

    def get_fragment(self, fragment_id: int) -> _FakeFragment | None:
        return self._fragment if fragment_id == _FRAGMENT_ID else None

    def scanner(self, **kwargs: object) -> _FakeScanner:
        assert kwargs["fragments"] == [self._fragment]
        projection = self._table.select(kwargs["columns"])
        row_addresses = pa.array(
            [(_FRAGMENT_ID << 32) | position for position in range(self._table.num_rows)],
            type=pa.uint64(),
        )
        return _FakeScanner(projection.append_column("_rowaddr", row_addresses))


class _FakeImageDataset:
    version = 4
    has_stable_row_ids = True

    def __init__(self) -> None:
        self.schema = pa.schema([pa.field("image", pa.large_binary())])
        self._payloads = pa.table({"image": pa.array([b"one", b"two", b"three"], type=pa.large_binary())})
        self.requests: list[tuple[int, ...]] = []
        self._stats_calls = 0

    def _take_rows(self, row_ids: list[int], *, columns: list[str]) -> pa.Table:
        self.requests.append(tuple(row_ids))
        indices = pa.array([row_id - 1 for row_id in row_ids], type=pa.int64())
        return self._payloads.take(indices).select(columns)

    def io_stats_incremental(self) -> SimpleNamespace:
        self._stats_calls += 1
        if self._stats_calls == 1:
            return SimpleNamespace(read_iops=0, read_bytes=0)
        return SimpleNamespace(read_iops=4, read_bytes=4096)


class _FakePayloadStreamer:
    def __init__(self, image: _FakeImageDataset) -> None:
        self.image = image
        self.last_metrics: dict[str, int | float | bool] = {}
        self.closed = False

    def iter_stable_row_ids(self, values: pa.Array):  # noqa: ANN202
        row_ids = [int(value) for value in values.to_pylist()]
        payload_bytes = 0
        batches = 0
        for start in range(0, len(row_ids), 2):
            batch_ids = row_ids[start : start + 2]
            payload = self.image._take_rows(batch_ids, columns=["image"])
            payload_bytes += payload["image"].nbytes
            batches += 1
            yield pa.Table.from_arrays(
                [pa.array(batch_ids, type=pa.uint64()), payload["image"]],
                schema=pa.schema(
                    [
                        pa.field("stable_row_id", pa.uint64(), nullable=False),
                        self.image.schema.field("image"),
                    ]
                ),
            )
        self.last_metrics = {
            "stream_complete": True,
            "input_stable_rows": len(row_ids),
            "stream_output_rows": len(row_ids),
            "payload_take_rows": len(row_ids),
            "payload_batches_emitted": batches,
            "payload_read_calls": batches,
            "payload_bytes": payload_bytes,
            "max_pending_payload_reads": min(2, batches),
            "max_retained_payload_batches": min(2, batches),
            "sparse_calls_avoided": max(0, len(row_ids) - batches),
            "payload_read_call_sum_seconds": 0.02,
            "payload_read_envelope_seconds": 0.01,
            "lance_read_iops": 4,
            "lance_read_bytes": 4096,
        }

    def close(self) -> None:
        self.closed = True


def _coordinate_task(root: Path):  # noqa: ANN202
    positions = [0, 2, 3, 5, 6]
    table = pa.Table.from_arrays(
        [
            pa.array([(_FRAGMENT_ID << 32) | position for position in positions], type=pa.uint64()),
            pa.array(positions, type=pa.uint64()),
            pa.array([2, None, 1, 2, 3], type=pa.uint64()),
        ],
        schema=lance_coordinate_plan_schema(allow_missing=True),
    )
    task = publish_coordinate_plan(
        root,
        table,
        CoordinatePlanIdentity(
            document_uri=_DOCUMENT_URI,
            document_version=1,
            image_uri=_IMAGE_URI,
            image_version=4,
            fragment_id=_FRAGMENT_ID,
            sidecar_manifest_sha256="a" * 64,
            fragment_manifest_sha256=_IMAGE_FRAGMENT_DIGEST,
        ),
        allow_missing=True,
    )
    task._stage_perf = {"coordinate": 1.0}
    return task


def _attach_fake_datasets(
    stage: LanceCoordinatePayloadPatchStage,
) -> tuple[LanceCoordinatePayloadPatchStage, _FakeImageDataset]:
    stage.output_root.mkdir()
    stage.node_local_spool_root.mkdir()
    image = _FakeImageDataset()
    stage._image_dataset = image
    stage._image_fragment_manifest_sha256 = _IMAGE_FRAGMENT_DIGEST
    stage._document_dataset = _FakeDocumentDataset()
    stage._document_identity = (_DOCUMENT_URI, 1)
    stage._payload_streamer = _FakePayloadStreamer(image)
    return stage, image


def _stage(tmp_path: Path) -> tuple[LanceCoordinatePayloadPatchStage, _FakeImageDataset]:
    return _attach_fake_datasets(
        LanceCoordinatePayloadPatchStage(
            image_uri=_IMAGE_URI,
            image_version=4,
            output_root=str(tmp_path / "patches"),
            node_local_spool_root=str(tmp_path / "spool"),
            payload_window_bytes="256MiB",
            bucket_rows=2,
            estimated_payload_bytes_per_row=1,
            fetch_batch_size=2,
            max_pending=2,
        )
    )


def _read_output(paths: list[str]) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in paths])


def test_setup_rejects_image_dataset_without_stable_row_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = LanceCoordinatePayloadPatchStage(
        image_uri=_IMAGE_URI,
        image_version=4,
        output_root=str(tmp_path / "patches"),
        node_local_spool_root=str(tmp_path / "spool"),
    )
    monkeypatch.setattr(
        lance_payload_patch_stage,
        "_open_lance_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(has_stable_row_ids=False),
    )

    with pytest.raises(ValueError, match="image dataset must have stable row IDs"):
        stage.setup()
    assert stage._image_dataset is None
    assert stage._payload_streamer is None


def test_lance_ray_reader_factory_is_lazy_and_uses_identity_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Config:
        def __init__(self, **kwargs: object) -> None:
            captured["config"] = kwargs

    class _Reader:
        def __init__(self, config: object, **kwargs: object) -> None:
            captured["reader"] = (config, kwargs)

    monkeypatch.setitem(
        sys.modules,
        "lance_ray",
        SimpleNamespace(
            LanceStableIdPayloadConfig=_Config,
            LanceStableIdPayloadStreamer=_Reader,
        ),
    )
    dataset = _FakeImageDataset()

    reader = lance_payload_patch_stage._create_stable_id_payload_streamer(
        dataset,
        dataset_uri=_IMAGE_URI,
        dataset_version=4,
        expected_rows=3,
        source_columns=("image", "width"),
        storage_options={"region": "us-west-2"},
        fetch_batch_size=1024,
        max_pending=16,
    )

    assert isinstance(reader, _Reader)
    config = captured["config"]
    assert isinstance(config, dict)
    assert config == {
        "dataset_uri": _IMAGE_URI,
        "dataset_version": 4,
        "expected_rows": 3,
        "columns": {"image": "image", "width": "width"},
        "dataset_storage_options": {"region": "us-west-2"},
        "fetch_batch_size": 1024,
        "io_threads": 16,
        "max_pending_fetch_batches": 16,
    }
    _, reader_kwargs = captured["reader"]
    assert reader_kwargs == {"dataset": dataset, "stable_row_id_output_column": "stable_row_id"}
    assert "LanceStableIdPayloadStreamer" not in lance_payload_patch_stage.__dict__


def test_setup_reuses_one_persistent_reader_and_teardown_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = LanceCoordinatePayloadPatchStage(
        image_uri=_IMAGE_URI,
        image_version=4,
        output_root=str(tmp_path / "patches"),
        node_local_spool_root=str(tmp_path / "spool"),
        image_columns={"image": "binary_content"},
        fetch_batch_size=7,
        max_pending=3,
    )
    image = _FakeImageDataset()
    reader = _FakePayloadStreamer(image)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(lance_payload_patch_stage, "_open_lance_dataset", lambda *_args, **_kwargs: image)
    monkeypatch.setattr(
        lance_payload_patch_stage,
        "_validate_stable_global_ordinal_manifest",
        lambda _dataset: SimpleNamespace(total_rows=3),
    )
    monkeypatch.setattr(
        lance_payload_patch_stage,
        "_stable_global_ordinal_manifest_sha256",
        lambda *_args, **_kwargs: _IMAGE_FRAGMENT_DIGEST,
    )

    def create(_dataset: object, **kwargs: object) -> _FakePayloadStreamer:
        calls.append(kwargs)
        return reader

    monkeypatch.setattr(lance_payload_patch_stage, "_create_stable_id_payload_streamer", create)

    stage.setup()
    stage.setup()

    assert stage._payload_streamer is reader
    assert calls == [
        {
            "dataset_uri": _IMAGE_URI,
            "dataset_version": 4,
            "expected_rows": 3,
            "source_columns": ("image",),
            "storage_options": {},
            "fetch_batch_size": 7,
            "max_pending": 3,
        }
    ]
    assert reader.closed is False

    stage.teardown()

    assert reader.closed is True
    assert stage._payload_streamer is None


def test_stage_rejects_unknown_payload_spool_sync_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported payload_spool_sync_mode"):
        LanceCoordinatePayloadPatchStage(
            image_uri=_IMAGE_URI,
            image_version=4,
            output_root=str(tmp_path / "patches"),
            node_local_spool_root=str(tmp_path / "spool"),
            payload_spool_sync_mode="unknown",  # type: ignore[arg-type]
        )


def test_stage_exposes_payload_actor_admission_and_estimated_reservation(tmp_path: Path) -> None:
    stage = LanceCoordinatePayloadPatchStage(
        image_uri=_IMAGE_URI,
        image_version=4,
        output_root=str(tmp_path / "patches"),
        node_local_spool_root=str(tmp_path / "spool"),
        payload_window_bytes="256MiB",
        estimated_payload_bytes_per_row=10,
        fetch_batch_size=2,
        max_pending=3,
        payload_actor_cpus=4,
        payload_patch_workers=2,
    )

    assert stage.resources.cpus == 4.0
    assert stage.num_workers() == 2
    assert stage.estimated_payload_actor_reservation_bytes == 256 * 1024**2 + 60


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_actor_cpus", 0),
        ("payload_actor_cpus", True),
        ("payload_patch_workers", 0),
        ("payload_patch_workers", True),
    ],
)
def test_stage_rejects_invalid_payload_actor_geometry(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        LanceCoordinatePayloadPatchStage(
            image_uri=_IMAGE_URI,
            image_version=4,
            output_root=str(tmp_path / "patches"),
            node_local_spool_root=str(tmp_path / "spool"),
            **{field: value},
        )


def test_stage_materializes_full_fragment_and_adopts_exact_retry(tmp_path: Path) -> None:
    task = _coordinate_task(tmp_path / "coordinates")
    stage, image = _stage(tmp_path)
    try:
        output = stage.process(task)
        requests_after_first = list(image.requests)
        retry = stage.process(task)
    finally:
        stage.teardown()

    result = _read_output(output.data)
    assert result[DOCUMENT_POSITION].to_pylist() == list(range(7))
    assert result["sample_id"].to_pylist() == ["a", "a", "b", "c", "c", "c", "d"]
    assert result["binary_content"].to_pylist() == [b"two", None, None, b"one", None, b"two", b"three"]
    assert sorted(requests_after_first) == [(1, 2), (3,)]
    assert image.requests == requests_after_first
    assert output._stage_perf == {"coordinate": 1.0}
    assert output._metadata["lance_coordinate_payload_patch"]["adopted"] is False
    assert output._metadata["lance_coordinate_payload_patch"]["sparse_calls_avoided"] == 1
    assert output._metadata["lance_coordinate_payload_patch"]["bucket_rows"] == 2
    assert output._metadata["lance_coordinate_payload_patch"]["payload_spool_sync_mode"] == "attempt_local"
    assert output._metadata["lance_coordinate_payload_patch"]["payload_spool_distinct_buckets"] == 4
    assert output._metadata["lance_coordinate_payload_patch"]["payload_spool_files"] == 4
    assert output._metadata["lance_coordinate_payload_patch"]["stream_complete"] is True
    assert output._metadata["lance_coordinate_payload_patch"]["input_stable_rows"] == 3
    assert output._metadata["lance_coordinate_payload_patch"]["stream_output_rows"] == 3
    assert output._metadata["lance_coordinate_payload_patch"]["payload_read_calls"] == 2
    assert output._metadata["lance_coordinate_payload_patch"]["take_calls"] == 2
    assert output._metadata["lance_coordinate_payload_patch"]["lance_read_iops"] == 4
    assert output._metadata["lance_coordinate_payload_patch"]["lance_read_bytes"] == 4096
    assert output._metadata["lance_coordinate_payload_patch"]["estimated_inflight_payload_bytes"] == 4
    assert output._metadata["lance_coordinate_payload_patch"]["estimated_payload_actor_reservation_bytes"] == (
        256 * 1024**2 + 4
    )
    assert output._metadata["lance_coordinate_payload_patch"]["payload_actor_cpus"] == 8
    assert (
        output._metadata["lance_coordinate_payload_patch"]["physical_read_operations_per_private_take_envelope_second"]
        > 0
    )
    assert output._metadata["lance_patch_artifact"]["adopted"] is False
    assert retry.data == output.data
    assert retry._metadata["lance_coordinate_payload_patch"] == {"adopted": True}
    assert retry._metadata["lance_patch_artifact"]["adopted"] is True
    assert not list((tmp_path / "spool").iterdir())
    assert not list((tmp_path / "patches").glob(".*.tmp"))


def test_stage_default_bucket_geometry_is_reported_without_large_input(tmp_path: Path) -> None:
    task = _coordinate_task(tmp_path / "coordinates")
    stage, _image = _attach_fake_datasets(
        LanceCoordinatePayloadPatchStage(
            image_uri=_IMAGE_URI,
            image_version=4,
            output_root=str(tmp_path / "patches"),
            node_local_spool_root=str(tmp_path / "spool"),
            payload_window_bytes="256MiB",
            estimated_payload_bytes_per_row=1,
            fetch_batch_size=2,
            max_pending=2,
        )
    )
    try:
        output = stage.process(task)
    finally:
        stage.teardown()

    metrics = output._metadata["lance_coordinate_payload_patch"]
    assert stage.bucket_rows == 131_072
    assert stage.payload_spool_sync_mode == "attempt_local"
    assert metrics["bucket_rows"] == 131_072
    assert metrics["payload_spool_sync_mode"] == "attempt_local"
    assert metrics["payload_spool_distinct_buckets"] == 1
    assert metrics["payload_spool_files"] == 1


def test_stage_failure_removes_attempt_and_retry_recomputes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _coordinate_task(tmp_path / "coordinates")
    stage, image = _stage(tmp_path)
    original = stage._append_splits
    raised = False

    def fail_after_append(*args: object, **kwargs: object) -> None:
        nonlocal raised
        original(*args, **kwargs)
        if not raised:
            raised = True
            msg = "synthetic patch failure"
            raise RuntimeError(msg)

    monkeypatch.setattr(stage, "_append_splits", fail_after_append)
    try:
        with pytest.raises(RuntimeError, match="synthetic patch failure"):
            stage.process(task)
        assert not [path for path in (tmp_path / "patches").iterdir() if path.is_dir()]
        assert not list((tmp_path / "spool").iterdir())

        lock_path = next((tmp_path / "patches").glob(".*.lock"))
        artifact_name = lock_path.name.removeprefix(".").removesuffix(".lock")
        orphan = tmp_path / "patches" / f".{artifact_name}.abandoned.tmp"
        orphan.mkdir()
        (orphan / "partial").write_bytes(b"partial")
        orphan_spool = tmp_path / "spool" / f".{artifact_name}.abandoned.payload-spool"
        orphan_spool.mkdir()
        (orphan_spool / "partial").write_bytes(b"partial")

        monkeypatch.setattr(stage, "_append_splits", original)
        output = stage.process(task)
    finally:
        stage.teardown()

    assert _read_output(output.data).num_rows == 7
    assert len(image.requests) == 4
    assert not orphan.exists()
    assert not orphan_spool.exists()


def test_stage_reader_failure_closes_partial_stream_and_retry_recomputes(tmp_path: Path) -> None:
    class _FailOncePayloadStreamer(_FakePayloadStreamer):
        def __init__(self, image: _FakeImageDataset) -> None:
            super().__init__(image)
            self.fail_once = True
            self.partial_iterator_closed = False

        def iter_stable_row_ids(self, values: pa.Array):  # noqa: ANN202
            if not self.fail_once:
                yield from super().iter_stable_row_ids(values)
                return
            self.fail_once = False
            iterator = super().iter_stable_row_ids(values)
            try:
                yield next(iterator)
                msg = "injected stable-ID stream failure"
                raise RuntimeError(msg)
            finally:
                iterator.close()
                self.partial_iterator_closed = True

    task = _coordinate_task(tmp_path / "coordinates")
    stage, image = _stage(tmp_path)
    reader = _FailOncePayloadStreamer(image)
    stage._payload_streamer = reader
    try:
        with pytest.raises(RuntimeError, match="injected stable-ID stream failure"):
            stage.process(task)
        assert reader.partial_iterator_closed is True
        assert reader.last_metrics == {}
        assert not [path for path in (tmp_path / "patches").iterdir() if path.is_dir()]
        assert not list((tmp_path / "spool").iterdir())

        output = stage.process(task)
    finally:
        stage.teardown()

    assert _read_output(output.data).num_rows == 7
    assert output._metadata["lance_coordinate_payload_patch"]["stream_complete"] is True
    assert reader.closed is True


def test_stage_rejects_coordinate_image_identity_before_fetch(tmp_path: Path) -> None:
    task = _coordinate_task(tmp_path / "coordinates")
    stage, image = _stage(tmp_path)
    stage._image_fragment_manifest_sha256 = "d" * 64
    try:
        with pytest.raises(ValueError, match="fragment-manifest digest"):
            stage.process(task)
    finally:
        stage.teardown()

    assert image.requests == []


def test_stage_surfaces_directory_fsync_failure_then_adopts_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _coordinate_task(tmp_path / "coordinates")
    stage, image = _stage(tmp_path)

    def fail_fsync(_path: Path) -> None:
        msg = "synthetic directory fsync failure"
        raise OSError(msg)

    monkeypatch.setattr(lance_payload_patch_stage, "fsync_directory", fail_fsync)
    try:
        with pytest.raises(OSError, match="synthetic directory fsync failure"):
            stage.process(task)
        requests_after_failure = list(image.requests)
        monkeypatch.undo()
        retry = stage.process(task)
    finally:
        stage.teardown()

    assert image.requests == requests_after_failure
    assert retry._metadata["lance_coordinate_payload_patch"] == {"adopted": True}
    assert _read_output(retry.data).num_rows == 7


def test_sample_stitcher_rejects_noncontiguous_sample_ids() -> None:
    stitcher = _SampleStitcher()
    table = pa.table(
        {
            "sample_id": pa.array(["a", "b", "a"], type=pa.string()),
            "value": pa.array([1, 2, 3], type=pa.int32()),
        }
    )

    with pytest.raises(ValueError, match="exactly one contiguous"):
        stitcher.push(table)
