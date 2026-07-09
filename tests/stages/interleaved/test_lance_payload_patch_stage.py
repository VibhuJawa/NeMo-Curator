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

from concurrent.futures import ThreadPoolExecutor
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


def _stage(tmp_path: Path) -> tuple[LanceCoordinatePayloadPatchStage, _FakeImageDataset]:
    stage = LanceCoordinatePayloadPatchStage(
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
    stage.output_root.mkdir()
    stage.node_local_spool_root.mkdir()
    image = _FakeImageDataset()
    stage._image_dataset = image
    stage._image_fragment_manifest_sha256 = _IMAGE_FRAGMENT_DIGEST
    stage._document_dataset = _FakeDocumentDataset()
    stage._document_identity = (_DOCUMENT_URI, 1)
    stage._executor = ThreadPoolExecutor(max_workers=2)
    return stage, image


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
    assert stage._executor is None


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
