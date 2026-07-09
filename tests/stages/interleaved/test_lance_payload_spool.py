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

import hashlib
import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved import lance_payload_spool as payload_spool_module
from nemo_curator.stages.interleaved.lance_payload_spool import (
    PayloadSpool,
    PayloadSpoolCoordinator,
    PayloadSpoolReader,
)

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.uint64(), nullable=False),
        pa.field("document_position", pa.uint64(), nullable=False),
        pa.field("image", pa.binary()),
        pa.field("attributes", pa.list_(pa.string())),
    ],
    metadata={b"purpose": b"payload-spool-test"},
)


def _table(
    stable_ids: list[int],
    positions: list[int],
    payloads: list[bytes],
    attributes: list[list[str]] | None = None,
) -> pa.Table:
    if attributes is None:
        attributes = [[f"tag-{stable_id}"] for stable_id in stable_ids]
    return pa.Table.from_arrays(
        [
            pa.array(stable_ids, type=pa.uint64()),
            pa.array(positions, type=pa.uint64()),
            pa.array(payloads, type=pa.binary()),
            pa.array(attributes, type=pa.list_(pa.string())),
        ],
        schema=_SCHEMA,
    )


def _append_deterministic_input(spool: PayloadSpool) -> None:
    spool.append(
        _table(
            [125, 102, 112],
            [25, 2, 12],
            [b"a" * 8, b"b" * 11, b"c" * 7],
        )
    )
    spool.append(
        _table(
            [101, 121, 111],
            [1, 21, 11],
            [b"d" * 9, b"e" * 6, b"f" * 10],
        )
    )


def test_payload_spool_is_bounded_bucketed_and_deterministic(tmp_path: Path) -> None:
    manifests = []
    outputs = []
    for name in ("first", "second"):
        spool = PayloadSpool(tmp_path / name, _SCHEMA, target_bytes=100, bucket_rows=10)
        _append_deterministic_input(spool)
        manifest = spool.finish()
        output = spool.read_all()
        manifests.append(manifest)
        outputs.append(output)

        assert manifest.total_rows == 6
        assert manifest.sync_mode == "fsync"
        assert manifest.peak_bounded_active_bytes <= manifest.target_bytes
        assert all(file.arrow_nbytes <= manifest.target_bytes for file in manifest.files)
        assert [(file.bucket, file.part) for file in manifest.files] == sorted(
            (file.bucket, file.part) for file in manifest.files
        )
        assert not list(manifest.root.glob(".*.tmp"))

    assert [file.path.name for file in manifests[0].files] == [file.path.name for file in manifests[1].files]
    assert [file.sha256 for file in manifests[0].files] == [file.sha256 for file in manifests[1].files]
    assert manifests[0].sha256 == manifests[1].sha256
    assert outputs[0].equals(outputs[1])
    assert sorted(outputs[0]["stable_id"].to_pylist()) == [101, 102, 111, 112, 121, 125]
    assert [position // 10 for position in outputs[0]["document_position"].to_pylist()] == sorted(
        position // 10 for position in outputs[0]["document_position"].to_pylist()
    )

    for manifest in manifests:
        PayloadSpoolReader(manifest).cleanup()
        assert not manifest.root.exists()


def test_payload_spool_groups_unsorted_bucket_runs_across_appends(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=4096, bucket_rows=10)
    spool.append(
        pa.concat_tables(
            [
                _table([20, 1], [20, 1], [b"twenty", b"one"]),
                _table([21, 10], [21, 10], [b"twenty-one", b"ten"]),
            ]
        )
    )
    spool.append(
        _table(
            [2, 22, 11],
            [2, 22, 11],
            [b"two", b"twenty-two", b"eleven"],
        )
    )

    manifest = spool.finish()
    output = spool.read_all()

    assert output["stable_id"].to_pylist() == [1, 2, 10, 11, 20, 21, 22]
    assert [(file.bucket, file.part) for file in manifest.files] == [(0, 0), (1, 0), (2, 0)]
    assert manifest.peak_bounded_active_bytes == sum(file.arrow_nbytes for file in manifest.files)


def test_payload_spool_flushes_exact_arrow_byte_boundary(tmp_path: Path) -> None:
    table = _table([1, 2, 3], [1, 2, 3], [b"a" * 7, b"b" * 11, b"c" * 13])
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=table.nbytes, bucket_rows=10)

    spool.append(table)
    manifest = spool.finish()

    assert manifest.peak_active_bytes == table.nbytes
    assert manifest.peak_bounded_active_bytes == table.nbytes
    assert len(manifest.files) == 1
    assert manifest.files[0].arrow_nbytes == table.nbytes
    assert not manifest.files[0].oversized


def test_payload_spool_public_flush_preserves_rows_and_lifecycle(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=200, bucket_rows=10)
    first = _table([1, 2], [1, 2], [b"first", b"second"])
    second = _table([3], [3], [b"third"])

    assert spool.active_bytes == 0
    spool.append(first)
    assert spool.active_bytes == first.nbytes
    assert not list(spool.root.glob("*.arrow"))

    spool.flush()
    assert spool.active_bytes == 0
    assert len(list(spool.root.glob("*.arrow"))) == 1
    spool.flush()
    assert len(list(spool.root.glob("*.arrow"))) == 1

    spool.append(second)
    assert spool.active_bytes == second.nbytes
    manifest = spool.finish()
    assert spool.active_bytes == 0
    assert manifest.total_rows == 3
    assert spool.read_all()["stable_id"].to_pylist() == [1, 2, 3]
    with pytest.raises(RuntimeError, match="already been finished"):
        spool.flush()

    spool.cleanup()
    assert spool.active_bytes == 0


def test_payload_spool_coordinator_enforces_aggregate_threshold_and_conservation(tmp_path: Path) -> None:
    coordinator = PayloadSpoolCoordinator(max_active_bytes=100)
    spools = [
        PayloadSpool(tmp_path / f"child-{index}", _SCHEMA, target_bytes=55, bucket_rows=10) for index in range(3)
    ]
    rows = [
        _table([1], [1], [b"a" * 10]),
        _table([2], [2], [b"b" * 10]),
        _table([3], [3], [b"c" * 10]),
    ]

    coordinator.append(spools[0], rows[0])
    assert coordinator.active_bytes == rows[0].nbytes
    coordinator.append(spools[1], rows[1])
    assert coordinator.active_bytes == rows[0].nbytes + rows[1].nbytes <= coordinator.max_active_bytes
    assert not list(spools[0].root.glob("*.arrow"))

    coordinator.append(spools[2], rows[2])
    assert coordinator.active_bytes == rows[1].nbytes + rows[2].nbytes <= coordinator.max_active_bytes
    assert spools[0].active_bytes == 0
    assert len(list(spools[0].root.glob("*.arrow"))) == 1

    coordinator.flush()
    assert coordinator.active_bytes == 0
    manifests = [spool.finish() for spool in spools]
    assert sum(manifest.total_rows for manifest in manifests) == 3
    assert [spool.read_all()["stable_id"].to_pylist() for spool in spools] == [[1], [2], [3]]


def test_payload_spool_coordinator_isolates_oversized_rows_outside_active_budget(tmp_path: Path) -> None:
    coordinator = PayloadSpoolCoordinator(max_active_bytes=100)
    buffered = PayloadSpool(tmp_path / "buffered", _SCHEMA, target_bytes=60, bucket_rows=10)
    oversized = PayloadSpool(tmp_path / "oversized", _SCHEMA, target_bytes=60, bucket_rows=10)
    small = _table([1], [1], [b"small"])
    large = _table([2], [2], [b"x" * 512])

    coordinator.append(buffered, small)
    assert coordinator.active_bytes == small.nbytes
    coordinator.append(oversized, large)

    assert coordinator.active_bytes == small.nbytes <= coordinator.max_active_bytes
    assert oversized.active_bytes == 0
    buffered_manifest = buffered.finish()
    oversized_manifest = oversized.finish()
    assert buffered_manifest.total_rows + oversized_manifest.total_rows == 2
    assert len(oversized_manifest.oversized_rows) == 1
    assert oversized_manifest.oversized_rows[0].stable_id == 2
    assert oversized_manifest.oversized_rows[0].arrow_nbytes == large.nbytes > coordinator.max_active_bytes
    assert oversized.read_all()["image"].to_pylist() == [b"x" * 512]


def test_payload_spool_coordinator_reserves_child_bound_for_multirow_slicing(tmp_path: Path) -> None:
    coordinator = PayloadSpoolCoordinator(max_active_bytes=150)
    first = PayloadSpool(tmp_path / "first", _SCHEMA, target_bytes=100, bucket_rows=10)
    second = PayloadSpool(tmp_path / "second", _SCHEMA, target_bytes=100, bucket_rows=10)
    first_row = _table([1], [1], [b"a" * 60], [[]])
    sliced_rows = _table([2, 3, 4], [2, 3, 4], [b"b" * 10, b"c" * 10, b"d" * 10], [[], [], []])

    coordinator.append(first, first_row)
    assert first.active_bytes == first_row.nbytes
    coordinator.append(second, sliced_rows)

    assert first.active_bytes == 0
    assert len(list(first.root.glob("*.arrow"))) == 1
    assert coordinator.active_bytes <= coordinator.max_active_bytes
    assert coordinator.peak_active_bytes <= coordinator.max_active_bytes
    coordinator.flush()
    manifests = (first.finish(), second.finish())
    assert sum(manifest.total_rows for manifest in manifests) == 4
    assert first.read_all()["stable_id"].to_pylist() == [1]
    assert second.read_all()["stable_id"].to_pylist() == [2, 3, 4]


def test_payload_spool_coordinator_retains_small_siblings_with_full_child_targets(tmp_path: Path) -> None:
    coordinator = PayloadSpoolCoordinator(max_active_bytes=1024)
    first = PayloadSpool(tmp_path / "first", _SCHEMA, target_bytes=1024, bucket_rows=10)
    second = PayloadSpool(tmp_path / "second", _SCHEMA, target_bytes=1024, bucket_rows=10)
    first_row = _table([1], [1], [b"a" * 32])
    second_row = _table([2], [2], [b"b" * 32])

    coordinator.append(first, first_row)
    coordinator.append(second, second_row)

    assert first.active_bytes == first_row.nbytes
    assert second.active_bytes == second_row.nbytes
    assert not list(first.root.glob("*.arrow"))
    assert not list(second.root.glob("*.arrow"))
    assert coordinator.active_bytes == first_row.nbytes + second_row.nbytes
    coordinator.flush()
    assert first.finish().total_rows == 1
    assert second.finish().total_rows == 1


def test_payload_spool_slice_calls_scale_with_bucket_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 100_000
    table = _table(
        list(range(row_count)),
        list(range(row_count)),
        [b"x"] * row_count,
        [[]] * row_count,
    )
    slice_calls: list[tuple[int, int | None]] = []
    original_slice_table = payload_spool_module._slice_table

    def tracked_slice(table: pa.Table, offset: int, length: int | None = None) -> pa.Table:
        slice_calls.append((offset, length))
        return original_slice_table(table, offset, length)

    monkeypatch.setattr(payload_spool_module, "_slice_table", tracked_slice)
    spool = PayloadSpool(
        tmp_path / "spool",
        _SCHEMA,
        target_bytes=table.nbytes,
        bucket_rows=row_count + 1,
    )

    spool.append(table)
    manifest = spool.finish()

    assert slice_calls == [(0, row_count)]
    assert manifest.total_rows == row_count
    assert manifest.peak_bounded_active_bytes == table.nbytes


def test_payload_spool_isolates_and_reports_one_oversized_row(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=90, bucket_rows=100)
    table = _table(
        [1, 2, 3],
        [10, 11, 12],
        [b"small", b"x" * 512, b"also-small"],
        [["a"], ["large"], ["c"]],
    )
    spool.append(table)

    manifest = spool.finish()

    assert manifest.total_rows == 3
    assert manifest.peak_bounded_active_bytes <= manifest.target_bytes
    assert manifest.peak_active_bytes > manifest.target_bytes
    assert len(manifest.oversized_rows) == 1
    oversized = manifest.oversized_rows[0]
    assert (oversized.stable_id, oversized.document_position) == (2, 11)
    assert oversized.arrow_nbytes == table.slice(1, 1).nbytes > manifest.target_bytes
    oversized_file = next(file for file in manifest.files if file.oversized)
    assert oversized_file.rows == 1
    assert oversized_file.path == oversized.path
    assert all(file.arrow_nbytes <= manifest.target_bytes for file in manifest.files if not file.oversized)

    reader = PayloadSpoolReader(manifest)
    output = reader.read_all()
    assert output["stable_id"].to_pylist() == [1, 2, 3]
    assert output["image"].to_pylist() == [b"small", b"x" * 512, b"also-small"]
    assert manifest.root.exists()
    reader.cleanup()
    reader.cleanup()
    assert not manifest.root.exists()


@pytest.mark.parametrize(
    ("sync_mode", "expected_calls"),
    [("fsync", ["file", "directory", "manifest", "directory"]), ("attempt_local", [])],
)
def test_payload_spool_sync_mode_controls_temporary_spool_fsyncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: str,
    expected_calls: list[str],
) -> None:
    sync_calls: list[str] = []
    monkeypatch.setattr(payload_spool_module, "_sync_file", lambda _path: sync_calls.append("file"))
    monkeypatch.setattr(payload_spool_module, "_sync_directory", lambda _path: sync_calls.append("directory"))
    monkeypatch.setattr(payload_spool_module.os, "fsync", lambda _descriptor: sync_calls.append("manifest"))
    spool = PayloadSpool(
        tmp_path / sync_mode,
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=10,
        sync_mode=sync_mode,  # type: ignore[arg-type]
    )
    spool.append(_table([1], [1], [b"payload"]))

    manifest = spool.finish()

    assert sync_calls == expected_calls
    assert manifest.sync_mode == sync_mode
    assert json.loads(manifest.path.read_text(encoding="utf-8"))["sync_mode"] == sync_mode
    assert PayloadSpoolReader(manifest).read_all()["image"].to_pylist() == [b"payload"]
    spool.cleanup()


def test_payload_spool_reader_rejects_file_sha256_mismatch_in_attempt_local_mode(tmp_path: Path) -> None:
    spool = PayloadSpool(
        tmp_path / "spool",
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=10,
        sync_mode="attempt_local",
    )
    spool.append(_table([1], [1], [b"payload"]))
    manifest = spool.finish()
    path = manifest.files[0].path
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PayloadSpoolReader(manifest).read_all()

    manifest.path.write_bytes(manifest.path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        PayloadSpoolReader(manifest)

    spool.cleanup()


def test_payload_spool_rejects_unknown_or_missing_sync_mode(tmp_path: Path) -> None:
    for invalid in ("unknown", ["fsync"]):
        with pytest.raises(ValueError, match="sync_mode must be"):
            PayloadSpool(
                tmp_path / f"invalid-{type(invalid).__name__}",
                _SCHEMA,
                target_bytes=1024,
                bucket_rows=10,
                sync_mode=invalid,  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="sync_mode must be"):
        PayloadSpool(
            tmp_path / "invalid-set",
            _SCHEMA,
            target_bytes=1024,
            bucket_rows=10,
            sync_mode={"fsync"},  # type: ignore[arg-type]
        )

    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=1024, bucket_rows=10)
    manifest = spool.finish()
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    del payload["sync_mode"]
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest sync_mode must be"):
        PayloadSpoolReader(manifest.path)

    spool.cleanup()


def test_payload_spool_reader_rejects_manifest_row_conservation_failure(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=1024, bucket_rows=10)
    spool.append(_table([1], [1], [b"payload"]))
    manifest = spool.finish()
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["total_rows"] += 1
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="row conservation"):
        PayloadSpoolReader(manifest.path)

    spool.cleanup()


def test_payload_spool_reader_rejects_schema_mismatch(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=1024, bucket_rows=10)
    spool.append(_table([1], [1], [b"payload"]))
    manifest = spool.finish()
    record = manifest.files[0]
    wrong_schema = pa.schema(
        [
            pa.field("stable_id", pa.uint64(), nullable=False),
            pa.field("document_position", pa.uint64(), nullable=False),
            pa.field("image", pa.string()),
            pa.field("attributes", pa.list_(pa.string())),
        ],
        metadata=_SCHEMA.metadata,
    )
    wrong_table = pa.Table.from_arrays(
        [
            pa.array([1], type=pa.uint64()),
            pa.array([1], type=pa.uint64()),
            pa.array(["payload"]),
            pa.array([["tag-1"]], type=pa.list_(pa.string())),
        ],
        schema=wrong_schema,
    )
    with pa.OSFile(str(record.path), "wb") as sink, pa.ipc.new_file(sink, wrong_schema) as writer:
        writer.write_table(wrong_table)

    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = hashlib.sha256(record.path.read_bytes()).hexdigest()
    payload["files"][0]["file_bytes"] = record.path.stat().st_size
    payload["files"][0]["arrow_nbytes"] = wrong_table.nbytes
    payload["total_arrow_nbytes"] = wrong_table.nbytes
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="schema mismatch"):
        PayloadSpoolReader(manifest.path).read_all()

    spool.cleanup()


def test_payload_spool_reader_rejects_file_row_count_mismatch(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=1024, bucket_rows=10)
    spool.append(_table([1], [1], [b"payload"]))
    manifest = spool.finish()
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["files"][0]["rows"] = 2
    payload["total_rows"] = 2
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="row-count mismatch"):
        PayloadSpoolReader(manifest.path).read_all()

    spool.cleanup()


def test_payload_spool_validates_schema_coordinates_and_lifecycle(tmp_path: Path) -> None:
    missing_coordinate = pa.schema(
        [pa.field("stable_id", pa.uint64(), nullable=False), pa.field("image", pa.binary())]
    )
    with pytest.raises(ValueError, match="missing coordinate"):
        PayloadSpool(tmp_path / "missing", missing_coordinate, target_bytes=100, bucket_rows=10)
    nullable_coordinates = pa.schema(
        [
            pa.field("stable_id", pa.uint64()),
            pa.field("document_position", pa.uint64()),
            pa.field("image", pa.binary()),
        ]
    )
    with pytest.raises(TypeError, match="non-nullable uint64"):
        PayloadSpool(tmp_path / "nullable", nullable_coordinates, target_bytes=100, bucket_rows=10)

    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=100, bucket_rows=10)
    with pytest.raises(RuntimeError, match="finish must be called"):
        spool.iter_tables()
    null_positions = pa.Table.from_arrays(
        [
            pa.array([1], type=pa.uint64()),
            pa.array([None], type=pa.uint64()),
            pa.array([b"payload"], type=pa.binary()),
            pa.array([["tag-1"]], type=pa.list_(pa.string())),
        ],
        schema=_SCHEMA,
    )
    with pytest.raises(ValueError, match="must not contain nulls"):
        spool.append(null_positions)

    spool.append(_table([1], [1], [b"payload"]))
    assert spool.finish() is spool.finish()
    with pytest.raises(RuntimeError, match="already been finished"):
        spool.append(_table([2], [2], [b"later"]))
    spool.cleanup()


def test_empty_payload_spool_preserves_schema_and_requires_explicit_cleanup(tmp_path: Path) -> None:
    spool = PayloadSpool(tmp_path / "spool", _SCHEMA, target_bytes=100, bucket_rows=10)

    manifest = spool.finish()
    output = spool.read_all()

    assert manifest.total_rows == 0
    assert manifest.total_arrow_nbytes == 0
    assert manifest.files == ()
    assert output.num_rows == 0
    assert output.schema.equals(_SCHEMA, check_metadata=True)
    assert manifest.root.exists()
    spool.cleanup()
    assert not manifest.root.exists()
