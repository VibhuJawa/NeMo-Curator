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

import gc
import json
import weakref
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.lance_patch_artifact import (
    LancePatchArtifactIdentity,
    LancePatchArtifactWriter,
    validate_lance_patch_artifact,
)
from nemo_curator.tasks import FileGroupTask

_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.uint64(), nullable=False),
        pa.field("document_position", pa.uint64(), nullable=False),
        pa.field("image", pa.binary()),
    ],
    metadata={b"artifact": b"lance-patch-test"},
)


def _identity(
    *,
    expected_rows: int = 5,
    coordinate: str = "c" * 64,
    patch_config: str = "e" * 64,
) -> LancePatchArtifactIdentity:
    return LancePatchArtifactIdentity(
        document_uri="s3://document-bucket/documents.lance",
        document_version=1,
        image_uri="s3://image-bucket/images.lance",
        image_version=4,
        fragment_id=7,
        coordinate_plan_sha256=coordinate,
        patch_config_sha256=patch_config,
        expected_rows=expected_rows,
    )


def _table(positions: list[int], *, payload_prefix: bytes = b"image") -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array([1000 + position for position in positions], type=pa.uint64()),
            pa.array(positions, type=pa.uint64()),
            pa.array([payload_prefix + str(position).encode() for position in positions], type=pa.binary()),
        ],
        schema=_SCHEMA,
    )


def _write_complete(root: Path) -> tuple[LancePatchArtifactWriter, FileGroupTask]:
    writer = LancePatchArtifactWriter(root, _SCHEMA, _identity())
    writer.append(_table([0, 1]), oversized_sample_count=1)
    writer.append(_table([2, 3, 4]), oversized_sample_count=2)
    return writer, writer.finish()


def test_patch_writer_publishes_deterministic_parts_manifest_and_task(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    tasks: list[FileGroupTask] = []
    manifests: list[dict[str, object]] = []
    part_hashes: list[list[str]] = []
    for root in roots:
        writer = LancePatchArtifactWriter(root, _SCHEMA, _identity())
        first = _table([0, 1])
        first_reference = weakref.ref(first)
        writer.append(first, oversized_sample_count=1)
        del first
        gc.collect()
        assert first_reference() is None
        writer.append(_table([2, 3, 4]), oversized_sample_count=2)

        task = writer.finish()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        tasks.append(task)
        manifests.append(manifest)
        part_hashes.append([part["sha256"] for part in manifest["parts"]])

        assert isinstance(task, FileGroupTask)
        assert [Path(path).name for path in task.data] == ["part-00000000.parquet", "part-00000001.parquet"]
        assert task._metadata["source_files"] == [*task.data, str(root / "manifest.json")]
        assert task._metadata["lance_patch_artifact"] == {
            "manifest_path": str(root / "manifest.json"),
            "manifest_sha256": task._metadata["lance_patch_artifact"]["manifest_sha256"],
            "coordinate_plan_sha256": "c" * 64,
            "patch_config_sha256": "e" * 64,
            "document_version": 1,
            "image_version": 4,
            "fragment_id": 7,
            "rows": 5,
            "size_bytes": manifest["total_size_bytes"],
            "oversized_sample_count": 3,
            "adopted": False,
        }
        assert [(part["row_start"], part["row_stop"]) for part in manifest["parts"]] == [(0, 2), (2, 5)]
        assert [part["schema_sha256"] for part in manifest["parts"]] == [manifest["schema"]["sha256"]] * 2
        assert manifest["total_rows"] == 5
        assert manifest["oversized_sample_count"] == 3
        assert manifest["coordinate_plan_sha256"] == "c" * 64
        assert not list(root.glob(".*.tmp"))

        combined = pa.concat_tables([pq.read_table(path) for path in task.data])
        assert combined["document_position"].to_pylist() == [0, 1, 2, 3, 4]

    assert part_hashes[0] == part_hashes[1]
    assert manifests[0] == manifests[1]


def test_complete_retry_is_adopted_only_after_full_validation(tmp_path: Path) -> None:
    root = tmp_path / "patch"
    _, first = _write_complete(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    retry = LancePatchArtifactWriter(root, _SCHEMA, _identity(), expected_oversized_sample_count=3)
    second = retry.finish()

    assert retry.adopted is True
    assert second.data == first.data
    assert second._metadata["lance_patch_artifact"]["adopted"] is True
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    with pytest.raises(RuntimeError, match="already complete"):
        retry.append(_table([10]))


def test_retry_rejects_partial_or_unexpected_state(tmp_path: Path) -> None:
    partial_root = tmp_path / "partial"
    writer = LancePatchArtifactWriter(partial_root, _SCHEMA, _identity())
    writer.append(_table([0, 1]))

    with pytest.raises(RuntimeError, match="publication is partial"):
        LancePatchArtifactWriter(partial_root, _SCHEMA, _identity())

    complete_root = tmp_path / "complete"
    _write_complete(complete_root)
    (complete_root / "unexpected.tmp").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="unexpected files"):
        LancePatchArtifactWriter(complete_root, _SCHEMA, _identity())


def test_retry_rejects_tampered_part_and_manifest_identity(tmp_path: Path) -> None:
    part_root = tmp_path / "part-tamper"
    _, task = _write_complete(part_root)
    part_path = Path(task.data[0])
    content = bytearray(part_path.read_bytes())
    content[len(content) // 2] ^= 1
    part_path.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256"):
        LancePatchArtifactWriter(part_root, _SCHEMA, _identity())

    manifest_root = tmp_path / "manifest-tamper"
    _write_complete(manifest_root)
    manifest_path = manifest_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinate_plan_sha256"] = "d" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match"):
        LancePatchArtifactWriter(manifest_root, _SCHEMA, _identity())


def test_append_rejects_schema_and_document_position_order_errors(tmp_path: Path) -> None:
    wrong_schema_writer = LancePatchArtifactWriter(tmp_path / "wrong-schema", _SCHEMA, _identity())
    wrong = _table([0]).drop(["image"])
    with pytest.raises(TypeError, match="schema does not match"):
        wrong_schema_writer.append(wrong)

    unordered_writer = LancePatchArtifactWriter(tmp_path / "unordered", _SCHEMA, _identity())
    with pytest.raises(ValueError, match="strictly ordered"):
        unordered_writer.append(_table([2, 1]))

    cross_part_writer = LancePatchArtifactWriter(tmp_path / "cross-part", _SCHEMA, _identity())
    cross_part_writer.append(_table([0, 1]))
    with pytest.raises(ValueError, match="parts must be strictly ordered"):
        cross_part_writer.append(_table([1, 2, 3]))

    gap_writer = LancePatchArtifactWriter(tmp_path / "gap", _SCHEMA, _identity())
    with pytest.raises(ValueError, match="cover exactly"):
        gap_writer.append(_table([0, 2]))


def test_finish_and_validation_fail_closed_on_row_conservation(tmp_path: Path) -> None:
    root = tmp_path / "short"
    writer = LancePatchArtifactWriter(root, _SCHEMA, _identity())
    writer.append(_table([0, 1]))
    with pytest.raises(RuntimeError, match="row conservation"):
        writer.finish()

    complete_root = tmp_path / "complete"
    _write_complete(complete_root)
    manifest_path = complete_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parts"][1]["row_start"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reconcile"):
        validate_lance_patch_artifact(complete_root)


def test_retry_rejects_requested_schema_or_coordinate_plan_conflict(tmp_path: Path) -> None:
    root = tmp_path / "patch"
    _write_complete(root)
    changed_schema = pa.schema(
        [
            pa.field("stable_id", pa.uint64(), nullable=False),
            pa.field("document_position", pa.uint64(), nullable=False),
            pa.field("image", pa.large_binary()),
        ],
        metadata=_SCHEMA.metadata,
    )

    with pytest.raises(TypeError, match="schema does not match"):
        LancePatchArtifactWriter(root, changed_schema, _identity())
    with pytest.raises(ValueError, match="identity does not match"):
        LancePatchArtifactWriter(root, _SCHEMA, _identity(coordinate="d" * 64))
    with pytest.raises(ValueError, match="identity does not match"):
        LancePatchArtifactWriter(root, _SCHEMA, _identity(patch_config="d" * 64))
    with pytest.raises(ValueError, match="oversized-sample count"):
        LancePatchArtifactWriter(root, _SCHEMA, _identity(), expected_oversized_sample_count=2)
