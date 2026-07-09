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

import json
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved import lance_coordinate_plan
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    LANCE_COORDINATE_PLAN_SCHEMA,
    STABLE_ROW_ID,
    CoordinatePlanIdentity,
    LanceCoordinatePlanTask,
    lance_coordinate_plan_schema,
    lance_coordinate_plan_sha256,
    load_coordinate_plan,
    publish_coordinate_plan,
    validate_lance_coordinate_plan,
)


def _identity(*, fragment_id: int = 7) -> CoordinatePlanIdentity:
    return CoordinatePlanIdentity(
        document_uri="s3://document-bucket/documents.lance",
        document_version=1,
        image_uri="s3://image-bucket/images.lance",
        image_version=4,
        fragment_id=fragment_id,
        sidecar_manifest_sha256="a" * 64,
        fragment_manifest_sha256="b" * 64,
    )


def _table(
    *,
    row_addresses: list[int] | None = None,
    positions: list[int] | None = None,
    stable_ids: list[int | None] | None = None,
    allow_missing: bool = False,
    chunked: bool = False,
) -> pa.Table:
    row_addresses = [100, 101, 102, 103] if row_addresses is None else row_addresses
    positions = [0, 2, 4, 6] if positions is None else positions
    stable_ids = [9, 9, 12, 15] if stable_ids is None else stable_ids
    schema = lance_coordinate_plan_schema(allow_missing=allow_missing)
    columns: list[pa.Array | pa.ChunkedArray] = []
    for values, field in zip((row_addresses, positions, stable_ids), schema, strict=True):
        if chunked:
            split = len(values) // 2
            columns.append(
                pa.chunked_array(
                    [
                        pa.array(values[:split], type=field.type),
                        pa.array(values[split:], type=field.type),
                    ],
                    type=field.type,
                )
            )
        else:
            columns.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def test_schema_requires_explicit_nullable_stable_row_id_policy() -> None:
    assert (
        pa.schema(
            [
                pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
                pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
                pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
            ]
        )
        == LANCE_COORDINATE_PLAN_SCHEMA
    )
    nullable_schema = lance_coordinate_plan_schema(allow_missing=True)
    assert nullable_schema.field(STABLE_ROW_ID).nullable is True

    nullable = _table(stable_ids=[9, None, 12, None], allow_missing=True)
    stats = validate_lance_coordinate_plan(nullable, missing_key_policy="null")
    assert stats.null_stable_row_ids == 2
    assert stats.non_null_stable_row_ids == 2

    with pytest.raises(TypeError, match="expected"):
        validate_lance_coordinate_plan(nullable, missing_key_policy="error")


@pytest.mark.parametrize(
    ("row_addresses", "positions", "message"),
    [
        ([100, 100, 102, 103], [0, 1, 2, 3], "document_rowaddr values must be unique"),
        ([100, 101, 102, 103], [0, 0, 2, 3], "document_position values must be unique"),
        ([100, 101, 102, 103], [0, 2, 1, 3], "strictly ordered"),
    ],
)
def test_validation_rejects_nondeterministic_document_position_order(
    row_addresses: list[int],
    positions: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_lance_coordinate_plan(_table(row_addresses=row_addresses, positions=positions))


def test_validation_rejects_null_hidden_behind_nonnullable_schema() -> None:
    malformed = _table(stable_ids=[9, None, 12, 15])

    with pytest.raises(ValueError, match="contains nulls"):
        validate_lance_coordinate_plan(malformed)


def test_canonical_digest_ignores_input_chunk_boundaries_and_counts_duplicates() -> None:
    contiguous = _table()
    chunked = _table(chunked=True)

    stats = validate_lance_coordinate_plan(chunked)

    assert stats.rows == 4
    assert stats.unique_document_rowaddrs == 4
    assert stats.unique_document_positions == 4
    assert stats.unique_stable_row_ids == 3
    assert stats.duplicate_stable_row_id_occurrences == 1
    assert stats.document_position_min == 0
    assert stats.document_position_max == 6
    assert lance_coordinate_plan_sha256(contiguous) == lance_coordinate_plan_sha256(chunked)


def test_publish_load_and_adopt_exact_existing_artifact(tmp_path: Path) -> None:
    identity = _identity()
    table = _table()

    first = publish_coordinate_plan(tmp_path, table, identity)
    parquet_before = Path(first.data).read_bytes()
    manifest_before = Path(first.manifest_path).read_bytes()
    loaded, manifest = load_coordinate_plan(
        first,
        expected_identity=identity,
        allow_missing=False,
    )
    second = publish_coordinate_plan(tmp_path, table, identity)

    assert isinstance(first, LanceCoordinatePlanTask)
    assert first.num_items == 1
    assert first.validate() is True
    assert first.get_deterministic_id() == second.get_deterministic_id()
    assert loaded.equals(table)
    assert manifest["document"] == {
        "uri": identity.document_uri,
        "version": identity.document_version,
        "fragment_id": identity.fragment_id,
    }
    assert manifest["image"] == {"uri": identity.image_uri, "version": identity.image_version}
    assert manifest["sidecar_manifest_sha256"] == identity.sidecar_manifest_sha256
    assert manifest["fragment_manifest_sha256"] == identity.fragment_manifest_sha256
    assert manifest["coordinates"]["rows"] == 4
    assert manifest["coordinates"]["unique_stable_row_ids"] == 3
    assert manifest["coordinates"]["duplicate_stable_row_id_occurrences"] == 1
    assert manifest["coordinates"]["canonical_ipc_sha256"] == lance_coordinate_plan_sha256(table)
    assert manifest["coordinates"]["schema"][-1] == {
        "name": STABLE_ROW_ID,
        "type": "uint64",
        "nullable": False,
    }
    assert first._metadata["lance_coordinate_plan"]["adopted"] is False
    assert second._metadata["lance_coordinate_plan"]["adopted"] is True
    assert Path(first.data).read_bytes() == parquet_before
    assert Path(first.manifest_path).read_bytes() == manifest_before
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_rejects_conflicting_and_recovers_exact_partial_artifact(tmp_path: Path) -> None:
    identity = _identity()
    task = publish_coordinate_plan(tmp_path, _table(), identity)

    with pytest.raises(ValueError, match="content does not match"):
        publish_coordinate_plan(tmp_path, _table(stable_ids=[9, 10, 12, 15]), identity)

    Path(task.manifest_path).unlink()
    recovered = publish_coordinate_plan(tmp_path, _table(), identity)
    assert recovered._metadata["lance_coordinate_plan"]["adopted"] is True
    assert load_coordinate_plan(recovered)[0].equals(_table())

    Path(recovered.manifest_path).unlink()
    with pytest.raises(ValueError, match="content does not match"):
        publish_coordinate_plan(tmp_path, _table(stable_ids=[9, 10, 12, 15]), identity)
    assert not Path(recovered.manifest_path).exists()


def test_load_rejects_tampered_manifest(tmp_path: Path) -> None:
    task = publish_coordinate_plan(tmp_path, _table(), _identity())
    manifest_path = Path(task.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinates"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reconcile"):
        load_coordinate_plan(task)


def test_load_rejects_wrong_identity_policy_and_corrupt_parquet(tmp_path: Path) -> None:
    identity = _identity()
    task = publish_coordinate_plan(
        tmp_path,
        _table(stable_ids=[9, None, 12, 15], allow_missing=True),
        identity,
        allow_missing=True,
    )

    with pytest.raises(ValueError, match="expected_identity"):
        load_coordinate_plan(task, expected_identity=_identity(fragment_id=8))
    with pytest.raises(ValueError, match="missing-key policy"):
        load_coordinate_plan(task, allow_missing=False)

    with Path(task.data).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match=r"unreadable|does not reconcile"):
        load_coordinate_plan(task)


def test_failed_parquet_write_leaves_no_published_or_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(*_: object, **__: object) -> None:
        msg = "injected write failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(lance_coordinate_plan.pq, "write_table", fail_write)

    with pytest.raises(RuntimeError, match="injected write failure"):
        publish_coordinate_plan(tmp_path, _table(), _identity())

    assert list(tmp_path.iterdir()) == []
