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
from dataclasses import replace
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
)
from nemo_curator.stages.interleaved.lance_payload_overlay import (
    LancePayloadOverlayIdentity,
    lance_payload_fetch_group,
    lance_payload_overlay_config_sha256,
    lance_payload_overlay_root,
    lance_payload_overlay_source_identity_sha256,
    lance_payload_overlay_task,
    payload_coordinate_sha256,
    publish_lance_payload_overlay,
    validate_lance_payload_overlay,
)
from nemo_curator.stages.interleaved.lance_payload_overlay_reader import LancePayloadOverlayReader
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool
from nemo_curator.tasks import EmptyTask

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_FRAGMENT_ID = 7
_IMAGE_COLUMNS = {"image": "binary_content"}
_SCHEMA = pa.schema(
    [
        pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
        pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
        pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
        pa.field("image", pa.large_binary(), nullable=False, metadata={b"content-type": b"image/jpeg"}),
    ]
)


def _table(positions: list[int], stable_ids: list[int]) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array([(_FRAGMENT_ID << 32) | value for value in positions], type=pa.uint64()),
            pa.array(positions, type=pa.uint64()),
            pa.array(stable_ids, type=pa.uint64()),
            pa.array([f"image-{value}".encode() for value in stable_ids], type=pa.large_binary()),
        ],
        schema=_SCHEMA,
    )


def _identity(coordinates: pa.Table, *, logical_rows: int | None = None) -> LancePayloadOverlayIdentity:
    logical_rows = coordinates.num_rows if logical_rows is None else logical_rows
    unique_rows = len(set(coordinates[STABLE_ROW_ID].to_pylist()))
    return LancePayloadOverlayIdentity(
        document_uri="s3://documents/dataset",
        document_version=1,
        image_uri="s3://images/dataset",
        image_version=4,
        fragment_id=_FRAGMENT_ID,
        coordinate_plan_sha256="a" * 64,
        coordinate_manifest_sha256="d" * 64,
        payload_coordinate_sha256=payload_coordinate_sha256(coordinates),
        sidecar_manifest_sha256="b" * 64,
        fragment_manifest_sha256="c" * 64,
        overlay_config_sha256=lance_payload_overlay_config_sha256(
            _IMAGE_COLUMNS,
            payload_schema=_SCHEMA,
            payload_window_bytes=1024,
            bucket_rows=4,
        ),
        expected_document_rows=12,
        expected_coordinate_rows=logical_rows + 1,
        expected_logical_rows=logical_rows,
        expected_unique_rows=unique_rows,
        expected_null_rows=1,
    )


def _producer_metrics(
    identity: LancePayloadOverlayIdentity,
    arrow_nbytes: int,
    payload_bytes: int,
    files: int,
) -> dict[str, object]:
    unique_rows = identity.expected_unique_rows
    return {
        "stream_complete": True,
        "completion_order_output": True,
        "batch_stable_ids_sorted": True,
        "exact_operation_coverage": True,
        "logical_rows": identity.expected_logical_rows,
        "unique_rows": unique_rows,
        "null_rows_skipped": identity.expected_null_rows,
        "scatter_input_rows": identity.expected_logical_rows,
        "input_stable_rows": unique_rows,
        "stream_output_rows": unique_rows,
        "payload_take_rows": unique_rows,
        "take_rows": unique_rows,
        "payload_batches_planned": unique_rows,
        "payload_batches_emitted": unique_rows,
        "payload_read_calls": unique_rows,
        "take_calls": unique_rows,
        "sparse_calls_avoided": 0,
        "payload_bytes": payload_bytes,
        "actual_payload_bytes": payload_bytes,
        "spooled_payload_bytes": payload_bytes,
        "spool_arrow_bytes": arrow_nbytes,
        "payload_spool_arrow_bytes": arrow_nbytes,
        "payload_spool_files": files,
        "payload_spool_oversized_rows": 0,
        "lance_read_iops": 0,
        "lance_read_bytes": 0,
    }


def _shared_local_metrics(
    identity: LancePayloadOverlayIdentity,
    arrow_nbytes: int,
    payload_bytes: int,
    files: int,
    peak_active_bytes: int,
) -> dict[str, object]:
    unique_rows = identity.expected_unique_rows
    return {
        "shared_fetch_group": True,
        "stream_complete": True,
        "completion_order_output": True,
        "batch_stable_ids_sorted": True,
        "exact_operation_coverage": True,
        "logical_rows": identity.expected_logical_rows,
        "unique_rows": unique_rows,
        "null_rows_skipped": identity.expected_null_rows,
        "scatter_input_rows": identity.expected_logical_rows,
        "duplicate_fanout": identity.expected_logical_rows / unique_rows if unique_rows else 0.0,
        "spooled_payload_bytes": payload_bytes,
        "payload_batches_contributed": 1 if unique_rows else 0,
        "spool_arrow_bytes": arrow_nbytes,
        "payload_spool_arrow_bytes": arrow_nbytes,
        "payload_spool_files": files,
        "payload_spool_oversized_rows": 0,
        "payload_spool_peak_active_bytes": peak_active_bytes,
    }


def _fetch_group_metrics(  # noqa: PLR0913
    identities: tuple[LancePayloadOverlayIdentity, ...],
    *,
    global_unique_rows: int,
    actual_payload_bytes: int,
    spooled_payload_bytes: int,
    spool_arrow_bytes: int,
    payload_spool_files: int,
    shared_spool_peak_active_bytes: int,
) -> dict[str, object]:
    logical_rows = sum(identity.expected_logical_rows for identity in identities)
    sum_plan_unique_rows = sum(identity.expected_unique_rows for identity in identities)
    take_calls = 1 if global_unique_rows else 0
    return {
        "stream_complete": True,
        "completion_order_output": True,
        "batch_stable_ids_sorted": True,
        "exact_operation_coverage": True,
        "logical_rows": logical_rows,
        "unique_rows": global_unique_rows,
        "sum_plan_unique_rows": sum_plan_unique_rows,
        "cross_plan_unique_ids_coalesced": sum_plan_unique_rows - global_unique_rows,
        "duplicate_fanout": logical_rows / global_unique_rows if global_unique_rows else 0.0,
        "scatter_input_rows": logical_rows,
        "input_stable_rows": global_unique_rows,
        "stream_output_rows": global_unique_rows,
        "payload_take_rows": global_unique_rows,
        "take_rows": global_unique_rows,
        "payload_batches_planned": take_calls,
        "payload_batches_emitted": take_calls,
        "payload_read_calls": take_calls,
        "take_calls": take_calls,
        "sparse_calls_avoided": global_unique_rows - take_calls,
        "payload_bytes": actual_payload_bytes,
        "actual_payload_bytes": actual_payload_bytes,
        "spooled_payload_bytes": spooled_payload_bytes,
        "spool_arrow_bytes": spool_arrow_bytes,
        "payload_spool_files": payload_spool_files,
        "payload_spool_oversized_rows": 0,
        "shared_spool_budget_bytes": 4096,
        "shared_spool_peak_active_bytes": shared_spool_peak_active_bytes,
        "shared_spool_peak_bounded_active_bytes": shared_spool_peak_active_bytes,
        "coordinate_member_count": len(identities),
        "coordinate_queue_rows": logical_rows,
        "coordinate_workspace_bytes": 512,
        "coordinate_workspace_estimated_bytes": 768,
        "max_coordinate_workspace_bytes": 1024,
        "lance_read_iops": 7,
        "lance_read_bytes": 8192,
    }


def _rehash_fetch_group(fetch_group: dict[str, object]) -> None:
    payload = {name: value for name, value in fetch_group.items() if name != "fetch_group_sha256"}
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    fetch_group["fetch_group_sha256"] = hashlib.sha256(content).hexdigest()


def _rewrite_manifest(final: Path, update: Callable[[dict[str, object]], None]) -> None:
    manifest = final / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    update(payload)
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _publish(tmp_path: Path) -> tuple[Path, LancePayloadOverlayIdentity]:
    coordinates = _table([1, 3, 8], [5, 2, 5])
    identity = _identity(coordinates)
    final = lance_payload_overlay_root(tmp_path / "output", identity)
    final.parent.mkdir()
    attempt = final.parent / f".{final.name}.attempt.tmp"
    attempt.mkdir()
    spool = PayloadSpool(
        attempt / "payload",
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=4,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
        sync_mode="fsync",
    )
    # Completion order is intentionally unrelated to document order.
    spool.append(coordinates.take(pa.array([2, 0])))
    spool.append(coordinates.take(pa.array([1])))
    payload = spool.finish()
    artifact = publish_lance_payload_overlay(
        attempt,
        final,
        identity=identity,
        image_columns=_IMAGE_COLUMNS,
        payload=payload,
        producer_metrics=_producer_metrics(
            identity,
            payload.total_arrow_nbytes,
            coordinates["image"].nbytes,
            len(payload.files),
        ),
    )
    assert artifact.adopted is False
    return final, identity


def _publish_shared(tmp_path: Path) -> tuple[Path, LancePayloadOverlayIdentity, dict[str, object]]:
    coordinates = _table([1, 3, 8], [5, 2, 5])
    identity = _identity(coordinates)
    sibling = replace(
        identity,
        fragment_id=_FRAGMENT_ID + 1,
        coordinate_plan_sha256="e" * 64,
        coordinate_manifest_sha256="f" * 64,
    )
    identities = (identity, sibling)
    final = lance_payload_overlay_root(tmp_path / "output", identity)
    final.parent.mkdir()
    attempt = final.parent / f".{final.name}.attempt.tmp"
    attempt.mkdir()
    spool = PayloadSpool(
        attempt / "payload",
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=4,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
        sync_mode="fsync",
    )
    spool.append(coordinates.take(pa.array([2, 0])))
    spool.append(coordinates.take(pa.array([1])))
    payload = spool.finish()
    local_payload_bytes = coordinates["image"].nbytes
    actual_payload_bytes = pa.array([b"image-2", b"image-5"], type=pa.large_binary()).nbytes
    group_metrics = _fetch_group_metrics(
        identities,
        global_unique_rows=2,
        actual_payload_bytes=actual_payload_bytes,
        spooled_payload_bytes=2 * local_payload_bytes,
        spool_arrow_bytes=2 * payload.total_arrow_nbytes,
        payload_spool_files=2 * len(payload.files),
        shared_spool_peak_active_bytes=payload.peak_active_bytes,
    )
    fetch_group = lance_payload_fetch_group(identities, group_metrics)
    assert fetch_group == lance_payload_fetch_group(tuple(reversed(identities)), group_metrics)
    artifact = publish_lance_payload_overlay(
        attempt,
        final,
        identity=identity,
        image_columns=_IMAGE_COLUMNS,
        payload=payload,
        producer_metrics=_shared_local_metrics(
            identity,
            payload.total_arrow_nbytes,
            local_payload_bytes,
            len(payload.files),
            payload.peak_active_bytes,
        ),
        fetch_group=fetch_group,
    )
    assert artifact.fetch_group == fetch_group
    return final, identity, fetch_group


def test_payload_overlay_publishes_and_adopts_completion_order_parts(tmp_path: Path) -> None:
    final, identity = _publish(tmp_path)

    artifact = validate_lance_payload_overlay(
        final,
        expected_identity=identity,
        expected_image_columns=_IMAGE_COLUMNS,
    )
    task = lance_payload_overlay_task(artifact, metadata={"upstream": "coordinate-plan"})
    tables = list(artifact.payload.files)

    assert artifact.adopted is True
    assert artifact.payload.total_rows == 3
    assert {record.bucket for record in tables} == {0, 2}
    assert task.data == [str(record.path) for record in artifact.payload.files]
    assert task._metadata["upstream"] == "coordinate-plan"
    assert task._metadata["lance_payload_overlay"]["duplicate_occurrences"] == 1
    assert task._metadata["lance_payload_overlay"]["null_rows"] == 1
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "lance_payload_overlay"
    assert "fetch_group" not in manifest
    assert artifact.fetch_group is None
    assert "fetch_group" not in task._metadata["lance_payload_overlay"]


def test_payload_overlay_publishes_hash_bound_shared_fetch_provenance(tmp_path: Path) -> None:
    final, identity, fetch_group = _publish_shared(tmp_path)

    artifact = validate_lance_payload_overlay(final, expected_identity=identity)
    task = lance_payload_overlay_task(artifact)
    stored = json.loads((final / "manifest.json").read_text(encoding="utf-8"))["fetch_group"]

    assert stored == fetch_group
    assert artifact.fetch_group == fetch_group
    assert task._metadata["lance_payload_overlay"]["fetch_group"] == fetch_group
    assert lance_payload_overlay_source_identity_sha256(identity) in fetch_group["member_source_identity_sha256"]
    assert fetch_group["metrics"]["cross_plan_unique_ids_coalesced"] == 2
    assert fetch_group["metrics"]["sparse_calls_avoided"] == 1
    assert "lance_read_iops" not in artifact.producer_metrics


def test_payload_overlay_rejects_tampered_fetch_group_hash(tmp_path: Path) -> None:
    final, _, _ = _publish_shared(tmp_path)

    def tamper(payload: dict[str, object]) -> None:
        payload["fetch_group"]["metrics"]["lance_read_iops"] += 1

    _rewrite_manifest(final, tamper)
    with pytest.raises(ValueError, match="fetch_group SHA-256 mismatch"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_rejects_fetch_group_without_current_member(tmp_path: Path) -> None:
    final, identity, _ = _publish_shared(tmp_path)

    def tamper(payload: dict[str, object]) -> None:
        fetch_group = payload["fetch_group"]
        current = lance_payload_overlay_source_identity_sha256(identity)
        fetch_group["member_source_identity_sha256"].remove(current)
        fetch_group["metrics"]["coordinate_member_count"] -= 1
        _rehash_fetch_group(fetch_group)

    _rewrite_manifest(final, tamper)
    with pytest.raises(ValueError, match="identity is not a member"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_rejects_rehashed_fetch_group_metric_inconsistency(tmp_path: Path) -> None:
    final, _, _ = _publish_shared(tmp_path)

    def tamper(payload: dict[str, object]) -> None:
        fetch_group = payload["fetch_group"]
        fetch_group["metrics"]["cross_plan_unique_ids_coalesced"] += 1
        _rehash_fetch_group(fetch_group)

    _rewrite_manifest(final, tamper)
    with pytest.raises(ValueError, match="cross-plan coalescing count"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_rejects_rehashed_impossible_coordinate_budget(tmp_path: Path) -> None:
    final, _identity, _fetch_group = _publish_shared(tmp_path)

    def tamper(payload: dict[str, object]) -> None:
        fetch_group = payload["fetch_group"]
        fetch_group["metrics"]["coordinate_workspace_estimated_bytes"] = 511
        _rehash_fetch_group(fetch_group)

    _rewrite_manifest(final, tamper)
    with pytest.raises(ValueError, match="coordinate workspace accounting"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_rejects_group_physical_metrics_repeated_locally(tmp_path: Path) -> None:
    final, _, _ = _publish_shared(tmp_path)

    def tamper(payload: dict[str, object]) -> None:
        payload["producer_metrics"]["lance_read_iops"] = payload["fetch_group"]["metrics"]["lance_read_iops"]

    _rewrite_manifest(final, tamper)
    with pytest.raises(ValueError, match="repeats fetch-group metrics locally"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_rejects_changed_part_bytes(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    artifact = validate_lance_payload_overlay(final, verify_payload=False)
    part = artifact.payload.files[0].path
    content = bytearray(part.read_bytes())
    content[len(content) // 2] ^= 1
    part.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_lance_payload_overlay(final)


def test_payload_overlay_task_rejects_manifest_only_validation(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    artifact = validate_lance_payload_overlay(final, verify_payload=False)

    assert artifact.payload_verified is False
    with pytest.raises(ValueError, match="requires full payload verification"):
        lance_payload_overlay_task(artifact)


def test_payload_overlay_rejects_extra_and_symlink_entries(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    (final / "unexpected").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected entries"):
        validate_lance_payload_overlay(final, verify_payload=False)

    (final / "unexpected").unlink()
    manifest = final / "manifest.json"
    manifest.rename(final / "real-manifest.json")
    manifest.symlink_to("real-manifest.json")
    with pytest.raises(ValueError, match="regular, non-symlink"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_rejects_wrong_coordinates_with_matching_counts(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    artifact = validate_lance_payload_overlay(final, verify_payload=False)
    part = artifact.payload.files[0].path
    with pa.memory_map(str(part), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    changed = table.set_column(
        table.schema.get_field_index(STABLE_ROW_ID),
        table.schema.field(STABLE_ROW_ID),
        pa.array([99] * table.num_rows, type=pa.uint64()),
    )
    with pa.OSFile(str(part), "wb") as sink, pa.ipc.new_file(sink, changed.schema) as writer:
        writer.write_table(changed)
    inner = final / "payload" / "manifest.json"
    inner_payload = json.loads(inner.read_text(encoding="utf-8"))
    import hashlib

    inner_payload["files"][0]["sha256"] = hashlib.sha256(part.read_bytes()).hexdigest()
    inner_payload["files"][0]["file_bytes"] = part.stat().st_size
    inner.write_text(json.dumps(inner_payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    outer = final / "manifest.json"
    outer_payload = json.loads(outer.read_text(encoding="utf-8"))
    outer_payload["payload"]["manifest_sha256"] = hashlib.sha256(inner.read_bytes()).hexdigest()
    outer.write_text(json.dumps(outer_payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pinned coordinate plan"):
        validate_lance_payload_overlay(final)


def test_payload_overlay_supports_an_all_missing_coordinate_plan(tmp_path: Path) -> None:
    empty = _table([], [])
    identity = _identity(empty, logical_rows=0)
    final = lance_payload_overlay_root(tmp_path / "output", identity)
    final.parent.mkdir()
    attempt = final.parent / f".{final.name}.attempt.tmp"
    attempt.mkdir()
    spool = PayloadSpool(
        attempt / "payload",
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=4,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
        sync_mode="fsync",
    )
    artifact = publish_lance_payload_overlay(
        attempt,
        final,
        identity=identity,
        image_columns=_IMAGE_COLUMNS,
        payload=spool.finish(),
        producer_metrics=_producer_metrics(identity, 0, 0, 0),
    )

    adopted = validate_lance_payload_overlay(final)
    assert artifact.payload.files == ()
    assert adopted.payload.total_rows == 0
    assert lance_payload_overlay_task(adopted).data == []


def test_payload_overlay_requires_fsync_spool_and_matching_identity(tmp_path: Path) -> None:
    coordinates = _table([1, 3, 8], [5, 2, 5])
    identity = _identity(coordinates)
    final = lance_payload_overlay_root(tmp_path / "output", identity)
    final.parent.mkdir()
    attempt = final.parent / f".{final.name}.attempt.tmp"
    attempt.mkdir()
    spool = PayloadSpool(
        attempt / "payload",
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=4,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
        sync_mode="attempt_local",
    )
    spool.append(coordinates)

    with pytest.raises(ValueError, match="require an fsync"):
        publish_lance_payload_overlay(
            attempt,
            final,
            identity=identity,
            image_columns=_IMAGE_COLUMNS,
            payload=spool.finish(),
            producer_metrics=_producer_metrics(
                identity,
                coordinates.nbytes,
                coordinates["image"].nbytes,
                1,
            ),
        )


def test_payload_overlay_rejects_corrupt_part_before_atomic_publication(tmp_path: Path) -> None:
    coordinates = _table([1, 3, 8], [5, 2, 5])
    identity = _identity(coordinates)
    final = lance_payload_overlay_root(tmp_path / "output", identity)
    final.parent.mkdir()
    attempt = final.parent / f".{final.name}.attempt.tmp"
    attempt.mkdir()
    spool = PayloadSpool(
        attempt / "payload",
        _SCHEMA,
        target_bytes=1024,
        bucket_rows=4,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
        sync_mode="fsync",
    )
    spool.append(coordinates)
    payload = spool.finish()
    part = payload.files[0].path
    content = bytearray(part.read_bytes())
    content[len(content) // 2] ^= 1
    part.write_bytes(content)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        publish_lance_payload_overlay(
            attempt,
            final,
            identity=identity,
            image_columns=_IMAGE_COLUMNS,
            payload=payload,
            producer_metrics=_producer_metrics(
                identity,
                payload.total_arrow_nbytes,
                coordinates["image"].nbytes,
                len(payload.files),
            ),
        )

    assert not final.exists()
    assert attempt.exists()


def test_payload_overlay_rejects_metrics_that_do_not_reconcile(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    manifest = final / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["producer_metrics"]["logical_rows"] += 1
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"logical_rows.*does not reconcile"):
        validate_lance_payload_overlay(final, verify_payload=False)


def test_payload_overlay_reader_emits_stable_checkpoint_source_task(tmp_path: Path) -> None:
    final, identity = _publish(tmp_path)
    reader = LancePayloadOverlayReader(
        overlay_root=str(final.parent),
        document_uri=identity.document_uri,
        document_version=identity.document_version,
        image_uri=identity.image_uri,
        image_version=identity.image_version,
        sidecar_manifest_sha256=identity.sidecar_manifest_sha256,
        fragment_manifest_sha256=identity.fragment_manifest_sha256,
        overlay_config_sha256=identity.overlay_config_sha256,
        image_columns=_IMAGE_COLUMNS,
        expected_fragment_ids=[_FRAGMENT_ID],
    )

    first = reader.process(EmptyTask())
    second = reader.process(EmptyTask())

    assert len(first) == 1
    assert first[0].manifest_path == str(final / "manifest.json")
    assert first[0].get_deterministic_id() == second[0].get_deterministic_id()
    assert first[0]._metadata["lance_payload_overlay_source"]["payload_verified"] is True
    assert reader.is_source_stage is True
    assert reader.is_resumable is True
    assert reader.num_workers() == 1


def test_payload_overlay_reader_rejects_inventory_mismatch_and_orphan_lock(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    reader = LancePayloadOverlayReader(
        overlay_root=str(final.parent),
        expected_fragment_ids=[_FRAGMENT_ID + 1],
    )
    with pytest.raises(ValueError, match="fragment inventory"):
        reader.process(EmptyTask())

    orphan = final.parent / ".fragment-00000099-overlay-aaaaaaaaaaaaaaaa.lock"
    orphan.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan locks"):
        LancePayloadOverlayReader(overlay_root=str(final.parent)).process(EmptyTask())


def test_payload_overlay_reader_rejects_identity_inconsistent_directory_name(tmp_path: Path) -> None:
    final, _ = _publish(tmp_path)
    renamed = final.with_name("fragment-00000007-overlay-0000000000000000")
    final.rename(renamed)

    with pytest.raises(ValueError, match="directory name"):
        LancePayloadOverlayReader(overlay_root=str(renamed.parent)).process(EmptyTask())
