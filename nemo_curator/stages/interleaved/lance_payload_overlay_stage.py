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

"""Checkpointable remote Lance payload fetch into durable Arrow overlays."""

from __future__ import annotations

import hashlib
import resource
import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    LanceCoordinatePlanTask,
    load_coordinate_plan,
)
from nemo_curator.stages.interleaved.lance_payload_materialize import materialize_lance_payload_to_spool
from nemo_curator.stages.interleaved.lance_payload_overlay import (
    LancePayloadOverlayIdentity,
    lance_payload_overlay_config_sha256,
    lance_payload_overlay_root,
    lance_payload_overlay_task,
    payload_coordinate_sha256,
    publish_lance_payload_overlay,
    validate_lance_payload_overlay,
)
from nemo_curator.stages.interleaved.lance_payload_patch_stage import (
    LanceCoordinatePayloadPatchStage,
    PayloadWindowBytes,
    _acquire_artifact_lock,
    _plan_identity,
    _release_artifact_lock,
    _remove_orphan_attempts,
)
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.tasks import FileGroupTask


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"coordinate plan manifest {name} section is invalid"
        raise TypeError(msg)
    return value


def _manifest_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"coordinate plan manifest {name} must be a nonnegative integer"
        raise ValueError(msg)
    return value


class LanceCoordinatePayloadOverlayStage(LanceCoordinatePayloadPatchStage):
    """Fetch one coordinate plan into an identity-bound Arrow overlay.

    The remote payload path ends when the overlay directory is atomically
    published. Full document reconstruction is deliberately a separate stage,
    so a retry can adopt this artifact without repeating any image reads.
    """

    name = "lance_coordinate_payload_overlay"
    is_resumable = True

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_uri: str,
        image_version: int,
        output_root: str,
        image_storage_options: Mapping[str, str] | None = None,
        image_columns: Mapping[str, str] | None = None,
        payload_window_bytes: PayloadWindowBytes | int = "1GiB",
        bucket_rows: int = 131_072,
        estimated_payload_bytes_per_row: int = 128 * 1024,
        fetch_batch_size: int = 1024,
        max_pending: int = 16,
        payload_actor_cpus: int = 8,
        payload_overlay_workers: int | None = None,
        document_storage_options: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            image_uri=image_uri,
            image_version=image_version,
            output_root=output_root,
            node_local_spool_root=output_root,
            image_storage_options=image_storage_options,
            image_columns=image_columns,
            payload_window_bytes=payload_window_bytes,
            bucket_rows=bucket_rows,
            payload_spool_sync_mode="fsync",
            estimated_payload_bytes_per_row=estimated_payload_bytes_per_row,
            fetch_batch_size=fetch_batch_size,
            max_pending=max_pending,
            payload_actor_cpus=payload_actor_cpus,
            payload_patch_workers=payload_overlay_workers,
            document_storage_options=document_storage_options,
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Open one persistent pinned image reader per actor."""

        super().setup(worker_metadata)

    def _payload_schema(self, image: object) -> pa.Schema:
        image_schema = getattr(image, "schema", None)
        if not isinstance(image_schema, pa.Schema):
            msg = "image dataset schema is invalid"
            raise TypeError(msg)
        missing = [name for name in self.image_columns if image_schema.get_field_index(name) < 0]
        if missing:
            msg = f"image dataset is missing payload columns: {missing}"
            raise ValueError(msg)
        return pa.schema(
            [
                pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
                pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
                pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
                *(image_schema.field(source) for source in self.image_columns),
            ]
        )

    def _overlay_identity(
        self,
        plan: pa.Table,
        manifest: Mapping[str, object],
        *,
        fragment_rows: int,
        payload_schema: pa.Schema,
        coordinate_manifest_path: Path,
    ) -> LancePayloadOverlayIdentity:
        plan_identity = _plan_identity(manifest)
        coordinates = _manifest_mapping(manifest.get("coordinates"), "coordinates")
        coordinate_rows = _manifest_integer(coordinates.get("rows"), "coordinates.rows")
        logical_rows = _manifest_integer(
            coordinates.get("non_null_stable_row_ids"),
            "coordinates.non_null_stable_row_ids",
        )
        unique_rows = _manifest_integer(
            coordinates.get("unique_stable_row_ids"),
            "coordinates.unique_stable_row_ids",
        )
        null_rows = _manifest_integer(coordinates.get("null_stable_row_ids"), "coordinates.null_stable_row_ids")
        if (
            coordinate_rows != plan.num_rows
            or logical_rows != plan.num_rows - plan[STABLE_ROW_ID].null_count
            or null_rows != plan[STABLE_ROW_ID].null_count
            or unique_rows != int(pc.count_distinct(plan[STABLE_ROW_ID], mode="only_valid").as_py())
        ):
            msg = "coordinate plan manifest counts do not reconcile with Arrow coordinates"
            raise ValueError(msg)
        sidecar_manifest_sha256 = manifest.get("sidecar_manifest_sha256")
        if not isinstance(sidecar_manifest_sha256, str):
            msg = "coordinate plan sidecar manifest digest is invalid"
            raise TypeError(msg)
        config_sha256 = lance_payload_overlay_config_sha256(
            self.image_columns,
            payload_schema=payload_schema,
            payload_window_bytes=self.payload_window_bytes,
            bucket_rows=self.bucket_rows,
        )
        return LancePayloadOverlayIdentity(
            document_uri=plan_identity.document_uri,
            document_version=plan_identity.document_version,
            image_uri=plan_identity.image_uri,
            image_version=plan_identity.image_version,
            fragment_id=plan_identity.fragment_id,
            coordinate_plan_sha256=plan_identity.coordinate_sha256,
            coordinate_manifest_sha256=_file_sha256(coordinate_manifest_path),
            payload_coordinate_sha256=payload_coordinate_sha256(plan),
            sidecar_manifest_sha256=sidecar_manifest_sha256,
            fragment_manifest_sha256=plan_identity.fragment_manifest_sha256,
            overlay_config_sha256=config_sha256,
            expected_document_rows=fragment_rows,
            expected_coordinate_rows=coordinate_rows,
            expected_logical_rows=logical_rows,
            expected_unique_rows=unique_rows,
            expected_null_rows=null_rows,
        )

    def _producer_metrics(  # noqa: PLR0913
        self,
        materialize_metrics: Mapping[str, int | float | bool],
        *,
        elapsed_seconds: float,
        plan: pa.Table,
        task: LanceCoordinatePlanTask,
        payload_files: int,
        payload_arrow_bytes: int,
        peak_active_bytes: int,
        oversized_rows: int,
    ) -> dict[str, int | float | bool]:
        read_iops = int(materialize_metrics["lance_read_iops"])
        read_bytes = int(materialize_metrics["lance_read_bytes"])
        actual_payload_bytes = int(materialize_metrics["actual_payload_bytes"])
        unique_rows = int(materialize_metrics["unique_rows"])
        logical_rows = int(materialize_metrics["logical_rows"])
        private_take_envelope = float(materialize_metrics["private_take_execution_envelope_seconds"])
        return {
            **materialize_metrics,
            "payload_materialize_seconds": elapsed_seconds,
            "payload_unique_images_per_second": unique_rows / elapsed_seconds if elapsed_seconds else 0.0,
            "payload_logical_images_per_second": logical_rows / elapsed_seconds if elapsed_seconds else 0.0,
            "average_physical_read_bytes": read_bytes / read_iops if read_iops else 0.0,
            "physical_reads_per_unique_payload": read_iops / unique_rows if unique_rows else 0.0,
            "physical_reads_per_logical_payload": read_iops / logical_rows if logical_rows else 0.0,
            "physical_read_operations_per_private_take_envelope_second": (
                read_iops / private_take_envelope if private_take_envelope else 0.0
            ),
            "read_amplification": read_bytes / actual_payload_bytes if actual_payload_bytes else 0.0,
            "coordinate_plan_arrow_bytes": plan.nbytes,
            "coordinate_plan_file_bytes": Path(task.data).stat().st_size,
            "coordinate_manifest_file_bytes": Path(task.manifest_path).stat().st_size,
            "payload_spool_files": payload_files,
            "payload_spool_arrow_bytes": payload_arrow_bytes,
            "payload_spool_peak_active_bytes": peak_active_bytes,
            "payload_spool_oversized_rows": oversized_rows,
            "payload_window_bytes": self.payload_window_bytes,
            "bucket_rows": self.bucket_rows,
            "estimated_inflight_payload_bytes": (
                self.max_pending * self.fetch_batch_size * self.estimated_payload_bytes_per_row
            ),
            "estimated_payload_actor_reservation_bytes": self.estimated_payload_actor_reservation_bytes,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }

    @staticmethod
    def _task_metadata(task: LanceCoordinatePlanTask) -> dict[str, object]:
        metadata = dict(task._metadata)
        metadata["coordinate_plan_source_files"] = metadata.get("source_files", [])
        return metadata

    def process(self, task: LanceCoordinatePlanTask) -> FileGroupTask:
        image, payload_streamer, image_fragment_digest = self._require_setup()
        plan, manifest = load_coordinate_plan(task)
        plan_identity = _plan_identity(manifest)
        if plan_identity.image_uri != self.image_uri or plan_identity.image_version != self.image_version:
            msg = "coordinate plan image identity does not match stage configuration"
            raise ValueError(msg)
        if plan_identity.fragment_manifest_sha256 != image_fragment_digest:
            msg = "coordinate plan image fragment-manifest digest does not match the opened image dataset"
            raise ValueError(msg)
        document = self._document(plan_identity)
        fragment = self._fragment(document, plan_identity.fragment_id)
        fragment_rows = int(fragment.physical_rows)
        self._validate_plan(plan, plan_identity, fragment_rows)
        payload_schema = self._payload_schema(image)
        identity = self._overlay_identity(
            plan,
            manifest,
            fragment_rows=fragment_rows,
            payload_schema=payload_schema,
            coordinate_manifest_path=Path(task.manifest_path),
        )
        artifact_root = lance_payload_overlay_root(self.output_root, identity)
        lock_descriptor = _acquire_artifact_lock(artifact_root)
        attempt_root: Path | None = None
        try:
            _remove_orphan_attempts(artifact_root)
            if artifact_root.exists() or artifact_root.is_symlink():
                artifact = validate_lance_payload_overlay(
                    artifact_root,
                    expected_identity=identity,
                    expected_image_columns=self.image_columns,
                    verify_payload=True,
                )
                output = lance_payload_overlay_task(artifact, metadata=self._task_metadata(task))
                output._stage_perf = task._stage_perf
                return output

            attempt_root = self.output_root / f".{artifact_root.name}.{uuid.uuid4().hex}.tmp"
            attempt_root.mkdir(exist_ok=False)
            spool = PayloadSpool(
                attempt_root / "payload",
                payload_schema,
                target_bytes=self.payload_window_bytes,
                bucket_rows=self.bucket_rows,
                stable_id_column=STABLE_ROW_ID,
                document_position_column=DOCUMENT_POSITION,
                sync_mode="fsync",
            )
            started = time.perf_counter()
            materialize_metrics = materialize_lance_payload_to_spool(
                payload_streamer,
                plan,
                tuple(self.image_columns),
                spool,
            )
            elapsed_seconds = time.perf_counter() - started
            payload_manifest = spool.finish()
            if int(materialize_metrics["logical_rows"]) != identity.expected_logical_rows:
                msg = "payload materializer logical rows do not match overlay identity"
                raise RuntimeError(msg)
            if int(materialize_metrics["unique_rows"]) != identity.expected_unique_rows:
                msg = "payload materializer unique rows do not match overlay identity"
                raise RuntimeError(msg)
            metrics = self._producer_metrics(
                materialize_metrics,
                elapsed_seconds=elapsed_seconds,
                plan=plan,
                task=task,
                payload_files=len(payload_manifest.files),
                payload_arrow_bytes=payload_manifest.total_arrow_nbytes,
                peak_active_bytes=payload_manifest.peak_active_bytes,
                oversized_rows=len(payload_manifest.oversized_rows),
            )
            artifact = publish_lance_payload_overlay(
                attempt_root,
                artifact_root,
                identity=identity,
                image_columns=self.image_columns,
                payload=payload_manifest,
                producer_metrics=metrics,
            )
            attempt_root = None
            output = lance_payload_overlay_task(artifact, metadata=self._task_metadata(task))
            output._stage_perf = task._stage_perf
            return output
        finally:
            if attempt_root is not None and attempt_root.exists() and not attempt_root.is_symlink():
                shutil.rmtree(attempt_root)
            _release_artifact_lock(lock_descriptor)
