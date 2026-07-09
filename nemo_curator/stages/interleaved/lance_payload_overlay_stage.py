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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from nemo_curator.stages.interleaved.lance_payload_materialize import (
    estimate_grouped_coordinate_workspace_bytes,
    materialize_lance_payload_group_to_spools,
)
from nemo_curator.stages.interleaved.lance_payload_overlay import (
    LancePayloadOverlayArtifact,
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
    from nemo_curator.stages.interleaved.lance_payload_materialize import _StableIdPayloadStreamer
    from nemo_curator.tasks import FileGroupTask


_MAX_COORDINATE_PLAN_BATCH_SIZE = 64
_COORDINATE_WINDOW_PROFILES = {
    "256MiB": 256 * 1024**2,
    "1GiB": 1024**3,
    "4GiB": 4 * 1024**3,
}


def _resolve_coordinate_window_bytes(value: PayloadWindowBytes | int) -> int:
    if isinstance(value, str):
        resolved = _COORDINATE_WINDOW_PROFILES.get(value)
        if resolved is None:
            msg = f"coordinate_window_bytes must be one of {tuple(_COORDINATE_WINDOW_PROFILES)}"
            raise ValueError(msg)
        return resolved
    if isinstance(value, bool) or value not in _COORDINATE_WINDOW_PROFILES.values():
        msg = "coordinate_window_bytes must be 256MiB, 1GiB, or 4GiB"
        raise ValueError(msg)
    return value


@dataclass
class _OverlayBatchMember:
    """Identity-only batch state; Arrow plans live only for one active group."""

    input_index: int
    task: LanceCoordinatePlanTask
    identity: LancePayloadOverlayIdentity
    source_identity_sha256: str
    artifact_root: Path
    singleton_workspace_estimate: int
    attempt_root: Path | None = None


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
    batch_size = _MAX_COORDINATE_PLAN_BATCH_SIZE

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_uri: str,
        image_version: int,
        output_root: str,
        image_storage_options: Mapping[str, str] | None = None,
        image_columns: Mapping[str, str] | None = None,
        payload_window_bytes: PayloadWindowBytes | int = "1GiB",
        coordinate_window_bytes: PayloadWindowBytes | int = "4GiB",
        bucket_rows: int = 131_072,
        estimated_payload_bytes_per_row: int = 128 * 1024,
        fetch_batch_size: int = 1024,
        max_pending: int = 16,
        payload_actor_cpus: int = 64,
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
        self.coordinate_window_bytes = _resolve_coordinate_window_bytes(coordinate_window_bytes)

    @property
    def estimated_retained_payload_bytes(self) -> int:
        """Return the reader's full retained-batch payload estimate."""

        retained_batches = 2 * self.max_pending + 1
        return retained_batches * self.fetch_batch_size * self.estimated_payload_bytes_per_row

    @property
    def estimated_payload_actor_reservation_bytes(self) -> int:
        """Return the retained payload, shared spool, and coordinate estimate."""

        return self.estimated_retained_payload_bytes + self.payload_window_bytes + self.coordinate_window_bytes

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
        plan: pa.Table,
        task: LanceCoordinatePlanTask,
        payload_files: int,
        payload_arrow_bytes: int,
        peak_active_bytes: int,
        oversized_rows: int,
    ) -> dict[str, int | float | bool]:
        return {
            **materialize_metrics,
            "coordinate_plan_arrow_bytes": plan.nbytes,
            "coordinate_plan_file_bytes": Path(task.data).stat().st_size,
            "coordinate_manifest_file_bytes": Path(task.manifest_path).stat().st_size,
            "payload_spool_files": payload_files,
            "payload_spool_arrow_bytes": payload_arrow_bytes,
            "payload_spool_peak_active_bytes": peak_active_bytes,
            "payload_spool_oversized_rows": oversized_rows,
            "payload_window_bytes": self.payload_window_bytes,
            "coordinate_window_bytes": self.coordinate_window_bytes,
            "bucket_rows": self.bucket_rows,
            "estimated_inflight_payload_bytes": (
                self.max_pending * self.fetch_batch_size * self.estimated_payload_bytes_per_row
            ),
            "estimated_retained_payload_bytes": self.estimated_retained_payload_bytes,
            "estimated_payload_actor_reservation_bytes": self.estimated_payload_actor_reservation_bytes,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }

    @staticmethod
    def _task_metadata(task: LanceCoordinatePlanTask) -> dict[str, object]:
        metadata = dict(task._metadata)
        metadata["coordinate_plan_source_files"] = metadata.get("source_files", [])
        return metadata

    def _validated_plan(
        self,
        task: LanceCoordinatePlanTask,
        *,
        image_fragment_digest: str,
        payload_schema: pa.Schema,
    ) -> tuple[pa.Table, LancePayloadOverlayIdentity, int]:
        """Load and fully validate one plan without retaining it on the member."""

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
        identity = self._overlay_identity(
            plan,
            manifest,
            fragment_rows=fragment_rows,
            payload_schema=payload_schema,
            coordinate_manifest_path=Path(task.manifest_path),
        )
        singleton_estimate = estimate_grouped_coordinate_workspace_bytes((plan,))
        if isinstance(singleton_estimate, bool) or not isinstance(singleton_estimate, int) or singleton_estimate < 0:
            msg = "coordinate workspace estimator returned an invalid singleton estimate"
            raise RuntimeError(msg)
        return plan, identity, singleton_estimate

    def _batch_member(
        self,
        input_index: int,
        task: LanceCoordinatePlanTask,
        *,
        image_fragment_digest: str,
        payload_schema: pa.Schema,
    ) -> _OverlayBatchMember:
        plan, identity, singleton_estimate = self._validated_plan(
            task,
            image_fragment_digest=image_fragment_digest,
            payload_schema=payload_schema,
        )
        try:
            artifact_root = lance_payload_overlay_root(self.output_root, identity)
            return _OverlayBatchMember(
                input_index=input_index,
                task=task,
                identity=identity,
                source_identity_sha256=lance_payload_overlay_source_identity_sha256(identity),
                artifact_root=artifact_root,
                singleton_workspace_estimate=singleton_estimate,
            )
        finally:
            del plan

    def _reload_member_plan(
        self,
        member: _OverlayBatchMember,
        *,
        image_fragment_digest: str,
        payload_schema: pa.Schema,
    ) -> pa.Table:
        plan, identity, singleton_estimate = self._validated_plan(
            member.task,
            image_fragment_digest=image_fragment_digest,
            payload_schema=payload_schema,
        )
        if identity != member.identity or singleton_estimate != member.singleton_workspace_estimate:
            del plan
            msg = "coordinate plan identity or workspace estimate changed after batch prevalidation"
            raise ValueError(msg)
        return plan

    def _output_task(
        self,
        artifact: LancePayloadOverlayArtifact,
        source: LanceCoordinatePlanTask,
    ) -> FileGroupTask:
        output = lance_payload_overlay_task(artifact, metadata=self._task_metadata(source))
        output._stage_perf = source._stage_perf
        return output

    def _coordinate_groups(
        self,
        pending: Sequence[_OverlayBatchMember],
    ) -> tuple[tuple[_OverlayBatchMember, ...], ...]:
        groups: list[tuple[_OverlayBatchMember, ...]] = []
        current: list[_OverlayBatchMember] = []
        current_bytes = 0
        for member in pending:
            singleton_bytes = member.singleton_workspace_estimate
            if singleton_bytes > self.coordinate_window_bytes:
                msg = (
                    f"coordinate plan workspace requires {singleton_bytes} bytes; "
                    f"coordinate_window_bytes is {self.coordinate_window_bytes}"
                )
                raise ValueError(msg)
            if current and current_bytes + singleton_bytes > self.coordinate_window_bytes:
                groups.append(tuple(current))
                current = []
                current_bytes = 0
            current.append(member)
            current_bytes += singleton_bytes
        if current:
            groups.append(tuple(current))
        return tuple(groups)

    @staticmethod
    def _group_fetch_metrics(
        materialize_metrics: Mapping[str, int | float | bool],
        *,
        elapsed_seconds: float,
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
        }

    def _materialize_group(
        self,
        group: tuple[_OverlayBatchMember, ...],
        *,
        payload_streamer: _StableIdPayloadStreamer,
        image_fragment_digest: str,
        payload_schema: pa.Schema,
    ) -> list[tuple[int, FileGroupTask]]:
        plans: list[pa.Table] = []
        spools: list[PayloadSpool] = []
        try:
            for member in group:
                plans.append(
                    self._reload_member_plan(
                        member,
                        image_fragment_digest=image_fragment_digest,
                        payload_schema=payload_schema,
                    )
                )

            for member in group:
                member.attempt_root = self.output_root / (f".{member.artifact_root.name}.{uuid.uuid4().hex}.tmp")
                member.attempt_root.mkdir(exist_ok=False)
                spools.append(
                    PayloadSpool(
                        member.attempt_root / "payload",
                        payload_schema,
                        target_bytes=self.payload_window_bytes,
                        bucket_rows=self.bucket_rows,
                        stable_id_column=STABLE_ROW_ID,
                        document_position_column=DOCUMENT_POSITION,
                        sync_mode="fsync",
                    )
                )

            started = time.perf_counter()
            grouped_result = materialize_lance_payload_group_to_spools(
                payload_streamer,
                tuple(plans),
                tuple(self.image_columns),
                tuple(spools),
                shared_spool_budget_bytes=self.payload_window_bytes,
                max_coordinate_workspace_bytes=self.coordinate_window_bytes,
            )
            elapsed_seconds = time.perf_counter() - started
            fetch_metrics = self._group_fetch_metrics(
                grouped_result.fetch_metrics,
                elapsed_seconds=elapsed_seconds,
            )
            fetch_group = lance_payload_fetch_group(
                tuple(member.identity for member in group),
                fetch_metrics,
            )

            payload_manifests = []
            producer_metrics = []
            for index, member in enumerate(group):
                payload_manifest = spools[index].finish()
                plan_metrics = grouped_result.plan_metrics[index].as_dict()
                if int(plan_metrics["logical_rows"]) != member.identity.expected_logical_rows:
                    msg = "payload materializer logical rows do not match overlay identity"
                    raise RuntimeError(msg)
                if int(plan_metrics["unique_rows"]) != member.identity.expected_unique_rows:
                    msg = "payload materializer unique rows do not match overlay identity"
                    raise RuntimeError(msg)
                metrics = self._producer_metrics(
                    plan_metrics,
                    plan=plans[index],
                    task=member.task,
                    payload_files=len(payload_manifest.files),
                    payload_arrow_bytes=payload_manifest.total_arrow_nbytes,
                    peak_active_bytes=payload_manifest.peak_active_bytes,
                    oversized_rows=len(payload_manifest.oversized_rows),
                )
                payload_manifests.append(payload_manifest)
                producer_metrics.append(metrics)

            plans.clear()
            produced: list[tuple[int, FileGroupTask]] = []
            for member, payload_manifest, metrics in zip(
                group,
                payload_manifests,
                producer_metrics,
                strict=True,
            ):
                if member.attempt_root is None:  # pragma: no cover - assigned above
                    msg = "payload overlay subgroup lost its attempt directory"
                    raise RuntimeError(msg)
                artifact = publish_lance_payload_overlay(
                    member.attempt_root,
                    member.artifact_root,
                    identity=member.identity,
                    image_columns=self.image_columns,
                    payload=payload_manifest,
                    producer_metrics=metrics,
                    fetch_group=fetch_group,
                )
                member.attempt_root = None
                produced.append((member.input_index, self._output_task(artifact, member.task)))
            return produced
        finally:
            plans.clear()

    def process(self, task: LanceCoordinatePlanTask) -> FileGroupTask:
        """Process one plan through the exact grouped implementation."""

        return self.process_batch([task])[0]

    def process_batch(  # noqa: C901, PLR0912
        self,
        tasks: list[LanceCoordinatePlanTask],
    ) -> list[FileGroupTask]:
        """Fetch up to 64 plans through deterministic shared stable-ID queues."""

        if not tasks:
            msg = "payload overlay process_batch requires at least one coordinate plan"
            raise ValueError(msg)
        if len(tasks) > _MAX_COORDINATE_PLAN_BATCH_SIZE:
            msg = f"payload overlay process_batch accepts at most {_MAX_COORDINATE_PLAN_BATCH_SIZE} plans"
            raise ValueError(msg)

        image, payload_streamer, image_fragment_digest = self._require_setup()
        payload_schema = self._payload_schema(image)
        members = [
            self._batch_member(
                input_index,
                task,
                image_fragment_digest=image_fragment_digest,
                payload_schema=payload_schema,
            )
            for input_index, task in enumerate(tasks)
        ]
        if len({member.source_identity_sha256 for member in members}) != len(members) or len(
            {member.artifact_root for member in members}
        ) != len(members):
            msg = "payload overlay batch contains duplicate semantic artifact identities"
            raise ValueError(msg)

        semantic_members = sorted(
            members,
            key=lambda member: (member.source_identity_sha256, str(member.artifact_root)),
        )
        outputs: list[FileGroupTask | None] = [None] * len(members)
        lock_descriptors: list[int] = []
        try:
            for member in semantic_members:
                lock_descriptors.append(_acquire_artifact_lock(member.artifact_root))
            for member in semantic_members:
                _remove_orphan_attempts(member.artifact_root)

            pending: list[_OverlayBatchMember] = []
            for member in semantic_members:
                if not (member.artifact_root.exists() or member.artifact_root.is_symlink()):
                    pending.append(member)
                    continue
                artifact = validate_lance_payload_overlay(
                    member.artifact_root,
                    expected_identity=member.identity,
                    expected_image_columns=self.image_columns,
                    verify_payload=True,
                )
                outputs[member.input_index] = self._output_task(artifact, member.task)

            for group in self._coordinate_groups(pending):
                for input_index, output in self._materialize_group(
                    group,
                    payload_streamer=payload_streamer,
                    image_fragment_digest=image_fragment_digest,
                    payload_schema=payload_schema,
                ):
                    outputs[input_index] = output

            if any(output is None for output in outputs):
                msg = "payload overlay batch did not produce one positional output per input"
                raise RuntimeError(msg)
            return [output for output in outputs if output is not None]
        finally:
            for member in semantic_members:
                if (
                    member.attempt_root is not None
                    and member.attempt_root.exists()
                    and not member.attempt_root.is_symlink()
                ):
                    shutil.rmtree(member.attempt_root)
            for descriptor in reversed(lock_descriptors):
                _release_artifact_lock(descriptor)
