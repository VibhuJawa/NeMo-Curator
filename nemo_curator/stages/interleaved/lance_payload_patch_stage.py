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

"""Ordinary Curator stage that materializes coordinate plans as patch files."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import shutil
import stat
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.gpu_key_lookup import _stable_global_ordinal_manifest_sha256
from nemo_curator.stages.interleaved.lance import _validate_stable_global_ordinal_manifest
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    LanceCoordinatePlanTask,
    load_coordinate_plan,
)
from nemo_curator.stages.interleaved.lance_document_patch import (
    LANCE_ROWADDR,
    SAMPLE_ID,
    apply_payload_part,
    split_interleaved_by_actual_bytes,
)
from nemo_curator.stages.interleaved.lance_patch_artifact import (
    LancePatchArtifactIdentity,
    LancePatchArtifactWriter,
)
from nemo_curator.stages.interleaved.lance_payload_materialize import materialize_lance_payload_to_spool
from nemo_curator.stages.interleaved.lance_payload_spool import (
    PayloadSpool,
    PayloadSpoolReader,
    PayloadSpoolSyncMode,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.atomic_io import fsync_directory
from nemo_curator.utils.uri import validate_credential_free_uri_identity

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.interleaved.lance_payload_materialize import _StableIdPayloadStreamer


ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]
PayloadWindowBytes = Literal["256MiB", "1GiB", "4GiB"]

_PAYLOAD_WINDOW_PROFILES = {
    "256MiB": 256 * 1024**2,
    "1GiB": 1024**3,
    "4GiB": 4 * 1024**3,
}
_ROW_OFFSET_MASK = (1 << 32) - 1


class _ScannerLike(Protocol):
    def to_batches(self) -> Iterator[pa.RecordBatch]: ...


class _FragmentLike(Protocol):
    fragment_id: int
    physical_rows: int
    num_deletions: int
    metadata: object

    def deletion_file(self) -> object | None: ...


class _DatasetLike(Protocol):
    uri: str
    schema: pa.Schema
    version: int
    has_stable_row_ids: bool

    def get_fragments(self) -> Sequence[_FragmentLike]: ...

    def get_fragment(self, fragment_id: int) -> _FragmentLike | None: ...

    def count_rows(self) -> int: ...

    def scanner(self, **kwargs: object) -> _ScannerLike: ...

    def io_stats_incremental(self) -> object: ...


@dataclass(frozen=True)
class _PlanIdentity:
    document_uri: str
    document_version: int
    image_uri: str
    image_version: int
    fragment_id: int
    coordinate_sha256: str
    fragment_manifest_sha256: str


@dataclass
class _PatchMetrics:
    parts: int = 0
    rows: int = 0
    arrow_bytes: int = 0
    file_bytes: int = 0
    oversized_samples: int = 0


class _SampleStitcher:
    """Hold only the trailing sample until its next boundary is observed."""

    def __init__(self) -> None:
        self._pending: pa.Table | None = None
        self._completed_sample_ids: set[str] = set()

    @property
    def completed_samples(self) -> int:
        return len(self._completed_sample_ids)

    def _record_completed(self, table: pa.Table) -> None:
        sample_ids = table[SAMPLE_ID].combine_chunks()
        if len(sample_ids) == 0:
            return
        changes = pc.indices_nonzero(
            pc.not_equal(
                sample_ids.slice(1),
                sample_ids.slice(0, len(sample_ids) - 1),
            )
        )
        starts = [0, *(int(changes[index].as_py()) + 1 for index in range(len(changes)))]
        completed = [str(sample_ids[index].as_py()) for index in starts]
        if len(set(completed)) != len(completed) or any(value in self._completed_sample_ids for value in completed):
            msg = "Each sample_id must occupy exactly one contiguous document row range"
            raise ValueError(msg)
        self._completed_sample_ids.update(completed)

    def push(self, table: pa.Table) -> pa.Table | None:
        if table.num_rows == 0:
            return None
        combined = pa.concat_tables([self._pending, table]) if self._pending is not None else table
        sample_ids = combined[SAMPLE_ID].combine_chunks()
        changes = pc.indices_nonzero(
            pc.not_equal(
                sample_ids.slice(1),
                sample_ids.slice(0, len(sample_ids) - 1),
            )
        )
        run_starts = [0, *(int(changes[index].as_py()) + 1 for index in range(len(changes)))]
        run_ids = [str(sample_ids[index].as_py()) for index in run_starts]
        if len(set(run_ids)) != len(run_ids) or any(value in self._completed_sample_ids for value in run_ids):
            msg = "Each sample_id must occupy exactly one contiguous document row range"
            raise ValueError(msg)
        if len(changes) == 0:
            self._pending = combined
            return None
        trailing_start = int(changes[len(changes) - 1].as_py()) + 1
        complete = combined.slice(0, trailing_start)
        self._pending = combined.slice(trailing_start)
        self._record_completed(complete)
        return complete

    def finish(self) -> pa.Table | None:
        pending = self._pending
        self._pending = None
        if pending is not None:
            self._record_completed(pending)
        return pending


def _open_lance_dataset(
    uri: str,
    version: int,
    storage_options: Mapping[str, str],
) -> _DatasetLike:
    import lance

    dataset = lance.dataset(
        uri,
        version=version,
        storage_options=dict(storage_options) or None,
    )
    if dataset.version != version:
        msg = f"Lance dataset resolved version {dataset.version}; expected {version}"
        raise RuntimeError(msg)
    return dataset


def _create_stable_id_payload_streamer(  # noqa: PLR0913
    dataset: _DatasetLike,
    *,
    dataset_uri: str,
    dataset_version: int,
    expected_rows: int,
    source_columns: Sequence[str],
    storage_options: Mapping[str, str],
    fetch_batch_size: int,
    max_pending: int,
) -> _StableIdPayloadStreamer:
    """Create the sidecar-free Lance-Ray reader with identity projection."""

    from lance_ray import LanceStableIdPayloadConfig, LanceStableIdPayloadStreamer

    config = LanceStableIdPayloadConfig(
        dataset_uri=dataset_uri,
        dataset_version=dataset_version,
        expected_rows=expected_rows,
        columns={name: name for name in source_columns},
        dataset_storage_options=dict(storage_options),
        fetch_batch_size=fetch_batch_size,
        io_threads=max_pending,
        max_pending_fetch_batches=max_pending,
    )
    return LanceStableIdPayloadStreamer(
        config,
        dataset=dataset,
        stable_row_id_output_column=STABLE_ROW_ID,
    )


def _resolve_payload_window_bytes(value: PayloadWindowBytes | int) -> int:
    if isinstance(value, str):
        resolved = _PAYLOAD_WINDOW_PROFILES.get(value)
        if resolved is None:
            msg = f"payload_window_bytes must be one of {tuple(_PAYLOAD_WINDOW_PROFILES)}"
            raise ValueError(msg)
        return resolved
    if isinstance(value, bool) or value not in _PAYLOAD_WINDOW_PROFILES.values():
        msg = "payload_window_bytes must be 256MiB, 1GiB, or 4GiB"
        raise ValueError(msg)
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _manifest_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"coordinate plan manifest {name} section is invalid"
        raise TypeError(msg)
    return value


def _plan_identity(manifest: Mapping[str, object]) -> _PlanIdentity:
    document = _manifest_mapping(manifest.get("document"), "document")
    image = _manifest_mapping(manifest.get("image"), "image")
    coordinates = _manifest_mapping(manifest.get("coordinates"), "coordinates")
    try:
        return _PlanIdentity(
            document_uri=str(document["uri"]),
            document_version=int(document["version"]),
            image_uri=str(image["uri"]),
            image_version=int(image["version"]),
            fragment_id=int(document["fragment_id"]),
            coordinate_sha256=str(coordinates["canonical_ipc_sha256"]),
            fragment_manifest_sha256=str(manifest["fragment_manifest_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = "coordinate plan manifest identity is invalid"
        raise ValueError(msg) from exc


def _patch_config_sha256(
    image_columns: Mapping[str, str],
    existing_column_policy: ExistingColumnPolicy,
) -> str:
    material = json.dumps(
        {
            "schema_version": 1,
            "image_columns": sorted(image_columns.items()),
            "existing_column_policy": existing_column_policy,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _artifact_root(output_root: Path, identity: _PlanIdentity, patch_config_sha256: str) -> Path:
    material = json.dumps(
        {
            "document_uri": identity.document_uri,
            "document_version": identity.document_version,
            "image_uri": identity.image_uri,
            "image_version": identity.image_version,
            "fragment_id": identity.fragment_id,
            "coordinate_sha256": identity.coordinate_sha256,
            "patch_config_sha256": patch_config_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(material).hexdigest()[:16]
    return output_root / f"fragment-{identity.fragment_id:08d}-{digest}"


def _validate_lock_descriptor(descriptor: int, lock_path: Path) -> None:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        msg = f"Patch artifact lock is not a regular file: {lock_path}"
        raise ValueError(msg)


def _acquire_artifact_lock(artifact_root: Path) -> int:
    lock_path = artifact_root.parent / f".{artifact_root.name}.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        _validate_lock_descriptor(descriptor, lock_path)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _release_artifact_lock(descriptor: int) -> None:
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def _remove_orphan_attempts(artifact_root: Path) -> None:
    pattern = f".{artifact_root.name}.*.tmp"
    for attempt in artifact_root.parent.glob(pattern):
        if attempt.is_symlink() or not attempt.is_dir():
            msg = f"Patch artifact attempt path is not a regular directory: {attempt}"
            raise ValueError(msg)
        shutil.rmtree(attempt)


def _remove_orphan_spools(spool_root: Path, artifact_root: Path) -> None:
    pattern = f".{artifact_root.name}.*.payload-spool"
    for attempt in spool_root.glob(pattern):
        if attempt.is_symlink() or not attempt.is_dir():
            msg = f"Payload spool attempt path is not a regular directory: {attempt}"
            raise ValueError(msg)
        shutil.rmtree(attempt)


def _require_scratch_capacity(root: Path, logical_rows: int, estimated_bytes_per_row: int) -> tuple[int, int]:
    estimated_bytes = logical_rows * estimated_bytes_per_row
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < estimated_bytes:
        msg = f"node-local payload scratch has {free_bytes} free bytes; estimated requirement is {estimated_bytes}"
        raise OSError(msg)
    return estimated_bytes, free_bytes


class LanceCoordinatePayloadPatchStage(ProcessingStage[LanceCoordinatePlanTask, FileGroupTask]):
    """Fetch one coordinate plan and stream a durable patched document fragment.

    ``payload_actor_cpus`` is a Ray scheduling admission control: on a node with
    64 advertised CPUs, the default reserves enough resources for at most eight
    concurrent patch actors. ``payload_patch_workers`` can additionally cap the
    actor pool across the cluster. ``estimated_payload_actor_reservation_bytes``
    reports the configured in-flight fetch estimate plus the normal spool
    buffer. It is not a hard actual-byte bound for variable-size payloads.
    """

    name = "lance_coordinate_payload_patch"
    resources = Resources(cpus=8.0)
    batch_size = 1

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_uri: str,
        image_version: int,
        output_root: str,
        node_local_spool_root: str,
        image_storage_options: Mapping[str, str] | None = None,
        image_columns: Mapping[str, str] | None = None,
        payload_window_bytes: PayloadWindowBytes | int = "1GiB",
        bucket_rows: int = 131_072,
        payload_spool_sync_mode: PayloadSpoolSyncMode = "attempt_local",
        estimated_payload_bytes_per_row: int = 128 * 1024,
        fetch_batch_size: int = 1024,
        max_pending: int = 16,
        payload_actor_cpus: int = 8,
        payload_patch_workers: int | None = None,
        document_projection: Sequence[str] | None = None,
        document_storage_options: Mapping[str, str] | None = None,
        existing_column_policy: ExistingColumnPolicy = "fill_null",
    ) -> None:
        super().__init__()
        self.image_uri = image_uri
        self.image_version = image_version
        self.output_root = Path(output_root)
        self.node_local_spool_root = Path(node_local_spool_root)
        self.image_storage_options = dict(image_storage_options or {})
        self.image_columns = {"image": "binary_content"} if image_columns is None else dict(image_columns)
        self.payload_window_bytes = _resolve_payload_window_bytes(payload_window_bytes)
        self.bucket_rows = _positive_integer(bucket_rows, "bucket_rows")
        self.payload_spool_sync_mode = payload_spool_sync_mode
        self.estimated_payload_bytes_per_row = _positive_integer(
            estimated_payload_bytes_per_row,
            "estimated_payload_bytes_per_row",
        )
        self.fetch_batch_size = _positive_integer(fetch_batch_size, "fetch_batch_size")
        self.max_pending = _positive_integer(max_pending, "max_pending")
        self.payload_actor_cpus = _positive_integer(payload_actor_cpus, "payload_actor_cpus")
        self.payload_patch_workers = (
            None
            if payload_patch_workers is None
            else _positive_integer(payload_patch_workers, "payload_patch_workers")
        )
        self.resources = Resources(cpus=float(self.payload_actor_cpus))
        self.document_projection = None if document_projection is None else tuple(document_projection)
        self.document_storage_options = dict(document_storage_options or {})
        self.existing_column_policy = existing_column_policy
        self._image_dataset: _DatasetLike | None = None
        self._image_fragment_manifest_sha256: str | None = None
        self._document_dataset: _DatasetLike | None = None
        self._document_identity: tuple[str, int] | None = None
        self._payload_streamer: _StableIdPayloadStreamer | None = None
        self._validate_config()

    @property
    def estimated_payload_actor_reservation_bytes(self) -> int:
        """Return the configured fetch-plus-spool estimate for one actor.

        Variable-size payloads can exceed the row-size estimate, and one
        oversized row may exceed the spool target, so this is deliberately not
        presented as a hard memory limit.
        """

        estimated_fetch_bytes = self.max_pending * self.fetch_batch_size * self.estimated_payload_bytes_per_row
        return estimated_fetch_bytes + self.payload_window_bytes

    def num_workers(self) -> int | None:
        """Return the optional cluster-wide patch actor cap."""

        return self.payload_patch_workers

    def _validate_config(self) -> None:  # noqa: C901
        if not self.image_uri:
            msg = "image_uri must not be empty"
            raise ValueError(msg)
        validate_credential_free_uri_identity(self.image_uri, "image Lance URI")
        _positive_integer(self.image_version, "image_version")
        if not self.output_root.is_absolute() or not self.node_local_spool_root.is_absolute():
            msg = "output_root and node_local_spool_root must be absolute filesystem paths"
            raise ValueError(msg)
        if not self.image_columns:
            msg = "image_columns must not be empty"
            raise ValueError(msg)
        if any(not source or not destination for source, destination in self.image_columns.items()):
            msg = "image_columns must contain non-empty source and destination names"
            raise ValueError(msg)
        if len(set(self.image_columns.values())) != len(self.image_columns):
            msg = "image_columns destination names must be unique"
            raise ValueError(msg)
        reserved = {LANCE_ROWADDR, DOCUMENT_POSITION, SAMPLE_ID}
        collisions = reserved & set(self.image_columns.values())
        if collisions:
            msg = f"image_columns destinations collide with reserved columns: {sorted(collisions)}"
            raise ValueError(msg)
        if self.existing_column_policy not in {"error", "fill_null", "overwrite"}:
            msg = f"Unsupported existing_column_policy: {self.existing_column_policy!r}"
            raise ValueError(msg)
        if not isinstance(self.payload_spool_sync_mode, str) or self.payload_spool_sync_mode not in {
            "fsync",
            "attempt_local",
        }:
            msg = f"Unsupported payload_spool_sync_mode: {self.payload_spool_sync_mode!r}"
            raise ValueError(msg)
        if self.document_projection is not None:
            if len(set(self.document_projection)) != len(self.document_projection):
                msg = "document_projection must not contain duplicates"
                raise ValueError(msg)
            if SAMPLE_ID not in self.document_projection:
                msg = f"document_projection must include {SAMPLE_ID!r}"
                raise ValueError(msg)
            if LANCE_ROWADDR in self.document_projection or DOCUMENT_POSITION in self.document_projection:
                msg = "document_projection must not include internal row-address columns"
                raise ValueError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._payload_streamer is not None:
            return
        if self.node_local_spool_root.is_symlink() or self.output_root.is_symlink():
            msg = "output_root and node_local_spool_root must not be symlinks"
            raise ValueError(msg)
        self.node_local_spool_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        image_dataset = _open_lance_dataset(
            self.image_uri,
            self.image_version,
            self.image_storage_options,
        )
        if not image_dataset.has_stable_row_ids:
            msg = "image dataset must have stable row IDs"
            raise ValueError(msg)
        image_manifest = _validate_stable_global_ordinal_manifest(image_dataset)
        self._image_fragment_manifest_sha256 = _stable_global_ordinal_manifest_sha256(
            self.image_uri,
            self.image_version,
            image_manifest,
        )
        payload_streamer = _create_stable_id_payload_streamer(
            image_dataset,
            dataset_uri=self.image_uri,
            dataset_version=self.image_version,
            expected_rows=image_manifest.total_rows,
            source_columns=tuple(self.image_columns),
            storage_options=self.image_storage_options,
            fetch_batch_size=self.fetch_batch_size,
            max_pending=self.max_pending,
        )
        self._image_dataset = image_dataset
        self._payload_streamer = payload_streamer

    def teardown(self) -> None:
        try:
            if self._payload_streamer is not None:
                self._payload_streamer.close()
        finally:
            self._payload_streamer = None
            self._image_dataset = None
            self._document_dataset = None
            self._document_identity = None

    def _require_setup(self) -> tuple[_DatasetLike, _StableIdPayloadStreamer, str]:
        if (
            self._image_dataset is None
            or self._payload_streamer is None
            or self._image_fragment_manifest_sha256 is None
        ):
            msg = "LanceCoordinatePayloadPatchStage.setup() must run before process()"
            raise RuntimeError(msg)
        return self._image_dataset, self._payload_streamer, self._image_fragment_manifest_sha256

    def _document(self, identity: _PlanIdentity) -> _DatasetLike:
        requested = (identity.document_uri, identity.document_version)
        if self._document_dataset is None or self._document_identity != requested:
            self._document_dataset = _open_lance_dataset(
                identity.document_uri,
                identity.document_version,
                self.document_storage_options,
            )
            self._document_identity = requested
        return self._document_dataset

    def _fragment(self, dataset: _DatasetLike, fragment_id: int) -> _FragmentLike:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None or int(fragment.fragment_id) != fragment_id:
            msg = f"document dataset does not contain fragment {fragment_id}"
            raise ValueError(msg)
        physical_rows = _positive_integer(int(fragment.physical_rows), "document fragment physical_rows")
        metadata = fragment.metadata
        metadata_rows = int(getattr(metadata, "physical_rows", -1))
        if metadata_rows != physical_rows:
            msg = "document fragment metadata physical_rows does not match"
            raise ValueError(msg)
        if (
            int(fragment.num_deletions) != 0
            or fragment.deletion_file() is not None
            or getattr(metadata, "deletion_file", None) is not None
        ):
            msg = "document fragment must be deletion-free"
            raise ValueError(msg)
        return fragment

    def _validate_plan(self, plan: pa.Table, identity: _PlanIdentity, fragment_rows: int) -> None:
        if plan.num_rows == 0:
            return
        rowaddrs = plan[DOCUMENT_ROWADDR]
        encoded_fragments = pc.shift_right(rowaddrs, 32)
        expected_fragment = pa.scalar(identity.fragment_id, type=pa.uint64())
        if pc.all(pc.equal(encoded_fragments, expected_fragment)).as_py() is not True:
            msg = "coordinate plan document_rowaddr values encode the wrong fragment"
            raise ValueError(msg)
        offsets = pc.bit_wise_and(
            rowaddrs,
            pa.scalar(_ROW_OFFSET_MASK, type=pa.uint64()),
        )
        if pc.all(pc.equal(offsets, plan[DOCUMENT_POSITION])).as_py() is not True:
            msg = "coordinate plan row-address offsets do not match document_position"
            raise ValueError(msg)
        maximum = int(pc.max(plan[DOCUMENT_POSITION]).as_py())
        if maximum >= fragment_rows:
            msg = "coordinate plan document_position lies outside the document fragment"
            raise ValueError(msg)

    def _schemas(
        self,
        document: _DatasetLike,
        image: _DatasetLike,
    ) -> tuple[tuple[str, ...], pa.Schema, pa.Schema, dict[str, str], dict[str, str]]:
        projection = tuple(document.schema.names) if self.document_projection is None else self.document_projection
        missing = [name for name in projection if document.schema.get_field_index(name) < 0]
        if missing:
            msg = f"document_projection contains unknown columns: {missing}"
            raise ValueError(msg)
        if SAMPLE_ID not in projection:
            msg = f"document projection must include {SAMPLE_ID!r}"
            raise ValueError(msg)
        image_missing = [name for name in self.image_columns if image.schema.get_field_index(name) < 0]
        if image_missing:
            msg = f"image dataset is missing payload columns: {image_missing}"
            raise ValueError(msg)

        base_fields = [document.schema.field(name) for name in projection]
        existing_pairs: dict[str, str] = {}
        new_pairs: dict[str, str] = {}
        output_fields = list(base_fields)
        for source, destination in self.image_columns.items():
            source_field = image.schema.field(source)
            destination_index = projection.index(destination) if destination in projection else -1
            if destination_index >= 0:
                destination_field = document.schema.field(destination)
                if self.existing_column_policy == "error":
                    msg = f"document already contains destination column {destination!r}"
                    raise ValueError(msg)
                if destination_field.type != source_field.type:
                    msg = f"document destination {destination!r} type does not match payload source"
                    raise TypeError(msg)
                existing_pairs[source] = destination
            else:
                output_fields.append(
                    pa.field(
                        destination,
                        source_field.type,
                        nullable=True,
                        metadata=source_field.metadata,
                    )
                )
                new_pairs[source] = destination
        output_fields.append(pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False))
        output_schema = pa.schema(output_fields, metadata=document.schema.metadata)
        spool_schema = pa.schema(
            [
                pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
                pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
                pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
                *(image.schema.field(source) for source in self.image_columns),
            ]
        )
        return projection, output_schema, spool_schema, existing_pairs, new_pairs

    def _iter_document_buckets(
        self,
        dataset: _DatasetLike,
        fragment: _FragmentLike,
        projection: tuple[str, ...],
    ) -> Iterator[tuple[int, pa.Table]]:
        scanner = dataset.scanner(
            columns=list(projection),
            fragments=[fragment],
            with_row_address=True,
            scan_in_order=True,
            batch_size=self.bucket_rows,
            batch_readahead=1,
            fragment_readahead=1,
        )
        expected_position = 0
        current_bucket: int | None = None
        bucket_tables: list[pa.Table] = []
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch])
            if LANCE_ROWADDR not in table.column_names:
                msg = "document scanner did not return _rowaddr"
                raise ValueError(msg)
            rowaddrs = table[LANCE_ROWADDR].cast(pa.uint64())
            encoded_fragments = pc.shift_right(rowaddrs, 32)
            expected_fragment = pa.scalar(int(fragment.fragment_id), type=pa.uint64())
            if pc.all(pc.equal(encoded_fragments, expected_fragment)).as_py() is not True:
                msg = "document scanner returned row addresses from another fragment"
                raise ValueError(msg)
            positions = pc.bit_wise_and(
                rowaddrs,
                pa.scalar(_ROW_OFFSET_MASK, type=pa.uint64()),
            )
            expected = pa.array(
                range(expected_position, expected_position + table.num_rows),
                type=pa.uint64(),
            )
            if pc.all(pc.equal(positions, expected)).as_py() is not True:
                msg = "document scanner is not in complete physical row order"
                raise ValueError(msg)
            table = table.set_column(
                table.schema.get_field_index(LANCE_ROWADDR),
                pa.field(LANCE_ROWADDR, pa.uint64(), nullable=False),
                rowaddrs,
            ).append_column(
                pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
                positions,
            )
            offset = 0
            while offset < table.num_rows:
                position = expected_position + offset
                bucket = position // self.bucket_rows
                bucket_stop = min(table.num_rows, offset + ((bucket + 1) * self.bucket_rows - position))
                piece = table.slice(offset, bucket_stop - offset)
                if current_bucket is not None and bucket != current_bucket:
                    yield (
                        current_bucket,
                        (pa.concat_tables(bucket_tables) if len(bucket_tables) > 1 else bucket_tables[0]),
                    )
                    bucket_tables = []
                current_bucket = bucket
                bucket_tables.append(piece)
                offset = bucket_stop
            expected_position += table.num_rows
        if bucket_tables and current_bucket is not None:
            yield current_bucket, pa.concat_tables(bucket_tables) if len(bucket_tables) > 1 else bucket_tables[0]
        if expected_position != int(fragment.physical_rows):
            msg = (
                f"document scanner row conservation failed: read {expected_position}, "
                f"expected {fragment.physical_rows}"
            )
            raise RuntimeError(msg)

    @staticmethod
    def _ensure_new_destinations(table: pa.Table, output_schema: pa.Schema, new_pairs: Mapping[str, str]) -> pa.Table:
        result = table
        for destination in new_pairs.values():
            if destination not in result.column_names:
                field = output_schema.field(destination)
                result = result.append_column(field, pa.nulls(result.num_rows, type=field.type))
        return result

    @staticmethod
    def _finalize_bucket(table: pa.Table, output_schema: pa.Schema) -> pa.Table:
        return pa.Table.from_arrays(
            [table[field.name].combine_chunks() for field in output_schema],
            schema=output_schema,
        )

    @staticmethod
    def _append_splits(
        writer: LancePatchArtifactWriter,
        table: pa.Table,
        target_bytes: int,
        metrics: _PatchMetrics,
    ) -> None:
        split = split_interleaved_by_actual_bytes(table, target_bytes)
        oversized_by_patch: dict[int, int] = {}
        for oversized in split.oversized_samples:
            oversized_by_patch[oversized.patch_index] = oversized_by_patch.get(oversized.patch_index, 0) + 1
        for patch_index, patch in enumerate(split.patches):
            part = writer.append(
                patch,
                oversized_sample_count=oversized_by_patch.get(patch_index, 0),
            )
            metrics.parts += 1
            metrics.rows += patch.num_rows
            metrics.arrow_bytes += patch.nbytes
            metrics.file_bytes += part.size_bytes
        metrics.oversized_samples += len(split.oversized_samples)

    def process(self, task: LanceCoordinatePlanTask) -> FileGroupTask:  # noqa: C901, PLR0912, PLR0915
        image, payload_streamer, image_fragment_digest = self._require_setup()
        plan, manifest = load_coordinate_plan(task)
        identity = _plan_identity(manifest)
        if identity.image_uri != self.image_uri or identity.image_version != self.image_version:
            msg = "coordinate plan image identity does not match stage configuration"
            raise ValueError(msg)
        if identity.fragment_manifest_sha256 != image_fragment_digest:
            msg = "coordinate plan image fragment-manifest digest does not match the opened image dataset"
            raise ValueError(msg)
        document = self._document(identity)
        fragment = self._fragment(document, identity.fragment_id)
        self._validate_plan(plan, identity, int(fragment.physical_rows))
        projection, output_schema, spool_schema, existing_pairs, new_pairs = self._schemas(document, image)

        patch_config_sha256 = _patch_config_sha256(
            self.image_columns,
            self.existing_column_policy,
        )
        patch_identity = LancePatchArtifactIdentity(
            document_uri=identity.document_uri,
            document_version=identity.document_version,
            image_uri=identity.image_uri,
            image_version=identity.image_version,
            fragment_id=identity.fragment_id,
            coordinate_plan_sha256=identity.coordinate_sha256,
            patch_config_sha256=patch_config_sha256,
            expected_rows=int(fragment.physical_rows),
        )
        artifact_root = _artifact_root(self.output_root, identity, patch_config_sha256)
        lock_descriptor = _acquire_artifact_lock(artifact_root)
        try:
            _remove_orphan_attempts(artifact_root)
            _remove_orphan_spools(self.node_local_spool_root, artifact_root)
        except BaseException:
            _release_artifact_lock(lock_descriptor)
            raise
        if artifact_root.exists() or artifact_root.is_symlink():
            try:
                writer = LancePatchArtifactWriter(
                    artifact_root,
                    output_schema,
                    patch_identity,
                    document_position_column=DOCUMENT_POSITION,
                )
                output = writer.finish()
                patch_metadata = dict(output._metadata)
                output._metadata = dict(task._metadata)
                output._metadata["coordinate_plan_source_files"] = output._metadata.get("source_files", [])
                output._metadata.update(patch_metadata)
                output._metadata["lance_coordinate_payload_patch"] = {"adopted": True}
                output._stage_perf = task._stage_perf
                return output
            finally:
                _release_artifact_lock(lock_descriptor)

        try:
            logical_payload_rows = plan.num_rows - plan[STABLE_ROW_ID].null_count
            estimated_scratch_bytes, scratch_free_bytes = _require_scratch_capacity(
                self.node_local_spool_root,
                logical_payload_rows,
                self.estimated_payload_bytes_per_row,
            )
            attempt_root = self.output_root / f".{artifact_root.name}.{uuid.uuid4().hex}.tmp"
            writer = LancePatchArtifactWriter(
                attempt_root,
                output_schema,
                patch_identity,
                document_position_column=DOCUMENT_POSITION,
            )
            spool_root = self.node_local_spool_root / (f".{artifact_root.name}.{uuid.uuid4().hex}.payload-spool")
        except BaseException:
            _release_artifact_lock(lock_descriptor)
            raise
        spool: PayloadSpool | None = None
        try:
            spool = PayloadSpool(
                spool_root,
                spool_schema,
                target_bytes=self.payload_window_bytes,
                bucket_rows=self.bucket_rows,
                stable_id_column=STABLE_ROW_ID,
                document_position_column=DOCUMENT_POSITION,
                sync_mode=self.payload_spool_sync_mode,
            )
            payload_fetch_started = time.perf_counter()
            materialize_metrics = materialize_lance_payload_to_spool(
                payload_streamer,
                plan,
                tuple(self.image_columns),
                spool,
            )
            payload_fetch_seconds = time.perf_counter() - payload_fetch_started
            spool_manifest = spool.finish()
            spool_reader = PayloadSpoolReader(spool_manifest)
            payload_items = iter(zip(spool_manifest.files, spool_reader.iter_tables(), strict=True))
            current_payload = next(payload_items, None)
            applied_payload_rows = 0
            stitcher = _SampleStitcher()
            patch_metrics = _PatchMetrics()

            for bucket, document_bucket in self._iter_document_buckets(document, fragment, projection):
                if current_payload is not None and current_payload[0].bucket < bucket:
                    msg = "payload spool contains a bucket before the current document bucket"
                    raise RuntimeError(msg)
                patched = self._ensure_new_destinations(document_bucket, output_schema, new_pairs)
                while current_payload is not None and current_payload[0].bucket == bucket:
                    record, payload_part = current_payload
                    if existing_pairs:
                        patched = apply_payload_part(
                            patched,
                            payload_part,
                            existing_pairs,
                            self.existing_column_policy,
                        )
                    if new_pairs:
                        patched = apply_payload_part(
                            patched,
                            payload_part,
                            new_pairs,
                            "fill_null",
                        )
                    applied_payload_rows += record.rows
                    current_payload = next(payload_items, None)
                complete = stitcher.push(self._finalize_bucket(patched, output_schema))
                if complete is not None:
                    self._append_splits(
                        writer,
                        complete,
                        self.payload_window_bytes,
                        patch_metrics,
                    )
            if current_payload is not None:
                msg = "payload spool contains rows outside the scanned document fragment"
                raise RuntimeError(msg)
            trailing = stitcher.finish()
            if trailing is not None:
                self._append_splits(
                    writer,
                    trailing,
                    self.payload_window_bytes,
                    patch_metrics,
                )
            if applied_payload_rows != spool_manifest.total_rows:
                msg = (
                    f"payload application row conservation failed: applied {applied_payload_rows}, "
                    f"spooled {spool_manifest.total_rows}"
                )
                raise RuntimeError(msg)
            if patch_metrics.rows != int(fragment.physical_rows):
                msg = "patched document row conservation failed"
                raise RuntimeError(msg)
            writer.finish()
            os.rename(attempt_root, artifact_root)
            fsync_directory(self.output_root)
            final_writer = LancePatchArtifactWriter(
                artifact_root,
                output_schema,
                patch_identity,
                document_position_column=DOCUMENT_POSITION,
                expected_oversized_sample_count=patch_metrics.oversized_samples,
            )
            output = final_writer.finish()
            read_iops = int(materialize_metrics["lance_read_iops"])
            read_bytes = int(materialize_metrics["lance_read_bytes"])
            actual_payload_bytes = int(materialize_metrics["actual_payload_bytes"])
            unique_payload_rows = int(materialize_metrics["unique_rows"])
            logical_payload_rows = int(materialize_metrics["logical_rows"])
            private_take_envelope = float(materialize_metrics["private_take_execution_envelope_seconds"])
            metrics: dict[str, int | float | bool] = {
                "adopted": False,
                "coordinate_rows": plan.num_rows,
                "document_rows": int(fragment.physical_rows),
                "payload_window_bytes": self.payload_window_bytes,
                "estimated_scratch_bytes": estimated_scratch_bytes,
                "scratch_free_bytes": scratch_free_bytes,
                **materialize_metrics,
                "lance_read_iops": read_iops,
                "lance_read_bytes": read_bytes,
                "average_physical_read_bytes": read_bytes / read_iops if read_iops else 0.0,
                "physical_read_operations_per_private_take_envelope_second": (
                    read_iops / private_take_envelope if private_take_envelope else 0.0
                ),
                "physical_reads_per_unique_payload": (read_iops / unique_payload_rows if unique_payload_rows else 0.0),
                "physical_reads_per_logical_payload": (
                    read_iops / logical_payload_rows if logical_payload_rows else 0.0
                ),
                "read_amplification": read_bytes / actual_payload_bytes if actual_payload_bytes else 0.0,
                "payload_materialize_seconds": payload_fetch_seconds,
                "coordinate_plan_arrow_bytes": plan.nbytes,
                "coordinate_plan_file_bytes": Path(task.data).stat().st_size,
                "coordinate_manifest_file_bytes": Path(task.manifest_path).stat().st_size,
                "estimated_inflight_payload_bytes": (
                    self.max_pending * self.fetch_batch_size * self.estimated_payload_bytes_per_row
                ),
                "estimated_payload_actor_reservation_bytes": self.estimated_payload_actor_reservation_bytes,
                "payload_actor_cpus": self.payload_actor_cpus,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "payload_spool_files": len(spool_manifest.files),
                "payload_spool_distinct_buckets": len({item.bucket for item in spool_manifest.files}),
                "bucket_rows": self.bucket_rows,
                "payload_spool_sync_mode": spool_manifest.sync_mode,
                "payload_spool_peak_active_bytes": spool_manifest.peak_active_bytes,
                "payload_spool_peak_bounded_active_bytes": spool_manifest.peak_bounded_active_bytes,
                "payload_spool_oversized_rows": len(spool_manifest.oversized_rows),
                "applied_payload_rows": applied_payload_rows,
                "patch_parts": patch_metrics.parts,
                "patch_rows": patch_metrics.rows,
                "patch_arrow_bytes": patch_metrics.arrow_bytes,
                "patch_file_bytes": patch_metrics.file_bytes,
                "patch_oversized_samples": patch_metrics.oversized_samples,
                "completed_samples": stitcher.completed_samples,
            }
            patch_metadata = dict(output._metadata)
            patch_metadata["lance_patch_artifact"] = dict(patch_metadata["lance_patch_artifact"])
            patch_metadata["lance_patch_artifact"]["adopted"] = False
            output._metadata = dict(task._metadata)
            output._metadata["coordinate_plan_source_files"] = output._metadata.get("source_files", [])
            output._metadata.update(patch_metadata)
            output._metadata["lance_coordinate_payload_patch"] = metrics
            output._stage_perf = task._stage_perf
            return output
        finally:
            if spool is not None and spool.root.exists():
                spool.cleanup()
            if attempt_root.exists() and not attempt_root.is_symlink():
                shutil.rmtree(attempt_root)
            _release_artifact_lock(lock_descriptor)
