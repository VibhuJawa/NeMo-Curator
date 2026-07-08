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

"""Two-shuffle GPU URL resolution and locality-aware Lance payload fetches.

This is an actor-pool shuffle stage, rather than an ordinary
``ProcessingStage``:

1. document image URLs and their origin coordinates are hash shuffled;
2. each owner performs a cuDF merge with its matching hash-sharded sidecar;
3. resolved stable row IDs are explicitly shuffled back to the origin rank;
4. the executor advances all ranks through bounded task windows; each origin
   rescans only the current window and fetches its image columns before the
   next pair of collectives starts.

Only coordinates cross the network.  Image payloads remain in Lance and are
read directly by the actor that owns the output document task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.gpu_lance_shuffle_actor import GpuLanceShuffleActor
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.lance import LanceReadTask
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.uri import validate_credential_free_uri_identity

ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]
MissingKeyPolicy = Literal["error", "null"]
FetchWindowBytes = Literal["256MiB", "1GiB", "4GiB"]

_FETCH_WINDOW_BYTE_PROFILES: dict[str, int] = {
    "256MiB": 256 * 1024**2,
    "1GiB": 1024**3,
    "4GiB": 4 * 1024**3,
}

_INTERNAL_RECONSTRUCTION_COLUMNS = {"_rowaddr", "__nemo_fetched_stable_row_id"}


class _ShuffleActorProtocol(Protocol):
    def read_and_insert_tasks(self, tasks: list[LanceReadTask]) -> None: ...

    def insert_finished(self) -> None: ...

    def resolve_return_and_fetch(self) -> list[InterleavedBatch]: ...

    def cleanup(self) -> None: ...


def _normalise_index_shards(
    index_shards: Sequence[str | Sequence[str]] | Mapping[int, str | Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Return one non-empty path tuple per hash partition."""
    if isinstance(index_shards, Mapping):
        if not index_shards:
            msg = "index_shards must not be empty"
            raise ValueError(msg)
        expected = list(range(len(index_shards)))
        actual = sorted(index_shards)
        if actual != expected:
            msg = f"index_shards mapping keys must be contiguous partition IDs {expected}; got {actual}"
            raise ValueError(msg)
        values = [index_shards[partition_id] for partition_id in expected]
    else:
        values = list(index_shards)
        if not values:
            msg = "index_shards must not be empty"
            raise ValueError(msg)

    result: list[tuple[str, ...]] = []
    for partition_id, value in enumerate(values):
        paths = (value,) if isinstance(value, str) else tuple(value)
        if not paths or any(not path for path in paths):
            msg = f"Index partition {partition_id} must contain at least one non-empty path"
            raise ValueError(msg)
        result.append(paths)
    flattened = [path for paths in result for path in paths]
    if len(flattened) != len(set(flattened)):
        msg = "index_shards must not reuse a sidecar path across hash partitions"
        raise ValueError(msg)
    return tuple(result)


def _resolve_fetch_window_bytes(value: FetchWindowBytes | int) -> int:
    """Resolve one supported no-deadline private-take byte profile."""
    if isinstance(value, str):
        resolved = _FETCH_WINDOW_BYTE_PROFILES.get(value)
        if resolved is None:
            msg = f"fetch_window_bytes must be one of {tuple(_FETCH_WINDOW_BYTE_PROFILES)}, got {value!r}"
            raise ValueError(msg)
        return resolved
    if isinstance(value, bool) or value not in _FETCH_WINDOW_BYTE_PROFILES.values():
        msg = (
            "fetch_window_bytes must be 256MiB, 1GiB, or 4GiB "
            f"({tuple(_FETCH_WINDOW_BYTE_PROFILES.values())} bytes), got {value!r}"
        )
        raise ValueError(msg)
    return value


class GpuLanceShuffleFetchStage(ProcessingStage[LanceReadTask, InterleavedBatch]):
    """Resolve document image URLs on GPU and fetch image payloads by coordinate.

    ``index_shards[p]`` must contain the sidecar rows whose cuDF Murmur3 hash
    partition is ``p`` for exactly ``len(index_shards)`` partitions.  Each
    sidecar row contains the URL and a ``uint64`` stable image row ID.
    Partition ``p`` is loaded by MPF rank ``p % number_of_ranks``.

    The input is a stream of pinned :class:`LanceReadTask` manifests.  The
    documents themselves are deliberately rescanned only after resolved image
    coordinates return to their origin rank.  Set ``document_projection`` to
    the columns needed downstream; ``None`` retains all document columns.
    ``fetch_task_window`` bounds the number of retained manifests and
    payload-bearing outputs per MPF rank.  The executor advances all ranks
    through one window at a time, including empty participation on ranks with
    fewer tasks, and carries emitted batches as Ray object references through
    downstream actor stages.  The executor materializes those references only
    at its final public return; placing a sink next keeps intermediate payloads
    out of the driver heap.  Increasing the window can coalesce more stable IDs
    into each private Lance take at the cost of host payload memory.
    ``fetch_window_bytes`` and ``estimated_payload_bytes_per_row`` bound the
    estimated payload for the entire rank window without a time deadline.  The
    actor also fails closed when fetched bytes scaled by duplicate fan-out
    exceed that target. The wider window is split into sorted
    ``fetch_batch_size`` private takes with at most ``max_pending_takes``
    submitted or running, so locality accumulation does not create one giant
    sparse call. The initial byte target remains an estimate because the
    two-column sidecar deliberately carries no per-image size.

    This class stays importable in a CPU-only environment.  cuDF, RMM,
    RAPIDS-MPF, and Lance are imported lazily inside the Ray GPU actor.
    """

    name = "gpu_lance_shuffle_fetch"
    resources = Resources(cpus=1.0, gpus=1.0)
    batch_size = 1
    actor_class = GpuLanceShuffleActor
    is_resumable = False

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_uri: str,
        image_version: int,
        index_shards: Sequence[str | Sequence[str]] | Mapping[int, str | Sequence[str]],
        index_manifest_uri: str,
        index_manifest_sha256: str,
        image_columns: Mapping[str, str] | None = None,
        document_uri: str | None = None,
        document_version: int | None = None,
        document_url_column: str = "source_ref",
        document_filter: str | None = "modality = 'image'",
        document_projection: Sequence[str] | None = None,
        index_url_column: str = "url",
        index_stable_row_id_column: str = "stable_row_id",
        stable_row_id_output_column: str | None = None,
        document_storage_options: Mapping[str, str] | None = None,
        image_storage_options: Mapping[str, str] | None = None,
        index_storage_options: Mapping[str, str] | None = None,
        existing_column_policy: ExistingColumnPolicy = "fill_null",
        missing_key_policy: MissingKeyPolicy = "error",
        scan_batch_size: int = 65_536,
        fetch_task_window: int = 8,
        fetch_window_bytes: FetchWindowBytes | int = "1GiB",
        estimated_payload_bytes_per_row: int = 128 * 1024,
        fetch_batch_size: int = 1024,
        max_pending_takes: int = 16,
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        enable_statistics: bool = False,
    ) -> None:
        super().__init__()
        self.image_uri = image_uri
        self.image_version = image_version
        self.index_shards = _normalise_index_shards(index_shards)
        self.index_manifest_uri = index_manifest_uri
        self.index_manifest_sha256 = index_manifest_sha256
        self.image_columns = {"image": "binary_content"} if image_columns is None else dict(image_columns)
        self.document_uri = document_uri
        self.document_version = document_version
        self.document_url_column = document_url_column
        self.document_filter = document_filter
        self.document_projection = None if document_projection is None else tuple(document_projection)
        self.index_url_column = index_url_column
        self.index_stable_row_id_column = index_stable_row_id_column
        self.stable_row_id_output_column = stable_row_id_output_column
        self.document_storage_options = dict(document_storage_options or {})
        self.image_storage_options = dict(image_storage_options or {})
        self.index_storage_options = dict(index_storage_options or {})
        self.existing_column_policy = existing_column_policy
        self.missing_key_policy = missing_key_policy
        self.scan_batch_size = scan_batch_size
        self.fetch_task_window = fetch_task_window
        # The Ray actor-pool executor uses this opt-in marker to schedule one
        # collective-safe task window per rank instead of one unbounded run.
        self._shuffle_task_window_size = fetch_task_window
        self.fetch_window_bytes = _resolve_fetch_window_bytes(fetch_window_bytes)
        self.estimated_payload_bytes_per_row = estimated_payload_bytes_per_row
        self.fetch_batch_size = fetch_batch_size
        self.max_pending_takes = max_pending_takes
        self.rmm_pool_size = rmm_pool_size
        self.spill_memory_limit = spill_memory_limit
        self.enable_statistics = enable_statistics
        self._validate_config()

        # ``ShuffleStageAdapter`` injects ``nranks``.  The first collective has
        # one partition per sidecar shard; the actor creates the rank-directed
        # return collective separately with operation ID 1.
        self.actor_kwargs: dict[str, Any] = {
            "total_nparts": len(self.index_shards),
            "image_uri": self.image_uri,
            "image_version": self.image_version,
            "index_shards": self.index_shards,
            "index_manifest_uri": self.index_manifest_uri,
            "index_manifest_sha256": self.index_manifest_sha256,
            "image_columns": self.image_columns,
            "document_uri": self.document_uri,
            "document_version": self.document_version,
            "document_url_column": self.document_url_column,
            "document_filter": self.document_filter,
            "document_projection": self.document_projection,
            "index_url_column": self.index_url_column,
            "index_stable_row_id_column": self.index_stable_row_id_column,
            "stable_row_id_output_column": self.stable_row_id_output_column,
            "document_storage_options": self.document_storage_options,
            "image_storage_options": self.image_storage_options,
            "index_storage_options": self.index_storage_options,
            "existing_column_policy": self.existing_column_policy,
            "missing_key_policy": self.missing_key_policy,
            "scan_batch_size": self.scan_batch_size,
            "fetch_task_window": self.fetch_task_window,
            "fetch_window_bytes": self.fetch_window_bytes,
            "estimated_payload_bytes_per_row": self.estimated_payload_bytes_per_row,
            "fetch_batch_size": self.fetch_batch_size,
            "max_pending_takes": self.max_pending_takes,
            "rmm_pool_size": self.rmm_pool_size,
            "spill_memory_limit": self.spill_memory_limit,
            "enable_statistics": self.enable_statistics,
        }

    def _validate_config(self) -> None:  # noqa: C901, PLR0912, PLR0915
        if not self.image_uri:
            msg = "image_uri must not be empty"
            raise ValueError(msg)
        validate_credential_free_uri_identity(self.image_uri, "image Lance URI")
        if self.image_version <= 0:
            msg = "image_version must be greater than zero"
            raise ValueError(msg)
        if self.document_version is not None and self.document_version <= 0:
            msg = "document_version must be greater than zero"
            raise ValueError(msg)
        if not self.index_manifest_uri or not self.index_manifest_sha256:
            msg = "index_manifest_uri and index_manifest_sha256 must not be empty"
            raise ValueError(msg)
        validate_credential_free_uri_identity(self.index_manifest_uri, "index manifest URI")
        if self.document_uri is not None:
            validate_credential_free_uri_identity(self.document_uri, "document Lance URI")
        for paths in self.index_shards:
            for path in paths:
                validate_credential_free_uri_identity(path, "index sidecar shard URI")
        names = {
            "document_url_column": self.document_url_column,
            "index_url_column": self.index_url_column,
            "index_stable_row_id_column": self.index_stable_row_id_column,
        }
        empty = sorted(name for name, value in names.items() if not value)
        if empty:
            msg = f"Column names must not be empty: {empty}"
            raise ValueError(msg)
        if len(set(names.values())) != len(names):
            msg = f"Coordinate and index column names must be distinct: {names}"
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
        collisions = sorted(
            _INTERNAL_RECONSTRUCTION_COLUMNS & (set(self.image_columns) | set(self.image_columns.values()))
        )
        if collisions:
            msg = f"image_columns collide with internal coordinate columns: {collisions}"
            raise ValueError(msg)
        if self.stable_row_id_output_column is not None:
            if not isinstance(self.stable_row_id_output_column, str):
                msg = "stable_row_id_output_column must be a string or None"
                raise TypeError(msg)
            if not self.stable_row_id_output_column.strip():
                msg = "stable_row_id_output_column must not be empty"
                raise ValueError(msg)
        if self.stable_row_id_output_column in _INTERNAL_RECONSTRUCTION_COLUMNS:
            msg = "stable_row_id_output_column must not use an internal reconstruction column"
            raise ValueError(msg)
        if self.stable_row_id_output_column in self.image_columns.values():
            msg = "stable_row_id_output_column must not collide with an image_columns destination"
            raise ValueError(msg)
        if self.existing_column_policy not in {"error", "fill_null", "overwrite"}:
            msg = f"Unsupported existing_column_policy: {self.existing_column_policy}"
            raise ValueError(msg)
        if self.missing_key_policy not in {"error", "null"}:
            msg = f"Unsupported missing_key_policy: {self.missing_key_policy}"
            raise ValueError(msg)
        if (
            self.scan_batch_size <= 0
            or self.fetch_task_window <= 0
            or self.estimated_payload_bytes_per_row <= 0
            or self.fetch_batch_size <= 0
            or self.max_pending_takes <= 0
        ):
            msg = (
                "scan_batch_size, fetch_task_window, estimated_payload_bytes_per_row, "
                "fetch_batch_size, and max_pending_takes must be positive"
            )
            raise ValueError(msg)
        if self.document_projection is not None:
            if len(set(self.document_projection)) != len(self.document_projection):
                msg = "document_projection must not contain duplicate columns"
                raise ValueError(msg)
            missing = sorted(InterleavedBatch.REQUIRED_COLUMNS - set(self.document_projection))
            if missing:
                msg = f"document_projection omits required InterleavedBatch columns: {missing}"
                raise ValueError(msg)

    def ray_stage_spec(self) -> dict[str, bool]:
        return {RayStageSpecKeys.IS_SHUFFLE_STAGE: True}

    def num_workers(self) -> int:
        """Default to one MPF rank per immutable sidecar partition."""
        return len(self.index_shards)

    def process(self, task: LanceReadTask) -> InterleavedBatch:
        del task
        msg = "GpuLanceShuffleFetchStage requires RayActorPoolExecutor's RAPIDS-MPF shuffle lifecycle"
        raise NotImplementedError(msg)

    def _actor(self) -> _ShuffleActorProtocol:
        actor = getattr(self, "_actor_obj", None)
        if actor is None:
            msg = "GPU Lance shuffle actor is not initialized; use RayActorPoolExecutor"
            raise RuntimeError(msg)
        return actor

    def read_and_insert_batch(self, tasks: list[LanceReadTask]) -> list[LanceReadTask]:
        """Stream compact coordinates for a batch of document manifests."""
        for task in tasks:
            if not isinstance(task, LanceReadTask):
                msg = f"Expected LanceReadTask, got {type(task).__name__}"
                raise TypeError(msg)
        self._actor().read_and_insert_tasks(tasks)
        return tasks

    def insert_finished(self) -> None:
        self._actor().insert_finished()

    def extract_and_write(self) -> list[InterleavedBatch]:
        """Run both extracts and return in-memory interleaved tasks.

        The method name is imposed by the generic shuffle adapter.  This stage
        never writes payloads or intermediate shuffle files.
        """
        return self._actor().resolve_return_and_fetch()

    def teardown(self) -> None:
        self._actor().cleanup()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = list(self.image_columns.values())
        if self.stable_row_id_output_column is not None:
            columns.append(self.stable_row_id_output_column)
        return ["data"], columns

    def process_batch(self, tasks: list[LanceReadTask]) -> list[InterleavedBatch]:
        """Reject ordinary actor execution with a direct, actionable error."""
        del tasks
        msg = "GpuLanceShuffleFetchStage is a collective shuffle stage, not an ordinary batch stage"
        raise NotImplementedError(msg)
