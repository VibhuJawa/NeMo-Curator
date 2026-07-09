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

"""Lazy GPU actor implementation for the two-shuffle Lance fetch stage."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.interleaved.gpu_key_lookup import (
    _load_and_validate_sidecar_contract,
    _stable_global_ordinal_manifest_sha256,
    _validate_mpf_partition_ownership,
)
from nemo_curator.stages.interleaved.lance import (
    _bounded_parallel_map,
    _validate_stable_global_ordinal_manifest,
)
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    CoordinatePlanIdentity,
    LanceCoordinatePlanTask,
    lance_coordinate_plan_schema,
    publish_coordinate_plan,
)
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence

    import cudf
    import pylibcudf as plc
    from rapidsmpf.shuffler import Shuffler

    from nemo_curator.stages.text.io.reader.lance import LanceReadTask

_URL = "url"
_ORIGIN_RANK = "origin_rank"
_ORIGIN_SLOT = "origin_slot"
_DOCUMENT_ROWADDR = "document_rowaddr"
_DOCUMENT_POSITION = "document_position"
_STABLE_ROW_ID = "stable_row_id"
_LANCE_ROWADDR = "_rowaddr"
_FETCH_STABLE_ROW_ID = "__nemo_fetched_stable_row_id"
_LANCE_ROW_OFFSET_MASK = (1 << 32) - 1

_REQUEST_COLUMNS = [_URL, _ORIGIN_RANK, _ORIGIN_SLOT, _DOCUMENT_ROWADDR, _DOCUMENT_POSITION]
_RETURN_COLUMNS = [_ORIGIN_RANK, _ORIGIN_SLOT, _DOCUMENT_ROWADDR, _DOCUMENT_POSITION, _STABLE_ROW_ID]


@dataclass(frozen=True)
class _OriginManifest:
    slot: int
    task: LanceReadTask
    uri: str
    version: int


class _StableIdTakeDataset(Protocol):
    def _take_rows(self, stable_ids: list[int], *, columns: list[str]) -> pa.Table: ...


@dataclass(frozen=True)
class _PrivateTakeMeasurement:
    logical_requests: int
    unique_payloads: int
    estimated_bytes_by_take: tuple[int, ...]
    actual_bytes_by_take: tuple[int, ...]
    read_bytes: int
    read_iops: int
    seconds: float
    peak_pending_takes: int


def _array_has_nulls(array: pa.Array | pa.ChunkedArray) -> bool:
    return array.null_count > 0


def _sorted_unique_stable_ids(coordinates: pa.Table) -> list[int]:
    """Return non-null stable IDs in locality-friendly global ordinal order."""
    values = coordinates[_STABLE_ROW_ID].combine_chunks().to_pylist()
    return sorted({int(value) for value in values if value is not None})


def _coordinate_plan_table(coordinates: pa.Table, *, allow_missing: bool) -> pa.Table:
    """Return one slot's compact coordinates in deterministic document order."""
    required = [_DOCUMENT_ROWADDR, _DOCUMENT_POSITION, _STABLE_ROW_ID]
    missing = sorted(set(required) - set(coordinates.column_names))
    if missing:
        msg = f"Returned coordinates omit coordinate-plan columns: {missing}"
        raise ValueError(msg)
    columns = [coordinates[name].combine_chunks().cast(pa.uint64()) for name in required]
    plan = pa.Table.from_arrays(
        columns,
        schema=lance_coordinate_plan_schema(allow_missing=allow_missing),
    )
    if plan.num_rows > 1:
        order = pc.sort_indices(plan, sort_keys=[(_DOCUMENT_POSITION, "ascending")])
        plan = plan.take(order)
    return plan


def _allow_single_partition_replicated_sidecar(nranks: int, total_nparts: int) -> bool:
    """Allow the replicated index only when every key has the same sole owner."""
    return nranks == total_nparts == 1


def _take_rows_by_stable_id(
    dataset: _StableIdTakeDataset,
    stable_ids: list[int],
    columns: list[str],
) -> pa.Table:
    """Fetch one sorted stable-ID window and attach its reconstruction key."""
    table = dataset._take_rows(stable_ids, columns=columns)
    if table.num_rows != len(stable_ids):
        msg = f"Image Lance dataset returned {table.num_rows} rows for {len(stable_ids)} stable IDs"
        raise RuntimeError(msg)
    return table.append_column(_FETCH_STABLE_ROW_ID, pa.array(stable_ids, type=pa.uint64()))


def _stable_id_fetch_chunks(
    stable_ids: list[int],
    fetch_batch_size: int,
    fetch_window_bytes: int,
    estimated_payload_bytes_per_row: int,
) -> list[list[int]]:
    """Split one large coordinate window into bounded private takes."""
    if fetch_batch_size <= 0 or fetch_window_bytes <= 0 or estimated_payload_bytes_per_row <= 0:
        msg = "fetch_batch_size, fetch_window_bytes, and estimated_payload_bytes_per_row must be positive"
        raise ValueError(msg)
    rows_per_take = min(
        fetch_batch_size,
        max(1, fetch_window_bytes // estimated_payload_bytes_per_row),
    )
    return [stable_ids[start : start + rows_per_take] for start in range(0, len(stable_ids), rows_per_take)]


def _take_stable_id_chunks(
    dataset: _StableIdTakeDataset,
    chunks: Sequence[list[int]],
    columns: list[str],
    executor: ThreadPoolExecutor,
    max_pending_takes: int,
) -> tuple[list[pa.Table], int]:
    """Fetch bounded chunks concurrently while retaining fragment-major order."""

    def take(chunk: list[int]) -> pa.Table:
        return _take_rows_by_stable_id(dataset, chunk, columns)

    return _bounded_parallel_map(executor, take, chunks, max_pending_takes)


def _validate_payload_window_bound(
    *,
    logical_requests: int,
    unique_payloads: int,
    estimated_payload_bytes_per_row: int,
    unique_payload_bytes: int | None,
    fetch_window_bytes: int,
) -> int:
    """Fail closed when one rank window can exceed its payload-memory target."""
    estimated_bytes = logical_requests * estimated_payload_bytes_per_row
    if estimated_bytes > fetch_window_bytes:
        msg = (
            f"Payload window estimate is {estimated_bytes} bytes for {logical_requests} requests; "
            f"configured fetch_window_bytes is {fetch_window_bytes}. Reduce fetch_task_window or increase the profile."
        )
        raise MemoryError(msg)
    if unique_payload_bytes is None or unique_payloads == 0:
        return estimated_bytes
    materialized_bytes = ceil(unique_payload_bytes * logical_requests / unique_payloads)
    if materialized_bytes > fetch_window_bytes:
        msg = (
            f"Payload window is estimated to materialize {materialized_bytes} bytes after duplicate fan-out; "
            f"configured fetch_window_bytes is {fetch_window_bytes}. Increase estimated_payload_bytes_per_row "
            "or reduce fetch_task_window."
        )
        raise MemoryError(msg)
    return materialized_bytes


def _update_private_take_metrics(
    metrics: MutableMapping[str, float],
    measurement: _PrivateTakeMeasurement,
) -> None:
    """Accumulate logical coalescing and physical storage measurements."""

    def increment(name: str, value: float) -> None:
        metrics[name] = metrics.get(name, 0.0) + float(value)

    take_calls = len(measurement.actual_bytes_by_take)
    estimated_payload_bytes = sum(measurement.estimated_bytes_by_take)
    payload_bytes = sum(measurement.actual_bytes_by_take)
    increment("logical_payload_requests", measurement.logical_requests)
    increment("unique_payloads", measurement.unique_payloads)
    increment("logical_duplicate_requests", measurement.logical_requests - measurement.unique_payloads)
    increment("private_take_calls", take_calls)
    increment("sparse_calls_avoided", max(0, measurement.unique_payloads - take_calls))
    increment("estimated_payload_bytes", estimated_payload_bytes)
    increment("payload_bytes", payload_bytes)
    increment("lance_read_bytes", measurement.read_bytes)
    increment("lance_read_iops", measurement.read_iops)
    increment("private_take_seconds", measurement.seconds)
    metrics["max_pending_private_takes"] = max(
        metrics.get("max_pending_private_takes", 0.0),
        float(measurement.peak_pending_takes),
    )

    if measurement.actual_bytes_by_take:
        metrics["max_estimated_private_take_bytes"] = max(
            metrics.get("max_estimated_private_take_bytes", 0.0),
            float(max(measurement.estimated_bytes_by_take)),
        )
        metrics["max_actual_private_take_bytes"] = max(
            metrics.get("max_actual_private_take_bytes", 0.0),
            float(max(measurement.actual_bytes_by_take)),
        )
        metrics["max_private_take_target_overshoot_bytes"] = max(
            metrics.get("max_private_take_target_overshoot_bytes", 0.0),
            float(max(0, max(measurement.actual_bytes_by_take) - metrics["private_take_target_bytes"])),
        )

    logical_requests = metrics["logical_payload_requests"]
    unique_payloads = metrics["unique_payloads"]
    total_take_calls = metrics["private_take_calls"]
    total_payload_bytes = metrics["payload_bytes"]
    total_estimated_bytes = metrics["estimated_payload_bytes"]
    read_bytes = metrics["lance_read_bytes"]
    read_iops = metrics["lance_read_iops"]
    take_seconds = metrics["private_take_seconds"]
    metrics["logical_duplicate_fanout"] = logical_requests / unique_payloads if unique_payloads else 0.0
    metrics["unique_payloads_per_private_take"] = unique_payloads / total_take_calls if total_take_calls else 0.0
    metrics["actual_to_estimated_payload_ratio"] = (
        total_payload_bytes / total_estimated_bytes if total_estimated_bytes else 0.0
    )
    metrics["payload_estimation_error_bytes"] = total_payload_bytes - total_estimated_bytes
    metrics["average_physical_read_bytes"] = read_bytes / read_iops if read_iops else 0.0
    metrics["physical_reads_per_unique_payload"] = read_iops / unique_payloads if unique_payloads else 0.0
    metrics["read_amplification"] = read_bytes / total_payload_bytes if total_payload_bytes else 0.0
    metrics["payload_bytes_per_second"] = total_payload_bytes / take_seconds if take_seconds else 0.0
    metrics["physical_read_bytes_per_second"] = read_bytes / take_seconds if take_seconds else 0.0
    metrics["physical_read_operations_per_second"] = read_iops / take_seconds if take_seconds else 0.0


def _set_projected_column(
    table: pa.Table,
    destination: str,
    projected: pa.Array,
    matched: pa.Array,
    existing_column_policy: str,
) -> pa.Table:
    column_index = table.schema.get_field_index(destination)
    if column_index < 0:
        return table.append_column(destination, projected)
    if existing_column_policy == "error":
        msg = f"Document projection already contains destination column {destination!r}"
        raise ValueError(msg)

    existing = table[destination]
    if projected.type != existing.type:
        try:
            projected = projected.cast(existing.type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            msg = (
                f"Cannot project image type {projected.type} into existing document column "
                f"{destination!r} with type {existing.type}"
            )
            raise TypeError(msg) from exc
    replace = pc.and_(matched, pc.is_null(existing)) if existing_column_policy == "fill_null" else matched
    values = pc.if_else(replace, projected, existing)
    return table.set_column(column_index, destination, values)


def _apply_payloads_to_document(  # noqa: PLR0913
    document: pa.Table,
    coordinates: pa.Table,
    payloads: pa.Table,
    *,
    image_columns: Mapping[str, str],
    stable_row_id_output_column: str | None,
    existing_column_policy: str,
) -> pa.Table:
    """Reconstruct one document in source order from rank-returned coordinates."""
    document_addresses = document[_LANCE_ROWADDR].combine_chunks().cast(pa.uint64())
    coordinate_documents = coordinates[_DOCUMENT_ROWADDR].combine_chunks().cast(pa.uint64())
    if pc.count_distinct(coordinate_documents).as_py() != coordinates.num_rows:
        msg = "Returned coordinates contain duplicate document row addresses"
        raise ValueError(msg)
    coordinate_positions = pc.index_in(coordinate_documents, value_set=document_addresses)
    if coordinate_positions.null_count:
        missing = pc.filter(coordinate_documents, pc.is_null(coordinate_positions)).to_pylist()[:10]
        msg = f"Resolved coordinates refer to document rows outside their retained manifest: {missing}"
        raise ValueError(msg)

    document_coordinate_indices = pc.index_in(document_addresses, value_set=coordinate_documents)
    matched = pc.is_valid(document_coordinate_indices)
    coordinate_stable_ids = coordinates[_STABLE_ROW_ID].combine_chunks().cast(pa.uint64())
    document_stable_ids = pc.take(coordinate_stable_ids, document_coordinate_indices)
    fetched_stable_ids = payloads[_FETCH_STABLE_ROW_ID].combine_chunks()
    payload_indices = pc.index_in(document_stable_ids, value_set=fetched_stable_ids)
    missing_payload = pc.and_(pc.is_valid(document_stable_ids), pc.is_null(payload_indices))
    if pc.any(missing_payload).as_py():
        examples = pc.filter(document_stable_ids, missing_payload).to_pylist()[:10]
        msg = f"Private Lance take omitted resolved stable IDs: {examples}"
        raise RuntimeError(msg)

    result = document
    for source, destination in image_columns.items():
        projected = pc.take(payloads[source], payload_indices)
        result = _set_projected_column(
            result,
            destination,
            projected,
            matched,
            existing_column_policy,
        )
    if stable_row_id_output_column is not None:
        result = _set_projected_column(
            result,
            stable_row_id_output_column,
            document_stable_ids,
            matched,
            existing_column_policy,
        )
    return result.drop_columns([_LANCE_ROWADDR])


@lru_cache(maxsize=1)
def _actor_implementation() -> type:  # noqa: C901
    """Build the real actor class only in a GPU worker process."""
    try:
        import cudf
        from rapidsmpf.integrations.cudf.partition import (
            split_and_pack,
            unpack_and_concat,
            unspill_partitions,
        )
        from rapidsmpf.shuffler import Shuffler
        from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe
    except ImportError as exc:  # pragma: no cover - exercised only in a misconfigured GPU worker
        msg = "GpuLanceShuffleFetchStage requires cudf-cu12==26.6.* and rapidsmpf-cu12==26.6.* in every GPU actor"
        raise ImportError(msg) from exc

    from nemo_curator.stages.interleaved.rapidsmpf_2606_shuffler import GpuLanceRapidsMPFShuffler

    class _GpuLanceShuffleActorImpl(GpuLanceRapidsMPFShuffler):
        """Hash-owner join, rank-directed return, and stable-ID payload fetch."""

        def __init__(  # noqa: PLR0913, PLR0915
            self,
            nranks: int,
            total_nparts: int,
            image_uri: str,
            image_version: int,
            index_shards: Sequence[Sequence[str]],
            index_manifest_uri: str,
            index_manifest_sha256: str,
            image_columns: Mapping[str, str],
            document_uri: str | None,
            document_version: int | None,
            document_url_column: str,
            document_filter: str | None,
            document_projection: Sequence[str] | None,
            index_url_column: str,
            index_stable_row_id_column: str,
            stable_row_id_output_column: str | None,
            document_storage_options: Mapping[str, str],
            image_storage_options: Mapping[str, str],
            index_storage_options: Mapping[str, str],
            existing_column_policy: str,
            missing_key_policy: str,
            scan_batch_size: int,
            fetch_task_window: int,
            fetch_window_bytes: int,
            estimated_payload_bytes_per_row: int,
            fetch_batch_size: int,
            max_pending_takes: int,
            coordinate_plan_output_path: str | None,
            rmm_pool_size: int | str | None,
            spill_memory_limit: int | str | None,
            *,
            enable_statistics: bool,
        ) -> None:
            super().__init__(
                nranks=nranks,
                total_nparts=total_nparts,
                shuffle_on=[_URL],
                rmm_pool_size=rmm_pool_size,
                spill_memory_limit=spill_memory_limit,
                enable_statistics=enable_statistics,
            )
            self._nranks = nranks
            self._index_shards = tuple(tuple(paths) for paths in index_shards)
            self._index_manifest_uri = index_manifest_uri
            self._index_manifest_sha256 = index_manifest_sha256
            self._image_uri = image_uri
            self._image_version = image_version
            self._image_columns = dict(image_columns)
            self._document_uri = document_uri
            self._document_version = document_version
            self._document_url_column = document_url_column
            self._document_filter = document_filter
            self._document_projection = None if document_projection is None else tuple(document_projection)
            self._index_url_column = index_url_column
            self._index_stable_row_id_column = index_stable_row_id_column
            self._stable_row_id_output_column = stable_row_id_output_column
            self._document_storage_options = dict(document_storage_options)
            self._image_storage_options = dict(image_storage_options)
            self._index_storage_options = dict(index_storage_options)
            self._existing_column_policy = existing_column_policy
            self._missing_key_policy = missing_key_policy
            self._scan_batch_size = scan_batch_size
            self._fetch_task_window = fetch_task_window
            self._fetch_window_bytes = fetch_window_bytes
            self._estimated_payload_bytes_per_row = estimated_payload_bytes_per_row
            self._fetch_batch_size = fetch_batch_size
            self._max_pending_takes = max_pending_takes
            self._coordinate_plan_output_path = coordinate_plan_output_path

            self._rank: int | None = None
            self._return_shuffler: Shuffler | None = None
            self._indexes: dict[int, cudf.DataFrame] = {}
            self._origins: dict[int, _OriginManifest] = {}
            self._document_datasets: dict[tuple[str, int], Any] = {}
            self._image_dataset: Any | None = None
            self._payload_executor: ThreadPoolExecutor | None = None
            self._image_rows = 0
            self._image_fragment_manifest_sha256 = ""
            self._owned_index_rows_expected = 0
            self._window_index = 0
            self._window_request_rows = 0
            self._window_missing_urls = 0
            self._window_missing_examples: list[object] = []
            self._next_slot = 0
            self._first_inserted = False
            self._cleaned = False
            self._metrics: dict[str, float] = defaultdict(float)
            self._metrics["fetch_window_target_bytes"] = float(fetch_window_bytes)
            self._metrics["private_take_target_bytes"] = float(
                min(fetch_window_bytes, fetch_batch_size * estimated_payload_bytes_per_row)
            )
            self._metrics["private_take_row_limit"] = float(fetch_batch_size)
            self._metrics["max_pending_takes_configured"] = float(max_pending_takes)
            self._metrics["estimated_payload_bytes_per_row"] = float(estimated_payload_bytes_per_row)

        def setup_worker(self, root_address_bytes: bytes) -> None:
            started = time.perf_counter()
            super().setup_worker(root_address_bytes)
            self._rank = int(self.comm.rank)
            if self.total_nparts < self._nranks:
                msg = (
                    f"The URL index has {self.total_nparts} partitions for {self._nranks} ranks; "
                    "provide at least one hash shard per rank"
                )
                raise ValueError(msg)

            # Operation 0 is constructed by the GPU Lance MPF base. Operation 1
            # is live concurrently and receives direct (not hashed) partitions.
            self._return_shuffler = Shuffler(
                self.comm,
                1,
                total_num_partitions=self._nranks,
                br=self.br,
            )
            self._open_image_dataset()
            self._validate_owned_index_contract()
            self._load_owned_indexes()
            if self._coordinate_plan_output_path is None:
                self._payload_executor = ThreadPoolExecutor(
                    max_workers=self._max_pending_takes,
                    thread_name_prefix=f"gpu-lance-rank-{self._rank}-take",
                )
            self._metrics["setup_seconds"] = time.perf_counter() - started

        def _start_next_shuffle_window(self) -> None:
            if self.shuffler is not None or self._return_shuffler is not None:
                msg = "Cannot start an MPF task window while the previous window is active"
                raise RuntimeError(msg)
            self.shuffler = Shuffler(
                self.comm,
                0,
                total_num_partitions=self.total_nparts,
                br=self.br,
            )
            self._return_shuffler = Shuffler(
                self.comm,
                1,
                total_num_partitions=self._nranks,
                br=self.br,
            )

        def _validate_owned_index_contract(self) -> None:
            if self._rank is None or self._image_dataset is None:
                msg = "Image dataset and actor rank must be initialized before sidecar validation"
                raise RuntimeError(msg)
            manifest = _validate_stable_global_ordinal_manifest(self._image_dataset)
            self._image_rows = manifest.total_rows
            fragment_manifest_sha256 = _stable_global_ordinal_manifest_sha256(
                self._image_uri,
                self._image_version,
                manifest,
            )
            self._image_fragment_manifest_sha256 = fragment_manifest_sha256
            owned_coordinates = {
                (partition_id, ordinal)
                for partition_id, paths in enumerate(self._index_shards)
                if partition_id % self._nranks == self._rank
                for ordinal in range(len(paths))
            }
            contract = _load_and_validate_sidecar_contract(
                manifest_uri=self._index_manifest_uri,
                manifest_sha256=self._index_manifest_sha256,
                dataset_uri=self._image_uri,
                dataset_version=self._image_version,
                fragment_manifest_sha256=fragment_manifest_sha256,
                total_rows=manifest.total_rows,
                key_column=self._index_url_column,
                row_id_column=self._index_stable_row_id_column,
                layout="hash_partitioned",
                partition_files=self._index_shards,
                storage_options=self._index_storage_options,
                verify_file_coordinates=owned_coordinates,
                allow_single_partition_replicated=_allow_single_partition_replicated_sidecar(
                    self._nranks,
                    self.total_nparts,
                ),
            )
            self._owned_index_rows_expected = sum(
                identity.rows
                for identity in contract.files
                if (identity.partition_id, identity.ordinal) in owned_coordinates
            )
            self._metrics["pinned_image_rows"] = float(manifest.total_rows)
            self._metrics["verified_index_files"] = float(len(owned_coordinates))

        def _load_owned_indexes(self) -> None:  # noqa: C901
            if self._rank is None:
                msg = "Actor rank is unavailable before setup_worker"
                raise RuntimeError(msg)
            started = time.perf_counter()
            total_rows = 0
            for partition_id, paths in enumerate(self._index_shards):
                if partition_id % self._nranks != self._rank:
                    continue
                frame = cudf.read_parquet(
                    list(paths),
                    columns=[self._index_url_column, self._index_stable_row_id_column],
                    storage_options=self._index_storage_options or None,
                ).rename(
                    columns={
                        self._index_url_column: _URL,
                        self._index_stable_row_id_column: _STABLE_ROW_ID,
                    }
                )
                if len(frame) == 0:
                    msg = f"Hash-sharded URL index partition {partition_id} is empty: {paths}"
                    raise ValueError(msg)
                null_columns = [name for name in [_URL, _STABLE_ROW_ID] if frame[name].isnull().any()]
                if null_columns:
                    msg = f"URL index partition {partition_id} has null columns: {null_columns}"
                    raise ValueError(msg)
                if frame[_URL].duplicated().any():
                    msg = f"URL index partition {partition_id} contains duplicate URLs"
                    raise ValueError(msg)
                _validate_mpf_partition_ownership(
                    frame,
                    key_column=_URL,
                    expected_partition=partition_id,
                    partition_count=self.total_nparts,
                )
                if str(frame[_STABLE_ROW_ID].dtype) != "uint64":
                    msg = (
                        f"URL index column {_STABLE_ROW_ID!r} has type {frame[_STABLE_ROW_ID].dtype}; expected uint64"
                    )
                    raise TypeError(msg)
                minimum_stable_id = int(frame[_STABLE_ROW_ID].min())
                maximum_stable_id = int(frame[_STABLE_ROW_ID].max())
                if minimum_stable_id < 0 or maximum_stable_id >= self._image_rows:
                    msg = (
                        f"URL index partition {partition_id} stable IDs span "
                        f"[{minimum_stable_id}, {maximum_stable_id}], outside the pinned image manifest"
                    )
                    raise ValueError(msg)
                frame = frame[[_URL, _STABLE_ROW_ID]]
                self._indexes[partition_id] = frame
                total_rows += len(frame)
            if not self._indexes:
                msg = f"Rank {self._rank} does not own an index partition"
                raise RuntimeError(msg)
            if total_rows != self._owned_index_rows_expected:
                msg = (
                    f"Rank {self._rank} loaded {total_rows} sidecar rows; "
                    f"the pinned sidecar manifest declares {self._owned_index_rows_expected}"
                )
                raise ValueError(msg)
            self._metrics["local_index_rows"] = float(total_rows)
            self._metrics["local_index_partitions"] = float(len(self._indexes))
            self._metrics["index_load_seconds"] = time.perf_counter() - started

        def _open_image_dataset(self) -> None:
            try:
                import lance
            except ImportError as exc:  # pragma: no cover - optional dependency failure in worker
                msg = "GpuLanceShuffleFetchStage requires the lance Python package"
                raise ImportError(msg) from exc
            dataset = lance.dataset(
                self._image_uri,
                version=self._image_version,
                storage_options=self._image_storage_options or None,
            )
            if dataset.version != self._image_version:
                msg = f"Image Lance dataset resolved version {dataset.version}; expected {self._image_version}"
                raise RuntimeError(msg)
            if not dataset.has_stable_row_ids:
                msg = "GpuLanceShuffleFetchStage requires an image dataset with stable row IDs"
                raise ValueError(msg)
            _validate_stable_global_ordinal_manifest(dataset)
            missing = sorted(set(self._image_columns) - set(dataset.schema.names))
            if missing:
                msg = f"Image Lance projection contains missing columns: {missing}"
                raise ValueError(msg)
            self._image_dataset = dataset

        def _task_identity(self, task: LanceReadTask) -> tuple[str, int]:
            lance_metadata = task._metadata.get("lance") or {}
            uri = self._document_uri or lance_metadata.get("path") or task.dataset_name
            version = self._document_version or lance_metadata.get("version")
            if not uri:
                msg = f"Lance task {task.task_id!r} does not identify a document dataset"
                raise ValueError(msg)
            if version is None:
                msg = f"Lance task {task.task_id!r} does not contain a pinned document version"
                raise ValueError(msg)
            task_uri = lance_metadata.get("path")
            task_version = lance_metadata.get("version")
            if self._document_uri is not None and task_uri is not None and task_uri != self._document_uri:
                msg = f"Document URI mismatch: configured={self._document_uri!r}, task={task_uri!r}"
                raise ValueError(msg)
            if (
                self._document_version is not None
                and task_version is not None
                and task_version != self._document_version
            ):
                msg = f"Document version mismatch: configured={self._document_version}, task={task_version}"
                raise ValueError(msg)
            return str(uri), int(version)

        def _document_dataset(self, uri: str, version: int) -> object:
            key = (uri, version)
            dataset = self._document_datasets.get(key)
            if dataset is None:
                try:
                    import lance
                except ImportError as exc:  # pragma: no cover - optional dependency failure in worker
                    msg = "GpuLanceShuffleFetchStage requires the lance Python package"
                    raise ImportError(msg) from exc
                dataset = lance.dataset(
                    uri,
                    version=version,
                    storage_options=self._document_storage_options or None,
                )
                if dataset.version != version:
                    msg = f"Document Lance dataset resolved version {dataset.version}; expected {version}"
                    raise RuntimeError(msg)
                if self._document_url_column not in dataset.schema.names:
                    msg = f"Document URL column {self._document_url_column!r} is missing from {uri}@{version}"
                    raise ValueError(msg)
                self._document_datasets[key] = dataset
            return dataset

        def _fragments(self, dataset: object, task: LanceReadTask) -> list[object]:
            fragments = []
            for fragment_id in task.data:
                fragment = dataset.get_fragment(fragment_id)
                if fragment is None:
                    msg = f"Document dataset does not contain fragment {fragment_id} from task {task.task_id!r}"
                    raise ValueError(msg)
                fragments.append(fragment)
            return fragments

        def _task_fragments_for_scan(self, dataset: object, task: LanceReadTask) -> list[object]:
            fragments = self._fragments(dataset, task)
            if self._coordinate_plan_output_path is None:
                return fragments
            if len(fragments) != 1:
                msg = (
                    "Coordinate-plan mode requires one document fragment per task; "
                    f"task {task.task_id!r} contains {task.data}"
                )
                raise ValueError(msg)
            deletion_file = getattr(fragments[0].metadata, "deletion_file", None)
            if deletion_file is not None:
                msg = f"Coordinate-plan mode rejects document fragment {task.data[0]} with deletions"
                raise ValueError(msg)
            return fragments

        def _request_frame(self, table: pa.Table, task: LanceReadTask, slot: int) -> cudf.DataFrame:
            if _array_has_nulls(table[self._document_url_column]):
                msg = f"Document task {task.task_id!r} contains null image URLs"
                raise ValueError(msg)
            frame = cudf.DataFrame.from_arrow(table).rename(columns={self._document_url_column: _URL})
            if frame[_URL].eq("").any():
                msg = f"Document task {task.task_id!r} contains empty image URLs"
                raise ValueError(msg)
            frame[_ORIGIN_RANK] = self._rank
            frame[_ORIGIN_RANK] = frame[_ORIGIN_RANK].astype("int32")
            frame[_ORIGIN_SLOT] = slot
            frame[_ORIGIN_SLOT] = frame[_ORIGIN_SLOT].astype("uint64")
            frame = frame.rename(columns={_LANCE_ROWADDR: _DOCUMENT_ROWADDR})
            frame[_DOCUMENT_ROWADDR] = frame[_DOCUMENT_ROWADDR].astype("uint64")
            if self._coordinate_plan_output_path is not None:
                encoded_fragment_ids = set((frame[_DOCUMENT_ROWADDR] >> 32).drop_duplicates().to_arrow().to_pylist())
                if encoded_fragment_ids != {task.data[0]}:
                    msg = (
                        f"Document row addresses encode fragments {sorted(encoded_fragment_ids)}; "
                        f"expected fragment {task.data[0]}"
                    )
                    raise ValueError(msg)
            # Lance row addresses encode the physical row offset in the low
            # 32 bits. Coordinate-plan mode requires one deletion-free
            # fragment, so this is also the full document scan position.
            frame[_DOCUMENT_POSITION] = frame[_DOCUMENT_ROWADDR] & _LANCE_ROW_OFFSET_MASK
            frame[_DOCUMENT_POSITION] = frame[_DOCUMENT_POSITION].astype("uint64")
            return frame[_REQUEST_COLUMNS]

        def _scan_and_insert_task(self, task: LanceReadTask) -> int:
            uri, version = self._task_identity(task)
            dataset = self._document_dataset(uri, version)
            fragments = self._task_fragments_for_scan(dataset, task)
            slot = self._next_slot
            self._next_slot += 1
            self._origins[slot] = _OriginManifest(slot=slot, task=task, uri=uri, version=version)
            scanner = dataset.scanner(
                columns=[self._document_url_column],
                filter=self._document_filter,
                fragments=fragments,
                with_row_address=True,
                scan_in_order=True,
                batch_size=self._scan_batch_size,
                batch_readahead=1,
                fragment_readahead=1,
            )
            task_rows = 0
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue
                frame = self._request_frame(pa.Table.from_batches([batch]), task, slot)
                self.insert_chunk(frame, _REQUEST_COLUMNS)
                self._first_inserted = True
                task_rows += len(frame)
            return task_rows

        def read_and_insert_tasks(self, tasks: list[LanceReadTask]) -> None:
            if self._rank is None:
                msg = "GPU Lance shuffle actor is not set up"
                raise RuntimeError(msg)
            if self.shuffler is None:
                self._start_next_shuffle_window()
            retained_tasks = len(self._origins) + len(tasks)
            if retained_tasks > self._fetch_task_window:
                msg = (
                    f"Rank {self._rank} received {retained_tasks} tasks in one MPF window; "
                    f"fetch_task_window is {self._fetch_task_window}"
                )
                raise ValueError(msg)
            for task in tasks:
                task_rows = self._scan_and_insert_task(task)
                self._metrics["document_tasks"] += 1.0
                self._metrics["request_rows"] += float(task_rows)
                self._window_request_rows += task_rows
            self._metrics["max_origin_manifests"] = max(
                self._metrics["max_origin_manifests"],
                float(len(self._origins)),
            )

        def _empty_request_frame(self) -> cudf.DataFrame:
            index = next(iter(self._indexes.values()))
            frame = index[[_URL]].head(0).copy(deep=True)
            frame[_ORIGIN_RANK] = cudf.Series([], dtype="int32")
            frame[_ORIGIN_SLOT] = cudf.Series([], dtype="uint64")
            frame[_DOCUMENT_ROWADDR] = cudf.Series([], dtype="uint64")
            frame[_DOCUMENT_POSITION] = cudf.Series([], dtype="uint64")
            return frame[_REQUEST_COLUMNS]

        def insert_finished(self) -> None:
            if self.shuffler is None:
                msg = "No active MPF task window to finish"
                raise RuntimeError(msg)
            # An explicit empty chunk makes globally empty/filter-only inputs a
            # valid streaming collective instead of an empty concatenate.
            if not self._first_inserted:
                self.insert_chunk(self._empty_request_frame(), _REQUEST_COLUMNS)
                self._first_inserted = True
            super().insert_finished()

        def _extract_from(self, shuffler: Shuffler) -> Iterator[tuple[int, plc.Table]]:
            from rmm.pylibrmm.stream import DEFAULT_STREAM

            # MPF 26.06 exposes bulk wait/local-partition extraction. The
            # stage's bounded task windows remain the memory/backpressure unit.
            shuffler.wait()
            for partition_id in shuffler.local_partitions():
                packed_chunks = shuffler.extract(partition_id)
                partition = unpack_and_concat(
                    unspill_partitions(
                        packed_chunks,
                        br=self.br,
                        allow_overbooking=True,
                    ),
                    br=self.br,
                    stream=DEFAULT_STREAM,
                )
                yield partition_id, partition

        def _owner_join(self, partition_id: int, table: plc.Table) -> cudf.DataFrame:
            requests = pylibcudf_to_cudf_dataframe(table, column_names=_REQUEST_COLUMNS)
            index = self._indexes.get(partition_id)
            if index is None:
                msg = f"Rank {self._rank} extracted unowned URL partition {partition_id}"
                raise RuntimeError(msg)
            started = time.perf_counter()
            resolved = requests.merge(index, on=_URL, how="left", sort=False)
            self._metrics["cudf_merge_seconds"] += time.perf_counter() - started
            if len(resolved) != len(requests):
                msg = (
                    f"URL owner merge changed row count for partition {partition_id}: "
                    f"requests={len(requests)}, resolved={len(resolved)}"
                )
                raise RuntimeError(msg)
            missing = int(resolved[_STABLE_ROW_ID].isnull().sum())
            self._metrics["missing_urls"] += float(missing)
            if missing and self._missing_key_policy == "error":
                examples = resolved.loc[resolved[_STABLE_ROW_ID].isnull(), _URL].head(10).to_arrow().to_pylist()
                self._window_missing_urls += missing
                remaining = 10 - len(self._window_missing_examples)
                self._window_missing_examples.extend(examples[:remaining])
            self._metrics["resolved_rows"] += float(len(resolved) - missing)
            return resolved[_RETURN_COLUMNS]

        def _insert_rank_directed(self, frame: cudf.DataFrame) -> None:
            if self._return_shuffler is None:
                msg = "Return shuffler is not initialized"
                raise RuntimeError(msg)
            from rmm.pylibrmm.stream import DEFAULT_STREAM

            frame = frame.sort_values(_ORIGIN_RANK, ignore_index=True)
            counts = frame[_ORIGIN_RANK].value_counts(sort=False)
            count_by_rank = dict(zip(counts.index.to_arrow().to_pylist(), counts.to_arrow().to_pylist(), strict=True))
            bad_ranks = sorted(int(rank) for rank in count_by_rank if not 0 <= int(rank) < self._nranks)
            if bad_ranks:
                msg = f"Resolved rows contain invalid origin ranks: {bad_ranks}"
                raise ValueError(msg)
            running = 0
            splits = []
            for rank in range(self._nranks - 1):
                running += int(count_by_rank.get(rank, 0))
                splits.append(running)
            packed = split_and_pack(
                cudf_to_pylibcudf_table(frame),
                splits=splits,
                br=self.br,
                stream=DEFAULT_STREAM,
            )
            self._return_shuffler.insert_chunks(packed)
            self._metrics["return_rows_inserted"] += float(len(frame))

        def _finish_return_shuffle(self) -> pa.Table:
            if self._return_shuffler is None or self._rank is None:
                msg = "Return shuffle is not initialized"
                raise RuntimeError(msg)
            self._return_shuffler.insert_finished()
            outputs = list(self._extract_from(self._return_shuffler))
            if len(outputs) != 1 or outputs[0][0] != self._rank:
                partitions = [partition_id for partition_id, _ in outputs]
                msg = f"Rank {self._rank} expected only its return partition; extracted {partitions}"
                raise RuntimeError(msg)
            frame = pylibcudf_to_cudf_dataframe(outputs[0][1], column_names=_RETURN_COLUMNS)
            ranks = frame[_ORIGIN_RANK].drop_duplicates().to_arrow().to_pylist()
            if ranks and ranks != [self._rank]:
                msg = f"Return partition on rank {self._rank} contains origins {ranks}"
                raise RuntimeError(msg)
            self._metrics["return_rows_extracted"] = float(len(frame))
            return frame.to_arrow()

        def _resolve_coordinates(self) -> pa.Table:
            if self.shuffler is None:
                msg = "No active URL shuffle to resolve"
                raise RuntimeError(msg)
            started = time.perf_counter()
            for partition_id, table in self._extract_from(self.shuffler):
                self._insert_rank_directed(self._owner_join(partition_id, table))
            self._metrics["owner_shuffle_and_merge_seconds"] = time.perf_counter() - started

            # The first collective is window-scoped.  The sidecar remains
            # resident and is reused by the next bounded task window.
            self.shuffler.shutdown()
            self.shuffler = None
            started = time.perf_counter()
            coordinates = self._finish_return_shuffle()
            if self._return_shuffler is not None:
                self._return_shuffler.shutdown()
                self._return_shuffler = None
            self._metrics["return_shuffle_seconds"] = time.perf_counter() - started
            return coordinates

        def _empty_payload_table(self) -> pa.Table:
            if self._image_dataset is None:
                msg = "Image Lance dataset is closed"
                raise RuntimeError(msg)
            if self._payload_executor is None:
                msg = "Image payload executor is not initialized"
                raise RuntimeError(msg)
            arrays = {
                source: pa.array([], type=self._image_dataset.schema.field(source).type)
                for source in self._image_columns
            }
            arrays[_FETCH_STABLE_ROW_ID] = pa.array([], type=pa.uint64())
            return pa.table(arrays)

        def _fetch_payloads(self, coordinates: pa.Table) -> pa.Table:
            if self._image_dataset is None:
                msg = "Image Lance dataset is closed"
                raise RuntimeError(msg)
            requested = coordinates[_STABLE_ROW_ID].combine_chunks()
            stable_ids = _sorted_unique_stable_ids(coordinates)
            requested_rows = len(requested) - requested.null_count
            estimated_window_bytes = _validate_payload_window_bound(
                logical_requests=requested_rows,
                unique_payloads=len(stable_ids),
                estimated_payload_bytes_per_row=self._estimated_payload_bytes_per_row,
                unique_payload_bytes=None,
                fetch_window_bytes=self._fetch_window_bytes,
            )
            chunks = _stable_id_fetch_chunks(
                stable_ids,
                self._fetch_batch_size,
                self._fetch_window_bytes,
                self._estimated_payload_bytes_per_row,
            )
            self._image_dataset.io_stats_incremental()
            take_started = time.perf_counter()
            tables, peak_pending_takes = _take_stable_id_chunks(
                self._image_dataset,
                chunks,
                list(self._image_columns),
                self._payload_executor,
                self._max_pending_takes,
            )
            take_seconds = time.perf_counter() - take_started
            if len(tables) > 1:
                result = pa.concat_tables(tables)
            elif tables:
                result = tables[0]
            else:
                result = self._empty_payload_table()
            estimated_bytes_by_take = [len(chunk) * self._estimated_payload_bytes_per_row for chunk in chunks]
            payload_bytes_by_take = [sum(table[source].nbytes for source in self._image_columns) for table in tables]
            materialized_window_bytes = _validate_payload_window_bound(
                logical_requests=requested_rows,
                unique_payloads=len(stable_ids),
                estimated_payload_bytes_per_row=self._estimated_payload_bytes_per_row,
                unique_payload_bytes=sum(payload_bytes_by_take),
                fetch_window_bytes=self._fetch_window_bytes,
            )
            self._metrics["max_estimated_window_payload_bytes"] = max(
                self._metrics.get("max_estimated_window_payload_bytes", 0.0),
                float(estimated_window_bytes),
            )
            self._metrics["max_materialized_window_payload_bytes"] = max(
                self._metrics.get("max_materialized_window_payload_bytes", 0.0),
                float(materialized_window_bytes),
            )
            stats = self._image_dataset.io_stats_incremental()
            _update_private_take_metrics(
                self._metrics,
                _PrivateTakeMeasurement(
                    logical_requests=requested_rows,
                    unique_payloads=len(stable_ids),
                    estimated_bytes_by_take=tuple(estimated_bytes_by_take),
                    actual_bytes_by_take=tuple(payload_bytes_by_take),
                    read_bytes=int(stats.read_bytes),
                    read_iops=int(stats.read_iops),
                    seconds=take_seconds,
                    peak_pending_takes=peak_pending_takes,
                ),
            )
            return result

        def _rescan_document(self, manifest: _OriginManifest) -> pa.Table:
            dataset = self._document_dataset(manifest.uri, manifest.version)
            scanner = dataset.scanner(
                columns=None if self._document_projection is None else list(self._document_projection),
                fragments=self._fragments(dataset, manifest.task),
                with_row_address=True,
                scan_in_order=True,
                batch_size=self._scan_batch_size,
                batch_readahead=1,
                fragment_readahead=1,
            )
            return scanner.to_table()

        @staticmethod
        def _slot_coordinates(coordinates: pa.Table, slot: int) -> pa.Table:
            mask = pc.equal(coordinates[_ORIGIN_SLOT], pa.scalar(slot, type=pa.uint64()))
            return coordinates.filter(mask)

        def _apply_payloads(
            self,
            document: pa.Table,
            coordinates: pa.Table,
            payloads: pa.Table,
        ) -> pa.Table:
            return _apply_payloads_to_document(
                document,
                coordinates,
                payloads,
                image_columns=self._image_columns,
                stable_row_id_output_column=self._stable_row_id_output_column,
                existing_column_policy=self._existing_column_policy,
            )

        def _build_outputs(self, coordinates: pa.Table) -> list[InterleavedBatch]:
            manifests = list(self._origins.items())
            if len(manifests) > self._fetch_task_window:
                msg = (
                    f"Rank {self._rank} retained {len(manifests)} task manifests; "
                    f"fetch_task_window is {self._fetch_task_window}"
                )
                raise RuntimeError(msg)
            if not manifests:
                return []

            outputs: list[InterleavedBatch] = []
            rescan_started = time.perf_counter()
            payloads = self._fetch_payloads(coordinates)
            self._metrics["fetch_windows"] += 1.0
            for slot, manifest in manifests:
                task_coordinates = self._slot_coordinates(coordinates, slot)
                document = self._rescan_document(manifest)
                result = self._apply_payloads(document, task_coordinates, payloads)
                metadata = dict(manifest.task._metadata)
                stage_metadata = dict(self._metrics)
                stage_metadata.update(
                    {
                        "origin_rank": self._rank,
                        "origin_slot": slot,
                        "task_rows": result.num_rows,
                        "task_image_requests": task_coordinates.num_rows,
                        "task_window": self._window_index,
                    }
                )
                metadata["gpu_lance_shuffle_fetch"] = stage_metadata
                output = InterleavedBatch(
                    dataset_name=manifest.task.dataset_name,
                    data=result,
                    _metadata=metadata,
                    _stage_perf=manifest.task._stage_perf,
                )
                if result.num_rows and not output.validate():
                    msg = f"Document task {manifest.task.task_id!r} did not produce a valid InterleavedBatch"
                    raise ValueError(msg)
                outputs.append(output)
            self._metrics["max_window_outputs"] = max(
                self._metrics["max_window_outputs"],
                float(len(outputs)),
            )
            self._metrics["document_rescan_and_projection_seconds"] += time.perf_counter() - rescan_started
            return outputs

        def _build_coordinate_plans(self, coordinates: pa.Table) -> list[LanceCoordinatePlanTask]:
            if self._coordinate_plan_output_path is None:
                msg = "Coordinate-plan output path is not configured"
                raise RuntimeError(msg)
            if not self._image_fragment_manifest_sha256:
                msg = "Image fragment manifest identity is unavailable"
                raise RuntimeError(msg)
            outputs: list[LanceCoordinatePlanTask] = []
            for slot, manifest in self._origins.items():
                if len(manifest.task.data) != 1:
                    msg = (
                        "Coordinate-plan mode requires exactly one document fragment per LanceReadTask; "
                        f"task {manifest.task.task_id!r} contains {manifest.task.data}"
                    )
                    raise ValueError(msg)
                allow_missing = self._missing_key_policy == "null"
                plan = _coordinate_plan_table(
                    self._slot_coordinates(coordinates, slot),
                    allow_missing=allow_missing,
                )
                output = publish_coordinate_plan(
                    Path(self._coordinate_plan_output_path),
                    plan,
                    CoordinatePlanIdentity(
                        document_uri=manifest.uri,
                        document_version=manifest.version,
                        image_uri=self._image_uri,
                        image_version=self._image_version,
                        fragment_id=manifest.task.data[0],
                        sidecar_manifest_sha256=self._index_manifest_sha256,
                        fragment_manifest_sha256=self._image_fragment_manifest_sha256,
                    ),
                    allow_missing=allow_missing,
                )
                plan_metadata = dict(output._metadata)
                document_sources = manifest.task._metadata.get("source_files")
                output._metadata = dict(manifest.task._metadata)
                if document_sources is not None:
                    output._metadata["document_source_files"] = document_sources
                output._metadata.update(plan_metadata)
                output._metadata["gpu_lance_coordinate_plan"] = {
                    "origin_rank": self._rank,
                    "origin_slot": slot,
                    "task_window": self._window_index,
                    "coordinate_rows": plan.num_rows,
                }
                output._stage_perf = manifest.task._stage_perf
                outputs.append(output)
            self._metrics["coordinate_plan_outputs"] += float(len(outputs))
            self._metrics["coordinate_plan_rows"] += float(coordinates.num_rows)
            return outputs

        def resolve_return_and_fetch(self) -> list[InterleavedBatch | LanceCoordinatePlanTask]:
            total_started = time.perf_counter()
            try:
                coordinates = self._resolve_coordinates()
                if coordinates.num_rows != self._window_request_rows:
                    msg = (
                        f"Rank {self._rank} received {coordinates.num_rows} returned coordinates in task window "
                        f"{self._window_index}; expected {self._window_request_rows}"
                    )
                    raise RuntimeError(msg)
                if self._window_missing_urls:
                    msg = (
                        f"URL sidecar did not resolve {self._window_missing_urls} requests in task window "
                        f"{self._window_index}; examples={self._window_missing_examples}"
                    )
                    raise KeyError(msg)
                outputs = (
                    self._build_coordinate_plans(coordinates)
                    if self._coordinate_plan_output_path is not None
                    else self._build_outputs(coordinates)
                )
                elapsed = time.perf_counter() - total_started
                self._metrics["total_resolve_and_output_seconds"] += elapsed
                if self._coordinate_plan_output_path is None:
                    self._metrics["total_resolve_and_fetch_seconds"] += elapsed
                for output in outputs:
                    for metadata_key in ("gpu_lance_shuffle_fetch", "gpu_lance_coordinate_plan"):
                        metadata = output._metadata.get(metadata_key)
                        if isinstance(metadata, dict):
                            metadata.update(self._metrics)
                return outputs
            finally:
                # Drop all origin and document state before the driver is
                # allowed to schedule the next pair of collectives.
                self._origins.clear()
                self._document_datasets.clear()
                self._next_slot = 0
                self._first_inserted = False
                self._window_request_rows = 0
                self._window_missing_urls = 0
                self._window_missing_examples.clear()
                self._window_index += 1

        def cleanup(self) -> None:
            if self._cleaned:
                return
            self._cleaned = True
            self._indexes.clear()
            self._origins.clear()
            self._document_datasets.clear()
            if self._payload_executor is not None:
                self._payload_executor.shutdown(wait=True, cancel_futures=True)
                self._payload_executor = None
            self._image_dataset = None
            if self._return_shuffler is not None:
                self._return_shuffler.shutdown()
                self._return_shuffler = None
            super().cleanup()

    return _GpuLanceShuffleActorImpl


class GpuLanceShuffleActor:
    """CPU-safe constructor proxy for the lazily imported GPU actor class."""

    def __new__(cls, *args: object, **kwargs: object) -> object:
        return _actor_implementation()(*args, **kwargs)
