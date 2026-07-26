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

"""Indexed Lance column fetches for row-wise interleaved batches."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.interleaved.fetch_budget import (
    is_overload_error,
    register_worker,
    report_outcome,
    unregister_worker,
    worker_id,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage, LanceReaderStage, LanceReadTask
from nemo_curator.tasks import EmptyTask, InterleavedBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.text.io.reader.base import ReaderOutput

_ROW_ID_COLUMN = "_rowid"
_NODE_READY_FILE = ".nemo_curator_lance_index_ready.json"

ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]
MissingKeyPolicy = Literal["mark", "error"]


@dataclass(frozen=True)
class LanceDatasetConfig:
    """Identity and indexed-key configuration for a pinned Lance dataset."""

    uri: str
    version: int
    key_column: str
    index_name: str
    storage_options: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uri:
            msg = "uri must not be empty"
            raise ValueError(msg)
        if self.version <= 0:
            msg = "version must be greater than 0"
            raise ValueError(msg)
        if not self.key_column:
            msg = "key_column must not be empty"
            raise ValueError(msg)
        if not self.index_name:
            msg = "index_name must not be empty"
            raise ValueError(msg)
        object.__setattr__(self, "storage_options", dict(self.storage_options or {}))


@dataclass(frozen=True)
class LanceIndexCacheConfig:
    """Worker cache and optional node-local mirror settings for Lance indices."""

    mirror_path: str | None = None
    copy_to_node_local: bool = False
    node_local_root: str = "/local/lance-indexes"
    prewarm: bool = True
    index_cache_size_bytes: int = 32 * 1024**3
    metadata_cache_size_bytes: int = 1024**3

    def __post_init__(self) -> None:
        if self.copy_to_node_local and not self.mirror_path:
            msg = "copy_to_node_local requires mirror_path"
            raise ValueError(msg)
        if self.copy_to_node_local and not self.node_local_root:
            msg = "node_local_root must not be empty"
            raise ValueError(msg)
        if self.index_cache_size_bytes <= 0:
            msg = "index_cache_size_bytes must be greater than 0"
            raise ValueError(msg)
        if self.metadata_cache_size_bytes <= 0:
            msg = "metadata_cache_size_bytes must be greater than 0"
            raise ValueError(msg)

    def node_local_path(self, dataset: LanceDatasetConfig) -> Path:
        identity = f"{dataset.uri}\n{dataset.version}\n{dataset.index_name}".encode()
        cache_id = hashlib.sha256(identity).hexdigest()[:24]
        return Path(self.node_local_root) / cache_id / "dataset"

    def resolved_index_uri(self, dataset: LanceDatasetConfig) -> str:
        if self.copy_to_node_local:
            return str(self.node_local_path(dataset))
        return self.mirror_path or dataset.uri


@dataclass
class _FetchResult:
    rows_by_key: dict[object, dict[str, object]]
    lookup_seconds: float
    fetch_seconds: float
    fetched_bytes_by_column: dict[str, int]
    read_bytes: int = 0
    read_iops: int = 0


@dataclass(frozen=True)
class _PreparedFetchTask:
    task: InterleavedBatch
    table: pa.Table
    keys: list[object]
    requested_keys: list[object]


class _LanceColumnFetcher:
    """Persistent worker-local Lance session and row-ID fetch executor."""

    def __init__(  # noqa: PLR0913
        self,
        dataset_config: LanceDatasetConfig,
        index_cache: LanceIndexCacheConfig,
        columns: dict[str, str],
        lookup_batch_size: int,
        fetch_batch_size: int,
        io_threads: int,
    ) -> None:
        import lance

        self.config = dataset_config
        self.index_cache = index_cache
        self.columns = columns
        self.lookup_batch_size = lookup_batch_size
        self.fetch_batch_size = fetch_batch_size
        self.session = lance.Session(
            index_cache_size_bytes=index_cache.index_cache_size_bytes,
            metadata_cache_size_bytes=index_cache.metadata_cache_size_bytes,
        )

        index_uri = index_cache.resolved_index_uri(dataset_config)
        index_options: dict[str, Any] = {"version": dataset_config.version, "session": self.session}
        if index_uri == dataset_config.uri and dataset_config.storage_options:
            index_options["storage_options"] = dataset_config.storage_options
        self.index_dataset = lance.dataset(index_uri, **index_options)
        self.remote_dataset = lance.dataset(
            dataset_config.uri,
            version=dataset_config.version,
            storage_options=dataset_config.storage_options or None,
            session=self.session,
        )
        self._validate_datasets()
        if not callable(getattr(self.remote_dataset, "_take_rows", None)):
            msg = "Pinned PyLance build does not expose dataset._take_rows"
            raise TypeError(msg)

        self.prewarm_seconds = 0.0
        if index_cache.prewarm:
            started = time.perf_counter()
            self.index_dataset.prewarm_index(dataset_config.index_name)
            self.prewarm_seconds = time.perf_counter() - started
        self.io_threads = io_threads
        self.executor = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="lance-column-fetch")

    def set_io_threads(self, io_threads: int) -> None:
        """Resize the IO pool between batches; fetches never span this call."""
        if io_threads == self.io_threads:
            return
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.io_threads = io_threads
        self.executor = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="lance-column-fetch")

    @property
    def key_type(self) -> pa.DataType:
        return self.remote_dataset.schema.field(self.config.key_column).type

    @property
    def source_types(self) -> dict[str, pa.DataType]:
        return {source: self.remote_dataset.schema.field(source).type for source in self.columns}

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.index_dataset = None
        self.remote_dataset = None
        self.session = None

    def _validate_datasets(self) -> None:
        key_column = self.config.key_column
        index_schema = self.index_dataset.schema
        remote_schema = self.remote_dataset.schema
        if key_column not in index_schema.names or key_column not in remote_schema.names:
            msg = f"Lance key column {key_column!r} is missing"
            raise ValueError(msg)
        if index_schema.field(key_column).type != remote_schema.field(key_column).type:
            msg = "Index mirror and remote Lance key column types do not match"
            raise TypeError(msg)

        missing = sorted(set(self.columns) - set(remote_schema.names))
        if missing:
            msg = f"Requested Lance columns do not exist: {missing}"
            raise ValueError(msg)

        indices = {index.name: index for index in self.index_dataset.describe_indices()}
        index = indices.get(self.config.index_name)
        if index is None:
            msg = f"Lance index {self.config.index_name!r} does not exist"
            raise ValueError(msg)
        if key_column not in index.field_names:
            msg = f"Lance index {self.config.index_name!r} does not cover {key_column!r}"
            raise ValueError(msg)
        if not self.remote_dataset.has_stable_row_ids or not self.index_dataset.has_stable_row_ids:
            msg = "LanceColumnFetchStage requires stable row IDs"
            raise ValueError(msg)

    def _resolve_row_ids(self, keys: list[object]) -> list[int]:
        row_ids: list[int] = []
        for start in range(0, len(keys), self.lookup_batch_size):
            key_chunk = keys[start : start + self.lookup_batch_size]
            key_array = pa.array(key_chunk, type=self.key_type, from_pandas=True)
            table = self.index_dataset.scanner(
                columns=[],
                filter=pc.field(self.config.key_column).isin(key_array),
                prefilter=True,
                with_row_id=True,
                use_scalar_index=True,
                fast_search=False,
            ).to_table()
            if _ROW_ID_COLUMN not in table.column_names:
                msg = "Lance index lookup did not return stable row IDs"
                raise RuntimeError(msg)
            row_ids.extend(int(value) for value in table[_ROW_ID_COLUMN].combine_chunks().to_pylist())
        return row_ids

    def _take_rows(self, row_ids: list[int]) -> list[pa.Table]:
        projected = [self.config.key_column, *self.columns]
        # Stable row IDs preserve fragment locality when ordered. Sorting once
        # prevents independent take calls from repeatedly reopening the same
        # remote fragments; output order is reconstructed from the key column.
        row_ids = sorted(row_ids)
        chunks = [
            row_ids[start : start + self.fetch_batch_size] for start in range(0, len(row_ids), self.fetch_batch_size)
        ]
        return list(self.executor.map(lambda ids: self.remote_dataset._take_rows(ids, columns=projected), chunks))

    def fetch(self, keys: list[object]) -> _FetchResult:
        if not keys:
            return _FetchResult({}, 0.0, 0.0, {})

        before = self.remote_dataset.io_stats_incremental()
        del before
        lookup_started = time.perf_counter()
        row_ids = self._resolve_row_ids(keys)
        lookup_seconds = time.perf_counter() - lookup_started

        fetch_started = time.perf_counter()
        tables = self._take_rows(row_ids) if row_ids else []
        fetch_seconds = time.perf_counter() - fetch_started

        rows_by_key: dict[object, dict[str, object]] = {}
        fetched_bytes = dict.fromkeys(self.columns, 0)
        requested = set(keys)
        for table in tables:
            for source in self.columns:
                fetched_bytes[source] += table[source].nbytes
            key_values = table[self.config.key_column].combine_chunks().to_pylist()
            source_values = {source: table[source].combine_chunks().to_pylist() for source in self.columns}
            for row_index, key in enumerate(key_values):
                if key not in requested:
                    msg = f"Lance index returned unexpected key {key!r}"
                    raise RuntimeError(msg)
                if key in rows_by_key:
                    msg = f"Multiple Lance rows matched key {key!r}"
                    raise ValueError(msg)
                rows_by_key[key] = {source: values[row_index] for source, values in source_values.items()}

        io_stats = self.remote_dataset.io_stats_incremental()
        return _FetchResult(
            rows_by_key=rows_by_key,
            lookup_seconds=lookup_seconds,
            fetch_seconds=fetch_seconds,
            fetched_bytes_by_column=fetched_bytes,
            read_bytes=int(io_stats.read_bytes),
            read_iops=int(io_stats.read_iops),
        )


@dataclass
class LanceColumnFetchStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Fetch selected columns from a pinned Lance table using exact indexed keys.

    ``io_threads`` sizes IO concurrency per worker, so in-flight requests against
    the object store grow with the worker count. Setting ``fetch_budget`` instead
    caps the cluster-wide total: each worker registers for a share of it at setup
    and adapts that share as the store signals overload, down to
    ``min_fetch_budget``.
    """

    dataset: LanceDatasetConfig
    index_cache: LanceIndexCacheConfig = field(default_factory=LanceIndexCacheConfig)
    input_key_column: str = "source_ref"
    columns: dict[str, str] = field(default_factory=dict)
    presence_column: str | None = None
    existing_column_policy: ExistingColumnPolicy = "error"
    missing_key_policy: MissingKeyPolicy = "mark"
    lookup_batch_size: int = 2_000
    fetch_batch_size: int = 128
    io_threads: int = 16
    fetch_budget: int | None = None
    min_fetch_budget: int = 8
    name: str = "lance_column_fetch"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _fetcher: _LanceColumnFetcher | None = field(default=None, init=False, repr=False)
    _prewarm_metric_pending: float | None = field(default=None, init=False, repr=False)
    _budget_worker_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns = dict(self.columns or {})
        if not self.input_key_column:
            msg = "input_key_column must not be empty"
            raise ValueError(msg)
        if not self.columns and not self.presence_column:
            msg = "columns may be empty only when presence_column is configured"
            raise ValueError(msg)
        if len(set(self.columns.values())) != len(self.columns):
            msg = "Each Lance source column must map to a distinct destination column"
            raise ValueError(msg)
        if self.presence_column in self.columns.values():
            msg = "presence_column must not also be a projected destination column"
            raise ValueError(msg)
        if self.existing_column_policy not in {"error", "fill_null", "overwrite"}:
            msg = f"Unsupported existing_column_policy: {self.existing_column_policy}"
            raise ValueError(msg)
        if self.missing_key_policy not in {"mark", "error"}:
            msg = f"Unsupported missing_key_policy: {self.missing_key_policy}"
            raise ValueError(msg)
        if self.missing_key_policy == "mark" and not self.presence_column:
            msg = "missing_key_policy='mark' requires presence_column"
            raise ValueError(msg)
        self._validate_limits()

    def _validate_limits(self) -> None:
        for name, value in {
            "lookup_batch_size": self.lookup_batch_size,
            "fetch_batch_size": self.fetch_batch_size,
            "io_threads": self.io_threads,
        }.items():
            if value <= 0:
                msg = f"{name} must be greater than 0"
                raise ValueError(msg)
        if self.fetch_budget is not None and not 0 < self.min_fetch_budget <= self.fetch_budget:
            msg = "min_fetch_budget must be in (0, fetch_budget]"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_columns = list(self.columns.values())
        if self.presence_column:
            output_columns.append(self.presence_column)
        return ["data"], output_columns

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if not self.index_cache.copy_to_node_local:
            return
        source = Path(self.index_cache.mirror_path or "")
        if not source.is_dir():
            msg = f"Lance index mirror does not exist: {source}"
            raise FileNotFoundError(msg)
        target = self.index_cache.node_local_path(self.dataset)
        ready = target / _NODE_READY_FILE
        if ready.is_file():
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        shutil.rmtree(temporary)
        try:
            shutil.copytree(source, temporary)
            (temporary / _NODE_READY_FILE).write_text(
                json.dumps({"uri": self.dataset.uri, "version": self.dataset.version}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                os.rename(temporary, target)
            except FileExistsError:
                if not ready.is_file():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.index_cache.copy_to_node_local:
            ready = self.index_cache.node_local_path(self.dataset) / _NODE_READY_FILE
            if not ready.is_file():
                msg = f"Node-local Lance index mirror is not ready: {ready.parent}"
                raise RuntimeError(msg)
        if self.fetch_budget is not None:
            self._budget_worker_id = worker_id()
            self._apply_budget_share(register_worker(self.fetch_budget, self.min_fetch_budget, self._budget_worker_id))
        self._fetcher = _LanceColumnFetcher(
            self.dataset,
            self.index_cache,
            self.columns,
            self.lookup_batch_size,
            self.fetch_batch_size,
            self.io_threads,
        )
        self._prewarm_metric_pending = self._fetcher.prewarm_seconds

    def teardown(self) -> None:
        if self._fetcher is not None:
            self._fetcher.close()
            self._fetcher = None
        if self._budget_worker_id is not None:
            unregister_worker(self._budget_worker_id)
            self._budget_worker_id = None

    def _apply_budget_share(self, share: int | None) -> None:
        """Adopt a share of the cluster-wide budget, or stop budgeting if it is gone."""
        if share is None:
            self._budget_worker_id = None
            return
        self.io_threads = share
        if self._fetcher is not None:
            self._fetcher.set_io_threads(share)

    def _report_budget(self, *, overloaded: bool) -> None:
        if self._budget_worker_id is None:
            return
        self._apply_budget_share(report_outcome(overloaded=overloaded))

    def _ensure_fetcher(self) -> _LanceColumnFetcher:
        if self._fetcher is None:
            self.setup()
        if self._fetcher is None:  # pragma: no cover - setup either returns a fetcher or raises
            msg = "Lance column fetcher setup did not initialize the worker"
            raise RuntimeError(msg)
        return self._fetcher

    def _validate_input_table(self, table: pa.Table, source_types: dict[str, pa.DataType]) -> None:
        if self.input_key_column not in table.column_names:
            msg = f"Input key column {self.input_key_column!r} does not exist"
            raise ValueError(msg)
        collisions = sorted(set(self.columns.values()) & set(table.column_names))
        if collisions and self.existing_column_policy == "error":
            msg = f"Projected destination columns already exist: {collisions}"
            raise ValueError(msg)
        for source, destination in self.columns.items():
            if destination in table.column_names and table.schema.field(destination).type != source_types[source]:
                msg = (
                    f"Destination column {destination!r} has type {table.schema.field(destination).type}; "
                    f"Lance column {source!r} has type {source_types[source]}"
                )
                raise TypeError(msg)
        if self.presence_column in table.column_names and not pa.types.is_boolean(
            table.schema.field(self.presence_column).type
        ):
            msg = f"Presence column {self.presence_column!r} must have boolean type"
            raise TypeError(msg)

    @staticmethod
    def _key_is_present(key: object) -> bool:
        return key is not None and (not isinstance(key, str) or key != "")

    def _requested_keys(
        self,
        table: pa.Table,
        keys: list[object],
        presence: list[bool | None] | None,
    ) -> list[object]:
        destination_values = {
            destination: table[destination].combine_chunks().to_pylist()
            for destination in self.columns.values()
            if destination in table.column_names
        }
        requested: list[object] = []
        seen: set[object] = set()
        for index, key in enumerate(keys):
            if not self._key_is_present(key) or (presence is not None and presence[index] is False):
                continue
            if self.existing_column_policy == "fill_null" and self.columns:
                all_populated = all(
                    destination in destination_values and destination_values[destination][index] is not None
                    for destination in self.columns.values()
                )
                presence_populated = presence is None or presence[index] is not None
                if all_populated and presence_populated:
                    continue
            elif not self.columns and presence is not None and presence[index] is not None:
                continue
            try:
                if key not in seen:
                    seen.add(key)
                    requested.append(key)
            except TypeError as exc:
                msg = f"Input Lance key is not hashable: {key!r}"
                raise TypeError(msg) from exc
        return requested

    def _apply_projection(
        self,
        table: pa.Table,
        keys: list[object],
        fetch_result: _FetchResult,
        source_types: dict[str, pa.DataType],
    ) -> pa.Table:
        result = table
        for source, destination in self.columns.items():
            if destination in result.column_names:
                values = result[destination].combine_chunks().to_pylist()
            else:
                values = [None] * result.num_rows
            for index, key in enumerate(keys):
                row = fetch_result.rows_by_key.get(key)
                if row is None:
                    continue
                if self.existing_column_policy == "fill_null" and values[index] is not None:
                    continue
                values[index] = row[source]
            array = pa.array(values, type=source_types[source], from_pandas=True)
            column_index = result.schema.get_field_index(destination)
            if column_index >= 0:
                result = result.set_column(column_index, destination, array)
            else:
                result = result.append_column(destination, array)
        return result

    def _apply_presence(
        self,
        table: pa.Table,
        keys: list[object],
        requested_keys: list[object],
        found_keys: set[object],
    ) -> pa.Table:
        if not self.presence_column:
            return table
        if self.presence_column in table.column_names:
            values = table[self.presence_column].combine_chunks().to_pylist()
        else:
            values = [None] * table.num_rows
        requested = set(requested_keys)
        for index, key in enumerate(keys):
            if key in requested:
                values[index] = key in found_keys
        presence = pa.array(values, type=pa.bool_(), from_pandas=True)
        column_index = table.schema.get_field_index(self.presence_column)
        if column_index >= 0:
            return table.set_column(column_index, self.presence_column, presence)
        return table.append_column(self.presence_column, presence)

    def _prepare_task(
        self,
        task: InterleavedBatch,
        fetcher: _LanceColumnFetcher,
        source_types: dict[str, pa.DataType],
    ) -> _PreparedFetchTask:
        table = task.to_pyarrow()
        self._validate_input_table(table, source_types)
        if table.schema.field(self.input_key_column).type != fetcher.key_type:
            msg = (
                f"Input key column has type {table.schema.field(self.input_key_column).type}; "
                f"Lance key column has type {fetcher.key_type}"
            )
            raise TypeError(msg)

        keys = table[self.input_key_column].combine_chunks().to_pylist()
        presence = (
            table[self.presence_column].combine_chunks().to_pylist()
            if self.presence_column and self.presence_column in table.column_names
            else None
        )
        requested_keys = self._requested_keys(table, keys, presence)
        return _PreparedFetchTask(task=task, table=table, keys=keys, requested_keys=requested_keys)

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []

        fetcher = self._ensure_fetcher()
        source_types = fetcher.source_types
        prepared = [self._prepare_task(task, fetcher, source_types) for task in tasks]

        # An actor owns one Lance session. Fold the Curator task batch into one
        # deterministic lookup so the B-tree is traversed serially and keys
        # shared by adjacent partitions are resolved and fetched only once.
        requested_keys = list(dict.fromkeys(key for prepared_task in prepared for key in prepared_task.requested_keys))
        try:
            fetch_result = fetcher.fetch(requested_keys)
        except Exception as exc:
            self._report_budget(overloaded=is_overload_error(exc))
            raise
        self._report_budget(overloaded=False)
        found_keys = set(fetch_result.rows_by_key)
        missing_keys = [key for key in requested_keys if key not in found_keys]
        if missing_keys and self.missing_key_policy == "error":
            sample = ", ".join(repr(key) for key in missing_keys[:5])
            msg = f"{len(missing_keys)} Lance keys were not found; examples: {sample}"
            raise KeyError(msg)

        outputs: list[InterleavedBatch] = []
        for prepared_task in prepared:
            result = self._apply_projection(
                prepared_task.table,
                prepared_task.keys,
                fetch_result,
                source_types,
            )
            result = self._apply_presence(
                result,
                prepared_task.keys,
                prepared_task.requested_keys,
                found_keys,
            )
            outputs.append(
                InterleavedBatch(
                    dataset_name=prepared_task.task.dataset_name,
                    data=result,
                    _metadata=prepared_task.task._metadata,
                    _stage_perf=prepared_task.task._stage_perf,
                )
            )

        metrics = {
            "input_tasks": float(len(prepared)),
            "input_rows": float(sum(prepared_task.table.num_rows for prepared_task in prepared)),
            "requested_unique_keys": float(len(requested_keys)),
            "found_unique_keys": float(len(fetch_result.rows_by_key)),
            "missing_unique_keys": float(len(missing_keys)),
            "lance_lookup_seconds": fetch_result.lookup_seconds,
            "lance_fetch_seconds": fetch_result.fetch_seconds,
            "lance_fetched_bytes": float(sum(fetch_result.fetched_bytes_by_column.values())),
            "lance_read_bytes": float(fetch_result.read_bytes),
            "lance_read_iops": float(fetch_result.read_iops),
            "lance_io_threads": float(self.io_threads),
            "peak_rss_bytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        }
        for source, value in fetch_result.fetched_bytes_by_column.items():
            metrics[f"lance_fetched_{source}_bytes"] = float(value)
        if self._prewarm_metric_pending is not None:
            metrics["lance_index_prewarm_seconds"] = self._prewarm_metric_pending
            self._prewarm_metric_pending = None
        self._log_metrics(metrics)
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        """Resolve and fetch one deduplicated key set for a Curator task batch."""
        return self._process_tasks(tasks)


@dataclass
class InterleavedLanceReaderStage(LanceReaderStage):
    """Read Lance fragments into validated ``InterleavedBatch`` objects."""

    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fields is not None:
            missing = sorted(InterleavedBatch.REQUIRED_COLUMNS - set(self.fields))
            if missing:
                msg = f"Interleaved Lance fields omit required columns: {missing}"
                raise ValueError(msg)

    def process(self, task: LanceReadTask) -> InterleavedBatch:
        output: ReaderOutput = self.read_task(task, self._effective_read_kwargs(), self.fields)
        self._validate_result(task, output.data)
        batch = InterleavedBatch(
            dataset_name=task.dataset_name,
            data=output.data,
            _metadata=self._output_metadata(task, output),
            _stage_perf=task._stage_perf,
        )
        if batch.to_pyarrow().num_rows and not batch.validate():
            msg = f"Lance fragment task {task.task_id} is not a valid InterleavedBatch"
            raise ValueError(msg)
        return batch


@dataclass
class InterleavedLanceReader(CompositeStage[EmptyTask, InterleavedBatch]):
    """Partition and read a Lance dataset as row-wise interleaved batches."""

    path: str
    fragments_per_partition: int = 1
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        return [
            LancePartitioningStage(
                path=self.path,
                fragments_per_partition=self.fragments_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            InterleavedLanceReaderStage(
                path=self.path,
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
