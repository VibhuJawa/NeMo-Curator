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

"""GPU exact-key resolution followed by streaming Lance column fetches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import pyarrow as pa

from nemo_curator.stages.interleaved.gpu_key_lookup import (
    _GpuExactKeyMapper,
    _load_and_validate_sidecar_contract,
    _stable_global_ordinal_manifest_sha256,
)
from nemo_curator.stages.interleaved.lance import (
    LanceColumnFetchStage,
    LanceDatasetConfig,
    LanceIndexCacheConfig,
    PayloadReadMode,
    _LancePayloadFetcher,
    _validate_stable_global_ordinal_manifest,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.utils.uri import validate_credential_free_uri_identity

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_NODE_READY_FILE = ".nemo_curator_gpu_lance_index_ready.json"


def _byte_windows(values: pa.Array, max_bytes: int) -> Iterator[pa.Array]:
    """Yield non-empty Arrow slices bounded by their encoded byte size."""
    if max_bytes <= 0:
        msg = "max_bytes must be greater than zero"
        raise ValueError(msg)

    start = 0
    while start < len(values):
        if values.slice(start, 1).nbytes > max_bytes:
            msg = f"One encoded lookup key exceeds max_lookup_bytes={max_bytes}"
            raise MemoryError(msg)
        low, high = 1, len(values) - start
        while low < high:
            middle = (low + high + 1) // 2
            if values.slice(start, middle).nbytes <= max_bytes:
                low = middle
            else:
                high = middle - 1
        yield values.slice(start, low)
        start += low


@dataclass(frozen=True)
class GpuLanceIndexCacheConfig:
    """Optional one-copy-per-node staging for immutable GPU index files."""

    copy_to_node_local: bool = False
    node_local_root: str = "/local/nemo-curator/gpu-lance-indexes"

    def __post_init__(self) -> None:
        if self.copy_to_node_local and not self.node_local_root:
            msg = "node_local_root must not be empty when copy_to_node_local is enabled"
            raise ValueError(msg)

    def node_local_path(
        self,
        dataset: LanceDatasetConfig,
        reference_files: list[str],
        reference_key_column: str,
        reference_row_id_column: str,
        reference_manifest_sha256: str,
    ) -> Path:
        identity = json.dumps(
            {
                "dataset_uri": dataset.uri,
                "dataset_version": dataset.version,
                "reference_files": reference_files,
                "reference_key_column": reference_key_column,
                "reference_row_id_column": reference_row_id_column,
                "reference_manifest_sha256": reference_manifest_sha256,
            },
            sort_keys=True,
        ).encode()
        cache_id = hashlib.sha256(identity).hexdigest()[:24]
        return Path(self.node_local_root) / cache_id

    def resolved_files(
        self,
        dataset: LanceDatasetConfig,
        reference_files: list[str],
        reference_key_column: str,
        reference_row_id_column: str,
        reference_manifest_sha256: str,
    ) -> list[str]:
        if not self.copy_to_node_local:
            return list(reference_files)
        target = self.node_local_path(
            dataset,
            reference_files,
            reference_key_column,
            reference_row_id_column,
            reference_manifest_sha256,
        )
        return [str(target / Path(path).name) for path in reference_files]


class _GpuLancePayloadFetcher(_LancePayloadFetcher):
    """Resolve keys on GPU and use the common sorted Lance payload reader."""

    def __init__(  # noqa: PLR0913
        self,
        dataset_config: LanceDatasetConfig,
        index_cache: LanceIndexCacheConfig,
        columns: dict[str, str],
        fetch_batch_size: int,
        max_pending_takes: int,
        payload_read_mode: PayloadReadMode,
        medium_density_threshold: float,
        high_density_threshold: float,
        max_coalesced_range_gap: int,
        take_scan_batch_readahead: int,
        validate_payload_keys: bool,
        max_lookup_bytes: int,
        mapper: _GpuExactKeyMapper,
    ) -> None:
        self.mapper = mapper
        self.max_lookup_bytes = max_lookup_bytes
        self._setup_metrics_pending = True
        try:
            super().__init__(
                dataset_config,
                index_cache,
                columns,
                fetch_batch_size,
                max_pending_takes,
                payload_read_mode,
                medium_density_threshold,
                high_density_threshold,
                max_coalesced_range_gap,
                take_scan_batch_readahead,
                validate_payload_keys,
            )
        except Exception:
            mapper.close()
            raise
        input_type = self.key_type
        reference_type = mapper.reference_type
        both_string = (pa.types.is_string(input_type) or pa.types.is_large_string(input_type)) and (
            pa.types.is_string(reference_type) or pa.types.is_large_string(reference_type)
        )
        if input_type != reference_type and not both_string:
            self.close()
            msg = f"Lance key column has type {input_type}; GPU reference key column has type {reference_type}"
            raise TypeError(msg)

    def close(self) -> None:
        if self.mapper is not None:
            self.mapper.close()
            self.mapper = None
        super().close()

    def _resolve_row_ids(self, keys: list[object]) -> tuple[dict[object, int], dict[str, float]]:
        if self.mapper is None:
            msg = "GPU key mapper is closed"
            raise RuntimeError(msg)
        key_array = pa.array(keys, type=self.mapper.reference_type, from_pandas=True)
        key_to_row_id: dict[object, int] = {}
        metrics = {
            "gpu_lookup_windows": 0.0,
            "gpu_eligible_keys": float(len(keys)),
            "gpu_mapped_keys": 0.0,
            "gpu_key_transfer_seconds": 0.0,
            "gpu_key_probe_seconds": 0.0,
            "gpu_row_id_search_seconds": 0.0,
            "gpu_row_id_gather_seconds": 0.0,
        }
        offset = 0
        for window in _byte_windows(key_array, self.max_lookup_bytes):
            mapped = self.mapper.map(window)
            window_keys = keys[offset : offset + len(window)]
            key_to_row_id.update(
                {
                    key: int(row_id)
                    for key, row_id, matched in zip(window_keys, mapped.row_ids, mapped.matched, strict=True)
                    if matched
                }
            )
            offset += len(window)
            metrics["gpu_lookup_windows"] += 1.0
            metrics["gpu_mapped_keys"] += float(mapped.matched.sum())
            metrics["gpu_key_transfer_seconds"] += mapped.transfer_seconds
            metrics["gpu_key_probe_seconds"] += mapped.probe_seconds
            metrics["gpu_row_id_search_seconds"] += mapped.search_seconds
            metrics["gpu_row_id_gather_seconds"] += mapped.gather_seconds
        if self._setup_metrics_pending:
            metrics.update(
                {
                    "gpu_reference_rows": float(self.mapper.reference_rows),
                    "gpu_reference_load_seconds": self.mapper.load_seconds,
                    "gpu_hash_build_seconds": self.mapper.build_seconds,
                    "gpu_reference_bytes": float(self.mapper.gpu_bytes),
                    "gpu_total_bytes": float(self.mapper.gpu_total_bytes),
                }
            )
            self._setup_metrics_pending = False
        return key_to_row_id, metrics


@dataclass
class GpuLanceColumnFetchStage(LanceColumnFetchStage):
    """Fetch Lance columns after persistent GPU key-to-stable-row-ID mapping.

    The immutable reference files contain a sorted exact-key column aligned
    with a ``uint64`` stable Lance row-ID column. Each GPU actor loads that
    compact index once. Streamed task batches are deduplicated by key, mapped
    on the GPU, sorted by stable row ID, and fetched through the same payload
    backend as :class:`LanceColumnFetchStage`. Image bytes remain on the CPU
    and never enter GPU memory or a distributed shuffle.

    Use ``ProcessingStage.with_(batch_size=...)`` to enlarge the coalescing
    window. Larger windows reduce sparse payload reads but retain task
    boundaries and row order in the returned ``InterleavedBatch`` objects.
    """

    reference_files: list[str] = field(default_factory=list)
    reference_key_column: str = "url"
    reference_row_id_column: str = "stable_row_id"
    reference_storage_options: dict[str, str] = field(default_factory=dict)
    reference_manifest_uri: str = ""
    reference_manifest_sha256: str = ""
    expected_reference_rows: int | None = None
    load_factor: float = 0.5
    max_lookup_bytes: int = 256 * 1024**2
    gpu_index_cache: GpuLanceIndexCacheConfig = field(default_factory=GpuLanceIndexCacheConfig)
    name: str = "gpu_lance_column_fetch"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reference_files = list(self.reference_files)
        self.reference_storage_options = dict(self.reference_storage_options or {})
        if not self.reference_files:
            msg = "reference_files must not be empty"
            raise ValueError(msg)
        for reference_file in self.reference_files:
            validate_credential_free_uri_identity(reference_file, "reference sidecar file URI")
        if len(set(self.reference_files)) != len(self.reference_files):
            msg = "reference_files must not contain duplicates"
            raise ValueError(msg)
        basenames = [Path(path).name for path in self.reference_files]
        if len(set(basenames)) != len(basenames):
            msg = "reference_files must have unique basenames for node-local staging"
            raise ValueError(msg)
        if not self.reference_key_column or not self.reference_row_id_column:
            msg = "reference_key_column and reference_row_id_column must not be empty"
            raise ValueError(msg)
        if not self.reference_manifest_uri or not self.reference_manifest_sha256:
            msg = "reference_manifest_uri and reference_manifest_sha256 must not be empty"
            raise ValueError(msg)
        validate_credential_free_uri_identity(self.reference_manifest_uri, "reference manifest URI")
        if self.expected_reference_rows is None or self.expected_reference_rows <= 0:
            msg = "expected_reference_rows is required and must be greater than zero"
            raise ValueError(msg)
        if not 0.0 < self.load_factor <= 1.0:
            msg = "load_factor must be in the interval (0, 1]"
            raise ValueError(msg)
        if self.max_lookup_bytes <= 0:
            msg = "max_lookup_bytes must be greater than zero"
            raise ValueError(msg)

    def _resolved_reference_files(self) -> list[str]:
        return self.gpu_index_cache.resolved_files(
            self.dataset,
            self.reference_files,
            self.reference_key_column,
            self.reference_row_id_column,
            self.reference_manifest_sha256,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if not self.gpu_index_cache.copy_to_node_local:
            return
        target = self.gpu_index_cache.node_local_path(
            self.dataset,
            self.reference_files,
            self.reference_key_column,
            self.reference_row_id_column,
            self.reference_manifest_sha256,
        )
        ready = target / _NODE_READY_FILE
        if ready.is_file():
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            for source in self.reference_files:
                destination = temporary / Path(source).name
                with (
                    fsspec.open(source, "rb", **self.reference_storage_options) as source_stream,
                    destination.open("wb") as destination_stream,
                ):
                    shutil.copyfileobj(source_stream, destination_stream, length=16 * 1024**2)
            (temporary / _NODE_READY_FILE).write_text(
                json.dumps(
                    {
                        "dataset_uri": self.dataset.uri,
                        "dataset_version": self.dataset.version,
                        "reference_files": self.reference_files,
                        "reference_manifest_sha256": self.reference_manifest_sha256,
                    },
                    sort_keys=True,
                )
                + "\n",
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
        try:
            import lance
        except ImportError as exc:  # pragma: no cover - optional dependency failure in worker
            msg = "GpuLanceColumnFetchStage requires the lance Python package"
            raise ImportError(msg) from exc
        reference_files = self._resolved_reference_files()
        if self.gpu_index_cache.copy_to_node_local:
            ready = Path(reference_files[0]).parent / _NODE_READY_FILE
            if not ready.is_file():
                msg = f"Node-local GPU Lance index is not ready: {ready.parent}"
                raise RuntimeError(msg)
        dataset = lance.dataset(
            self.dataset.uri,
            version=self.dataset.version,
            storage_options=self.dataset.storage_options or None,
        )
        if dataset.version != self.dataset.version:
            msg = f"Lance dataset resolved version {dataset.version}; expected {self.dataset.version}"
            raise RuntimeError(msg)
        if not dataset.has_stable_row_ids:
            msg = "GpuLanceColumnFetchStage requires a Lance dataset with stable row IDs"
            raise ValueError(msg)
        manifest = _validate_stable_global_ordinal_manifest(dataset)
        if manifest.total_rows != self.expected_reference_rows:
            msg = (
                f"Reference sidecar expects {self.expected_reference_rows} rows; "
                f"pinned Lance manifest contains {manifest.total_rows}"
            )
            raise ValueError(msg)
        fragment_manifest_sha256 = _stable_global_ordinal_manifest_sha256(
            self.dataset.uri,
            self.dataset.version,
            manifest,
        )
        _load_and_validate_sidecar_contract(
            manifest_uri=self.reference_manifest_uri,
            manifest_sha256=self.reference_manifest_sha256,
            dataset_uri=self.dataset.uri,
            dataset_version=self.dataset.version,
            fragment_manifest_sha256=fragment_manifest_sha256,
            total_rows=manifest.total_rows,
            key_column=self.reference_key_column,
            row_id_column=self.reference_row_id_column,
            layout="replicated_sorted",
            partition_files=(tuple(self.reference_files),),
            storage_options=self.reference_storage_options,
            actual_files=(tuple(reference_files),),
            actual_storage_options={} if self.gpu_index_cache.copy_to_node_local else self.reference_storage_options,
        )
        mapper = _GpuExactKeyMapper(
            reference_files,
            self.reference_key_column,
            self.reference_row_id_column,
            {} if self.gpu_index_cache.copy_to_node_local else self.reference_storage_options,
            self.expected_reference_rows,
            self.load_factor,
        )
        self._fetcher = _GpuLancePayloadFetcher(
            self.dataset,
            self.index_cache,
            self.columns,
            self.fetch_batch_size,
            self.max_pending_takes,
            self.payload_read_mode,
            self.medium_density_threshold,
            self.high_density_threshold,
            self.max_coalesced_range_gap,
            self.take_scan_batch_readahead,
            self.validate_payload_keys,
            self.max_lookup_bytes,
            mapper,
        )
        self._prewarm_metric_pending = None
