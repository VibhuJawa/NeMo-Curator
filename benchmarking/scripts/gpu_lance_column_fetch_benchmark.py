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

"""Benchmark sparse, ordered image-column fetches from a pinned Lance dataset.

The query manifest is a Parquet table with ``source_ref`` and, optionally,
``expected_md5``/``md5``, ``expected_width``/``width``, and
``expected_height``/``height``. All arm boundaries use ``pyarrow.Table``.
Optional GPU, Ray, or lance-ray dependencies are isolated per arm so one
unsupported baseline does not prevent the remaining measurements.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.utils.uri import redact_uri_identity, validate_credential_free_uri_identity

_ORDINAL = "_benchmark_query_ordinal"
_PRESENT = "_benchmark_fetched_present"
_FETCHED = {
    "image": "_benchmark_fetched_image",
    "md5": "_benchmark_fetched_md5",
    "width": "_benchmark_fetched_width",
    "height": "_benchmark_fetched_height",
}
_ACTOR_METRICS = (
    "lance_lookup_seconds",
    "lance_fetch_seconds",
    "lance_fetched_bytes",
    "lance_read_bytes",
    "lance_read_iops",
    "requested_unique_keys",
    "found_unique_keys",
    "gpu_key_transfer_seconds",
    "gpu_key_probe_seconds",
    "gpu_row_id_search_seconds",
    "gpu_row_id_gather_seconds",
    "fragments",
    "fragment_take_calls",
    "payload_take_calls",
    "payload_read_calls",
    "payload_take_rows",
    "rows_per_payload_take",
    "rows_per_payload_read",
    "private_take_calls",
    "private_take_rows",
    "rows_per_private_take",
    "max_pending_payload_reads",
    "max_pending_private_takes",
    "strategy_sparse_fragments",
    "strategy_range_fragments",
    "strategy_sequential_fragments",
    "take_rows_calls",
    "fragment_scan_calls",
    "fragment_scan_batches",
    "fragment_take_ranges",
    "planned_fragment_read_rows",
    "take_scan_calls",
    "take_scan_ranges",
    "planned_scan_rows",
    "range_overread_rows",
    "duplicate_queries_coalesced",
    "average_physical_read_bytes",
    "read_amplification",
    "sparse_calls_avoided",
)
_ACTOR_MAX_METRICS = frozenset({"max_pending_payload_reads", "max_pending_private_takes"})
_ACTOR_DERIVED_METRICS = frozenset(
    {
        "rows_per_payload_take",
        "rows_per_payload_read",
        "rows_per_private_take",
        "average_physical_read_bytes",
        "read_amplification",
    }
)
_ACTOR_PREFIX = "_benchmark_actor_"
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_MISMATCH_EXAMPLES = 20
_SHA256_HEX_LENGTH = 64
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_CLASSES = (
    "adhoc_benchmark",
    "scaling_rank",
    "locality_sensitivity",
    "primary_saturation",
)
_SECRET_OPTION_PARTS = ("access_key", "secret", "token", "password", "credential")
_URI_IN_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_PRIMARY_THROUGHPUT_TIMING = "arm_run_wall_seconds"
_PUBLIC_BASELINE_ARMS = frozenset({"naive_pylance_scalar", "lance_ray_datasource"})
_THREAD_ENVIRONMENT_KEYS = (
    "LANCE_CPU_THREADS",
    "LANCE_IO_THREADS",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


class ArmUnavailableError(RuntimeError):
    """An optional benchmark arm cannot run in the current environment."""


class _IndexStatsProtocol(Protocol):
    def index_stats(self, index_name: str) -> dict[str, object]: ...


class _IndexCoverageDatasetProtocol(Protocol):
    stats: _IndexStatsProtocol

    def count_rows(self) -> int: ...

    def get_fragments(self) -> list[object]: ...


@dataclass(frozen=True)
class QueryManifest:
    """Normalized query input and optional ground-truth columns."""

    table: pa.Table
    expected_columns: dict[str, str]
    digest: str


@dataclass
class ArmRun:
    """One arm's Arrow output plus backend measurements."""

    table: pa.Table
    metrics: dict[str, float | int | None]


@dataclass(frozen=True)
class BenchmarkSettings:
    """Serializable settings shared by benchmark arms."""

    image_lance_uri: str
    image_lance_version: int
    storage_options: dict[str, str]
    key_column: str
    index_name: str
    source_columns: dict[str, str]
    index_mirror: str | None
    index_mirror_contract: dict[str, object] | None
    copy_index_to_node_local: bool
    index_node_local_root: str
    prewarm_index: bool
    index_cache_size_bytes: int
    metadata_cache_size_bytes: int
    lookup_batch_size: int
    fetch_batch_size: int
    io_threads: int
    task_rows: int
    coalesce_tasks: int
    reference_files: list[str]
    reference_storage_options: dict[str, str]
    reference_manifest_uri: str | None
    reference_manifest_sha256: str | None
    reference_key_column: str
    reference_row_id_column: str
    expected_reference_rows: int | None
    gpu_load_factor: float
    max_lookup_bytes: int
    max_pending_fetch_batches: int
    payload_read_mode: str
    medium_density_threshold: float
    high_density_threshold: float
    max_coalesced_range_gap: int
    take_scan_batch_readahead: int
    validate_payload_keys: bool
    copy_reference_to_node_local: bool
    reference_node_local_root: str
    ray_address: str | None
    ray_temp_dir: str | None
    ray_filter_batch_size: int
    ray_concurrency: int | None
    ray_worker_dataset_cache_size: int
    public_index_fast_search: bool
    ray_gpu_actors: int
    actor_warmup_rows: int

    @property
    def window_rows(self) -> int:
        """Rows resolved in one coalesced stage call."""
        return self.task_rows * self.coalesce_tasks


def _json_options(value: str) -> dict[str, str]:
    """Parse an inline JSON object or ``@path`` without exposing it in output."""
    if not value:
        return {}
    raw = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "storage options JSON must contain an object"
        raise TypeError(msg)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()):
        msg = "storage options JSON keys and values must be strings"
        raise TypeError(msg)
    secret_keys = sorted(key for key in parsed if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        msg = (
            f"storage options contain credential-like keys {secret_keys}; "
            "load credentials through the process environment instead"
        )
        raise ValueError(msg)
    return parsed


def _redact_uri_for_report(value: str | None) -> str | None:
    """Remove URI userinfo, query, and fragment fields from persisted reports."""
    if value is None:
        return None
    return redact_uri_identity(value)


def _credential_free_uri(value: str) -> str:
    try:
        return validate_credential_free_uri_identity(value, "URI identity")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _redact_error_message(value: str) -> str:
    """Scrub credential-bearing URI components while retaining useful errors."""
    return _URI_IN_TEXT.sub(lambda match: _redact_uri_for_report(match.group(0)) or "", value)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(path: Path, *arguments: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
            [git, "-C", str(path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _path_code_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    identity: dict[str, object] = {"source_sha256": _file_sha256(resolved)}
    repository = _git_output(resolved.parent, "rev-parse", "--show-toplevel")
    if repository is None:
        return identity
    repository_path = Path(repository)
    commit = _git_output(repository_path, "rev-parse", "HEAD")
    status = _git_output(repository_path, "status", "--porcelain=v1", "--untracked-files=no")
    if commit is not None:
        identity["git_commit"] = commit
    identity["git_dirty"] = bool(status)
    try:
        identity["source_path"] = str(resolved.relative_to(repository_path))
    except ValueError:
        identity["source_path"] = resolved.name
    return identity


def _module_code_identity(module_name: str) -> dict[str, object] | None:
    try:
        specification = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    if specification is None or specification.origin is None:
        return None
    source = Path(specification.origin)
    if not source.is_file():
        return None
    return _path_code_identity(source)


def _code_identity() -> dict[str, object]:
    identity: dict[str, object] = {"benchmark": _path_code_identity(Path(__file__))}
    implementation_modules = {
        "nemo_curator_lance": "nemo_curator.stages.interleaved.lance",
        "lance_ray_datasource": "lance_ray.datasource",
        "lance_ray_gpu": "lance_ray.gpu",
        "lance_ray_io": "lance_ray.io",
    }
    for label, module_name in implementation_modules.items():
        module_identity = _module_code_identity(module_name)
        if module_identity is not None:
            identity[label] = module_identity
    return identity


def _index_mirror_contract(value: str | None) -> dict[str, object] | None:
    """Parse and validate a caller-pinned mirror contract before setup."""
    if not value:
        return None
    raw = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        msg = "index mirror contract JSON must contain an object with string keys"
        raise TypeError(msg)
    contract_format = parsed.pop("format", None)
    if contract_format not in (None, "nemo-curator-lance-index-mirror-v1"):
        msg = f"unsupported index mirror contract format: {contract_format!r}"
        raise ValueError(msg)

    from nemo_curator.stages.interleaved import LanceIndexMirrorContract

    contract = LanceIndexMirrorContract(**parsed)
    return {key: item for key, item in contract.as_dict().items() if key != "format"}


def _resolve_reference_files(files: list[str], patterns: list[str]) -> tuple[list[str], list[str]]:
    resolved = list(files)
    unmatched: list[str] = []
    for pattern in patterns:
        if "://" not in pattern:
            matches = sorted(glob.glob(pattern))
        else:
            try:
                import fsspec

                fs, path = fsspec.core.url_to_fs(pattern)
                matches = [fs.unstrip_protocol(item) for item in sorted(fs.glob(path))]
            except Exception:
                matches = []
        if matches:
            resolved.extend(matches)
        else:
            unmatched.append(pattern)
    return list(dict.fromkeys(resolved)), unmatched


def _arrow_digest(table: pa.Table, columns: list[str]) -> str:
    sink = pa.BufferOutputStream()
    selected = table.select(columns)
    with pa.ipc.new_stream(sink, selected.schema) as writer:
        writer.write_table(selected)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def _find_expected_column(table: pa.Table, requested: str | None, aliases: tuple[str, ...]) -> str | None:
    if requested:
        if requested not in table.column_names:
            msg = f"Expected correctness column {requested!r} is missing from the query manifest"
            raise ValueError(msg)
        return requested
    return next((name for name in aliases if name in table.column_names), None)


def _load_manifest(args: argparse.Namespace) -> QueryManifest:
    table = pq.read_table(args.query_manifest)
    if "source_ref" not in table.column_names:
        msg = "query manifest must contain a source_ref column"
        raise ValueError(msg)
    if args.max_queries is not None:
        table = table.slice(0, args.max_queries)
    if table.num_rows == 0:
        msg = "query manifest contains no rows"
        raise ValueError(msg)
    if table["source_ref"].null_count:
        msg = "query manifest source_ref contains nulls"
        raise ValueError(msg)

    expected = {
        name: column
        for name, column in {
            "md5": _find_expected_column(table, args.expected_md5_column, ("expected_md5", "md5")),
            "width": _find_expected_column(table, args.expected_width_column, ("expected_width", "width")),
            "height": _find_expected_column(table, args.expected_height_column, ("expected_height", "height")),
        }.items()
        if column is not None
    }
    collisions = sorted({_ORDINAL, _PRESENT, *_FETCHED.values()} & set(table.column_names))
    if collisions:
        msg = f"query manifest uses reserved benchmark columns: {collisions}"
        raise ValueError(msg)
    table = table.append_column(_ORDINAL, pa.array(range(table.num_rows), type=pa.int64()))
    digest_columns = ["source_ref", _ORDINAL, *expected.values()]
    return QueryManifest(table=table, expected_columns=expected, digest=_arrow_digest(table, digest_columns))


def _replace_or_append(table: pa.Table, name: str, values: pa.Array | pa.ChunkedArray) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, values) if index >= 0 else table.append_column(name, values)


def _interleaved_input(table: pa.Table, key_type: pa.DataType) -> pa.Table:
    source_ref = table["source_ref"].combine_chunks()
    if source_ref.type != key_type:
        source_ref = pc.cast(source_ref, key_type)
    result = _replace_or_append(table, "source_ref", source_ref)
    result = _replace_or_append(result, "sample_id", pa.repeat(pa.scalar("lance-fetch-benchmark"), table.num_rows))
    result = _replace_or_append(result, "position", pa.repeat(pa.scalar(0, type=pa.int32()), table.num_rows))
    return _replace_or_append(result, "modality", pa.repeat(pa.scalar("image"), table.num_rows))


def _chunk_tables(table: pa.Table, rows: int) -> list[pa.Table]:
    return [table.slice(start, rows) for start in range(0, table.num_rows, rows)]


def _as_arrow_table(batch: pa.Table | pa.RecordBatch) -> pa.Table:
    return batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])


def _payload_projection(settings: BenchmarkSettings) -> dict[str, str]:
    return {source: _FETCHED[kind] for kind, source in settings.source_columns.items()}


def _arm_payload_projection(arm_name: str, settings: BenchmarkSettings) -> dict[str, object]:
    include_key = arm_name in _PUBLIC_BASELINE_ARMS or settings.validate_payload_keys
    source_columns = list(settings.source_columns.values())
    projected_columns = ([settings.key_column] if include_key else []) + source_columns
    return {
        "mode": ("url_image_matched" if include_key and list(settings.source_columns) == ["image"] else "custom"),
        "include_key": include_key,
        "key_column": settings.key_column if include_key else None,
        "source_columns": dict(settings.source_columns),
        "projected_columns": list(dict.fromkeys(projected_columns)),
    }


def _output_types(schema: pa.Schema, settings: BenchmarkSettings) -> dict[str, pa.DataType]:
    result: dict[str, pa.DataType] = {}
    for kind, source in settings.source_columns.items():
        source_type = schema.field(source).type
        result[source] = pa.large_binary() if kind == "image" else source_type
    return result


def _rows_from_tables(tables: list[pa.Table], settings: BenchmarkSettings) -> dict[object, dict[str, object]]:
    rows: dict[object, dict[str, object]] = {}
    for table in tables:
        keys = table[settings.key_column].combine_chunks().to_pylist()
        values = {source: table[source].combine_chunks().to_pylist() for source in settings.source_columns.values()}
        for row_index, key in enumerate(keys):
            if key in rows:
                msg = f"Lance dataset contains multiple rows for key {key!r}"
                raise ValueError(msg)
            rows[key] = {source: column[row_index] for source, column in values.items()}
    return rows


def _assemble_output(
    query: pa.Table,
    rows: dict[object, dict[str, object]],
    settings: BenchmarkSettings,
    source_types: dict[str, pa.DataType],
) -> pa.Table:
    result = query
    keys = query["source_ref"].combine_chunks().to_pylist()
    for kind, source in settings.source_columns.items():
        values = [rows[key][source] if key in rows else None for key in keys]
        result = result.append_column(_FETCHED[kind], pa.array(values, type=source_types[source], from_pandas=True))
    return result.append_column(_PRESENT, pa.array([key in rows for key in keys], type=pa.bool_()))


def _merge_metrics(total: dict[str, float], current: dict[str, float]) -> None:
    for name, value in current.items():
        if name in {"peak_rss_bytes", "gpu_reference_rows", "gpu_reference_bytes", "gpu_total_bytes"}:
            total[name] = max(total.get(name, 0.0), float(value))
        elif name in {"gpu_reference_load_seconds", "gpu_hash_build_seconds", "lance_index_prewarm_seconds"}:
            total.setdefault(name, float(value))
        else:
            total[name] = total.get(name, 0.0) + float(value)


def _stage_setup_metrics(stage: object) -> dict[str, float]:
    fetcher = getattr(stage, "_fetcher", None)
    if fetcher is None:
        return {}
    result: dict[str, float] = {}
    if getattr(fetcher, "prewarm_seconds", 0.0):
        result["lance_index_prewarm_seconds"] = float(fetcher.prewarm_seconds)
    mapper = getattr(fetcher, "mapper", None)
    if mapper is not None:
        result.update(
            {
                "gpu_reference_rows": float(mapper.reference_rows),
                "gpu_reference_load_seconds": float(mapper.load_seconds),
                "gpu_hash_build_seconds": float(mapper.build_seconds),
                "gpu_reference_bytes": float(mapper.gpu_bytes),
                "gpu_total_bytes": float(mapper.gpu_total_bytes),
            }
        )
    return result


class BenchmarkArm(ABC):
    """Persistent benchmark arm with separately measured setup and runs."""

    name: str

    def __init__(self, manifest: QueryManifest, settings: BenchmarkSettings) -> None:
        self.manifest = manifest
        self.settings = settings
        self.setup_metrics: dict[str, float | int | bool | str] = {}

    @abstractmethod
    def setup(self) -> None:
        """Initialize persistent state outside timed warm runs."""

    @abstractmethod
    def run(self) -> ArmRun:
        """Fetch one full copy of the query manifest."""

    def close(self) -> None:
        """Release persistent state."""
        _ = self


def _require_complete_index_coverage(
    dataset: _IndexCoverageDatasetProtocol,
    index_name: str,
) -> dict[str, int | str]:
    """Fail closed before allowing a pinned public reader to skip unindexed data."""
    statistics = dataset.stats.index_stats(index_name)
    expected_rows = int(dataset.count_rows())
    expected_fragments = len(dataset.get_fragments())
    coverage: dict[str, int | str] = {
        "index_type": str(statistics["index_type"]),
        "num_indexed_rows": int(statistics["num_indexed_rows"]),
        "num_unindexed_rows": int(statistics["num_unindexed_rows"]),
        "num_indexed_fragments": int(statistics["num_indexed_fragments"]),
        "num_unindexed_fragments": int(statistics["num_unindexed_fragments"]),
        "dataset_rows": expected_rows,
        "dataset_fragments": expected_fragments,
    }
    if (
        coverage["num_indexed_rows"] != expected_rows
        or coverage["num_unindexed_rows"] != 0
        or coverage["num_indexed_fragments"] != expected_fragments
        or coverage["num_unindexed_fragments"] != 0
    ):
        msg = f"fast search requires complete pinned index coverage, got {coverage}"
        raise ValueError(msg)
    return coverage


class NaivePylanceArm(BenchmarkArm):
    """One scalar-index scanner and one sparse take per unique query key."""

    name = "naive_pylance_scalar"

    def setup(self) -> None:
        try:
            import lance
        except ImportError as exc:
            msg = f"PyLance is unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        self.session = lance.Session(
            index_cache_size_bytes=self.settings.index_cache_size_bytes,
            metadata_cache_size_bytes=self.settings.metadata_cache_size_bytes,
        )
        self.dataset = lance.dataset(
            self.settings.image_lance_uri,
            version=self.settings.image_lance_version,
            storage_options=self.settings.storage_options or None,
            session=self.session,
        )
        indices = {item.name: item for item in self.dataset.describe_indices()}
        if self.settings.index_name not in indices:
            msg = f"Lance scalar index {self.settings.index_name!r} does not exist"
            raise ValueError(msg)
        if not self.dataset.has_stable_row_ids:
            msg = "naive two-phase baseline requires stable Lance row IDs"
            raise ValueError(msg)
        self.source_types = _output_types(self.dataset.schema, self.settings)
        self.input_table = _interleaved_input(
            self.manifest.table, self.dataset.schema.field(self.settings.key_column).type
        )
        if self.settings.public_index_fast_search:
            self.setup_metrics["fast_search_index_coverage"] = _require_complete_index_coverage(
                self.dataset,
                self.settings.index_name,
            )
        if self.settings.prewarm_index:
            started = time.perf_counter()
            self.dataset.prewarm_index(self.settings.index_name)
            self.setup_metrics["lance_index_prewarm_seconds"] = time.perf_counter() - started

    def run(self) -> ArmRun:
        self.dataset.io_stats_incremental()
        unique_keys = list(dict.fromkeys(self.input_table["source_ref"].combine_chunks().to_pylist()))
        projected = [self.settings.key_column, *self.settings.source_columns.values()]
        rows: dict[object, dict[str, object]] = {}
        lookup_seconds = 0.0
        fetch_seconds = 0.0
        fetch_calls = 0
        for key in unique_keys:
            lookup_started = time.perf_counter()
            expression = pc.field(self.settings.key_column) == pa.scalar(
                key, type=self.dataset.schema.field(self.settings.key_column).type
            )
            matches = self.dataset.scanner(
                columns=[],
                filter=expression,
                prefilter=True,
                with_row_id=True,
                use_scalar_index=True,
                fast_search=self.settings.public_index_fast_search,
            ).to_table()
            lookup_seconds += time.perf_counter() - lookup_started
            row_ids = [int(value) for value in matches["_rowid"].combine_chunks().to_pylist()]
            if not row_ids:
                continue
            fetch_started = time.perf_counter()
            fetched = self.dataset._take_rows(row_ids, columns=projected)
            fetch_seconds += time.perf_counter() - fetch_started
            fetch_calls += 1
            _merge_unique_rows(rows, _rows_from_tables([fetched], self.settings))
        stats = self.dataset.io_stats_incremental()
        output = _assemble_output(self.input_table, rows, self.settings, self.source_types)
        return ArmRun(
            output,
            {
                "lookup_seconds": lookup_seconds,
                "fetch_seconds": fetch_seconds,
                "lance_read_bytes": int(stats.read_bytes),
                "lance_read_iops": int(stats.read_iops),
                "lookup_calls": len(unique_keys),
                "fetch_calls": fetch_calls,
                "requested_unique_keys": len(unique_keys),
            },
        )

    def close(self) -> None:
        self.dataset = None
        self.session = None


def _merge_unique_rows(target: dict[object, dict[str, object]], incoming: dict[object, dict[str, object]]) -> None:
    duplicate = set(target) & set(incoming)
    if duplicate:
        msg = f"Lance query returned duplicate keys: {list(duplicate)[:5]}"
        raise ValueError(msg)
    target.update(incoming)


class CuratorStageArm(BenchmarkArm):
    """CPU or GPU Curator column-fetch stage with streaming coalescing."""

    def __init__(self, manifest: QueryManifest, settings: BenchmarkSettings, *, gpu: bool) -> None:
        super().__init__(manifest, settings)
        self.gpu = gpu
        self.name = "gpu_lance_column_fetch_stage" if gpu else "cpu_lance_column_fetch_stage"

    def setup(self) -> None:
        try:
            from nemo_curator.stages.interleaved import (
                GpuLanceColumnFetchStage,
                GpuLanceIndexCacheConfig,
                LanceColumnFetchStage,
                LanceDatasetConfig,
                LanceIndexCacheConfig,
                LanceIndexMirrorContract,
            )
        except ImportError as exc:
            msg = f"Curator Lance stages are unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        if self.gpu and not self.settings.reference_files:
            msg = "GPU sidecar reference files were not provided"
            raise ArmUnavailableError(msg)
        if self.gpu and (not self.settings.reference_manifest_uri or not self.settings.reference_manifest_sha256):
            msg = "GPU sidecar manifest URI and SHA-256 were not provided"
            raise ArmUnavailableError(msg)
        dataset = LanceDatasetConfig(
            uri=self.settings.image_lance_uri,
            version=self.settings.image_lance_version,
            key_column=self.settings.key_column,
            index_name=self.settings.index_name,
            storage_options=self.settings.storage_options,
        )
        cache = LanceIndexCacheConfig(
            mirror_path=self.settings.index_mirror,
            mirror_contract=(
                LanceIndexMirrorContract(**self.settings.index_mirror_contract)
                if self.settings.index_mirror_contract is not None
                else None
            ),
            copy_to_node_local=self.settings.copy_index_to_node_local,
            node_local_root=self.settings.index_node_local_root,
            prewarm=self.settings.prewarm_index,
            index_cache_size_bytes=self.settings.index_cache_size_bytes,
            metadata_cache_size_bytes=self.settings.metadata_cache_size_bytes,
        )
        common: dict[str, Any] = {
            "dataset": dataset,
            "index_cache": cache,
            "input_key_column": "source_ref",
            "columns": _payload_projection(self.settings),
            "presence_column": _PRESENT,
            "missing_key_policy": "mark",
            "lookup_batch_size": self.settings.lookup_batch_size,
            "fetch_batch_size": self.settings.fetch_batch_size,
            "max_pending_takes": self.settings.max_pending_fetch_batches,
            "payload_read_mode": self.settings.payload_read_mode,
            "medium_density_threshold": self.settings.medium_density_threshold,
            "high_density_threshold": self.settings.high_density_threshold,
            "max_coalesced_range_gap": self.settings.max_coalesced_range_gap,
            "take_scan_batch_readahead": self.settings.take_scan_batch_readahead,
            "validate_payload_keys": self.settings.validate_payload_keys,
        }
        if self.gpu:
            self.stage = GpuLanceColumnFetchStage(
                **common,
                reference_files=self.settings.reference_files,
                reference_key_column=self.settings.reference_key_column,
                reference_row_id_column=self.settings.reference_row_id_column,
                reference_storage_options=self.settings.reference_storage_options,
                reference_manifest_uri=self.settings.reference_manifest_uri,
                reference_manifest_sha256=self.settings.reference_manifest_sha256,
                expected_reference_rows=self.settings.expected_reference_rows,
                load_factor=self.settings.gpu_load_factor,
                gpu_index_cache=GpuLanceIndexCacheConfig(
                    copy_to_node_local=self.settings.copy_reference_to_node_local,
                    node_local_root=self.settings.reference_node_local_root,
                ),
            )
        else:
            self.stage = LanceColumnFetchStage(**common)
        self.stage.setup_on_node()
        self.stage.setup()
        self.setup_metrics.update(_stage_setup_metrics(self.stage))
        if self.stage._fetcher is None:
            msg = "Curator stage setup did not create a fetcher"
            raise RuntimeError(msg)
        self.input_table = _interleaved_input(self.manifest.table, self.stage._fetcher.key_type)

    def run(self) -> ArmRun:
        from nemo_curator.tasks import InterleavedBatch

        task_tables = _chunk_tables(self.input_table, self.settings.task_rows)
        outputs: list[pa.Table] = []
        metrics: dict[str, float] = {}
        calls = 0
        for start in range(0, len(task_tables), self.settings.coalesce_tasks):
            tasks = [
                InterleavedBatch(dataset_name="gpu-lance-column-fetch-benchmark", data=table)
                for table in task_tables[start : start + self.settings.coalesce_tasks]
            ]
            outputs.extend(output.to_pyarrow() for output in self.stage.process_batch(tasks))
            _merge_metrics(metrics, self.stage._consume_custom_metrics())
            calls += 1
        metrics.update(
            {
                "lookup_seconds": metrics.get("lance_lookup_seconds", 0.0),
                "fetch_seconds": metrics.get("lance_fetch_seconds", 0.0),
                "lookup_calls": calls,
                "fetch_calls": calls,
            }
        )
        return ArmRun(pa.concat_tables(outputs), metrics)

    def close(self) -> None:
        if hasattr(self, "stage"):
            self.stage.teardown()


def _ensure_ray(address: str | None, temp_dir: str | None) -> tuple[Any, bool, float]:
    try:
        import ray
    except ImportError as exc:
        msg = f"Ray is unavailable: {exc}"
        raise ArmUnavailableError(msg) from exc
    if ray.is_initialized():
        return ray, False, 0.0
    started = time.perf_counter()
    init_options: dict[str, Any] = {
        "address": address,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": False,
    }
    if temp_dir and address is None:
        init_options["_temp_dir"] = temp_dir
    ray.init(**init_options)
    return ray, True, time.perf_counter() - started


def _sql_literal(value: object) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            msg = f"non-finite key {value!r} cannot be represented in a Lance SQL filter"
            raise ValueError(msg)
        return repr(value)
    msg = f"lance-ray SQL baseline does not support key type {type(value).__name__}"
    raise TypeError(msg)


class LanceRayDatasourceArm(BenchmarkArm):
    """Ray Data scans through lance-ray's public LanceDatasource path."""

    name = "lance_ray_datasource"

    def setup(self) -> None:
        self.ray, started, startup_seconds = _ensure_ray(self.settings.ray_address, self.settings.ray_temp_dir)
        try:
            import lance
            import lance_ray
        except ImportError as exc:
            msg = f"lance-ray is unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        if not hasattr(lance_ray, "read_lance"):
            msg = "installed lance-ray does not expose read_lance"
            raise ArmUnavailableError(msg)
        try:
            from lance_ray.datasource import LanceDatasource
        except ImportError as exc:
            msg = f"lance-ray public datasource is unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        parameters = inspect.signature(lance_ray.read_lance).parameters
        if "dataset_options" not in parameters:
            msg = "installed lance-ray cannot pin a dataset version"
            raise ArmUnavailableError(msg)
        required_cache_parameters = {
            "index_cache_size_bytes",
            "metadata_cache_size_bytes",
            "worker_dataset_cache_size",
        }
        if missing_cache_parameters := sorted(required_cache_parameters.difference(parameters)):
            msg = f"installed lance-ray cannot bind worker cache policy: {missing_cache_parameters}"
            raise ArmUnavailableError(msg)
        if not _SQL_IDENTIFIER.fullmatch(self.settings.key_column):
            msg = "lance-ray SQL baseline requires a simple identifier key column"
            raise ArmUnavailableError(msg)
        self.lance_ray = lance_ray
        self.dataset = lance.dataset(
            self.settings.image_lance_uri,
            version=self.settings.image_lance_version,
            storage_options=self.settings.storage_options or None,
        )
        self.source_types = _output_types(self.dataset.schema, self.settings)
        self.input_table = _interleaved_input(
            self.manifest.table, self.dataset.schema.field(self.settings.key_column).type
        )
        if self.settings.public_index_fast_search:
            self.setup_metrics["fast_search_index_coverage"] = _require_complete_index_coverage(
                self.dataset,
                self.settings.index_name,
            )
        cache_identity = LanceDatasource(
            uri=self.settings.image_lance_uri,
            storage_options=self.settings.storage_options or None,
            dataset_options={"version": self.settings.image_lance_version},
            index_cache_size_bytes=self.settings.index_cache_size_bytes,
            metadata_cache_size_bytes=self.settings.metadata_cache_size_bytes,
            worker_dataset_cache_size=self.settings.ray_worker_dataset_cache_size,
        )
        self.setup_metrics.update(
            {
                "ray_started_by_benchmark": started,
                "ray_startup_seconds": startup_seconds,
                "ray_datasource_name": cache_identity.get_name(),
                "ray_worker_dataset_cache": cache_identity.worker_cache_config,
                "ray_worker_dataset_cache_reuse_scope": "opportunistic_per_worker_without_affinity",
                "ray_worker_dataset_cache_hits_observed": False,
            }
        )

    def run(self) -> ArmRun:
        started = time.perf_counter()
        unique_keys = list(dict.fromkeys(self.input_table["source_ref"].combine_chunks().to_pylist()))
        tables: list[pa.Table] = []
        scan_jobs = 0
        for start in range(0, len(unique_keys), self.settings.ray_filter_batch_size):
            keys = unique_keys[start : start + self.settings.ray_filter_batch_size]
            expression = f"{self.settings.key_column} IN ({', '.join(_sql_literal(key) for key in keys)})"
            dataset = self.lance_ray.read_lance(
                uri=self.settings.image_lance_uri,
                columns=[self.settings.key_column, *self.settings.source_columns.values()],
                filter=expression,
                storage_options=self.settings.storage_options or None,
                dataset_options={"version": self.settings.image_lance_version},
                index_cache_size_bytes=self.settings.index_cache_size_bytes,
                metadata_cache_size_bytes=self.settings.metadata_cache_size_bytes,
                worker_dataset_cache_size=self.settings.ray_worker_dataset_cache_size,
                scanner_options={
                    "batch_size": self.settings.fetch_batch_size,
                    "fast_search": self.settings.public_index_fast_search,
                },
                concurrency=self.settings.ray_concurrency,
            )
            tables.extend(_as_arrow_table(batch) for batch in dataset.iter_batches(batch_format="pyarrow"))
            scan_jobs += 1
        rows = _rows_from_tables(tables, self.settings)
        output = _assemble_output(self.input_table, rows, self.settings, self.source_types)
        fetch_seconds = time.perf_counter() - started
        return ArmRun(
            output,
            {
                "lookup_seconds": None,
                "fetch_seconds": fetch_seconds,
                "lance_read_bytes": None,
                "lance_read_iops": None,
                "lookup_calls": scan_jobs,
                "fetch_calls": scan_jobs,
                "requested_unique_keys": len(unique_keys),
            },
        )


def _first_value_array(value: float | None, rows: int) -> pa.Array:
    if rows == 0:
        return pa.array([], type=pa.float64())
    return pa.concat_arrays([pa.array([value], type=pa.float64()), pa.nulls(rows - 1, type=pa.float64())])


def _aggregate_actor_metric(name: str, values: list[float]) -> float | None:
    if not values or name in _ACTOR_DERIVED_METRICS:
        return None
    if name in _ACTOR_MAX_METRICS:
        return max(values)
    return float(sum(values))


def _metric_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if not isinstance(numerator, int | float) or not isinstance(denominator, int | float):
        return None
    return float(numerator) / float(denominator) if denominator else 0.0


class _PersistentGpuFetchActor:
    """Ray Data callable actor owning one persistent GPU index and Lance session."""

    def __init__(self, config: dict[str, Any], warmup_rows: int) -> None:
        from nemo_curator.stages.interleaved import (
            GpuLanceColumnFetchStage,
            GpuLanceIndexCacheConfig,
            LanceDatasetConfig,
            LanceIndexCacheConfig,
            LanceIndexMirrorContract,
        )

        started = time.perf_counter()
        dataset = LanceDatasetConfig(**config["dataset"])
        cache_config = dict(config["index_cache"])
        mirror_contract = cache_config.pop("mirror_contract", None)
        cache = LanceIndexCacheConfig(
            **cache_config,
            mirror_contract=(LanceIndexMirrorContract(**mirror_contract) if mirror_contract is not None else None),
        )
        self.stage = GpuLanceColumnFetchStage(
            dataset=dataset,
            index_cache=cache,
            input_key_column="source_ref",
            columns=config["columns"],
            presence_column=_PRESENT,
            missing_key_policy="mark",
            lookup_batch_size=config["lookup_batch_size"],
            fetch_batch_size=config["fetch_batch_size"],
            max_pending_takes=config["max_pending_takes"],
            payload_read_mode=config["payload_read_mode"],
            medium_density_threshold=config["medium_density_threshold"],
            high_density_threshold=config["high_density_threshold"],
            max_coalesced_range_gap=config["max_coalesced_range_gap"],
            take_scan_batch_readahead=config["take_scan_batch_readahead"],
            validate_payload_keys=config["validate_payload_keys"],
            reference_files=config["reference_files"],
            reference_key_column=config["reference_key_column"],
            reference_row_id_column=config["reference_row_id_column"],
            reference_storage_options=config["reference_storage_options"],
            reference_manifest_uri=config["reference_manifest_uri"],
            reference_manifest_sha256=config["reference_manifest_sha256"],
            expected_reference_rows=config["expected_reference_rows"],
            load_factor=config["gpu_load_factor"],
            gpu_index_cache=GpuLanceIndexCacheConfig(**config["gpu_index_cache"]),
        )
        self.stage.setup_on_node()
        self.stage.setup()
        self.setup_seconds = time.perf_counter() - started
        self.warmup_rows = warmup_rows
        self.first_batch = True

    def __call__(self, table: pa.Table) -> pa.Table:
        from nemo_curator.tasks import InterleavedBatch

        warmup_seconds = 0.0
        if self.first_batch and self.warmup_rows:
            warmup = table.slice(0, min(table.num_rows, self.warmup_rows))
            started = time.perf_counter()
            self.stage.process(InterleavedBatch(dataset_name="gpu-lance-ray-actor-warmup", data=warmup))
            warmup_seconds = time.perf_counter() - started
            self.stage._consume_custom_metrics()
        started = time.perf_counter()
        process_started_epoch = time.time()
        output = self.stage.process(InterleavedBatch(dataset_name="gpu-lance-ray-actor", data=table)).to_pyarrow()
        process_ended_epoch = time.time()
        process_seconds = time.perf_counter() - started
        metrics = self.stage._consume_custom_metrics()
        setup_seconds = self.setup_seconds if self.first_batch else None
        self.first_batch = False
        output = output.append_column(
            f"{_ACTOR_PREFIX}setup_seconds", _first_value_array(setup_seconds, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}warmup_seconds", _first_value_array(warmup_seconds, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}process_seconds", _first_value_array(process_seconds, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}process_started_epoch", _first_value_array(process_started_epoch, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}process_ended_epoch", _first_value_array(process_ended_epoch, output.num_rows)
        )
        for name in _ACTOR_METRICS:
            output = output.append_column(
                f"{_ACTOR_PREFIX}{name}", _first_value_array(metrics.get(name), output.num_rows)
            )
        return output

    def __del__(self) -> None:
        if hasattr(self, "stage"):
            self.stage.teardown()


class _PersistentLanceRayGpuFetchActor:
    """Ray Data callable actor owning lance-ray's public GPU fetcher."""

    def __init__(self, config: dict[str, Any], warmup_rows: int) -> None:
        from lance_ray import GpuLanceColumnFetcher, GpuLanceFetchConfig

        started = time.perf_counter()
        self.fetcher = GpuLanceColumnFetcher(GpuLanceFetchConfig(**config))
        self.setup_seconds = time.perf_counter() - started
        self.warmup_rows = warmup_rows
        self.first_batch = True

    @staticmethod
    def _benchmark_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
        read_iops = int(metrics.get("lance_read_iops", 0))
        read_bytes = int(metrics.get("lance_read_bytes", 0))
        return {
            "lance_lookup_seconds": metrics.get("lookup_seconds"),
            "lance_fetch_seconds": metrics.get("payload_fetch_seconds"),
            "lance_fetched_bytes": metrics.get("payload_bytes"),
            "lance_read_bytes": read_bytes,
            "lance_read_iops": read_iops,
            "requested_unique_keys": metrics.get("unique_keys"),
            "found_unique_keys": metrics.get("found_unique_keys"),
            "gpu_key_transfer_seconds": metrics.get("gpu_key_transfer_seconds"),
            "gpu_key_probe_seconds": metrics.get("gpu_key_probe_seconds"),
            "gpu_row_id_search_seconds": metrics.get("gpu_row_id_search_seconds"),
            "gpu_row_id_gather_seconds": metrics.get("gpu_row_id_gather_seconds"),
            "payload_take_calls": metrics.get("payload_take_calls"),
            "payload_read_calls": metrics.get("payload_read_calls"),
            "payload_take_rows": metrics.get("payload_take_rows"),
            "rows_per_payload_take": metrics.get("rows_per_payload_take"),
            "rows_per_payload_read": metrics.get("rows_per_payload_read"),
            "max_pending_payload_reads": metrics.get("max_pending_payload_reads"),
            "strategy_sparse_fragments": metrics.get("strategy_sparse_fragments"),
            "strategy_range_fragments": metrics.get("strategy_range_fragments"),
            "strategy_sequential_fragments": metrics.get("strategy_sequential_fragments"),
            "take_rows_calls": metrics.get("take_rows_calls"),
            "fragment_take_calls": metrics.get("fragment_take_calls"),
            "fragment_scan_calls": metrics.get("fragment_scan_calls"),
            "fragment_scan_batches": metrics.get("fragment_scan_batches"),
            "fragment_take_ranges": metrics.get("fragment_take_ranges"),
            "planned_fragment_read_rows": metrics.get("planned_fragment_read_rows"),
            "take_scan_calls": metrics.get("take_scan_calls"),
            "take_scan_ranges": metrics.get("take_scan_ranges"),
            "planned_scan_rows": metrics.get("planned_scan_rows"),
            "range_overread_rows": metrics.get("range_overread_rows"),
            "duplicate_queries_coalesced": metrics.get("duplicate_queries_coalesced"),
            "average_physical_read_bytes": read_bytes / read_iops if read_iops else 0.0,
            "read_amplification": metrics.get("read_amplification"),
            "sparse_calls_avoided": metrics.get("sparse_calls_avoided"),
        }

    def __call__(self, table: pa.Table) -> pa.Table:
        warmup_seconds = 0.0
        if self.first_batch and self.warmup_rows:
            warmup = table.slice(0, min(table.num_rows, self.warmup_rows))
            started = time.perf_counter()
            self.fetcher(warmup)
            warmup_seconds = time.perf_counter() - started

        started = time.perf_counter()
        process_started_epoch = time.time()
        output = self.fetcher(table)
        process_ended_epoch = time.time()
        process_seconds = time.perf_counter() - started
        metrics = self._benchmark_metrics(self.fetcher.last_metrics)
        setup_seconds = self.setup_seconds if self.first_batch else None
        self.first_batch = False
        output = output.append_column(
            f"{_ACTOR_PREFIX}setup_seconds", _first_value_array(setup_seconds, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}warmup_seconds", _first_value_array(warmup_seconds, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}process_seconds", _first_value_array(process_seconds, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}process_started_epoch", _first_value_array(process_started_epoch, output.num_rows)
        )
        output = output.append_column(
            f"{_ACTOR_PREFIX}process_ended_epoch", _first_value_array(process_ended_epoch, output.num_rows)
        )
        for name in _ACTOR_METRICS:
            output = output.append_column(
                f"{_ACTOR_PREFIX}{name}", _first_value_array(metrics.get(name), output.num_rows)
            )
        return output

    def process(self, table: pa.Table) -> pa.Table:
        """Explicit Ray actor method; the instance remains alive across runs."""
        return self(table)

    def ready(self) -> dict[str, float]:
        """Block actor-pool setup until the GPU index is fully resident."""
        return {"setup_seconds": self.setup_seconds}

    def close(self) -> None:
        self.fetcher.close()

    def __del__(self) -> None:
        if hasattr(self, "fetcher"):
            self.fetcher.close()


def _lance_ray_gpu_config(settings: BenchmarkSettings) -> dict[str, Any]:
    expected_rows = settings.expected_reference_rows
    if expected_rows is None:
        msg = "lance-ray GPU fetch requires --expected-reference-rows"
        raise ArmUnavailableError(msg)
    return {
        "dataset_uri": settings.image_lance_uri,
        "dataset_version": settings.image_lance_version,
        "sidecar_files": tuple(settings.reference_files),
        "sidecar_manifest_uri": settings.reference_manifest_uri,
        "sidecar_manifest_sha256": settings.reference_manifest_sha256,
        "columns": _payload_projection(settings),
        "expected_reference_rows": expected_rows,
        "input_key_column": "source_ref",
        "dataset_key_column": settings.key_column,
        "sidecar_key_column": settings.reference_key_column,
        "sidecar_row_id_column": settings.reference_row_id_column,
        "presence_column": _PRESENT,
        "missing_key_policy": "mark",
        "dataset_storage_options": settings.storage_options,
        "sidecar_storage_options": settings.reference_storage_options,
        "load_factor": settings.gpu_load_factor,
        "max_lookup_bytes": settings.max_lookup_bytes,
        "fetch_batch_size": settings.fetch_batch_size,
        "io_threads": settings.io_threads,
        "max_pending_fetch_batches": settings.max_pending_fetch_batches,
        "payload_read_mode": settings.payload_read_mode,
        "medium_density_threshold": settings.medium_density_threshold,
        "high_density_threshold": settings.high_density_threshold,
        "max_coalesced_range_gap": settings.max_coalesced_range_gap,
        "take_scan_batch_readahead": settings.take_scan_batch_readahead,
        "validate_payload_keys": settings.validate_payload_keys,
        "index_cache_size_bytes": settings.index_cache_size_bytes,
        "metadata_cache_size_bytes": settings.metadata_cache_size_bytes,
        "include_metrics_metadata": False,
    }


class LanceRayGpuFetcherArm(BenchmarkArm):
    """One persistent public lance-ray GPU fetcher without Ray scheduling."""

    name = "lance_ray_gpu_fetcher"

    def setup(self) -> None:
        if not self.settings.reference_files:
            msg = "GPU sidecar reference files were not provided"
            raise ArmUnavailableError(msg)
        try:
            import lance
            from lance_ray import GpuLanceColumnFetcher, GpuLanceFetchConfig
        except ImportError as exc:
            msg = f"lance-ray GPU API is unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        config = GpuLanceFetchConfig(**_lance_ray_gpu_config(self.settings))
        self.fetcher = GpuLanceColumnFetcher(config)
        dataset = lance.dataset(
            self.settings.image_lance_uri,
            version=self.settings.image_lance_version,
            storage_options=self.settings.storage_options or None,
        )
        self.input_table = _interleaved_input(
            self.manifest.table,
            dataset.schema.field(self.settings.key_column).type,
        )
        self.setup_metrics.update(
            {
                "reference_rows": self.fetcher._index.reference_rows,
                "reference_load_seconds": self.fetcher._index.load_seconds,
                "reference_build_seconds": self.fetcher._index.build_seconds,
                "reference_gpu_bytes": self.fetcher._index.gpu_bytes,
            }
        )

    def run(self) -> ArmRun:
        output = self.fetcher(self.input_table)
        metrics = _PersistentLanceRayGpuFetchActor._benchmark_metrics(self.fetcher.last_metrics)
        metrics.update(
            {
                "lookup_seconds": self.fetcher.last_metrics["lookup_seconds"],
                "fetch_seconds": self.fetcher.last_metrics["payload_fetch_seconds"],
                "lookup_calls": self.fetcher.last_metrics["lookup_windows"],
                "fetch_calls": self.fetcher.last_metrics["payload_take_calls"],
            }
        )
        return ArmRun(output, metrics)

    def close(self) -> None:
        if hasattr(self, "fetcher"):
            self.fetcher.close()


class RayDataActorArm(BenchmarkArm):
    """GPU fetch stage executed by a persistent Ray Data map_batches actor pool."""

    name = "ray_data_persistent_gpu_actor"
    actor_class = _PersistentGpuFetchActor

    def _build_actor_config(self) -> dict[str, Any]:
        return {
            "dataset": {
                "uri": self.settings.image_lance_uri,
                "version": self.settings.image_lance_version,
                "key_column": self.settings.key_column,
                "index_name": self.settings.index_name,
                "storage_options": self.settings.storage_options,
            },
            "index_cache": {
                "mirror_path": self.settings.index_mirror,
                "mirror_contract": self.settings.index_mirror_contract,
                "copy_to_node_local": self.settings.copy_index_to_node_local,
                "node_local_root": self.settings.index_node_local_root,
                "prewarm": self.settings.prewarm_index,
                "index_cache_size_bytes": self.settings.index_cache_size_bytes,
                "metadata_cache_size_bytes": self.settings.metadata_cache_size_bytes,
            },
            "columns": _payload_projection(self.settings),
            "lookup_batch_size": self.settings.lookup_batch_size,
            "fetch_batch_size": self.settings.fetch_batch_size,
            "max_pending_takes": self.settings.max_pending_fetch_batches,
            "payload_read_mode": self.settings.payload_read_mode,
            "medium_density_threshold": self.settings.medium_density_threshold,
            "high_density_threshold": self.settings.high_density_threshold,
            "max_coalesced_range_gap": self.settings.max_coalesced_range_gap,
            "take_scan_batch_readahead": self.settings.take_scan_batch_readahead,
            "validate_payload_keys": self.settings.validate_payload_keys,
            "reference_files": self.settings.reference_files,
            "reference_key_column": self.settings.reference_key_column,
            "reference_row_id_column": self.settings.reference_row_id_column,
            "reference_storage_options": self.settings.reference_storage_options,
            "reference_manifest_uri": self.settings.reference_manifest_uri,
            "reference_manifest_sha256": self.settings.reference_manifest_sha256,
            "expected_reference_rows": self.settings.expected_reference_rows,
            "gpu_load_factor": self.settings.gpu_load_factor,
            "gpu_index_cache": {
                "copy_to_node_local": self.settings.copy_reference_to_node_local,
                "node_local_root": self.settings.reference_node_local_root,
            },
        }

    def setup(self) -> None:
        if not self.settings.reference_files:
            msg = "GPU sidecar reference files were not provided"
            raise ArmUnavailableError(msg)
        if not self.settings.reference_manifest_uri or not self.settings.reference_manifest_sha256:
            msg = "GPU sidecar manifest URI and SHA-256 were not provided"
            raise ArmUnavailableError(msg)
        self.ray, started, startup_seconds = _ensure_ray(self.settings.ray_address, self.settings.ray_temp_dir)
        resources = self.ray.cluster_resources()
        advertised_gpus = float(resources.get("GPU", 0.0))
        if advertised_gpus < self.settings.ray_gpu_actors:
            msg = (
                f"Ray cluster advertises {advertised_gpus:g} GPUs; "
                f"--ray-gpu-actors requires {self.settings.ray_gpu_actors}"
            )
            raise ArmUnavailableError(msg)
        try:
            import cudf  # noqa: F401
            import cupy  # noqa: F401
            import lance
            import pylibcudf  # noqa: F401
            from ray.data import ActorPoolStrategy
        except ImportError as exc:
            msg = f"persistent GPU actor dependencies are unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        self.actor_pool_strategy = ActorPoolStrategy(
            size=self.settings.ray_gpu_actors,
            max_tasks_in_flight_per_actor=1,
        )
        dataset = lance.dataset(
            self.settings.image_lance_uri,
            version=self.settings.image_lance_version,
            storage_options=self.settings.storage_options or None,
        )
        self.input_table = _interleaved_input(self.manifest.table, dataset.schema.field(self.settings.key_column).type)
        self.actor_config = self._build_actor_config()
        self.setup_metrics.update(
            {
                "ray_started_by_benchmark": started,
                "ray_startup_seconds": startup_seconds,
                "ray_cluster_gpus": advertised_gpus,
                "ray_gpu_actors": self.settings.ray_gpu_actors,
            }
        )

    def run(self) -> ArmRun:
        # Each left-table chunk remains one Ray input block. Ray may bundle
        # adjacent blocks up to window_rows before assigning an actor call.
        input_blocks = _chunk_tables(self.input_table, self.settings.task_rows)
        dataset = self.ray.data.from_arrow(input_blocks)
        mapped = dataset.map_batches(
            self.actor_class,
            batch_format="pyarrow",
            batch_size=self.settings.window_rows,
            compute=self.actor_pool_strategy,
            fn_constructor_kwargs={
                "config": self.actor_config,
                "warmup_rows": self.settings.actor_warmup_rows,
            },
            num_cpus=1,
            num_gpus=1,
        )
        output_batches = [_as_arrow_table(batch) for batch in mapped.iter_batches(batch_format="pyarrow")]
        return self._finish_actor_run(output_batches, len(input_blocks))

    def _finish_actor_run(
        self,
        output_batches: list[pa.Table],
        input_block_count: int,
        *,
        actors_used: int | None = None,
    ) -> ArmRun:
        """Aggregate actor metadata and restore the original Arrow row order."""
        output = pa.concat_tables(output_batches)
        metrics: dict[str, float | int | None] = {}
        metadata_columns = [name for name in output.column_names if name.startswith(_ACTOR_PREFIX)]
        metric_values: dict[str, list[float]] = {}
        for name in metadata_columns:
            values = [float(value) for value in pc.drop_null(output[name]).to_pylist()]
            metric_name = name.removeprefix(_ACTOR_PREFIX)
            metric_values[metric_name] = values
            metrics[metric_name] = _aggregate_actor_metric(metric_name, values)
        output = output.drop_columns(metadata_columns)

        setup_values = metric_values.get("setup_seconds", [])
        warmup_values = metric_values.get("warmup_seconds", [])
        process_values = metric_values.get("process_seconds", [])
        process_starts = metric_values.get("process_started_epoch", [])
        process_ends = metric_values.get("process_ended_epoch", [])
        metrics["actor_setup_seconds_sum"] = sum(setup_values)
        metrics["actor_warmup_seconds_sum"] = sum(warmup_values)
        metrics["actor_process_seconds_sum"] = sum(process_values)
        metrics["setup_seconds"] = max(setup_values, default=0.0)
        metrics["warmup_seconds"] = max(warmup_values, default=0.0)
        if process_starts and process_ends:
            metrics["process_seconds"] = max(process_ends) - min(process_starts)
        else:
            metrics["process_seconds"] = sum(process_values)
        metrics.pop("process_started_epoch", None)
        metrics.pop("process_ended_epoch", None)
        metrics["rows_per_payload_take"] = _metric_ratio(
            metrics.get("payload_take_rows"), metrics.get("payload_take_calls")
        )
        metrics["rows_per_payload_read"] = _metric_ratio(
            metrics.get("payload_take_rows"), metrics.get("payload_read_calls")
        )
        metrics["rows_per_private_take"] = _metric_ratio(
            metrics.get("private_take_rows"), metrics.get("private_take_calls")
        )
        metrics["average_physical_read_bytes"] = _metric_ratio(
            metrics.get("lance_read_bytes"), metrics.get("lance_read_iops")
        )
        metrics["read_amplification"] = _metric_ratio(
            metrics.get("lance_read_bytes"), metrics.get("lance_fetched_bytes")
        )

        raw_ordinals = output[_ORDINAL].combine_chunks().to_pylist()
        raw_order_matches = raw_ordinals == list(range(self.input_table.num_rows))
        if not raw_order_matches:
            output = output.take(pc.sort_indices(output, sort_keys=[(_ORDINAL, "ascending")]))
        actor_calls = len(process_values)
        metrics.update(
            {
                "lookup_seconds": metrics.get("lance_lookup_seconds"),
                "fetch_seconds": metrics.get("lance_fetch_seconds"),
                "lance_read_bytes": metrics.get("lance_read_bytes"),
                "lance_read_iops": metrics.get("lance_read_iops"),
                "lookup_calls": actor_calls,
                "fetch_calls": actor_calls,
                "ray_input_blocks": input_block_count,
                "ray_gpu_actors_requested": self.settings.ray_gpu_actors,
                "ray_gpu_actors_used": len(setup_values) if actors_used is None else actors_used,
                "ray_raw_output_order_matches": float(raw_order_matches),
            }
        )
        return ArmRun(output, metrics)


class LanceRayGpuActorArm(RayDataActorArm):
    """lance-ray's public cuDF/private-PyLance actor in a Ray Data pool."""

    name = "lance_ray_gpu_actor"
    actor_class = _PersistentLanceRayGpuFetchActor

    def setup(self) -> None:
        try:
            from lance_ray import GpuLanceColumnFetcher, GpuLanceFetchConfig
        except ImportError as exc:
            msg = f"lance-ray GPU API is unavailable: {exc}"
            raise ArmUnavailableError(msg) from exc
        del GpuLanceColumnFetcher, GpuLanceFetchConfig
        if self.settings.expected_reference_rows is None:
            msg = "lance-ray GPU arm requires --expected-reference-rows"
            raise ArmUnavailableError(msg)
        super().setup()
        input_blocks = _chunk_tables(self.input_table, self.settings.task_rows)
        actor_count = self.settings.ray_gpu_actors
        if len(input_blocks) % actor_count:
            msg = f"{len(input_blocks)} input blocks cannot be balanced across {actor_count} persistent actors"
            raise ValueError(msg)
        self._actor_blocks = [[] for _ in range(actor_count)]
        for block_index, block in enumerate(input_blocks):
            self._actor_blocks[block_index % actor_count].append(block)
        if len({len(blocks) for blocks in self._actor_blocks}) != 1:
            msg = "persistent actor input blocks are not balanced"
            raise RuntimeError(msg)

        remote_actor = self.ray.remote(num_cpus=1, num_gpus=1)(self.actor_class)
        self._persistent_actors = [remote_actor.remote(self.actor_config, 0) for _ in range(actor_count)]
        try:
            ready = self.ray.get([actor.ready.remote() for actor in self._persistent_actors])
            prewarm_started = time.perf_counter()
            if self.settings.actor_warmup_rows:
                warmup_refs = [
                    actor.process.remote(blocks[0].slice(0, self.settings.actor_warmup_rows))
                    for actor, blocks in zip(
                        self._persistent_actors,
                        self._actor_blocks,
                        strict=True,
                    )
                ]
                self.ray.get(warmup_refs)
            prewarm_seconds = time.perf_counter() - prewarm_started
        except Exception:
            for actor in self._persistent_actors:
                self.ray.kill(actor, no_restart=True)
            self._persistent_actors = []
            raise
        self.setup_metrics.update(
            {
                "persistent_actor_pool": True,
                "persistent_actor_count": actor_count,
                "persistent_actor_setup_seconds_max": max(
                    (float(item["setup_seconds"]) for item in ready),
                    default=0.0,
                ),
                "persistent_actor_prewarm_seconds": prewarm_seconds,
            }
        )

    def run(self) -> ArmRun:
        outputs: list[pa.Table] = []
        blocks_per_actor = len(self._actor_blocks[0])
        blocks_per_wave = min(self.settings.coalesce_tasks, blocks_per_actor)
        for start in range(0, blocks_per_actor, blocks_per_wave):
            pending = [
                actor.process.remote(pa.concat_tables(blocks[start : start + blocks_per_wave]))
                for actor, blocks in zip(
                    self._persistent_actors,
                    self._actor_blocks,
                    strict=True,
                )
            ]
            outputs.extend(self.ray.get(pending))
        return self._finish_actor_run(
            outputs,
            sum(len(blocks) for blocks in self._actor_blocks),
            actors_used=len(self._persistent_actors),
        )

    def close(self) -> None:
        actors = getattr(self, "_persistent_actors", [])
        try:
            if actors:
                self.ray.get([actor.close.remote() for actor in actors])
        finally:
            for actor in actors:
                self.ray.kill(actor, no_restart=True)
            self._persistent_actors = []

    def _build_actor_config(self) -> dict[str, Any]:
        return _lance_ray_gpu_config(self.settings)


def _as_bytes(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return None


def _normalized_md5(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("ascii").lower()
        except UnicodeDecodeError:
            return value.hex().lower()
    return str(value).lower()


def _validate_output(output: pa.Table, manifest: QueryManifest) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    expected_rows = manifest.table.num_rows
    ordinal = output[_ORDINAL].combine_chunks().to_pylist() if _ORDINAL in output.column_names else []
    output_keys = output["source_ref"].combine_chunks().to_pylist() if "source_ref" in output.column_names else []
    expected_keys = manifest.table["source_ref"].combine_chunks().to_pylist()
    order_matches = ordinal == list(range(expected_rows)) and output_keys == expected_keys
    present = (
        output[_PRESENT].combine_chunks().to_pylist() if _PRESENT in output.column_names else [False] * output.num_rows
    )
    payloads = (
        output[_FETCHED["image"]].combine_chunks().to_pylist() if _FETCHED["image"] in output.column_names else []
    )
    fetched_md5 = (
        output[_FETCHED["md5"]].combine_chunks().to_pylist()
        if _FETCHED["md5"] in output.column_names
        else [None] * output.num_rows
    )
    fetched_width = (
        output[_FETCHED["width"]].combine_chunks().to_pylist()
        if _FETCHED["width"] in output.column_names
        else [None] * output.num_rows
    )
    fetched_height = (
        output[_FETCHED["height"]].combine_chunks().to_pylist()
        if _FETCHED["height"] in output.column_names
        else [None] * output.num_rows
    )
    expected_md5 = (
        manifest.table[manifest.expected_columns["md5"]].combine_chunks().to_pylist()
        if "md5" in manifest.expected_columns
        else [None] * expected_rows
    )
    expected_width = (
        manifest.table[manifest.expected_columns["width"]].combine_chunks().to_pylist()
        if "width" in manifest.expected_columns
        else [None] * expected_rows
    )
    expected_height = (
        manifest.table[manifest.expected_columns["height"]].combine_chunks().to_pylist()
        if "height" in manifest.expected_columns
        else [None] * expected_rows
    )

    digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    payload_bytes = 0
    missing_payloads = 0
    md5_checked = 0
    md5_mismatch_count = 0
    md5_mismatches: list[int] = []
    metadata_dimension_checked = 0
    metadata_dimension_mismatch_count = 0
    metadata_dimension_mismatches: list[int] = []
    decoded_checked = 0
    decoded_mismatch_count = 0
    decoded_mismatches: list[int] = []
    decode_safety_skipped_count = 0
    decode_safety_skipped_rows: list[int] = []
    pillow_error: str | None = None
    try:
        from PIL import Image
    except ImportError as exc:
        image_available = False
        pillow_error = str(exc)
    else:
        image_available = True

    for index in range(min(output.num_rows, expected_rows)):
        payload = _as_bytes(payloads[index]) if index < len(payloads) else None
        digest.update(
            json.dumps(
                [
                    ordinal[index],
                    output_keys[index],
                    present[index],
                    fetched_md5[index],
                    fetched_width[index],
                    fetched_height[index],
                ],
                separators=(",", ":"),
                default=str,
            ).encode()
        )
        payload_digest.update(
            json.dumps(
                [ordinal[index], output_keys[index], present[index]],
                separators=(",", ":"),
                default=str,
            ).encode()
        )
        if payload is None:
            missing_payloads += 1
            digest.update(b"null")
            payload_digest.update(b"null")
            continue
        payload_bytes += len(payload)
        payload_sha = hashlib.sha256(payload).digest()
        digest.update(payload_sha)
        payload_digest.update(payload_sha)
        actual_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        expected_hashes = {
            value
            for value in (_normalized_md5(expected_md5[index]), _normalized_md5(fetched_md5[index]))
            if value is not None
        }
        if expected_hashes:
            md5_checked += 1
            if actual_md5 not in expected_hashes or len(expected_hashes) > 1:
                md5_mismatch_count += 1
                if len(md5_mismatches) < _MAX_MISMATCH_EXAMPLES:
                    md5_mismatches.append(index)

        expected_size = (expected_width[index], expected_height[index])
        fetched_size = (fetched_width[index], fetched_height[index])
        if all(value is not None for value in expected_size):
            metadata_dimension_checked += 1
            if not all(value is not None for value in fetched_size) or tuple(map(int, fetched_size)) != tuple(
                map(int, expected_size)
            ):
                metadata_dimension_mismatch_count += 1
                if len(metadata_dimension_mismatches) < _MAX_MISMATCH_EXAMPLES:
                    metadata_dimension_mismatches.append(index)
        if image_available:
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    actual_size = image.size
                comparison_size = expected_size if all(value is not None for value in expected_size) else fetched_size
                if all(value is not None for value in comparison_size):
                    decoded_checked += 1
                    if actual_size != tuple(map(int, comparison_size)):
                        decoded_mismatch_count += 1
                        if len(decoded_mismatches) < _MAX_MISMATCH_EXAMPLES:
                            decoded_mismatches.append(index)
            except Image.DecompressionBombError:
                decode_safety_skipped_count += 1
                if len(decode_safety_skipped_rows) < _MAX_MISMATCH_EXAMPLES:
                    decode_safety_skipped_rows.append(index)
            except Exception:
                decoded_mismatch_count += 1
                if len(decoded_mismatches) < _MAX_MISMATCH_EXAMPLES:
                    decoded_mismatches.append(index)

    correct = (
        output.num_rows == expected_rows
        and order_matches
        and missing_payloads == 0
        and md5_mismatch_count == 0
        and metadata_dimension_mismatch_count == 0
        and decoded_mismatch_count == 0
    )
    return {
        "correct": correct,
        "row_count": output.num_rows,
        "expected_row_count": expected_rows,
        "order_matches_manifest": order_matches,
        "output_digest_sha256": digest.hexdigest(),
        "payload_digest_sha256": payload_digest.hexdigest(),
        "present_rows": sum(value is True for value in present),
        "missing_payload_rows": missing_payloads,
        "payload_bytes": payload_bytes,
        "md5": {"checked": md5_checked, "mismatch_count": md5_mismatch_count, "mismatch_rows": md5_mismatches},
        "dimensions": {
            "metadata_checked": metadata_dimension_checked,
            "metadata_mismatch_count": metadata_dimension_mismatch_count,
            "metadata_mismatch_rows": metadata_dimension_mismatches,
            "decoded_checked": decoded_checked,
            "decoded_mismatch_count": decoded_mismatch_count,
            "decoded_mismatch_rows": decoded_mismatches,
            "decode_safety_skipped_count": decode_safety_skipped_count,
            "decode_safety_skipped_rows": decode_safety_skipped_rows,
            "decode_skipped_reason": f"Pillow unavailable: {pillow_error}" if pillow_error else None,
        },
    }


def _error_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": _redact_error_message(str(exc)),
    }


def _run_once(arm: BenchmarkArm, manifest: QueryManifest, repeat: int, order_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    arm_run = arm.run()
    wall_seconds = time.perf_counter() - started
    correctness = _validate_output(arm_run.table, manifest)
    payload_bytes = int(correctness["payload_bytes"])
    rows = manifest.table.num_rows
    metrics = dict(arm_run.metrics)
    process_seconds = metrics.get("process_seconds")
    actor_process_span_seconds = (
        float(process_seconds)
        if isinstance(process_seconds, int | float) and not isinstance(process_seconds, bool) and process_seconds > 0
        else None
    )
    return {
        "status": "completed" if correctness["correct"] else "incorrect",
        "repeat": repeat,
        "order_index": order_index,
        "wall_seconds": wall_seconds,
        "warm_process_seconds": wall_seconds,
        "actor_process_span_seconds": actor_process_span_seconds,
        "cold_setup_seconds": metrics.pop("setup_seconds", None),
        "internal_warmup_seconds": metrics.pop("warmup_seconds", None),
        "lookup_seconds": metrics.pop("lookup_seconds", None),
        "fetch_seconds": metrics.pop("fetch_seconds", None),
        "images_per_second": rows / wall_seconds if wall_seconds else None,
        "payload_mib_per_second": payload_bytes / (1024**2 * wall_seconds) if wall_seconds else None,
        "payload_bytes": payload_bytes,
        "lance_read_iops": metrics.pop("lance_read_iops", None),
        "lance_read_bytes": metrics.pop("lance_read_bytes", None),
        "lookup_calls": metrics.pop("lookup_calls", None),
        "fetch_calls": metrics.pop("fetch_calls", None),
        "correctness": correctness,
        "backend_metrics": metrics,
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _backend_median(repeats: list[dict[str, Any]], *metric_names: str) -> float | None:
    values = []
    for repeat in repeats:
        metrics = repeat.get("backend_metrics") or {}
        value = next((metrics.get(metric) for metric in metric_names if metrics.get(metric) is not None), None)
        if isinstance(value, int | float):
            values.append(float(value))
    return statistics.median(values) if values else None


def _repeat_eligibility_errors(repeat: dict[str, Any], expected_rows: int, repeat_index: int) -> list[str]:
    prefix = f"repeat[{repeat_index}]"
    errors: list[str] = []
    if repeat.get("status") != "completed":
        errors.append(f"{prefix}.status is not completed")
    correctness = repeat.get("correctness")
    if not isinstance(correctness, dict):
        errors.append(f"{prefix}.correctness is missing")
        return errors
    expected = {
        "correct": True,
        "row_count": expected_rows,
        "expected_row_count": expected_rows,
        "order_matches_manifest": True,
        "present_rows": expected_rows,
        "missing_payload_rows": 0,
    }
    for field, value in expected.items():
        if correctness.get(field) != value:
            errors.append(f"{prefix}.correctness.{field}={correctness.get(field)!r}; expected {value!r}")
    for field in ("output_digest_sha256", "payload_digest_sha256"):
        digest = correctness.get(field)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            errors.append(f"{prefix}.correctness.{field} is not a lowercase SHA-256")
    return errors


def _arm_summary_eligibility_errors(report: dict[str, Any], arm_name: str, arm_result: dict[str, Any]) -> list[str]:  # noqa: C901
    errors: list[str] = []
    if report.get("status") != "completed":
        errors.append(f"benchmark status is {report.get('status')!r}, not 'completed'")
    if arm_result.get("status") != "completed":
        errors.append(f"arm status is {arm_result.get('status')!r}, not 'completed'")
    repeat_count = report.get("configuration", {}).get("repeat_count")
    if not isinstance(repeat_count, int) or isinstance(repeat_count, bool) or repeat_count <= 0:
        errors.append("configuration.repeat_count is not a positive integer")
        return errors
    repeats = arm_result.get("repeats")
    if not isinstance(repeats, list):
        errors.append("arm repeats are missing")
        return errors
    if len(repeats) != repeat_count:
        errors.append(f"arm has {len(repeats)} repeats; expected exactly {repeat_count}")
    expected_rows = report.get("manifest", {}).get("rows")
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows <= 0:
        errors.append("manifest.rows is not a positive integer")
        return errors
    for repeat_index, repeat in enumerate(repeats):
        if not isinstance(repeat, dict):
            errors.append(f"repeat[{repeat_index}] is not an object")
            continue
        errors.extend(_repeat_eligibility_errors(repeat, expected_rows, repeat_index))
    for digest_field in ("output_digest_sha256", "payload_digest_sha256"):
        digests = {
            repeat.get("correctness", {}).get(digest_field)
            for repeat in repeats
            if isinstance(repeat, dict) and isinstance(repeat.get("correctness"), dict)
        }
        if len(digests) != 1 or any(
            not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None for digest in digests
        ):
            errors.append(f"arm {arm_name} does not have one stable {digest_field}")
    return list(dict.fromkeys(errors))


def _independent_md5_complete(arm_result: dict[str, Any], expected_rows: int) -> bool:
    for repeat in arm_result.get("repeats", []):
        correctness = repeat.get("correctness", {})
        md5 = correctness.get("md5", {})
        if md5.get("checked") != expected_rows or md5.get("mismatch_count") != 0:
            return False
    return True


def _summarize(report: dict[str, Any]) -> None:  # noqa: C901, PLR0912, PLR0915
    correctness_digests: dict[str, str] = {}
    payload_digests: dict[str, str] = {}
    eligible_arms: set[str] = set()
    for arm_name, arm_result in report["arms"].items():
        eligibility_errors = _arm_summary_eligibility_errors(report, arm_name, arm_result)
        arm_result["summary_eligibility"] = {
            "eligible": not eligibility_errors,
            "errors": eligibility_errors,
        }
        if eligibility_errors:
            arm_result["summary"] = {"eligible": False}
            continue
        completed = arm_result["repeats"]
        arm_result["summary"] = {"eligible": True}
        arm_result["summary"].update(
            {
                name: _stats([float(item[name]) for item in completed if isinstance(item.get(name), int | float)])
                for name in (
                    "wall_seconds",
                    "warm_process_seconds",
                    "actor_process_span_seconds",
                    "cold_setup_seconds",
                    "internal_warmup_seconds",
                    "lookup_seconds",
                    "fetch_seconds",
                    "images_per_second",
                    "payload_mib_per_second",
                    "lance_read_iops",
                    "lance_read_bytes",
                    "lookup_calls",
                    "fetch_calls",
                )
            }
        )
        output_digest = completed[0]["correctness"]["output_digest_sha256"]
        payload_digest = completed[0]["correctness"]["payload_digest_sha256"]
        arm_result["summary"]["stable_correctness_digest"] = True
        arm_result["summary"]["stable_payload_digest"] = True
        correctness_digests[arm_name] = output_digest
        payload_digests[arm_name] = payload_digest
        eligible_arms.add(arm_name)

    report["correctness_digests"] = correctness_digests
    report["payload_digests"] = payload_digests
    report["cross_arm_correctness_digest_match"] = (
        len(eligible_arms) == len(report["arms"]) and len(set(correctness_digests.values())) == 1
        if correctness_digests
        else None
    )
    report["cross_arm_payload_digest_match"] = (
        len(eligible_arms) == len(report["arms"]) and len(set(payload_digests.values())) == 1
        if payload_digests
        else None
    )

    report["speedups"] = {}
    report["comparison_eligibility"] = {}
    baselines = ("naive_pylance_scalar", "lance_ray_datasource")
    for baseline in baselines:
        baseline_result = report["arms"].get(baseline)
        if not baseline_result or baseline not in eligible_arms:
            continue
        baseline_stats = baseline_result["summary"]["wall_seconds"]
        for name, arm_result in report["arms"].items():
            if name == baseline:
                continue
            reasons: list[str] = []
            if name not in eligible_arms:
                reasons.append("candidate summary is not terminally eligible")
            if baseline_result.get("payload_projection") != arm_result.get("payload_projection"):
                reasons.append("payload projection differs")
            baseline_projection = baseline_result.get("payload_projection", {})
            if not isinstance(baseline_projection, dict) or baseline_projection.get("mode") != "url_image_matched":
                reasons.append("comparison projection is not matched url+image")
            if payload_digests.get(baseline) != payload_digests.get(name):
                reasons.append("payload correctness digest differs")
            if baseline == "lance_ray_datasource" or name == "lance_ray_datasource":
                reasons.append(
                    "lance-ray DataSource cache reuse is opportunistic per Ray worker and cache hits are not "
                    "bound into this report"
                )
            expected_rows = report["manifest"]["rows"]
            expected_columns = report["manifest"].get("expected_columns", {})
            if "md5" not in expected_columns:
                reasons.append("query manifest has no independent expected MD5 column")
            elif not _independent_md5_complete(baseline_result, expected_rows) or not _independent_md5_complete(
                arm_result, expected_rows
            ):
                reasons.append("independent MD5 validation is incomplete")
            comparison_name = f"{name}:vs_{baseline}"
            report["comparison_eligibility"][comparison_name] = {
                "eligible": not reasons,
                "errors": reasons,
            }
            candidate = arm_result.get("summary", {}).get("wall_seconds")
            if not reasons and candidate and candidate["median"]:
                report["speedups"].setdefault(name, {})[f"vs_{baseline}"] = (
                    baseline_stats["median"] / candidate["median"]
                )

    report["sparse_read_goal"] = {}
    for name, arm_result in report["arms"].items():
        if name not in eligible_arms:
            report["sparse_read_goal"][name] = {"eligible": False}
            continue
        summary = arm_result.get("summary", {})
        stage_windows = summary.get("fetch_calls")
        read_iops = summary.get("lance_read_iops")
        repeats = arm_result.get("repeats", [])
        private_takes = _backend_median(repeats, "private_take_calls", "payload_take_calls")
        report["sparse_read_goal"][name] = {
            "eligible": True,
            "median_stage_windows": stage_windows["median"] if stage_windows else None,
            "median_private_take_calls": private_takes,
            "median_take_rows_calls": _backend_median(repeats, "take_rows_calls"),
            "median_take_scan_calls": _backend_median(repeats, "take_scan_calls"),
            "median_physical_reads": read_iops["median"] if read_iops else None,
            "query_rows_per_private_take": (report["manifest"]["rows"] / private_takes if private_takes else None),
        }


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def _finalize_report_status(report: dict[str, Any]) -> None:
    arm_results = report["arms"].values()
    all_arms_completed = bool(report["arms"]) and all(result.get("status") == "completed" for result in arm_results)
    if not all_arms_completed or report.get("teardown_errors"):
        report["status"] = "failed"
        return
    report["status"] = "completed"
    terminal_errors = {
        arm_name: errors
        for arm_name, arm_result in report["arms"].items()
        if (errors := _arm_summary_eligibility_errors(report, arm_name, arm_result))
    }
    if terminal_errors:
        report["status"] = "failed"
        report["terminal_eligibility_errors"] = terminal_errors


def _package_versions() -> dict[str, str | None]:
    packages = ("nemo-curator", "pylance", "lance-ray", "ray", "pyarrow", "cudf-cu12", "pillow")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _make_settings(args: argparse.Namespace, references: list[str]) -> BenchmarkSettings:
    validate_credential_free_uri_identity(args.image_lance_uri, "image Lance URI")
    if args.reference_manifest_uri is not None:
        validate_credential_free_uri_identity(args.reference_manifest_uri, "reference manifest URI")
    if args.index_mirror is not None:
        validate_credential_free_uri_identity(args.index_mirror, "index mirror URI")
    for reference in references:
        validate_credential_free_uri_identity(reference, "reference sidecar URI")
    source_columns = {
        name: value
        for name, value in {
            "image": args.image_column,
            "md5": args.md5_column,
            "width": args.width_column,
            "height": args.height_column,
        }.items()
        if value
    }
    if "image" not in source_columns:
        msg = "--image-column must not be empty"
        raise ValueError(msg)
    mirror_contract = _index_mirror_contract(args.index_mirror_contract_json)
    if bool(args.index_mirror) != bool(mirror_contract):
        msg = "--index-mirror and --index-mirror-contract-json must be configured together"
        raise ValueError(msg)
    if args.copy_index_to_node_local and not args.index_mirror:
        msg = "--copy-index-to-node-local requires a contract-pinned --index-mirror"
        raise ValueError(msg)
    return BenchmarkSettings(
        image_lance_uri=args.image_lance_uri,
        image_lance_version=args.image_lance_version,
        storage_options=_json_options(args.storage_options_json),
        key_column=args.key_column,
        index_name=args.index_name,
        source_columns=source_columns,
        index_mirror=args.index_mirror,
        index_mirror_contract=mirror_contract,
        copy_index_to_node_local=args.copy_index_to_node_local,
        index_node_local_root=args.index_node_local_root,
        prewarm_index=not args.no_prewarm_index,
        index_cache_size_bytes=args.index_cache_size_gib * 1024**3,
        metadata_cache_size_bytes=args.metadata_cache_size_mib * 1024**2,
        lookup_batch_size=args.lookup_batch_size,
        fetch_batch_size=args.fetch_batch_size,
        io_threads=args.io_threads,
        task_rows=args.task_rows,
        coalesce_tasks=args.coalesce_tasks,
        reference_files=references,
        reference_storage_options=_json_options(args.reference_storage_options_json),
        reference_manifest_uri=args.reference_manifest_uri,
        reference_manifest_sha256=args.reference_manifest_sha256,
        reference_key_column=args.reference_key_column,
        reference_row_id_column=args.reference_row_id_column,
        expected_reference_rows=args.expected_reference_rows,
        gpu_load_factor=args.gpu_load_factor,
        max_lookup_bytes=args.max_lookup_bytes_mib * 1024**2,
        max_pending_fetch_batches=args.max_pending_fetch_batches,
        payload_read_mode=args.payload_read_mode,
        medium_density_threshold=args.medium_density_threshold,
        high_density_threshold=args.high_density_threshold,
        max_coalesced_range_gap=args.max_coalesced_range_gap,
        take_scan_batch_readahead=args.take_scan_batch_readahead,
        validate_payload_keys=args.validate_payload_keys,
        copy_reference_to_node_local=args.copy_reference_to_node_local,
        reference_node_local_root=args.reference_node_local_root,
        ray_address=args.ray_address,
        ray_temp_dir=args.ray_temp_dir,
        ray_filter_batch_size=args.ray_filter_batch_size,
        ray_concurrency=args.ray_concurrency,
        ray_worker_dataset_cache_size=args.ray_worker_dataset_cache_size,
        public_index_fast_search=args.public_index_fast_search,
        ray_gpu_actors=args.ray_gpu_actors,
        actor_warmup_rows=args.actor_warmup_rows,
    )


def _construct_arms(
    selected: list[str], manifest: QueryManifest, settings: BenchmarkSettings
) -> dict[str, BenchmarkArm]:
    factories = {
        "naive_pylance_scalar": lambda: NaivePylanceArm(manifest, settings),
        "cpu_lance_column_fetch_stage": lambda: CuratorStageArm(manifest, settings, gpu=False),
        "gpu_lance_column_fetch_stage": lambda: CuratorStageArm(manifest, settings, gpu=True),
        "lance_ray_datasource": lambda: LanceRayDatasourceArm(manifest, settings),
        "lance_ray_gpu_fetcher": lambda: LanceRayGpuFetcherArm(manifest, settings),
        "lance_ray_gpu_actor": lambda: LanceRayGpuActorArm(manifest, settings),
        "ray_data_persistent_gpu_actor": lambda: RayDataActorArm(manifest, settings),
    }
    return {name: factories[name]() for name in selected}


def _rotated_order(names: list[str], seed: int, round_index: int) -> list[str]:
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - deterministic benchmark scheduling
    offset = round_index % len(shuffled)
    return shuffled[offset:] + shuffled[:offset]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    manifest = _load_manifest(args)
    for reference in (*args.reference_file, *args.reference_glob):
        validate_credential_free_uri_identity(reference, "reference sidecar URI")
    references, unmatched_patterns = _resolve_reference_files(args.reference_file, args.reference_glob)
    settings = _make_settings(args, references)
    selected = list(dict.fromkeys(args.arm))
    rank_identity_values = (args.rank_id, args.rank_count, args.slurm_job_id)
    if args.evidence_class == "scaling_rank":
        if args.rank_id is None or args.rank_count is None or not args.slurm_job_id:
            msg = "scaling_rank evidence requires rank-id, rank-count, and slurm-job-id"
            raise ValueError(msg)
        if args.rank_id >= args.rank_count:
            msg = f"rank-id {args.rank_id} is outside rank-count {args.rank_count}"
            raise ValueError(msg)
    elif any(value is not None for value in rank_identity_values):
        msg = "rank identity arguments are valid only for scaling_rank evidence"
        raise ValueError(msg)
    arms = _construct_arms(selected, manifest, settings)
    code_identity = _code_identity()
    benchmark_git_commit = code_identity.get("benchmark", {}).get("git_commit")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "evidence_class": args.evidence_class,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "packages": _package_versions(),
            "code": code_identity,
        },
        "manifest": {
            "path": str(Path(args.query_manifest).resolve()),
            "rows": manifest.table.num_rows,
            "schema": str(manifest.table.schema),
            "digest_sha256": manifest.digest,
            "expected_columns": manifest.expected_columns,
        },
        "dataset": {
            "uri": _redact_uri_for_report(settings.image_lance_uri),
            "version": settings.image_lance_version,
            "key_column": settings.key_column,
            "index_name": settings.index_name,
            "source_columns": settings.source_columns,
            "storage_option_keys": sorted(settings.storage_options),
            "storage_options_identity_sha256": _canonical_json_sha256(settings.storage_options),
        },
        "configuration": {
            "repeat_count": args.repeat_count,
            "warmup_count": args.warmup_count,
            "order_seed": args.order_seed,
            "task_rows": settings.task_rows,
            "coalesce_tasks": settings.coalesce_tasks,
            "rows_per_coalesced_fetch": settings.window_rows,
            "lookup_batch_size": settings.lookup_batch_size,
            "fetch_batch_size": settings.fetch_batch_size,
            "ray_filter_batch_size": settings.ray_filter_batch_size,
            "ray_concurrency": settings.ray_concurrency,
            "ray_worker_dataset_cache_size": settings.ray_worker_dataset_cache_size,
            "public_index_fast_search": settings.public_index_fast_search,
            "max_lookup_bytes": settings.max_lookup_bytes,
            "max_pending_fetch_batches": settings.max_pending_fetch_batches,
            "payload_read_mode": settings.payload_read_mode,
            "medium_density_threshold": settings.medium_density_threshold,
            "high_density_threshold": settings.high_density_threshold,
            "max_coalesced_range_gap": settings.max_coalesced_range_gap,
            "take_scan_batch_readahead": settings.take_scan_batch_readahead,
            "validate_payload_keys": settings.validate_payload_keys,
            "io_threads": settings.io_threads,
            "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_KEYS},
            "prewarm_index": settings.prewarm_index,
            "index_cache_size_bytes": settings.index_cache_size_bytes,
            "metadata_cache_size_bytes": settings.metadata_cache_size_bytes,
            "index_mirror": settings.index_mirror,
            "index_mirror_contract": settings.index_mirror_contract,
            "copy_index_to_node_local": settings.copy_index_to_node_local,
            "index_node_local_root": settings.index_node_local_root,
            "reference_files": settings.reference_files,
            "reference_file_inventory_sha256": _canonical_json_sha256(settings.reference_files),
            "reference_manifest_uri": _redact_uri_for_report(settings.reference_manifest_uri),
            "reference_manifest_sha256": settings.reference_manifest_sha256,
            "reference_key_column": settings.reference_key_column,
            "reference_row_id_column": settings.reference_row_id_column,
            "expected_reference_rows": settings.expected_reference_rows,
            "unmatched_reference_globs": unmatched_patterns,
            "reference_storage_option_keys": sorted(settings.reference_storage_options),
            "reference_storage_options_identity_sha256": _canonical_json_sha256(settings.reference_storage_options),
            "copy_reference_to_node_local": settings.copy_reference_to_node_local,
            "reference_node_local_root": settings.reference_node_local_root,
            "ray_address": settings.ray_address,
            "ray_temp_dir": settings.ray_temp_dir,
            "ray_gpu_actors": settings.ray_gpu_actors,
            "actor_warmup_rows": settings.actor_warmup_rows,
            "ray_actor_pool_size": settings.ray_gpu_actors,
            "ray_actor_input_blocks": math.ceil(manifest.table.num_rows / settings.task_rows),
            "ray_actor_input_block_rows": settings.task_rows,
            "ray_actor_coalesce_tasks": settings.coalesce_tasks,
            "ray_actor_target_batch_rows": settings.window_rows,
            "ray_actor_batching": "Ray approximately bundles adjacent Arrow blocks to the target batch rows",
            "throughput_timing_basis": _PRIMARY_THROUGHPUT_TIMING,
            "warm_process_timing": "same arm.run wall envelope as wall_seconds",
            "ray_actor_process_span_timing": "max(process_end_epoch)-min(process_start_epoch)",
        },
        "order_schedule": {"setup": [], "warmup": [], "repeat": []},
        "arms": {
            name: {
                "status": "pending",
                "payload_projection": _arm_payload_projection(name, settings),
                "cold_setup": None,
                "warmups": [],
                "repeats": [],
            }
            for name in selected
        },
    }
    if isinstance(benchmark_git_commit, str):
        report["environment"]["git_commit"] = benchmark_git_commit
    if args.evidence_class == "scaling_rank":
        report["run_identity"] = {
            "rank_id": args.rank_id,
            "rank_count": args.rank_count,
            "slurm_job_id": args.slurm_job_id,
        }
    _write_report(report, args.output)

    active: list[str] = []
    setup_order = _rotated_order(selected, args.order_seed, 0)
    report["order_schedule"]["setup"] = setup_order
    for name in setup_order:
        started = time.perf_counter()
        try:
            arms[name].setup()
            report["arms"][name]["cold_setup"] = {
                "wall_seconds": time.perf_counter() - started,
                "backend_metrics": arms[name].setup_metrics,
            }
            report["arms"][name]["status"] = "ready"
            active.append(name)
        except (ArmUnavailableError, ImportError, ModuleNotFoundError) as exc:
            report["arms"][name]["status"] = "skipped"
            report["arms"][name]["skip_reason"] = str(exc)
        except Exception as exc:
            report["arms"][name]["status"] = "setup_failed"
            report["arms"][name]["error"] = _error_record(exc)
        _write_report(report, args.output)

    failed: set[str] = set()
    for warmup in range(args.warmup_count):
        order = _rotated_order(active, args.order_seed, warmup) if active else []
        report["order_schedule"]["warmup"].append(order)
        for order_index, name in enumerate(order):
            if name in failed:
                continue
            try:
                result = _run_once(arms[name], manifest, warmup, order_index)
                report["arms"][name]["warmups"].append(
                    {
                        "status": result["status"],
                        "wall_seconds": result["wall_seconds"],
                        "correctness": result["correctness"],
                    }
                )
                if result["status"] != "completed":
                    failed.add(name)
                    report["arms"][name]["status"] = "warmup_incorrect"
            except Exception as exc:
                failed.add(name)
                report["arms"][name]["status"] = "warmup_failed"
                report["arms"][name]["warmups"].append({"status": "failed", "error": _error_record(exc)})
            _write_report(report, args.output)

    for repeat in range(args.repeat_count):
        order = _rotated_order(active, args.order_seed, args.warmup_count + repeat) if active else []
        report["order_schedule"]["repeat"].append(order)
        for order_index, name in enumerate(order):
            if name in failed:
                continue
            try:
                result = _run_once(arms[name], manifest, repeat, order_index)
                report["arms"][name]["repeats"].append(result)
                if result["status"] != "completed":
                    failed.add(name)
                    report["arms"][name]["status"] = "incorrect"
            except Exception as exc:
                failed.add(name)
                report["arms"][name]["status"] = "run_failed"
                report["arms"][name]["repeats"].append(
                    {"status": "failed", "repeat": repeat, "order_index": order_index, "error": _error_record(exc)}
                )
            _write_report(report, args.output)

    for name in active:
        if name not in failed:
            report["arms"][name]["status"] = "completed"
    report["status"] = "tearing_down"
    _write_report(report, args.output)
    for arm in arms.values():
        try:
            arm.close()
        except Exception as exc:
            report.setdefault("teardown_errors", {})[arm.name] = _error_record(exc)
    _finalize_report_status(report)
    _summarize(report)
    _write_report(report, args.output)
    return report


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be greater than zero"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _sha256(value: str) -> str:
    if len(value) != _SHA256_HEX_LENGTH or value != value.lower():
        msg = "value must be a lowercase SHA-256 hex digest"
        raise argparse.ArgumentTypeError(msg)
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        msg = "value must be a lowercase SHA-256 hex digest"
        raise argparse.ArgumentTypeError(msg) from exc
    return value


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        msg = "value must be nonnegative"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--image-lance-uri", required=True, type=_credential_free_uri)
    parser.add_argument("--image-lance-version", required=True, type=_positive)
    parser.add_argument("--storage-options-json", default="{}", help="Inline JSON object or @path")
    parser.add_argument("--key-column", default="url")
    parser.add_argument("--index-name", default="url_btree")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--md5-column", default="md5", help="Empty string disables this Lance projection")
    parser.add_argument("--width-column", default="width", help="Empty string disables this Lance projection")
    parser.add_argument("--height-column", default="height", help="Empty string disables this Lance projection")
    parser.add_argument("--expected-md5-column")
    parser.add_argument("--expected-width-column")
    parser.add_argument("--expected-height-column")
    parser.add_argument("--max-queries", type=_positive)

    parser.add_argument("--index-mirror")
    parser.add_argument(
        "--index-mirror-contract-json",
        help="Inline contract JSON or @path; required together with --index-mirror",
    )
    parser.add_argument("--copy-index-to-node-local", action="store_true")
    parser.add_argument("--index-node-local-root", default="/local/lance-indexes")
    parser.add_argument("--no-prewarm-index", action="store_true")
    parser.add_argument("--index-cache-size-gib", type=_positive, default=32)
    parser.add_argument("--metadata-cache-size-mib", type=_positive, default=1024)

    parser.add_argument("--reference-file", action="append", default=[])
    parser.add_argument("--reference-glob", action="append", default=[])
    parser.add_argument("--reference-storage-options-json", default="{}", help="Inline JSON object or @path")
    parser.add_argument("--reference-manifest-uri", type=_credential_free_uri)
    parser.add_argument("--reference-manifest-sha256", type=_sha256)
    parser.add_argument("--reference-key-column", default="url")
    parser.add_argument("--reference-row-id-column", default="stable_row_id")
    parser.add_argument("--expected-reference-rows", type=_positive)
    parser.add_argument("--gpu-load-factor", type=float, default=0.5)
    parser.add_argument(
        "--max-lookup-bytes-mib",
        type=_positive,
        default=256,
        help="Hard Arrow key-window bound for the lance-ray GPU actor",
    )
    parser.add_argument(
        "--max-pending-fetch-batches",
        type=_positive,
        default=16,
        help="Maximum concurrent private Lance take batches per lance-ray actor",
    )
    parser.add_argument(
        "--payload-read-mode",
        choices=("sparse", "adaptive_unmeasured"),
        default="sparse",
        help="Private payload strategy; adaptive mode is opt-in until remotely benchmarked",
    )
    parser.add_argument("--medium-density-threshold", type=float, default=0.25)
    parser.add_argument("--high-density-threshold", type=float, default=0.75)
    parser.add_argument("--max-coalesced-range-gap", type=_nonnegative, default=0)
    parser.add_argument("--take-scan-batch-readahead", type=_positive, default=16)
    parser.add_argument(
        "--validate-payload-keys",
        action="store_true",
        help="Include the Lance key in timed payload reads and validate stable-ID mappings",
    )
    parser.add_argument("--copy-reference-to-node-local", action="store_true")
    parser.add_argument("--reference-node-local-root", default="/local/nemo-curator/gpu-lance-indexes")

    parser.add_argument("--lookup-batch-size", type=_positive, default=2_000)
    parser.add_argument("--fetch-batch-size", type=_positive, default=1024)
    parser.add_argument("--io-threads", type=_positive, default=16)
    parser.add_argument("--task-rows", type=_positive, default=2_000)
    parser.add_argument("--coalesce-tasks", type=_positive, default=8)
    parser.add_argument("--ray-filter-batch-size", type=_positive, default=2_000)
    parser.add_argument("--ray-concurrency", type=_positive)
    parser.add_argument(
        "--ray-worker-dataset-cache-size",
        type=_nonnegative,
        default=1,
        help="Exact Lance dataset/session reconstructions retained per Ray worker process",
    )
    parser.add_argument(
        "--public-index-fast-search",
        action="store_true",
        help=(
            "Allow naive PyLance and public lance-ray readers to skip unindexed data only after "
            "verifying complete row and fragment coverage on the pinned snapshot"
        ),
    )
    parser.add_argument("--ray-address")
    parser.add_argument("--ray-temp-dir", help="Short local Ray temp path when starting a standalone cluster")
    parser.add_argument(
        "--ray-gpu-actors",
        type=_positive,
        default=1,
        help="Persistent Ray Data GPU actors; each actor reserves one GPU",
    )
    parser.add_argument("--actor-warmup-rows", type=_nonnegative, default=128)

    parser.add_argument("--evidence-class", choices=_EVIDENCE_CLASSES, default="adhoc_benchmark")
    parser.add_argument("--rank-id", type=_nonnegative)
    parser.add_argument("--rank-count", type=_positive)
    parser.add_argument("--slurm-job-id")

    parser.add_argument("--repeat-count", type=_positive, default=3)
    parser.add_argument("--warmup-count", type=_nonnegative, default=1)
    parser.add_argument("--order-seed", type=int, default=17)
    parser.add_argument(
        "--arm",
        action="append",
        choices=(
            "naive_pylance_scalar",
            "cpu_lance_column_fetch_stage",
            "gpu_lance_column_fetch_stage",
            "lance_ray_datasource",
            "lance_ray_gpu_fetcher",
            "lance_ray_gpu_actor",
            "ray_data_persistent_gpu_actor",
        ),
        default=[],
        help="Repeat to select arms; the default runs every arm",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.arm:
        args.arm = [
            "naive_pylance_scalar",
            "cpu_lance_column_fetch_stage",
            "gpu_lance_column_fetch_stage",
            "lance_ray_datasource",
            "lance_ray_gpu_fetcher",
            "lance_ray_gpu_actor",
            "ray_data_persistent_gpu_actor",
        ]
    report = run_benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
