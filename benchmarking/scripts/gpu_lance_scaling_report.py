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

# ruff: noqa: PLR2004, S108
"""Aggregate validated GPU Lance harness results into scaling reports.

Each ``--result`` uses ``LABEL=PATH``. ``PATH`` may be one file, a glob, or a
comma-separated list of files for per-rank results. Resource metadata can live in the
harness JSON or in a structured label such as::

    --result 'h100-1[nodes=1,gpus=1]=results/h100-1.json'
    --result 'cpu-4[backend=cpu_lance_column_fetch_stage,nodes=4,gpus=0,workload=mint-fixed]=rank-*.json'

An optional ``backend=ARM_NAME`` label field selects one arm after the entire
harness result has passed correctness validation. Compact labels such as
``h100-1n-8g`` are also recognized. Projections are accepted only through
``--projection`` and remain in a separate, explicitly non-measured section.

The script uses only the Python standard library. It intentionally rejects
running, failed, skipped, partially repeated, or incorrect harness outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from glob import glob, has_magic
from itertools import product
from pathlib import Path

SCHEMA_VERSION = 1
MIB = 1024**2
GPU_BACKENDS = frozenset(
    {
        "gpu_lance_column_fetch_stage",
        "lance_ray_gpu_actor",
        "lance_ray_gpu_fetcher",
        "ray_data_persistent_gpu_actor",
    }
)
CPU_BACKENDS = frozenset({"naive_pylance_scalar", "cpu_lance_column_fetch_stage", "lance_ray_datasource"})
KNOWN_BACKENDS = GPU_BACKENDS | CPU_BACKENDS
_LABEL_PATTERN = re.compile(r"^(?P<name>.*?)(?:\[(?P<meta>[^]]+)\])?$")
_COMPACT_NODE_PATTERNS = (
    re.compile(r"(?:^|[-_.])n(?:odes?)?[-_]?([0-9]+)(?:$|[-_.])", re.IGNORECASE),
    re.compile(r"(?:^|[-_.])([0-9]+)[-_]?n(?:odes?)?(?:$|[-_.])", re.IGNORECASE),
)
_COMPACT_GPU_PATTERNS = (
    re.compile(r"(?:^|[-_.])g(?:pus?)?[-_]?([0-9]+)(?:$|[-_.])", re.IGNORECASE),
    re.compile(r"(?:^|[-_.])([0-9]+)[-_]?g(?:pus?)?(?:$|[-_.])", re.IGNORECASE),
    re.compile(r"(?:^|[-_.])([0-9]+)[-_]?h100(?:s)?(?:$|[-_.])", re.IGNORECASE),
)
_RANK_FILE_PATTERN = re.compile(r"^rank[-_]?([0-9]+)\.json$", re.IGNORECASE)
_RANK_DIRECTORY_PATTERN = re.compile(r"(?:^|_)([0-9]+)_ranks(?:$|_)", re.IGNORECASE)
_SLURM_DIRECTORY_PATTERN = re.compile(r"^([0-9]+)(?:[_-].*)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRIMARY_SATURATION_EVIDENCE_CLASS = "primary_saturation"
_PRIMARY_SATURATION_WAVES = frozenset({4, 8})
_TERMINAL_ELIGIBILITY_SCHEMA_VERSION = 2


class ReportInputError(ValueError):
    """Raised when a result is incomplete, incorrect, or ambiguous."""


@dataclass(frozen=True)
class LabeledPath:
    label: str
    display_name: str
    path_expression: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class MetricStats:
    minimum: float
    median: float
    mean: float
    maximum: float
    stdev: float

    @classmethod
    def from_values(cls, values: Sequence[float], *, allow_zero: bool = False) -> MetricStats:
        if not values:
            message = "cannot summarize an empty metric series"
            raise ReportInputError(message)
        invalid = any(not math.isfinite(value) or (value < 0 if allow_zero else value <= 0) for value in values)
        if invalid:
            qualifier = "nonnegative" if allow_zero else "positive"
            message = f"metric series must contain finite {qualifier} values: {values!r}"
            raise ReportInputError(message)
        return cls(
            minimum=min(values),
            median=statistics.median(values),
            mean=statistics.fmean(values),
            maximum=max(values),
            stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "min": self.minimum,
            "median": self.median,
            "mean": self.mean,
            "max": self.maximum,
            "stdev": self.stdev,
        }


@dataclass(frozen=True)
class RepeatMeasurement:
    wall_seconds: float
    warm_process_seconds: float
    images_per_second: float
    payload_mib_per_second: float
    payload_bytes: int
    output_digest_sha256: str
    lance_read_iops: float | None
    lance_read_bytes: float | None
    lookup_calls: float | None
    fetch_calls: float | None


@dataclass(frozen=True)
class Measurement:
    label: str
    sources: tuple[str, ...]
    source_sha256: tuple[str, ...]
    backend: str
    backend_class: str
    node_count: int
    gpu_count: int
    resource_source: Mapping[str, str]
    rank_count: int
    rank_task_counts: tuple[int, ...]
    cold_setup_seconds_max: float
    cold_setup_seconds_sum: float
    manifest_rows: int
    manifest_digest_sha256: tuple[str, ...]
    workload_id: str
    workload_id_source: str
    dataset_uri: str
    dataset_version: str
    task_rows: int
    coalesce_tasks: int
    task_window_rows: int
    lookup_batch_size: int
    fetch_batch_size: int
    io_threads: int
    evidence_identity: Mapping[str, object]
    comparison_eligibility_errors: tuple[str, ...]
    rank_ids: tuple[int, ...]
    slurm_run_id: str | None
    repeats: tuple[RepeatMeasurement, ...]

    def metric_stats(self) -> dict[str, MetricStats]:
        nodes = self.node_count
        metrics = {
            "wall_seconds": MetricStats.from_values([item.wall_seconds for item in self.repeats]),
            "warm_process_seconds": MetricStats.from_values([item.warm_process_seconds for item in self.repeats]),
            "images_per_second": MetricStats.from_values([item.images_per_second for item in self.repeats]),
            "payload_mib_per_second": MetricStats.from_values([item.payload_mib_per_second for item in self.repeats]),
            "images_per_second_per_node": MetricStats.from_values(
                [item.images_per_second / nodes for item in self.repeats]
            ),
            "payload_mib_per_second_per_node": MetricStats.from_values(
                [item.payload_mib_per_second / nodes for item in self.repeats]
            ),
            "payload_bytes": MetricStats.from_values([float(item.payload_bytes) for item in self.repeats]),
        }
        for name in ("lance_read_iops", "lance_read_bytes", "lookup_calls", "fetch_calls"):
            values = [getattr(item, name) for item in self.repeats]
            if all(value is not None for value in values):
                metrics[name] = MetricStats.from_values(
                    [float(value) for value in values if value is not None],
                    allow_zero=True,
                )
        return metrics

    def workload_signature(self, *, include_window: bool) -> tuple[object, ...]:
        signature: tuple[object, ...] = (
            self.workload_id,
            self.manifest_rows,
            self.dataset_uri,
            self.dataset_version,
            self.lookup_batch_size,
            self.fetch_batch_size,
            self.io_threads,
            json.dumps(self.evidence_identity, sort_keys=True, separators=(",", ":")),
        )
        if include_window:
            return (*signature, self.task_rows, self.coalesce_tasks, self.task_window_rows)
        return signature

    def as_dict(self) -> dict[str, object]:
        metrics = {name: stats.as_dict() for name, stats in self.metric_stats().items()}
        dataset_uri_sha256 = hashlib.sha256(self.dataset_uri.encode("utf-8")).hexdigest()
        return {
            "classification": "measured",
            "label": self.label,
            "sources": list(self.sources),
            "source_sha256": list(self.source_sha256),
            "backend": self.backend,
            "backend_class": self.backend_class,
            "resources": {
                "node_count": self.node_count,
                "gpu_count": self.gpu_count,
                "rank_count": self.rank_count,
                "source": dict(self.resource_source),
            },
            "setup_seconds": {
                "max_rank_wall": self.cold_setup_seconds_max,
                "sum_rank_wall": self.cold_setup_seconds_sum,
                "classification": "measured",
            },
            "workload": {
                "manifest_rows": self.manifest_rows,
                "rank_manifest_digest_sha256": list(self.manifest_digest_sha256),
                "workload_id": self.workload_id,
                "workload_id_source": self.workload_id_source,
                "dataset_uri_sha256": dataset_uri_sha256,
                "dataset_version": self.dataset_version,
            },
            "task_window": {
                "task_rows": self.task_rows,
                "coalesce_tasks": self.coalesce_tasks,
                "rows_per_coalesced_fetch": self.task_window_rows,
                "per_rank_rows_per_coalesced_fetch": self.task_window_rows,
                "per_rank_task_counts": list(self.rank_task_counts),
                "global_task_count": sum(self.rank_task_counts),
            },
            "configuration": {
                "lookup_batch_size": self.lookup_batch_size,
                "fetch_batch_size": self.fetch_batch_size,
                "io_threads": self.io_threads,
            },
            "evidence_identity": dict(self.evidence_identity),
            "comparison_eligible": not self.comparison_eligibility_errors,
            "comparison_eligibility_errors": list(self.comparison_eligibility_errors),
            "rank_identity": {
                "rank_ids": list(self.rank_ids),
                "slurm_run_id": self.slurm_run_id,
            },
            "repeat_count": len(self.repeats),
            "metrics": metrics,
        }


@dataclass(frozen=True)
class AcceptedResult:
    label: str
    display_name: str
    sources: tuple[str, ...]
    source_sha256: tuple[str, ...]
    node_count: int
    gpu_count: int
    resource_source: Mapping[str, str]
    rank_id: int | None
    expected_rank_count: int | None
    slurm_run_id: str | None
    rank_identity_source: Mapping[str, str]
    measurements: tuple[Measurement, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "display_name": self.display_name,
            "sources": list(self.sources),
            "source_sha256": list(self.source_sha256),
            "rank_count": len(self.sources),
            "node_count": self.node_count,
            "gpu_count": self.gpu_count,
            "resource_source": dict(self.resource_source),
            "rank_identity": {
                "rank_ids": list(self.measurements[0].rank_ids) if self.measurements else [],
                "expected_rank_count": self.expected_rank_count,
                "slurm_run_id": self.slurm_run_id,
                "source": dict(self.rank_identity_source),
            },
            "backends": [measurement.backend for measurement in self.measurements],
            "classification": "measured",
        }


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        message = f"{context} must be an object"
        raise ReportInputError(message)
    if not all(isinstance(key, str) for key in value):
        message = f"{context} keys must be strings"
        raise ReportInputError(message)
    return value


def _as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        message = f"{context} must be an array"
        raise ReportInputError(message)
    return value


def _required(mapping: Mapping[str, object], key: str, context: str) -> object:
    if key not in mapping:
        message = f"{context} is missing {key!r}"
        raise ReportInputError(message)
    return mapping[key]


def _as_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        message = f"{context} must be a non-empty string"
        raise ReportInputError(message)
    return value


def _as_number(value: object, context: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"{context} must be numeric"
        raise ReportInputError(message)
    result = float(value)
    minimum_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "nonnegative" if allow_zero else "positive"
        message = f"{context} must be finite and {qualifier}, got {value!r}"
        raise ReportInputError(message)
    return result


def _as_int(value: object, context: str, *, allow_zero: bool = False) -> int:
    result = _as_number(value, context, allow_zero=allow_zero)
    if not result.is_integer():
        message = f"{context} must be an integer, got {value!r}"
        raise ReportInputError(message)
    return int(result)


def _lookup(mapping: Mapping[str, object], path: Sequence[str]) -> object | None:
    current: object = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_int(mapping: Mapping[str, object], paths: Sequence[Sequence[str]]) -> tuple[int, str] | None:
    found: list[tuple[int, str]] = []
    for path in paths:
        value = _lookup(mapping, path)
        if value is not None:
            location = ".".join(path)
            found.append((_as_int(value, location, allow_zero=True), location))
    if not found:
        return None
    distinct = {value for value, _ in found}
    if len(distinct) != 1:
        message = f"conflicting resource counts: {found!r}"
        raise ReportInputError(message)
    return found[0]


def parse_labeled_path(raw: str) -> LabeledPath:
    """Parse ``LABEL=PATH`` plus optional bracket metadata."""

    label, separator, path_text = raw.rpartition("=")
    if not separator or not label or not path_text:
        message = f"expected LABEL=PATH, got {raw!r}"
        raise ReportInputError(message)
    match = _LABEL_PATTERN.fullmatch(label)
    if match is None or not match.group("name"):
        message = f"invalid result label {label!r}"
        raise ReportInputError(message)
    metadata: dict[str, str] = {}
    raw_metadata = match.group("meta")
    if raw_metadata:
        for token in raw_metadata.split(","):
            key, token_separator, value = token.strip().partition("=")
            if not token_separator or not key or not value:
                message = f"invalid label metadata token {token!r} in {label!r}"
                raise ReportInputError(message)
            normalized_key = {"node_count": "nodes", "gpu_count": "gpus"}.get(key, key)
            if normalized_key not in {"nodes", "gpus", "ranks", "slurm_job_id", "backend", "workload"}:
                message = f"unsupported label metadata key {key!r}"
                raise ReportInputError(message)
            if normalized_key in metadata:
                message = f"duplicate label metadata key {key!r}"
                raise ReportInputError(message)
            metadata[normalized_key] = value
    return LabeledPath(
        label=label,
        display_name=match.group("name"),
        path_expression=path_text,
        metadata=metadata,
    )


def _compact_count(label: str, patterns: Sequence[re.Pattern[str]], context: str) -> int | None:
    values = {int(match.group(1)) for pattern in patterns if (match := pattern.search(label))}
    if len(values) > 1:
        message = f"{label!r} contains conflicting compact {context} counts: {sorted(values)}"
        raise ReportInputError(message)
    if not values:
        return None
    value = next(iter(values))
    if value <= 0:
        message = f"compact {context} count in {label!r} must be positive"
        raise ReportInputError(message)
    return value


def _metadata_count(labeled: LabeledPath, key: str, *, allow_zero: bool = False) -> int | None:
    if key not in labeled.metadata:
        return None
    raw = labeled.metadata[key]
    try:
        parsed = int(raw)
    except ValueError as error:
        message = f"{labeled.label}.{key} must be an integer, got {raw!r}"
        raise ReportInputError(message) from error
    return _as_int(parsed, f"{labeled.label}.{key}", allow_zero=allow_zero)


def _canonical_mapping(value: object, context: str) -> dict[str, object]:
    mapping = _as_mapping(value, context)
    try:
        return json.loads(json.dumps(mapping, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as error:
        message = f"{context} must contain JSON-compatible identity values"
        raise ReportInputError(message) from error


def _identity_field(
    mapping: Mapping[str, object],
    key: str,
    context: str,
    errors: list[str],
    *,
    allow_none: bool = False,
) -> object | None:
    if key not in mapping:
        errors.append(f"missing {context}.{key}")
        return None
    value = mapping[key]
    if value is None and not allow_none:
        errors.append(f"{context}.{key} is null")
    return value


def _comparison_identity(  # noqa: C901, PLR0912
    report: Mapping[str, object],
    dataset: Mapping[str, object],
    configuration: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Return the full comparison identity and reasons it is not trustworthy."""

    errors: list[str] = []
    raw_columns = _identity_field(dataset, "source_columns", "dataset", errors)
    source_columns: dict[str, object] = {}
    if raw_columns is not None:
        try:
            source_columns = _canonical_mapping(raw_columns, "dataset.source_columns")
        except ReportInputError as error:
            errors.append(str(error))
        if not source_columns or not all(
            isinstance(key, str) and key and isinstance(value, str) and value for key, value in source_columns.items()
        ):
            errors.append("dataset.source_columns must be a non-empty string mapping")

    sidecar_uri = _identity_field(configuration, "reference_manifest_uri", "configuration", errors)
    if sidecar_uri is not None and (not isinstance(sidecar_uri, str) or not sidecar_uri):
        errors.append("configuration.reference_manifest_uri must be a non-empty string")
    sidecar_sha256 = _identity_field(configuration, "reference_manifest_sha256", "configuration", errors)
    if sidecar_sha256 is not None and (
        not isinstance(sidecar_sha256, str) or _SHA256_PATTERN.fullmatch(sidecar_sha256) is None
    ):
        errors.append("configuration.reference_manifest_sha256 must be a lowercase SHA-256")

    concurrency_keys = (
        "io_threads",
        "max_lookup_bytes",
        "max_pending_fetch_batches",
        "take_scan_batch_readahead",
    )
    concurrency = {key: _identity_field(configuration, key, "configuration", errors) for key in concurrency_keys}
    if "ray_concurrency" in configuration:
        concurrency["ray_concurrency"] = configuration["ray_concurrency"]

    index_mirror = _identity_field(configuration, "index_mirror", "configuration", errors, allow_none=True)
    cache_policy: dict[str, object | None] = {
        "index_mirror": index_mirror,
        "copy_index_to_node_local": _identity_field(
            configuration,
            "copy_index_to_node_local",
            "configuration",
            errors,
        ),
    }
    if "index_mirror_contract" in configuration:
        cache_policy["index_mirror_contract"] = configuration["index_mirror_contract"]
    elif index_mirror is not None:
        errors.append("missing configuration.index_mirror_contract for configured index mirror")

    validation_policy = {
        "validate_payload_keys": _identity_field(
            configuration,
            "validate_payload_keys",
            "configuration",
            errors,
        )
    }
    payload_read_mode = _identity_field(configuration, "payload_read_mode", "configuration", errors)
    environment_value = _identity_field(report, "environment", "benchmark", errors)
    environment = _as_mapping(environment_value, "benchmark.environment") if environment_value is not None else {}
    python_identity = _identity_field(environment, "python", "environment", errors) if environment else None
    package_value = _identity_field(environment, "packages", "environment", errors) if environment else None
    packages: dict[str, object] = {}
    if package_value is not None:
        try:
            packages = _canonical_mapping(package_value, "environment.packages")
        except ReportInputError as error:
            errors.append(str(error))
        required_packages = ("nemo-curator", "pyarrow", "pylance")
        missing_packages = [
            package
            for package in required_packages
            if not isinstance(packages.get(package), str) or not packages[package]
        ]
        if missing_packages:
            errors.append(f"environment.packages is missing version identity for {missing_packages}")

    code_identity = {
        key: environment[key]
        for key in ("git_commit", "git_sha", "source_commit", "code_commit")
        if key in environment
    }
    dataset_storage_keys = _identity_field(dataset, "storage_option_keys", "dataset", errors)
    sidecar_storage_keys = _identity_field(
        configuration,
        "reference_storage_option_keys",
        "configuration",
        errors,
    )
    for value, name in (
        (dataset_storage_keys, "dataset.storage_option_keys"),
        (sidecar_storage_keys, "configuration.reference_storage_option_keys"),
    ):
        if value is not None and (
            not isinstance(value, list) or not all(isinstance(item, str) and item for item in value)
        ):
            errors.append(f"{name} must be an array of non-empty strings")
    return (
        {
            "payload_projection": source_columns,
            "sidecar": {
                "manifest_uri": sidecar_uri,
                "manifest_sha256": sidecar_sha256,
            },
            "read_policy": {"payload_read_mode": payload_read_mode},
            "concurrency_policy": concurrency,
            "cache_policy": cache_policy,
            "validation_policy": validation_policy,
            "storage_option_keys": {
                "dataset": dataset_storage_keys,
                "sidecar": sidecar_storage_keys,
            },
            "runtime_identity": {
                "python": python_identity,
                "packages": packages,
                "code": code_identity,
            },
        },
        tuple(dict.fromkeys(errors)),
    )


def _first_optional_int(
    report: Mapping[str, object],
    paths: Sequence[Sequence[str]],
    context: str,
) -> tuple[int | None, str | None]:
    found = _first_int(report, paths)
    if found is None:
        return None, None
    value, source = found
    if value < 0:
        message = f"{context} must be nonnegative"
        raise ReportInputError(message)
    return value, source


def _rank_identity(  # noqa: C901
    labeled: LabeledPath,
    report: Mapping[str, object],
    source: Path,
) -> tuple[int | None, int | None, str | None, dict[str, str]]:
    rank_id, rank_source = _first_optional_int(
        report,
        (
            ("run_identity", "rank_id"),
            ("run_identity", "rank"),
            ("configuration", "rank_id"),
            ("configuration", "scale_rank"),
            ("rank_id",),
        ),
        "rank id",
    )
    file_match = _RANK_FILE_PATTERN.fullmatch(source.name)
    path_rank = int(file_match.group(1)) if file_match is not None else None
    if rank_id is not None and path_rank is not None and rank_id != path_rank:
        message = f"{labeled.label}: embedded rank id {rank_id} conflicts with filename rank id {path_rank}"
        raise ReportInputError(message)
    if rank_id is None and path_rank is not None:
        rank_id, rank_source = path_rank, "filename"

    expected_rank_count, count_source = _first_optional_int(
        report,
        (
            ("run_identity", "rank_count"),
            ("run_identity", "expected_rank_count"),
            ("configuration", "rank_count"),
            ("configuration", "scale_ranks"),
            ("rank_count",),
        ),
        "expected rank count",
    )
    directory_match = _RANK_DIRECTORY_PATTERN.search(source.parent.parent.name)
    path_count = int(directory_match.group(1)) if directory_match is not None else None
    if expected_rank_count is not None and path_count is not None and expected_rank_count != path_count:
        message = (
            f"{labeled.label}: embedded rank count {expected_rank_count} conflicts with "
            f"directory rank count {path_count}"
        )
        raise ReportInputError(message)
    if expected_rank_count is None and path_count is not None:
        expected_rank_count, count_source = path_count, "directory"
    label_count = _metadata_count(labeled, "ranks")
    if label_count is not None and expected_rank_count is not None and label_count != expected_rank_count:
        message = f"{labeled.label}: label ranks={label_count} conflicts with {count_source}={expected_rank_count}"
        raise ReportInputError(message)
    if expected_rank_count is None and label_count is not None:
        expected_rank_count, count_source = label_count, "label"

    slurm_values: list[tuple[str, str]] = []
    for path in (
        ("run_identity", "slurm_job_id"),
        ("configuration", "slurm_job_id"),
        ("slurm_job_id",),
    ):
        value = _lookup(report, path)
        if value is not None:
            slurm_values.append((str(value), ".".join(path)))
    parent_match = _SLURM_DIRECTORY_PATTERN.fullmatch(source.parent.name)
    if parent_match is not None:
        slurm_values.append((parent_match.group(1), "parent_directory"))
    label_slurm = labeled.metadata.get("slurm_job_id")
    if label_slurm is not None:
        slurm_values.append((label_slurm, "label"))
    distinct_slurm = {value for value, _ in slurm_values}
    if len(distinct_slurm) > 1:
        message = f"{labeled.label}: conflicting Slurm run identities: {slurm_values!r}"
        raise ReportInputError(message)
    slurm_run_id = next(iter(distinct_slurm), None)
    slurm_source = ",".join(source for _, source in slurm_values) if slurm_values else "absent"
    return (
        rank_id,
        expected_rank_count,
        slurm_run_id,
        {
            "rank_id": str(rank_source or "absent"),
            "expected_rank_count": str(count_source or "absent"),
            "slurm_run_id": slurm_source,
        },
    )


def _resources(  # noqa: C901
    labeled: LabeledPath,
    report: Mapping[str, object],
    gpus_per_node: int,
) -> tuple[int, int, dict[str, str]]:
    explicit_nodes = _first_int(
        report,
        (
            ("configuration", "node_count"),
            ("configuration", "nodes"),
            ("resources", "node_count"),
            ("resources", "nodes"),
            ("cluster", "node_count"),
            ("cluster", "nodes"),
            ("node_count",),
        ),
    )
    explicit_gpus = _first_int(
        report,
        (
            ("configuration", "gpu_count"),
            ("configuration", "gpus"),
            ("configuration", "total_gpu_count"),
            ("resources", "gpu_count"),
            ("resources", "gpus"),
            ("cluster", "gpu_count"),
            ("cluster", "gpus"),
            ("gpu_count",),
        ),
    )
    label_nodes = _metadata_count(labeled, "nodes")
    if label_nodes is None:
        label_nodes = _compact_count(labeled.display_name, _COMPACT_NODE_PATTERNS, "node")
    label_gpus = _metadata_count(labeled, "gpus", allow_zero=True)
    if label_gpus is None:
        label_gpus = _compact_count(labeled.display_name, _COMPACT_GPU_PATTERNS, "GPU")

    if label_nodes is not None and explicit_nodes is not None and label_nodes != explicit_nodes[0]:
        message = f"{labeled.label}: label nodes={label_nodes} conflicts with {explicit_nodes[1]}={explicit_nodes[0]}"
        raise ReportInputError(message)
    if label_gpus is not None and explicit_gpus is not None and label_gpus != explicit_gpus[0]:
        message = f"{labeled.label}: label gpus={label_gpus} conflicts with {explicit_gpus[1]}={explicit_gpus[0]}"
        raise ReportInputError(message)

    gpu_source = "label" if label_gpus is not None else None
    gpu_count = label_gpus
    if gpu_count is None and explicit_gpus is not None:
        gpu_count, gpu_source = explicit_gpus
    if gpu_count is None:
        actors = _lookup(report, ("configuration", "ray_gpu_actors"))
        if actors is not None:
            gpu_count = _as_int(actors, "configuration.ray_gpu_actors", allow_zero=True)
            gpu_source = "configuration.ray_gpu_actors"
    if gpu_count is None:
        gpu_count = 0
        gpu_source = "default_zero"

    node_source = "label" if label_nodes is not None else None
    node_count = label_nodes
    if node_count is None and explicit_nodes is not None:
        node_count, node_source = explicit_nodes
    if node_count is None and gpu_count > 0:
        node_count = math.ceil(gpu_count / gpus_per_node)
        node_source = f"inferred_ceil(gpus/{gpus_per_node})"
    if node_count is None:
        message = (
            f"{labeled.label}: node count is missing; add [nodes=N,gpus=G] or a configuration/resources node_count"
        )
        raise ReportInputError(message)
    return (
        node_count,
        gpu_count,
        {
            "nodes": str(node_source),
            "gpus": str(gpu_source),
        },
    )


def _optional_counter(item: Mapping[str, object], key: str, context: str) -> float | None:
    value = item.get(key)
    if value is None:
        return None
    return _as_number(value, f"{context}.{key}", allow_zero=True)


def _validate_repeat(
    item: Mapping[str, object],
    *,
    context: str,
    manifest_rows: int,
) -> RepeatMeasurement:
    if item.get("status") != "completed":
        message = f"{context}.status must be 'completed', got {item.get('status')!r}"
        raise ReportInputError(message)
    wall_seconds = _as_number(_required(item, "wall_seconds", context), f"{context}.wall_seconds")
    warm_seconds = _as_number(
        _required(item, "warm_process_seconds", context),
        f"{context}.warm_process_seconds",
    )
    images_per_second = _as_number(
        _required(item, "images_per_second", context),
        f"{context}.images_per_second",
    )
    payload_rate = _as_number(
        _required(item, "payload_mib_per_second", context),
        f"{context}.payload_mib_per_second",
    )
    payload_bytes = _as_int(_required(item, "payload_bytes", context), f"{context}.payload_bytes")
    correctness = _as_mapping(_required(item, "correctness", context), f"{context}.correctness")
    if correctness.get("correct") is not True:
        message = f"{context}.correctness.correct is not true"
        raise ReportInputError(message)
    for key in ("row_count", "expected_row_count", "present_rows"):
        value = _as_int(_required(correctness, key, f"{context}.correctness"), f"{context}.correctness.{key}")
        if value != manifest_rows:
            message = f"{context}.correctness.{key}={value} does not match manifest rows={manifest_rows}"
            raise ReportInputError(message)
    missing = _as_int(
        _required(correctness, "missing_payload_rows", f"{context}.correctness"),
        f"{context}.correctness.missing_payload_rows",
        allow_zero=True,
    )
    if missing != 0:
        message = f"{context} has {missing} missing payload rows"
        raise ReportInputError(message)
    correctness_payload = _as_int(
        _required(correctness, "payload_bytes", f"{context}.correctness"),
        f"{context}.correctness.payload_bytes",
    )
    if correctness_payload != payload_bytes:
        message = f"{context} payload byte counters disagree"
        raise ReportInputError(message)
    digest = _as_string(
        _required(correctness, "output_digest_sha256", f"{context}.correctness"),
        f"{context}.correctness.output_digest_sha256",
    )
    expected_images_per_second = manifest_rows / warm_seconds
    expected_payload_rate = payload_bytes / (MIB * warm_seconds)
    if not math.isclose(images_per_second, expected_images_per_second, rel_tol=1e-9):
        message = (
            f"{context}.images_per_second={images_per_second} does not match "
            f"rows/warm_process_seconds={expected_images_per_second}"
        )
        raise ReportInputError(message)
    if not math.isclose(payload_rate, expected_payload_rate, rel_tol=1e-9):
        message = (
            f"{context}.payload_mib_per_second={payload_rate} does not match "
            f"payload_bytes/warm_process_seconds={expected_payload_rate}"
        )
        raise ReportInputError(message)
    return RepeatMeasurement(
        wall_seconds=wall_seconds,
        warm_process_seconds=warm_seconds,
        images_per_second=images_per_second,
        payload_mib_per_second=payload_rate,
        payload_bytes=payload_bytes,
        output_digest_sha256=digest,
        lance_read_iops=_optional_counter(item, "lance_read_iops", context),
        lance_read_bytes=_optional_counter(item, "lance_read_bytes", context),
        lookup_calls=_optional_counter(item, "lookup_calls", context),
        fetch_calls=_optional_counter(item, "fetch_calls", context),
    )


def _validate_warmups(
    arm: Mapping[str, object],
    *,
    expected_count: int,
    context: str,
) -> None:
    warmups = _as_list(_required(arm, "warmups", context), f"{context}.warmups")
    if len(warmups) != expected_count:
        message = f"{context} has {len(warmups)} warmups, expected {expected_count}"
        raise ReportInputError(message)
    for index, raw_warmup in enumerate(warmups):
        warmup = _as_mapping(raw_warmup, f"{context}.warmups[{index}]")
        if warmup.get("status") != "completed":
            message = f"{context}.warmups[{index}] is not completed"
            raise ReportInputError(message)
        correctness = _as_mapping(
            _required(warmup, "correctness", f"{context}.warmups[{index}]"),
            f"{context}.warmups[{index}].correctness",
        )
        if correctness.get("correct") is not True:
            message = f"{context}.warmups[{index}] is incorrect"
            raise ReportInputError(message)


def validate_harness_result(  # noqa: C901, PLR0912, PLR0915
    labeled: LabeledPath,
    report: Mapping[str, object],
    source: Path,
    source_sha256: str,
    *,
    gpus_per_node: int = 8,
) -> AcceptedResult:
    """Validate one complete harness output and extract measured arm records."""

    if report.get("schema_version") != 1:
        message = f"{labeled.label}: unsupported harness schema_version={report.get('schema_version')!r}"
        raise ReportInputError(message)
    if report.get("status") != "completed":
        message = f"{labeled.label}: harness status must be 'completed', got {report.get('status')!r}"
        raise ReportInputError(message)
    teardown_errors = report.get("teardown_errors")
    if teardown_errors:
        message = f"{labeled.label}: teardown_errors is non-empty"
        raise ReportInputError(message)

    manifest = _as_mapping(_required(report, "manifest", labeled.label), f"{labeled.label}.manifest")
    manifest_rows = _as_int(
        _required(manifest, "rows", f"{labeled.label}.manifest"),
        f"{labeled.label}.manifest.rows",
    )
    manifest_digest = _as_string(
        _required(manifest, "digest_sha256", f"{labeled.label}.manifest"),
        f"{labeled.label}.manifest.digest_sha256",
    )
    dataset = _as_mapping(_required(report, "dataset", labeled.label), f"{labeled.label}.dataset")
    dataset_uri = _as_string(
        _required(dataset, "uri", f"{labeled.label}.dataset"),
        f"{labeled.label}.dataset.uri",
    )
    dataset_version = str(_required(dataset, "version", f"{labeled.label}.dataset"))
    configuration = _as_mapping(
        _required(report, "configuration", labeled.label),
        f"{labeled.label}.configuration",
    )
    repeat_count = _as_int(
        _required(configuration, "repeat_count", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.repeat_count",
    )
    if repeat_count < 2:
        message = f"{labeled.label}: repeat_count={repeat_count}; measured evidence requires at least two repeats"
        raise ReportInputError(message)
    warmup_count = _as_int(
        _required(configuration, "warmup_count", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.warmup_count",
        allow_zero=True,
    )
    task_rows = _as_int(
        _required(configuration, "task_rows", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.task_rows",
    )
    coalesce_tasks = _as_int(
        _required(configuration, "coalesce_tasks", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.coalesce_tasks",
    )
    task_window_rows = _as_int(
        _required(configuration, "rows_per_coalesced_fetch", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.rows_per_coalesced_fetch",
    )
    if task_window_rows != task_rows * coalesce_tasks:
        message = (
            f"{labeled.label}: rows_per_coalesced_fetch={task_window_rows} does not equal "
            f"task_rows*coalesce_tasks={task_rows * coalesce_tasks}"
        )
        raise ReportInputError(message)
    rank_task_count = math.ceil(manifest_rows / task_rows)
    configured_task_count = configuration.get("ray_actor_input_blocks")
    if configured_task_count is not None:
        configured_task_count = _as_int(
            configured_task_count,
            f"{labeled.label}.configuration.ray_actor_input_blocks",
        )
        if configured_task_count != rank_task_count:
            message = (
                f"{labeled.label}: ray_actor_input_blocks={configured_task_count} does not equal "
                f"ceil(manifest_rows/task_rows)={rank_task_count}"
            )
            raise ReportInputError(message)
    lookup_batch_size = _as_int(
        _required(configuration, "lookup_batch_size", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.lookup_batch_size",
    )
    fetch_batch_size = _as_int(
        _required(configuration, "fetch_batch_size", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.fetch_batch_size",
    )
    io_threads = _as_int(
        _required(configuration, "io_threads", f"{labeled.label}.configuration"),
        f"{labeled.label}.configuration.io_threads",
    )
    unmatched = configuration.get("unmatched_reference_globs", [])
    if unmatched:
        message = f"{labeled.label}: unmatched_reference_globs is non-empty"
        raise ReportInputError(message)

    evidence_identity, comparison_eligibility_errors = _comparison_identity(report, dataset, configuration)
    rank_id, expected_rank_count, slurm_run_id, rank_identity_source = _rank_identity(labeled, report, source)
    node_count, gpu_count, resource_source = _resources(labeled, report, gpus_per_node)
    arms = _as_mapping(_required(report, "arms", labeled.label), f"{labeled.label}.arms")
    if not arms:
        message = f"{labeled.label}: arms is empty"
        raise ReportInputError(message)
    selected_backend = labeled.metadata.get("backend")
    if selected_backend is not None and selected_backend not in arms:
        message = f"{labeled.label}: selected backend {selected_backend!r} is absent from arms"
        raise ReportInputError(message)

    all_arm_digests: set[str] = set()
    measurements: list[Measurement] = []
    for backend, raw_arm in arms.items():
        if backend not in KNOWN_BACKENDS:
            message = f"{labeled.label}: unknown backend arm {backend!r}"
            raise ReportInputError(message)
        arm = _as_mapping(raw_arm, f"{labeled.label}.arms.{backend}")
        if arm.get("status") != "completed":
            message = f"{labeled.label}.arms.{backend}.status is not completed"
            raise ReportInputError(message)
        cold_setup = _as_mapping(
            _required(arm, "cold_setup", f"{labeled.label}.arms.{backend}"),
            f"{labeled.label}.arms.{backend}.cold_setup",
        )
        cold_setup_seconds = _as_number(
            _required(cold_setup, "wall_seconds", f"{labeled.label}.arms.{backend}.cold_setup"),
            f"{labeled.label}.arms.{backend}.cold_setup.wall_seconds",
        )
        _validate_warmups(
            arm,
            expected_count=warmup_count,
            context=f"{labeled.label}.arms.{backend}",
        )
        raw_repeats = _as_list(
            _required(arm, "repeats", f"{labeled.label}.arms.{backend}"),
            f"{labeled.label}.arms.{backend}.repeats",
        )
        if len(raw_repeats) != repeat_count:
            message = f"{labeled.label}.arms.{backend} has {len(raw_repeats)} repeats, expected {repeat_count}"
            raise ReportInputError(message)
        repeats = tuple(
            _validate_repeat(
                _as_mapping(item, f"{labeled.label}.arms.{backend}.repeats[{index}]"),
                context=f"{labeled.label}.arms.{backend}.repeats[{index}]",
                manifest_rows=manifest_rows,
            )
            for index, item in enumerate(raw_repeats)
        )
        digests = {repeat.output_digest_sha256 for repeat in repeats}
        if len(digests) != 1:
            message = f"{labeled.label}.arms.{backend} output digest changes across repeats"
            raise ReportInputError(message)
        all_arm_digests.update(digests)
        summary = _as_mapping(
            _required(arm, "summary", f"{labeled.label}.arms.{backend}"),
            f"{labeled.label}.arms.{backend}.summary",
        )
        if summary.get("stable_correctness_digest") is not True:
            message = f"{labeled.label}.arms.{backend}.summary does not confirm a stable correctness digest"
            raise ReportInputError(message)
        if backend in GPU_BACKENDS and gpu_count <= 0:
            message = f"{labeled.label}: GPU backend {backend!r} requires gpu_count > 0"
            raise ReportInputError(message)
        if selected_backend is None or selected_backend == backend:
            measurements.append(
                Measurement(
                    label=labeled.label,
                    sources=(_portable_source_path(source),),
                    source_sha256=(source_sha256,),
                    backend=backend,
                    backend_class="gpu" if backend in GPU_BACKENDS else "cpu",
                    node_count=node_count,
                    gpu_count=gpu_count,
                    resource_source=resource_source,
                    rank_count=1,
                    rank_task_counts=(rank_task_count,),
                    cold_setup_seconds_max=cold_setup_seconds,
                    cold_setup_seconds_sum=cold_setup_seconds,
                    manifest_rows=manifest_rows,
                    manifest_digest_sha256=(manifest_digest,),
                    workload_id=labeled.metadata.get(
                        "workload",
                        (
                            f"dataset-{hashlib.sha256(dataset_uri.encode('utf-8')).hexdigest()[:16]}"
                            f"@{dataset_version}:rows={manifest_rows}"
                        ),
                    ),
                    workload_id_source=(
                        "label" if "workload" in labeled.metadata else "dataset_version_and_total_rows"
                    ),
                    dataset_uri=dataset_uri,
                    dataset_version=dataset_version,
                    task_rows=task_rows,
                    coalesce_tasks=coalesce_tasks,
                    task_window_rows=task_window_rows,
                    lookup_batch_size=lookup_batch_size,
                    fetch_batch_size=fetch_batch_size,
                    io_threads=io_threads,
                    evidence_identity=evidence_identity,
                    comparison_eligibility_errors=comparison_eligibility_errors,
                    rank_ids=(() if rank_id is None else (rank_id,)),
                    slurm_run_id=slurm_run_id,
                    repeats=repeats,
                )
            )

    if len(all_arm_digests) != 1:
        message = f"{labeled.label}: correctness digests differ across arms"
        raise ReportInputError(message)
    if len(arms) > 1 and report.get("cross_arm_correctness_digest_match") is not True:
        message = f"{labeled.label}: cross_arm_correctness_digest_match is not true"
        raise ReportInputError(message)
    return AcceptedResult(
        label=labeled.label,
        display_name=labeled.display_name,
        sources=(_portable_source_path(source),),
        source_sha256=(source_sha256,),
        node_count=node_count,
        gpu_count=gpu_count,
        resource_source=resource_source,
        rank_id=rank_id,
        expected_rank_count=expected_rank_count,
        slurm_run_id=slurm_run_id,
        rank_identity_source=rank_identity_source,
        measurements=tuple(measurements),
    )


def _signature_id(signature: tuple[object, ...]) -> str:
    encoded = json.dumps(signature, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def strong_scaling(measurements: Sequence[Measurement]) -> list[dict[str, object]]:
    """Calculate strong-scaling speedup and efficiency for compatible records."""

    groups: dict[tuple[object, ...], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.comparison_eligibility_errors:
            continue
        axis = "gpus" if measurement.backend_class == "gpu" else "nodes"
        key = (
            measurement.backend,
            axis,
            *measurement.workload_signature(include_window=True),
            sum(measurement.rank_task_counts),
        )
        groups[key].append(measurement)
    output: list[dict[str, object]] = []
    for signature, records in sorted(groups.items(), key=lambda item: repr(item[0])):
        axis = str(signature[1])
        resource = (lambda item: item.gpu_count) if axis == "gpus" else (lambda item: item.node_count)
        if len({resource(record) for record in records}) < 2:
            continue
        ordered = sorted(records, key=lambda item: (resource(item), item.label))
        baseline = ordered[0]
        baseline_resources = resource(baseline)
        baseline_metrics = baseline.metric_stats()
        points = []
        for record in ordered:
            metrics = record.metric_stats()
            resource_ratio = resource(record) / baseline_resources
            speedup = baseline_metrics["wall_seconds"].median / metrics["wall_seconds"].median
            points.append(
                {
                    "label": record.label,
                    "resource_count": resource(record),
                    "median_wall_seconds": metrics["wall_seconds"].median,
                    "median_images_per_second": metrics["images_per_second"].median,
                    "summed_images": record.manifest_rows,
                    "global_task_count": sum(record.rank_task_counts),
                    "per_rank_task_counts": list(record.rank_task_counts),
                    "per_rank_window_rows": record.task_window_rows,
                    "rank_count": record.rank_count,
                    "task_rows": record.task_rows,
                    "coalesce_tasks": record.coalesce_tasks,
                    "lookup_batch_size": record.lookup_batch_size,
                    "fetch_batch_size": record.fetch_batch_size,
                    "io_threads": record.io_threads,
                    "correctness_digest_sha256": record.repeats[0].output_digest_sha256,
                    "speedup_vs_baseline": speedup,
                    "strong_scaling_efficiency": speedup / resource_ratio,
                    "resource_ratio": resource_ratio,
                    "classification": "derived_from_measured",
                }
            )
        output.append(
            {
                "group_id": _signature_id(signature),
                "backend": baseline.backend,
                "resource_axis": axis,
                "baseline_label": baseline.label,
                "baseline_resource_count": baseline_resources,
                "workload_id": baseline.workload_id,
                "summed_images": baseline.manifest_rows,
                "compatibility": (
                    "same workload, dataset/version, payload projection, immutable sidecar, read/concurrency/"
                    "cache/validation policies, code/package identity, task/coalescing window, lookup/fetch "
                    "batches, I/O threads, global task count, backend, and summed image count; every rank "
                    "has a stable validated correctness digest and a complete rank identity"
                ),
                "correctness_digest_note": (
                    "Digest values are reported per point but are not compared across shardings because "
                    "SHA-256 stream digests are not composable."
                ),
                "points": points,
                "classification": "derived_from_measured",
            }
        )
    return output


def task_window_sweeps(measurements: Sequence[Measurement]) -> list[dict[str, object]]:
    """Compare task/coalescing windows at fixed backend and resources."""

    groups: dict[tuple[object, ...], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.comparison_eligibility_errors:
            continue
        key = (
            measurement.backend,
            measurement.node_count,
            measurement.gpu_count,
            *measurement.workload_signature(include_window=False),
        )
        groups[key].append(measurement)
    output: list[dict[str, object]] = []
    for signature, records in sorted(groups.items(), key=lambda item: repr(item[0])):
        if len({record.task_window_rows for record in records}) < 2:
            continue
        ordered = sorted(records, key=lambda item: (item.task_window_rows, item.label))
        baseline = ordered[0]
        baseline_metrics = baseline.metric_stats()
        best_wall = min(record.metric_stats()["wall_seconds"].median for record in ordered)
        points = []
        for record in ordered:
            metrics = record.metric_stats()
            points.append(
                {
                    "label": record.label,
                    "task_rows": record.task_rows,
                    "coalesce_tasks": record.coalesce_tasks,
                    "task_window_rows": record.task_window_rows,
                    "global_task_count": sum(record.rank_task_counts),
                    "median_wall_seconds": metrics["wall_seconds"].median,
                    "median_images_per_second": metrics["images_per_second"].median,
                    "wall_speedup_vs_smallest_window": (
                        baseline_metrics["wall_seconds"].median / metrics["wall_seconds"].median
                    ),
                    "wall_time_relative_to_best": metrics["wall_seconds"].median / best_wall,
                    "classification": "derived_from_measured",
                }
            )
        output.append(
            {
                "group_id": _signature_id(signature),
                "backend": baseline.backend,
                "node_count": baseline.node_count,
                "gpu_count": baseline.gpu_count,
                "baseline_label": baseline.label,
                "points": points,
                "classification": "derived_from_measured",
            }
        )
    return output


def cpu_gpu_speedups(measurements: Sequence[Measurement]) -> list[dict[str, object]]:
    """Pair workload-identical CPU and GPU records and report measured ratios."""

    groups: dict[tuple[object, ...], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.comparison_eligibility_errors:
            continue
        groups[measurement.workload_signature(include_window=True)].append(measurement)
    output: list[dict[str, object]] = []
    for signature, records in sorted(groups.items(), key=lambda item: repr(item[0])):
        cpu_records = sorted(
            (record for record in records if record.backend_class == "cpu"),
            key=lambda item: (item.backend, item.label),
        )
        gpu_records = sorted(
            (record for record in records if record.backend_class == "gpu"),
            key=lambda item: (item.backend, item.label),
        )
        for cpu, gpu in product(cpu_records, gpu_records):
            cpu_metrics = cpu.metric_stats()
            gpu_metrics = gpu.metric_stats()
            output.append(
                {
                    "group_id": _signature_id(signature),
                    "cpu": {
                        "label": cpu.label,
                        "backend": cpu.backend,
                        "node_count": cpu.node_count,
                        "gpu_count": cpu.gpu_count,
                        "median_wall_seconds": cpu_metrics["wall_seconds"].median,
                    },
                    "gpu": {
                        "label": gpu.label,
                        "backend": gpu.backend,
                        "node_count": gpu.node_count,
                        "gpu_count": gpu.gpu_count,
                        "median_wall_seconds": gpu_metrics["wall_seconds"].median,
                    },
                    "wall_time_speedup": (cpu_metrics["wall_seconds"].median / gpu_metrics["wall_seconds"].median),
                    "images_per_second_speedup": (
                        gpu_metrics["images_per_second"].median / cpu_metrics["images_per_second"].median
                    ),
                    "payload_mib_per_second_speedup": (
                        gpu_metrics["payload_mib_per_second"].median / cpu_metrics["payload_mib_per_second"].median
                    ),
                    "classification": "derived_from_measured",
                }
            )
    return output


def build_report(
    accepted_results: Sequence[AcceptedResult],
    projections: Sequence[Mapping[str, object]],
    *,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build the JSON-serializable aggregate report."""

    measurements = [measurement for result in accepted_results for measurement in result.measurements]
    if not measurements:
        message = "no measured backend records remain after validation/selection"
        raise ReportInputError(message)
    comparison_exclusions = [
        {
            "label": measurement.label,
            "backend": measurement.backend,
            "reasons": list(measurement.comparison_eligibility_errors),
        }
        for measurement in sorted(measurements, key=lambda item: (item.backend, item.label))
        if measurement.comparison_eligibility_errors
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "report": "gpu_lance_scaling_report",
        "generated_at": generated_at or datetime.now(tz=UTC).isoformat(),
        "classification_legend": {
            "measured": "Directly observed by a completed and correctness-validated harness run.",
            "derived_from_measured": "Arithmetic comparison of compatible measured records.",
            "projection": "A separate extrapolation document; never included in measured comparisons.",
        },
        "measured": {
            "accepted_results": [result.as_dict() for result in accepted_results],
            "performance": [
                measurement.as_dict()
                for measurement in sorted(measurements, key=lambda item: (item.backend, item.label))
            ],
            "strong_scaling": strong_scaling(measurements),
            "task_window_sweeps": task_window_sweeps(measurements),
            "cpu_vs_gpu_speedups": cpu_gpu_speedups(measurements),
            "comparison_exclusions": comparison_exclusions,
            "classification": "measured_and_derived_from_measured",
        },
        "projections": {
            "classification": "projection",
            "excluded_from_measured_comparisons": True,
            "documents": list(projections),
        },
        "validation": {
            "policy": (
                "Only completed schema-v1 harnesses with at least two complete repeats, correct rows/payloads, "
                "stable repeat digests, and matching cross-arm digests are accepted. Comparisons additionally "
                "require a complete payload/sidecar/policy/runtime identity and complete multi-rank identity."
            ),
            "multi_rank_aggregation": {
                "images": "sum rank manifest rows",
                "wall_seconds": "max rank wall_seconds per repeat",
                "warm_process_seconds": "max rank warm_process_seconds per repeat",
                "payload_and_io": "sum rank bytes/counters per repeat",
                "cold_setup": "both max rank wall and sum rank wall",
                "classification": "derived_from_measured",
            },
            "accepted_result_count": len(accepted_results),
            "accepted_measurement_count": len(measurements),
        },
    }


def _format_number(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return f"{value:.3g}"


def _label_prefix(value: object) -> str:
    return str(value).split("[", maxsplit=1)[0]


def _metric_cell(metric: Mapping[str, object], suffix: str = "") -> str:
    median = _as_number(_required(metric, "median", "metric"), "metric.median", allow_zero=True)
    minimum = _as_number(_required(metric, "min", "metric"), "metric.min", allow_zero=True)
    maximum = _as_number(_required(metric, "max", "metric"), "metric.max", allow_zero=True)
    return f"{_format_number(median)}{suffix} [{_format_number(minimum)}, {_format_number(maximum)}]"


def _optional_metric_cell(metrics: Mapping[str, object], name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "n/a"
    return _metric_cell(_as_mapping(value, f"metrics.{name}"))


def render_markdown(report: Mapping[str, object]) -> str:  # noqa: C901, PLR0912, PLR0915
    """Render measured and projected evidence in visibly separate sections."""

    measured = _as_mapping(_required(report, "measured", "report"), "report.measured")
    performance = _as_list(_required(measured, "performance", "report.measured"), "report.measured.performance")
    lines = [
        "# GPU Lance scaling report",
        "",
        "> Measured tables contain only completed, correctness-validated harness runs. Projections are separate below.",
        "",
        "## Measured performance",
        "",
        "Values are median `[min, max]` across measured repeats.",
        "",
        "| Label | Backend | Ranks | Nodes | GPUs | Global tasks | Per-rank window | Wall | Images/s | Payload MiB/s | Images/s/node |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for raw_record in performance:
        record = _as_mapping(raw_record, "measured.performance[]")
        resources = _as_mapping(_required(record, "resources", "performance"), "performance.resources")
        window = _as_mapping(_required(record, "task_window", "performance"), "performance.task_window")
        metrics = _as_mapping(_required(record, "metrics", "performance"), "performance.metrics")
        lines.append(
            "| {label} | {backend} | {ranks} | {nodes} | {gpus} | {global_tasks} | {window} | {wall} | {images} | {payload} | {per_node} |".format(
                label=_label_prefix(record["label"]),
                backend=record["backend"],
                ranks=resources["rank_count"],
                nodes=resources["node_count"],
                gpus=resources["gpu_count"],
                global_tasks=window["global_task_count"],
                window=window["per_rank_rows_per_coalesced_fetch"],
                wall=_metric_cell(_as_mapping(metrics["wall_seconds"], "metrics.wall_seconds"), " s"),
                images=_metric_cell(_as_mapping(metrics["images_per_second"], "metrics.images_per_second")),
                payload=_metric_cell(_as_mapping(metrics["payload_mib_per_second"], "metrics.payload_mib_per_second")),
                per_node=_metric_cell(
                    _as_mapping(metrics["images_per_second_per_node"], "metrics.images_per_second_per_node")
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Measured setup and I/O",
            "",
            "For multi-rank inputs, setup max is the critical path; setup sum and I/O counters are sums across ranks.",
            "",
            "| Label | Backend | Setup max (s) | Setup sum (rank-s) | Payload bytes | Lance IOPS | Lance read bytes | Lookup calls | Fetch calls |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for raw_record in performance:
        record = _as_mapping(raw_record, "measured.performance[]")
        setup = _as_mapping(_required(record, "setup_seconds", "performance"), "performance.setup_seconds")
        metrics = _as_mapping(_required(record, "metrics", "performance"), "performance.metrics")
        lines.append(
            "| {label} | {backend} | {setup_max:.4g} | {setup_sum:.4g} | {payload} | {iops} | {read_bytes} | {lookups} | {fetches} |".format(
                label=_label_prefix(record["label"]),
                backend=record["backend"],
                setup_max=float(setup["max_rank_wall"]),
                setup_sum=float(setup["sum_rank_wall"]),
                payload=_optional_metric_cell(metrics, "payload_bytes"),
                iops=_optional_metric_cell(metrics, "lance_read_iops"),
                read_bytes=_optional_metric_cell(metrics, "lance_read_bytes"),
                lookups=_optional_metric_cell(metrics, "lookup_calls"),
                fetches=_optional_metric_cell(metrics, "fetch_calls"),
            )
        )

    exclusions = _as_list(
        _required(measured, "comparison_exclusions", "report.measured"),
        "report.measured.comparison_exclusions",
    )
    lines.extend(
        [
            "",
            "## Comparison eligibility",
            "",
            "Measured rows remain visible, but records below are excluded from every derived comparison.",
            "",
            "| Label | Backend | Fail-closed reasons |",
            "|---|---|---|",
        ]
    )
    if not exclusions:
        lines.append("| All measured rows eligible | | |")
    for raw_exclusion in exclusions:
        exclusion = _as_mapping(raw_exclusion, "measured.comparison_exclusions[]")
        reasons = _as_list(exclusion["reasons"], "comparison_exclusion.reasons")
        lines.append(
            f"| {_label_prefix(exclusion['label'])} | {exclusion['backend']} | {'; '.join(map(str, reasons))} |"
        )

    scaling = _as_list(_required(measured, "strong_scaling", "report.measured"), "measured.strong_scaling")
    lines.extend(
        [
            "",
            "## Strong scaling",
            "",
            "Speedup uses median measured wall time; efficiency is speedup divided by resource growth.",
            "",
            "| Backend | Label | Resource | Count | Global tasks | Per-rank window | Wall (s) | Speedup | Efficiency |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if not scaling:
        lines.append("| No compatible multi-resource measurements | | | | | | | | |")
    for raw_group in scaling:
        group = _as_mapping(raw_group, "strong_scaling[]")
        points = _as_list(group["points"], "strong_scaling.points")
        for raw_point in points:
            point = _as_mapping(raw_point, "strong_scaling.points[]")
            lines.append(
                "| {backend} | {label} | {axis} | {count} | {global_tasks} | {window} | {wall:.4g} | {speedup:.3f}x | {efficiency:.1%} |".format(
                    backend=group["backend"],
                    label=_label_prefix(point["label"]),
                    axis=group["resource_axis"],
                    count=point["resource_count"],
                    global_tasks=point["global_task_count"],
                    window=point["per_rank_window_rows"],
                    wall=float(point["median_wall_seconds"]),
                    speedup=float(point["speedup_vs_baseline"]),
                    efficiency=float(point["strong_scaling_efficiency"]),
                )
            )

    sweeps = _as_list(
        _required(measured, "task_window_sweeps", "report.measured"),
        "measured.task_window_sweeps",
    )
    lines.extend(
        [
            "",
            "## Task-window sweep",
            "",
            "| Backend | Label | Global tasks | Task rows | Coalesced tasks | Per-rank window rows | Wall (s) | Speedup vs smallest | Relative to best |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if not sweeps:
        lines.append("| No compatible multi-window measurements | | | | | | | | |")
    for raw_group in sweeps:
        group = _as_mapping(raw_group, "task_window_sweeps[]")
        for raw_point in _as_list(group["points"], "task_window_sweeps.points"):
            point = _as_mapping(raw_point, "task_window_sweeps.points[]")
            lines.append(
                "| {backend} | {label} | {global_tasks} | {task_rows} | {tasks} | {window} | {wall:.4g} | {speedup:.3f}x | {relative:.3f}x |".format(
                    backend=group["backend"],
                    label=_label_prefix(point["label"]),
                    global_tasks=point["global_task_count"],
                    task_rows=point["task_rows"],
                    tasks=point["coalesce_tasks"],
                    window=point["task_window_rows"],
                    wall=float(point["median_wall_seconds"]),
                    speedup=float(point["wall_speedup_vs_smallest_window"]),
                    relative=float(point["wall_time_relative_to_best"]),
                )
            )

    comparisons = _as_list(
        _required(measured, "cpu_vs_gpu_speedups", "report.measured"),
        "measured.cpu_vs_gpu_speedups",
    )
    lines.extend(
        [
            "",
            "## CPU vs GPU",
            "",
            "| CPU | GPU | CPU resources | GPU resources | Wall speedup | Images/s speedup | Payload speedup |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not comparisons:
        lines.append("| No workload-compatible CPU/GPU measurements | | | | | | |")
    for raw_comparison in comparisons:
        comparison = _as_mapping(raw_comparison, "cpu_vs_gpu_speedups[]")
        cpu = _as_mapping(comparison["cpu"], "cpu_vs_gpu.cpu")
        gpu = _as_mapping(comparison["gpu"], "cpu_vs_gpu.gpu")
        lines.append(
            "| {cpu_backend} ({cpu_label}) | {gpu_backend} ({gpu_label}) | {cpu_nodes}n | {gpu_nodes}n/{gpu_count}g | {wall:.3f}x | {images:.3f}x | {payload:.3f}x |".format(
                cpu_backend=cpu["backend"],
                cpu_label=_label_prefix(cpu["label"]),
                gpu_backend=gpu["backend"],
                gpu_label=_label_prefix(gpu["label"]),
                cpu_nodes=cpu["node_count"],
                gpu_nodes=gpu["node_count"],
                gpu_count=gpu["gpu_count"],
                wall=float(comparison["wall_time_speedup"]),
                images=float(comparison["images_per_second_speedup"]),
                payload=float(comparison["payload_mib_per_second_speedup"]),
            )
        )

    projections = _as_mapping(_required(report, "projections", "report"), "report.projections")
    projection_documents = _as_list(
        _required(projections, "documents", "report.projections"),
        "report.projections.documents",
    )
    lines.extend(
        [
            "",
            "## Projections (not measured)",
            "",
            "Projection documents are retained for provenance and are excluded from every table above.",
            "",
            "| Label | Source | Model |",
            "|---|---|---|",
        ]
    )
    if not projection_documents:
        lines.append("| None supplied | | |")
    for raw_projection in projection_documents:
        projection = _as_mapping(raw_projection, "projections.documents[]")
        document = _as_mapping(projection["document"], "projection.document")
        lines.append(
            f"| {_label_prefix(projection['label'])} | {projection['source']} | {document.get('model', 'unspecified')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def _read_json_object(path: Path, context: str) -> Mapping[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"cannot read {context} {path}: {error}"
        raise ReportInputError(message) from error
    return _as_mapping(loaded, context)


def _expanded_paths(labeled: LabeledPath) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw_part in labeled.path_expression.split(","):
        part = str(Path(raw_part.strip()).expanduser())
        if not part:
            message = f"{labeled.label}: empty path in comma-separated path expression"
            raise ReportInputError(message)
        matches = sorted(glob(part))
        if not matches and has_magic(part):
            message = f"{labeled.label}: path glob matched no files: {part}"
            raise ReportInputError(message)
        paths.extend(Path(match) for match in matches)
        if not matches:
            paths.append(Path(part))
    resolved = tuple(path.resolve() for path in paths)
    if not resolved:
        message = f"{labeled.label}: no result paths were supplied"
        raise ReportInputError(message)
    if len(resolved) != len(set(resolved)):
        message = f"{labeled.label}: path expression resolves the same file more than once"
        raise ReportInputError(message)
    return resolved


def _sum_optional(values: Sequence[float | None], context: str) -> float | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        message = f"{context} is present on only some ranks"
        raise ReportInputError(message)
    return sum(float(value) for value in values if value is not None)


def _combine_rank_results(  # noqa: C901, PLR0912, PLR0915
    labeled: LabeledPath, ranks: Sequence[AcceptedResult]
) -> AcceptedResult:
    if not ranks:
        message = f"{labeled.label}: cannot combine an empty rank set"
        raise ReportInputError(message)
    if len(ranks) == 1:
        explicitly_expected = _metadata_count(labeled, "ranks") or ranks[0].expected_rank_count
        if explicitly_expected is not None and explicitly_expected != 1:
            message = f"{labeled.label}: found 1 rank file; expected exactly {explicitly_expected}"
            raise ReportInputError(message)
        return ranks[0]
    if "nodes" not in labeled.metadata:
        message = f"{labeled.label}: multi-rank results require nodes=N in bracket label metadata"
        raise ReportInputError(message)

    expected_counts = {rank.expected_rank_count for rank in ranks if rank.expected_rank_count is not None}
    labeled_rank_count = _metadata_count(labeled, "ranks")
    if labeled_rank_count is not None:
        expected_counts.add(labeled_rank_count)
    if not expected_counts:
        # The launch contract is one rank per node unless ranks=N is stated. Filename
        # identities below independently corroborate the label-derived expectation.
        expected_counts.add(_metadata_count(labeled, "nodes") or 0)
    if len(expected_counts) != 1:
        message = f"{labeled.label}: rank files disagree on expected rank count: {sorted(expected_counts)}"
        raise ReportInputError(message)
    expected_rank_count = next(iter(expected_counts))
    if expected_rank_count <= 0 or len(ranks) != expected_rank_count:
        message = f"{labeled.label}: found {len(ranks)} rank files; expected exactly {expected_rank_count}"
        raise ReportInputError(message)
    rank_ids = [rank.rank_id for rank in ranks]
    if any(rank_id is None for rank_id in rank_ids):
        message = f"{labeled.label}: every multi-rank artifact requires an embedded or filename rank id"
        raise ReportInputError(message)
    concrete_rank_ids = [int(rank_id) for rank_id in rank_ids if rank_id is not None]
    if len(set(concrete_rank_ids)) != len(concrete_rank_ids):
        message = f"{labeled.label}: duplicate rank ids: {concrete_rank_ids}"
        raise ReportInputError(message)
    if set(concrete_rank_ids) != set(range(expected_rank_count)):
        message = (
            f"{labeled.label}: rank ids must be exactly 0..{expected_rank_count - 1}, got {sorted(concrete_rank_ids)}"
        )
        raise ReportInputError(message)
    ranks = tuple(rank for _, rank in sorted(zip(concrete_rank_ids, ranks, strict=True)))
    slurm_ids = [rank.slurm_run_id for rank in ranks]
    if any(slurm_ids) and (any(value is None for value in slurm_ids) or len(set(slurm_ids)) != 1):
        message = f"{labeled.label}: rank files do not share one complete Slurm run identity: {slurm_ids}"
        raise ReportInputError(message)
    slurm_run_id = slurm_ids[0] if slurm_ids and slurm_ids[0] is not None else None

    first = ranks[0]
    for rank in ranks[1:]:
        if (rank.node_count, rank.gpu_count) != (first.node_count, first.gpu_count):
            message = f"{labeled.label}: rank files disagree on node/GPU resources"
            raise ReportInputError(message)
        if [item.backend for item in rank.measurements] != [item.backend for item in first.measurements]:
            message = f"{labeled.label}: rank files contain different backend arms"
            raise ReportInputError(message)

    combined_measurements: list[Measurement] = []
    for backend_index, first_measurement in enumerate(first.measurements):
        rank_measurements = [rank.measurements[backend_index] for rank in ranks]
        comparable_fields = (
            "backend",
            "backend_class",
            "dataset_uri",
            "dataset_version",
            "task_rows",
            "coalesce_tasks",
            "task_window_rows",
            "lookup_batch_size",
            "fetch_batch_size",
            "io_threads",
            "evidence_identity",
        )
        for rank_measurement in rank_measurements[1:]:
            differences = [
                field
                for field in comparable_fields
                if getattr(rank_measurement, field) != getattr(first_measurement, field)
            ]
            if differences:
                message = (
                    f"{labeled.label}: rank configuration differs for {first_measurement.backend}: "
                    f"{', '.join(differences)}"
                )
                raise ReportInputError(message)
            if len(rank_measurement.repeats) != len(first_measurement.repeats):
                message = f"{labeled.label}: ranks have different repeat counts"
                raise ReportInputError(message)

        total_images = sum(item.manifest_rows for item in rank_measurements)
        combined_repeats: list[RepeatMeasurement] = []
        for repeat_index in range(len(first_measurement.repeats)):
            rank_repeats = [item.repeats[repeat_index] for item in rank_measurements]
            warm_seconds = max(item.warm_process_seconds for item in rank_repeats)
            payload_bytes = sum(item.payload_bytes for item in rank_repeats)
            digest_material = "|".join(item.output_digest_sha256 for item in rank_repeats).encode()
            combined_repeats.append(
                RepeatMeasurement(
                    wall_seconds=max(item.wall_seconds for item in rank_repeats),
                    warm_process_seconds=warm_seconds,
                    images_per_second=total_images / warm_seconds,
                    payload_mib_per_second=payload_bytes / (MIB * warm_seconds),
                    payload_bytes=payload_bytes,
                    output_digest_sha256=hashlib.sha256(digest_material).hexdigest(),
                    lance_read_iops=_sum_optional(
                        [item.lance_read_iops for item in rank_repeats],
                        f"{labeled.label}.{first_measurement.backend}.lance_read_iops",
                    ),
                    lance_read_bytes=_sum_optional(
                        [item.lance_read_bytes for item in rank_repeats],
                        f"{labeled.label}.{first_measurement.backend}.lance_read_bytes",
                    ),
                    lookup_calls=_sum_optional(
                        [item.lookup_calls for item in rank_repeats],
                        f"{labeled.label}.{first_measurement.backend}.lookup_calls",
                    ),
                    fetch_calls=_sum_optional(
                        [item.fetch_calls for item in rank_repeats],
                        f"{labeled.label}.{first_measurement.backend}.fetch_calls",
                    ),
                )
            )
        workload_id = labeled.metadata.get(
            "workload",
            (
                f"dataset-{hashlib.sha256(first_measurement.dataset_uri.encode('utf-8')).hexdigest()[:16]}"
                f"@{first_measurement.dataset_version}:rows={total_images}"
            ),
        )
        combined_measurements.append(
            Measurement(
                label=labeled.label,
                sources=tuple(source for item in rank_measurements for source in item.sources),
                source_sha256=tuple(digest for item in rank_measurements for digest in item.source_sha256),
                backend=first_measurement.backend,
                backend_class=first_measurement.backend_class,
                node_count=first.node_count,
                gpu_count=first.gpu_count,
                resource_source=first.resource_source,
                rank_count=len(ranks),
                rank_task_counts=tuple(
                    task_count for item in rank_measurements for task_count in item.rank_task_counts
                ),
                cold_setup_seconds_max=max(item.cold_setup_seconds_max for item in rank_measurements),
                cold_setup_seconds_sum=sum(item.cold_setup_seconds_sum for item in rank_measurements),
                manifest_rows=total_images,
                manifest_digest_sha256=tuple(
                    digest for item in rank_measurements for digest in item.manifest_digest_sha256
                ),
                workload_id=workload_id,
                workload_id_source=("label" if "workload" in labeled.metadata else "dataset_version_and_total_rows"),
                dataset_uri=first_measurement.dataset_uri,
                dataset_version=first_measurement.dataset_version,
                task_rows=first_measurement.task_rows,
                coalesce_tasks=first_measurement.coalesce_tasks,
                task_window_rows=first_measurement.task_window_rows,
                lookup_batch_size=first_measurement.lookup_batch_size,
                fetch_batch_size=first_measurement.fetch_batch_size,
                io_threads=first_measurement.io_threads,
                evidence_identity=first_measurement.evidence_identity,
                comparison_eligibility_errors=tuple(
                    dict.fromkeys(error for item in rank_measurements for error in item.comparison_eligibility_errors)
                ),
                rank_ids=tuple(rank_id for item in rank_measurements for rank_id in item.rank_ids),
                slurm_run_id=slurm_run_id,
                repeats=tuple(combined_repeats),
            )
        )
    return AcceptedResult(
        label=labeled.label,
        display_name=labeled.display_name,
        sources=tuple(source for rank in ranks for source in rank.sources),
        source_sha256=tuple(digest for rank in ranks for digest in rank.source_sha256),
        node_count=first.node_count,
        gpu_count=first.gpu_count,
        resource_source=first.resource_source,
        rank_id=None,
        expected_rank_count=expected_rank_count,
        slurm_run_id=slurm_run_id,
        rank_identity_source={
            "rank_id": "validated_artifact_set",
            "expected_rank_count": "rank metadata or nodes plus contiguous rank ids",
            "slurm_run_id": "common across ranks" if slurm_run_id is not None else "absent",
        },
        measurements=tuple(combined_measurements),
    )


def _require_terminal_eligibility(path: Path, source_sha256: str, label: str) -> None:
    parent = path.parent
    indicators = (
        parent / "run_identity.json",
        parent / "telemetry_validation.json",
        parent / "eligibility.json",
    )
    if not any(indicator.exists() for indicator in indicators):
        return
    eligibility_path = parent / "eligibility.json"
    if not eligibility_path.is_file():
        message = (
            f"{label}: saturation-adjacent artifact lacks required terminal eligibility.json; "
            "telemetry status alone is not benchmark eligibility"
        )
        raise ReportInputError(message)
    eligibility = _read_json_object(eligibility_path, f"terminal eligibility for {label}")
    eligibility_schema_version = eligibility.get("schema_version")
    if isinstance(eligibility_schema_version, bool) or eligibility_schema_version not in {
        1,
        _TERMINAL_ELIGIBILITY_SCHEMA_VERSION,
    }:
        message = f"{label}: terminal eligibility schema_version is {eligibility_schema_version!r}, not 1 or 2"
        raise ReportInputError(message)
    if eligibility.get("terminal") is not True or eligibility.get("status") != "eligible":
        message = f"{label}: terminal eligibility status is {eligibility.get('status')!r}, not 'eligible'"
        raise ReportInputError(message)
    evidence_class = eligibility.get("evidence_class")
    benchmark_validation = eligibility.get("benchmark_validation")
    validation_waves = benchmark_validation.get("waves") if isinstance(benchmark_validation, Mapping) else None
    if eligibility_schema_version == 1 and evidence_class is None and validation_waves in _PRIMARY_SATURATION_WAVES:
        evidence_class = _PRIMARY_SATURATION_EVIDENCE_CLASS
    if evidence_class != _PRIMARY_SATURATION_EVIDENCE_CLASS or validation_waves not in _PRIMARY_SATURATION_WAVES:
        message = (
            f"{label}: terminal eligibility evidence_class={evidence_class!r}, "
            f"benchmark_validation.waves={validation_waves!r}; expected primary_saturation with 4 or 8 waves"
        )
        raise ReportInputError(message)
    artifacts = _as_mapping(_required(eligibility, "artifacts", label), f"{label}.eligibility.artifacts")
    benchmark = _as_mapping(_required(artifacts, "benchmark", label), f"{label}.eligibility.artifacts.benchmark")
    recorded_sha256 = _as_string(
        _required(benchmark, "sha256", f"{label}.eligibility.artifacts.benchmark"),
        f"{label}.eligibility.artifacts.benchmark.sha256",
    )
    if recorded_sha256 != source_sha256:
        message = f"{label}: terminal eligibility benchmark digest does not match the loaded artifact"
        raise ReportInputError(message)


def load_result(labeled: LabeledPath, *, gpus_per_node: int = 8) -> AcceptedResult:
    paths = _expanded_paths(labeled)
    ranks = []
    for rank_index, path in enumerate(paths):
        report = _read_json_object(path, f"result {labeled.label} rank {rank_index}")
        source_sha256 = _file_sha256(path)
        _require_terminal_eligibility(path, source_sha256, labeled.label)
        ranks.append(
            validate_harness_result(
                labeled,
                report,
                path,
                source_sha256,
                gpus_per_node=gpus_per_node,
            )
        )
    return _combine_rank_results(labeled, ranks)


def load_projection(labeled: LabeledPath) -> dict[str, object]:
    if labeled.metadata:
        message = f"projection label {labeled.label!r} must not contain resource/backend metadata"
        raise ReportInputError(message)
    paths = _expanded_paths(labeled)
    if len(paths) != 1:
        message = f"projection {labeled.label!r} must resolve to exactly one file"
        raise ReportInputError(message)
    path = paths[0]
    document = _read_json_object(path, f"projection {labeled.label}")
    return {
        "label": labeled.label,
        "source": _portable_source_path(path),
        "source_sha256": _file_sha256(path),
        "classification": "projection",
        "document": dict(document),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _self_test_fixture(
    *,
    scale: float = 1.0,
    coalesce_tasks: int = 1,
    correct: bool = True,
) -> dict[str, object]:
    rows = 100
    payload_bytes = 100 * MIB

    def arm(warm_seconds: float, wall_seconds: float) -> dict[str, object]:
        repeat = {
            "status": "completed",
            "wall_seconds": wall_seconds,
            "warm_process_seconds": warm_seconds,
            "images_per_second": rows / warm_seconds,
            "payload_mib_per_second": payload_bytes / (MIB * warm_seconds),
            "payload_bytes": payload_bytes,
            "lance_read_iops": 4,
            "lance_read_bytes": payload_bytes,
            "lookup_calls": 1,
            "fetch_calls": 2,
            "correctness": {
                "correct": correct,
                "row_count": rows,
                "expected_row_count": rows,
                "present_rows": rows,
                "missing_payload_rows": 0,
                "payload_bytes": payload_bytes,
                "output_digest_sha256": "digest",
            },
        }
        return {
            "status": "completed",
            "cold_setup": {"wall_seconds": 1.0, "backend_metrics": {}},
            "warmups": [],
            "repeats": [dict(repeat), dict(repeat)],
            "summary": {"stable_correctness_digest": True},
        }

    return {
        "schema_version": 1,
        "status": "completed",
        "environment": {
            "python": "3.12.0",
            "packages": {
                "nemo-curator": "1.3.0+test",
                "pyarrow": "22.0.0",
                "pylance": "9.0.0b11",
            },
        },
        "manifest": {"rows": rows, "digest_sha256": "manifest"},
        "dataset": {
            "uri": "s3://example/images",
            "version": 1,
            "source_columns": {"image": "image"},
            "storage_option_keys": ["endpoint"],
        },
        "configuration": {
            "repeat_count": 2,
            "warmup_count": 0,
            "task_rows": 100,
            "coalesce_tasks": coalesce_tasks,
            "rows_per_coalesced_fetch": rows * coalesce_tasks,
            "lookup_batch_size": 2_000,
            "fetch_batch_size": 128,
            "io_threads": 16,
            "max_lookup_bytes": 1024,
            "max_pending_fetch_batches": 4,
            "payload_read_mode": "sparse",
            "take_scan_batch_readahead": 16,
            "index_mirror": None,
            "copy_index_to_node_local": False,
            "validate_payload_keys": False,
            "reference_manifest_uri": "/indexes/sidecar-manifest.json",
            "reference_manifest_sha256": "a" * 64,
            "reference_storage_option_keys": [],
            "unmatched_reference_globs": [],
            "ray_gpu_actors": max(1, round(scale)),
        },
        "arms": {
            "cpu_lance_column_fetch_stage": arm(20.0, 22.0),
            "gpu_lance_column_fetch_stage": arm(10.0 / scale, 12.0 / scale),
        },
        "cross_arm_correctness_digest_match": True,
    }


def _self_check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_tests() -> None:
    """Exercise validation, scaling, task-window, and projection separation."""

    one_label = parse_labeled_path("one[nodes=1,gpus=1]=/tmp/one.json")
    two_label = parse_labeled_path("two[nodes=1,gpus=2]=/tmp/two.json")
    window_label = parse_labeled_path("window[nodes=1,gpus=1]=/tmp/window.json")
    one = validate_harness_result(one_label, _self_test_fixture(), Path("/tmp/one.json"), "a")
    two = validate_harness_result(
        two_label,
        _self_test_fixture(scale=2),
        Path("/tmp/two.json"),
        "b",
    )
    window = validate_harness_result(
        window_label,
        _self_test_fixture(coalesce_tasks=2),
        Path("/tmp/window.json"),
        "c",
    )
    aggregate = build_report(
        [one, two, window],
        [
            {
                "label": "future",
                "source": "/tmp/future.json",
                "classification": "projection",
                "document": {"model": "test"},
            }
        ],
        generated_at="test",
    )
    measured = _as_mapping(aggregate["measured"], "self_test.measured")
    scaling = _as_list(measured["strong_scaling"], "self_test.strong_scaling")
    sweeps = _as_list(measured["task_window_sweeps"], "self_test.task_window_sweeps")
    comparisons = _as_list(measured["cpu_vs_gpu_speedups"], "self_test.cpu_vs_gpu_speedups")
    _self_check(bool(scaling), "self-test expected a strong-scaling group")
    _self_check(bool(sweeps), "self-test expected a task-window sweep")
    _self_check(bool(comparisons), "self-test expected CPU/GPU comparisons")
    scaling_group = _as_mapping(scaling[0], "self_test.scaling[0]")
    points = _as_list(scaling_group["points"], "self_test.scaling.points")
    last_point = _as_mapping(points[-1], "self_test.scaling.points[-1]")
    _self_check(
        math.isclose(float(last_point["speedup_vs_baseline"]), 2.0),
        "self-test strong-scaling speedup mismatch",
    )
    _self_check(
        int(last_point["per_rank_window_rows"])
        == int(_as_mapping(points[0], "self_test.point[0]")["per_rank_window_rows"]),
        "self-test strong scaling accepted different per-rank windows",
    )
    scaling_labels = {
        str(point["label"])
        for group in scaling
        for point in _as_list(_as_mapping(group, "self_test.scaling[]")["points"], "self_test.scaling.points")
    }
    _self_check("window" not in scaling_labels, "self-test strong scaling accepted a task-window mismatch")
    projection_section = _as_mapping(aggregate["projections"], "self_test.projections")
    _self_check(
        projection_section["excluded_from_measured_comparisons"] is True,
        "self-test projection separation mismatch",
    )

    cluster_label = parse_labeled_path("cluster[nodes=2,gpus=2,backend=gpu_lance_column_fetch_stage]=/tmp/rank-*.json")
    cluster_ranks = [
        validate_harness_result(
            cluster_label,
            _self_test_fixture(),
            Path(f"/tmp/rank-{index}.json"),
            f"rank-{index}",
        )
        for index in range(2)
    ]
    cluster = _combine_rank_results(cluster_label, cluster_ranks)
    cluster_measurement = cluster.measurements[0]
    _self_check(cluster_measurement.rank_count == 2, "self-test cluster rank count mismatch")
    _self_check(cluster_measurement.manifest_rows == 200, "self-test cluster image sum mismatch")
    _self_check(
        math.isclose(cluster_measurement.metric_stats()["images_per_second"].median, 20.0),
        "self-test cluster throughput mismatch",
    )
    _self_check(
        math.isclose(cluster_measurement.cold_setup_seconds_sum, 2.0),
        "self-test setup sum mismatch",
    )
    _self_check(
        math.isclose(cluster_measurement.metric_stats()["lance_read_iops"].median, 8.0),
        "self-test cluster I/O sum mismatch",
    )

    incomplete = _self_test_fixture()
    incomplete["status"] = "running"
    try:
        validate_harness_result(one_label, incomplete, Path("/tmp/bad.json"), "bad")
    except ReportInputError:
        pass
    else:
        message = "self-test accepted an incomplete harness"
        raise AssertionError(message)

    try:
        validate_harness_result(
            one_label,
            _self_test_fixture(correct=False),
            Path("/tmp/incorrect.json"),
            "incorrect",
        )
    except ReportInputError:
        pass
    else:
        message = "self-test accepted an incorrect harness"
        raise AssertionError(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Repeat LABEL=PATH; PATH may be one harness JSON, a glob, or comma-separated rank files",
    )
    parser.add_argument(
        "--projection",
        action="append",
        default=[],
        help="Repeat LABEL=PATH for separately reported projection JSON",
    )
    parser.add_argument("--output", type=Path, help="Atomic report destination")
    parser.add_argument("--format", choices=("auto", "json", "markdown"), default="auto")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="Used only to infer nodes when labels omit them")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _output_format(requested: str, output: Path) -> str:
    if requested != "auto":
        return requested
    if output.suffix.lower() == ".json":
        return "json"
    if output.suffix.lower() in {".md", ".markdown"}:
        return "markdown"
    message = "--format auto requires an output suffix of .json, .md, or .markdown"
    raise ReportInputError(message)


def _validate_unique_labels(specs: Sequence[LabeledPath]) -> None:
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        message = "--result/--projection labels must be unique"
        raise ReportInputError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_tests()
        print(json.dumps({"status": "ok", "self_tests": "passed"}))
        return 0
    if not args.result:
        parser.error("at least one --result LABEL=PATH is required")
    if args.output is None:
        parser.error("--output is required")
    if args.gpus_per_node <= 0:
        parser.error("--gpus-per-node must be positive")
    try:
        result_specs = [parse_labeled_path(raw) for raw in args.result]
        projection_specs = [parse_labeled_path(raw) for raw in args.projection]
        _validate_unique_labels((*result_specs, *projection_specs))
        accepted = [load_result(spec, gpus_per_node=args.gpus_per_node) for spec in result_specs]
        projections = [load_projection(spec) for spec in projection_specs]
        report = build_report(accepted, projections)
        output_format = _output_format(args.format, args.output)
    except ReportInputError as error:
        parser.error(str(error))
    content = (
        json.dumps(report, indent=2, sort_keys=True) + "\n" if output_format == "json" else render_markdown(report)
    )
    _atomic_write(args.output, content)
    print(f"wrote {output_format} report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
