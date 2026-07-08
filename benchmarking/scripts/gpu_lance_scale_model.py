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

# ruff: noqa: ANN401, EM101, EM102, PLR0913, PLR2004
"""Build a fail-closed queue-diagnostic scale model for GPU Lance fetches.

The model has two deliberately separate outputs:

* index capacity, which is a memory-sizing calculation; and
* throughput profiles, which are explicit reader/node assumptions anchored by a
  completed one-reader queue benchmark.

The queue benchmark is not a storage-saturation measurement. All projected
runtimes retain the ``queue_diagnostic_extrapolation`` classification until a
matched, repeated node-saturation run replaces this evidence source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEGACY_INPUT_SCHEMA_VERSION = 3
INPUT_SCHEMA_VERSION = 4
MODEL_SCHEMA_VERSION = 3
MODEL_INPUT_KIND = "gpu_lance_queue_diagnostic"
MODEL_NAME = "gpu_lance_column_fetch_scale_model"
BENCHMARK_ARM = "gpu_lance_column_fetch_stage"
DEFAULT_H100_MEMORY_BYTES = 80 * 1024**3
_PRIMARY_SATURATION_EVIDENCE_CLASS = "primary_saturation"
_PRIMARY_SATURATION_WAVES = frozenset({4, 8})
_TERMINAL_ELIGIBILITY_SCHEMA_VERSION = 2
TARGET_IMAGE_REFERENCES = (
    ("6B", 6_000_000_000),
    ("20B", 20_000_000_000),
    ("100B+", 100_000_000_000),
)


class ModelInputError(ValueError):
    """Raised when benchmark evidence or model input is incomplete or invalid."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelInputError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ModelInputError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ModelInputError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelInputError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ModelInputError(f"{name} must be > 0")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ModelInputError(f"{name} must be >= 0")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _positive(value, name)
    if not result.is_integer():
        raise ModelInputError(f"{name} must be an integer")
    return int(result)


def _nonnegative_int(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if not result.is_integer():
        raise ModelInputError(f"{name} must be an integer")
    return int(result)


def _required(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ModelInputError(f"{name}.{key} is required")
    return mapping[key]


def _required_text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = _required(mapping, key, name)
    if not isinstance(value, str) or not value:
        raise ModelInputError(f"{name}.{key} must be a non-empty string")
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ModelInputError(f"{name} must be {expected!r}, got {actual!r}")


def _require_close(actual: Any, expected: Any, name: str) -> None:
    left = _finite(actual, name)
    right = _finite(expected, f"expected {name}")
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        raise ModelInputError(f"{name} is inconsistent: {left!r} != {right!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ModelInputError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ModelInputError(f"{name} must be a SHA-256 hex digest") from error
    return value


def _terminal_eligibility_identity(  # noqa: C901, PLR0915
    source_path: str, source_sha256: str
) -> dict[str, Any] | None:
    path = Path(source_path)
    if not path.is_file():
        return None
    parent = path.parent
    required_paths = {
        "run_identity": parent / "run_identity.json",
        "telemetry_validation": parent / "telemetry_validation.json",
        "eligibility": parent / "eligibility.json",
    }
    indicators = tuple(required_paths.values())
    if not any(candidate.exists() for candidate in indicators):
        return None
    missing = [candidate.name for candidate in required_paths.values() if not candidate.is_file()]
    if missing:
        raise ModelInputError(f"saturation artifact family is missing required files: {missing}")
    eligibility_path = required_paths["eligibility"]
    try:
        eligibility = _mapping(json.loads(eligibility_path.read_text(encoding="utf-8")), "terminal eligibility")
    except (OSError, json.JSONDecodeError) as error:
        raise ModelInputError(f"terminal eligibility is unreadable: {error}") from error
    eligibility_schema_version = eligibility.get("schema_version")
    if isinstance(eligibility_schema_version, bool) or eligibility_schema_version not in {
        1,
        _TERMINAL_ELIGIBILITY_SCHEMA_VERSION,
    }:
        raise ModelInputError(
            f"terminal eligibility.schema_version must be 1 or 2, got {eligibility_schema_version!r}"
        )
    _require_equal(
        eligibility.get("artifact_kind"),
        "gpu_lance_saturation_terminal_eligibility",
        "terminal eligibility.artifact_kind",
    )
    _require_equal(eligibility.get("terminal"), True, "terminal eligibility.terminal")
    _require_equal(eligibility.get("status"), "eligible", "terminal eligibility.status")
    _require_equal(_required(eligibility, "failures", "terminal eligibility"), [], "terminal eligibility.failures")
    evidence_class = eligibility.get("evidence_class")
    benchmark_validation = _mapping(
        _required(eligibility, "benchmark_validation", "terminal eligibility"),
        "terminal eligibility.benchmark_validation",
    )
    validation_waves = benchmark_validation.get("waves")
    if eligibility_schema_version == 1 and evidence_class is None and validation_waves in _PRIMARY_SATURATION_WAVES:
        evidence_class = _PRIMARY_SATURATION_EVIDENCE_CLASS
    if evidence_class != _PRIMARY_SATURATION_EVIDENCE_CLASS or validation_waves not in _PRIMARY_SATURATION_WAVES:
        raise ModelInputError(
            f"terminal eligibility evidence_class={evidence_class!r}, "
            f"benchmark_validation.waves={validation_waves!r}; expected primary_saturation with 4 or 8 waves"
        )
    _require_equal(benchmark_validation.get("status"), "passed", "benchmark_validation.status")
    _require_equal(benchmark_validation.get("failures"), [], "benchmark_validation.failures")
    validation_evidence_class = benchmark_validation.get("evidence_class")
    if eligibility_schema_version == 1 and validation_evidence_class is None:
        validation_evidence_class = evidence_class
    _require_equal(validation_evidence_class, evidence_class, "benchmark_validation.evidence_class")
    identity_validation = _mapping(
        _required(eligibility, "identity_validation", "terminal eligibility"),
        "terminal eligibility.identity_validation",
    )
    _require_equal(identity_validation.get("status"), "passed", "identity_validation.status")
    _require_equal(identity_validation.get("failures"), [], "identity_validation.failures")
    _require_equal(
        eligibility.get("telemetry_validation_status"),
        "passed",
        "terminal eligibility.telemetry_validation_status",
    )
    policy = _mapping(_required(eligibility, "policy", "terminal eligibility"), "terminal eligibility.policy")
    for name in (
        "requires_benchmark_validation",
        "requires_run_identity_validation",
        "requires_telemetry_validation",
        "telemetry_pass_is_not_benchmark_eligibility",
    ):
        _require_equal(policy.get(name), True, f"terminal eligibility.policy.{name}")
    _require_equal(policy.get("minimum_repeat_count"), 2, "terminal eligibility.policy.minimum_repeat_count")
    if eligibility_schema_version == _TERMINAL_ELIGIBILITY_SCHEMA_VERSION:
        _require_equal(policy.get("evidence_class"), evidence_class, "terminal eligibility.policy.evidence_class")
        _require_equal(
            policy.get("primary_saturation_waves"),
            [4, 8],
            "terminal eligibility.policy.primary_saturation_waves",
        )
        _require_equal(
            policy.get("locality_sensitivity_waves"),
            [1, 2],
            "terminal eligibility.policy.locality_sensitivity_waves",
        )
    artifacts = _mapping(_required(eligibility, "artifacts", "terminal eligibility"), "terminal eligibility.artifacts")
    expected_artifacts = {
        "benchmark": (path, source_sha256),
        "run_identity": (required_paths["run_identity"], _sha256_file(required_paths["run_identity"])),
        "telemetry_validation": (
            required_paths["telemetry_validation"],
            _sha256_file(required_paths["telemetry_validation"]),
        ),
    }
    for name, (artifact_path, expected_sha256) in expected_artifacts.items():
        artifact = _mapping(
            _required(artifacts, name, "terminal eligibility.artifacts"),
            f"terminal eligibility.artifacts.{name}",
        )
        _require_equal(artifact.get("path"), artifact_path.name, f"eligibility {name}.path")
        _require_equal(
            _validate_sha256(artifact.get("sha256"), f"eligibility {name}.sha256"),
            expected_sha256,
            f"eligibility {name}.sha256",
        )
    try:
        telemetry = _mapping(
            json.loads(required_paths["telemetry_validation"].read_text(encoding="utf-8")),
            "telemetry validation",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ModelInputError(f"telemetry validation is unreadable: {error}") from error
    _require_equal(telemetry.get("status"), "passed", "telemetry validation.status")
    return {
        "path": eligibility_path.name,
        "sha256": _sha256_file(eligibility_path),
        "status": "eligible",
        "evidence_class": _PRIMARY_SATURATION_EVIDENCE_CLASS,
    }


@dataclass(frozen=True)
class ObservedRange:
    """Minimum, median, and maximum across completed correct repeats."""

    low: float
    observed: float
    high: float

    def __post_init__(self) -> None:
        low = _positive(self.low, "range.low")
        observed = _positive(self.observed, "range.observed")
        high = _positive(self.high, "range.high")
        if not low <= observed <= high:
            raise ModelInputError("range must satisfy 0 < low <= observed <= high")

    @classmethod
    def from_values(cls, values: Sequence[float]) -> ObservedRange:
        checked = [_positive(value, "observed value") for value in values]
        if len(checked) < 2:
            raise ModelInputError("repeat-derived ranges require at least two values")
        return cls(min(checked), statistics.median(checked), max(checked))

    def scaled(self, factor: float) -> ObservedRange:
        scale = _positive(factor, "range scale")
        return ObservedRange(self.low * scale, self.observed * scale, self.high * scale)

    def divided_by(self, denominator: ObservedRange) -> ObservedRange:
        return ObservedRange(
            self.low / denominator.high,
            self.observed / denominator.observed,
            self.high / denominator.low,
        )

    def as_dict(self, classification: str, *, repeat_count: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "low": self.low,
            "observed": self.observed,
            "high": self.high,
            "classification": classification,
        }
        if repeat_count is not None:
            result["repeat_count"] = repeat_count
        return result


@dataclass(frozen=True)
class ModelConfig:
    """Explicit assumptions for memory capacity and throughput profiles."""

    h100_memory_bytes: int = DEFAULT_H100_MEMORY_BYTES
    usable_gpu_memory_fraction: float = 0.80
    gpus_per_node: int = 8
    readers_per_node: int = 8
    throughput_node_counts: tuple[int, ...] = (1, 2, 4, 8)
    marginal_reader_scaling_efficiency: float = 0.80

    def __post_init__(self) -> None:
        _positive_int(self.h100_memory_bytes, "h100_memory_bytes")
        _positive_int(self.gpus_per_node, "gpus_per_node")
        _positive_int(self.readers_per_node, "readers_per_node")
        if not self.throughput_node_counts:
            raise ModelInputError("throughput_node_counts cannot be empty")
        for value in self.throughput_node_counts:
            _positive_int(value, "throughput_node_count")
        fraction = _positive(self.usable_gpu_memory_fraction, "usable_gpu_memory_fraction")
        efficiency = _positive(
            self.marginal_reader_scaling_efficiency,
            "marginal_reader_scaling_efficiency",
        )
        if fraction > 1 or efficiency > 1:
            raise ModelInputError("memory fraction and scaling efficiency must be <= 1")


def _validate_correctness(
    correctness: Mapping[str, Any],
    *,
    rows: int,
    name: str,
) -> tuple[str, str, int]:
    _require_equal(_required(correctness, "correct", name), True, f"{name}.correct")
    _require_equal(
        _required(correctness, "order_matches_manifest", name),
        True,
        f"{name}.order_matches_manifest",
    )
    for field in ("row_count", "expected_row_count", "present_rows"):
        _require_equal(_positive_int(_required(correctness, field, name), f"{name}.{field}"), rows, f"{name}.{field}")
    _require_equal(
        _nonnegative_int(_required(correctness, "missing_payload_rows", name), f"{name}.missing_payload_rows"),
        0,
        f"{name}.missing_payload_rows",
    )
    payload_digest = _validate_sha256(
        _required(correctness, "payload_digest_sha256", name),
        f"{name}.payload_digest_sha256",
    )
    output_digest = _validate_sha256(
        _required(correctness, "output_digest_sha256", name),
        f"{name}.output_digest_sha256",
    )
    payload_bytes = _positive_int(
        _required(correctness, "payload_bytes", name),
        f"{name}.payload_bytes",
    )
    return payload_digest, output_digest, payload_bytes


def _raw_repeat(repeat: Mapping[str, Any], *, rows: int, ordinal: int) -> dict[str, Any]:
    name = f"repeats[{ordinal}]"
    _require_equal(_required_text(repeat, "status", name), "completed", f"{name}.status")
    _require_equal(_nonnegative_int(_required(repeat, "repeat", name), f"{name}.repeat"), ordinal, f"{name}.repeat")
    correctness = _mapping(_required(repeat, "correctness", name), f"{name}.correctness")
    payload_digest, output_digest, correctness_payload_bytes = _validate_correctness(
        correctness,
        rows=rows,
        name=f"{name}.correctness",
    )
    logical_payload_bytes = _positive_int(
        _required(repeat, "payload_bytes", name),
        f"{name}.payload_bytes",
    )
    _require_equal(logical_payload_bytes, correctness_payload_bytes, f"{name}.payload_bytes")

    backend = _mapping(_required(repeat, "backend_metrics", name), f"{name}.backend_metrics")
    logical_requests = _positive_int(
        _required(backend, "logical_payload_requests", f"{name}.backend_metrics"),
        f"{name}.backend_metrics.logical_payload_requests",
    )
    _require_equal(logical_requests, rows, f"{name}.backend_metrics.logical_payload_requests")
    for field in ("found_unique_keys", "unique_payloads", "requested_unique_keys"):
        _require_equal(
            _positive_int(_required(backend, field, f"{name}.backend_metrics"), f"{name}.backend_metrics.{field}"),
            rows,
            f"{name}.backend_metrics.{field}",
        )
    for field in ("missing_unique_keys", "logical_duplicate_requests"):
        _require_equal(
            _nonnegative_int(_required(backend, field, f"{name}.backend_metrics"), f"{name}.backend_metrics.{field}"),
            0,
            f"{name}.backend_metrics.{field}",
        )

    warm_seconds = _positive(_required(repeat, "warm_process_seconds", name), f"{name}.warm_process_seconds")
    fetch_seconds = _positive(_required(repeat, "fetch_seconds", name), f"{name}.fetch_seconds")
    lookup_seconds = _positive(_required(repeat, "lookup_seconds", name), f"{name}.lookup_seconds")
    if fetch_seconds > warm_seconds:
        raise ModelInputError(f"{name}.fetch_seconds exceeds warm_process_seconds")
    physical_read_calls = _positive_int(
        _required(repeat, "lance_read_iops", name),
        f"{name}.lance_read_iops",
    )
    physical_read_bytes = _positive_int(
        _required(repeat, "lance_read_bytes", name),
        f"{name}.lance_read_bytes",
    )
    lance_projected_bytes = _positive_int(
        _required(backend, "lance_fetched_bytes", f"{name}.backend_metrics"),
        f"{name}.backend_metrics.lance_fetched_bytes",
    )
    _require_close(
        _required(backend, "lance_fetch_seconds", f"{name}.backend_metrics"),
        fetch_seconds,
        f"{name}.backend_metrics.lance_fetch_seconds",
    )
    _require_close(
        _required(backend, "lance_lookup_seconds", f"{name}.backend_metrics"),
        lookup_seconds,
        f"{name}.backend_metrics.lance_lookup_seconds",
    )
    _require_close(
        _required(backend, "physical_reads_per_payload", f"{name}.backend_metrics"),
        physical_read_calls / logical_requests,
        f"{name}.backend_metrics.physical_reads_per_payload",
    )
    _require_close(
        _required(backend, "average_physical_read_bytes", f"{name}.backend_metrics"),
        physical_read_bytes / physical_read_calls,
        f"{name}.backend_metrics.average_physical_read_bytes",
    )
    _require_close(
        _required(backend, "read_amplification", f"{name}.backend_metrics"),
        physical_read_bytes / lance_projected_bytes,
        f"{name}.backend_metrics.read_amplification",
    )
    _require_close(
        _required(repeat, "images_per_second", name),
        rows / warm_seconds,
        f"{name}.images_per_second",
    )
    return {
        "repeat": ordinal,
        "status": "completed",
        "correct": True,
        "payload_digest_sha256": payload_digest,
        "output_digest_sha256": output_digest,
        "logical_payload_requests": logical_requests,
        "warm_process_seconds": warm_seconds,
        "fetch_seconds": fetch_seconds,
        "lookup_seconds": lookup_seconds,
        "logical_payload_bytes": logical_payload_bytes,
        "lance_projected_bytes": lance_projected_bytes,
        "physical_read_bytes": physical_read_bytes,
        "physical_read_calls": physical_read_calls,
        "peak_rss_bytes": _positive_int(
            _required(backend, "peak_rss_bytes", f"{name}.backend_metrics"),
            f"{name}.backend_metrics.peak_rss_bytes",
        ),
    }


def queue_model_input_from_benchmark(  # noqa: C901, PLR0912, PLR0915
    report: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    reference_keys: int,
    sidecar_bytes: int,
    reference_manifest_uri: str,
    reference_manifest_sha256: str,
    current_mint_image_references: int,
    current_mint_unique_reference_keys: int,
    measurement_h100_count: int = 1,
    measurement_physical_node_count: int = 1,
    active_payload_readers: int = 1,
    compact_shuffle_bytes_per_probe_per_pass: int = 12,
    compact_shuffle_coordinate_passes: int = 2,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create v3 input while rejecting partial, unidentified, or incorrect evidence."""

    _require_equal(_required_text(report, "status", "benchmark"), "completed", "benchmark.status")
    if report.get("teardown_errors"):
        raise ModelInputError("benchmark has teardown_errors")
    config = _mapping(_required(report, "configuration", "benchmark"), "benchmark.configuration")
    configured_repeats = _positive_int(
        _required(config, "repeat_count", "benchmark.configuration"),
        "benchmark.configuration.repeat_count",
    )
    if configured_repeats < 2:
        raise ModelInputError("benchmark requires at least two configured repeats")
    manifest = _mapping(_required(report, "manifest", "benchmark"), "benchmark.manifest")
    rows = _positive_int(_required(manifest, "rows", "benchmark.manifest"), "benchmark.manifest.rows")
    manifest_digest = _validate_sha256(
        _required(manifest, "digest_sha256", "benchmark.manifest"),
        "benchmark.manifest.digest_sha256",
    )
    arms = _mapping(_required(report, "arms", "benchmark"), "benchmark.arms")
    arm = _mapping(_required(arms, BENCHMARK_ARM, "benchmark.arms"), f"benchmark.arms.{BENCHMARK_ARM}")
    _require_equal(_required_text(arm, "status", BENCHMARK_ARM), "completed", f"{BENCHMARK_ARM}.status")
    summary = _mapping(_required(arm, "summary", BENCHMARK_ARM), f"{BENCHMARK_ARM}.summary")
    _require_equal(
        _required(summary, "stable_correctness_digest", f"{BENCHMARK_ARM}.summary"),
        True,
        f"{BENCHMARK_ARM}.summary.stable_correctness_digest",
    )
    repeats = _sequence(_required(arm, "repeats", BENCHMARK_ARM), f"{BENCHMARK_ARM}.repeats")
    if len(repeats) != configured_repeats:
        raise ModelInputError(f"benchmark has {len(repeats)} repeats but configuration requires {configured_repeats}")
    raw_repeats = [
        _raw_repeat(_mapping(repeat, f"repeats[{ordinal}]"), rows=rows, ordinal=ordinal)
        for ordinal, repeat in enumerate(repeats)
    ]
    if len({item["payload_digest_sha256"] for item in raw_repeats}) != 1:
        raise ModelInputError("payload digest differs across repeats")
    if len({item["output_digest_sha256"] for item in raw_repeats}) != 1:
        raise ModelInputError("output digest differs across repeats")
    if len({item["logical_payload_bytes"] for item in raw_repeats}) != 1:
        raise ModelInputError("logical payload bytes differ for the same manifest")

    for ordinal, warmup_value in enumerate(arm.get("warmups", [])):
        warmup = _mapping(warmup_value, f"warmups[{ordinal}]")
        _require_equal(
            _required_text(warmup, "status", f"warmups[{ordinal}]"), "completed", f"warmups[{ordinal}].status"
        )
        _validate_correctness(
            _mapping(_required(warmup, "correctness", f"warmups[{ordinal}]"), f"warmups[{ordinal}].correctness"),
            rows=rows,
            name=f"warmups[{ordinal}].correctness",
        )

    reference_keys = _positive_int(reference_keys, "reference_keys")
    sidecar_bytes = _positive_int(sidecar_bytes, "sidecar_bytes")
    current_mint_image_references = _positive_int(
        current_mint_image_references,
        "current_mint_image_references",
    )
    current_mint_unique_reference_keys = _positive_int(
        current_mint_unique_reference_keys,
        "current_mint_unique_reference_keys",
    )
    cold_setup = _mapping(_required(arm, "cold_setup", BENCHMARK_ARM), f"{BENCHMARK_ARM}.cold_setup")
    cold_backend = _mapping(
        _required(cold_setup, "backend_metrics", f"{BENCHMARK_ARM}.cold_setup"),
        f"{BENCHMARK_ARM}.cold_setup.backend_metrics",
    )
    _require_equal(
        _positive_int(
            _required(cold_backend, "gpu_reference_rows", "cold_setup.backend_metrics"),
            "cold_setup.backend_metrics.gpu_reference_rows",
        ),
        reference_keys,
        "cold_setup.backend_metrics.gpu_reference_rows",
    )
    dataset = _mapping(_required(report, "dataset", "benchmark"), "benchmark.dataset")
    source_columns = _mapping(
        _required(dataset, "source_columns", "benchmark.dataset"),
        "benchmark.dataset.source_columns",
    )
    if set(source_columns) != {"image"}:
        raise ModelInputError("queue diagnostic must use the image-only timed projection")
    if not all(
        isinstance(key, str) and key and isinstance(value, str) and value for key, value in source_columns.items()
    ):
        raise ModelInputError("benchmark.dataset.source_columns must be a string mapping")

    reference_manifest_uri = str(reference_manifest_uri)
    if not reference_manifest_uri:
        raise ModelInputError("reference_manifest_uri must be a non-empty immutable identity")
    reference_manifest_sha256 = _validate_sha256(reference_manifest_sha256, "reference_manifest_sha256")
    recorded_reference_uri = config.get("reference_manifest_uri")
    recorded_reference_sha256 = config.get("reference_manifest_sha256")
    if recorded_reference_uri is not None:
        _require_equal(
            recorded_reference_uri, reference_manifest_uri, "benchmark.configuration.reference_manifest_uri"
        )
    if recorded_reference_sha256 is not None:
        _require_equal(
            _validate_sha256(recorded_reference_sha256, "benchmark.configuration.reference_manifest_sha256"),
            reference_manifest_sha256,
            "benchmark.configuration.reference_manifest_sha256",
        )
    reference_files = _sequence(
        _required(config, "reference_files", "benchmark.configuration"),
        "benchmark.configuration.reference_files",
    )
    if not reference_files or not all(isinstance(path, str) and path for path in reference_files):
        raise ModelInputError("benchmark.configuration.reference_files must identify the measured sidecar files")
    reference_file_inventory_sha256 = hashlib.sha256(
        json.dumps(sorted(reference_files), separators=(",", ":")).encode()
    ).hexdigest()

    policy_keys = (
        "task_rows",
        "coalesce_tasks",
        "rows_per_coalesced_fetch",
        "lookup_batch_size",
        "fetch_batch_size",
        "max_lookup_bytes",
        "max_pending_fetch_batches",
        "payload_read_mode",
        "take_scan_batch_readahead",
        "validate_payload_keys",
        "io_threads",
        "index_mirror",
        "copy_index_to_node_local",
        "warmup_count",
        "repeat_count",
    )
    queue_configuration = {key: _required(config, key, "benchmark.configuration") for key in policy_keys}
    if queue_configuration["payload_read_mode"] != "sparse":
        raise ModelInputError("queue diagnostic requires payload_read_mode='sparse'")
    if queue_configuration["validate_payload_keys"] is not False:
        raise ModelInputError("image-only queue diagnostic requires validate_payload_keys=false")
    if not isinstance(queue_configuration["copy_index_to_node_local"], bool):
        raise ModelInputError("benchmark.configuration.copy_index_to_node_local must be boolean")
    if queue_configuration["index_mirror"] is not None and "index_mirror_contract" not in config:
        raise ModelInputError("configured index mirror requires index_mirror_contract identity")
    queue_configuration["index_mirror_contract"] = config.get("index_mirror_contract")
    queue_configuration["ray_concurrency"] = config.get("ray_concurrency")
    queue_configuration["dataset_storage_option_keys"] = _required(
        dataset,
        "storage_option_keys",
        "benchmark.dataset",
    )
    queue_configuration["reference_storage_option_keys"] = _required(
        config,
        "reference_storage_option_keys",
        "benchmark.configuration",
    )

    environment = _mapping(_required(report, "environment", "benchmark"), "benchmark.environment")
    packages = _mapping(_required(environment, "packages", "benchmark.environment"), "benchmark.environment.packages")
    for package in ("nemo-curator", "pyarrow", "pylance"):
        _required_text(packages, package, "benchmark.environment.packages")
    runtime_identity = {
        "python": _required_text(environment, "python", "benchmark.environment"),
        "platform": _required_text(environment, "platform", "benchmark.environment"),
        "packages": dict(packages),
        "code": {
            key: environment[key]
            for key in ("git_commit", "git_sha", "source_commit", "code_commit")
            if key in environment
        },
    }
    validated_source_sha256 = _validate_sha256(source_sha256, "source_sha256")
    terminal_eligibility = _terminal_eligibility_identity(source_path, validated_source_sha256)

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "input_kind": MODEL_INPUT_KIND,
        "generated_at": generated_at or datetime.now(tz=UTC).isoformat(),
        "source": {
            "artifact_path": source_path,
            "artifact_sha256": validated_source_sha256,
            "artifact_status": "completed",
            "arm": BENCHMARK_ARM,
            "arm_status": "completed",
            "repeat_count": configured_repeats,
            "manifest_digest_sha256": manifest_digest,
            "manifest_rows": rows,
            "payload_digest_sha256": raw_repeats[0]["payload_digest_sha256"],
            "output_digest_sha256": raw_repeats[0]["output_digest_sha256"],
            "dataset_uri_sha256": hashlib.sha256(
                _required_text(dataset, "uri", "benchmark.dataset").encode("utf-8")
            ).hexdigest(),
            "dataset_version": _positive_int(
                _required(dataset, "version", "benchmark.dataset"),
                "benchmark.dataset.version",
            ),
            "projection": source_columns,
            "sidecar_identity": {
                "manifest_uri": reference_manifest_uri,
                "manifest_sha256": reference_manifest_sha256,
                "reference_file_count": len(reference_files),
                "reference_file_inventory_sha256": reference_file_inventory_sha256,
                "identity_source": (
                    "benchmark_and_caller_pin"
                    if recorded_reference_uri is not None and recorded_reference_sha256 is not None
                    else "caller_pin_cross_check_unavailable_in_legacy_benchmark"
                ),
            },
            "queue_configuration": queue_configuration,
            "runtime_identity": runtime_identity,
            "terminal_eligibility": terminal_eligibility,
        },
        "measurement_resources": {
            "h100_count": _positive_int(measurement_h100_count, "measurement_h100_count"),
            "physical_node_count": _positive_int(
                measurement_physical_node_count,
                "measurement_physical_node_count",
            ),
            "active_payload_readers": _positive_int(active_payload_readers, "active_payload_readers"),
            "node_saturation_status": "not_measured",
            "classification": "measured_geometry_not_node_saturation",
        },
        "reference_inventory": {
            "keys": reference_keys,
            "sidecar_bytes": sidecar_bytes,
            "gpu_index_bytes": _positive_int(
                _required(cold_backend, "gpu_reference_bytes", "cold_setup.backend_metrics"),
                "cold_setup.backend_metrics.gpu_reference_bytes",
            ),
            "cold_setup_seconds": _positive(
                _required(cold_setup, "wall_seconds", "cold_setup"),
                "cold_setup.wall_seconds",
            ),
            "classification": "measured_inventory_and_single_setup_observation",
        },
        "current_mint": {
            "image_references": {
                "value": current_mint_image_references,
                "classification": "measured_inventory",
            },
            "unique_reference_keys": {
                "value": current_mint_unique_reference_keys,
                "classification": "measured_inventory",
            },
            "payload_reads": {
                "value": current_mint_image_references,
                "classification": "modeled_no_cross_window_reuse_upper_bound",
                "policy": "one_payload_fetch_per_image_reference",
            },
        },
        "shuffle_assumption": {
            "bytes_per_probe_per_coordinate_pass": _positive_int(
                compact_shuffle_bytes_per_probe_per_pass,
                "compact_shuffle_bytes_per_probe_per_pass",
            ),
            "coordinate_passes": _positive_int(
                compact_shuffle_coordinate_passes,
                "compact_shuffle_coordinate_passes",
            ),
            "classification": "modeled_schema_floor_excluding_transport_overhead",
        },
        "raw_repeats": raw_repeats,
    }


def _validate_model_input(data: Mapping[str, Any]) -> None:  # noqa: C901, PLR0912, PLR0915
    input_schema_version = _positive_int(_required(data, "schema_version", "input"), "input.schema_version")
    if input_schema_version not in {LEGACY_INPUT_SCHEMA_VERSION, INPUT_SCHEMA_VERSION}:
        raise ModelInputError(
            f"input.schema_version must be {LEGACY_INPUT_SCHEMA_VERSION} or {INPUT_SCHEMA_VERSION}, "
            f"got {input_schema_version!r}"
        )
    _require_equal(_required_text(data, "input_kind", "input"), MODEL_INPUT_KIND, "input.input_kind")
    source = _mapping(_required(data, "source", "input"), "input.source")
    _require_equal(
        _required_text(source, "artifact_status", "input.source"), "completed", "input.source.artifact_status"
    )
    _require_equal(_required_text(source, "arm_status", "input.source"), "completed", "input.source.arm_status")
    _require_equal(_required_text(source, "arm", "input.source"), BENCHMARK_ARM, "input.source.arm")
    _validate_sha256(_required(source, "artifact_sha256", "input.source"), "input.source.artifact_sha256")
    manifest_rows = _positive_int(_required(source, "manifest_rows", "input.source"), "input.source.manifest_rows")
    repeat_count = _positive_int(_required(source, "repeat_count", "input.source"), "input.source.repeat_count")
    if repeat_count < 2:
        raise ModelInputError("input requires at least two repeats")
    projection = _mapping(_required(source, "projection", "input.source"), "input.source.projection")
    _require_equal(dict(projection), {"image": "image"}, "input.source.projection")
    sidecar_identity = _mapping(
        _required(source, "sidecar_identity", "input.source"),
        "input.source.sidecar_identity",
    )
    _required_text(sidecar_identity, "manifest_uri", "input.source.sidecar_identity")
    _validate_sha256(
        _required(sidecar_identity, "manifest_sha256", "input.source.sidecar_identity"),
        "input.source.sidecar_identity.manifest_sha256",
    )
    _positive_int(
        _required(sidecar_identity, "reference_file_count", "input.source.sidecar_identity"),
        "input.source.sidecar_identity.reference_file_count",
    )
    _validate_sha256(
        _required(sidecar_identity, "reference_file_inventory_sha256", "input.source.sidecar_identity"),
        "input.source.sidecar_identity.reference_file_inventory_sha256",
    )
    _required_text(sidecar_identity, "identity_source", "input.source.sidecar_identity")

    queue_configuration = _mapping(
        _required(source, "queue_configuration", "input.source"),
        "input.source.queue_configuration",
    )
    for field in (
        "task_rows",
        "coalesce_tasks",
        "rows_per_coalesced_fetch",
        "lookup_batch_size",
        "fetch_batch_size",
        "max_lookup_bytes",
        "max_pending_fetch_batches",
        "take_scan_batch_readahead",
        "io_threads",
        "repeat_count",
    ):
        _positive_int(_required(queue_configuration, field, "input.source.queue_configuration"), f"queue.{field}")
    _require_equal(
        queue_configuration["rows_per_coalesced_fetch"],
        queue_configuration["task_rows"] * queue_configuration["coalesce_tasks"],
        "input.source.queue_configuration.rows_per_coalesced_fetch",
    )
    _require_equal(
        _required_text(queue_configuration, "payload_read_mode", "input.source.queue_configuration"),
        "sparse",
        "input.source.queue_configuration.payload_read_mode",
    )
    _require_equal(
        _required(queue_configuration, "validate_payload_keys", "input.source.queue_configuration"),
        False,
        "input.source.queue_configuration.validate_payload_keys",
    )
    if not isinstance(
        _required(queue_configuration, "copy_index_to_node_local", "input.source.queue_configuration"),
        bool,
    ):
        raise ModelInputError("input.source.queue_configuration.copy_index_to_node_local must be boolean")
    for field in (
        "index_mirror",
        "index_mirror_contract",
        "ray_concurrency",
        "dataset_storage_option_keys",
        "reference_storage_option_keys",
    ):
        _required(queue_configuration, field, "input.source.queue_configuration")
    _nonnegative_int(
        _required(queue_configuration, "warmup_count", "input.source.queue_configuration"),
        "input.source.queue_configuration.warmup_count",
    )
    for field in ("dataset_storage_option_keys", "reference_storage_option_keys"):
        values = _sequence(queue_configuration[field], f"input.source.queue_configuration.{field}")
        if not all(isinstance(value, str) and value for value in values):
            raise ModelInputError(f"input.source.queue_configuration.{field} must contain strings")
    _require_equal(
        queue_configuration["repeat_count"],
        repeat_count,
        "input.source.queue_configuration.repeat_count",
    )

    runtime_identity = _mapping(
        _required(source, "runtime_identity", "input.source"),
        "input.source.runtime_identity",
    )
    _required_text(runtime_identity, "python", "input.source.runtime_identity")
    _required_text(runtime_identity, "platform", "input.source.runtime_identity")
    packages = _mapping(
        _required(runtime_identity, "packages", "input.source.runtime_identity"),
        "input.source.runtime_identity.packages",
    )
    for package in ("nemo-curator", "pyarrow", "pylance"):
        _required_text(packages, package, "input.source.runtime_identity.packages")
    _mapping(
        _required(runtime_identity, "code", "input.source.runtime_identity"), "input.source.runtime_identity.code"
    )
    terminal_eligibility = _required(source, "terminal_eligibility", "input.source")
    if terminal_eligibility is not None:
        terminal = _mapping(terminal_eligibility, "input.source.terminal_eligibility")
        _require_equal(
            _required_text(terminal, "status", "input.source.terminal_eligibility"),
            "eligible",
            "input.source.terminal_eligibility.status",
        )
        embedded_evidence_class = terminal.get("evidence_class")
        if embedded_evidence_class is None and input_schema_version == LEGACY_INPUT_SCHEMA_VERSION:
            embedded_evidence_class = _PRIMARY_SATURATION_EVIDENCE_CLASS
        _require_equal(
            embedded_evidence_class,
            _PRIMARY_SATURATION_EVIDENCE_CLASS,
            "input.source.terminal_eligibility.evidence_class",
        )
        _validate_sha256(
            _required(terminal, "sha256", "input.source.terminal_eligibility"),
            "input.source.terminal_eligibility.sha256",
        )
        _required_text(terminal, "path", "input.source.terminal_eligibility")
    resources = _mapping(_required(data, "measurement_resources", "input"), "input.measurement_resources")
    _require_equal(
        _required_text(resources, "node_saturation_status", "input.measurement_resources"),
        "not_measured",
        "input.measurement_resources.node_saturation_status",
    )
    h100_count = _positive_int(
        _required(resources, "h100_count", "input.measurement_resources"),
        "input.measurement_resources.h100_count",
    )
    _positive_int(
        _required(resources, "physical_node_count", "input.measurement_resources"),
        "input.measurement_resources.physical_node_count",
    )
    active_readers = _positive_int(
        _required(resources, "active_payload_readers", "input.measurement_resources"),
        "input.measurement_resources.active_payload_readers",
    )
    if active_readers > h100_count:
        raise ModelInputError("active_payload_readers cannot exceed measured H100 count")
    inventory = _mapping(_required(data, "reference_inventory", "input"), "input.reference_inventory")
    for field in ("keys", "sidecar_bytes", "gpu_index_bytes"):
        _positive_int(_required(inventory, field, "input.reference_inventory"), f"input.reference_inventory.{field}")
    _positive(
        _required(inventory, "cold_setup_seconds", "input.reference_inventory"),
        "input.reference_inventory.cold_setup_seconds",
    )
    current = _mapping(_required(data, "current_mint", "input"), "input.current_mint")
    for field in ("image_references", "unique_reference_keys", "payload_reads"):
        item = _mapping(_required(current, field, "input.current_mint"), f"input.current_mint.{field}")
        _positive_int(_required(item, "value", f"input.current_mint.{field}"), f"input.current_mint.{field}.value")
        _required_text(item, "classification", f"input.current_mint.{field}")
    _require_equal(
        current["payload_reads"]["classification"],
        "modeled_no_cross_window_reuse_upper_bound",
        "input.current_mint.payload_reads.classification",
    )
    _require_equal(
        current["payload_reads"]["policy"],
        "one_payload_fetch_per_image_reference",
        "input.current_mint.payload_reads.policy",
    )
    _require_equal(
        current["payload_reads"]["value"],
        current["image_references"]["value"],
        "input.current_mint.payload_reads.value",
    )
    repeats = _sequence(_required(data, "raw_repeats", "input"), "input.raw_repeats")
    if len(repeats) != repeat_count:
        raise ModelInputError("input raw repeat count does not match source.repeat_count")
    payload_digests: set[str] = set()
    output_digests: set[str] = set()
    logical_payload_bytes: set[int] = set()
    lance_projected_bytes: set[int] = set()
    for ordinal, repeat_value in enumerate(repeats):
        repeat = _mapping(repeat_value, f"input.raw_repeats[{ordinal}]")
        _require_equal(
            _required_text(repeat, "status", f"raw_repeats[{ordinal}]"), "completed", f"raw_repeats[{ordinal}].status"
        )
        _require_equal(
            _required(repeat, "correct", f"raw_repeats[{ordinal}]"), True, f"raw_repeats[{ordinal}].correct"
        )
        _require_equal(
            _nonnegative_int(_required(repeat, "repeat", f"raw_repeats[{ordinal}]"), f"raw_repeats[{ordinal}].repeat"),
            ordinal,
            f"raw_repeats[{ordinal}].repeat",
        )
        _require_equal(
            _positive_int(
                _required(repeat, "logical_payload_requests", f"raw_repeats[{ordinal}]"),
                f"raw_repeats[{ordinal}].logical_payload_requests",
            ),
            manifest_rows,
            f"raw_repeats[{ordinal}].logical_payload_requests",
        )
        for field in (
            "warm_process_seconds",
            "fetch_seconds",
            "lookup_seconds",
            "logical_payload_bytes",
            "lance_projected_bytes",
            "physical_read_bytes",
            "physical_read_calls",
            "peak_rss_bytes",
        ):
            _positive(_required(repeat, field, f"raw_repeats[{ordinal}]"), f"raw_repeats[{ordinal}].{field}")
        payload_digests.add(
            _validate_sha256(
                _required(repeat, "payload_digest_sha256", f"raw_repeats[{ordinal}]"),
                f"raw_repeats[{ordinal}].payload_digest_sha256",
            )
        )
        output_digests.add(
            _validate_sha256(
                _required(repeat, "output_digest_sha256", f"raw_repeats[{ordinal}]"),
                f"raw_repeats[{ordinal}].output_digest_sha256",
            )
        )
        logical_payload_bytes.add(
            _positive_int(
                _required(repeat, "logical_payload_bytes", f"raw_repeats[{ordinal}]"),
                f"raw_repeats[{ordinal}].logical_payload_bytes",
            )
        )
        lance_projected_bytes.add(
            _positive_int(
                _required(repeat, "lance_projected_bytes", f"raw_repeats[{ordinal}]"),
                f"raw_repeats[{ordinal}].lance_projected_bytes",
            )
        )
    if (
        len(payload_digests) != 1
        or len(output_digests) != 1
        or len(logical_payload_bytes) != 1
        or len(lance_projected_bytes) != 1
    ):
        raise ModelInputError("repeat digests and logical/projected payload bytes must be stable")
    _require_equal(next(iter(payload_digests)), source["payload_digest_sha256"], "input.source.payload_digest_sha256")
    _require_equal(next(iter(output_digests)), source["output_digest_sha256"], "input.source.output_digest_sha256")


def _repeat_ranges(data: Mapping[str, Any]) -> dict[str, Any]:
    repeats = [_mapping(value, "raw repeat") for value in _sequence(data["raw_repeats"], "raw_repeats")]
    count = len(repeats)

    def values(function: Callable[[Mapping[str, Any]], float]) -> list[float]:
        return [float(function(repeat)) for repeat in repeats]

    def measured(function: Callable[[Mapping[str, Any]], float]) -> dict[str, Any]:
        raw = values(function)
        return {
            **ObservedRange.from_values(raw).as_dict("measured_queue_diagnostic", repeat_count=count),
            "raw_values": raw,
        }

    end_to_end = {
        "time_basis": "warm_process_seconds",
        "images_per_second": measured(lambda item: item["logical_payload_requests"] / item["warm_process_seconds"]),
        "logical_payload_bytes_per_second": measured(
            lambda item: item["logical_payload_bytes"] / item["warm_process_seconds"]
        ),
        "physical_read_bytes_per_second": measured(
            lambda item: item["physical_read_bytes"] / item["warm_process_seconds"]
        ),
        "physical_read_calls_per_second": measured(
            lambda item: item["physical_read_calls"] / item["warm_process_seconds"]
        ),
        "seconds": measured(lambda item: item["warm_process_seconds"]),
    }
    fetch_only = {
        "time_basis": "fetch_seconds",
        "images_per_second": measured(lambda item: item["logical_payload_requests"] / item["fetch_seconds"]),
        "logical_payload_bytes_per_second": measured(
            lambda item: item["logical_payload_bytes"] / item["fetch_seconds"]
        ),
        "physical_read_bytes_per_second": measured(lambda item: item["physical_read_bytes"] / item["fetch_seconds"]),
        "physical_read_calls_per_second": measured(lambda item: item["physical_read_calls"] / item["fetch_seconds"]),
        "seconds": measured(lambda item: item["fetch_seconds"]),
    }
    workload = {
        "logical_payload_bytes_per_payload": measured(
            lambda item: item["logical_payload_bytes"] / item["logical_payload_requests"]
        ),
        "physical_read_bytes_per_payload": measured(
            lambda item: item["physical_read_bytes"] / item["logical_payload_requests"]
        ),
        "physical_read_calls_per_payload": measured(
            lambda item: item["physical_read_calls"] / item["logical_payload_requests"]
        ),
        "average_physical_read_bytes": measured(
            lambda item: item["physical_read_bytes"] / item["physical_read_calls"]
        ),
        "read_amplification": measured(lambda item: item["physical_read_bytes"] / item["logical_payload_bytes"]),
        "lance_backend_read_amplification": measured(
            lambda item: item["physical_read_bytes"] / item["lance_projected_bytes"]
        ),
        "peak_rss_bytes": measured(lambda item: item["peak_rss_bytes"]),
    }
    lookup = {
        "time_basis": "lookup_seconds",
        "probes_per_second": measured(lambda item: item["logical_payload_requests"] / item["lookup_seconds"]),
        "seconds": measured(lambda item: item["lookup_seconds"]),
    }
    return {
        "repeat_count": count,
        "end_to_end": end_to_end,
        "fetch_only": fetch_only,
        "lookup_only": lookup,
        "workload_shape": workload,
    }


def _observed_range(data: Mapping[str, Any]) -> ObservedRange:
    return ObservedRange(
        _positive(data["low"], "range.low"),
        _positive(data["observed"], "range.observed"),
        _positive(data["high"], "range.high"),
    )


def capacity_for_index(
    gpu_index_bytes: int,
    h100_memory_bytes: int,
    usable_fraction: float,
    gpus_per_node: int,
) -> dict[str, int]:
    index_bytes = _positive_int(gpu_index_bytes, "gpu_index_bytes")
    memory_bytes = _positive_int(h100_memory_bytes, "h100_memory_bytes")
    fraction = _positive(usable_fraction, "usable_fraction")
    if fraction > 1:
        raise ModelInputError("usable_fraction must be <= 1")
    per_gpu = math.floor(memory_bytes * fraction)
    if per_gpu <= 0:
        raise ModelInputError("usable GPU memory rounds to zero bytes")
    minimum_h100s = math.ceil(index_bytes / per_gpu)
    nodes = math.ceil(minimum_h100s / _positive_int(gpus_per_node, "gpus_per_node"))
    return {
        "usable_bytes_per_h100": per_gpu,
        "minimum_h100_count": minimum_h100s,
        "minimum_node_count": nodes,
        "allocated_h100_count_at_capacity_minimum": nodes * gpus_per_node,
    }


def scaled_rate_multiplier(resources: int, measured_resources: int, efficiency: float) -> float:
    resources = _positive_int(resources, "resources")
    measured_resources = _positive_int(measured_resources, "measured_resources")
    efficiency = _positive(efficiency, "efficiency")
    if efficiency > 1:
        raise ModelInputError("efficiency must be <= 1")
    ratio = resources / measured_resources
    if ratio <= 1:
        return ratio
    return 1 + efficiency * (ratio - 1)


def _runtime_from_rate(items: int, rate: ObservedRange) -> ObservedRange:
    count = _positive_int(items, "items")
    return ObservedRange(count / rate.high, count / rate.observed, count / rate.low)


def _sum_ranges(left: ObservedRange, right: ObservedRange) -> ObservedRange:
    return ObservedRange(
        left.low + right.low,
        left.observed + right.observed,
        left.high + right.high,
    )


def _dominant_term(terms: Mapping[str, ObservedRange]) -> str:
    return max(terms, key=lambda name: terms[name].observed)


def _scenario(
    name: str,
    image_references: int,
    *,
    data: Mapping[str, Any],
    measured: Mapping[str, Any],
    config: ModelConfig,
    current: bool,
) -> dict[str, Any]:
    inventory = _mapping(data["reference_inventory"], "reference_inventory")
    mint = _mapping(data["current_mint"], "current_mint")
    current_refs = _positive_int(mint["image_references"]["value"], "current refs")
    current_unique = _positive_int(mint["unique_reference_keys"]["value"], "current unique")
    reference_keys = _positive_int(inventory["keys"], "reference keys")
    scale = image_references / current_refs
    unique_keys = current_unique if current else math.ceil(current_unique * scale)
    payload_reads = image_references
    reference_scale = unique_keys / reference_keys
    sidecar_bytes = math.ceil(_positive_int(inventory["sidecar_bytes"], "sidecar bytes") * reference_scale)
    gpu_index_bytes = math.ceil(_positive_int(inventory["gpu_index_bytes"], "gpu index bytes") * reference_scale)
    capacity = capacity_for_index(
        gpu_index_bytes,
        config.h100_memory_bytes,
        config.usable_gpu_memory_fraction,
        config.gpus_per_node,
    )

    shape = _mapping(measured["workload_shape"], "workload_shape")
    logical_bytes = _observed_range(shape["logical_payload_bytes_per_payload"]).scaled(payload_reads)
    physical_bytes = _observed_range(shape["physical_read_bytes_per_payload"]).scaled(payload_reads)
    physical_calls = _observed_range(shape["physical_read_calls_per_payload"]).scaled(payload_reads)

    measured_index_bytes = _positive_int(inventory["gpu_index_bytes"], "measured gpu index bytes")
    allocated_h100s = capacity["allocated_h100_count_at_capacity_minimum"]
    sharded_bytes_per_h100 = math.ceil(gpu_index_bytes / allocated_h100s)
    setup_scale = sharded_bytes_per_h100 / measured_index_bytes
    setup_observation = _positive(inventory["cold_setup_seconds"], "cold setup seconds")
    modeled_setup = ObservedRange(setup_observation, setup_observation, setup_observation).scaled(setup_scale)

    resources = _mapping(data["measurement_resources"], "measurement_resources")
    measured_readers = _positive_int(resources["active_payload_readers"], "active payload readers")
    end_to_end = _mapping(measured["end_to_end"], "end_to_end")
    profiles: list[dict[str, Any]] = []
    profile_specs = [("measured_one_reader_geometry", 1, measured_readers)]
    profile_specs.extend(
        (f"modeled_{nodes}_node_{nodes * config.readers_per_node}_readers", nodes, nodes * config.readers_per_node)
        for nodes in config.throughput_node_counts
    )
    seen: set[tuple[int, int]] = set()
    for profile_name, node_count, reader_count in profile_specs:
        if (node_count, reader_count) in seen:
            continue
        seen.add((node_count, reader_count))
        rate_factor = scaled_rate_multiplier(
            reader_count,
            measured_readers,
            config.marginal_reader_scaling_efficiency,
        )
        image_rate = _observed_range(end_to_end["images_per_second"]).scaled(rate_factor)
        logical_rate = _observed_range(end_to_end["logical_payload_bytes_per_second"]).scaled(rate_factor)
        physical_byte_rate = _observed_range(end_to_end["physical_read_bytes_per_second"]).scaled(rate_factor)
        physical_call_rate = _observed_range(end_to_end["physical_read_calls_per_second"]).scaled(rate_factor)
        direct_runtime = _runtime_from_rate(payload_reads, image_rate)
        arithmetic_terms = {
            "logical_payload_byte_rate_term": logical_bytes.divided_by(logical_rate),
            "physical_read_byte_rate_term": physical_bytes.divided_by(physical_byte_rate),
            "physical_read_call_rate_term": physical_calls.divided_by(physical_call_rate),
        }
        profiles.append(
            {
                "name": profile_name,
                "throughput_node_count": node_count,
                "active_payload_reader_count": reader_count,
                "reader_rate_multiplier": rate_factor,
                "capacity_feasible": node_count >= capacity["minimum_node_count"],
                "classification": "queue_diagnostic_extrapolation",
                "end_to_end_stage_seconds": direct_runtime.as_dict("queue_diagnostic_extrapolation"),
                "end_to_end_with_modeled_parallel_setup_seconds": _sum_ranges(
                    modeled_setup,
                    direct_runtime,
                ).as_dict("queue_diagnostic_extrapolation"),
                "effective_rates": {
                    "images_per_second": image_rate.as_dict("modeled_reader_scaling"),
                    "logical_payload_bytes_per_second": logical_rate.as_dict("modeled_reader_scaling"),
                    "physical_read_bytes_per_second": physical_byte_rate.as_dict("modeled_reader_scaling"),
                    "physical_read_calls_per_second": physical_call_rate.as_dict("modeled_reader_scaling"),
                },
                "arithmetic_diagnostics": {
                    **{
                        term_name: term.as_dict("derived_queue_diagnostic")
                        for term_name, term in arithmetic_terms.items()
                    },
                    "dominant_arithmetic_term": _dominant_term(arithmetic_terms),
                    "storage_saturation_proven": False,
                    "note": (
                        "These correlated arithmetic terms are diagnostics, not independent "
                        "storage ceilings and not evidence of storage saturation."
                    ),
                },
            }
        )

    shuffle = _mapping(data["shuffle_assumption"], "shuffle_assumption")
    shuffle_bytes = (
        image_references
        * _positive_int(shuffle["bytes_per_probe_per_coordinate_pass"], "shuffle bytes")
        * _positive_int(shuffle["coordinate_passes"], "shuffle passes")
    )
    return {
        "name": name,
        "scenario_classification": (
            "inventory_measured_runtime_queue_diagnostic" if current else "queue_diagnostic_extrapolation"
        ),
        "scales": {
            "image_references": {
                "count": image_references,
                "classification": "measured_inventory" if current else "target_scenario",
            },
            "unique_reference_keys": {
                "count": unique_keys,
                "classification": "measured_inventory" if current else "linear_extrapolation",
            },
            "payload_reads": {
                "count": payload_reads,
                "classification": "modeled_no_cross_window_reuse_upper_bound",
                "policy": "one_payload_fetch_per_image_reference",
            },
            "logical_payload_bytes": logical_bytes.as_dict("queue_sample_extrapolation"),
            "physical_read_bytes": physical_bytes.as_dict("queue_sample_extrapolation"),
            "physical_read_calls": physical_calls.as_dict("queue_sample_extrapolation"),
        },
        "index_capacity": {
            "sidecar_bytes": sidecar_bytes,
            "gpu_index_bytes": gpu_index_bytes,
            "classification": "modeled_linear_index_density_and_even_sharding",
            **capacity,
            "modeled_sharded_setup_seconds": modeled_setup.as_dict("modeled_linear_setup_and_even_sharding"),
            "note": (
                "Index-memory capacity is independent of throughput provisioning. "
                "Even sharding and linear setup are architecture assumptions, not benchmark results."
            ),
        },
        "compact_coordinate_shuffle": {
            "total_bytes": shuffle_bytes,
            "bytes_per_probe_per_coordinate_pass": shuffle["bytes_per_probe_per_coordinate_pass"],
            "coordinate_passes": shuffle["coordinate_passes"],
            "classification": shuffle["classification"],
        },
        "throughput_profiles": profiles,
    }


def _sparse_call_sensitivity(scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scenario in scenarios:
        calls = _observed_range(scenario["scales"]["physical_read_calls"])
        rows = []
        for factor in (1, 2, 4, 8, 16):
            adjusted = calls.scaled(1 / factor)
            rows.append(
                {
                    "payloads_per_physical_call_factor": factor,
                    "physical_read_calls": adjusted.as_dict("modeled_call_count_only"),
                    "relative_calls": 1 / factor,
                    "runtime_estimate": None,
                    "note": (
                        "Runtime is intentionally not estimated: changing call locality also "
                        "changes physical bytes and effective rates."
                    ),
                }
            )
        result.append({"scenario": scenario["name"], "coalescing": rows})
    return result


def build_scale_model(
    input_data: Mapping[str, Any],
    config: ModelConfig,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Validate v3 evidence and build capacity plus queue-diagnostic projections."""

    _validate_model_input(input_data)
    measured = _repeat_ranges(input_data)
    mint_refs = _positive_int(
        input_data["current_mint"]["image_references"]["value"],
        "current MINT image references",
    )
    scenarios = [
        _scenario(
            "current MINT",
            mint_refs,
            data=input_data,
            measured=measured,
            config=config,
            current=True,
        )
    ]
    scenarios.extend(
        _scenario(
            name,
            count,
            data=input_data,
            measured=measured,
            config=config,
            current=False,
        )
        for name, count in TARGET_IMAGE_REFERENCES
    )
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "generated_at": generated_at or datetime.now(tz=UTC).isoformat(),
        "model_status": "queue_diagnostic_not_storage_saturation",
        "source": input_data["source"],
        "evidence_legend": {
            "measured_inventory": "Direct dataset/document inventory.",
            "measured_queue_diagnostic": "Repeated correct one-reader queue measurement.",
            "modeled": "Arithmetic using explicit assumptions; not benchmark evidence.",
            "queue_diagnostic_extrapolation": (
                "Projected from one active reader; not a saturated per-node rate or SLA."
            ),
        },
        "measurement_resources": input_data["measurement_resources"],
        "measured_queue_evidence": measured,
        "configuration": {
            "h100_memory_bytes": config.h100_memory_bytes,
            "usable_gpu_memory_fraction": config.usable_gpu_memory_fraction,
            "gpus_per_node": config.gpus_per_node,
            "readers_per_node": config.readers_per_node,
            "throughput_node_counts": list(config.throughput_node_counts),
            "marginal_reader_scaling_efficiency": config.marginal_reader_scaling_efficiency,
        },
        "scenarios": scenarios,
        "sensitivity": {
            "sparse_physical_calls": _sparse_call_sensitivity(scenarios),
            "note": (
                "Sparse-call reductions are first-class, but runtime is not inferred without "
                "a measured locality/read-size response curve."
            ),
        },
        "assumptions": [
            {
                "id": "remote_primary",
                "value": "The source evidence is remote object-storage Lance I/O.",
            },
            {
                "id": "no_node_saturation",
                "value": (
                    "The source run used one active payload reader on one physical node; "
                    "all multi-reader and multi-node rates are queue-diagnostic extrapolations."
                ),
            },
            {
                "id": "capacity_is_not_throughput",
                "value": (
                    "Minimum nodes needed to hold an evenly sharded index never choose the "
                    "throughput profile; throughput node counts are explicit."
                ),
            },
            {
                "id": "payload_read_upper_bound",
                "value": (
                    "Payload reads equal image references, modeling no reuse across streaming "
                    "windows. This is an upper-bound policy, not measured current-MINT I/O."
                ),
            },
            {
                "id": "correlated_rate_terms",
                "value": (
                    "Logical-byte, physical-byte, and physical-call rates come from the same "
                    "repeats. Their dominant arithmetic term does not prove storage saturation."
                ),
            },
            {
                "id": "scope",
                "value": (
                    "Runtime excludes scheduler startup, skew, retries, spill, output writes, "
                    "object-store throttling changes, and failures."
                ),
            },
        ],
    }


def _human_count(value: float) -> str:
    number = float(value)
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(number) >= divisor:
            return f"{number / divisor:.3g}{suffix}"
    return f"{number:.3g}"


def _human_bytes(value: float) -> str:
    number = float(value)
    for suffix, divisor in (
        ("PiB", 1024**5),
        ("TiB", 1024**4),
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if abs(number) >= divisor:
            return f"{number / divisor:.3g} {suffix}"
    return f"{number:.3g} B"


def _human_seconds(value: float) -> str:
    if value >= 86400:
        return f"{value / 86400:.3g} d"
    if value >= 3600:
        return f"{value / 3600:.3g} h"
    if value >= 60:
        return f"{value / 60:.3g} min"
    return f"{value:.3g} s"


def _range_text(data: Mapping[str, Any], formatter: Callable[[float], str]) -> str:
    return " / ".join(formatter(float(data[key])) for key in ("low", "observed", "high"))


def render_markdown(model: Mapping[str, Any]) -> str:
    """Render the evidence, capacity, and explicit throughput profiles."""

    evidence = model["measured_queue_evidence"]
    end_to_end = evidence["end_to_end"]
    fetch_only = evidence["fetch_only"]
    workload = evidence["workload_shape"]
    lines = [
        "# GPU Lance queue-diagnostic scale model v3",
        "",
        "**This is not a storage-saturation result or an SLA.** Ranges are minimum / median / maximum across two correct repeats.",
        "",
        "## Measured Queue Evidence",
        "",
        "| Timing basis | Images/s | Logical payload rate | Physical read rate | Physical calls/s |",
        "|---|---:|---:|---:|---:|",
        "| End-to-end stage | {images} | {logical} | {physical} | {calls} |".format(
            images=_range_text(end_to_end["images_per_second"], _human_count),
            logical=_range_text(end_to_end["logical_payload_bytes_per_second"], _human_bytes),
            physical=_range_text(end_to_end["physical_read_bytes_per_second"], _human_bytes),
            calls=_range_text(end_to_end["physical_read_calls_per_second"], _human_count),
        ),
        "| Lance fetch only | {images} | {logical} | {physical} | {calls} |".format(
            images=_range_text(fetch_only["images_per_second"], _human_count),
            logical=_range_text(fetch_only["logical_payload_bytes_per_second"], _human_bytes),
            physical=_range_text(fetch_only["physical_read_bytes_per_second"], _human_bytes),
            calls=_range_text(fetch_only["physical_read_calls_per_second"], _human_count),
        ),
        "",
        "Workload shape: {calls} physical reads/payload, {size} average physical read, {amp}x read amplification.".format(
            calls=_range_text(workload["physical_read_calls_per_payload"], lambda value: f"{value:.4g}"),
            size=_range_text(workload["average_physical_read_bytes"], _human_bytes),
            amp=_range_text(workload["read_amplification"], lambda value: f"{value:.4g}"),
        ),
        "",
        "## Scale And Capacity",
        "",
        "| Scenario | Classification | Image refs | Unique keys | Payload reads | Logical payload bytes | Physical read bytes | Minimum index nodes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in model["scenarios"]:
        scales = scenario["scales"]
        capacity = scenario["index_capacity"]
        lines.append(
            "| {name} | {classification} | {refs} | {keys} | {reads} | {logical} | {physical} | {nodes} |".format(
                name=scenario["name"],
                classification=scenario["scenario_classification"],
                refs=_human_count(scales["image_references"]["count"]),
                keys=_human_count(scales["unique_reference_keys"]["count"]),
                reads=_human_count(scales["payload_reads"]["count"]),
                logical=_range_text(scales["logical_payload_bytes"], _human_bytes),
                physical=_range_text(scales["physical_read_bytes"], _human_bytes),
                nodes=capacity["minimum_node_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Explicit Throughput Profiles",
            "",
            "| Scenario | Profile | Nodes | Readers | Holds index | End-to-end stage runtime | Dominant arithmetic term |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for scenario in model["scenarios"]:
        for profile in scenario["throughput_profiles"]:
            lines.append(
                "| {scenario} | {profile} | {nodes} | {readers} | {feasible} | {runtime} | {term} |".format(
                    scenario=scenario["name"],
                    profile=profile["name"],
                    nodes=profile["throughput_node_count"],
                    readers=profile["active_payload_reader_count"],
                    feasible="yes" if profile["capacity_feasible"] else "no",
                    runtime=_range_text(profile["end_to_end_stage_seconds"], _human_seconds),
                    term=profile["arithmetic_diagnostics"]["dominant_arithmetic_term"],
                )
            )
    lines.extend(
        [
            "",
            "The dominant arithmetic term compares correlated equations from the same queue repeats. It is not evidence of storage saturation.",
            "",
            "## Assumptions",
            "",
        ]
    )
    lines.extend(f"- **{item['id']}**: {item['value']}" for item in model["assumptions"])
    return "\n".join(lines) + "\n"


def _write_text(destination: str | Path, content: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _node_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("node counts must be comma-separated integers") from error
    if not counts or any(item <= 0 for item in counts):
        raise argparse.ArgumentTypeError("node counts must be positive")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Validated v3 model-input JSON")
    source.add_argument("--benchmark-artifact", type=Path, help="Completed benchmark JSON")
    parser.add_argument("--write-input", type=Path, help="Write generated v3 model input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--reference-keys", type=int)
    parser.add_argument("--sidecar-bytes", type=int)
    parser.add_argument("--reference-manifest-uri")
    parser.add_argument("--reference-manifest-sha256")
    parser.add_argument("--current-mint-image-references", type=int)
    parser.add_argument("--current-mint-unique-reference-keys", type=int)
    parser.add_argument("--measurement-h100-count", type=int, default=1)
    parser.add_argument("--measurement-physical-node-count", type=int, default=1)
    parser.add_argument("--active-payload-readers", type=int, default=1)
    parser.add_argument("--compact-shuffle-bytes-per-probe-per-pass", type=int, default=12)
    parser.add_argument("--compact-shuffle-coordinate-passes", type=int, default=2)
    parser.add_argument("--h100-memory-gib", type=float, default=80.0)
    parser.add_argument("--usable-gpu-memory-fraction", type=float, default=0.80)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--readers-per-node", type=int, default=8)
    parser.add_argument("--throughput-node-counts", type=_node_counts, default=(1, 2, 4, 8))
    parser.add_argument("--marginal-reader-scaling-efficiency", type=float, default=0.80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.benchmark_artifact is not None:
        required = {
            "--reference-keys": args.reference_keys,
            "--sidecar-bytes": args.sidecar_bytes,
            "--reference-manifest-uri": args.reference_manifest_uri,
            "--reference-manifest-sha256": args.reference_manifest_sha256,
            "--current-mint-image-references": args.current_mint_image_references,
            "--current-mint-unique-reference-keys": args.current_mint_unique_reference_keys,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"benchmark input generation requires: {', '.join(missing)}")
        artifact = json.loads(args.benchmark_artifact.read_text(encoding="utf-8"))
        input_data = queue_model_input_from_benchmark(
            _mapping(artifact, "benchmark"),
            source_path=str(args.benchmark_artifact),
            source_sha256=_sha256_file(args.benchmark_artifact),
            reference_keys=args.reference_keys,
            sidecar_bytes=args.sidecar_bytes,
            reference_manifest_uri=args.reference_manifest_uri,
            reference_manifest_sha256=args.reference_manifest_sha256,
            current_mint_image_references=args.current_mint_image_references,
            current_mint_unique_reference_keys=args.current_mint_unique_reference_keys,
            measurement_h100_count=args.measurement_h100_count,
            measurement_physical_node_count=args.measurement_physical_node_count,
            active_payload_readers=args.active_payload_readers,
            compact_shuffle_bytes_per_probe_per_pass=args.compact_shuffle_bytes_per_probe_per_pass,
            compact_shuffle_coordinate_passes=args.compact_shuffle_coordinate_passes,
        )
        if args.write_input is None:
            raise SystemExit("--benchmark-artifact requires --write-input")
        _write_text(args.write_input, json.dumps(input_data, indent=2, sort_keys=True) + "\n")
    else:
        input_data = _mapping(
            json.loads(args.input.read_text(encoding="utf-8")),
            "input",
        )
    config = ModelConfig(
        h100_memory_bytes=round(_positive(args.h100_memory_gib, "h100_memory_gib") * 1024**3),
        usable_gpu_memory_fraction=args.usable_gpu_memory_fraction,
        gpus_per_node=args.gpus_per_node,
        readers_per_node=args.readers_per_node,
        throughput_node_counts=args.throughput_node_counts,
        marginal_reader_scaling_efficiency=args.marginal_reader_scaling_efficiency,
    )
    model = build_scale_model(input_data, config)
    _write_text(args.output, json.dumps(model, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        _write_text(args.markdown, render_markdown(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
