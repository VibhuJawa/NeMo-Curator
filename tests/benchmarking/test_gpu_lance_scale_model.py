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

# ruff: noqa: PLR0913
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "benchmarking" / "scripts" / "gpu_lance_scale_model.py"
SPEC = importlib.util.spec_from_file_location("gpu_lance_scale_model", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


def _correctness(rows: int, payload_bytes: int, suffix: str = "a") -> dict[str, Any]:
    return {
        "correct": True,
        "order_matches_manifest": True,
        "row_count": rows,
        "expected_row_count": rows,
        "present_rows": rows,
        "missing_payload_rows": 0,
        "payload_bytes": payload_bytes,
        "payload_digest_sha256": suffix * 64,
        "output_digest_sha256": "b" * 64,
    }


def _repeat(
    ordinal: int,
    *,
    rows: int = 100,
    warm_seconds: float,
    fetch_seconds: float,
    lookup_seconds: float,
    payload_bytes: int = 10_000,
    physical_bytes: int,
    physical_calls: int,
) -> dict[str, Any]:
    backend = {
        "logical_payload_requests": rows,
        "found_unique_keys": rows,
        "unique_payloads": rows,
        "requested_unique_keys": rows,
        "missing_unique_keys": 0,
        "logical_duplicate_requests": 0,
        "lance_fetch_seconds": fetch_seconds,
        "lance_lookup_seconds": lookup_seconds,
        "lance_fetched_bytes": payload_bytes,
        "physical_reads_per_payload": physical_calls / rows,
        "average_physical_read_bytes": physical_bytes / physical_calls,
        "read_amplification": physical_bytes / payload_bytes,
        "peak_rss_bytes": 1_000_000 + ordinal,
    }
    return {
        "status": "completed",
        "repeat": ordinal,
        "warm_process_seconds": warm_seconds,
        "fetch_seconds": fetch_seconds,
        "lookup_seconds": lookup_seconds,
        "images_per_second": rows / warm_seconds,
        "payload_bytes": payload_bytes,
        "lance_read_iops": physical_calls,
        "lance_read_bytes": physical_bytes,
        "correctness": _correctness(rows, payload_bytes),
        "backend_metrics": backend,
    }


def _benchmark() -> dict[str, Any]:
    rows = 100
    return {
        "status": "completed",
        "environment": {
            "python": "3.12.0",
            "platform": "linux-test",
            "packages": {
                "nemo-curator": "1.3.0+test",
                "pyarrow": "22.0.0",
                "pylance": "9.0.0b11",
            },
        },
        "configuration": {
            "repeat_count": 2,
            "warmup_count": 0,
            "task_rows": 10,
            "coalesce_tasks": 10,
            "rows_per_coalesced_fetch": rows,
            "lookup_batch_size": 2_000,
            "fetch_batch_size": 25,
            "max_lookup_bytes": 1024,
            "max_pending_fetch_batches": 4,
            "payload_read_mode": "sparse",
            "take_scan_batch_readahead": 16,
            "validate_payload_keys": False,
            "io_threads": 8,
            "index_mirror": None,
            "copy_index_to_node_local": False,
            "reference_files": ["/indexes/part-000.parquet"],
            "reference_manifest_uri": "/indexes/manifest.json",
            "reference_manifest_sha256": "e" * 64,
            "reference_storage_option_keys": [],
        },
        "manifest": {"rows": rows, "digest_sha256": "c" * 64},
        "dataset": {
            "uri": "s3://example/dataset",
            "version": 4,
            "source_columns": {"image": "image"},
            "storage_option_keys": ["endpoint"],
        },
        "arms": {
            MODEL.BENCHMARK_ARM: {
                "status": "completed",
                "summary": {"stable_correctness_digest": True},
                "cold_setup": {
                    "wall_seconds": 10.0,
                    "backend_metrics": {
                        "gpu_reference_rows": 100,
                        "gpu_reference_bytes": 400,
                    },
                },
                "warmups": [],
                "repeats": [
                    _repeat(
                        0,
                        warm_seconds=20.0,
                        fetch_seconds=10.0,
                        lookup_seconds=1.0,
                        physical_bytes=15_000,
                        physical_calls=200,
                    ),
                    _repeat(
                        1,
                        warm_seconds=25.0,
                        fetch_seconds=12.5,
                        lookup_seconds=2.0,
                        physical_bytes=16_000,
                        physical_calls=220,
                    ),
                ],
            }
        },
    }


def _input(
    report: dict[str, Any] | None = None,
    *,
    source_path: str = "benchmark.json",
    source_sha256: str = "d" * 64,
) -> dict[str, Any]:
    return MODEL.queue_model_input_from_benchmark(
        report or _benchmark(),
        source_path=source_path,
        source_sha256=source_sha256,
        reference_keys=100,
        sidecar_bytes=1_000,
        reference_manifest_uri="/indexes/manifest.json",
        reference_manifest_sha256="e" * 64,
        current_mint_image_references=200,
        current_mint_unique_reference_keys=100,
        generated_at="test",
    )


def _write_terminal_benchmark(
    root: Path,
    evidence_class: str | None,
    *,
    benchmark_waves: int | None = None,
    schema_version: int = 2,
) -> tuple[dict[str, Any], Path, str]:
    root.mkdir()
    benchmark = _benchmark()
    benchmark_path = root / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    source_sha256 = MODEL._sha256_file(benchmark_path)
    eligibility = {
        "schema_version": schema_version,
        "terminal": True,
        "status": "eligible",
        "artifacts": {"benchmark": {"sha256": source_sha256}},
    }
    if evidence_class is not None:
        eligibility["evidence_class"] = evidence_class
    if benchmark_waves is not None:
        eligibility["benchmark_validation"] = {"waves": benchmark_waves}
    (root / "eligibility.json").write_text(json.dumps(eligibility), encoding="utf-8")
    return benchmark, benchmark_path, source_sha256


def test_input_generation_preserves_repeats_and_uses_coherent_time_bases() -> None:
    input_data = _input()
    model = MODEL.build_scale_model(input_data, MODEL.ModelConfig(), generated_at="test")

    assert input_data["schema_version"] == 4
    assert len(input_data["raw_repeats"]) == 2
    assert input_data["raw_repeats"][0]["physical_read_calls"] == 200
    assert input_data["source"]["sidecar_identity"]["reference_file_count"] == 1
    assert "/indexes/part-000.parquet" not in json.dumps(input_data)
    measured = model["measured_queue_evidence"]
    assert measured["end_to_end"]["images_per_second"]["raw_values"] == [5.0, 4.0]
    assert measured["end_to_end"]["physical_read_calls_per_second"]["raw_values"] == [10.0, 8.8]
    assert measured["fetch_only"]["physical_read_calls_per_second"]["raw_values"] == [20.0, 17.6]
    assert measured["end_to_end"]["time_basis"] == "warm_process_seconds"
    assert measured["fetch_only"]["time_basis"] == "fetch_seconds"


def test_scale_model_accepts_only_primary_saturation_terminal_evidence(tmp_path: Path) -> None:
    benchmark, path, source_sha256 = _write_terminal_benchmark(
        tmp_path / "primary",
        "primary_saturation",
        benchmark_waves=8,
    )
    input_data = _input(benchmark, source_path=str(path), source_sha256=source_sha256)

    assert input_data["source"]["terminal_eligibility"]["evidence_class"] == "primary_saturation"

    legacy, legacy_path, legacy_sha256 = _write_terminal_benchmark(
        tmp_path / "legacy-primary",
        None,
        benchmark_waves=4,
        schema_version=1,
    )
    legacy_input = _input(legacy, source_path=str(legacy_path), source_sha256=legacy_sha256)
    assert legacy_input["source"]["terminal_eligibility"]["evidence_class"] == "primary_saturation"

    legacy_embedded_input = json.loads(json.dumps(input_data))
    legacy_embedded_input["schema_version"] = 3
    legacy_embedded_input["source"]["terminal_eligibility"].pop("evidence_class")
    MODEL.build_scale_model(legacy_embedded_input, MODEL.ModelConfig())

    current_missing_class = json.loads(json.dumps(input_data))
    current_missing_class["source"]["terminal_eligibility"].pop("evidence_class")
    with pytest.raises(MODEL.ModelInputError, match="evidence_class"):
        MODEL.build_scale_model(current_missing_class, MODEL.ModelConfig())

    input_data["source"]["terminal_eligibility"]["evidence_class"] = "locality_sensitivity"
    with pytest.raises(MODEL.ModelInputError, match="evidence_class"):
        MODEL.build_scale_model(input_data, MODEL.ModelConfig())

    rejected = (
        ("locality", "locality_sensitivity", 1, 2),
        ("misclassified-locality", "primary_saturation", 2, 2),
        ("legacy-locality", None, 1, 1),
        ("current-missing", None, 8, 2),
        ("missing", None, None, 2),
    )
    for name, evidence_class, benchmark_waves, schema_version in rejected:
        benchmark, path, source_sha256 = _write_terminal_benchmark(
            tmp_path / name,
            evidence_class,
            benchmark_waves=benchmark_waves,
            schema_version=schema_version,
        )
        with pytest.raises(MODEL.ModelInputError, match="evidence_class"):
            _input(benchmark, source_path=str(path), source_sha256=source_sha256)


@pytest.mark.parametrize("status", ["running", "failed", "tearing_down"])
def test_input_generation_rejects_noncompleted_artifact(status: str) -> None:
    report = _benchmark()
    report["status"] = status
    with pytest.raises(MODEL.ModelInputError, match=r"benchmark\.status"):
        _input(report)


def test_input_generation_rejects_partial_or_incorrect_repeats() -> None:
    partial = _benchmark()
    partial["arms"][MODEL.BENCHMARK_ARM]["repeats"].pop()
    with pytest.raises(MODEL.ModelInputError, match="configuration requires"):
        _input(partial)

    incorrect = _benchmark()
    incorrect["arms"][MODEL.BENCHMARK_ARM]["repeats"][1]["correctness"]["correct"] = False
    with pytest.raises(MODEL.ModelInputError, match="correct"):
        _input(incorrect)


def test_input_generation_rejects_unstable_digest_and_teardown_errors() -> None:
    unstable = _benchmark()
    unstable["arms"][MODEL.BENCHMARK_ARM]["repeats"][1]["correctness"]["payload_digest_sha256"] = "e" * 64
    with pytest.raises(MODEL.ModelInputError, match="payload digest differs"):
        _input(unstable)

    teardown = _benchmark()
    teardown["teardown_errors"] = {"arm": {"message": "boom"}}
    with pytest.raises(MODEL.ModelInputError, match="teardown_errors"):
        _input(teardown)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("dataset", "source_columns"),
        ("configuration", "reference_files"),
        ("configuration", "payload_read_mode"),
        ("configuration", "max_pending_fetch_batches"),
        ("configuration", "index_mirror"),
        ("configuration", "validate_payload_keys"),
        ("environment", "packages"),
    ],
)
def test_input_generation_requires_complete_evidence_identity(section: str, field: str) -> None:
    report = _benchmark()
    report[section].pop(field)

    with pytest.raises(MODEL.ModelInputError):
        _input(report)


def test_model_validation_rejects_tampered_policy_and_sidecar_identity() -> None:
    policy = _input()
    policy["source"]["queue_configuration"].pop("payload_read_mode")
    with pytest.raises(MODEL.ModelInputError, match="payload_read_mode"):
        MODEL.build_scale_model(policy, MODEL.ModelConfig())

    sidecar = _input()
    sidecar["source"]["sidecar_identity"].pop("manifest_sha256")
    with pytest.raises(MODEL.ModelInputError, match="manifest_sha256"):
        MODEL.build_scale_model(sidecar, MODEL.ModelConfig())


def test_model_validation_rejects_tampered_embedded_evidence() -> None:
    input_data = _input()
    input_data["raw_repeats"][1]["status"] = "failed"
    with pytest.raises(MODEL.ModelInputError, match="status"):
        MODEL.build_scale_model(input_data, MODEL.ModelConfig())


def test_capacity_and_throughput_resources_are_independent() -> None:
    model = MODEL.build_scale_model(
        _input(),
        MODEL.ModelConfig(
            h100_memory_bytes=1_000,
            usable_gpu_memory_fraction=0.5,
            gpus_per_node=2,
            readers_per_node=2,
            throughput_node_counts=(1, 4),
        ),
        generated_at="test",
    )
    target = next(item for item in model["scenarios"] if item["name"] == "6B")
    assert target["scenario_classification"] == "queue_diagnostic_extrapolation"
    assert target["index_capacity"]["minimum_node_count"] > 4
    assert [item["throughput_node_count"] for item in target["throughput_profiles"]] == [1, 1, 4]
    assert target["throughput_profiles"][1]["capacity_feasible"] is False
    assert target["throughput_profiles"][2]["capacity_feasible"] is False

    rendered = json.dumps(model).lower()
    assert "bandwidth" not in rendered
    assert "dominant_arithmetic_term" in rendered
    assert model["model_status"] == "queue_diagnostic_not_storage_saturation"


def test_payload_reads_and_shuffle_are_classified_as_assumptions() -> None:
    input_data = _input()
    model = MODEL.build_scale_model(input_data, MODEL.ModelConfig(), generated_at="test")
    current = model["scenarios"][0]

    assert input_data["current_mint"]["payload_reads"]["classification"] == (
        "modeled_no_cross_window_reuse_upper_bound"
    )
    assert current["scales"]["payload_reads"]["classification"] == ("modeled_no_cross_window_reuse_upper_bound")
    assert current["compact_coordinate_shuffle"]["classification"] == (
        "modeled_schema_floor_excluding_transport_overhead"
    )
    assert all(
        row["runtime_estimate"] is None for row in model["sensitivity"]["sparse_physical_calls"][0]["coalescing"]
    )
