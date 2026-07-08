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

import json
from pathlib import Path

import pytest

from benchmarking.scripts import gpu_lance_scaling_report as report


def _measurements(
    root: Path,
    label: str,
    *,
    gpus: int,
    coalesce_tasks: int = 1,
    io_threads: int = 16,
) -> tuple[report.Measurement, ...]:
    fixture = report._self_test_fixture(scale=float(gpus), coalesce_tasks=coalesce_tasks)
    fixture["configuration"]["io_threads"] = io_threads
    path = root / f"{label}.json"
    labeled = report.parse_labeled_path(f"{label}[nodes=1,gpus={gpus}]={path}")
    accepted = report.validate_harness_result(labeled, fixture, path, label)
    return accepted.measurements


def _measurements_for_gpu_backend(
    root: Path,
    label: str,
    backend: str,
    *,
    gpus: int = 1,
    validate_payload_keys: bool = False,
) -> tuple[report.Measurement, ...]:
    fixture = report._self_test_fixture(scale=float(gpus))
    arms = fixture["arms"]
    assert isinstance(arms, dict)
    arms[backend] = arms.pop("gpu_lance_column_fetch_stage")
    configuration = fixture["configuration"]
    assert isinstance(configuration, dict)
    configuration["validate_payload_keys"] = validate_payload_keys
    path = root / f"{label}.json"
    labeled = report.parse_labeled_path(f"{label}[nodes=1,gpus={gpus},backend={backend}]={path}")
    return report.validate_harness_result(labeled, fixture, path, label).measurements


@pytest.mark.parametrize("backend", ["lance_ray_gpu_fetcher", "lance_ray_gpu_actor"])
def test_lance_ray_gpu_backends_are_accepted_and_classified_as_gpu(tmp_path: Path, backend: str) -> None:
    measurements = _measurements_for_gpu_backend(tmp_path, backend, backend)

    assert len(measurements) == 1
    assert measurements[0].backend == backend
    assert measurements[0].backend_class == "gpu"
    assert measurements[0].comparison_eligibility_errors == ()


def test_lance_ray_gpu_backends_do_not_form_cross_backend_strong_scaling_group(tmp_path: Path) -> None:
    measurements = (
        *_measurements_for_gpu_backend(tmp_path, "fetcher", "lance_ray_gpu_fetcher", gpus=1),
        *_measurements_for_gpu_backend(tmp_path, "actor", "lance_ray_gpu_actor", gpus=2),
    )

    assert report.strong_scaling(measurements) == []


@pytest.mark.parametrize("backend", ["lance_ray_gpu_fetcher", "lance_ray_gpu_actor"])
def test_lance_ray_gpu_backends_preserve_comparison_identity_gate(tmp_path: Path, backend: str) -> None:
    cpu = tuple(
        measurement
        for measurement in _measurements(tmp_path, "cpu", gpus=1)
        if measurement.backend == "cpu_lance_column_fetch_stage"
    )
    gpu = _measurements_for_gpu_backend(
        tmp_path,
        backend,
        backend,
        validate_payload_keys=True,
    )

    assert cpu[0].comparison_eligibility_errors == ()
    assert gpu[0].comparison_eligibility_errors == ()
    assert report.cpu_gpu_speedups((*cpu, *gpu)) == []


def test_strong_scaling_excludes_window_and_io_thread_mismatches(tmp_path: Path) -> None:
    measurements = (
        *_measurements(tmp_path, "one", gpus=1),
        *_measurements(tmp_path, "two-compatible", gpus=2),
        *_measurements(tmp_path, "two-window-mismatch", gpus=2, coalesce_tasks=2),
        *_measurements(tmp_path, "two-io-mismatch", gpus=2, io_threads=32),
    )

    groups = report.strong_scaling(measurements)

    assert len(groups) == 1
    assert [point["label"].split("[", 1)[0] for point in groups[0]["points"]] == [
        "one",
        "two-compatible",
    ]


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("dataset", "source_columns"),
        ("dataset", "storage_option_keys"),
        ("configuration", "reference_manifest_sha256"),
        ("configuration", "payload_read_mode"),
        ("configuration", "max_pending_fetch_batches"),
        ("configuration", "copy_index_to_node_local"),
        ("configuration", "validate_payload_keys"),
        ("configuration", "reference_storage_option_keys"),
        ("environment", "packages"),
    ],
)
def test_missing_identity_field_is_measured_but_comparison_ineligible(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    baseline = _measurements(tmp_path, "one", gpus=1)
    fixture = report._self_test_fixture(scale=2.0)
    fixture[section].pop(field)
    path = tmp_path / f"missing-{field}.json"
    labeled = report.parse_labeled_path(f"missing-{field}[nodes=1,gpus=2]={path}")
    candidate = report.validate_harness_result(labeled, fixture, path, "candidate").measurements

    assert candidate[0].comparison_eligibility_errors
    assert report.strong_scaling((*baseline, *candidate)) == []
    aggregate = report.build_report(
        [
            report.validate_harness_result(
                report.parse_labeled_path(f"one-copy[nodes=1,gpus=1]={tmp_path / 'one-copy.json'}"),
                report._self_test_fixture(),
                tmp_path / "one-copy.json",
                "one-copy",
            ),
            report.validate_harness_result(labeled, fixture, path, "candidate"),
        ],
        [],
        generated_at="test",
    )
    assert aggregate["measured"]["comparison_exclusions"]


def test_harness_requires_at_least_two_repeats(tmp_path: Path) -> None:
    fixture = report._self_test_fixture()
    fixture["configuration"]["repeat_count"] = 1
    for arm in fixture["arms"].values():
        arm["repeats"] = arm["repeats"][:1]
    labeled = report.parse_labeled_path(f"single[nodes=1,gpus=1]={tmp_path / 'single.json'}")

    with pytest.raises(report.ReportInputError, match="at least two repeats"):
        report.validate_harness_result(labeled, fixture, tmp_path / "single.json", "single")


def _write_terminal_result(
    root: Path,
    evidence_class: str | None,
    *,
    benchmark_waves: int | None = None,
    schema_version: int = 2,
) -> report.LabeledPath:
    root.mkdir()
    benchmark_path = root / "benchmark.json"
    benchmark = report._self_test_fixture()
    arms = benchmark["arms"]
    assert isinstance(arms, dict)
    arms["lance_ray_gpu_actor"] = arms.pop("gpu_lance_column_fetch_stage")
    if schema_version == 2:
        benchmark["evidence_class"] = (
            "primary_saturation" if benchmark_waves in report._PRIMARY_SATURATION_WAVES else "locality_sensitivity"
        )
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    run_identity_path = root / "run_identity.json"
    run_identity_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    telemetry_path = root / "telemetry_validation.json"
    telemetry_path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    eligibility = {
        "schema_version": schema_version,
        "artifact_kind": "gpu_lance_saturation_terminal_eligibility",
        "terminal": True,
        "status": "eligible",
        "failures": [],
        "identity_validation": {"status": "passed", "failures": []},
        "telemetry_validation_status": "passed",
        "policy": {
            "minimum_repeat_count": 2,
            "requires_benchmark_validation": True,
            "requires_run_identity_validation": True,
            "requires_telemetry_validation": True,
            "telemetry_pass_is_not_benchmark_eligibility": True,
        },
        "artifacts": {
            "benchmark": {"path": benchmark_path.name, "sha256": report._file_sha256(benchmark_path)},
            "run_identity": {
                "path": run_identity_path.name,
                "sha256": report._file_sha256(run_identity_path),
            },
            "telemetry_validation": {
                "path": telemetry_path.name,
                "sha256": report._file_sha256(telemetry_path),
            },
        },
    }
    if evidence_class is not None:
        eligibility["evidence_class"] = evidence_class
    if benchmark_waves is not None:
        eligibility["benchmark_validation"] = {
            "status": "passed",
            "failures": [],
            "waves": benchmark_waves,
        }
        if schema_version == 2:
            eligibility["benchmark_validation"]["evidence_class"] = evidence_class
    if schema_version == 2:
        eligibility["policy"].update(
            {
                "evidence_class": evidence_class,
                "primary_saturation_waves": [4, 8],
                "locality_sensitivity_waves": [1, 2],
            }
        )
    (root / "eligibility.json").write_text(json.dumps(eligibility), encoding="utf-8")
    return report.parse_labeled_path(f"terminal[nodes=1,gpus=1]={benchmark_path}")


def test_scaling_report_accepts_only_primary_saturation_terminal_evidence(tmp_path: Path) -> None:
    accepted = report.load_result(
        _write_terminal_result(tmp_path / "primary", "primary_saturation", benchmark_waves=8)
    )
    legacy = report.load_result(
        _write_terminal_result(tmp_path / "legacy-primary", None, benchmark_waves=8, schema_version=1)
    )

    assert accepted.measurements
    assert legacy.measurements

    rejected = (
        ("locality", "locality_sensitivity", 2, 2),
        ("misclassified-locality", "primary_saturation", 2, 2),
        ("legacy-locality", None, 2, 1),
        ("current-missing", None, 8, 2),
        ("missing", None, None, 2),
    )
    for name, evidence_class, benchmark_waves, schema_version in rejected:
        with pytest.raises(report.ReportInputError):
            report.load_result(
                _write_terminal_result(
                    tmp_path / name,
                    evidence_class,
                    benchmark_waves=benchmark_waves,
                    schema_version=schema_version,
                )
            )


def test_scaling_report_rejects_orphan_one_wave_actor_benchmark(tmp_path: Path) -> None:
    root = tmp_path / "orphan"
    root.mkdir()
    benchmark = report._self_test_fixture()
    arms = benchmark["arms"]
    assert isinstance(arms, dict)
    arms["lance_ray_gpu_actor"] = arms.pop("gpu_lance_column_fetch_stage")
    benchmark["evidence_class"] = "locality_sensitivity"
    path = root / "benchmark.json"
    path.write_text(json.dumps(benchmark), encoding="utf-8")

    with pytest.raises(report.ReportInputError, match="missing required files"):
        report.load_result(report.parse_labeled_path(f"orphan[nodes=1,gpus=1]={path}"))


def test_scaling_report_accepts_intrinsically_identified_actor_rank(tmp_path: Path) -> None:
    path = tmp_path / "lance_ray_gpu_actor/1_nodes_1_ranks/123/rank_00.json"
    path.parent.mkdir(parents=True)
    benchmark = report._self_test_fixture()
    arms = benchmark["arms"]
    assert isinstance(arms, dict)
    arms["lance_ray_gpu_actor"] = arms.pop("gpu_lance_column_fetch_stage")
    benchmark["evidence_class"] = "scaling_rank"
    benchmark["run_identity"] = {"rank_id": 0, "rank_count": 1, "slurm_job_id": "123"}
    path.write_text(json.dumps(benchmark), encoding="utf-8")
    labeled = report.parse_labeled_path(f"rank[nodes=1,gpus=1,ranks=1,backend=lance_ray_gpu_actor]={path}")

    accepted = report.load_result(labeled)

    assert accepted.rank_id == 0
    assert accepted.measurements[0].backend == "lance_ray_gpu_actor"


def test_scaling_report_rejects_terminal_family_digest_tampering(tmp_path: Path) -> None:
    labeled = _write_terminal_result(tmp_path / "tampered", "primary_saturation", benchmark_waves=8)
    telemetry = tmp_path / "tampered/telemetry_validation.json"
    telemetry.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    with pytest.raises(report.ReportInputError, match="telemetry_validation identity"):
        report.load_result(labeled)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("artifact_kind",), "other"),
        (("failures",), ["dummy failure"]),
        (("benchmark_validation", "status"), "failed"),
        (("identity_validation", "status"), "failed"),
        (("telemetry_validation_status",), "failed"),
        (("policy", "requires_run_identity_validation"), False),
    ],
)
def test_scaling_report_rejects_incomplete_terminal_verdict(
    tmp_path: Path, field_path: tuple[str, ...], value: object
) -> None:
    root = tmp_path / "terminal"
    labeled = _write_terminal_result(root, "primary_saturation", benchmark_waves=8)
    path = root / "eligibility.json"
    eligibility = json.loads(path.read_text(encoding="utf-8"))
    target = eligibility
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    path.write_text(json.dumps(eligibility), encoding="utf-8")

    with pytest.raises(report.ReportInputError):
        report.load_result(labeled)


def test_multi_rank_requires_exact_unique_rank_ids_and_common_slurm_run(tmp_path: Path) -> None:
    labeled = report.parse_labeled_path(
        f"cluster[nodes=2,gpus=2,ranks=2,backend=gpu_lance_column_fetch_stage]={tmp_path}/rank-*.json"
    )

    def accepted(job: str, rank_id: int) -> report.AcceptedResult:
        source = tmp_path / job / f"rank_{rank_id:02d}.json"
        fixture = report._self_test_fixture()
        return report.validate_harness_result(labeled, fixture, source, f"sha-{job}-{rank_id}")

    combined = report._combine_rank_results(labeled, [accepted("123", 0), accepted("123", 1)])
    assert combined.measurements[0].rank_ids == (0, 1)
    assert combined.slurm_run_id == "123"

    with pytest.raises(report.ReportInputError, match="duplicate rank ids"):
        report._combine_rank_results(labeled, [accepted("123", 0), accepted("123", 0)])
    with pytest.raises(report.ReportInputError, match="one complete Slurm run identity"):
        report._combine_rank_results(labeled, [accepted("123", 0), accepted("124", 1)])


def test_multi_rank_explicit_count_rejects_partial_set(tmp_path: Path) -> None:
    labeled = report.parse_labeled_path(f"cluster[nodes=2,gpus=2,ranks=2]={tmp_path / '123/rank_00.json'}")
    rank = report.validate_harness_result(
        labeled,
        report._self_test_fixture(),
        tmp_path / "123/rank_00.json",
        "rank-0",
    )

    with pytest.raises(report.ReportInputError, match="expected exactly 2"):
        report._combine_rank_results(labeled, [rank])
