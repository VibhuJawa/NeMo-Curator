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
