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

"""Summarize one NVMe/Lustre/S3 GPU image-format benchmark session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ENTRY_PREFIX = "gpu_image_table_formats_"
STORAGE_ORDER = ("nvme", "lustre", "s3")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def load_matrix(session_root: Path) -> dict[str, Any]:
    """Load and validate all storage entries in a runner session."""

    storages: dict[str, Any] = {}
    for storage in STORAGE_ORDER:
        root = session_root / f"{ENTRY_PREFIX}{storage}"
        params = _read_json(root / "params.json")
        metrics = _read_json(root / "metrics.json")
        trials = _read_json(root / "trials.json")
        fractions = _read_json(root / "fraction_summaries.json")
        indexes = sorted(int(trial["actor_task_index"]) for trial in trials)
        expected_indexes = list(range(2, 2 + len(trials)))
        if indexes != expected_indexes:
            message = f"{storage} measured actor indexes are {indexes}; expected {expected_indexes}"
            raise ValueError(message)
        if not metrics["is_success"] or not metrics["output_parity_valid"]:
            message = f"{storage} benchmark or output parity failed"
            raise ValueError(message)
        storages[storage] = {
            "params": params,
            "runtime": {
                "runtime_setup_s": metrics["runtime_setup_s"],
                "warmup_end_to_end_s_total": metrics["warmup_end_to_end_s_total"],
                "measured_end_to_end_s_total": metrics["measured_end_to_end_s_total"],
                "pipeline_wall_s": metrics["pipeline_wall_s"],
                "pipeline_non_task_overhead_s": metrics["pipeline_non_task_overhead_s"],
                "measured_actor_task_indexes": indexes,
            },
            "fractions": fractions,
        }
    return {"session": session_root.name, "storages": storages}


def _spread(summary: dict[str, Any], arm: str) -> str:
    mean = summary[f"{arm}_end_to_end_s_mean"]
    low = summary[f"{arm}_end_to_end_s_min"]
    high = summary[f"{arm}_end_to_end_s_max"]
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def render_markdown(matrix: dict[str, Any]) -> str:
    """Render the comparison and initialization audit as Markdown."""

    lines = [
        f"# GPU image table format matrix: `{matrix['session']}`",
        "",
        "Measured task time excludes actor setup and both warmup tasks. Values are mean [min, max] seconds.",
        "",
        "| Storage | Fraction | Rows | Parquet e2e | Lance e2e | Lance/Parquet | Parquet read | Lance read |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for storage in STORAGE_ORDER:
        for summary in matrix["storages"][storage]["fractions"].values():
            lines.append(
                "| "
                + " | ".join(
                    (
                        storage,
                        f"{summary['sample_fraction']:.0%}",
                        str(summary["sample_rows"]),
                        _spread(summary, "parquet"),
                        _spread(summary, "lance"),
                        f"{summary['lance_over_parquet_end_to_end_ratio']:.3f}",
                        f"{summary['parquet_source_read_s_mean']:.3f}",
                        f"{summary['lance_source_read_s_mean']:.3f}",
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## Initialization audit",
            "",
            "| Storage | Runtime setup (excluded) | Warmups (excluded) | Measured tasks | Actor index range | Pipeline wall |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for storage in STORAGE_ORDER:
        runtime = matrix["storages"][storage]["runtime"]
        indexes = runtime["measured_actor_task_indexes"]
        lines.append(
            f"| {storage} | {runtime['runtime_setup_s']:.3f} | "
            f"{runtime['warmup_end_to_end_s_total']:.3f} | {len(indexes)} | "
            f"{indexes[0]}-{indexes[-1]} | {runtime['pipeline_wall_s']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    matrix = load_matrix(args.session_root)
    output_dir = args.output_dir or args.session_root
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "storage_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    (output_dir / "storage_matrix.md").write_text(render_markdown(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
