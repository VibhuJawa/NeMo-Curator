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

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarking" / "scripts"))

from summarize_gpu_image_table_format_matrix import load_matrix, render_markdown


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_matrix_summary_validates_persistent_actor_and_renders_table(tmp_path: Path) -> None:
    summary = {
        "sample_fraction": 0.1,
        "sample_rows": 10,
        "parquet_end_to_end_s_mean": 1.0,
        "parquet_end_to_end_s_min": 0.9,
        "parquet_end_to_end_s_max": 1.1,
        "lance_end_to_end_s_mean": 0.8,
        "lance_end_to_end_s_min": 0.7,
        "lance_end_to_end_s_max": 0.9,
        "lance_over_parquet_end_to_end_ratio": 0.8,
        "parquet_source_read_s_mean": 0.2,
        "lance_source_read_s_mean": 0.1,
    }
    for storage in ("nvme", "lustre", "s3"):
        root = tmp_path / f"gpu_image_table_formats_{storage}"
        root.mkdir()
        _write_json(root / "params.json", {"storage_label": storage})
        _write_json(
            root / "metrics.json",
            {
                "is_success": True,
                "output_parity_valid": True,
                "runtime_setup_s": 2.0,
                "warmup_end_to_end_s_total": 3.0,
                "measured_end_to_end_s_total": 5.0,
                "pipeline_wall_s": 11.0,
                "pipeline_non_task_overhead_s": 3.0,
            },
        )
        _write_json(root / "trials.json", [{"actor_task_index": 2}, {"actor_task_index": 3}])
        _write_json(root / "fraction_summaries.json", {"010pct": summary})

    matrix = load_matrix(tmp_path)
    markdown = render_markdown(matrix)

    assert matrix["storages"]["nvme"]["runtime"]["measured_actor_task_indexes"] == [2, 3]
    assert "| nvme | 10% | 10 | 1.000 [0.900, 1.100] |" in markdown
    assert "Runtime setup (excluded)" in markdown
