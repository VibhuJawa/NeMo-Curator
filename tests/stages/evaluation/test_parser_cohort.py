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

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@lru_cache
def _cohort_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3] / "tutorials" / "text" / "llm-as-a-judge" / "build_html_parser_cohort.py"
    )
    spec = importlib.util.spec_from_file_location("build_html_parser_cohort_test", path)
    if spec is None or spec.loader is None:
        msg = f"Could not load tutorial module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(path: Path) -> None:
    pd.DataFrame(
        [
            {"url": "empty", "text": "", "justext_extracted_text": ""},
            {"url": "mineru-only", "text": "m" * 300, "justext_extracted_text": ""},
            {"url": "justext-only", "text": "", "justext_extracted_text": "j" * 300},
            {"url": "short", "text": "m" * 20, "justext_extracted_text": "j" * 30},
            {"url": "similar", "text": "m" * 300, "justext_extracted_text": "j" * 290},
            {"url": "mineru-long", "text": "m" * 500, "justext_extracted_text": "j" * 200},
            {"url": "justext-long", "text": "m" * 200, "justext_extracted_text": "j" * 500},
            {"url": "moderate", "text": "m" * 300, "justext_extracted_text": "j" * 450},
        ]
    ).to_parquet(path, index=False)


def test_cohort_covers_parser_behavior_strata_deterministically(tmp_path: Path) -> None:
    module = _cohort_module()
    source = tmp_path / "source.parquet"
    _write_fixture(source)

    first, populations = module.build_cohort([source], rows_per_stratum=1, batch_size=3)
    second, _ = module.build_cohort([source], rows_per_stratum=1, batch_size=5)

    expected = {
        "both_empty",
        "mineru_html_only",
        "justext_only",
        "both_short",
        "both_similar_length",
        "mineru_html_much_longer",
        "justext_much_longer",
        "both_moderate_difference",
    }
    assert set(first["parser_comparison_stratum"]) == expected
    assert populations == dict.fromkeys(expected, 1)
    assert first[["url", "_eval_source_row"]].to_dict("records") == second[["url", "_eval_source_row"]].to_dict(
        "records"
    )


def test_cohort_keeps_smallest_stable_priorities_per_stratum(tmp_path: Path) -> None:
    module = _cohort_module()
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        [{"url": f"u-{index}", "text": "m" * 300, "justext_extracted_text": "j" * 300} for index in range(20)]
    ).to_parquet(source, index=False)

    cohort, populations = module.build_cohort([source], rows_per_stratum=3, batch_size=4)

    assert len(cohort) == 3
    assert populations["both_similar_length"] == 20
    assert cohort["_eval_stable_priority"].is_monotonic_increasing


def test_manifest_is_written_atomically_next_to_cohort(tmp_path: Path) -> None:
    module = _cohort_module()
    output = tmp_path / "cohort.parquet"
    output.touch()

    manifest_path = module.write_manifest({"schema_version": "test_v1", "selected_rows": 3}, output)

    assert manifest_path == tmp_path / "cohort.parquet.manifest.json"
    assert json.loads(manifest_path.read_text()) == {"schema_version": "test_v1", "selected_rows": 3}


def test_population_sample_is_exact_weighted_and_batch_independent(tmp_path: Path) -> None:
    module = _cohort_module()
    files = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    for file_index, source in enumerate(files):
        pd.DataFrame(
            [
                {
                    "url": f"{file_index}-{row}",
                    "text": "m" * (250 + row),
                    "justext_extracted_text": "j" * (240 + row),
                }
                for row in range(10)
            ]
        ).to_parquet(source, index=False)

    first, total_rows = module.build_population_sample(files, 5, batch_size=3, seed=19)
    second, _ = module.build_population_sample(files, 5, batch_size=7, seed=19)

    assert total_rows == 20
    assert len(first) == 5
    assert first["_eval_inclusion_probability"].unique().tolist() == [0.25]
    assert first["_eval_sample_weight"].unique().tolist() == [4.0]
    assert first[["_eval_source_file", "_eval_source_row"]].to_dict("records") == second[
        ["_eval_source_file", "_eval_source_row"]
    ].to_dict("records")


def test_population_sample_rejects_target_larger_than_snapshot(tmp_path: Path) -> None:
    module = _cohort_module()
    source = tmp_path / "source.parquet"
    _write_fixture(source)

    with pytest.raises(ValueError, match="exceeds"):
        module.build_population_sample([source], 9, batch_size=2)
