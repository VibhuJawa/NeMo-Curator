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
from pathlib import Path

import pandas as pd


def module():
    path = Path(__file__).resolve().parents[3] / "tutorials/text/llm-as-a-judge/build_html_parser_cohort.py"
    spec = importlib.util.spec_from_file_location("cohort", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_balanced_and_population_cohorts(tmp_path: Path) -> None:
    mod, source = module(), tmp_path / "data.parquet"
    rows = [
        {"url": f"u{i}", "text": "m" * left, "justext_extracted_text": "j" * right}
        for i, (left, right) in enumerate(
            [(0, 0), (300, 0), (0, 300), (20, 30), (300, 290), (500, 200), (200, 500), (300, 450)] * 3
        )
    ]
    pd.DataFrame(rows).to_parquet(source, index=False)
    balanced, populations = mod.stratified([source], 1, 3)
    sample_a, total = mod.population([source], 5, 3, 17)
    sample_b, _ = mod.population([source], 5, 7, 17)
    assert len(balanced) == len(populations) == 8
    assert total == 24
    assert sample_a["_eval_sample_weight"].sum() == 24
    assert sample_a["_eval_source_row"].tolist() == sample_b["_eval_source_row"].tolist()
