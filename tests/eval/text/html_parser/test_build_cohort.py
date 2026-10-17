# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pandas as pd
import pytest

from eval.text.html_parser import build_cohort
from eval.text.html_parser.judge import create_html_parser_judge


def test_balanced_and_population_cohorts(tmp_path: Path) -> None:
    source = tmp_path / "data.parquet"
    rows = [
        {"url": f"u{i}", "text": "m" * left, "justext_extracted_text": "j" * right}
        for i, (left, right) in enumerate(
            [(0, 0), (300, 0), (0, 300), (20, 30), (300, 290), (500, 200), (200, 500), (300, 450)] * 3
        )
    ]
    pd.DataFrame(rows).to_parquet(source, index=False)
    fragments = build_cohort._fragments(source)
    balanced, populations = build_cohort.stratified(fragments, 1, 3)
    sample_a, total = build_cohort.population(fragments, 5, 3, 17)
    sample_b, _ = build_cohort.population(fragments, 5, 7, 17)
    assert len(balanced) == len(populations) == 8
    assert total == 24
    assert sample_a["_eval_sample_weight"].sum() == 24
    assert sample_a["_eval_source_row"].tolist() == sample_b["_eval_source_row"].tolist()
    pd.DataFrame({"text": ["missing fields"]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="missing"):
        build_cohort._fragments(tmp_path / "bad.parquet")

    empty = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["url", "text", "justext_extracted_text"]).to_parquet(empty)
    empty_fragments = build_cohort._fragments(empty)
    with pytest.raises(ValueError, match="contains no rows"):
        build_cohort.stratified(empty_fragments, 1, 3)
    with pytest.raises(ValueError, match="contains no rows"):
        build_cohort.population(empty_fragments, 1, 3, 17)


def test_standard_judge_is_bidirectional() -> None:
    judge = create_html_parser_judge("test/model")
    assert judge.inputs()[1] == ["text", "justext_extracted_text", "url"]
    assert len(judge.config_builder.build().columns) == 2
