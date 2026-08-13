# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("data_designer.config")
from data_designer.config.preview_results import PreviewResults

from eval.text.llm_judge import JudgeCriterion, PairwiseLLMJudgeStage
from nemo_curator.tasks import DocumentBatch


def stage(max_chars: int = 12000) -> PairwiseLLMJudgeStage:
    return PairwiseLLMJudgeStage(
        model_name="test/model",
        left_field="text",
        right_field="justext",
        left_label="MinerU-HTML",
        right_label="jusText",
        criteria=[JudgeCriterion("quality", "overall extraction quality")],
        output_prefix="parser",
        max_candidate_chars=max_chars,
    )


def result(score: str, reasoning: str = "evidence") -> dict:
    return {"quality": {"score": score, "reasoning": reasoning}}


def run(judge: PairwiseLLMJudgeStage, generated: pd.DataFrame, left: str = "main") -> pd.Series:
    judge.data_designer.preview = MagicMock(
        return_value=PreviewResults(config_builder=judge.config_builder, dataset=generated)
    )
    batch = DocumentBatch(dataset_name="cc", data=pd.DataFrame([{"text": left, "justext": "other"}]))
    return judge.process(batch).to_pandas().iloc[0]


def test_uses_structured_judge_bidirectionally_and_maps_labels() -> None:
    judge = stage()
    columns = judge.config_builder.build().columns
    assert [column.column_type for column in columns] == ["llm-structured", "llm-structured"]
    assert columns[0].output_format["properties"]["quality"]["properties"]["reasoning"]["maxLength"] == 220
    generated = pd.DataFrame([{"__parser_row_id": 0, "parser_ab": result("A"), "parser_ba": result("B")}])
    row = run(judge, generated)
    assert row["parser_winner"] == "MinerU-HTML"
    assert json.loads(row["parser_directional_winners"]) == ["MinerU-HTML", "MinerU-HTML"]
    assert bool(row["parser_order_consistent"])
    assert row["parser_error"] is None


def test_order_disagreement_context_overflow_and_dropped_row() -> None:
    judge = stage(20)
    generated = pd.DataFrame([{"__parser_row_id": 0, "parser_ab": result("A"), "parser_ba": result("A")}])
    row = run(judge, generated, "x" * 100)
    assert row["parser_winner"] == "order_sensitive"
    assert "original_chars=100" in row["parser_context_issue"]
    assert "model_token_limit_status=unverified" in row["parser_context_issue"]
    dropped = run(stage(), pd.DataFrame())
    assert dropped["parser_error"] == "data_designer_generation_failed_or_dropped"
