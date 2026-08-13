# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("data_designer.config")
from data_designer.config.preview_results import PreviewResults

from nemo_curator.tasks import DocumentBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tutorials/text/llm-as-a-judge"))
from pretraining_readiness import (
    DEFAULT_PRETRAINING_TAXONOMY,
    DEFAULT_QUALITY_CRITERIA,
    PretrainingReadinessLLMJudgeStage,
)


def payload() -> dict:
    return {
        "taxonomy_version": "pretraining_phase2_v2",
        "primary_topic": "clinical_medicine",
        "secondary_topics": ["biology"],
        "topic_confidence": 0.9,
        "content_form": "explanatory_article",
        "training_value_tags": ["domain_expert_knowledge"],
        "phase2_bucket": "specialized_domain",
        "language": {
            "primary": {
                "iso639_3": "eng",
                "script_iso15924": "Latn",
                "bcp47": "en",
                "estimated_share": 1.0,
                "confidence": 0.9,
            },
            "others": [],
            "multilingual_mode": "monolingual",
            "register": "technical",
        },
        "quality_scores": dict.fromkeys((criterion.name for criterion in DEFAULT_QUALITY_CRITERIA), 4),
        "knowledge_depth": "advanced",
        "reasoning_density": "medium",
        "temporal_profile": "dated_but_stable",
        "quality_flags": ["none"],
        "risk_flags": ["medical_high_stakes"],
        "phase2_action": "include",
        "action_confidence": 0.8,
        "rationale": "substantive clinical material",
    }


def run(value: dict, text: str = "medical article", max_chars: int = 24000) -> pd.Series:
    judge = PretrainingReadinessLLMJudgeStage(
        model_name="test/model", output_prefix="pretrain", max_document_chars=max_chars
    )
    generated = pd.DataFrame([{"__pretrain_row_id": 0, "pretrain_structured": value}])
    judge.data_designer.preview = MagicMock(
        return_value=PreviewResults(config_builder=judge.config_builder, dataset=generated)
    )
    return judge.process(DocumentBatch(dataset_name="cc", data=pd.DataFrame([{"text": text}]))).to_pandas().iloc[0]


def test_structured_phase2_quality_language_and_context() -> None:
    judge = PretrainingReadinessLLMJudgeStage(model_name="test/model")
    assert judge.config_builder.build().columns[0].column_type == "llm-structured"
    row = run(payload(), "x" * 100, 30)
    assert len(DEFAULT_PRETRAINING_TAXONOMY.topic_to_family) == 65
    assert row["pretrain_topic_family"] == "medicine_health"
    assert row["pretrain_quality_score"] == 4
    assert row["pretrain_quality_tier"] == "medium_high"
    assert json.loads(row["pretrain_language"])["primary"]["iso639_3"] == "eng"
    assert "model_token_limit_status=unverified" in row["pretrain_context_issue"]


def test_recoverable_output_is_normalized_and_semantic_error_is_row_scoped() -> None:
    value = deepcopy(payload())
    value["secondary_topics"] = [value["primary_topic"], "biology"]
    value["quality_flags"] = ["none", "truncated"]
    row = run(value)
    assert json.loads(row["pretrain_secondary_topics"]) == ["biology"]
    assert json.loads(row["pretrain_quality_flags"]) == ["truncated"]

    value["language"]["others"] = [{"iso639_3": "spa", "script_iso15924": "Latn", "estimated_share": 0.2}]
    row = run(value)
    assert "language shares exceed one" in row["pretrain_error"]
    assert row["pretrain_primary_topic"] is None
