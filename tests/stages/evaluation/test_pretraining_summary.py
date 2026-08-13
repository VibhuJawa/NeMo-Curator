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

import pandas as pd
import pytest

from nemo_curator.stages.evaluation import PretrainingJudgeSummary


def _row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "pretrain_judge_taxonomy_version": "pretraining_phase2_v2",
        "pretrain_judge_topic_family": "mathematics_formal",
        "pretrain_judge_primary_topic": "pure_mathematics",
        "pretrain_judge_secondary_topics": json.dumps(["physics"]),
        "pretrain_judge_topic_confidence": 0.9,
        "pretrain_judge_content_form": "problem_solution",
        "pretrain_judge_training_value_tags": json.dumps(["mathematical_content", "analytical_reasoning"]),
        "pretrain_judge_phase2_bucket": "math",
        "pretrain_judge_language_code": "eng",
        "pretrain_judge_language_script": "Latn",
        "pretrain_judge_language_bcp47": "en",
        "pretrain_judge_language_share": 1.0,
        "pretrain_judge_other_language_codes": "[]",
        "pretrain_judge_multilingual": False,
        "pretrain_judge_multilingual_mode": "monolingual",
        "pretrain_judge_language_register": "technical",
        "pretrain_judge_locale_region": None,
        "pretrain_judge_language_confidence": 0.99,
        "pretrain_judge_action_confidence": 0.8,
        "pretrain_judge_quality_score": 4.2,
        "pretrain_judge_quality_scores": json.dumps({"clarity": 4, "transformability": 5}),
        "pretrain_judge_quality_tier": "medium_high",
        "pretrain_judge_knowledge_depth": "advanced",
        "pretrain_judge_reasoning_density": "high",
        "pretrain_judge_temporal_profile": "timeless",
        "pretrain_judge_quality_flags": json.dumps(["none"]),
        "pretrain_judge_risk_flags": json.dumps(["none"]),
        "pretrain_judge_phase2_action": "upweight",
        "pretrain_judge_original_chars": 1200,
        "pretrain_judge_judged_chars": 1200,
        "pretrain_judge_partial_document": False,
        "pretrain_judge_judge_view_strategy": "full",
        "pretrain_judge_error": None,
    }
    row.update(overrides)
    return row


def _distribution(report: dict, name: str) -> dict[str, dict]:
    return {entry["label"]: entry for entry in report["distributions"][name]}


def test_summary_aggregates_categories_quality_confidence_and_errors() -> None:
    summary = PretrainingJudgeSummary()
    summary.update(
        pd.DataFrame(
            [
                _row(),
                _row(
                    pretrain_judge_primary_topic="programming_software_engineering",
                    pretrain_judge_topic_family="code_computing",
                    pretrain_judge_secondary_topics="[]",
                    pretrain_judge_training_value_tags=json.dumps(["code_and_algorithms", "analytical_reasoning"]),
                    pretrain_judge_phase2_bucket="code",
                    pretrain_judge_topic_confidence=0.7,
                    pretrain_judge_action_confidence=0.6,
                    pretrain_judge_quality_score=3.2,
                    pretrain_judge_quality_scores=json.dumps({"clarity": 3, "transformability": 4}),
                    pretrain_judge_quality_tier="medium",
                    pretrain_judge_phase2_action="include",
                ),
                _row(
                    **{
                        column: None
                        for column in PretrainingJudgeSummary().required_columns()
                        if column != "pretrain_judge_error"
                    },
                    pretrain_judge_error="parse_failed: ValueError: bad JSON",
                ),
            ]
        )
    )

    report = summary.as_dict()

    assert report["rows"] == {
        "total": 3,
        "successful": 2,
        "failed": 1,
        "success_rate": 0.666667,
        "estimated_total": None,
        "estimated_successful": None,
        "estimated_failed": None,
    }
    topics = _distribution(report, "primary_topic")
    assert topics["pure_mathematics"]["count"] == 1
    assert topics["pure_mathematics"]["share_of_successful_rows"] == 0.5
    values = _distribution(report, "training_value_tags")
    assert values["analytical_reasoning"]["count"] == 2
    assert _distribution(report, "errors")["parse_failed"]["count"] == 1
    assert report["numeric"]["quality_score"]["mean"] == 3.7
    assert report["numeric"]["topic_confidence"]["mean"] == 0.8
    assert report["numeric"]["quality_dimension.clarity"]["mean"] == 3.5
    assert {entry["label"]: entry["count"] for entry in report["quality_bands"]} == {
        "3_to_lt_4": 1,
        "4_to_5": 1,
        "1_to_lt_2": 0,
        "2_to_lt_3": 0,
    }
    assert report["mean_quality_by_primary_topic"] == [
        {
            "label": "programming_software_engineering",
            "count": 1,
            "mean": 3.2,
            "estimated_weight": None,
            "weighted_mean": None,
        },
        {
            "label": "pure_mathematics",
            "count": 1,
            "mean": 4.2,
            "estimated_weight": None,
            "weighted_mean": None,
        },
    ]


def test_summary_merges_batches_and_audits_malformed_nested_json() -> None:
    summary = PretrainingJudgeSummary(output_prefix="judge")
    first = {key.replace("pretrain_judge_", "judge_"): value for key, value in _row().items()}
    second = dict(first)
    second["judge_training_value_tags"] = "not-json"

    summary.update(pd.DataFrame([first]))
    summary.update(pd.DataFrame([second]))
    report = summary.as_dict()

    assert report["rows"]["successful"] == 2
    assert _distribution(report, "primary_topic")["pure_mathematics"]["count"] == 2
    assert report["invalid_structured_values"] == {"training_value_tags": 1}


def test_summary_rejects_incompatible_schema() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        PretrainingJudgeSummary().update(pd.DataFrame({"text": ["x"]}))


def test_summary_uses_inverse_probability_weights_for_snapshot_estimates() -> None:
    summary = PretrainingJudgeSummary(weight_column="_eval_sample_weight")
    summary.update(
        pd.DataFrame(
            [
                _row(_eval_sample_weight=10.0),
                _row(
                    _eval_sample_weight=1.0,
                    pretrain_judge_primary_topic="programming_software_engineering",
                    pretrain_judge_topic_family="code_computing",
                    pretrain_judge_quality_score=3.2,
                ),
                _row(
                    **{
                        column: None
                        for column in PretrainingJudgeSummary().required_columns()
                        if column != "pretrain_judge_error"
                    },
                    _eval_sample_weight=5.0,
                    pretrain_judge_error="request_failed: TimeoutError",
                ),
            ]
        )
    )

    report = summary.as_dict()
    topics = _distribution(report, "primary_topic")

    assert report["rows"]["estimated_total"] == 16.0
    assert report["rows"]["estimated_successful"] == 11.0
    assert report["rows"]["estimated_failed"] == 5.0
    assert topics["pure_mathematics"]["estimated_count"] == 10.0
    assert topics["pure_mathematics"]["estimated_share_of_successful_rows"] == 0.909091
    assert report["weighted_quality"] == {"estimated_weight": 11.0, "weighted_mean": 4.1091}


def test_summary_rejects_invalid_sample_weights() -> None:
    frame = pd.DataFrame([_row(_eval_sample_weight=0.0)])
    with pytest.raises(ValueError, match="finite positive weights"):
        PretrainingJudgeSummary(weight_column="_eval_sample_weight").update(frame)


def test_summary_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PretrainingJudgeSummary(output_prefix=" ")
