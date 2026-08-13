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
"""MinerU-HTML versus jusText judge configuration."""

from typing import Any

from eval.text.llm_judge import JudgeCriterion, PairwiseLLMJudgeStage

HTML_PARSER_CRITERIA = (
    JudgeCriterion("content_recall", "Retains the page's substantive main content without omissions."),
    JudgeCriterion("content_precision", "Excludes navigation, ads, boilerplate, and unrelated page chrome."),
    JudgeCriterion("structure", "Preserves readable section, list, table, code, and paragraph structure."),
    JudgeCriterion("coherence", "Produces coherent text in the correct reading order without extraction artifacts."),
)


def create_html_parser_judge(
    model_name: str,
    *,
    model_configs: list[Any] | None = None,
    model_providers: list[Any] | None = None,
    max_candidate_chars: int = 12000,
) -> PairwiseLLMJudgeStage:
    """Create the standard bidirectional MinerU-HTML/jusText judge."""
    return PairwiseLLMJudgeStage(
        model_name=model_name,
        model_configs=model_configs,
        model_providers=model_providers,
        left_field="text",
        right_field="justext_extracted_text",
        left_label="MinerU-HTML",
        right_label="jusText",
        context_fields=("url",),
        criteria=HTML_PARSER_CRITERIA,
        output_prefix="html_parser_judge",
        max_candidate_chars=max_candidate_chars,
    )
