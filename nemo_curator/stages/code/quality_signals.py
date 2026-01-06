# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Code quality signals stage based on OpenCoder methodology.

This stage computes various quality signals for code data including:
- Comment line/character fractions (Rust implementation)
- Tokenization with custom vocabularies
- Decontamination via n-gram matching
"""

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

# Try to import code_annotation, but provide fallback
try:
    import code_annotation

    CODE_ANNOTATION_AVAILABLE = True
except ImportError:
    CODE_ANNOTATION_AVAILABLE = False


@dataclass
class CodeQualitySignalsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Compute code quality signals including comment fractions.

    This stage adds quality signal columns including:
    - ors_comment_lines_frac: Fraction of lines that are comments
    - ors_comment_chars_frac: Fraction of characters in comments

    Optionally adds tokenization and decontamination results.

    Attributes:
        name: Stage name identifier
        resources: Compute resources required
        batch_size: Number of tasks to process in a batch
        include_tokenization: Whether to include tokenization
        tokenizer_name: Name of tokenizer to use
        include_decontamination: Whether to check for benchmark contamination
        decontamination_ngrams: N-grams to check for contamination
        decontamination_ngram_order: Order of n-grams for matching
    """

    name: str = "CodeQualitySignalsStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1
    include_tokenization: bool = False
    tokenizer_name: str = "github_o200k_base"
    include_decontamination: bool = False
    decontamination_ngrams: dict[str, list[str]] = field(default_factory=dict)
    decontamination_ngram_order: int = 13

    def __post_init__(self) -> None:
        """Initialize the stage and verify dependencies."""
        if not CODE_ANNOTATION_AVAILABLE:
            msg = (
                "code_annotation package not installed. "
                "Please build and install it using: ./scripts/build_code_annotation.sh --install"
            )
            raise ImportError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define required input attributes."""
        return ["data"], ["code_content", "language"]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output attributes added by this stage."""
        output_cols = ["ors_comment_lines_frac", "ors_comment_chars_frac"]

        if self.include_tokenization:
            output_cols.extend(
                [
                    "tokenized_content",
                    f"num_tokens_{self.tokenizer_name}",
                ]
            )

        if self.include_decontamination:
            for label in self.decontamination_ngrams:
                output_cols.append(f"{label}_matched_ngrams")

        return ["data"], output_cols

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a document batch to compute quality signals.

        Args:
            task: Input document batch with source code

        Returns:
            Document batch with quality signal columns added
        """
        df = task.data

        # Extract code and languages as lists of strings
        codes: list[str] = df["code_content"].tolist()
        languages: list[str] = df["language"].tolist()

        # Build dict of new columns
        new_columns = {}

        # OpenCoder Rust annotations (comment fractions)
        ors_results = code_annotation.compute_opencoder_rs(codes, languages)
        new_columns["ors_comment_lines_frac"] = [r["comment_lines_frac"] for r in ors_results]
        new_columns["ors_comment_chars_frac"] = [r["comment_chars_frac"] for r in ors_results]

        # Optional tokenization
        if self.include_tokenization:
            token_results = code_annotation.compute_tokenization(
                codes,
                self.tokenizer_name,
            )
            new_columns["tokenized_content"] = [r["tokens"] for r in token_results]
            new_columns[f"num_tokens_{self.tokenizer_name}"] = [r["num_tokens"] for r in token_results]

        # Optional decontamination
        if self.include_decontamination and self.decontamination_ngrams:
            ngram_results = code_annotation.compute_ngram_matches(
                codes,
                self.decontamination_ngrams,
                self.decontamination_ngram_order,
            )
            for label, matches in ngram_results.items():
                new_columns[f"{label}_matched_ngrams"] = matches

        # Create new DataFrame with added columns
        result_df = df.assign(**new_columns)

        return DocumentBatch(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=result_df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def get_config(self) -> dict[str, Any]:
        """Get configuration for this stage."""
        config = super().get_config()
        config["include_tokenization"] = self.include_tokenization
        config["tokenizer_name"] = self.tokenizer_name
        config["include_decontamination"] = self.include_decontamination
        config["decontamination_ngram_order"] = self.decontamination_ngram_order
        return config
