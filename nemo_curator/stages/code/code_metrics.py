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

"""Code metrics stage for computing software quality metrics.

This stage uses Mozilla's rust-code-analysis library to compute:
- Cyclomatic complexity
- Cognitive complexity
- Halstead metrics
- Maintainability Index
- Lines of Code metrics
- OOP metrics (WMC, CDA, COA)
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
class CodeMetricsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Compute software quality metrics for source code.

    This stage adds multiple metric columns to the data including:
    - software_metrics_cyclomatic_complexity
    - software_metrics_cognitive_complexity
    - software_metrics_maintainability_index
    - software_metrics_halstead_difficulty
    - software_metrics_comment_lines
    - software_metrics_blank_lines
    - software_metrics_parsed_ok

    Attributes:
        name: Stage name identifier
        resources: Compute resources required
        batch_size: Number of tasks to process in a batch
        include_basic_annotations: Whether to include basic stats (byte counts, patterns)
        xml_header_search_length: Length to search for XML headers
        max_decompressed_byte_size: Maximum size for decompressed content (None for no limit)
    """

    name: str = "CodeMetricsStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1
    include_basic_annotations: bool = True
    xml_header_search_length: int = 100
    max_decompressed_byte_size: int | None = None

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
        return ["data"], ["compressed_content", "representative_filename"]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output attributes added by this stage."""
        output_cols = [
            "language",
            "language_detector",
            "software_metrics_cyclomatic_complexity",
            "software_metrics_cognitive_complexity",
            "software_metrics_exits_average",
            "software_metrics_maintainability_index",
            "software_metrics_halstead_difficulty",
            "software_metrics_comment_lines",
            "software_metrics_comment_lines_frac",
            "software_metrics_comment_lines_per_space",
            "software_metrics_blank_lines",
            "software_metrics_blank_lines_per_space",
            "software_metrics_args_average",
            "software_metrics_functions_closures_per_space",
            "software_metrics_total_cda",
            "software_metrics_total_wmc",
            "software_metrics_total_coa",
            "software_metrics_parsed_ok",
        ]

        if self.include_basic_annotations:
            output_cols.extend(
                [
                    "basic_num_bytes",
                    "basic_valid_utf8",
                    "basic_max_line_length",
                    "basic_alpha_percent",
                    "basic_alnum_percent",
                    "basic_base64_percent",
                    "basic_hex_percent",
                    "basic_unicode_percent",
                    "basic_num_lines",
                    "basic_average_line_length",
                    "basic_contains_xml_header",
                ]
            )

        return ["data"], output_cols

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a document batch to compute code metrics.

        Args:
            task: Input document batch with compressed source code

        Returns:
            Document batch with code metrics columns added
        """
        import polars as pl

        df = task.data
        if isinstance(df, pl.DataFrame):
            # Build the annotation configuration
            annotations = {}

            if self.include_basic_annotations:
                annotations["basic"] = {
                    "xml_header_search_length": self.xml_header_search_length,
                    "max_decompressed_byte_size": self.max_decompressed_byte_size,
                }

            # Language detection is required for software metrics
            annotations["detect_language"] = {}
            annotations["software_metrics"] = {}

            # Run the Rust-based annotations
            result_df = code_annotation.annotate(annotations, df)

            return DocumentBatch(
                task_id=f"{task.task_id}_{self.name}",
                dataset_name=task.dataset_name,
                data=result_df,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )

        msg = f"Unsupported data type: {type(df)}"
        raise TypeError(msg)

    def get_config(self) -> dict[str, Any]:
        """Get configuration for this stage."""
        config = super().get_config()
        config["include_basic_annotations"] = self.include_basic_annotations
        config["xml_header_search_length"] = self.xml_header_search_length
        config["max_decompressed_byte_size"] = self.max_decompressed_byte_size
        return config
