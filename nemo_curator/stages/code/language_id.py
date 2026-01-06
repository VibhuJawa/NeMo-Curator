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

"""Language identification stage for code files.

This stage uses hyperpolyglot to detect programming languages from:
1. Filename (exact matches like Makefile)
2. Extension (like .py -> Python)
3. Shebang (like #!/usr/bin/python)
4. Heuristics (content-based patterns)
5. Classifier (Naive Bayes for ambiguous cases)
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
class LanguageIdentificationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Detect programming languages in source code files.

    This stage adds two columns to the data:
    - language: The detected programming language name
    - language_detector: The detection method used (Filename, Extension, Shebang, Heuristics, Classifier)

    Attributes:
        name: Stage name identifier
        resources: Compute resources required
        batch_size: Number of tasks to process in a batch
    """

    name: str = "LanguageIdentificationStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

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
        return ["data"], ["language", "language_detector"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a document batch to detect languages.

        Args:
            task: Input document batch with compressed source code

        Returns:
            Document batch with language and language_detector columns added
        """
        df = task.data

        # Extract code and filenames as lists of strings
        codes: list[str] = df["compressed_content"].tolist()
        filenames: list[str] = df["representative_filename"].tolist()

        # Run the Rust-based language detection
        lang_results = code_annotation.compute_language_detection(codes, filenames)

        # Extract results into separate lists
        languages = [r["language"] for r in lang_results]
        detectors = [r["language_detector"] for r in lang_results]

        # Create new DataFrame with added columns
        result_df = df.assign(
            language=languages,
            language_detector=detectors,
        )

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
        config["code_annotation_available"] = CODE_ANNOTATION_AVAILABLE
        return config
