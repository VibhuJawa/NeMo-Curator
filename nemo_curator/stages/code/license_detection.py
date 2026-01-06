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

"""License detection stage for source code files.

This stage uses scancode-toolkit to detect software licenses
by matching against 1800+ license templates.
"""

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

# Try to import scancode, but provide fallback
try:
    from licensedcode.detection import detect_licenses

    SCANCODE_AVAILABLE = True
except ImportError:
    SCANCODE_AVAILABLE = False


@dataclass
class LicenseDetectionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Detect software licenses in source code files.

    This stage adds two columns to the data:
    - license_num_licenses: Number of licenses detected
    - license_spans: List of license span information (start/end lines)

    Attributes:
        name: Stage name identifier
        resources: Compute resources required
        batch_size: Number of tasks to process in a batch
        content_window_size: Number of characters to search for licenses
        deadline: Timeout for license detection per document (seconds)
    """

    name: str = "LicenseDetectionStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1
    content_window_size: int = 3000
    deadline: int = 100

    def __post_init__(self) -> None:
        """Initialize the stage and verify dependencies."""
        if not SCANCODE_AVAILABLE:
            msg = "scancode-toolkit package not installed. Please install it using: pip install scancode-toolkit"
            raise ImportError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define required input attributes."""
        return ["data"], ["content"]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output attributes added by this stage."""
        return ["data"], ["license_num_licenses", "license_spans"]

    def _detect_licenses(self, content: str | None) -> dict[str, Any]:
        """Detect licenses in content.

        Args:
            content: Source code content to analyze

        Returns:
            Dictionary with num_licenses and license_spans
        """
        if content is None:
            return {"license_num_licenses": 0, "license_spans": None}

        text_windowed = content[: self.content_window_size]
        detected = list(detect_licenses(query_string=text_windowed, deadline=self.deadline))
        num_licenses = len(detected)

        license_spans = None
        if num_licenses > 0:
            license_spans = []
            for lic in detected:
                if lic.matches:
                    lines = lic.matches[0].lines()
                    license_spans.append({"start": lines[0], "end": lines[1]})

        return {
            "license_num_licenses": num_licenses,
            "license_spans": license_spans,
        }

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a document batch to detect licenses.

        Args:
            task: Input document batch with source code content

        Returns:
            Document batch with license detection columns added
        """
        import polars as pl

        df = task.data
        if isinstance(df, pl.DataFrame):
            # Process each document
            results = []
            content_col = "content"

            for row in df.iter_rows(named=True):
                content = row.get(content_col)
                result = self._detect_licenses(content)
                results.append(result)

            # Create result columns
            result_df = pl.from_dicts(results)

            # Combine with original data
            combined_df = df.hstack(result_df)

            return DocumentBatch(
                task_id=f"{task.task_id}_{self.name}",
                dataset_name=task.dataset_name,
                data=combined_df,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )

        msg = f"Unsupported data type: {type(df)}"
        raise TypeError(msg)

    def get_config(self) -> dict[str, Any]:
        """Get configuration for this stage."""
        config = super().get_config()
        config["content_window_size"] = self.content_window_size
        config["deadline"] = self.deadline
        config["scancode_available"] = SCANCODE_AVAILABLE
        return config
