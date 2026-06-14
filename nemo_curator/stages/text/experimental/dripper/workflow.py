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

"""DripperHTMLWorkflow — end-to-end HTML content extraction pipeline.

Chains GPU-accelerated layout clustering with LLM inference to extract
main content from HTML pages at Common Crawl scale.

Usage::

    workflow = DripperHTMLWorkflow(
        input_path="/lustre/cc_manifest.parquet",
        output_path="/lustre/cc_output/",
        client=my_llm_client,
        model_name="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact",
    )
    result = workflow.run(executor)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.stages.text.experimental.dripper.extraction import DripperHTMLExtractionStage  # noqa: F401
from nemo_curator.stages.text.experimental.dripper.inference import DripperHTMLInferenceStage
from nemo_curator.stages.text.experimental.dripper.preprocessing import (
    DripperHTMLPostprocessStage,
    DripperHTMLPreprocessStage,
)
from nemo_curator.stages.text.experimental.dripper.stage import DripperHTMLLayoutTemplateStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor
    from nemo_curator.models.client.llm_client import AsyncLLMClient
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import Task


@dataclass(kw_only=True)
class DripperHTMLWorkflow:
    """End-to-end HTML content extraction pipeline.

    Orchestrates layout clustering, LLM inference, and postprocessing to
    extract main content from HTML at Common Crawl scale.  Timing lives
    here (not inside individual stage classes) following the SemanticDedup
    workflow pattern.

    Args:
        client: AsyncLLMClient used for MinerU-HTML inference.
        model_name: HuggingFace model ID for MinerU-HTML inference.
        html_col: Column containing raw HTML (default: ``"html"``).
        url_col: Column containing page URL (default: ``"url"``).
        output_col: Column for extracted content (default: ``"dripper_content"``).
        perform_layout_clustering: Whether to run layout template clustering
            before the main extraction stages (default: ``True``).
        layout_cluster_threshold: Cosine similarity threshold for layout
            clustering (default: ``0.95``).
        fallback: Fallback strategy when LLM extraction fails —
            ``"trafilatura"``, ``"bypass"``, or ``"empty"``
            (default: ``"trafilatura"``).
        output_format: Output content format (default: ``"mm_md"``).
        max_concurrent_requests: Maximum in-flight LLM requests per worker
            (default: ``64``).
        health_check: Run a model health check on setup (default: ``True``).
        verbose: Log progress and timing (default: ``True``).
    """

    # Required — caller must supply a configured LLM client and model name
    client: AsyncLLMClient | None
    model_name: str

    # Column names
    html_col: str = "html"
    url_col: str | None = "url"
    output_col: str = "dripper_content"

    # Layout clustering options
    perform_layout_clustering: bool = True
    layout_cluster_threshold: float = 0.95

    # Extraction options
    fallback: str = "trafilatura"
    output_format: str = "mm_md"
    max_concurrent_requests: int = 64
    health_check: bool = True

    # General options
    verbose: bool = True

    def run(self, executor: BaseExecutor, initial_tasks: list[Task] | None = None) -> WorkflowRunResult:
        """Run the full extraction pipeline and return a WorkflowRunResult.

        Args:
            executor: Executor to use (e.g. ``RayActorPoolExecutor``).
            initial_tasks: Optional pre-built task list.  Pass ``None`` to
                build a pipeline with no initial tasks (the first stage must
                be a reader/source stage in that case).

        Returns:
            WorkflowRunResult with timing, stage names, and output tasks.
        """
        start = time.time()

        if self.verbose:
            logger.info(
                "DripperHTMLWorkflow starting — model={}, layout_clustering={}",
                self.model_name,
                self.perform_layout_clustering,
            )

        stages = self._build_stages()
        pipeline = Pipeline(name="dripper_html_extraction")
        for stage in stages:
            pipeline.add_stage(stage)

        output_tasks = pipeline.run(executor=executor, initial_tasks=initial_tasks)

        elapsed = time.time() - start

        if self.verbose:
            logger.info(
                "DripperHTMLWorkflow complete in {:.1f}s",
                elapsed,
            )

        result = WorkflowRunResult(workflow_name="dripper_html_extraction")
        result.add_metadata("elapsed_s", elapsed)
        result.add_metadata("stages", [s.name for s in stages])
        result.add_pipeline_tasks("dripper_html_extraction", output_tasks)
        return result

    def _build_stages(self) -> list[ProcessingStage]:
        """Construct the ordered list of processing stages."""
        stages: list[ProcessingStage] = []

        if self.perform_layout_clustering:
            stages.append(
                DripperHTMLLayoutTemplateStage(
                    client=self.client,
                    model_name=self.model_name,
                    html_col=self.html_col,
                    url_col=self.url_col,
                    layout_cluster_threshold=self.layout_cluster_threshold,
                    fallback=self.fallback,
                    output_format=self.output_format,
                    max_concurrent_requests=self.max_concurrent_requests,
                    health_check=self.health_check,
                )
            )

        # Standalone (non-layout) extraction path
        stages.extend(
            [
                DripperHTMLPreprocessStage(
                    html_col=self.html_col,
                    url_col=self.url_col,
                ),
                DripperHTMLInferenceStage(
                    client=self.client,
                    model_name=self.model_name,
                    max_concurrent_requests=self.max_concurrent_requests,
                ),
                DripperHTMLPostprocessStage(
                    html_col=self.html_col,
                    url_col=self.url_col,
                    fallback=self.fallback,
                    output_format=self.output_format,
                    output_content_col=self.output_col,
                ),
            ]
        )

        return stages
