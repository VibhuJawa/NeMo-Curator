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

"""Tests for DripperHTMLWorkflow — the end-to-end extraction pipeline.

Matches the style of tests/stages/text/deduplication/test_semantic.py.
Tests instantiation, field access, stage list construction, and the
layout-clustering toggle — all without requiring GPU, Ray, or LLM servers.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import (
    AsyncLLMClient,
    GenerationConfig,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper import DripperHTMLWorkflow

# ---------------------------------------------------------------------------
# Minimal stub LLM client — satisfies non-None client check without a server
# ---------------------------------------------------------------------------


class _StubLLMClient(AsyncLLMClient):
    """Stub client that returns an empty string for every inference call.

    Required because DripperHTMLInferenceStage and DripperHTMLLayoutTemplateStage
    validate ``client is not None`` in their ``__post_init__`` methods.
    """

    def __init__(self) -> None:
        super().__init__(max_concurrent_requests=1, max_retries=0, base_delay=0.0)

    def setup(self) -> None:
        pass

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: object = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        return [""]


@pytest.fixture
def stub_client() -> _StubLLMClient:
    """Reusable stub LLM client fixture."""
    return _StubLLMClient()


@pytest.fixture
def synthetic_html_df() -> pd.DataFrame:
    """Small synthetic HTML dataset for workflow tests."""
    return pd.DataFrame(
        [
            {
                "url": f"https://example.com/page{i}",
                "url_host_name": "example.com",
                "html": (f"<html><body><h1>Title {i}</h1><p>Body text for page {i}.</p></body></html>"),
            }
            for i in range(20)
        ]
    )


# ---------------------------------------------------------------------------
# TestDripperHTMLWorkflow
# ---------------------------------------------------------------------------


class TestDripperHTMLWorkflow:
    """Workflow-level unit tests — no GPU, Ray, or LLM server required."""

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_workflow_instantiation_with_defaults(self, stub_client: _StubLLMClient) -> None:
        """DripperHTMLWorkflow can be constructed with only required args."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
        )
        assert workflow is not None

    def test_workflow_default_field_values(self, stub_client: _StubLLMClient) -> None:
        """Default dataclass fields match documented defaults."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
        )
        assert workflow.perform_layout_clustering is True
        assert workflow.layout_cluster_threshold == pytest.approx(0.95)
        assert workflow.fallback == "trafilatura"
        assert workflow.output_format == "mm_md"
        assert workflow.max_concurrent_requests == 64
        assert workflow.health_check is True
        assert workflow.verbose is True
        assert workflow.html_col == "html"
        assert workflow.url_col == "url"
        assert workflow.output_col == "dripper_content"

    def test_workflow_custom_fields(self, stub_client: _StubLLMClient) -> None:
        """Custom field values are stored correctly."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="custom-model",
            layout_cluster_threshold=0.85,
            perform_layout_clustering=False,
            fallback="bypass",
            output_format="text",
            max_concurrent_requests=32,
            health_check=False,
            verbose=False,
        )
        assert workflow.model_name == "custom-model"
        assert workflow.layout_cluster_threshold == pytest.approx(0.85)
        assert workflow.perform_layout_clustering is False
        assert workflow.fallback == "bypass"
        assert workflow.output_format == "text"
        assert workflow.max_concurrent_requests == 32
        assert workflow.health_check is False
        assert workflow.verbose is False

    # ------------------------------------------------------------------
    # Stage construction
    # ------------------------------------------------------------------

    def test_build_stages_returns_nonempty_list(self, stub_client: _StubLLMClient) -> None:
        """_build_stages() returns a non-empty list of ProcessingStage instances."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
        )
        stages = workflow._build_stages()
        assert len(stages) > 0
        for stage in stages:
            assert isinstance(stage, ProcessingStage)

    def test_build_stages_all_have_names(self, stub_client: _StubLLMClient) -> None:
        """Every stage returned by _build_stages() has a non-empty name string."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
        )
        for stage in workflow._build_stages():
            assert isinstance(stage.name, str)
            assert stage.name.strip(), f"Stage {stage!r} has an empty name"

    def test_build_stages_with_clustering(self, stub_client: _StubLLMClient) -> None:
        """With layout clustering enabled the stage list includes the layout stage."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            perform_layout_clustering=True,
            health_check=False,
        )
        stage_names = [s.name for s in workflow._build_stages()]
        assert any("Layout" in name for name in stage_names), f"Expected a layout stage in {stage_names!r}"

    def test_build_stages_without_clustering(self, stub_client: _StubLLMClient) -> None:
        """With layout clustering disabled the stage list omits the layout stage."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            perform_layout_clustering=False,
            health_check=False,
        )
        stage_names = [s.name for s in workflow._build_stages()]
        assert not any("Layout" in name for name in stage_names), f"Unexpected layout stage in {stage_names!r}"

    def test_clustering_toggle_changes_stage_count(self, stub_client: _StubLLMClient) -> None:
        """Enabling layout clustering adds at least one stage compared to disabling it."""
        with_clust = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            perform_layout_clustering=True,
            health_check=False,
        )
        without_clust = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            perform_layout_clustering=False,
            health_check=False,
        )
        assert len(with_clust._build_stages()) > len(without_clust._build_stages())

    def test_build_stages_without_clustering_has_preprocess_inference_postprocess(
        self, stub_client: _StubLLMClient
    ) -> None:
        """Without clustering, the three core stages are present in order."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            perform_layout_clustering=False,
            health_check=False,
        )
        names = [s.name for s in workflow._build_stages()]
        assert "DripperHTMLPreprocessStage" in names
        assert "DripperHTMLInferenceStage" in names
        assert "DripperHTMLPostprocessStage" in names
        # Preprocess must precede inference, inference must precede postprocess
        assert names.index("DripperHTMLPreprocessStage") < names.index("DripperHTMLInferenceStage")
        assert names.index("DripperHTMLInferenceStage") < names.index("DripperHTMLPostprocessStage")

    # ------------------------------------------------------------------
    # Column name propagation
    # ------------------------------------------------------------------

    def test_custom_column_names_propagate_to_stages(self, stub_client: _StubLLMClient) -> None:
        """Column name overrides on the workflow propagate to the underlying stages."""
        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            html_col="raw_html",
            url_col="page_url",
            output_col="extracted_text",
            perform_layout_clustering=False,
            health_check=False,
        )
        stages = workflow._build_stages()
        # PreprocessStage should use the overridden html_col and url_col
        preprocess = next(s for s in stages if s.name == "DripperHTMLPreprocessStage")
        assert preprocess.html_col == "raw_html"
        assert preprocess.url_col == "page_url"
        # PostprocessStage should use the overridden output_col
        postprocess = next(s for s in stages if s.name == "DripperHTMLPostprocessStage")
        assert postprocess.output_content_col == "extracted_text"

    # ------------------------------------------------------------------
    # run() contract (dict keys)
    # ------------------------------------------------------------------

    def test_run_returns_dict_with_expected_keys(
        self, stub_client: _StubLLMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """workflow.run() returns a dict containing 'elapsed_s', 'stages', 'output_tasks'."""
        from nemo_curator.pipeline import Pipeline

        # Monkeypatch Pipeline.run to avoid actually executing the pipeline
        def _noop_run(_self, _executor, _initial_tasks=None):
            return []

        monkeypatch.setattr(Pipeline, "run", _noop_run)

        workflow = DripperHTMLWorkflow(
            client=stub_client,
            model_name="test-model",
            perform_layout_clustering=False,
            health_check=False,
            verbose=False,
        )

        from nemo_curator.backends.xenna import XennaExecutor

        result = workflow.run(executor=XennaExecutor())
        assert isinstance(result, dict)
        assert "elapsed_s" in result
        assert "stages" in result
        assert "output_tasks" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0
        assert isinstance(result["stages"], list)
        assert len(result["stages"]) > 0
