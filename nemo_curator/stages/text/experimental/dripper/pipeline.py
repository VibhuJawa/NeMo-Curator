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

"""Dripper Common Crawl pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader
from nemo_curator.stages.text.download.common_crawl.warc_parse import WARCParseStage
from nemo_curator.stages.text.experimental.dripper.stages.clustering import DripperHTMLLayoutClusteringStage
from nemo_curator.stages.text.experimental.dripper.stages.grouping import HostDomainGroupingStage
from nemo_curator.stages.text.experimental.dripper.stages.inference import DripperHTMLInferenceStage
from nemo_curator.stages.text.experimental.dripper.stages.layout_finalize import DripperHTMLLayoutFinalizeStage
from nemo_curator.stages.text.experimental.dripper.stages.layout_plan import DripperHTMLLayoutPlanStage
from nemo_curator.stages.text.experimental.dripper.stages.postprocess import DripperHTMLPostprocessStage
from nemo_curator.stages.text.experimental.dripper.stages.preprocess import DripperHTMLPreprocessStage


@dataclass(kw_only=True)
class DripperCommonCrawlPipeline(CompositeStage):
    """Full streaming Dripper pipeline: group -> WARC fetch -> parse -> preprocess -> cluster -> plan -> infer -> finalize -> infer2 -> postprocess.

    Host grouping runs before WARC fetch so the StreamingRepartition barrier only
    blocks on fast parquet manifest reads (~seconds) rather than S3/HTTPS WARC
    fetches (~15-20 min for 100k rows).  This keeps vLLM inference actors active
    from the first host-group batch onward rather than waiting for all WARCs.

    DAG:
      1. HostDomainGroupingStage     -- group by host_domain (IS_FANOUT_STAGE); url_host_name in manifest
      2. CommonCrawlWARCReader       -- fetch WARC bytes per host-group batch
      3. WARCParseStage              -- parse HTTP response bytes -> html text
      4. DripperHTMLPreprocessStage  -- simplify HTML, build prompts
      5. DripperHTMLLayoutClusteringStage -- DBSCAN clustering per host
      6. DripperHTMLLayoutPlanStage  -- plan which rows need LLM (_dripper_needs_llm)
      7. DripperHTMLInferenceStage   -- inference for representatives + validation
      8. DripperHTMLLayoutFinalizeStage   -- propagate templates, defer failures
      9. DripperHTMLInferenceStage   -- inference for deferred rows
      10. DripperHTMLPostprocessStage -- parse responses, extract content
    """

    # LLM client for inference
    client: AsyncLLMClient | None
    model_name: str

    # Column names
    warc_filename_col: str = "warc_filename"
    warc_record_offset_col: str = "warc_record_offset"
    warc_record_length_col: str = "warc_record_length"
    binary_content_col: str = "binary_content"
    html_col: str = "html"
    url_col: str | None = "url"
    host_domain_col: str = "host_domain"

    # HostDomainGroupingStage config
    min_rows_per_batch: int = 1000

    # Preprocess config
    prompt_version: str = "short_compact"
    generation_config: GenerationConfig | None = None
    dynamic_max_tokens: bool = False

    # Clustering config
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_page_signature_mode: str = "none"
    layout_exact_query_value_keys: str | None = None
    layout_template_feature_source: Literal["raw_html", "simpled_html", "mapped_html"] = "raw_html"

    # Layout template config (shared between Plan and Finalize)
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_max_representative_selected_item_ratio: float | None = None
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_validation_aggregation: str = "min"
    layout_template_validation_signature_mode: str = "none"
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_representative_candidates: int = 1
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_propagation_content_source: Literal["converted", "layout_text"] = "converted"
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    layout_template_prompt_dedup_fallback_min_fraction: float = 0.0
    layout_template_min_saved_call_pages: int = 0
    layout_template_propagation_concurrency: int = 1
    dynamic_classid_similarity_threshold: float = 0.85

    # Postprocess config
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False

    # Inference config
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    health_check: bool = True

    # Worker counts
    worker_count: int | None = None

    # WARC fetch config
    use_s3: bool = False
    warc_max_workers: int = 16

    # Phase 2 mode: when True, skip group/WARC/parse/preprocess/cluster/plan and run
    # only the LLM stages (infer -> finalize -> infer -> postprocess) on an input that
    # already has precomputed layout_id + prompts (from a Phase 1 clustering run).
    inference_only: bool = False

    def __post_init__(self) -> None:
        super().__init__()
        if self.client is None:
            msg = "DripperCommonCrawlPipeline requires a non-None 'client'"
            raise ValueError(msg)
        self.model_name = self.model_name.strip()
        if not self.model_name:
            msg = "DripperCommonCrawlPipeline requires a non-empty 'model_name'"
            raise ValueError(msg)
        self.stages = self._build_stages()

    def _build_stages(self) -> list[ProcessingStage]:
        """Construct the ordered list of sub-stages for the Dripper pipeline.

        Host grouping runs before WARC fetch so that the StreamingRepartition barrier
        (which collects all data before emitting per-host blocks) only blocks on fast
        parquet manifest reads rather than slow S3/HTTPS WARC fetches. This ensures
        vLLM inference actors receive the first host-group batch within ~seconds of
        job start rather than after all WARCs are fetched (~15-20 min for 100k rows).
        """
        shared_kwargs = {
            "html_col": self.html_col,
            "url_col": self.url_col,
        }
        template_kwargs = {
            "layout_cluster_threshold": self.layout_cluster_threshold,
            "layout_template_min_cluster_size": self.layout_template_min_cluster_size,
            "layout_template_max_selected_item_ratio": self.layout_template_max_selected_item_ratio,
            "layout_template_max_representative_selected_item_ratio": self.layout_template_max_representative_selected_item_ratio,
            "layout_template_validation_rows": self.layout_template_validation_rows,
            "layout_template_validation_min_content_f1": self.layout_template_validation_min_content_f1,
            "layout_template_validation_aggregation": self.layout_template_validation_aggregation,
            "layout_template_validation_signature_mode": self.layout_template_validation_signature_mode,
            "layout_template_large_cluster_validation_rows": self.layout_template_large_cluster_validation_rows,
            "layout_template_large_cluster_min_size": self.layout_template_large_cluster_min_size,
            "layout_template_representative_candidates": self.layout_template_representative_candidates,
            "layout_template_feature_source": self.layout_template_feature_source,
            "layout_template_propagation_target": self.layout_template_propagation_target,
            "layout_template_propagation_content_source": self.layout_template_propagation_content_source,
            "layout_template_min_main_html_sim": self.layout_template_min_main_html_sim,
            "layout_template_min_content_length_ratio": self.layout_template_min_content_length_ratio,
            "layout_template_max_content_length_ratio": self.layout_template_max_content_length_ratio,
            "layout_page_signature_mode": self.layout_page_signature_mode,
            "layout_exact_query_value_keys": self.layout_exact_query_value_keys,
            "layout_template_prompt_dedup_fallback_min_fraction": self.layout_template_prompt_dedup_fallback_min_fraction,
            "layout_template_min_saved_call_pages": self.layout_template_min_saved_call_pages,
            "layout_template_propagation_concurrency": self.layout_template_propagation_concurrency,
            "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
            "worker_count": self.worker_count,
        }

        all_stages = [
            # HostDomainGroupingStage runs first: url_host_name is already in the manifest
            # parquet so no WARC fetch is needed.  The StreamingRepartition barrier inside
            # this stage only blocks on fast parquet reads, not S3 WARC fetches.
            HostDomainGroupingStage(
                host_domain_col=self.host_domain_col,
                min_rows_per_batch=self.min_rows_per_batch,
            ),
            CommonCrawlWARCReader(
                warc_filename_col=self.warc_filename_col,
                warc_record_offset_col=self.warc_record_offset_col,
                warc_record_length_col=self.warc_record_length_col,
                binary_content_col=self.binary_content_col,
                use_s3=self.use_s3,
                max_workers=self.warc_max_workers,
            ),
            WARCParseStage(
                binary_content_col=self.binary_content_col,
                html_col=self.html_col,
            ),
            DripperHTMLPreprocessStage(
                **shared_kwargs,
                prompt_version=self.prompt_version,
                generation_config=self.generation_config,
                dynamic_max_tokens=self.dynamic_max_tokens,
                worker_count=self.worker_count,
            ),
            DripperHTMLLayoutClusteringStage(
                **shared_kwargs,
                layout_cluster_threshold=self.layout_cluster_threshold,
                layout_template_min_cluster_size=self.layout_template_min_cluster_size,
                layout_page_signature_mode=self.layout_page_signature_mode,
                layout_exact_query_value_keys=self.layout_exact_query_value_keys,
                layout_feature_source=self.layout_template_feature_source,
                worker_count=self.worker_count,
            ),
            DripperHTMLLayoutPlanStage(
                **shared_kwargs,
                **template_kwargs,
            ),
            DripperHTMLInferenceStage(
                client=self.client,
                model_name=self.model_name,
                generation_config=self.generation_config,
                structured_output_mode=self.structured_output_mode,
                max_concurrent_requests=self.max_concurrent_requests,
                health_check=self.health_check,
                worker_count=self.worker_count,
            ),
            DripperHTMLLayoutFinalizeStage(
                **shared_kwargs,
                **template_kwargs,
                fallback=self.fallback,
                output_format=self.output_format,
                keep_intermediate=self.keep_intermediate,
            ),
            DripperHTMLInferenceStage(
                client=self.client,
                model_name=self.model_name,
                generation_config=self.generation_config,
                structured_output_mode=self.structured_output_mode,
                max_concurrent_requests=self.max_concurrent_requests,
                health_check=False,
                worker_count=self.worker_count,
            ),
            DripperHTMLPostprocessStage(
                **shared_kwargs,
                fallback=self.fallback,
                output_format=self.output_format,
                keep_intermediate=self.keep_intermediate,
                worker_count=self.worker_count,
            ),
        ]

        if self.inference_only:
            # Phase 2: input already has precomputed layout_id + prompts (from Phase 1),
            # so drop group/WARC/parse/preprocess/cluster/plan (the first 6) and run only
            # the LLM stages: infer -> finalize -> infer -> postprocess.
            return all_stages[6:]
        return all_stages

    def decompose(self) -> list[ProcessingStage]:
        """Return the ordered sub-stages for pipeline execution."""
        return self.stages
