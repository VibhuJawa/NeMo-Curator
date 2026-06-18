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

"""Abstract base class shared by DripperHTMLLayoutPlanStage and DripperHTMLLayoutFinalizeStage."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper.stages._layout_mixin import _LayoutRowKeyMixin
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import (
    _coerce_positive_int,
    _item_id_response,
    _item_ids_in_html,
    _json_safe_layout_mapping_data,
    _labels_to_webkit_response,
    _normalize_query_value_keys,
    _selected_item_ratio_from_labels,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _LAYOUT_PAGE_SIGNATURE_MODES,
    _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES,
    _LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES,
    _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES,
    _STRUCTURED_OUTPUT_MODES,
    _DripperInferenceResult,
    _DripperPostResult,
    _LayoutGroupPlan,
    _LayoutTemplateRowResult,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import (
    _append_warning,
    _coerce_html,
    _coerce_optional_float,
    _coerce_optional_str,
    _coerce_usage_int,
    _is_empty_document_error,
    _is_missing,
    _sanitize_case_output_html,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import GenerationConfig
    from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
        _LLMWebKitBindings,
        _MinerUHTMLBindings,
    )

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DripperHTMLLayoutTemplateStage(_LayoutRowKeyMixin, ProcessingStage[DocumentBatch, DocumentBatch]):
    """Abstract base class for DripperHTMLLayoutPlanStage and DripperHTMLLayoutFinalizeStage.

    This class contains all shared layout-template grouping, clustering, propagation,
    and postprocessing logic.  Subclasses must implement process(), setup(), and outputs().
    """

    name: str = "DripperHTMLLayoutTemplateStage"
    html_col: str = "html"
    url_col: str | None = "url"
    host_col: str | None = None
    layout_id_col: str | None = None
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    generation_config: GenerationConfig | None = None
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_max_representative_selected_item_ratio: float | None = None
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_validation_signature_mode: str = "none"
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_representative_candidates: int = 1
    layout_template_feature_source: Literal["raw_html", "simpled_html", "mapped_html"] = "raw_html"
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_propagation_content_source: Literal["converted", "layout_text"] = "converted"
    # Similarity gate. When True, propagation RAISES (discards content) whenever the layout parser
    # flags main_html_success=False -- which rejects nearly every cluster. The validated-good config
    # (the 0.8444-F1 run) uses False: keep main_html_body regardless of the similarity flag. This was
    # previously a getattr-only read defaulting True (unsettable) -- a regression from the stage.py split.
    layout_template_require_success: bool = False
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    layout_page_signature_mode: str = "none"
    layout_exact_query_value_keys: str | Iterable[str] | None = None
    layout_template_prompt_dedup_fallback_min_fraction: float = 0.0
    layout_template_min_saved_call_pages: int = 0
    layout_template_propagation_concurrency: int = 1
    dynamic_classid_similarity_threshold: float = 0.85
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        if not 0.0 < self.layout_cluster_threshold <= 1.0:
            msg = "layout_cluster_threshold must be in (0, 1]"
            raise ValueError(msg)
        if self.layout_template_min_cluster_size <= 1:
            msg = "layout_template_min_cluster_size must be greater than 1"
            raise ValueError(msg)
        if self.layout_template_max_selected_item_ratio is not None and not (
            0.0 < self.layout_template_max_selected_item_ratio <= 1.0
        ):
            msg = "layout_template_max_selected_item_ratio must be in (0, 1] when set"
            raise ValueError(msg)
        if self.layout_template_max_representative_selected_item_ratio is not None and not (
            0.0 < self.layout_template_max_representative_selected_item_ratio <= 1.0
        ):
            msg = "layout_template_max_representative_selected_item_ratio must be in (0, 1] when set"
            raise ValueError(msg)
        if self.layout_template_validation_rows < 0:
            msg = "layout_template_validation_rows must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_cluster_validation_rows < 0:
            msg = "layout_template_large_cluster_validation_rows must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_cluster_min_size < 0:
            msg = "layout_template_large_cluster_min_size must be non-negative"
            raise ValueError(msg)
        if self.layout_template_representative_candidates <= 0:
            msg = "layout_template_representative_candidates must be positive"
            raise ValueError(msg)
        if self.layout_template_feature_source not in _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES:
            msg = f"layout_template_feature_source must be one of {sorted(_LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES)}"
            raise ValueError(msg)
        if self.layout_template_propagation_target not in _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES:
            msg = (
                "layout_template_propagation_target must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_propagation_content_source not in _LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES:
            msg = (
                "layout_template_propagation_content_source must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES)}"
            )
            raise ValueError(msg)
        if (
            self.layout_template_propagation_target == "mapped_item_ids"
            and self.layout_template_propagation_content_source == "layout_text"
        ):
            msg = "layout_template_propagation_content_source='layout_text' requires raw_html propagation target"
            raise ValueError(msg)
        if self.layout_template_min_main_html_sim is not None and not (
            0.0 <= self.layout_template_min_main_html_sim <= 1.0
        ):
            msg = "layout_template_min_main_html_sim must be in [0, 1] when set"
            raise ValueError(msg)
        if not 0.0 <= self.layout_template_validation_min_content_f1 <= 1.0:
            msg = "layout_template_validation_min_content_f1 must be in [0, 1]"
            raise ValueError(msg)
        if self.layout_template_validation_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = f"layout_template_validation_signature_mode must be one of {sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            raise ValueError(msg)
        if (
            self.layout_template_min_content_length_ratio is not None
            and self.layout_template_min_content_length_ratio < 0
        ):
            msg = "layout_template_min_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if (
            self.layout_template_max_content_length_ratio is not None
            and self.layout_template_max_content_length_ratio < 0
        ):
            msg = "layout_template_max_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if (
            self.layout_template_min_content_length_ratio is not None
            and self.layout_template_max_content_length_ratio is not None
            and self.layout_template_min_content_length_ratio > self.layout_template_max_content_length_ratio
        ):
            msg = "layout_template_min_content_length_ratio must be <= layout_template_max_content_length_ratio"
            raise ValueError(msg)
        if self.layout_page_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = f"layout_page_signature_mode must be one of {sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            raise ValueError(msg)
        self.layout_exact_query_value_keys = _normalize_query_value_keys(self.layout_exact_query_value_keys)
        if not 0.0 <= self.layout_template_prompt_dedup_fallback_min_fraction <= 1.0:
            msg = "layout_template_prompt_dedup_fallback_min_fraction must be in [0, 1]"
            raise ValueError(msg)
        if self.layout_template_min_saved_call_pages < 0:
            msg = "layout_template_min_saved_call_pages must be non-negative"
            raise ValueError(msg)
        if self.layout_template_propagation_concurrency <= 0:
            msg = "layout_template_propagation_concurrency must be positive"
            raise ValueError(msg)
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            msg = f"structured_output_mode must be one of {sorted(_STRUCTURED_OUTPUT_MODES)}"
            raise ValueError(msg)
        if self.dynamic_classid_similarity_threshold <= 0:
            msg = "dynamic_classid_similarity_threshold must be positive"
            raise ValueError(msg)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.html_col,
            self.raw_response_col,
            self.preprocess_time_col,
            self.warning_col,
            self.item_count_col,
            self.request_max_tokens_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        raise NotImplementedError

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        raise NotImplementedError

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Layout group building
    # ------------------------------------------------------------------

    def _build_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan]:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        if len(df) < self.layout_template_min_cluster_size:
            return []
        # Clustering is a separate stage (DripperHTMLLayoutClusteringStage) that precomputes
        # `layout_id_col`. The plan/finalize stages ALWAYS consume it -- there is no on-the-fly
        # CPU re-clustering fallback (a missing column means the clustering stage didn't run).
        if not self.layout_id_col:
            _msg = "layout_id_col must be set to the precomputed layout column from DripperHTMLLayoutClusteringStage"
            raise RuntimeError(_msg)
        if self.layout_id_col not in df.columns:
            _msg = (
                f"layout_id_col={self.layout_id_col!r} is missing from the batch; "
                "run DripperHTMLLayoutClusteringStage upstream."
            )
            raise RuntimeError(_msg)
        return self._build_precomputed_layout_group_plans(df)

    def _build_precomputed_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan]:
        by_layout: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = _coerce_html(row.get(self.html_col, ""))
            if not html_text.strip():
                continue
            layout_key = self._row_layout_id_key(row)
            if not layout_key:
                continue
            by_layout[(self._row_host_key(row), layout_key)].append(int(idx))

        plans: list[_LayoutGroupPlan] = []
        for (host_key, layout_key), indexes in sorted(by_layout.items(), key=lambda item: (min(item[1]), item[0])):
            sorted_indexes = sorted(indexes)
            if len(sorted_indexes) < self.layout_template_min_cluster_size:
                continue
            plans.append(
                _LayoutGroupPlan(
                    indexes=sorted_indexes,
                    host_key=host_key,
                    source=f"precomputed_layout:{layout_key}",
                )
            )
        logger.info(
            "Dripper layout-template used precomputed layout column {} to build {} group plans",
            self.layout_id_col,
            len(plans),
        )
        return plans

    @property
    def _feature_source(self) -> str:
        return self.layout_template_feature_source

    def _row_layout_id_key(self, row: pd.Series) -> str:
        if not self.layout_id_col:
            return ""
        value = row.get(self.layout_id_col)
        text = "" if _is_missing(value) else str(value).strip()
        if not text or text in {"-1", "-2"} or text.endswith(("_-1", "_-2")):
            return ""
        return text

    # ------------------------------------------------------------------
    # Fallback / low-return helpers
    # ------------------------------------------------------------------

    def _low_return_fallback_reason(self, indexes: list[int]) -> str:
        threshold = self.layout_template_min_saved_call_pages
        if threshold <= 0:
            return ""
        max_saved = self._max_saved_call_pages(len(indexes))
        if max_saved >= threshold:
            return ""
        return (
            "layout template low-return fallback: max_saved_call_pages="
            f"{max_saved} threshold={threshold} rows={len(indexes)} "
            f"validation_rows={min(self._effective_validation_rows(len(indexes)), max(0, len(indexes) - 1))}"
        )

    def _max_saved_call_pages(self, cluster_size: int) -> int:
        validation_rows = min(self._effective_validation_rows(cluster_size), max(0, cluster_size - 1))
        return max(0, cluster_size - 1 - validation_rows)

    def _prompt_dedup_fallback_reason(self, df: pd.DataFrame, indexes: list[int]) -> str:
        threshold = self.layout_template_prompt_dedup_fallback_min_fraction
        if threshold <= 0.0 or len(indexes) < self.layout_template_min_cluster_size:
            return ""

        prompt_keys: list[tuple[str, int]] = []
        for idx in indexes:
            row = df.iloc[idx]
            prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
            if not prompt.strip():
                continue
            row_max_tokens = _coerce_usage_int(row.get(self.request_max_tokens_col, 0))
            prompt_keys.append((prompt, row_max_tokens))
        if len(prompt_keys) < self.layout_template_min_cluster_size:
            return ""

        unique_prompts = len(set(prompt_keys))
        duplicate_fraction = (len(prompt_keys) - unique_prompts) / len(prompt_keys)
        if duplicate_fraction < threshold:
            return ""
        return (
            "layout template prompt dedup fallback: duplicate_prompt_fraction="
            f"{duplicate_fraction:.3f} unique_prompt_keys={unique_prompts}/{len(prompt_keys)} "
            f"threshold={threshold:.3f}"
        )

    def _effective_validation_rows(self, cluster_size: int) -> int:
        rows = self.layout_template_validation_rows
        if (
            self.layout_template_large_cluster_validation_rows > 0
            and self.layout_template_large_cluster_min_size > 0
            and cluster_size >= self.layout_template_large_cluster_min_size
        ):
            rows = max(rows, self.layout_template_large_cluster_validation_rows)
        return rows

    # ------------------------------------------------------------------
    # Representative selection
    # ------------------------------------------------------------------

    def _select_representative_index(self, df: pd.DataFrame, indexes: list[int]) -> int:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        candidates = [
            {
                "track_id": str(idx),
                "html": self._row_feature_html(df.iloc[idx]),
            }
            for idx in indexes
        ]
        try:
            representative = self._web_bindings.select_representative_html(candidates)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper representative selection failed: {}", exc)
            representative = None
        if representative is None:
            return indexes[0]
        try:
            selected = int(representative["track_id"])
        except (KeyError, TypeError, ValueError):
            return indexes[0]
        return selected if selected in indexes else indexes[0]

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    async def _propagate_layout_template_async(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
    ) -> _LayoutTemplateRowResult:
        async with semaphore:
            return await asyncio.to_thread(self._propagate_layout_template, row, mapping_data, cluster_id)

    def _propagate_layout_template(  # noqa: C901, PLR0912, PLR0915
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
    ) -> _LayoutTemplateRowResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        started = time.perf_counter()
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        use_mapped_item_ids = (
            self.layout_template_propagation_target == "mapped_item_ids" and "_item_id" in mapped_html
        )
        html_source = mapped_html if use_mapped_item_ids else html_text
        layout_template_more_noise_enable = getattr(self, "layout_template_more_noise_enable", False)
        try:
            if self.layout_template_max_representative_selected_item_ratio is not None:
                representative_selected_item_ratio = _coerce_optional_float(
                    mapping_data.get("_dripper_representative_selected_item_ratio")
                )
                if representative_selected_item_ratio is None:
                    _msg = "layout propagation representative selected item ratio unavailable"
                    raise RuntimeError(_msg)  # noqa: TRY301
                if representative_selected_item_ratio > self.layout_template_max_representative_selected_item_ratio:
                    _msg = (
                        "layout propagation representative selected item ratio "
                        f"{representative_selected_item_ratio:.3f} exceeds "
                        f"{self.layout_template_max_representative_selected_item_ratio:.3f}"
                    )
                    raise RuntimeError(_msg)  # noqa: TRY301
            task_data = dict(mapping_data)
            task_data.setdefault("labels", mapping_data.get("_dripper_webkit_response", {}))
            task_data.update(
                {
                    "html_source": html_source,
                    "dynamic_id_enable": True,
                    "dynamic_classid_enable": True,
                    "more_noise_enable": layout_template_more_noise_enable,
                    "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
                }
            )
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
            layout_template_require_success = getattr(self, "layout_template_require_success", True)
            if layout_template_require_success and parts.get("main_html_success") is False:
                _msg = f"layout propagation similarity below threshold: {parts.get('main_html_sim')}"
                raise RuntimeError(_msg)  # noqa: TRY301
            if self.layout_template_min_main_html_sim is not None:
                main_html_sim = _coerce_optional_float(parts.get("main_html_sim"))
                if main_html_sim is not None and main_html_sim < self.layout_template_min_main_html_sim:
                    _msg = (
                        "layout propagation main_html_sim "
                        f"{main_html_sim:.3f} below {self.layout_template_min_main_html_sim:.3f}"
                    )
                    raise RuntimeError(_msg)  # noqa: TRY301
            main_html = str(parts.get("main_html_body") or "")
            layout_text = str(parts.get("main_html") or "")
            raw_response = ""
            if use_mapped_item_ids:
                all_item_ids = _item_ids_in_html(mapped_html)
                main_item_ids = set(_item_ids_in_html(main_html))
                if not all_item_ids:
                    _msg = "layout propagation target mapped HTML has no item ids"
                    raise RuntimeError(_msg)  # noqa: TRY301
                if not main_item_ids:
                    _msg = "layout propagation produced no target item ids"
                    raise RuntimeError(_msg)  # noqa: TRY301
                selected_item_ratio = len(main_item_ids) / len(all_item_ids)
                if (
                    self.layout_template_max_selected_item_ratio is not None
                    and selected_item_ratio > self.layout_template_max_selected_item_ratio
                ):
                    _msg = (
                        "layout propagation selected item ratio "
                        f"{selected_item_ratio:.3f} exceeds "
                        f"{self.layout_template_max_selected_item_ratio:.3f}"
                    )
                    raise RuntimeError(_msg)  # noqa: TRY301
                raw_response = _item_id_response(all_item_ids, main_item_ids)
                post_result = self._postprocess_raw_response(row, raw_response)
            elif self.layout_template_propagation_content_source == "layout_text":
                if not layout_text.strip() and main_html.strip():
                    _msg = "layout propagation produced empty layout text content"
                    raise RuntimeError(_msg)  # noqa: TRY301
                post_result = _DripperPostResult(main_html=main_html, main_content=layout_text)
            else:
                post_result = self._convert_main_html(row, main_html)
            content_ratio_error = self._propagated_content_length_ratio_error(
                post_result.main_content,
                mapping_data,
            )
            if content_ratio_error:
                _msg = content_ratio_error
                raise RuntimeError(_msg)  # noqa: TRY301
            return _LayoutTemplateRowResult(
                raw_response=raw_response,
                main_html=post_result.main_html,
                main_content=post_result.main_content,
                postprocess_time_s=time.perf_counter() - started,
                error=post_result.error,
                warning=post_result.warning,
                layout_cluster=cluster_id,
                layout_propagated=True,
                layout_propagation_success=not bool(post_result.error),
            )
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper layout propagation failed: {}", primary_error)
            fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
            return _LayoutTemplateRowResult(
                main_html=fallback_result.main_html,
                main_content=fallback_result.main_content,
                postprocess_time_s=time.perf_counter() - started,
                error=fallback_result.error or primary_error,
                warning=fallback_result.warning,
                primary_error=primary_error,
                layout_cluster=cluster_id,
                layout_propagated=True,
            )

    def _propagated_content_length_ratio_error(
        self,
        propagated_content: object,
        mapping_data: dict[str, Any],
    ) -> str:
        if (
            self.layout_template_min_content_length_ratio is None
            and self.layout_template_max_content_length_ratio is None
        ):
            return ""
        rep_len = _coerce_positive_int(mapping_data.get("_dripper_representative_content_len"))
        if rep_len <= 0:
            return ""
        content_len = len(str(propagated_content or ""))
        ratio = content_len / rep_len
        if (
            self.layout_template_min_content_length_ratio is not None
            and ratio < self.layout_template_min_content_length_ratio
        ):
            return (
                "layout propagation content length ratio "
                f"{ratio:.3f} below {self.layout_template_min_content_length_ratio:.3f}"
            )
        if (
            self.layout_template_max_content_length_ratio is not None
            and ratio > self.layout_template_max_content_length_ratio
        ):
            return (
                "layout propagation content length ratio "
                f"{ratio:.3f} exceeds {self.layout_template_max_content_length_ratio:.3f}"
            )
        return ""

    # ------------------------------------------------------------------
    # Postprocessing helpers
    # ------------------------------------------------------------------

    def _postprocess_raw_response(self, row: pd.Series, raw_response: str) -> _DripperPostResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        started = time.perf_counter()
        case = self._build_case(row)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=raw_response)
            case = self._bindings.parse_result(case)
            case = self._bindings.extract_main_html_single(case)
            result = self._convert_case(case)
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper parse/extract failed, applying {} fallback: {}", self.fallback, primary_error)
            result = self._fallback_and_convert(row, primary_error=primary_error)
        return replace(result, postprocess_time_s=time.perf_counter() - started)

    def _postprocess_error_row(  # noqa: PLR0913
        self,
        row: pd.Series,
        inference_result: _DripperInferenceResult,
        layout_cluster: str,
        *,
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
        primary_error: str = "",
    ) -> _LayoutTemplateRowResult:
        primary_error = _append_warning(primary_error, inference_result.primary_error)
        fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
        return _LayoutTemplateRowResult(
            raw_response=inference_result.raw_response,
            inference_time_s=inference_result.inference_time_s,
            prompt_tokens=inference_result.prompt_tokens,
            completion_tokens=inference_result.completion_tokens,
            total_tokens=inference_result.total_tokens,
            main_html=fallback_result.main_html,
            main_content=fallback_result.main_content,
            postprocess_time_s=fallback_result.postprocess_time_s,
            error=fallback_result.error,
            warning=fallback_result.warning,
            primary_error=primary_error,
            layout_cluster=layout_cluster,
            layout_fallback_llm=layout_fallback_llm,
            layout_standalone_llm=layout_standalone_llm,
        )

    def _defer_row(  # noqa: PLR0913
        self,
        row: pd.Series,
        *,
        primary_error: str = "",
        layout_cluster: str = "",
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
        force_needs_llm: bool = False,
    ) -> _LayoutTemplateRowResult:
        needs_llm = force_needs_llm or bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
        return _LayoutTemplateRowResult(
            raw_response=str(row.get(self.raw_response_col, "") or ""),
            inference_time_s=float(row.get(self.inference_time_col, 0.0) or 0.0),
            prompt_tokens=_coerce_usage_int(row.get(self.prompt_tokens_col, 0)),
            completion_tokens=_coerce_usage_int(row.get(self.completion_tokens_col, 0)),
            total_tokens=_coerce_usage_int(row.get(self.total_tokens_col, 0)),
            error=str(row.get(self.error_col, "") or ""),
            warning=_append_warning(str(row.get(self.warning_col, "") or ""), primary_error),
            primary_error=primary_error,
            deferred_llm=needs_llm,
            layout_finalized=False,
            layout_cluster=layout_cluster,
            layout_fallback_llm=layout_fallback_llm and needs_llm,
            layout_standalone_llm=layout_standalone_llm and needs_llm,
        )

    # ------------------------------------------------------------------
    # Representative inference and mapping
    # ------------------------------------------------------------------

    def _representative_mapping_from_inference(
        self,
        row: pd.Series,
        inference_result: _DripperInferenceResult,
        cluster_id: str,
    ) -> tuple[_LayoutTemplateRowResult, dict[str, Any] | None]:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        started = time.perf_counter()
        if inference_result.primary_error:
            return self._postprocess_error_row(row, inference_result, cluster_id), None
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        case = self._build_case(row)
        layout_template_require_success = getattr(self, "layout_template_require_success", True)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=inference_result.raw_response)
            case = self._bindings.parse_result(case)
            item_labels = getattr(case.parse_result, "item_label", {})
            representative_selected_item_ratio = _selected_item_ratio_from_labels(item_labels)
            webkit_response = _labels_to_webkit_response(item_labels)
            case = self._bindings.extract_main_html_single(case)
            post_result = self._convert_case(case)
            mapping_data = self._web_bindings.map_parser_cls({}).parse(
                {
                    "typical_raw_tag_html": mapped_html,
                    "typical_raw_html": html_text,
                    "llm_response": webkit_response,
                }
            )
            mapping_failure_reason = ""
            if layout_template_require_success and mapping_data.get("typical_main_html_success") is False:
                mapping_failure_reason = "typical_main_html_success=false"
                mapping_data = None
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper representative mapping failed: {}", primary_error)
            fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
            return (
                _LayoutTemplateRowResult(
                    raw_response=inference_result.raw_response,
                    inference_time_s=inference_result.inference_time_s,
                    prompt_tokens=inference_result.prompt_tokens,
                    completion_tokens=inference_result.completion_tokens,
                    total_tokens=inference_result.total_tokens,
                    main_html=fallback_result.main_html,
                    main_content=fallback_result.main_content,
                    postprocess_time_s=time.perf_counter() - started,
                    error=fallback_result.error,
                    warning=fallback_result.warning,
                    primary_error=primary_error,
                    layout_cluster=cluster_id,
                ),
                None,
            )

        warning = post_result.warning
        if mapping_data is None:
            primary_error = f"layout template mapping failed: {mapping_failure_reason or 'template unusable'}"
            warning = _append_warning(warning, primary_error)
        else:
            primary_error = ""
            mapping_data = dict(mapping_data)
            mapping_data = _json_safe_layout_mapping_data(mapping_data)
            mapping_data["_dripper_webkit_response"] = webkit_response
            mapping_data["_dripper_representative_content_len"] = len(str(post_result.main_content or ""))
            mapping_data["_dripper_representative_selected_item_ratio"] = representative_selected_item_ratio
        return (
            _LayoutTemplateRowResult(
                raw_response=inference_result.raw_response,
                inference_time_s=inference_result.inference_time_s,
                prompt_tokens=inference_result.prompt_tokens,
                completion_tokens=inference_result.completion_tokens,
                total_tokens=inference_result.total_tokens,
                main_html=post_result.main_html,
                main_content=post_result.main_content,
                postprocess_time_s=time.perf_counter() - started,
                error=post_result.error,
                warning=warning,
                primary_error=primary_error,
                layout_cluster=cluster_id,
            ),
            mapping_data,
        )

    # ------------------------------------------------------------------
    # Case / conversion helpers
    # ------------------------------------------------------------------

    def _build_case(self, row: pd.Series) -> object:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        html_text = _coerce_html(row.get(self.html_col, ""))
        url = _coerce_optional_str(row.get(self.url_col) if self.url_col else None)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html_text, url=url))
        simplified_html = str(row.get(self.simplified_html_col, "") or "")
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _fallback_and_convert(self, row: pd.Series, *, primary_error: str = "") -> _DripperPostResult:
        started = time.perf_counter()
        case = self._build_case(row)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not _coerce_html(row.get(self.html_col, "")).strip():
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                warning=_append_warning(primary_error, "empty HTML input"),
            )
        fallback_result = self._apply_fallback(case, primary_error)
        case = fallback_result[0]
        if fallback_result[2]:
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                error=fallback_result[2],
                warning=fallback_result[1],
            )
        result = self._convert_case(case, warning=fallback_result[1])
        return replace(result, postprocess_time_s=time.perf_counter() - started)

    def _convert_main_html(self, row: pd.Series, main_html: str) -> _DripperPostResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        case = self._build_case(row)
        case.output_data = self._bindings.output_cls(main_html=main_html)
        return self._convert_case(case)

    def _convert_case(self, case: object, *, warning: str = "") -> _DripperPostResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        conversion_error = ""
        try:
            _sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
        except Exception as exc:  # noqa: BLE001
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)

        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        if main_content is None:
            main_content = ""
        error = ""
        if conversion_error:
            if _is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error
        return _DripperPostResult(main_html=main_html, main_content=main_content, error=error, warning=warning)

    def _apply_fallback(self, case: object, primary_error: str) -> tuple[object, str, str]:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        try:
            case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
        except Exception as fallback_exc:  # noqa: BLE001
            if primary_error:
                return case, primary_error, f"{primary_error}; fallback failed: {fallback_exc}"
            return case, "", f"fallback failed: {fallback_exc}"
        else:
            return case, primary_error, ""
