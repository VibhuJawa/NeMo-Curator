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

"""DripperHTMLLayoutTemplateStage — layout clustering + template propagation.

This module owns the layout-template extraction path end-to-end:
  - DripperHTMLLayoutTemplateStage  (main class)
  - _LLMWebKitBindings              (llm-web-kit runtime bindings)
  - All layout-group dataclasses    (_LayoutGroupPlan, _LayoutGroupRun, …)
  - All layout-specific helpers     (URL keying, DOM fingerprinting, …)

Shared utilities (_append_warning, _coerce_html, _rebuild_batch, …) and
shared dataclasses (_MinerUHTMLBindings, _DripperInferenceResult, …) live
in stage.py and are imported from there.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlparse

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _STRUCTURED_OUTPUT_MODES,
    _append_warning,
    _apply_fallback_extraction,
    _coerce_html,
    _coerce_optional_str,
    _coerce_usage_int,
    _DripperInferenceResult,
    _DripperPostResult,
    _is_empty_document_error,
    _is_missing,
    _item_ids_in_html,
    _LLMWebKitBindings,
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
    _numeric_series_or_zero,
    _query_dripper_model,
    _rebuild_batch,
    _run_dripper_health_check,
    _sanitize_case_output_html,
    _with_structured_output_config,
)
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient


# ---------------------------------------------------------------------------
# Layout-template dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LayoutTemplateRowResult:
    """Per-row output from layout-template extraction."""

    raw_response: str = ""
    inference_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    main_html: str = ""
    main_content: Any = ""
    postprocess_time_s: float = 0.0
    error: str = ""
    warning: str = ""
    primary_error: str = ""
    deferred_llm: bool = False
    layout_finalized: bool = True
    layout_cluster: str = ""
    layout_representative: bool = False
    layout_propagated: bool = False
    layout_propagation_success: bool = False
    layout_fallback_llm: bool = False
    layout_standalone_llm: bool = False
    layout_pending_propagation: bool = False
    layout_mapping_json: str = ""


@dataclass(frozen=True)
class _LayoutGroupPlan:
    """A layout group to try, plus safer fallback groups if the attempt fails."""

    indexes: list[int]
    host_key: str = ""
    source: str = "dom"
    fallback_groups: tuple[list[int], ...] = ()


@dataclass(frozen=True)
class _LayoutGroupOutcome:
    """Result of processing one layout group."""

    results: dict[int, _LayoutTemplateRowResult]
    accepted: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class _LayoutProcessContext:
    """Shared async context for layout-template group processing."""

    df: pd.DataFrame
    semaphore: asyncio.Semaphore
    propagation_semaphore: asyncio.Semaphore
    inference_cache: _InferenceCache
    inference_cache_lock: asyncio.Lock
    needs_llm: list[bool]


@dataclass(frozen=True)
class _LayoutGroupAttempt:
    """A single layout-group attempt plus its fallback configuration."""

    indexes: list[int]
    cluster_id: str
    host_key: str
    source: str
    fallback_groups: tuple[list[int], ...]
    split_failed_host_fallback: bool


@dataclass(frozen=True)
class _LayoutGroupRun:
    """Per-group processing parameters for a single layout-template attempt."""

    ctx: _LayoutProcessContext
    indexes: list[int]
    cluster_id: str
    emit_failure_fallback: bool


@dataclass(frozen=True)
class _ValidationOutcome:
    """Result of validating propagated rows against per-row LLM extraction."""

    failed: bool = False
    error: str = ""


@dataclass(frozen=True)
class _InferContext:
    """Inference context bundle for per-row inference and postprocessing."""

    semaphore: asyncio.Semaphore | None = None
    cache: _InferenceCache | None = None
    cache_lock: asyncio.Lock | None = None
    layout_cluster: str = ""
    layout_fallback_llm: bool = False
    layout_standalone_llm: bool = False
    primary_error: str = ""


@dataclass
class _SelectorState:
    """Mutable accumulation state for validation index selection."""

    selected: list[int]
    selected_set: set[int]
    count: int
    url_col: str | None
    item_count_col: str

    def add(self, idx: int) -> None:
        if len(self.selected) >= self.count or idx in self.selected_set:
            return
        self.selected.append(idx)
        self.selected_set.add(idx)

    def is_full(self) -> bool:
        return len(self.selected) >= self.count


_ColSpec = tuple[str | None, str]

_InferenceCache = dict[tuple[str, int], asyncio.Task[_DripperInferenceResult]]


def _inference_token_fields(r: _DripperInferenceResult) -> dict[str, object]:
    """Return the shared token/timing fields from an inference result for use in _LayoutTemplateRowResult(**...)."""
    return {
        "raw_response": r.raw_response,
        "inference_time_s": r.inference_time_s,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
    }


# ---------------------------------------------------------------------------
# Validation helpers (only used by DripperHTMLLayoutTemplateStage)
# ---------------------------------------------------------------------------


def _check_enum_field(value: object, valid_set: set, field_name: str) -> None:
    if value not in valid_set:
        msg = f"{field_name} must be one of {sorted(valid_set)}"
        raise ValueError(msg)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# DripperHTMLLayoutTemplateStage
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DripperHTMLLayoutTemplateStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Infer layout representatives, then propagate their template on CPU."""

    name: str = "DripperHTMLLayoutTemplateStage"
    client: AsyncLLMClient | None
    model_name: str
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
    max_concurrent_requests: int = 64
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_template_fallback_llm: bool = True
    layout_template_require_success: bool = True
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_more_noise_enable: bool = True
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_validation_signature_mode: str = "none"
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_representative_candidates: int = 1
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    layout_template_defer_fallback_llm: bool = False
    layout_template_defer_propagation: bool = False
    layout_page_signature_mode: str = "none"
    layout_template_failed_host_fallback_signature_mode: str = "none"
    layout_template_failed_layout_fallback_signature_mode: str = "none"
    layout_template_host_single_cluster_min_pages: int = 0
    layout_template_host_single_cluster_max_pages: int = 0
    layout_template_max_exact_host_pages: int = 0
    layout_template_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    layout_template_propagation_concurrency: int = 32
    dynamic_classid_similarity_threshold: float = 0.85
    health_check: bool = False
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        _require(
            self.client is not None, "DripperHTMLLayoutTemplateStage requires a non-None 'client' (AsyncLLMClient)"
        )
        self.model_name = self.model_name.strip()
        _require(bool(self.model_name), "DripperHTMLLayoutTemplateStage requires a non-empty 'model_name'")
        _require(self.max_concurrent_requests > 0, "max_concurrent_requests must be positive")
        self._validate_layout_template_thresholds()
        self._validate_layout_template_modes()
        self._validate_layout_template_host_config()

    def _validate_layout_template_thresholds(self) -> None:
        _require(0.0 < self.layout_cluster_threshold <= 1.0, "layout_cluster_threshold must be in (0, 1]")
        _require(self.layout_template_min_cluster_size > 1, "layout_template_min_cluster_size must be greater than 1")
        _require(
            self.layout_template_max_selected_item_ratio is None
            or 0.0 < self.layout_template_max_selected_item_ratio <= 1.0,
            "layout_template_max_selected_item_ratio must be in (0, 1] when set",
        )
        _require(
            self.layout_template_representative_candidates > 0,
            "layout_template_representative_candidates must be positive",
        )
        _require(
            self.layout_template_min_main_html_sim is None or 0.0 <= self.layout_template_min_main_html_sim <= 1.0,
            "layout_template_min_main_html_sim must be in [0, 1] when set",
        )
        _require(
            0.0 <= self.layout_template_validation_min_content_f1 <= 1.0,
            "layout_template_validation_min_content_f1 must be in [0, 1]",
        )
        _require(
            self.dynamic_classid_similarity_threshold > 0, "dynamic_classid_similarity_threshold must be positive"
        )
        self._validate_layout_template_row_limits()
        self._validate_layout_template_content_length_ratios()

    def _validate_layout_template_row_limits(self) -> None:
        _require(self.layout_template_validation_rows >= 0, "layout_template_validation_rows must be non-negative")
        _require(
            self.layout_template_large_cluster_validation_rows >= 0,
            "layout_template_large_cluster_validation_rows must be non-negative",
        )
        _require(
            self.layout_template_large_cluster_min_size >= 0,
            "layout_template_large_cluster_min_size must be non-negative",
        )

    def _validate_layout_template_content_length_ratios(self) -> None:
        min_ratio = self.layout_template_min_content_length_ratio
        max_ratio = self.layout_template_max_content_length_ratio
        _require(
            min_ratio is None or min_ratio >= 0,
            "layout_template_min_content_length_ratio must be non-negative when set",
        )
        _require(
            max_ratio is None or max_ratio >= 0,
            "layout_template_max_content_length_ratio must be non-negative when set",
        )
        _require(
            min_ratio is None or max_ratio is None or min_ratio <= max_ratio,
            "layout_template_min_content_length_ratio must be <= layout_template_max_content_length_ratio",
        )

    def _validate_layout_template_modes(self) -> None:
        _check_enum_field(
            self.layout_template_propagation_target,
            _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES,
            "layout_template_propagation_target",
        )
        _check_enum_field(
            self.layout_template_validation_signature_mode,
            _LAYOUT_PAGE_SIGNATURE_MODES,
            "layout_template_validation_signature_mode",
        )
        _check_enum_field(self.layout_page_signature_mode, _LAYOUT_PAGE_SIGNATURE_MODES, "layout_page_signature_mode")
        _check_enum_field(
            self.layout_template_failed_host_fallback_signature_mode,
            _LAYOUT_PAGE_SIGNATURE_MODES,
            "layout_template_failed_host_fallback_signature_mode",
        )
        _check_enum_field(
            self.layout_template_failed_layout_fallback_signature_mode,
            _LAYOUT_PAGE_SIGNATURE_MODES,
            "layout_template_failed_layout_fallback_signature_mode",
        )
        _check_enum_field(
            self.layout_template_large_host_mode, _LAYOUT_TEMPLATE_LARGE_HOST_MODES, "layout_template_large_host_mode"
        )
        _check_enum_field(self.structured_output_mode, _STRUCTURED_OUTPUT_MODES, "structured_output_mode")

    def _validate_layout_template_host_config(self) -> None:
        _require(
            self.layout_template_host_single_cluster_min_pages >= 0,
            "layout_template_host_single_cluster_min_pages must be non-negative",
        )
        _require(
            self.layout_template_host_single_cluster_max_pages >= 0,
            "layout_template_host_single_cluster_max_pages must be non-negative",
        )
        _require(
            self.layout_template_host_single_cluster_max_pages == 0
            or self.layout_template_host_single_cluster_min_pages
            <= self.layout_template_host_single_cluster_max_pages,
            "layout_template_host_single_cluster_min_pages must be less than or equal to "
            "layout_template_host_single_cluster_max_pages when the max is set",
        )
        _require(
            self.layout_template_max_exact_host_pages >= 0, "layout_template_max_exact_host_pages must be non-negative"
        )
        _require(
            self.layout_template_propagation_concurrency > 0,
            "layout_template_propagation_concurrency must be positive",
        )
        _require(self.worker_count is None or self.worker_count > 0, "worker_count must be positive when set")

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
        columns = [
            self.output_html_col,
            self.output_content_col,
            self.raw_response_col,
            self.inference_time_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            "dripper_layout_cluster",
            "dripper_layout_representative",
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            "dripper_layout_fallback_llm",
            "dripper_layout_standalone_llm",
            _DRIPPER_LAYOUT_FINALIZED_COL,
        ]
        if self.layout_template_defer_propagation:
            columns.extend(["dripper_layout_pending_propagation", "dripper_layout_mapping_json"])
        if self.layout_template_defer_fallback_llm:
            columns.extend(
                [
                    self.simplified_html_col,
                    self.mapped_html_col,
                    _DRIPPER_PROMPT_COL,
                    _DRIPPER_NEEDS_LLM_COL,
                    _DRIPPER_PRIMARY_ERROR_COL,
                    _DRIPPER_EMPTY_INPUT_COL,
                ]
            )
        if self.keep_intermediate and not self.layout_template_defer_fallback_llm:
            columns.extend([self.simplified_html_col, self.mapped_html_col])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self.client.setup()  # type: ignore[union-attr]
        if self.health_check:
            self._run_health_check()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        results = run_async_safe(lambda: self._process_all_async(df))
        preprocess_times = _numeric_series_or_zero(df, self.preprocess_time_col)
        inference_times = pd.Series([r.inference_time_s for r in results], index=df.index)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)

        for _col, _attr in [
            (self.output_html_col, "main_html"),
            (self.output_content_col, "main_content"),
            (self.raw_response_col, "raw_response"),
            (self.error_col, "error"),
            (self.prompt_tokens_col, "prompt_tokens"),
            (self.completion_tokens_col, "completion_tokens"),
            (self.total_tokens_col, "total_tokens"),
        ]:
            df[_col] = [getattr(r, _attr) for r in results]
        df[self.inference_time_col] = inference_times
        df[self.postprocess_time_col] = postprocess_times
        df[self.total_time_col] = preprocess_times + inference_times + postprocess_times
        df[self.warning_col] = [
            _append_warning(str(existing or ""), result.warning)
            for existing, result in zip(
                df.get(self.warning_col, pd.Series([""] * len(df))).tolist(), results, strict=True
            )
        ]
        for _col, _attr in [
            ("dripper_layout_cluster", "layout_cluster"),
            ("dripper_layout_representative", "layout_representative"),
            ("dripper_layout_propagated", "layout_propagated"),
            ("dripper_layout_propagation_success", "layout_propagation_success"),
            ("dripper_layout_fallback_llm", "layout_fallback_llm"),
            ("dripper_layout_standalone_llm", "layout_standalone_llm"),
            (_DRIPPER_LAYOUT_FINALIZED_COL, "layout_finalized"),
        ]:
            df[_col] = [getattr(r, _attr) for r in results]

        if self.layout_template_defer_propagation:
            df["dripper_layout_pending_propagation"] = [r.layout_pending_propagation for r in results]
            df["dripper_layout_mapping_json"] = [r.layout_mapping_json for r in results]

        if self.layout_template_defer_fallback_llm:
            existing_primary_errors = df[_DRIPPER_PRIMARY_ERROR_COL].astype(str).tolist()
            df[_DRIPPER_NEEDS_LLM_COL] = [r.deferred_llm for r in results]
            df[_DRIPPER_PRIMARY_ERROR_COL] = [
                _append_warning(existing_error, result.primary_error)
                for existing_error, result in zip(existing_primary_errors, results, strict=True)
            ]

        drop_cols = [_DRIPPER_PROMPT_COL, _DRIPPER_NEEDS_LLM_COL, _DRIPPER_PRIMARY_ERROR_COL, _DRIPPER_EMPTY_INPUT_COL]
        if not self.layout_template_defer_fallback_llm:
            drop_cols.append(_DRIPPER_LAYOUT_FINALIZED_COL)
        else:
            drop_cols = []
        if not self.keep_intermediate and not self.layout_template_defer_fallback_llm:
            drop_cols.extend([self.simplified_html_col, self.mapped_html_col])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        _metric_attrs = [
            ("layout_template_representative_rows", "layout_representative"),
            ("layout_template_propagated_rows", "layout_propagated"),
            ("layout_template_success_rows", "layout_propagation_success"),
            ("layout_template_fallback_llm_rows", "layout_fallback_llm"),
            ("layout_template_standalone_llm_rows", "layout_standalone_llm"),
            ("layout_template_deferred_llm_rows", "deferred_llm"),
            ("layout_template_finalized_rows", "layout_finalized"),
        ]
        self._log_metrics(
            {"layout_template_rows": float(len(df))}
            | {k: float(sum(getattr(r, a) for r in results)) for k, a in _metric_attrs}
        )
        return _rebuild_batch(batch, df)

    def _run_health_check(self) -> None:
        run_async_safe(lambda: _run_dripper_health_check(self.client, self.model_name, self.generation_config))

    async def _process_all_async(self, df: pd.DataFrame) -> list[_LayoutTemplateRowResult]:
        propagation_semaphore = asyncio.Semaphore(
            min(self.max_concurrent_requests, self.layout_template_propagation_concurrency)
        )
        ctx = _LayoutProcessContext(
            df=df,
            semaphore=asyncio.Semaphore(self.max_concurrent_requests),
            propagation_semaphore=propagation_semaphore,
            inference_cache={},
            inference_cache_lock=asyncio.Lock(),
            needs_llm=df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist(),
        )
        build_started = time.perf_counter()
        layout_plans = self._build_layout_group_plans(df)
        build_elapsed_s = time.perf_counter() - build_started
        grouped_indexes = {idx for plan in layout_plans for idx in plan.indexes}
        logger.info(
            "Dripper layout-template built {} group plans covering {}/{} rows in {:.3f}s; standalone rows={}",
            len(layout_plans),
            len(grouped_indexes),
            len(df),
            build_elapsed_s,
            len(df) - len(grouped_indexes),
        )

        async def _handle_plan(plan_index: int, plan: _LayoutGroupPlan) -> dict[int, _LayoutTemplateRowResult]:
            return await self._handle_group_attempt_async(
                ctx,
                _LayoutGroupAttempt(
                    indexes=plan.indexes,
                    cluster_id=f"layout-{plan_index:06d}",
                    host_key=plan.host_key,
                    source=plan.source,
                    fallback_groups=plan.fallback_groups,
                    split_failed_host_fallback=True,
                ),
            )

        tasks: list[Any] = [_handle_plan(plan_index, plan) for plan_index, plan in enumerate(layout_plans)]
        tasks.extend(self._handle_standalone_async(ctx, idx) for idx in range(len(df)) if idx not in grouped_indexes)
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results_by_index: dict[int, _LayoutTemplateRowResult] = {}
        for raw_result in raw_results:
            if isinstance(raw_result, BaseException):
                logger.error("Dripper layout-template task failed: {}", raw_result)
                continue
            if isinstance(raw_result, tuple):
                idx, result = raw_result
                results_by_index[idx] = result
            else:
                results_by_index.update(raw_result)

        return [
            results_by_index[idx] if idx in results_by_index else self._missing_layout_result(df.iloc[idx])
            for idx in range(len(df))
        ]

    async def _handle_standalone_async(
        self, ctx: _LayoutProcessContext, idx: int
    ) -> tuple[int, _LayoutTemplateRowResult]:
        if self.layout_template_defer_fallback_llm:
            return idx, self._defer_row(
                ctx.df.iloc[idx],
                layout_standalone_llm=ctx.needs_llm[idx],
                primary_error="layout template standalone row",
            )
        if ctx.needs_llm[idx]:
            result = await self._infer_and_postprocess_row(
                ctx.df.iloc[idx],
                _InferContext(
                    semaphore=ctx.semaphore,
                    cache=ctx.inference_cache,
                    cache_lock=ctx.inference_cache_lock,
                    layout_standalone_llm=True,
                ),
            )
        else:
            result = self._fallback_row(ctx.df.iloc[idx])
        return idx, result

    async def _handle_group_attempt_async(
        self,
        ctx: _LayoutProcessContext,
        attempt: _LayoutGroupAttempt,
    ) -> dict[int, _LayoutTemplateRowResult]:
        fallback_groups = attempt.fallback_groups
        outcome = await self._process_layout_group_with_status(
            ctx,
            attempt.indexes,
            attempt.cluster_id,
            emit_failure_fallback=not fallback_groups,
        )
        if outcome.accepted or not fallback_groups:
            return outcome.results

        logger.info(
            "Dripper layout attempt {} host={} source={} rows={} failed ({}); falling back to {} child groups",
            attempt.cluster_id,
            attempt.host_key,
            attempt.source,
            len(attempt.indexes),
            outcome.failure_reason,
            len(fallback_groups),
        )

        child_groups = list(fallback_groups)
        if attempt.split_failed_host_fallback and self.layout_template_failed_host_fallback_signature_mode != "none":
            child_groups = self._split_fallback_groups_by_signature(
                ctx.df, child_groups, self.layout_template_failed_host_fallback_signature_mode
            )
            logger.info(
                "Dripper layout attempt {} host={} split fallback into {} groups by {}",
                attempt.cluster_id,
                attempt.host_key,
                len(child_groups),
                self.layout_template_failed_host_fallback_signature_mode,
            )

        fallback_results: dict[int, _LayoutTemplateRowResult] = {}
        fallback_grouped_indexes: set[int] = set()
        fallback_tasks = [
            self._handle_group_attempt_async(
                ctx,
                _LayoutGroupAttempt(
                    indexes=fallback_indexes,
                    cluster_id=f"{attempt.cluster_id}-fallback-{fallback_index:06d}",
                    host_key=attempt.host_key,
                    source="fallback",
                    fallback_groups=tuple(self._build_failed_layout_fallback_groups(ctx.df, fallback_indexes)),
                    split_failed_host_fallback=False,
                ),
            )
            for fallback_index, fallback_indexes in enumerate(child_groups)
        ]
        if fallback_tasks:
            for group_result in await asyncio.gather(*fallback_tasks):
                fallback_results.update(group_result)
            fallback_grouped_indexes = {idx for group in child_groups for idx in group}

        standalone_tasks = [
            self._handle_standalone_async(ctx, idx) for idx in attempt.indexes if idx not in fallback_grouped_indexes
        ]
        if standalone_tasks:
            fallback_results.update(dict(await asyncio.gather(*standalone_tasks)))
        return fallback_results

    def _missing_layout_result(self, row: pd.Series) -> _LayoutTemplateRowResult:
        primary_error = "layout template task produced no result"
        if self.layout_template_defer_fallback_llm:
            return self._defer_row(row, primary_error=primary_error, layout_fallback_llm=True)
        return self._fallback_row(row, primary_error=primary_error)

    def _build_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan]:
        if len(df) < self.layout_template_min_cluster_size:
            return []
        precomputed_plans = self._build_precomputed_layout_group_plans(df)
        if precomputed_plans is not None:
            return precomputed_plans

        samples_by_host = self._build_host_samples(df)
        return self._build_plans_from_host_samples(df, samples_by_host)

    def _build_host_samples(self, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in df.iterrows():
            if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = _coerce_html(row.get(self.html_col, ""))
            if not html_text.strip():
                continue
            try:
                feature = self._web_bindings.get_feature(html_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            samples_by_host[self._row_host_key(row)].append(
                {"track_id": str(idx), "html": html_text, "feature": feature}
            )
        return samples_by_host

    def _build_plans_from_host_samples(
        self, df: pd.DataFrame, samples_by_host: dict[str, list[dict[str, Any]]]
    ) -> list[_LayoutGroupPlan]:
        plans: list[_LayoutGroupPlan] = []
        for host_key, samples in samples_by_host.items():
            if len(samples) < self.layout_template_min_cluster_size:
                continue
            host_indexes = sorted(int(sample["track_id"]) for sample in samples)
            fallback_groups = self._build_layout_groups_for_host_samples(df, host_key, samples)
            if self._should_try_host_single_cluster(len(samples)):
                plans.append(
                    _LayoutGroupPlan(
                        indexes=host_indexes,
                        host_key=host_key,
                        source="host_single_cluster",
                        fallback_groups=tuple(fallback_groups),
                    )
                )
                logger.debug(
                    "Dripper layout host={} rows={} will try single-template host group with {} fallback groups",
                    host_key,
                    len(host_indexes),
                    len(fallback_groups),
                )
                continue
            for indexes in fallback_groups:
                plans.append(
                    _LayoutGroupPlan(
                        indexes=indexes,
                        host_key=host_key,
                        source="dom",
                        fallback_groups=tuple(self._build_failed_layout_fallback_groups(df, indexes)),
                    )
                )
        return plans

    def _build_precomputed_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan] | None:
        if not self.layout_id_col or self.layout_id_col not in df.columns:
            return None

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
            plan_groups = self._split_large_precomputed_layout_group(df, host_key, layout_key, sorted_indexes)
            for plan_indexes in plan_groups:
                if len(plan_indexes) < self.layout_template_min_cluster_size:
                    continue
                plans.append(
                    _LayoutGroupPlan(
                        indexes=plan_indexes,
                        host_key=host_key,
                        source=f"precomputed_layout:{layout_key}",
                        fallback_groups=tuple(self._build_failed_layout_fallback_groups(df, plan_indexes)),
                    )
                )
        logger.info(
            "Dripper layout-template used precomputed layout column {} to build {} group plans",
            self.layout_id_col,
            len(plans),
        )
        return plans

    def _split_large_precomputed_layout_group(
        self,
        df: pd.DataFrame,
        host_key: str,
        layout_key: str,
        indexes: list[int],
    ) -> list[list[int]]:
        if not self.layout_template_max_exact_host_pages or len(indexes) <= self.layout_template_max_exact_host_pages:
            return [indexes]
        if self.layout_template_large_host_mode == "standalone":
            logger.debug(
                "Dripper precomputed layout group host={} layout={} rows={} exceeds max_exact_host_pages={}; leaving standalone",
                host_key,
                layout_key,
                len(indexes),
                self.layout_template_max_exact_host_pages,
            )
            return []

        samples: list[dict[str, Any]] = []
        for idx in indexes:
            html_text = _coerce_html(df.iloc[idx].get(self.html_col, ""))
            if not html_text.strip():
                continue
            sample: dict[str, Any] = {"track_id": str(idx), "html": html_text}
            if self.layout_template_large_host_mode == "feature_hash":
                try:
                    feature = self._web_bindings.get_feature(html_text) if self._web_bindings else None
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper precomputed layout feature extraction failed for row {}: {}", idx, exc)
                    continue
                if feature is None:
                    continue
                sample["feature"] = feature
            samples.append(sample)
        fingerprint_fn = (
            (lambda sample: _layout_feature_fingerprint(sample.get("feature")))
            if self.layout_template_large_host_mode == "feature_hash"
            else (lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")))
        )
        groups = self._build_fingerprint_groups(df, host_key, samples, fingerprint_fn=fingerprint_fn)
        logger.debug(
            "Dripper precomputed layout group host={} layout={} rows={} exceeded max_exact_host_pages={}; split into {} {} group(s)",
            host_key,
            layout_key,
            len(indexes),
            self.layout_template_max_exact_host_pages,
            len(groups),
            self.layout_template_large_host_mode,
        )
        return groups

    def _row_host_key(self, row: pd.Series) -> str:
        if self.host_col and self.host_col in row:
            host_key = _url_host_key(row.get(self.host_col))
            if host_key:
                return host_key
        return _url_host_key(row.get(self.url_col) if self.url_col else None)

    def _row_layout_id_key(self, row: pd.Series) -> str:
        if not self.layout_id_col:
            return ""
        value = row.get(self.layout_id_col)
        text = "" if _is_missing(value) else str(value).strip()
        if not text or text in {"-1", "-2"} or text.endswith(("_-1", "_-2")):
            return ""
        return text

    def _should_try_host_single_cluster(self, host_pages: int) -> bool:
        if self.layout_template_host_single_cluster_min_pages <= 0:
            return False
        if host_pages < self.layout_template_host_single_cluster_min_pages:
            return False
        return not (
            self.layout_template_host_single_cluster_max_pages > 0
            and host_pages > self.layout_template_host_single_cluster_max_pages
        )

    def _build_layout_groups_for_host_samples(
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
    ) -> list[list[int]]:
        if len(samples) < self.layout_template_min_cluster_size:
            return []

        large_host_groups = self._build_large_host_groups(df, host_key, samples)
        if large_host_groups is not None:
            return large_host_groups

        try:
            clustered_samples, _layout_ids = self._web_bindings.cluster_html_struct(
                samples,
                threshold=self.layout_cluster_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper layout clustering failed for host {}: {}", host_key, exc)
            return []

        if not clustered_samples:
            return []
        return self._build_clustered_host_groups(df, host_key, clustered_samples)

    def _build_large_host_groups(
        self, df: pd.DataFrame, host_key: str, samples: list[dict[str, Any]]
    ) -> list[list[int]] | None:
        if not self.layout_template_max_exact_host_pages or len(samples) <= self.layout_template_max_exact_host_pages:
            return None

        groups: list[list[int]] = []
        if self.layout_template_large_host_mode == "feature_hash":
            fingerprint_fn = lambda sample: _layout_feature_fingerprint(sample.get("feature"))  # noqa: E731
        elif self.layout_template_large_host_mode == "dom_path_hash":
            fingerprint_fn = lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or ""))  # noqa: E731
        else:
            logger.debug(
                "Dripper layout host={} rows={} exceeds max_exact_host_pages={}; leaving standalone",
                host_key,
                len(samples),
                self.layout_template_max_exact_host_pages,
            )
            return groups
        groups.extend(self._build_fingerprint_groups(df, host_key, samples, fingerprint_fn=fingerprint_fn))
        return groups

    def _build_clustered_host_groups(
        self, df: pd.DataFrame, host_key: str, clustered_samples: list[dict[str, Any]]
    ) -> list[list[int]]:
        max_layer_n = int(
            next((s.get("max_layer_n") for s in clustered_samples if int(s.get("layout_id", -1)) >= 0), None) or 5
        )
        exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = int(sample.get("layout_id", -1))
            if layout_id < 0:
                continue
            if len(exemplars_by_layout[layout_id]) < _MAX_EXEMPLARS_PER_LAYOUT:
                exemplars_by_layout[layout_id].append(sample)

        by_layout: dict[tuple[int, str], list[int]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = self._assign_layout_by_exemplar_similarity(
                sample.get("feature"),
                exemplars_by_layout,
                max_layer_n,
            )
            if layout_id < 0:
                continue
            row_idx = int(sample["track_id"])
            signature_key = self._layout_page_signature_key(df.iloc[row_idx])
            by_layout[(layout_id, signature_key)].append(row_idx)
        groups: list[list[int]] = []
        for (layout_id, signature_key), indexes in sorted(by_layout.items()):
            if len(indexes) >= self.layout_template_min_cluster_size:
                groups.append(sorted(indexes))
                logger.debug(
                    "Dripper layout group host={} layout_id={} signature={} rows={}",
                    host_key,
                    layout_id,
                    signature_key,
                    len(indexes),
                )
        return groups

    def _build_failed_layout_fallback_groups(self, df: pd.DataFrame, indexes: list[int]) -> list[list[int]]:
        mode = self.layout_template_failed_layout_fallback_signature_mode
        if mode == "none" or len(indexes) < self.layout_template_min_cluster_size:
            return []

        children = self._split_fallback_groups_by_signature(df, [indexes], mode)
        parent_set = set(indexes)
        return [child for child in children if set(child) != parent_set]

    def _assign_layout_by_exemplar_similarity(
        self,
        feature: object,
        exemplars_by_layout: dict[int, list[dict[str, Any]]],
        max_layer_n: int,
    ) -> int:
        for layout_id, exemplars in sorted(exemplars_by_layout.items()):
            for exemplar in exemplars:
                try:
                    score = self._web_bindings.similarity(feature, exemplar.get("feature"), max_layer_n)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper layout similarity failed for layout {}: {}", layout_id, exc)
                    continue
                if score is not None and score >= self.layout_cluster_threshold:
                    return layout_id
        return -2

    def _build_fingerprint_groups(
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
        *,
        fingerprint_fn: Callable[[dict[str, Any]], str],
    ) -> list[list[int]]:
        by_fingerprint: dict[str, list[int]] = defaultdict(list)
        for sample in samples:
            by_fingerprint[fingerprint_fn(sample)].append(int(sample["track_id"]))

        groups: list[list[int]] = []
        for fingerprint, indexes in sorted(by_fingerprint.items(), key=lambda item: (min(item[1]), item[0])):
            by_signature: dict[str, list[int]] = defaultdict(list)
            for row_idx in indexes:
                signature_key = self._layout_page_signature_key(df.iloc[row_idx])
                by_signature[signature_key].append(row_idx)
            for signature_key, signature_indexes in sorted(by_signature.items()):
                if len(signature_indexes) < self.layout_template_min_cluster_size:
                    continue
                groups.append(sorted(signature_indexes))
                logger.debug(
                    "Dripper layout fingerprint group host={} signature={} rows={} fingerprint_chars={}",
                    host_key,
                    signature_key,
                    len(signature_indexes),
                    len(fingerprint),
                )
        return groups

    def _layout_page_signature_key(self, row: pd.Series) -> str:
        return _layout_page_signature_key(
            row.get(self.url_col) if self.url_col else None,
            row.get(self.item_count_col),
            self.layout_page_signature_mode,
        )

    def _split_fallback_groups_by_signature(
        self,
        df: pd.DataFrame,
        groups: list[list[int]],
        mode: str,
    ) -> list[list[int]]:
        split_groups: list[list[int]] = []
        for group in groups:
            low_card_query_keys: set[str] = set()
            if "url_low_card_query_shape" in mode and self.url_col:
                low_card_query_keys = _low_card_query_value_keys(
                    [df.iloc[row_idx].get(self.url_col) for row_idx in group]
                )
            by_signature: dict[str, list[int]] = defaultdict(list)
            use_low_card = "url_low_card_query_shape" in mode
            for row_idx in group:
                row = df.iloc[row_idx]
                url = row.get(self.url_col) if self.url_col else None
                if use_low_card:
                    signature_key = _layout_page_signature_key_with_low_card_queries(
                        url, row.get(self.item_count_col), mode, low_card_query_keys
                    )
                else:
                    signature_key = _layout_page_signature_key(url, row.get(self.item_count_col), mode)
                by_signature[signature_key].append(row_idx)
            for _signature, indexes in sorted(by_signature.items(), key=lambda item: (min(item[1]), item[0])):
                if len(indexes) >= self.layout_template_min_cluster_size:
                    split_groups.append(sorted(indexes))
        return split_groups

    async def _process_layout_group_with_status(
        self,
        ctx: _LayoutProcessContext,
        indexes: list[int],
        cluster_id: str,
        *,
        emit_failure_fallback: bool,
    ) -> _LayoutGroupOutcome:
        run = _LayoutGroupRun(
            ctx=ctx, indexes=indexes, cluster_id=cluster_id, emit_failure_fallback=emit_failure_fallback
        )
        df = ctx.df
        group_started = time.perf_counter()
        representative_idx, mapping_data, results, mapping_failures = await self._infer_representative_candidates(run)

        if mapping_data is None:
            warning = "layout template mapping failed"
            if mapping_failures:
                warning = f"{warning}: {'; '.join(mapping_failures[:3])}"
            return await self._handle_mapping_failure(run, results, warning)

        if representative_idx is None:
            msg = "representative_idx must not be None"
            raise RuntimeError(msg)
        sibling_indexes = [idx for idx in indexes if idx not in results]
        validation_rows = self._effective_validation_rows(len(indexes))
        validation_indexes = _select_validation_indexes(
            df,
            sibling_indexes,
            validation_rows,
            (self.url_col, self.item_count_col),
            signature_mode=self.layout_template_validation_signature_mode,
        )
        validation_index_set = set(validation_indexes)
        remaining_indexes = [idx for idx in sibling_indexes if idx not in validation_index_set]
        validation = _ValidationOutcome()
        if validation_indexes:
            validation = await self._run_validation_rows_async(run, validation_indexes, mapping_data, results)
            if validation.failed:
                logger.debug("Dripper layout validation failed for {}: {}", cluster_id, validation.error)
                if not emit_failure_fallback:
                    return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=validation.error)

        sibling_outcome = await self._propagate_sibling_rows_async(
            run, remaining_indexes, mapping_data, results, validation
        )
        if sibling_outcome is not None:
            return sibling_outcome
        logger.info(
            "Dripper layout-template group {} rows={} representative={} propagated={} fallback_llm={} elapsed_s={:.3f}",
            cluster_id,
            len(indexes),
            representative_idx,
            sum(result.layout_propagated for result in results.values()),
            sum(result.layout_fallback_llm for result in results.values()),
            time.perf_counter() - group_started,
        )
        return _LayoutGroupOutcome(results=results)

    async def _infer_representative_candidates(
        self, run: _LayoutGroupRun
    ) -> tuple[int | None, dict[str, Any] | None, dict[int, _LayoutTemplateRowResult], list[str]]:
        ctx = run.ctx
        df = ctx.df
        cluster_id = run.cluster_id
        representative_indexes = self._select_representative_indexes(df, run.indexes)
        representative_idx: int | None = None
        mapping_data: dict[str, Any] | None = None
        candidate_results: dict[int, _LayoutTemplateRowResult] = {}
        mapping_failures: list[str] = []

        for candidate_idx in representative_indexes:
            candidate_result, candidate_mapping = await self._infer_representative_and_mapping(
                df.iloc[candidate_idx], ctx.semaphore, cluster_id, ctx.inference_cache, ctx.inference_cache_lock
            )
            candidate_results[candidate_idx] = candidate_result
            if candidate_mapping is not None:
                representative_idx = candidate_idx
                mapping_data = candidate_mapping
                break
            mapping_failures.append(
                f"{candidate_idx}:{candidate_result.primary_error or candidate_result.warning or 'mapping failed'}"
            )

        results: dict[int, _LayoutTemplateRowResult] = {}
        mapping_json_for_representative = (
            json.dumps(mapping_data, default=str)
            if self.layout_template_defer_propagation and mapping_data is not None
            else ""
        )
        for candidate_idx, candidate_result in candidate_results.items():
            is_representative = candidate_idx == representative_idx
            results[candidate_idx] = replace(
                candidate_result,
                layout_cluster=cluster_id,
                layout_representative=is_representative,
                layout_fallback_llm=not is_representative,
                layout_mapping_json=mapping_json_for_representative if is_representative else "",
            )
        return representative_idx, mapping_data, results, mapping_failures

    async def _handle_mapping_failure(
        self,
        run: _LayoutGroupRun,
        results: dict[int, _LayoutTemplateRowResult],
        warning: str,
    ) -> _LayoutGroupOutcome:
        df = run.ctx.df
        cluster_id = run.cluster_id
        if not run.emit_failure_fallback:
            return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=warning)
        fallback_indexes = [idx for idx in run.indexes if idx not in results]
        if self.layout_template_defer_fallback_llm:
            for idx in fallback_indexes:
                results[idx] = self._defer_row(
                    df.iloc[idx], primary_error=warning, layout_cluster=cluster_id, layout_fallback_llm=True
                )
        elif self.layout_template_fallback_llm:
            fallback_results = await asyncio.gather(
                *(
                    self._infer_and_postprocess_row(
                        df.iloc[idx], self._fallback_infer_context(run.ctx, cluster_id, warning)
                    )
                    for idx in fallback_indexes
                )
            )
            results.update(zip(fallback_indexes, fallback_results, strict=True))
        else:
            for idx in fallback_indexes:
                results[idx] = replace(
                    self._fallback_row(df.iloc[idx], primary_error=warning), layout_cluster=cluster_id
                )
        return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=warning)

    async def _run_validation_rows_async(
        self,
        run: _LayoutGroupRun,
        validation_indexes: list[int],
        mapping_data: dict[str, Any],
        results: dict[int, _LayoutTemplateRowResult],
    ) -> _ValidationOutcome:
        df = run.ctx.df
        cluster_id = run.cluster_id
        validation_propagated, validation_llm_results = await asyncio.gather(
            asyncio.gather(
                *(
                    self._propagate_layout_template_async(
                        df.iloc[idx], mapping_data, cluster_id, run.ctx.propagation_semaphore
                    )
                    for idx in validation_indexes
                )
            ),
            asyncio.gather(
                *(
                    self._infer_and_postprocess_row(
                        df.iloc[idx],
                        self._fallback_infer_context(run.ctx, cluster_id, "layout template validation LLM"),
                    )
                    for idx in validation_indexes
                )
            ),
        )
        validation = _ValidationOutcome()
        for idx, propagated, llm_result in zip(
            validation_indexes, validation_propagated, validation_llm_results, strict=True
        ):
            results[idx] = llm_result
            content_f1 = _token_f1(propagated.main_content, llm_result.main_content)
            failure_reasons = []
            if propagated.error:
                failure_reasons.append(f"propagation_error={propagated.error[:160]}")
            if content_f1 < self.layout_template_validation_min_content_f1:
                failure_reasons.append(f"content_f1={content_f1:.3f}")
            if failure_reasons:
                validation = _ValidationOutcome(
                    failed=True,
                    error=f"layout template validation failed: {' '.join(failure_reasons)} min={self.layout_template_validation_min_content_f1:.3f}",
                )
        return validation

    async def _propagate_sibling_rows_async(
        self,
        run: _LayoutGroupRun,
        remaining_indexes: list[int],
        mapping_data: dict[str, Any],
        results: dict[int, _LayoutTemplateRowResult],
        validation: _ValidationOutcome,
    ) -> _LayoutGroupOutcome | None:
        df = run.ctx.df
        cluster_id = run.cluster_id
        propagated_results: list[_LayoutTemplateRowResult] = []
        if remaining_indexes and not validation.failed:
            if self.layout_template_defer_propagation:
                for idx in remaining_indexes:
                    results[idx] = _LayoutTemplateRowResult(
                        layout_cluster=cluster_id,
                        layout_pending_propagation=True,
                        layout_finalized=False,
                    )
                return _LayoutGroupOutcome(results=results)
            propagated_results = await asyncio.gather(
                *(
                    self._propagate_layout_template_async(
                        df.iloc[idx], mapping_data, cluster_id, run.ctx.propagation_semaphore
                    )
                    for idx in remaining_indexes
                )
            )

        fallback_tasks: list[Any] = []
        fallback_indexes: list[int] = []
        for i, idx in enumerate(remaining_indexes):
            if validation.failed:
                fallback = self._apply_validation_failed_row(run, idx, results, validation.error)
            else:
                fallback = self._apply_propagated_row(run, idx, propagated_results[i], results)
            if fallback is not None:
                fallback_indexes.append(idx)
                fallback_tasks.append(fallback)
        if fallback_tasks:
            fallback_results = await asyncio.gather(*fallback_tasks)
            results.update(zip(fallback_indexes, fallback_results, strict=True))
        return None

    def _apply_validation_failed_row(
        self,
        run: _LayoutGroupRun,
        idx: int,
        results: dict[int, _LayoutTemplateRowResult],
        error: str,
    ) -> Awaitable[_LayoutTemplateRowResult] | None:
        df = run.ctx.df
        cluster_id = run.cluster_id
        if self.layout_template_defer_fallback_llm:
            results[idx] = self._defer_row(
                df.iloc[idx], primary_error=error, layout_cluster=cluster_id, layout_fallback_llm=True
            )
            return None
        if self.layout_template_fallback_llm:
            return self._infer_and_postprocess_row(
                df.iloc[idx], self._fallback_infer_context(run.ctx, cluster_id, error)
            )
        results[idx] = replace(self._fallback_row(df.iloc[idx], primary_error=error), layout_cluster=cluster_id)
        return None

    def _apply_propagated_row(
        self,
        run: _LayoutGroupRun,
        idx: int,
        propagated: _LayoutTemplateRowResult,
        results: dict[int, _LayoutTemplateRowResult],
    ) -> Awaitable[_LayoutTemplateRowResult] | None:
        df = run.ctx.df
        cluster_id = run.cluster_id
        if propagated.error and self.layout_template_defer_fallback_llm:
            results[idx] = self._defer_row(
                df.iloc[idx], primary_error=propagated.error, layout_cluster=cluster_id, layout_fallback_llm=True
            )
            return None
        if propagated.error and self.layout_template_fallback_llm:
            return self._infer_and_postprocess_row(
                df.iloc[idx], self._fallback_infer_context(run.ctx, cluster_id, propagated.error)
            )
        results[idx] = propagated
        return None

    def _fallback_infer_context(
        self, ctx: _LayoutProcessContext, cluster_id: str, primary_error: str
    ) -> _InferContext:
        return _InferContext(
            semaphore=ctx.semaphore,
            cache=ctx.inference_cache,
            cache_lock=ctx.inference_cache_lock,
            layout_cluster=cluster_id,
            layout_fallback_llm=True,
            primary_error=primary_error,
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

    async def _propagate_layout_template_async(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
    ) -> _LayoutTemplateRowResult:
        async with semaphore:
            return await asyncio.to_thread(self._propagate_layout_template, row, mapping_data, cluster_id)

    def _select_representative_indexes(self, df: pd.DataFrame, indexes: list[int]) -> list[int]:
        selected = self._select_representative_index(df, indexes)
        representative_indexes = [selected]
        if self.layout_template_representative_candidates <= 1:
            return representative_indexes

        remaining_indexes = [idx for idx in indexes if idx != selected]
        representative_indexes.extend(
            _select_validation_indexes(
                df,
                remaining_indexes,
                self.layout_template_representative_candidates - 1,
                (self.url_col, self.item_count_col),
            )
        )
        return representative_indexes

    def _select_representative_index(self, df: pd.DataFrame, indexes: list[int]) -> int:
        candidates = [
            {"track_id": str(idx), "html": _coerce_html(df.iloc[idx].get(self.html_col, ""))} for idx in indexes
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

    async def _infer_representative_and_mapping(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        cluster_id: str,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> tuple[_LayoutTemplateRowResult, dict[str, Any] | None]:
        inference_result = await self._infer_row_cached(row, semaphore, inference_cache, inference_cache_lock)
        started = time.perf_counter()
        if inference_result.primary_error:
            return self._postprocess_error_row(row, inference_result, _InferContext(layout_cluster=cluster_id)), None

        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        case = self._build_case(row)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=inference_result.raw_response)
            case = self._bindings.parse_result(case)
            webkit_response = _labels_to_webkit_response(getattr(case.parse_result, "item_label", {}))
            case = self._bindings.extract_main_html_single(case)
            post_result = self._convert_case(case)
            mapping_data = self._web_bindings.map_parser_cls({}).parse(
                {"typical_raw_tag_html": mapped_html, "typical_raw_html": html_text, "llm_response": webkit_response}
            )
            mapping_failure_reason = (
                "typical_main_html_success=false"
                if self.layout_template_require_success and mapping_data.get("typical_main_html_success") is False
                else ""
            )
            if mapping_failure_reason:
                mapping_data = None
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper representative mapping failed: {}", primary_error)
            fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
            return (
                _LayoutTemplateRowResult(
                    **_inference_token_fields(inference_result),
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
            mapping_data["_dripper_representative_content_len"] = len(str(post_result.main_content or ""))
        return (
            _LayoutTemplateRowResult(
                **_inference_token_fields(inference_result),
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

    def _propagate_layout_template(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
    ) -> _LayoutTemplateRowResult:
        started = time.perf_counter()
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        use_mapped_item_ids = (
            self.layout_template_propagation_target == "mapped_item_ids" and "_item_id" in mapped_html
        )
        html_source = mapped_html if use_mapped_item_ids else html_text
        try:
            task_data = dict(mapping_data) | {
                "html_source": html_source,
                "dynamic_id_enable": True,
                "dynamic_classid_enable": True,
                "more_noise_enable": self.layout_template_more_noise_enable,
                "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
            }
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
            if self.layout_template_require_success and parts.get("main_html_success") is False:
                raise RuntimeError(f"layout propagation similarity below threshold: {parts.get('main_html_sim')}")  # noqa: TRY301, EM102
            if self.layout_template_min_main_html_sim is not None:
                main_html_sim = _coerce_optional_float(parts.get("main_html_sim"))
                if main_html_sim is not None and main_html_sim < self.layout_template_min_main_html_sim:
                    msg = f"layout propagation main_html_sim {main_html_sim:.3f} below {self.layout_template_min_main_html_sim:.3f}"
                    raise RuntimeError(msg)  # noqa: TRY301
            main_html = str(parts.get("main_html_body") or "")
            raw_response = ""
            if use_mapped_item_ids:
                all_item_ids = _item_ids_in_html(mapped_html)
                main_item_ids = set(_item_ids_in_html(main_html))
                if not all_item_ids:
                    raise RuntimeError("layout propagation target mapped HTML has no item ids")  # noqa: TRY301, EM101
                if not main_item_ids:
                    raise RuntimeError("layout propagation produced no target item ids")  # noqa: TRY301, EM101
                selected_item_ratio = len(main_item_ids) / len(all_item_ids)
                if (
                    self.layout_template_max_selected_item_ratio is not None
                    and selected_item_ratio > self.layout_template_max_selected_item_ratio
                ):
                    msg = f"layout propagation selected item ratio {selected_item_ratio:.3f} exceeds {self.layout_template_max_selected_item_ratio:.3f}"
                    raise RuntimeError(msg)  # noqa: TRY301
                raw_response = _item_id_response(all_item_ids, main_item_ids)
                post_result = self._postprocess_raw_response(row, raw_response)
            else:
                post_result = self._convert_main_html(row, main_html)
            content_ratio_error = self._propagated_content_length_ratio_error(post_result.main_content, mapping_data)
            if content_ratio_error:
                raise RuntimeError(content_ratio_error)  # noqa: TRY301
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
            return f"layout propagation content length ratio {ratio:.3f} below {self.layout_template_min_content_length_ratio:.3f}"
        if (
            self.layout_template_max_content_length_ratio is not None
            and ratio > self.layout_template_max_content_length_ratio
        ):
            return f"layout propagation content length ratio {ratio:.3f} exceeds {self.layout_template_max_content_length_ratio:.3f}"
        return ""

    async def _infer_and_postprocess_row(
        self,
        row: pd.Series,
        infer_ctx: _InferContext,
    ) -> _LayoutTemplateRowResult:
        semaphore = infer_ctx.semaphore
        if infer_ctx.cache is None or infer_ctx.cache_lock is None:
            inference_result = await self._infer_row(row, semaphore)
        else:
            inference_result = await self._infer_row_cached(row, semaphore, infer_ctx.cache, infer_ctx.cache_lock)
        if inference_result.primary_error:
            merged_ctx = replace(
                infer_ctx, primary_error=_append_warning(infer_ctx.primary_error, inference_result.primary_error)
            )
            return self._postprocess_error_row(row, inference_result, merged_ctx)

        post_result = self._postprocess_raw_response(row, inference_result.raw_response)
        return _LayoutTemplateRowResult(
            **_inference_token_fields(inference_result),
            main_html=post_result.main_html,
            main_content=post_result.main_content,
            postprocess_time_s=post_result.postprocess_time_s,
            error=post_result.error,
            warning=_append_warning(infer_ctx.primary_error, post_result.warning),
            layout_cluster=infer_ctx.layout_cluster,
            layout_fallback_llm=infer_ctx.layout_fallback_llm,
            layout_standalone_llm=infer_ctx.layout_standalone_llm,
        )

    async def _infer_row(self, row: pd.Series, semaphore: asyncio.Semaphore) -> _DripperInferenceResult:
        prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
        row_max_tokens = _coerce_usage_int(row.get(self.request_max_tokens_col, 0))
        return await self._infer_prompt(prompt, row_max_tokens, semaphore)

    async def _infer_row_cached(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> _DripperInferenceResult:
        prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
        row_max_tokens = _coerce_usage_int(row.get(self.request_max_tokens_col, 0))
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")

        key = (prompt, row_max_tokens)
        async with inference_cache_lock:
            task = inference_cache.get(key)
            owns_request = task is None
            if task is None:
                task = asyncio.create_task(self._infer_prompt(prompt, row_max_tokens, semaphore))
                inference_cache[key] = task

        result = await task
        if owns_request:
            return result
        return replace(result, inference_time_s=0.0, prompt_tokens=0, completion_tokens=0, total_tokens=0)

    async def _infer_prompt(
        self,
        prompt: str,
        row_max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> _DripperInferenceResult:
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")
        async with semaphore:
            started = time.perf_counter()
            try:
                generation_config = self.generation_config or GenerationConfig()
                if row_max_tokens > 0 and generation_config.max_tokens != row_max_tokens:
                    generation_config = replace(generation_config, max_tokens=row_max_tokens)
                generation_config = _with_structured_output_config(
                    generation_config, prompt, self.structured_output_mode
                )
                raw_response, prompt_tokens, completion_tokens, total_tokens = await _query_dripper_model(
                    self.client, self.model_name, [{"role": "user", "content": prompt}], generation_config
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                logger.debug("Dripper inference failed; postprocess stage will apply fallback: {}", error)
                return _DripperInferenceResult(
                    inference_time_s=time.perf_counter() - started,
                    primary_error=error,
                    warning=error,
                )
            return _DripperInferenceResult(
                raw_response=raw_response,
                inference_time_s=time.perf_counter() - started,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    def _postprocess_raw_response(self, row: pd.Series, raw_response: str) -> _DripperPostResult:
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

    def _postprocess_error_row(
        self,
        row: pd.Series,
        inference_result: _DripperInferenceResult,
        ctx: _InferContext,
    ) -> _LayoutTemplateRowResult:
        primary_error = _append_warning(ctx.primary_error, inference_result.primary_error)
        fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
        return _LayoutTemplateRowResult(
            **_inference_token_fields(inference_result),
            main_html=fallback_result.main_html,
            main_content=fallback_result.main_content,
            postprocess_time_s=fallback_result.postprocess_time_s,
            error=fallback_result.error,
            warning=fallback_result.warning,
            primary_error=primary_error,
            layout_cluster=ctx.layout_cluster,
            layout_fallback_llm=ctx.layout_fallback_llm,
            layout_standalone_llm=ctx.layout_standalone_llm,
        )

    def _fallback_row(self, row: pd.Series, *, primary_error: str = "") -> _LayoutTemplateRowResult:
        result = self._fallback_and_convert(
            row,
            primary_error=_append_warning(primary_error, str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")),
        )
        return _LayoutTemplateRowResult(
            main_html=result.main_html,
            main_content=result.main_content,
            postprocess_time_s=result.postprocess_time_s,
            error=result.error,
            warning=result.warning,
            primary_error=primary_error,
        )

    def _defer_row(
        self,
        row: pd.Series,
        *,
        primary_error: str = "",
        layout_cluster: str = "",
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
    ) -> _LayoutTemplateRowResult:
        needs_llm = bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
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

    def _build_case(self, row: pd.Series) -> object:
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
        case = self._build_case(row)
        case.output_data = self._bindings.output_cls(main_html=main_html)
        return self._convert_case(case)

    def _convert_case(self, case: object, *, warning: str = "") -> _DripperPostResult:
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
        return _apply_fallback_extraction(self._bindings, self._fallback_handler, case, primary_error)


# ---------------------------------------------------------------------------
# Layout-template private helpers (only used by DripperHTMLLayoutTemplateStage)
# ---------------------------------------------------------------------------


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _parse_url(value: object) -> tuple[str, object]:
    """Return (raw_text, ParseResult) for a URL column value, or ('', None) if missing/empty."""
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return "", None
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    return text, parsed


def _url_host_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _layout_page_signature_key(url_value: object, item_count_value: object, mode: str) -> str:
    return _layout_page_signature_key_with_low_card_queries(url_value, item_count_value, mode, set())


def _layout_page_signature_key_with_low_card_queries(
    url_value: object,
    item_count_value: object,
    mode: str,
    low_card_query_keys: set[str],
) -> str:
    if not mode or mode == "none":
        return ""
    parts: list[str] = []
    if "url_low_card_query_shape" in mode:
        parts.append(f"url={_url_low_card_query_shape_key(url_value, low_card_query_keys)}")
    elif "url_semantic_shape" in mode:
        parts.append(f"url={_url_semantic_shape_key(url_value)}")
    elif "url_shape" in mode:
        parts.append(f"url={_url_shape_key(url_value)}")
    if "item_count_exact" in mode:
        parts.append(f"items={_coerce_item_count(item_count_value)}")
    elif "item_count_bucket" in mode:
        parts.append(f"items={_item_count_bucket(item_count_value)}")
    return "|".join(parts)


def _url_shape_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def _url_low_card_query_shape_key(value: object, low_card_query_keys: set[str]) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]

    include_all_query_values = bool(parsed.query) and not low_card_query_keys
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if (
            include_all_query_values
            or lowered_key in low_card_query_keys
            or lowered_key in _LAYOUT_EXACT_QUERY_VALUE_KEYS
        ):
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _url_semantic_shape_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    normalized_segments = [_normalize_semantic_url_path_segment(segment) for segment in raw_segments]
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.lower()
        if lowered_key in _LAYOUT_SEMANTIC_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={_normalize_semantic_url_query_value(query_value)}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_semantic_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        stem, extension = segment.rsplit(".", 1)
        segment = stem
        suffix = f".{extension}"
    if (
        segment.isdigit()
        or _LAYOUT_RE_MD5.fullmatch(segment)
        or _LAYOUT_RE_SHA1.fullmatch(segment)
        or _LAYOUT_RE_UUID.fullmatch(segment)
        or _LAYOUT_RE_TIMESTAMP.fullmatch(segment)
    ):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _normalize_semantic_url_query_value(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if (
        text.isdigit()
        or _LAYOUT_RE_MD5.fullmatch(text)
        or _LAYOUT_RE_SHA1.fullmatch(text)
        or _LAYOUT_RE_UUID.fullmatch(text)
        or _LAYOUT_RE_TIMESTAMP.fullmatch(text)
    ):
        return "#num"
    return text


def _item_count_bucket(value: object) -> str:
    count = _coerce_item_count(value)
    if count <= 0:
        return "0"
    for threshold, label in _ITEM_COUNT_BUCKET_THRESHOLDS:
        if count <= threshold:
            return str(count) if label is None else label
    return "129+"


def _coerce_item_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _coerce_positive_int(value: object) -> int:
    return max(0, _coerce_item_count(value))


def _labels_to_webkit_response(labels: object) -> dict[str, int]:
    if not isinstance(labels, dict):
        return {}
    response: dict[str, int] = {}
    for item_id, label in labels.items():
        normalized = str(label).strip().lower()
        response[f"item_id {item_id}"] = 1 if normalized in {"main", "1", "true"} else 0
    return response


def _item_id_response(all_item_ids: list[str], main_item_ids: set[str]) -> str:
    labels = {item_id: ("main" if item_id in main_item_ids else "other") for item_id in all_item_ids}
    if all(item_id.isdigit() for item_id in all_item_ids):
        return "".join(f"{item_id}{label}" for item_id, label in labels.items())
    return json.dumps(labels, ensure_ascii=False, separators=(",", ":"))


def _layout_feature_fingerprint(feature: object) -> str:
    if not isinstance(feature, dict):
        return ""

    def normalize_part(part: str) -> dict[str, list[tuple[str, int]]]:
        raw_layers = feature.get(part, {})
        if not isinstance(raw_layers, dict):
            return {}
        normalized: dict[str, list[tuple[str, int]]] = {}
        for layer, values in raw_layers.items():
            if not isinstance(values, list):
                continue
            counts = Counter(str(value) for value in values)
            normalized[str(layer)] = sorted(counts.items())
        return normalized

    payload = {"tags": normalize_part("tags"), "attrs": normalize_part("attrs")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_dynamic_attribute(value: str) -> str:
    lowered = value.strip().lower()
    for pattern, label in (
        (_LAYOUT_RE_MD5, "[MD5]"),
        (_LAYOUT_RE_SHA1, "[SHA1]"),
        (_LAYOUT_RE_UUID, "[UUID]"),
        (_LAYOUT_RE_TIMESTAMP, "[TIMESTAMP]"),
    ):
        if pattern.fullmatch(lowered):
            return label
    return _LAYOUT_RE_NUM.sub("", lowered)


def _normalize_attr_tokens(value: str | None) -> str:
    if not value:
        return ""
    tokens = value.split()
    if len(tokens) > 1:
        normalized = [token.lower() for token in tokens if not _LAYOUT_RE_NUM.search(token)]
    else:
        normalized = [_normalize_dynamic_attribute(tokens[0])] if tokens else []
    return " ".join(token for token in normalized if token)


def _walk_dom_element(element: object) -> object:
    raw_tag = getattr(element, "tag", None)
    if not isinstance(raw_tag, str):
        return None
    tag = raw_tag.lower()
    if tag in _LAYOUT_TAGS_TO_IGNORE:
        return None
    attrs: list[tuple[str, str]] = []
    if tag not in _LAYOUT_TAGS_IGNORE_ATTR:
        class_attr = _normalize_attr_tokens(element.get("class"))
        id_attr = _normalize_attr_tokens(element.get("id"))
        if class_attr:
            attrs.append(("class", class_attr))
        if id_attr:
            attrs.append(("id", id_attr))
    children = [child for child in (_walk_dom_element(child) for child in element) if child is not None]
    return [tag, attrs, children]


def _layout_dom_path_fingerprint(html_text: str) -> str:
    try:
        from lxml.html import HTMLParser, fromstring
    except ModuleNotFoundError:
        return ""

    try:
        parser = HTMLParser(collect_ids=False, encoding="utf-8", remove_comments=True, remove_pis=True)
        root = fromstring(html_text.encode("utf-8", errors="ignore"), parser=parser)
        body_nodes = root.xpath("//body")
        root = body_nodes[0] if body_nodes else root
    except Exception:  # noqa: BLE001
        return ""

    return json.dumps(_walk_dom_element(root), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_response_regex(item_ids: list[str]) -> str:
    item_pattern = "".join(f"{re.escape(item_id)}(main|other)" for item_id in item_ids)
    return f"<answer>\\s*{item_pattern}\\s*</answer>"


def _token_f1(candidate: object, reference: object) -> float:
    candidate_tokens = Counter(_TOKEN_RE.findall(str(candidate or "").lower()))
    reference_tokens = Counter(_TOKEN_RE.findall(str(reference or "").lower()))
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((candidate_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _select_by_signature(
    df: pd.DataFrame,
    indexes: list[int],
    *,
    signature_mode: str,
    state: _SelectorState,
) -> bool:
    """Fill state from signature-grouped indexes. Returns True if count reached."""
    url_col = state.url_col
    item_count_col = state.item_count_col
    low_card_query_keys: set[str] = set()
    if "url_low_card_query_shape" in signature_mode and url_col:
        low_card_query_keys = _low_card_query_value_keys([df.iloc[idx].get(url_col) for idx in indexes])
    by_signature: dict[str, list[int]] = defaultdict(list)
    for idx in indexes:
        row = df.iloc[idx]
        signature_key = _layout_page_signature_key_with_low_card_queries(
            row.get(url_col) if url_col else None,
            row.get(item_count_col) if item_count_col in row else None,
            signature_mode,
            low_card_query_keys,
        )
        by_signature[signature_key].append(idx)
    signature_groups = sorted(
        by_signature.values(),
        key=lambda group: (-len(group), _validation_sample_key(df.iloc[group[0]], group[0], url_col, item_count_col)),
    )
    for group in signature_groups:
        for idx in _select_validation_indexes(df, sorted(group), 1, (url_col, item_count_col), signature_mode="none"):
            state.add(idx)
            break
        if state.is_full():
            return True
    return False


def _select_by_url(
    df: pd.DataFrame,
    indexes: list[int],
    *,
    state: _SelectorState,
) -> None:
    url_col = state.url_col
    count = state.count
    query_value_rows: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx in indexes:
        url_text = str(df.iloc[idx].get(url_col) or "")
        for key, value in _validation_query_values(url_text):
            query_value_rows[key].append((value, idx))
    for key in sorted(query_value_rows):
        entries = sorted(query_value_rows[key])
        query_positions = _QUERY_POSITIONS_HIGH if count >= _QUERY_POSITIONS_THRESHOLD else _QUERY_POSITIONS_LOW
        for position in _spread_positions(len(entries), min(count, query_positions)):
            state.add(entries[position][1])
        if state.is_full():
            return

    url_sorted = sorted(indexes, key=lambda idx: (str(df.iloc[idx].get(url_col) or ""), idx))
    for position in _spread_positions(len(url_sorted), count):
        state.add(url_sorted[position])
        if state.is_full():
            return


def _select_validation_indexes(
    df: pd.DataFrame,
    indexes: list[int],
    count: int,
    cols: _ColSpec,
    *,
    signature_mode: str = "none",
) -> list[int]:
    url_col, item_count_col = cols
    if count <= 0 or not indexes:
        return []
    if count >= len(indexes):
        return list(indexes)
    if count == 1:
        return [indexes[-1]]

    state = _SelectorState(
        selected=[], selected_set=set(), count=count, url_col=url_col, item_count_col=item_count_col
    )

    if (
        signature_mode
        and signature_mode != "none"
        and _select_by_signature(df, indexes, signature_mode=signature_mode, state=state)
    ):
        return sorted(state.selected)

    state.add(indexes[0])
    state.add(indexes[-1])

    item_sorted = sorted(indexes, key=lambda idx: (_coerce_item_count(df.iloc[idx].get(item_count_col)), idx))
    state.add(item_sorted[0])
    state.add(item_sorted[-1])

    if url_col:
        _select_by_url(df, indexes, state=state)
        if state.is_full():
            return sorted(state.selected)

    remaining = [idx for idx in indexes if idx not in state.selected_set]
    remaining.sort(key=lambda idx: _validation_sample_key(df.iloc[idx], idx, url_col, item_count_col))
    for idx in remaining:
        state.add(idx)
        if state.is_full():
            break
    return sorted(state.selected)


def _spread_positions(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(slot * (length - 1) / (count - 1)) for slot in range(count)})


def _validation_query_values(url_text: str) -> list[tuple[str, str]]:
    _text, parsed = _parse_url(url_text)
    if parsed is None:
        return []
    return [
        (key.strip().lower(), value.strip().lower())
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.strip()
    ]


def _low_card_query_value_keys(url_values: list[Any], max_distinct: int = 16) -> set[str]:
    values_by_key: dict[str, set[str]] = defaultdict(set)
    for url_value in url_values:
        url_text = "" if _is_missing(url_value) else str(url_value)
        for key, value in _validation_query_values(url_text):
            values_by_key[key].add(value)
    return {key for key, values in values_by_key.items() if 1 < len(values) <= max_distinct}


def _validation_sample_key(
    row: pd.Series,
    row_index: int,
    url_col: str | None,
    item_count_col: str,
) -> tuple[int, int]:
    url_text = str(row.get(url_col) or "") if url_col else ""
    item_count = str(row.get(item_count_col) or "")
    payload = f"{url_text}\0{item_count}\0{row_index}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False), row_index


# ---------------------------------------------------------------------------
# Layout-template constants (only used within this module)
# ---------------------------------------------------------------------------

# XML character range constants
_XML_CHAR_SINGLE = {0x09, 0x0A, 0x0D}
_XML_CHAR_RANGE_1_LO = 0x20
_XML_CHAR_RANGE_1_HI = 0xD7FF
_XML_CHAR_RANGE_2_LO = 0xE000
_XML_CHAR_RANGE_2_HI = 0xFFFD
_XML_CHAR_RANGE_3_LO = 0x10000
_XML_CHAR_RANGE_3_HI = 0x10FFFF

# Item count bucket thresholds: (upper_bound, label) where label=None means str(count)
_ITEM_COUNT_BUCKET_THRESHOLDS = [(8, None), (16, "9-16"), (32, "17-32"), (64, "33-64"), (128, "65-128")]

# Query position constants for validation index selection
_QUERY_POSITIONS_THRESHOLD = 8
_QUERY_POSITIONS_HIGH = 4
_QUERY_POSITIONS_LOW = 3

# Maximum exemplars per layout cluster when building exemplar sets
_MAX_EXEMPLARS_PER_LAYOUT = 3

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LAYOUT_PAGE_SIGNATURE_MODES = {
    "none",
    "url_shape",
    "url_low_card_query_shape",
    "url_semantic_shape",
    "item_count_bucket",
    "item_count_exact",
    "url_shape_item_count_bucket",
    "url_shape_item_count_exact",
    "url_low_card_query_shape_item_count_bucket",
    "url_low_card_query_shape_item_count_exact",
    "url_semantic_shape_item_count_bucket",
    "url_semantic_shape_item_count_exact",
}
_LAYOUT_SEMANTIC_QUERY_VALUE_KEYS = {"hl", "lang", "language", "locale"}
_LAYOUT_EXACT_QUERY_VALUE_KEYS = {"id"}
_LAYOUT_TAGS_TO_IGNORE = {"script", "style", "meta", "link", "br", "noscript"}
_LAYOUT_TAGS_IGNORE_ATTR = {"a", "i", "b", "li", "tr", "td", "img", "p", "body"}
_LAYOUT_RE_MD5 = re.compile(r"^[0-9a-f]{32}$")
_LAYOUT_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LAYOUT_RE_UUID = re.compile(r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$")
_LAYOUT_RE_TIMESTAMP = re.compile(r"^\d{10,13}$")
_LAYOUT_RE_NUM = re.compile(r"\d+")
_LAYOUT_TEMPLATE_LARGE_HOST_MODES = {"standalone", "feature_hash", "dom_path_hash"}
_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES = {"raw_html", "mapped_item_ids"}
# Note: _STRUCTURED_OUTPUT_MODES is imported from stage.py (shared with other stages)
