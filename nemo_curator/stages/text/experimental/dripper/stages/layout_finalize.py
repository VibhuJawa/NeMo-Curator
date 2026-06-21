from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
)
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import _token_f1
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_DEFERRED_LLM_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL,
    _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
    _DRIPPER_LAYOUT_SPLIT_PLANNED_COL,
    _DRIPPER_LAYOUT_TEMPLATE_JSON_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _DripperInferenceResult,
    _LayoutTemplateRowResult,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import (
    _append_warning,
    _coerce_usage_int,
    _numeric_series_or_zero,
)
from nemo_curator.stages.text.experimental.dripper.stages.layout_template import DripperHTMLLayoutTemplateStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass(kw_only=True)
class DripperHTMLLayoutFinalizeStage(DripperHTMLLayoutTemplateStage):
    """Finalize split layout-template plans from already-inferred responses."""

    name: str = "DripperHTMLLayoutFinalizeStage"

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self._initialized = True

    def inputs(self) -> tuple[list[str], list[str]]:
        columns = [
            self.html_col,
            self.raw_response_col,
            self.preprocess_time_col,
            self.inference_time_col,
            self.warning_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
            "dripper_layout_cluster",
            "dripper_layout_representative",
            "dripper_layout_fallback_llm",
            "dripper_layout_validation_llm",
            "dripper_layout_standalone_llm",
            _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
            _DRIPPER_LAYOUT_SPLIT_PLANNED_COL,
        ]
        return ["data"], columns

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
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
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            "dripper_layout_cluster",
            "dripper_layout_representative",
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            "dripper_layout_fallback_llm",
            "dripper_layout_validation_llm",
            "dripper_layout_standalone_llm",
            _DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL,
            _DRIPPER_LAYOUT_DEFERRED_LLM_COL,
            _DRIPPER_LAYOUT_FINALIZED_COL,
            _DRIPPER_LAYOUT_TEMPLATE_JSON_COL,
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        results = self._finalize_split_results(df)
        preprocess_times = _numeric_series_or_zero(df, self.preprocess_time_col)
        inference_times = pd.Series([r.inference_time_s for r in results], index=df.index)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)

        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.raw_response_col] = [r.raw_response for r in results]
        df[self.inference_time_col] = inference_times
        df[self.postprocess_time_col] = postprocess_times
        df[self.total_time_col] = preprocess_times + inference_times + postprocess_times
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [
            _append_warning(str(existing or ""), result.warning)
            for existing, result in zip(
                df.get(self.warning_col, pd.Series([""] * len(df))).tolist(), results, strict=True
            )
        ]
        df[self.prompt_tokens_col] = [r.prompt_tokens for r in results]
        df[self.completion_tokens_col] = [r.completion_tokens for r in results]
        df[self.total_tokens_col] = [r.total_tokens for r in results]
        df["dripper_layout_cluster"] = [r.layout_cluster for r in results]
        df["dripper_layout_representative"] = [r.layout_representative for r in results]
        df["dripper_layout_propagated"] = [r.layout_propagated for r in results]
        df["dripper_layout_propagation_success"] = [r.layout_propagation_success for r in results]
        df["dripper_layout_fallback_llm"] = [r.layout_fallback_llm for r in results]
        df["dripper_layout_validation_llm"] = [r.layout_validation_llm for r in results]
        df["dripper_layout_standalone_llm"] = [r.layout_standalone_llm for r in results]
        df[_DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL] = [r.layout_finalized for r in results]
        df[_DRIPPER_LAYOUT_DEFERRED_LLM_COL] = [r.deferred_llm for r in results]
        df[_DRIPPER_LAYOUT_FINALIZED_COL] = [r.layout_finalized for r in results]
        # Additive template side-table column: non-empty JSON only on representative rows whose cluster
        # passed validation; "" (defer sentinel) everywhere else. Phase 2b reads
        # [dripper_layout_cluster, _dripper_layout_template_json] from these rows.
        df[_DRIPPER_LAYOUT_TEMPLATE_JSON_COL] = [r.template_json for r in results]
        df[_DRIPPER_NEEDS_LLM_COL] = [r.deferred_llm for r in results]
        existing_primary_errors = df[_DRIPPER_PRIMARY_ERROR_COL].astype(str).tolist()
        df[_DRIPPER_PRIMARY_ERROR_COL] = [
            _append_warning(existing_error, result.primary_error)
            for existing_error, result in zip(existing_primary_errors, results, strict=True)
        ]
        df = df.drop(
            columns=[
                col
                for col in (
                    _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
                    _DRIPPER_LAYOUT_SPLIT_PLANNED_COL,
                )
                if col in df.columns
            ]
        )

        self._log_metrics(
            {
                "layout_finalize_rows": float(len(df)),
                "layout_finalize_representative_rows": float(sum(r.layout_representative for r in results)),
                "layout_finalize_validation_llm_rows": float(sum(r.layout_validation_llm for r in results)),
                "layout_finalize_success_rows": float(sum(r.layout_propagation_success for r in results)),
                "layout_finalize_deferred_llm_rows": float(sum(r.deferred_llm for r in results)),
                "layout_finalize_finalized_rows": float(sum(r.layout_finalized for r in results)),
            }
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _finalize_split_results(self, df: pd.DataFrame) -> list[_LayoutTemplateRowResult]:
        propagation_semaphore = asyncio.Semaphore(self.layout_template_propagation_concurrency)
        return run_async_safe(lambda: self._finalize_split_results_async(df, propagation_semaphore))

    async def _finalize_split_results_async(  # noqa: C901, PLR0912, PLR0915
        self,
        df: pd.DataFrame,
        propagation_semaphore: asyncio.Semaphore,
    ) -> list[_LayoutTemplateRowResult]:
        results_by_index: dict[int, _LayoutTemplateRowResult] = {}
        cluster_groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            cluster_id = str(row.get("dripper_layout_cluster", "") or "")
            if cluster_id and bool(row.get(_DRIPPER_LAYOUT_SPLIT_PLANNED_COL, False)):
                cluster_groups[cluster_id].append(int(idx))

        for cluster_id, indexes in sorted(cluster_groups.items(), key=lambda item: (min(item[1]), item[0])):
            representative_indexes = [
                idx for idx in indexes if bool(df.iloc[idx].get("dripper_layout_representative", False))
            ]
            if not representative_indexes:
                continue
            representative_idx = representative_indexes[0]
            representative_result, mapping_data = self._representative_mapping_from_inference(
                df.iloc[representative_idx],
                self._inference_result_from_row(df.iloc[representative_idx]),
                cluster_id,
            )
            validation_indexes = [
                idx for idx in indexes if bool(df.iloc[idx].get("dripper_layout_validation_llm", False))
            ]
            pending_indexes = [
                idx for idx in indexes if bool(df.iloc[idx].get(_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL, False))
            ]

            validation_failed: bool = False
            validation_error: str = representative_result.primary_error

            if mapping_data is not None:
                validation_propagated = await asyncio.gather(
                    *(
                        self._propagate_layout_template_async(
                            df.iloc[idx],
                            mapping_data,
                            cluster_id,
                            propagation_semaphore,
                        )
                        for idx in validation_indexes
                    )
                )
                # Collect per-row validation token-F1 so the gate aggregate (configurable below) and a
                # log scan can both consume it. A hard per-row ERROR (propagation/parse failure, not a
                # low F1) is tracked separately because it MUST force failure regardless of the
                # aggregation rule -- a row that could not be propagated at all is never "averaged away".
                validation_row_f1s: list[float] = []
                validation_hard_error: bool = False
                for idx, propagated in zip(validation_indexes, validation_propagated, strict=True):
                    llm_result = replace(
                        self._postprocess_existing_llm_row(
                            df.iloc[idx],
                            cluster_id,
                            layout_fallback_llm=True,
                            layout_validation_llm=True,
                        ),
                        layout_validation_llm=True,
                    )
                    results_by_index[idx] = llm_result
                    content_f1 = _token_f1(propagated.main_content, llm_result.main_content)
                    validation_row_f1s.append(content_f1)
                    logger.info(
                        "VALIDATION_F1 cluster={} f1={:.4f} threshold={:.4f} decision={}",
                        cluster_id,
                        content_f1,
                        self.layout_template_validation_min_content_f1,
                        "accept" if content_f1 >= self.layout_template_validation_min_content_f1 else "reject",
                    )
                    failure_reasons = []
                    if propagated.error:
                        failure_reasons.append(f"propagation_error={propagated.error[:160]}")
                        validation_hard_error = True
                    if content_f1 < self.layout_template_validation_min_content_f1:
                        failure_reasons.append(f"content_f1={content_f1:.3f}")
                    if failure_reasons:
                        # Record the most recent per-row failure detail. With aggregation="min" the
                        # gate decision (set after the loop) matches the legacy "fail on any row"
                        # rule exactly, so this preserves the legacy validation_error text as well.
                        validation_error = (
                            "layout template validation failed"
                            f": {' '.join(failure_reasons)}"
                            f" min={self.layout_template_validation_min_content_f1:.3f}"
                        )
                # Apply the configurable aggregation across all valid per-row F1s. min(...) reproduces
                # the legacy "reject if ANY row < threshold" semantics byte-for-byte (min < t iff some
                # f1 < t); mean/median relax it. A hard propagation/parse error always forces failure.
                # Empty validation_row_f1s (no validation rows) leaves validation_failed untouched, so
                # behavior there is unchanged.
                if validation_row_f1s:
                    aggregate_fn = {"min": min, "mean": statistics.mean, "median": statistics.median}[
                        self.layout_template_validation_aggregation
                    ]
                    aggregate_f1 = aggregate_fn(validation_row_f1s)
                    validation_failed = validation_hard_error or (
                        aggregate_f1 < self.layout_template_validation_min_content_f1
                    )
                    if validation_failed and not validation_error:
                        validation_error = (
                            "layout template validation failed"
                            f": aggregate_{self.layout_template_validation_aggregation}_f1={aggregate_f1:.3f}"
                            f" min={self.layout_template_validation_min_content_f1:.3f}"
                        )
                    logger.info(
                        "VALIDATION_F1_ROWS cluster={} f1s={} agg={} agg_f1={:.4f} min_f1={:.4f} "
                        "threshold={:.4f} hard_error={} decision={}",
                        cluster_id,
                        [round(f1, 4) for f1 in validation_row_f1s],
                        self.layout_template_validation_aggregation,
                        aggregate_f1,
                        min(validation_row_f1s),
                        self.layout_template_validation_min_content_f1,
                        validation_hard_error,
                        "reject" if validation_failed else "accept",
                    )

            if mapping_data is None:
                validation_failed = True
                validation_error = validation_error or "layout template mapping failed"

            if validation_failed:
                # Validation gate FAILED (or mapping was None): emit the defer sentinel ("") to the
                # rep's template column so Phase 2b never replays an unvalidated template. In-block
                # members defer to the LLM exactly as before -- the side-table is purely additive.
                results_by_index[representative_idx] = replace(
                    representative_result,
                    layout_cluster=cluster_id,
                    layout_representative=True,
                    template_json="",
                )
                for idx in pending_indexes:
                    results_by_index[idx] = self._defer_row(
                        df.iloc[idx],
                        primary_error=validation_error,
                        layout_cluster=cluster_id,
                        layout_fallback_llm=True,
                        force_needs_llm=True,
                    )
                continue

            # Validation gate PASSED and mapping_data is not None: emit the JSON-serialized mapping_data
            # to the rep's template column so Phase 2b can replay propagation off-GPU with the identical
            # template. This is ADDITIVE -- the in-block propagation below is unchanged.
            results_by_index[representative_idx] = replace(
                representative_result,
                layout_cluster=cluster_id,
                layout_representative=True,
                template_json=self._serialize_template_json(mapping_data),
            )

            propagated_results = await asyncio.gather(
                *(
                    self._propagate_pending_row_async(
                        df.iloc[idx],
                        mapping_data,
                        cluster_id,
                        propagation_semaphore,
                    )
                    for idx in pending_indexes
                )
            )
            results_by_index.update(zip(pending_indexes, propagated_results, strict=True))

        for idx, row in df.iterrows():
            if idx in results_by_index:
                continue
            if bool(row.get("dripper_layout_standalone_llm", False)):
                results_by_index[idx] = self._postprocess_existing_llm_row(
                    row,
                    str(row.get("dripper_layout_cluster", "") or ""),
                    layout_standalone_llm=True,
                )
            elif bool(row.get("dripper_layout_fallback_llm", False)) or bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                results_by_index[idx] = self._postprocess_existing_llm_row(
                    row,
                    str(row.get("dripper_layout_cluster", "") or ""),
                    layout_fallback_llm=bool(row.get("dripper_layout_fallback_llm", False)),
                    layout_validation_llm=bool(row.get("dripper_layout_validation_llm", False)),
                )
            else:
                results_by_index[idx] = _LayoutTemplateRowResult(
                    raw_response=str(row.get(self.raw_response_col, "") or ""),
                    inference_time_s=float(row.get(self.inference_time_col, 0.0) or 0.0),
                    prompt_tokens=_coerce_usage_int(row.get(self.prompt_tokens_col, 0)),
                    completion_tokens=_coerce_usage_int(row.get(self.completion_tokens_col, 0)),
                    total_tokens=_coerce_usage_int(row.get(self.total_tokens_col, 0)),
                    warning=str(row.get(self.warning_col, "") or ""),
                    primary_error=str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or ""),
                    layout_finalized=False,
                    layout_cluster=str(row.get("dripper_layout_cluster", "") or ""),
                )

        return [results_by_index[idx] for idx in range(len(df))]

    def _inference_result_from_row(self, row: pd.Series) -> _DripperInferenceResult:
        primary_error = str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")
        raw_response = str(row.get(self.raw_response_col, "") or "")
        return _DripperInferenceResult(
            raw_response=raw_response,
            inference_time_s=float(row.get(self.inference_time_col, 0.0) or 0.0),
            primary_error=primary_error if primary_error and not raw_response else "",
            warning=primary_error if primary_error and not raw_response else "",
            prompt_tokens=_coerce_usage_int(row.get(self.prompt_tokens_col, 0)),
            completion_tokens=_coerce_usage_int(row.get(self.completion_tokens_col, 0)),
            total_tokens=_coerce_usage_int(row.get(self.total_tokens_col, 0)),
        )

    def _postprocess_existing_llm_row(
        self,
        row: pd.Series,
        layout_cluster: str,
        *,
        layout_fallback_llm: bool = False,
        layout_validation_llm: bool = False,
        layout_standalone_llm: bool = False,
    ) -> _LayoutTemplateRowResult:
        inference_result = self._inference_result_from_row(row)
        primary_error = str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")
        if inference_result.primary_error:
            return replace(
                self._postprocess_error_row(
                    row,
                    inference_result,
                    layout_cluster,
                    layout_fallback_llm=layout_fallback_llm,
                    layout_standalone_llm=layout_standalone_llm,
                    primary_error=primary_error,
                ),
                layout_validation_llm=layout_validation_llm,
            )
        post_result = self._postprocess_raw_response(row, inference_result.raw_response)
        return _LayoutTemplateRowResult(
            raw_response=inference_result.raw_response,
            inference_time_s=inference_result.inference_time_s,
            prompt_tokens=inference_result.prompt_tokens,
            completion_tokens=inference_result.completion_tokens,
            total_tokens=inference_result.total_tokens,
            main_html=post_result.main_html,
            main_content=post_result.main_content,
            postprocess_time_s=post_result.postprocess_time_s,
            error=post_result.error,
            warning=_append_warning(primary_error, post_result.warning),
            layout_cluster=layout_cluster,
            layout_fallback_llm=layout_fallback_llm,
            layout_validation_llm=layout_validation_llm,
            layout_standalone_llm=layout_standalone_llm,
        )
