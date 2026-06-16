from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.stages.text.experimental.dripper.stages._bindings import _load_llm_web_kit_bindings
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import _select_validation_indexes
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
    _DRIPPER_LAYOUT_SPLIT_PLANNED_COL,
    _DRIPPER_NEEDS_LLM_COL,
)
from nemo_curator.stages.text.experimental.dripper.stages.layout_template import DripperHTMLLayoutTemplateStage
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass(kw_only=True)
class DripperHTMLLayoutPlanStage(DripperHTMLLayoutTemplateStage):
    """Plan layout-template first-pass LLM rows without issuing model calls.

    This experimental split stage lets the normal inference stage batch layout
    representatives, validation rows, standalone rows, and known fallback rows
    together. Rows that may be satisfied by template propagation are held back
    until :class:`DripperHTMLLayoutFinalizeStage` has representative responses.
    """

    name: str = "DripperHTMLLayoutPlanStage"

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._web_bindings = _load_llm_web_kit_bindings()
        self._initialized = True
        logger.info("DripperHTMLLayoutPlanStage setup complete")

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            _DRIPPER_NEEDS_LLM_COL,
            "dripper_layout_cluster",
            "dripper_layout_representative",
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            "dripper_layout_fallback_llm",
            "dripper_layout_validation_llm",
            "dripper_layout_standalone_llm",
            _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
            _DRIPPER_LAYOUT_SPLIT_PLANNED_COL,
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:  # noqa: C901, PLR0912, PLR0915
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        logger.debug("Plan: {} rows", len(df))
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        original_needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        first_pass_needs_llm = [False] * len(df)
        layout_cluster = [""] * len(df)
        layout_representative = [False] * len(df)
        layout_propagated = [False] * len(df)
        layout_propagation_success = [False] * len(df)
        layout_fallback_llm = [False] * len(df)
        layout_validation_llm = [False] * len(df)
        layout_standalone_llm = [False] * len(df)
        layout_pending_propagation = [False] * len(df)
        layout_split_planned = [False] * len(df)

        build_started = time.perf_counter()
        layout_plans = self._build_layout_group_plans(df)
        grouped_indexes = {idx for plan in layout_plans for idx in plan.indexes}

        for plan_index, plan in enumerate(layout_plans):
            cluster_id = f"layout-{plan_index:06d}"
            indexes = [idx for idx in plan.indexes if original_needs_llm[idx]]
            if len(indexes) < self.layout_template_min_cluster_size:
                continue
            low_return_reason = self._low_return_fallback_reason(indexes)
            prompt_dedup_reason = self._prompt_dedup_fallback_reason(df, indexes)
            for idx in indexes:
                layout_cluster[idx] = cluster_id
                layout_split_planned[idx] = True
            if low_return_reason or prompt_dedup_reason:
                for idx in indexes:
                    first_pass_needs_llm[idx] = True
                    layout_fallback_llm[idx] = True
                continue

            representative_idx = self._select_representative_index(df, indexes)
            sibling_indexes = [idx for idx in indexes if idx != representative_idx]
            validation_indexes = _select_validation_indexes(
                df,
                sibling_indexes,
                self._effective_validation_rows(len(indexes)),
                self.url_col,
                self.item_count_col,
                self.layout_template_validation_signature_mode,
                exact_query_value_keys=self.layout_exact_query_value_keys,
            )
            validation_set = set(validation_indexes)

            first_pass_needs_llm[representative_idx] = True
            layout_representative[representative_idx] = True
            for idx in validation_indexes:
                first_pass_needs_llm[idx] = True
                layout_fallback_llm[idx] = True
                layout_validation_llm[idx] = True
            for idx in sibling_indexes:
                if idx in validation_set:
                    continue
                layout_pending_propagation[idx] = True

        for idx, needs_llm in enumerate(original_needs_llm):
            if idx in grouped_indexes:
                continue
            if needs_llm:
                first_pass_needs_llm[idx] = True
                layout_standalone_llm[idx] = True
                layout_split_planned[idx] = True

        df[_DRIPPER_NEEDS_LLM_COL] = first_pass_needs_llm
        df["dripper_layout_cluster"] = layout_cluster
        df["dripper_layout_representative"] = layout_representative
        df["dripper_layout_propagated"] = layout_propagated
        df["dripper_layout_propagation_success"] = layout_propagation_success
        df["dripper_layout_fallback_llm"] = layout_fallback_llm
        df["dripper_layout_validation_llm"] = layout_validation_llm
        df["dripper_layout_standalone_llm"] = layout_standalone_llm
        df[_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL] = layout_pending_propagation
        df[_DRIPPER_LAYOUT_SPLIT_PLANNED_COL] = layout_split_planned

        self._log_metrics(
            {
                "layout_plan_rows": float(len(df)),
                "layout_plan_groups": float(len(layout_plans)),
                "layout_plan_first_pass_llm_rows": float(sum(first_pass_needs_llm)),
                "layout_plan_representative_rows": float(sum(layout_representative)),
                "layout_plan_validation_llm_rows": float(sum(layout_validation_llm)),
                "layout_plan_pending_propagation_rows": float(sum(layout_pending_propagation)),
                "layout_plan_standalone_llm_rows": float(sum(layout_standalone_llm)),
                "layout_plan_build_time_s": time.perf_counter() - build_started,
            }
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
