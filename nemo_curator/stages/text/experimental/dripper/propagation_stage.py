"""DripperHTMLLayoutPropagationStage — CPU-only stage for deferred template propagation.

Reads the output of DripperHTMLLayoutTemplateStage with defer_propagation=True,
finds sibling rows marked dripper_layout_pending_propagation=True, and runs
LayoutBatchParser against the cluster's representative mapping data.

This moves the expensive CPU propagation (~11s/row) completely off the H100
critical path. GPU stage does only LLM inference; this stage runs afterwards
on cheap CPU nodes.

Estimated impact: GPU stage drops from ~600s → ~250s (removes 23,000s of CPU
work from 8-GPU job), projecting H100-hours from 387K → ~160K.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper.stage import (
    DripperHTMLExtractionStage,
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
    _rebuild_batch,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    import pandas as pd


_PENDING_COL = "dripper_layout_pending_propagation"
_MAPPING_COL = "dripper_layout_mapping_json"
_CLUSTER_COL = "dripper_layout_cluster"
_REPRESENTATIVE_COL = "dripper_layout_representative"


@dataclass(kw_only=True)
class DripperHTMLLayoutPropagationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU-only stage: apply layout templates to rows deferred by the GPU stage.

    Requires the GPU output parquet to have been produced with
    ``layout_template_defer_propagation=True``, which writes:
    - ``dripper_layout_pending_propagation``: True for sibling rows
    - ``dripper_layout_mapping_json``: serialized mapping_data on representative rows
    - ``dripper_layout_cluster``: cluster ID on all layout rows

    This stage propagates templates to pending rows, validates quality,
    and marks failed rows for a downstream LLM fallback pass.
    """

    html_col: str = "html"
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    error_col: str = "dripper_error"
    url_col: str = "url"

    dynamic_classid_similarity_threshold: float = 0.85
    more_noise_enable: bool = True
    layout_template_validation_min_content_f1: float = 0.95
    layout_template_min_content_length_ratio: float | None = 0.25
    layout_template_max_content_length_ratio: float | None = 4.0
    propagation_target: str = "raw_html"

    _bindings: Any = None
    _web_bindings: Any = None
    _initialized: bool = False

    def output_batches(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.output_html_col,
            self.output_content_col,
            self.postprocess_time_col,
            self.error_col,
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            _PENDING_COL,
        ]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ANN401, ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:  # noqa: C901
        if not self._initialized:
            self.setup()
        df = batch.to_pandas().copy()

        if _PENDING_COL not in df.columns:
            return batch

        pending_mask = df[_PENDING_COL].astype(bool)
        if not pending_mask.any():
            return batch

        # Build cluster → representative mapping_data lookup
        mapping_by_cluster: dict[str, dict[str, Any]] = {}
        if _MAPPING_COL in df.columns and _REPRESENTATIVE_COL in df.columns:
            rep_rows = df[df[_REPRESENTATIVE_COL].astype(bool)]
            for _, row in rep_rows.iterrows():
                mapping_json = str(row.get(_MAPPING_COL) or "")
                cluster = str(row.get(_CLUSTER_COL) or "")
                if mapping_json and cluster:
                    with contextlib.suppress(Exception):
                        mapping_by_cluster[cluster] = json.loads(mapping_json)

        # Propagate each pending row
        for idx in df.index[pending_mask]:
            row = df.iloc[idx] if hasattr(df.iloc[idx], "get") else df.loc[idx]
            cluster_id = str(row.get(_CLUSTER_COL) or "")
            mapping_data = mapping_by_cluster.get(cluster_id)

            t0 = time.perf_counter()
            propagated_html = ""
            propagated_content = ""
            error = ""
            success = False

            if mapping_data is None:
                error = f"no_mapping_data_for_cluster={cluster_id}"
            else:
                try:
                    propagated_html, propagated_content, error = self._run_propagation(row, mapping_data)
                    if not error:
                        success = True
                except Exception as exc:  # noqa: BLE001
                    error = f"propagation_exception={exc!s:.200}"

            elapsed = time.perf_counter() - t0

            df.loc[idx, self.output_html_col] = propagated_html
            df.loc[idx, self.output_content_col] = propagated_content
            df.loc[idx, self.postprocess_time_col] = elapsed
            df.loc[idx, self.error_col] = error
            df.loc[idx, "dripper_layout_propagated"] = True
            df.loc[idx, "dripper_layout_propagation_success"] = success
            df.loc[idx, _PENDING_COL] = False  # consumed

        n_pending = int(pending_mask.sum())
        n_success = (
            int(df["dripper_layout_propagation_success"].sum())
            if "dripper_layout_propagation_success" in df.columns
            else 0
        )
        logger.info(
            "DripperHTMLLayoutPropagationStage: propagated {}/{} rows in batch",
            n_success,
            n_pending,
        )
        return _rebuild_batch(batch, df)

    def _run_propagation(  # noqa: PLR0911
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Run LayoutBatchParser on one sibling row. Returns (html, content, error)."""
        assert self._web_bindings is not None  # noqa: S101
        assert self._bindings is not None  # noqa: S101

        if self.propagation_target == "mapped_item_ids":
            mapped_html = str(row.get("dripper_mapped_html") or row.get("html") or "")
            html_source = mapped_html
        else:
            html_source = DripperHTMLExtractionStage._coerce_html(row.get("html") or "")

        if not html_source.strip():
            return "", "", "empty_html_source"

        task_data = dict(mapping_data)
        task_data.update(
            {
                "html_source": html_source,
                "dynamic_id_enable": True,
                "dynamic_classid_enable": True,
                "more_noise_enable": self.more_noise_enable,
                "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
            }
        )

        try:
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
        except Exception as exc:  # noqa: BLE001
            return "", "", f"layout_parser_error={exc!s:.200}"

        if parts.get("main_html_success") is False:
            return "", "", "main_html_success_false"

        main_html = str(parts.get("main_html_body") or "")

        # Content-length ratio guard
        rep_content_len = mapping_data.get("_dripper_representative_content_len")
        if rep_content_len and rep_content_len > 0:
            from nemo_curator.stages.text.experimental.dripper.stage import _convert_main_html

            content = _convert_main_html(self._bindings, main_html, row.get("url"))
            content_len = len(str(content))
            ratio = content_len / rep_content_len
            if self.layout_template_min_content_length_ratio and ratio < self.layout_template_min_content_length_ratio:
                return "", "", f"content_length_ratio_low={ratio:.3f}"
            if self.layout_template_max_content_length_ratio and ratio > self.layout_template_max_content_length_ratio:
                return "", "", f"content_length_ratio_high={ratio:.3f}"
            return main_html, str(content), ""

        try:
            from nemo_curator.stages.text.experimental.dripper.stage import _convert_main_html

            content = _convert_main_html(self._bindings, main_html, row.get("url"))
        except Exception as exc:  # noqa: BLE001
            return main_html, "", f"content_conversion_error={exc!s:.200}"

        return main_html, str(content), ""
