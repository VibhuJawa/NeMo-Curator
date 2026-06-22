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

"""CPU-only broadcast-template propagation stage (Phase 2b).

``DripperHTMLBroadcastPropagateStage`` decouples the CPU-heavy per-page content extraction
(template propagation, ~78%% of runtime) from the GPU vLLM inference. The finalize stage emits a
small per-cluster template side-table (``_dripper_layout_template_json``); this stage loads that
table once at ``setup()`` and replays propagation for every pending-propagation member on a CPU
partition, calling the IDENTICAL ``_propagate_pending_row_async`` the in-block finalize uses. The
template is only ever written for clusters whose validation gate PASSED, so replaying it here is
F1-neutral by construction: same ``_propagate_layout_template``, same ``mapping_data``, same gate.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_DEFERRED_LLM_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL,
    _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
    _DRIPPER_LAYOUT_TEMPLATE_JSON_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _LayoutTemplateRowResult,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import (
    _append_warning,
    _numeric_series_or_zero,
)
from nemo_curator.stages.text.experimental.dripper.stages.layout_template import DripperHTMLLayoutTemplateStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass(kw_only=True)
class DripperHTMLBroadcastPropagateStage(DripperHTMLLayoutTemplateStage):
    """Replay layout-template propagation for pending members on a CPU partition (Phase 2b).

    Loads the finalize-emitted template side-table (parquet of
    ``[dripper_layout_cluster, _dripper_layout_template_json]``) into a ``{cluster_id ->
    mapping_data}`` map, then for every row flagged ``_dripper_layout_pending_propagation`` calls the
    same ``_propagate_pending_row_async`` the finalize uses. Rows whose cluster has no usable template
    (missing / defer sentinel) are deferred to the LLM via ``_defer_row``. All other rows pass through
    unchanged. This stage issues NO model calls; it is the CPU half of the broadcast-template split.
    """

    name: str = "DripperHTMLBroadcastPropagateStage"
    template_table_path: str = ""
    # Optional ray ObjectRef to a pre-loaded, deduped Arrow side-table (dripper_layout_cluster,
    # _dripper_layout_template_json). When set, every actor on a node SHARES one zero-copy plasma
    # copy instead of each loading the full ~GB side-table into a per-actor parsed dict -- the
    # per-actor load capped the postprocess fan-out at ~16 actors and stalled the pool. The driver
    # builds + ray.puts it once (see pipeline_cpu_only.py); templates are parsed lazily per lookup
    # (cheap vs convert2content). Falls back to per-actor path load when unset (tests/standalone).
    template_table_ref: Any = None

    _template_by_cluster: dict[str, dict[str, Any]] = field(init=False, repr=False, default_factory=dict)
    _template_table: Any = field(init=False, repr=False, default=None)
    _cluster_index: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.template_table_path and self.template_table_ref is None:
            msg = "template_table_path or template_table_ref must be set to the template side-table"
            raise ValueError(msg)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        if self.template_table_ref is not None:
            # Shared path: fetch the ONE deduped Arrow side-table from plasma (zero-copy on this
            # node) and build only a tiny {cluster_id -> row} index. The (large) template_json
            # strings stay in the shared buffer and are parsed lazily per lookup (_mapping_for).
            import ray

            self._template_table = ray.get(self.template_table_ref)
            self._cluster_index = {}
            for i, cluster_id in enumerate(self._template_table.column("dripper_layout_cluster").to_pylist()):
                key = str(cluster_id or "")
                if key and key not in self._cluster_index:
                    self._cluster_index[key] = i
            n_templates = len(self._cluster_index)
            source = "shared Arrow side-table (zero-copy)"
        else:
            self._template_by_cluster = self._load_template_table(self.template_table_path)
            n_templates = len(self._template_by_cluster)
            source = self.template_table_path
        self._initialized = True
        logger.info(
            "DripperHTMLBroadcastPropagateStage setup complete: {} cluster templates from {}",
            n_templates,
            source,
        )

    @staticmethod
    def _load_template_table(template_table_path: str) -> dict[str, dict[str, Any]]:
        """Load the side-table into ``{cluster_id -> mapping_data}``, dropping the defer sentinel ("").

        Inverts ``DripperHTMLLayoutTemplateStage._serialize_template_json`` with ``json.loads``: the
        recovered dict is byte-for-byte the JSON-safe ``mapping_data`` the finalize fed to
        ``_propagate_layout_template`` (``html_element_dict`` stays a JSON string, which
        ``LayoutBatchParser`` accepts), so propagation downstream is identical.
        """
        # template_table_path is a directory of *.parquet shards that ALSO contains a
        # non-parquet metrics.json; a bare directory read trips on it (ArrowInvalid). Read
        # only the parquet shards (concat is byte-identical to reading the dir as one table).
        from pathlib import Path

        _p = Path(template_table_path)
        _files = sorted(str(f) for f in _p.glob("*.parquet")) if _p.is_dir() else [template_table_path]
        table = pd.concat(
            [
                pd.read_parquet(f, columns=["dripper_layout_cluster", _DRIPPER_LAYOUT_TEMPLATE_JSON_COL])
                for f in _files
            ],
            ignore_index=True,
        )
        templates: dict[str, dict[str, Any]] = {}
        for cluster_id, template_json in zip(
            table["dripper_layout_cluster"].tolist(),
            table[_DRIPPER_LAYOUT_TEMPLATE_JSON_COL].tolist(),
            strict=True,
        ):
            cluster_key = str(cluster_id or "")
            text = str(template_json or "")
            if not cluster_key or not text:
                continue
            if cluster_key in templates:
                continue
            templates[cluster_key] = json.loads(text)
        return templates

    def _mapping_for(self, cluster_id: str) -> dict[str, Any] | None:
        """Resolve a cluster's parsed mapping_data.

        Shared-ref mode: {cluster_id -> row} index -> the template_json string (a small per-lookup
        copy out of the zero-copy plasma Arrow buffer) -> json.loads. Parsing per lookup is cheap
        vs the convert2content postprocess and avoids holding the full parsed side-table per actor.
        Path/fallback mode: the per-actor parsed dict.
        """
        if self._template_table is not None:
            idx = self._cluster_index.get(cluster_id)
            if idx is None:
                return None
            text = self._template_table.column(_DRIPPER_LAYOUT_TEMPLATE_JSON_COL)[idx].as_py()
            return json.loads(text) if text else None
        return self._template_by_cluster.get(cluster_id)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.html_col,
            self.mapped_html_col,
            self.simplified_html_col,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
            "dripper_layout_cluster",
            _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.output_html_col,
            self.output_content_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            "dripper_layout_cluster",
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            _DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL,
            _DRIPPER_LAYOUT_DEFERRED_LLM_COL,
            _DRIPPER_LAYOUT_FINALIZED_COL,
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        logger.debug("BroadcastPropagate: {} rows", len(df))
        results = self._propagate_pending_rows(df)

        preprocess_times = _numeric_series_or_zero(df, self.preprocess_time_col)
        inference_times = _numeric_series_or_zero(df, self.inference_time_col)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)

        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.postprocess_time_col] = postprocess_times
        df[self.total_time_col] = preprocess_times + inference_times + postprocess_times
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [
            _append_warning(str(existing or ""), result.warning)
            for existing, result in zip(
                df.get(self.warning_col, pd.Series([""] * len(df))).tolist(), results, strict=True
            )
        ]
        df["dripper_layout_cluster"] = [r.layout_cluster for r in results]
        df["dripper_layout_propagated"] = [r.layout_propagated for r in results]
        df["dripper_layout_propagation_success"] = [r.layout_propagation_success for r in results]
        df[_DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL] = [r.layout_finalized for r in results]
        df[_DRIPPER_LAYOUT_DEFERRED_LLM_COL] = [r.deferred_llm for r in results]
        df[_DRIPPER_LAYOUT_FINALIZED_COL] = [r.layout_finalized for r in results]
        df[_DRIPPER_NEEDS_LLM_COL] = [r.deferred_llm for r in results]
        existing_primary_errors = df[_DRIPPER_PRIMARY_ERROR_COL].astype(str).tolist()
        df[_DRIPPER_PRIMARY_ERROR_COL] = [
            _append_warning(existing_error, result.primary_error)
            for existing_error, result in zip(existing_primary_errors, results, strict=True)
        ]
        df = df.drop(columns=[col for col in (_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,) if col in df.columns])

        self._log_metrics(
            {
                "broadcast_propagate_rows": float(len(df)),
                "broadcast_propagate_pending_rows": float(sum(r.layout_propagated for r in results)),
                "broadcast_propagate_success_rows": float(sum(r.layout_propagation_success for r in results)),
                "broadcast_propagate_deferred_llm_rows": float(sum(r.deferred_llm for r in results)),
                "broadcast_propagate_finalized_rows": float(sum(r.layout_finalized for r in results)),
            }
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _propagate_pending_rows(self, df: pd.DataFrame) -> list[_LayoutTemplateRowResult]:
        propagation_semaphore = asyncio.Semaphore(self.layout_template_propagation_concurrency)
        return run_async_safe(lambda: self._propagate_pending_rows_async(df, propagation_semaphore))

    async def _propagate_pending_rows_async(
        self,
        df: pd.DataFrame,
        propagation_semaphore: asyncio.Semaphore,
    ) -> list[_LayoutTemplateRowResult]:
        results_by_index: dict[int, _LayoutTemplateRowResult] = {}
        pending: list[tuple[int, pd.Series, dict[str, Any], str]] = []
        for idx, row in df.iterrows():
            int_idx = int(idx)
            if not bool(row.get(_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL, False)):
                results_by_index[int_idx] = self._passthrough_row(row)
                continue
            cluster_id = str(row.get("dripper_layout_cluster", "") or "")
            mapping_data = self._mapping_for(cluster_id)
            if mapping_data is None:
                # Missing template or defer sentinel ("" was dropped at load): the cluster's validation
                # gate did not pass (or its rep landed in another shard), so defer the row to the LLM --
                # the identical _defer_row(force_needs_llm=True) outcome the finalize uses on failure.
                results_by_index[int_idx] = self._defer_row(
                    row,
                    primary_error="layout template unavailable for broadcast propagation",
                    layout_cluster=cluster_id,
                    layout_fallback_llm=True,
                    force_needs_llm=True,
                )
                continue
            pending.append((int_idx, row, mapping_data, cluster_id))

        propagated_results = await asyncio.gather(
            *(
                self._propagate_pending_row_async(row, mapping_data, cluster_id, propagation_semaphore)
                for _idx, row, mapping_data, cluster_id in pending
            )
        )
        for (int_idx, _row, _mapping_data, _cluster_id), propagated in zip(pending, propagated_results, strict=True):
            results_by_index[int_idx] = propagated

        return [results_by_index[idx] for idx in range(len(df))]

    def _passthrough_row(self, row: pd.Series) -> _LayoutTemplateRowResult:
        """Carry a non-pending row through, preserving any content/finalized state it already has.

        In the Phase 2b chain the input is plan/cluster output (no content yet), so standalone /
        fallback / non-LLM rows pass through with ``layout_finalized=False`` for
        ``DripperHTMLPostprocessStage`` to handle, and ``deferred_llm`` mirrors the row's existing
        ``_dripper_needs_llm``. If a row was ALREADY finalized upstream (it carries
        ``_dripper_layout_finalized`` + content), that state is preserved so this opt-in stage never
        clobbers a prior result.
        """
        already_finalized = bool(row.get(_DRIPPER_LAYOUT_FINALIZED_COL, False))
        return _LayoutTemplateRowResult(
            main_html=str(row.get(self.output_html_col, "") or "") if already_finalized else "",
            main_content=(row.get(self.output_content_col, "") or "") if already_finalized else "",
            postprocess_time_s=float(row.get(self.postprocess_time_col, 0.0) or 0.0) if already_finalized else 0.0,
            error=str(row.get(self.error_col, "") or "") if already_finalized else "",
            warning=str(row.get(self.warning_col, "") or ""),
            primary_error=str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or ""),
            deferred_llm=bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)),
            layout_finalized=already_finalized,
            layout_cluster=str(row.get("dripper_layout_cluster", "") or ""),
        )
