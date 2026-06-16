"""Dripper HTML layout clustering stage.

This module extracts _gpu_cluster_html_struct and DripperHTMLLayoutClusteringStage
from stage.py so they can be used in the streaming pipeline where each DocumentBatch
may contain pages from MULTIPLE host_domains (from bin-packing).

Critical fix: process() now groups by host_domain before clustering, so each host
is clustered independently. The layout_id becomes "{host_domain}::{cluster_id}"
where cluster_id is the stable layout hash.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
    _LLMWebKitBindings,
    _load_llm_web_kit_bindings,
)
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import (
    _layout_dom_path_fingerprint,
    _layout_feature_fingerprint,
    _layout_page_signature_key,
    _layout_page_signature_key_with_low_card_queries,
    _low_card_query_value_keys,
    _normalize_query_value_keys,
    _uses_low_card_query_shape,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_NEEDS_LLM_COL,
    _LAYOUT_PAGE_SIGNATURE_MODES,
    _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES,
    _LAYOUT_TEMPLATE_LARGE_HOST_MODES,
    _LayoutClusterAssignment,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import _coerce_html, _url_host_key
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from nemo_curator.backends.base import WorkerMetadata


_GPU_CLUSTER_MIN_N = 128  # use GPU matmul for hosts with >= this many pages


def _gpu_cluster_html_struct(  # noqa: C901
    samples: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[int]] | None:
    """GPU-accelerated drop-in for web_bindings.cluster_html_struct().

    Ports the llm-webkit O(n²) cosine loop to a single CUDA matmul.
    Returns None if GPU is unavailable or features are empty (caller falls back to CPU).
    """
    try:
        import numpy as np
        import torch
        import torch.nn.functional as nn_functional
        from sklearn.cluster import DBSCAN
        from sklearn.feature_extraction import DictVectorizer
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    features_list = [s.get("feature") or {} for s in samples]

    # Port __parse_valid_layer: find layer with most tags per page, then take majority vote
    layer_counter: Counter[int] = Counter()
    for f in features_list:
        tags = f.get("tags") or {}
        if not tags:
            continue
        widest = max(tags.items(), key=lambda kv: len(kv[1]) if isinstance(kv[1], list) else 0)[0]
        layer_counter[int(widest)] += 1
    if not layer_counter:
        return None
    layer_n = layer_counter.most_common(1)[0][0]

    # Port __simp_tags: flatten layers <= layer_n into count dicts
    def _simp(d: dict, ln: int) -> dict:
        out: dict[str, int] = {}
        for k, items in d.items():
            if int(k) <= ln:
                for item in items if isinstance(items, list) else []:
                    key = f"{k}_{item}"
                    out[key] = out.get(key, 0) + 1
        return out

    tags_dicts = [_simp(f.get("tags") or {}, layer_n) for f in features_list]
    attrs_dicts = [_simp(f.get("attrs") or {}, layer_n) for f in features_list]

    # Port __parse_vectors + GPU cosine sim (k=0.7 tags, 0.3 attrs -- matches __cosin_simil)
    k = 0.7

    def _cosine_sim_gpu(dicts: list[dict]) -> torch.Tensor:
        if all(not d for d in dicts):
            return torch.ones(len(dicts), len(dicts))
        feature_matrix = np.array(
            DictVectorizer(sparse=False).fit_transform(dicts),
            dtype=np.float32,
        )
        t = torch.from_numpy(feature_matrix).cuda()
        t_norm = nn_functional.normalize(t, dim=1)
        return t_norm @ t_norm.T

    sim = k * _cosine_sim_gpu(tags_dicts) + (1 - k) * _cosine_sim_gpu(attrs_dicts)
    dist = np.clip(1.0 - sim.cpu().numpy(), 0.0, 1.0)

    labels: list[int] = (
        DBSCAN(
            eps=1.0 - threshold,
            min_samples=2,
            metric="precomputed",
        )
        .fit_predict(dist)
        .tolist()
    )

    for sample, label in zip(samples, labels, strict=False):
        sample["layout_id"] = label
        sample["max_layer_n"] = layer_n

    return samples, labels


@dataclass(kw_only=True)
class DripperHTMLLayoutClusteringStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Precompute host-bounded llm-webkit DOM layout IDs on CPU.

    Running this as a separate pass lets the downstream template stage use
    ``layout_id_col`` instead of rebuilding DBSCAN clusters inside every
    representative/propagation actor.

    In the streaming pipeline each DocumentBatch may contain pages from MULTIPLE
    host_domains (from bin-packing). This stage groups by host_domain and clusters
    each host independently before assigning layout IDs.
    """

    name: str = "DripperHTMLLayoutClusteringStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.5))
    html_col: str = "html"
    url_col: str | None = "url"
    host_col: str | None = None
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    item_count_col: str = "dripper_item_count"
    layout_id_col: str = "dripper_layout_id"
    layout_feature_source: Literal["raw_html", "simpled_html", "mapped_html"] = "raw_html"
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_page_signature_mode: str = "none"
    layout_exact_query_value_keys: str | Iterable[str] | None = None
    layout_template_max_exact_host_pages: int = 0
    layout_template_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    worker_count: int | None = None

    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.layout_cluster_threshold <= 1.0:
            msg = "layout_cluster_threshold must be in (0, 1]"
            raise ValueError(msg)
        if self.layout_template_min_cluster_size <= 1:
            msg = "layout_template_min_cluster_size must be greater than 1"
            raise ValueError(msg)
        if self.layout_page_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = f"layout_page_signature_mode must be one of {sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            raise ValueError(msg)
        if self.layout_feature_source not in _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES:
            msg = f"layout_feature_source must be one of {sorted(_LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES)}"
            raise ValueError(msg)
        self.layout_exact_query_value_keys = _normalize_query_value_keys(self.layout_exact_query_value_keys)
        if self.layout_template_max_exact_host_pages < 0:
            msg = "layout_template_max_exact_host_pages must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_host_mode not in _LAYOUT_TEMPLATE_LARGE_HOST_MODES:
            msg = f"layout_template_large_host_mode must be one of {sorted(_LAYOUT_TEMPLATE_LARGE_HOST_MODES)}"
            raise ValueError(msg)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        columns = [self.html_col]
        if self.url_col:
            columns.append(self.url_col)
        if self.host_col:
            columns.append(self.host_col)
        if self.layout_feature_source == "simpled_html":
            columns.append(self.simplified_html_col)
        elif self.layout_feature_source == "mapped_html":
            columns.append(self.mapped_html_col)
        return ["data"], columns

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.layout_id_col]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._web_bindings = _load_llm_web_kit_bindings()
        self._initialized = True
        logger.info("DripperHTMLLayoutClusteringStage setup complete")

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        logger.debug("Clustering: {} rows", len(df))
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        started = time.perf_counter()
        layout_ids = [""] * len(df)

        # Group by host_domain and cluster each host independently.
        # In the streaming pipeline each DocumentBatch may contain pages from MULTIPLE
        # host_domains (from bin-packing), so we must cluster per-host to avoid
        # layout IDs bleeding across unrelated domains.
        host_col = self.host_col or self.url_col
        if host_col and host_col in df.columns:
            for _host_val, host_df in df.groupby(host_col):
                host_assignments = self._build_layout_assignments(host_df)
                for assignment in host_assignments:
                    layout_ids[assignment.row_index] = assignment.layout_id
        else:
            # Fallback: treat all rows as same host
            assignments = self._build_layout_assignments(df)
            for assignment in assignments:
                layout_ids[assignment.row_index] = assignment.layout_id

        df[self.layout_id_col] = layout_ids

        assigned_rows = sum(bool(layout_id) for layout_id in layout_ids)
        elapsed_s = time.perf_counter() - started
        self._log_metrics(
            {
                "layout_clustering_rows": float(len(df)),
                "layout_clustering_assigned_rows": float(assigned_rows),
                "layout_clustering_unassigned_rows": float(len(df) - assigned_rows),
                "layout_clustering_elapsed_s": elapsed_s,
            }
        )
        logger.info(
            "Dripper layout clustering assigned {}/{} row(s) to {} layout ID(s) in {:.3f}s",
            assigned_rows,
            len(df),
            len({layout_id for layout_id in layout_ids if layout_id}),
            elapsed_s,
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _build_layout_assignments(self, df: pd.DataFrame) -> list[_LayoutClusterAssignment]:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in df.iterrows():
            if _DRIPPER_NEEDS_LLM_COL in df.columns and not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = self._row_feature_html(row)
            if not html_text.strip():
                continue
            try:
                feature = self._web_bindings.get_feature(html_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper pre-layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            samples_by_host[self._row_host_key(row)].append(
                {"track_id": str(idx), "html": html_text, "feature": feature}
            )

        assignments: list[_LayoutClusterAssignment] = []
        for host_key, samples in samples_by_host.items():
            assignments.extend(self._build_host_layout_assignments(df, host_key, samples))
        return assignments

    def _build_host_layout_assignments(  # noqa: C901, PLR0912
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
    ) -> list[_LayoutClusterAssignment]:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        if len(samples) < self.layout_template_min_cluster_size:
            return []

        grouped_samples: dict[str, list[int]] = defaultdict(list)
        if self.layout_template_max_exact_host_pages and len(samples) > self.layout_template_max_exact_host_pages:
            if self.layout_template_large_host_mode == "standalone":
                logger.debug(
                    "Dripper pre-layout host={} rows={} exceeds max_exact_host_pages={}; leaving unassigned",
                    host_key,
                    len(samples),
                    self.layout_template_max_exact_host_pages,
                )
                return []
            fingerprint_fn = (
                (lambda sample: _layout_feature_fingerprint(sample.get("feature")))
                if self.layout_template_large_host_mode == "feature_hash"
                else (lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")))
            )
            by_fingerprint: dict[str, list[int]] = defaultdict(list)
            for sample in samples:
                by_fingerprint[fingerprint_fn(sample)].append(int(sample["track_id"]))
            for fingerprint, indexes in by_fingerprint.items():
                self._add_signature_grouped_indexes(
                    df,
                    grouped_samples,
                    host_key=host_key,
                    layout_key="fingerprint",
                    fingerprint=fingerprint,
                    indexes=indexes,
                )
        else:
            clustered_samples = None
            if len(samples) >= _GPU_CLUSTER_MIN_N:
                try:
                    gpu_result = _gpu_cluster_html_struct(samples, threshold=self.layout_cluster_threshold)
                    if gpu_result is not None:
                        clustered_samples, _layout_ids = gpu_result
                        logger.debug(
                            "Dripper GPU clustering host={} n={} layout_ids={}",
                            host_key,
                            len(samples),
                            len(set(_layout_ids)),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper GPU clustering failed for host {}, falling back to CPU: {}", host_key, exc)
                    clustered_samples = None

            if clustered_samples is None:
                try:
                    clustered_samples, _layout_ids = self._web_bindings.cluster_html_struct(
                        samples,
                        threshold=self.layout_cluster_threshold,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper pre-layout clustering failed for host {}: {}", host_key, exc)
                    return []
            if not clustered_samples:
                return []

            max_layer_n = int(clustered_samples[0].get("max_layer_n") or 5)
            exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for sample in clustered_samples:
                layout_id = int(sample.get("layout_id", -1))
                if layout_id < 0:
                    continue
                if len(exemplars_by_layout[layout_id]) < 3:  # noqa: PLR2004
                    exemplars_by_layout[layout_id].append(sample)

            for sample in clustered_samples:
                layout_id = self._assign_layout_by_exemplar_similarity(
                    sample.get("feature"),
                    exemplars_by_layout,
                    max_layer_n,
                )
                if layout_id < 0:
                    continue
                row_idx = int(sample["track_id"])
                grouped_samples[f"__pending_dom_{layout_id:06d}"].append(row_idx)

            pending_groups = [
                (key, indexes) for key, indexes in list(grouped_samples.items()) if key.startswith("__pending_dom_")
            ]
            grouped_samples.clear()
            for pending_key, indexes in pending_groups:
                self._add_signature_grouped_indexes(
                    df,
                    grouped_samples,
                    host_key=host_key,
                    layout_key=pending_key.removeprefix("__pending_"),
                    fingerprint="",
                    indexes=indexes,
                )

        assignments: list[_LayoutClusterAssignment] = []
        for layout_key, indexes in grouped_samples.items():
            if len(indexes) < self.layout_template_min_cluster_size:
                continue
            assignments.extend(_LayoutClusterAssignment(row_index=idx, layout_id=layout_key) for idx in indexes)
        return assignments

    def _assign_layout_by_exemplar_similarity(
        self,
        feature: object,
        exemplars_by_layout: dict[int, list[dict[str, Any]]],
        max_layer_n: int,
    ) -> int:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        for layout_id, exemplars in exemplars_by_layout.items():
            for exemplar in exemplars:
                try:
                    score = self._web_bindings.similarity(feature, exemplar.get("feature"), max_layer_n)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper pre-layout similarity failed for layout {}: {}", layout_id, exc)
                    continue
                if score is not None and score >= self.layout_cluster_threshold:
                    return layout_id
        return -2

    def _row_host_key(self, row: pd.Series) -> str:
        if self.host_col and self.host_col in row:
            host_key = _url_host_key(row.get(self.host_col))
            if host_key:
                return host_key
        return _url_host_key(row.get(self.url_col) if self.url_col else None)

    def _row_feature_html(self, row: pd.Series) -> str:
        if self.layout_feature_source == "simpled_html":
            return _coerce_html(row.get(self.simplified_html_col, ""))
        if self.layout_feature_source == "mapped_html":
            return _coerce_html(row.get(self.mapped_html_col, ""))
        return _coerce_html(row.get(self.html_col, ""))

    def _layout_page_signature_key(self, row: pd.Series) -> str:
        return _layout_page_signature_key(
            row.get(self.url_col) if self.url_col else None,
            row.get(self.item_count_col) if self.item_count_col in row else None,
            self.layout_page_signature_mode,
            exact_query_value_keys=self.layout_exact_query_value_keys,
        )

    def _add_signature_grouped_indexes(  # noqa: PLR0913
        self,
        df: pd.DataFrame,
        grouped_samples: dict[str, list[int]],
        *,
        host_key: str,
        layout_key: str,
        fingerprint: str,
        indexes: list[int],
    ) -> None:
        low_card_query_keys: set[str] = set()
        if _uses_low_card_query_shape(self.layout_page_signature_mode) and self.url_col:
            low_card_query_keys = _low_card_query_value_keys(
                [df.iloc[row_idx].get(self.url_col) for row_idx in indexes]
            )
        for row_idx in indexes:
            row = df.iloc[row_idx]
            if _uses_low_card_query_shape(self.layout_page_signature_mode):
                signature_key = _layout_page_signature_key_with_low_card_queries(
                    row.get(self.url_col) if self.url_col else None,
                    row.get(self.item_count_col) if self.item_count_col in row else None,
                    self.layout_page_signature_mode,
                    low_card_query_keys,
                    exact_query_value_keys=self.layout_exact_query_value_keys,
                )
            else:
                signature_key = self._layout_page_signature_key(row)
            stable_layout_key = self._stable_layout_id(host_key, layout_key, fingerprint, signature_key)
            grouped_samples[stable_layout_key].append(row_idx)

    @staticmethod
    def _stable_layout_id(host_key: str, layout_key: str, fingerprint: str, signature_key: str) -> str:
        payload = f"{host_key}\n{layout_key}\n{fingerprint}\n{signature_key}"
        digest = hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:20]  # noqa: S324
        return f"layout-{digest}"
