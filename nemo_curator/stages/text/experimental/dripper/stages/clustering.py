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
import json
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
from nemo_curator.stages.text.experimental.dripper.stages._layout_mixin import _LayoutRowKeyMixin
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import (
    _layout_page_signature_key_with_low_card_queries,
    _low_card_query_value_keys,
    _normalize_query_value_keys,
    _uses_low_card_query_shape,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_NEEDS_LLM_COL,
    _LAYOUT_PAGE_SIGNATURE_MODES,
    _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES,
    _LayoutClusterAssignment,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import torch

    from nemo_curator.backends.base import WorkerMetadata


_GPU_CLUSTER_MIN_N = 128  # use GPU matmul for hosts with >= this many pages
# Defensive bound on the dense n x n cosine-similarity matrix (sparse vectorize itself is
# cardinality-independent -- see _cosine_sim_gpu). A host's samples are normally kept well under
# this by Phase-B grouping (max_rows_per_batch splits mega-hosts into sub-blocks), so this only
# triggers if a large host reaches clustering un-split; it then chunks into GPU-sized pieces and
# deterministic _stable_layout_id collapses identical layouts across chunks (no dedup loss).
# (The ~13-min tgcom24 stall was NOT the cosine -- it was the per-row html decompress in the
# sample-build loop above, now skipped when the feature is precomputed.)
_GPU_CLUSTER_MAX_N = 8000


def _dbscan_precomputed_cosine(sim: torch.Tensor, eps: float) -> list[int]:
    """DBSCAN over a precomputed cosine-distance matrix (``dist = 1 - sim``).

    Prefers cuML's GPU DBSCAN, keeping the distance matrix on-device (zero host
    copy via DLPack).  Falls back to sklearn on CPU when cuML/cupy or a GPU is
    unavailable -- e.g. no GPU allocated to this stage, or the precomputed metric
    is unsupported by the installed cuML version.
    """
    import torch

    dist_t = torch.clamp(1.0 - sim, 0.0, 1.0)

    if dist_t.is_cuda:
        try:
            import cupy as cp
            from cuml.cluster import DBSCAN as _CuDBSCAN  # noqa: N811

            dist_cp = cp.from_dlpack(dist_t.contiguous())
            labels = _CuDBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist_cp)
            return cp.asnumpy(labels).astype(int).tolist()
        except Exception as exc:  # noqa: BLE001
            logger.debug("cuML DBSCAN unavailable ({}); falling back to sklearn CPU", exc)

    import numpy as np
    from sklearn.cluster import DBSCAN

    dist = np.clip(dist_t.detach().cpu().numpy(), 0.0, 1.0)
    return DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist).tolist()


def _gpu_exemplar_reassign(
    sim: torch.Tensor, labels: list[int], threshold: float, max_exemplars: int = 3
) -> list[int]:
    """Reassign each page to a layout using the precomputed GPU cosine ``sim`` matrix.

    Mirrors the CPU ``_assign_layout_by_exemplar_similarity`` path but reuses ``sim`` (the
    same cosine DBSCAN clustered on) instead of recomputing web_bindings.similarity per page
    on CPU -- the prior O(n * layouts * exemplars) bottleneck. For each page, pick the FIRST
    layout (in label-first-encounter order, using up to ``max_exemplars`` exemplars per
    layout) whose exemplar similarity is >= ``threshold``; else leave it unassigned (-2).
    """
    import torch

    n = len(labels)
    order: list[int] = []
    exemplar_cols: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab < 0:
            continue
        if lab not in exemplar_cols:
            exemplar_cols[lab] = []
            order.append(lab)
        if len(exemplar_cols[lab]) < max_exemplars:
            exemplar_cols[lab].append(i)
    if not order:
        return [-2] * n

    # (n, n_layouts): max similarity of each page to each layout's exemplars, layouts in order.
    max_sim = torch.stack([sim[:, exemplar_cols[lab]].max(dim=1)[0] for lab in order], dim=1)
    meets = max_sim >= threshold
    any_meet = meets.any(dim=1).tolist()
    first_match = meets.to(torch.int8).argmax(dim=1).tolist()  # first True column (0 if none)
    return [order[first_match[i]] if any_meet[i] else -2 for i in range(n)]


def _gpu_cluster_html_struct(  # noqa: C901, PLR0915
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
    if layer_n <= 1:
        # llm_web_kit __parse_valid_layer fallback: when the modal widest layer is <= 1,
        # use the number of layers in the first feature instead.
        layer_n = len((features_list[0].get("tags") if features_list else {}) or {})

    # Faithful port of llm_web_kit __simp_tags + __list_to_dict (must match exactly so the
    # GPU cosine == web_bindings.similarity): keep layers (in dict order) with key <= ln;
    # index each kept layer by its 0-based POSITION in the filtered list (NOT its original
    # layer number); emit f"{pos}_{path}" = 1 -- BINARY presence, repetition within a layer
    # collapsed to a single 1.
    def _simp(d: dict, ln: int) -> dict:
        out: dict[str, int] = {}
        pos = 0
        for k, items in d.items():
            if int(k) <= ln:
                for item in dict.fromkeys(items if isinstance(items, list) else []):
                    out[f"{pos}_{item}"] = 1
                pos += 1
        return out

    _t0 = time.perf_counter()
    tags_dicts = [_simp(f.get("tags") or {}, layer_n) for f in features_list]
    attrs_dicts = [_simp(f.get("attrs") or {}, layer_n) for f in features_list]
    _t_feat = time.perf_counter()

    # Port __parse_vectors + GPU cosine sim (k=0.7 tags, 0.3 attrs -- matches __cosin_simil)
    k = 0.7

    def _cosine_sim_gpu(dicts: list[dict]) -> torch.Tensor:
        n = len(dicts)
        if all(not d for d in dicts):
            return torch.ones(n, n, device="cuda")
        # Sparse vectorize (n x nnz storage, independent of vocab cardinality) so
        # high-cardinality giant hosts don't blow up a dense n x vocab matrix.
        import cupy as cp
        import cupyx.scipy.sparse as cusparse

        x_sparse = DictVectorizer(sparse=True).fit_transform(dicts).tocsr().astype(np.float32)
        x_gpu = cusparse.csr_matrix(x_sparse)
        n_rows, n_feat = x_gpu.shape
        norms = cp.asarray(cp.sqrt(x_gpu.multiply(x_gpu).sum(axis=1))).ravel()
        norms[norms == 0] = 1.0
        x_norm = cusparse.diags(1.0 / norms) @ x_gpu  # L2-normalized rows, (n x f) sparse

        # The Gram matrix G = x_norm @ x_norm.T is DENSE n x n (real templated pages
        # share many features). Do NOT use cuSPARSE SpGEMM (sparse @ sparse): it raises
        # CUSPARSE_STATUS_INSUFFICIENT_RESOURCES trying to allocate the near-dense output.
        # Densify when the n x f matrix fits a GEMM budget; else tile SpMM (sparse @ dense)
        # so we never materialize the full n x f for high-cardinality hosts.
        dense_budget = 2_000_000_000  # ~2 GB of float32
        if n_rows * n_feat * 4 <= dense_budget:
            x_dense = x_norm.toarray()
            sim = x_dense @ x_dense.T  # cuBLAS GEMM -> dense (n x n)
        else:
            sim = cp.empty((n_rows, n_rows), dtype=np.float32)
            tile = max(256, dense_budget // (n_feat * 4))
            for j in range(0, n_rows, tile):
                rhs = cp.ascontiguousarray(x_norm[j : j + tile].toarray().T)  # (f x t) dense
                sim[:, j : j + tile] = x_norm @ rhs  # SpMM -> dense (n x t)
        return torch.from_dlpack(cp.ascontiguousarray(sim))

    sim = k * _cosine_sim_gpu(tags_dicts) + (1 - k) * _cosine_sim_gpu(attrs_dicts)
    _t_sim = time.perf_counter()
    labels: list[int] = _dbscan_precomputed_cosine(sim, eps=1.0 - threshold)
    _t_dbscan = time.perf_counter()

    # Exemplar reassignment on GPU, reusing `sim` (the same cosine DBSCAN used) instead of
    # recomputing web_bindings.similarity per page on CPU (the prior ~O(n*layouts) bottleneck).
    reassigned: list[int] = _gpu_exemplar_reassign(sim, labels, threshold)
    _t_reassign = time.perf_counter()

    logger.info(
        "GPU cluster timing n={}: feat_vectorize={:.1f}s cosine_matmul={:.1f}s dbscan={:.1f}s "
        "reassign={:.1f}s total={:.1f}s",
        len(samples),
        _t_feat - _t0,
        _t_sim - _t_feat,
        _t_dbscan - _t_sim,
        _t_reassign - _t_dbscan,
        _t_reassign - _t0,
    )

    for sample, lab in zip(samples, reassigned, strict=False):
        sample["layout_id"] = lab
        sample["max_layer_n"] = layer_n

    return samples, reassigned


@dataclass(kw_only=True)
class DripperHTMLLayoutClusteringStage(_LayoutRowKeyMixin, ProcessingStage[DocumentBatch, DocumentBatch]):
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
    worker_count: int | None = None
    # Phase split (avoids the GPU-idle watchdog): when feature_only is set this stage runs on a
    # CPU partition and ONLY precomputes the llm-webkit DOM feature into layout_feature_col (as a
    # JSON string), without clustering. The GPU phase then reads that column and skips the
    # CPU-heavy get_feature() pass entirely (see _build_layout_assignments).
    feature_only: bool = False
    layout_feature_col: str = "_dripper_layout_feature"

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
        # Surface GPU visibility: the cuML/torch clustering path is gated on
        # torch.cuda.is_available(); if this logs cuda=False the actor silently
        # falls back to CPU clustering (slow on large hosts).
        import os as _os

        try:
            import torch as _torch

            _cuda = _torch.cuda.is_available()
            _dev = _torch.cuda.device_count() if _cuda else 0
        except Exception as _exc:  # noqa: BLE001
            _cuda, _dev = f"import-error: {_exc}", 0
        logger.info(
            "DripperHTMLLayoutClusteringStage setup complete (torch.cuda.is_available={}, device_count={}, "
            "CUDA_VISIBLE_DEVICES={!r}, gpu_cluster_min_n={})",
            _cuda,
            _dev,
            _os.environ.get("CUDA_VISIBLE_DEVICES"),
            _GPU_CLUSTER_MIN_N,
        )

    def process(self, batch: DocumentBatch) -> DocumentBatch:  # noqa: C901
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        # Normalize the index to 0..len-1: in the streaming pipeline, bin-packed blocks
        # arrive with non-contiguous (or duplicate) pandas labels. Both `track_id` and the
        # positional `layout_ids` list below key off this index, so a label >= len(df) would
        # raise IndexError (and duplicate labels would silently collide).
        df = df.reset_index(drop=True)
        logger.debug("Clustering: {} rows", len(df))
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        if self.feature_only:
            # Phase A (CPU partition): precompute the llm-webkit DOM feature per row into
            # layout_feature_col (JSON string) and return WITHOUT clustering. The GPU phase
            # reads this column so no CPU get_feature() runs on the GPU node.
            if self._web_bindings is None:
                self.setup()
            features: list[str] = [""] * len(df)
            for idx, row in df.iterrows():
                html_text = self._row_feature_html(row)
                if not html_text.strip():
                    continue
                try:
                    feat = self._web_bindings.get_feature(html_text)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper feature precompute failed for row {}: {}", idx, exc)
                    feat = None
                features[idx] = json.dumps(feat) if feat else ""
            df[self.layout_feature_col] = features
            # Drop the raw WARC bytes (binary_content): parse already consumed them and they are
            # dead weight (~250 KB/row). Removing them here keeps the persisted Phase-A output and
            # the GPU phase that reads it lean (the prior compaction-time drop silently no-op'd).
            df = df.drop(columns=[c for c in ("binary_content",) if c in df.columns])
            return DocumentBatch(
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

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

    def _build_layout_assignments(self, df: pd.DataFrame) -> list[_LayoutClusterAssignment]:  # noqa: C901
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Per-row signature lookup keyed by full-batch label, built once here so the downstream
        # signature groupby (_add_signature_grouped_indexes) never pays a per-row df.loc.
        rows_by_label: dict[Any, dict[str, Any]] = {}
        # df.iterrows() builds a full Series per row, copying the ~50KB html column each time --
        # ~tens of ms/row, the dominant cost on big hosts (tgcom24's 20k rows ~= 13 min). Instead
        # pull the needed columns to lists once and hand the row helpers a tiny dict per row
        # (string refs, no copy); they use row.get()/`in`, which behave the same on a dict.
        needed_cols = [
            c
            for c in (
                self.host_col,
                self.url_col,
                self.simplified_html_col,
                self.mapped_html_col,
                self.html_col,
                self.item_count_col,
                self.layout_feature_col,
                _DRIPPER_NEEDS_LLM_COL,
            )
            if c and c in df.columns
        ]
        col_lists = {c: df[c].tolist() for c in needed_cols}
        # CRITICAL: track_id must be the row's FULL-BATCH position, not its positional offset
        # within this (possibly per-host groupby) subframe. process() writes results back via
        # layout_ids[assignment.row_index] into the whole-batch array, so for any host that is
        # not first in a bin-packed multi-host batch, a positional track_id (0..len(host_df)-1)
        # would overwrite an EARLIER host's rows -> cross-host cluster contamination. The batch
        # is reset_index'd upstream (label == full-batch position), so df.index gives it.
        index_labels = df.index.tolist()
        has_needs_llm = _DRIPPER_NEEDS_LLM_COL in df.columns
        has_feature_col = self.layout_feature_col in df.columns
        for idx in range(len(df)):
            row = {c: col_lists[c][idx] for c in needed_cols}
            if has_needs_llm and not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            try:
                # Prefer the Phase-A precomputed feature column (JSON string). When present, SKIP
                # _row_feature_html entirely: decoding the html (zlib.decompress on the compressed
                # column, ~tens of ms x N rows) is pure waste here -- the feature is already computed
                # and the sample's html is unused downstream -- and on a mega-host it stalls the GPU
                # node (CPU-bound) into the watchdog. Only the no-feature fallback decodes html for
                # get_feature(). Keys come back as strings after the JSON round-trip;
                # _simp/_gpu_cluster_html_struct coerce them via int(k).
                raw = row.get(self.layout_feature_col) if has_feature_col else None
                if isinstance(raw, str) and raw:
                    feature = json.loads(raw)
                    html_text = ""
                else:
                    html_text = self._row_feature_html(row)
                    if not html_text.strip():
                        continue
                    feature = self._web_bindings.get_feature(html_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper pre-layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            samples_by_host[self._row_host_key(row)].append(
                {"track_id": str(index_labels[idx]), "html": html_text, "feature": feature}
            )
            rows_by_label[index_labels[idx]] = row

        assignments: list[_LayoutClusterAssignment] = []
        for host_key, samples in samples_by_host.items():
            # Safety net: only pathological hosts (> _GPU_CLUSTER_MAX_N pages) get chunked, to
            # bound the dense n x n similarity matrix. Sparse GPU vectorize handles the rest
            # whole. No pages dropped -- every chunk is fully clustered into its own namespace.
            if len(samples) > _GPU_CLUSTER_MAX_N:
                for ci in range(0, len(samples), _GPU_CLUSTER_MAX_N):
                    chunk = samples[ci : ci + _GPU_CLUSTER_MAX_N]
                    chunk_key = f"{host_key}#c{ci // _GPU_CLUSTER_MAX_N}"
                    assignments.extend(self._build_host_layout_assignments(rows_by_label, chunk_key, chunk))
            else:
                assignments.extend(self._build_host_layout_assignments(rows_by_label, host_key, samples))
        return assignments

    def _build_host_layout_assignments(  # noqa: C901, PLR0912
        self,
        rows_by_label: dict[Any, dict[str, Any]],
        host_key: str,
        samples: list[dict[str, Any]],
    ) -> list[_LayoutClusterAssignment]:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        if len(samples) < self.layout_template_min_cluster_size:
            return []

        grouped_samples: dict[str, list[int]] = defaultdict(list)
        clustered_samples = None
        gpu_reassigned = False
        if len(samples) >= _GPU_CLUSTER_MIN_N:
            try:
                gpu_result = _gpu_cluster_html_struct(samples, threshold=self.layout_cluster_threshold)
                if gpu_result is not None:
                    clustered_samples, _layout_ids = gpu_result
                    gpu_reassigned = True  # sample["layout_id"] is already the exemplar-reassigned layout
                    logger.info(
                        "Dripper GPU clustering host={} n={} layout_ids={}",
                        host_key,
                        len(samples),
                        len({lid for lid in _layout_ids if lid >= 0}),
                    )
                else:
                    logger.warning(
                        "Dripper GPU clustering returned None for host={} n={} (torch.cuda unavailable?); "
                        "falling back to CPU",
                        host_key,
                        len(samples),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Dripper GPU clustering failed for host {} (n={}), falling back to CPU: {}",
                    host_key,
                    len(samples),
                    exc,
                )
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

        if gpu_reassigned:
            # GPU path already assigned each page to its exemplar-matched layout (or -2),
            # reusing the similarity matrix -- no per-page CPU similarity recompute needed.
            for sample in clustered_samples:
                layout_id = int(sample.get("layout_id", -2))
                if layout_id < 0:
                    continue
                grouped_samples[f"__pending_dom_{layout_id:06d}"].append(int(sample["track_id"]))
        else:
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
                rows_by_label,
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

    @property
    def _feature_source(self) -> str:
        return self.layout_feature_source

    def _add_signature_grouped_indexes(  # noqa: PLR0913
        self,
        rows_by_label: dict[Any, dict[str, Any]],
        grouped_samples: dict[str, list[int]],
        *,
        host_key: str,
        layout_key: str,
        fingerprint: str,
        indexes: list[int],
    ) -> None:
        # Fast path: a constant page-signature ("none") maps every row in the group to the SAME
        # stable id -- assign the whole group in one shot, skipping the per-row signature loop.
        if not self.layout_page_signature_mode or self.layout_page_signature_mode == "none":
            grouped_samples[self._stable_layout_id(host_key, layout_key, fingerprint, "")].extend(indexes)
            return
        # rows_by_label gives a tiny per-row dict (url/item_count refs, no html copy) keyed by label,
        # so this groupby never pays the per-row df.loc that copies the ~50KB html column -- the
        # dominant Phase-B cost on big hosts (tgcom24's 20k rows ~= 13 min of pure row-copying).
        low_card_query_keys: set[str] = set()
        if _uses_low_card_query_shape(self.layout_page_signature_mode) and self.url_col:
            low_card_query_keys = _low_card_query_value_keys(
                [rows_by_label[row_idx].get(self.url_col) for row_idx in indexes]
            )
        for row_idx in indexes:
            row = rows_by_label[row_idx]
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
