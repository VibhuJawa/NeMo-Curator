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

"""
gpu_layout_clustering.py — GPU-accelerated layout clustering using cuML DBSCAN.

Replaces the O(N²) Python loop in llm-webkit's cluster_html_struct with:
  1. Vectorized cosine similarity on GPU via cupy matrix ops
  2. cuML DBSCAN (GPU-accelerated, replaces sklearn DBSCAN)

Drop-in replacement for cluster_html_struct — same inputs/outputs.

Performance:
  - CPU (sklearn): N=3000 pages → ~25 min (4.5M cosine calls in Python loop)
  - GPU (cuML):    N=3000 pages → ~5-10s  (batched cuBLAS matmul on H100)

Falls back gracefully to sklearn when:
  - CUDA not available
  - cuML / cupy not installed
  - Cluster smaller than GPU_MIN_SIZE (overhead not worth it)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import cupy as cp

# Minimum cluster size to use GPU path (smaller clusters faster on CPU)
GPU_MIN_SIZE = 200


def _gpu_available() -> bool:
    """Return True if a CUDA device and cupy are usable in this process."""
    try:
        import cupy as cp

        _ = cp.cuda.Device(0).compute_capability  # raises if no GPU
    except Exception:  # noqa: BLE001 - any import/runtime error means no usable GPU
        return False
    return True


def _feature_matrices(features_vec: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack vectorized feature dicts into (tag_matrix, attr_matrix) float32 arrays."""
    tags = np.stack([f["tags"] for f in features_vec]).astype(np.float32)  # (N, D_tag)
    attrs = np.stack([f["attrs"] for f in features_vec]).astype(np.float32)  # (N, D_attr)
    return tags, attrs


def _cosine_similarity_gpu(x: cp.ndarray) -> cp.ndarray:
    """Compute the full NxN cosine similarity matrix on GPU using cuBLAS matmul.

    For N=3000: one batched matmul vs 4.5M Python loop iterations.
    """
    import cupy as cp

    norms = cp.linalg.norm(x, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-10)
    x_norm = x / norms
    return x_norm @ x_norm.T  # (N, D) @ (D, N) -> (N, N) cosine similarity


def cluster_html_struct_gpu(
    sampled_list: list[dict],
    threshold: float = 0.95,
    gpu_min_size: int = GPU_MIN_SIZE,
    tag_weight: float = 0.7,
) -> tuple[list[dict], list[int]]:
    """GPU-accelerated drop-in replacement for llm-webkit's cluster_html_struct.

    Uses cuML DBSCAN + cupy batched cosine similarity for large clusters.
    Falls back to sklearn for small clusters or when GPU unavailable.

    Args:
        sampled_list: same format as cluster_html_struct — list of dicts with 'feature' key
        threshold: cosine similarity threshold, default 0.95 (eps = 1 - threshold)
        gpu_min_size: use GPU path only for clusters with >= this many pages
        tag_weight: weight for tag features (attr weight = 1 - tag_weight)

    Returns:
        (success, layout_ids) — identical format to cluster_html_struct
    """
    n = len(sampled_list)

    # ── Build feature vectors (CPU, reuse llm-webkit logic) ──────────────────
    # Import internal helpers from the installed llm-webkit package
    import llm_web_kit.html_layout.html_layout_cosin as _cosin_mod
    from llm_web_kit.html_layout.html_layout_cosin import cluster_html_struct as _sklearn_cluster

    # Small clusters: use sklearn (GPU overhead not worth it)
    use_gpu = n >= gpu_min_size and _gpu_available()

    if not use_gpu:
        logger.debug(
            "cluster_html_struct_gpu: n={} < gpu_min_size={} or no GPU — using sklearn",
            n,
            gpu_min_size,
        )
        return _sklearn_cluster(sampled_list, threshold)

    # ── GPU path ──────────────────────────────────────────────────────────────
    logger.info("cluster_html_struct_gpu: n={} pages — using GPU (cuML DBSCAN + cupy cosine)", n)
    try:
        return _cluster_gpu(sampled_list, threshold, tag_weight, _cosin_mod)
    except Exception as exc:  # noqa: BLE001 - fall back to sklearn on any GPU failure
        logger.warning("GPU clustering failed ({}) — falling back to sklearn", exc)
        return _sklearn_cluster(sampled_list, threshold)


def _cluster_gpu(
    sampled_list: list[dict],
    threshold: float,
    tag_weight: float,
    cosin_mod: ModuleType,
) -> tuple[list[dict], list[int]]:
    """Core GPU clustering implementation."""
    import cuml.cluster
    import cupy as cp

    features = [s["feature"] for s in sampled_list]

    # Step 1: Vectorize features on CPU (DictVectorizer, same as sklearn path)
    _simp_features_fn = _get_simp_features(cosin_mod)
    layer_n, features_vec = _simp_features_fn(features)

    tags, attrs = _feature_matrices(features_vec)

    # Step 2: GPU cosine similarity — one matmul per feature type
    tags_gpu = cp.asarray(tags)
    attrs_gpu = cp.asarray(attrs)

    tag_sim = _cosine_similarity_gpu(tags_gpu)  # (N, N) on GPU
    attr_sim = _cosine_similarity_gpu(attrs_gpu)  # (N, N) on GPU

    # Step 3: Weighted combination (tag=0.7, attr=0.3)
    # For rows where attr norm == 0, use tag_sim only (matches __cosin_simil logic)
    attr_norms = cp.linalg.norm(attrs_gpu, axis=1)  # (N,)
    no_attr = attr_norms == 0  # (N,) bool mask

    sim_matrix = tag_weight * tag_sim + (1 - tag_weight) * attr_sim  # (N, N)

    # Override rows/cols with no attrs to use tag_sim only
    if cp.any(no_attr):
        sim_matrix[no_attr, :] = tag_sim[no_attr, :]
        sim_matrix[:, no_attr] = tag_sim[:, no_attr]

    sim_matrix = cp.clip(sim_matrix, 0, 1)
    dist_matrix = 1.0 - sim_matrix  # distance = 1 - cosine_similarity

    # Step 4: DBSCAN on precomputed distance matrix
    # GPU matmul already computed the full NxN matrix — sklearn DBSCAN on
    # the precomputed numpy array is O(N²) table lookup, not O(N²) Python loop.
    # cuML DBSCAN with metric='precomputed' is also supported in ≥22.06.
    eps = float(1.0 - threshold)
    dist_np = cp.asnumpy(dist_matrix)  # NxN float32 numpy array

    try:
        # Prefer cuML for the final DBSCAN step (stays GPU-adjacent)
        dbscan = cuml.cluster.DBSCAN(
            eps=eps,
            min_samples=2,
            metric="precomputed",
            output_type="numpy",
        )
        layout_ids = dbscan.fit_predict(dist_np)
    except Exception as exc:  # noqa: BLE001 - fall back to sklearn on any cuML failure
        # Fall back to sklearn — still faster than O(N²) Python loop because
        # the expensive cosine similarity step was already done on GPU.
        logger.debug("cuML DBSCAN precomputed failed ({}), using sklearn", exc)
        layout_ids = _sklearn_dbscan(dist_np, eps)

    layout_ids = [int(x) for x in layout_ids]

    success = []
    for idd, sample in zip(layout_ids, sampled_list, strict=False):
        sample["layout_id"] = idd
        sample["max_layer_n"] = layer_n
        success.append(sample)

    n_clusters = len({x for x in layout_ids if x >= 0})
    n_noise = sum(1 for x in layout_ids if x < 0)
    logger.info("cluster_html_struct_gpu: n={} → {} clusters ({} noise)", len(sampled_list), n_clusters, n_noise)
    return success, list(set(layout_ids))


def _get_simp_features(cosin_mod: ModuleType) -> Callable:
    """Return llm-webkit's feature-vectorization function.

    The helper that turns raw layout features into the (tags, attrs) vectors lives
    in ``llm_web_kit.html_layout.html_layout_cosin`` as a module-private function.
    Python name-mangles a module-level ``__simp_features`` to
    ``_<module>__simp_features``, so we look up both that mangled name and the
    bare name explicitly. We raise a clear error if neither is present (rather
    than silently scanning ``dir()``) so an upstream rename surfaces immediately.
    """
    for name in ("_html_layout_cosin__simp_features", "__simp_features", "simp_features"):
        fn = getattr(cosin_mod, name, None)
        if callable(fn):
            return fn
    msg = (
        "Could not find the feature-vectorization helper (__simp_features) in "
        "llm_web_kit.html_layout.html_layout_cosin; the GPU clustering path needs it. "
        "The llm_web_kit internal API may have changed."
    )
    raise RuntimeError(msg)


def _sklearn_dbscan(dist_matrix: np.ndarray, eps: float) -> list[int]:
    """Thin sklearn DBSCAN wrapper for fallback."""
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    return clustering.fit_predict(dist_matrix).tolist()
