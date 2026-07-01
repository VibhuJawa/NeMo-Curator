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

"""GPU-accelerated layout clustering using cuML DBSCAN + cupy cosine similarity."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import cupy as cp


def _gpu_available() -> bool:
    try:
        import cupy as cp

        _ = cp.cuda.Device(0).compute_capability  # raises if no GPU
    except Exception:  # noqa: BLE001 - any import/runtime error means no usable GPU
        return False
    return True


def _feature_matrices(features_vec: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    tags = np.stack([f["tags"] for f in features_vec]).astype(np.float32)
    attrs = np.stack([f["attrs"] for f in features_vec]).astype(np.float32)
    return tags, attrs


def _cosine_similarity_gpu(x: cp.ndarray) -> cp.ndarray:
    import cupy as cp

    norms = cp.linalg.norm(x, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-10)
    x_norm = x / norms
    return x_norm @ x_norm.T  # (N, D) @ (D, N) -> (N, N) cosine similarity


def cluster_html_struct_gpu(
    sampled_list: list[dict],
    threshold: float = 0.95,
    tag_weight: float = 0.7,
) -> tuple[list[dict], list[int]]:
    n = len(sampled_list)
    if n == 0:
        return [], []

    import llm_web_kit.html_layout.html_layout_cosin as _cosin_mod

    if not _gpu_available():
        raise RuntimeError("GPU layout clustering requires a usable CUDA device")
    logger.info("cluster_html_struct_gpu: n={} pages using GPU (cuML DBSCAN + cupy cosine)", n)
    return _cluster_gpu(sampled_list, threshold, tag_weight, _cosin_mod)


def _cluster_gpu(
    sampled_list: list[dict],
    threshold: float,
    tag_weight: float,
    cosin_mod: ModuleType,
) -> tuple[list[dict], list[int]]:
    import cuml.cluster
    import cupy as cp

    try:
        features = [s["feature"] for s in sampled_list]
        _simp_features_fn = _get_simp_features(cosin_mod)
        layer_n, features_vec = _simp_features_fn(features)
        tags, attrs = _feature_matrices(features_vec)

        tags_gpu = cp.asarray(tags)
        attrs_gpu = cp.asarray(attrs)
        tag_sim = _cosine_similarity_gpu(tags_gpu)
        attr_sim = _cosine_similarity_gpu(attrs_gpu)

        attr_norms = cp.linalg.norm(attrs_gpu, axis=1)
        no_attr = attr_norms == 0
        sim_matrix = tag_weight * tag_sim + (1 - tag_weight) * attr_sim
        if cp.any(no_attr):
            sim_matrix[no_attr, :] = tag_sim[no_attr, :]
            sim_matrix[:, no_attr] = tag_sim[:, no_attr]

        dist_matrix = cp.ascontiguousarray((1.0 - cp.clip(sim_matrix, 0, 1)).astype(cp.float32, copy=False))
        eps = float(1.0 - threshold)

        dbscan = cuml.cluster.DBSCAN(
            eps=eps,
            min_samples=2,
            metric="precomputed",
            output_type="cupy",
        )
        raw_layout_ids = dbscan.fit_predict(dist_matrix)
        if hasattr(raw_layout_ids, "to_numpy"):
            layout_ids_arr = raw_layout_ids.to_numpy()
        else:
            with contextlib.suppress(Exception):
                layout_ids_arr = cp.asnumpy(raw_layout_ids)
            if "layout_ids_arr" not in locals():
                layout_ids_arr = np.asarray(raw_layout_ids)
        layout_ids = layout_ids_arr.astype(np.int32, copy=False).tolist()
    finally:
        with contextlib.suppress(Exception):
            del tags_gpu, attrs_gpu, tag_sim, attr_sim, sim_matrix, dist_matrix
        with contextlib.suppress(Exception):
            cp.get_default_memory_pool().free_all_blocks()
        with contextlib.suppress(Exception):
            cp.get_default_pinned_memory_pool().free_all_blocks()

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
    # llm-webkit's __simp_features is module-private; Python mangles it to _<module>__simp_features.
    # We look up both forms so upstream renames surface immediately rather than silently failing.
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
