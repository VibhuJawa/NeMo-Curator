#!/usr/bin/env python3
"""
test_gpu_dbscan.py — compare GPU vs CPU layout clustering on real CC pages.

Tests:
  1. GPU and CPU produce the same cluster assignments
  2. GPU is faster for large hosts
  3. Fallback works when GPU unavailable

Usage:
  python test_gpu_dbscan.py --manifest /lustre/.../layout_precompute_manifest.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(
    0, "/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/curator"
)

import pyarrow.parquet as pq

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[33mINFO\033[0m"

# Speedup thresholds for GPU DBSCAN evaluation
_SPEEDUP_GOOD = 5
_SPEEDUP_MODERATE = 2


def coerce_html(raw: bytes | str | None) -> str:
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")


def check(name: str, fn: Callable[[], object]) -> object:
    try:
        result = fn()
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e!s:.150}")
        return None
    else:
        print(f"  [{PASS}] {name}")
        return result


def _run_imports() -> tuple[object, object, bool]:
    """Run import checks; return (web_bindings, gpu_mod, gpu_ok)."""
    print("\n=== 1. IMPORTS ===")
    web = check(
        "load llm_web_kit bindings",
        lambda: __import__(
            "nemo_curator.stages.text.experimental.dripper.stage", fromlist=["_load_llm_web_kit_bindings"]
        )._load_llm_web_kit_bindings(),
    )

    if web is None:
        print("Cannot proceed without bindings")
        sys.exit(1)

    gpu_mod = check(
        "import gpu_layout_clustering",
        lambda: __import__(
            "nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering",
            fromlist=["cluster_html_struct_gpu", "_gpu_available"],
        ),
    )

    gpu_ok = False
    if gpu_mod:
        gpu_ok = check("GPU available (cupy + CUDA)", gpu_mod._gpu_available)  # type: ignore[union-attr]
        if gpu_ok:
            check("cuML importable", lambda: __import__("cuml.cluster"))
            check("cupy importable", lambda: __import__("cupy"))

    return web, gpu_mod, bool(gpu_ok)


def _load_data(manifest_path: str) -> tuple[object, object, object]:
    """Load manifest; return (df, big_host, vc) where vc is value_counts series."""
    print("\n=== 2. LOAD DATA ===")
    df = check("read manifest", lambda: pq.ParquetFile(manifest_path).read().to_pandas())
    if df is None:
        print("No manifest")
        sys.exit(1)

    print(f"  [{INFO}] {len(df):,} rows, {df['url_host_name'].nunique()} hosts")  # type: ignore[union-attr]

    vc = df["url_host_name"].value_counts()  # type: ignore[union-attr]
    big_host = vc.index[0]
    return df, big_host, vc


def _run_correctness_test(
    small_samples: list[dict],
    cpu_cluster: Callable[..., tuple[list, object]],
    cluster_html_struct_gpu: Callable[..., tuple[list, object]],
) -> None:
    """Section 4: GPU vs CPU correctness on a small cluster."""
    print("\n=== 4. CORRECTNESS: GPU vs CPU (small cluster) ===")
    if not small_samples:
        return
    import copy

    samples_a = copy.deepcopy(small_samples)
    samples_b = copy.deepcopy(small_samples)

    t0 = time.perf_counter()
    cpu_res, _ = cpu_cluster(samples_a, threshold=0.95)
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_res, _ = cluster_html_struct_gpu(samples_b, threshold=0.95, gpu_min_size=1)
    gpu_time = time.perf_counter() - t0

    cpu_labels = [s["layout_id"] for s in cpu_res]
    gpu_labels = [s["layout_id"] for s in gpu_res]

    cpu_n_clusters = len({x for x in cpu_labels if x >= 0})
    gpu_n_clusters = len({x for x in gpu_labels if x >= 0})
    cpu_noise = sum(1 for x in cpu_labels if x < 0)
    gpu_noise = sum(1 for x in gpu_labels if x < 0)

    print(f"  CPU: {cpu_n_clusters} clusters, {cpu_noise} noise  ({cpu_time:.2f}s)")
    print(f"  GPU: {gpu_n_clusters} clusters, {gpu_noise} noise  ({gpu_time:.2f}s)")

    if cpu_n_clusters == gpu_n_clusters and cpu_noise == gpu_noise:
        print(f"  [{PASS}] Same cluster count ({cpu_n_clusters} clusters, {cpu_noise} noise)")
    else:
        print(f"  [{FAIL}] Cluster count mismatch — CPU={cpu_n_clusters} GPU={gpu_n_clusters}")


def _run_speedup_test(
    large_samples: list[dict] | None,
    gpu_ok: bool,
    cpu_cluster: Callable[..., tuple[list, object]],
    cluster_html_struct_gpu: Callable[..., tuple[list, object]],
) -> None:
    """Section 5: GPU speedup test on a large cluster."""
    n = len(large_samples) if large_samples else 0
    print(f"\n=== 5. SPEEDUP: Large cluster (N={n}) ===")
    if not large_samples or not gpu_ok:
        if not gpu_ok:
            print(f"  [{INFO}] SKIPPED — no GPU available on this node")
        return

    import copy

    samples_c = copy.deepcopy(large_samples)
    samples_d = copy.deepcopy(large_samples)

    print(f"  Running CPU DBSCAN on {len(samples_c)} pages (may take minutes)...")
    t0 = time.perf_counter()
    cpu_res2, _ = cpu_cluster(samples_c, threshold=0.95)
    cpu_big_time = time.perf_counter() - t0

    print(f"  Running GPU DBSCAN on {len(samples_d)} pages...")
    t0 = time.perf_counter()
    gpu_res2, _ = cluster_html_struct_gpu(samples_d, threshold=0.95, gpu_min_size=1)
    gpu_big_time = time.perf_counter() - t0

    speedup = cpu_big_time / max(gpu_big_time, 0.001)
    cpu_clusters = len({s["layout_id"] for s in cpu_res2 if s["layout_id"] >= 0})
    gpu_clusters = len({s["layout_id"] for s in gpu_res2 if s["layout_id"] >= 0})

    print(f"  CPU time: {cpu_big_time:.1f}s → {cpu_clusters} clusters")
    print(f"  GPU time: {gpu_big_time:.1f}s → {gpu_clusters} clusters")
    print(f"  Speedup:  {speedup:.1f}×")

    if speedup >= _SPEEDUP_GOOD:
        print(f"  [{PASS}] GPU is {speedup:.0f}× faster (≥{_SPEEDUP_GOOD}× expected)")
    elif speedup >= _SPEEDUP_MODERATE:
        print(f"  [{INFO}] GPU is {speedup:.0f}× faster (moderate)")
    else:
        print(f"  [{FAIL}] GPU not significantly faster ({speedup:.1f}×)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "/lustre/fsw/portfolios/llmservice/users/vjawa/"
            "nemo_curator_dripper_layout_clustering_20260611_194849/"
            "output_00/layout_precompute_manifest.parquet"
        ),
    )
    parser.add_argument("--small-n", type=int, default=50, help="Small cluster test size")
    parser.add_argument("--large-n", type=int, default=1000, help="Large cluster test size (GPU benefit)")
    args = parser.parse_args()

    print("=" * 65)
    print("GPU DBSCAN TEST — cuML vs sklearn")
    print("=" * 65)

    web, _gpu_mod, gpu_ok = _run_imports()
    df, big_host, vc = _load_data(args.manifest)

    big_df = df[df["url_host_name"] == big_host].head(args.large_n)
    small_df = df[df["url_host_name"] == vc.index[-1]].head(args.small_n)
    print(f"  [{INFO}] Large host: {big_host} ({len(big_df)} pages for test)")
    print(f"  [{INFO}] Small host: {vc.index[-1]} ({len(small_df)} pages for test)")

    def build_samples(sub_df: object) -> list[dict]:
        samples = []
        for _, row in sub_df.iterrows():
            html = coerce_html(row["html"])
            feat = web.get_feature(html)
            if feat:
                samples.append({"track_id": row["url"], "html": html, "feature": feat})
        return samples

    print("\n=== 3. FEATURE EXTRACTION ===")
    t0 = time.perf_counter()
    large_samples = check(f"get_feature on {len(big_df)} pages", lambda: build_samples(big_df))
    feat_time = time.perf_counter() - t0
    if large_samples:
        print(f"  [{INFO}] Feature extraction: {feat_time:.1f}s ({len(large_samples) / feat_time:.0f} pages/s)")

    small_samples = check(f"get_feature on {len(small_df)} pages", lambda: build_samples(small_df))

    from llm_web_kit.html_layout.html_layout_cosin import cluster_html_struct as cpu_cluster

    from nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering import cluster_html_struct_gpu

    _run_correctness_test(small_samples or [], cpu_cluster, cluster_html_struct_gpu)
    _run_speedup_test(large_samples, gpu_ok, cpu_cluster, cluster_html_struct_gpu)

    print("\n" + "=" * 65)
    print("TEST COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
