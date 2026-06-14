#!/usr/bin/env python3
"""validate_stage3_fix.py — fast correctness probe for the Stage 3 input-dir fix.

Confirms that stage2b's mapping_json, fed through the Stage 3 propagation kernel,
actually produces non-empty content for sibling pages (i.e. the _sanitize() JSON
round-trip did not break LayoutBatchParser, and html is present for siblings).

Runs on a SAMPLE of clusters only — meant for a <5 min cpu_short job.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import stage3_cpu_propagation as s3

# Maximum sibling pages to sample per cluster, for diverse coverage.
_MAX_SIBLING_PER_CLUSTER = 8
# Minimum non-empty dripper_content length to count as a successful extraction.
_MIN_CONTENT_LEN = 5


def _load_sibling_sample(
    stage1b_path: str,
    gpu_lookup: dict,
    max_siblings: int,
    max_clusters: int,
) -> tuple[dict, int]:
    """Stream stage1b parquet; collect a capped sample of sibling rows."""
    f1 = sorted(glob.glob(f"{stage1b_path}/shard_*.parquet") or glob.glob(f"{stage1b_path}/*.parquet"))[0]
    pf = pq.ParquetFile(f1)
    cols = [c for c in ["url", "url_host_name", "cluster_id", "cluster_role", "html"] if c in pf.schema_arrow.names]

    by_cluster: dict[str, list] = defaultdict(list)
    n_sib = 0
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        recs = batch.to_pylist()
        for r in recs:
            if str(r.get("cluster_role")) != "sibling":
                continue
            cid = r.get("cluster_id")
            if cid is None:
                continue
            cid = str(cid)
            if cid not in gpu_lookup:
                continue
            if len(by_cluster[cid]) >= _MAX_SIBLING_PER_CLUSTER:
                continue
            by_cluster[cid].append(r)
            n_sib += 1
            if n_sib >= max_siblings or len(by_cluster) >= max_clusters:
                break
        if n_sib >= max_siblings or len(by_cluster) >= max_clusters:
            break
    return by_cluster, n_sib


def _print_sample_cluster_info(cid: str, xpath_rules: object, mapping_data: object, rep_len: int) -> None:
    """Print diagnostic info for the first cluster processed."""
    print(
        f"[validate] sample cluster {cid}: xpath_rules={'yes' if xpath_rules else 'no'} "
        f"mapping_data={'yes' if mapping_data else 'no'} rep_content_len={rep_len}",
        flush=True,
    )
    if mapping_data:
        print(f"[validate]   mapping_data keys: {list(mapping_data.keys())[:12]}", flush=True)  # type: ignore[union-attr]


def _process_clusters(
    by_cluster: dict,
    gpu_lookup: dict,
) -> tuple[dict, int, dict, int]:
    """Run propagation on sampled clusters; return (methods, content_ok, errors, processed)."""
    methods: dict[str, int] = defaultdict(int)
    content_ok = 0
    errors: dict[str, int] = defaultdict(int)
    processed = 0

    for cid, rows in by_cluster.items():
        gpu_row = gpu_lookup[cid]
        xpath_rules = s3._parse_xpath_rules(gpu_row.get("xpath_rules"))
        mapping_data = s3._parse_mapping_json(gpu_row.get("mapping_json") or gpu_row.get("llm_output_raw"))
        rep_len = len(str(gpu_row.get("dripper_content", "")))
        if processed == 0:
            _print_sample_cluster_info(cid, xpath_rules, mapping_data, rep_len)
        for r in rows:
            out = s3._process_sibling_row(r, xpath_rules, mapping_data, rep_len)
            methods[out["propagation_method"]] += 1
            if out["dripper_content"] and len(out["dripper_content"]) > _MIN_CONTENT_LEN:
                content_ok += 1
            if out["dripper_error"]:
                errors[out["dripper_error"][:60]] += 1
            processed += 1

    return methods, content_ok, errors, processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1b", required=True)
    ap.add_argument("--stage2b", required=True)
    ap.add_argument("--max-siblings", type=int, default=200)
    ap.add_argument("--max-clusters", type=int, default=40)
    args = ap.parse_args()

    # Init the worker bindings in-process (no pool — we want tracebacks)
    s3._worker_init(0.70, True, 0.25, 4.0, "INFO")
    print(f"[validate] llm_web_kit bindings: {'OK' if s3._WORKER_BINDINGS else 'MISSING'}", flush=True)
    print(f"[validate] mineru bindings:      {'OK' if s3._WORKER_MINERU_BINDINGS else 'MISSING'}", flush=True)

    # --- Load stage2b gpu results, build cluster_id -> row lookup ---
    b2 = sorted(glob.glob(f"{args.stage2b}/shard_*.parquet") or glob.glob(f"{args.stage2b}/*.parquet"))[0]
    gpu_df = s3._load_inference_results(b2)
    gpu_lookup = s3._build_gpu_lookup(gpu_df)
    print(f"[validate] stage2b rows={len(gpu_df)}  cluster lookup={len(gpu_lookup)}", flush=True)

    by_cluster, n_sib = _load_sibling_sample(args.stage1b, gpu_lookup, args.max_siblings, args.max_clusters)
    print(f"[validate] sampled {n_sib} sibling pages across {len(by_cluster)} clusters", flush=True)

    t0 = time.perf_counter()
    methods, content_ok, errors, processed = _process_clusters(by_cluster, gpu_lookup)
    elapsed = time.perf_counter() - t0

    print(
        f"\n[validate] === RESULTS ({processed} siblings, {elapsed:.1f}s, "
        f"{processed / max(elapsed, 1e-6):.2f} pages/s) ===",
        flush=True,
    )
    print(f"[validate] content_ok (non-empty): {content_ok}/{processed}", flush=True)
    print(f"[validate] methods: {dict(methods)}", flush=True)
    print("[validate] top errors:", flush=True)
    for e, c in sorted(errors.items(), key=lambda x: -x[1])[:10]:
        print(f"    {c:>5}  {e}", flush=True)


if __name__ == "__main__":
    main()
