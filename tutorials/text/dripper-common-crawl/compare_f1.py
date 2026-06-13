#!/usr/bin/env python3
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

"""compare_f1.py — token-level F1 of the clustering pipeline vs standalone Dripper.

Treats the standalone Dripper output (run B) as the reference and the 3-stage
clustering+propagation pipeline (Stage 3 output) as the prediction. Reports the
F1 distribution overall and broken down by cluster_role, so we can quantify how
much accuracy clustering+propagation costs vs running the LLM on every page.

F1 is multiset token overlap:
    precision = |pred ∩ ref| / |pred|
    recall    = |pred ∩ ref| / |ref|
    F1        = 2PR / (P+R)
Both-empty → F1=1.0 (agreement). One-empty → F1=0.0.
"""
import argparse, glob, re
from collections import Counter

import pyarrow.parquet as pq

_TOK = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> Counter:
    return Counter(_TOK.findall(text.lower())) if text else Counter()


def f1(pred: str, ref: str) -> float:
    pc, rc = tokenize(pred), tokenize(ref)
    if not pc and not rc:
        return 1.0
    if not pc or not rc:
        return 0.0
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    p = common / sum(pc.values())
    r = common / sum(rc.values())
    return 2 * p * r / (p + r)


def load_url_content(path_glob, content_col):
    out = {}
    for f in sorted(glob.glob(path_glob)):
        pf = pq.ParquetFile(f)
        cols = [c for c in ["url", content_col, "cluster_role"] if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=4000, columns=cols):
            for r in batch.to_pylist():
                u = r.get("url")
                if u is None:
                    continue
                out[str(u)] = (str(r.get(content_col) or ""), str(r.get("cluster_role") or ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="standalone dripper_results.parquet")
    ap.add_argument("--pipeline", required=True, help="Stage 3 output dir (shard_*.parquet)")
    ap.add_argument("--baseline-col", default="dripper_content")
    ap.add_argument("--pipeline-col", default="dripper_content")
    args = ap.parse_args()

    print("[f1] loading baseline...", flush=True)
    base = load_url_content(args.baseline, args.baseline_col)
    print(f"[f1] baseline urls: {len(base):,}", flush=True)

    print("[f1] loading pipeline...", flush=True)
    pglob = args.pipeline if args.pipeline.endswith(".parquet") else f"{args.pipeline.rstrip('/')}/*.parquet"
    pipe = load_url_content(pglob, args.pipeline_col)
    print(f"[f1] pipeline urls: {len(pipe):,}", flush=True)

    common_urls = set(base) & set(pipe)
    print(f"[f1] common urls: {len(common_urls):,}  "
          f"(baseline-only={len(set(base)-set(pipe)):,}  pipeline-only={len(set(pipe)-set(base)):,})",
          flush=True)

    scores = []
    by_role = {}
    n_f0 = n_f80 = n_both_empty = 0
    for u in common_urls:
        pred, role = pipe[u]
        ref, _ = base[u]
        s = f1(pred, ref)
        scores.append(s)
        by_role.setdefault(role or "unknown", []).append(s)
        if s == 0.0:
            n_f0 += 1
        if s >= 0.80:
            n_f80 += 1
        if not pred and not ref:
            n_both_empty += 1

    scores.sort()
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    median = scores[n // 2] if n else 0.0
    p10 = scores[int(0.10 * n)] if n else 0.0
    p25 = scores[int(0.25 * n)] if n else 0.0

    print("\n" + "=" * 64)
    print("  F1: clustering pipeline vs standalone Dripper (reference)")
    print("=" * 64)
    print(f"  pages compared:        {n:,}")
    print(f"  mean F1:               {mean:.4f}")
    print(f"  median F1:             {median:.4f}")
    print(f"  p25 / p10 F1:          {p25:.4f} / {p10:.4f}")
    print(f"  pages F1 >= 0.80:      {n_f80:,}  ({n_f80/max(n,1)*100:.1f}%)")
    print(f"  pages F1 == 0:         {n_f0:,}  ({n_f0/max(n,1)*100:.1f}%)")
    print(f"  both-empty (agree):    {n_both_empty:,}")
    print("  " + "-" * 60)
    print(f"  {'role':<16}{'pages':>10}{'mean F1':>10}{'>=0.80':>10}{'F1==0':>10}")
    for role, ss in sorted(by_role.items()):
        m = sum(ss) / len(ss)
        ge = sum(1 for x in ss if x >= 0.80) / len(ss) * 100
        z = sum(1 for x in ss if x == 0.0) / len(ss) * 100
        print(f"  {role:<16}{len(ss):>10,}{m:>10.4f}{ge:>9.1f}%{z:>9.1f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
