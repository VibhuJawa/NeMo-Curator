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

"""stage3b_fallback_llm.py — route Stage 3 propagation failures to the LLM.

The standalone Dripper uses `--layout-template-fallback-llm`: when layout
propagation fails for a sibling, it runs the LLM on that page instead of leaving
it empty. Our pipeline left `propagation_method=="fallback"` siblings with empty
content (F1==0), which is the dominant drag on overall F1. This stage closes that
gap:

  mode=build : read Stage 3 output, select the fallback siblings, attach their raw
               HTML (from the Stage 1b manifest), and emit a fallback-input parquet
               shaped like Stage 1b output with cluster_role="singleton" so the
               existing Stage 1c → Stage 2 → Stage 2b chain re-infers them.

  mode=merge : read the original Stage 3 output and the Stage 2b output of the
               re-inferred fallbacks, and replace each fallback row's content with
               the LLM result (propagation_method="fallback_llm"). Writes the final
               merged Stage 3 parquet.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _read_concat(path_glob, columns=None):
    files = sorted(glob.glob(path_glob))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        names = pq.read_schema(f).names
        cols = [c for c in columns if c in names] if columns else None
        frames.append(pq.read_table(f, columns=cols).to_pandas())
    return pd.concat(frames, ignore_index=True)


def build(args):
    s3 = _read_concat(
        f"{args.stage3.rstrip('/')}/*.parquet", ["url", "url_host_name", "cluster_id", "propagation_method"]
    )
    fb = s3[s3["propagation_method"] == "fallback"]
    print(
        f"[stage3b] {len(fb):,} fallback siblings of {len(s3):,} stage3 rows ({len(fb) / max(len(s3), 1) * 100:.1f}%)",
        flush=True,
    )
    fb_urls = set(fb["url"].astype(str))
    if not fb_urls:
        print("[stage3b] no fallbacks — nothing to re-infer", flush=True)

    # Attach HTML + WARC locators from the Stage 1b manifest for the fallback urls.
    man_cols = ["url", "url_host_name", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    rows = []
    seen = set()
    for f in sorted(glob.glob(f"{args.stage1b.rstrip('/')}/*.parquet")):
        names = pq.read_schema(f).names
        cols = [c for c in man_cols if c in names]
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4000, columns=cols):
            for r in batch.to_pylist():
                u = str(r.get("url", ""))
                if u in fb_urls and u not in seen:
                    seen.add(u)
                    r["cluster_id"] = ""  # treat as singleton for re-inference
                    r["cluster_role"] = "singleton"
                    rows.append(r)
    out_df = pd.DataFrame(rows)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) / "shard_0000.parquet"
    out_df.to_parquet(str(out_path), index=False, compression="snappy")
    print(f"[stage3b] build: wrote {len(out_df):,} fallback pages → {out_path}", flush=True)


def merge(args):
    s3 = _read_concat(f"{args.stage3.rstrip('/')}/*.parquet")
    llm = _read_concat(
        f"{args.fallback_stage2b.rstrip('/')}/*.parquet", ["url", "dripper_content", "dripper_html", "dripper_error"]
    )
    print(f"[stage3b] merge: stage3={len(s3):,} rows, re-inferred fallbacks={len(llm):,}", flush=True)
    llm = llm.drop_duplicates(subset="url", keep="first").set_index("url")
    content_map = llm["dripper_content"].to_dict()
    html_map = llm["dripper_html"].to_dict() if "dripper_html" in llm.columns else {}

    n_replaced = 0
    s3 = s3.copy()
    s3_url = s3["url"].astype(str)
    is_fb = s3["propagation_method"] == "fallback"
    for idx in s3.index[is_fb]:
        u = s3_url.loc[idx]
        content = content_map.get(u)
        if isinstance(content, str) and content:
            s3.at[idx, "dripper_content"] = content
            if html_map.get(u):
                s3.at[idx, "dripper_html"] = html_map[u]
            s3.at[idx, "propagation_method"] = "fallback_llm"
            s3.at[idx, "propagation_success"] = True
            s3.at[idx, "dripper_error"] = ""
            n_replaced += 1
    print(f"[stage3b] merge: replaced {n_replaced:,} fallback rows with LLM content", flush=True)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) / "shard_0000.parquet"
    s3.to_parquet(str(out_path), index=False, compression="snappy")
    vc = s3["propagation_method"].value_counts().to_dict()
    print(f"[stage3b] merge: wrote {len(s3):,} rows → {out_path}", flush=True)
    print(f"[stage3b] propagation_method: {vc}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["build", "merge"])
    p.add_argument("--stage3", required=True, help="Stage 3 output dir")
    p.add_argument("--stage1b", help="Stage 1b manifest dir (build mode: HTML source)")
    p.add_argument("--fallback-stage2b", help="Stage 2b output of re-inferred fallbacks (merge mode)")
    p.add_argument("--output", required=True, help="Output dir")
    args = p.parse_args()
    if args.mode == "build":
        if not args.stage1b:
            p.error("--stage1b required for build mode")
        build(args)
    else:
        if not args.fallback_stage2b:
            p.error("--fallback-stage2b required for merge mode")
        merge(args)


if __name__ == "__main__":
    main()
