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

"""
stage1c_cpu_preprocess.py — CPU-only preprocessing for Stage 2 GPU inference.

RUNS ON: cpu_short partition (no GPU needed).

Reads Stage 1b cluster assignments (representatives + their HTML), runs:
  1. simplify_single_input(case) → simplified HTML with _item_id labels
  2. build_prompt(case, prompt_version) → formatted LLM prompt string

Output per representative: url, cluster_id, cluster_role, prompt, simp_html, map_html, html

Stage 2 GPU reads this and ONLY calls vLLM — no CPU preprocessing on GPU node.

PERFORMANCE:
  ~200-500 pages/s per CPU core for simplification
  Embarrassingly parallel across 64 cores
"""
import argparse, os, re, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_metrics import StageMetrics

OUTPUT_COLS = [
    "url", "url_host_name", "cluster_id", "cluster_role",
    "prompt",       # formatted LLM prompt → fed to vLLM in Stage 2
    "item_count",   # # of _item_id labels → Stage 2 dynamic max_tokens (perf)
    "simp_html",    # simplified HTML with _item_ids → for map_parser_cls in Stage 2b
    "map_html",     # tag-mapped HTML → for map_parser_cls in Stage 2b
    "html",         # original raw HTML → for map_parser_cls in Stage 2b
    "warc_filename", "warc_record_offset", "warc_record_length",
]

_ITEM_ID_RE = re.compile(r"_item_id")

_BINDINGS = None

def _init_worker():
    global _BINDINGS
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    try:
        from nemo_curator.stages.text.experimental.dripper.stage import (
            _load_mineru_html_bindings,
        )
        _BINDINGS = _load_mineru_html_bindings()
    except Exception as e:
        print(f"[stage1c] WARNING: bindings unavailable: {e}", flush=True)
        _BINDINGS = None


def _get_attr(case, attr: str) -> str:
    """Read attribute from case.process_data or case.output_data."""
    for data in (getattr(case, "process_data", None), getattr(case, "output_data", None)):
        if data is not None:
            val = getattr(data, attr, None)
            if val:
                return str(val)
    return ""


def _preprocess_one(rec: dict) -> dict:
    """Run simplify_single_input + build_prompt for one representative page."""
    url   = rec.get("url", "")
    html  = rec.get("html", "") or ""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")

    out = {
        "url":           url,
        "url_host_name": rec.get("url_host_name", ""),
        "cluster_id":    rec.get("cluster_id", ""),
        "cluster_role":  rec.get("cluster_role", ""),
        "prompt":        "",
        "item_count":    0,
        "simp_html":     "",
        "map_html":      "",
        "html":          html,
        "warc_filename":      rec.get("warc_filename"),
        "warc_record_offset": rec.get("warc_record_offset"),
        "warc_record_length": rec.get("warc_record_length"),
    }

    if not _BINDINGS or not html.strip():
        return out

    try:
        case = _BINDINGS.case_cls(_BINDINGS.input_cls(raw_html=html, url=url))
        case = _BINDINGS.simplify_single_input(case)
        simp_html = _get_attr(case, "simpled_html")  # uses module-level helper, no monkey-patch
        map_html  = _get_attr(case, "map_html")
        case = _BINDINGS.build_prompt(case, "short_compact")
        generate_in = getattr(case, "generate_input", None)
        prompt = (str(generate_in.full_prompt) if generate_in and generate_in.full_prompt else "")
        # item_count = # of _item_id labels the model must emit → drives Stage 2
        # dynamic max_tokens (output length scales with item count, not 2048).
        item_count = len(_ITEM_ID_RE.findall(map_html or simp_html or ""))
        out.update({"prompt": prompt, "item_count": item_count,
                    "simp_html": simp_html, "map_html": map_html})
    except Exception as e:
        import traceback
        out["prompt"] = f"ERROR:{type(e).__name__}:{str(e)[:100]}"
        print(f"[stage1c] preprocess error for {url[:60]}: {traceback.format_exc()[-200:]}", flush=True)

    return out


def run(args):
    tracker = StageMetrics("stage1c", shard_index=args.shard_index,
                           num_shards=args.num_shards, n_workers=args.workers)
    tracker.start()

    # Load Stage 1b output — representatives + singletons only
    inp = Path(args.input)
    if inp.is_dir():
        import glob as _g
        files = sorted(_g.glob(str(inp / f"shard_{args.shard_index:04d}.parquet")))
        if not files:
            files = sorted(_g.glob(str(inp / "shard_*.parquet")))
        inp = Path(files[0]) if files else inp

    pf = pq.ParquetFile(str(inp))
    df = pf.read().to_pandas()

    # Filter to pages that need GPU inference
    if "cluster_role" in df.columns:
        mask = df["cluster_role"].isin(["representative", "singleton"])
    elif "is_representative" in df.columns:
        mask = df["is_representative"].astype(bool)
    else:
        mask = pd.Series(True, index=df.index)
    df = df[mask].reset_index(drop=True)

    print(f"[stage1c] {len(df):,} representative/singleton pages to preprocess "
          f"({args.workers} workers)", flush=True)

    if len(df) == 0:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / (f"shard_{args.shard_index:04d}.parquet"
                          if args.num_shards > 1 else "shard_0000.parquet")
        pd.DataFrame(columns=OUTPUT_COLS).to_parquet(str(out_path), index=False)
        tracker.finish(total_pages=0, errors=0)
        tracker.extra = {"prompts_ok": 0}
        tracker.save(args.output)
        return

    records = df.to_dict("records")
    results = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_preprocess_one, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 500 == 0:
                ok_so_far = sum(1 for r in results if len(r.get("prompt", "")) > 10)
                tracker.checkpoint(pages_done=done,
                                   label=f"prompts_ok={ok_so_far}")

    result_df = pd.DataFrame(results)

    # Ensure all output columns present
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet"
                      if args.num_shards > 1 else "shard_0000.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    ok = int((result_df["prompt"].astype(str).str.len() > 10).sum())
    err = len(result_df) - ok
    tracker.finish(total_pages=len(result_df), errors=err)
    tracker.extra = {"prompts_ok": ok}
    tracker.save(args.output)
    print(f"[stage1c] output → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, help="Stage 1b output dir or parquet")
    p.add_argument("--output",      required=True, help="Output dir")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards",  type=int, default=1)
    p.add_argument("--workers",     type=int, default=max(1, (os.cpu_count() or 4) - 2))
    run(p.parse_args())


if __name__ == "__main__":
    main()
