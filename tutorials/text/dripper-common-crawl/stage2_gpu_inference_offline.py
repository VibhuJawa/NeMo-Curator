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

"""stage2_gpu_inference_offline.py — GPU-ONLY vLLM inference, OFFLINE BATCHED.

Productionized H1 serving rewrite. Replaces the Ray-Serve per-request dispatch
(the throughput bottleneck — ~27 pages/s/node) with offline batched generation:
one vllm.LLM engine per GPU, in its own subprocess, fed its whole prompt slice via
a single LLM.generate() call. vLLM does continuous batching internally with zero
per-request IPC. Validated at ~12.8 pages/s/GPU → ~102 pages/s/node (3.8x).

INPUT:  Stage 1c output (url, cluster_id, cluster_role, prompt, item_count,
        simp_html, map_html, html, ...)
OUTPUT: adds llm_response → inference_results.parquet (Stage 2b reads this).

Architecture: parent splits the shard into N GPU slices, spawns N worker
subprocesses (CUDA_VISIBLE_DEVICES pinned), each writes a sub-parquet; parent
merges. F1-safe: identical model / chat-template / dynamic-max-tokens as the
Ray-Serve path — only the request transport differs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "llm_response",
    "simp_html",
    "map_html",
    "html",
    "dripper_error",
    "inference_time_s",
]


def _chat_format(tok, prompt, supports_think):
    msgs = [{"role": "user", "content": prompt}]
    if supports_think[0]:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            supports_think[0] = False
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_worker(args):
    """Subprocess: one GPU, offline batched generate over a slice parquet."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pq.ParquetFile(args.slice).read().to_pandas()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.perf_counter()
    llm_kw = dict(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    # FP8 (H2): online dynamic W8A8 of the bf16 checkpoint — extra prefill compute
    # headroom on H100. kv_cache_dtype=fp8 frees KV memory for bigger batches.
    if args.quantization and args.quantization != "none":
        llm_kw["quantization"] = args.quantization
    if args.kv_cache_dtype and args.kv_cache_dtype != "auto":
        llm_kw["kv_cache_dtype"] = args.kv_cache_dtype
    llm = LLM(**llm_kw)
    setup_s = time.perf_counter() - t0

    rows = df.to_dict("records")
    supports_think = [True]
    prompts, samplings, ridx, n_trunc = [], [], [], 0
    results = [None] * len(rows)
    for i, r in enumerate(rows):
        p = str(r.get("prompt", "") or "")
        if not p or p.startswith("ERROR:"):
            results[i] = {
                **{k: r.get(k, "") for k in OUTPUT_COLS},
                "llm_response": "",
                "dripper_error": p if p.startswith("ERROR:") else "empty_prompt",
                "inference_time_s": 0.0,
            }
            continue
        try:
            ic = int(r.get("item_count", 0) or 0)
        except (TypeError, ValueError):
            ic = 0
        max_tok = min(args.max_tokens, max(32, ic * 6 + 16) if ic > 0 else args.max_tokens)
        text = _chat_format(tok, p, supports_think)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cap = args.max_model_len - max_tok - 8
        if len(ids) > cap:
            ids = ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": ids})
        samplings.append(SamplingParams(temperature=0.0, max_tokens=max_tok))
        ridx.append(i)

    print(f"[s2-offline gpu{args.gpu}] {len(prompts)} prompts ({n_trunc} truncated), setup={setup_s:.1f}s", flush=True)
    t1 = time.perf_counter()
    outs = llm.generate(prompts, samplings) if prompts else []
    infer_s = time.perf_counter() - t1

    passthrough = ("url", "url_host_name", "cluster_id", "cluster_role", "simp_html", "map_html", "html")
    for j, o in enumerate(outs):
        i = ridx[j]
        r = rows[i]
        resp = o.outputs[0].text if o.outputs else ""
        results[i] = {
            **{k: r.get(k, "") for k in passthrough},
            "llm_response": resp,
            "dripper_error": "" if resp else "empty_response",
            "inference_time_s": infer_s / max(len(outs), 1),
        }
    results = [x for x in results if x is not None]
    pd.DataFrame(results).to_parquet(args.out, index=False, compression="snappy")
    rate = len(prompts) / max(infer_s, 1e-6)
    # sidecar so the parent can compute the true pure-inference per-node rate
    # (= total_pages / max worker infer_s) — setup amortizes away at CC scale.
    Path(args.out + ".meta.json").write_text(
        json.dumps(
            {
                "infer_s": round(infer_s, 2),
                "setup_s": round(setup_s, 2),
                "pages": len(results),
                "rate_gpu": round(rate, 2),
            }
        )
    )
    print(
        f"[s2-offline gpu{args.gpu}] DONE {len(results)} pages  {rate:.1f} pages/s/GPU  "
        f"infer={infer_s:.1f}s → {args.out}",
        flush=True,
    )


def _detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True).stdout
        n = sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU "))
        return max(n, 1)
    except Exception:
        return 1


def run(args):
    inp = Path(args.input)
    if inp.is_dir():
        import glob as _g

        files = sorted(_g.glob(str(inp / f"shard_{args.shard_index:04d}.parquet"))) or sorted(
            _g.glob(str(inp / "shard_*.parquet"))
        )
        inp = Path(files[0]) if files else inp
    df = pq.ParquetFile(str(inp)).read().to_pandas()
    n_gpus = args.replicas if args.replicas > 0 else _detect_gpus()
    print(f"[s2-offline] {len(df):,} pages over {n_gpus} GPUs (offline batched)", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_slices"
    tmp.mkdir(exist_ok=True)

    # Balance slices by prompt LENGTH (prefill-dominated cost) via greedy LPT
    # bin-packing so all GPUs finish together — contiguous equal-page slices left
    # the slowest GPU at 54s while the fastest finished in 32s (~70% imbalance).
    t0 = time.perf_counter()
    cost = df["prompt"].astype(str).str.len().to_numpy() if "prompt" in df.columns else [1] * len(df)
    order = sorted(range(len(df)), key=lambda i: -cost[i])
    bins = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for i in order:
        g = min(range(n_gpus), key=lambda k: load[k])
        bins[g].append(i)
        load[g] += int(cost[i])

    procs, slice_paths, out_paths = [], [], []
    for g in range(n_gpus):
        sp = tmp / f"slice_{g}.parquet"
        op = tmp / f"out_{g}.parquet"
        df.iloc[bins[g]].to_parquet(sp, index=False)
        slice_paths.append(sp)
        out_paths.append(op)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--slice",
            str(sp),
            "--out",
            str(op),
            "--gpu",
            str(g),
            "--model",
            args.model,
            "--max-tokens",
            str(args.max_tokens),
            "--gpu-mem-util",
            str(args.gpu_mem_util),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--quantization",
            args.quantization,
            "--kv-cache-dtype",
            args.kv_cache_dtype,
        ]
        procs.append(subprocess.Popen(cmd))
    rc = [p.wait() for p in procs]
    print(f"[s2-offline] workers exit codes: {rc}", flush=True)

    frames = [pq.ParquetFile(str(op)).read().to_pandas() for op in out_paths if op.exists()]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLS)
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "inference_results.parquet")
    result_df.to_parquet(str(out_path), index=False, compression="snappy")

    elapsed = time.perf_counter() - t0
    ok = int((result_df["llm_response"].astype(str).str.len() > 0).sum())
    wall_rate = len(result_df) / max(elapsed, 1e-6)
    # Pure-inference per-node rate (setup amortizes to ~0 at CC scale): total pages
    # over the SLOWEST worker's inference time. Also report setup + imbalance.
    metas = []
    for op in out_paths:
        mp = Path(str(op) + ".meta.json")
        if mp.exists():
            try:
                metas.append(json.loads(mp.read_text()))
            except Exception:
                pass
    max_infer = max((m["infer_s"] for m in metas), default=elapsed)
    min_infer = min((m["infer_s"] for m in metas), default=elapsed)
    max_setup = max((m.get("setup_s", 0) for m in metas), default=0)
    pure_per_node = len(result_df) / max(max_infer, 1e-6)
    imbalance = max_infer / max(min_infer, 1e-6)
    print(
        f"[s2-offline] DONE {len(result_df):,} pages ok={ok}  "
        f"PURE={pure_per_node:.1f} pages/s/node (gated by slowest GPU {max_infer:.1f}s)  "
        f"wall={elapsed:.1f}s ({wall_rate:.1f} incl setup~{max_setup:.0f}s+merge)  "
        f"imbalance={imbalance:.2f}x → {out_path}",
        flush=True,
    )
    metrics = {
        "stage": "stage2",
        "shard_index": args.shard_index,
        "total_pages": len(result_df),
        "successful_pages": ok,
        "elapsed_s": round(elapsed, 2),
        "pages_per_s_per_node": round(pure_per_node, 2),
        "wall_pages_per_s_per_node": round(wall_rate, 2),
        "setup_s": round(max_setup, 1),
        "imbalance_x": round(imbalance, 2),
        "n_gpus": n_gpus,
        "serving": "offline_batched",
    }
    (out / f"metrics_stage2_shard_{args.shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true", help="internal: run one GPU worker")
    p.add_argument("--slice")
    p.add_argument("--out")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--replicas", type=int, default=int(os.environ.get("N_GPU_REPLICAS", "0")))
    p.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--hf-cache", default=os.environ.get("HF_HOME"), help="HuggingFace cache dir (default: $HF_HOME)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--quantization", default="none", help="none|fp8 (online W8A8)")
    p.add_argument("--kv-cache-dtype", default="auto", help="auto|fp8")
    args = p.parse_args()
    if args.hf_cache:
        os.environ.setdefault("HF_HOME", args.hf_cache)
    if args.worker:
        run_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
