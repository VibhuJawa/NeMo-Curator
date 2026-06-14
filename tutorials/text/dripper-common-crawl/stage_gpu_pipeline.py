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

"""Combined Stage 1c + Stage 2 + Stage 2b in a single GPU job.

Eliminates two intermediate parquet round-trips and two Slurm queue waits.
INPUT:  Stage 1b output dir. OUTPUT: combined parquet with Stage 2b schema.
RUNS ON: batch GPU partition (8xH100). Replaces JOB1c + JOB2 + JOB2b.

NOTE: The CPU stages (1c preprocessing and 2b postprocessing) use library stages:
    DripperHTMLPreprocessStage  -- from nemo_curator.stages.text.experimental.dripper
    DripperHTMLPostprocessStage -- from nemo_curator.stages.text.experimental.dripper

The GPU inference (Stage 2) uses offline vLLM batching (LLM.generate) for maximum
throughput on multi-GPU nodes. For online/server inference, use DripperHTMLInferenceStage
with an OpenAI-compatible client (e.g., vLLM server, NIM).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper import DripperHTMLPostprocessStage, DripperHTMLPreprocessStage
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "dripper_inference_time_s",
]

_MIN_CONTENT_LEN = 5
_MIN_PROMPT_LEN = 10


def run_stage1c(df: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 1c HTML preprocessing via DripperHTMLPreprocessStage."""
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    t0 = time.perf_counter()
    chunk = max(1, len(df) // n_workers)
    initial_tasks = [
        DocumentBatch(dataset_name="stage1c", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    # Simple Curator pattern: library stage -> pipeline -> run()
    stage = DripperHTMLPreprocessStage(html_col="html", url_col="url", worker_count=n_workers)
    pipeline = Pipeline(name="stage1c")
    pipeline.add_stage(stage)
    output_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=initial_tasks) or []

    result_df = pd.concat([t.to_pandas() for t in output_tasks], ignore_index=True)
    elapsed = time.perf_counter() - t0
    ok = (
        int((result_df["_dripper_prompt"].astype(str).str.len() > _MIN_PROMPT_LEN).sum())
        if "_dripper_prompt" in result_df.columns
        else 0
    )
    print(f"[gpu-pipeline] Stage 1c: {ok:,}/{len(df):,} prompts in {elapsed:.1f}s", flush=True)
    return result_df


def _chat_format(tok: object, prompt: str, supports_think: list[bool]) -> str:
    msgs = [{"role": "user", "content": prompt}]
    if supports_think[0]:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            supports_think[0] = False
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@dataclass
class _WorkerConfig:
    model: str
    gpu_mem_util: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_tokens: int
    kv_cache_dtype: str


def run_stage2_worker(gpu_id: int, slice_path: str, out_path: str, cfg: _WorkerConfig) -> None:
    """One GPU worker: offline-batched LLM.generate over its prompt slice."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from nemo_curator.utils.vllm_utils import pick_free_port, resolve_local_model_path

    local_model = resolve_local_model_path(cfg.model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pq.ParquetFile(slice_path).read().to_pandas()
    tok = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
    llm_kw = {
        "model": local_model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": cfg.gpu_mem_util,
        "max_model_len": cfg.max_model_len,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    if cfg.kv_cache_dtype and cfg.kv_cache_dtype != "auto":
        llm_kw["kv_cache_dtype"] = cfg.kv_cache_dtype

    t_setup = time.perf_counter()
    os.environ["MASTER_PORT"] = str(pick_free_port())
    llm = LLM(**llm_kw)
    setup_s = time.perf_counter() - t_setup

    rows = df.to_dict("records")
    supports_think = [True]
    prompts, samplings, ridx, results, n_trunc = [], [], [], [None] * len(rows), 0

    # Use _dripper_prompt column (produced by DripperHTMLPreprocessStage)
    prompt_col = "_dripper_prompt" if "_dripper_prompt" in df.columns else "prompt"
    item_count_col = "dripper_item_count" if "dripper_item_count" in df.columns else "item_count"

    for i, r in enumerate(rows):
        p = str(r.get(prompt_col, "") or "")
        if not p or p.startswith("ERROR:"):
            results[i] = {
                **r,
                "dripper_response": "",
                "dripper_error": p if p.startswith("ERROR:") else "empty_prompt",
                "dripper_inference_time_s": 0.0,
            }
            continue
        try:
            ic = int(r.get(item_count_col, 0) or 0)
        except (TypeError, ValueError):
            ic = 0
        max_tok = min(cfg.max_tokens, max(32, ic * 6 + 16) if ic > 0 else cfg.max_tokens)
        text = _chat_format(tok, p, supports_think)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cap = cfg.max_model_len - max_tok - 8
        if len(ids) > cap:
            ids = ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": ids})
        samplings.append(SamplingParams(temperature=0.0, max_tokens=max_tok))
        ridx.append(i)

    t1 = time.perf_counter()
    outs = llm.generate(prompts, samplings) if prompts else []
    infer_s = time.perf_counter() - t1

    for j, o in enumerate(outs):
        i = ridx[j]
        resp = o.outputs[0].text if o.outputs else ""
        results[i] = {
            **rows[i],
            "dripper_response": resp,
            "dripper_error": "" if resp else "empty_response",
            "dripper_inference_time_s": infer_s / max(len(outs), 1),
        }

    pd.DataFrame([x for x in results if x is not None]).to_parquet(out_path, index=False, compression="snappy")
    rate = len(prompts) / max(infer_s, 1e-6)
    print(
        f"[gpu-pipeline gpu{gpu_id}] DONE {len(prompts)} prompts ({n_trunc} trunc)"
        f" setup={setup_s:.1f}s infer={infer_s:.1f}s {rate:.1f} pages/s/GPU",
        flush=True,
    )


def run_stage2(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Dispatch Stage 2 across all GPUs (LPT balanced, offline batched)."""
    n_gpus = args.replicas if args.replicas > 0 else _detect_gpus()
    print(f"[gpu-pipeline] Stage 2: {len(df):,} pages over {n_gpus} GPUs", flush=True)
    tmp = Path(args.output) / "_gpu_slices"
    tmp.mkdir(parents=True, exist_ok=True)
    # Use _dripper_prompt column (produced by DripperHTMLPreprocessStage)
    prompt_col = "_dripper_prompt" if "_dripper_prompt" in df.columns else "prompt"
    cost = df[prompt_col].astype(str).str.len().to_numpy() if prompt_col in df.columns else [1] * len(df)
    order = sorted(range(len(df)), key=lambda i: -cost[i])
    bins: list[list[int]] = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for i in order:
        g = min(range(n_gpus), key=lambda k: load[k])
        bins[g].append(i)
        load[g] += int(cost[i])

    slice_paths, out_paths = [], []
    for g in range(n_gpus):
        sp = str(tmp / f"slice_{g}.parquet")
        op = str(tmp / f"out_{g}.parquet")
        df.iloc[bins[g]].to_parquet(sp, index=False)
        slice_paths.append(sp)
        out_paths.append(op)
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--gpu",
                str(g),
                "--slice",
                slice_paths[g],
                "--slice-out",
                out_paths[g],
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
                "--kv-cache-dtype",
                args.kv_cache_dtype,
            ]
        )
        for g in range(n_gpus)
    ]
    rcs = [p.wait() for p in procs]
    print(f"[gpu-pipeline] Stage 2 workers done in {time.perf_counter() - t0:.1f}s codes={rcs}", flush=True)
    frames = [pq.ParquetFile(op).read().to_pandas() for op in out_paths if Path(op).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _detect_gpus() -> int:
    n = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS_PER_NODE", "")
    if n:
        try:
            return int(n.split(":")[-1])
        except ValueError:
            pass
    try:
        r = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=5)
        return max(1, sum(1 for ln in r.stdout.splitlines() if ln.startswith("GPU")))
    except OSError:
        return 1


def run_stage2b(df: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 2b postprocessing via DripperHTMLPostprocessStage."""
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    t0 = time.perf_counter()
    chunk = max(1, len(df) // n_workers)
    initial_tasks = [
        DocumentBatch(dataset_name="stage2b", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    # Simple Curator pattern: library stage -> pipeline -> run()
    stage = DripperHTMLPostprocessStage(
        html_col="html",
        url_col="url",
        raw_response_col="dripper_response",
        fallback="trafilatura",
        output_format="mm_md",
        worker_count=n_workers,
    )
    pipeline = Pipeline(name="stage2b")
    pipeline.add_stage(stage)
    output_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=initial_tasks) or []

    result_df = pd.concat([t.to_pandas() for t in output_tasks], ignore_index=True)
    elapsed = time.perf_counter() - t0
    content_ok = int(
        (result_df["dripper_content"].astype(str).str.len() > _MIN_CONTENT_LEN).sum()
        if "dripper_content" in result_df.columns
        else 0
    )
    print(f"[gpu-pipeline] Stage 2b: content_ok={content_ok:,} in {elapsed:.1f}s", flush=True)
    return result_df


def run(args: argparse.Namespace) -> None:
    t_total = time.perf_counter()
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        inp = exact if exact.exists() else sorted(inp.glob("shard_*.parquet"))[0]
    all_df = pq.ParquetFile(str(inp)).read().to_pandas()
    if "cluster_role" in all_df.columns:
        rep_df = all_df[all_df["cluster_role"].isin(["representative", "singleton"])].reset_index(drop=True)
    else:
        rep_df = all_df.reset_index(drop=True)
    print(
        f"[gpu-pipeline] {len(rep_df):,}/{len(all_df):,} pages sent to LLM "
        f"({len(rep_df) / max(len(all_df), 1) * 100:.1f}%)",
        flush=True,
    )

    t1c = time.perf_counter()
    rep_df = run_stage1c(rep_df)
    t1c_s = time.perf_counter() - t1c

    t2 = time.perf_counter()
    infer_df = run_stage2(rep_df, args)
    t2_s = time.perf_counter() - t2

    # Merge 1c HTML back into inference output for postprocessing
    t2b = time.perf_counter()
    html_cols = ["url"] + [
        c for c in ["dripper_simplified_html", "dripper_mapped_html", "html"] if c in rep_df.columns
    ]
    infer_df = infer_df.merge(rep_df[html_cols], on="url", how="left", suffixes=("", "_1c"))
    for c in ["dripper_simplified_html", "dripper_mapped_html", "html"]:
        if f"{c}_1c" in infer_df.columns:
            infer_df[c] = infer_df[c].fillna(infer_df[f"{c}_1c"])
            infer_df = infer_df.drop(columns=[f"{c}_1c"])
    result_df = run_stage2b(infer_df)
    t2b_s = time.perf_counter() - t2b

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "pipeline_results.parquet")
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    total_s = time.perf_counter() - t_total
    ok = int(
        (result_df["dripper_content"].astype(str).str.len() > _MIN_CONTENT_LEN).sum()
        if "dripper_content" in result_df.columns
        else 0
    )
    print(
        f"[gpu-pipeline] ALL DONE: {len(result_df):,} pages ok={ok} "
        f"total={total_s:.1f}s (1c={t1c_s:.1f}s 2={t2_s:.1f}s 2b={t2b_s:.1f}s) -> {out_path}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--slice")
    p.add_argument("--slice-out")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--replicas", type=int, default=int(os.environ.get("N_GPU_REPLICAS", "0")))
    p.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--hf-cache", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--kv-cache-dtype", default="fp8")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)

    if args.worker:
        cfg = _WorkerConfig(
            model=args.model,
            gpu_mem_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_tokens=args.max_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
        )
        run_stage2_worker(args.gpu, args.slice, args.slice_out, cfg)
    else:
        if not args.input or not args.output:
            p.error("--input and --output required in main mode")
        run(args)


if __name__ == "__main__":
    main()
