#!/usr/bin/env python3
"""
stage2_serving_proto.py — Serving-architecture prototype for Stage 2 (H1 track).

PURPOSE
  Demonstrate / benchmark the *fastest* serving design for the prefill-heavy,
  short-decode 0.5B MinerU-HTML workload, and quantify it against the current
  custom Ray-Serve `handle.infer.remote` per-request path (27 pages/s/node).

  This file is ILLUSTRATIVE and single-GPU testable. It does NOT touch the
  production stage scripts. Run it on ONE H100 with a small shard to measure
  pages/s/GPU; multiply by 8 for per-node, derate by ~0.85 for the cluster.

THE FINDING (why current Stage 2 is slow)
  The standalone baseline (nemo_curator.core.serve) deploys vLLM via
  `ray.serve.llm.build_openai_app` (the production OpenAI ingress + router with
  its OWN continuous batcher) and drives it with an OpenAI HTTP client at
  `max_concurrent_requests` concurrency. The custom Stage 2, by contrast, sends
  EVERY page through `handle.infer.remote(prompt, rid, ic)` — a Ray *actor
  method RPC*. Each call pays:
    - Python-object (cloudpickle) serialization of prompt+args, both ways,
    - a hop through the Ray object store / actor inbox queue,
    - one async actor task per request, scheduled by Ray's core worker.
  That per-request overhead (~ms-scale each) throttles how many requests are
  actually *in flight* at the vLLM engine, so vLLM's continuous batcher runs
  with a starved batch. The model is tiny (0.5B); the GPU is idle waiting on the
  RPC pipe, not on compute. That is the 27-vs-62 gap.

  => The fix is NOT a different model or generation config. It is to put the
     rows directly into the vLLM engine with hundreds in flight, with no Ray
     actor RPC between the data and the engine.

THREE CANDIDATES (this script can run A and B; C is sketched)
  A) OFFLINE BATCHED  `LLM.generate(list_of_prompts, sampling)`  [RECOMMENDED]
     One vLLM `LLM` per GPU, in the same process as the data shard. Hand the
     engine the ENTIRE shard's prompt list at once; vLLM's scheduler does
     continuous batching internally with zero IPC. This is the lowest-overhead
     path for a batch (non-serving) workload — which Stage 2 is (read a parquet
     shard, write a parquet shard). No HTTP, no Ray Serve, no actor RPC.
  B) ASYNC + SEMAPHORE  AsyncLLM(.generate) with Semaphore(N), N high (~512)
     Same in-process engine, but async streaming. Equivalent throughput to A
     when N is large; useful if you need per-request early-exit/streaming. Still
     no Ray RPC. This is what Stage 2 *should* have been instead of routing
     through a Serve deployment handle.
  C) RAY SERVE OpenAI ingress (`build_openai_app`) + OpenAI HTTP client
     The standalone's path. Works, but adds an HTTP round-trip + router hop per
     request vs. A/B. Use only if you need a long-lived shared server across
     many client processes. For a one-shot shard job, A is strictly simpler and
     at least as fast.

HOW TO DECIDE PER GPU
  Stage 2 is embarrassingly data-parallel: 1 vLLM engine per GPU, each owns a
  disjoint set of shards. Use Ray ONLY to place 8 tasks (one per GPU) — inside
  each task use candidate A (offline `LLM.generate`). No cross-GPU request
  routing. This removes the central Serve router entirely.

USAGE (single GPU, on the cluster)
  PY=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv/bin/python3
  $PY stage2_serving_proto.py \
      --input  /path/to/stage1c_out \
      --shard-index 0 \
      --mode offline \
      --max-pages 4000
  # compare:
  $PY stage2_serving_proto.py ... --mode async --in-flight 512
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_shard(input_dir: str, shard_index: int, max_pages: int) -> pd.DataFrame:
    inp = Path(input_dir)
    if inp.is_dir():
        cand = inp / f"shard_{shard_index:04d}.parquet"
        files = [cand] if cand.exists() else sorted(inp.glob("shard_*.parquet"))
        inp = files[0] if files else inp
    df = pq.ParquetFile(str(inp)).read().to_pandas()
    if max_pages and max_pages > 0:
        df = df.head(max_pages)
    return df


def sampling_for(sampling_params: type, item_count: int, hard_cap: int) -> object:
    """Dynamic max_tokens — proven F1-safe; mirrors stage.py and stage2."""
    cap = max(32, int(item_count) * 6 + 16) if item_count and item_count > 0 else hard_cap
    return sampling_params(temperature=0.0, max_tokens=min(hard_cap, cap))


def chat_format(tokenizer: object, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_engine_common(args: Namespace) -> dict[str, object]:
    """Engine kwargs that mirror the proven standalone config (main.py:1626)."""
    return {
        "model": args.model,
        "tensor_parallel_size": 1,  # data-parallel: 1 engine / GPU
        "gpu_memory_utilization": args.gpu_mem_util,  # 0.90 — bigger KV cache
        "max_model_len": args.max_model_len,  # 32768 — do NOT lower (F1: truncation)
        "max_num_seqs": args.max_num_seqs,  # 512 — raise concurrency; 0.5B under-utilizes default
        "max_num_batched_tokens": args.max_num_batched_tokens,  # 16384
        "enable_chunked_prefill": True,  # smooth long prefills into decode batches
        "enable_prefix_caching": True,  # caches shared template prefix (cheap)
        "enforce_eager": False,  # CUDA graphs on — cuts per-decode-step launch overhead
        "trust_remote_code": True,
        "disable_log_stats": True,
    }


# --------------------------------------------------------------------------- #
# Candidate A: OFFLINE BATCHED  (recommended)
# --------------------------------------------------------------------------- #
def run_offline(args: Namespace, df: pd.DataFrame) -> float:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.perf_counter()
    llm = LLM(**build_engine_common(args))
    setup_s = time.perf_counter() - t0

    rows = df.to_dict("records")
    prompts, samplings, idx = [], [], []
    n_trunc = 0
    for i, r in enumerate(rows):
        p = str(r.get("prompt", "") or "")
        if not p or p.startswith("ERROR:"):
            continue
        try:
            ic = int(r.get("item_count", 0) or 0)
        except (TypeError, ValueError):
            ic = 0
        sp = sampling_for(SamplingParams, ic, args.max_tokens)
        text = chat_format(tok, p)
        # Tokenize and truncate over-length prompts to fit max_model_len, keeping
        # the FRONT (instruction header + as many _item_ids as fit). vLLM hard-errors
        # on prompt+out > max_model_len and kills the engine, so we must clamp here.
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cap = args.max_model_len - (sp.max_tokens or 64) - 8
        if len(ids) > cap:
            ids = ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": ids})
        samplings.append(sp)
        idx.append(i)

    print(
        f"[offline] {len(prompts)} prompts ready; {n_trunc} truncated to fit max_model_len={args.max_model_len}",
        flush=True,
    )
    t1 = time.perf_counter()
    # ONE call. vLLM does continuous batching over the whole list internally,
    # keeping max_num_seqs in flight with zero IPC per request.
    outs = llm.generate(prompts, samplings)
    infer_s = time.perf_counter() - t1

    ok = sum(1 for o in outs if o.outputs and o.outputs[0].text)
    rate = len(prompts) / max(infer_s, 1e-6)
    print(
        f"[offline] pages={len(prompts)} ok={ok} setup_s={setup_s:.1f} "
        f"infer_s={infer_s:.1f}  {rate:.1f} pages/s/GPU  "
        f"=> ~{rate * 8:.0f} pages/s/node (x8 GPU)  "
        f"=> ~{rate * 8 * 0.85:.0f} pages/s/node @85% eff",
        flush=True,
    )
    return rate


# --------------------------------------------------------------------------- #
# Candidate B: ASYNC + high-concurrency SEMAPHORE (in-process, no Ray RPC)
# --------------------------------------------------------------------------- #
def run_async(args: Namespace, df: pd.DataFrame) -> float:
    import uuid

    from transformers import AutoTokenizer

    # vLLM >=0.6: from vllm.v1.engine.async_llm import AsyncLLM
    # vLLM <0.6 : AsyncLLMEngine.from_engine_args(AsyncEngineArgs(...))
    try:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        _new_api = True
    except ImportError:
        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        _new_api = False

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eargs = AsyncEngineArgs(**build_engine_common(args))
    t0 = time.perf_counter()
    engine = AsyncLLM.from_engine_args(eargs) if _new_api else AsyncLLMEngine.from_engine_args(eargs)
    setup_s = time.perf_counter() - t0

    rows = df.to_dict("records")
    t1 = time.perf_counter()

    async def one(r: dict[str, object], sem: asyncio.Semaphore) -> bool:
        p = str(r.get("prompt", "") or "")
        if not p or p.startswith("ERROR:"):
            return False
        try:
            ic = int(r.get("item_count", 0) or 0)
        except (TypeError, ValueError):
            ic = 0
        text = chat_format(tok, p)
        sp = sampling_for(SamplingParams, ic, args.max_tokens)
        rid = uuid.uuid4().hex
        async with sem:
            final = None
            async for out in engine.generate(text, sp, rid):
                final = out
            return bool(final and final.outputs and final.outputs[0].text)

    async def drive() -> int:
        sem = asyncio.Semaphore(args.in_flight)  # hundreds in flight — the key knob
        tasks = [asyncio.ensure_future(one(r, sem)) for r in rows]
        ok = 0
        for f in asyncio.as_completed(tasks):
            ok += 1 if await f else 0
        return ok

    ok = asyncio.run(drive())
    infer_s = time.perf_counter() - t1
    n = len(rows)
    rate = n / max(infer_s, 1e-6)
    print(
        f"[async] in_flight={args.in_flight} pages={n} ok={ok} setup_s={setup_s:.1f} "
        f"infer_s={infer_s:.1f}  {rate:.1f} pages/s/GPU  "
        f"=> ~{rate * 8:.0f} pages/s/node  => ~{rate * 8 * 0.85:.0f} @85% eff",
        flush=True,
    )
    return rate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Stage 1c output dir")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-pages", type=int, default=4000, help="0 = whole shard")
    p.add_argument("--mode", choices=["offline", "async"], default="offline")
    p.add_argument("--in-flight", type=int, default=512, help="async semaphore size")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/lustre/fsw/portfolios/llmservice/users/vjawa/hf_cache")
    df = load_shard(args.input, args.shard_index, args.max_pages)
    print(f"[proto] mode={args.mode} pages={len(df)}", flush=True)
    (run_offline if args.mode == "offline" else run_async)(args, df)


if __name__ == "__main__":
    main()
