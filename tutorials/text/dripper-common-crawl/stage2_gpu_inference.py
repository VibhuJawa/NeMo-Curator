#!/usr/bin/env python3
"""
stage2_gpu_inference.py — GPU-ONLY vLLM inference.

RUNS ON: batch partition with 8×H100.
ALL work here is GPU inference. Zero CPU preprocessing on this node.

INPUT:  Stage 1c output (url, cluster_id, cluster_role, prompt, simp_html, map_html, html)
OUTPUT: Adds llm_response column → (url, cluster_id, cluster_role, llm_response,
         simp_html, map_html, html, dripper_error)

Stage 2b (CPU) reads this output and runs map_parser_cls to build mapping_json.

DESIGN:
  8 Ray Serve replicas (one vLLM per GPU) with async dispatch.
  Pure inference — no simplification, no prompt building, no postprocessing.
  GPU stays >90% busy → no watchdog kills.
"""
import argparse, json, os, time, asyncio
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

OUTPUT_COLS = [
    "url", "url_host_name", "cluster_id", "cluster_role",
    "llm_response",  # raw vLLM output → fed to map_parser_cls in Stage 2b
    "simp_html",     # passed through for Stage 2b
    "map_html",      # passed through for Stage 2b
    "html",          # passed through for Stage 2b
    "dripper_error",
    "inference_time_s",
]


def run_stage2(args):
    import ray
    from ray import serve

    # ── Start Ray + 8 vLLM replicas ──────────────────────────────────────────
    t_startup_begin = time.perf_counter()
    ray.init(ignore_reinit_error=True,
             runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""}})

    hf_cache = args.hf_cache
    os.environ.update({"HF_HOME": hf_cache, "TRANSFORMERS_CACHE": hf_cache})

    @serve.deployment(num_replicas=args.replicas, ray_actor_options={"num_gpus": 1})
    class VLLMWorker:
        def __init__(self):
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            engine_args = AsyncEngineArgs(
                model=args.model,
                tensor_parallel_size=1,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                disable_log_stats=True,
                trust_remote_code=True,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            from vllm import SamplingParams
            self._SamplingParams = SamplingParams
            self.sampling = SamplingParams(temperature=0.0, max_tokens=2048)
            self._sampling_cache = {}
            # Load the tokenizer directly (transformers) so the chat template is
            # applied without depending on vLLM's version-specific get_tokenizer API.
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            self._supports_enable_thinking = True

        def _sampling_for(self, item_count: int):
            # Dynamic max tokens: the compact model emits ~one short label per item,
            # so cap output at item_count*per_item + padding (min floor), instead of
            # the 2048 default. This is the standalone baseline's trick and is the
            # dominant Stage 2 speedup (decode length, not prefill, is the cost).
            n = max(args.dyn_min_tokens,
                    int(item_count) * args.dyn_tokens_per_item + args.dyn_token_padding)
            n = min(n, args.max_tokens)
            s = self._sampling_cache.get(n)
            if s is None:
                s = self._SamplingParams(temperature=0.0, max_tokens=n)
                self._sampling_cache[n] = s
            return s

        def _chat_format(self, prompt: str) -> str:
            # The standalone Dripper sends the prompt as a chat message
            # (messages=[{"role":"user","content":prompt}]), so the model's chat
            # template (system prompt + turn markers, thinking disabled) is applied.
            # Feeding the raw prompt to engine.generate() bypasses this → degenerate
            # output. Reproduce the chat template here.
            msgs = [{"role": "user", "content": prompt}]
            if self._supports_enable_thinking:
                try:
                    return self._tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    self._supports_enable_thinking = False
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

        async def infer(self, prompt: str, request_id: str, item_count: int = 0) -> str:
            text = self._chat_format(prompt)
            sampling = self._sampling_for(item_count) if item_count else self.sampling
            gen = self.engine.generate(text, sampling, request_id)
            async for out in gen:
                pass
            return out.outputs[0].text if out.outputs else ""

    handle = serve.run(VLLMWorker.bind(), name="stage2_vllm")
    startup_s = time.perf_counter() - t_startup_begin
    print(f"[stage2] {args.replicas} vLLM replicas ready  startup_s={startup_s:.1f}  "
          f"(model load + Ray init)", flush=True)

    # ── Load Stage 1c pre-processed prompts ──────────────────────────────────
    inp = Path(args.input)
    if inp.is_dir():
        import glob as _g
        files = sorted(_g.glob(str(inp / f"shard_{args.shard_index:04d}.parquet")))
        if not files:
            files = sorted(_g.glob(str(inp / "shard_*.parquet")))
        inp = Path(files[0]) if files else inp

    df = pq.ParquetFile(str(inp)).read().to_pandas()
    print(f"[stage2] {len(df):,} pages to infer", flush=True)

    rows = df.to_dict("records")
    t_load = time.perf_counter()  # start of inference (after startup)

    def _result(row, *, llm_response, dripper_error, inference_time_s):
        passthrough = ("url", "url_host_name", "cluster_id", "cluster_role",
                       "simp_html", "map_html", "html")
        return {
            **{k: row.get(k, "") for k in passthrough},
            "llm_response": llm_response,
            "dripper_error": dripper_error,
            "inference_time_s": inference_time_s,
        }

    async def call_one(row, sem):
        prompt = str(row.get("prompt", "") or "")
        if not prompt or prompt.startswith("ERROR:"):
            return _result(row, llm_response="",
                           dripper_error=prompt if prompt.startswith("ERROR:") else "empty_prompt",
                           inference_time_s=0.0)
        t0 = time.perf_counter()
        try:
            rid = f"{str(row.get('url',''))[:32]}_{id(row)}"
            try:
                ic = int(row.get("item_count", 0) or 0)
            except (TypeError, ValueError):
                ic = 0
            async with sem:
                response = await handle.infer.remote(prompt, rid, ic)
            return _result(row, llm_response=response, dripper_error="",
                           inference_time_s=time.perf_counter() - t0)
        except Exception as e:
            return _result(row, llm_response="",
                           dripper_error=f"infer_error:{type(e).__name__}:{str(e)[:100]}",
                           inference_time_s=time.perf_counter() - t0)

    async def run_all():
        # One bounded-concurrency stream (semaphore) keeps ~batch_size requests in
        # flight so vLLM's continuous batcher stays saturated — no per-batch barrier
        # where the slowest of N requests stalls the next batch.
        sem = asyncio.Semaphore(args.batch_size)
        out = []
        futs = [asyncio.ensure_future(call_one(r, sem)) for r in rows]
        done = 0
        for fut in asyncio.as_completed(futs):
            out.append(await fut)
            done += 1
            if done % 512 == 0 or done == len(rows):
                rate = done / max(time.perf_counter() - t_load, 1e-6)
                ok = sum(1 for r in out if r.get("llm_response"))
                print(f"[stage2] {done:>6}/{len(rows)} pages  {rate:.1f} pages/s  ok={ok}",
                      flush=True)
        return out

    results = asyncio.get_event_loop().run_until_complete(run_all())

    serve.shutdown()
    ray.shutdown()

    # ── Write output ──────────────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet"
                      if args.num_shards > 1 else "inference_results.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    inference_s = time.perf_counter() - t_load
    ok = int((result_df["llm_response"].astype(str).str.len() > 0).sum())
    err = int((result_df["dripper_error"].astype(str).str.len() > 2).sum())
    pure_rate = len(result_df) / max(inference_s, 1e-6)
    wall_rate  = len(result_df) / max(inference_s + startup_s, 1e-6)
    print(f"[stage2] DONE: {len(result_df):,} pages  ok={ok}  errors={err}  "
          f"inference_only={pure_rate:.1f} pages/s  wall(incl_startup)={wall_rate:.1f} pages/s  "
          f"inference_s={inference_s:.1f}s  startup_s={startup_s:.1f}s  → {out_path}", flush=True)

    metrics = {
        "stage": "stage2", "shard_index": args.shard_index,
        "total_pages": len(result_df), "successful_pages": ok, "errors": err,
        "elapsed_s": round(inference_s, 2),
        "setup_time_s": round(startup_s, 2),
        "inference_time_s": round(inference_s, 2),
        "pages_per_s_per_node": round(pure_rate, 2),
        "pure_inference_pages_per_s": round(pure_rate, 2),
        "wall_pages_per_s_incl_startup": round(wall_rate, 2),
        "n_gpus": args.replicas,
    }
    (out_path.with_name(f"metrics_stage2_shard_{args.shard_index:04d}.json")
     .write_text(json.dumps(metrics, indent=2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, help="Stage 1c output dir")
    p.add_argument("--output",      required=True, help="Output dir")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards",  type=int, default=1)
    p.add_argument("--replicas",    type=int, default=int(os.environ.get("N_GPU_REPLICAS", "8")))
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--max-tokens",          type=int, default=2048, help="hard cap on output tokens")
    p.add_argument("--dyn-tokens-per-item", type=int, default=6,  help="dynamic max_tokens per _item_id")
    p.add_argument("--dyn-token-padding",   type=int, default=16, help="dynamic max_tokens padding")
    p.add_argument("--dyn-min-tokens",      type=int, default=32, help="dynamic max_tokens floor")
    p.add_argument("--gpu-mem-util",          type=float, default=0.90)
    p.add_argument("--max-model-len",         type=int,   default=32768)
    p.add_argument("--max-num-seqs",          type=int,   default=256)
    p.add_argument("--max-num-batched-tokens",type=int,   default=16384)
    p.add_argument("--model",       default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--hf-cache",    default=os.environ.get("HF_HOME",
                   os.path.expanduser("~/.cache/huggingface")))
    run_stage2(p.parse_args())


if __name__ == "__main__":
    main()
