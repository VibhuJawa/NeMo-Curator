# Stage 2 Serving Architecture (Track H1)

**Question:** Is the 27 vs ~62 pages/s/node gap the *serving architecture* (custom Ray-Serve `handle.infer.remote` per request), not the model? **Yes.**

## 1. Root cause — what the current Stage 2 does vs the standalone baseline

**Current Stage 2 (`stage2_gpu_inference.py`):** 8 `VLLMWorker` Serve replicas (1/GPU, each wraps an `AsyncLLMEngine`). The driver loop calls, per page:
```python
async with sem:                                   # sem = Semaphore(batch_size=256)
    response = await handle.infer.remote(prompt, rid, ic)   # Ray ACTOR METHOD RPC
```
Every page is a **Ray actor-method RPC**. Each call pays: cloudpickle-serialize `(prompt, rid, ic)` and the result string, a hop through the Ray object store / actor inbox queue, and one async actor task scheduled by the core worker. Prompts here are thousands of chars; serializing them both ways per request, plus the queue hop, costs on the order of milliseconds *per request*. That overhead, multiplied across the request stream, **caps how many requests are actually in flight at the vLLM scheduler**, so vLLM's continuous batcher runs a starved batch. The 0.5B model is FLOP-light (~1 GFLOP/token); the H100 sits idle waiting on the RPC pipe, not on compute.

**Standalone baseline (`nemo_curator.core.serve` + tutorial `main.py`):** deploys vLLM through `ray.serve.llm.build_openai_app` (`ray_serve/backend.py:96`) — the production OpenAI ingress with its own router and continuous batcher — and drives it with an `AsyncOpenAIClient` (httpx) at `max_concurrent_requests` (`stage.py:454`, `Semaphore`). vLLM receives a saturated request stream over a tuned ingress, so its batcher stays full. Same model, same `dynamic_max_tokens`, same `gpu_memory_utilization=0.9`, same prefix caching — **the only material difference is the request path**. That is the gap.

Confirmation that generation length is NOT the cause: the project already measured that dynamic `max_tokens` gives no gain because temp=0 already stops at EOS in tens of tokens. So the wall is purely **how fast rows reach a full vLLM batch**.

## 2. The insight: Stage 2 is a BATCH job, not a service

Stage 2 reads a parquet shard and writes a parquet shard. There is no external client, no need for a long-lived shared server, no need for a cross-GPU router. A serving framework (Ray Serve deployment handle, or even the OpenAI HTTP ingress) only adds an IPC/RPC layer between data that is *already in the same process tree* as the GPU and the engine that consumes it. For a one-shot shard job the correct architecture is **offline batched inference**: one `vllm.LLM` engine per GPU, in the same process as its shard, fed the whole prompt list in one `LLM.generate(prompts, samplings)` call. vLLM then does continuous batching internally with **zero per-request IPC**.

## 3. Recommended design (ONE)

**Offline batched, data-parallel, 1 engine per GPU. No Ray Serve, no actor RPC, no HTTP.**

- Launch 8 processes per node (one per GPU; pin `CUDA_VISIBLE_DEVICES`). Use Ray *only* to place these 8 tasks across GPUs (or just `srun`/`torchrun`-style 8-way launch). No central router, no deployment handle.
- Inside each process: `LLM(**engine_kwargs)`, then a single `llm.generate(prompts, samplings)` over that GPU's whole assigned prompt list. Write results to the shard parquet.
- Engine kwargs (mirror the proven standalone, `main.py:1626`): `tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=32768, max_num_seqs=512, max_num_batched_tokens=16384, enable_chunked_prefill=True, enable_prefix_caching=True, enforce_eager=False, trust_remote_code=True`.
- Sampling: keep dynamic `max_tokens = min(2048, max(32, item_count*6+16))` (F1-safe, already in place).
- Keep the `enable_thinking=False` chat template (the correctness fix) — apply it once to all prompts before `generate`.

Prototype: `stage2_serving_proto.py` (`--mode offline`, runnable on 1 GPU; `--mode async` benchmarks Candidate B for comparison).

**Why offline over Candidate B (AsyncLLM + Semaphore) or C (OpenAI ingress):**
- B is in-process too and removes the Ray RPC; at high `in_flight` (~512) it should match offline. But offline `LLM.generate` is simpler (no event loop, no per-request task objects, no semaphore tuning) and lets vLLM see the *whole* workload up front for optimal scheduling. Keep B as the fallback if you need streaming/early-exit.
- C (the standalone's `build_openai_app` + HTTP) is proven but still pays an HTTP round-trip + router hop per request — strictly more overhead than A for a shard job. Only justified for a shared multi-client server, which Stage 2 is not.

## 4. Expected throughput (arithmetic)

Removing the actor-RPC bottleneck recovers at least the standalone's measured rate. Two anchors exist in the docs: the plan doc cites **45 pages/s/node** (job 335168), the project brief cites **~62**. Offline batched eliminates *even the HTTP/router overhead the standalone still pays*, so the floor is the higher of these.

- **Floor (match standalone HTTP path):** 45–62 pages/s/GPU-aggregate → **45–62 pages/s/node**. That alone is **1.7–2.3x** over today's 27.
- **Offline, fully saturated:** prefill is the only real work. At ~3,000 input tok/page and an H100 sustaining conservatively ~150K prefill tok/s/GPU for a 0.5B model: 150,000 / 3,000 = **~50 pages/s/GPU = ~400 pages/s/node** compute-bound ceiling. Decode adds tens of tokens/page (negligible vs prefill). Realistic sustained, accounting for scheduler/KV limits and prompt-size variance: **~80–140 pages/s/node**.
- Arithmetic on the prefill side confirms compute is not the wall: 512 seqs * (tens of decode tokens) is trivial; the batched prefill of 16384 tokens/step at ~150K tok/s clears the 211M-page (8.8%) workload's required 19K–28K input tok/s/GPU (plan §2) with large margin.

**Conservative engineering estimate: 27 → ~80–120 pages/s/node (3–4.4x).**

## 5. Reaching the targets

| Target | Per-node need | This design (offline batched) |
|---|---|---|
| 8.8% LLM coverage, 16 nodes, 2 days | ~76 floor / ~90 @85% eff | **MET.** ~80–120 p/s/node clears 76; clears ~90 once the batch saturates (no FP8 needed). |
| 14% coverage (project's projected F1~0.91 routing) | 336M / 172800 / 16 = **122 floor; ~143 @85%** | **TIGHT/marginal at bf16.** Offline batched lands ~80–120; needs the top of the range + good shard balance, or +25% headroom from FP8 weights / fp8 KV cache, or 18–20 nodes. |
| 20% coverage | ~174 floor / ~204 @85% | **NOT met by serving change alone** — requires FP8 (verify F1 parity) and/or scale-out. |

The serving-architecture fix alone gets the **8.8% target comfortably** and gets the **14% target into reach** (combine with FP8 or a few more nodes). It does NOT by itself hit 20%. It is independent of and additive to the F1 work (Stage 3.5 LLM fallback) — F1 is unaffected because generation config (chat template + dynamic max_tokens, temp=0) is unchanged.

## 6. Validation steps (light, single-GPU; respects the no-heavy-GPU-jobs constraint)
1. Run `stage2_serving_proto.py --mode offline --max-pages 4000` on **one** free GPU → record pages/s/GPU; x8x0.85 = projected per-node.
2. Run `--mode async --in-flight 512` on the same shard → confirm it matches offline (validates that the win is removing the Ray RPC, not anything else).
3. Compare both against the current Stage 2's 27/node (= ~3.4 pages/s/GPU). Expected: offline/async ≥ 6–15 pages/s/GPU.
4. If offline ≈ async ≈ 6+ /GPU while current handle.infer ≈ 3.4 /GPU, the actor-RPC diagnosis is confirmed and the recommendation stands.
