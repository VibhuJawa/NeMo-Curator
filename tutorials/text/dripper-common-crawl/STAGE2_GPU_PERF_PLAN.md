# Stage 2 (GPU vLLM Inference) Performance Plan

**Goal:** Complete GPU inference for full CC-MAIN (2.4B pages) in **2 days on 16 nodes (8×H100 each = 128 GPUs)**, running the LLM only on cluster representatives + singletons.

**Model:** `opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact`
HunYuanDenseV1, 24 layers, hidden 1024, 16 attn heads, 8 KV heads (GQA), head_dim 128, bf16, vocab 120818, tie_word_embeddings, `max_position_embeddings=262144`. A genuine ~0.5B dense model — tiny relative to an H100.

**Measured current state:** Stage 2 = **27 pages/s/node** (8×H100, corrected chat-template fix).
**Standalone baseline (job 335168, same model):** **45 pages/s/node** (44,117 pages / 987 s / 8 GPUs) with `--dynamic-max-tokens` (per-item cap), `--max-concurrent-requests 64`, `gpu-mem-util 0.9`, prefix caching, thinking disabled.

---

## 1. Target math (pages/s/node)

Window = 2 days = **172,800 s**. Nodes = 16.

| LLM fraction | LLM pages | Required agg rate | Per-node @ 100% | **Per-node @ 85% eff** |
|---|---|---|---|---|
| **8.8%** of 2.4B | **211 M** | 1,221 p/s | 76.4 | **89.9 p/s/node** |
| **20%** of 2.4B | **480 M** | 2,778 p/s | 173.6 | **204.2 p/s/node** |

**Verification of the spoiler (76 / 174):** those numbers are the **raw** requirement with NO efficiency derating (211M / 172800 / 16 = 76.4; 480M / 172800 / 16 = 173.6). The "~85% efficiency" must therefore be applied as **headroom on top of the spoiler**, i.e. the *real* sustained per-node throughput you must hit to absorb 15% lost time (startup, stragglers, I/O, shard skew) is:

- 8.8% case: **~90 pages/s/node** sustained (spoiler 76 is the zero-overhead floor).
- 20% case: **~204 pages/s/node** sustained (spoiler 174 is the floor).

I use the 85%-derated targets (**90 / 204**) as the engineering targets below; meeting the raw 76/174 is necessary but not sufficient.

**Gap from today (27 p/s/node):**
- To 90 (8.8%): **3.3×**. To 76 floor: 2.8×.
- To 204 (20%): **7.6×**. To 174 floor: 6.4×.

From the standalone 45 p/s/node: **2.0×** (to 90) and **4.5×** (to 204).

---

## 2. Decode vs prefill profile — where the time goes

This workload is **prefill-heavy with a short decode tail**:

- **Input:** simplified-HTML prompt = thousands of input tokens (estimate ~2,000–4,000 tok/page; varies with page size, capped by `max_model_len=32768`).
- **Output:** the compact model emits **one short label per `_item_id`** (e.g. `1main`, `2other`). For typical pages with tens of `_item_id`s, the true output is **tens of tokens**, not thousands.

**The current bottleneck is decode length, not prefill.** With fixed `max_tokens=2048` and greedy decoding, vLLM keeps each sequence in the decode loop until it emits EOS or hits 2048. If the model fails to emit a clean stop on some pages (degenerate repetition, no EOS), those requests run to 2048 steps. Even when EOS fires early, the scheduler reserves KV slots for up to 2048 tokens, shrinking the effective batch. Decode is memory-bandwidth-bound and **serialized per token**, so over-long decode dominates wall time.

**Prefill feasibility check** (after decode is fixed) — required *input* token throughput:

| prompt size | @90 p/s/node | @204 p/s/node |
|---|---|---|
| 2,000 tok | 19K tok/s/GPU | 44K tok/s/GPU |
| 3,000 tok | 28K tok/s/GPU | 65K tok/s/GPU |
| 4,000 tok | 38K tok/s/GPU | 87K tok/s/GPU |

A 0.5B model on an H100 sustains **hundreds of thousands of prefill tokens/s/GPU** (it is FLOP-light; ~1 GFLOP/token). Even the worst cell (87K tok/s/GPU) is comfortably within H100 prefill capacity. **Prefill is NOT the wall** for either target — the levers are (a) stop wasting decode steps, and (b) keep the batch full so the GPU isn't idle between the python-side batches.

**Prefix caching gives ~zero benefit here:** different pages → different prompts → no shared prefix beyond the (short) system/template prelude. Keep it enabled (cheap, caches the shared template prefix) but do not count on it.

---

## 3. Optimization levers (prioritized)

Effort: S = config-only, M = needs a column/plumbing change, L = larger work.
F1 risk: whether it can change extraction quality.

| # | Lever | What it does | Expected p/s/node after | Effort | F1 risk |
|---|---|---|---|---|---|
| **1** | **Dynamic max_tokens** | Cap `max_tokens = min(2048, item_count*6 + 16)`, floor 32 | **~50–70** (gets us to ≈ standalone+; this is THE win) | M | **None** (output is bounded by design; only truncates pathological runaway) |
| **2** | **Add hard stop tokens** | Stop on EOS + structural stop string so no request runs to the cap | folds into #1; removes runaway tail | S | None |
| **3** | **Replace python 256-batch loop with continuous batching** | Stream all rows into vLLM via a bounded semaphore (≈256–512 in flight) instead of `asyncio.gather` over fixed 256-row blocks | +15–30% (kills inter-batch GPU idle / tail effect) | M | None |
| **4** | **Tune `max_num_seqs` / `max_num_batched_tokens`** | Raise concurrency so the 0.5B model saturates the H100 | +20–40% on top | S | None |
| **5** | **`enforce_eager=False` (CUDA graphs)** + bump `gpu_memory_utilization` 0.85→0.90 | More KV cache → bigger batch; graphs cut per-step launch overhead for short decode | +10–20% | S | None |
| **6** | **FP8 weights (optional, 20% case)** | W8A8 / fp8 KV cache → larger batch, faster decode | +15–30% | L | Low–Med (verify F1 parity) |
| 7 | Multi-instance per GPU | N/A — 0.5B leaves memory, but a single replica with large `max_num_seqs` already saturates; data-parallel 1/GPU stays | — | — | — |

### Lever 1 detail — dynamic max tokens (highest value)
The standalone proved this: identical model + identical vLLM settings, the **only** generation difference vs our config is `--dynamic-max-tokens --dynamic-max-tokens-per-item 6 --dynamic-min-max-tokens 32 --dynamic-max-token-padding 16`, and it ran at **45 vs our 27** (1.67×). The reference implementation is already in `stage.py`:

```python
# _generation_config_for_item_count (stage.py:678-687, mirrored 909-918)
dynamic_max_tokens = max(
    self.dynamic_min_max_tokens,                                   # 32
    item_count * self.dynamic_max_tokens_per_item                  # 6 per item
        + self.dynamic_max_token_padding,                          # + 16
)
return replace(base, max_tokens=min(base.max_tokens, dynamic_max_tokens))
```

`item_count = len(set(_ITEM_ID_RE.findall(simpled_html or map_html)))` (`_count_item_ids`, stage.py:673-676).

**Multiplier estimate.** Effective decode work per page scales with the realized output length. Today we budget 2048; the model truly needs `~item_count*6+16`. For a page with, say, 40 items → cap = 256 tokens (8× tighter budget than 2048); a page with 100 items → 616 tokens (3.3× tighter). Because greedy decode usually emits EOS well before the cap, the *primary* gain is (a) eliminating runaway-to-2048 sequences and (b) shrinking the KV reservation so more sequences fit per batch. The empirically observed effect (standalone) is **~1.7×**. Combined with proper continuous batching and concurrency tuning (levers 3–5) the realistic landing is **2.0–2.8× over 27 → ~55–75 p/s/node.**

**Plumbing:** `item_count` must be available per request in Stage 2.
- **Recommended:** Stage 1c emits an `item_count` column (it already produces `simp_html`/`map_html`; add `item_count = len(set(_ITEM_ID_RE.findall(simp_html or map_html)))`). Stage 2 then sets `max_tokens` per request with zero CPU cost on the GPU node.
- **Fallback:** compute the same regex count in Stage 2 from `simp_html` (already passed through) — cheap, but adds a tiny CPU step on the GPU node.

---

## 4. Recommended configuration

### Stage 1c (`stage1c_cpu_preprocess.py`) — emit item_count
Add to `OUTPUT_COLS` and `_preprocess_one`:
```python
import re
_ITEM_ID_RE = re.compile(r'_item_id="(\d+)"')   # match the regex used by stage.py _count_item_ids
# after simplify:
item_count = len(set(_ITEM_ID_RE.findall(simp_html or map_html or "")))
out["item_count"] = item_count
```
(Confirm the exact `_ITEM_ID_RE` pattern by importing `_ITEM_ID_RE` from `nemo_curator/.../dripper/stage.py` rather than re-deriving it.)

### Stage 2 (`stage2_gpu_inference.py`) — engine + sampling (you are editing this; spec only)
**AsyncEngineArgs:**
```python
AsyncEngineArgs(
    model=args.model,
    tensor_parallel_size=1,                 # data-parallel: 1 replica/GPU (keep)
    gpu_memory_utilization=0.90,            # 0.85 -> 0.90 (bigger KV cache)
    max_model_len=32768,                    # keep (see truncation note §5)
    enable_prefix_caching=True,             # keep (caches shared template prefix; cheap)
    enable_chunked_prefill=True,            # smooth long prompts into decode batches
    max_num_seqs=256,                       # raise concurrency (0.5B under-utilizes default)
    max_num_batched_tokens=16384,           # large; lets long prefills + many decodes co-batch
    enforce_eager=False,                    # CUDA graphs on for short-decode speed
    disable_log_stats=True,
    trust_remote_code=True,
)
```
**Per-request SamplingParams (dynamic):**
```python
def _sampling_for(item_count: int) -> SamplingParams:
    cap = max(32, item_count * 6 + 16) if item_count and item_count > 0 else 2048
    return SamplingParams(
        temperature=0.0,
        max_tokens=min(2048, cap),
        # add stop tokens matching the compact format so decode halts promptly:
        # stop=[...] / stop_token_ids=[<eos for this template>]
    )
```
**Dispatch:** replace the fixed 256-row `asyncio.gather` blocks with a single bounded-concurrency pump (one `asyncio.Semaphore(N)` with N≈256–384) feeding all rows continuously, so vLLM's continuous batcher — not the python loop boundaries — controls batching. Keep `enable_thinking=False` chat template (the correctness fix) unchanged.

### Knob alignment with the standalone (mirror these exactly, they are proven)
- `max-concurrent-requests 64` was the *standalone* per-process semaphore. With 8 in-process replicas and continuous batching, set the in-flight cap per replica to ~256 and let `max_num_seqs` bound the GPU; the 64 figure is a client-side throttle, not a GPU limit. Tune up from 64 → 256 and watch GPU util.
- `gpu-memory-utilization 0.9` and dynamic-max-tokens: adopt as-is.

---

## 5. Truncation risk (cross-concern, flag only)
- Prompts are capped at `max_model_len=32768`. Long HTML pages whose simplified prompt exceeds 32768 input tokens are **silently truncated** by vLLM, dropping trailing `_item_id`s → those items can never be labeled "main" → **potential F1/recall loss on very large pages.** This is independent of the throughput work but worth measuring: log `prompt_tokens` and count pages at/above the cap. If a non-trivial fraction truncates, raise `max_model_len` (the model supports 262144 positions) at the cost of KV memory, or chunk large pages. Do NOT lower `max_model_len` for speed — it would trade F1 for throughput.
- Dynamic-max-tokens does **not** truncate legitimate output: the cap (`item_count*6+16`) is sized to the number of labels the model must emit, with 6 tokens of slack per item. Only genuinely runaway generations are cut, which is the desired behavior.

---

## 6. Feasibility verdict

**8.8% case (target ~90 p/s/node, floor 76): FEASIBLE.**
Dynamic max tokens alone reaches the standalone's 45; adding continuous batching + concurrency/`gpu_mem_util`/CUDA-graph tuning (levers 1–5, all config/plumbing, no F1 risk) is conservatively **2.0–2.8× over 27 → 55–75 p/s/node**, and realistically clears 76–90 once the GPU is kept saturated (the 0.5B model has large untapped headroom on H100). **Minimal changes:** lever 1 (dynamic max_tokens + item_count column) + lever 3 (continuous-batching dispatch). These two should land ≥76; add levers 4–5 for the 85%-efficiency cushion to ~90.

**20% case (target ~204 p/s/node, floor 174): FEASIBLE BUT TIGHT — needs the full stack + likely FP8.**
This is ~4.5× over the standalone 45 and 7.6× over current 27. Levers 1–5 plausibly reach ~80–120 p/s/node. Closing to ~174–204 likely requires **lever 6 (FP8 weights + fp8 KV cache)** for a larger batch and faster decode, and/or **scaling out** (more nodes or a longer window). Recommended hedge: validate levers 1–5 first, measure actual sustained p/s/node and prompt-token distribution, then decide between FP8 (verify F1 parity) vs. allocating ~20–24 nodes instead of 16 for the 20% routing experiment. At 16 nodes / 2 days, 20% is achievable only with FP8 landing its expected 1.2–1.3× on top of a well-tuned bf16 baseline.

---

## 7. Action checklist (minimal path)
1. **Stage 1c:** add `item_count` column (import `_ITEM_ID_RE` from `dripper/stage.py`). [M, no F1 risk]
2. **Stage 2:** per-request dynamic `max_tokens = min(2048, max(32, item_count*6+16))` + stop tokens. [M]
3. **Stage 2:** continuous-batching dispatch (single bounded semaphore, ~256 in flight) instead of 256-row gather blocks. [M]
4. **Stage 2 engine:** `gpu_memory_utilization=0.90`, `max_num_seqs=256`, `max_num_batched_tokens=16384`, `enable_chunked_prefill=True`, `enforce_eager=False`. [S]
5. **Measure:** sustained p/s/node, prompt-token histogram, % at `max_model_len` cap. [S]
6. **If 20% routing is adopted and step 5 < 174:** evaluate FP8 (F1 parity check) or scale to 20–24 nodes. [L]
```
```
