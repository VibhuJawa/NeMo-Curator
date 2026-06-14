# FP8 / Quantization Plan — Stage 2 vLLM Inference (Track H2)

**Model:** `opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact` (HunYuanDenseV1, arch `hunyuan_v1` in vLLM; 24 layers, hidden 1024, 16 attn heads / 8 KV heads GQA, head_dim 128, bf16 weights, tie_word_embeddings).

**Hypothesis under test:** FP8 roughly doubles throughput for this 0.5B model on H100 with negligible F1 loss.

**Verdict (short):** FP8 is *supported and applicable* here (online dynamic W8A8, no pre-quantized checkpoint needed), but the realistic multiplier for THIS workload is **~1.1–1.4×, not ~2×**. The 2× figure applies to large compute-bound models; a 0.5B model on H100 is tiny and the measured bottleneck is the **serving/batching architecture**, not weight FLOPs or weight-memory traffic. FP8 is a *secondary* lever to be stacked on top of the serving fix — it does **not** on its own close the 27→143 p/s/node gap, and is most useful for the aggressive 20% routing case.

---

## 1. Cluster + vLLM support (verified, light inspection)

Verified live on `nb-hel-cs-001-login-01` via the venv at
`/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv`:

- **vLLM `0.18.1`**, **torch `2.10.0+cu129`**, CUDA build 12.9. Target GPUs are H100 = **sm_90**, which has **native FP8 (E4M3) tensor-core support**. (Confirmed device-capability call returns None on the login node only because login has no GPU; H100 sm_90 FP8 is well established.)
- vLLM ships the FP8 quantization method: `vllm/model_executor/layers/quantization/fp8.py`.
  - `class Fp8Config`: `ACTIVATION_SCHEMES = ["static", "dynamic"]`, default `activation_scheme="dynamic"`, and it explicitly supports `is_checkpoint_fp8_serialized=False` — i.e. **online quantization of a bf16 checkpoint at load time** (the comment in the file: "supports loading quantized FP16/BF16 model checkpoints with dynamic ... activation scale"). No pre-quantized weights required.
  - KV-cache FP8 path present: `kv_cache.py` (`BaseKVCacheMethod`), enabling `kv_cache_dtype="fp8"`.
- **Architecture supports quantization:** `vllm/model_executor/models/hunyuan_v1.py` threads `quant_config: QuantizationConfig | None` through every linear layer (q/k/v/o proj at lines 121/128/195/203/219, gate/up/down MLP at 300/308/324, etc.). So passing `quantization="fp8"` will FP8-quantize the attention + MLP GEMMs. (The router/embedding/lm_head stay higher precision — standard, and lm_head is tied here.)

**Conclusion:** `quantization="fp8"` + optional `kv_cache_dtype="fp8"` is a one-line engine-arg change, requires no offline conversion, and is compatible with this model and this vLLM build.

---

## 2. Why ~2× does NOT hold for this workload (the honest estimate)

The 2× rule-of-thumb for FP8 applies to **large, compute-bound models** where matmul FLOPs dominate. Two facts break that here:

**(a) The model is 0.5B — already FLOP-light and far from compute-bound.**
~1 GFLOP/token. Prefill at any realistic batch is nowhere near H100's bf16 tensor-core roofline. FP8 doubles *peak* matmul throughput, but if you're at, say, 20–30% of the bf16 roofline, doubling the roofline buys little. Prefill is already NOT the wall (STAGE2_GPU_PERF_PLAN §2 confirms: even 87K tok/s/GPU is comfortably within bf16 capacity).

**(b) The measured bottleneck is serving/batching, not generation or weight FLOPs.**
Per the project state: dynamic max_tokens gave **no** gain (temp=0 model already EOS-stops in tens of tokens), and the standalone got ~2.3× purely from a better serving/batching architecture (max-concurrent-requests dispatch in nemo_curator's `LLMServer` vs our per-request `handle.infer.remote`). When the GPU is idle waiting on the python dispatch loop, making each GEMM faster with FP8 changes nothing — you're not GEMM-bound, you're **dispatch/occupancy-bound**.

**Where FP8 *does* help this workload, quantified:**

- **Decode (memory-bandwidth-bound):** per decoded token you read all weights once. bf16 weights ≈ 0.5B × 2B = **~1.0 GB**; FP8 ≈ **~0.5 GB**. H100 HBM3 ≈ 3.35 TB/s. At batch B, decode step time floor ≈ weight_bytes / BW (weights read once per step regardless of B) + KV reads (scale with B). Halving weight bytes lowers the weight-traffic component of the per-step floor, which **only matters at small batch** (low B → weight traffic dominates). At the large batches we *want* (max_num_seqs 256+), KV-cache and activation traffic dominate and the weight saving is diluted. Net decode speedup: **~1.1–1.3×**, larger only if batches stay small.
- **fp8 KV cache:** halves KV bytes → **~2× more KV slots** for the same `gpu_memory_utilization`. For a 0.5B model the KV cache is already tiny relative to 80 GB, so this rarely unblocks batch (we're seq-count / dispatch limited, not KV-limited). Marginal here; main value is the 20% case at very high concurrency. **~1.0–1.1×**, with an F1-parity risk (see §4).
- **Prefill (compute):** FP8 GEMM ~2× peak, but we're well below roofline → realized **~1.05–1.2×**.

**Stacked, realistic FP8 multiplier on a *well-tuned bf16 baseline*: ~1.1–1.4×.** Use **1.2×** as the planning point estimate; **1.4×** is optimistic-but-plausible if the serving fix pushes us into a more GEMM/decode-bound regime (which itself would mean FP8 helps more).

---

## 3. Throughput projection — does FP8 + serving fix reach ~143 p/s/node?

Baselines: current custom serving = **27 p/s/node**; standalone (better serving, same model) = **~62 p/s/node** (project state) / 45 p/s/node (STAGE2 doc, conservative). The serving fix is the dominant lever and is FP8-independent.

| Scenario | bf16 p/s/node | × FP8 (1.2) | × FP8 (1.4) |
|---|---|---|---|
| Today (custom serving) | 27 | 32 | 38 |
| Serving fix → standalone-class (62) | 62 | 74 | 87 |
| Serving fix + concurrency/CUDA-graph tuning (est. 80–100) | 90 | 108 | 126 |

**Against the 143 p/s/node target (14% LLM coverage, 16 nodes, 2 days, 0.85 eff):**

- FP8 **alone** (32–38 p/s/node): **does not** reach 143. Not even close. Rules out FP8 as a standalone fix.
- Serving fix to standalone-class **+ FP8**: 74–87 p/s/node — **still short of 143** (~1.6–1.9× gap remains).
- Serving fix + full concurrency/CUDA-graph tuning to ~90 **+ FP8 1.2–1.4×**: **108–126 p/s/node** — **approaches but likely still misses 143** by ~12–25%.

**So FP8 contributes meaningfully but is not sufficient.** To hit 143/node you need: (1) the serving/batching rewrite (biggest lever, must land first), (2) full concurrency + CUDA-graph + gpu_mem_util tuning, (3) FP8 as the final ~1.2–1.4× multiplier, and very likely (4) reduce LLM coverage below 14% (Stage-3.5 routing efficiency) or add a couple of nodes. FP8 is best understood as the lever that converts a ~108–126 result into a comfortable cushion *if* coverage drops to ~11–12%, where the required rate falls accordingly (e.g. 12% coverage → ~123 p/s/node target, which 108–126 *does* span).

---

## 4. F1-parity risk and cheap validation

**Risk level: LOW for weight-only FP8; LOW–MEDIUM for fp8 KV cache.**

- **W8A8 dynamic weight FP8** (`quantization="fp8"`, dynamic per-tensor/per-token activation scales): for greedy/temp=0 decoding, FP8 weight error is small; the main failure mode is a *small fraction* of pages where a near-tie label flips (main vs other), changing the extracted span. Because reps/singletons sit at the 0.97 nondeterminism ceiling, even a tiny perturbation reads as noise — the metric to watch is the **per-bucket token-F1 delta**, not exact-match.
- **fp8 KV cache** is the higher-risk knob: it quantizes attention K/V and can degrade long-context recall — relevant because some MinerU prompts are thousands of input tokens and a few near the 32768 cap. This is exactly where label recall on trailing `_item_id`s could drop. **Recommend testing it separately** and only adopting if its incremental F1 delta is ~0.

**Cheap validation protocol (no heavy/long job; respects the GPU-contention constraint):**
1. Take a **small fixed sample** (e.g. 2,000–5,000 pages) of Stage-1c outputs that already have ground-truth/baseline labels (reuse the same set `compare_f1.py` already scores).
2. Run Stage 2 **twice on one GPU** (single replica, short job): (a) bf16 baseline, (b) `quantization="fp8"`. Then optionally (c) `quantization="fp8", kv_cache_dtype="fp8"`.
3. Score all three with `compare_f1.py` against the standalone baseline (job 335168). Report **overall + per-bucket token-F1** (rep / singleton / sibling) and the **fp8−bf16 delta**.
4. **Accept FP8 weights if overall delta ≥ −0.005** (within nondeterminism noise). **Accept fp8 KV cache only if its additional delta ≥ −0.003**, else ship weight-FP8 only.
5. Also log the per-page `prompt_tokens` histogram during the FP8 run to confirm no new truncation interaction.

This is a single-GPU, few-thousand-page job (minutes), safe to run alongside the existing validation chain on a spare GPU or queued briefly.

---

## 5. Exact config changes (Stage 2 engine — spec only; do NOT edit production script)

In `stage2_gpu_inference.py`, the `AsyncEngineArgs` (currently lines 53–64) becomes:

```python
engine_args = AsyncEngineArgs(
    model=args.model,
    tensor_parallel_size=1,
    gpu_memory_utilization=args.gpu_mem_util,   # 0.90 recommended
    max_model_len=args.max_model_len,           # keep 32768 (do NOT lower for speed)
    max_num_seqs=args.max_num_seqs,             # 256+ (serving fix; FP8-independent)
    max_num_batched_tokens=args.max_num_batched_tokens,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    disable_log_stats=True,
    trust_remote_code=True,
    # --- FP8 additions ---
    quantization="fp8",                 # online dynamic W8A8; no pre-quantized weights needed
    # kv_cache_dtype="fp8",             # OPTIONAL, gate behind the §4 KV-cache F1 check
)
```

Add CLI flags so it's A/B-testable without code edits:
```python
p.add_argument("--quantization", default=None, choices=[None, "fp8"])
p.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
# then: quantization=args.quantization, kv_cache_dtype=args.kv_cache_dtype
```

Notes:
- `activation_scheme` defaults to `"dynamic"` in `Fp8Config` — correct for an online (non-serialized) checkpoint; do not set `"static"` (it requires a serialized fp8 checkpoint and would raise).
- No tokenizer/sampling/chat-template changes. The `enable_thinking=False` correctness fix and temp=0 sampling are unchanged.
- Sequence to validate independently: **(A) bf16 baseline → (B) +fp8 weights → (C) +fp8 KV** — adopt the largest prefix that holds F1 parity per §4.

---

## 6. Summary

- FP8 is **supported, applicable, and one engine-arg away** for this model on this vLLM/H100 stack (online dynamic W8A8; optional fp8 KV cache).
- The ~2× hypothesis is **not** borne out for a 0.5B model whose bottleneck is serving/batching, not weight FLOPs. Honest estimate: **~1.2× (plan), up to ~1.4× (optimistic)**.
- FP8 **alone reaches only ~32–38 p/s/node** — far from 143. It is a **stacking multiplier**: serving fix (→~90) × FP8 (1.2–1.4) → **~108–126 p/s/node**, which **approaches but likely misses 143** unless LLM coverage drops to ~11–12% or 1–2 nodes are added.
- F1 risk is **low for weight FP8, low–medium for fp8 KV cache**; validate cheaply with a 2–5K-page single-GPU A/B against `compare_f1.py`, accepting only deltas within nondeterminism noise.
