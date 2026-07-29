# MinerU-HTML on Curator: measurements and where to go next

Measured on Common Crawl `CC-MAIN-2025-26` (100k raw HTML pages, 61 KiB/page
mean) with `opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact`, on a **power-capped
NVIDIA L4**. Read the methodology note before trusting any absolute number.

## Workload shape

Properties of the data and the model, not the hardware — these carry to any GPU.

| Quantity | Mean | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw page (KiB) | 61 | 33 | 126 | 611 | 1384 |
| Prompt (tokens) | 4190 | 2313 | 7937 | 40265 | 175000 |
| `_item_id` elements | 74 | 42 | 155 | 529 | 2172 |
| Answer (tokens) | 149 | 84 | 310 | 1058 | 5517 |

- **Prefill-bound**: 100k documents is ~419M prompt tokens against ~15M
  generated — 28:1. Decode-side tricks barely register.
- **6.4% of documents produce zero labelled elements.** The inference stage
  skips them (0.3% of tokens, but 6.4% of request slots).
- DOM simplification compresses 5.2x (61 KiB page -> 12.7 KiB simplified).

## Where the GPU work actually goes

Per prefill token the model costs `2P` FLOPs in the linear layers (`P` = 415M
non-embedding params) and `2·n_heads·head_dim·L·s` in causal attention. Those
cross over at `415e6 / (16·128·24)` = **8443 tokens**.

The corpus mean is 4190 tokens — below that — but the mean is the wrong
statistic, because attention is paid per token *and* scales with the length of
the document that token belongs to. **Token-weighted, the average document this
workload sees is 22,158 tokens long.**

| | TFLOP per 1000 docs | Share |
| --- | ---: | ---: |
| Linear layers | 3478 | 27.6% |
| Causal attention | 9128 | **72.4%** |

| Documents | % of docs | % of prefill tokens | % of prefill FLOPs |
| --- | ---: | ---: | ---: |
| ≥ 8k tokens | 9.6% | 46.8% | 78.5% |
| ≥ 16k tokens | 3.4% | 29.8% | 67.2% |
| ≥ 32k tokens | 1.2% | 18.2% | **55.3%** |

**1.2% of documents account for 55% of the GPU work.**

### …and the engine is nowhere near its roofline

The 283-document benchmark set is 1388 TFLOP of arithmetic. At 130 s that is
**10.7 TFLOP/s achieved**, against ~77 TFLOPS available at the clocks the L4
actually ran — **~14% of peak**. Hidden size is 1024, so every GEMM has K=1024:
shapes that leave most of a tensor core idle no matter how many tokens you
batch. Confirmed empirically — doubling `max_num_batched_tokens` 16k -> 32k
changed nothing (85.4 s -> 85.7 s). It is a GEMM *shape* limit, not a batching
limit.

This is why weight quantization underperforms: you cannot buy much by making a
multiply cheaper when the multiplier is already 86% idle.

## GPU results

> **Methodology — required reading for small cards.** This L4 has a 72 W power
> cap; under load its SM clock falls 2040 -> ~1100 MHz, so the *same* config
> measures ~25% faster running first in a sequence than fifth. Run-ordered
> comparisons are meaningless. Every number below uses: a throwaway warm-up per
> round, two rounds with the config order reversed in the second, and the mean
> of the two positions. Sub-1% spread between a config's two positions is the
> signal that the control worked. All configs process an **identical document
> set** (`--force-drop-budget 16384`), otherwise the per-document answer budget
> makes the better config accept the expensive long tail and look slower.

| Configuration | mean | spread | speedup | identical `main` sets vs bf16 |
| --- | ---: | ---: | ---: | ---: |
| Reference implementation | 141.1 s | 2.6% | 1.00x | — |
| Curator default | 136.4 s | 0.1% | 1.03x | — |
| + FP8 W8A8 weights | 130.2 s | 0.2% | 1.08x | 89.4% |
| bf16 KV reference (post `flashinfer-jit-cache`) | 133.7 s | 0.1% | 1.06x | (reference) |
| **+ FP8 KV cache** | **85.4 s** | 2.8% | **1.65x** | **94.0%** |
| + FP8 KV + FP8 weights | 82.0 s | 1.4% | 1.72x | 89.0% |
| + FP8 KV + FP8 weights + 32k batch | 81.8 s | 0.3% | 1.73x | 89.0% |

**Recommended: `kv_cache_dtype="fp8"` and nothing else.** It is 1.57x over its
own bf16 reference and perturbs labels least. Adding weight quantization buys a
further 4% while dropping label agreement 94.0% -> 89.0% — a bad trade on a
corpus you intend to train on. Prefill batch size does nothing at either
precision.

The bf16 reference row exists deliberately: installing `flashinfer-jit-cache`
could in principle have switched the attention backend for bf16 too, so bf16 KV
was re-measured afterwards. 133.7 s vs 134.6 s before — unchanged — so the win
is the KV dtype, not the install.

That result is what the roofline predicts. KV capacity, not arithmetic, binds:
96 KiB/token against ~17 GB of usable cache caps the L4 near 55 concurrent
sequences of average length. Halving the KV dtype doubles that, and an engine
at 14% of peak has headroom to use it.

### Enabling FP8 KV cache (no CUDA toolkit needed)

On Ada, `kv_cache_dtype="fp8"` sends vLLM down FlashInfer's JIT path and dies
with `/usr/local/cuda/bin/nvcc: not found`. Installing a CUDA toolkit is **not**
the fix — both PyPI's and NVIDIA's `nvidia-cuda-nvcc-cu12` wheels ship `ptxas`
and headers but no `nvcc`, and `flashinfer-cubin` does not cover the sm89 FP8
prefill kernel. The fix is FlashInfer's prebuilt kernel cache, version-matched
to `flashinfer-python`:

```bash
pip install --extra-index-url https://flashinfer.ai/whl/cu129/ \
    "flashinfer-jit-cache==0.6.6+cu129"
```

## CPU cost, and when it starts to matter

| Step | ms/doc/core |
| --- | ---: |
| `simplify_html` (upstream, unchanged) | 20.5 |
| Tokenize | 7.9 |
| `extract_main_html` — **Curator** (upstream: 14.3) | 6.3 |
| Markdown conversion incl. maths/LaTeX | 45.2 |
| **Total** | **80** → **12.5 docs/s/core** |

Throughput per GPU is `min(cores × 12.5, gpu_rate)`, so:

| Scenario | GPU docs/s | cores/GPU before CPU-bound |
| --- | ---: | ---: |
| L4, bf16 KV (measured) | 2.1 | 0.2 |
| L4, FP8 KV (measured) | 3.4 | 0.3 |
| H100 ≈8x L4, bf16 KV | 16.8 | 1.3 |
| H100 ≈8x L4, FP8 KV | 26.9 | 2.1 |
| H100 + FP8 KV + 2.6x chunking | 69.9 | **5.6** |

CPU is nowhere near binding today. It becomes a provisioning question only once
the GPU wins stack: fully optimized on H100 you would want ~6 cores/GPU, ~45 on
an 8-GPU node.

Markdown conversion is the largest CPU item (41% of it `pylatexenc` detecting
LaTeX), but **keep it** — that pass is what preserves formulae, tables and code,
and the whole CPU side still fits in ~2 cores per GPU. `output_format="none"`
exists for pipelines that consume HTML directly, not to buy throughput.

### The stage split is what makes GPU work pay off

The reference implementation runs all six steps in one process, so ~80 ms of CPU
per document adds to GPU time instead of overlapping it. The penalty grows with
every GPU optimization:

| Scenario | Serial | Pipelined | Gain |
| --- | ---: | ---: | ---: |
| L4, bf16 KV | 1.8 docs/s | 2.1 | 1.17x |
| H100 ≈8x, bf16 KV | 7.2 | 16.8 | 2.34x |
| H100 ≈8x + FP8 KV | 8.5 | 26.9 | 3.15x |
| H100 + FP8 KV + 2.6x chunking | 10.6 | 69.9 | **6.58x** |

At the bottom row the GPU spends 14 ms/document and the serial design spends
80 ms of CPU on top — 85% of wall clock with the GPU idle.

## Two upstream bugs worth knowing

**The reference implementation applies the chat template twice** — once in
`InferenceBackend.process`, again in `VLLMInferenceBackend.generate` — leaving a
stray `<|hy_begin_of_sentence|><|hy_User|>` inside the user turn. Token cost is
trivial (~7/doc) but it changes the answer on ~20% of documents against a ~1%
noise floor. Curator applies it once; `chat_template_mode="upstream_double"`
reproduces the original if you need to match its published numbers.

**`extract_main_html` was O(elements × DOM size)** — the reference resolves each
`main` label with its own `//*[@_item_id="N"]` XPath. Curator indexes in one
pass: **4.5x faster, byte-identical** on 282 real documents and on all 64 label
assignments of a fixture document.

Note that greedy decoding is **not** bit-reproducible across engine configs —
batch composition changes float reduction order and a 0.5B model doing
per-element binary classification has many near-ties. Treat any A/B below ~1% as
noise, and validate quantization on *label agreement*, not output equality.

## Gotcha: reading raw HTML

`ParquetReader` defaults to `dtype_backend="pyarrow"`. An Arrow-backed `binary`
column over 2 GB in one partition — ~25k Common Crawl pages, i.e. one 673 MB
Parquet file — cannot be pickled when handed to the next stage
(`ArrowInvalid: offset overflow while concatenating arrays`). `run_pipeline.py`
reads with `dtype_backend="numpy_nullable"`.

## What to do on H100

**Carries over unchanged, or improves:** the CPU/GPU stage split (gain grows
with GPU speed), the `extract_main_html` rewrite, pre-tokenizing on CPU workers,
per-document `max_tokens` (fallbacks 17/300 -> 7/300; and 80 GB lets you raise
`max_model_len` to recover more tail), skipping zero-item documents, the single
chat template.

**Re-measure:**

| Knob | L4 result | Why H100 differs |
| --- | --- | --- |
| FP8 KV cache | 1.57x | Largely a 24 GB artefact — 80 GB relaxes the KV constraint ~3.3x on its own, so expect it to buy context length more than throughput. Check whether Hopper still needs `flashinfer-jit-cache`. |
| FP8 W8A8 weights | 1.08x | H100 FP8 tensor cores are ~2x bf16 dense and the card is not power-starved. Still capped at 1.38x by the linear FLOP share. Re-check the 10.6% label churn. |
| CUDA graphs | no change | Per-step CPU overhead is fixed while the GPU gets ~8x faster, so graph capture matters relatively more. |
| `max_num_batched_tokens` | no change 16k->32k | Shape-limited here; worth one confirmation on H100. |
| Absolute throughput | ~2-3.4 docs/s | Do **not** scale by core count — the L4 was power-throttled throughout. |

**The biggest lever is not implemented.** Attention is quadratic and 1.2% of
documents carry 55% of the FLOPs, so splitting long documents into chunks —
labelling each independently and merging the `{item_id: label}` maps, which is a
plain `dict.update()` since ids are globally unique — models out at **2.2–2.6x
while keeping every document**. Capping `max_model_len` at 16k is worth 3.1x if
you accept a 3.4% fallback rate.

Note that vLLM's own chunked prefill does **not** do this: it splits a long
prompt across engine steps for scheduling, but every chunk still attends to the
full preceding KV cache, so total attention FLOPs are identical. It must be —
the engine is obliged to reproduce the unchunked output. Cutting attention FLOPs
requires *changing* the output, so no engine can do it for you. That is also why
chunking needs a WebMainBench evaluation before it ships, not just a throughput
number.
