# MinerU-HTML on Curator: measurements and where to go next

Measured on Common Crawl `CC-MAIN-2025-26` (100k raw HTML pages, 61 KiB/page
mean) with `opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact`, on a **power-capped
NVIDIA L4**. Read the methodology note before trusting any absolute number.

> **Where these knobs live now.** The Curator pipeline is CPU-only and talks to a
> vLLM server over HTTP, so every engine-level result below is a **`vllm serve`
> flag on the host you run the server on**, not a pipeline argument. The
> measurements themselves were taken with an engine driven directly, but they are
> properties of the engine and the model, so they carry over; where a result names
> a Python argument in the original tooling, the corresponding server flag is given.

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

## Server-side engine tuning

Every configuration in this section is something you pass to `vllm serve`.

| Result below | `vllm serve` flag |
| --- | --- |
| FP8 KV cache | `--kv-cache-dtype fp8` |
| FP8 W8A8 weights | `--quantization fp8` |
| prefill batch size | `--max-num-batched-tokens` |
| context length | `--max-model-len` (match the pipeline's `--max-model-len`) |

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
| Curator default (no engine flags) | 136.4 s | 0.1% | 1.03x | — |
| + FP8 W8A8 weights | 130.2 s | 0.2% | 1.08x | 89.4% |
| bf16 KV reference (post `flashinfer-jit-cache`) | 133.7 s | 0.1% | 1.06x | (reference) |
| **+ FP8 KV cache** | **85.4 s** | 2.8% | **1.65x** | **94.0%** |
| + FP8 KV + FP8 weights | 82.0 s | 1.4% | 1.72x | 89.0% |
| + FP8 KV + FP8 weights + 32k batch | 81.8 s | 0.3% | 1.73x | 89.0% |

**Recommended: `--kv-cache-dtype fp8` and nothing else.** It is 1.57x over its
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

On the **server host**, and on Ada only: `--kv-cache-dtype fp8` sends vLLM down
FlashInfer's JIT path and dies with `/usr/local/cuda/bin/nvcc: not found`.
Installing a CUDA toolkit is **not** the fix — both PyPI's and NVIDIA's
`nvidia-cuda-nvcc-cu12` wheels ship `ptxas` and headers but no `nvcc`, and
`flashinfer-cubin` does not cover the sm89 FP8 prefill kernel. The fix is
FlashInfer's prebuilt kernel cache, version-matched to `flashinfer-python`:

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

Throughput per GPU is `min(cores × 12.5, gpu_rate)`, where "per GPU" now means
per GPU *of server capacity* — the cores are on the pipeline nodes, the GPUs are
on the server host, and the two are sized independently:

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

## Why a server instead of an in-process engine

Curator briefly shipped both, so the choice was measured rather than assumed.
On 8x H100 over 10k Common Crawl documents, at identical extraction quality
(0.810):

| Configuration | docs/s e2e | inference stage | inference wall |
| --- | ---: | ---: | ---: |
| in-process engine, fp8 KV, 8 pinned workers | 33.6 | 624.7 s / 8 workers | 78.1 s |
| server, `--data-parallel-size 8`, 16 CPU workers | 62.3 | 1207.6 s / 16 workers | 75.5 s |

**The end-to-end gap is not a throughput win.** The inference work costs the same
either way — 78.1 s against 75.5 s of wall time. What differs is that the
in-process path pays ~78 s of vLLM engine startup *inside* its measured window,
while a persistent server pays it once, outside. Per document the server path is
in fact ~2x slower (120.8 ms vs 62.5 ms) from HTTP and serialization overhead,
and only keeps up because that latency is spread across twice as many workers —
which is affordable precisely because they are CPU-only.

So the reason to run against a server is operational, not speed: startup is
amortized across runs, the engines scale, restart and are shared independently
of the pipeline, and a Curator job needs no GPU allocation at all. That is why
the in-process stage was removed rather than kept as an option — one backend
that is never slower and much easier to operate beats two that must be kept in
sync.

One correctness consequence of the move: an OpenAI server applies the
checkpoint's `generation_config.json` as request defaults, which an in-process
engine building `SamplingParams` directly never sees. Its `repetition_penalty`
of 1.05 attacks the deliberately repetitive answer format and collapses
`extraction_rate` to **0.015 against 0.809**. Start the server with
`--generation-config vllm`; the stage also pins `repetition_penalty` and `top_k`
per request so a forgotten flag cannot silently ruin a corpus.

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

## Pipeline tuning on 8x H100

Everything above this section is single-L4 work. This section is 8x H100 against
a persistent `--data-parallel-size 8` server, on 10k-40k Common Crawl documents.

### First: `docs/s` is meaningless without the corpus size

A run carries **~85 s of fixed, GPU-idle overhead inside the measured window**
(`time_taken_s` starts after Ray cluster setup, so this is *not* cluster startup).
It is ~64 s of ramp — actor pools constructing, each importing transformers/lxml
and loading a tokenizer — plus 23-39 s of tail, the last partitions draining
through the CPU stages after inference has finished.

Fitting `n/R + T = elapsed` on two scales gives T = 84 s from the 10k/20k pair and
T = 85 s from the 20k/40k pair, so it is stable. The consequence is that **the same
configuration reports wildly different throughput depending only on corpus size**:

| documents | overhead as share of window | reported |
| ---: | ---: | ---: |
| 10 000 | 51% | ~60 docs/s |
| 20 000 | 34% | ~87 docs/s |
| 40 000 | 23% | ~110 docs/s |
| 100 000 | 7% | ~125 docs/s |

Steady-state is ~138 docs/s. **Benchmark at 40k documents or more** (100k if you are
tuning the engine rather than the pipeline), never compare `docs/s` across corpus
sizes, and treat differences under ~1.3% as noise. At 10k the overhead compresses a
real 12% win into ~6%, which is why an entire batch of engine flags first measured
"flat".

Which scale you pick also decides *what you can see*. Below 40k the fixed ramp and
tail dominate, so pipeline structure is what moves the number; at 100k they are 7% of
the window and the engine is what moves it. The two largest wins in this document were
each invisible at the other's scale.

### What actually moves throughput

Ranked, all at unchanged extraction quality:

Ranked. Every figure below is a **within-scale, same-server** comparison; chaining
deltas measured at different corpus sizes or against different server flags
inflates them badly, because corpus size alone moves the reported number ~80%.

| Lever | Gain | Measured at | Where |
| --- | ---: | --- | --- |
| Partition size 312 -> 156 rows | **+8.5%** | 40k, w48, block64 | input sharding |
| ... same, replicated at another scale | **+6.7%** | 20k, w32, block64 | input sharding |
| Total CPU actors 64 -> 32 | **+2.0%** | 40k, 156 rows, 32 workers | `simplify_workers` + `extract_workers` |
| Inference workers 16 -> 32 | +2.7% | 20k, 312 rows, block16 | `inference_workers` |
| Inference workers past 32 | **nothing** | 40k, 156 rows, block64 | `inference_workers` |
| `--block-size 64` | +2.6% *at 16 workers, -1.2% at 32* | 20k, 312 rows | `vllm serve` |

**Know the noise floor before believing any of this.** Repeat runs of an identical
config differ by ~1.3% at 20k and ~2.1% at 40k. Only the partition-size result clears
that comfortably.

**Partition size is the single largest knob, and it is not a flag** — it is how many
shards the input is written as, since `FilePartitioningStage` groups whole files and
never splits one. It is also the only result here that replicates cleanly at two
scales: 98.5 -> 106.9 docs/s at 40k and 81.6 -> 87.1 at 20k, both from nothing but
halving rows per partition, with workers, server flags and corpus held fixed.

The mechanism is the tail, and it is visible in the GPU trace: halving rows per
partition cut the drain from 31 s to 23 s and lifted GPU-busy time from 59.9% to
62.7%. It does not touch the ~64 s ramp, which is why the gain is ~8% rather than the
~35% that eliminating all idle time would buy.

**There is an optimum, so do not just keep splitting.** Rows per partition against
throughput at 40k: 312 -> 98.5, 156 -> 107.7, 79 -> 103.9 docs/s. Halving past 156
costs 3.5%, well outside the 0.6% spread of repeat runs at that config. (Partly
confounded: dividing 4 source shards 507 ways leaves a 3.04x row skew against 1.29x
for the 255-shard build, and skew hurts too.) **Target ~156 rows per partition.**

**Use FEWER CPU actors than the node can hold.** This is the counter-intuitive one.
Every Ray Data actor costs ~0.35 s of GPU-idle ramp at startup — it imports
transformers and lxml and constructs an `AutoTokenizer`, and on shared storage those
loads contend. Measured across 25 runs, ramp tracks *total* actor count
(`simplify_workers + extract_workers`) almost linearly:

| total actors | ramp | throughput at 40k |
| ---: | ---: | ---: |
| 24 | 47 s | 109.1 (tail grows: extract starved) |
| **32** | **46-47 s** | **109.8-111.4 docs/s** |
| 48 | 54 s | 109.5 |
| 64 | 59-67 s | 107.4-108.0 |
| 72-80 | 72 s | (20k runs, lost 5-7%) |

**The ramp floors at ~47 s** with actor count alone. Dropping from 32 to 24 actors
saved nothing, so actor count is only the part *above* ~32. Going below 32 also costs:
at 16 extract actors the tail grew 25 s -> 33 s. **32 total actors is the optimum.**

### The ramp is one Python import — set three environment variables

That ~47 s floor is not Ray. `import mineru_html` eagerly pulls in `transformers`,
which pulls `torch` and (via `generation.candidate_generator`) `sklearn`. Measured
standalone in this venv, a single actor's `setup()` costs **25.8 s** for extract and
**28.4 s** for simplify. None of it is needed: inference is a remote HTTP call and no
stage runs a local model. Fitting the ramp across runs gives
`ramp ~= 26 s + 0.6 s x N_actors` — the intercept is one actor's construction cost,
and the slope is contention between actors constructing concurrently.

```bash
export USE_TORCH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`USE_TORCH=0` stops `transformers` importing torch. The two offline flags stop every
simplify actor making HuggingFace hub HEAD requests, since
`AutoTokenizer.from_pretrained` is called without `local_files_only` and the model is
already cached. Measured end to end at 40k, same config otherwise:

| | ramp | busy | tail | docs/s |
| --- | ---: | ---: | ---: | ---: |
| without | 47 s | 297 s | 25 s | 109.8, 111.1 |
| **with** | **37 s** | 295 s | 23 s | **114.4, 114.5** |

Busy and tail are unchanged — the variables cut exactly the ramp and nothing else,
worth **+3.5%** for no code change. Do not set `USE_TORCH=0` in a pipeline that runs
a local torch model; it is safe here only because inference is remote.

The remaining ~37 s is still GPU-idle and still mostly imports. Removing
`import mineru_html` from the extract stage entirely (it needs only a thin
`trafilatura` wrapper, and `mineru_utils.py` already reimplements the rest of that
path) measured 25.8 s -> 1.4 s standalone and should take a further large bite.

The steady-state work is **unchanged** — GPU-busy time is 293-297 s in every one of
these — so the entire difference is ramp. Both stages are wildly over-provisioned
anyway: simplify measures 578 docs/s at 32 actors against a pipeline running at ~110.

**Inference workers do not matter past 32,** and the apparent effect was actor count in
disguise: fitting more workers into 128 cores forces actors down. Held at 32 actors,
32 vs 80 *requested* workers measures 109.8 vs 110.5 — noise. Held at 32 workers,
halving actors 64 -> 32 is worth +2.0%.

> **`inference_workers` is silently capped, so read the above carefully.** The stage
> declares `cpus=2.0` (`mineru_server.py`), and Ray Data charges an actor pool's CPU
> reservation against the cluster total for the whole run whether the actors are
> working or not. Concurrent inference tasks are therefore capped at
> `(128 - simplify - extract) / 2`, so `--inference-workers 80` with 32 actors really
> ran 48, and 48 with 64 actors really ran 32. What is demonstrated above is that
> **32 vs 48 actual workers makes no difference**; values beyond that were never
> tested. Lowering `cpus` for this stage — it is `asyncio` HTTP I/O and burns almost
> no CPU while awaiting — would both uncap it and remove the actor-pool deadlock
> cliff described below.

Queue *depth* is not a lever either: at fixed 16 workers, 384/768/1536 gave
76.3/80.4/77.3 — a peak with losses on both sides.

This also explains an earlier failed experiment. Shifting CPU workers from simplify to
extract (24/48 and 16/64) lost 7.5% and 5.1% — not because extract got worse, but
because total actors rose to 72 and 80 and the ramp went to 72 s.

`--block-size 64` is worth +2.6% only while the client is under-provisioned; with 32
workers it measures -1.2%, inside noise. Both relieve the same constraint, so they do
not stack. Recommended: raise workers, skip the flag.

### Speculative decoding is the one engine win that is large

Everything else in this section moves throughput by a few percent. This moves it by
**28%**. Measured at 100k documents, all on one node against the same control:

| draft tokens | docs/s | vs control | tokens per forward pass | draft accepted | extraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| none (control) | 125.6 | — | 1.00 | — | 0.8075 |
| 3 | 140.1 | +11.5% | 3.68 | 91.0% | 0.8075 |
| 6 | 150.0 | +19.4% | 5.58 | 82.2% | 0.8074 |
| 10 | 157.2 | +25.2% | 7.51 | 73.2% | 0.8075 |
| **16** | **160.5** | **+27.8%** | **9.19** | 64.1% | 0.8076 |

```bash
pip install arctic-inference   # provides the suffix proposer
vllm serve ... --speculative-config \
  '{"method":"suffix","num_speculative_tokens":16,
    "suffix_decoding_max_spec_factor":2.0,"suffix_decoding_max_cached_requests":10000}'
```

Returns flatten past ~10 draft tokens — 16 buys only 2.1% more than 10 — so either is
defensible. Note acceptance *falls* as you draft deeper while throughput still rises:
a rejected draft token costs ~2.4 µs of arithmetic, and each accepted one saves a
~147 µs KV re-read, so the trade stays favourable long after the hit rate stops
looking impressive.

**Why it works here, when speculative decoding usually does not help throughput at
high batch sizes.** This 0.5B model stores **48 KiB of KV per token** — about 75% of
Llama-3.1-8B's footprint with 1/16 the FLOPs — so decode re-reads the whole KV cache
for every token and is memory-bound by ~30x at the mean context. A drafted token costs
~2.4 µs of arithmetic; the KV re-read it avoids costs ~147 µs. Drafting is nearly free.

And the answers are highly predictable: the response is `1main2other3main…`, so the ID
literals are forced by the output format and the labels are a binary choice. Suffix
decoding matches against a tree built from *past* requests, so `["13","other"] -> "14"`
holds across every document.

How predictable, concretely: at 3 draft tokens the *third* position was still accepted
81% of the time (559k / 487k / 456k acceptances at positions 0/1/2 over 561,706 drafts),
which is what suggested drafting deeper would keep paying — and it did, up to ~10.

**It costs nothing in quality.** Every drafted token is verified against the target
model and only exact matches are accepted, so the output is the model's own. Over 100k
documents the status histogram is identical to within one document: ok 97 652 / 97 652 /
97 653 and too_long 1 495 / 1 495 / 1 495 for stock / 3 tokens / 6 tokens, with
`extraction_rate` moving -0.02% and mean characters +0.04% — the same greedy-decoding
batch-composition noise described earlier, not a speculative artefact.

**Check acceptance, not just throughput.** `vllm:spec_decode_num_drafts_total` and
`..._num_accepted_tokens_total` (note the `_total` suffix) are the ground truth — an
inert proposer and a proposer that did not help look identical in docs/s alone.
Accepted-per-draft must clear ~0.6 to pay for async scheduling, which vLLM
auto-disables with suffix speculation and warns about at startup.

The token ratio misleads here. 26.9:1 prefill:decode by *token count* corresponds to
roughly **43:57 prefill:decode by GPU time**, because each decode token drags the whole
KV cache through HBM. Every engine flag we tested before this was aimed at the smaller
half of the time budget, which is why they all measured flat.

### Measured dead ends

| Tried | Result |
| --- | ---: |
| `--no-enable-prefix-caching` | -0.9% |
| `--prefix-caching-hash-algo xxhash` | -1.3% |
| `--api-server-count 16` | -2.4% |
| CPU workers shifted simplify -> extract (24/48) | -7.5% |
| ... (16/64) | -5.1% |
| `--max-num-partial-prefills 4` | won't start: "Concurrent Partial Prefill is not supported" |

Prefix caching is worth keeping despite a **4.16% hit rate** — the corpus is 27:1
prefill-dominated, so 4% of prompt tokens served from cache outweighs the hashing.
Do not "optimise" the hash: it is not on the critical path.

The simplify -> extract rebalance failed because the premise was wrong. Extract's
observed 146 docs/s was its *arrival* rate, not its capacity — it was being fed by
inference at ~136 docs/s. Only simplify's 578 docs/s was a true capacity measurement,
because simplify genuinely runs ahead and drains every partition early. **A stage that
is not saturated tells you nothing about its ceiling.**

### Footgun: Ray Data actors hold their CPU for the whole run

`simplify_workers + extract_workers + inference_workers` must stay well under the core
count. On a 128-core node, 64 simplify + 64 extract actors deadlocks: the actor pools
take all 128 slots and the inference task pool never schedules. The run does not fail,
it hangs. Budget ~112 of 128 and leave headroom for the fused reader and the writer.

### Recommended starting point on 8x H100

```bash
# server, started once, independently of any pipeline run
vllm serve opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact \
    --port 8000 --served-model-name mineru \
    --data-parallel-size 8 --max-model-len 32768 \
    --kv-cache-dtype fp8 --trust-remote-code --generation-config vllm
```

Start the server with speculative decoding (see above) — it is the single largest win
and needs nothing from the pipeline:

```bash
pip install arctic-inference
vllm serve ... --speculative-config \
  '{"method":"suffix","num_speculative_tokens":16,
    "suffix_decoding_max_spec_factor":2.0,"suffix_decoding_max_cached_requests":10000}'
```

On the **pipeline host**, before launching the job:

```bash
export USE_TORCH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Write the input as **~156 rows per shard**, and set `inference_workers=32`,
`server_concurrency=48`, `simplify_workers=8`, `extract_workers=24`.

Two numbers, because the pipeline settings and the server setting were measured at
different scales and it is worth keeping them apart:

| | | |
| --- | ---: | --- |
| the three **pipeline** settings, 40k, no speculation | **114.5 docs/s** | +16% over 98.5 |
| **+ speculative decoding**, 100k | **160.5 docs/s** | +28% over a 125.6 control |

The 40k figure is the tightest replication in this document — two runs at 114.4 and
114.5 — and isolates the pipeline work, none of which is a `vllm serve` flag:

| change | gain (40k) |
| --- | ---: |
| 312 -> 156 rows per partition | +8.5% |
| 64 -> 32 CPU actors | +2.6% |
| `USE_TORCH=0` + HF offline | +3.5% |

The actor-count result is as clean as anything here. Grouping every 40k run by total
actors: four runs at 32 actors average 110.5 docs/s whether `inference_workers` is 32
or 80, and two runs at 64 actors average 107.7. The worker count contributes nothing;
the actor count is the whole effect.

The three pipeline rules are: **partition to ~156 rows**, **run as few CPU actors as
the stages need** (both are ~5x over-provisioned by default), and **keep torch out of
the workers**. Nothing else on the pipeline side moved throughput outside the noise
floor — not queue depth, not inference workers past 32.

**Together with speculative decoding on the server, 98.5 -> 160.5 docs/s.**

The two halves are worth keeping straight, because each is invisible at the other's
scale. At 40k the fixed ramp and tail are ~23% of the window, so pipeline structure
dominates and GPU-busy time is 290-297 s in *every* run regardless of configuration —
which is what makes "the engine is not the constraint" true *there*. At 100k the
overhead is 7% and the engine sets the number.

And when the engine did turn out to matter, it was not where any flag was pointing.
Five `vllm serve` flags aimed at prefill measured flat, because by GPU *time* this
workload is ~43% prefill and ~57% decode despite being 26.9:1 prefill by token count.
The one change that addressed decode was worth more than everything else combined.

## What to do on H100

**Carries over unchanged, or improves:** the CPU/GPU stage split (gain grows
with GPU speed), the `extract_main_html` rewrite, pre-tokenizing on CPU workers,
per-document `max_tokens` (fallbacks 17/300 -> 7/300; and 80 GB lets you raise
`max_model_len` to recover more tail), skipping zero-item documents, the single
chat template.

**Re-measure** — all of these are `vllm serve` flags on the server host:

| Knob | L4 result | Why H100 differs |
| --- | --- | --- |
| `--kv-cache-dtype fp8` | 1.57x | Largely a 24 GB artefact — 80 GB relaxes the KV constraint ~3.3x on its own, so expect it to buy context length more than throughput. Check whether Hopper still needs `flashinfer-jit-cache`. |
| `--quantization fp8` (W8A8) | 1.08x | H100 FP8 tensor cores are ~2x bf16 dense and the card is not power-starved. Still capped at 1.38x by the linear FLOP share. Re-check the 10.6% label churn. |
| CUDA graphs | no change | Per-step CPU overhead is fixed while the GPU gets ~8x faster, so graph capture matters relatively more. |
| `--max-num-batched-tokens` | no change 16k->32k | **Settled: leave it alone.** The default is 8192, so the "8k baseline" was the default and 32k only probed one-shot prefill. `--long-prefill-token-threshold` redistributes the same 8192-token budget, giving identical GEMM shapes — there is no mechanism for either to help. |
| Absolute throughput | ~2-3.4 docs/s | Do **not** scale by core count — the L4 was power-throttled throughout. Measured on 8x H100: ~110 docs/s over 40k documents, but see the scale caveat above. |

Two defaults worth knowing before designing an H100 sweep, both read from
vLLM 0.18.1 source rather than `--help` (which is abbreviated, and crashes on a
driverless login node): `--max-num-seqs` defaults to **1024** on H100, so testing
256 -> 512 tests values *below* the default; and `--api-server-count` defaults to
`--data-parallel-size`, so a DP=8 server already runs 8 front ends behind one
`SO_REUSEPORT` socket rather than the single front end one might assume.

`--data-parallel-size` cannot exceed the GPU count **of one node** — vLLM maps DP
rank to device as `dp_local_rank * TP*PP + tp_rank` and asserts against the visible
device count. Going wider needs `--data-parallel-size-local` plus a `--headless`
second node. For a dense (non-MoE) model vLLM treats DP ranks as fully independent
with no cross-rank collective, so two independent `-dp 8` servers are
throughput-equivalent to one `-dp 16` group and strictly simpler to operate.

**The biggest lever is not implemented.** Attention is quadratic and 1.2% of
documents carry 55% of the FLOPs, so splitting long documents into chunks —
labelling each independently and merging the `{item_id: label}` maps, which is a
plain `dict.update()` since ids are globally unique — models out at **2.2–2.6x
while keeping every document**. It would live in the simplify stage, which is
where the `_item_id` numbering is assigned, and needs no engine support. Capping
`--max-model-len` at 16k (on both the server and the pipeline) is worth 3.1x if
you accept a 3.4% fallback rate.

Note that vLLM's own chunked prefill does **not** do this: it splits a long
prompt across engine steps for scheduling, but every chunk still attends to the
full preceding KV cache, so total attention FLOPs are identical. It must be —
the engine is obliged to reproduce the unchunked output. Cutting attention FLOPs
requires *changing* the output, so no engine can do it for you. That is also why
chunking needs a WebMainBench evaluation before it ships, not just a throughput
number.

## Curator-managed Dynamo vs a standalone `vllm serve`

Everything above assumes an endpoint you start yourself. Curator can instead own
the engines: an `InferenceServer` with a `DynamoVLLMModelConfig` brings up
`num_replicas` vLLM engines as Ray actors inside the same cluster the pipeline
runs on, and the pipeline talks to `server.endpoint`. No separate server to start,
no GPU allocation to coordinate.

Measured on the same 100k corpus, both paths with fp8 KV and suffix speculative
decoding at 16 draft tokens, both through `benchmarking/run.py`, replicated:

| path | docs/s | extraction |
| --- | ---: | ---: |
| Dynamo, `M=32 x B=256` | 169.5, 167.4 (mean **168.4**) | 0.8089 |
| standalone, `M=32 x B=48` | 160.5, 160.2 (mean **160.3**) | 0.8076 |

**+4.9% for Dynamo.** The obvious objection is that the two ran at different queue
depths, so standalone was simply under-fed. It was not -- re-running standalone at
Dynamo's depth, on one node against one server, moves nothing:

| standalone, same node, same server | docs/s |
| --- | ---: |
| `B=48` | 158.2 |
| `B=256` | 158.1 |

A 0.06% difference. Standalone does not benefit from a deeper client queue; Dynamo
needs one. Node-to-node variation is ~1.5% (158.2 here against 160.3 on the node the
original pair ran on), comfortably below the 4.3% gap between Dynamo's *worst* run
and standalone's *best*.

Request-level timing separates the serving layer from the rest of the pipeline.
The CPU stages are identical across all four runs (simplify 37.9-40.2 ms/doc,
extract 65.7-72.5), so the whole difference is in inference:

| path | inference ms/doc | inference-actor idle |
| --- | ---: | ---: |
| Dynamo | 141.4 | 1717s, 3528s |
| standalone | 171.4 | **0s, 0s** |

The idle column is the load-bearing one. Standalone's inference actors were never
idle -- the server was the constraint. Dynamo's idled for 29-59 minutes of actor
time, i.e. its serving layer finished faster than the CPU stages could feed it.
Do not read the ms/doc gap as pure serving efficiency: `process_time` is actor
wall-clock with B requests in flight, so a larger B mechanically compresses
actor-seconds per document, and these two ran at different B.

### The knobs are not interchangeable

Dynamo wants a far deeper client queue: 1024 requests per replica against 192 for
a standalone server. Its frontend adds per-request latency that only more
in-flight work covers. Feeding Dynamo the standalone's depth understates it badly.

| requests/replica | B | no spec | spec=16 |
| ---: | ---: | ---: | ---: |
| 128 | 32 | 105.0 | -- |
| 256 | 64 | 104.5 | -- |
| 512 | 128 | 109.4 | 130.8 |
| 1024 | 256 | 110.0 | **134.4** |

(40k corpus, so not comparable with the 100k figures above -- see the corpus-size
warning earlier in this file.)

### What it costs

* **Startup is ~210-320s against ~125s.** Ray builds each engine actor its own uv
  venv, and nothing installed in the driver venv is visible to the engine. That is
  setup, outside the measured window, but it is paid per run.
* **`structured_outputs` is unavailable.** Dynamo's frontend validates request
  bodies strictly and rejects the extra body `vllm serve` accepts, with
  `400 Validation: Unsupported parameter(s)`. Measured at 40k this costs nothing --
  extraction 0.8043-0.8047 unconstrained against 0.8026-0.8032 constrained -- but
  it is a real capability gap if you need guaranteed well-formed answers.
* **`etcd` and `nats-server` must be on PATH.** They are static Go binaries the
  control plane shells out to, not Python packages.
* **Speculative decoding needs `arctic-inference` in the actor venv**, declared via
  `runtime_env={"uv": {"packages": ["arctic-inference"]}}`. Leave it unpinned:
  vLLM's own error message recommends `==0.1.1`, but every release is sdist-only,
  so it always builds from source, and 0.1.1's build needs `torch==2.7.0` --
  unsatisfiable against the torch vLLM pulls in, which fails the entire
  `runtime_env` setup rather than just that package.
