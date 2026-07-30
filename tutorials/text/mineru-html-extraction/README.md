# MinerU-HTML main content extraction

Extract the main content of raw HTML pages using
[MinerU-HTML](https://github.com/opendatalab/MinerU-HTML), a small language model
that labels each element of a simplified DOM as `main` or `other`.

Compared with heuristic extractors (trafilatura, resiliparse, jusText), the model
handles forums, Q&A threads and product pages that break rule-based boilerplate
removal — at the cost of a GPU somewhere in the system.

## What the pipeline does

| Stage | Hardware | Work |
| --- | --- | --- |
| `MinerUHtmlSimplifyStage` | CPU | Simplify the DOM, tag every element with `_item_id`, build the prompt, tokenize |
| `MinerUHtmlServerInferenceStage` | CPU | POST the prompts to an OpenAI-compatible vLLM endpoint and collect `1main2other3main…` labels |
| `MinerUHtmlExtractStage` | CPU | Parse labels, prune the DOM to `main` subtrees, render Markdown |

Two things are being split here. First the CPU/GPU boundary: the reference
implementation runs all six of its steps in one process, so the GPU idles through
the DOM work, whereas in Curator the CPU stages scale out independently and keep
the model fed. Second the *process* boundary — the model runs in a vLLM server
you start and own, not inside a pipeline actor. **No stage in this pipeline
requests a GPU.** That means engine startup (~78 s) is paid once instead of on
every run, engines scale and restart independently of the pipeline, and the
Curator job needs no GPU allocation. Measured against an in-process engine on
8x H100, the two are at parity on inference wall time; the win here is
operational.

## Install

The pipeline itself never imports vLLM. On the machine that runs the pipeline:

```bash
pip install nemo_curator mineru_html
```

`mineru_html` supplies the DOM simplifier and the Markdown converter. The
`mineru_html[vllm]` extra (which pins an old vLLM) is not needed, and neither is
`nemo_curator[vllm]` — the HTTP client is `openai`, already a core Curator
dependency.

vLLM is needed only on the machine that **hosts** the model.

## Start the server

```bash
vllm serve opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact \
    --port 8000 --served-model-name mineru \
    --data-parallel-size <N_GPUS> --max-model-len 32768 \
    --kv-cache-dtype fp8 --trust-remote-code --generation-config vllm
```

- **`--generation-config vllm` is load-bearing.** The checkpoint ships a
  `generation_config.json` with `temperature 0.7`, `top_k 20` and
  `repetition_penalty 1.05`, and an OpenAI server applies those as request
  defaults. `repetition_penalty` is the damaging one: the answer format is
  deliberately repetitive (`1main2other3main…`), so penalising repeats attacks
  exactly the tokens the model must emit, and the longer the document the worse
  it gets. Measured with the checkpoint's defaults left in place:
  `extraction_rate` **0.015 vs 0.809**. The stage also pins
  `repetition_penalty=1.0` and `top_k=-1` per request, so correctness does not
  depend on remembering this flag — but set it anyway.
- **`--data-parallel-size` cannot exceed the GPU count.** vLLM maps one replica
  per physical GPU; asking for 16 on 8 GPUs fails with
  `DP adjusted local rank N out of bounds`. Data parallel, not tensor parallel,
  is what you want for a 0.5B model: the weights fit on one GPU many times over,
  and the workload is throughput-bound, not latency-bound.
- **`--kv-cache-dtype fp8`** is the largest single engine win measured (1.57x),
  and perturbs labels less than quantizing weights (94.0% vs 89.4% identical
  `main` sets). On Ada it needs FlashInfer's prebuilt kernels first:
  `pip install --extra-index-url https://flashinfer.ai/whl/cu129/ "flashinfer-jit-cache==0.6.6+cu129"`,
  version-matched to `flashinfer-python`. Drop the flag if the server fails to
  start. See [BENCHMARKS.md](BENCHMARKS.md) for the rest of the server-side
  tuning.
- `--max-model-len` must match the pipeline's `--max-model-len`: the simplify
  stage uses it to route documents that will not fit to the fallback instead of
  sending a request that the server would reject.

## Run

```bash
python run_pipeline.py \
    --input /path/to/html-parquet \
    --output /path/to/out \
    --server-url http://127.0.0.1:8000
```

The input must be Parquet with a `content` column of raw HTML (`bytes` or `str`)
and a `url` column. The output is the same rows plus a `text` column.

> `run_pipeline.py` reads with `dtype_backend="numpy_nullable"` on purpose.
> `ParquetReader` defaults to `"pyarrow"`, and an Arrow-backed `binary` column
> holding more than 2 GB of HTML — about 25k Common Crawl pages, i.e. one
> partition — overflows its 32-bit offsets when the batch is pickled for the
> next stage (`offset overflow while concatenating arrays`). Keep this override
> for any raw-HTML input.

Quick smoke run over a few hundred documents:

```bash
python run_pipeline.py --input /path/to/html-parquet --output ./out \
    --server-url http://127.0.0.1:8000 --limit 200
```

## Tuning

Engine knobs (`--kv-cache-dtype`, `--quantization`, `--gpu-memory-utilization`,
`--max-num-batched-tokens`) are now arguments to `vllm serve`, not to this
pipeline. What is left here is prompt shape and how hard the CPU stages push.

| Flag | Default | Notes |
| --- | --- | --- |
| `--server-url` | required | Root of the endpoint, with or without a trailing `/v1`. There is no in-process fallback. |
| `--server-concurrency` | `64` | In-flight requests per inference worker. Server queue depth is this × `--inference-workers`; that product is what keeps the engines saturated. |
| `--inference-workers` | auto | CPU workers that do nothing but hold HTTP requests open, so the useful count is set by the server's capacity, not the node's. Unset, the backend autoscales from one worker, which under-feeds the server for the first part of a short run — pin it for benchmarks. 16 workers × 48 concurrency fed a DP=8 H100 server. |
| `--structured-outputs` | `per_request` | The reference implementation's per-document regex. It guarantees every element id appears exactly once and costs only a few percent, so it is the default. `none` is a little faster but lets ~7% of answers contain out-of-range ids (harmless — they match no element). |
| `--max-model-len` | `32768` | **The highest-leverage knob on GPU cost.** Covers ~99% of Common Crawl documents; longer ones fall back to trafilatura without being sent. Dropping to 16k loses 3.4% of documents and cuts GPU work ~3x, because attention is quadratic and the tail dominates — see below. Must match the server. |
| `--output-format` | `mm_md` | Markdown with maths and images. Conversion is the most expensive CPU step (~45 ms/document, 41% of it in `pylatexenc` detecting LaTeX) — but that is exactly what preserves formulae, tables and code, and the whole CPU side still fits in under two cores per GPU of server capacity. Use `none` (emit pruned HTML) only when downstream code consumes HTML, not to save CPU. |
| `--simplify-workers` / `--extract-workers` | auto | Size the CPU stages so they keep the server fed. Per core: ~20 ms/document to simplify, ~8 ms to tokenize, ~6 ms to prune, ~45 ms to render Markdown — ~12.5 documents/s/core in total. |
| `--fallback` | `trafilatura` | Applied to documents that fail to parse or exceed the context window. `empty` or `bypass` let the pipeline drop the raw HTML column after simplification, roughly halving inter-stage bytes. |
| `--chat-template-mode` | `single` | The reference implementation applies the chat template twice, leaving a stray `<\|hy_begin_of_sentence\|><\|hy_User\|>` inside the user turn. Curator applies it once by default. The two prompts disagree on the `main` set for ~20% of documents (against a ~1% run-to-run noise floor), so use `upstream_double` if you need to reproduce the reference implementation's published numbers. |

## Cost profile

Measured on Common Crawl `CC-MAIN-2025-26` (100k documents, 61 KiB/page mean):

- Simplified DOM is ~5x smaller than the source page: 12.7 KiB, 4190 prompt tokens on average.
- Answers are short: ~74 elements, ~150 output tokens.
- Over 100k documents that is **419M prefill tokens against 15M decode tokens** —
  the run is prefill-bound, so prefill batching and weight precision matter far
  more than decode-side tricks.
- The long tail is real: p99 is 40k prompt tokens and the maximum seen was 175k.
  1.2% of documents do not fit a 32k context.
- CPU is ~80 ms/document/core end to end (~12.5 documents/s/core), including
  full Markdown conversion with maths and LaTeX.

### Where the GPU time goes

Causal attention overtakes this model's linear layers at ~8.4k tokens of
context, and while the *document*-weighted mean prompt is 4190 tokens, the
*token*-weighted mean is 22,158 — so attention is **72% of prefill FLOPs**, and

> **1.2% of documents (those over 32k tokens) account for 55% of all GPU work.**

And the engine only reaches ~14% of the L4's available FLOPs, because a 0.5B
model with hidden size 1024 leaves most of a tensor core idle no matter how you
batch it — so this is not compute-bound either.

That ranks the levers. Weight quantization tops out at 1.38x on arithmetic
alone, and returns far less than that in practice for the same reason. Splitting
long documents into chunks would be worth 2.2–2.6x while keeping every document;
capping `--max-model-len` at 16k is worth 3.1x if you accept a 3.4% fallback
rate.
See [BENCHMARKS.md](BENCHMARKS.md) for the full derivation, the measurements
behind each server flag, and the things that turned out **not** to help.
