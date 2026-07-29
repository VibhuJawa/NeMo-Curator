# MinerU-HTML main content extraction

Extract the main content of raw HTML pages using
[MinerU-HTML](https://github.com/opendatalab/MinerU-HTML), a small language model
that labels each element of a simplified DOM as `main` or `other`.

Compared with heuristic extractors (trafilatura, resiliparse, jusText), the model
handles forums, Q&A threads and product pages that break rule-based boilerplate
removal — at the cost of a GPU.

## What the pipeline does

| Stage | Hardware | Work |
| --- | --- | --- |
| `MinerUHtmlSimplifyStage` | CPU | Simplify the DOM, tag every element with `_item_id`, build the prompt, tokenize |
| `MinerUHtmlInferenceStage` | GPU | vLLM batch generation of `1main2other3main…` labels |
| `MinerUHtmlExtractStage` | CPU | Parse labels, prune the DOM to `main` subtrees, render Markdown |

Splitting the work this way is the point: the reference implementation runs all
six of its steps in one process, so the GPU idles through the DOM work. In
Curator the CPU stages scale out independently and keep the GPU saturated.

## Install

```bash
pip install "nemo_curator[vllm]" mineru_html
```

`mineru_html` supplies the DOM simplifier and the Markdown converter; Curator
drives vLLM itself, so the `mineru_html[vllm]` extra (which pins an old vLLM) is
not needed.

## Run

```bash
python run_pipeline.py --input /path/to/html-parquet --output /path/to/out
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
python run_pipeline.py --input /path/to/html-parquet --output /tmp/out --limit 200 --verbose
```

## Tuning

| Flag | Default | Notes |
| --- | --- | --- |
| `--structured-outputs` | `per_request` | The reference implementation's per-document regex. It guarantees every element id appears exactly once and costs only a few percent, so it is the default. `none` is a little faster but lets ~7% of answers contain out-of-range ids (harmless — they match no element). |
| `--max-model-len` | `32768` | **The highest-leverage knob here.** Covers ~99% of Common Crawl documents; longer ones fall back to trafilatura. Dropping to 16k loses 3.4% of documents and cuts GPU work ~3x, because attention is quadratic and the tail dominates — see below. |
| `--max-num-batched-tokens` | `8192` | Prefill budget per engine step. Measured: raising it 16k → 32k changes nothing (inside noise, both with and without quantization). The engine sits at ~14% of peak FLOPs because of GEMM *shape* — hidden size 1024 — not batch size. Re-check on H100, but don't expect much. |
| `--kv-cache-dtype` | `auto` | **`fp8` is the biggest GPU win measured: 1.56×**, and it costs less fidelity than quantizing weights (94.0% identical label sets vs 89.4%). The bf16 KV cache is 96 KiB/token and the token-weighted mean sequence is ~22k tokens, so KV capacity — not arithmetic — caps concurrency. Not the default only because Ada needs FlashInfer's prebuilt kernels first: `pip install --extra-index-url https://flashinfer.ai/whl/cu129/ "flashinfer-jit-cache==0.6.6+cu129"`, version-matched to `flashinfer-python`. No CUDA toolkit required. |
| `--quantization` | none | `fp8` enables W8A8 on Hopper/Ada. **Not recommended.** Stacked on top of FP8 KV it buys a further 4% (1.57× → 1.63×) while dropping label agreement from 94.0% to 89.0%. Use `--kv-cache-dtype fp8` and stop there. |
| `--output-format` | `mm_md` | Markdown with maths and images. Conversion is the most expensive CPU step (~45 ms/document, 41% of it in `pylatexenc` detecting LaTeX) — but that is exactly what preserves formulae, tables and code, and the whole CPU side still fits in under two cores per GPU. Use `none` (emit pruned HTML) only when downstream code consumes HTML, not to save CPU. |
| `--simplify-workers` / `--extract-workers` | auto | Size the CPU stages so they keep the GPU fed. Per core: ~20 ms/document to simplify, ~8 ms to tokenize, ~6 ms to prune, ~45 ms to render Markdown — ~12.5 documents/s/core in total. |
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
behind each default, and the things that turned out **not** to help.
