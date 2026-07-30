# MinerU-HTML main content extraction

Turn raw web pages into clean Markdown, keeping the article and dropping the
navigation, sidebars, cookie banners and footers.

[MinerU-HTML](https://github.com/opendatalab/MinerU-HTML) does this with a small
language model (0.5B) instead of hand-written rules, so it adapts to page
layouts that heuristic extractors get wrong. This tutorial runs it as a Curator
pipeline in which **no stage needs a GPU** — the model lives in a `vllm serve`
you start once, and the pipeline talks to it over HTTP.

## How it works

The model never sees your raw HTML, and it never writes any text. It plays a
labelling game on a simplified copy of the page.

**1. Simplify.** The page is stripped to its structure and every block of
content gets a numbered tag, `_item_id`. Boilerplate like the `<nav>` links is
already gone at this point:

```html
<html><head><title>Sourdough</title></head><body>
<nav><a href="/">Home</a><a href="/recipes">Recipes</a></nav>
<article><h1 _item_id="1">How to bake sourdough</h1>
<p _item_id="2">Mix flour, water and starter until no dry flour remains.</p>
<p _item_id="3">Rest the dough 30 minutes, then fold it four times.</p></article>
<footer _item_id="4">Copyright 2026 Example Corp</footer></body></html>
```

**2. Label.** The model reads that and answers with one `main` or `other` per
numbered element, in order, and nothing else:

```
<answer>1main2main3main4other</answer>
```

Read it as: *the heading and both paragraphs are the article; the copyright line
is not.* That is the model's entire job — four elements, four labels. A typical
Common Crawl page has ~74.

**3. Prune and render.** Elements labelled `other` are deleted, and what remains
is converted to Markdown:

```markdown
# How to bake sourdough

Mix flour, water and starter until no dry flour remains.

Rest the dough 30 minutes, then fold it four times.
```

Two things follow from this design and explain most of the options below. The
answer is tiny and rigidly formatted, so the pipeline constrains it with a regex
and sizes each request's token budget individually. And a page that fails at any
step — unparseable HTML, too long for the context window, a lost request — falls
back to [trafilatura](https://trafilatura.readthedocs.io/) rather than being
dropped, with the reason recorded in a `_mineru_status` column.

## Install

```bash
pip install 'nemo_curator[mineru_html]'
```

That extra pulls in `mineru-html` (the DOM simplifier used by step 1) and
`mineru-webkit`, whose `webpage_converter` renders the Markdown in step 3.
You do **not** need upstream's `mineru_html[vllm]` extra, which pins an old vLLM,
nor `nemo_curator[vllm]` — the HTTP client is `openai`, already a core
dependency. vLLM is needed only on the machine that *hosts* the model.

One system package: the Markdown converter imports `cairosvg` on its first
conversion, which dlopens `libcairo.so.2`. Install it with your package manager
(`apt install libcairo2`). Only `--output-format none` avoids that path.

## Start the server

Start it once; it outlives any number of pipeline runs.

```bash
pip install arctic-inference   # enables speculative decoding, see below

vllm serve opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact \
    --port 8000 --served-model-name mineru \
    --data-parallel-size <N_GPUS> --max-model-len 32768 \
    --kv-cache-dtype fp8 --trust-remote-code --generation-config vllm \
    --speculative-config '{"method":"suffix","num_speculative_tokens":16,
      "suffix_decoding_max_spec_factor":2.0,"suffix_decoding_max_cached_requests":10000}'
```

Every flag here earns its place:

- **`--generation-config vllm`** — the checkpoint ships sampling defaults
  (`temperature 0.7`, `top_k 20`, `repetition_penalty 1.05`) that an OpenAI
  server would otherwise apply to every request. `repetition_penalty` is
  actively harmful: the answer is *supposed* to repeat (`1main2other3main…`), so
  penalising repeats attacks exactly the tokens the model must emit. Leaving it
  on measured **extraction rate 0.015 against 0.809**. The pipeline also pins
  these per request, so a forgotten flag cannot silently ruin a corpus — but set
  it anyway.
- **`--speculative-config …`** — worth **+28% throughput at identical output**,
  the largest single win available. The answer format is so predictable that the
  model can guess several tokens ahead and verify them in one pass. Drop this
  flag if you skip `arctic-inference`; everything else still works.
- **`--kv-cache-dtype fp8`** — halves KV cache memory, which this model uses a
  lot of. On Ada GPUs it needs FlashInfer's prebuilt kernels first
  (`pip install --extra-index-url https://flashinfer.ai/whl/cu129/ "flashinfer-jit-cache==0.6.6+cu129"`);
  drop the flag if the server won't start.
- **`--data-parallel-size`** — one model replica per GPU. Use data parallel, not
  tensor parallel: a 0.5B model fits on one GPU many times over, and this is a
  throughput problem, not a latency one. It cannot exceed the GPU count *of one
  node*; to go wider, run one server per node and point different pipeline jobs
  at each.
- **`--max-model-len 32768`** — must match the pipeline's `--max-model-len`.
  The pipeline uses it to route documents that won't fit to the fallback instead
  of sending a request the server would reject.

## Run

```bash
python run_pipeline.py \
    --input /path/to/html-parquet \
    --output /path/to/out \
    --server-url http://127.0.0.1:8000
```

Input is Parquet with a `content` column of raw HTML (`bytes` or `str`) and a
`url` column. Output is the same rows plus `text` (the Markdown) and
`_mineru_status` (`ok`, or why the row fell back).

Try it on a few hundred documents first:

```bash
python run_pipeline.py --input /path/to/html-parquet --output ./out \
    --server-url http://127.0.0.1:8000 --limit 200
```

The defaults are the configuration that measured fastest on an 8×H100 server,
scaled to your machine's core count, so there is normally nothing to tune. Two
things are worth knowing if you are running at scale:

- **Shard your input to roughly 150 rows per file.** Curator never splits a
  file across workers, so file count sets how finely the work divides. This was
  worth ~8% and is the one input-side choice that matters.
- **Raw HTML is read with `dtype_backend="numpy_nullable"`.** Keep this. Arrow's
  default 32-bit offsets overflow once a partition holds more than 2 GB of HTML
  — about 25k Common Crawl pages — and the run dies with
  `offset overflow while concatenating arrays`.

## Options

| Flag | Default | What it's for |
| --- | --- | --- |
| `--server-url` | required | Endpoint root, with or without `/v1`. There is no in-process fallback. |
| `--max-model-len` | `32768` | Must match the server. Documents whose prompt plus answer won't fit go to the fallback instead. Lowering it makes runs cheaper and sends more documents to the fallback: attention is quadratic, so the longest ~1% of pages dominate GPU cost. |
| `--fallback` | `trafilatura` | What to do with documents the model could not label. `bypass` keeps the original HTML, `empty` writes an empty string. Only `empty` lets the pipeline drop the raw HTML column after simplification. |
| `--output-format` | `mm_md` | Markdown including maths and images. `none` emits pruned HTML for downstream code that wants HTML — not a way to save CPU. |
| `--structured-outputs` | `per_request` | Constrains the answer to one label per element via a regex. Keep it: without it a few percent of answers reference elements that don't exist. |
| `--simplify-workers` / `--inference-workers` / `--extract-workers` | auto | CPU worker counts, sized from your core count by default. Raise `--inference-workers` if the GPUs are idle. |
| `--server-concurrency` | `48` | In-flight requests per inference worker. Server queue depth is this × `--inference-workers`. |
| `--chat-template-mode` | `single` | The reference implementation applies the chat template twice, which changes the answer on ~20% of documents. Use `upstream_double` only to reproduce its published numbers. |

## Going further

Performance measurements — a 35-run sweep on 8×H100, what helped, and the many
things that didn't — are in
[benchmarking/mineru-html-BENCHMARKS.md](../../../benchmarking/mineru-html-BENCHMARKS.md),
alongside the benchmark config used to produce them.

## Papers

**The method.** [MinerU-HTML / Dripper](https://arxiv.org/abs/2511.23119) —
the extractor this tutorial runs. It reports ROUGE-N F1 of 0.84 on
[WebMainBench](https://github.com/opendatalab/WebMainBench) (7,887 annotated
pages) against 0.64 for Trafilatura, which is the gap that justifies spending a
GPU on extraction at all — and also why Trafilatura is a sensible *fallback*
rather than the primary path.

**The baseline, and the fallback.**
[Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction](https://aclanthology.org/2021.acl-demo.15/)
(Barbaresi, ACL 2021) — the rule-based extractor every document falls back to
when the model cannot label it.

**Why extraction quality is worth this trouble.**
[The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
(Penedo et al., 2023) — properly filtered web data alone can match models
trained on curated corpora. Boilerplate that survives extraction is boilerplate
you train on.

**How the server works.**
[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
(Kwon et al., SOSP 2023) — the paper behind vLLM. Worth reading if you want to
understand why `--kv-cache-dtype fp8` and the KV cache matter so much for a
model this small: its per-token KV footprint is large relative to its compute.

**Why speculative decoding pays here.**
[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
(Leviathan et al., 2023) introduces the draft-then-verify idea and, importantly,
proves the output distribution is unchanged — which is why `--speculative-config`
costs nothing in quality.
[SuffixDecoding](https://arxiv.org/abs/2411.04975) (Oliaro et al., 2024) is the
specific variant used here: it needs no draft model, instead matching against a
suffix tree of previously generated tokens. That is a good fit for this workload
because the answers are highly repetitive across documents (`1main2other3main…`),
so the tree predicts them well.
