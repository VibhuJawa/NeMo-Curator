# MinerU-HTML main content extraction

Turn raw web pages into clean Markdown, keeping the article and dropping the
navigation, sidebars, cookie banners and footers.

[MinerU-HTML](https://github.com/opendatalab/MinerU-HTML) does this with a small
language model (0.5B) instead of hand-written rules, so it adapts to page
layouts that heuristic extractors get wrong. This tutorial runs it as a Curator
pipeline in which **no stage needs a GPU** — the model lives in a `vllm serve`
you start once, and the pipeline talks to it over HTTP.

## How it works

The model never sees your raw HTML, and it never writes a word of the output.
It plays a labelling game: it is shown a stripped-down page where each block of
content is numbered, and it answers with one word per number.

Everything below is the real output of each stage for this toy page.

### 1. The input

```html
<!DOCTYPE html><html><head><title>Sourdough</title></head><body>
<nav><a href="/">Home</a><a href="/recipes">Recipes</a></nav>
<article><h1>How to bake sourdough</h1>
<p>Mix flour, water and starter until no dry flour remains.</p>
<p>Rest the dough 30 minutes, then fold it four times.</p></article>
<footer>Copyright 2026 Example Corp</footer></body></html>
```

### 2. Simplify

Scripts, styles and attributes go; long text is truncated; obvious chrome like
the `<nav>` block is dropped outright. What survives gets a numbered
`_item_id`. This is what the model will be asked about:

```html
<html><head><meta charset="utf-8"></head><body>
<h1 _item_id="1">How to bake sourdough</h1>
<p _item_id="2">Mix flour, water and starter until no dry flour remains.</p>
<p _item_id="3">Rest the dough 30 minutes, then fold it four times.</p>
<footer _item_id="4">Copyright 2026 Example Corp</footer></body></html>
```

Note the `<nav>` is already gone, but the `<footer>` is not — the simplifier only
removes what is unambiguous, and leaves the judgement calls to the model. That is
the whole division of labour.

### 3. The prompt

The simplified page is pasted into a fixed instruction. This is verbatim what
gets tokenized and sent, minus the chat-template markers:

```text
As an HTML expert, classify elements with "_item_id" as "main" or "other",
keeping only the main content and removing nav, metadata, etc.
Guidelines:
"Main": Includes primary content like article text, images in the article,
original posts in forums, Q&A questions, and answers.
"Other": Includes navigation, metadata, ads, user info, and related content
(e.g., sidebars, timestamps, suggested articles).
Output Format:
Return each _item_id and its corresponding category in the following format:
{_item_id}{category}{_item_id}{category}...
Here, {_item_id} may be 1, 2, 3..., and {category} is either "main" or "other",
as shown in the example below:
"1main2other3other4main"
Input HTML:
<html><head><meta charset="utf-8"></head><body><h1 _item_id="1">How to bake …
```

That is 283 tokens for this toy page; a real Common Crawl page averages ~4,900.
The pipeline tokenizes this on the CPU workers and sends token ids, so the
server never re-tokenizes.

### 4. The answer

The model replies with the labels and nothing else:

```text
<answer>1main2main3main4other</answer>
```

Read it as four `{id}{label}` pairs:

| id | element | label | meaning |
| --- | --- | --- | --- |
| 1 | `<h1>How to bake sourdough` | `main` | keep — article heading |
| 2 | `<p>Mix flour, water…` | `main` | keep — article body |
| 3 | `<p>Rest the dough…` | `main` | keep — article body |
| 4 | `<footer>Copyright 2026…` | `other` | drop — boilerplate |

That is the model's entire job. Four elements here; a typical page has ~74, and
the answer is ~150 tokens. Because the format is this rigid, the pipeline
constrains generation with a regex built from the element count, so the model
*cannot* skip an id, invent one, or start writing prose.

### 5. Prune and render

Elements labelled `other` are deleted from the annotated page, and what remains
is converted to Markdown:

```markdown
# How to bake sourdough

Mix flour, water and starter until no dry flour remains.

Rest the dough 30 minutes, then fold it four times.
```

The footer is gone; the heading kept its level and the paragraphs their breaks.

**When it doesn't work.** A page that fails at any step — unparseable HTML, too
long for the context window, a request lost to the server — falls back to
[trafilatura](https://trafilatura.readthedocs.io/) instead of being dropped, and
the reason is recorded per row in a `_mineru_status` column (`ok`, `too_long`,
`empty_input`, `inference_error`, …). Check that column before trusting a run:
a document that fell back is still a document, but it was extracted by rules
rather than by the model.

### A harder case: equations

Prose is the easy case. Here is the same walkthrough on a page full of maths:
section 3.4 of [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v4),
which sets up a Poisson problem and a PINN loss. The input is that section left
inside the real arXiv page — announcement banner, header, licence line, footer,
funder logos and all. That is ~43 KB of HTML, 36 numbered elements, a
4,752-token prompt.

arXiv's HTML carries every formula twice: as MathML for the browser, and as the
author's original LaTeX in an `alttext` attribute.

```html
<p class="ltx_p">We consider a Poisson equation with zero Dirichlet boundary data.
For <math class="ltx_Math" alttext="\Omega=[-1,1]^{2}" display="inline"><semantics>
  <mrow><mi mathvariant="normal">Ω</mi><mo>=</mo>…</mrow>
  <annotation encoding="application/x-tex">\Omega=[-1,1]^{2}</annotation>
</semantics></math>, consider the PDE</p>
<table id="S3.E2" class="ltx_equationgroup ltx_eqn_table">…
  <math class="ltx_Math" alttext="\displaystyle u_{xx}+u_{yy}" …>…</math>…
</table>
```

The model labels items 1–10 `other` (a bug-report modal, the announcement
banner, arXiv's header and licence line), 11–23 `main`, and 24–36 `other` (the
footer, the HTML-feedback instructions, the funder logos). Rendering what
survives as `mm_md` gives this — elided at `…` and cut short of the closing
paragraph, otherwise verbatim:

```markdown
# KAN: Kolmogorov–Arnold Networks

## 3 KANs are accurate

### 3.4 Solving partial differential equations

We consider a Poisson equation with zero Dirichlet boundary data. For $\Omega=[-1,1]^{2}$ , consider the PDE

<table><tbody><tr><td>$\displaystyle u_{xx}+u_{yy}$</td><td>$\displaystyle=f\quad\text{in}\,\,\Omega\,,$</td><td rowspan="2">(3.2)</td></tr><tr><td>$\displaystyle u$</td><td>$\displaystyle=0\quad\text{on}\,\,\partial\Omega\,.$</td></tr></tbody></table>

We consider the data $f=-\pi^{2}(1+4y^{2})\sin(\pi x)\sin(\pi y^{2})+2\pi\sin(\pi x)\cos(\pi y^{2})$ for which $u=\sin(\pi x)\sin(\pi y^{2})$ is the true solution. We use the framework of physics-informed neural networks (PINNs)[38, 39] to solve this PDE, with the loss function given by

$$\text{loss}_{\text{pde}}=\alpha\text{loss}_{i}+\text{loss}_{b}\coloneqq\alpha\frac{1}{n_{i}}\sum_{i=1}^{n_{i}}|u_{xx}(z_{i})+u_{yy}(z_{i})-f(z_{i})|^{2}+\frac{1}{n_{b}}\sum_{i=1}^{n_{b}}u^{2}\,,$$

where we use $\text{loss}_{i}$ to denote the interior loss, … $\alpha$ is the hyperparameter balancing the effect of the two terms.
```

The same page through the trafilatura fallback, trimmed at the same point:

```markdown
# KAN: Kolmogorov–Arnold Networks

## 3 KANs are accurate

### 3.4 Solving partial differential equations

We consider a Poisson equation with zero Dirichlet boundary data. For , consider the PDE

(3.2)

We consider the data for which is the true solution. We use the framework of physics-informed neural networks (PINNs) [38, 39] to solve this PDE, with the loss function given by

where we use to denote the interior loss, … is the hyperparameter balancing the effect of the two terms.
```

Both extractors found the same article and dropped the same boilerplate — on
this page the fallback's *selection* is fine, and the difference is entirely in
what reaches the renderer. The page has 23 `<math>` elements; the pruned page
still has 23 and trafilatura's has none, so the equation *number* `(3.2)`
survives and no equation does, leaving sentences like "We consider the data for
which is the true solution" that read as complete but say nothing. Note what
this is and isn't: the win is not the model's labelling but the fact that
MinerU-HTML *deletes* nodes from the original DOM, so the MathML and its
`alttext` survive for `mm_md` to turn into LaTeX, whereas trafilatura rebuilds a clean tree
with no place to put a formula — its own Markdown mode drops them too, so this
is not an artefact of how the pipeline calls it.

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
