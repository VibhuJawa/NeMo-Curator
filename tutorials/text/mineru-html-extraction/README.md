# MinerU-HTML main-content extraction

MinerU-HTML labels the numbered elements of a simplified DOM as `main` or
`other`, prunes boilerplate, and renders the surviving original DOM as Markdown.
Keeping the original DOM is important for maths and structured content; see the
[equation example](EQUATIONS.md).

Curator separates the workflow at the CPU/GPU boundary:

1. CPU workers simplify HTML, build the prompt, and tokenize it.
2. CPU HTTP workers submit token IDs to an OpenAI-compatible server.
3. CPU workers prune the DOM and render `mm_md`, Markdown, JSON, text, or HTML.

No pipeline stage owns a GPU. Use an external vLLM endpoint for local work, or
`InferenceServer`/Dynamo through the production benchmark.

## Install

```bash
pip install 'nemo_curator[mineru_html]'
```

The Markdown converter requires `libcairo.so.2`. The end-to-end managed serving
environment is available as `nemo_curator[mineru_html_inference]`.

## External server

```bash
vllm serve opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact \
  --port 8000 \
  --served-model-name mineru \
  --data-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --trust-remote-code \
  --generation-config vllm \
  --speculative-config '{"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_spec_factor":2.0,"suffix_decoding_max_cached_requests":10000}'
```

`--generation-config vllm` is required: the checkpoint's sampling defaults
penalize the deliberately repetitive `1main2other3main…` answer. Data parallel
is preferable to tensor parallel for this 0.5B throughput workload.

## Run

```bash
python run_pipeline.py \
  --input /path/to/html-parquet \
  --output /path/to/output \
  --server-url http://127.0.0.1:8000
```

Input requires an HTML column (`content` by default) and optionally a URL
column. Output preserves the input fields and adds `text` plus `_mineru_status`.
Any row that cannot use the model is retained and routed through the configured
fallback; its status records the reason.

For one independently Zstandard-compressed frame per row:

```bash
python run_pipeline.py \
  --input /path/to/html-zstd-parquet \
  --output /path/to/output \
  --html-field _html_zstd \
  --html-compression zstd \
  --server-url http://127.0.0.1:8000
```

No uncompressed-size field is needed. Only the cell being processed is expanded,
so the batch retains its compact payload instead of materializing a second raw
HTML column.

Useful options:

- `--structured-outputs per_request` constrains every answer to one label for
  each element ID.
- `--fallback trafilatura` extracts model failures with rules; `bypass` keeps
  original HTML and `empty` emits nothing.
- `--output-format mm_md` preserves maths and images; `none` emits pruned HTML.
- `--max-model-len` must match the server. Oversize rows go to the fallback.
- `--files-per-partition 1` is appropriate when the source was pre-sharded to
  roughly 150–200 rows per Parquet file.

For a full Common Crawl snapshot, use the manifest, Slurm, atomic-publish, and
verification workflow in
[benchmarking/mineru-html-snapshot.md](../../../benchmarking/mineru-html-snapshot.md).
