# PR 2075 E2E Wall-Time Prompt

## Objective

Optimize the current NVIDIA-NeMo/Curator PR 2075 Common Crawl Dripper pipeline for one representative `CC-MAIN-2025-26` subset. The active goal is clean, reproducible wall-time measurement and then targeted CPU/GPU optimization.

Work from the current staged DAG only. Keep each stage independently runnable and measured before considering fusion.

## Working Rules

- Use `/lustre` for environments, data, caches, and outputs.
- Do not run Ray workloads on the login/shared host. Use SLURM CPU nodes for CPU stages and SLURM GPU nodes for GPU stages.
- Keep HTML as per-row `html_zlib` with `html_chars`; do not carry raw `html` through stage outputs.
- Use strict disk contracts between stages. Join by `record_id` plus `cluster_id`, not URL.
- Missing required GPU/CuPy/cuML/llm_web_kit/vLLM dependencies should fail fast.
- Keep benchmark-stage status explicit: rows that cannot produce prompts, responses, mappings, propagated content, or retry content should remain counted error/status rows.

## Checkout And Environment

- Repo: `/home/vjawa/nemo-curator-html-parser`
- Branch: `pr-2075-dripper-cc`
- Venv: `/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv`
- Python:

```bash
source /lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/activate
cd /home/vjawa/nemo-curator-html-parser
```

Useful config:

```bash
source /lustre/fsw/portfolios/llmservice/users/vjawa/secrets/swiftstack_commoncrawl.env
source /lustre/fsw/portfolios/llmservice/users/vjawa/configs/cc-main-2025-26.env
```

Installed runtime includes `ray`, `nemo_curator`, `mineru_html`, `webpage_extractor`, `trafilatura`, `warcio`, `xxhash`, `vllm`, `cupy`, `cuml`, `cudf`, `lance`, and `llm_web_kit`.

## Current Data Paths

200k working subset root:

```text
/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_20260626_174637/streaming_stage1b_20260630_115606
```

Current canonical inputs/outputs:

```text
Stage 1b:
/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_20260626_174637/streaming_stage1b_20260630_115606/stage1b_streaming_4gpu_retry4_cupy_singletons

Stage 2 split root:
/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_20260626_174637/streaming_stage1b_20260630_115606/stage2_split_8gpu_bench_20260630_143226

Stage 2c templates:
/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_20260626_174637/streaming_stage1b_20260630_115606/stage2_split_8gpu_bench_20260630_143226/stage2c_templates_8gpu

Stage 3a:
/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_20260626_174637/streaming_stage1b_20260630_115606/stage3a_retry_full_20260630_161905

Stage 3b smoke:
/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_stage3b_smoke_20260630_170139
```

## Current DAG

```text
Stage 1a CPU feature extraction
  input: content parquet with html_zlib
  output: host_bucket_*.parquet with dom_feature

Stage 1b GPU clustering
  input: Stage 1a host buckets
  output: host_bucket_*.parquet with cluster_id and cluster_role

Stage 2a CPU prompt prep
  input: Stage 1b representatives and singletons
  output: prompt_*.parquet

Stage 2b GPU LLM inference
  input: prompt_*.parquet
  output: response_*.parquet

Stage 2c CPU postprocess and template build
  input: Stage 2a prompts plus Stage 2b responses
  output: stage2c_host_bucket_*.parquet with content and representative mapping_json

Stage 3a CPU propagation
  input: Stage 1b host bucket plus matching Stage 2c host bucket
  output: stage3_host_bucket_*.parquet
  retry side channel: stage3b_retry_input/retry_host_bucket_*.parquet

Stage 3b-a CPU retry prompt prep
  input: stage3b_retry_input/retry_host_bucket_*.parquet
  output: prompt_retry_host_bucket_*.parquet

Stage 3b-b GPU retry LLM inference
  input: prompt_retry_host_bucket_*.parquet
  output: response_retry_host_bucket_*.parquet

Stage 3b-c CPU retry postprocess
  input: Stage 3b-a prompts plus Stage 3b-b responses
  output: stage3b_retry_host_bucket_*.parquet

Stage 3b-d CPU final merge
  input: Stage 3a output plus Stage 3b-c retry output
  output: stage3_final_host_bucket_*.parquet
```

## Current Scripts

```text
tutorials/text/dripper-common-crawl/stage2a_prompt_prep_cpu.py
tutorials/text/dripper-common-crawl/stage2b_llm_inference_gpu.py
tutorials/text/dripper-common-crawl/stage2c_postprocess_templates_cpu.py
tutorials/text/dripper-common-crawl/stage3_cpu_propagation.py
tutorials/text/dripper-common-crawl/stage3b_retry_prompt_prep_cpu.py
tutorials/text/dripper-common-crawl/stage3b_retry_postprocess_cpu.py
tutorials/text/dripper-common-crawl/stage3b_merge_retry_outputs.py
```

SLURM wrappers:

```text
tutorials/text/dripper-common-crawl/slurm/pr2075_stage2a_prompt_prep.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage2b_llm_1gpu.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage2b_llm_4gpu.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage2b_llm_8gpu.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage2c_postprocess.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage3_subset.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage3b_prompt_prep.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage3b_postprocess.sbatch
tutorials/text/dripper-common-crawl/slurm/pr2075_stage3b_merge.sbatch
```

## Latest Verified Metrics

Stage 1b / Stage 3 role counts on the 200k subset:

```text
rows:              200,000
representatives:    2,470
siblings:         195,708
singletons:         1,822
Stage 2 LLM rows:   4,292
```

Stage 2c 8-GPU template output:

```text
output rows:          4,292
ok rows:              4,192
postprocess errors:      82
no_item_ids rows:         18
mapping_json rows:     2,421
```

Stage 3a full run:

```text
SLURM job:           384223
state:               COMPLETED
SLURM elapsed:       00:36:45
script elapsed:      2177.7s
MaxRSS:              82,317,540K
bucket files:        622 / 622
rows:                200,000
success rows:        175,789
error rows:           24,211
retry rows:           24,060

method_lbp_static:             139,266
method_layout_batch_parser:     32,382
method_llm_representative:       2,470
method_llm_singleton:            1,822
method_failed:                  24,060
```

Stage 3b-a smoke on bucket `00209`:

```text
SLURM job:       384277
state:           COMPLETED
SLURM elapsed:   00:01:23
MaxRSS:          3,810,608K
input rows:      390
retry rows:      390
prompt ok rows:   91
no_item_ids:     299
prompt chars:    1,572,468
```

Queued Stage 3b smoke continuation:

```text
384281  Stage 3b-b GPU inference  PENDING (Priority)
384282  Stage 3b-c postprocess    PENDING (afterok:384281)
384283  Stage 3b-d merge          PENDING (afterok:384282)
```

Latest local validation:

```text
py_compile: PASS for Dripper helpers and Stage 2/3 scripts
bash -n:    PASS for Stage 2/3 SLURM wrappers
diff check: PASS
pytest:     PASS, 13 tests, SLURM job 384286, cpu_short, 30.50s
```

## Commands For Current Stage 3b Smoke

```bash
ROOT=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_stage3b_smoke_20260630_170139
STAGE3A=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_20260626_174637/streaming_stage1b_20260630_115606/stage3a_retry_full_20260630_161905

squeue -j 384281,384282,384283 \
  -o '%.18i %.9P %.35j %.8T %.10M %.6D %R'

tail -f /lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/pr2075-stage2b-1gpu-384281.out
```

After the dependency chain completes:

```bash
/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_stage3b_smoke_20260630_170139")
for rel in [
    "stage3b_b_responses/_stage2b_summary.json",
    "stage3b_c_postprocess/_stage3b_postprocess_summary.json",
    "stage3b_d_final/_stage3b_merge_summary.json",
]:
    path = root / rel
    print(path)
    print(json.dumps(json.loads(path.read_text()), indent=2)[:2000])
PY
```

## Optimization Questions

1. Stage 3a has a long-tail bucket issue: `stage3_host_bucket_01578` took about `1601s` while most buckets completed much sooner. Profile per-row and per-host propagation cost inside slow buckets.
2. Stage 3a memory rose to about `82GB` MaxRSS on a 64-CPU node. Confirm whether this is Ray/task concurrency overhead, per-task HTML decompression, or transient parser state.
3. Stage 3b-a produced prompts for only `91/390` retry rows in bucket `00209`; quantify this across all retry files before running a full GPU retry pass.
4. Stage 2b GPU scheduling currently balances prompt files by prompt characters. Compare that against observed token counts and per-GPU inference time.
5. Keep all stage metrics separate until bottlenecks are clear; only then consider fusing adjacent CPU stages or reducing intermediate Parquet writes.

## Validation Commands

```bash
python -m py_compile \
  tutorials/text/dripper-common-crawl/stage3_cpu_propagation.py \
  tutorials/text/dripper-common-crawl/stage3b_retry_prompt_prep_cpu.py \
  tutorials/text/dripper-common-crawl/stage3b_retry_postprocess_cpu.py \
  tutorials/text/dripper-common-crawl/stage3b_merge_retry_outputs.py

bash -n \
  tutorials/text/dripper-common-crawl/slurm/pr2075_stage3_subset.sbatch \
  tutorials/text/dripper-common-crawl/slurm/pr2075_stage3b_prompt_prep.sbatch \
  tutorials/text/dripper-common-crawl/slurm/pr2075_stage3b_postprocess.sbatch \
  tutorials/text/dripper-common-crawl/slurm/pr2075_stage3b_merge.sbatch

git diff --check
```
