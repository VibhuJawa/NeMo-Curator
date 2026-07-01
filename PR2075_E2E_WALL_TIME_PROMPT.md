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

## Actual 200k Run Times

These are the current actual-scale measurements for the representative 200k-page, 672-host, 622-host-bucket subset. Use SLURM elapsed as the optimization wall-time baseline; script elapsed is the in-script measurement where available.

Input materialization is separated from the core Dripper benchmark because it includes SwiftStack/Common Crawl WARC fetch and sample construction.

```text
Input prep:
build content input      job 376196  cpu_long  64 CPU, 220G       SLURM 04:08:35  script 14897.7s  MaxRSS 205,610,776K  rows 200,000
compress html_zlib       job 383638  cpu_long  32 CPU, 128G       SLURM 00:03:42  script   202.3s  MaxRSS  51,919,796K  rows 200,000

Core Dripper stages:
Stage 1a feature extract job 383895  cpu_long  64 CPU, 228G       SLURM 00:07:00  script n/a       MaxRSS  57,875,988K  rows 200,000
Stage 1b clustering      job 383906  batch     64 CPU, 4 GPU, 600G SLURM 00:01:22  script    52.4s  MaxRSS  24,804,272K  rows 200,000
Stage 2a prompt prep     job 384074  cpu_long  64 CPU, 228G       SLURM 00:02:10  script   102.6s  MaxRSS  36,786,040K  rows 200,000 -> 4,292 candidates
Stage 2b LLM inference   job 384121  batch    128 CPU, 8 GPU      SLURM 00:04:14  script   237.9s  MaxRSS  25,359,932K  prompts 4,274
Stage 2c template build  job 384122  cpu_long  64 CPU, 220G       SLURM 00:02:29  script   111.3s  MaxRSS  33,295,620K  rows 4,292
Stage 3a propagation     job 384223  cpu_long  64 CPU, 220G       SLURM 00:36:45  script  2177.7s  MaxRSS  82,317,540K  rows 200,000

Core total after compressed input: 00:54:00 summed SLURM runtime.
Total including input materialization and html_zlib compression: 05:06:17 summed SLURM runtime.
```

Stage 1a's summary does not currently record `elapsed_s`; the log shows about `04:52` in actor-pool processing, while the SLURM job wall time is `00:07:00` including Ray startup, output writing, and shutdown.

Core-stage wall-time share:

```text
Stage 3a propagation:   36:45 / 54:00 = 68.1%
Stage 1a extraction:     7:00 / 54:00 = 13.0%
Stage 2b inference:      4:14 / 54:00 =  7.8%
Stage 2c template build: 2:29 / 54:00 =  4.6%
Stage 2a prompt prep:    2:10 / 54:00 =  4.0%
Stage 1b clustering:     1:22 / 54:00 =  2.5%
```

Stage 2b 8-GPU worker skew:

```text
gpu  prompts  prompt_files  inference_s  setup_s  truncated
0       454       73            106.5      38.8      17
1       527       76            137.4      38.9      21
2       470       78            137.7      38.8      19
3       806       79            139.2      38.8      22
4       555       79            119.2      38.8      11
5       485       79            132.3      38.8      17
6       505       79            151.2      38.8      27
7       472       79            146.4      38.8      18
```

Stage 3a long-tail buckets dominate the current wall time:

```text
bucket  elapsed_s  rows  success  retry  errors  lbp_static  lbp_parser  mapping_clusters
00225    1705.7    300     300       0      0       298          0             2
01578    1601.1    300     286      13     14       228         55             3
00249    1115.5    300     264      36     36       262          0             2
00474     916.3    300     299       1      1       295          0             4
01719     908.7    600     508      92     92       209        290             4
02852     888.8    600     521      79     79       481          6            15
00975     839.7    300     297       3      3         0        290             4
01717     785.7    600     558      42     42       285        266             6
```

Stage 3b is not yet measured at full 200k retry scale. The current Stage 3b numbers are only the one-bucket smoke run on bucket `00209`; do not use them as a full-scale benchmark.

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
