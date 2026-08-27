# CC Index URL List

Create one global unique URL list from an explicit set of Common Crawl Index snapshots. This tutorial points NeMo Curator exact deduplication and duplicate removal workflows at the PBSS-hosted CC Index parquet table, reads `url` and `warc_filename`, and writes a Parquet dataset that can be shared with a team.

The run is intentionally split into two scripts:

| Script | Partition | Purpose |
|--------|-----------|---------|
| `identify_cc_index_url_duplicates.py` | GPU `batch` | Runs `ExactDeduplicationWorkflow(text_field="url", assign_id=True)` and writes duplicate-ID side outputs under `_dedup_ids/` |
| `remove_cc_index_url_duplicates.py` | CPU `cpu_dataprocessing` | Runs `TextDuplicatesRemovalWorkflow` with the saved exact-dedup IDs and writes `global_unique_urls/` |

The default config uses these snapshots:

```text
CC-MAIN-2026-17
CC-MAIN-2026-12
CC-MAIN-2026-08
CC-MAIN-2026-04
CC-MAIN-2025-51
CC-MAIN-2025-47
CC-MAIN-2025-43
CC-MAIN-2025-38
CC-MAIN-2025-33
CC-MAIN-2025-30
```

`CC-MAIN-2025-26` and `CC-MAIN-2025-08` are not present because the config is authoritative: include only the crawls you want in `selected_crawls.yaml`.

## Install

Create a CUDA environment with Curator text deduplication dependencies and the S3 fsspec backend. Keep the environment and uv cache on Lustre so smoke tests and full jobs reuse the same install:

```bash
export UV_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/vjawa/uv_cache
uv venv /lustre/fsw/portfolios/llmservice/users/vjawa/cc-index-url-list-venv
source /lustre/fsw/portfolios/llmservice/users/vjawa/cc-index-url-list-venv/bin/activate
uv pip install -e ".[text_cuda12]" "s3fs>=2024.12.0"
```

For local smoke tests, keep Ray's temporary directory short enough for Unix socket paths:

```bash
export RAY_TMPDIR=/tmp/vjawa_ray
```

For iteration, run smoke commands on a short Slurm CPU allocation:

```bash
srun -A nemotron_n4_pre -p cpu_dataprocessing \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:20:00 --pty bash
```

Use CPU nodes for config checks and PBSS listing checks. The exact deduplication identification phase uses RAPIDS/cuDF and should run in the CUDA/RAPIDS environment used for full Curator text deduplication jobs.

## PBSS Credentials

The tutorial reads from the `commoncrawl` PBSS namespace at `https://pdx.s8k.io`. Set PBSS credentials before running:

```bash
export PBSS_ACCESS_KEY_ID=commoncrawl
export PBSS_SECRET_ACCESS_KEY=<secret>
```

The scripts also accept `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as fallbacks, but PBSS-prefixed variables are preferred so they do not collide with unrelated AWS credentials.

The scripts map the selected credentials into the process environment for `s3fs` and keep only endpoint/region in Curator storage options so workflow logs do not print secrets.

## Config

Edit `selected_crawls.yaml` to choose the exact snapshots:

```yaml
cc_index_base_uri: s3://cc-index/table/cc-main/warc
endpoint_url: https://pdx.s8k.io
output_name: global_unique_urls
included_crawls:
  - CC-MAIN-2026-17
  - CC-MAIN-2026-12
```

Each crawl expands to:

```text
s3://cc-index/table/cc-main/warc/crawl=<crawl>/subset=warc/
```

## Dry Run

Print the configured crawl directories without launching the Curator pipeline. This mode only expands the YAML config, so it does not need PBSS credentials unless you also pass `--max-files-per-crawl`.

```bash
python tutorials/text/cc-index-url-list/identify_cc_index_url_duplicates.py \
  --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
  --output /lustre/$USER/cc_index_url_list \
  --dry-run
```

For a small smoke test, cap each crawl to one source parquet file. Use the same output root for both phases.

```bash
export SMOKE_OUTPUT=/lustre/$USER/cc_index_url_list_smoke

srun -A nemotron_n4_pre -p batch \
  --nodes=1 --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=64 \
  --time=01:00:00 \
  python tutorials/text/cc-index-url-list/identify_cc_index_url_duplicates.py \
    --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
    --output "${SMOKE_OUTPUT}" \
    --max-files-per-crawl 1 \
    --slurm-ray \
    --ray-temp-dir /tmp/$USER-ray \
    --ray-num-cpus 64 \
    --ray-num-gpus 8 \
    --disable-ray-dashboard

srun -A nemotron_n4_pre -p cpu_dataprocessing \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
  --time=01:00:00 \
  python tutorials/text/cc-index-url-list/remove_cc_index_url_duplicates.py \
    --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
    --output "${SMOKE_OUTPUT}" \
    --max-files-per-crawl 1 \
    --slurm-ray \
    --ray-temp-dir /tmp/$USER-ray \
    --ray-num-cpus 16 \
    --disable-ray-dashboard
```

## Full Run

For production-sized runs, split the work into two Slurm jobs with the same `--output` root. `ExactDeduplicationWorkflow` uses GPU actors, while `TextDuplicatesRemovalWorkflow` is CPU-only. Splitting the phases keeps the GPU allocation short and avoids holding idle GPUs during duplicate removal.

The full-run examples launch one Ray process per node. Non-head Ray workers can return exit code 1 after the head node shuts down cleanly, so the `srun` wrapper below treats that worker-only shutdown code as success while preserving failures from the head process.

First, identify duplicate URLs on the GPU `batch` partition:

```bash
export OUTPUT_ROOT=/lustre/$USER/cc_index_url_list

sbatch <<'SBATCH'
#!/bin/bash
#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=cc-index-url-identify

set -euo pipefail

srun --ntasks-per-node=1 bash -lc '
set -uo pipefail
python tutorials/text/cc-index-url-list/identify_cc_index_url_duplicates.py \
  --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
  --output "${OUTPUT_ROOT}" \
  --slurm-ray \
  --ray-temp-dir /tmp/$USER-ray \
  --ray-num-cpus 64 \
  --ray-num-gpus 8 \
  --disable-ray-dashboard
status=$?
if [[ "${SLURM_NODEID:-0}" != "0" && "${status}" -eq 1 ]]; then
  echo "Worker Ray process exited 1 after head shutdown; treating worker exit as success."
  exit 0
fi
exit "${status}"
'
SBATCH
```

Then remove duplicates and write the final URL list on the CPU data-processing partition:

```bash
export OUTPUT_ROOT=/lustre/$USER/cc_index_url_list

sbatch <<'SBATCH'
#!/bin/bash
#SBATCH -A nemotron_n4_pre
#SBATCH -p cpu_dataprocessing
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --job-name=cc-index-url-remove

set -euo pipefail

srun --ntasks-per-node=1 bash -lc '
set -uo pipefail
python tutorials/text/cc-index-url-list/remove_cc_index_url_duplicates.py \
  --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
  --output "${OUTPUT_ROOT}" \
  --slurm-ray \
  --ray-temp-dir /tmp/$USER-ray \
  --ray-num-cpus 16 \
  --disable-ray-dashboard
status=$?
if [[ "${SLURM_NODEID:-0}" != "0" && "${status}" -eq 1 ]]; then
  echo "Worker Ray process exited 1 after head shutdown; treating worker exit as success."
  exit 0
fi
exit "${status}"
'
SBATCH
```

Output layout:

```text
/lustre/$USER/cc_index_url_list/
|-- _dedup_ids/                # ExactDeduplicationWorkflow duplicate IDs, ID generator, and filegroup signature
`-- global_unique_urls/        # final URL/WARC filename Parquet dataset
```

The final Parquet files contain two columns:

| Column | Description |
|--------|-------------|
| `url` | Globally unique page URL across all configured snapshots |
| `warc_filename` | WARC object path; includes `crawl-data/CC-MAIN-*` snapshot ID |

Because the deduplication key is only `url`, a URL that appears in several snapshots is written once. `warc_filename` is a retained-row provenance hint for debugging, not a complete list of every snapshot that contained that URL. The CC snapshot ID can be extracted from the `crawl-data/CC-MAIN-*` segment of `warc_filename`.

The Curator-generated row ID is used internally by exact deduplication and duplicate removal, and its side outputs remain under `_dedup_ids/`. It is not written to the final shareable dataset because it is an implementation detail of the dedup run rather than URL-list metadata.

## How It Works

1. Expand each configured crawl ID to its CC Index parquet directory.
2. List source parquet files in deterministic crawl/path order.
3. Create one Curator `FileGroupTask` per source parquet shard.
4. Run `ExactDeduplicationWorkflow(text_field="url", assign_id=True)` on those file groups to identify duplicates.
5. Persist the filegroup order signature beside the exact-dedup side outputs.
6. Run `TextDuplicatesRemovalWorkflow(input_fields=["url", "warc_filename"], output_fields=["url", "warc_filename"])` on the same file groups to write the global unique URL dataset.

Each CC Index parquet shard is already large enough for the exact-dedup pipeline, so the tutorial keeps one source shard per Curator file group. Exact deduplication and duplicate removal receive that same ordered list, which keeps ID assignment stable and avoids oversized Arrow string batches during duplicate removal.

Exact deduplication and duplicate removal are run as the clean Curator workflow pair used elsewhere in the repository. `ExactDeduplicationWorkflow` writes duplicate-ID side outputs plus an ID generator mapping, and `TextDuplicatesRemovalWorkflow` consumes those outputs to write the final parquet dataset.

Each phase script owns a Curator `RayClient`/`SlurmRayClient` lifecycle. The GPU script writes the exact-dedup side outputs first, then the CPU script validates the persisted SHA256 filegroup signature before removal so the ID generator maps back to the same source parquet groups used during exact deduplication.

When `--slurm-ray` is set, the head-node port handoff defaults to a shared `_ray_port_broadcast/` directory under `--output`. Override it with `--ray-port-broadcast-dir` if the output location is not shared across nodes.
