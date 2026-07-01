#!/usr/bin/env bash
#SBATCH --job-name=pr2075-build-200k
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.err

set -euo pipefail

source /lustre/fsw/portfolios/llmservice/users/vjawa/secrets/swiftstack_commoncrawl.env
source /lustre/fsw/portfolios/llmservice/users/vjawa/configs/cc-main-2025-26.env

export PYTHON=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python
export PATH=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin:${PATH}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

cd /home/vjawa/nemo-curator-html-parser

INPUT=${INPUT:-/lustre/fsw/portfolios/llmservice/users/vjawa/cc-main-2025-26-index-host-buckets}
OUT=${1:?usage: sbatch pr2075_build_content_subset.sbatch /lustre/.../content_input}
MANIFEST_FILE_INDICES=${MANIFEST_FILE_INDICES:-143,1000,2000,3000,4000,5000,6000}
TARGET_PAGES=${TARGET_PAGES:-200000}
MAX_PAGES_PER_HOST=${MAX_PAGES_PER_HOST:-300}
MIN_PAGES_PER_HOST=${MIN_PAGES_PER_HOST:-100}
CANDIDATE_MULTIPLIER=${CANDIDATE_MULTIPLIER:-1.15}
EXCLUDE_HOST_REGEX=${EXCLUDE_HOST_REGEX:-workplace\\.com}
FETCH_WORKERS=${FETCH_WORKERS:-$SLURM_CPUS_PER_TASK}

"$PYTHON" tutorials/text/dripper-common-crawl/build_host_bucket_content_subset.py \
  --input "$INPUT" \
  --output "$OUT" \
  --manifest-file-indices "$MANIFEST_FILE_INDICES" \
  --target-pages "$TARGET_PAGES" \
  --candidate-multiplier "$CANDIDATE_MULTIPLIER" \
  --min-pages-per-host "$MIN_PAGES_PER_HOST" \
  --max-pages-per-host "$MAX_PAGES_PER_HOST" \
  --exclude-host-regex "$EXCLUDE_HOST_REGEX" \
  --fetch-workers "$FETCH_WORKERS" \
  --log-level INFO
