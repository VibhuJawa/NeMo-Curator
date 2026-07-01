#!/usr/bin/env bash
#SBATCH --job-name=pr2075-stage1a-200k
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=228G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.err

set -euo pipefail

export PYTHON=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python
export PATH=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin:${PATH}
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

cd /home/vjawa/nemo-curator-html-parser

INPUT=${1:?usage: sbatch pr2075_stage1a_subset.sbatch INPUT_DIR OUTPUT_DIR}
OUT=${2:?usage: sbatch pr2075_stage1a_subset.sbatch INPUT_DIR OUTPUT_DIR}
CPUS_PER_ACTOR=${CPUS_PER_ACTOR:-2}
RAY_OBJECT_STORE_MEMORY_BYTES=${RAY_OBJECT_STORE_MEMORY_BYTES:-}
if [[ -n "$RAY_OBJECT_STORE_MEMORY_BYTES" ]]; then
  export RAY_OBJECT_STORE_MEMORY_BYTES
fi

"$PYTHON" tutorials/text/dripper-common-crawl/stage1a_feature_extraction.py \
  --input "$INPUT" \
  --output "$OUT" \
  --num-shards 1 \
  --shard-index 0 \
  --cpus-per-actor "$CPUS_PER_ACTOR"
