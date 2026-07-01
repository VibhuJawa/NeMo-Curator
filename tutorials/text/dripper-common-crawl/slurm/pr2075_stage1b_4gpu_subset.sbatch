#!/usr/bin/env bash
#SBATCH --job-name=pr2075-stage1b-4gpu
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=600G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.err

set -euo pipefail

export PYTHON=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python
export PATH=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin:${PATH}
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
export RAY_DEDUP_LOGS=0
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
RAY_LOG_ARCHIVE=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_ray_logs/${SLURM_JOB_ID}
archive_ray_logs() {
  mkdir -p "$RAY_LOG_ARCHIVE"
  if [[ -d "$RAY_TMPDIR/session_latest/logs" ]]; then
    cp -a "$RAY_TMPDIR/session_latest/logs" "$RAY_LOG_ARCHIVE/"
  fi
}
trap archive_ray_logs EXIT

cd /home/vjawa/nemo-curator-html-parser

INPUT=${1:?usage: sbatch pr2075_stage1b_4gpu_subset.sbatch INPUT_DIR OUTPUT_DIR}
OUT=${2:?usage: sbatch pr2075_stage1b_4gpu_subset.sbatch INPUT_DIR OUTPUT_DIR}
THRESHOLD=${THRESHOLD:-0.95}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-2}
export STAGE1B_MAX_HOST_SIZE=${STAGE1B_MAX_HOST_SIZE:-3000}
CPUS_PER_ACTOR=${CPUS_PER_ACTOR:-4}

"$PYTHON" tutorials/text/dripper-common-crawl/stage1b_gpu_dbscan.py \
  --input "$INPUT" \
  --output "$OUT" \
  --num-shards 1 \
  --shard-index 0 \
  --threshold "$THRESHOLD" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --max-host-size "$STAGE1B_MAX_HOST_SIZE" \
  --cpus-per-actor "$CPUS_PER_ACTOR"
