#!/usr/bin/env bash
#SBATCH --job-name=pr2075-stage3b-merge
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.err

set -euo pipefail

export PYTHON=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python
export PATH=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin:${PATH}
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

cd /home/vjawa/nemo-curator-html-parser

STAGE3=${1:?usage: sbatch pr2075_stage3b_merge.sbatch STAGE3A_DIR RETRY_OUTPUTS_DIR FINAL_OUTPUT_DIR}
RETRY=${2:?usage: sbatch pr2075_stage3b_merge.sbatch STAGE3A_DIR RETRY_OUTPUTS_DIR FINAL_OUTPUT_DIR}
OUT=${3:?usage: sbatch pr2075_stage3b_merge.sbatch STAGE3A_DIR RETRY_OUTPUTS_DIR FINAL_OUTPUT_DIR}
TASK_CPUS=${TASK_CPUS:-1}

"$PYTHON" tutorials/text/dripper-common-crawl/stage3b_merge_retry_outputs.py \
  --stage3 "$STAGE3" \
  --retry-outputs "$RETRY" \
  --output "$OUT" \
  --num-shards 1 \
  --shard-index 0 \
  --task-cpus "$TASK_CPUS"
