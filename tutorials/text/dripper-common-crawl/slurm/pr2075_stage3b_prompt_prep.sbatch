#!/usr/bin/env bash
#SBATCH --job-name=pr2075-stage3b-prompt
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

export PYTHON=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python
export PATH=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin:${PATH}
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

cd /home/vjawa/nemo-curator-html-parser

INPUT=${1:?usage: sbatch pr2075_stage3b_prompt_prep.sbatch STAGE3B_RETRY_INPUT OUTPUT_DIR}
OUT=${2:?usage: sbatch pr2075_stage3b_prompt_prep.sbatch STAGE3B_RETRY_INPUT OUTPUT_DIR}
CPUS_PER_ACTOR=${CPUS_PER_ACTOR:-2}

"$PYTHON" tutorials/text/dripper-common-crawl/stage3b_retry_prompt_prep_cpu.py \
  --input "$INPUT" \
  --output "$OUT" \
  --num-shards 1 \
  --shard-index 0 \
  --cpus-per-actor "$CPUS_PER_ACTOR"
