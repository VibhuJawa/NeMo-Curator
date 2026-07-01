#!/usr/bin/env bash
#SBATCH --job-name=pr2075-stage2c-post
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

PROMPTS=${1:?usage: sbatch pr2075_stage2c_postprocess.sbatch STAGE2A_PROMPTS_DIR STAGE2B_RESPONSES_DIR OUTPUT_DIR}
RESPONSES=${2:?usage: sbatch pr2075_stage2c_postprocess.sbatch STAGE2A_PROMPTS_DIR STAGE2B_RESPONSES_DIR OUTPUT_DIR}
OUT=${3:?usage: sbatch pr2075_stage2c_postprocess.sbatch STAGE2A_PROMPTS_DIR STAGE2B_RESPONSES_DIR OUTPUT_DIR}
CPUS_PER_ACTOR=${CPUS_PER_ACTOR:-2}

"$PYTHON" tutorials/text/dripper-common-crawl/stage2c_postprocess_templates_cpu.py \
  --prompts "$PROMPTS" \
  --responses "$RESPONSES" \
  --output "$OUT" \
  --num-shards 1 \
  --shard-index 0 \
  --cpus-per-actor "$CPUS_PER_ACTOR"
