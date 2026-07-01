#!/usr/bin/env bash
#SBATCH --job-name=pr2075-stage2-4gpu
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
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/vjawa/hf_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/lustre/fsw/portfolios/llmservice/users/vjawa/cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/vjawa/cache/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/vjawa/cache/triton}
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd /home/vjawa/nemo-curator-html-parser

INPUT=${1:?usage: sbatch pr2075_stage_gpu_pipeline_4gpu_subset.sbatch INPUT_DIR OUTPUT_DIR}
OUT=${2:?usage: sbatch pr2075_stage_gpu_pipeline_4gpu_subset.sbatch INPUT_DIR OUTPUT_DIR}
N_GPU_REPLICAS=${N_GPU_REPLICAS:-4}
DRIPPER_CPU_BACKEND=${DRIPPER_CPU_BACKEND:-ray}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_TOKENS=${MAX_TOKENS:-2048}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}

"$PYTHON" tutorials/text/dripper-common-crawl/stage_gpu_pipeline.py \
  --input "$INPUT" \
  --output "$OUT" \
  --num-shards 1 \
  --shard-index 0 \
  --replicas "$N_GPU_REPLICAS" \
  --cpu-backend "$DRIPPER_CPU_BACKEND" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_CACHE_DTYPE"
