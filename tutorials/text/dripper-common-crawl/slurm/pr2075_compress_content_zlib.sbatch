#!/usr/bin/env bash
#SBATCH --job-name=pr2075-html-zlib
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_logs/%x-%j.err

set -euo pipefail

export PYTHON=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin/python
export PATH=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo-curator-html-parser-pr2075-venv/bin:${PATH}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

cd /home/vjawa/nemo-curator-html-parser

INPUT=${1:?usage: sbatch pr2075_compress_content_zlib.sbatch INPUT_DIR OUTPUT_DIR}
OUT=${2:?usage: sbatch pr2075_compress_content_zlib.sbatch INPUT_DIR OUTPUT_DIR}
WORKERS=${WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-2048}

"$PYTHON" tutorials/text/dripper-common-crawl/compress_html_parquet.py \
  --input "$INPUT" \
  --output "$OUT" \
  --workers "$WORKERS" \
  --batch-size "$BATCH_SIZE"
