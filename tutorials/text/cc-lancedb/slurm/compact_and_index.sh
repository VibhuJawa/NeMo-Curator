#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm job: compact the cc_url_index LanceDB table and build scalar indexes.
# Submitted automatically by submit_array.sh after all 34 array tasks succeed.
# All I/O is PBSS ↔ memory ↔ PBSS — no local disk writes.
#
# Manual submit:
#   sbatch --dependency=afterokarray:<ARRAY_JOB_ID> slurm/compact_and_index.sh

#SBATCH --job-name=cc-lancedb-index
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_dataprocessing
#SBATCH --output=logs/cc-lancedb-compact-%j.out
#SBATCH --error=logs/cc-lancedb-compact-%j.err

set -euo pipefail

VENV=${VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv}
LANCEDB_URI=${LANCEDB_URI:-s3://vjawa-cc-lance}
TABLE_NAME=${TABLE_NAME:-cc_url_index}
REPO=${REPO:-/home/vjawa/nemo-curator-adlr-mm}

# PBSS credentials — all writes go to SwiftStack, zero local disk
export AWS_ENDPOINT_URL_S3=${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}
: "${AWS_ACCESS_KEY_ID:?Need AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Need AWS_SECRET_ACCESS_KEY}"
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

mkdir -p logs

echo "========================================================"
echo "Job       : ${SLURM_JOB_ID}"
echo "LanceDB   : ${LANCEDB_URI}/${TABLE_NAME}"
echo "Node      : $(hostname)"
echo "Start     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Storage   : all I/O via PBSS (https://pdx.s8k.io) — no local writes"
echo "========================================================"

export PYTHONPATH=$REPO:${PYTHONPATH:-}
cd "$REPO/tutorials/text/cc-lancedb"

"$VENV/bin/python" compact_and_index.py \
    --lancedb-uri "$LANCEDB_URI" \
    --table-name  "$TABLE_NAME" \
    --num-threads "${SLURM_CPUS_PER_TASK:-32}"

echo "Compaction complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
