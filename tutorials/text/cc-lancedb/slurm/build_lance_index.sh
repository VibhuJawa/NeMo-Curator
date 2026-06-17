#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Distributed Lance scalar index build — runs after all 121 write tasks succeed.
# Uses lance fragment_ids API: each Ray worker builds one segment, coordinator merges.
#
#   sbatch --dependency=afterokarray:<ARRAY_JOB_ID> slurm/build_lance_index.sh

#SBATCH --job-name=cc-lance-index
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/cc-lance-index-%j.out
#SBATCH --error=logs/cc-lance-index-%j.err

set -euo pipefail

VENV=${VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv}
LANCEDB_URI=${LANCEDB_URI:-s3://vjawa-cc-lance}
TABLE_NAME=${TABLE_NAME:-cc_url_index}
REPO=${REPO:-/home/vjawa/nemo-curator-adlr-mm}
N_WORKERS=${N_WORKERS:-128}   # 4 nodes × 32 CPUs = 128 workers

export AWS_ENDPOINT_URL_S3=${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}
: "${AWS_ACCESS_KEY_ID:?Need AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Need AWS_SECRET_ACCESS_KEY}"
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

mkdir -p logs

export RAY_TMPDIR=/tmp/ray_vjawa_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR"
export PYTHONPATH=$REPO:${PYTHONPATH:-}

echo "=========================================================="
echo "Job         : ${SLURM_JOB_ID}"
echo "LanceDB     : ${LANCEDB_URI}/${TABLE_NAME}"
echo "Workers     : ${N_WORKERS}"
echo "Nodes       : ${SLURM_NNODES}"
echo "Start       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="

# Start Ray cluster across all allocated nodes
if [[ "$SLURM_NODEID" == "0" ]]; then
    ray start --head --num-cpus="${SLURM_CPUS_PER_TASK}" --temp-dir="$RAY_TMPDIR"
    RAY_ADDRESS=$(ray status | grep "Ray cluster started" | awk '{print $NF}')
    export RAY_ADDRESS
else
    sleep 10
    ray start --address="${SLURM_NODELIST%,*}:6379" --num-cpus="${SLURM_CPUS_PER_TASK}" --temp-dir="$RAY_TMPDIR"
fi

trap 'ray stop' EXIT

cd "$REPO/tutorials/text/cc-lancedb"
"$VENV/bin/python" build_lance_index.py \
    --lancedb-uri "$LANCEDB_URI" \
    --table-name  "$TABLE_NAME" \
    --n-workers   "$N_WORKERS"

echo "Index build complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
