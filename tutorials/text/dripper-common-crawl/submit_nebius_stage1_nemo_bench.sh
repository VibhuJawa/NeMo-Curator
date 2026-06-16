#!/bin/bash
# Stage 1 NeMo Curator benchmark — CommonCrawlWARCReader via Ray Data.
#
# Compares against the serial ThreadPool approach (submit_nebius_dripper_stage1_global.sh)
# by distributing WARC fetch across multiple Ray actors in parallel.
#
# Submit manually (single shard):
#   bash submit_nebius_stage1_nemo_bench.sh  [HOST [SHARD_ID]]
# Or submit via sbatch directly on the login node:
#   sbatch --export=ALL,SHARD_ID=0001 submit_nebius_stage1_nemo_bench.sh
#
#SBATCH --job-name=dripper-s1-nemo
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=dripper_s1_nemo_%j.log
#SBATCH --error=dripper_s1_nemo_%j.log

set -euo pipefail

# ── If invoked locally (not inside Slurm), SSH to the cluster and submit ──────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
    SHARD_ID="${2:-0001}"
    REMOTE_USER="${REMOTE_USER:-vjawa}"
    USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}"
    SHARED_CODE="${USER_CACHE_ROOT}/nemo_curator_shared"

    echo "Syncing scripts to ${HOST}..."
    rsync -az --mkpath \
        /Users/vjawa/Documents/codex/nemo_curator_domain_cluster/tutorials/text/dripper-common-crawl/benchmark_stage1_nemo.py \
        /Users/vjawa/Documents/codex/nemo_curator_domain_cluster/tutorials/text/dripper-common-crawl/submit_nebius_stage1_nemo_bench.sh \
        "${HOST}:${SHARED_CODE}/tutorials/text/dripper-common-crawl/"

    ssh -o "ControlMaster=auto" \
        -o "ControlPath=/tmp/.nebius_ctl/%C.sock" \
        -o "ConnectTimeout=30" \
        "${HOST}" \
        "sbatch --parsable \
            --export=ALL,SHARD_ID=${SHARD_ID},REMOTE_USER=${REMOTE_USER} \
            --chdir=${SHARED_CODE}/logs \
            ${SHARED_CODE}/tutorials/text/dripper-common-crawl/submit_nebius_stage1_nemo_bench.sh"
    exit 0
fi

# ── Running inside Slurm ──────────────────────────────────────────────────────
REMOTE_USER="${REMOTE_USER:-${USER}}"
SHARED_CODE="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared"
SHARED_VENV="${SHARED_CODE}/.venv"
USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}"

SHARD_ID="${SHARD_ID:-0001}"
SHARD_DIR="${SHARD_DIR:-${USER_CACHE_ROOT}/nemo_curator_dripper_sorted_host_buckets_20260611}"
GLOBAL_OUTPUT_ROOT="${GLOBAL_OUTPUT_ROOT:-${USER_CACHE_ROOT}/dripper_global}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
MAX_WORKERS="${MAX_WORKERS:-256}"
MAX_ROWS="${MAX_ROWS:-0}"

INPUT_MANIFEST="${SHARD_DIR}/host_bucket=${SHARD_ID}.parquet"
OUTPUT_DIR="${GLOBAL_OUTPUT_ROOT}/stage1_nemo/shard_${SHARD_ID}"

set +u; source "${HOME}/.bashrc"; set -u
if [ -f "${USER_CACHE_ROOT}/cache_env.sh" ]; then
    set -a; set +u; source "${USER_CACHE_ROOT}/cache_env.sh"; set -u; set +a
fi

export AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
[ -n "${PBSS_ACCESS_KEY_ID:-}" ] && export AWS_ACCESS_KEY_ID="${PBSS_ACCESS_KEY_ID}"
[ -n "${PBSS_SECRET_ACCESS_KEY:-}" ] && export AWS_SECRET_ACCESS_KEY="${PBSS_SECRET_ACCESS_KEY}"
export UV_PROJECT_ENVIRONMENT="${SHARED_VENV}"
export PATH="${SHARED_VENV}/bin:${PATH}"
export RAY_TMPDIR="/tmp/ray_s1_nemo_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}" "${SHARED_CODE}/logs" "${RAY_TMPDIR}"

echo "=================================================="
echo "  Dripper Stage 1 NeMo Bench — shard ${SHARD_ID}"
echo "  Job     : ${SLURM_JOB_ID}"
echo "  Host    : $(hostname)"
echo "  Input   : ${INPUT_MANIFEST}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Chunks  : ${CHUNK_SIZE} rows, ${MAX_WORKERS} threads/actor"
echo "=================================================="

if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo "ERROR: shard not found: ${INPUT_MANIFEST}" >&2
    exit 1
fi

cd "${SHARED_CODE}"

srun --ntasks-per-node=1 "${SHARED_VENV}/bin/python" \
    tutorials/text/dripper-common-crawl/benchmark_stage1_nemo.py \
    --manifest-path "${INPUT_MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-workers "${MAX_WORKERS}" \
    --num-cpus 64 \
    --ray-tmpdir "${RAY_TMPDIR}" \
    --max-rows "${MAX_ROWS}"

echo "Stage 1 NeMo bench complete: ${OUTPUT_DIR}"
