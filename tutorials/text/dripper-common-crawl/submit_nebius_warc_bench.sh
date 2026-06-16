#!/bin/bash
# WARC fetch throughput benchmark — standalone Slurm job.
# No Ray, no GPU — pure boto3 ThreadPool range-GET benchmark.
#
# Env vars (all have defaults):
#   WORKERS      - boto3 worker count (default 64)
#   MAX_PAGES    - pages to fetch per run (default 2000)
#   PARTITION    - Slurm partition (default cpu_short)
#   CPUS         - CPUs to request (default 64)
#   MANIFEST     - path to host_bucket parquet (default shard 0001)
#
# Usage:
#   WORKERS=128 PARTITION=cpu_short sbatch submit_nebius_warc_bench.sh
#
#SBATCH --job-name=warc-bench
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/warc_bench/%j/bench.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/warc_bench/%j/bench.log

set -euo pipefail

# ── configurable via env vars ────────────────────────────────────────────────
WORKERS="${WORKERS:-64}"
MAX_PAGES="${MAX_PAGES:-2000}"
MANIFEST="${MANIFEST:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_sorted_host_buckets_20260611/host_bucket=0001.parquet}"

REMOTE_USER="${REMOTE_USER:-${USER}}"
SHARED_CODE="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared"
SHARED_VENV="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared/.venv"
USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}"
OUTPUT_DIR="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/warc_bench/${SLURM_JOB_ID}"

# Override partition/cpus if provided at submit time via #SBATCH env injection
# (these are just informational here — actual resources are set by sbatch flags)
PARTITION="${PARTITION:-cpu_short}"
CPUS="${CPUS:-64}"

# ── environment setup ────────────────────────────────────────────────────────
set +u; source "${HOME}/.bashrc" 2>/dev/null || true; set -u

if [ -f "${USER_CACHE_ROOT}/cache_env.sh" ]; then
    set -a; set +u; source "${USER_CACHE_ROOT}/cache_env.sh"; set -u; set +a
else
    echo "ERROR: cache_env.sh not found at ${USER_CACHE_ROOT}/cache_env.sh" >&2
    exit 1
fi

export AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
[ -n "${PBSS_ACCESS_KEY_ID:-}" ] && export AWS_ACCESS_KEY_ID="${PBSS_ACCESS_KEY_ID}"
[ -n "${PBSS_SECRET_ACCESS_KEY:-}" ] && export AWS_SECRET_ACCESS_KEY="${PBSS_SECRET_ACCESS_KEY}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${USER_CACHE_ROOT}/uv_cache}"
export HF_HOME="${HF_HOME:-${USER_CACHE_ROOT}/hf_cache}"
export TMPDIR="/tmp"
export UV_PROJECT_ENVIRONMENT="${SHARED_VENV}"
export PATH="${SHARED_VENV}/bin:${PATH}"

# ── fail-fast venv check ─────────────────────────────────────────────────────
if ! "${SHARED_VENV}/bin/python" -c "import boto3, pandas" >/dev/null 2>&1; then
    echo "ERROR: shared venv missing boto3 or pandas. Check ${SHARED_VENV}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ── banner ───────────────────────────────────────────────────────────────────
echo "=================================================="
echo "  WARC Fetch Benchmark"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Host     : $(hostname)"
echo "  Workers  : ${WORKERS}"
echo "  MaxPages : ${MAX_PAGES}"
echo "  Manifest : ${MANIFEST}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Endpoint : ${AWS_ENDPOINT_URL_S3}"
echo "=================================================="

# ── run benchmark ────────────────────────────────────────────────────────────
cd "${SHARED_CODE}"

"${SHARED_VENV}/bin/python" \
    tutorials/text/dripper-common-crawl/benchmark_warc_fetch.py \
    --manifest-path "${MANIFEST}" \
    --workers "${WORKERS}" \
    --max-pages "${MAX_PAGES}" \
    --bucket crawl-data \
    --endpoint-url "${AWS_ENDPOINT_URL_S3}" \
    --region "${AWS_REGION}" \
    2>&1 | tee "${OUTPUT_DIR}/bench_output.txt"

# ── extract JSON metrics to a standalone file ────────────────────────────────
grep "^BENCH_METRICS_JSON:" "${OUTPUT_DIR}/bench_output.txt" \
    | sed 's/^BENCH_METRICS_JSON://' \
    > "${OUTPUT_DIR}/metrics.json"

echo ""
echo "Metrics written to: ${OUTPUT_DIR}/metrics.json"
echo "Job complete."
