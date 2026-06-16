#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Dripper Stage 1 CPU — WARC fetch + HTML extraction (no GPU needed).
# Uses the shared pre-built venv on Lustre; no uv sync at job start.
#
# Prerequisite: run scripts/setup_nebius_shared_env.sh once to build the venv.
#
# Submitting (from a Nebius login / dc node):
#   sbatch scripts/submit_nebius_dripper_stage1_cpu.sh
# Or from Mac (using SSH):
#   ssh dc-01 "cd /path/to/repo && sbatch scripts/submit_nebius_dripper_stage1_cpu.sh"

#SBATCH --job-name=curator-dripper-stage1-cpu
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=logs/dripper_stage1_cpu_%j.log
#SBATCH --error=logs/dripper_stage1_cpu_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REMOTE_USER="${REMOTE_USER:-${USER}}"
SHARED_CODE="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared"
SHARED_VENV="${SHARED_VENV:-/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared/.venv}"
USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}"

# Per-run output dirs (logs, parquets, metrics remain job-specific)
OUTPUT_DIR="${OUTPUT_DIR:-${USER_CACHE_ROOT}/dripper_stage1_cpu/${SLURM_JOB_ID}}"

# ---------------------------------------------------------------------------
# Stage 1 parameters
# ---------------------------------------------------------------------------
MAX_WARCS="${MAX_WARCS:-32}"
MAX_PAGES="${MAX_PAGES:-0}"                      # 0 = unlimited
MANIFEST_WARC_BUCKET="${MANIFEST_WARC_BUCKET:-crawl-data}"
MANIFEST_FETCH_WORKERS="${MANIFEST_FETCH_WORKERS:-64}"
INPUT_MANIFEST_PATH="${INPUT_MANIFEST_PATH:-}"
PIPELINE_SHARD_SIZE="${PIPELINE_SHARD_SIZE:-64}"
PIPELINE_SHARD_STRATEGY="${PIPELINE_SHARD_STRATEGY:-sequential}"
PIPELINE_PREPROCESS_WORKERS="${PIPELINE_PREPROCESS_WORKERS:-16}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-mm_md}"
FALLBACK="${FALLBACK:-trafilatura}"

# ---------------------------------------------------------------------------
# Source environment
# ---------------------------------------------------------------------------
set +u
source "${HOME}/.bashrc"
set -u

if [ -f "${USER_CACHE_ROOT}/cache_env.sh" ]; then
    set -a
    set +u
    # shellcheck disable=SC1090
    source "${USER_CACHE_ROOT}/cache_env.sh"
    set -u
    set +a
fi

export AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -n "${PBSS_ACCESS_KEY_ID:-}" ]; then
    export AWS_ACCESS_KEY_ID="${PBSS_ACCESS_KEY_ID}"
fi
if [ -n "${PBSS_SECRET_ACCESS_KEY:-}" ]; then
    export AWS_SECRET_ACCESS_KEY="${PBSS_SECRET_ACCESS_KEY}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${USER_CACHE_ROOT}/uv_cache}"
export HF_HOME="${HF_HOME:-${USER_CACHE_ROOT}/hf_cache}"
export TMPDIR="/tmp"

mkdir -p "${OUTPUT_DIR}" "${SHARED_CODE}/logs"

# ---------------------------------------------------------------------------
# Validate shared venv — fail fast; do NOT silently fall back to uv sync.
# Run setup_nebius_shared_env.sh first if this check fails.
# ---------------------------------------------------------------------------
echo "=================================================="
echo "  NeMo Curator Dripper Stage 1 CPU"
echo "=================================================="
echo "  Host        : $(hostname)"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Shared code : ${SHARED_CODE}"
echo "  Shared venv : ${SHARED_VENV}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  Max WARCs   : ${MAX_WARCS}"
echo "  Max pages   : ${MAX_PAGES}"
echo "  Manifest    : ${INPUT_MANIFEST_PATH:-none} bucket=${MANIFEST_WARC_BUCKET}"
echo "=================================================="

if [ ! -x "${SHARED_VENV}/bin/python" ]; then
    echo "ERROR: shared venv not found at ${SHARED_VENV}" >&2
    echo "Run scripts/setup_nebius_shared_env.sh to build it first." >&2
    exit 1
fi

echo "Validating shared venv imports..."
if ! "${SHARED_VENV}/bin/python" -c \
    "import s3fs, sklearn, mineru_html, llm_web_kit, nemo_curator" \
    >/dev/null 2>&1; then
    echo "ERROR: shared venv at ${SHARED_VENV} is missing required packages." >&2
    echo "Expected: s3fs, sklearn, mineru_html, llm_web_kit, nemo_curator" >&2
    echo "Run scripts/setup_nebius_shared_env.sh to rebuild the venv." >&2
    exit 1
fi
echo "Shared venv OK."

"${SHARED_VENV}/bin/python" --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# ---------------------------------------------------------------------------
# Run Stage 1 from the shared code directory
# ---------------------------------------------------------------------------
cd "${SHARED_CODE}"

extra_args=()
if [ -n "${INPUT_MANIFEST_PATH}" ]; then
    extra_args+=(--input-manifest-path "${INPUT_MANIFEST_PATH}")
fi
if [ "${MAX_PAGES}" != "0" ]; then
    extra_args+=(--max-pages "${MAX_PAGES}")
fi
extra_args+=(--manifest-warc-bucket "${MANIFEST_WARC_BUCKET}")
extra_args+=(--manifest-fetch-workers "${MANIFEST_FETCH_WORKERS}")
extra_args+=(--pipeline-shard-size "${PIPELINE_SHARD_SIZE}")
extra_args+=(--pipeline-shard-strategy "${PIPELINE_SHARD_STRATEGY}")
if [ -n "${PIPELINE_PREPROCESS_WORKERS}" ]; then
    extra_args+=(--pipeline-preprocess-workers "${PIPELINE_PREPROCESS_WORKERS}")
fi
extra_args+=(--output-format "${OUTPUT_FORMAT}")
extra_args+=(--fallback "${FALLBACK}")
# Stage 1 is CPU-only: precompute HTML, no inference
extra_args+=(--precompute-html-only)

"${SHARED_VENV}/bin/python" tutorials/text/dripper-common-crawl/main.py \
    --output-dir "${OUTPUT_DIR}" \
    --max-warcs "${MAX_WARCS}" \
    "${extra_args[@]}"

echo "=================================================="
echo "  Stage 1 DONE"
echo "  Output : ${OUTPUT_DIR}"
echo "  Pass OUTPUT_DIR=${OUTPUT_DIR} to Stage 2 GPU job."
echo "=================================================="
