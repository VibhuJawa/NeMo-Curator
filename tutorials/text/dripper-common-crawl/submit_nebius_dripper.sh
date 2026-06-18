#!/usr/bin/env bash
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
# Submit the Dripper streaming pipeline to Nebius Slurm.
#
# Usage:
#   bash submit_nebius_dripper.sh HOST MANIFEST_PATH OUTPUT_DIR [MAX_ROWS]
#
# Example — smoke test (200 rows):
#   bash submit_nebius_dripper.sh \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../nemo_curator_dripper_layout_cpu_parallel_manifest_20260609 \
#     /lustre/.../dripper_streaming_smoke \
#     200
#
# Example — full host shard (all rows):
#   bash submit_nebius_dripper.sh \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../nemo_curator_dripper_sorted_host_buckets_20260611/shard_0001.parquet \
#     /lustre/.../dripper_streaming_shard0001
#
# Key env vars (all optional, defaults match 1×8×H100 node):
#   NODES GPUS_PER_NODE CPUS_PER_TASK TIME_LIMIT PARTITION ACCOUNT
#   REPLICAS TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION MAX_MODEL_LEN
#   MAX_TOKENS MAX_CONCURRENT_REQUESTS MAX_NUM_SEQS MIN_ROWS_PER_BATCH OUTPUT_SHARDS
#   USE_S3 WORKER_COUNT
#   LAYOUT_TEMPLATE_VALIDATION_ROWS LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1
#   MODEL_IDENTIFIER LOG_DIR SHARED_VENV

set -euo pipefail

HOST="${1:?Usage: $0 <host> <manifest_or_input_path> <output_dir> [max_rows]}"
MANIFEST_PATH="${2:?}"
OUTPUT_DIR="${3:?}"
MAX_ROWS="${4:-0}"

# Phase 2 (inference-only): set INFERENCE_ONLY=1 and pass a Phase 1 clustering-output
# parquet as arg 2 — the pipeline reads precomputed layout_id + prompts and skips
# group/WARC/parse/preprocess/cluster/plan (no clustering contention in the vLLM run).
INFERENCE_ONLY="${INFERENCE_ONLY:-}"
if [ -n "${INFERENCE_ONLY}" ]; then
    INPUT_FLAG="--input-parquet ${MANIFEST_PATH}"
else
    INPUT_FLAG="--manifest-path ${MANIFEST_PATH}"
fi

# Derive remote username from user@host syntax; override with REMOTE_USER env var.
_host_user="$(echo "${HOST}" | cut -d@ -f1)"
REMOTE_USER="${REMOTE_USER:-${_host_user}}"
USER_CACHE_ROOT="${USER_CACHE_ROOT:-/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}}"
SHARED_CODE="${SHARED_CODE:-${USER_CACHE_ROOT}/nemo_curator_shared}"
LOG_DIR="${LOG_DIR:-${SHARED_CODE}/logs}"
SHARED_VENV="${SHARED_VENV:-${SHARED_CODE}/.venv}"

# Slurm resources
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"

# vLLM server
MODEL_IDENTIFIER="${MODEL_IDENTIFIER:-opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact}"
# Leave 1 GPU for DripperHTMLLayoutClusteringStage (gpus=0.5); vLLM gets the rest.
REPLICAS="${REPLICAS:-$((GPUS_PER_NODE - 1))}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
# vLLM engine log level: set INFO to surface generation throughput / KV-cache stats.
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# Pipeline
MIN_ROWS_PER_BATCH="${MIN_ROWS_PER_BATCH:-1000}"
OUTPUT_SHARDS="${OUTPUT_SHARDS:-24}"
WARC_MAX_WORKERS="${WARC_MAX_WORKERS:-64}"
USE_S3="${USE_S3:-}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
WORKER_COUNT="${WORKER_COUNT:-}"

# Layout template
LAYOUT_CLUSTER_THRESHOLD="${LAYOUT_CLUSTER_THRESHOLD:-0.95}"
LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE="${LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE:-2}"
LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO="${LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO:-0.50}"
LAYOUT_TEMPLATE_VALIDATION_ROWS="${LAYOUT_TEMPLATE_VALIDATION_ROWS:-2}"
LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1="${LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1:-0.98}"
LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE="${LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS="${LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS:-0}"
LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE="${LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE:-0}"
LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES="${LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES:-1}"
LAYOUT_TEMPLATE_FEATURE_SOURCE="${LAYOUT_TEMPLATE_FEATURE_SOURCE:-raw_html}"
LAYOUT_TEMPLATE_PROPAGATION_TARGET="${LAYOUT_TEMPLATE_PROPAGATION_TARGET:-raw_html}"
LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE="${LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE:-converted}"
LAYOUT_PAGE_SIGNATURE_MODE="${LAYOUT_PAGE_SIGNATURE_MODE:-none}"
LAYOUT_EXACT_QUERY_VALUE_KEYS="${LAYOUT_EXACT_QUERY_VALUE_KEYS:-entityid,id}"
LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY="${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY:-1}"
DYNAMIC_CLASSID_SIMILARITY_THRESHOLD="${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD:-0.85}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Sync tutorial entry point + full nemo_curator package to remote.
# The remote uses an editable install pointing at SHARED_CODE, so rsyncing
# Python source is sufficient — no reinstall needed.
echo "=== Syncing code to ${HOST}:${SHARED_CODE}/ ==="
rsync -az \
    "${SCRIPT_DIR}/pipeline.py" \
    "${HOST}:${SHARED_CODE}/tutorials/text/dripper-common-crawl/"
rsync -az --exclude='__pycache__' --exclude='*.pyc' \
    "${REPO_ROOT}/nemo_curator/" \
    "${HOST}:${SHARED_CODE}/nemo_curator/"

# Clear stale bytecode so editable installs pick up the synced changes.
ssh "${HOST}" bash <<PYCACHE
find "${SHARED_CODE}/nemo_curator" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
PYCACHE

echo "=== Submitting Dripper streaming pipeline ==="
echo "  Host      : ${HOST}"
echo "  Manifest  : ${MANIFEST_PATH}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Max rows  : ${MAX_ROWS} (0=all rows)"
echo "  Nodes     : ${NODES} × ${GPUS_PER_NODE} H100s  Replicas: ${REPLICAS}"
echo "  Time      : ${TIME_LIMIT}"
echo ""

JOB_ID=$(ssh "${HOST}" bash <<REMOTE
set -euo pipefail
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Write the job body to a temp file.  The inner heredoc uses a *quoted* delimiter
# ('JOBSCRIPT') so that remote bash writes the content literally — SLURM_JOB_ID and
# RAY_TMPDIR are preserved as unexpanded placeholders for the job's own shell.
# The outer <<REMOTE is unquoted so the local shell expands all config variables
# (SHARED_VENV, MANIFEST_PATH, etc.) before sending anything over SSH.
TMPJOB=\$(mktemp "${LOG_DIR}/dripper_XXXXXX.sh")
cat > "\${TMPJOB}" << 'JOBSCRIPT'
#!/bin/bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=${SHARED_VENV}
export PATH=${SHARED_VENV}/bin:\${PATH}
export HF_HOME=${USER_CACHE_ROOT}/hf_cache
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export TMPDIR=/tmp
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}
[ -f ${USER_CACHE_ROOT}/cache_env.sh ] && set -a && source ${USER_CACHE_ROOT}/cache_env.sh && set +a || true

# Override AWS credentials with PBSS Common Crawl WARC credentials.
# data.commoncrawl.org (HTTPS) is blocked from Nebius; use PBSS S3 instead.
# crawl-data/<path> filenames → bucket=crawl-data, key=<path> (strip prefix).
if [ -n "\${PBSS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="\${PBSS_ACCESS_KEY_ID}"
  export AWS_SECRET_ACCESS_KEY="\${PBSS_SECRET_ACCESS_KEY}"
  export AWS_ENDPOINT_URL_S3="https://pdx.s8k.io"
  export CC_USE_S3="1"
  export CC_S3_BUCKET="crawl-data"
  export CC_S3_KEY_PREFIX="crawl-data/"
fi

echo "=== Dripper Streaming Pipeline ===" && hostname && date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Clear stale bytecode immediately before execution so synced .py files are used.
find ${SHARED_CODE}/nemo_curator -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

cd ${SHARED_CODE}
srun --ntasks-per-node=1 \
  ${SHARED_VENV}/bin/python \
    tutorials/text/dripper-common-crawl/pipeline.py \
    --slurm \
    ${INPUT_FLAG} \
    --output-dir ${OUTPUT_DIR} \
    --max-rows ${MAX_ROWS} \
    --model-identifier ${MODEL_IDENTIFIER} \
    --replicas ${REPLICAS} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-tokens ${MAX_TOKENS} \
    --max-concurrent-requests ${MAX_CONCURRENT_REQUESTS} \
    ${USE_S3:+--use-s3} \
    --min-rows-per-batch ${MIN_ROWS_PER_BATCH} \
    --output-shards ${OUTPUT_SHARDS} \
    --warc-max-workers ${WARC_MAX_WORKERS} \
    --layout-cluster-threshold ${LAYOUT_CLUSTER_THRESHOLD} \
    --layout-template-min-cluster-size ${LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE} \
    --layout-template-max-selected-item-ratio ${LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO} \
    --layout-template-validation-rows ${LAYOUT_TEMPLATE_VALIDATION_ROWS} \
    --layout-template-validation-min-content-f1 ${LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1} \
    --layout-template-validation-signature-mode ${LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE} \
    --layout-template-large-cluster-validation-rows ${LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS} \
    --layout-template-large-cluster-min-size ${LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE} \
    --layout-template-representative-candidates ${LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES} \
    --layout-template-feature-source ${LAYOUT_TEMPLATE_FEATURE_SOURCE} \
    --layout-template-propagation-target ${LAYOUT_TEMPLATE_PROPAGATION_TARGET} \
    --layout-template-propagation-content-source ${LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE} \
    --layout-page-signature-mode ${LAYOUT_PAGE_SIGNATURE_MODE} \
    --layout-exact-query-value-keys ${LAYOUT_EXACT_QUERY_VALUE_KEYS} \
    --layout-template-propagation-concurrency ${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY} \
    --dynamic-classid-similarity-threshold ${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD} \
    --ray-temp-dir /tmp/ray_\${SLURM_JOB_ID} \
    --ray-num-cpus ${CPUS_PER_TASK} \
    --ray-num-gpus ${GPUS_PER_NODE} \
    --ray-port 6379 \
    --log-level ${LOG_LEVEL} \
    ${MAX_NUM_SEQS:+--max-num-seqs ${MAX_NUM_SEQS}} \
    ${WORKER_COUNT:+--worker-count ${WORKER_COUNT}}
JOBSCRIPT

sbatch --parsable \
  --job-name=dripper-streaming \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes="${NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --time="${TIME_LIMIT}" \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  --output="${LOG_DIR}/dripper_streaming_%j.log" \
  --error="${LOG_DIR}/dripper_streaming_%j.log" \
  "\${TMPJOB}"
rm -f "\${TMPJOB}"
REMOTE
)

echo ""
echo "=== Submitted ==="
echo "  Job ID   : ${JOB_ID}"
echo "  Log      : ${LOG_DIR}/dripper_streaming_${JOB_ID}.log"
echo "  Output   : ${OUTPUT_DIR}"
echo ""
echo "Monitor with:"
echo "  ! ssh ${HOST} 'tail -f ${LOG_DIR}/dripper_streaming_${JOB_ID}.log'"
echo "  ! ssh ${HOST} 'squeue -j ${JOB_ID} --format=\"%.10i %.20j %.8T %.10M\"'"
echo "  ! ssh ${HOST} 'cat ${OUTPUT_DIR}/metrics.json 2>/dev/null || echo not-done-yet'"
