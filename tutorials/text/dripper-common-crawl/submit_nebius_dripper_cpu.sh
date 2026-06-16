#!/usr/bin/env bash
# Submit the Dripper CPU-only pipeline (no vLLM) to Nebius Slurm.
# Runs WARC fetch → parse → group → preprocess → cluster → plan.
# Useful for fast failure detection without the 3-4 min vLLM startup.
#
# Usage:
#   bash submit_nebius_dripper_cpu.sh HOST MANIFEST_PATH OUTPUT_DIR [MAX_ROWS]
#
# Example — smoke (all rows, CPU only):
#   bash submit_nebius_dripper_cpu.sh \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../nemo_curator_dripper_layout_cpu_parallel_manifest_20260609 \
#     /lustre/.../dripper_cpu_smoke
#
# Key env vars (all optional):
#   NODES CPUS_PER_TASK TIME_LIMIT PARTITION ACCOUNT
#   MIN_ROWS_PER_BATCH OUTPUT_SHARDS WARC_MAX_WORKERS LOG_LEVEL
#   LAYOUT_CLUSTER_THRESHOLD LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE ...

set -euo pipefail

HOST="${1:?Usage: $0 <host> <manifest_path> <output_dir> [max_rows]}"
MANIFEST_PATH="${2:?}"
OUTPUT_DIR="${3:?}"
MAX_ROWS="${4:-0}"

_host_user="$(echo "${HOST}" | cut -d@ -f1)"
REMOTE_USER="${REMOTE_USER:-${_host_user}}"
USER_CACHE_ROOT="${USER_CACHE_ROOT:-/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}}"
SHARED_CODE="${SHARED_CODE:-${USER_CACHE_ROOT}/nemo_curator_shared}"
LOG_DIR="${LOG_DIR:-${SHARED_CODE}/logs}"
SHARED_VENV="${SHARED_VENV:-${SHARED_CODE}/.venv}"

# Slurm resources — 1 GPU requested only to satisfy partition requirement; pipeline won't use it
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"

# Pipeline
MIN_ROWS_PER_BATCH="${MIN_ROWS_PER_BATCH:-1000}"
OUTPUT_SHARDS="${OUTPUT_SHARDS:-8}"
WARC_MAX_WORKERS="${WARC_MAX_WORKERS:-64}"
WORKER_COUNT="${WORKER_COUNT:-}"
USE_S3="${USE_S3:-}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# Layout clustering
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
LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE="${LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE="${LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES="${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES:-0}"
LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES="${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES:-0}"
LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES="${LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES:-0}"
LAYOUT_TEMPLATE_LARGE_HOST_MODE="${LAYOUT_TEMPLATE_LARGE_HOST_MODE:-standalone}"
LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY="${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY:-1}"
DYNAMIC_CLASSID_SIMILARITY_THRESHOLD="${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD:-0.85}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=== Syncing code to ${HOST}:${SHARED_CODE}/ ==="
rsync -az \
    "${SCRIPT_DIR}/pipeline_cpu_only.py" \
    "${HOST}:${SHARED_CODE}/tutorials/text/dripper-common-crawl/"
rsync -az --exclude='__pycache__' --exclude='*.pyc' \
    "${REPO_ROOT}/nemo_curator/" \
    "${HOST}:${SHARED_CODE}/nemo_curator/"

# Clear stale bytecode so editable installs pick up the synced changes.
ssh "${HOST}" bash <<PYCACHE
find "${SHARED_CODE}/nemo_curator" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
PYCACHE

echo "=== Submitting Dripper CPU-only pipeline ==="
echo "  Host      : ${HOST}"
echo "  Manifest  : ${MANIFEST_PATH}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Max rows  : ${MAX_ROWS} (0=all)"
echo "  CPUs      : ${CPUS_PER_TASK}  Time: ${TIME_LIMIT}  Log: ${LOG_LEVEL}"
echo ""

# Build GPU flag before heredoc so ${GPU_ARGS} expands locally into the unquoted heredoc
GPU_ARGS=""
[ "${GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null && GPU_ARGS="--gpus-per-node=${GPUS_PER_NODE}" || true

JOB_ID=$(ssh "${HOST}" bash <<REMOTE
set -euo pipefail
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

TMPJOB=\$(mktemp "${LOG_DIR}/dripper_cpu_XXXXXX.sh")
cat > "\${TMPJOB}" << 'JOBSCRIPT'
#!/bin/bash
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=${SHARED_VENV}
export PATH=${SHARED_VENV}/bin:\${PATH}
export HF_HOME=${USER_CACHE_ROOT}/hf_cache
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export TMPDIR=/tmp
[ -f ${USER_CACHE_ROOT}/cache_env.sh ] && set -a && source ${USER_CACHE_ROOT}/cache_env.sh && set +a || true

echo "=== Dripper CPU-only Pipeline ===" && hostname && date

# Clear stale bytecode immediately before execution so synced .py files are used.
find ${SHARED_CODE}/nemo_curator -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

cd ${SHARED_CODE}
srun --ntasks-per-node=1 \
  ${SHARED_VENV}/bin/python \
    tutorials/text/dripper-common-crawl/pipeline_cpu_only.py \
    --slurm \
    --manifest-path ${MANIFEST_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --max-rows ${MAX_ROWS} \
    --min-rows-per-batch ${MIN_ROWS_PER_BATCH} \
    --output-shards ${OUTPUT_SHARDS} \
    --warc-max-workers ${WARC_MAX_WORKERS} \
    --log-level ${LOG_LEVEL} \
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
    --layout-template-failed-host-fallback-signature-mode ${LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE} \
    --layout-template-failed-layout-fallback-signature-mode ${LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE} \
    --layout-template-host-single-cluster-min-pages ${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES} \
    --layout-template-host-single-cluster-max-pages ${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES} \
    --layout-template-max-exact-host-pages ${LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES} \
    --layout-template-large-host-mode ${LAYOUT_TEMPLATE_LARGE_HOST_MODE} \
    --layout-template-propagation-concurrency ${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY} \
    --dynamic-classid-similarity-threshold ${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD} \
    --ray-temp-dir /tmp/ray_\${SLURM_JOB_ID} \
    --ray-num-cpus ${CPUS_PER_TASK} \
    --ray-port 6379 \
    ${USE_S3:+--use-s3} \
    ${WORKER_COUNT:+--worker-count ${WORKER_COUNT}}
JOBSCRIPT

sbatch --parsable \
  --job-name=dripper-cpu \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes="${NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  ${GPU_ARGS} \
  --time="${TIME_LIMIT}" \
  --output="${LOG_DIR}/dripper_cpu_%j.log" \
  --error="${LOG_DIR}/dripper_cpu_%j.log" \
  "\${TMPJOB}"
rm -f "\${TMPJOB}"
REMOTE
)

echo ""
echo "=== Submitted ==="
echo "  Job ID   : ${JOB_ID}"
echo "  Log      : ${LOG_DIR}/dripper_cpu_${JOB_ID}.log"
echo "  Output   : ${OUTPUT_DIR}"
echo ""
echo "Monitor with:"
echo "  ! ssh ${HOST} 'tail -f ${LOG_DIR}/dripper_cpu_${JOB_ID}.log'"
echo "  ! ssh ${HOST} 'cat ${OUTPUT_DIR}/metrics.json 2>/dev/null || echo not-done-yet'"
