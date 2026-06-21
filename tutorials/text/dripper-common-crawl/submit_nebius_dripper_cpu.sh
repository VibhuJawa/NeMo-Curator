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

# Pipeline phase flags — read FIRST so the Slurm resource defaults below can be
# phase-aware (CPU-only phases must avoid the GPU watchdog partition).
CLUSTER_ONLY="${CLUSTER_ONLY:-}"  # Phase 1a: group->WARC->parse->preprocess->cluster (GPU), no plan
PLAN_ONLY="${PLAN_ONLY:-}"        # Phase 1b: plan stage only on a clustered parquet (CPU, no watchdog)
PREPROCESS_ONLY="${PREPROCESS_ONLY:-}"                  # Phase A: WARC->parse->preprocess + feature precompute (CPU)
CLUSTER_FROM_PREPROCESSED="${CLUSTER_FROM_PREPROCESSED:-}"  # Phase B: group->cluster from precomputed features (GPU)
BROADCAST_PROPAGATE_ONLY="${BROADCAST_PROPAGATE_ONLY:-}"  # Phase 2b: broadcast-propagate templates off-GPU (CPU)
TEMPLATE_TABLE_PATH="${TEMPLATE_TABLE_PATH:-}"            # Phase 2b: finalize-emitted template side-table

# CPU-only phases (preprocess / plan / broadcast-propagate) do NO GPU work, so they MUST run on the
# cpu partition. The batch/GPU partition runs a GPU-utilization watchdog that scancels
# jobs whose GPU stays idle (it killed EXP-008's cluster-only run at 31 min) — a
# CPU-only Phase A there would be cancelled long before its big-host tail finishes.
# Default such phases to the cpu partition + account with no GPU and a long limit;
# GPU phases keep batch + 1 GPU. All overridable via env.
_cpu_only_phase=""
if [ -n "${PREPROCESS_ONLY}" ] || [ -n "${PLAN_ONLY}" ] || [ -n "${BROADCAST_PROPAGATE_ONLY}" ]; then
  _cpu_only_phase=1
fi

# Slurm resources
NODES="${NODES:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
if [ -n "${_cpu_only_phase}" ]; then
  PARTITION="${PARTITION:-cpu}"
  ACCOUNT="${ACCOUNT:-llmservice_fm_text}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-0}"     # no GPU on the cpu partition
  TIME_LIMIT="${TIME_LIMIT:-04:00:00}"    # WARC fetch + per-row preprocess incl. serial big-host tail
  # cpu nodes have ~235 GB RAM but Ray defaults the object store to ~36 GiB, which
  # OVERFLOWS on a mega-host's single feature block (tgcom24's 20k rows) and wedges
  # the write stage. Give plasma plenty of room -- but the store lives in /dev/shm
  # (~135 GB on these nodes), so stay safely under that (the wedge peaked ~46 GiB).
  OBJECT_STORE_MEMORY_GB="${OBJECT_STORE_MEMORY_GB:-100}"
  # GUARD: preprocess/plan have NO GPU stage (only Phase-B clustering uses cuML). If an override put
  # this CPU-only phase on a GPU node (GPUS_PER_NODE>0 / PARTITION=batch), warn loudly: a GPU node
  # wastes the GPU AND the ~31-min GPU-idle watchdog can scancel a long CPU job (plan 357125 cleared
  # it by only ~2 min). Prefer a cpu* partition with GPUS_PER_NODE=0.
  if [ "${GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null || [ "${PARTITION}" = "batch" ]; then
    echo "WARNING: CPU-only phase on partition='${PARTITION}' GPUS_PER_NODE='${GPUS_PER_NODE}' -- no GPU stage runs" >&2
    echo "         here; a GPU node wastes the GPU and the ~31-min idle-watchdog may scancel it." >&2
    echo "         Prefer PARTITION=cpu (or cpu_dataprocessing) with GPUS_PER_NODE=0." >&2
  fi
else
  PARTITION="${PARTITION:-batch}"
  ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-1}"     # 1 GPU for cuML clustering / vLLM
  TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
  OBJECT_STORE_MEMORY_GB="${OBJECT_STORE_MEMORY_GB:-}"  # Ray default on GPU nodes
fi

# Ray temp/session dir — also where the object store SPILLS when plasma fills. The
# default /tmp is small/limited; the nodes have a big fast local disk at /raid
# (~1 TB on the cpu nodes, world-writable /raid/scratch), so spill there to remove
# the /dev/shm ceiling on mega-host blocks. Override RAY_TEMP_BASE for nodes w/o /raid.
RAY_TEMP_BASE="${RAY_TEMP_BASE:-/raid/scratch}"

# Pipeline
MIN_ROWS_PER_BATCH="${MIN_ROWS_PER_BATCH:-1000}"
MAX_ROWS_PER_BATCH="${MAX_ROWS_PER_BATCH:-}"  # Phase A balance: split big hosts into <=this; finalize re-groups whole
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

# Fail fast if the synced files do not match local byte-for-byte. A partial/failed rsync (or a
# stale editable install) would otherwise silently run OLD code -- the failure mode that wasted
# hours on tgcom24. Check the edited files directly (content sha256; macOS shasum == Linux
# sha256sum for identical bytes), so remote-only/orphan files can't cause a false mismatch.
echo "=== Verifying code sync (local vs remote sha) ==="
_FILES="nemo_curator/stages/text/experimental/dripper/stages/clustering.py nemo_curator/stages/text/experimental/dripper/stages/grouping.py tutorials/text/dripper-common-crawl/pipeline_cpu_only.py"
# one ssh round-trip: hash all files remotely (bare hashes, in $_FILES order)
_REMOTE=$(ssh "${HOST}" "cd '${SHARED_CODE}' && sha256sum ${_FILES} 2>/dev/null | cut -d' ' -f1")
_i=0
for _f in ${_FILES}; do
  _i=$((_i + 1))
  _L=$(shasum -a 256 "${REPO_ROOT}/${_f}" | cut -d' ' -f1)
  _R=$(printf '%s\n' "${_REMOTE}" | sed -n "${_i}p")
  if [ "${_L}" != "${_R}" ]; then
    echo "FATAL: sync mismatch on ${_f} (local=${_L} remote=${_R}) -- aborting before running stale code" >&2
    exit 1
  fi
  echo "  verified ${_f} sha=${_L}"
done
echo "CODE SYNC VERIFIED"

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
mkdir -p ${RAY_TEMP_BASE}/ray_\${SLURM_JOB_ID} 2>/dev/null || true
export RAY_TMPDIR=${RAY_TEMP_BASE}/ray_\${SLURM_JOB_ID}
export TMPDIR=/tmp
# The Ray object store lives in /dev/shm. If --object-store-memory-gb exceeds the
# node's /dev/shm, Ray raises a FATAL ValueError at init. This flag downgrades that
# to a warning + disk-backed spill so a too-large request never crashes the job.
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
[ -f ${USER_CACHE_ROOT}/cache_env.sh ] && set -a && source ${USER_CACHE_ROOT}/cache_env.sh && set +a || true

if [ -n "\${PBSS_ACCESS_KEY_ID:-}" ]; then
  export AWS_ACCESS_KEY_ID="\${PBSS_ACCESS_KEY_ID}"
  export AWS_SECRET_ACCESS_KEY="\${PBSS_SECRET_ACCESS_KEY}"
  export AWS_ENDPOINT_URL_S3="https://pdx.s8k.io"
  export CC_USE_S3="1"
  export CC_S3_BUCKET="crawl-data"
  export CC_S3_KEY_PREFIX="crawl-data/"
fi

echo "=== Dripper CPU-only Pipeline ===" && hostname && date

# Clear stale bytecode immediately before execution so synced .py files are used.
find ${SHARED_CODE}/nemo_curator -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

cd ${SHARED_CODE}
srun --ntasks-per-node=1 \
  ${SHARED_VENV}/bin/python \
    tutorials/text/dripper-common-crawl/pipeline_cpu_only.py \
    --slurm \
    ${CLUSTER_ONLY:+--cluster-only} \
    ${PLAN_ONLY:+--plan-only} \
    ${PREPROCESS_ONLY:+--preprocess-only} \
    ${CLUSTER_FROM_PREPROCESSED:+--cluster-from-preprocessed} \
    ${BROADCAST_PROPAGATE_ONLY:+--broadcast-propagate-only} \
    ${TEMPLATE_TABLE_PATH:+--template-table-path ${TEMPLATE_TABLE_PATH}} \
    --manifest-path ${MANIFEST_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --max-rows ${MAX_ROWS} \
    --min-rows-per-batch ${MIN_ROWS_PER_BATCH} \
    ${MAX_ROWS_PER_BATCH:+--max-rows-per-batch ${MAX_ROWS_PER_BATCH}} \
    --output-shards ${OUTPUT_SHARDS} \
    ${MAX_PROPAGATION_GROUP_PAGES:+--layout-template-max-propagation-group-pages ${MAX_PROPAGATION_GROUP_PAGES}} \
    --warc-max-workers ${WARC_MAX_WORKERS} \
    ${OBJECT_STORE_MEMORY_GB:+--object-store-memory-gb ${OBJECT_STORE_MEMORY_GB}} \
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
    --layout-template-propagation-concurrency ${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY} \
    --dynamic-classid-similarity-threshold ${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD} \
    --ray-temp-dir ${RAY_TEMP_BASE}/ray_\${SLURM_JOB_ID} \
    --ray-num-cpus ${CPUS_PER_TASK} \
    ${RAY_NUM_GPUS:+--ray-num-gpus ${RAY_NUM_GPUS}} \
    ${CLUSTER_GPUS:+--cluster-gpus ${CLUSTER_GPUS}} \
    --ray-port 6379 \
    ${USE_S3:+--use-s3} \
    ${WORKER_COUNT:+--worker-count ${WORKER_COUNT}}
JOBSCRIPT

sbatch --parsable \
  --job-name=dripper-cpu \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  ${QOS:+--qos="${QOS}"} \
  --nodes="${NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  ${GPU_ARGS} \
  --time="${TIME_LIMIT}" \
  ${DEPENDENCY:+--dependency=${DEPENDENCY}} \
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
