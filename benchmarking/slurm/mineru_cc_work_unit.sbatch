#!/bin/bash
# One array element owns one deterministic subset of complete Common Crawl WARCs.
# Submit through submit_mineru_cc_snapshot.sh rather than invoking this directly.
#SBATCH --job-name=mineru-cc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=mineru-cc-%A_%a.log
#SBATCH --error=mineru-cc-%A_%a.log

set -euo pipefail
umask 022

required=(CURATOR_DIR MINERU_RESULTS_ROOT MINERU_OUTPUT_PATH MINERU_CHECKPOINT_PATH MINERU_MODEL_CACHE MINERU_SNAPSHOT MINERU_WARC_MANIFEST MINERU_TOTAL_SHARDS SLURM_ARRAY_TASK_ID)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
done

logical_shard="$(( ${MINERU_SHARD_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
export NEMO_CURATOR_SLURM_ARRAY_ENABLED=1
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${logical_shard}"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="${MINERU_TOTAL_SHARDS}"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX=0

export MINERU_HTML_FIELD="${MINERU_HTML_FIELD:-content}"
export MINERU_HTML_COMPRESSION=zstd
export MINERU_URL_FIELD="${MINERU_URL_FIELD:-url}"
export MINERU_DOWNLOAD_WORKERS="${MINERU_DOWNLOAD_WORKERS:-2}"
export MINERU_WARC_RECORDS_PER_BATCH="${MINERU_WARC_RECORDS_PER_BATCH:-1024}"
export MINERU_INFERENCE_WORKERS="${MINERU_INFERENCE_WORKERS:-32}"
export MINERU_SIMPLIFY_WORKERS="${MINERU_SIMPLIFY_WORKERS:-32}"
export MINERU_EXTRACT_WORKERS="${MINERU_EXTRACT_WORKERS:-32}"
export MINERU_OBJECT_STORE_SIZE="${MINERU_OBJECT_STORE_SIZE:-274877906944}"
export MINERU_CC_TRANSPORT="${MINERU_CC_TRANSPORT:-s3}"
export MINERU_CC_S3_BUCKET="${MINERU_CC_S3_BUCKET:-crawl-data}"
export MINERU_CC_S3_KEY_PREFIX="${MINERU_CC_S3_KEY_PREFIX:-crawl-data/}"
export MINERU_CC_S5CMD_CONCURRENCY="${MINERU_CC_S5CMD_CONCURRENCY:-8}"
export MINERU_CC_S5CMD_PART_SIZE_MB="${MINERU_CC_S5CMD_PART_SIZE_MB:-256}"
export MINERU_DM_STORAGE_LOCATION="${MINERU_DM_STORAGE_LOCATION:-pdx-commoncrawl}"
export MINERU_DM_STORAGE_CONFIG="${MINERU_DM_STORAGE_CONFIG:-${HOME}/.config/datamover/storage_locations}"
node_local_base="${MINERU_NODE_LOCAL_BASE:-}"
if [[ -z "${node_local_base}" ]]; then
    if [[ -d /raid/scratch && -w /raid/scratch ]]; then
        node_local_base=/raid/scratch
    else
        node_local_base="${SLURM_TMPDIR:-/tmp}"
    fi
fi
export MINERU_NODE_LOCAL_ROOT="${MINERU_NODE_LOCAL_ROOT:-${node_local_base}/${USER}/mineru-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}}"
export MINERU_DOWNLOAD_DIR="${MINERU_DOWNLOAD_DIR:-${MINERU_NODE_LOCAL_ROOT}/common-crawl-warcs}"
export USE_TORCH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Ray appends long session and socket names; keep this root short enough for
# Linux's 107-byte AF_UNIX path limit.
export RAY_TMPDIR="${MINERU_RAY_TMPDIR:-${node_local_base}/${USER}/r${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}}"
export RAY_PORT_BROADCAST_DIR="${MINERU_CHECKPOINT_PATH}/ray-ports"
mkdir -p "${MINERU_DOWNLOAD_DIR}" "${RAY_TMPDIR}" "${RAY_PORT_BROADCAST_DIR}"
echo "Node-local storage: $(df -h "${MINERU_NODE_LOCAL_ROOT}" | tail -1)"

PYTHON_BIN="${CURATOR_PYTHON:-${CURATOR_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
PYTHON_ENV="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
export PATH="${PYTHON_ENV}/bin:${PATH}"
if [[ -f "${PYTHON_ENV}/pyvenv.cfg" ]]; then
    export VIRTUAL_ENV="${PYTHON_ENV}"
fi
if [[ -n "${CURATOR_DYNAMO_BIN_DIR:-}" ]]; then
    export PATH="${CURATOR_DYNAMO_BIN_DIR}:${PATH}"
fi
for binary in ray etcd nats-server; do
    if ! command -v "${binary}" >/dev/null; then
        echo "ERROR: ${binary} is required on PATH" >&2
        exit 2
    fi
done
if [[ "${MINERU_CC_TRANSPORT}" == "s3" ]] && ! command -v s5cmd >/dev/null; then
    echo "ERROR: s5cmd is required for MINERU_CC_TRANSPORT=s3" >&2
    exit 2
fi
if [[ "${MINERU_CC_TRANSPORT}" == "s3" ]]; then
    if [[ ! -r "${MINERU_DM_STORAGE_CONFIG}" ]]; then
        echo "ERROR: Data Mover storage config is not readable: ${MINERU_DM_STORAGE_CONFIG}" >&2
        exit 2
    fi
    dm_location_value() {
        local field="$1"
        awk -v location="${MINERU_DM_STORAGE_LOCATION}" -v field="${field}" '
            $0 == location ":" { active = 1; next }
            active && /^[^[:space:]]/ { exit }
            active {
                line = $0
                sub(/^[[:space:]]*/, "", line)
                if (index(line, field ":") == 1) {
                    sub(/^[^:]+:[[:space:]]*/, "", line)
                    gsub(/^"|"$/, "", line)
                    print line
                    exit
                }
            }
        ' "${MINERU_DM_STORAGE_CONFIG}"
    }
    export AWS_ACCESS_KEY_ID="$(dm_location_value access_key_id)"
    export AWS_SECRET_ACCESS_KEY="$(dm_location_value secret_access_key)"
    export AWS_DEFAULT_REGION="$(dm_location_value region)"
    export AWS_REGION="${AWS_DEFAULT_REGION}"
    export S3_ENDPOINT_URL="$(dm_location_value endpoint)"
    export AWS_ENDPOINT_URL_S3="${S3_ENDPOINT_URL}"
    export MINERU_CC_S3_ENDPOINT_URL="${S3_ENDPOINT_URL}"
    unset AWS_SESSION_TOKEN
    if [[ -z "${AWS_ACCESS_KEY_ID}" || -z "${AWS_SECRET_ACCESS_KEY}" || -z "${S3_ENDPOINT_URL}" ]]; then
        echo "ERROR: incomplete Data Mover location: ${MINERU_DM_STORAGE_LOCATION}" >&2
        exit 2
    fi

    first_warc="$(awk 'NF { print; exit }' "${MINERU_WARC_MANIFEST}")"
    first_key="${first_warc}"
    if [[ "${first_key}" == *://* ]]; then
        first_key="${first_key#*://}"
        first_key="${first_key#*/}"
    fi
    if [[ "${MINERU_CC_S3_KEY_PREFIX}" != "-" && "${first_key}" == "${MINERU_CC_S3_KEY_PREFIX}"* ]]; then
        first_key="${first_key#"${MINERU_CC_S3_KEY_PREFIX}"}"
    fi
    echo "Common Crawl source: ${MINERU_DM_STORAGE_LOCATION} ${MINERU_CC_S3_ENDPOINT_URL} s3://${MINERU_CC_S3_BUCKET}/${first_key}"
    s5cmd --endpoint-url "${MINERU_CC_S3_ENDPOINT_URL}" ls "s3://${MINERU_CC_S3_BUCKET}/${first_key}" >/dev/null
fi

cd "${CURATOR_DIR}"
srun --ntasks-per-node=1 "${PYTHON_BIN}" benchmarking/run.py \
    --config benchmarking/mineru-html-benchmark.yaml \
    --strict-config-check \
    --entries-exact mineru_cc_work_unit \
    --session-name "mineru-${SLURM_ARRAY_JOB_ID}-${logical_shard}" \
    --reason "Common Crawl native shard ${logical_shard}/${MINERU_TOTAL_SHARDS}"
