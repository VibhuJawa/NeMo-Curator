#!/bin/bash
# Submit with: sbatch --array=0-$((N-1))%MAX benchmarking/slurm/mineru_cc_work_unit.sbatch
# Required environment: CURATOR_DIR, MINERU_RESULTS_ROOT,
# MINERU_WORK_UNIT_MANIFEST, MINERU_MODEL_CACHE.
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

required=(CURATOR_DIR MINERU_RESULTS_ROOT MINERU_WORK_UNIT_MANIFEST MINERU_MODEL_CACHE SLURM_ARRAY_TASK_ID)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
done

MINERU_WORK_UNIT_INDEX="$(( ${MINERU_WORK_UNIT_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
export MINERU_WORK_UNIT_INDEX
# The immutable manifest already assigns complete input files to this task.
# Disable Curator's generic source-task sharding to avoid partitioning them a
# second time by SLURM_ARRAY_TASK_ID.
export NEMO_CURATOR_SLURM_ARRAY_ENABLED=0
export MINERU_HTML_FIELD="${MINERU_HTML_FIELD:-content}"
export MINERU_HTML_COMPRESSION="${MINERU_HTML_COMPRESSION:-zstd}"
export MINERU_URL_FIELD="${MINERU_URL_FIELD:-url}"
export USE_TORCH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray-${SLURM_JOB_ID}"
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

cd "${CURATOR_DIR}"
"${PYTHON_BIN}" benchmarking/run.py \
    --config benchmarking/mineru-html-benchmark.yaml \
    --strict-config-check \
    --entries-exact mineru_cc_work_unit \
    --session-name "mineru-${SLURM_ARRAY_JOB_ID}-${MINERU_WORK_UNIT_INDEX}" \
    --reason "Common Crawl work unit ${MINERU_WORK_UNIT_INDEX}"
