#!/bin/bash
# Submitted by submit_mineru_cc_snapshot.sh after every array succeeds.
# Required environment: CURATOR_DIR, MINERU_RESULTS_ROOT,
# MINERU_OUTPUT_PATH, MINERU_CHECKPOINT_PATH, MINERU_SNAPSHOT_SUCCESS_PATH.
#SBATCH --job-name=mineru-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=04:00:00
#SBATCH --output=mineru-verify-%j.log
#SBATCH --error=mineru-verify-%j.log

set -euo pipefail
umask 022

required=(CURATOR_DIR MINERU_RESULTS_ROOT MINERU_OUTPUT_PATH MINERU_CHECKPOINT_PATH MINERU_SNAPSHOT MINERU_WARC_MANIFEST MINERU_SNAPSHOT_SUCCESS_PATH)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
done

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

cd "${CURATOR_DIR}"
"${PYTHON_BIN}" benchmarking/run.py \
    --config benchmarking/mineru-html-snapshot-verify.yaml \
    --strict-config-check \
    --entries-exact mineru_cc_snapshot_verify \
    --session-name "mineru-snapshot-verify-${SLURM_JOB_ID}" \
    --reason "Verify all Common Crawl work units"
