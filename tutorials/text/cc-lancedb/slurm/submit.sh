#!/bin/bash
# =============================================================================
# submit.sh — SLURM batch job for the CC LanceDB pipeline
#
# Usage:
#   sbatch slurm/submit.sh
#
# All parameters can be overridden via environment variables before submission:
#   CC_SNAPSHOT, DOWNLOAD_DIR, LANCEDB_URI, TABLE_NAME, EXTRACTOR,
#   NUM_WORKERS, VENV
#
# PBSS credentials must be set in your shell before calling sbatch:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_ENDPOINT_URL_S3=https://pdx.s8k.io   # optional, this is the default
#
# To override the SLURM account without editing this file:
#   SBATCH_ACCOUNT=my_other_account sbatch slurm/submit.sh
# =============================================================================

#SBATCH --job-name=cc-lancedb
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/cc-lancedb-%j.out
#SBATCH --error=logs/cc-lancedb-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# PBSS / SwiftStack credentials — must be present in the environment
# ---------------------------------------------------------------------------
: "${AWS_ACCESS_KEY_ID:?Need AWS_ACCESS_KEY_ID set before calling sbatch}"
: "${AWS_SECRET_ACCESS_KEY:?Need AWS_SECRET_ACCESS_KEY set before calling sbatch}"
AWS_ENDPOINT_URL_S3=${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
export AWS_ENDPOINT_URL_S3 AWS_DEFAULT_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
SNAPSHOT=${CC_SNAPSHOT:-CC-MAIN-2025-26}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-/tmp/cc_warcs_${SLURM_JOB_ID}}
LANCEDB_URI=${LANCEDB_URI:-s3://pdx-commoncrawl/cc_lancedb}
TABLE_NAME=${TABLE_NAME:-cc_snapshot_index}
EXTRACTOR=${EXTRACTOR:-trafilatura}
VENV=${VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/venvs/nemo_curator}

# Resolve the directory containing this script so pipeline.py can be found
# regardless of where sbatch is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$DOWNLOAD_DIR" logs

echo "Job ID        : ${SLURM_JOB_ID}"
echo "Snapshot      : ${SNAPSHOT}"
echo "Download dir  : ${DOWNLOAD_DIR}"
echo "LanceDB URI   : ${LANCEDB_URI}"
echo "Table name    : ${TABLE_NAME}"
echo "Extractor     : ${EXTRACTOR}"
echo "CPUs per task : ${SLURM_CPUS_PER_TASK}"

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "WARNING: venv not found at ${VENV}; falling back to system Python" >&2
else
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

# ---------------------------------------------------------------------------
# Start Ray head node and ensure it is stopped on exit
# ---------------------------------------------------------------------------
trap 'echo "Stopping Ray..."; ray stop' EXIT

ray start --head --num-cpus="${SLURM_CPUS_PER_TASK}"

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
python "${SCRIPT_DIR}/../pipeline.py" \
    --start-snapshot "${SNAPSHOT}" \
    --end-snapshot   "${SNAPSHOT}" \
    --download-dir   "${DOWNLOAD_DIR}" \
    --lancedb-uri    "${LANCEDB_URI}" \
    --table-name     "${TABLE_NAME}" \
    --extractor-lib  "${EXTRACTOR}"

echo "CC LanceDB pipeline completed successfully for snapshot ${SNAPSHOT}."
