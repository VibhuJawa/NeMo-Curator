#!/bin/bash
# One-node Ray cluster per logical Slurm-array shard. Pass site-specific
# --account and --partition options to sbatch rather than hard-coding them here.

#SBATCH --job-name=image-lance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=image_lance_%A_%a.log
#SBATCH --error=image_lance_%A_%a.log

set -euo pipefail

: "${MANIFEST_DIR:?Set MANIFEST_DIR to the frozen manifest directory}"
: "${DATASET_URI:?Set DATASET_URI to the output Lance URI}"
: "${LANCE_COMMIT_PATH:?Set LANCE_COMMIT_PATH to shared fragment checkpoint storage}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to shared Curator checkpoint storage}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch --array}"
: "${SLURM_ARRAY_TASK_COUNT:?Submit this script with sbatch --array}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="${CURATOR_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SOURCE_STORAGE_OPTIONS="${SOURCE_STORAGE_OPTIONS:-{}}"
LANCE_STORAGE_OPTIONS="${LANCE_STORAGE_OPTIONS:-{}}"
CPUS="${SLURM_CPUS_PER_TASK:-32}"
SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET:-0}"
SHARD_INDEX="${SHARD_INDEX:-$((SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET))}"
TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT}}"
MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX:-0}"

export NEMO_CURATOR_SLURM_ARRAY_ENABLED=1
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${SHARD_INDEX}"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="${TOTAL_SHARDS}"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}"
export RAY_TMPDIR="/tmp/ray_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTHONUNBUFFERED=1

cd "${CURATOR_DIR}"
srun --ntasks=1 python -m tutorials.image.lance_writer.pipeline ingest \
    --manifest-dir "${MANIFEST_DIR}" \
    --dataset-uri "${DATASET_URI}" \
    --lance-commit-path "${LANCE_COMMIT_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --source-storage-options "${SOURCE_STORAGE_OPTIONS}" \
    --lance-storage-options "${LANCE_STORAGE_OPTIONS}" \
    --cpus "${CPUS}" \
    --ray-temp-dir "${RAY_TMPDIR}"
