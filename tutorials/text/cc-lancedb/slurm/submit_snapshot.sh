#!/usr/bin/env bash
# Submit phase-1 array + phase-2 commit for one CC snapshot.
# Usage:
#   bash slurm/submit_snapshot.sh CC-MAIN-2025-26 [total_splits]
#
# Output: prints array job ID and commit job ID.

set -euo pipefail

SNAPSHOT="${1:?Usage: bash submit_snapshot.sh CC-MAIN-2025-26 [total_splits]}"
TOTAL_SPLITS="${2:-40}"
ARRAY_END=$((TOTAL_SPLITS - 1))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting $SNAPSHOT ($TOTAL_SPLITS splits)..."

ARRAY_JOB=$(sbatch --parsable \
    --array="0-${ARRAY_END}" \
    "$SCRIPT_DIR/process_snapshot_array.sh" \
    "$SNAPSHOT" "$TOTAL_SPLITS")

COMMIT_JOB=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB}" \
    "$SCRIPT_DIR/commit_snapshot.sh" \
    "$SNAPSHOT" "$TOTAL_SPLITS")

echo "Snapshot : $SNAPSHOT"
echo "Array    : $ARRAY_JOB  (splits 0-${ARRAY_END})"
echo "Commit   : $COMMIT_JOB (runs after all splits complete)"
echo "Watch    : tail -f /lustre/.../cc_lance_logs/cc-lance-commit-${COMMIT_JOB}.log"
