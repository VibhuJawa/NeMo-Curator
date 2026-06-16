#!/usr/bin/env bash
# Cancel one or more Nebius Slurm jobs.
#
# Usage:
#   bash cancel_nebius_jobs.sh HOST JOB_ID [JOB_ID ...]
#
# Example:
#   bash cancel_nebius_jobs.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346475 346476

set -euo pipefail

HOST="${1:?Usage: $0 <host> <job_id> [job_id ...]}"
shift
JOB_IDS=("$@")

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "Error: at least one job ID required"
    exit 1
fi

echo "=== Cancelling jobs on ${HOST}: ${JOB_IDS[*]} ==="
ssh "${HOST}" "scancel ${JOB_IDS[*]} && echo 'scancel sent'"

echo ""
echo "=== Verifying ==="
ssh "${HOST}" "squeue -j $(IFS=,; echo "${JOB_IDS[*]}") --format='%.10i %.20j %.8T %.10M' 2>/dev/null || echo '(jobs no longer in queue)'"
