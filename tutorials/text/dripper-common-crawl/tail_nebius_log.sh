#!/usr/bin/env bash
# Tail the log for one or more Nebius Slurm job IDs.
#
# Usage:
#   bash tail_nebius_log.sh HOST JOB_ID [JOB_ID ...] [-n LINES]
#
# Options:
#   -n LINES   Number of lines to tail (default: 100)
#   -f         Follow mode (tail -f); only works with a single job ID
#
# Examples:
#   bash tail_nebius_log.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346472
#   bash tail_nebius_log.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346472 346473 -n 200
#   bash tail_nebius_log.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346472 -f

set -euo pipefail

HOST="${1:?Usage: $0 <host> <job_id> [job_id ...] [-n LINES] [-f]}"
shift

JOB_IDS=()
LINES=100
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) LINES="${2:?-n requires a number}"; shift 2 ;;
        -f) FOLLOW=true; shift ;;
        *)  JOB_IDS+=("$1"); shift ;;
    esac
done

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "Error: at least one job ID required"
    exit 1
fi

LOG_DIR="/lustre/fsw/portfolios/llmservice/users/\$(whoami)/nemo_curator_shared/logs"

if $FOLLOW; then
    if [[ ${#JOB_IDS[@]} -gt 1 ]]; then
        echo "Warning: -f only works with a single job ID; using ${JOB_IDS[0]}"
    fi
    JOB="${JOB_IDS[0]}"
    ssh "${HOST}" "tail -f ${LOG_DIR}/dripper_streaming_${JOB}.log"
else
    ssh "${HOST}" bash <<REMOTE
for JOB in ${JOB_IDS[*]}; do
    LOG="${LOG_DIR}/dripper_streaming_\${JOB}.log"
    echo "=== Job \${JOB}: \${LOG} ==="
    tail -${LINES} "\${LOG}" 2>/dev/null || echo "(log not found: \${LOG})"
    echo ""
done
REMOTE
fi
