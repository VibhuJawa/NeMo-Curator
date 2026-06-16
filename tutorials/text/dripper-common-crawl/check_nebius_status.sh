#!/usr/bin/env bash
# Show queue status + last log lines + metrics for one or more Nebius Slurm jobs.
#
# Usage:
#   bash check_nebius_status.sh HOST JOB_ID [JOB_ID ...] [-n LINES]
#
# Examples:
#   bash check_nebius_status.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346476 346477 346479
#   bash check_nebius_status.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346479 -n 50

set -euo pipefail

HOST="${1:?Usage: $0 <host> <job_id> [job_id ...] [-n LINES]}"
shift

JOB_IDS=()
LINES=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) LINES="${2:?-n requires a number}"; shift 2 ;;
        *)  JOB_IDS+=("$1"); shift ;;
    esac
done

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "Error: at least one job ID required"
    exit 1
fi

ssh "${HOST}" bash <<REMOTE
LOG_DIR="/lustre/fsw/portfolios/llmservice/users/\$(whoami)/nemo_curator_shared/logs"

echo "=== My Jobs ==="
squeue -u "\$(whoami)" --format='%.8i %.20j %.8T %.10M %.10L %R %N' 2>/dev/null || echo "(squeue failed)"
echo ""
echo "=== Cluster (running jobs soonest to finish) ==="
squeue --states=RUNNING --format='%.8i %.10u %.8T %.10M %.10L %R' --sort=L 2>/dev/null | head -15 || echo "(squeue failed)"
echo ""

for JOB in ${JOB_IDS[*]}; do
    echo "======================================================"
    echo "  Job \${JOB}"
    echo "======================================================"

    # Find the log — try streaming, then cpu patterns
    LOG=""
    for CANDIDATE in \
        "\${LOG_DIR}/dripper_streaming_\${JOB}.log" \
        "\${LOG_DIR}/dripper_cpu_\${JOB}.log" \
        "\${LOG_DIR}/dripper_\${JOB}.log"; do
        if [[ -f "\${CANDIDATE}" ]]; then
            LOG="\${CANDIDATE}"
            break
        fi
    done

    if [[ -z "\${LOG}" ]]; then
        echo "  Log : not found (checked streaming/cpu/generic patterns)"
    else
        echo "  Log : \${LOG}"
        echo ""
        tail -${LINES} "\${LOG}"
    fi

    # metrics.json — search common output dirs
    echo ""
    echo "  --- metrics.json ---"
    find /lustre/fsw/portfolios/llmservice/users/\$(whoami) \
        -maxdepth 4 -name "metrics.json" \
        -newer "\${LOG_DIR}/dripper_streaming_346000.log" \
        2>/dev/null | head -10 | while read -r M; do
        echo "  \${M}:"
        cat "\${M}"
        echo ""
    done 2>/dev/null || true
    echo ""
done
REMOTE
