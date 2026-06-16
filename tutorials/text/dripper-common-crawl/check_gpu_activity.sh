#!/usr/bin/env bash
# Check GPU activity for one or more Nebius Slurm jobs.
# Uses srun --overlap to run nvidia-smi inside the job's existing allocation,
# so GPU visibility is guaranteed without needing direct node SSH.
# Prints a ACTIVE / IDLE / HUNG verdict per job.
#
# Usage:
#   bash check_gpu_activity.sh HOST JOB_ID [JOB_ID ...]
#
# Examples:
#   bash check_gpu_activity.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346476
#   bash check_gpu_activity.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com 346476 346477 346479

set -euo pipefail

HOST="${1:?Usage: $0 <host> <job_id> [job_id ...]}"
shift
JOB_IDS=("$@")

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "Error: at least one job ID required"
    exit 1
fi

ssh "${HOST}" bash <<REMOTE
for JOB in ${JOB_IDS[*]}; do
    echo "======================================================"
    echo "  Job \${JOB}"
    echo "======================================================"

    # --- state + elapsed time ---
    INFO=\$(squeue -j "\${JOB}" -h --format="%T %M %R %N" 2>/dev/null || true)
    if [[ -z "\${INFO}" ]]; then
        echo "  State     : not in queue (completed or failed)"
        echo ""
        continue
    fi
    STATE=\$(echo "\${INFO}" | awk '{print \$1}')
    ELAPSED=\$(echo "\${INFO}" | awk '{print \$2}')
    REASON=\$(echo "\${INFO}" | awk '{print \$3}')
    NODELIST=\$(echo "\${INFO}" | awk '{print \$4}')
    echo "  State     : \${STATE}  elapsed=\${ELAPSED}  reason=\${REASON}"
    echo "  Node(s)   : \${NODELIST}"

    if [[ "\${STATE}" != "RUNNING" ]]; then
        echo "  (not running — skip GPU check)"
        echo ""
        continue
    fi

    # --- per-GPU utilisation via srun --overlap (runs inside the job's allocation) ---
    echo ""
    echo "  GPU utilisation (srun --overlap inside allocation):"
    GPU_INFO=\$(srun --jobid="\${JOB}" --overlap --ntasks=1 --nodes=1 \
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "UNAVAILABLE")

    if [[ "\${GPU_INFO}" == "UNAVAILABLE" ]]; then
        echo "  (nvidia-smi unavailable — CPU-only job or srun failed)"
        echo "  VERDICT   : N/A (no GPUs or srun error)"
        echo ""
        continue
    fi

    echo "\${GPU_INFO}" | awk -F', ' '
    {
        gpu=\$1; util_gpu=\$3+0; util_mem=\$4+0; mem_used=\$5+0; mem_total=\$6+0; temp=\$7+0
        status = (util_gpu > 5) ? "ACTIVE" : "IDLE"
        printf "  GPU %s  util=%3d%%  mem_util=%3d%%  mem=%dMiB/%dMiB  temp=%dC  [%s]\n", \
               gpu, util_gpu, util_mem, mem_used, mem_total, temp, status
    }'

    # --- GPU compute processes ---
    echo ""
    echo "  GPU compute processes:"
    srun --jobid="\${JOB}" --overlap --ntasks=1 --nodes=1 \
        nvidia-smi --query-compute-apps=pid,used_memory,name \
        --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' 'NR>0{printf "    pid=%-8s mem=%sMiB  %s\n", \$1, \$2, \$3}' || \
    echo "    (none)"

    # --- Hang verdict: 2 samples 3s apart; if both zero => HUNG ---
    echo ""
    S1=\$(srun --jobid="\${JOB}" --overlap --ntasks=1 --nodes=1 \
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk '{s+=\$1} END{printf "%d", s+0}')
    S1=\${S1:-0}

    sleep 3

    S2=\$(srun --jobid="\${JOB}" --overlap --ntasks=1 --nodes=1 \
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk '{s+=\$1} END{printf "%d", s+0}')
    S2=\${S2:-0}

    TOTAL=\$(( S1 + S2 ))
    if [[ \${TOTAL} -gt 0 ]]; then
        echo "  VERDICT   : ACTIVE  (GPU util: \${S1}% -> \${S2}%)"
    else
        echo "  VERDICT   : HUNG / IDLE  (GPU util 0% on both samples)"
    fi
    echo ""
done
REMOTE
