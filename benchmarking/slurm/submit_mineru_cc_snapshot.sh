#!/bin/bash
# Submit every manifest work unit plus one afterok snapshot verifier. Arrays are
# offset because Slurm task IDs cannot exceed MaxArraySize - 1.

set -euo pipefail

required=(CURATOR_DIR MINERU_RESULTS_ROOT MINERU_WORK_UNIT_MANIFEST MINERU_MODEL_CACHE MINERU_SNAPSHOT_SUCCESS_PATH)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
done

units="$(awk 'NF { count++ } END { print count + 0 }' "${MINERU_WORK_UNIT_MANIFEST}")"
if (( units < 1 )); then
    echo "ERROR: manifest contains no work units" >&2
    exit 2
fi
cluster_max="$(scontrol show config | awk '$1 == "MaxArraySize" { print $3; exit }')"
array_size="${MINERU_MAX_ARRAY_SIZE:-${cluster_max}}"
max_nodes="${MINERU_MAX_GPU_NODES:-32}"
for value in "${units}" "${cluster_max}" "${array_size}" "${max_nodes}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Slurm limits and launcher sizes must be integers" >&2
        exit 2
    fi
done
if (( array_size < 1 || array_size > cluster_max || max_nodes < 1 )); then
    echo "ERROR: require 1 <= MINERU_MAX_ARRAY_SIZE <= ${cluster_max} and MINERU_MAX_GPU_NODES >= 1" >&2
    exit 2
fi
chunks="$(( (units + array_size - 1) / array_size ))"
if (( chunks > max_nodes )); then
    echo "ERROR: ${chunks} arrays cannot share a ${max_nodes}-node cap; increase --target-rows or MINERU_MAX_GPU_NODES" >&2
    exit 2
fi
nodes_per_array="$(( max_nodes / chunks ))"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
array_jobs=()
for (( offset = 0; offset < units; offset += array_size )); do
    count="$(( units - offset ))"
    (( count > array_size )) && count="${array_size}"
    job_id="$(sbatch --parsable \
        --array="0-$(( count - 1 ))%${nodes_per_array}" \
        --export="ALL,MINERU_WORK_UNIT_OFFSET=${offset}" \
        "${script_dir}/mineru_cc_work_unit.sbatch")"
    array_jobs+=("${job_id%%;*}")
done

dependency="$(IFS=:; echo "${array_jobs[*]}")"
verify_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
    "${script_dir}/mineru_cc_snapshot_verify.sbatch")"
printf 'array_jobs=%s\nverify_job=%s\n' "${array_jobs[*]}" "${verify_job%%;*}"
