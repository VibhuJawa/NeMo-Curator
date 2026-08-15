#!/bin/bash
# Submit native Curator source shards plus one dependent snapshot verifier.

set -euo pipefail

required=(CURATOR_DIR MINERU_RESULTS_ROOT MINERU_OUTPUT_PATH MINERU_CHECKPOINT_PATH MINERU_MODEL_CACHE MINERU_SNAPSHOT MINERU_WARC_MANIFEST MINERU_SNAPSHOT_SUCCESS_PATH)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required" >&2
        exit 2
    fi
done

total_shards="${MINERU_TOTAL_SHARDS:-1400}"
cluster_max="$(scontrol show config | awk '$1 == "MaxArraySize" { print $3; exit }')"
array_size="${MINERU_MAX_ARRAY_SIZE:-${cluster_max}}"
max_nodes="${MINERU_MAX_GPU_NODES:-32}"
for value in "${total_shards}" "${cluster_max}" "${array_size}" "${max_nodes}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: shard and Slurm limits must be positive integers" >&2
        exit 2
    fi
done
if (( total_shards < 1 || array_size < 1 || array_size > cluster_max || max_nodes < 1 )); then
    echo "ERROR: invalid shard, array, or node limit" >&2
    exit 2
fi
export MINERU_TOTAL_SHARDS="${total_shards}"

chunks="$(( (total_shards + array_size - 1) / array_size ))"
if (( chunks > max_nodes )); then
    echo "ERROR: ${chunks} arrays cannot share a ${max_nodes}-node concurrency cap" >&2
    exit 2
fi
nodes_per_array="$(( max_nodes / chunks ))"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
array_jobs=()
for (( offset = 0; offset < total_shards; offset += array_size )); do
    count="$(( total_shards - offset ))"
    (( count > array_size )) && count="${array_size}"
    job_id="$(sbatch --parsable \
        --array="0-$(( count - 1 ))%${nodes_per_array}" \
        --export="ALL,MINERU_SHARD_OFFSET=${offset},MINERU_TOTAL_SHARDS=${total_shards}" \
        "${script_dir}/mineru_cc_work_unit.sbatch")"
    array_jobs+=("${job_id%%;*}")
done

dependency="$(IFS=:; echo "${array_jobs[*]}")"
verify_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
    "${script_dir}/mineru_cc_snapshot_verify.sbatch")"
printf 'total_shards=%s\narray_jobs=%s\nverify_job=%s\n' \
    "${total_shards}" "${array_jobs[*]}" "${verify_job%%;*}"
