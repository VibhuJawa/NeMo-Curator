#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
set +x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmarking/scripts/gpu_lance_slurm_preflight.sh
source "${SCRIPT_DIR}/gpu_lance_slurm_preflight.sh"

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  gpu_lance_fail "DRY_RUN must be 0 or 1"
fi
gpu_lance_reject_scheduler_replay
unset RAY_ADDRESS
if [[ "${DRY_RUN}" != "1" ]]; then
  gpu_lance_validate_live_slurm_allocation
fi

REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_bin_parent="${PYTHON_BIN%/*}"
  [[ -n "${python_bin_parent}" ]] || python_bin_parent="/"
  if ! python_bin_dir="$(cd -- "${python_bin_parent}" && pwd -P)"; then
    gpu_lance_fail "PYTHON_BIN directory does not exist: ${python_bin_parent}"
  fi
  export PATH="${python_bin_dir}${PATH:+:${PATH}}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmarking/results/gpu_lance_column_fetch/saturation}"

NODES="${NODES:-${SLURM_NNODES:-}}"
WAVES="${WAVES:-8}"
ARM="${ARM:-lance_ray_gpu_actor}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_CPUS_PER_NODE="${RAY_CPUS_PER_NODE:-64}"
LANCE_CPU_THREADS="${LANCE_CPU_THREADS:-32}"
LANCE_IO_THREADS="${LANCE_IO_THREADS:-64}"
IO_THREADS_PER_ACTOR="${IO_THREADS_PER_ACTOR:-4}"
FETCH_BATCH_SIZE="${FETCH_BATCH_SIZE:-1024}"
MAX_PENDING_FETCH_BATCHES="${MAX_PENDING_FETCH_BATCHES:-16}"
MAX_LOOKUP_BYTES_MIB="${MAX_LOOKUP_BYTES_MIB:-256}"
PAYLOAD_PROJECTION="${PAYLOAD_PROJECTION:-image_only}"
WARMUP_COUNT="${WARMUP_COUNT:-1}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"
TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-5}"
STORAGE_AXIS="${STORAGE_AXIS:-remote_s3}"

gpu_lance_require_positive_integer "NODES" "${NODES}"
case "${NODES}" in
  1|2|4|8) ;;
  *) gpu_lance_fail "NODES must be 1, 2, 4, or 8 to match a generated saturation preset" ;;
esac
if [[ "${DRY_RUN}" != "1" && "${SLURM_NNODES}" != "${NODES}" ]]; then
  gpu_lance_fail "NODES=${NODES} does not match SLURM_NNODES=${SLURM_NNODES}"
fi
case "${WAVES}" in
  1|2|4|8) ;;
  *) gpu_lance_fail "WAVES must be 1, 2, 4, or 8" ;;
esac
case "${ARM}" in
  lance_ray_gpu_actor|ray_data_persistent_gpu_actor) ;;
  *) gpu_lance_fail "ARM must be lance_ray_gpu_actor or ray_data_persistent_gpu_actor" ;;
esac
for pair in \
  "CPUS_PER_NODE:${CPUS_PER_NODE}" \
  "GPUS_PER_NODE:${GPUS_PER_NODE}" \
  "RAY_CPUS_PER_NODE:${RAY_CPUS_PER_NODE}" \
  "LANCE_CPU_THREADS:${LANCE_CPU_THREADS}" \
  "LANCE_IO_THREADS:${LANCE_IO_THREADS}" \
  "IO_THREADS_PER_ACTOR:${IO_THREADS_PER_ACTOR}" \
  "FETCH_BATCH_SIZE:${FETCH_BATCH_SIZE}" \
  "MAX_PENDING_FETCH_BATCHES:${MAX_PENDING_FETCH_BATCHES}" \
  "MAX_LOOKUP_BYTES_MIB:${MAX_LOOKUP_BYTES_MIB}" \
  "REPEAT_COUNT:${REPEAT_COUNT}"; do
  gpu_lance_require_positive_integer "${pair%%:*}" "${pair#*:}"
done
gpu_lance_require_nonnegative_integer "WARMUP_COUNT" "${WARMUP_COUNT}"
if (( REPEAT_COUNT < 2 )); then
  gpu_lance_fail "REPEAT_COUNT must be at least 2"
fi
if [[ "${GPUS_PER_NODE}" != "8" ]]; then
  gpu_lance_fail "GPUS_PER_NODE must be 8 for the fixed eight-actor saturation geometry"
fi
case "${PAYLOAD_PROJECTION}" in
  image_only|image_url|full) ;;
  *) gpu_lance_fail "PAYLOAD_PROJECTION must be image_only, image_url, or full" ;;
esac
case "${STORAGE_AXIS}" in
  remote_s3|lustre|node_local_nvme) ;;
  *) gpu_lance_fail "STORAGE_AXIS must be remote_s3, lustre, or node_local_nvme" ;;
esac
gpu_lance_require_positive_number "TELEMETRY_INTERVAL_SECONDS" "${TELEMETRY_INTERVAL_SECONDS}"
COPY_REFERENCE_TO_NODE_LOCAL="${COPY_REFERENCE_TO_NODE_LOCAL:-0}"
if [[ "${COPY_REFERENCE_TO_NODE_LOCAL}" != "0" && "${COPY_REFERENCE_TO_NODE_LOCAL}" != "1" ]]; then
  gpu_lance_fail "COPY_REFERENCE_TO_NODE_LOCAL must be 0 or 1"
fi
if [[ "${COPY_REFERENCE_TO_NODE_LOCAL}" == "1" ]]; then
  gpu_lance_require_nonempty "REFERENCE_NODE_LOCAL_ROOT" "${REFERENCE_NODE_LOCAL_ROOT:-}"
fi

gpu_lance_require_nonempty "MANIFEST_DIR" "${MANIFEST_DIR:-}"
if [[ ! -f "${MANIFEST_DIR}/manifest.json" || ! -f "${MANIFEST_DIR}/manifest.parquet" ]]; then
  gpu_lance_fail "MANIFEST_DIR is missing manifest.json or manifest.parquet: ${MANIFEST_DIR}"
fi
gpu_lance_require_nonempty "IMAGE_LANCE_URI" "${IMAGE_LANCE_URI:-}"
gpu_lance_require_positive_integer "IMAGE_LANCE_VERSION" "${IMAGE_LANCE_VERSION:-}"
gpu_lance_require_nonempty "REFERENCE_GLOB" "${REFERENCE_GLOB:-}"
gpu_lance_require_nonempty "REFERENCE_MANIFEST_URI" "${REFERENCE_MANIFEST_URI:-}"
gpu_lance_require_sha256 "REFERENCE_MANIFEST_SHA256" "${REFERENCE_MANIFEST_SHA256:-}"
gpu_lance_require_positive_integer "EXPECTED_REFERENCE_ROWS" "${EXPECTED_REFERENCE_ROWS:-}"
gpu_lance_validate_storage_options_json "${PYTHON_BIN}" "STORAGE_OPTIONS_JSON" "${STORAGE_OPTIONS_JSON:-}"
gpu_lance_validate_storage_options_json \
  "${PYTHON_BIN}" "REFERENCE_STORAGE_OPTIONS_JSON" "${REFERENCE_STORAGE_OPTIONS_JSON:-}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-dry-run}}"
gpu_lance_require_path_component "RUN_ID" "${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}_${NODES}n_${WAVES}w_${ARM}}"
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  gpu_lance_fail "refusing to reuse an existing saturation output directory: ${OUTPUT_DIR}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  MINIMUM_REMAINING_SLURM_SECONDS="${MINIMUM_REMAINING_SLURM_SECONDS:-}"
  ALLOCATION_END_EPOCH="${ALLOCATION_END_EPOCH:-${SLURM_JOB_END_TIME:-}}"
  gpu_lance_require_positive_integer \
    "MINIMUM_REMAINING_SLURM_SECONDS" "${MINIMUM_REMAINING_SLURM_SECONDS}"
  gpu_lance_require_positive_integer "ALLOCATION_END_EPOCH" "${ALLOCATION_END_EPOCH}"
  remaining_seconds=$((ALLOCATION_END_EPOCH - $(date +%s)))
  if (( remaining_seconds < MINIMUM_REMAINING_SLURM_SECONDS )); then
    gpu_lance_fail \
      "Slurm allocation has ${remaining_seconds}s remaining; requires at least ${MINIMUM_REMAINING_SLURM_SECONDS}s"
  fi
fi

export PYTHONPATH="${REPO_ROOT}"
export LANCE_CPU_THREADS LANCE_IO_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

runner_args=(
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarking/scripts/gpu_lance_saturation_runner.py"
  --manifest-dir "${MANIFEST_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --nodes "${NODES}"
  --waves "${WAVES}"
  --arm "${ARM}"
  --image-lance-uri "${IMAGE_LANCE_URI}"
  --image-lance-version "${IMAGE_LANCE_VERSION}"
  --storage-options-json "${STORAGE_OPTIONS_JSON}"
  --reference-storage-options-json "${REFERENCE_STORAGE_OPTIONS_JSON}"
  --reference-manifest-uri "${REFERENCE_MANIFEST_URI}"
  --reference-manifest-sha256 "${REFERENCE_MANIFEST_SHA256}"
  --reference-glob "${REFERENCE_GLOB}"
  --expected-reference-rows "${EXPECTED_REFERENCE_ROWS}"
  --ray-cpus-per-node "${RAY_CPUS_PER_NODE}"
  --lance-cpu-threads "${LANCE_CPU_THREADS}"
  --lance-io-threads "${LANCE_IO_THREADS}"
  --io-threads-per-actor "${IO_THREADS_PER_ACTOR}"
  --fetch-batch-size "${FETCH_BATCH_SIZE}"
  --max-pending-fetch-batches "${MAX_PENDING_FETCH_BATCHES}"
  --max-lookup-bytes-mib "${MAX_LOOKUP_BYTES_MIB}"
  --payload-projection "${PAYLOAD_PROJECTION}"
  --warmup-count "${WARMUP_COUNT}"
  --repeat-count "${REPEAT_COUNT}"
  --telemetry-interval-seconds "${TELEMETRY_INTERVAL_SECONDS}"
  --storage-axis "${STORAGE_AXIS}"
  --filesystem-path "${MANIFEST_DIR}"
  --filesystem-path "${OUTPUT_DIR}"
)

if [[ "${COPY_REFERENCE_TO_NODE_LOCAL}" == "1" ]]; then
  runner_args+=(
    --copy-reference-to-node-local
    --reference-node-local-root "${REFERENCE_NODE_LOCAL_ROOT}"
    --filesystem-path "${REFERENCE_NODE_LOCAL_ROOT}"
  )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  exec "${runner_args[@]}" --dry-run
fi

runner_args+=(
  --minimum-remaining-slurm-seconds "${MINIMUM_REMAINING_SLURM_SECONDS}"
  --allocation-end-epoch "${ALLOCATION_END_EPOCH}"
)
exec srun \
  --nodes="${NODES}" \
  --ntasks="${NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_NODE}" \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --exclusive \
  --kill-on-bad-exit=1 \
  "${runner_args[@]}"
