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

gpu_lance_reject_scheduler_replay
gpu_lance_validate_live_slurm_allocation
unset RAY_ADDRESS

REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCALE_NODES="${SCALE_NODES:-${SLURM_NNODES}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-1}"
gpu_lance_require_positive_integer "SCALE_NODES" "${SCALE_NODES}"
gpu_lance_require_positive_integer "TASKS_PER_NODE" "${TASKS_PER_NODE}"
if [[ "${SCALE_NODES}" != "${SLURM_NNODES}" ]]; then
  gpu_lance_fail "SCALE_NODES=${SCALE_NODES} does not match SLURM_NNODES=${SLURM_NNODES}"
fi
expected_ranks=$((SCALE_NODES * TASKS_PER_NODE))
SCALE_RANKS="${SCALE_RANKS:-${expected_ranks}}"
gpu_lance_require_positive_integer "SCALE_RANKS" "${SCALE_RANKS}"
if (( SCALE_RANKS != expected_ranks )); then
  gpu_lance_fail "SCALE_RANKS must equal SCALE_NODES * TASKS_PER_NODE (${expected_ranks})"
fi

RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
gpu_lance_require_path_component "RUN_ID" "${RUN_ID}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-64}}"
GPUS_PER_TASK="${GPUS_PER_TASK:-0}"
BENCHMARK_ARM="${BENCHMARK_ARM:-gpu_lance_column_fetch_stage}"
gpu_lance_validate_benchmark_arm "${BENCHMARK_ARM}"
gpu_lance_require_positive_integer "CPUS_PER_TASK" "${CPUS_PER_TASK}"
gpu_lance_require_nonnegative_integer "GPUS_PER_TASK" "${GPUS_PER_TASK}"
if gpu_lance_arm_uses_gpu "${BENCHMARK_ARM}" && (( GPUS_PER_TASK <= 0 )); then
  gpu_lance_fail "GPUS_PER_TASK must be positive for GPU benchmark arm ${BENCHMARK_ARM}"
fi

MANIFEST_ROOT="${MANIFEST_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmarking/results/gpu_lance_column_fetch/scaling}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
LOG_DIR="${LOG_ROOT}/${RUN_ID}"
IMAGE_LANCE_URI="${IMAGE_LANCE_URI:-}"
IMAGE_LANCE_VERSION="${IMAGE_LANCE_VERSION:-}"
STORAGE_OPTIONS_JSON="${STORAGE_OPTIONS_JSON:-}"

gpu_lance_require_nonempty "MANIFEST_ROOT" "${MANIFEST_ROOT}"
gpu_lance_require_nonempty "IMAGE_LANCE_URI" "${IMAGE_LANCE_URI}"
gpu_lance_require_positive_integer "IMAGE_LANCE_VERSION" "${IMAGE_LANCE_VERSION}"
gpu_lance_validate_storage_options_json "${PYTHON_BIN}" "STORAGE_OPTIONS_JSON" "${STORAGE_OPTIONS_JSON}"

if gpu_lance_arm_uses_gpu "${BENCHMARK_ARM}"; then
  gpu_lance_require_nonempty "REFERENCE_GLOB" "${REFERENCE_GLOB:-}"
  gpu_lance_require_nonempty "REFERENCE_MANIFEST_URI" "${REFERENCE_MANIFEST_URI:-}"
  gpu_lance_require_sha256 "REFERENCE_MANIFEST_SHA256" "${REFERENCE_MANIFEST_SHA256:-}"
  gpu_lance_require_positive_integer "EXPECTED_REFERENCE_ROWS" "${EXPECTED_REFERENCE_ROWS:-}"
  gpu_lance_validate_storage_options_json \
    "${PYTHON_BIN}" "REFERENCE_STORAGE_OPTIONS_JSON" "${REFERENCE_STORAGE_OPTIONS_JSON:-}"
fi

TASK_ROWS="${TASK_ROWS:-256}"
TOTAL_LEFT_TABLES="${TOTAL_LEFT_TABLES:-64}"
FETCH_BATCH_SIZE="${FETCH_BATCH_SIZE:-1024}"
IO_THREADS="${IO_THREADS:-16}"
MAX_PENDING_FETCH_BATCHES="${MAX_PENDING_FETCH_BATCHES:-16}"
WARMUP_COUNT="${WARMUP_COUNT:-1}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
for pair in \
  "TASK_ROWS:${TASK_ROWS}" \
  "TOTAL_LEFT_TABLES:${TOTAL_LEFT_TABLES}" \
  "FETCH_BATCH_SIZE:${FETCH_BATCH_SIZE}" \
  "IO_THREADS:${IO_THREADS}" \
  "MAX_PENDING_FETCH_BATCHES:${MAX_PENDING_FETCH_BATCHES}" \
  "REPEAT_COUNT:${REPEAT_COUNT}"; do
  gpu_lance_require_positive_integer "${pair%%:*}" "${pair#*:}"
done
gpu_lance_require_nonnegative_integer "WARMUP_COUNT" "${WARMUP_COUNT}"
if (( REPEAT_COUNT < 2 )); then
  gpu_lance_fail "REPEAT_COUNT must be at least 2"
fi
if (( TOTAL_LEFT_TABLES % SCALE_RANKS != 0 )); then
  gpu_lance_fail "TOTAL_LEFT_TABLES must be divisible by SCALE_RANKS"
fi
COALESCE_TASKS="${COALESCE_TASKS:-$((TOTAL_LEFT_TABLES / SCALE_RANKS))}"
gpu_lance_require_positive_integer "COALESCE_TASKS" "${COALESCE_TASKS}"
LANCE_CPU_THREADS="${LANCE_CPU_THREADS:-32}"
LANCE_IO_THREADS="${LANCE_IO_THREADS:-64}"
MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
gpu_lance_require_positive_integer "LANCE_CPU_THREADS" "${LANCE_CPU_THREADS}"
gpu_lance_require_positive_integer "LANCE_IO_THREADS" "${LANCE_IO_THREADS}"
gpu_lance_require_positive_integer "MALLOC_ARENA_MAX" "${MALLOC_ARENA_MAX}"

output_dir="${OUTPUT_ROOT}/${BENCHMARK_ARM}/${SCALE_NODES}_nodes_${SCALE_RANKS}_ranks/${RUN_ID}"
if [[ -e "${LOG_DIR}" || -L "${LOG_DIR}" ]]; then
  gpu_lance_fail "refusing to reuse an existing scaling log directory: ${LOG_DIR}"
fi
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  gpu_lance_fail "refusing to reuse an existing scaling output directory: ${output_dir}"
fi
for ((rank = 0; rank < SCALE_RANKS; rank++)); do
  manifest="${MANIFEST_ROOT}/shards_${SCALE_RANKS}/rank_$(printf '%02d' "${rank}").parquet"
  if [[ ! -f "${manifest}" ]]; then
    gpu_lance_fail "missing rank manifest before srun: ${manifest}"
  fi
done

export REPO_ROOT PYTHON_BIN SCALE_NODES SCALE_RANKS RUN_ID BENCHMARK_ARM
export MANIFEST_ROOT OUTPUT_ROOT IMAGE_LANCE_URI IMAGE_LANCE_VERSION STORAGE_OPTIONS_JSON
export TASK_ROWS TOTAL_LEFT_TABLES COALESCE_TASKS FETCH_BATCH_SIZE IO_THREADS MAX_PENDING_FETCH_BATCHES
export WARMUP_COUNT REPEAT_COUNT
export LANCE_CPU_THREADS LANCE_IO_THREADS MALLOC_ARENA_MAX
if gpu_lance_arm_uses_gpu "${BENCHMARK_ARM}"; then
  export REFERENCE_GLOB REFERENCE_MANIFEST_URI REFERENCE_MANIFEST_SHA256
  export EXPECTED_REFERENCE_ROWS REFERENCE_STORAGE_OPTIONS_JSON
fi

mkdir -p "${LOG_DIR}"
srun_args=(
  srun
  --nodes="${SCALE_NODES}"
  --ntasks="${SCALE_RANKS}"
  --ntasks-per-node="${TASKS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --exclusive
  --kill-on-bad-exit=1
  --output="${LOG_DIR}/rank_%t.out"
  --error="${LOG_DIR}/rank_%t.err"
)
if (( GPUS_PER_TASK > 0 )); then
  srun_args+=(--gpus-per-task="${GPUS_PER_TASK}")
fi
srun_args+=(bash "${SCRIPT_DIR}/run_gpu_lance_scaling_rank.sh")

exec "${srun_args[@]}"
