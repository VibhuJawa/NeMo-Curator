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
MANIFEST_ROOT="${MANIFEST_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmarking/results/gpu_lance_column_fetch/scaling}"

gpu_lance_require_positive_integer "SLURM_NTASKS" "${SLURM_NTASKS:-}"
gpu_lance_require_nonnegative_integer "SLURM_PROCID" "${SLURM_PROCID:-}"
SCALE_NODES="${SCALE_NODES:-${SLURM_NNODES}}"
SCALE_RANKS="${SCALE_RANKS:-${SLURM_NTASKS}}"
SCALE_RANK="${SCALE_RANK:-${SLURM_PROCID}}"
gpu_lance_require_positive_integer "SCALE_NODES" "${SCALE_NODES}"
gpu_lance_require_positive_integer "SCALE_RANKS" "${SCALE_RANKS}"
gpu_lance_require_nonnegative_integer "SCALE_RANK" "${SCALE_RANK}"
if [[ "${SCALE_NODES}" != "${SLURM_NNODES}" ]]; then
  gpu_lance_fail "SCALE_NODES=${SCALE_NODES} does not match SLURM_NNODES=${SLURM_NNODES}"
fi
if [[ "${SCALE_RANKS}" != "${SLURM_NTASKS}" ]]; then
  gpu_lance_fail "SCALE_RANKS=${SCALE_RANKS} does not match SLURM_NTASKS=${SLURM_NTASKS}"
fi
if (( SCALE_RANK >= SCALE_RANKS )); then
  gpu_lance_fail "SCALE_RANK=${SCALE_RANK} is outside 0..$((SCALE_RANKS - 1))"
fi

BENCHMARK_ARM="${BENCHMARK_ARM:-gpu_lance_column_fetch_stage}"
gpu_lance_validate_benchmark_arm "${BENCHMARK_ARM}"
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

gpu_lance_require_nonempty "MANIFEST_ROOT" "${MANIFEST_ROOT}"
gpu_lance_require_nonempty "IMAGE_LANCE_URI" "${IMAGE_LANCE_URI:-}"
gpu_lance_require_positive_integer "IMAGE_LANCE_VERSION" "${IMAGE_LANCE_VERSION:-}"
gpu_lance_validate_storage_options_json "${PYTHON_BIN}" "STORAGE_OPTIONS_JSON" "${STORAGE_OPTIONS_JSON:-}"
if gpu_lance_arm_uses_gpu "${BENCHMARK_ARM}"; then
  gpu_lance_require_nonempty "REFERENCE_GLOB" "${REFERENCE_GLOB:-}"
  gpu_lance_require_nonempty "REFERENCE_MANIFEST_URI" "${REFERENCE_MANIFEST_URI:-}"
  gpu_lance_require_sha256 "REFERENCE_MANIFEST_SHA256" "${REFERENCE_MANIFEST_SHA256:-}"
  gpu_lance_require_positive_integer "EXPECTED_REFERENCE_ROWS" "${EXPECTED_REFERENCE_ROWS:-}"
  gpu_lance_validate_storage_options_json \
    "${PYTHON_BIN}" "REFERENCE_STORAGE_OPTIONS_JSON" "${REFERENCE_STORAGE_OPTIONS_JSON:-}"
fi

MANIFEST="${MANIFEST_ROOT}/shards_${SCALE_RANKS}/rank_$(printf '%02d' "${SCALE_RANK}").parquet"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
gpu_lance_require_path_component "RUN_ID" "${RUN_ID}"
OUTPUT_DIR="${OUTPUT_ROOT}/${BENCHMARK_ARM}/${SCALE_NODES}_nodes_${SCALE_RANKS}_ranks/${RUN_ID}"
OUTPUT="${OUTPUT_DIR}/rank_$(printf '%02d' "${SCALE_RANK}").json"
if [[ ! -f "${MANIFEST}" ]]; then
  gpu_lance_fail "missing rank manifest: ${MANIFEST}"
fi
if [[ -e "${OUTPUT}" || -L "${OUTPUT}" ]]; then
  gpu_lance_fail "refusing to overwrite an existing benchmark artifact: ${OUTPUT}"
fi

export PYTHONPATH="${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

args=(
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarking/scripts/gpu_lance_column_fetch_benchmark.py"
  --query-manifest "${MANIFEST}"
  --image-lance-uri "${IMAGE_LANCE_URI}"
  --image-lance-version "${IMAGE_LANCE_VERSION}"
  --storage-options-json "${STORAGE_OPTIONS_JSON}"
  --task-rows "${TASK_ROWS}"
  --coalesce-tasks "${COALESCE_TASKS}"
  --fetch-batch-size "${FETCH_BATCH_SIZE}"
  --io-threads "${IO_THREADS}"
  --max-pending-fetch-batches "${MAX_PENDING_FETCH_BATCHES}"
  --warmup-count "${WARMUP_COUNT}"
  --repeat-count "${REPEAT_COUNT}"
  --arm "${BENCHMARK_ARM}"
  --output "${OUTPUT}"
)

if gpu_lance_arm_uses_gpu "${BENCHMARK_ARM}"; then
  args+=(
    --reference-glob "${REFERENCE_GLOB}"
    --reference-storage-options-json "${REFERENCE_STORAGE_OPTIONS_JSON}"
    --reference-manifest-uri "${REFERENCE_MANIFEST_URI}"
    --reference-manifest-sha256 "${REFERENCE_MANIFEST_SHA256}"
    --expected-reference-rows "${EXPECTED_REFERENCE_ROWS}"
  )
fi

echo \
  "mode=fixed_global_latency rank=${SCALE_RANK}/${SCALE_RANKS} nodes=${SCALE_NODES} arm=${BENCHMARK_ARM} manifest=${MANIFEST} output=${OUTPUT}"
exec "${args[@]}"
