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

# Shared fail-closed checks for the GPU Lance benchmark launchers. This file is
# sourced by the launchers and intentionally performs no work on its own.

gpu_lance_fail() {
  echo "$1" >&2
  exit 2
}

gpu_lance_reject_scheduler_replay() {
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    gpu_lance_fail \
      "GPU Lance benchmark sweeps must not run as Slurm array elements; use one allocation and run points sequentially"
  fi
  if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
    gpu_lance_fail \
      "GPU Lance benchmarks do not resume requeued jobs; submit with --no-requeue and a fresh RUN_ID"
  fi
}

gpu_lance_require_nonempty() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    gpu_lance_fail "${name} is required"
  fi
}

gpu_lance_require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    gpu_lance_fail "${name} must be a positive integer"
  fi
}

gpu_lance_require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    gpu_lance_fail "${name} must be a nonnegative integer"
  fi
}

gpu_lance_require_positive_number() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ || "${value}" =~ ^0*([.]0*)?$ ]]; then
    gpu_lance_fail "${name} must be a positive number"
  fi
}

gpu_lance_require_path_component() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9_.-]+$ || "${value}" == "." || "${value}" == ".." ]]; then
    gpu_lance_fail "${name} must be a single safe path component"
  fi
}

gpu_lance_require_sha256() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    gpu_lance_fail "${name} must be a lowercase SHA-256 digest"
  fi
}

gpu_lance_arm_uses_gpu() {
  case "$1" in
    gpu_lance_column_fetch_stage|lance_ray_gpu_fetcher|lance_ray_gpu_actor|ray_data_persistent_gpu_actor)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

gpu_lance_validate_benchmark_arm() {
  case "$1" in
    naive_pylance_scalar|cpu_lance_column_fetch_stage|gpu_lance_column_fetch_stage|lance_ray_datasource|lance_ray_gpu_fetcher|lance_ray_gpu_actor|ray_data_persistent_gpu_actor)
      ;;
    *)
      gpu_lance_fail "unsupported GPU Lance benchmark arm: $1"
      ;;
  esac
}

gpu_lance_validate_storage_options_json() {
  local python_bin="$1"
  local label="$2"
  local value="$3"

  gpu_lance_require_nonempty "${label}" "${value}"
  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    gpu_lance_fail "PYTHON_BIN is not executable: ${python_bin}"
  fi
  if ! printf '%s' "${value}" | "${python_bin}" -c '
import json
import sys
from pathlib import Path

label = sys.argv[1]
raw = sys.stdin.read()
try:
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    payload = json.loads(raw)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"{label} must be valid JSON or @path: {type(exc).__name__}") from exc
if not isinstance(payload, dict) or not all(
    isinstance(key, str) and isinstance(item, str) for key, item in payload.items()
):
    raise SystemExit(f"{label} must be a JSON object with string keys and values")
secret_parts = ("access_key", "secret", "token", "password", "credential")
secret_keys = sorted(key for key in payload if any(part in key.casefold() for part in secret_parts))
if secret_keys:
    raise SystemExit(f"{label} contains credential-like keys: {secret_keys}")
' "${label}"; then
    gpu_lance_fail "${label} failed nonsecret JSON validation"
  fi
}

gpu_lance_validate_live_slurm_allocation() {
  local record=""
  local state=""
  local requeue=""
  local oversubscribe=""
  local token=""

  gpu_lance_require_nonempty "SLURM_JOB_ID" "${SLURM_JOB_ID:-}"
  gpu_lance_require_nonempty "SLURM_JOB_NODELIST" "${SLURM_JOB_NODELIST:-}"
  gpu_lance_require_positive_integer "SLURM_NNODES" "${SLURM_NNODES:-}"
  if ! command -v scontrol >/dev/null 2>&1; then
    gpu_lance_fail "scontrol is required to validate the live Slurm allocation"
  fi
  if ! record="$(scontrol show job --oneliner "${SLURM_JOB_ID}" 2>/dev/null)"; then
    gpu_lance_fail "scontrol could not inspect Slurm job ${SLURM_JOB_ID}"
  fi
  for token in ${record}; do
    case "${token}" in
      JobState=*) state="${token#JobState=}" ;;
      Requeue=*) requeue="${token#Requeue=}" ;;
      OverSubscribe=*) oversubscribe="${token#OverSubscribe=}" ;;
    esac
  done
  if [[ "${state}" != "RUNNING" ]]; then
    gpu_lance_fail "Slurm allocation must be RUNNING; observed ${state:-missing JobState}"
  fi
  if [[ "${requeue}" != "0" ]]; then
    gpu_lance_fail "Slurm allocation must be submitted with --no-requeue; observed Requeue=${requeue:-missing}"
  fi
  if [[ "${oversubscribe}" != "NO" ]]; then
    gpu_lance_fail \
      "Slurm allocation must be node-exclusive with OverSubscribe=NO; observed ${oversubscribe:-missing}"
  fi
}
