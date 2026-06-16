#!/usr/bin/env bash
# Submit the smoke test + one host shard to Nebius Slurm.
#
# Usage:
#   bash submit_smoke_and_shard.sh HOST [SMOKE_OUTPUT_DIR] [SHARD_OUTPUT_DIR]
#
# All manifest / output paths default to the standard Lustre layout for the
# remote user inferred from HOST (user@host). Override any of them via env vars:
#
#   SMOKE_MANIFEST   SHARD_MANIFEST
#   SMOKE_OUTPUT_DIR SHARD_OUTPUT_DIR
#
# All other env vars (NODES, GPUS_PER_NODE, MODEL_IDENTIFIER, ...) are forwarded
# to submit_nebius_dripper.sh — see that script's header for the full list.
#
# Example:
#   bash submit_smoke_and_shard.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com
#
#   SMOKE_OUTPUT_DIR=/lustre/.../my_smoke \
#   bash submit_smoke_and_shard.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com

set -euo pipefail

HOST="${1:?Usage: $0 <host> [smoke_output_dir] [shard_output_dir]}"

# Derive remote user + base Lustre root from HOST (user@host).
_host_user="$(echo "${HOST}" | cut -d@ -f1)"
REMOTE_USER="${REMOTE_USER:-${_host_user}}"
USER_CACHE_ROOT="${USER_CACHE_ROOT:-/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}}"

SMOKE_MANIFEST="${SMOKE_MANIFEST:-${USER_CACHE_ROOT}/nemo_curator_dripper_layout_cpu_parallel_manifest_20260609}"
SHARD_MANIFEST="${SHARD_MANIFEST:-${USER_CACHE_ROOT}/nemo_curator_dripper_sorted_host_buckets_20260611/shard_0001.parquet}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${2:-${USER_CACHE_ROOT}/dripper_streaming_smoke}}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-${3:-${USER_CACHE_ROOT}/dripper_streaming_shard0001}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_nebius_dripper.sh"

echo "=== Smoke test ==="
echo "  Manifest : ${SMOKE_MANIFEST}"
echo "  Output   : ${SMOKE_OUTPUT_DIR}"
bash "${SUBMIT}" "${HOST}" "${SMOKE_MANIFEST}" "${SMOKE_OUTPUT_DIR}"

echo ""
echo "=== Host shard 0001 ==="
echo "  Manifest : ${SHARD_MANIFEST}"
echo "  Output   : ${SHARD_OUTPUT_DIR}"
bash "${SUBMIT}" "${HOST}" "${SHARD_MANIFEST}" "${SHARD_OUTPUT_DIR}"
