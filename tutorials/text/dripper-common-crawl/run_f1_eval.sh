#!/usr/bin/env bash
# Run F1 evaluation against standalone baseline on a remote Nebius node.
#
# Usage:
#   bash run_f1_eval.sh HOST PIPELINE_OUTPUT_DIR [SAMPLE]
#
# Example:
#   bash run_f1_eval.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../dripper_streaming_f1_comparison_20260616
set -euo pipefail

HOST="${1:?usage: $0 <host> <pipeline_output_dir> [sample]}"
PIPELINE_OUTPUT="${2:?}"
SAMPLE="${3:-0}"

REMOTE_USER="${REMOTE_USER:-$(echo "${HOST}" | cut -d@ -f1)}"
SHARED_CODE="${SHARED_CODE:-/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared}"
BASELINE="${BASELINE:-/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/pipeline_standalone_output_03/baseline_merged.parquet}"
OUT_CSV="${OUT_CSV:-${PIPELINE_OUTPUT}/f1_scores.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Syncing eval script to ${HOST} ==="
rsync -az "${SCRIPT_DIR}/eval_f1_vs_standalone.py" \
  "${HOST}:${SHARED_CODE}/tutorials/text/dripper-common-crawl/"

echo "=== Running F1 evaluation on ${HOST} ==="
echo "  Pipeline output : ${PIPELINE_OUTPUT}"
echo "  Baseline        : ${BASELINE}"
echo "  Out CSV         : ${OUT_CSV}"
echo "  Sample          : ${SAMPLE} (0=all)"
echo ""

ssh "${HOST}" bash <<REMOTE
set -euo pipefail
VENV="${SHARED_CODE}/.venv/bin/python3"
"\${VENV}" "${SHARED_CODE}/tutorials/text/dripper-common-crawl/eval_f1_vs_standalone.py" \
  --pipeline-output "${PIPELINE_OUTPUT}" \
  --baseline        "${BASELINE}" \
  --out-csv         "${OUT_CSV}" \
  --sample          "${SAMPLE}"
REMOTE
