#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/lustre/fsw/portfolios/llmservice/users/vjawa/pr2075_multigpu_200k_$(date +%Y%m%d_%H%M%S)}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$BASE"

content_job=$(sbatch --parsable "$SCRIPT_DIR/pr2075_build_content_subset.sbatch" "$BASE/content_input")
stage1a_job=$(sbatch --parsable --dependency=afterok:"$content_job" "$SCRIPT_DIR/pr2075_stage1a_subset.sbatch" "$BASE/content_input" "$BASE/stage1a")
stage1b_job=$(sbatch --parsable --dependency=afterok:"$stage1a_job" "$SCRIPT_DIR/pr2075_stage1b_4gpu_subset.sbatch" "$BASE/stage1a" "$BASE/stage1b_4gpu")
stage2_job=$(sbatch --parsable --dependency=afterok:"$stage1b_job" "$SCRIPT_DIR/pr2075_stage_gpu_pipeline_4gpu_subset.sbatch" "$BASE/stage1b_4gpu" "$BASE/stage_gpu_pipeline_4gpu")
stage3_job=$(sbatch --parsable --dependency=afterok:"$stage2_job" "$SCRIPT_DIR/pr2075_stage3_subset.sbatch" "$BASE/stage1b_4gpu" "$BASE/stage_gpu_pipeline_4gpu" "$BASE/stage3_propagation")

cat <<EOF
BASE=$BASE
content_job=$content_job
stage1a_job=$stage1a_job
stage1b_job=$stage1b_job
stage2_job=$stage2_job
stage3_job=$stage3_job
EOF
