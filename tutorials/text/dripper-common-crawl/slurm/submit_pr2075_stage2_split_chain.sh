#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

STAGE1B_DIR=${1:?usage: submit_pr2075_stage2_split_chain.sh STAGE1B_DIR OUT_ROOT}
OUT_ROOT=${2:?usage: submit_pr2075_stage2_split_chain.sh STAGE1B_DIR OUT_ROOT}

mkdir -p "$OUT_ROOT"

stage2a_job=$(sbatch --parsable "$SCRIPT_DIR/pr2075_stage2a_prompt_prep.sbatch" "$STAGE1B_DIR" "$OUT_ROOT/stage2a_prompts")
stage2b_job=$(sbatch --parsable --dependency=afterok:"$stage2a_job" "$SCRIPT_DIR/pr2075_stage2b_llm_4gpu.sbatch" "$OUT_ROOT/stage2a_prompts" "$OUT_ROOT/stage2b_responses")
stage2c_job=$(sbatch --parsable --dependency=afterok:"$stage2b_job" "$SCRIPT_DIR/pr2075_stage2c_postprocess.sbatch" "$OUT_ROOT/stage2a_prompts" "$OUT_ROOT/stage2b_responses" "$OUT_ROOT/stage2c_templates")

cat <<EOF
Submitted split Stage 2 chain:
stage2a_job=$stage2a_job
stage2b_job=$stage2b_job
stage2c_job=$stage2c_job

Outputs:
stage2a=$OUT_ROOT/stage2a_prompts
stage2b=$OUT_ROOT/stage2b_responses
stage2c=$OUT_ROOT/stage2c_templates
EOF
