#!/usr/bin/env bash
# submit_mineru_standalone_array.sh
# Submit MinerU-HTML standalone as a Slurm array (1 GPU per task).
#
# Usage:
#   bash submit_mineru_standalone_array.sh HOST INPUT_MANIFEST OUTPUT_DIR [NUM_SHARDS]
#
# Example:
#   bash submit_mineru_standalone_array.sh \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../layout_precompute_manifest.parquet \
#     /lustre/.../mineru_c_array_output \
#     32
set -euo pipefail

HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
DC_HOST="${DC_HOST:-vjawa@nb-hel-cs-001-dc-01.nvidia.com}"
INPUT_MANIFEST="${2:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/output_00/layout_precompute_manifest.parquet}"
OUTPUT_DIR="${3:-/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cc_mineru_array_$(date -u +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${4:-32}"

NEBIUS_SSH_CONTROL_DIR="${NEBIUS_SSH_CONTROL_DIR:-/tmp/.nebius_ctl}"
CTL="-o ControlMaster=auto -o ControlPath=$NEBIUS_SSH_CONTROL_DIR/%C.sock -o StrictHostKeyChecking=no"

# Use the venv from the working Dripper codex run (has vllm 0.18.1 + Gemma3Config-compatible transformers)
# The cached venv has a newer vllm that breaks on older transformers
CACHED_VENV="${MINERU_VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv}"
REMOTE_REPO=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/curator
SCRIPT=$REMOTE_REPO/tutorials/text/dripper-common-crawl/run_mineru_html_standalone.py
LAST_ARRAY_IDX=$(( NUM_SHARDS - 1 ))

echo "=== Syncing run_mineru_html_standalone.py via dc-01 ==="
rsync -az -e "ssh $CTL" \
  "$(dirname "$0")/run_mineru_html_standalone.py" \
  "$DC_HOST:$SCRIPT"

echo "=== Creating output dir on Lustre ==="
ssh $CTL "$HOST" "mkdir -p $OUTPUT_DIR"

echo "=== Writing SBATCH array script ==="
SBATCH_SCRIPT="$OUTPUT_DIR/job_array.sh"

ssh $CTL "$HOST" "cat > $SBATCH_SCRIPT" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=mineru-array
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-${LAST_ARRAY_IDX}
#SBATCH --output=${OUTPUT_DIR}/shard_%04a.out
#SBATCH --error=${OUTPUT_DIR}/shard_%04a.err

source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh 2>/dev/null || true

# Expose nvidia package libs for cupy (needed if GPU ops used)
SITE_PKGS="${CACHED_VENV}/lib/python3.12/site-packages"
for pkg_dir in "\${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "\${pkg_dir}" ] && export LD_LIBRARY_PATH="\${pkg_dir}:\${LD_LIBRARY_PATH:-}"
done

export TENSOR_PARALLEL_SIZE=1
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}

echo "=== MinerU-HTML array task \${SLURM_ARRAY_TASK_ID}/${LAST_ARRAY_IDX} ==="
echo "Host: \$(hostname)  GPU: \$(nvidia-smi -L | head -1)"
echo "Output: ${OUTPUT_DIR}"

${CACHED_VENV}/bin/python3 ${SCRIPT} \\
    --input   ${INPUT_MANIFEST} \\
    --output  ${OUTPUT_DIR} \\
    --shard-index \${SLURM_ARRAY_TASK_ID} \\
    --num-shards  ${NUM_SHARDS} \\
    --batch-size  64 \\
    --model   opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact

echo "=== shard \${SLURM_ARRAY_TASK_ID} DONE ==="
HEREDOC

echo ""
echo "=== Submitting array job (${NUM_SHARDS} tasks, 1 GPU each) ==="
ARRAY_JOB_ID=$(ssh $CTL "$HOST" "sbatch --parsable $SBATCH_SCRIPT")
echo ""
echo "ARRAY_JOB_ID=$ARRAY_JOB_ID"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOGS=${OUTPUT_DIR}/shard_NNNN.out"
echo ""
echo "Monitor:  squeue -j ${ARRAY_JOB_ID} --format='%.10i %.4K %.8T %.10M %R'"
echo "Merge when done:"
echo "  python3 merge_mineru_shards.py --input-dir ${OUTPUT_DIR} --output ${OUTPUT_DIR}/dripper_results.parquet"
