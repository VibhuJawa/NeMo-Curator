#!/usr/bin/env bash
# submit_stage2_gpu_inference.sh
#
# Stage 2: GPU inference on cluster representatives only.
#
# This script is the second stage of the three-stage CC-scale pipeline:
#
#   Stage 1 (CPU array, 80 nodes): DOM clustering + representative selection
#   Stage 2 (GPU array, 8 nodes):  MinerU-HTML LLM inference on ~0.4-5% of pages
#   Stage 3 (CPU array, 80 nodes): XPath propagation to siblings
#
# Architecture:
#   - 64 Slurm array tasks, 1 GPU (H100) per task, TP=1
#   - Each task reads a slice of representatives from cluster_assignments/
#   - No Ray / NeMo Curator infrastructure — pure vLLM + PyArrow
#   - GPU util stays >20% watchdog threshold because no CPU propagation is mixed in
#
# Usage:
#   # Standalone (after Stage 1 completes):
#   bash submit_stage2_gpu_inference.sh \
#     HOST \
#     /lustre/.../cc_scale_run_YYYYMMDD/cluster_assignments \
#     /lustre/.../cc_scale_run_YYYYMMDD/gpu_results
#
#   # With Slurm dependency on Stage 1 merge job:
#   bash submit_stage2_gpu_inference.sh HOST INPUT_DIR OUTPUT_DIR [NUM_SHARDS] [STAGE1_MERGE_JOB_ID]
#
# Outputs per shard:
#   gpu_results/shard_NNNN_of_0064.parquet  — inference results
#   gpu_results/metrics_shard_NNNN.json     — per-task metrics
#
# Output columns:
#   url, url_host_name, layout_cluster_id, cluster_role, host_bucket,
#   dripper_content (mm_md text), dripper_html, dripper_error,
#   dripper_time_s, xpath_rules (JSON for Stage 3 lxml eval),
#   template_html, inference_time_s

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
DC_HOST="${DC_HOST:-vjawa@nb-hel-cs-001-dc-01.nvidia.com}"

# Stage 1 output directory containing cluster_assignments/ shards
INPUT_DIR="${2:-/lustre/fsw/portfolios/llmservice/users/vjawa/cc_scale_run/cluster_assignments}"

# Stage 2 output directory for inference results
OUTPUT_DIR="${3:-/lustre/fsw/portfolios/llmservice/users/vjawa/cc_scale_run/gpu_results}"

# Number of GPU array tasks (= number of H100 GPUs used concurrently).
# With 8 nodes x 8 GPUs = 64 total, set 64 for full throughput.
NUM_SHARDS="${4:-64}"

# Optional: Slurm job ID of Stage 1 merge job to express --dependency=afterok
STAGE1_MERGE_JOB_ID="${5:-}"

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_pre}"
PARTITION="${SLURM_PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL="${MODEL:-opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/llmservice/users/vjawa/hf_cache}"

# Working venv with vllm 0.18.1 + mineru_html installed
CACHED_VENV="${MINERU_VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv}"

REMOTE_REPO=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/curator
SCRIPT=$REMOTE_REPO/tutorials/text/dripper-common-crawl/run_mineru_html_standalone.py

LAST_ARRAY_IDX=$(( NUM_SHARDS - 1 ))

NEBIUS_SSH_CONTROL_DIR="${NEBIUS_SSH_CONTROL_DIR:-/tmp/.nebius_ctl}"
CTL="-o ControlMaster=auto -o ControlPath=$NEBIUS_SSH_CONTROL_DIR/%C.sock -o StrictHostKeyChecking=no"

# ── Sync script to Lustre ──────────────────────────────────────────────────────
echo "=== Stage 2: GPU inference on representatives ==="
echo "HOST=$HOST"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "STAGE1_MERGE_JOB_ID=${STAGE1_MERGE_JOB_ID:-<none>}"
echo "TIME_LIMIT=$TIME_LIMIT"
echo ""

echo "=== Syncing run_mineru_html_standalone.py via dc-01 ==="
rsync -az -e "ssh $CTL" \
  "$(dirname "$0")/run_mineru_html_standalone.py" \
  "$DC_HOST:$SCRIPT"

echo "=== Creating output dir on Lustre ==="
ssh $CTL "$HOST" "mkdir -p $OUTPUT_DIR"

# ── Write SBATCH script ────────────────────────────────────────────────────────
SBATCH_SCRIPT="$OUTPUT_DIR/stage2_job_array.sh"

ssh $CTL "$HOST" "cat > $SBATCH_SCRIPT" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=mineru-stage2-gpu
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=0-${LAST_ARRAY_IDX}
#SBATCH --output=${OUTPUT_DIR}/shard_%04a.out
#SBATCH --error=${OUTPUT_DIR}/shard_%04a.err

# ── Environment ─────────────────────────────────────────────────────────────
source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh 2>/dev/null || true

# Expose nvidia package libs for cupy / CUDA symbols
SITE_PKGS="${CACHED_VENV}/lib/python3.12/site-packages"
for pkg_dir in "\${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "\${pkg_dir}" ] && export LD_LIBRARY_PATH="\${pkg_dir}:\${LD_LIBRARY_PATH:-}"
done

export HF_HOME=${HF_CACHE}
export TRANSFORMERS_CACHE=${HF_CACHE}

# TP=1: model fits on 1 GPU; no inter-GPU communication → GPU util stays >20%
export TENSOR_PARALLEL_SIZE=1

# Isolate Ray temp dirs per task to avoid cross-task collisions
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}
mkdir -p "\${RAY_TMPDIR}"

echo "=== MinerU Stage 2 task \${SLURM_ARRAY_TASK_ID}/${LAST_ARRAY_IDX} ==="
echo "Host:  \$(hostname)"
echo "GPU:   \$(nvidia-smi -L | head -1)"
echo "Start: \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ── Stage 2 inference ────────────────────────────────────────────────────────
# --representatives-only: reads cluster_assignments/, filters to
#   cluster_role in {representative, singleton}, skips HTML > 500 KB,
#   writes inference_results with xpath_rules column for Stage 3.
${CACHED_VENV}/bin/python3 ${SCRIPT} \
    --input              ${INPUT_DIR} \
    --output             ${OUTPUT_DIR} \
    --representatives-only \
    --shard-index        \${SLURM_ARRAY_TASK_ID} \
    --num-shards         ${NUM_SHARDS} \
    --batch-size         ${BATCH_SIZE} \
    --model              ${MODEL} \
    --hf-cache           ${HF_CACHE}

EXIT_CODE=\$?
echo ""
echo "=== task \${SLURM_ARRAY_TASK_ID} finished with exit code \${EXIT_CODE} at \$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit \${EXIT_CODE}
HEREDOC

# ── Submit ────────────────────────────────────────────────────────────────────
DEPENDENCY_FLAG=""
if [[ -n "${STAGE1_MERGE_JOB_ID}" ]]; then
    DEPENDENCY_FLAG="--dependency=afterok:${STAGE1_MERGE_JOB_ID}"
    echo "=== Submitting Stage 2 with dependency on Stage 1 merge job ${STAGE1_MERGE_JOB_ID} ==="
else
    echo "=== Submitting Stage 2 immediately (no Stage 1 dependency) ==="
fi

ARRAY_JOB_ID=$(ssh $CTL "$HOST" "sbatch --parsable ${DEPENDENCY_FLAG} $SBATCH_SCRIPT")

echo ""
echo "STAGE2_JOB_ID=$ARRAY_JOB_ID"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOGS=${OUTPUT_DIR}/shard_NNNN.out"
echo ""
echo "Monitor progress:"
echo "  ssh $HOST 'squeue -j ${ARRAY_JOB_ID} --format=\"%.10i %.4K %.8T %.10M %R\"'"
echo ""
echo "Check GPU utilization (pick any running node):"
echo "  ssh <node> 'nvidia-smi dmon -s u -d 5'"
echo ""
echo "Merge when all tasks complete:"
echo "  python3 merge_stage2_results.py \\"
echo "    --input-dir ${OUTPUT_DIR} \\"
echo "    --output ${OUTPUT_DIR}/inference_results.parquet"
echo ""
echo "Then submit Stage 3:"
echo "  bash submit_stage3_propagation.sh $HOST \\"
echo "    <cluster_assignments_dir> \\"
echo "    ${OUTPUT_DIR}/inference_results.parquet \\"
echo "    <stage3_output_dir> \\"
echo "    \${ARRAY_JOB_ID}"  # depends on this job completing
