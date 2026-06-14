#!/usr/bin/env bash
# submit_stage3_cpu_propagation.sh
# Submit Stage 3 (CPU template propagation) as a Slurm array job on cpu_long partition.
#
# Usage:
#   bash submit_stage3_cpu_propagation.sh [HOST] [CLUSTER_MANIFEST_DIR] [INFERENCE_RESULTS_DIR] [OUTPUT_BASE]
#
# Positional args (all optional, can override via env vars):
#   HOST                  — Nebius login node  (default: vscode-01)
#   CLUSTER_MANIFEST_DIR  — Stage 1 output: cluster_assignments/ dir on Lustre
#   INFERENCE_RESULTS_DIR — Stage 2 output: gpu_results/ dir on Lustre
#   OUTPUT_BASE           — Base output path; a timestamped subdir is created here
#
# Environment overrides:
#   STAGE2_JOB_ID    — If set, adds --dependency=afterok:$STAGE2_JOB_ID to the sbatch
#   NUM_SHARDS       — Override the default 80 array tasks
#   NUM_WORKERS      — Override the default 64 parallel workers per node
#   DC_HOST          — dc-01/dc-02 node for rsync (faster than vscode for bulk)
#
# Example (standalone, after Stage 2 is done):
#   bash submit_stage3_cpu_propagation.sh \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../cc_scale_run_20260611/cluster_assignments \
#     /lustre/.../cc_scale_run_20260611/gpu_results \
#     /lustre/.../cc_scale_run_20260611
#
# Example (chained from Stage 2, job 999999):
#   STAGE2_JOB_ID=999999 bash submit_stage3_cpu_propagation.sh ...
#
set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
DC_HOST="${DC_HOST:-vjawa@nb-hel-cs-001-dc-01.nvidia.com}"

CLUSTER_MANIFEST_DIR="${2:-}"
INFERENCE_RESULTS_DIR="${3:-}"
OUTPUT_BASE="${4:-/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cc_stage3_$(date -u +%Y%m%d_%H%M%S)}"

NUM_SHARDS="${NUM_SHARDS:-80}"
NUM_WORKERS="${NUM_WORKERS:-64}"
STAGE2_JOB_ID="${STAGE2_JOB_ID:-}"

# Validate required dirs
if [[ -z "${CLUSTER_MANIFEST_DIR}" ]]; then
    echo "ERROR: CLUSTER_MANIFEST_DIR must be provided as \$2 or set via env" >&2
    exit 1
fi
if [[ -z "${INFERENCE_RESULTS_DIR}" ]]; then
    echo "ERROR: INFERENCE_RESULTS_DIR must be provided as \$3 or set via env" >&2
    exit 1
fi

# ── SSH multiplexing ──────────────────────────────────────────────────────────
NEBIUS_SSH_CONTROL_DIR="${NEBIUS_SSH_CONTROL_DIR:-/tmp/.nebius_ctl}"
mkdir -p "$NEBIUS_SSH_CONTROL_DIR"
CTL="-o ControlMaster=auto -o ControlPath=${NEBIUS_SSH_CONTROL_DIR}/%C.sock -o StrictHostKeyChecking=no"

# Use the venv from the working codex run (vllm 0.18.1 + Gemma3Config-compatible transformers)
CACHED_VENV="${MINERU_VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv}"
REMOTE_REPO="/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/curator"
SCRIPT="${REMOTE_REPO}/tutorials/text/dripper-common-crawl/stage3_cpu_propagation.py"

LAST_ARRAY_IDX=$(( NUM_SHARDS - 1 ))
OUTPUT_DIR="${OUTPUT_BASE}/propagation_results"

echo "=== Stage 3: CPU Template Propagation ==="
echo "  HOST:                  $HOST"
echo "  CLUSTER_MANIFEST_DIR:  $CLUSTER_MANIFEST_DIR"
echo "  INFERENCE_RESULTS_DIR: $INFERENCE_RESULTS_DIR"
echo "  OUTPUT_DIR:            $OUTPUT_DIR"
echo "  NUM_SHARDS (array):    $NUM_SHARDS"
echo "  NUM_WORKERS (per node): $NUM_WORKERS"
echo "  STAGE2_JOB_ID:         ${STAGE2_JOB_ID:-none}"
echo ""

# ── Sync stage3 script via dc-01 ──────────────────────────────────────────────
echo "=== Syncing stage3_cpu_propagation.py via dc-01 ==="
rsync -az -e "ssh $CTL" \
  "$(dirname "$0")/stage3_cpu_propagation.py" \
  "${DC_HOST}:${SCRIPT}"

# ── Ensure output dir exists ──────────────────────────────────────────────────
echo "=== Creating output dir on Lustre ==="
ssh $CTL "$HOST" "mkdir -p ${OUTPUT_DIR}"

# ── Write SBATCH array script on remote ──────────────────────────────────────
SBATCH_SCRIPT="${OUTPUT_DIR}/stage3_job_array.sh"
LOGS_DIR="${OUTPUT_DIR}/logs"

ssh $CTL "$HOST" "mkdir -p ${LOGS_DIR}"

ssh $CTL "$HOST" "cat > ${SBATCH_SCRIPT}" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=stage3-cpu-prop
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${NUM_WORKERS}
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --array=0-${LAST_ARRAY_IDX}
#SBATCH --output=${LOGS_DIR}/shard_%04a.out
#SBATCH --error=${LOGS_DIR}/shard_%04a.err

# ── Environment ───────────────────────────────────────────────────────────────
source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh 2>/dev/null || true

SITE_PKGS="${CACHED_VENV}/lib/python3.12/site-packages"
for pkg_dir in "\${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "\${pkg_dir}" ] && export LD_LIBRARY_PATH="\${pkg_dir}:\${LD_LIBRARY_PATH:-}"
done

export UV_PROJECT_ENVIRONMENT="${CACHED_VENV}"
export PATH="${CACHED_VENV}/bin:\${PATH}"

# Use spawn context to avoid lxml/lxml_bindings fork-safety issues
export PYTHONFAULTHANDLER=1

echo "=== Stage 3 array task \${SLURM_ARRAY_TASK_ID}/${LAST_ARRAY_IDX} ==="
echo "Host: \$(hostname)"
echo "CPUs: \${SLURM_CPUS_PER_TASK}"
echo "Memory: \${SLURM_MEM_PER_NODE}MB"
echo "Output: ${OUTPUT_DIR}"
echo ""

${CACHED_VENV}/bin/python3 ${SCRIPT} \\
    --cluster-manifest    "${CLUSTER_MANIFEST_DIR}" \\
    --inference-results   "${INFERENCE_RESULTS_DIR}" \\
    --output-dir          "${OUTPUT_DIR}" \\
    --shard-index         \${SLURM_ARRAY_TASK_ID} \\
    --num-shards          ${NUM_SHARDS} \\
    --num-workers         ${NUM_WORKERS} \\
    --dynamic-classid-similarity-threshold 0.70 \\
    --more-noise-enable \\
    --min-content-length-ratio 0.25 \\
    --max-content-length-ratio 4.0 \\
    --log-level           INFO \\
    --cluster-chunk-size  500

echo "=== shard \${SLURM_ARRAY_TASK_ID} DONE ==="
HEREDOC

ssh $CTL "$HOST" "chmod +x ${SBATCH_SCRIPT}"

# ── Submit with optional Stage 2 dependency ───────────────────────────────────
echo ""
echo "=== Submitting Stage 3 array (${NUM_SHARDS} tasks, 1 CPU node each) ==="

if [[ -n "${STAGE2_JOB_ID}" ]]; then
    ARRAY_JOB_ID=$(ssh $CTL "$HOST" \
        "sbatch --parsable --dependency=afterok:${STAGE2_JOB_ID} ${SBATCH_SCRIPT}")
    echo "  (dependency: afterok:${STAGE2_JOB_ID})"
else
    ARRAY_JOB_ID=$(ssh $CTL "$HOST" "sbatch --parsable ${SBATCH_SCRIPT}")
fi

echo ""
echo "================================================================"
echo "  STAGE3_ARRAY_JOB_ID = ${ARRAY_JOB_ID}"
echo "  NUM_SHARDS           = ${NUM_SHARDS}"
echo "  OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "  LOGS                 = ${LOGS_DIR}/shard_NNNN.out"
echo ""
echo "  Monitor:  squeue -j ${ARRAY_JOB_ID} --format='%.10i %.4K %.8T %.10M %R'"
echo "  Watch 1:  ssh $HOST 'tail -f ${LOGS_DIR}/shard_0000.out'"
echo ""
echo "  After completion, merge with:"
echo "    python3 merge_stage3_shards.py \\"
echo "      --input-dir  ${OUTPUT_DIR} \\"
echo "      --output     ${OUTPUT_BASE}/final_results.parquet"
echo ""
echo "  Check fallback rate:"
echo "    python3 -c \""
echo "      import pandas as pd, glob"
echo "      dfs = [pd.read_parquet(f) for f in sorted(glob.glob('${OUTPUT_DIR}/shard_*.parquet'))]"
echo "      df = pd.concat(dfs)"
echo "      print(df.groupby('propagation_method').size())"
echo "      print('fallback rate:', (df.propagation_method=='fallback').mean())"
echo "    \""
echo "================================================================"

# ── Export job ID for downstream chaining ─────────────────────────────────────
echo ""
echo "STAGE3_ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "OUTPUT_BASE=${OUTPUT_BASE}"
