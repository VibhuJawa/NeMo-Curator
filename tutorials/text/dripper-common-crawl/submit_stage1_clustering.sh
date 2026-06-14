#!/usr/bin/env bash
# submit_stage1_clustering.sh
#
# Sync stage1_cpu_clustering.py to Nebius and submit as a Slurm CPU array job.
#
# Usage:
#   bash submit_stage1_clustering.sh [login-host] [INPUT_MANIFEST] [OUTPUT_DIR] [NUM_SHARDS]
#
# Examples:
#   # Smoke test: 1 shard, 1000 pages on cpu_short
#   bash submit_stage1_clustering.sh \
#       vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#       /lustre/.../layout_precompute_manifest.parquet \
#       /lustre/.../stage1_output \
#       1
#
#   # Full CC scale: 80 shards on cpu_long
#   bash submit_stage1_clustering.sh \
#       vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#       /lustre/.../layout_precompute_manifest.parquet \
#       /lustre/.../stage1_output_YYYYMMDD \
#       80
#
# Environment overrides (set before calling this script):
#   SMOKE_TEST=1             use cpu_short (1h) + --max-pages 1000
#   PARTITION=cpu_long       override partition (default: cpu_long)
#   DC_HOST                  rsync host (default: dc-01)
#   NEBIUS_SSH_CONTROL_DIR   SSH multiplex socket dir (default: /tmp/.nebius_ctl)
#
set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
INPUT_MANIFEST="${2:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/output_00/layout_precompute_manifest.parquet}"
OUTPUT_DIR="${3:-/lustre/fsw/portfolios/llmservice/users/vjawa/cc_scale_stage1_$(date -u +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${4:-80}"

# ── Config ────────────────────────────────────────────────────────────────────
DC_HOST="${DC_HOST:-vjawa@nb-hel-cs-001-dc-01.nvidia.com}"
NEBIUS_SSH_CONTROL_DIR="${NEBIUS_SSH_CONTROL_DIR:-/tmp/.nebius_ctl}"
CTL="-o ControlMaster=auto -o ControlPath=$NEBIUS_SSH_CONTROL_DIR/%C.sock -o StrictHostKeyChecking=no"

SMOKE_TEST="${SMOKE_TEST:-0}"
if [[ "$SMOKE_TEST" == "1" ]]; then
    PARTITION="${PARTITION:-cpu_short}"
    TIME_LIMIT="01:00:00"
    MAX_PAGES_ARG="--max-pages 1000"
    echo "=== SMOKE TEST MODE (cpu_short, 1000 pages per shard) ==="
else
    PARTITION="${PARTITION:-cpu_long}"
    TIME_LIMIT="04:00:00"   # 3h expected + 1h buffer
    MAX_PAGES_ARG=""
fi

# Paths on the remote Lustre filesystem
REMOTE_REPO=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/curator
# Use the working venv (vllm 0.18.1 + cuML-compatible CUDA libs)
CACHED_VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv

LAST_ARRAY_IDX=$(( NUM_SHARDS - 1 ))
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================================"
echo "  Stage 1 CPU Clustering — Slurm Array Submit"
echo "========================================================"
echo "  Login host:    $HOST"
echo "  DC host:       $DC_HOST"
echo "  Input:         $INPUT_MANIFEST"
echo "  Output:        $OUTPUT_DIR"
echo "  Shards:        $NUM_SHARDS  (array 0-$LAST_ARRAY_IDX)"
echo "  Partition:     $PARTITION  (time: $TIME_LIMIT)"
echo "  Smoke test:    ${SMOKE_TEST:-0}"
echo ""

# ── 1. Ensure SSH multiplex socket dir exists ─────────────────────────────────
mkdir -p "$NEBIUS_SSH_CONTROL_DIR"

# ── 2. Sync the clustering script and gpu_layout_clustering via dc-01 ─────────
echo "=== Syncing stage1_cpu_clustering.py via dc-01 ==="
rsync -az -e "ssh $CTL" \
    "$LOCAL_DIR/stage1_cpu_clustering.py" \
    "$DC_HOST:$REMOTE_REPO/tutorials/text/dripper-common-crawl/stage1_cpu_clustering.py"

# Also sync the GPU clustering module (needed on GPU-capable nodes)
GPU_MOD_LOCAL="$(cd "$LOCAL_DIR/../../.." && pwd)/nemo_curator/stages/text/experimental/dripper/gpu_layout_clustering.py"
if [[ -f "$GPU_MOD_LOCAL" ]]; then
    echo "=== Syncing gpu_layout_clustering.py ==="
    rsync -az -e "ssh $CTL" \
        "$GPU_MOD_LOCAL" \
        "$DC_HOST:$REMOTE_REPO/nemo_curator/stages/text/experimental/dripper/gpu_layout_clustering.py"
fi

# ── 3. Create output dir on Lustre ────────────────────────────────────────────
echo "=== Creating output dir on Lustre: $OUTPUT_DIR ==="
ssh $CTL "$HOST" "mkdir -p $OUTPUT_DIR"

# ── 4. Write SBATCH array script on remote ────────────────────────────────────
echo "=== Writing SBATCH array script ==="
SBATCH_SCRIPT="$OUTPUT_DIR/stage1_array.sh"

ssh $CTL "$HOST" "cat > $SBATCH_SCRIPT" << HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=cc-stage1-cluster
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=235G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=0-${LAST_ARRAY_IDX}
#SBATCH --output=${OUTPUT_DIR}/shard_%04a.out
#SBATCH --error=${OUTPUT_DIR}/shard_%04a.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh 2>/dev/null || true

CACHED_VENV=${CACHED_VENV}
REMOTE_REPO=${REMOTE_REPO}

# Expose nvidia libs for cupy / cuML (needed even on CPU nodes for cosine sim)
SITE_PKGS="\${CACHED_VENV}/lib/python3.12/site-packages"
for pkg_dir in "\${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "\${pkg_dir}" ] && export LD_LIBRARY_PATH="\${pkg_dir}:\${LD_LIBRARY_PATH:-}"
done

export PYTHONPATH="\${REMOTE_REPO}:\${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="\${CACHED_VENV}"
export PATH="\${CACHED_VENV}/bin:\${PATH}"

# Suppress noisy tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

echo "========================================================="
echo "Stage 1 CPU Clustering — array task \${SLURM_ARRAY_TASK_ID}/${LAST_ARRAY_IDX}"
echo "Host: \$(hostname)"
echo "CPUs: \$(nproc)  MEM: \$(free -h | awk '/^Mem/{print \$2}')"
echo "========================================================="

# ── Run Stage 1 ───────────────────────────────────────────────────────────────
"\${CACHED_VENV}/bin/python3" \
    "\${REMOTE_REPO}/tutorials/text/dripper-common-crawl/stage1_cpu_clustering.py" \
    --input   "${INPUT_MANIFEST}" \
    --output  "${OUTPUT_DIR}" \
    --shard-index "\${SLURM_ARRAY_TASK_ID}" \
    --num-shards  "${NUM_SHARDS}" \
    --workers 62 \
    --threshold 0.95 \
    --min-cluster-size 2 \
    --max-host-pages 4096 \
    --gpu-min-size 200 \
    ${MAX_PAGES_ARG}

echo "=== shard \${SLURM_ARRAY_TASK_ID} DONE ==="
HEREDOC

ssh $CTL "$HOST" "chmod +x $SBATCH_SCRIPT"

# ── 5. Submit the array job ────────────────────────────────────────────────────
echo ""
echo "=== Submitting array job ($NUM_SHARDS tasks) ==="
ARRAY_JOB_ID=$(ssh $CTL "$HOST" "sbatch --parsable $SBATCH_SCRIPT")

echo ""
echo "========================================================"
echo "  ARRAY_JOB_ID = $ARRAY_JOB_ID"
echo "  NUM_SHARDS   = $NUM_SHARDS"
echo "  PARTITION    = $PARTITION"
echo "  OUTPUT_DIR   = $OUTPUT_DIR"
echo "  LOGS         = $OUTPUT_DIR/shard_NNNN.out"
echo ""
echo "  Monitor:  ssh $HOST \"squeue -j ${ARRAY_JOB_ID} --format='%.10i %.4K %.8T %.10M %R'\""
echo "  Tail log: ssh $HOST \"tail -f ${OUTPUT_DIR}/shard_0000.out\""
echo ""
echo "  After all tasks complete, verify with:"
echo "    ssh $HOST \"ls $OUTPUT_DIR/shard_*.parquet | wc -l\"   # should be $NUM_SHARDS"
echo "    ssh $HOST \"ls $OUTPUT_DIR/metrics_shard_*.json | wc -l\"  # same"
echo ""
echo "  Then submit Stage 2 GPU inference with:"
echo "    bash submit_stage2_gpu_inference.sh $HOST $OUTPUT_DIR <stage2-output-dir>"
echo "========================================================"

# ── 6. Optional: submit a merge/sentinel job after all shards complete ────────
# This writes a _SUCCESS sentinel that Stage 2 can use as a dependency check.
MERGE_SBATCH="$OUTPUT_DIR/stage1_merge.sh"
ssh $CTL "$HOST" "cat > $MERGE_SBATCH" << MERGE_HEREDOC
#!/usr/bin/env bash
#SBATCH --job-name=cc-stage1-merge
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}
#SBATCH --output=${OUTPUT_DIR}/merge.out
#SBATCH --error=${OUTPUT_DIR}/merge.err

set -euo pipefail

echo "=== Stage 1 Merge / Validation ==="
echo "Checking output: ${OUTPUT_DIR}"

# Count completed shards
SHARDS_FOUND=\$(ls "${OUTPUT_DIR}"/shard_*.parquet 2>/dev/null | wc -l)
echo "Shards found: \$SHARDS_FOUND / ${NUM_SHARDS}"

if [ "\$SHARDS_FOUND" -lt "${NUM_SHARDS}" ]; then
    echo "ERROR: Only \$SHARDS_FOUND of ${NUM_SHARDS} shards complete" >&2
    exit 1
fi

# Aggregate metrics across shards
CACHED_VENV=${CACHED_VENV}
"\${CACHED_VENV}/bin/python3" - << 'PYEOF'
import json, glob, sys
from pathlib import Path

output_dir = "${OUTPUT_DIR}"
metrics_files = sorted(glob.glob(f"{output_dir}/metrics_shard_*.json"))
if not metrics_files:
    print("No metrics files found", file=sys.stderr)
    sys.exit(1)

totals = {
    "total_pages": 0,
    "clustered_pages": 0,
    "singleton_pages": 0,
    "representative_pages": 0,
    "feature_error_pages": 0,
    "shards": len(metrics_files),
}
for mf in metrics_files:
    m = json.loads(Path(mf).read_text())
    for k in ["total_pages", "clustered_pages", "singleton_pages",
              "representative_pages", "feature_error_pages"]:
        totals[k] += m.get(k, 0)

llm_pages = totals["representative_pages"] + totals["singleton_pages"]
total = totals["total_pages"]
totals["llm_call_pages"] = llm_pages
totals["call_reduction_pct"] = 100.0 * (1.0 - llm_pages / max(total, 1))

print(json.dumps(totals, indent=2))
summary_path = Path(output_dir) / "stage1_summary.json"
summary_path.write_text(json.dumps(totals, indent=2))
print(f"Summary written: {summary_path}")
PYEOF

# Write _SUCCESS sentinel for downstream dependency
touch "${OUTPUT_DIR}/_SUCCESS"
echo "=== Stage 1 COMPLETE — wrote _SUCCESS sentinel ==="
MERGE_HEREDOC

ssh $CTL "$HOST" "chmod +x $MERGE_SBATCH"
MERGE_JOB_ID=$(ssh $CTL "$HOST" "sbatch --parsable $MERGE_SBATCH")

echo ""
echo "  Merge/validation job: $MERGE_JOB_ID"
echo "  (auto-submitted with --dependency=afterok:$ARRAY_JOB_ID)"
echo ""
echo "  Stage 2 GPU inference can depend on: $MERGE_JOB_ID"
echo "  Use: sbatch --dependency=afterok:$MERGE_JOB_ID <stage2_script>"
echo "========================================================"
