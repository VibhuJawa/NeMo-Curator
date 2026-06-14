#!/usr/bin/env bash
# submit_run_a_v2.sh
# Local script — syncs code to Nebius and submits the SBATCH job.
#
# Usage:
#   bash submit_run_a_v2.sh [nebius-host]
#
set -euo pipefail

HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
DC_HOST="${DC_HOST:-vjawa@nb-hel-cs-001-dc-01.nvidia.com}"
NEBIUS_SSH_CONTROL_DIR="${NEBIUS_SSH_CONTROL_DIR:-/tmp/.nebius_ctl}"
CTL="-o ControlMaster=auto -o ControlPath=$NEBIUS_SSH_CONTROL_DIR/%C.sock -o StrictHostKeyChecking=no"

LOCAL_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
REMOTE_REPO=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/curator
CACHED_VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv
SMOKE_BASE=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cc_main_2025_26_smoke
LOGS_DIR="$SMOKE_BASE/logs"

# ── 1. Sync code ──────────────────────────────────────────────────────────────
echo "=== Syncing code via dc-01 ==="
rsync -az -e "ssh $CTL" \
  --exclude='.git/' --exclude='.claude/' --exclude='.venv/' \
  --exclude='__pycache__/' --exclude='*.egg-info/' \
  "$LOCAL_REPO/" "$DC_HOST:$REMOTE_REPO/"

# ── 2. Ensure logs dir exists ─────────────────────────────────────────────────
ssh $CTL "$HOST" "mkdir -p $LOGS_DIR"

# ── 3. Write SBATCH script on remote ─────────────────────────────────────────
REMOTE_SBATCH="$REMOTE_REPO/tutorials/text/dripper-common-crawl/run_a_v2_sbatch.sh"

ssh $CTL "$HOST" "cat > $REMOTE_SBATCH" << SBATCH_HEREDOC
#!/bin/bash
#SBATCH --job-name=dripper-run-a-v2
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00
#SBATCH --output=$LOGS_DIR/run_a_v2_%j.log
#SBATCH --error=$LOGS_DIR/run_a_v2_%j.log

set -euo pipefail
source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh

# Use the venv from the working codex run (vllm 0.18.1 + compatible transformers)
# The dripper_cached_venv has a newer vllm incompatible with its transformers version
CACHED_VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv
CURATOR_DIR=$REMOTE_REPO
OUTPUT_DIR=$SMOKE_BASE/\${SLURM_JOB_ID}

mkdir -p "\${OUTPUT_DIR}"
# Symlink so the job log appears in the output dir too
ln -sf "$LOGS_DIR/run_a_v2_\${SLURM_JOB_ID}.log" "\${OUTPUT_DIR}/job.out" 2>/dev/null || true

# Expose bundled nvidia libs (cupy/cuML)
SITE_PKGS="\${CACHED_VENV}/lib/python3.12/site-packages"
for d in "\${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "\${d}" ] && export LD_LIBRARY_PATH="\${d}:\${LD_LIBRARY_PATH:-}"
done

export UV_PROJECT_ENVIRONMENT="\${CACHED_VENV}"
export PATH="\${CACHED_VENV}/bin:\${PATH}"
export RAY_TMPDIR="/tmp/ray_\${SLURM_JOB_ID}"
export OUTPUT_DIR
mkdir -p "\${RAY_TMPDIR}"

echo "Job \${SLURM_JOB_ID} starting on \$(hostname)"
echo "Output: \${OUTPUT_DIR}"
echo "ray binary: \$(which ray 2>/dev/null || echo 'NOT FOUND')"

cd "\${CURATOR_DIR}"
"\${CACHED_VENV}/bin/python3" \
    tutorials/text/dripper-common-crawl/main_run_a_v2.py

echo "Job \${SLURM_JOB_ID} complete. Output: \${OUTPUT_DIR}"
SBATCH_HEREDOC

ssh $CTL "$HOST" "chmod +x $REMOTE_SBATCH"

# ── 4. Submit ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Submitting Run A v2 ==="
JOB_ID=$(ssh $CTL "$HOST" "sbatch --parsable $REMOTE_SBATCH")
echo ""
echo "========================================================"
echo "  JOB_ID    = $JOB_ID"
echo "  LOG       = $LOGS_DIR/run_a_v2_${JOB_ID}.log"
echo "  OUTPUT    = $SMOKE_BASE/${JOB_ID}/"
echo ""
echo "  Watch:  ssh $HOST 'tail -f $LOGS_DIR/run_a_v2_${JOB_ID}.log'"
echo "  Status: bash scripts/check_nebius_jobs_compact.sh nb-hel-cs-001-login-01.nvidia.com ${JOB_ID}"
echo "========================================================"
