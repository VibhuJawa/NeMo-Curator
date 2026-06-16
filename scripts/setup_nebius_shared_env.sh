#!/bin/bash
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
#
# setup_nebius_shared_env.sh — build a shared venv on Lustre once; all jobs reuse it.
#
# Usage:
#   bash scripts/setup_nebius_shared_env.sh HOST LOCAL_REPO
#
#   HOST        SSH alias / hostname for the Nebius login or dc node
#               (e.g. nebius-login or dc-01)
#   LOCAL_REPO  Local path to the nemo_curator repo to rsync
#               (defaults to the directory containing this script's parent)
#
# What this script does (runs on your Mac / orchestration host):
#   1. Rsync LOCAL_REPO → nemo_curator_shared/ on Lustre (via SSH)
#   2. SSH to HOST and sbatch a cpu_short Slurm job that:
#        a. Checks whether the shared venv already has all required packages
#        b. If yes  → prints ENV_READY=yes and exits
#        c. If no   → runs full uv sync + pip install chain under flock

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
HOST="${1:-}"
LOCAL_REPO="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ -z "${HOST}" ]; then
    echo "Usage: $0 HOST [LOCAL_REPO]" >&2
    echo "  HOST       — SSH alias for Nebius node (e.g. dc-01)" >&2
    echo "  LOCAL_REPO — local path to repo (default: parent of scripts/)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths (all on Lustre)
# ---------------------------------------------------------------------------
REMOTE_USER="${REMOTE_USER:-vjawa}"
SHARED_DIR="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}/nemo_curator_shared"
SHARED_VENV="${SHARED_DIR}/.venv"
LOG_DIR="${SHARED_DIR}/logs"

LLM_WEB_KIT_PACKAGE="${LLM_WEB_KIT_PACKAGE:-git+https://github.com/ccprocessor/llm-webkit.git@dev}"

# ---------------------------------------------------------------------------
# Step 1: rsync local repo → shared dir
# ---------------------------------------------------------------------------
echo "================================================================"
echo "  Step 1: rsync ${LOCAL_REPO} → ${HOST}:${SHARED_DIR}"
echo "================================================================"

# Ensure the remote directory exists
ssh "${HOST}" "mkdir -p '${SHARED_DIR}'"

rsync -avz --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.uv-cache' \
    --exclude='.uv-python' \
    --exclude='logs/' \
    "${LOCAL_REPO}/" \
    "${HOST}:${SHARED_DIR}/"

echo "Rsync complete."

# ---------------------------------------------------------------------------
# Step 2: submit the env-build Slurm job
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Step 2: submit env-build job to ${HOST}"
echo "================================================================"

# Build the job script as a here-doc and pipe it into sbatch over SSH.
# We use 'EOF' (quoted) so that the heredoc is NOT expanded locally —
# all \$VAR references are expanded on the remote node by bash.
JOB_ID=$(ssh "${HOST}" bash <<OUTER
set -euo pipefail
mkdir -p '${LOG_DIR}'
sbatch --parsable <<'EOF'
#!/bin/bash
#SBATCH --job-name=nemo-curator-shared-env-build
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/shared_env_build_%j.log
#SBATCH --error=${LOG_DIR}/shared_env_build_%j.log

set -euo pipefail

SHARED_DIR="${SHARED_DIR}"
SHARED_VENV="${SHARED_VENV}"
USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}"
LLM_WEB_KIT_PACKAGE="${LLM_WEB_KIT_PACKAGE}"

echo "=================================================="
echo "  NeMo Curator shared env build"
echo "=================================================="
echo "  Host       : \$(hostname)"
echo "  Job ID     : \${SLURM_JOB_ID}"
echo "  Shared dir : \${SHARED_DIR}"
echo "  Shared venv: \${SHARED_VENV}"
echo "=================================================="

# Source PBSS credentials and user environment
set +u
source "\${HOME}/.bashrc"
set -u

if [ -f "\${USER_CACHE_ROOT}/cache_env.sh" ]; then
    set -a
    set +u
    # shellcheck disable=SC1090
    source "\${USER_CACHE_ROOT}/cache_env.sh"
    set -u
    set +a
fi

export UV_CACHE_DIR="\${UV_CACHE_DIR:-\${USER_CACHE_ROOT}/uv_cache}"
export UV_PROJECT_ENVIRONMENT="\${SHARED_VENV}"
export HF_HOME="\${HF_HOME:-\${USER_CACHE_ROOT}/hf_cache}"

cd "\${SHARED_DIR}"
uv --version
python --version || true

# ---------------------------------------------------------------------------
# Check whether the shared venv already has all required packages.
# If it does, skip the full install.
# ---------------------------------------------------------------------------
ENV_READY=0
if [ -x "\${SHARED_VENV}/bin/python" ]; then
    echo "Checking existing venv..."
    if "\${SHARED_VENV}/bin/python" -c \
        "import vllm, s3fs, sklearn, mineru_html, llm_web_kit, nemo_curator" \
        >/dev/null 2>&1; then
        ENV_READY=1
    fi
fi

if [ "\${ENV_READY}" = "1" ]; then
    echo "ENV_READY=yes — all packages present, skipping rebuild."
    exit 0
fi

echo "ENV_READY=no — building shared venv under flock..."

# ---------------------------------------------------------------------------
# Full install under flock so concurrent jobs don't collide
# ---------------------------------------------------------------------------
env_lock="\${SHARED_VENV}.lock"
(
    flock 9
    echo "Acquired lock, starting uv sync..."
    uv sync --inexact --extra inference_server --extra text_cpu

    echo "Installing s3fs, scikit-learn, aiohttp..."
    uv pip install \
        --python "\${SHARED_VENV}/bin/python" \
        "s3fs>=2024.2.0" \
        "scikit-learn>=1.6.1" \
        "aiohttp>=3.9"

    echo "Installing mineru_html..."
    if ! "\${SHARED_VENV}/bin/python" -c "import mineru_html" >/dev/null 2>&1; then
        uv pip install \
            --python "\${SHARED_VENV}/bin/python" \
            "mineru_html>=1.1.2"
    fi

    echo "Installing llm-webkit (no-deps)..."
    if ! "\${SHARED_VENV}/bin/python" -c "import llm_web_kit" >/dev/null 2>&1; then
        uv pip install \
            --python "\${SHARED_VENV}/bin/python" \
            "selectolax==0.3.33" \
            "scikit-learn>=1.6.1"
        uv pip install \
            --python "\${SHARED_VENV}/bin/python" \
            --no-deps \
            "\${LLM_WEB_KIT_PACKAGE}"
    fi
) 9>"\${env_lock}"

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
echo "Verifying shared venv..."
"\${SHARED_VENV}/bin/python" -c \
    "import vllm, s3fs, sklearn, mineru_html, llm_web_kit, nemo_curator; print('All imports OK')"

echo "=================================================="
echo "  ENV_READY=yes"
echo "  Shared venv: \${SHARED_VENV}"
echo "=================================================="
EOF
OUTER
)

echo "Submitted env-build job: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  ssh ${HOST} squeue -j ${JOB_ID}"
echo "  ssh ${HOST} tail -f '${LOG_DIR}/shared_env_build_${JOB_ID}.log'"
echo ""
echo "Shared venv will be at: ${SHARED_VENV}"
echo "Shared code will be at: ${SHARED_DIR}"
