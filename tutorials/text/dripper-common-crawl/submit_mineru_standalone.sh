#!/usr/bin/env bash
# submit_mineru_standalone.sh
# Submit a Slurm job that runs MinerU-HTML directly (no Curator infrastructure).
# Usage: bash submit_mineru_standalone.sh HOST [INPUT_MANIFEST] [OUTPUT_DIR] [MAX_PAGES]
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/lib_nebius_ssh.sh"

HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
INPUT_MANIFEST="${2:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_20260611_194849/output_00/layout_precompute_manifest.parquet}"
OUTPUT_DIR="${3:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_mineru_standalone_$(date -u +%Y%m%d_%H%M%S)}"
MAX_PAGES="${MAX_PAGES:-${4:-2000}}"

ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_pre}"
PARTITION="${SLURM_PARTITION:-batch}"
H100_COUNT="${H100_COUNT:-8}"
TIME="${TIME_LIMIT:-01:00:00}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL="${MODEL:-opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact}"
HF_CACHE="/lustre/fsw/portfolios/llmservice/users/vjawa/hf_cache"

# The venv that has mineru_html + vllm installed
# Use the Curator venv which already has mineru_html from earlier setup
VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cc_main_2025_26_smoke/.venv

resolved_host="$(nebius_resolve_ssh_host "$HOST")"
rsync_host="$(nebius_resolve_rsync_host "$resolved_host")"
rsync_ssh="$(nebius_ssh_command_string "$rsync_host" 30)"

REMOTE_SCRIPT=/lustre/fsw/portfolios/llmservice/users/vjawa/run_mineru_html_standalone.py

echo "SUBMIT_MINERU_STANDALONE_BEGIN"
echo "HOST=$resolved_host"
echo "INPUT_MANIFEST=$INPUT_MANIFEST"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_PAGES=$MAX_PAGES"
echo "H100_COUNT=$H100_COUNT"
echo "PARTITION=$PARTITION"
echo "MODEL=$MODEL"

# Create output dir and sync script to Lustre
nebius_ssh_command "$resolved_host" "mkdir -p '$(printf "%q" "$OUTPUT_DIR")'"
rsync -a -e "$rsync_ssh" "${script_dir}/run_mineru_html_standalone.py" "$rsync_host:$REMOTE_SCRIPT"

# Generate SBATCH script locally then copy
LOCAL_JOB=/tmp/mineru_standalone_job.sh
cat > "$LOCAL_JOB" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=mineru-standalone
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${H100_COUNT}
#SBATCH --time=${TIME}
#SBATCH --output=${OUTPUT_DIR}/job.out
#SBATCH --error=${OUTPUT_DIR}/job.err

source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh
export HF_HOME=${HF_CACHE}
export TRANSFORMERS_CACHE=${HF_CACHE}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

# Use the smoke run venv (has mineru_html, vllm, torch already installed)
VENV=${VENV}
export PATH="\$VENV/bin:\$PATH"
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
mkdir -p \$RAY_TMPDIR

echo "=== MinerU-HTML Standalone Baseline ==="
echo "Host: \$(hostname)"
echo "GPUs: \$(nvidia-smi -L | wc -l)"
nvidia-smi -L

echo ""
echo "Starting extraction at \$(date -u)"

\$VENV/bin/python3 ${REMOTE_SCRIPT} \
  --input   "${INPUT_MANIFEST}" \
  --output  "${OUTPUT_DIR}" \
  --max-pages ${MAX_PAGES} \
  --batch-size ${BATCH_SIZE} \
  --model   "${MODEL}" \
  --hf-cache ${HF_CACHE}

echo "Finished at \$(date -u)"
echo "Output:"
ls -lh ${OUTPUT_DIR}/
SBATCH

REMOTE_JOB_SCRIPT="${OUTPUT_DIR}/job_script.sh"
rsync -a -e "$rsync_ssh" "$LOCAL_JOB" "$rsync_host:$REMOTE_JOB_SCRIPT"

JOB_ID=$(nebius_ssh_command "$resolved_host" "sbatch --parsable '$REMOTE_JOB_SCRIPT'")
echo "JOB_ID=$JOB_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_OUT=${OUTPUT_DIR}/job.out"
echo "LOG_ERR=${OUTPUT_DIR}/job.err"
echo "SUBMIT_MINERU_STANDALONE_END"
