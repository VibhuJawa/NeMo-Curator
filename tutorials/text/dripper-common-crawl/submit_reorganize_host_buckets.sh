#!/usr/bin/env bash
# submit_reorganize_host_buckets.sh
# Submit 100 Slurm jobs (one per host_bucket_group) to produce 10,000 sorted parquets.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/lib_nebius_ssh.sh"

HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
resolved_host="$(nebius_resolve_ssh_host "$HOST")"
rsync_host="$(nebius_resolve_rsync_host "$resolved_host")"

VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_precompute_manifest_20260609/curator/.venv
INPUT_BASE=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_host_bucket_map_20260608_003146/host_bucket_shards
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_sorted_host_buckets_20260611}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_n4_pre}
PARTITION=${SLURM_PARTITION:-cpu_dataprocessing}
CPUS=${CPUS_PER_TASK:-8}
MEM=${MEM_PER_NODE:-64G}
TIME=${TIME_LIMIT:-02:00:00}

REMOTE_SCRIPT=/tmp/reorganize_host_buckets.py

echo "HOST:       $resolved_host"
echo "INPUT:      $INPUT_BASE"
echo "OUTPUT:     $OUTPUT_DIR"
echo "PARTITION:  $PARTITION  CPUS=$CPUS  MEM=$MEM  TIME=$TIME"
echo ""

# Sync the Python script to remote
rsync_ssh="$(nebius_ssh_command_string "$rsync_host" 30)"
rsync -a -e "$rsync_ssh" "${script_dir}/reorganize_host_buckets.py" "$rsync_host:$REMOTE_SCRIPT"
echo "Script synced to $REMOTE_SCRIPT"

# Create output dir
nebius_ssh_command "$resolved_host" "mkdir -p '$OUTPUT_DIR'"

# Submit array job: 100 tasks, one per group_id (0-99)
JOB_SCRIPT=$(nebius_ssh_command "$resolved_host" "mktemp /tmp/reorganize_XXXXXX.sh")

nebius_ssh_command "$resolved_host" "cat > '$JOB_SCRIPT'" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=host-bucket-sort
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --array=0-99
#SBATCH --output=$OUTPUT_DIR/logs/group_%a.out
#SBATCH --error=$OUTPUT_DIR/logs/group_%a.err

mkdir -p $OUTPUT_DIR/logs
GROUP_ID=\$SLURM_ARRAY_TASK_ID
echo "Starting group \$GROUP_ID on \$(hostname) at \$(date -u)"
$VENV/bin/python3 $REMOTE_SCRIPT \$GROUP_ID $INPUT_BASE $OUTPUT_DIR
echo "Finished group \$GROUP_ID at \$(date -u)"
SBATCH

JOB_ID=$(nebius_ssh_command "$resolved_host" "sbatch --parsable '$JOB_SCRIPT'")
echo ""
echo "JOB_ID=$JOB_ID (array 0-99)"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOGS=$OUTPUT_DIR/logs/group_{0..99}.{out,err}"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f $OUTPUT_DIR/logs/group_0.out"
echo ""
echo "When done, verify:"
echo "  ls $OUTPUT_DIR/*.parquet | wc -l   # should be 10000"
