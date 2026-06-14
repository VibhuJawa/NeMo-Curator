#!/usr/bin/env bash
# split_and_submit_clustering.sh
# Split host_bucket=NNNN.parquet into N chunks by host, submit N parallel
# layout-precompute jobs, each fetching WARCs + running DBSCAN on its hosts.
#
# Usage:
#   bash split_and_submit_clustering.sh HOST SHARD_PATH [N_NODES] [OUTPUT_BASE]
#
# Example:
#   N_NODES=4 bash split_and_submit_clustering.sh \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     /lustre/.../host_bucket=0000.parquet 4
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/lib_nebius_ssh.sh"

HOST="${1:?Usage: $0 HOST SHARD_PATH [N_NODES] [OUTPUT_BASE]}"
SHARD_PATH="${2:?}"
N_NODES="${N_NODES:-${3:-4}}"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-${4:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_clustering_${TS}}}"

VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_precompute_manifest_20260609/curator/.venv
SPLIT_DIR="${OUTPUT_BASE}/input_splits"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$script_dir/../../../.." && pwd)}"  # nemo_curator_dc_v2

resolved_host="$(nebius_resolve_ssh_host "$HOST")"
rsync_host="$(nebius_resolve_rsync_host "$resolved_host")"
rsync_ssh="$(nebius_ssh_command_string "$rsync_host" 30) -o StrictHostKeyChecking=no"

echo "HOST:           $resolved_host"
echo "SHARD:          $SHARD_PATH"
echo "N_NODES:        $N_NODES"
echo "OUTPUT_BASE:    $OUTPUT_BASE"
echo "LOCAL_REPO:     $LOCAL_REPO"
echo ""

# ── Step 1: Create split dir and run Python split script on remote ────────────
nebius_ssh_command "$resolved_host" "mkdir -p '$SPLIT_DIR' '${OUTPUT_BASE}/logs'"

REMOTE_SPLIT_SCRIPT=/lustre/fsw/portfolios/llmservice/users/vjawa/split_shard_by_host.py
cat > /tmp/split_shard_by_host_local.py << 'PYEOF'
#!/usr/bin/env python3
"""Split a host-sorted parquet into N chunks by url_host_name range."""
import sys, os
import pyarrow.parquet as pq
import pandas as pd

shard_path  = sys.argv[1]
output_dir  = sys.argv[2]
n_chunks    = int(sys.argv[3])

df = pq.ParquetFile(shard_path).read().to_pandas()
print(f"Loaded: {len(df):,} rows, {df['url_host_name'].nunique():,} hosts")

hosts = sorted(df['url_host_name'].unique())
chunk_size = len(hosts) // n_chunks
splits = []
for i in range(n_chunks):
    start = i * chunk_size
    end   = (i + 1) * chunk_size if i < n_chunks - 1 else len(hosts)
    chunk_hosts = hosts[start:end]
    chunk_df = df[df['url_host_name'].isin(chunk_hosts)].reset_index(drop=True)
    out = os.path.join(output_dir, f"chunk_{i:02d}.parquet")
    chunk_df.to_parquet(out, index=False, compression='snappy')
    print(f"chunk_{i:02d}: {len(chunk_hosts)} hosts, {len(chunk_df):,} rows → {out}")
    splits.append(out)

print(f"\nWrote {n_chunks} splits to {output_dir}")
PYEOF

rsync -a -e "$rsync_ssh" /tmp/split_shard_by_host_local.py "$rsync_host:$REMOTE_SPLIT_SCRIPT"

echo "=== Splitting shard into $N_NODES chunks ==="
nebius_ssh_command "$resolved_host" \
  "$VENV/bin/python3 $REMOTE_SPLIT_SCRIPT '$SHARD_PATH' '$SPLIT_DIR' $N_NODES"

echo ""

# ── Step 2: Sync local repo to remote (reuse for all nodes) ─────────────────
REMOTE_REPO="${OUTPUT_BASE}/curator"
nebius_ssh_command "$resolved_host" "mkdir -p '$REMOTE_REPO'"

echo "=== Syncing Curator code ==="
rsync -a -e "$rsync_ssh" \
  --exclude='.git/' --exclude='.github/' --exclude='.claude/' \
  --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$LOCAL_REPO/" "$rsync_host:$REMOTE_REPO/"

# ── Step 3: Submit array job ─────────────────────────────────────────────────
ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_pre}"
PARTITION="${SLURM_PARTITION:-cpu_short}"
CPUS="${CPUS_PER_TASK:-64}"
MEM="${MEM_PER_NODE:-32G}"
TIME="${TIME_LIMIT:-02:00:00}"
FETCH_WORKERS="${MANIFEST_FETCH_WORKERS:-64}"

echo "=== Submitting Slurm array job (0-$((N_NODES-1))) ==="
LOCAL_JOB_SCRIPT=/tmp/layout_cluster_array_job.sh
JOB_SCRIPT="${OUTPUT_BASE}/logs/array_job.sh"

# Generate job script locally then rsync to Lustre
cat > "$LOCAL_JOB_SCRIPT" << SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=layout-cluster
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-$((N_NODES-1))
#SBATCH --output=${OUTPUT_BASE}/logs/chunk_%a.out
#SBATCH --error=${OUTPUT_BASE}/logs/chunk_%a.err

source /lustre/fsw/portfolios/llmservice/users/vjawa/cache_env.sh
export AWS_ACCESS_KEY_ID=\$PBSS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=\$PBSS_SECRET_ACCESS_KEY
export UV_PROJECT_ENVIRONMENT="${VENV}"
export PYTHONPATH="${REMOTE_REPO}:\${PYTHONPATH:-}"
# Short RAY_TMPDIR — Unix sockets can't exceed 107 bytes
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
mkdir -p \$RAY_TMPDIR
# uv lives on Lustre (set by cache_env.sh UV_TOOL_DIR)
UV="${VENV}/../../../uv_tools/bin/uv"
if [ ! -f "\$UV" ]; then UV=\$(which uv 2>/dev/null || echo ""); fi
if [ -z "\$UV" ]; then echo "ERROR: uv not found" >&2; exit 1; fi
echo "Using uv: \$UV"

CHUNK_ID=\$(printf "%02d" \$SLURM_ARRAY_TASK_ID)
INPUT=${SPLIT_DIR}/chunk_\${CHUNK_ID}.parquet
OUTPUT=${OUTPUT_BASE}/output_\${CHUNK_ID}
mkdir -p \$OUTPUT

echo "[chunk \$CHUNK_ID] starting on \$(hostname) at \$(date -u)"
cd ${REMOTE_REPO}
\$UV run --no-sync python tutorials/text/dripper-common-crawl/main.py \
  --input-manifest-path "\$INPUT" \
  --manifest-warc-bucket crawl-data \
  --manifest-fetch-workers ${FETCH_WORKERS} \
  --output-dir "\$OUTPUT" \
  --precompute-layout-manifest-only \
  --layout-template-layout-id-col dripper_layout_id \
  --layout-cluster-threshold 0.95 \
  --layout-template-min-cluster-size 2 \
  --layout-page-signature-mode none \
  --pipeline-shard-strategy layout_complete \
  --pipeline-shard-size 256 \
  --pipeline-layout-workers ${CPUS} \
  --max-pages 0

echo "[chunk \$CHUNK_ID] done at \$(date -u)"
ls -lh \$OUTPUT/
SBATCH

rsync -a -e "$rsync_ssh" "$LOCAL_JOB_SCRIPT" "$rsync_host:$JOB_SCRIPT"
chmod +x "$LOCAL_JOB_SCRIPT"

JOB_ID=$(nebius_ssh_command "$resolved_host" "sbatch --parsable '$JOB_SCRIPT'")
echo ""
echo "JOB_ID=${JOB_ID} (array 0-$((N_NODES-1)))"
echo "OUTPUT_BASE=${OUTPUT_BASE}"
echo ""
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     ${OUTPUT_BASE}/logs/chunk_{0..3}.out"
echo ""
echo "When done, merge with:"
echo "  python3 - << 'EOF'"
echo "  import pandas as pd, glob"
echo "  parts = [pd.read_parquet(f) for f in sorted(glob.glob('${OUTPUT_BASE}/output_*/layout_precompute_manifest.parquet'))]"
echo "  merged = pd.concat(parts, ignore_index=True)"
echo "  merged.to_parquet('${OUTPUT_BASE}/layout_precompute_manifest_full.parquet', index=False)"
echo "  print('Merged:', len(merged), 'rows,', merged['dripper_layout_id'].str.startswith('layout-',na=False).sum(), 'clustered')"
echo "  EOF"
