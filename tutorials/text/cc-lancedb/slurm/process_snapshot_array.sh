#!/usr/bin/env bash
# Phase 1: Write lance fragments for one split of one CC snapshot.
# Submit as a job array:
#   sbatch --array=0-39 slurm/process_snapshot_array.sh CC-MAIN-2025-26
#
# Each task handles WARCs[SLURM_ARRAY_TASK_ID::TOTAL_SPLITS] for the snapshot.
# Fragments are staged to $STAGING_DIR/<snapshot>/ — not committed yet.
# Phase 2 (commit_snapshot.sh) reads all staged files and issues one commit.

#SBATCH --job-name=cc-lance-write
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_dataprocessing
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_logs/%x-%A_%a.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_logs/%x-%A_%a.log

set -euo pipefail

SNAPSHOT="${1:?Usage: sbatch --array=0-39 process_snapshot_array.sh CC-MAIN-2025-26}"
TOTAL_SPLITS="${2:-40}"
SPLIT="$SLURM_ARRAY_TASK_ID"

VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv
REPO=/home/vjawa/nemo-curator-adlr-mm
DOWNLOAD_DIR=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_warcs_${SNAPSHOT}_split${SPLIT}
STAGING_DIR=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_staging
MANIFEST=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_manifest.jsonl
LANCE_URI=s3://vjawa-cc-lance/cc_all

# Read CC PBSS creds from dm config
CC_CFG=$($VENV/bin/python3 -c "
import yaml, os
cfg = yaml.safe_load(open(os.path.expanduser('~/.config/datamover/storage_locations')))
r = cfg['pdx-commoncrawl']['secrets']['local']
w = cfg['pdx-multimodal']['secrets']['local']
print(r['access_key_id'], r['secret_access_key'], w['access_key_id'], w['secret_access_key'])
")
export CC_PBSS_ACCESS_KEY_ID=$(echo $CC_CFG | awk '{print $1}')
export CC_PBSS_SECRET_ACCESS_KEY=$(echo $CC_CFG | awk '{print $2}')
export AWS_ACCESS_KEY_ID=$(echo $CC_CFG | awk '{print $3}')
export AWS_SECRET_ACCESS_KEY=$(echo $CC_CFG | awk '{print $4}')
export AWS_ENDPOINT_URL_S3=https://pdx.s8k.io
export AWS_DEFAULT_REGION=us-east-1
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export RAY_TMPDIR=/local/ray_${USER}

mkdir -p $RAY_TMPDIR $DOWNLOAD_DIR
mkdir -p /lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_logs

echo "============================="
echo "Snapshot : $SNAPSHOT"
echo "Split    : $SPLIT / $TOTAL_SPLITS"
echo "Job      : $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID)"
echo "Node     : $(hostname)"
echo "CPUs     : $(nproc)"
echo "Start    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================="

cd $REPO/tutorials/text/cc-lancedb

$VENV/bin/python -u build_cc_lancedb.py \
    --snapshot      "$SNAPSHOT" \
    --download-dir  "$DOWNLOAD_DIR" \
    --lance-uri     "$LANCE_URI" \
    --pbss \
    --split         "$SPLIT" \
    --total-splits  "$TOTAL_SPLITS" \
    --stage-only \
    --staging-dir   "$STAGING_DIR" \
    --manifest-file "$MANIFEST"

# Clean up per-split download dir to save space
rm -rf "$DOWNLOAD_DIR"
echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
