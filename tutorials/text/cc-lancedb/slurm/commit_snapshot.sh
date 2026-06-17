#!/usr/bin/env bash
# Phase 2: Commit all staged fragments for one CC snapshot.
# Submit after the phase-1 array completes:
#   sbatch --dependency=afterok:$ARRAY_JOB_ID slurm/commit_snapshot.sh CC-MAIN-2025-26

#SBATCH --job-name=cc-lance-commit
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_dataprocessing
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_logs/%x-%j.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_logs/%x-%j.log

set -euo pipefail

SNAPSHOT="${1:?Usage: sbatch commit_snapshot.sh CC-MAIN-2025-26}"
TOTAL_SPLITS="${2:-40}"

VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv
REPO=/home/vjawa/nemo-curator-adlr-mm
STAGING_DIR=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_staging
MANIFEST=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_manifest.jsonl
LANCE_URI=s3://vjawa-cc-lance/cc_all

CC_CFG=$($VENV/bin/python3 -c "
import yaml, os
cfg = yaml.safe_load(open(os.path.expanduser('~/.config/datamover/storage_locations')))
w = cfg['pdx-multimodal']['secrets']['local']
print(w['access_key_id'], w['secret_access_key'])
")
export AWS_ACCESS_KEY_ID=$(echo $CC_CFG | awk '{print $1}')
export AWS_SECRET_ACCESS_KEY=$(echo $CC_CFG | awk '{print $2}')
export AWS_ENDPOINT_URL_S3=https://pdx.s8k.io
export AWS_DEFAULT_REGION=us-east-1
export PYTHONPATH=$REPO:${PYTHONPATH:-}

echo "Committing $SNAPSHOT ($TOTAL_SPLITS splits) → $LANCE_URI"

cd $REPO/tutorials/text/cc-lancedb

$VENV/bin/python -u commit_snapshot.py \
    --snapshot-id   "$SNAPSHOT" \
    --lance-uri     "$LANCE_URI" \
    --staging-dir   "$STAGING_DIR" \
    --total-splits  "$TOTAL_SPLITS" \
    --manifest-file "$MANIFEST"

echo "Commit done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
