#!/usr/bin/env bash
# One-shard smoke test — survives SSH disconnect, logs to Lustre.
# Submit: sbatch slurm/test_1pct.sh
# Watch:  tail -f /lustre/fsw/portfolios/llmservice/users/vjawa/cc_lancedb_test_1pct.log

#SBATCH --job-name=cc-lancedb-test
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_dataprocessing
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lancedb_test_1pct.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lancedb_test_1pct.log

set -euo pipefail

VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv
REPO=/home/vjawa/nemo-curator-adlr-mm

# CC read creds (commoncrawl account)
CC_CONFIG=$(python3 -c "
import yaml,os
cfg = yaml.safe_load(open(os.path.expanduser('~/.config/datamover/storage_locations')))
c = cfg['pdx-commoncrawl']['secrets']['local']
w = cfg['pdx-multimodal']['secrets']['local']
print(c['access_key_id'], c['secret_access_key'], w['access_key_id'], w['secret_access_key'])
")
export CC_PBSS_ACCESS_KEY_ID=$(echo $CC_CONFIG | awk '{print $1}')
export CC_PBSS_SECRET_ACCESS_KEY=$(echo $CC_CONFIG | awk '{print $2}')
# Write creds (team-adlr-codistillation owns vjawa-cc-lance)
export AWS_ACCESS_KEY_ID=$(echo $CC_CONFIG | awk '{print $3}')
export AWS_SECRET_ACCESS_KEY=$(echo $CC_CONFIG | awk '{print $4}')
export AWS_ENDPOINT_URL_S3=https://pdx.s8k.io
export AWS_DEFAULT_REGION=us-east-1
export PYTHONPATH=$REPO:${PYTHONPATH:-}

echo "=============================="
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Start : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================="

cd $REPO/tutorials/text/cc-lancedb

# -u for unbuffered Python output so logs appear immediately
$VENV/bin/python -u build_url_index.py \
    --snapshot          CC-MAIN-2025-26 \
    --lancedb-uri       s3://vjawa-cc-lance \
    --table-name        cc_url_index_test \
    --max-shards        1 \
    --min-warc-length   5000

echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
