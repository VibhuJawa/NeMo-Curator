#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm array job: build a LanceDB URL index for every Common Crawl snapshot.
# Usage:
#   sbatch slurm/submit_array.sh
#   # Override defaults via environment:
#   LANCEDB_URI=s3://my-bucket TABLE_NAME=my_table sbatch slurm/submit_array.sh

#SBATCH --job-name=cc-lancedb-index
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_dataprocessing
#SBATCH --array=0-120%4
#SBATCH --output=logs/cc-lancedb-%A_%a.out
#SBATCH --error=logs/cc-lancedb-%A_%a.err

# ---------------------------------------------------------------------------
# All 34 snapshots available under pdx-commoncrawl cc-index
# ---------------------------------------------------------------------------
SNAPSHOTS=(
  CC-MAIN-2013-20 CC-MAIN-2013-48 CC-MAIN-2014-10 CC-MAIN-2014-15
  CC-MAIN-2014-23 CC-MAIN-2014-35 CC-MAIN-2014-41 CC-MAIN-2014-42
  CC-MAIN-2014-49 CC-MAIN-2014-52 CC-MAIN-2015-06 CC-MAIN-2015-11
  CC-MAIN-2015-14 CC-MAIN-2015-18 CC-MAIN-2015-22 CC-MAIN-2015-27
  CC-MAIN-2015-32 CC-MAIN-2015-35 CC-MAIN-2015-40 CC-MAIN-2015-48
  CC-MAIN-2016-07 CC-MAIN-2016-18 CC-MAIN-2016-22 CC-MAIN-2016-26
  CC-MAIN-2016-30 CC-MAIN-2016-36 CC-MAIN-2016-40 CC-MAIN-2016-44
  CC-MAIN-2016-50 CC-MAIN-2017-04 CC-MAIN-2017-09 CC-MAIN-2017-13
  CC-MAIN-2017-17 CC-MAIN-2017-22 CC-MAIN-2017-26 CC-MAIN-2017-30
  CC-MAIN-2017-34 CC-MAIN-2017-39 CC-MAIN-2017-43 CC-MAIN-2017-47
  CC-MAIN-2017-51 CC-MAIN-2018-05 CC-MAIN-2018-09 CC-MAIN-2018-13
  CC-MAIN-2018-17 CC-MAIN-2018-22 CC-MAIN-2018-26 CC-MAIN-2018-30
  CC-MAIN-2018-34 CC-MAIN-2018-39 CC-MAIN-2018-43 CC-MAIN-2018-47
  CC-MAIN-2018-51 CC-MAIN-2019-04 CC-MAIN-2019-09 CC-MAIN-2019-13
  CC-MAIN-2019-18 CC-MAIN-2019-22 CC-MAIN-2019-26 CC-MAIN-2019-30
  CC-MAIN-2019-35 CC-MAIN-2019-39 CC-MAIN-2019-43 CC-MAIN-2019-47
  CC-MAIN-2019-51 CC-MAIN-2020-05 CC-MAIN-2020-10 CC-MAIN-2020-16
  CC-MAIN-2020-24 CC-MAIN-2020-29 CC-MAIN-2020-34 CC-MAIN-2020-40
  CC-MAIN-2020-45 CC-MAIN-2020-50 CC-MAIN-2021-04 CC-MAIN-2021-10
  CC-MAIN-2021-17 CC-MAIN-2021-21 CC-MAIN-2021-25 CC-MAIN-2021-31
  CC-MAIN-2021-39 CC-MAIN-2021-43 CC-MAIN-2021-49 CC-MAIN-2022-05
  CC-MAIN-2022-21 CC-MAIN-2022-27 CC-MAIN-2022-33 CC-MAIN-2022-40
  CC-MAIN-2022-49 CC-MAIN-2023-06 CC-MAIN-2023-14 CC-MAIN-2023-23
  CC-MAIN-2023-40 CC-MAIN-2023-50 CC-MAIN-2024-10 CC-MAIN-2024-18
  CC-MAIN-2024-22 CC-MAIN-2024-26 CC-MAIN-2024-30 CC-MAIN-2024-33
  CC-MAIN-2024-38 CC-MAIN-2024-42 CC-MAIN-2024-46 CC-MAIN-2024-51
  CC-MAIN-2025-05 CC-MAIN-2025-08 CC-MAIN-2025-13 CC-MAIN-2025-18
  CC-MAIN-2025-21 CC-MAIN-2025-26 CC-MAIN-2025-30 CC-MAIN-2025-33
  CC-MAIN-2025-38 CC-MAIN-2025-43 CC-MAIN-2025-47 CC-MAIN-2025-51
  CC-MAIN-2026-04 CC-MAIN-2026-08 CC-MAIN-2026-12 CC-MAIN-2026-17
  CC-MAIN-2026-21
)

SNAPSHOT=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Paths and tunables
# ---------------------------------------------------------------------------
VENV=${VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv}
LANCEDB_URI=${LANCEDB_URI:-s3://vjawa-cc-lance}
TABLE_NAME=${TABLE_NAME:-cc_url_index}
REPO=${REPO:-/home/vjawa/nemo-curator-adlr-mm}

# ---------------------------------------------------------------------------
# Credential guards — fail fast if required secrets are absent
# ---------------------------------------------------------------------------
# CC PBSS read credentials (Common Crawl source bucket)
: "${CC_PBSS_ACCESS_KEY_ID:?Need CC_PBSS_ACCESS_KEY_ID}"
: "${CC_PBSS_SECRET_ACCESS_KEY:?Need CC_PBSS_SECRET_ACCESS_KEY}"
export CC_PBSS_ACCESS_KEY_ID CC_PBSS_SECRET_ACCESS_KEY

# PBSS write credentials — team-adlr-codistillation account owns vjawa-cc-lance.
# Set by submitter: sbatch --export=ALL,AWS_ACCESS_KEY_ID=...,AWS_SECRET_ACCESS_KEY=...
# Or pre-export in your shell before calling sbatch.
export AWS_ENDPOINT_URL_S3=${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}
: "${AWS_ACCESS_KEY_ID:?Need AWS_ACCESS_KEY_ID (team-adlr-codistillation)}"
: "${AWS_SECRET_ACCESS_KEY:?Need AWS_SECRET_ACCESS_KEY (team-adlr-codistillation)}"
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------
mkdir -p logs
mkdir -p "/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lancedb_checkpoints/${TABLE_NAME}"

export RAY_TMPDIR=/tmp/ray_vjawa_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR"

export PYTHONPATH=$REPO:${PYTHONPATH:-}

echo "========================================================"
echo "Job        : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Snapshot   : ${SNAPSHOT}"
echo "LanceDB URI: ${LANCEDB_URI}"
echo "Table      : ${TABLE_NAME}"
echo "Node       : $(hostname)"
echo "Start      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Run the indexing script
# ---------------------------------------------------------------------------
cd "$REPO/tutorials/text/cc-lancedb"

"$VENV/bin/python" build_url_index.py \
    --snapshot          "$SNAPSHOT" \
    --lancedb-uri       "$LANCEDB_URI" \
    --table-name        "$TABLE_NAME" \
    --fragment-bytes    8589934592 \
    --num-threads       256 \
    --checkpoint-dir    "/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lancedb_checkpoints/${TABLE_NAME}"

EXIT_CODE=$?
echo "Finished ${SNAPSHOT} at $(date -u +%Y-%m-%dT%H:%M:%SZ) with exit code ${EXIT_CODE}"

# On the last array task (index 120), submit the distributed index build.
# afterokarray=JOBID runs only after ALL 121 write tasks succeed.
if [[ "${SLURM_ARRAY_TASK_ID}" == "120" ]]; then
    echo "All 121 snapshots written. Submitting distributed index build ..."
    sbatch \
        --account="${SLURM_JOB_ACCOUNT}" \
        --dependency="afterokarray:${SLURM_ARRAY_JOB_ID}" \
        --export="ALL,LANCEDB_URI=${LANCEDB_URI},TABLE_NAME=${TABLE_NAME}" \
        slurm/build_lance_index.sh
fi

exit $EXIT_CODE
