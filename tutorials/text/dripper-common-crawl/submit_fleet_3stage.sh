#!/usr/bin/env bash
# =============================================================================
# submit_fleet_3stage.sh — Fleet submission wrapper for run_mineru_pipeline.sh
#
# Usage:
#   bash submit_fleet_3stage.sh <SEGMENT>
#
#   SEGMENT — integer 0–7; each segment covers 100 host_bucket parquet files
#
# What it does:
#   1. Selects 100 host_bucket parquets from the sorted bucket directory
#      (files are named host_bucket_NNNN.parquet, sorted lexicographically)
#   2. Merges them with PyArrow into a single manifest parquet under OUTPUT_BASE
#   3. Calls run_mineru_pipeline.sh <merged_manifest> <output_dir> fleet
#
# Example: process segments 0–7 to cover all 800 host_bucket files
#   for seg in {0..7}; do bash submit_fleet_3stage.sh $seg; done
# =============================================================================

set -euo pipefail

SEGMENT="${1:?Usage: $0 <SEGMENT_NUMBER (0-7)>}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST_BUCKET_DIR="/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_sorted_host_buckets_20260611"
OUTPUT_BASE="/lustre/fsw/portfolios/llmservice/users/vjawa/fleet_pipeline_3stage"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_CPU="/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv"
PYTHON_CPU="${VENV_CPU}/bin/python3"

BUCKETS_PER_SEGMENT=100

# ---------------------------------------------------------------------------
# Validate segment
# ---------------------------------------------------------------------------
if ! [[ "${SEGMENT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEGMENT must be a non-negative integer, got: '${SEGMENT}'" >&2
    exit 1
fi

START_IDX=$(( SEGMENT * BUCKETS_PER_SEGMENT ))
END_IDX=$(( START_IDX + BUCKETS_PER_SEGMENT - 1 ))   # inclusive

echo "[fleet] Segment ${SEGMENT}: host_bucket files ${START_IDX}–${END_IDX}"

# ---------------------------------------------------------------------------
# Locate source host_bucket parquet files
# ---------------------------------------------------------------------------
# Enumerate all parquets in sorted order, then slice [START_IDX, END_IDX]
mapfile -t ALL_BUCKETS < <(find "${HOST_BUCKET_DIR}" -maxdepth 1 -name '*.parquet' | sort)

TOTAL_BUCKETS="${#ALL_BUCKETS[@]}"
echo "[fleet] Total host_bucket files found: ${TOTAL_BUCKETS}"

if (( START_IDX >= TOTAL_BUCKETS )); then
    echo "ERROR: SEGMENT ${SEGMENT} (start_idx=${START_IDX}) exceeds total files (${TOTAL_BUCKETS})." >&2
    exit 1
fi

# Slice: bash array is 0-based
SLICE=( "${ALL_BUCKETS[@]:${START_IDX}:${BUCKETS_PER_SEGMENT}}" )
N_SELECTED="${#SLICE[@]}"
echo "[fleet] Selected ${N_SELECTED} files for segment ${SEGMENT}"
echo "[fleet]   First: ${SLICE[0]}"
echo "[fleet]   Last:  ${SLICE[-1]}"

# ---------------------------------------------------------------------------
# Merge selected parquets into a single manifest
# ---------------------------------------------------------------------------
SEGMENT_DIR="${OUTPUT_BASE}/seg_$(printf '%02d' "${SEGMENT}")"
mkdir -p "${SEGMENT_DIR}"
MERGED_MANIFEST="${SEGMENT_DIR}/merged_manifest.parquet"

if [[ -f "${MERGED_MANIFEST}" ]]; then
    echo "[fleet] Merged manifest already exists — reusing: ${MERGED_MANIFEST}"
else
    echo "[fleet] Merging ${N_SELECTED} host_bucket parquets → ${MERGED_MANIFEST} ..."

    # Write the file list to a temp file so we don't exceed ARG_MAX
    FILELIST=$(mktemp /tmp/fleet_filelist_XXXXXX.txt)
    printf '%s\n' "${SLICE[@]}" > "${FILELIST}"

    "${PYTHON_CPU}" - "${FILELIST}" "${MERGED_MANIFEST}" <<'PYEOF'
import sys
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq

filelist_path = sys.argv[1]
out_path      = sys.argv[2]

with open(filelist_path) as f:
    files = [l.strip() for l in f if l.strip()]

print(f"[merge] Reading {len(files)} parquet files...")
tables = []
for i, fpath in enumerate(files):
    try:
        tbl = pq.read_table(fpath)
        tables.append(tbl)
        if (i + 1) % 20 == 0:
            print(f"[merge]   {i+1}/{len(files)} loaded")
    except Exception as exc:
        print(f"[merge] WARNING: skipping {fpath}: {exc}", file=sys.stderr)

if not tables:
    print("ERROR: no tables loaded — check HOST_BUCKET_DIR path", file=sys.stderr)
    sys.exit(1)

merged = pa.concat_tables(tables, promote_options="default")
print(f"[merge] Merged: {len(merged):,} rows from {len(tables)} files")

tmp = out_path + ".tmp"
pq.write_table(merged, tmp, compression="snappy")
pathlib.Path(tmp).rename(out_path)
print(f"[merge] Written: {out_path}")
PYEOF

    rm -f "${FILELIST}"
    echo "[fleet] Merge complete: ${MERGED_MANIFEST}"
fi

# ---------------------------------------------------------------------------
# Launch 3-stage pipeline on merged manifest
# ---------------------------------------------------------------------------
PIPELINE_OUTPUT="${SEGMENT_DIR}/pipeline_output"
mkdir -p "${PIPELINE_OUTPUT}"

echo "[fleet] Launching run_mineru_pipeline.sh for segment ${SEGMENT}..."
echo "[fleet]   INPUT:  ${MERGED_MANIFEST}"
echo "[fleet]   OUTPUT: ${PIPELINE_OUTPUT}"
echo "[fleet]   MODE:   fleet"

bash "${SCRIPT_DIR}/run_mineru_pipeline.sh" \
    "${MERGED_MANIFEST}" \
    "${PIPELINE_OUTPUT}" \
    fleet
