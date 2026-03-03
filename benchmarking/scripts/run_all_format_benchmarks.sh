#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="${RESULTS_BASE:-/raid/vjawa/NeMo-Curator/benchmark_results}"
MINT1T_WDS_PATH="${MINT1T_WDS_PATH:-/datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/}"
OBELICS_PATH="${OBELICS_PATH:-/datasets/vjawa/obelics_raw_10gb/data}"
EXECUTOR="${EXECUTOR:-xenna}"

DO_CLEAN=false
DO_MATERIALIZE=false
DO_VERIFY_LANCE=false

for arg in "$@"; do
    case "$arg" in
        --clean) DO_CLEAN=true ;;
        --materialization) DO_MATERIALIZE=true ;;
        --verify-lance) DO_VERIFY_LANCE=true ;;
        --results-base=*) RESULTS_BASE="${arg#*=}" ;;
        --help|-h)
            echo "Usage: $0 [--clean] [--materialization] [--verify-lance] [--results-base=PATH]"
            echo ""
            echo "  --clean             Remove previous results before running"
            echo "  --materialization   Also run materialization variants (MINT-1T only)"
            echo "  --verify-lance      Verify Lance shard row counts after runs"
            echo "  --results-base=PATH Override results directory (default: $RESULTS_BASE)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

log() { echo ""; echo "===== $(date '+%Y-%m-%d %H:%M:%S') $* ====="; echo ""; }

if [ "$DO_CLEAN" = true ]; then
    log "Cleaning previous results at $RESULTS_BASE"
    rm -rf "$RESULTS_BASE"
fi

mkdir -p "$RESULTS_BASE"

# --------------------------------------------------------------------------
# Dataset A: MINT-1T WebDataset (10GB tar shards)
# --------------------------------------------------------------------------

# A1: WDS -> Parquet
log "A1: MINT-1T WDS -> Parquet"
python "$SCRIPT_DIR/multimodal_mint1t_benchmark.py" \
    --benchmark-results-path "$RESULTS_BASE/mint1t_wds_to_parquet" \
    --executor "$EXECUTOR" \
    --input-path "$MINT1T_WDS_PATH" \
    --output-path "$RESULTS_BASE/mint1t_wds_to_parquet/output" \
    --no-materialize-on-read \
    --no-materialize-on-write \
    --mode overwrite

MINT1T_PQ="$RESULTS_BASE/mint1t_wds_to_parquet/output"

# A2: Parquet -> WebDataset
log "A2: MINT-1T Parquet -> WebDataset"
python "$SCRIPT_DIR/multimodal_parquet_to_wds_benchmark.py" \
    --benchmark-results-path "$RESULTS_BASE/mint1t_pq_to_wds" \
    --executor "$EXECUTOR" \
    --input-path "$MINT1T_PQ" \
    --output-path "$RESULTS_BASE/mint1t_pq_to_wds/output" \
    --no-materialize-on-write \
    --on-materialize-error warn \
    --mode overwrite

# A3: Parquet -> Lance
log "A3: MINT-1T Parquet -> Lance"
python "$SCRIPT_DIR/multimodal_parquet_3format_benchmark.py" \
    --benchmark-results-path "$RESULTS_BASE/mint1t_pq_to_lance" \
    --executor "$EXECUTOR" \
    --input-path "$MINT1T_PQ" \
    --output-path "$RESULTS_BASE/mint1t_pq_to_lance/output" \
    --formats lance \
    --no-filter \
    --no-materialize-on-write \
    --on-materialize-error warn \
    --mode overwrite

# --------------------------------------------------------------------------
# Materialization variants (MINT-1T only -- has local tar source_refs)
# --------------------------------------------------------------------------
if [ "$DO_MATERIALIZE" = true ]; then
    # A1m: WDS -> Parquet (materialize)
    log "A1m: MINT-1T WDS -> Parquet (materialize-on-write)"
    python "$SCRIPT_DIR/multimodal_mint1t_benchmark.py" \
        --benchmark-results-path "$RESULTS_BASE/mint1t_wds_to_parquet_materialize" \
        --executor "$EXECUTOR" \
        --input-path "$MINT1T_WDS_PATH" \
        --output-path "$RESULTS_BASE/mint1t_wds_to_parquet_materialize/output" \
        --materialize-on-write \
        --mode overwrite

    MINT1T_PQ_MAT="$RESULTS_BASE/mint1t_wds_to_parquet_materialize/output"

    # A2m: Parquet -> WebDataset (materialize)
    log "A2m: MINT-1T Parquet -> WebDataset (materialize-on-write)"
    python "$SCRIPT_DIR/multimodal_parquet_to_wds_benchmark.py" \
        --benchmark-results-path "$RESULTS_BASE/mint1t_pq_to_wds_materialize" \
        --executor "$EXECUTOR" \
        --input-path "$MINT1T_PQ_MAT" \
        --output-path "$RESULTS_BASE/mint1t_pq_to_wds_materialize/output" \
        --materialize-on-write \
        --on-materialize-error warn \
        --mode overwrite

    # A3m: Parquet -> Lance (materialize)
    log "A3m: MINT-1T Parquet -> Lance (materialize-on-write)"
    python "$SCRIPT_DIR/multimodal_parquet_3format_benchmark.py" \
        --benchmark-results-path "$RESULTS_BASE/mint1t_pq_to_lance_materialize" \
        --executor "$EXECUTOR" \
        --input-path "$MINT1T_PQ_MAT" \
        --output-path "$RESULTS_BASE/mint1t_pq_to_lance_materialize/output" \
        --formats lance \
        --no-filter \
        --materialize-on-write \
        --on-materialize-error warn \
        --mode overwrite
fi

# --------------------------------------------------------------------------
# Dataset B: OBELICS raw parquet (~10GB)
# --------------------------------------------------------------------------

# B1: OBELICS -> Parquet
log "B1: OBELICS Parquet -> Parquet"
python "$SCRIPT_DIR/multimodal_parquet_3format_benchmark.py" \
    --benchmark-results-path "$RESULTS_BASE/obelics_pq_to_parquet" \
    --executor "$EXECUTOR" \
    --input-path "$OBELICS_PATH" \
    --output-path "$RESULTS_BASE/obelics_pq_to_parquet/output" \
    --formats parquet \
    --reader-type obelics \
    --no-filter \
    --no-materialize-on-write \
    --on-materialize-error warn \
    --mode overwrite

# B2: OBELICS -> WebDataset
log "B2: OBELICS Parquet -> WebDataset"
python "$SCRIPT_DIR/multimodal_parquet_3format_benchmark.py" \
    --benchmark-results-path "$RESULTS_BASE/obelics_pq_to_wds" \
    --executor "$EXECUTOR" \
    --input-path "$OBELICS_PATH" \
    --output-path "$RESULTS_BASE/obelics_pq_to_wds/output" \
    --formats webdataset \
    --reader-type obelics \
    --no-filter \
    --no-materialize-on-write \
    --on-materialize-error warn \
    --mode overwrite

# B3: OBELICS -> Lance
log "B3: OBELICS Parquet -> Lance"
python "$SCRIPT_DIR/multimodal_parquet_3format_benchmark.py" \
    --benchmark-results-path "$RESULTS_BASE/obelics_pq_to_lance" \
    --executor "$EXECUTOR" \
    --input-path "$OBELICS_PATH" \
    --output-path "$RESULTS_BASE/obelics_pq_to_lance/output" \
    --formats lance \
    --reader-type obelics \
    --no-filter \
    --no-materialize-on-write \
    --on-materialize-error warn \
    --mode overwrite

# --------------------------------------------------------------------------
# Verify Lance shards
# --------------------------------------------------------------------------
if [ "$DO_VERIFY_LANCE" = true ]; then
    log "Verifying Lance shard integrity"
    python -c "
import lance
from pathlib import Path

for name, base in [
    ('mint1t', '$RESULTS_BASE/mint1t_pq_to_lance/output/lance'),
    ('obelics', '$RESULTS_BASE/obelics_pq_to_lance/output/lance'),
]:
    p = Path(base)
    if not p.exists():
        print(f'{name}: SKIPPED (directory not found)')
        continue
    lance_dirs = sorted(d for d in p.rglob('*.lance') if d.is_dir())
    row_counts = [lance.dataset(str(d)).count_rows() for d in lance_dirs]
    total = sum(row_counts)
    print(f'{name}: {len(lance_dirs)} shards, {total} total rows, '
          f'min={min(row_counts)}, max={max(row_counts)}, avg={total/len(row_counts):.0f}')
"
    if [ "$DO_MATERIALIZE" = true ]; then
        python -c "
import lance
from pathlib import Path

for name, base in [
    ('mint1t_mat', '$RESULTS_BASE/mint1t_pq_to_lance_materialize/output/lance'),
]:
    p = Path(base)
    if not p.exists():
        print(f'{name}: SKIPPED (directory not found)')
        continue
    lance_dirs = sorted(d for d in p.rglob('*.lance') if d.is_dir())
    row_counts = [lance.dataset(str(d)).count_rows() for d in lance_dirs]
    total = sum(row_counts)
    print(f'{name}: {len(lance_dirs)} shards, {total} total rows, '
          f'min={min(row_counts)}, max={max(row_counts)}, avg={total/len(row_counts):.0f}')
"
    fi
fi

# --------------------------------------------------------------------------
# Generate comparison
# --------------------------------------------------------------------------
log "Generating comparison markdown"
python "$SCRIPT_DIR/generate_benchmark_comparison.py" \
    --results-base "$RESULTS_BASE" \
    --output "$RESULTS_BASE/COMPARISON.md"

log "All benchmarks complete. Results in $RESULTS_BASE"
