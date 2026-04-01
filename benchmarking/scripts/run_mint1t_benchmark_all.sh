#!/usr/bin/env bash
# Run all 4 interleaved IO benchmark paths and print a summary table.
set -euo pipefail

WDS=/raid/vjawa/interleaved_test/mint1t_pdf_cc2024_wds_80shards
PQ=/raid/vjawa/interleaved_test/mint1t_pdf_cc2024_parquet_80shards
BASE=/raid/vjawa/benchamrk_ouput_dir

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source /home/nfs/vjawa/.bashrc
source /raid/vjawa/NeMo-Curator/.venv/bin/activate

run_path() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "=== $name ==="
    echo "================================================================"
    python multimodal_mint1t_benchmark.py \
        --benchmark-results-path "$BASE/$name" \
        --output-path "$BASE/$name/output" \
        --on-materialize-error error \
        --mode overwrite \
        "$@"
}

run_path wds_to_parquet  --reader-type wds     --writer-format parquet    --input-path "$WDS"
run_path wds_to_wds      --reader-type wds     --writer-format webdataset --input-path "$WDS"
run_path pq_to_parquet   --reader-type parquet --writer-format parquet    --input-path "$PQ"
run_path pq_to_wds       --reader-type parquet --writer-format webdataset --input-path "$PQ"

echo ""
echo "================================================================"
echo "=== SUMMARY ==="
echo "================================================================"
for name in wds_to_parquet wds_to_wds pq_to_parquet pq_to_wds; do
    python - <<EOF
import json, pathlib
p = pathlib.Path("$BASE/$name/metrics.json")
if not p.exists():
    print("$name: no metrics.json found")
else:
    m = json.loads(p.read_text())
    rows_in  = m.get("input_num_rows", "N/A")
    mb_in    = m.get("input_total_mb", 0)
    rows_out = m.get("num_rows", "N/A")
    mb_out   = m.get("output_total_mb", 0)
    elapsed  = m["time_taken_s"]
    samples  = m.get("modality_counts", {}).get("metadata", "N/A")
    sps      = int(samples) / elapsed if isinstance(samples, int) and elapsed > 0 else 0
    print(f"$name")
    print(f"  success={m['is_success']}  time={elapsed:.1f}s  throughput={sps:.0f} samples/s")
    print(f"  input  rows={rows_in}  size={mb_in:.1f} MB  files={m.get('input_num_files','N/A')}")
    print(f"  output rows={rows_out}  size={mb_out:.1f} MB  files={m.get('num_output_files','N/A')}")
    print(f"  output modalities={m.get('modality_counts')}")
    print(f"  materialize_errors={m.get('materialize_error_count', 'N/A')}")
EOF
done
