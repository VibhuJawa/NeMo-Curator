#!/usr/bin/env python3
"""
merge_mineru_shards.py — Concatenate shard_NNNN_of_MMMM.parquet files from
a MinerU-HTML array job into a single dripper_results.parquet + merged metrics.json.

Usage:
  python merge_mineru_shards.py --input-dir /lustre/.../output --output /lustre/.../dripper_results.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)

    shards = sorted(input_dir.glob("shard_*_of_*.parquet"))
    if not shards:
        print(f"ERROR: no shard_*_of_*.parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shards)} shard files in {input_dir}")

    tables = []
    for s in shards:
        t = pq.ParquetFile(s).read()
        tables.append(t)
        print(f"  {s.name}: {len(t):,} rows")

    combined = pa.concat_tables(tables)
    print(f"\nTotal rows: {len(combined):,}")

    pq.write_table(combined, str(out_path), compression="snappy")
    print(f"Written: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Merge metrics
    metric_files = sorted(input_dir.glob("metrics_shard_*.json"))
    if metric_files:
        all_metrics = [json.loads(p.read_text()) for p in metric_files]
        total_pages = sum(m.get("total_pages", 0) for m in all_metrics)
        total_errors = sum(m.get("error_pages", 0) for m in all_metrics)
        total_inf = sum(m.get("inference_s", 0) for m in all_metrics)
        avg_tput = sum(m.get("throughput_pages_per_s", 0) for m in all_metrics) / len(all_metrics)
        merged = {
            "extractor": "MinerU-HTML-standalone-array",
            "model": all_metrics[0].get("model", ""),
            "input_manifest_path": all_metrics[0].get("input_manifest_path", ""),
            "num_shards": len(all_metrics),
            "total_pages": total_pages,
            "successful_pages": total_pages - total_errors,
            "error_pages": total_errors,
            "total_inference_s": total_inf,
            "avg_throughput_per_gpu": avg_tput,
            "output_parquet": str(out_path),
        }
        merged_metrics_path = out_path.parent / "metrics.json"
        merged_metrics_path.write_text(json.dumps(merged, indent=2))
        print(f"Merged metrics: {merged_metrics_path}")
        print(f"  total_pages={total_pages:,}  errors={total_errors}  avg_tput={avg_tput:.1f} pages/s/gpu")


if __name__ == "__main__":
    main()
