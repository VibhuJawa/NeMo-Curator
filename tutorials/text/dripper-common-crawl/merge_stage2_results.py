#!/usr/bin/env python3
"""
merge_stage2_results.py — Concatenate Stage 2 shard_NNNN_of_0064.parquet files
into a single inference_results.parquet, and write merged metrics.json.

Usage:
  python merge_stage2_results.py \
    --input-dir /lustre/.../gpu_results \
    --output    /lustre/.../gpu_results/inference_results.parquet

Output parquet columns:
  url, url_host_name, layout_cluster_id, cluster_role, host_bucket,
  dripper_content, dripper_html, dripper_error, dripper_time_s,
  xpath_rules, template_html, inference_time_s

The merged file is what Stage 3 joins against cluster_assignments/ to
propagate XPath rules to siblings.
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Minimum JSON-serialised xpath_rules length that indicates a non-empty rule set
_XPATH_MIN_LEN = 2


def _merge_metrics(out_path: Path, all_metrics: list[dict]) -> None:
    """Write merged metrics.json from per-shard metric dicts."""
    total_pages = sum(m.get("total_pages", 0) for m in all_metrics)
    total_errors = sum(m.get("error_pages", 0) for m in all_metrics)
    total_too_long = sum(m.get("too_long_pages", 0) for m in all_metrics)
    total_inf_s = sum(m.get("inference_s", 0) for m in all_metrics)
    avg_tput = sum(m.get("throughput_pages_per_s", 0) for m in all_metrics) / len(all_metrics)
    merged = {
        "extractor": "MinerU-HTML-stage2-representatives-merged",
        "model": all_metrics[0].get("model", ""),
        "input_path": all_metrics[0].get("input_path", ""),
        "num_shards": len(all_metrics),
        "total_pages": total_pages,
        "successful_pages": total_pages - total_errors - total_too_long,
        "error_pages": total_errors,
        "too_long_pages": total_too_long,
        "total_inference_s": total_inf_s,
        "avg_throughput_per_gpu": avg_tput,
        "estimated_total_throughput": avg_tput * len(all_metrics),
        "output_parquet": str(out_path),
    }
    merged_metrics_path = out_path.parent / "metrics.json"
    merged_metrics_path.write_text(json.dumps(merged, indent=2))
    print(f"\nMerged metrics: {merged_metrics_path}")
    print(
        f"  total_pages={total_pages:,}  "
        f"errors={total_errors:,}  "
        f"too_long={total_too_long:,}  "
        f"avg_tput_per_gpu={avg_tput:.1f} pages/s  "
        f"estimated_total={avg_tput * len(all_metrics):.1f} pages/s"
    )


def _print_column_summary(combined: pa.Table, total_rows: int) -> None:
    """Print a per-column breakdown of the merged parquet table."""
    import pandas as pd  # imported here to keep top-level imports minimal

    df = combined.to_pandas()
    error_counts = df["dripper_error"].value_counts() if "dripper_error" in df.columns else pd.Series(dtype=object)
    has_xpath = int((df["xpath_rules"].str.len() > _XPATH_MIN_LEN).sum()) if "xpath_rules" in df.columns else 0

    print("\nColumn summary:")
    print(f"  Total rows:         {total_rows:,}")
    if "cluster_role" in df.columns:
        print(f"  Representatives:    {(df['cluster_role'] == 'representative').sum():,}")
        print(f"  Singletons/noise:   {(df['cluster_role'] == 'singleton').sum():,}")
    print(f"  With xpath_rules:   {has_xpath:,}")
    if error_counts:
        print("  Error breakdown:")
        for err, cnt in error_counts.head(10).items():
            if err:
                print(f"    {err}: {cnt:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing shard_*_of_*.parquet files")
    parser.add_argument("--output", required=True, help="Output merged parquet path")
    parser.add_argument("--pattern", default="shard_*_of_*.parquet", help="Glob pattern for shard files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(input_dir.glob(args.pattern))
    if not shards:
        # Also try inference_results.parquet from single-shard runs
        single = input_dir / "inference_results.parquet"
        if single.exists():
            shards = [single]
        else:
            print(f"ERROR: no {args.pattern} files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(shards)} shard files in {input_dir}")

    tables = []
    for s in shards:
        try:
            t = pq.ParquetFile(str(s)).read()
            tables.append(t)
            print(f"  {s.name}: {len(t):,} rows")
        except (OSError, ValueError) as exc:
            print(f"  WARNING: could not read {s.name}: {exc}", file=sys.stderr)

    if not tables:
        print("ERROR: no readable shard files found", file=sys.stderr)
        sys.exit(1)

    combined = pa.concat_tables(tables, promote_options="default")
    total_rows = len(combined)
    print(f"\nTotal rows: {total_rows:,}")

    # Atomic write
    tmp_path = out_path.with_suffix(".parquet.tmp")
    pq.write_table(combined, str(tmp_path), compression="snappy")
    tmp_path.rename(out_path)
    print(f"Written: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    _print_column_summary(combined, total_rows)

    # Merge metrics
    metric_files = sorted(input_dir.glob("metrics_shard_*.json"))
    if metric_files:
        all_metrics = [json.loads(p.read_text()) for p in metric_files]
        _merge_metrics(out_path, all_metrics)


if __name__ == "__main__":
    main()
