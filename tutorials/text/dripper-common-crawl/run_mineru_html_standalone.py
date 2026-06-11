#!/usr/bin/env python3
"""
run_mineru_html_standalone.py

Pure MinerU-HTML baseline — runs the upstream library directly on pages from
a manifest parquet, with no NeMo Curator infrastructure.

This is the true "Dripper standalone" baseline:
  - Reads pages from a manifest (url, html columns)
  - Optionally fetches HTML from WARCs if html column is missing
  - Batches pages and calls MinerUHTML.process() directly
  - Writes results to a parquet + metrics JSON

Usage (Slurm):
  python run_mineru_html_standalone.py \
    --input   /lustre/.../layout_precompute_manifest.parquet \
    --output  /lustre/.../mineru_standalone_output \
    --max-pages 2000 \
    --batch-size 64 \
    --model opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact
"""
import argparse, json, os, sys, time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def read_parquet(path):
    return pq.ParquetFile(str(path)).read().to_pandas()


def coerce_html(raw):
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,  help="Input manifest parquet (must have url + html columns)")
    parser.add_argument("--output",     required=True,  help="Output directory")
    parser.add_argument("--max-pages",  type=int, default=0, help="0 = all pages")
    parser.add_argument("--batch-size", type=int, default=32, help="Pages per MinerUHTML batch")
    parser.add_argument("--model",      default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    parser.add_argument("--hf-cache",   default=os.environ.get("HF_HOME", "/lustre/fsw/portfolios/llmservice/users/vjawa/hf_cache"))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    print(f"[mineru_standalone] input:      {args.input}")
    print(f"[mineru_standalone] output:     {args.output}")
    print(f"[mineru_standalone] max_pages:  {args.max_pages or 'all'}")
    print(f"[mineru_standalone] batch_size: {args.batch_size}")
    print(f"[mineru_standalone] model:      {args.model}")
    print(f"[mineru_standalone] hf_cache:   {args.hf_cache}")
    print()

    # ── Load input ────────────────────────────────────────────────────────────
    print("[mineru_standalone] loading manifest...")
    df = read_parquet(args.input)
    if args.max_pages > 0:
        df = df.head(args.max_pages)
    print(f"[mineru_standalone] {len(df):,} pages to process")

    if "html" not in df.columns:
        print("[mineru_standalone] ERROR: manifest missing 'html' column. Need WARC fetch first.", file=sys.stderr)
        sys.exit(1)

    # ── Load MinerU-HTML ──────────────────────────────────────────────────────
    print("[mineru_standalone] loading MinerUHTML extractor...")
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    from mineru_html import MinerUHTML
    extractor = MinerUHTML(model_path=args.model)

    t_load = time.perf_counter()
    print(f"[mineru_standalone] extractor ready in {t_load-t_start:.1f}s")

    # ── Run inference in batches ──────────────────────────────────────────────
    rows = df.to_dict("records")
    results = []
    errors = 0

    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        html_list = [coerce_html(r.get("html", "")) for r in batch]

        t0 = time.perf_counter()
        try:
            batch_results = extractor.process(html_list)
        except Exception as e:
            print(f"[mineru_standalone] batch {batch_start//args.batch_size} ERROR: {e}", file=sys.stderr)
            batch_results = [None] * len(batch)
            errors += len(batch)

        elapsed = time.perf_counter() - t0

        for row, result in zip(batch, batch_results):
            if result is not None:
                try:
                    main_content = str(result.output_data.main_content or "")
                    main_html    = str(getattr(result.output_data, "main_html", "") or "")
                    error        = ""
                except Exception as e:
                    main_content = ""
                    main_html    = ""
                    error        = str(e)[:200]
                    errors += 1
            else:
                main_content = ""
                main_html    = ""
                error        = "batch_failed"

            results.append({
                "url":              row.get("url", ""),
                "url_host_name":    row.get("url_host_name", ""),
                "dripper_layout_id": row.get("dripper_layout_id", ""),
                "dripper_content":   main_content,
                "dripper_html":      main_html,
                "dripper_error":     error,
                "dripper_time_s":    elapsed / len(batch),
            })

        done = min(batch_start + args.batch_size, len(rows))
        rate = done / (time.perf_counter() - t_load) if time.perf_counter() > t_load else 0
        print(f"[mineru_standalone] {done:>6}/{len(rows)} pages  {rate:.1f} pages/s  batch={elapsed:.1f}s")

    # ── Write outputs ─────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    result_df = pd.DataFrame(results)
    out_parquet = output_dir / "dripper_results.parquet"
    result_df.to_parquet(str(out_parquet), index=False, compression="snappy")

    total_s = t_end - t_start
    pages_s = len(rows) / max(t_end - t_load, 1)
    metrics = {
        "extractor":           "MinerU-HTML-standalone",
        "model":               args.model,
        "input_manifest_path": str(args.input),
        "total_pages":         len(rows),
        "successful_pages":    len(rows) - errors,
        "error_pages":         errors,
        "elapsed_s":           total_s,
        "load_s":              t_load - t_start,
        "inference_s":         t_end - t_load,
        "throughput_pages_per_s": pages_s,
        "batch_size":          args.batch_size,
        "output_parquet":      str(out_parquet),
    }

    out_metrics = output_dir / "metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print(f"[mineru_standalone] DONE")
    print(f"  pages:      {len(rows):,}  ({errors} errors)")
    print(f"  elapsed:    {total_s:.1f}s  (load={metrics['load_s']:.1f}s  inference={metrics['inference_s']:.1f}s)")
    print(f"  throughput: {pages_s:.1f} pages/s")
    print(f"  output:     {out_parquet}")
    print(f"  metrics:    {out_metrics}")


if __name__ == "__main__":
    main()
