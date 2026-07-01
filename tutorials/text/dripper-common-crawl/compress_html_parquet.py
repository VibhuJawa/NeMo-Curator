#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Convert raw-HTML parquet shards to per-row zlib-compressed HTML."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.text.experimental.dripper._html_compression import (
    HTML_CHARS_COL,
    HTML_COL,
    HTML_ZLIB_COL,
    coerce_html_text,
    compress_html_zlib,
)


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("shard_*.parquet")) or sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")
    return files


def _compress_one(value: object) -> tuple[bytes, int]:
    html = coerce_html_text(value)
    return compress_html_zlib(html), len(html)


def _convert_batch(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    out = df.copy()
    if HTML_ZLIB_COL in out.columns:
        if HTML_CHARS_COL not in out.columns:
            out[HTML_CHARS_COL] = [len(coerce_html_text(v)) for v in out[HTML_ZLIB_COL].tolist()]
    elif HTML_COL in out.columns:
        values = out[HTML_COL].tolist()
        if workers > 1 and len(values) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pairs = list(pool.map(_compress_one, values, chunksize=max(1, len(values) // (workers * 4))))
        else:
            pairs = [_compress_one(v) for v in values]
        out[HTML_ZLIB_COL] = [p[0] for p in pairs]
        out[HTML_CHARS_COL] = [p[1] for p in pairs]
    else:
        raise ValueError(f"Input batch is missing required column: {HTML_ZLIB_COL!r}")
    return out.drop(columns=[HTML_COL], errors="ignore")


def convert_file(input_path: Path, output_path: Path, batch_size: int, workers: int) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    if tmp.exists():
        tmp.unlink()

    pf = pq.ParquetFile(str(input_path))
    schema_names = set(pf.schema_arrow.names)
    if HTML_ZLIB_COL not in schema_names and HTML_COL not in schema_names:
        raise ValueError(f"{input_path} is missing required column: {HTML_ZLIB_COL!r}")

    writer: pq.ParquetWriter | None = None
    rows = 0
    html_chars = 0
    t0 = time.perf_counter()
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            out_df = _convert_batch(batch.to_pandas(), workers=workers)
            rows += len(out_df)
            if HTML_CHARS_COL in out_df.columns:
                html_chars += int(pd.to_numeric(out_df[HTML_CHARS_COL], errors="coerce").fillna(0).sum())
            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
            writer.write_table(table)
            logger.info("converted {:,}/{:,} rows from {}", rows, pf.metadata.num_rows, input_path.name)
    finally:
        if writer is not None:
            writer.close()

    tmp.rename(output_path)
    elapsed_s = time.perf_counter() - t0
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": rows,
        "html_chars": html_chars,
        "elapsed_s": round(elapsed_s, 3),
        "input_bytes": input_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
    }


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    inp = Path(args.input)
    out = Path(args.output)
    files = _input_files(inp)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, src in enumerate(files):
        dst = out / src.name
        if dst.exists() and not args.overwrite:
            logger.info("SKIP {} already exists", dst)
            continue
        logger.info("Converting {}/{}: {} -> {}", i + 1, len(files), src, dst)
        summaries.append(convert_file(src, dst, args.batch_size, args.workers))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(inp),
        "output": str(out),
        "files": summaries,
        "total_rows": int(sum(item["rows"] for item in summaries)),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "html_column": HTML_ZLIB_COL,
        "html_codec": "zlib",
    }
    (out / "_compression_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote compressed HTML dataset to {}", out)


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Input parquet file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
