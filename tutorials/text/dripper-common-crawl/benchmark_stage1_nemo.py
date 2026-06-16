#!/usr/bin/env python3  # noqa: EXE001
"""
Stage 1 benchmark using NeMo Curator CommonCrawlWARCReader + Ray Data.

Distributes WARC fetch across Ray actors in parallel. Each actor processes
one chunk of rows using CommonCrawlWARCReader's ThreadPoolExecutor (default 256
threads per actor → high S3 concurrency).

Output: sharded parquet with binary_content column + all original manifest
columns. Stage 2 reads binary_content directly (line 2486 in main.py).

Usage:
    python benchmark_stage1_nemo.py \\
        --manifest-path /path/to/host_bucket=0001.parquet \\
        --output-dir /path/to/stage1_nemo/shard_0001 \\
        --chunk-size 20000 --max-workers 256 --num-cpus 64

Env (inherited from Slurm scripts / cache_env.sh):
    AWS_ENDPOINT_URL_S3   https://pdx.s8k.io  (PBSS)
    AWS_ACCESS_KEY_ID     PBSS key
    AWS_SECRET_ACCESS_KEY PBSS secret
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import ray
import ray.data

from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader


class WARCFetchActor:
    """Ray actor wrapping CommonCrawlWARCReader for parallel WARC fetch.

    Instantiated once per actor process; reuses the S3 client across calls.
    Calls _read_warc_records_batch directly to avoid DocumentBatch construction
    (task_id is a required positional arg in the current Task base class, which
    is being removed upstream — see PR #2036).
    """

    def __init__(self, max_workers: int, s3_bucket: str, s3_key_prefix: str) -> None:
        self.reader = CommonCrawlWARCReader(
            use_s3=True,
            s3_bucket=s3_bucket,
            s3_key_prefix=s3_key_prefix,
            max_workers=max_workers,
            drop_failed=False,
        )

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch = batch.copy()
        batch["binary_content"] = self.reader._read_warc_records_batch(batch)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 NeMo Curator WARC fetch benchmark")
    parser.add_argument("--manifest-path", required=True, help="Path to host_bucket .parquet shard")
    parser.add_argument("--output-dir", required=True, help="Output directory for sharded parquet")
    parser.add_argument("--chunk-size", type=int, default=20000, help="Rows per Ray actor batch")
    parser.add_argument("--max-workers", type=int, default=256, help="Boto3 threads per actor")
    parser.add_argument("--num-cpus", type=int, default=64, help="Ray CPUs / max concurrent actors")
    parser.add_argument("--ray-tmpdir", default="/tmp/ray_s1_nemo", help="Ray temp dir")  # noqa: S108
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for testing (0=all)")
    parser.add_argument("--s3-bucket", default="crawl-data", help="S3 bucket name")
    parser.add_argument("--s3-key-prefix", default="crawl-data/", help="Prefix to strip from warc_filename")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ray.init(
        num_cpus=args.num_cpus,
        include_dashboard=False,
        _temp_dir=args.ray_tmpdir,
        ignore_reinit_error=True,
    )

    df = pd.read_parquet(args.manifest_path)
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    total_rows = len(df)
    num_blocks = max(1, (total_rows + args.chunk_size - 1) // args.chunk_size)
    concurrency = min(num_blocks, args.num_cpus)

    print(f"[stage1-nemo] Manifest: {args.manifest_path} ({total_rows} rows)")
    print(f"[stage1-nemo] Chunk size: {args.chunk_size} → {num_blocks} blocks, concurrency={concurrency}")
    print(f"[stage1-nemo] Threads per actor: {args.max_workers}")
    print(f"[stage1-nemo] Max concurrent S3 requests: {concurrency * args.max_workers:,}")
    print(f"[stage1-nemo] Output: {output_dir}")
    print(f"[stage1-nemo] S3: bucket={args.s3_bucket} prefix={args.s3_key_prefix!r}")

    start = time.monotonic()

    ds = ray.data.from_pandas(df).repartition(num_blocks)
    ds = ds.map_batches(
        WARCFetchActor,
        fn_constructor_args=(args.max_workers, args.s3_bucket, args.s3_key_prefix),
        batch_size=None,
        batch_format="pandas",
        concurrency=concurrency,
        num_cpus=1,
    )
    ds.write_parquet(str(output_dir))

    elapsed = time.monotonic() - start

    output_files = sorted(output_dir.glob("*.parquet"))
    try:
        output_rows = sum(len(pd.read_parquet(f, columns=["url"])) for f in output_files)
    except Exception:  # noqa: BLE001
        output_rows = -1

    metrics = {
        "total_input_rows": total_rows,
        "output_rows": output_rows,
        "failed_rows": total_rows - output_rows if output_rows >= 0 else -1,
        "elapsed_s": round(elapsed, 2),
        "rows_per_second": round(output_rows / elapsed, 1) if elapsed > 0 and output_rows >= 0 else 0,
        "chunk_size": args.chunk_size,
        "num_blocks": num_blocks,
        "concurrency": concurrency,
        "max_workers_per_actor": args.max_workers,
        "num_cpus": args.num_cpus,
        "max_concurrent_s3_requests": concurrency * args.max_workers,
        "num_output_files": len(output_files),
    }
    (output_dir / "stage1_nemo_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(
        f"\n[stage1-nemo] Done: {output_rows}/{total_rows} rows in {elapsed:.1f}s ({metrics['rows_per_second']} rows/s)"
    )


if __name__ == "__main__":
    main()
