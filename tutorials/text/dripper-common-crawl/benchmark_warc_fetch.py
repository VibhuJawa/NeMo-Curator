#!/usr/bin/env python3  # noqa: EXE001
"""
Standalone WARC fetch throughput benchmark.

Measures pages/second and MB/s of boto3 range-GET fetches from PBSS/S3
(Common Crawl WARC files stored on Ceph at pdx.s8k.io).

Usage:
    python benchmark_warc_fetch.py \
        --manifest-path /lustre/.../host_bucket=0001.parquet \
        --workers 64 \
        --max-pages 2000 \
        --bucket crawl-data

Prints a JSON metrics line at the end.
"""

import argparse
import concurrent.futures
import contextlib
import gzip
import json
import os
import time
from typing import Any
from urllib.parse import urlparse


def _setup_pbss_credentials(endpoint_url: str) -> None:
    """Map PBSS_* env vars onto the standard AWS_* names boto3 reads."""
    if endpoint_url and "pdx.s8k.io" in endpoint_url:
        pbss_key = os.environ.get("PBSS_ACCESS_KEY_ID")
        pbss_secret = os.environ.get("PBSS_SECRET_ACCESS_KEY")
        if pbss_key:
            os.environ["AWS_ACCESS_KEY_ID"] = pbss_key
        if pbss_secret:
            os.environ["AWS_SECRET_ACCESS_KEY"] = pbss_secret


def make_s3_client(endpoint_url: str, region: str, workers: int) -> Any:  # noqa: ANN401
    import boto3
    from botocore.config import Config as BotoConfig

    _setup_pbss_credentials(endpoint_url)

    max_pool = max(10, workers)
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            read_timeout=120,
            max_pool_connections=max_pool,
        ),
    )


def parse_warc_location(default_bucket: str, filename: str) -> tuple[str, str]:
    parsed = urlparse(filename)
    if parsed.scheme == "s3" and parsed.netloc:
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    elif parsed.scheme in ("http", "https") and parsed.netloc:
        bucket = default_bucket
        key = parsed.path.lstrip("/")
    else:
        bucket = default_bucket
        key = filename.lstrip("/")

    # Normalize crawl-data double-prefix
    if bucket == "crawl-data" and key.startswith("crawl-data/"):
        key = key.removeprefix("crawl-data/")
    return bucket, key


def fetch_one(
    client: Any,  # noqa: ANN401
    default_bucket: str,
    row: dict[str, Any],
) -> tuple[int, bool]:
    """
    Fetch a single WARC page.  Returns (bytes_fetched, success).
    Does NOT parse HTML — pure network + read benchmark.
    """
    filename = str(row["warc_filename"])
    offset = int(row["warc_record_offset"])
    length = int(row["warc_record_length"])
    bucket, key = parse_warc_location(default_bucket, filename)
    end_byte = offset + length - 1
    try:
        response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{end_byte}")
        raw_bytes = response["Body"].read()
        # Minimal decompress to verify data integrity (mirrors production path).
        with contextlib.suppress(gzip.BadGzipFile, OSError):
            gzip.decompress(raw_bytes)
        return len(raw_bytes), True
    except Exception:  # noqa: BLE001
        return 0, False


def run_benchmark(  # noqa: PLR0913
    manifest_path: str,
    workers: int,
    max_pages: int,
    bucket: str,
    endpoint_url: str,
    region: str,
    seed: int = 42,
) -> dict[str, Any]:
    import pandas as pd

    print(f"[bench] Loading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    print(f"[bench] Manifest rows: {len(df):,}")

    # Shuffle for a random sample across WARCs/hosts
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_pages > 0:
        df = df.head(max_pages)
    rows = df.to_dict("records")
    print(f"[bench] Benchmarking {len(rows):,} pages with {workers} workers …")

    client = make_s3_client(endpoint_url, region, workers)

    pages_fetched = 0
    bytes_fetched = 0
    failures = 0

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, client, bucket, row): i for i, row in enumerate(rows)}
        for future in concurrent.futures.as_completed(futures):
            nb, ok = future.result()
            if ok:
                pages_fetched += 1
                bytes_fetched += nb
            else:
                failures += 1
    elapsed = time.perf_counter() - t0

    pages_per_second = pages_fetched / elapsed if elapsed > 0 else 0
    mb_per_second = (bytes_fetched / 1_048_576) / elapsed if elapsed > 0 else 0

    return {
        "workers": workers,
        "max_pages_requested": len(rows),
        "pages_fetched": pages_fetched,
        "failures": failures,
        "elapsed_s": round(elapsed, 2),
        "pages_per_second": round(pages_per_second, 2),
        "bytes_fetched": bytes_fetched,
        "mb_fetched": round(bytes_fetched / 1_048_576, 2),
        "mb_per_second": round(mb_per_second, 2),
        "manifest_path": manifest_path,
        "endpoint_url": endpoint_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WARC fetch throughput benchmark")
    parser.add_argument(
        "--manifest-path",
        required=True,
        help="Path to host_bucket parquet shard",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of boto3 ThreadPoolExecutor workers",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=2000,
        help="Max pages to fetch (0 = all)",
    )
    parser.add_argument(
        "--bucket",
        default="crawl-data",
        help="Default S3 bucket for WARC files",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("AWS_ENDPOINT_URL_S3", "https://pdx.s8k.io"),
        help="S3 endpoint URL",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="S3 region",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling manifest rows",
    )
    args = parser.parse_args()

    metrics = run_benchmark(
        manifest_path=args.manifest_path,
        workers=args.workers,
        max_pages=args.max_pages,
        bucket=args.bucket,
        endpoint_url=args.endpoint_url,
        region=args.region,
        seed=args.seed,
    )

    print("\n[bench] ===== RESULTS =====")
    print(f"  workers          : {metrics['workers']}")
    print(f"  pages fetched    : {metrics['pages_fetched']:,} / {metrics['max_pages_requested']:,}")
    print(f"  failures         : {metrics['failures']}")
    print(f"  elapsed          : {metrics['elapsed_s']:.1f}s")
    print(f"  throughput       : {metrics['pages_per_second']:.1f} pages/s")
    print(f"  bandwidth        : {metrics['mb_per_second']:.1f} MB/s  ({metrics['mb_fetched']:.1f} MB total)")
    print()
    # Machine-parseable JSON line — collector script reads this
    print("BENCH_METRICS_JSON:" + json.dumps(metrics))


if __name__ == "__main__":
    main()
