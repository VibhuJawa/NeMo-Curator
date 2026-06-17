#!/usr/bin/env python3
"""Benchmark PBSS S3 connectivity with multiple clients to isolate the bottleneck.

Run on the compute node (via Slurm) so results reflect actual network path:
    python benchmark_pbss.py

Outputs a table of latency / throughput per client configuration.
"""

from __future__ import annotations

import concurrent.futures
import io
import time
from pathlib import Path
from statistics import mean, median, stdev

import yaml

# ── credentials ──────────────────────────────────────────────────────────────


def _creds() -> tuple[str, str]:
    dm = Path.home() / ".config" / "datamover" / "storage_locations"
    cfg = yaml.safe_load(dm.read_text())
    c = cfg["pdx-commoncrawl"]["secrets"]["local"]
    return c["access_key_id"], c["secret_access_key"]


KEY_ID, SECRET = _creds()
ENDPOINT = "https://pdx.s8k.io"

# A known-good CC index shard (small-ish Parquet — ~900 MB; use for throughput)
SHARD_KEY = (
    "table/cc-main/warc/crawl=CC-MAIN-2025-26/subset=warc/"
    "part-00000-aff71553-4da7-4a36-92b4-e5300d3e4422.c000.gz.parquet"
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _warc_key_and_range(shard_bytes: bytes) -> tuple[str, int, int]:
    """Pull first valid WARC row from the shard to use as a byte-range target."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(io.BytesIO(shard_bytes))
    for batch in pf.iter_batches(
        batch_size=100,
        columns=["warc_filename", "warc_record_offset", "warc_record_length"],
    ):
        df = batch.to_pydict()
        for fn, off, ln in zip(df["warc_filename"], df["warc_record_offset"], df["warc_record_length"]):
            if ln and ln > 1000:
                return fn, off, ln
    raise RuntimeError("No usable WARC row found in shard")


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.2f}s"


def _report(label: str, times: list[float], size_bytes: int | None = None) -> None:
    if not times:
        print(f"  {label:40s}  NO DATA")
        return
    avg = mean(times)
    med = median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    sd = stdev(times) if len(times) > 1 else 0
    tput = f"{size_bytes / avg / 1e6:.1f} MB/s" if size_bytes else ""
    print(
        f"  {label:40s}  avg={_fmt(avg)}  med={_fmt(med)}  p95={_fmt(p95)}  stddev={_fmt(sd)}  n={len(times)}  {tput}"
    )


# ── benchmarks ────────────────────────────────────────────────────────────────


def bench_boto3_shard_download(n: int = 3) -> None:
    """Full shard download via boto3 — measures bulk S3 throughput."""
    import boto3
    import botocore.config

    print("\n── boto3 shard download ──────────────────────────────────────────────")
    configs = {
        "path-style (default timeouts)": botocore.config.Config(s3={"addressing_style": "path"}),
        "path-style (10s/30s timeouts)": botocore.config.Config(
            s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=30
        ),
        "path-style + max-pool-64": botocore.config.Config(
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=30,
            max_pool_connections=64,
        ),
    }
    for label, cfg in configs.items():
        cl = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET,
            region_name="us-east-1",
            config=cfg,
        )
        times: list[float] = []
        size = 0
        for _ in range(n):
            t = time.perf_counter()
            resp = cl.get_object(Bucket="cc-index", Key=SHARD_KEY)
            data = resp["Body"].read()
            times.append(time.perf_counter() - t)
            size = len(data)
        _report(label, times, size)


def bench_boto3_warc_fetch(warc_key: str, offset: int, length: int, n: int = 10, workers: int = 16) -> None:
    """Concurrent WARC byte-range fetches — measures latency at concurrency."""
    import boto3
    import botocore.config

    print("\n── boto3 WARC byte-range fetch ───────────────────────────────────────")

    def _single_fetch(client) -> float:
        rng = f"bytes={offset}-{offset + length - 1}"
        t = time.perf_counter()
        r = client.get_object(Bucket="crawl-data", Key=warc_key, Range=rng)
        r["Body"].read()
        return time.perf_counter() - t

    for label, cfg in {
        "sequential (1 thread)": (
            1,
            botocore.config.Config(s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=60),
        ),
        f"concurrent ({workers} threads)": (
            workers,
            botocore.config.Config(
                s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=60, max_pool_connections=workers + 4
            ),
        ),
    }.items():
        concurrency, bcfg = cfg
        cl = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET,
            region_name="us-east-1",
            config=bcfg,
        )
        times: list[float] = []
        if concurrency == 1:
            for _ in range(n):
                times.append(_single_fetch(cl))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(_single_fetch, cl) for _ in range(n)]
                wall_t = time.perf_counter()
                for f in concurrent.futures.as_completed(futs):
                    try:
                        times.append(f.result())
                    except Exception as e:
                        print(f"    fetch error: {e}")
                total_wall = time.perf_counter() - wall_t
            print(
                f"    {concurrency}-thread wall time for {n} fetches: {_fmt(total_wall)} → {n / total_wall:.1f} fetches/s"
            )
        _report(label, times, length)


def bench_requests_warc_fetch(warc_key: str, offset: int, length: int, n: int = 5) -> None:
    """Same WARC fetch via requests (no S3 signing overhead)."""
    import requests

    print("\n── requests WARC byte-range ──────────────────────────────────────────")
    # PBSS supports both S3 and HTTPS presigned; try unsigned path-style URL directly
    url = f"{ENDPOINT}/crawl-data/{warc_key}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    # Try with AWS SigV4 via requests-aws4auth if available
    try:
        from requests_aws4auth import AWS4Auth

        auth = AWS4Auth(KEY_ID, SECRET, "us-east-1", "s3", service="s3")
        auth_label = "SigV4"
    except ImportError:
        auth = None
        auth_label = "no-auth (may 403)"

    times: list[float] = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            resp = requests.get(url, headers=headers, auth=auth, timeout=30)
            resp.raise_for_status()
            _ = resp.content
            times.append(time.perf_counter() - t)
        except Exception as e:
            print(f"    requests ({auth_label}) error: {e}")
            break
    if times:
        _report(f"requests {auth_label}", times, length)


def bench_s3fs_shard_download(n: int = 2) -> None:
    """Full shard download via s3fs (fsspec backend)."""
    print("\n── s3fs shard download ───────────────────────────────────────────────")
    try:
        import s3fs
    except ImportError:
        print("  s3fs not installed — skip")
        return

    fs = s3fs.S3FileSystem(
        key=KEY_ID,
        secret=SECRET,
        endpoint_url=ENDPOINT,
        config_kwargs={"s3": {"addressing_style": "path"}},
    )
    times: list[float] = []
    size = 0
    for _ in range(n):
        t = time.perf_counter()
        with fs.open(f"cc-index/{SHARD_KEY}", "rb") as fh:
            data = fh.read()
        times.append(time.perf_counter() - t)
        size = len(data)
    _report("s3fs", times, size)


def bench_public_http_warc(warc_key: str, offset: int, length: int, n: int = 5) -> None:
    """Same WARC via public CC HTTP endpoint — baseline with no auth overhead."""
    import requests

    print("\n── public CC HTTP (baseline, no auth) ────────────────────────────────")
    url = f"https://data.commoncrawl.org/{warc_key}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    times: list[float] = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            _ = resp.content
            times.append(time.perf_counter() - t)
        except Exception as e:
            print(f"    public HTTP error: {e}")
            break
    if times:
        _report("public data.commoncrawl.org", times, length)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 72)
    print("PBSS Connectivity Benchmark")
    print(f"Endpoint : {ENDPOINT}")
    print(f"Key ID   : {KEY_ID}")
    print("=" * 72)

    # Step 1: download shard to get a real WARC key
    print("\nDownloading shard to extract WARC key...")
    import boto3
    import botocore.config

    cl = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
        region_name="us-east-1",
        config=botocore.config.Config(s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=120),
    )
    t0 = time.perf_counter()
    resp = cl.get_object(Bucket="cc-index", Key=SHARD_KEY)
    shard_bytes = resp["Body"].read()
    shard_t = time.perf_counter() - t0
    print(f"  Shard: {len(shard_bytes) / 1e6:.1f} MB in {shard_t:.2f}s ({len(shard_bytes) / shard_t / 1e6:.1f} MB/s)")

    warc_key, offset, length = _warc_key_and_range(shard_bytes)
    print(f"  WARC target: {warc_key.split('/')[-1]} @{offset}+{length}")

    # Check if WARC exists on PBSS
    try:
        cl.head_object(Bucket="crawl-data", Key=warc_key)
        warc_on_pbss = True
        print("  WARC exists on PBSS ✓")
    except Exception as e:
        warc_on_pbss = False
        print(f"  WARC NOT on PBSS ({e}) — skipping PBSS WARC benchmarks")

    bench_boto3_shard_download(n=2)

    if warc_on_pbss:
        bench_boto3_warc_fetch(warc_key, offset, length, n=10, workers=16)
        bench_requests_warc_fetch(warc_key, offset, length, n=5)
        bench_s3fs_shard_download(n=2)

    bench_public_http_warc(warc_key, offset, length, n=5)

    print("\n" + "=" * 72)
    print("Benchmark complete.")
    if shard_t > 10:
        print(f"⚠  Shard download took {shard_t:.1f}s — PBSS appears degraded (normal <5s)")
    else:
        print(f"✓  PBSS healthy — {len(shard_bytes) / shard_t / 1e6:.0f} MB/s")


if __name__ == "__main__":
    main()
