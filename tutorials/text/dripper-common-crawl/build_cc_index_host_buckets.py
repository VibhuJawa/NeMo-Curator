#!/usr/bin/env python3
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

"""Build host-bucketed Common Crawl index manifests without fetching page bodies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.fs as pafs
import ray
import ray.data
import xxhash
from loguru import logger
from ray.data._internal.savemode import SaveMode

INDEX_COLS = [
    "url",
    "url_host_name",
    "fetch_status",
    "content_mime_type",
    "content_mime_detected",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]

OUTPUT_COLS = [
    "snapshot",
    "url",
    "url_host_name",
    "host_hash64",
    "host_bucket",
    "host_bucket_label",
    "fetch_status",
    "content_mime_type",
    "content_mime_detected",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


def _s3_filesystem() -> pafs.S3FileSystem:
    endpoint_url = os.environ["AWS_ENDPOINT_URL_S3"]
    endpoint = endpoint_url.removeprefix("https://").removeprefix("http://")
    scheme = "https" if endpoint_url.startswith("https://") else "http"
    return pafs.S3FileSystem(
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_override=endpoint,
        scheme=scheme,
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _s3_path(uri_or_path: str) -> str:
    if uri_or_path.startswith("s3://"):
        parsed = urlparse(uri_or_path)
        return f"{parsed.netloc}/{parsed.path.lstrip('/')}"
    return uri_or_path.lstrip("/")


def _list_index_parts(s3: pafs.S3FileSystem, prefix_uri: str, max_parts: int) -> list[str]:
    prefix = _s3_path(prefix_uri).rstrip("/") + "/"
    infos = s3.get_file_info(pafs.FileSelector(prefix, recursive=True))
    parts = sorted(info.path for info in infos if info.is_file and info.path.endswith(".parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet index parts under {prefix_uri}")
    if max_parts > 0:
        return parts[:max_parts]
    return parts


def _derive_host(url: object) -> str:
    if not isinstance(url, str):
        return ""
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _transform_index_batch(
    batch: pd.DataFrame,
    *,
    snapshot: str,
    num_buckets: int,
    min_warc_record_length: int,
) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = batch.copy()
    status = pd.to_numeric(df["fetch_status"], errors="coerce").fillna(0).astype("int64")
    length = pd.to_numeric(df["warc_record_length"], errors="coerce").fillna(0).astype("int64")
    mime = (
        df.get("content_mime_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
        + " "
        + df.get("content_mime_detected", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    )

    hosts = df.get("url_host_name", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    missing_hosts = hosts.eq("") | hosts.eq("nan")
    if missing_hosts.any():
        derived = df.loc[missing_hosts, "url"].map(_derive_host)
        hosts.loc[missing_hosts] = derived

    keep = status.eq(200) & mime.str.contains("html", regex=False) & length.ge(min_warc_record_length) & hosts.ne("")
    out = df.loc[keep, [c for c in INDEX_COLS if c in df.columns]].copy()
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    out["snapshot"] = snapshot
    out["url_host_name"] = hosts.loc[out.index].to_numpy()
    out["fetch_status"] = status.loc[out.index].astype("int64").to_numpy()
    out["warc_record_offset"] = pd.to_numeric(out["warc_record_offset"], errors="coerce").fillna(0).astype("int64")
    out["warc_record_length"] = pd.to_numeric(out["warc_record_length"], errors="coerce").fillna(0).astype("int64")

    hashes = [xxhash.xxh64_hexdigest(host, seed=0) for host in out["url_host_name"].astype(str)]
    buckets = [int(host_hash, 16) % num_buckets for host_hash in hashes]
    out["host_hash64"] = hashes
    out["host_bucket"] = buckets
    out["host_bucket_label"] = [f"{bucket:05d}" for bucket in buckets]
    out = out[OUTPUT_COLS]
    return out.sort_values(["host_bucket", "url_host_name", "url"], kind="stable").reset_index(drop=True)


def _init_ray(args: argparse.Namespace) -> None:
    ray_kwargs: dict[str, object] = {"ignore_reinit_error": True}
    if args.ray_address:
        ray_kwargs["address"] = args.ray_address
    else:
        ray_kwargs["num_cpus"] = args.num_cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        ray_tmpdir = os.environ.get("RAY_TMPDIR")
        if ray_tmpdir:
            ray_kwargs["_temp_dir"] = ray_tmpdir
    ray.init(**ray_kwargs)


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    snapshot = args.snapshot or os.environ["CC_SNAPSHOT"]
    index_prefix = args.index_prefix or os.environ["CC_INDEX_TABLE_PREFIX"]
    output = Path(args.output)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    s3 = _s3_filesystem()
    parts = _list_index_parts(s3, index_prefix, args.max_parts)
    logger.info("Using {} CC index parquet part(s) from {}", len(parts), index_prefix)
    logger.info("Output: {}", output)

    _init_ray(args)
    ds = ray.data.read_parquet(
        parts,
        filesystem=s3,
        columns=INDEX_COLS,
        concurrency=args.read_concurrency,
        override_num_blocks=args.read_blocks if args.read_blocks > 0 else None,
    )
    ds = ds.map_batches(
        _transform_index_batch,
        batch_format="pandas",
        batch_size=args.batch_size,
        zero_copy_batch=False,
        fn_kwargs={
            "snapshot": snapshot,
            "num_buckets": args.num_buckets,
            "min_warc_record_length": args.min_warc_record_length,
        },
        num_cpus=args.cpus_per_transform,
        concurrency=args.map_concurrency,
    )
    if args.max_rows > 0:
        ds = ds.limit(args.max_rows)
    if args.repartition_blocks > 0:
        ds = ds.repartition(args.repartition_blocks, shuffle=args.shuffle_repartition)
    if args.global_sort:
        ds = ds.sort(["host_bucket", "url_host_name", "url"])

    write_kwargs = {
        "compression": "snappy",
        "mode": SaveMode.OVERWRITE,
        "min_rows_per_file": args.min_rows_per_file,
        "max_rows_per_file": args.max_rows_per_file,
    }
    if args.partition_by_bucket:
        write_kwargs["partition_cols"] = ["host_bucket_label"]
    ds.write_parquet(str(output / "parquet"), **write_kwargs)

    manifest = {
        "snapshot": snapshot,
        "index_prefix": index_prefix,
        "num_index_parts": len(parts),
        "max_parts": args.max_parts,
        "num_buckets": args.num_buckets,
        "partition_by_bucket": args.partition_by_bucket,
        "min_warc_record_length": args.min_warc_record_length,
        "output": str(output),
        "parquet_path": str(output / "parquet"),
        "columns": OUTPUT_COLS,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    (output / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote manifest to {}", output / "_manifest.json")
    logger.info("Ray execution stats:\n{}", ds.stats())


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", required=True, help="Output directory under /lustre")
    parser.add_argument("--snapshot", default=None, help="Snapshot name; defaults to CC_SNAPSHOT")
    parser.add_argument("--index-prefix", default=None, help="S3 prefix; defaults to CC_INDEX_TABLE_PREFIX")
    parser.add_argument("--max-parts", type=int, default=1, help="Number of index parts to scan; 0 means all")
    parser.add_argument("--num-buckets", type=int, default=512, help="Stable host hash buckets")
    parser.add_argument("--min-warc-record-length", type=int, default=0)
    parser.add_argument("--partition-by-bucket", action="store_true", help="Write Hive dirs by host_bucket_label")
    parser.add_argument("--global-sort", action="store_true", help="Globally sort output; expensive at snapshot scale")
    parser.add_argument("--repartition-blocks", type=int, default=0, help="Ray output blocks before write; 0 leaves as-is")
    parser.add_argument("--shuffle-repartition", action="store_true", help="Shuffle during repartition")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit output rows after filtering; 0 means no limit")
    parser.add_argument("--read-blocks", type=int, default=0, help="Override Ray read blocks; 0 uses Ray default")
    parser.add_argument("--read-concurrency", type=int, default=8)
    parser.add_argument("--map-concurrency", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--cpus-per-transform", type=float, default=1.0)
    parser.add_argument("--min-rows-per-file", type=int, default=250_000)
    parser.add_argument("--max-rows-per-file", type=int, default=1_000_000)
    parser.add_argument("--ray-address", default=None, help="Ray address; omit to start a local Ray runtime on the node")
    parser.add_argument("--num-cpus", type=int, default=0, help="Local Ray CPUs; defaults to SLURM_CPUS_PER_TASK")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
