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

"""Materialize a tiny content-bearing smoke shard from one host-bucket manifest file."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL, compress_html_zlib

MANIFEST_COLS = [
    "snapshot",
    "url",
    "url_host_name",
    "host_bucket",
    "host_bucket_label",
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
    "host_bucket",
    "host_bucket_label",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
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


def _parse_host_buckets(raw: str | None) -> list[int]:
    if not raw:
        return []
    buckets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        buckets.append(int(part))
    return sorted(set(buckets))


def _read_manifest(input_parquet: Path, min_warc_record_length: int) -> pd.DataFrame:
    pf = pq.ParquetFile(input_parquet)
    cols = [col for col in MANIFEST_COLS if col in pf.schema_arrow.names]
    missing = {"url", "url_host_name", "host_bucket", "warc_filename", "warc_record_offset", "warc_record_length"} - set(
        cols
    )
    if missing:
        raise ValueError(f"{input_parquet} is missing required columns: {sorted(missing)}")

    df = pq.read_table(input_parquet, columns=cols).to_pandas()
    if "snapshot" not in df.columns:
        df["snapshot"] = os.environ.get("CC_SNAPSHOT", "")
    if "host_bucket_label" not in df.columns:
        df["host_bucket_label"] = df["host_bucket"].map(lambda value: f"{int(value):05d}")

    df["host_bucket"] = pd.to_numeric(df["host_bucket"], errors="coerce").astype("Int64")
    df["warc_record_offset"] = pd.to_numeric(df["warc_record_offset"], errors="coerce").astype("Int64")
    df["warc_record_length"] = pd.to_numeric(df["warc_record_length"], errors="coerce").astype("Int64")
    df["url"] = df["url"].fillna("").astype(str)
    df["url_host_name"] = df["url_host_name"].fillna("").astype(str)
    df["warc_filename"] = df["warc_filename"].fillna("").astype(str)

    keep = (
        df["host_bucket"].notna()
        & df["warc_record_offset"].notna()
        & df["warc_record_length"].notna()
        & (df["warc_record_length"] >= min_warc_record_length)
        & (df["url"] != "")
        & (df["url_host_name"] != "")
        & (df["warc_filename"] != "")
    )
    if "fetch_status" in df.columns:
        keep &= pd.to_numeric(df["fetch_status"], errors="coerce").fillna(0).astype("int64") == 200
    mime_cols = [col for col in ("content_mime_type", "content_mime_detected") if col in df.columns]
    if mime_cols:
        mime = pd.Series("", index=df.index)
        for col in mime_cols:
            mime += " " + df[col].fillna("").astype(str).str.lower()
        keep &= mime.str.contains("html", regex=False)

    out = df[keep].copy()
    out["host_bucket"] = out["host_bucket"].astype("int64")
    out["warc_record_offset"] = out["warc_record_offset"].astype("int64")
    out["warc_record_length"] = out["warc_record_length"].astype("int64")
    out["host_bucket_label"] = out["host_bucket"].map(lambda value: f"{int(value):05d}")
    logger.info("Read {:,} eligible manifest rows from {}", len(out), input_parquet)
    return out


def _choose_buckets(df: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    requested = _parse_host_buckets(args.host_buckets)
    if requested:
        present = sorted(set(df["host_bucket"].astype(int)) & set(requested))
        missing = sorted(set(requested) - set(present))
        if missing:
            logger.warning("Requested host buckets not present in input file: {}", missing)
        if not present:
            raise RuntimeError("None of the requested host buckets are present in the input file")
        return present[: args.num_host_buckets]

    host_counts = df.groupby(["host_bucket", "url_host_name"], sort=False).size().rename("rows").reset_index()
    eligible_hosts = host_counts[host_counts["rows"] >= args.min_pages_per_host].copy()
    if eligible_hosts.empty:
        raise RuntimeError(f"No hosts have at least {args.min_pages_per_host} eligible rows in this file")
    bucket_scores = (
        eligible_hosts.groupby("host_bucket")
        .agg(repeated_hosts=("url_host_name", "nunique"), repeated_rows=("rows", "sum"), max_host_rows=("rows", "max"))
        .reset_index()
        .sort_values(["repeated_rows", "repeated_hosts", "max_host_rows", "host_bucket"], ascending=[False, False, False, True])
    )
    chosen = bucket_scores["host_bucket"].astype(int).head(args.num_host_buckets).tolist()
    logger.info("Auto-selected host buckets: {}", chosen)
    return chosen


def _select_rows(df: pd.DataFrame, buckets: list[int], args: argparse.Namespace) -> pd.DataFrame:
    per_bucket_target = max(args.min_pages_per_host, math.ceil(args.target_pages / max(len(buckets), 1)))
    selected: list[pd.DataFrame] = []
    bucket_df = df[df["host_bucket"].isin(buckets)].copy()
    host_counts = bucket_df.groupby(["host_bucket", "url_host_name"], sort=False).size().rename("rows").reset_index()

    for bucket in buckets:
        hosts = (
            host_counts[(host_counts["host_bucket"] == bucket) & (host_counts["rows"] >= args.min_pages_per_host)]
            .sort_values(["rows", "url_host_name"], ascending=[False, True])
            .head(args.max_hosts_per_bucket)
        )
        kept_for_bucket = 0
        for host in hosts["url_host_name"].tolist():
            rows = bucket_df[(bucket_df["host_bucket"] == bucket) & (bucket_df["url_host_name"] == host)]
            rows = rows.sort_values(["url"], kind="stable").head(args.max_pages_per_host)
            selected.append(rows)
            kept_for_bucket += len(rows)
            if kept_for_bucket >= per_bucket_target:
                break

    if not selected:
        raise RuntimeError("No candidate rows selected for the requested buckets")
    out = pd.concat(selected, ignore_index=True)
    out = out.sort_values(["host_bucket", "url_host_name", "url"], kind="stable").head(args.target_pages)
    logger.info(
        "Selected {:,} candidate rows across {:,} buckets and {:,} hosts",
        len(out),
        out["host_bucket"].nunique(),
        out["url_host_name"].nunique(),
    )
    return out.reset_index(drop=True)


def _extract_html_from_warc_record(record_bytes: bytes) -> str:
    with gzip.GzipFile(fileobj=BytesIO(record_bytes)) as gz:
        payload = gz.read()
    for record in ArchiveIterator(BytesIO(payload)):
        if record.rec_type != "response":
            continue
        return record.content_stream().read().decode("utf-8", errors="replace")
    return ""


def _fetch_html(s3: pafs.S3FileSystem, row: dict) -> str:
    offset = int(row["warc_record_offset"])
    length = int(row["warc_record_length"])
    with s3.open_input_file(_s3_path(str(row["warc_filename"]))) as infile:
        infile.seek(offset)
        return _extract_html_from_warc_record(infile.read(length))


def _materialize_html(s3: pafs.S3FileSystem, rows: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    out_rows = []
    failures = 0
    for i, row in enumerate(rows.to_dict("records"), start=1):
        try:
            html = _fetch_html(s3, row)
        except Exception as exc:  # noqa: BLE001 - smoke builder should report and continue
            failures += 1
            logger.warning("Failed to fetch WARC record for {}: {}", row.get("url"), exc)
            continue
        if len(html.strip()) < args.min_html_chars:
            continue
        out_rows.append(
            {
                "snapshot": str(row.get("snapshot") or os.environ.get("CC_SNAPSHOT", "")),
                "url": row["url"],
                "url_host_name": row["url_host_name"],
                "host_bucket": int(row["host_bucket"]),
                "host_bucket_label": f"{int(row['host_bucket']):05d}",
                HTML_ZLIB_COL: compress_html_zlib(html),
                HTML_CHARS_COL: len(html),
                "warc_filename": row["warc_filename"],
                "warc_record_offset": int(row["warc_record_offset"]),
                "warc_record_length": int(row["warc_record_length"]),
            }
        )
        if i % 10 == 0:
            logger.info("Fetched {:,}/{:,} candidate records; kept {:,}", i, len(rows), len(out_rows))
    out = pd.DataFrame(out_rows, columns=OUTPUT_COLS)
    if not out.empty:
        out = out.sort_values(["host_bucket", "url_host_name", "url"], kind="stable").reset_index(drop=True)
    return out, failures


def _write_summary(
    *,
    output: Path,
    input_parquet: Path,
    selected_rows: pd.DataFrame,
    html_rows: pd.DataFrame,
    failures: int,
    args: argparse.Namespace,
) -> None:
    if html_rows.empty:
        per_bucket: dict[str, int] = {}
        per_host: dict[str, int] = {}
    else:
        per_bucket = {f"{int(k):05d}": int(v) for k, v in html_rows.groupby("host_bucket").size().to_dict().items()}
        per_host = {str(k): int(v) for k, v in html_rows.groupby("url_host_name").size().sort_values(ascending=False).to_dict().items()}
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_parquet": str(input_parquet),
        "output_parquet": str(output / "shard_0000.parquet"),
        "target_pages": args.target_pages,
        "selected_candidate_rows": int(len(selected_rows)),
        "materialized_rows": int(len(html_rows)),
        "fetch_failures": int(failures),
        "host_buckets": sorted(per_bucket),
        "host_count": int(html_rows["url_host_name"].nunique()) if not html_rows.empty else 0,
        "rows_per_bucket": per_bucket,
        "rows_per_host": per_host,
        "min_html_chars": args.min_html_chars,
        "min_pages_per_host": args.min_pages_per_host,
    }
    with (output / "_smoke_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run(args: argparse.Namespace) -> None:
    input_parquet = Path(args.input_parquet)
    if not input_parquet.exists():
        raise FileNotFoundError(input_parquet)

    manifest = _read_manifest(input_parquet, args.min_warc_record_length)
    buckets = _choose_buckets(manifest, args)
    selected = _select_rows(manifest, buckets, args)
    html_rows, failures = _materialize_html(_s3_filesystem(), selected, args)
    if html_rows.empty:
        raise RuntimeError("No HTML rows were materialized")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "shard_0000.parquet"
    tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    html_rows.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(out_path)
    _write_summary(output=output, input_parquet=input_parquet, selected_rows=selected, html_rows=html_rows, failures=failures, args=args)
    logger.info(
        "Wrote {:,} rows across {:,} buckets and {:,} hosts to {}",
        len(html_rows),
        html_rows["host_bucket"].nunique(),
        html_rows["url_host_name"].nunique(),
        out_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-parquet", required=True, help="One local host-bucket manifest parquet file")
    parser.add_argument("--output", required=True, help="Output directory; writes shard_0000.parquet")
    parser.add_argument("--host-buckets", default=None, help="Comma-separated host_bucket ids to sample")
    parser.add_argument("--num-host-buckets", type=int, default=3)
    parser.add_argument("--target-pages", type=int, default=60)
    parser.add_argument("--min-pages-per-host", type=int, default=5)
    parser.add_argument("--max-hosts-per-bucket", type=int, default=4)
    parser.add_argument("--max-pages-per-host", type=int, default=20)
    parser.add_argument("--min-html-chars", type=int, default=200)
    parser.add_argument("--min-warc-record-length", type=int, default=5_000)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
