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

"""Build a small host-bucketed Common Crawl HTML Parquet shard for Dripper smoke runs."""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL, compress_html_zlib

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
    "url",
    "url_host_name",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


def _s3_filesystem() -> pafs.S3FileSystem:
    endpoint = os.environ["AWS_ENDPOINT_URL_S3"].removeprefix("https://").removeprefix("http://")
    scheme = "https" if os.environ["AWS_ENDPOINT_URL_S3"].startswith("https://") else "http"
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
    return parts[:max_parts]


def _is_html_index_batch(df: pd.DataFrame, min_warc_record_length: int) -> pd.Series:
    status_ok = pd.to_numeric(df["fetch_status"], errors="coerce").fillna(0).astype("int64") == 200
    mime = (
        df.get("content_mime_type", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
        + " "
        + df.get("content_mime_detected", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    )
    length_ok = pd.to_numeric(df["warc_record_length"], errors="coerce").fillna(0).astype("int64")
    return status_ok & mime.str.contains("html", regex=False) & (length_ok >= min_warc_record_length)


def _load_candidate_rows(
    *,
    s3: pafs.S3FileSystem,
    index_parts: list[str],
    max_index_rows: int,
    max_candidates: int,
    batch_size: int,
    min_warc_record_length: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rows_seen = 0
    candidates = 0
    for part in index_parts:
        logger.info("Scanning index part {}", part)
        with s3.open_input_file(part) as infile:
            pf = pq.ParquetFile(infile)
            for batch in pf.iter_batches(batch_size=batch_size, columns=INDEX_COLS):
                df = batch.to_pandas()
                rows_seen += len(df)
                html_df = df[_is_html_index_batch(df, min_warc_record_length)]
                if not html_df.empty:
                    frames.append(html_df[INDEX_COLS])
                    candidates += len(html_df)
                if rows_seen >= max_index_rows or candidates >= max_candidates:
                    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLS)
                    logger.info("Index scan stopped: rows_seen={:,} html_candidates={:,}", rows_seen, len(out))
                    return out
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLS)
    logger.info("Index scan complete: rows_seen={:,} html_candidates={:,}", rows_seen, len(out))
    return out


def _choose_repeated_host_rows(df: pd.DataFrame, target_pages: int, min_pages_per_host: int) -> pd.DataFrame:
    counts = Counter(df["url_host_name"].fillna("").astype(str))
    hosts = [host for host, count in counts.most_common() if host and count >= min_pages_per_host]
    if not hosts:
        raise RuntimeError(f"No hosts with at least {min_pages_per_host} HTML candidates")

    per_host_seen: dict[str, int] = defaultdict(int)
    selected = []
    host_set = set(hosts)
    for row in df.to_dict("records"):
        host = str(row.get("url_host_name") or "")
        if host not in host_set:
            continue
        selected.append(row)
        per_host_seen[host] += 1
        if len(selected) >= target_pages:
            break
    out = pd.DataFrame(selected)
    out = out.sort_values(["url_host_name", "url"], kind="stable").reset_index(drop=True)
    logger.info("Selected {:,} index rows across {:,} hosts", len(out), out["url_host_name"].nunique())
    return out


def _extract_html_from_warc_record(record_bytes: bytes) -> str:
    with gzip.GzipFile(fileobj=BytesIO(record_bytes)) as gz:
        payload = gz.read()
    for record in ArchiveIterator(BytesIO(payload)):
        if record.rec_type != "response":
            continue
        content = record.content_stream().read()
        return content.decode("utf-8", errors="replace")
    return ""


def _fetch_html(s3: pafs.S3FileSystem, row: dict) -> str:
    offset = int(row["warc_record_offset"])
    length = int(row["warc_record_length"])
    with s3.open_input_file(_s3_path(str(row["warc_filename"]))) as infile:
        infile.seek(offset)
        data = infile.read(length)
    return _extract_html_from_warc_record(data)


def _materialize_html(
    s3: pafs.S3FileSystem,
    rows: pd.DataFrame,
    min_html_chars: int,
    target_pages: int,
) -> pd.DataFrame:
    out_rows = []
    for i, row in enumerate(rows.to_dict("records"), start=1):
        try:
            html = _fetch_html(s3, row)
        except Exception as exc:  # noqa: BLE001 - keep smoke builder moving past bad records
            logger.warning("Failed to fetch WARC record for {}: {}", row.get("url"), exc)
            continue
        if len(html.strip()) < min_html_chars:
            continue
        out_rows.append(
            {
                "url": row["url"],
                "url_host_name": row["url_host_name"],
                HTML_ZLIB_COL: compress_html_zlib(html),
                HTML_CHARS_COL: len(html),
                "warc_filename": row["warc_filename"],
                "warc_record_offset": int(row["warc_record_offset"]),
                "warc_record_length": int(row["warc_record_length"]),
            }
        )
        if i % 25 == 0:
            logger.info("Fetched {:,}/{:,} records; kept {:,}", i, len(rows), len(out_rows))
        if len(out_rows) >= target_pages:
            logger.info("Reached target of {:,} kept HTML records", target_pages)
            break
    return pd.DataFrame(out_rows, columns=OUTPUT_COLS)


def run(args: argparse.Namespace) -> None:
    s3 = _s3_filesystem()
    index_prefix = args.index_prefix or os.environ["CC_INDEX_TABLE_PREFIX"]
    parts = _list_index_parts(s3, index_prefix, args.index_parts)
    candidates = _load_candidate_rows(
        s3=s3,
        index_parts=parts,
        max_index_rows=args.max_index_rows,
        max_candidates=args.max_candidates,
        batch_size=args.index_batch_size,
        min_warc_record_length=args.min_warc_record_length,
    )
    selected = _choose_repeated_host_rows(candidates, args.target_pages * args.oversample, args.min_pages_per_host)
    html_df = _materialize_html(s3, selected, args.min_html_chars, args.target_pages)
    if len(html_df) > args.target_pages:
        html_df = html_df.groupby("url_host_name", group_keys=False).head(args.target_pages).head(args.target_pages)
    if html_df.empty:
        raise RuntimeError("No HTML records materialized")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shard_0000.parquet"
    tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    html_df.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(out_path)
    logger.info(
        "Wrote {} rows across {} hosts to {}",
        len(html_df),
        html_df["url_host_name"].nunique(),
        out_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", required=True, help="Output directory; writes shard_0000.parquet")
    parser.add_argument("--index-prefix", default=None, help="S3 prefix for CC index parquet parts")
    parser.add_argument("--index-parts", type=int, default=1, help="Number of index parquet parts to scan")
    parser.add_argument("--max-index-rows", type=int, default=1_000_000)
    parser.add_argument("--max-candidates", type=int, default=50_000)
    parser.add_argument("--index-batch-size", type=int, default=200_000)
    parser.add_argument("--target-pages", type=int, default=200)
    parser.add_argument("--oversample", type=int, default=3)
    parser.add_argument("--min-pages-per-host", type=int, default=5)
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
