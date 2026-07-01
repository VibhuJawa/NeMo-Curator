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

"""Build a larger content-bearing host-bucket subset for PR 2075 benchmarking.

This is intentionally still a subset builder, not a full snapshot materializer:
it reads selected files from the index-only host-bucket manifest, chooses repeated
hosts, fetches their WARC records concurrently, and writes one grouped Parquet
shard with real HTML bodies.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
OUTPUT_COLS = [
    "snapshot",
    "record_id",
    "url",
    "url_host_name",
    "host_hash64",
    "host_bucket",
    "host_bucket_label",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "source_manifest_file",
]

_THREAD_LOCAL = threading.local()


def _parquet_files(root: Path, indices: str | None, max_files: int, file_stride: int) -> list[Path]:
    if root.is_file():
        return [root]
    base = root / "parquet" if (root / "parquet").is_dir() else root
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {base}")
    if indices:
        wanted = {int(part.strip()) for part in indices.split(",") if part.strip()}
        selected = []
        for idx in wanted:
            matches = [path for path in files if f"_{idx:06d}-" in path.name]
            if not matches:
                raise FileNotFoundError(f"No manifest parquet file matching index {idx:06d}")
            selected.extend(matches)
        return sorted(set(selected))
    stride = max(1, file_stride)
    selected = files[::stride]
    if max_files > 0:
        selected = selected[:max_files]
    return selected


def _s3_filesystem() -> pafs.S3FileSystem:
    fs = getattr(_THREAD_LOCAL, "s3", None)
    if fs is not None:
        return fs
    endpoint_url = os.environ["AWS_ENDPOINT_URL_S3"]
    endpoint = endpoint_url.removeprefix("https://").removeprefix("http://")
    scheme = "https" if endpoint_url.startswith("https://") else "http"
    fs = pafs.S3FileSystem(
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_override=endpoint,
        scheme=scheme,
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    _THREAD_LOCAL.s3 = fs
    return fs


def _s3_path(uri_or_path: str) -> str:
    if uri_or_path.startswith("s3://"):
        parsed = urlparse(uri_or_path)
        return f"{parsed.netloc}/{parsed.path.lstrip('/')}"
    return uri_or_path.lstrip("/")


def _record_id(row: dict) -> str:
    parts = [row.get("warc_filename"), row.get("warc_record_offset"), row.get("warc_record_length")]
    if all(part is not None and str(part) for part in parts):
        return "|".join(str(part) for part in parts)
    return str(row.get("url") or "")


def _eligible_rows(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    schema = pq.read_schema(path).names
    cols = [col for col in MANIFEST_COLS if col in schema]
    required = {"url", "url_host_name", "host_bucket", "warc_filename", "warc_record_offset", "warc_record_length"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = pq.read_table(path, columns=cols).to_pandas()
    if "snapshot" not in df.columns:
        df["snapshot"] = os.environ.get("CC_SNAPSHOT", "")
    if "host_hash64" not in df.columns:
        df["host_hash64"] = None
    if "host_bucket_label" not in df.columns:
        df["host_bucket_label"] = df["host_bucket"].map(lambda value: f"{int(value):05d}")

    df["host_bucket"] = pd.to_numeric(df["host_bucket"], errors="coerce")
    df["warc_record_offset"] = pd.to_numeric(df["warc_record_offset"], errors="coerce")
    df["warc_record_length"] = pd.to_numeric(df["warc_record_length"], errors="coerce")
    for col in ("url", "url_host_name", "warc_filename"):
        df[col] = df[col].fillna("").astype(str)

    keep = (
        df["host_bucket"].notna()
        & df["warc_record_offset"].notna()
        & df["warc_record_length"].notna()
        & (df["warc_record_length"] >= args.min_warc_record_length)
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
    if args.exclude_host_regex:
        keep &= ~df["url_host_name"].str.contains(args.exclude_host_regex, regex=True, case=False, na=False)

    out = df[keep].copy()
    out["host_bucket"] = out["host_bucket"].astype("int64")
    out["warc_record_offset"] = out["warc_record_offset"].astype("int64")
    out["warc_record_length"] = out["warc_record_length"].astype("int64")
    out["host_bucket_label"] = out["host_bucket"].map(lambda value: f"{int(value):05d}")
    out["source_manifest_file"] = path.name
    return out


def _select_from_file(df: pd.DataFrame, per_file_target: int, args: argparse.Namespace) -> pd.DataFrame:
    host_counts = df.groupby(["host_bucket", "url_host_name"], sort=False).size().rename("rows").reset_index()
    hosts = host_counts[host_counts["rows"] >= args.min_pages_per_host].copy()
    if hosts.empty:
        return pd.DataFrame(columns=df.columns)
    hosts = hosts.sort_values(["rows", "host_bucket", "url_host_name"], ascending=[False, True, True])
    selected = []
    kept = 0
    for rec in hosts.to_dict("records"):
        rows = df[(df["host_bucket"] == rec["host_bucket"]) & (df["url_host_name"] == rec["url_host_name"])]
        rows = rows.sort_values(["warc_filename", "warc_record_offset", "url"], kind="stable").head(args.max_pages_per_host)
        selected.append(rows)
        kept += len(rows)
        if kept >= per_file_target:
            break
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=df.columns)


def _balanced_limit(df: pd.DataFrame, target_rows: int, group_col: str) -> pd.DataFrame:
    if len(df) <= target_rows or group_col not in df.columns:
        return df.head(target_rows).copy()
    groups = list(df.groupby(group_col, sort=False))
    per_group = max(1, math.ceil(target_rows / max(len(groups), 1)))
    picked = [group.sort_values("_selection_order", kind="stable").head(per_group) for _, group in groups]
    out = pd.concat(picked, ignore_index=False).sort_values("_selection_order", kind="stable")
    if len(out) < target_rows:
        missing = target_rows - len(out)
        remaining = df.drop(index=out.index, errors="ignore").sort_values("_selection_order", kind="stable")
        out = pd.concat([out, remaining.head(missing)], ignore_index=False)
    return out.sort_values("_selection_order", kind="stable").head(target_rows).copy()


def _select_rows(files: list[Path], args: argparse.Namespace) -> pd.DataFrame:
    candidate_target = math.ceil(args.target_pages * args.candidate_multiplier)
    per_file_target = max(args.min_pages_per_host, math.ceil(candidate_target / max(len(files), 1)))
    selected = []
    for i, path in enumerate(files, start=1):
        t0 = time.perf_counter()
        eligible = _eligible_rows(path, args)
        picked = _select_from_file(eligible, per_file_target, args)
        logger.info(
            "Manifest {}/{} {}: eligible={:,} picked={:,} elapsed={:.1f}s",
            i,
            len(files),
            path.name,
            len(eligible),
            len(picked),
            time.perf_counter() - t0,
        )
        if not picked.empty:
            selected.append(picked)
    if not selected:
        raise RuntimeError("No eligible repeated-host rows selected")
    out = pd.concat(selected, ignore_index=True)
    out = out.sort_values(
        ["source_manifest_file", "host_bucket", "url_host_name", "url", "warc_record_offset"], kind="stable"
    )
    out = out.reset_index(drop=True)
    out["_selection_order"] = range(len(out))
    out = _balanced_limit(out, candidate_target, "source_manifest_file")
    out = out.sort_values(
        ["source_manifest_file", "host_bucket", "url_host_name", "url", "warc_record_offset"], kind="stable"
    ).reset_index(drop=True)
    out["_selection_order"] = range(len(out))
    out["record_id"] = [_record_id(row) for row in out.to_dict("records")]
    logger.info(
        "Selected {:,} candidates for target {:,}: hosts={:,} buckets={:,} manifest_files={:,}",
        len(out),
        args.target_pages,
        out["url_host_name"].nunique(),
        out["host_bucket"].nunique(),
        out["source_manifest_file"].nunique(),
    )
    return out


def _extract_html_from_warc_record(record_bytes: bytes) -> str:
    with gzip.GzipFile(fileobj=BytesIO(record_bytes)) as gz:
        payload = gz.read()
    for record in ArchiveIterator(BytesIO(payload)):
        if record.rec_type == "response":
            return record.content_stream().read().decode("utf-8", errors="replace")
    return ""


def _fetch_group(item: tuple[str, list[dict]], min_html_chars: int) -> tuple[list[dict], list[dict]]:
    warc_filename, records = item
    records = sorted(records, key=lambda rec: int(rec["warc_record_offset"]))
    rows = []
    failures = []
    try:
        infile = _s3_filesystem().open_input_file(_s3_path(warc_filename))
    except Exception as exc:  # noqa: BLE001
        return [], [{"url": rec.get("url"), "warc_filename": warc_filename, "error": repr(exc)} for rec in records]
    with infile:
        for rec in records:
            try:
                infile.seek(int(rec["warc_record_offset"]))
                raw = infile.read(int(rec["warc_record_length"]))
                html = _extract_html_from_warc_record(raw)
                if len(html.strip()) < min_html_chars:
                    continue
                rows.append(
                    {
                        "snapshot": str(rec.get("snapshot") or os.environ.get("CC_SNAPSHOT", "")),
                        "record_id": rec["record_id"],
                        "url": rec["url"],
                        "url_host_name": rec["url_host_name"],
                        "host_hash64": rec.get("host_hash64"),
                        "host_bucket": int(rec["host_bucket"]),
                        "host_bucket_label": f"{int(rec['host_bucket']):05d}",
                        HTML_ZLIB_COL: compress_html_zlib(html),
                        HTML_CHARS_COL: len(html),
                        "warc_filename": rec["warc_filename"],
                        "warc_record_offset": int(rec["warc_record_offset"]),
                        "warc_record_length": int(rec["warc_record_length"]),
                        "source_manifest_file": rec["source_manifest_file"],
                        "_selection_order": int(rec["_selection_order"]),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"url": rec.get("url"), "warc_filename": warc_filename, "error": repr(exc)})
    return rows, failures


def _materialize_html(selected: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict]]:
    t_group = time.perf_counter()
    grouped: dict[str, list[dict]] = {}
    for rec in selected.to_dict("records"):
        grouped.setdefault(str(rec["warc_filename"]), []).append(rec)
    groups = list(grouped.items())
    logger.info(
        "Fetching {:,} candidate records from {:,} WARC files with {} workers (grouped in {:.1f}s)",
        len(selected),
        len(groups),
        args.fetch_workers,
        time.perf_counter() - t_group,
    )
    rows: list[dict] = []
    failures: list[dict] = []
    done_groups = 0
    done_candidates = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        futures = {pool.submit(_fetch_group, item, args.min_html_chars): item for item in groups}
        for future in as_completed(futures):
            group_rows, group_failures = future.result()
            rows.extend(group_rows)
            failures.extend(group_failures)
            done_groups += 1
            done_candidates += len(futures[future][1])
            if done_groups % args.log_every_groups == 0 or done_groups == len(groups):
                logger.info(
                    "Fetched groups {:,}/{:,}; candidates {:,}/{:,}; kept {:,}; failures {:,}; elapsed {:.1f}s",
                    done_groups,
                    len(groups),
                    done_candidates,
                    len(selected),
                    len(rows),
                    len(failures),
                    time.perf_counter() - t0,
                )
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLS), failures
    out = pd.DataFrame(rows)
    out = _balanced_limit(out.sort_values("_selection_order", kind="stable"), args.target_pages, "source_manifest_file")
    out = out.sort_values(
        ["source_manifest_file", "host_bucket", "url_host_name", "_selection_order"], kind="stable"
    ).reset_index(drop=True)
    return out[OUTPUT_COLS], failures


def _write_summary(
    output: Path,
    files: list[Path],
    selected: pd.DataFrame,
    html_rows: pd.DataFrame,
    failures: list[dict],
    args: argparse.Namespace,
    elapsed_s: float,
) -> None:
    per_host = (
        html_rows.groupby("url_host_name").size().sort_values(ascending=False).to_dict() if not html_rows.empty else {}
    )
    per_file = (
        html_rows.groupby("source_manifest_file").size().sort_values(ascending=False).to_dict()
        if not html_rows.empty
        else {}
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "manifest_files": [path.name for path in files],
        "output_parquet": str(output / "shard_0000.parquet"),
        "target_pages": args.target_pages,
        "selected_candidate_rows": int(len(selected)),
        "materialized_rows": int(len(html_rows)),
        "fetch_failures": int(len(failures)),
        "host_count": int(html_rows["url_host_name"].nunique()) if not html_rows.empty else 0,
        "bucket_count": int(html_rows["host_bucket"].nunique()) if not html_rows.empty else 0,
        "rows_per_source_manifest_file": {str(k): int(v) for k, v in per_file.items()},
        "top_rows_per_host": {str(k): int(v) for k, v in list(per_host.items())[:100]},
        "min_pages_per_host": args.min_pages_per_host,
        "max_pages_per_host": args.max_pages_per_host,
        "candidate_multiplier": args.candidate_multiplier,
        "exclude_host_regex": args.exclude_host_regex,
        "elapsed_s": round(elapsed_s, 3),
    }
    (output / "_subset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if failures:
        with (output / "_fetch_failures_sample.jsonl").open("w", encoding="utf-8") as fh:
            for rec in failures[:1000]:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    files = _parquet_files(Path(args.input), args.manifest_file_indices, args.max_manifest_files, args.file_stride)
    logger.info("Using {} manifest parquet file(s)", len(files))
    selected = _select_rows(files, args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_selected = _balanced_limit(selected, args.target_pages, "source_manifest_file")
        dry_selected.drop(columns=["_selection_order"], errors="ignore").to_parquet(
            output / "selected_candidates.parquet", index=False, compression="snappy"
        )
        _write_summary(output, files, selected, pd.DataFrame(columns=OUTPUT_COLS), [], args, time.perf_counter() - t0)
        logger.info("Dry run wrote selected candidate metadata to {}", output)
        return
    html_rows, failures = _materialize_html(selected, args)
    if html_rows.empty:
        raise RuntimeError("No HTML rows were materialized")
    out_path = output / "shard_0000.parquet"
    tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    html_rows.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(out_path)
    _write_summary(output, files, selected, html_rows, failures, args, time.perf_counter() - t0)
    logger.info(
        "Wrote {:,} rows across {:,} hosts and {:,} buckets to {}",
        len(html_rows),
        html_rows["url_host_name"].nunique(),
        html_rows["host_bucket"].nunique(),
        out_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Host-bucketed index dataset root or one manifest parquet")
    parser.add_argument("--output", required=True, help="Output directory; writes shard_0000.parquet")
    parser.add_argument("--manifest-file-indices", default=None, help="Comma-separated manifest file indices, e.g. 143,1000")
    parser.add_argument("--max-manifest-files", type=int, default=0, help="Use at most this many files after stride; 0 means all")
    parser.add_argument("--file-stride", type=int, default=1)
    parser.add_argument("--target-pages", type=int, default=200_000)
    parser.add_argument("--candidate-multiplier", type=float, default=1.15)
    parser.add_argument("--min-pages-per-host", type=int, default=100)
    parser.add_argument("--max-pages-per-host", type=int, default=300)
    parser.add_argument("--min-html-chars", type=int, default=200)
    parser.add_argument("--min-warc-record-length", type=int, default=5_000)
    parser.add_argument("--exclude-host-regex", default=r"workplace\.com", help="Regex applied to url_host_name")
    parser.add_argument("--fetch-workers", type=int, default=32)
    parser.add_argument("--log-every-groups", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.candidate_multiplier < 1.0:
        raise ValueError("--candidate-multiplier must be >= 1.0")
    if args.exclude_host_regex:
        re.compile(args.exclude_host_regex)
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
