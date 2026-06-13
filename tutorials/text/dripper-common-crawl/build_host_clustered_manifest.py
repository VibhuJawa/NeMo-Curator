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

"""Build a host-clustered Dripper input manifest from Common Crawl URL Index parquet.

This is intentionally CPU-only.  The output manifest contains Common Crawl byte-range
columns and is consumed by ``main.py --input-manifest-path``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterator
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

INDEX_COLUMNS = [
    "url",
    "url_host_name",
    "fetch_status",
    "http_status",
    "content_mime_type",
    "content_mime_detected",
    "mime",
    "mime-detected",
    "content_languages",
    "languages",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "offset",
    "length",
]

REQUIRED_OUTPUT_COLUMNS = ["url", "warc_filename", "warc_record_offset", "warc_record_length"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a host-clustered CC URL Index manifest for Dripper")
    parser.add_argument(
        "--cc-index-path",
        required=True,
        help="Directory, parquet file, or glob for CC URL Index parquet files.",
    )
    parser.add_argument("--output", required=True, help="Output parquet manifest path")
    parser.add_argument("--max-pages", type=int, default=8192)
    parser.add_argument("--min-host-pages", type=int, default=8)
    parser.add_argument("--max-pages-per-host", type=int, default=64)
    parser.add_argument(
        "--max-hosts",
        type=int,
        default=0,
        help="Maximum hosts to include. Default chooses enough top hosts to fill max-pages.",
    )
    parser.add_argument("--host-bucket-mod", type=int, default=10000)
    parser.add_argument(
        "--host-buckets",
        default=None,
        help="Optional comma/range filter, e.g. '3,7,10-19'. Uses xxhash64(host) % host-bucket-mod.",
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument(
        "--max-index-rows",
        type=int,
        default=0,
        help="Optional raw index-row cap for quick smoke tests.",
    )
    parser.add_argument("--status", type=int, default=200)
    parser.add_argument("--html-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language substring filter over content_languages/languages, e.g. 'eng'.",
    )
    args = parser.parse_args()
    if args.max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    if args.min_host_pages <= 1:
        raise ValueError("--min-host-pages must be greater than 1")
    if args.max_pages_per_host <= 0:
        raise ValueError("--max-pages-per-host must be positive")
    if args.max_hosts < 0:
        raise ValueError("--max-hosts must be non-negative")
    if args.host_bucket_mod <= 0:
        raise ValueError("--host-bucket-mod must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_index_rows < 0:
        raise ValueError("--max-index-rows must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    host_buckets = parse_host_buckets(args.host_buckets)
    input_paths = resolve_input_paths(args.cc_index_path)
    print(f"INPUT_PATHS={input_paths[:8]} COUNT={len(input_paths)}")

    counts, first_pass_rows = count_hosts(args, input_paths, host_buckets)
    if not counts:
        raise RuntimeError("No eligible HTML rows found in the CC index input")

    requested_hosts = args.max_hosts or (math.ceil(args.max_pages / args.max_pages_per_host) + 16)
    eligible_hosts = {host for host, count in counts.most_common(requested_hosts) if count >= args.min_host_pages}
    if not eligible_hosts:
        raise RuntimeError(
            f"No host had at least {args.min_host_pages} filtered page(s). "
            "Use a larger index slice or lower --min-host-pages."
        )

    selected, second_pass_rows = select_manifest_rows(args, input_paths, host_buckets, eligible_hosts)
    if selected.empty:
        raise RuntimeError("No manifest rows selected after host filtering")

    selected = selected.sort_values(
        ["host_bucket", "url_host_name", "url", "warc_filename", "warc_record_offset"],
        kind="stable",
    ).reset_index(drop=True)
    selected = selected.head(args.max_pages)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_path, index=False)

    metrics = {
        "input_paths": input_paths,
        "first_pass_index_rows": first_pass_rows,
        "second_pass_index_rows": second_pass_rows,
        "filtered_hosts": len(counts),
        "eligible_hosts": len(eligible_hosts),
        "selected_rows": len(selected),
        "selected_hosts": int(selected["url_host_name"].nunique()),
        "min_host_pages": args.min_host_pages,
        "max_pages_per_host": args.max_pages_per_host,
        "host_bucket_mod": args.host_bucket_mod,
        "host_buckets": sorted(host_buckets) if host_buckets is not None else None,
        "p50_selected_host_pages": float(selected.groupby("url_host_name").size().quantile(0.5)),
        "p95_selected_host_pages": float(selected.groupby("url_host_name").size().quantile(0.95)),
        "max_selected_host_pages": int(selected.groupby("url_host_name").size().max()),
    }
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OUTPUT={output_path}")
    print(f"METRICS={metrics_path}")
    print(json.dumps(metrics, sort_keys=True))
    return 0


def count_hosts(
    args: argparse.Namespace,
    input_paths: list[str],
    host_buckets: set[int] | None,
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    rows_seen = 0
    for batch in iter_filtered_batches(args, input_paths, host_buckets):
        rows_seen += int(batch.attrs.get("raw_rows", len(batch)))
        counts.update(batch["url_host_name"].tolist())
        if args.max_index_rows and rows_seen >= args.max_index_rows:
            break
    print(f"FIRST_PASS_ROWS={rows_seen} FILTERED_HOSTS={len(counts)}")
    return counts, rows_seen


def select_manifest_rows(
    args: argparse.Namespace,
    input_paths: list[str],
    host_buckets: set[int] | None,
    eligible_hosts: set[str],
) -> tuple[pd.DataFrame, int]:
    selected_rows: list[dict[str, Any]] = []
    host_selected: Counter[str] = Counter()
    rows_seen = 0

    for batch in iter_filtered_batches(args, input_paths, host_buckets):
        rows_seen += int(batch.attrs.get("raw_rows", len(batch)))
        batch = batch[batch["url_host_name"].isin(eligible_hosts)]
        if batch.empty:
            if args.max_index_rows and rows_seen >= args.max_index_rows:
                break
            continue

        for row in batch.to_dict("records"):
            host = row["url_host_name"]
            if host_selected[host] >= args.max_pages_per_host:
                continue
            selected_rows.append(row)
            host_selected[host] += 1
            if len(selected_rows) >= args.max_pages:
                break
        if len(selected_rows) >= args.max_pages:
            break
        if args.max_index_rows and rows_seen >= args.max_index_rows:
            break

    print(f"SECOND_PASS_ROWS={rows_seen} SELECTED_ROWS={len(selected_rows)} SELECTED_HOSTS={len(host_selected)}")
    return pd.DataFrame(selected_rows), rows_seen


def iter_filtered_batches(
    args: argparse.Namespace,
    input_paths: list[str],
    host_buckets: set[int] | None,
) -> Iterator[pd.DataFrame]:
    rows_seen = 0
    for batch in iter_index_batches(input_paths, batch_size=args.batch_size):
        raw_rows = len(batch)
        if args.max_index_rows:
            remaining = args.max_index_rows - rows_seen
            if remaining <= 0:
                break
            batch = batch.head(remaining)
            raw_rows = len(batch)
        rows_seen += raw_rows
        filtered = normalize_and_filter_batch(batch, args, host_buckets)
        filtered.attrs["raw_rows"] = raw_rows
        if not filtered.empty:
            yield filtered
        if args.max_index_rows and rows_seen >= args.max_index_rows:
            break


def iter_index_batches(input_paths: list[str], *, batch_size: int) -> Iterator[pd.DataFrame]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError:
        for path in input_paths:
            if Path(path).is_dir():
                raise RuntimeError("pyarrow is required to scan a parquet directory dataset")
            df = pd.read_parquet(path)
            keep_columns = [column for column in INDEX_COLUMNS if column in df.columns]
            df = df[keep_columns]
            for start in range(0, len(df), batch_size):
                yield df.iloc[start : start + batch_size].copy()
        return

    dataset_input: str | list[str] = input_paths[0] if len(input_paths) == 1 else input_paths
    dataset = ds.dataset(dataset_input, format="parquet", partitioning="hive")
    columns = [column for column in INDEX_COLUMNS if column in dataset.schema.names]
    missing = sorted({"url", "warc_filename"}.difference(columns))
    if missing:
        raise ValueError(f"CC index input is missing required columns: {missing}")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        yield record_batch.to_pandas()


def normalize_and_filter_batch(
    df: pd.DataFrame,
    args: argparse.Namespace,
    host_buckets: set[int] | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "fetch_status" not in work.columns and "http_status" in work.columns:
        work["fetch_status"] = work["http_status"]
    if "warc_record_offset" not in work.columns and "offset" in work.columns:
        work["warc_record_offset"] = work["offset"]
    if "warc_record_length" not in work.columns and "length" in work.columns:
        work["warc_record_length"] = work["length"]
    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in work.columns:
            raise ValueError(f"CC index input is missing required column: {column}")

    if "fetch_status" in work.columns:
        work = work[pd.to_numeric(work["fetch_status"], errors="coerce") == args.status]
    if args.html_only:
        html_mask = pd.Series(False, index=work.index)
        for column in ("content_mime_type", "content_mime_detected", "mime", "mime-detected"):
            if column in work.columns:
                html_mask |= work[column].fillna("").astype(str).str.contains("html", case=False, regex=False)
        work = work[html_mask]
    if args.language:
        lang_mask = pd.Series(False, index=work.index)
        for column in ("content_languages", "languages"):
            if column in work.columns:
                lang_mask |= work[column].fillna("").astype(str).str.contains(args.language, case=False, regex=False)
        work = work[lang_mask]
    if work.empty:
        return work

    if "url_host_name" not in work.columns:
        work["url_host_name"] = work["url"].map(url_host_key)
    else:
        work["url_host_name"] = work["url_host_name"].fillna("").astype(str).map(normalize_host)
        missing_host = work["url_host_name"] == ""
        if missing_host.any():
            work.loc[missing_host, "url_host_name"] = work.loc[missing_host, "url"].map(url_host_key)
    work = work[work["url_host_name"] != ""]
    if work.empty:
        return work

    work["host_bucket"] = work["url_host_name"].map(lambda host: xxhash_host_bucket(host, args.host_bucket_mod))
    if host_buckets is not None:
        work = work[work["host_bucket"].isin(host_buckets)]
    if work.empty:
        return work

    output_columns = [
        "url",
        "url_host_name",
        "host_bucket",
        "content_mime_type" if "content_mime_type" in work.columns else None,
        "content_mime_detected" if "content_mime_detected" in work.columns else None,
        "content_languages" if "content_languages" in work.columns else None,
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
    ]
    output_columns = [column for column in output_columns if column is not None]
    work = work[output_columns].dropna(subset=REQUIRED_OUTPUT_COLUMNS)
    work["warc_record_offset"] = pd.to_numeric(work["warc_record_offset"], errors="coerce")
    work["warc_record_length"] = pd.to_numeric(work["warc_record_length"], errors="coerce")
    work = work.dropna(subset=["warc_record_offset", "warc_record_length"])
    work["warc_record_offset"] = work["warc_record_offset"].astype("int64")
    work["warc_record_length"] = work["warc_record_length"].astype("int64")
    return work


def resolve_input_paths(path_or_glob: str) -> list[str]:
    if any(char in path_or_glob for char in "*?["):
        paths = sorted(glob(path_or_glob))
    else:
        path = Path(path_or_glob)
        if path.is_dir():
            paths = [str(path)]
        else:
            paths = [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"No CC index paths matched {path_or_glob!r}")
    return paths


def url_host_key(url_value: Any) -> str:
    if pd.isna(url_value):
        return ""
    url_text = str(url_value).strip()
    if not url_text:
        return ""
    try:
        host = urlparse(url_text).hostname or ""
    except ValueError:
        host = ""
    if not host and "://" not in url_text:
        try:
            host = urlparse(f"//{url_text}").hostname or ""
        except ValueError:
            host = ""
    return normalize_host(host)


def normalize_host(host: Any) -> str:
    if pd.isna(host):
        return ""
    host_text = str(host).strip().rstrip(".").lower()
    if not host_text:
        return ""
    try:
        return host_text.encode("idna").decode("ascii")
    except UnicodeError:
        return host_text


def xxhash_host_bucket(host: str, modulus: int) -> int:
    try:
        import xxhash
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xxhash is required to build llm-webkit-compatible host buckets. "
            "Install xxhash in the execution environment."
        ) from exc
    return int(xxhash.xxh64_intdigest(host) % modulus)


def parse_host_buckets(value: str | None) -> set[int] | None:
    if not value:
        return None
    buckets: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid host bucket range: {part}")
            buckets.update(range(start, end + 1))
        else:
            buckets.add(int(part))
    return buckets


if __name__ == "__main__":
    raise SystemExit(main())
