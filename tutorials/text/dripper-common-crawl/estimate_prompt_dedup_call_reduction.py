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

"""Estimate Dripper call-reduction potential before GPU inference.

This is a CPU-only diagnostic for the Common Crawl Dripper workflow. It reads
host-bucketed CC index shards, selects high-reuse host samples, range-fetches
the corresponding WARC records, runs the MinerU/Dripper preprocessing stage,
hashes the exact ``(prompt, request_max_tokens)`` request surface, and can
optionally estimate host-bounded DOM-layout representative calls with the
llm-webkit clustering primitives used by the AICC §2.1.2 path.

The estimator deliberately stores prompt hashes and aggregate counts only. It
does not persist prompt text or LLM responses. When ``--sample-output`` is
provided, it writes a runnable manifest that keeps the selected page HTML/WARC
columns plus prompt hashes so the same sample can be used for GPU A/B tests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

PROMPT_COL = "_dripper_prompt"
NEEDS_LLM_COL = "_dripper_needs_llm"
EMPTY_INPUT_COL = "_dripper_empty_input"
PRIMARY_ERROR_COL = "_dripper_primary_error"
REQUIRED_WARC_COLUMNS = ["url", "url_host_name", "warc_filename", "warc_record_offset", "warc_record_length"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate exact Dripper prompt dedup from CC manifests")
    parser.add_argument("--input", required=True, help="Host-bucketed parquet shard dir, file, or glob")
    parser.add_argument("--output", required=True, help="Output JSON metrics path")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all matching files")
    parser.add_argument(
        "--host-bucket-groups",
        default=None,
        help="Optional comma/range filter over host_bucket_group values in file names, e.g. 0,7,10-19.",
    )
    parser.add_argument("--count-max-rows", type=int, default=0, help="Optional cap for the host-counting pass")
    parser.add_argument("--select-max-rows", type=int, default=0, help="Optional cap for the row-selection pass")
    parser.add_argument("--top-hosts", type=int, default=16)
    parser.add_argument("--min-host-pages", type=int, default=2)
    parser.add_argument("--max-pages-per-host", type=int, default=512)
    parser.add_argument("--max-pages", type=int, default=8192, help="Maximum WARC rows to fetch/preprocess")
    parser.add_argument("--manifest-warc-bucket", default=os.environ.get("DRIPPER_MANIFEST_WARC_BUCKET", "crawl-data"))
    parser.add_argument("--manifest-fetch-workers", type=int, default=64)
    parser.add_argument(
        "--s3-endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    )
    parser.add_argument("--s3-region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--html-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-html-bytes", type=int, default=1)
    parser.add_argument("--prompt-version", default="short_compact")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dynamic-max-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-max-token-padding", type=int, default=16)
    parser.add_argument("--dynamic-max-tokens-per-item", type=int, default=6)
    parser.add_argument("--dynamic-min-max-tokens", type=int, default=32)
    parser.add_argument("--preprocess-batch-size", type=int, default=128)
    parser.add_argument("--top-prompt-groups", type=int, default=20)
    parser.add_argument("--layout-estimate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layout-cluster-threshold", type=float, default=0.95)
    parser.add_argument("--layout-min-cluster-size", type=int, default=2)
    parser.add_argument("--layout-max-exact-host-pages", type=int, default=2048)
    parser.add_argument("--top-layout-clusters", type=int, default=20)
    parser.add_argument(
        "--sample-output",
        default=None,
        help="Optional parquet path for a GPU-runnable sample manifest plus per-row hash diagnostics",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_files < 0:
        raise ValueError("--max-files must be non-negative")
    if args.count_max_rows < 0 or args.select_max_rows < 0:
        raise ValueError("--count-max-rows and --select-max-rows must be non-negative")
    if args.top_hosts <= 0:
        raise ValueError("--top-hosts must be positive")
    if args.min_host_pages <= 0:
        raise ValueError("--min-host-pages must be positive")
    if args.max_pages_per_host <= 0:
        raise ValueError("--max-pages-per-host must be positive")
    if args.max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    if args.manifest_fetch_workers <= 0:
        raise ValueError("--manifest-fetch-workers must be positive")
    if args.min_html_bytes < 0:
        raise ValueError("--min-html-bytes must be non-negative")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.dynamic_max_token_padding < 0:
        raise ValueError("--dynamic-max-token-padding must be non-negative")
    if args.dynamic_max_tokens_per_item <= 0:
        raise ValueError("--dynamic-max-tokens-per-item must be positive")
    if args.dynamic_min_max_tokens <= 0:
        raise ValueError("--dynamic-min-max-tokens must be positive")
    if args.preprocess_batch_size <= 0:
        raise ValueError("--preprocess-batch-size must be positive")
    if args.top_prompt_groups < 0:
        raise ValueError("--top-prompt-groups must be non-negative")
    if not 0.0 < args.layout_cluster_threshold <= 1.0:
        raise ValueError("--layout-cluster-threshold must be in (0, 1]")
    if args.layout_min_cluster_size <= 1:
        raise ValueError("--layout-min-cluster-size must be greater than 1")
    if args.layout_max_exact_host_pages < 0:
        raise ValueError("--layout-max-exact-host-pages must be non-negative")
    if args.top_layout_clusters < 0:
        raise ValueError("--top-layout-clusters must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    manifest_files = resolve_manifest_files(args.input, parse_int_ranges(args.host_bucket_groups))
    if args.max_files:
        manifest_files = manifest_files[: args.max_files]
    if not manifest_files:
        raise FileNotFoundError(f"No manifest parquet files matched {args.input!r}")

    print(
        "PROMPT_DEDUP_ESTIMATE_INPUT "
        f"files={len(manifest_files)} top_hosts={args.top_hosts} max_pages={args.max_pages} "
        f"max_pages_per_host={args.max_pages_per_host}",
        flush=True,
    )

    count_started = time.perf_counter()
    host_counts, count_rows = count_hosts(manifest_files, batch_size=args.batch_size, max_rows=args.count_max_rows)
    selected_hosts = select_top_hosts(host_counts, top_hosts=args.top_hosts, min_host_pages=args.min_host_pages)
    count_elapsed_s = time.perf_counter() - count_started
    print(
        "PROMPT_DEDUP_ESTIMATE_HOSTS "
        f"count_rows={count_rows} total_hosts={len(host_counts)} selected_hosts={len(selected_hosts)} "
        f"top_host_pages={selected_hosts[0][1] if selected_hosts else 0}",
        flush=True,
    )

    select_started = time.perf_counter()
    candidate_df, selection_stats = select_manifest_rows(
        manifest_files,
        selected_hosts=[host for host, _count in selected_hosts],
        batch_size=args.batch_size,
        max_pages=args.max_pages,
        max_pages_per_host=args.max_pages_per_host,
        max_rows=args.select_max_rows,
    )
    if candidate_df.empty:
        raise RuntimeError("Selected no candidate WARC rows for prompt dedup estimation")

    fetch_started = time.perf_counter()
    pages, fetch_stats = fetch_manifest_warc_pages(candidate_df, args=args)
    if not pages:
        raise RuntimeError("Fetched no HTML pages for prompt dedup estimation")

    preprocess_started = time.perf_counter()
    processed_df = preprocess_pages(pages, args=args)
    row_df, prompt_metrics = hash_preprocessed_pages(processed_df, args=args)
    layout_metrics = estimate_layout_cluster_calls(processed_df, row_df, args=args) if args.layout_estimate else None

    metrics = {
        "input": args.input,
        "files": [str(path) for path in manifest_files],
        "file_count": len(manifest_files),
        "count_rows": count_rows,
        "total_hosts_seen": len(host_counts),
        "selected_hosts": [{"host": host, "count": count} for host, count in selected_hosts],
        "candidate_rows": len(candidate_df),
        "candidate_hosts": int(candidate_df["url_host_name"].map(normalize_host).nunique()),
        "selection_stats": selection_stats,
        "fetch_stats": fetch_stats,
        "prompt_metrics": prompt_metrics,
        "layout_metrics": layout_metrics,
        "timings_s": {
            "count_hosts_s": count_elapsed_s,
            "select_rows_s": fetch_started - select_started,
            "fetch_pages_s": preprocess_started - fetch_started,
            "preprocess_hash_s": time.perf_counter() - preprocess_started,
            "total_s": time.perf_counter() - started,
        },
        "args": {
            "batch_size": args.batch_size,
            "max_files": args.max_files,
            "host_bucket_groups": args.host_bucket_groups,
            "count_max_rows": args.count_max_rows,
            "select_max_rows": args.select_max_rows,
            "top_hosts": args.top_hosts,
            "min_host_pages": args.min_host_pages,
            "max_pages_per_host": args.max_pages_per_host,
            "max_pages": args.max_pages,
            "manifest_warc_bucket": args.manifest_warc_bucket,
            "manifest_fetch_workers": args.manifest_fetch_workers,
            "html_only": args.html_only,
            "min_html_bytes": args.min_html_bytes,
            "prompt_version": args.prompt_version,
            "max_tokens": args.max_tokens,
            "dynamic_max_tokens": args.dynamic_max_tokens,
            "preprocess_batch_size": args.preprocess_batch_size,
            "layout_estimate": args.layout_estimate,
            "layout_cluster_threshold": args.layout_cluster_threshold,
            "layout_min_cluster_size": args.layout_min_cluster_size,
            "layout_max_exact_host_pages": args.layout_max_exact_host_pages,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if args.sample_output:
        sample_path = Path(args.sample_output)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df = build_sample_output_dataframe(processed_df, row_df)
        sample_df.to_parquet(sample_path, index=False)
        metrics["sample_output"] = str(sample_path)
        metrics["sample_output_mode"] = "runnable_manifest_with_hash_diagnostics"
        metrics["sample_output_rows"] = len(sample_df)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("PROMPT_DEDUP_ESTIMATE_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("PROMPT_DEDUP_ESTIMATE_END")
    print(f"OUTPUT={output_path}")
    return 0


def build_sample_output_dataframe(processed_df: pd.DataFrame, row_df: pd.DataFrame) -> pd.DataFrame:
    """Build a GPU-runnable sample manifest without persisting prompt text."""
    if len(processed_df) != len(row_df):
        raise ValueError(
            "processed_df and row_df must have the same length to build a row-aligned sample output: "
            f"{len(processed_df)} != {len(row_df)}"
        )

    sample_df = processed_df.reset_index(drop=True).copy()
    sample_df = sample_df.drop(columns=[PROMPT_COL], errors="ignore")

    diagnostics = row_df.reset_index(drop=True).copy()
    renamed_columns: dict[str, str] = {}
    for column in diagnostics.columns:
        output_column = column
        if output_column in sample_df.columns:
            output_column = f"prompt_dedup_{column}"
        renamed_columns[column] = output_column
    diagnostics = diagnostics.rename(columns=renamed_columns)

    return pd.concat([sample_df, diagnostics], axis=1)


def count_hosts(manifest_files: list[Path], *, batch_size: int, max_rows: int) -> tuple[Counter[str], int]:
    import pyarrow.parquet as pq

    counts: Counter[str] = Counter()
    rows_seen = 0
    for path in manifest_files:
        parquet_file = pq.ParquetFile(path)
        require_columns(path, parquet_file.schema_arrow.names, ["url_host_name"])
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["url_host_name"], use_threads=True):
            hosts = batch.column("url_host_name").to_pylist()
            if max_rows and rows_seen + len(hosts) > max_rows:
                hosts = hosts[: max_rows - rows_seen]
            rows_seen += len(hosts)
            counts.update(host for host in (normalize_host(value) for value in hosts) if host)
            if max_rows and rows_seen >= max_rows:
                return counts, rows_seen
    return counts, rows_seen


def select_top_hosts(host_counts: Counter[str], *, top_hosts: int, min_host_pages: int) -> list[tuple[str, int]]:
    return [
        (host, count)
        for host, count in sorted(host_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_host_pages
    ][:top_hosts]


def select_manifest_rows(
    manifest_files: list[Path],
    *,
    selected_hosts: list[str],
    batch_size: int,
    max_pages: int,
    max_pages_per_host: int,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import pyarrow.parquet as pq

    selected_host_set = set(selected_hosts)
    selected_by_host: Counter[str] = Counter()
    rows_scanned = 0
    frames: list[pd.DataFrame] = []
    selected_total = 0
    columns = REQUIRED_WARC_COLUMNS

    for path in manifest_files:
        parquet_file = pq.ParquetFile(path)
        require_columns(path, parquet_file.schema_arrow.names, columns)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns, use_threads=True):
            df = batch.to_pandas()
            if max_rows and rows_scanned + len(df) > max_rows:
                df = df.head(max_rows - rows_scanned)
            rows_scanned += len(df)
            df["_normalized_host"] = df["url_host_name"].map(normalize_host)
            df = df[df["_normalized_host"].isin(selected_host_set)]
            if not df.empty:
                keep_indexes: list[int] = []
                for row_index, host in df["_normalized_host"].items():
                    if selected_by_host[host] >= max_pages_per_host:
                        continue
                    if selected_total >= max_pages:
                        break
                    selected_by_host[host] += 1
                    selected_total += 1
                    keep_indexes.append(row_index)
                if keep_indexes:
                    frames.append(df.loc[keep_indexes].drop(columns=["_normalized_host"]))
            if selected_total >= max_pages:
                return (
                    pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns),
                    {
                        "rows_scanned": rows_scanned,
                        "selected_by_host": dict(selected_by_host),
                        "stopped_by_max_pages": True,
                        "stopped_by_max_rows": bool(max_rows and rows_scanned >= max_rows),
                    },
                )
            if max_rows and rows_scanned >= max_rows:
                return (
                    pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns),
                    {
                        "rows_scanned": rows_scanned,
                        "selected_by_host": dict(selected_by_host),
                        "stopped_by_max_pages": False,
                        "stopped_by_max_rows": True,
                    },
                )

    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns),
        {
            "rows_scanned": rows_scanned,
            "selected_by_host": dict(selected_by_host),
            "stopped_by_max_pages": False,
            "stopped_by_max_rows": False,
        },
    )


def fetch_manifest_warc_pages(
    manifest_df: pd.DataFrame, *, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = make_s3_client(args)
    rows = manifest_df.to_dict("records")
    pages: list[dict[str, Any] | None] = [None] * len(rows)
    stats: dict[str, Any] = {
        "requested_rows": len(rows),
        "loaded_pages": 0,
        "fetch_failed": 0,
        "skipped_non_html": 0,
        "skipped_min_bytes": 0,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.manifest_fetch_workers) as executor:
        futures = {
            executor.submit(fetch_manifest_warc_page, client, args.manifest_warc_bucket, row, args): index
            for index, row in enumerate(rows)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                page = future.result()
            except Exception as exc:
                stats["fetch_failed"] += 1
                print(f"PROMPT_DEDUP_FETCH_WARNING row={index} error={exc!r}", flush=True)
                continue
            if page is None:
                stats["skipped_non_html"] += 1
                continue
            pages[index] = page

    loaded = [page for page in pages if page is not None]
    stats["loaded_pages"] = len(loaded)
    return loaded, stats


def fetch_manifest_warc_page(
    client: Any, default_bucket: str, row: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any] | None:
    from warcio.archiveiterator import ArchiveIterator

    filename = str(row["warc_filename"])
    offset = int(row["warc_record_offset"])
    length = int(row["warc_record_length"])
    bucket, key = parse_manifest_warc_location(default_bucket, filename)
    end_byte = offset + length - 1
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{end_byte}")
    raw_bytes = response["Body"].read()
    try:
        decompressed = gzip.decompress(raw_bytes)
    except gzip.BadGzipFile:
        decompressed = raw_bytes

    for record in ArchiveIterator(io.BytesIO(decompressed), arc2warc=True):
        if record.rec_type != "response":
            continue
        content_type = ""
        if record.http_headers is not None:
            content_type = record.http_headers.get_header("Content-Type") or ""
        if args.html_only and "html" not in content_type.lower():
            return None
        html = record.content_stream().read()
        if len(html) < args.min_html_bytes:
            return None
        warc_id = record.rec_headers.get_header("WARC-Record-ID") or ""
        return {
            **row,
            "url": row.get("url") or record.rec_headers.get_header("WARC-Target-URI"),
            "url_host_name": row.get("url_host_name") or normalize_host_from_url(row.get("url")),
            "warc_id": warc_id.strip("<>"),
            "warc_filename": key,
            "content_type": content_type,
            "html": html,
        }
    return None


def preprocess_and_hash_pages(
    pages: list[dict[str, Any]], *, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed_df = preprocess_pages(pages, args=args)
    return hash_preprocessed_pages(processed_df, args=args)


def preprocess_pages(pages: list[dict[str, Any]], *, args: argparse.Namespace) -> pd.DataFrame:
    from nemo_curator.models.client.llm_client import GenerationConfig
    from nemo_curator.stages.text.experimental.dripper import DripperHTMLPreprocessStage
    from nemo_curator.tasks import DocumentBatch

    generation_config = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0, top_p=args.top_p)
    stage = DripperHTMLPreprocessStage(
        html_col="html",
        url_col="url",
        prompt_version=args.prompt_version,
        generation_config=generation_config,
        dynamic_max_tokens=args.dynamic_max_tokens,
        dynamic_max_token_padding=args.dynamic_max_token_padding,
        dynamic_max_tokens_per_item=args.dynamic_max_tokens_per_item,
        dynamic_min_max_tokens=args.dynamic_min_max_tokens,
    )
    stage.setup()

    frames: list[pd.DataFrame] = []
    for batch_index, start in enumerate(range(0, len(pages), args.preprocess_batch_size)):
        batch_pages = pages[start : start + args.preprocess_batch_size]
        batch = DocumentBatch(
            task_id=f"prompt-dedup-estimate-{batch_index:06d}",
            dataset_name="CC-MAIN-2025-26-prompt-dedup-estimate",
            data=pd.DataFrame(batch_pages),
        )
        frames.append(stage.process(batch).to_pandas())
        print(
            f"PROMPT_DEDUP_PREPROCESS_BATCH index={batch_index} rows={len(batch_pages)}",
            flush=True,
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hash_preprocessed_pages(df: pd.DataFrame, *, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_records: list[dict[str, Any]] = []
    prompt_counts: Counter[str] = Counter()
    host_prompt_counts: Counter[str] = Counter()
    prompt_hosts: dict[str, set[str]] = defaultdict(set)
    prompt_example_urls: dict[str, list[str]] = defaultdict(list)
    item_counts: Counter[int] = Counter()
    prompt_char_counts: Counter[int] = Counter()
    request_max_tokens_counts: Counter[int] = Counter()

    for row_index, row in df.iterrows():
        host = normalize_host(row.get("url_host_name")) or normalize_host_from_url(row.get("url"))
        needs_llm = bool(row.get(NEEDS_LLM_COL, False))
        prompt = str(row.get(PROMPT_COL, "") or "")
        request_max_tokens = coerce_int(row.get("dripper_request_max_tokens"))
        prompt_hash = ""
        request_key = ""
        if needs_llm and prompt.strip():
            prompt_hash = hash_text(prompt)
            request_key = f"{prompt_hash}:{request_max_tokens}"
            prompt_counts[request_key] += 1
            host_prompt_counts[f"{host}\0{request_key}"] += 1
            prompt_hosts[request_key].add(host)
            if len(prompt_example_urls[request_key]) < 3:
                prompt_example_urls[request_key].append(str(row.get("url") or ""))
        item_counts[coerce_int(row.get("dripper_item_count"))] += 1
        prompt_char_counts[coerce_int(row.get("dripper_prompt_chars"))] += 1
        request_max_tokens_counts[request_max_tokens] += 1
        row_records.append(
            {
                "row_index": row_index,
                "url": row.get("url"),
                "url_host_name": host,
                "needs_llm": needs_llm,
                "empty_input": bool(row.get(EMPTY_INPUT_COL, False)),
                "warning": str(row.get("dripper_warning") or ""),
                "primary_error": str(row.get(PRIMARY_ERROR_COL) or ""),
                "item_count": coerce_int(row.get("dripper_item_count")),
                "prompt_chars": coerce_int(row.get("dripper_prompt_chars")),
                "request_max_tokens": request_max_tokens,
                "prompt_hash": prompt_hash,
                "request_key": request_key,
            }
        )

    row_df = pd.DataFrame(row_records)
    needs_llm_pages = int(row_df["needs_llm"].sum()) if "needs_llm" in row_df else 0
    unique_prompt_requests = len(prompt_counts)
    unique_host_prompt_requests = len(host_prompt_counts)
    exact_prompt_saved_pages = sum(count - 1 for count in prompt_counts.values() if count > 1)
    host_prompt_saved_pages = sum(count - 1 for count in host_prompt_counts.values() if count > 1)
    top_prompt_groups = [
        {
            "request_key": key,
            "pages": int(count),
            "hosts": len(prompt_hosts.get(key, set())),
            "example_urls": prompt_example_urls.get(key, []),
        }
        for key, count in prompt_counts.most_common(args.top_prompt_groups)
        if count > 1
    ]

    return row_df, {
        "pages": len(row_df),
        "needs_llm_pages": needs_llm_pages,
        "fallback_only_pages": int(len(row_df) - needs_llm_pages),
        "empty_input_pages": int(row_df["empty_input"].sum()) if "empty_input" in row_df else 0,
        "warning_pages": int((row_df["warning"].astype(str) != "").sum()) if "warning" in row_df else 0,
        "primary_error_pages": int((row_df["primary_error"].astype(str) != "").sum())
        if "primary_error" in row_df
        else 0,
        "unique_prompt_requests": unique_prompt_requests,
        "exact_prompt_saved_pages": int(exact_prompt_saved_pages),
        "exact_prompt_call_ratio": safe_ratio(unique_prompt_requests, needs_llm_pages),
        "exact_prompt_reduction_factor": safe_ratio(needs_llm_pages, unique_prompt_requests),
        "unique_host_prompt_requests": unique_host_prompt_requests,
        "host_prompt_saved_pages": int(host_prompt_saved_pages),
        "host_prompt_call_ratio": safe_ratio(unique_host_prompt_requests, needs_llm_pages),
        "host_prompt_reduction_factor": safe_ratio(needs_llm_pages, unique_host_prompt_requests),
        "prompt_group_size_quantiles": histogram_quantiles(Counter(prompt_counts.values())),
        "host_prompt_group_size_quantiles": histogram_quantiles(Counter(host_prompt_counts.values())),
        "item_count_quantiles": histogram_quantiles(item_counts),
        "prompt_chars_quantiles": histogram_quantiles(prompt_char_counts),
        "request_max_tokens_counts": dict(request_max_tokens_counts),
        "top_prompt_groups": top_prompt_groups,
    }


def estimate_layout_cluster_calls(
    processed_df: pd.DataFrame,
    row_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Estimate one-LLM-call-per-host-layout-cluster savings.

    This estimates the scheduling opportunity only. It does not claim CPU
    propagation accuracy; that still needs GPU representative inference and
    output comparison against pure Dripper.
    """
    from llm_web_kit.html_layout.html_layout_cosin import cluster_html_struct, get_feature
    from llm_web_kit.main_html_parser.typical_html.typical_html import select_representative_html

    if processed_df.empty or row_df.empty:
        return {
            "pages": 0,
            "needs_llm_pages": 0,
            "estimated_llm_requests_with_layout": 0,
            "layout_estimate_note": "empty input",
        }

    request_key_by_row = {
        int(row["row_index"]): str(row.get("request_key") or "")
        for _idx, row in row_df.iterrows()
        if bool(row.get("needs_llm", False)) and str(row.get("request_key") or "")
    }
    samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_error_pages = 0
    feature_none_pages = 0
    no_html_pages = 0
    needs_llm_pages = 0

    for row_index, row in processed_df.iterrows():
        if row_index not in request_key_by_row:
            continue
        needs_llm_pages += 1
        html_text = coerce_html(row.get("html", ""))
        if not html_text.strip():
            no_html_pages += 1
            continue
        try:
            feature = get_feature(html_text)
        except Exception as exc:
            feature_error_pages += 1
            print(f"LAYOUT_ESTIMATE_FEATURE_WARNING row={row_index} error={exc!r}", flush=True)
            continue
        if feature is None:
            feature_none_pages += 1
            continue
        host = normalize_host(row.get("url_host_name")) or normalize_host_from_url(row.get("url"))
        samples_by_host[host].append(
            {
                "track_id": str(row_index),
                "html": html_text,
                "feature": feature,
                "url": str(row.get("url") or ""),
            }
        )

    covered_by_layout: set[int] = set()
    representative_rows: set[int] = set()
    layout_call_keys: set[str] = set()
    layout_clusters: list[dict[str, Any]] = []
    host_metrics: list[dict[str, Any]] = []
    clustering_error_hosts = 0
    skipped_large_host_pages = 0

    sorted_hosts = sorted(samples_by_host.items(), key=lambda item: (-len(item[1]), item[0]))
    for host_rank, (host, samples) in enumerate(sorted_hosts):
        host_clustered_pages = 0
        host_cluster_count = 0
        host_representatives = 0
        host_errors = 0
        print(
            f"LAYOUT_ESTIMATE_HOST_BEGIN rank={host_rank} host={host!r} feature_pages={len(samples)}",
            flush=True,
        )
        if args.layout_max_exact_host_pages and len(samples) > args.layout_max_exact_host_pages:
            skipped_large_host_pages += len(samples)
            host_metrics.append(
                {
                    "host": host,
                    "feature_pages": len(samples),
                    "clustered_pages": 0,
                    "layout_clusters": 0,
                    "representative_calls": 0,
                    "standalone_pages": len(samples),
                    "skipped_large_host": True,
                }
            )
            print(
                "LAYOUT_ESTIMATE_HOST_END "
                f"rank={host_rank} host={host!r} feature_pages={len(samples)} "
                "skipped_large_host=1 clustered_pages=0 layout_clusters=0",
                flush=True,
            )
            continue
        if len(samples) >= args.layout_min_cluster_size:
            try:
                clustered_samples, _layout_ids = cluster_html_struct(
                    samples,
                    threshold=args.layout_cluster_threshold,
                )
            except Exception as exc:
                clustering_error_hosts += 1
                host_errors += 1
                print(f"LAYOUT_ESTIMATE_CLUSTER_WARNING host={host!r} error={exc!r}", flush=True)
                clustered_samples = []
        else:
            clustered_samples = []

        by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = int(sample.get("layout_id", -1))
            if layout_id >= 0:
                by_layout[layout_id].append(sample)

        for layout_id, cluster_samples in sorted(by_layout.items()):
            if len(cluster_samples) < args.layout_min_cluster_size:
                continue
            indexes = sorted(int(sample["track_id"]) for sample in cluster_samples)
            representative_idx = select_representative_row(cluster_samples, select_representative_html)
            request_key = request_key_by_row.get(representative_idx, "")
            if not request_key:
                continue
            covered_by_layout.update(indexes)
            representative_rows.add(representative_idx)
            layout_call_keys.add(request_key)
            host_clustered_pages += len(indexes)
            host_cluster_count += 1
            host_representatives += 1
            distinct_prompt_requests = len(
                {request_key_by_row.get(index, "") for index in indexes if request_key_by_row.get(index, "")}
            )
            layout_clusters.append(
                {
                    "host": host,
                    "layout_id": int(layout_id),
                    "pages": len(indexes),
                    "distinct_prompt_requests": distinct_prompt_requests,
                    "representative_row_index": representative_idx,
                    "representative_url": str(processed_df.loc[representative_idx].get("url") or ""),
                    "saved_vs_exact_prompt_requests": max(0, distinct_prompt_requests - 1),
                }
            )

        host_metrics.append(
            {
                "host": host,
                "feature_pages": len(samples),
                "clustered_pages": host_clustered_pages,
                "layout_clusters": host_cluster_count,
                "representative_calls": host_representatives,
                "standalone_pages": len(samples) - host_clustered_pages,
                "cluster_errors": host_errors,
            }
        )
        print(
            "LAYOUT_ESTIMATE_HOST_END "
            f"rank={host_rank} host={host!r} feature_pages={len(samples)} "
            f"clustered_pages={host_clustered_pages} layout_clusters={host_cluster_count} "
            f"representative_calls={host_representatives} cluster_errors={host_errors}",
            flush=True,
        )

    standalone_request_keys = {
        request_key
        for row_index, request_key in request_key_by_row.items()
        if row_index not in covered_by_layout and request_key
    }
    combined_request_keys = layout_call_keys | standalone_request_keys
    unique_prompt_requests = len(set(request_key_by_row.values()))
    estimated_llm_requests = len(combined_request_keys)
    clustered_pages = len(covered_by_layout)
    representative_pages = len(representative_rows)
    top_clusters = sorted(
        layout_clusters,
        key=lambda item: (
            -int(item["saved_vs_exact_prompt_requests"]),
            -int(item["pages"]),
            item["host"],
            item["layout_id"],
        ),
    )[: args.top_layout_clusters]

    return {
        "pages": len(row_df),
        "needs_llm_pages": needs_llm_pages,
        "feature_ok_pages": sum(len(samples) for samples in samples_by_host.values()),
        "feature_error_pages": feature_error_pages,
        "feature_none_pages": feature_none_pages,
        "no_html_pages": no_html_pages,
        "hosts_with_features": len(samples_by_host),
        "clustering_error_hosts": clustering_error_hosts,
        "skipped_large_host_pages": skipped_large_host_pages,
        "layout_cluster_threshold": args.layout_cluster_threshold,
        "layout_min_cluster_size": args.layout_min_cluster_size,
        "layout_cluster_count": len(layout_clusters),
        "layout_clustered_pages": clustered_pages,
        "layout_representative_pages": representative_pages,
        "layout_standalone_feature_pages": max(
            0, sum(len(samples) for samples in samples_by_host.values()) - clustered_pages
        ),
        "unique_prompt_requests": unique_prompt_requests,
        "estimated_llm_requests_with_layout": estimated_llm_requests,
        "layout_estimated_saved_pages": max(0, needs_llm_pages - estimated_llm_requests),
        "layout_estimated_call_ratio": safe_ratio(estimated_llm_requests, needs_llm_pages),
        "layout_estimated_reduction_factor": safe_ratio(needs_llm_pages, estimated_llm_requests),
        "layout_additional_saved_vs_exact_prompt_requests": max(0, unique_prompt_requests - estimated_llm_requests),
        "layout_call_ratio_vs_exact_prompt": safe_ratio(estimated_llm_requests, unique_prompt_requests),
        "top_layout_clusters": top_clusters,
        "top_hosts": sorted(
            host_metrics,
            key=lambda item: (
                -int(item.get("clustered_pages", 0)),
                -int(item.get("feature_pages", 0)),
                str(item.get("host", "")),
            ),
        )[:20],
        "layout_estimate_note": "call-reduction estimate only; CPU propagation accuracy must be validated against pure Dripper",
    }


def select_representative_row(cluster_samples: list[dict[str, Any]], selector: Any) -> int:
    representative = None
    try:
        representative = selector(
            [{"track_id": sample["track_id"], "html": sample["html"]} for sample in cluster_samples]
        )
    except Exception as exc:
        print(f"LAYOUT_ESTIMATE_REPRESENTATIVE_WARNING error={exc!r}", flush=True)
    if isinstance(representative, dict):
        try:
            return int(representative["track_id"])
        except (KeyError, TypeError, ValueError):
            pass
    return int(cluster_samples[0]["track_id"])


def make_s3_client(args: argparse.Namespace) -> Any:
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError("boto3 is required to stream Common Crawl WARC data from S3/PBSS") from exc

    if is_pbss_endpoint(args.s3_endpoint_url) and os.environ.get("PBSS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["PBSS_ACCESS_KEY_ID"]
    if is_pbss_endpoint(args.s3_endpoint_url) and os.environ.get("PBSS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["PBSS_SECRET_ACCESS_KEY"]  # pragma: allowlist secret

    return boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint_url,
        region_name=args.s3_region,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            read_timeout=120,
            max_pool_connections=max(10, int(args.manifest_fetch_workers)),
        ),
    )


def is_pbss_endpoint(endpoint_url: str | None) -> bool:
    return bool(endpoint_url and "pdx.s8k.io" in endpoint_url)


def parse_manifest_warc_location(default_bucket: str, filename: str) -> tuple[str, str]:
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
    if bucket == "crawl-data" and key.startswith("crawl-data/"):
        key = key.removeprefix("crawl-data/")
    return bucket, key


def resolve_manifest_files(input_value: str, host_bucket_groups: set[int] | None) -> list[Path]:
    if any(char in input_value for char in "*?["):
        paths = [Path(path) for path in glob(input_value)]
    else:
        path = Path(input_value)
        if path.is_dir():
            paths = sorted(path.glob("host_bucket_group=*.parquet"))
            if not paths:
                paths = sorted(path.glob("host_bucket_group=*/*.parquet"))
            if not paths:
                paths = sorted(path.rglob("*.parquet"))
        else:
            paths = [path]
    files = [path for path in paths if path.suffix == ".parquet" and not path.name.startswith("_")]
    if host_bucket_groups is not None:
        files = [path for path in files if host_bucket_group_from_path(path) in host_bucket_groups]
    return sorted(files)


def host_bucket_group_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"host_bucket_group=(\d+)", part)
        if match:
            return int(match.group(1))
    match = re.search(r"host_bucket_group=(\d+)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer host_bucket_group from path: {path}")


def parse_int_ranges(value: str | None) -> set[int] | None:
    if not value:
        return None
    numbers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return numbers


def require_columns(path: Path, schema_names: list[str], required: list[str]) -> None:
    missing = sorted(set(required).difference(schema_names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def normalize_host(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().rstrip(".")
    if not text or text == "nan":
        return ""
    try:
        return text.encode("idna").decode("ascii")
    except UnicodeError:
        return text


def normalize_host_from_url(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if not parsed.hostname and "://" not in text:
            parsed = urlparse(f"//{text}")
    except ValueError:
        return ""
    return normalize_host(parsed.hostname)


def coerce_html(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def coerce_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def histogram_quantiles(hist: Counter[int]) -> dict[str, float | int]:
    total = sum(hist.values())
    if total == 0:
        return {"count": 0}
    targets = {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    out: dict[str, float | int] = {"count": int(total), "mean": weighted_mean(hist), "max": max(hist)}
    seen = 0
    pending = sorted(targets.items(), key=lambda item: item[1])
    pending_index = 0
    for size, count in sorted(hist.items()):
        seen += count
        while pending_index < len(pending) and seen >= math.ceil(total * pending[pending_index][1]):
            out[pending[pending_index][0]] = int(size)
            pending_index += 1
    return out


def weighted_mean(hist: Counter[int]) -> float:
    total = sum(hist.values())
    if not total:
        return 0.0
    return sum(size * count for size, count in hist.items()) / total


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
