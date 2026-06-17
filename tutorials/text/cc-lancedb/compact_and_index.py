# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Compact a LanceDB table that has been populated by Slurm array tasks and
# build scalar indexes for efficient URL and snapshot lookups.
#
# Run after all array tasks have finished:
#   python compact_and_index.py \
#       --lancedb-uri s3://vjawa-cc-lance \
#       --table-name  cc_url_index
#
# Required environment variables for PBSS (S3-compatible) storage:
#   AWS_ENDPOINT_URL_S3       e.g. https://pdx.s8k.io
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY

import argparse
import os
import sys
import time
from datetime import timedelta


def build_storage_options() -> dict:
    """Collect PBSS / S3-compatible credentials from the environment.

    LanceDB 0.33 storage_options keys are passed to the object-store Rust layer,
    which uses "endpoint" (not boto3's "endpoint_url").  Path-style requests
    (virtual_hosted_style_request=false) are required for PBSS/SwiftStack.
    """
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", "https://pdx.s8k.io")
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

    missing = [
        name
        for name, val in [
            ("AWS_ACCESS_KEY_ID", key),
            ("AWS_SECRET_ACCESS_KEY", secret),
        ]
        if not val
    ]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    return {
        # LanceDB/object-store key is "endpoint", not "endpoint_url" (that's boto3).
        "endpoint": endpoint,
        "aws_access_key_id": key,
        "aws_secret_access_key": secret,
        # Required for PBSS / SwiftStack: disable virtual-hosted-style S3 requests.
        "virtual_hosted_style_request": "false",
        # Stable storage format and V2 manifest paths for new tables.
        "new_table_data_storage_version": "stable",
        "new_table_enable_v2_manifest_paths": "true",
        "io_threads": "128",
    }


def compact(table, num_threads: int) -> None:
    print("Running compact_files() ...")
    t0 = time.monotonic()
    table.compact_files(
        max_bytes_per_file=8_589_934_592,  # 8 GiB per file (~160K rows at 50KB/row avg)
        target_rows_per_fragment=4_000_000,  # 4 M rows (bytes limit wins for html data)
        materialize_deletions=True,
        defer_index_remap=True,
        num_threads=num_threads,
        batch_size=256,
    )
    elapsed = time.monotonic() - t0
    print(f"compact_files() finished in {elapsed:.1f}s")


def cleanup(table) -> None:
    print("Running cleanup_old_versions() ...")
    t0 = time.monotonic()
    table.cleanup_old_versions(older_than=timedelta(hours=1))
    elapsed = time.monotonic() - t0
    print(f"cleanup_old_versions() finished in {elapsed:.1f}s")


def create_indexes(table) -> None:
    """Create scalar indexes suitable for URL and snapshot lookups."""
    indexes = [
        ("cc_url", "BTREE"),
        ("cc_snapshot_id", "BITMAP"),
        ("url_host_name", "BTREE"),
    ]
    for column, index_type in indexes:
        print(f"Creating {index_type} index on '{column}' ...")
        t0 = time.monotonic()
        table.create_scalar_index(column, index_type=index_type, replace=True)
        elapsed = time.monotonic() - t0
        print(f"  Done in {elapsed:.1f}s")


def print_stats(table) -> None:
    row_count = table.count_rows()
    stats = table.stats()
    print("\n--- Table stats ---")
    print(f"  Row count : {row_count:,}")
    print(f"  Stats     : {stats}")
    print("-------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compact a LanceDB Common Crawl URL index table and build scalar "
            "indexes.  Run once after all Slurm array tasks have finished."
        )
    )
    parser.add_argument(
        "--lancedb-uri",
        required=True,
        help="LanceDB URI, e.g. s3://vjawa-cc-lance",
    )
    parser.add_argument(
        "--table-name",
        required=True,
        help="Name of the LanceDB table to compact and index",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of threads for compact_files() (default: 32)",
    )
    args = parser.parse_args()

    storage_options = build_storage_options()

    print(f"Connecting to LanceDB at {args.lancedb_uri} ...")
    import lancedb  # local import so argparse --help works without lancedb installed

    db = lancedb.connect(args.lancedb_uri, storage_options=storage_options)

    print(f"Opening table '{args.table_name}' ...")
    table = db.open_table(args.table_name)

    print_stats(table)

    compact(table, num_threads=args.num_threads)
    cleanup(table)
    create_indexes(table)

    print_stats(table)
    print("compact_and_index.py completed successfully.")


if __name__ == "__main__":
    main()
