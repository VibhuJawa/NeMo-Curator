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

"""Distributed Lance scalar index builder using Ray.

Protocol (lance 7.x fragment_ids API):
  1. Each Ray worker calls create_scalar_index(fragment_ids=batch, index_uuid=SHARED_UUID)
     — builds a partial index segment for its assigned fragments, writes to S3,
     does NOT commit.
  2. Coordinator calls ds.merge_index_metadata(index_uuid, index_type)
     — merges all segment metadata into one Index object.
  3. Coordinator commits via lance.LanceDataset.commit(LanceOperation.CreateIndex(...))

This replaces the single-node create_scalar_index() call for petabyte-scale tables.

Usage:
  python build_lance_index.py \\
      --lancedb-uri s3://vjawa-cc-lance \\
      --table-name  cc_url_index \\
      --n-workers   64

  # With explicit Ray cluster:
  python build_lance_index.py --ray-address auto --n-workers 128
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from loguru import logger

# Make repo root importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# PBSS storage options
# ---------------------------------------------------------------------------


def _storage_options() -> dict:
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", "https://pdx.s8k.io")
    return {
        "endpoint": endpoint,
        "virtual_hosted_style_request": "false",
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }


def _lance_uri(lancedb_uri: str, table_name: str) -> str:
    """Convert LanceDB URI + table_name into a raw Lance dataset path."""
    return f"{lancedb_uri.rstrip('/')}/{table_name}.lance"


# ---------------------------------------------------------------------------
# Ray remote: build one index segment over a batch of fragments
# ---------------------------------------------------------------------------


def _build_segment_remote(
    lance_uri: str,
    column: str,
    index_type: str,
    fragment_ids: list[int],
    index_uuid: str,
    storage_opts: dict,
    worker_id: int,
) -> str:
    """Ray remote task: build partial index for *fragment_ids*, do NOT commit."""
    import lance

    ds = lance.dataset(lance_uri, storage_options=storage_opts)
    ds.create_scalar_index(
        column,
        index_type,
        fragment_ids=fragment_ids,
        index_uuid=index_uuid,
        replace=False,
    )
    return f"worker {worker_id}: {len(fragment_ids)} fragments OK"


# ---------------------------------------------------------------------------
# Distributed index builder
# ---------------------------------------------------------------------------


def build_distributed_index(
    lance_uri: str,
    column: str,
    index_type: str,
    n_workers: int,
    storage_opts: dict,
) -> None:
    """Build a Lance scalar index distributed across *n_workers* Ray tasks."""
    import lance
    from lance import LanceOperation

    try:
        import ray
    except ImportError:
        raise RuntimeError("ray is required for distributed index building")

    ds = lance.dataset(lance_uri, storage_options=storage_opts)
    all_fragments = [f.fragment_id for f in ds.get_fragments()]

    logger.info(f"Building {index_type} index on '{column}': {len(all_fragments):,} fragments → {n_workers} workers")

    # Shared UUID ties all worker segments together
    index_uuid = str(uuid.uuid4())

    # Round-robin split into n_workers batches
    batches = [all_fragments[i::n_workers] for i in range(n_workers)]
    batches = [b for b in batches if b]  # drop empty batches

    # Register remote function lazily so ray.init() is already done
    build_segment = ray.remote(_build_segment_remote)

    # Launch all workers in parallel
    logger.info(f"Launching {len(batches)} Ray tasks …")
    futures = [
        build_segment.remote(lance_uri, column, index_type, batch, index_uuid, storage_opts, i)
        for i, batch in enumerate(batches)
    ]

    # Wait for all segments to be written
    results = ray.get(futures)
    for r in results:
        logger.info(f"  {r}")

    # Merge all segment metadata on the coordinator
    logger.info("Merging index segment metadata …")
    merged_index = ds.merge_index_metadata(
        index_uuid=index_uuid,
        index_type=index_type,
    )

    # Commit: publish the merged index into the dataset manifest
    logger.info("Committing index to manifest …")
    lance.LanceDataset.commit(
        lance_uri,
        LanceOperation.CreateIndex(
            new_indices=[merged_index],
            removed_indices=[],
        ),
        read_version=ds.version,
        storage_options=storage_opts,
    )

    logger.info(f"Index '{column}' ({index_type}) committed — {len(all_fragments):,} fragments covered.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INDEXES = [
    # (column,          index_type)
    ("cc_url", "BTREE"),  # high-cardinality URL lookup
    ("cc_snapshot_id", "BITMAP"),  # 121 distinct values, instant filter
    ("url_host_name", "BTREE"),  # domain-level queries
]


def main(args: argparse.Namespace) -> None:
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            logger.error(f"Missing required env var: {var}")
            sys.exit(1)

    storage_opts = _storage_options()
    lance_uri = _lance_uri(args.lancedb_uri, args.table_name)

    import ray

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    logger.info(f"Lance URI : {lance_uri}")
    logger.info(f"Workers   : {args.n_workers}")
    logger.info(f"Columns   : {[c for c, _ in INDEXES]}")

    for column, index_type in INDEXES:
        logger.info(f"=== Indexing: {column} ({index_type}) ===")
        build_distributed_index(
            lance_uri=lance_uri,
            column=column,
            index_type=index_type,
            n_workers=args.n_workers,
            storage_opts=storage_opts,
        )

    logger.info("All indexes built. Cleaning up old versions …")
    from datetime import timedelta

    import lance

    ds = lance.dataset(lance_uri, storage_options=storage_opts)
    ds.cleanup_old_versions(timedelta(hours=1))

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Lance scalar index builder (Ray + fragment_ids API)")
    parser.add_argument("--lancedb-uri", default="s3://vjawa-cc-lance")
    parser.add_argument("--table-name", default="cc_url_index")
    parser.add_argument(
        "--n-workers", type=int, default=64, help="Ray workers (each handles total_fragments / n_workers fragments)"
    )
    parser.add_argument("--ray-address", default=None, help="Ray cluster address (default: start local Ray)")
    main(parser.parse_args())
