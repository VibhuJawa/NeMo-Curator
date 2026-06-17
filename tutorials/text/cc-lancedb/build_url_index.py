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

"""Build a global CC URL index in LanceDB — thin Curator pipeline entry point.

Pipeline stages:
  1. CCIndexShardListStage    — list cc-index Parquet shards for the snapshot
  2. CCIndexParquetReaderStage — read each shard into DocumentBatches
  3. CCWarcByteRangeFetchStage — S3 byte-range fetch of raw HTML (128 workers)
  4. CCHtmlExtractStage        — trafilatura / justext / resiliparse extraction
  5. LanceDBWriter             — append to LanceDB table (batch_size=40 → 1 fragment)

CC read credentials come from the datamover config or env vars
(CC_PBSS_ACCESS_KEY_ID / CC_PBSS_SECRET_ACCESS_KEY).
LanceDB write credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.

Usage:
  python build_url_index.py --snapshot CC-MAIN-2024-10 \\
      --lancedb-uri s3://vjawa-cc-lance \\
      --table-name cc_url_index

  # Test run (2 shards only):
  python build_url_index.py --snapshot CC-MAIN-2024-10 --max-shards 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import ray  # noqa: E402
import yaml  # noqa: E402
from lancedb_writer import LANCEDB_URL_INDEX_SCHEMA  # noqa: E402
from loguru import logger  # noqa: E402

from nemo_curator.backends.ray_data import RayDataExecutor  # noqa: E402
from nemo_curator.pipeline import Pipeline  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.cc_html_extract import HtmlExtractStage  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.cc_index import (  # noqa: E402
    CCIndexParquetReaderStage,
    CCIndexShardListStage,
)
from nemo_curator.stages.text.download.common_crawl.warc_byte_range import (  # noqa: E402
    CCWarcByteRangeFetcher,
    CCWarcByteRangeFetchStage,
)
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.resiliparse import ResiliparseExtractor  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.trafilatura import TrafilaturaExtractor  # noqa: E402
from nemo_curator.stages.text.io.writer import LanceDBWriter  # noqa: E402
from nemo_curator.tasks import EmptyTask  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PBSS_ENDPOINT = "https://pdx.s8k.io"


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


def _load_cc_pbss_creds() -> tuple[str, str]:
    """Return (access_key_id, secret_access_key) for the PBSS pdx-commoncrawl namespace.

    Priority:
      1. Environment variables CC_PBSS_ACCESS_KEY_ID + CC_PBSS_SECRET_ACCESS_KEY.
      2. ~/.config/datamover/storage_locations YAML under key
         'pdx-commoncrawl' -> 'secrets' -> 'local' (or 'remote').
    Raises RuntimeError if credentials cannot be found.
    """
    key_id = os.environ.get("CC_PBSS_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("CC_PBSS_SECRET_ACCESS_KEY", "").strip()
    if key_id and secret:
        logger.debug("CC PBSS credentials loaded from environment variables.")
        return key_id, secret

    dm_path = Path.home() / ".config" / "datamover" / "storage_locations"
    if dm_path.exists():
        try:
            with dm_path.open() as fh:
                config = yaml.safe_load(fh) or {}
            ns = config.get("pdx-commoncrawl", {})
            secrets = ns.get("secrets", {})
            cred_block = secrets.get("local") or secrets.get("remote") or {}
            key_id = (cred_block.get("access_key_id") or cred_block.get("aws_access_key_id") or "").strip()
            secret = (cred_block.get("secret_access_key") or cred_block.get("aws_secret_access_key") or "").strip()
            if key_id and secret:
                logger.debug(f"CC PBSS credentials loaded from {dm_path}.")
                return key_id, secret
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to parse {dm_path}: {exc}")

    msg = (
        "CC PBSS credentials not found.  "
        "Set CC_PBSS_ACCESS_KEY_ID and CC_PBSS_SECRET_ACCESS_KEY, "
        "or populate ~/.config/datamover/storage_locations under 'pdx-commoncrawl.secrets.local'."
    )
    raise RuntimeError(msg)


def _build_storage_options(key_id: str, secret: str) -> dict:
    """Return the LanceDB 0.33 storage_options dict for PBSS (path-style S3-compat)."""
    return {
        "endpoint": _PBSS_ENDPOINT,
        "virtual_hosted_style_request": "false",
        "aws_access_key_id": key_id,
        "aws_secret_access_key": secret,
        "new_table_data_storage_version": "stable",
        "new_table_enable_v2_manifest_paths": "true",
        "io_threads": "128",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    # CC PBSS read credentials
    try:
        cc_key, cc_secret = _load_cc_pbss_creds()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # LanceDB write credentials
    write_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    write_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not write_key or not write_secret:
        logger.error("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — needed for LanceDB write")
        sys.exit(1)

    fetch_stage = CCWarcByteRangeFetchStage(
        fetcher=CCWarcByteRangeFetcher(
            use_s3=args.use_s3,
            s3_key_id=cc_key if args.use_s3 else None,
            s3_secret=cc_secret if args.use_s3 else None,
        ),
        # max_workers uses class default (16) — matches Curator's WARC downloader.
    )

    writer = LanceDBWriter(
        uri=args.lancedb_uri,
        table_name=args.table_name,
        schema=LANCEDB_URL_INDEX_SCHEMA,
        storage_options=_build_storage_options(write_key, write_secret),
        # Each process() call accumulates this many blocks before one tbl.add().
        # Tune so fragment_rows = batch_size x chunk_size ~= 100K-1M rows.
        # No compaction needed - each call creates one permanent, right-sized fragment.
        batch_size=args.lancedb_batch_size,
    )

    pipeline = Pipeline(
        name="cc_url_index",
        stages=[
            CCIndexShardListStage(
                snapshot=args.snapshot,
                cc_key=cc_key,
                cc_secret=cc_secret,
                max_shards=args.max_shards,
            ),
            CCIndexParquetReaderStage(
                cc_key=cc_key,
                cc_secret=cc_secret,
                chunk_size=args.chunk_size,
                min_warc_length=args.min_warc_length,
                max_batches=args.max_batches,
            ),
            fetch_stage,
            # Three independent actor stages - Ray pipelines them so all three extractors
            # run on different blocks simultaneously, using 3x more CPU than a single stage.
            HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura", name="trafilatura_extract"),
            HtmlExtractStage(JusTextExtractor, "cc_extracted_text_justext", name="justext_extract"),
            HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse", name="resiliparse_extract"),
            writer,
        ],
    )

    initial_tasks = [EmptyTask(dataset_name="cc_url_index")]

    # PBSS throttles at ~400 concurrent S3 connections.  Derive the CPU cap from
    # the measured safe limit so it auto-adjusts if max_workers is ever changed.
    pbss_connection_limit = 384  # ~400 throttle with 4 % headroom
    fetch_threads = fetch_stage.max_workers  # read from stage — stays in sync
    fetch_actors = pbss_connection_limit // fetch_threads  # = 24 with default 16
    reserved_cpus = 7  # 3 extract stages (1 CPU each) + 1 writer (4 CPUs)
    ray.init(
        num_cpus=fetch_actors + reserved_cpus,
        _temp_dir=f"/tmp/ray_{os.environ.get('USER', 'user')}",  # noqa: S108
        ignore_reinit_error=True,
    )

    pipeline.run(RayDataExecutor(), initial_tasks=initial_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a global CC URL index in LanceDB using the NeMo Curator pipeline. "
            "Reads CC index shards from PBSS, performs byte-range WARC fetches, "
            "extracts HTML text, and appends to a LanceDB table via RayDataExecutor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="CC snapshot to process (e.g. CC-MAIN-2024-10).",
    )
    parser.add_argument(
        "--lancedb-uri",
        type=str,
        required=True,
        help="LanceDB root URI (local path or S3-compatible URL, e.g. s3://my-bucket).",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="cc_url_index",
        help="LanceDB table name.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit Parquet shards per snapshot (testing only).",
    )
    parser.add_argument(
        "--min-warc-length",
        type=int,
        default=5_000,
        help="Skip WARC records smaller than this (bytes). "
        "Records < 5 KB are typically redirects with no extractable HTML. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many DocumentBatches per shard (testing only). "
        "Each batch is chunk_size rows (default 5000). E.g. --max-batches 3 = 15 000 rows.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Rows per DocumentBatch from the CC index Parquet reader.",
    )
    parser.add_argument(
        "--lancedb-batch-size",
        type=int,
        default=100,
        help="Number of DocumentBatches accumulated before each LanceDB tbl.add() call. "
        "Each call creates one permanent fragment: fragment_rows = batch_size x chunk_size. "
        "No compaction is needed - tune so fragments are 100K-1M rows each.",
    )
    parser.add_argument(
        "--use-s3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch WARCs from PBSS S3 (default). Pass --no-use-s3 to use the public CC HTTP endpoint.",
    )

    main(parser.parse_args())
