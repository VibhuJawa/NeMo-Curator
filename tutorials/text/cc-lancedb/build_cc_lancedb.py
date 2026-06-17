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

"""Build a global CC URL index in LanceDB using upstream Curator stages.

Pipeline (7 stages):
  1-3. DocumentDownloadExtractStage(extractor=None)
         -- internally: URL generation -> WARC download -> WARC iteration
         -- outputs raw records: url, content (HTML bytes), warc_id, source_id
  4.  HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura")
  5.  HtmlExtractStage(JusTextExtractor,     "cc_extracted_text_justext")
  6.  HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse")
         — three independent actor stages pipelined for 3x CPU utilisation
  7.  LanceDBWriter — appends to a LanceDB table on PBSS S3

Usage (public CC via HTTPS):
  python build_url_index.py \\
      --snapshot CC-MAIN-2025-26 \\
      --download-dir /lustre/fsw/.../tmp_warcs \\
      --lancedb-uri s3://vjawa-cc-lance \\
      --table-name cc_url_index

  # Use PBSS mirror (requires s5cmd):
  python build_url_index.py ... --pbss
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import ray  # noqa: E402
from loguru import logger  # noqa: E402

from nemo_curator.backends.ray_data import RayDataExecutor  # noqa: E402
from nemo_curator.pipeline import Pipeline  # noqa: E402
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.cc_html_extract import HtmlExtractStage  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader  # noqa: E402
from nemo_curator.stages.text.download.common_crawl.url_generation import (  # noqa: E402
    MainCommonCrawlUrlGenerator,
    NewsCommonCrawlUrlGenerator,
)
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.resiliparse import ResiliparseExtractor  # noqa: E402
from nemo_curator.stages.text.download.html_extractors.trafilatura import TrafilaturaExtractor  # noqa: E402
from nemo_curator.stages.text.io.writer import LanceDBWriter  # noqa: E402
from nemo_curator.tasks import EmptyTask  # noqa: E402

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

_PBSS_ENDPOINT = "https://pdx.s8k.io"
_PBSS_WARC_BUCKET = "crawl-data"  # PBSS mirror of CC WARCs

# PBSS S3 throttle: ~400 concurrent connections / 16 connections per actor = 24 fetch actors.
_FETCH_CONCURRENCY = 24


def _build_lancedb_storage_options(key_id: str, secret: str) -> dict:
    """LanceDB 0.33 storage_options dict for PBSS (path-style S3)."""
    return {
        "endpoint": _PBSS_ENDPOINT,
        "virtual_hosted_style_request": "false",
        "aws_access_key_id": key_id,
        "aws_secret_access_key": secret,
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "new_table_data_storage_version": "stable",
        "new_table_enable_v2_manifest_paths": "true",
        "io_threads": "128",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    write_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    write_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not write_key or not write_secret:
        logger.error("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — needed for LanceDB write")
        sys.exit(1)

    # URL generator for the target snapshot
    url_gen_cls = NewsCommonCrawlUrlGenerator if args.crawl_type == "news" else MainCommonCrawlUrlGenerator
    url_generator = url_gen_cls(
        start_snapshot_str=args.snapshot,
        end_snapshot_str=args.snapshot,
        limit=args.url_limit,
    )

    # Downloader: public CC (HTTPS/wget) or PBSS mirror (s5cmd + custom endpoint)
    downloader = CommonCrawlWARCDownloader(
        download_dir=args.download_dir,
        use_aws_to_download=args.pbss,
        s3_bucket=_PBSS_WARC_BUCKET if args.pbss else "commoncrawl",
        s3_endpoint_url=_PBSS_ENDPOINT if args.pbss else None,
    )

    # Download-only stage: extractor=None → raw WARC records flow downstream.
    # Records contain: url, content (HTML bytes), warc_id, source_id
    download_stage = DocumentDownloadExtractStage(
        url_generator=url_generator,
        downloader=downloader,
        iterator=CommonCrawlWarcIterator(),
        extractor=None,
        url_limit=args.url_limit,
    )

    writer = LanceDBWriter(
        uri=args.lancedb_uri,
        table_name=args.table_name,
        storage_options=_build_lancedb_storage_options(write_key, write_secret),
        batch_size=args.lancedb_batch_size,
    )

    pipeline = Pipeline(
        name="cc_url_index",
        stages=[
            download_stage,
            # Three independent actor stages — Ray pipelines them so all three
            # extractors run on different blocks simultaneously (3x CPU use).
            # input_column="content" matches the upstream CommonCrawlWarcIterator output.
            HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura", name="trafilatura_extract"),
            HtmlExtractStage(JusTextExtractor, "cc_extracted_text_justext", name="justext_extract"),
            HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse", name="resiliparse_extract"),
            writer,
        ],
    )

    # PBSS throttles at ~400 concurrent S3 connections (relevant for PBSS write).
    reserved_cpus = 7  # 3 extract stages (1 CPU each) + 1 writer (4 CPUs)
    ray.init(
        num_cpus=_FETCH_CONCURRENCY + reserved_cpus,
        _temp_dir=f"/tmp/ray_{os.environ.get('USER', 'user')}",  # noqa: S108
        ignore_reinit_error=True,
    )

    initial_tasks = [EmptyTask(dataset_name="cc_url_index")]
    pipeline.run(RayDataExecutor(), initial_tasks=initial_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a CC URL index in LanceDB using the Curator WARC download pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot", type=str, required=True, help="CC snapshot (e.g. CC-MAIN-2025-26).")
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Local directory to store downloaded WARC files (e.g. /lustre/.../tmp_warcs).",
    )
    parser.add_argument("--lancedb-uri", type=str, required=True, help="LanceDB root URI (e.g. s3://vjawa-cc-lance).")
    parser.add_argument("--table-name", type=str, default="cc_url_index", help="LanceDB table name.")
    parser.add_argument(
        "--pbss",
        action="store_true",
        default=False,
        help="Download WARCs from PBSS mirror instead of public CC. Requires s5cmd.",
    )
    parser.add_argument("--crawl-type", choices=["main", "news"], default="main", help="CC crawl type.")
    parser.add_argument("--url-limit", type=int, default=None, help="Limit WARC URLs to process (testing).")
    parser.add_argument(
        "--lancedb-batch-size",
        type=int,
        default=5_000,
        help="Rows per tbl.add() call. Match to the iterator chunk_size for one fragment per call.",
    )

    main(parser.parse_args())
