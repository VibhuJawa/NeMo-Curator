# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""CLI script to download Common Crawl WARC files, extract HTML bytes + text using
CCHTMLBytesExtractor, and write all records to a LanceDB table on PBSS/SwiftStack.
"""

import argparse
import os
from pathlib import Path

import ray
from cc_html_bytes_extractor import CCHTMLBytesExtractor
from lancedb_writer import LANCEDB_CC_SCHEMA
from loguru import logger

from nemo_curator.backends.ray_actor_pool.executor import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader
from nemo_curator.stages.text.download.common_crawl.url_generation import (
    MainCommonCrawlUrlGenerator,
    NewsCommonCrawlUrlGenerator,
)
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator
from nemo_curator.stages.text.io.writer import LanceDBWriter


def main(args):
    # 1. Warn if AWS credentials are not set
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        logger.warning("AWS_ACCESS_KEY_ID is not set in the environment.")
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        logger.warning("AWS_SECRET_ACCESS_KEY is not set in the environment.")

    # 2. Create local temp download directory
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    # 3. Initialize Ray
    # Override RAY_TMPDIR before init to avoid AF_UNIX 107-byte path limit on Lustre.
    if not args.ray_address:
        os.environ["RAY_TMPDIR"] = f"/tmp/ray_{os.environ.get('USER', 'user')}"
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    # 4. Build URL generator based on crawl type
    if args.crawl_type == "main":
        url_generator = MainCommonCrawlUrlGenerator(
            start_snapshot_str=args.start_snapshot,
            end_snapshot_str=args.end_snapshot,
            limit=args.url_limit,
        )
    else:
        url_generator = NewsCommonCrawlUrlGenerator(
            start_snapshot_str=args.start_snapshot,
            end_snapshot_str=args.end_snapshot,
            limit=args.url_limit,
        )

    # 5. Build downloader
    downloader = CommonCrawlWARCDownloader(args.download_dir, use_aws_to_download=args.use_aws)

    # 6. Build iterator
    iterator = CommonCrawlWarcIterator()

    # 7. Build extractor
    extractor = CCHTMLBytesExtractor(extractor_lib=args.extractor_lib)

    # 8. Build the download/extract stage
    cc_stage = DocumentDownloadExtractStage(
        url_generator=url_generator,
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        url_limit=args.url_limit,
        record_limit=args.record_limit,
    )
    cc_stage.name = f"cc_download_extract_{args.crawl_type}"

    # 9. Build the LanceDB writer stage with the CC-specific schema
    writer = LanceDBWriter(uri=args.lancedb_uri, table_name=args.table_name, schema=LANCEDB_CC_SCHEMA)

    # 10. Assemble the pipeline
    pipeline = Pipeline(name="cc_lancedb_pipeline", description="CC → LanceDB on PBSS")
    pipeline.add_stage(cc_stage)
    pipeline.add_stage(writer)

    # 11. Build the Ray actor pool executor
    # RayActorPoolExecutor does not accept num_workers; worker count is derived
    # automatically from the Ray cluster's available resources.
    executor = RayActorPoolExecutor()

    # 12. Run the pipeline
    results = pipeline.run(executor)

    # 13. Log summary
    total = sum(t.num_items for t in results) if results else 0
    logger.info(f"Pipeline complete. Total records written: {total}")
    logger.info(f"LanceDB table: {args.lancedb_uri}/{args.table_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CC LanceDB index on PBSS")

    parser.add_argument(
        "--start-snapshot",
        type=str,
        required=True,
        help='Starting Common Crawl snapshot, e.g. "CC-MAIN-2025-26"',
    )
    parser.add_argument(
        "--end-snapshot",
        type=str,
        required=True,
        help='Ending Common Crawl snapshot, e.g. "CC-MAIN-2025-26" (same as start = single snapshot)',
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Local temporary directory for downloaded WARC files",
    )
    parser.add_argument(
        "--lancedb-uri",
        type=str,
        default="s3://pdx-commoncrawl/cc_lancedb",
        help="LanceDB URI on PBSS/SwiftStack (default: s3://pdx-commoncrawl/cc_lancedb)",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="cc_snapshot_index",
        help="LanceDB table name (default: cc_snapshot_index)",
    )
    parser.add_argument(
        "--crawl-type",
        type=str,
        choices=["main", "news"],
        default="main",
        help="Common Crawl crawl type: main or news (default: main)",
    )
    parser.add_argument(
        "--extractor-lib",
        type=str,
        choices=["trafilatura", "justext", "resiliparse"],
        default="trafilatura",
        help="HTML text extraction library (default: trafilatura)",
    )
    parser.add_argument(
        "--url-limit",
        type=int,
        default=None,
        help="Maximum number of WARC URLs to process (default: no limit)",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=None,
        help="Maximum number of records per WARC file (default: no limit)",
    )
    parser.add_argument(
        "--use-aws",
        action="store_true",
        default=False,
        help="Use s5cmd instead of wget for downloading WARC files",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address; if not provided, a local Ray instance is started",
    )

    main(parser.parse_args())
