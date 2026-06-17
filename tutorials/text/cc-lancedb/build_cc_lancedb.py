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

Pipeline:
  1-3. DocumentDownloadExtractStage(extractor=None)
         — URL generation → WARC download → WARC iteration
         — outputs raw records: url, content (HTML bytes), warc_id, source_id
  4.   HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura")
  5.   HtmlExtractStage(JusTextExtractor,     "cc_extracted_text_justext")
  6.   HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse")
         — three independent actor stages pipelined for 3x CPU utilisation

Write: lance_ray.LanceDatasink via run_stages_to_lance()
  Workers write lance fragments in parallel (no manifest contention).
  A single LanceDataset.commit() closes the dataset atomically.

Usage (public CC via HTTPS):
  python build_cc_lancedb.py \\
      --snapshot CC-MAIN-2025-26 \\
      --download-dir /lustre/.../tmp_warcs \\
      --lancedb-uri s3://vjawa-cc-lance/cc_url_index

  # Use PBSS mirror (requires s5cmd + CC_PBSS_ACCESS_KEY_ID / CC_PBSS_SECRET_ACCESS_KEY):
  python build_cc_lancedb.py ... --pbss
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

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
from nemo_curator.stages.text.io.writer.lancedb import run_stages_to_lance  # noqa: E402
from nemo_curator.tasks import EmptyTask  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PBSS_ENDPOINT = "https://pdx.s8k.io"
_PBSS_WARC_BUCKET = "crawl-data"  # PBSS mirror of CC WARCs

# PBSS S3 throttle: ~400 concurrent connections / 16 connections per actor = 24 fetch actors.
_FETCH_CONCURRENCY = 24


def _build_lance_storage_options(key_id: str, secret: str) -> dict:
    """lance_ray storage_options dict for PBSS (path-style S3)."""
    return {
        "aws_endpoint": _PBSS_ENDPOINT,
        "aws_access_key_id": key_id,
        "aws_secret_access_key": secret,
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "virtual_hosted_style_request": "false",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    # LanceDB write credentials — always required.
    write_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    write_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not write_key or not write_secret:
        logger.error("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — needed for Lance write")
        sys.exit(1)

    # CC PBSS read credentials — may differ from write account.
    # Falls back to write creds for single-account setups.
    cc_key_id = os.environ.get("CC_PBSS_ACCESS_KEY_ID") or write_key
    cc_secret = os.environ.get("CC_PBSS_SECRET_ACCESS_KEY") or write_secret

    # MainCommonCrawlUrlGenerator expects YYYY-WW, not the full "CC-MAIN-2025-26".
    snapshot_id = args.snapshot.removeprefix("CC-MAIN-").removeprefix("CC-NEWS-")
    url_gen_cls = NewsCommonCrawlUrlGenerator if args.crawl_type == "news" else MainCommonCrawlUrlGenerator
    url_generator = url_gen_cls(
        start_snapshot_str=snapshot_id,
        end_snapshot_str=snapshot_id,
        limit=args.url_limit,
    )

    # s3_key_id/s3_secret are injected only into the s5cmd subprocess env,
    # keeping WARC read credentials isolated from LanceDB write credentials.
    downloader = CommonCrawlWARCDownloader(
        download_dir=args.download_dir,
        use_aws_to_download=args.pbss,
        s3_bucket=_PBSS_WARC_BUCKET if args.pbss else "commoncrawl",
        s3_endpoint_url=_PBSS_ENDPOINT if args.pbss else None,
        s3_key_id=cc_key_id if args.pbss else None,
        s3_secret=cc_secret if args.pbss else None,
    )

    # extractor=None: raw WARC records flow downstream as-is.
    # Output columns: url, content (HTML bytes), warc_id, source_id
    download_stage = DocumentDownloadExtractStage(
        url_generator=url_generator,
        downloader=downloader,
        iterator=CommonCrawlWarcIterator(),
        extractor=None,
        url_limit=args.url_limit,
    )

    # Three independent actor stages — Ray pipelines them so all three run on
    # different blocks simultaneously for 3x CPU utilisation.
    # input_column="content" matches the CommonCrawlWarcIterator output column.
    stages = [
        download_stage,
        HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura", name="trafilatura_extract"),
        HtmlExtractStage(JusTextExtractor, "cc_extracted_text_justext", name="justext_extract"),
        HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse", name="resiliparse_extract"),
    ]

    # lance_ray.LanceDatasink: workers write fragments in parallel; one atomic commit.
    from lance_ray import LanceDatasink

    datasink = LanceDatasink(
        uri=args.lancedb_uri,
        mode="create",
        storage_options=_build_lance_storage_options(write_key, write_secret),
    )

    # Cap Ray to the CPUs allocated by Slurm (--cpus-per-task=N).
    # _FETCH_CONCURRENCY download actors x 1 CPU each for S3 I/O.
    # reserved_cpus: 3 extract stages (1 CPU each) — writer is now the datasink.
    reserved_cpus = 3
    run_stages_to_lance(
        stages=stages,
        datasink=datasink,
        initial_tasks=[EmptyTask(dataset_name="cc_url_index")],
        ray_init_kwargs={
            "num_cpus": _FETCH_CONCURRENCY + reserved_cpus,
            "_temp_dir": os.environ["RAY_TMPDIR"],
        },
    )


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
        help="Local directory for downloaded WARC files.",
    )
    parser.add_argument("--lancedb-uri", type=str, required=True, help="Lance dataset URI (e.g. s3://bucket/table).")
    parser.add_argument(
        "--pbss",
        action="store_true",
        default=False,
        help="Download WARCs from PBSS mirror instead of public CC. Requires s5cmd.",
    )
    parser.add_argument("--crawl-type", choices=["main", "news"], default="main", help="CC crawl type.")
    parser.add_argument("--url-limit", type=int, default=None, help="Limit WARC URLs to process (testing).")

    main(parser.parse_args())
