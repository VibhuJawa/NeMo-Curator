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

"""Build a global CC HTML/text dataset in Lance format using NeMo Curator.

Pipeline stages:
  1-3. DocumentDownloadExtractStage(extractor=None)
         URL generation → WARC download → WARC record iteration
         Output: url, content (HTML bytes), warc_id, source_id
  4.   HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura")
  5.   HtmlExtractStage(JusTextExtractor,     "cc_extracted_text_justext")
  6.   HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse")
  7.   LanceFragmentWriterStage — workers write lance fragments in parallel,
         no manifest contention; returns fragment metadata rows.

After pipeline.run() the driver calls lance_commit_fragments() which issues
a single LanceDataset.commit() covering all fragments atomically.

Usage:
  python build_cc_lancedb.py \\
      --snapshot CC-MAIN-2025-26 \\
      --download-dir /lustre/.../tmp_warcs \\
      --lance-uri s3://vjawa-cc-lance/cc_url_index

  # PBSS mirror (requires s5cmd + CC_PBSS_ACCESS_KEY_ID/CC_PBSS_SECRET_ACCESS_KEY):
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
from nemo_curator.stages.text.io.writer.lancedb import LanceFragmentWriterStage, lance_commit_fragments  # noqa: E402
from nemo_curator.stages.text.io.writer.utils import s3_storage_options_from_env  # noqa: E402
from nemo_curator.tasks import EmptyTask  # noqa: E402

_PBSS_ENDPOINT = "https://pdx.s8k.io"
_PBSS_WARC_BUCKET = "crawl-data"
_FETCH_CONCURRENCY = 24


def main(args: argparse.Namespace) -> None:
    write_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    write_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not write_key or not write_secret:
        logger.error("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")
        sys.exit(1)

    # CC PBSS read credentials — may differ from write account.
    cc_key_id = os.environ.get("CC_PBSS_ACCESS_KEY_ID") or write_key
    cc_secret = os.environ.get("CC_PBSS_SECRET_ACCESS_KEY") or write_secret

    # MainCommonCrawlUrlGenerator expects YYYY-WW, not "CC-MAIN-2025-26".
    snapshot_id = args.snapshot.removeprefix("CC-MAIN-").removeprefix("CC-NEWS-")
    url_gen_cls = NewsCommonCrawlUrlGenerator if args.crawl_type == "news" else MainCommonCrawlUrlGenerator
    url_generator = url_gen_cls(
        start_snapshot_str=snapshot_id,
        end_snapshot_str=snapshot_id,
        limit=args.url_limit,
    )

    downloader = CommonCrawlWARCDownloader(
        download_dir=args.download_dir,
        use_aws_to_download=args.pbss,
        s3_bucket=_PBSS_WARC_BUCKET if args.pbss else "commoncrawl",
        s3_endpoint_url=_PBSS_ENDPOINT if args.pbss else None,
        # Injected into s5cmd subprocess env only — keeps read/write creds isolated.
        s3_key_id=cc_key_id if args.pbss else None,
        s3_secret=cc_secret if args.pbss else None,
    )

    lance_storage_options = s3_storage_options_from_env()

    pipeline = Pipeline(
        name="cc_lance",
        stages=[
            DocumentDownloadExtractStage(
                url_generator=url_generator,
                downloader=downloader,
                iterator=CommonCrawlWarcIterator(),
                extractor=None,
                url_limit=args.url_limit,
            ),
            # Three independent actor stages — Ray pipelines them for 3x CPU utilisation.
            HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura", name="trafilatura_extract"),
            HtmlExtractStage(JusTextExtractor, "cc_extracted_text_justext", name="justext_extract"),
            HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse", name="resiliparse_extract"),
            # Writes lance fragments in parallel — no manifest contention.
            # Returns fragment metadata rows; driver commits atomically below.
            LanceFragmentWriterStage(
                uri=args.lance_uri,
                storage_options=lance_storage_options,
            ),
        ],
    )

    # 3 extract stages (1 CPU each) + fragment writer (2 CPUs) = 5 reserved CPUs.
    tasks = pipeline.run(
        RayDataExecutor(),
        initial_tasks=[EmptyTask(dataset_name="cc_lance")],
    )

    # Single atomic commit — one manifest write covering all fragments.
    lance_commit_fragments(tasks, uri=args.lance_uri, storage_options=lance_storage_options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a CC HTML/text dataset in Lance format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot", required=True, help="CC snapshot (e.g. CC-MAIN-2025-26).")
    parser.add_argument("--download-dir", required=True, help="Local directory for WARC downloads.")
    parser.add_argument("--lance-uri", required=True, help="Lance dataset URI (e.g. s3://bucket/dataset).")
    parser.add_argument("--pbss", action="store_true", help="Use PBSS mirror instead of public CC.")
    parser.add_argument("--crawl-type", choices=["main", "news"], default="main")
    parser.add_argument("--url-limit", type=int, default=None, help="Limit WARCs (testing).")

    main(parser.parse_args())
