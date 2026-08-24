# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import os
from pathlib import Path
from typing import Literal

import pyarrow as pa
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.html_extractors import HTMLExtractorAlgorithm
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask

from .download import CommonCrawlWARCDownloader
from .extract import CommonCrawlHTMLExtractor
from .url_generation import MainCommonCrawlUrlGenerator, NewsCommonCrawlUrlGenerator
from .warc_iterator import CommonCrawlWarcIterator


class CommonCrawlWARCManifestSourceStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Emit deterministic WARC tasks from a frozen ``warc.paths`` file.

    The manifest is the immutable input contract. It is prepared once outside
    the GPU pipeline; this stage performs no snapshot discovery or index query.
    Lines may be full URLs or official Common Crawl paths such as
    ``crawl-data/CC-MAIN-...``.
    """

    resources = Resources(cpus=0.5)

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.name = "common_crawl_warc_manifest_source"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: EmptyTask) -> list[FileGroupTask]:
        manifest = Path(self.manifest_path)
        warcs = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
        if not warcs:
            msg = f"Common Crawl WARC manifest is empty: {manifest}"
            raise ValueError(msg)
        return [
            FileGroupTask(
                dataset_name=task.dataset_name,
                data=[path],
                _metadata={"warc_path": path, "warc_manifest": str(manifest)},
            )
            for path in warcs
        ]

    def num_workers(self) -> int:
        return 1


class CommonCrawlWARCDownloadAndReadStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Download and consume each WARC on the same Ray worker.

    Keeping these operations in one stage is required when ``download_dir`` is
    node-local storage: separate download and iteration stages may be scheduled
    on different nodes. Raw response bodies can be independently Zstandard
    compressed as they are read, so the resulting batch never retains a second
    uncompressed copy of an entire WARC. Records are emitted in bounded batches
    so downstream CPU stages can scale independently of WARC file boundaries.
    """

    batch_size = 1

    def __init__(  # noqa: PLR0913
        self,
        downloader: CommonCrawlWARCDownloader,
        *,
        content_field: str = "content",
        compression: Literal["none", "zstd"] = "none",
        cleanup: bool = True,
        workers_per_node: int = 2,
        records_per_batch: int = 1024,
    ):
        if workers_per_node < 1:
            msg = "workers_per_node must be positive"
            raise ValueError(msg)
        if compression not in ("none", "zstd"):
            msg = f"Unsupported content compression: {compression}"
            raise ValueError(msg)
        if records_per_batch < 1:
            msg = "records_per_batch must be positive"
            raise ValueError(msg)
        self.downloader = downloader
        self.iterator = CommonCrawlWarcIterator()
        self.content_field = content_field
        self.compression = compression
        self.cleanup = cleanup
        self.workers_per_node = workers_per_node
        self.records_per_batch = records_per_batch
        self.name = "common_crawl_warc_download_and_read"
        self.resources = Resources(cpus=1.0)
        self._compressor = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [self.content_field if column == "content" else column for column in self.iterator.output_columns()]
        return ["data"], columns

    def ray_stage_spec(self) -> dict:
        # Ray Data slices the returned task block for the downstream stage's
        # batch_size=1.  Marking this as a generic fan-out stage inserts an
        # extra one-row repartition, which can materialize many WARC batches
        # and starve downstream work under object-store backpressure.
        return {
            RayStageSpecKeys.IS_ACTOR_STAGE: True,
            RayStageSpecKeys.MIN_WORKERS: self.workers_per_node,
            RayStageSpecKeys.MAX_WORKERS: self.workers_per_node,
            RayStageSpecKeys.INITIAL_WORKERS: self.workers_per_node,
            # A call expands one complete WARC. Do not queue a second WARC on
            # an actor while its multi-GiB fan-out enters Ray Data.
            RayStageSpecKeys.MAX_TASKS_IN_FLIGHT_PER_ACTOR: 1,
        }

    def _compress(self, content: bytes) -> bytes:
        if self.compression == "none":
            return content
        if self._compressor is None:
            from zstandard import ZstdCompressor

            self._compressor = ZstdCompressor()
        return self._compressor.compress(content)

    def _batch(self, task: FileGroupTask, records: list[dict], chunk_index: int) -> DocumentBatch:
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pylist(records).select(self.outputs()[1]),
            _metadata={
                **task._metadata,
                "source_files": list(task.data),
                "warc_chunk_index": chunk_index,
            },
            _stage_perf=task._stage_perf,
        )

    def process(self, task: FileGroupTask) -> list[DocumentBatch]:
        batches = []
        records = []
        for source_url in task.data:
            local_path = self.downloader.download(source_url)
            if local_path is None:
                msg = f"Common Crawl WARC download failed: {source_url}"
                raise RuntimeError(msg)
            try:
                for record in self.iterator.iterate(local_path):
                    content = record.pop("content")
                    record[self.content_field] = self._compress(content)
                    records.append(record)
                    if len(records) == self.records_per_batch:
                        batches.append(self._batch(task, records, len(batches)))
                        records = []
            finally:
                if self.cleanup:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(local_path)

        if records:
            batches.append(self._batch(task, records, len(batches)))
        return batches


class CommonCrawlDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite stage for downloading and processing Common Crawl data.

    This pipeline:
    1. Generates WARC URLs (either from main or news crawls)
    2. Downloads WARC files
    3. Extracts content from WARC files
    4. Extracts text from HTML content
    """

    def __init__(  # noqa: PLR0913
        self,
        start_snapshot: str,
        end_snapshot: str,
        download_dir: str,
        crawl_type: Literal["main", "news"] = "main",
        html_extraction: HTMLExtractorAlgorithm | str | None = None,
        html_extraction_kwargs: dict | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
        use_aws_to_download: bool = False,
        verbose: bool = False,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
        extractor_max_calls_per_worker: int | None = None,
    ):
        self.crawl_type = crawl_type
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot

        if crawl_type == "main":
            self.url_generator = MainCommonCrawlUrlGenerator(
                start_snapshot_str=start_snapshot, end_snapshot_str=end_snapshot, limit=url_limit
            )
        else:
            self.url_generator = NewsCommonCrawlUrlGenerator(
                start_snapshot_str=start_snapshot, end_snapshot_str=end_snapshot, limit=url_limit
            )

        self.downloader = CommonCrawlWARCDownloader(
            download_dir=download_dir, use_aws_to_download=use_aws_to_download, verbose=verbose
        )
        self.iterator = CommonCrawlWarcIterator()
        self.extractor = CommonCrawlHTMLExtractor(
            algorithm=html_extraction,
            algorithm_kwargs=html_extraction_kwargs,
            stop_lists=stop_lists,
        )
        if extractor_max_calls_per_worker is None and isinstance(self.extractor.algorithm, JusTextExtractor):
            extractor_max_calls_per_worker = 2
            logger.info(
                "jusText extraction can cause memory fragmentation and lead to OOM errors. "
                "Setting extractor_max_calls_per_worker=2 for the iterate-extract stage. "
                "Pass extractor_max_calls_per_worker explicitly to override."
            )
        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
            extractor_max_calls_per_worker=extractor_max_calls_per_worker,
        )
        self.name = f"common_crawl_{self.crawl_type}_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return f"Common Crawl {self.crawl_type} pipeline: {self.start_snapshot} to {self.end_snapshot}"
