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

"""HTML extraction stages for the CC-LanceDB pipeline.

One parameterized actor stage, three instances — one per extractor library:

    HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura", name="trafilatura_extract")
    HtmlExtractStage(JusTextExtractor,     "cc_extracted_text_justext",     name="justext_extract")
    HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse", name="resiliparse_extract")

Each stage is an independent Ray actor pool.  Placing them sequentially in a
RayDataExecutor pipeline gives full pipeline parallelism: while block N is in
JusText, block N+1 is in Trafilatura and block N+2 is still being fetched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd  # noqa: TC002 — used at runtime for pd.DataFrame(...)
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.html_extractors.utils import get_stop_list_dict
from nemo_curator.stages.text.download.utils import decode_html, lang_detect
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.text.download.base.extract import DocumentExtractor


@dataclass
class HtmlExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Actor stage: run one HTML extractor over every document in a batch.

    Pass a zero-argument factory and the target output column name:

        HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura")
        HtmlExtractStage(JusTextExtractor,     "cc_extracted_text_justext")
        HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse")
        # Or with constructor args:
        HtmlExtractStage(lambda: JusTextExtractor(language="ENGLISH"), "cc_extracted_text_justext")

    Overriding :meth:`setup` makes this an actor stage — Ray spawns one OS
    process per available CPU, bypassing the GIL for true parallelism across
    actors while each actor runs its extractor serially per row.

    Attributes:
        extractor_factory: Zero-argument callable that returns a DocumentExtractor.
            Instantiation is deferred to :meth:`setup` so it runs inside the
            worker process, not the driver — avoids pickling a live extractor.
        output_column: Name of the DataFrame column to write extracted text to.
    """

    extractor_factory: Callable[[], DocumentExtractor]
    output_column: str
    # The DataFrame column containing HTML bytes. Use "content" for upstream
    # CommonCrawlWarcIterator output; "cc_html_bytes" for legacy byte-range fetch.
    input_column: str = "content"
    name: str = "html_extract"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    # Per-actor state — populated once in setup()
    _extractor: DocumentExtractor | None = field(init=False, repr=False, default=None)
    _stop_lists: dict | None = field(init=False, repr=False, default=None)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._extractor = self.extractor_factory()
        self._stop_lists = get_stop_list_dict()
        worker_id = worker_metadata.worker_id if worker_metadata is not None else "unknown"
        logger.info(f"HtmlExtractStage({type(self._extractor).__name__}) ready on {worker_id}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["cc_html_bytes"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_column]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df: pd.DataFrame = task.to_pandas()
        results: list[str] = []

        for html_bytes in df["cc_html_bytes"].tolist():
            html = decode_html(html_bytes) if html_bytes else None
            if html is None:
                results.append("")
                continue
            lang = lang_detect(html)
            stop_words = self._stop_lists.get(lang) or self._stop_lists.get("en", frozenset())
            try:
                texts = self._extractor.extract_text(html, stop_words, lang)
                results.append("\n\n".join(texts) if texts else "")
            except Exception:  # noqa: BLE001
                results.append("")

        df[self.output_column] = results
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
