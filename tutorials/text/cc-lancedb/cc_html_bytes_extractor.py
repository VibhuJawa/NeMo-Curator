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

from typing import Any

from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor


class CCHTMLBytesExtractor(CommonCrawlHTMLExtractor):
    # Subclass of CommonCrawlHTMLExtractor that preserves raw HTML bytes alongside
    # extracted text and renames fields to LanceDB column names.

    def __init__(
        self,
        extractor_lib: str = "trafilatura",
        algorithm_kwargs: dict[str, Any] | None = None,
        stop_lists: Any | None = None,
    ) -> None:
        self._extractor_lib = extractor_lib
        super().__init__(
            algorithm=extractor_lib,
            algorithm_kwargs=algorithm_kwargs,
            stop_lists=stop_lists,
        )

    def extract(self, record: dict) -> dict | None:
        # Capture raw HTML bytes before calling super(), which drops "content".
        raw_html = record.get("content", b"")
        result = super().extract(record)
        if result is None:
            return None
        return {
            "cc_snapshot_id": result["source_id"],
            "cc_url": result["url"],
            "cc_html_bytes": raw_html,
            "cc_extracted_text": result["text"],
            "warc_id": result["warc_id"],
            "language": result.get("language", ""),
            "extractor_lib": self._extractor_lib,
        }

    def input_columns(self) -> list[str]:
        return ["url", "warc_id", "source_id", "content"]

    def output_columns(self) -> list[str]:
        return [
            "cc_snapshot_id",
            "cc_url",
            "cc_html_bytes",
            "cc_extracted_text",
            "warc_id",
            "language",
            "extractor_lib",
        ]
