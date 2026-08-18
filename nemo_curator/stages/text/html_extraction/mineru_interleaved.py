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

"""Interleaved output for MinerU-HTML: one row per content item, not one blob per page.

The extract stage renders main content as `mm_md` — markdown *including maths and
images*, per the tutorial — which is one string per document. That is the right thing to
read and the wrong thing to train on: it says nothing about where an image sits, what it
points at, or how to fetch it. This stage turns that same string into
:class:`InterleavedBatch` rows, alternating `text` and `image` in document order.

It splits the pipeline's own output rather than re-walking the DOM, deliberately. The
text rows are then byte-identical to what the pipeline already produced — no second
conversion, no second opinion about what markdown a `<table>` deserves — and the only
thing this module decides is where one item ends and the next begins.

Image rows carry a `source_ref` locator and no bytes. `binary_content` is documented as
"populated by materialization", and the converter has already absolutised every `src`
against the page URL, so `{"path": "https://…"}` is a locator the existing helpers can
fetch. Fetching here would put an HTTP client in a CPU stage that has no business owning
one, and would refetch on every re-run of a pipeline whose whole point is that inference
is the expensive part.
"""

from __future__ import annotations

import re

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.html_extraction.mineru_utils import STATUS_FIELD
from nemo_curator.tasks import DocumentBatch, InterleavedBatch

# `![alt](src "title")` and a bare `<img src=...>`, because `mm_md` emits the first and
# passes the second through when a figure carries markup markdown cannot express.
_MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*?\bsrc=[\"'](?P<src2>[^\"']+)[\"'][^>]*>", re.IGNORECASE)
_IMAGE_RE = re.compile(f"{_MD_IMAGE_RE.pattern}|{_HTML_IMAGE_RE.pattern}")

_CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "bmp": "image/bmp",
    "avif": "image/avif",
}

TEXT_MODALITY = "text"
IMAGE_MODALITY = "image"


def guess_content_type(src: str) -> str | None:
    """From the extension, or `None` — which is honest, and not `application/octet-stream`."""
    tail = src.split("?", 1)[0].split("#", 1)[0].rsplit(".", 1)
    return _CONTENT_TYPES.get(tail[-1].lower()) if len(tail) == 2 else None  # noqa: PLR2004


def split_items(markdown: str) -> list[tuple[str, str]]:
    """
    `[(modality, payload), …]` in document order, tiling the markdown completely.

    Every character of the input lands in exactly one item: the text between images, and
    each image's src. Whitespace-only gaps are dropped rather than emitted as empty text
    rows, which is the one thing removed — an interleaved record whose every other row
    is `"\\n\\n"` is harder to read and no more faithful.
    """
    items: list[tuple[str, str]] = []
    at = 0
    for match in _IMAGE_RE.finditer(markdown):
        before = markdown[at : match.start()]
        if before.strip():
            items.append((TEXT_MODALITY, before.strip()))
        src = match.group("src") or match.group("src2") or ""
        if src:
            items.append((IMAGE_MODALITY, src))
        at = match.end()
    tail = markdown[at:]
    if tail.strip():
        items.append((TEXT_MODALITY, tail.strip()))
    return items


class MinerUHtmlInterleavedStage(ProcessingStage[DocumentBatch, InterleavedBatch]):
    """Extracted markdown -> row-wise interleaved records."""

    def __init__(
        self,
        text_field: str = "text",
        id_field: str | None = None,
        url_field: str | None = "url",
        keep_fields: tuple[str, ...] = (),
        cpus: float = 1.0,
    ):
        """
        Args:
            text_field: Column holding the extracted markdown.
            id_field: Column to use as `sample_id`. When absent the URL is used, and
                when that is absent too the row's position in the batch — stated here
                because a `sample_id` that silently falls back to an ordinal is one
                that stops being stable the moment the input is repartitioned.
            url_field: Column holding the page URL, carried through as a user column.
            keep_fields: Extra input columns to carry onto every row of a document.
            cpus: This is string splitting; one core is ample.
        """
        self.text_field = text_field
        self.id_field = id_field
        self.url_field = url_field
        self.keep_fields = keep_fields

        self.resources = Resources(cpus=cpus)
        self.name = "mineru_html_interleaved"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sample_id", "position", "modality", "content_type", "text_content", "source_ref"]

    def _sample_id(self, row: pd.Series, index: int) -> str:
        for field in (self.id_field, self.url_field):
            if field and field in row.index and row[field] not in (None, ""):
                return str(row[field])
        return str(index)

    def process(self, batch: DocumentBatch) -> InterleavedBatch:
        df = batch.to_pandas()
        rows: list[dict[str, object]] = []

        for index, (_, row) in enumerate(df.iterrows()):
            sample_id = self._sample_id(row, index)
            carried = {name: row[name] for name in self.keep_fields if name in row.index}
            if self.url_field and self.url_field in row.index:
                carried.setdefault("url", row[self.url_field])
            if STATUS_FIELD in row.index:
                # The extract stage's verdict travels with the rows it produced: a
                # document that fell back is not a document the model labelled, and
                # once these are rows among millions there is nothing else to ask.
                carried.setdefault("mineru_status", row[STATUS_FIELD])

            for position, (modality, payload) in enumerate(split_items(str(row[self.text_field] or ""))):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "position": position,
                        "modality": modality,
                        "content_type": "text/markdown" if modality == TEXT_MODALITY else guess_content_type(payload),
                        "text_content": payload if modality == TEXT_MODALITY else None,
                        "binary_content": None,
                        "source_ref": (
                            None
                            if modality == TEXT_MODALITY
                            else InterleavedBatch.build_source_ref(path=payload, member=None)
                        ),
                        "materialize_error": None,
                        **carried,
                    }
                )

        return InterleavedBatch(
            dataset_name=batch.dataset_name,
            data=pd.DataFrame(rows, columns=_columns(rows)),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def _columns(rows: list[dict[str, object]]) -> list[str]:
    """Reserved columns first, then whatever was carried — stable across empty batches."""
    reserved = [
        "sample_id",
        "position",
        "modality",
        "content_type",
        "text_content",
        "binary_content",
        "source_ref",
        "materialize_error",
    ]
    extra = sorted({name for row in rows for name in row} - set(reserved))
    return reserved + extra


__all__ = ["MinerUHtmlInterleavedStage", "guess_content_type", "split_items"]
