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

"""Interleaved output: the markdown must come apart into items and lose nothing."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nemo_curator.stages.text.html_extraction.mineru_interleaved import (
    MinerUHtmlInterleavedStage,
    guess_content_type,
    split_items,
)
from nemo_curator.stages.text.html_extraction.mineru_utils import STATUS_FIELD
from nemo_curator.tasks import DocumentBatch

MARKDOWN = (
    "# Title\n\nIntro.\n\n![a chart](https://x/y/chart.png)\n\n"
    'Middle.\n\n<img src="https://x/z/photo.jpeg" alt="p">\n\nTail.'
)


def test_items_alternate_in_document_order():
    assert [modality for modality, _ in split_items(MARKDOWN)] == [
        "text",
        "image",
        "text",
        "image",
        "text",
    ]


def test_both_image_spellings_are_found():
    # mm_md emits markdown images, and passes HTML through where markdown cannot
    # express the figure. Reading only one spelling would drop half the images and
    # silently glue the surrounding paragraphs together.
    assert split_items("![a](p.png)")[0] == ("image", "p.png")
    assert split_items('<img src="q.png">')[0] == ("image", "q.png")


def test_no_prose_is_lost_between_the_images():
    """The text items hold every word that was not part of an image tag.

    Losing prose at an image boundary is the failure that would matter most here and
    the one hardest to see: the document still looks plausible, just shorter.
    """
    import re

    items = split_items(MARKDOWN)
    prose = " ".join(payload for modality, payload in items if modality == "text")
    without_images = re.sub(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>", " ", MARKDOWN)
    assert prose.split() == without_images.split()


def test_every_image_becomes_exactly_one_item():
    srcs = [payload for modality, payload in split_items(MARKDOWN) if modality == "image"]
    assert srcs == ["https://x/y/chart.png", "https://x/z/photo.jpeg"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", []), ("   \n\n  ", []), ("just text", [("text", "just text")])],
)
def test_degenerate_inputs(text: str, expected: list) -> None:
    assert split_items(text) == expected


def test_whitespace_between_images_is_not_an_empty_text_row():
    assert split_items("![a](1.png)\n\n![b](2.png)") == [("image", "1.png"), ("image", "2.png")]


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("https://x/a.png", "image/png"),
        ("https://x/a.JPEG", "image/jpeg"),
        ("https://x/a.webp?v=2", "image/webp"),
        ("https://x/a.svg#frag", "image/svg+xml"),
        # Unknown means unknown. `application/octet-stream` would be a guess wearing
        # the clothes of a fact.
        ("https://x/no-extension", None),
    ],
)
def test_content_type_is_guessed_or_absent(src: str, expected: str | None) -> None:
    assert guess_content_type(src) == expected


def batch(rows: list[dict]) -> DocumentBatch:
    return DocumentBatch(dataset_name="t", data=pd.DataFrame(rows))


def test_rows_carry_the_reserved_schema():
    out = (
        MinerUHtmlInterleavedStage()
        .process(batch([{"url": "https://p/1", "text": MARKDOWN, STATUS_FIELD: "ok"}]))
        .to_pandas()
    )

    assert list(out["modality"]) == ["text", "image", "text", "image", "text"]
    assert list(out["position"]) == [0, 1, 2, 3, 4]
    assert set(out["sample_id"]) == {"https://p/1"}
    # Text rows carry text and no locator; image rows the reverse.
    text_rows = out[out.modality == "text"]
    image_rows = out[out.modality == "image"]
    assert text_rows["source_ref"].isna().all()
    assert image_rows["text_content"].isna().all()
    assert image_rows["binary_content"].isna().all()


def test_image_rows_carry_a_materialisable_locator():
    out = (
        MinerUHtmlInterleavedStage()
        .process(batch([{"url": "https://p/1", "text": "![a](https://x/y/chart.png)"}]))
        .to_pandas()
    )
    ref = json.loads(out.iloc[0]["source_ref"])
    assert ref["path"] == "https://x/y/chart.png"
    assert ref["member"] is None


def test_status_travels_with_the_rows_it_produced():
    out = (
        MinerUHtmlInterleavedStage()
        .process(batch([{"url": "https://p/1", "text": "hi", STATUS_FIELD: "trafilatura_fallback"}]))
        .to_pandas()
    )
    # A document that fell back is not a document the model labelled, and once these
    # are rows among millions there is nothing else left to ask.
    assert set(out["mineru_status"]) == {"trafilatura_fallback"}


def test_sample_id_falls_back_from_id_to_url_to_position():
    stage = MinerUHtmlInterleavedStage(id_field="doc_id")
    with_id = stage.process(batch([{"doc_id": "D1", "url": "https://p/1", "text": "a"}])).to_pandas()
    assert set(with_id["sample_id"]) == {"D1"}

    without = MinerUHtmlInterleavedStage().process(batch([{"text": "a"}, {"text": "b"}])).to_pandas()
    assert list(without["sample_id"]) == ["0", "1"]


def test_a_document_that_extracted_nothing_produces_no_rows():
    out = (
        MinerUHtmlInterleavedStage()
        .process(batch([{"url": "https://p/1", "text": ""}, {"url": "https://p/2", "text": "kept"}]))
        .to_pandas()
    )
    assert list(out["sample_id"]) == ["https://p/2"]
