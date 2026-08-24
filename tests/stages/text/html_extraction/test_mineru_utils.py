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

import importlib.util

import pytest

from nemo_curator.stages.text.html_extraction.mineru_utils import (
    count_item_ids,
    extract_main_html,
    extract_other_html,
    parse_compact_response,
)

MAP_HTML = (
    "<html><body>"
    '<nav _item_id="1">Home | About</nav>'
    '<article _item_id="2"><h1 _item_id="3">Title</h1>'
    '<p _item_id="4">First paragraph.</p>'
    '<p _item_id="5">Second paragraph.</p></article>'
    '<footer _item_id="6">(c) 2026</footer>'
    "</body></html>"
)


class TestParseCompactResponse:
    def test_parses_answer_block(self) -> None:
        assert parse_compact_response("<answer>\n1other2main3main\n</answer>") == {
            "1": "other",
            "2": "main",
            "3": "main",
        }

    def test_ignores_surrounding_noise(self) -> None:
        assert parse_compact_response("Sure! 1main 2other done") == {"1": "main", "2": "other"}

    def test_empty_response(self) -> None:
        assert parse_compact_response("") == {}

    def test_later_duplicate_wins(self) -> None:
        assert parse_compact_response("1main1other") == {"1": "other"}

    def test_multi_digit_ids(self) -> None:
        assert parse_compact_response("10main200other")["200"] == "other"


class TestCountItemIds:
    def test_counts_attributes(self) -> None:
        assert count_item_ids(MAP_HTML) == 6

    def test_no_ids(self) -> None:
        assert count_item_ids("<p>hello</p>") == 0


class TestExtractMainHtml:
    def test_keeps_only_main_subtrees(self) -> None:
        labels = {"1": "other", "2": "main", "3": "main", "4": "main", "5": "main", "6": "other"}
        out = extract_main_html(MAP_HTML, labels)
        assert "First paragraph." in out
        assert "Second paragraph." in out
        assert "Home | About" not in out
        assert "(c) 2026" not in out

    def test_ancestors_of_main_are_retained(self) -> None:
        # Only a leaf is labelled main; its <article> ancestor must survive so
        # the fragment stays well formed.
        out = extract_main_html(MAP_HTML, {"4": "main"})
        assert "<article" in out
        assert "First paragraph." in out
        assert "Second paragraph." not in out

    def test_all_other_yields_no_content(self) -> None:
        labels = dict.fromkeys(map(str, range(1, 7)), "other")
        out = extract_main_html(MAP_HTML, labels)
        assert "First paragraph." not in out
        assert "Home | About" not in out

    def test_unknown_ids_are_ignored(self) -> None:
        out = extract_main_html(MAP_HTML, {"99": "main", "4": "main"})
        assert "First paragraph." in out

    def test_empty_labels(self) -> None:
        assert "First paragraph." not in extract_main_html(MAP_HTML, {})

    def test_br_adjacent_to_main_is_recalled(self) -> None:
        html = '<html><body><div _item_id="1">a<br/>b</div></body></html>'
        assert "<br" in extract_main_html(html, {"1": "main"})

    def test_tail_block_tag_is_unwrapped(self) -> None:
        html = '<html><body><div><cc-alg-uc-text _item_id="1">text</cc-alg-uc-text></div></body></html>'
        out = extract_main_html(html, {"1": "main"})
        assert "cc-alg-uc-text" not in out
        assert "text" in out

    def test_empty_document(self) -> None:
        assert extract_main_html("", {"1": "main"}) is not None

    def test_other_label_projection_is_complementary(self) -> None:
        labels = {"1": "other", "2": "main", "3": "main", "4": "main", "5": "main", "6": "other"}
        out = extract_other_html(MAP_HTML, labels)
        assert "Home | About" in out
        assert "(c) 2026" in out
        assert "First paragraph." not in out


# find_spec, not importorskip: importorskip raises Skipped when it fails, and a
# decorator is evaluated at import time, so using it here skipped the whole module --
# including the tests above, which need only lxml.
@pytest.mark.skipif(
    importlib.util.find_spec("mineru_html") is None,
    reason="mineru_html not installed",
)
class TestUpstreamParity:
    """Curator's rewritten extractor must match mineru_html byte for byte."""

    @pytest.mark.parametrize("mask", range(1 << 6))
    def test_matches_upstream_for_every_label_assignment(self, mask: int) -> None:
        from mineru_html.process.map_to_main import extract_main_html as upstream_extract
        from mineru_html.process.parse_result import parse_llm_response as upstream_parse

        n = count_item_ids(MAP_HTML)
        assert n == 6, "mask enumeration assumes six labelled elements"
        response = (
            "<answer>"
            + "".join(f"{i}{'main' if mask >> (i - 1) & 1 else 'other'}" for i in range(1, n + 1))
            + "</answer>"
        )

        assert extract_main_html(MAP_HTML, parse_compact_response(response)) == upstream_extract(
            MAP_HTML, upstream_parse(response)
        )
