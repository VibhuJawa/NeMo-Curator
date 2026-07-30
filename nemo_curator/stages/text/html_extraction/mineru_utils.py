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

"""CPU helpers for the MinerU-HTML extraction pipeline.

These mirror the semantics of ``mineru_html.process`` but are rewritten for
throughput on large web crawls:

* :func:`extract_main_html` resolves ``_item_id`` -> element with a single
  document walk instead of one ``//*[@_item_id="N"]`` XPath scan per label,
  and prunes the tree iteratively instead of recursively.
* :func:`parse_compact_response` skips the JSON-brace probing that the
  upstream parser attempts before falling back to the compact regex, which is
  the only format the ``*-compact`` checkpoints emit.

``mineru_html`` remains the source of truth for HTML simplification; only the
post-inference path is reimplemented here.
"""

from __future__ import annotations

import re

from lxml import html as lxml_html
from lxml.etree import ParserError

ITEM_ID_ATTR = "_item_id"
TAIL_BLOCK_TAG = "cc-alg-uc-text"
MAIN_LABEL = "main"

_COMPACT_PAIR_RE = re.compile(r"(\d+)(main|other)")
_ITEM_ID_RE = re.compile(rf'\s{ITEM_ID_ATTR}="(\d+)"')


def count_item_ids(text: str) -> int:
    """Number of ``_item_id`` attributes in a simplified-HTML string."""
    return len(_ITEM_ID_RE.findall(text))


def parse_compact_response(response: str) -> dict[str, str]:
    """Parse a compact ``1main2other3other`` response into ``{item_id: label}``.

    The model wraps its answer in ``<answer>...</answer>``; the regex ignores
    the wrapper. Later duplicates of an id win, matching upstream's dict
    comprehension over ``re.finditer``.
    """
    return {m.group(1): m.group(2) for m in _COMPACT_PAIR_RE.finditer(response)}


def html_to_element(html_str: str) -> lxml_html.HtmlElement:
    """Parse an HTML string with the same parser options as upstream."""
    parser = lxml_html.HTMLParser(
        collect_ids=False,
        encoding="utf-8",
        remove_blank_text=True,
        remove_comments=True,
        remove_pis=True,
    )
    # lxml ignores the parser encoding for ``str`` input, so documents that
    # declare their own encoding must be handed over as bytes.
    html_input = (
        html_str.encode("utf-8")
        if isinstance(html_str, str)
        and ("<?xml" in html_str or "<meta charset" in html_str or "encoding=" in html_str)
        else html_str
    )
    try:
        return lxml_html.fromstring(html_input, parser=parser)
    except ParserError as e:
        if "Document is empty" in str(e):
            return lxml_html.HtmlElement()
        raise


def element_to_html(root: lxml_html.HtmlElement) -> str:
    # tostring() returns str only for encoding=str/"unicode"; a codec name gives bytes.
    return lxml_html.tostring(root, pretty_print=False, encoding="utf-8").decode("utf-8")


_URL_ATTR_RE = re.compile(r'(href="|src=")(.*?)(")', flags=re.IGNORECASE | re.DOTALL)
_HTTP_PREFIXES = ("http://", "https://", "ftp://", "HTTP://", "HTTPS://", "FTP://", "//")


def decode_http_urls_only(html_str: str) -> str:
    """Unescape HTML entities inside ``href``/``src`` values that are URLs."""
    from html import unescape

    def _decode(match: re.Match[str]) -> str:
        url = match.group(2)
        if url.startswith(_HTTP_PREFIXES):
            return f"{match.group(1)}{unescape(url)}{match.group(3)}"
        return match.group(0)

    return _URL_ATTR_RE.sub(_decode, html_str)


def _build_item_id_index(root: lxml_html.HtmlElement) -> dict[str, lxml_html.HtmlElement]:
    """Map ``_item_id`` -> first element carrying it, in document order."""
    index: dict[str, lxml_html.HtmlElement] = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            # Comments / processing instructions carry no attributes.
            continue
        item_id = el.get(ITEM_ID_ATTR)
        if item_id is not None and item_id not in index:
            index[item_id] = el
    return index


def _prune_to_kept(root: lxml_html.HtmlElement, kept: set) -> None:
    """Drop every subtree whose root is not in ``kept`` (iterative)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node not in kept:
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
                continue
        # Materialise children before mutating, so removals can't cut the walk short.
        stack.extend(node.iterchildren())


def extract_main_html(map_html: str, item_label: dict[str, str]) -> str:
    """Keep only the elements the model labelled ``main`` (plus their context).

    Args:
        map_html: Original HTML annotated with ``_item_id`` attributes.
        item_label: ``{item_id: "main"|"other"}`` from the model.

    Returns:
        HTML string containing the main content.
    """
    root = html_to_element(map_html)
    index = _build_item_id_index(root)

    kept: set = set()
    for item_id, label in item_label.items():
        if label != MAIN_LABEL:
            continue
        elem = index.get(item_id)
        if elem is None:
            continue
        kept.update(elem.iter())
        kept.update(elem.iterancestors())

    # Recall <br> tags directly adjacent to kept content so line breaks survive.
    # Both arms of the loop need a <br>, so a C-level existence check (~5 us) skips
    # a second full Python-level walk (~180 us) for every document that has none.
    if root.find(".//br") is not None:
        previous = None
        for element in root.iter():
            if previous is not None:
                if element.tag == "br" and previous.tag != "br" and previous in kept:
                    kept.add(element)
                if previous.tag == "br" and element.tag != "br" and element in kept:
                    kept.add(previous)
            previous = element

    _prune_to_kept(root, kept)

    # Materialise before dropping: drop_tag() splices children into the parent.
    for tail_block in list(root.iter(TAIL_BLOCK_TAG)):
        tail_block.drop_tag()

    return decode_http_urls_only(element_to_html(root))
