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

"""Shared vocabulary and CPU helpers for the MinerU-HTML extraction pipeline.

This module is the foundation the stage modules are built on: it holds the
column names and status values they hand to each other, the prompt and answer
helpers, and the pure-CPU post-inference path. Both stage modules import this
one and neither is imported by it, so the edges run
``mineru_html -> mineru_server -> mineru_utils`` with no cycle.

The helpers mirror the semantics of ``mineru_html.process`` but are rewritten for
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

import hashlib
import json
import pathlib
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

import pandas as pd
from loguru import logger
from lxml import html as lxml_html
from lxml.etree import ParserError

DEFAULT_MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"

# Columns the three stages hand to each other. They live here, not in the stage
# modules, so that the stages form a DAG: both import this module and neither
# imports the other.
TOKENS_FIELD = "_mineru_prompt_tokens"
MAP_HTML_FIELD = "_mineru_map_html"
N_ITEMS_FIELD = "_mineru_n_items"
RESPONSE_FIELD = "_mineru_response"
STATUS_FIELD = "_mineru_status"

# Everything except STATUS_FIELD is scratch; the extract stage drops these.
PROMPT_FIELD = "_mineru_prompt"
"""The prompt as text. Written instead of :data:`TOKENS_FIELD` when the model is behind
a hosted chat endpoint, which has no tokenizer we can run and wants messages, not ids."""

CHUNK_IDS_FIELD = "_mineru_chunk_ids"
"""JSON list of the element ids in each window, when a document was chunked. Empty for
the whole-document path, which is what the inference stage branches on."""

COVERAGE_FIELD = "_mineru_coverage"
"""Share of the document's elements the answer actually labelled. Internal, so the output
schema is unchanged; `--keep-internal-fields` surfaces it for an analysis run, and the
coarse signal is always in the status."""

INTERNAL_FIELDS = (TOKENS_FIELD, MAP_HTML_FIELD, N_ITEMS_FIELD, RESPONSE_FIELD, PROMPT_FIELD, COVERAGE_FIELD)

# The values STATUS_FIELD takes. It is the one _mineru_* column that survives into
# the output, so these are public: callers filter on them and benchmarks bucket by
# them. Anything other than "ok" means the row went through the fallback.
Status = Literal[
    "ok",
    "empty_input",  # no HTML in the input cell
    "simplify_error",  # upstream simplify_html raised
    "too_long",  # prompt + answer budget exceeds max_model_len
    "inference_error",  # the request to the server was lost
    "extract_error",  # label parsing or DOM pruning raised
    "convert_error",  # Markdown/JSON conversion raised
]

ITEM_ID_ATTR = "_item_id"
TAIL_BLOCK_TAG = "cc-alg-uc-text"
MAIN_LABEL = "main"

FallbackMode = Literal["trafilatura", "bypass", "empty"]

# The compact form the grammar produces — `1main2other` — plus the punctuation a model
# reaches for when nothing constrains it: `1: main`, `1 = other`, `"1": "main"`, one per
# line. On a genuinely compact string every variant collapses to the original match, so
# the vLLM path is unchanged; the tolerance only ever adds labels a stricter reader
# would have thrown away.
# The trailing lookahead, not `\b`: in the compact form the next character after a label
# is a digit, and `\b` sees no boundary between `main` and `2`, so requiring one silently
# reduced `1main2other3main` to a single label. It must accept a digit, a non-word
# character, or end of string — and reject a letter, so `1mainly` is not read as a label.
_COMPACT_PAIR_RE = re.compile(r"(\d+)\s*[\"']?\s*[:=.\-)]?\s*[\"']?\s*(main|other)(?=\d|\W|$)", re.IGNORECASE)
_ITEM_ID_RE = re.compile(rf'\s{ITEM_ID_ATTR}="(\d+)"')


# Whitespace either side of the answer body, bounded. An unbounded ``\s*`` keeps
# whitespace a legal next token forever, and a small model will take that offer: it
# opens the tag and emits blank lines until the budget runs out, which arrives as an
# empty answer rather than as an error. Seen on 3 of 25 windows of the first chunked
# run. Two characters is enough for the newline the models actually emit.
_PAD = r"\s{0,2}"


@lru_cache(maxsize=4096)
def compact_answer_regex(n_items: int) -> str:
    """Regex constraining the answer to one ``{id}{label}`` pair per element, in order."""
    pattern = "".join(f"{i}(main|other)" for i in range(1, int(n_items) + 1))
    return rf"<answer>{_PAD}{pattern}{_PAD}</answer>"


def compact_response_budget(n_items: int) -> int:
    """Token budget for a compact ``{id}{label}`` answer over ``n_items`` elements.

    Measured on Common Crawl, a compact answer costs ~2.1 tokens per element;
    4 plus a fixed 64-token slack is a comfortable ceiling. Sizing each request
    individually (instead of the reference implementation's flat 16k) is what
    lets a 32k-context engine accept documents longer than 16k tokens at all.
    """
    return max(64, n_items * 4 + 64)


# What upstream's ``truncate_html_element_selective`` writes where it cut, so an
# abridged element is marked the way a text-truncated one already is.
_ELLIPSIS = "..."


def _serialised_len(node: lxml_html.HtmlElement) -> int:
    """Characters this element costs in a prompt — markup, text and tail together."""
    return len(lxml_html.tostring(node, encoding="unicode"))


def _abridge(node: lxml_html.HtmlElement, budget: int) -> None:
    """Shrink `node` in place until it serialises to at most `budget` characters.

    Trailing children go first, and the cut is marked with `...`. The head of a table
    is also the informative end of it — the header row and the first few data rows say
    what the table is, which is the whole question the model is being asked — and it is
    the end upstream keeps when it truncates an element's text.

    The child that straddles the boundary is abridged rather than dropped, so a
    `<table>` whose only child is a `<tbody>` keeps rows instead of collapsing to an
    empty tag. If it cannot be made to fit what is left it goes with the rest, so the
    budget is a bound and not a wish.
    """
    size = _serialised_len(node)
    if size <= budget:
        return

    children = list(node)
    sizes = [_serialised_len(child) for child in children]
    # tostring() concatenates, so whatever is not a child is the start tag, the
    # attributes, `node.text`, the end tag and the tail. One subtraction beats
    # serialising a stripped copy of a 1.5M-character table to find out.
    overhead = size - sum(sizes)

    marked = False
    if node.text and overhead + len(_ELLIPSIS) > budget:
        # The element's own text is what does not fit, so no amount of dropping
        # children helps. Reached only inside a `table` or `math` subtree, which
        # `cutoff_length` exempts from truncation.
        room = max(0, budget - (overhead - len(node.text)) - len(_ELLIPSIS))
        # Unless the start tag alone is over budget, in which case shortening the text
        # buys nothing and would delete a caption to no purpose.
        if len(node.text) > room + len(_ELLIPSIS):
            node.text = node.text[:room] + _ELLIPSIS
            overhead = _serialised_len(node) - sum(sizes)
            marked = True

    used = overhead + len(_ELLIPSIS)
    keep = 0
    for i, child_size in enumerate(sizes):
        if used + child_size <= budget:
            used += child_size
            keep = i + 1
            continue
        _abridge(children[i], budget - used)
        if _serialised_len(children[i]) <= budget - used:
            keep = i + 1
        break

    for child in children[keep:]:
        node.remove(child)
    if keep < len(children) and not marked:
        if keep:
            children[keep - 1].tail = (children[keep - 1].tail or "") + _ELLIPSIS
        else:
            node.text = (node.text or "") + _ELLIPSIS


def abridge_oversized_elements(simplified: str, max_chars: int) -> str:
    """Cap how many characters of any one element the model is shown.

    `cutoff_length` bounds an element's *text* and exempts `table` and `math` outright
    (`no_calc_text_tags` upstream), so nothing bounds an element's *markup* — and on
    this corpus the markup is the payload. The largest element of 1608.00232 is one
    `<table>`: 192,762 serialised characters carrying 9,563 characters of text, and
    byte-identical at cutoff 500 and at cutoff 64. Over the 16 documents of 5,000 that
    a 32k window could not hold, the largest element runs from 62,866 to 1,514,442
    characters; it is a `<table>` in 13 of them, a `<td>` in two — one of inline SVG,
    one of LaTeXML `xmtok` markup — and a `<p>` of MathML in one.

    Windows are cut at element boundaries and never inside an element, so one oversized
    element pins its window at whatever size it is however small the chunk budget —
    which is what `too_long` counts. A cap at or below `chunk_max_chars` is what makes
    that budget hold.

    Only the prompt is abridged. `simplify_html` returns `(simplified, map_html)` and
    the output is rebuilt from `map_html`, so a table the model labels `main` is still
    emitted whole: what shrinks is what it is asked to read, not what it gets.

    Returns `simplified` unchanged when nothing exceeds `max_chars`, so a document that
    did not need this pays no re-serialisation and its prompt is byte-identical. It
    still pays the parse: 6.09 ms per document against `simplify_html`'s 172 ms, over
    200 arXiv pages averaging 169,771 simplified characters.
    """
    # A document shorter than the cap cannot hold an element longer than it, and the
    # comparison is free. It fires on small pages rather than on this corpus, where the
    # mean document is seven times the cap.
    if max_chars <= 0 or len(simplified) <= max_chars:
        return simplified

    root = html_to_element(simplified)
    # An id is one labellable unit and the answer carries one label per id, so an id
    # nested inside another must not be dropped along with the markup around it — the
    # grammar would then demand a label for an element the window never showed. Upstream
    # ids do not nest (0 of the 16 too_long documents, 0 of a 200-document sample), so
    # this is a guard rather than a case, and the walk it guards runs only on the
    # elements already known to be over the cap.
    oversized = [
        node
        for node in root.xpath(f"//*[@{ITEM_ID_ATTR}]")
        if _serialised_len(node) > max_chars and not node.xpath(f".//*[@{ITEM_ID_ATTR}]")
    ]
    if not oversized:
        return simplified

    for node in oversized:
        _abridge(node, max_chars)
    return element_to_html(root)


def _overlap_step(window: Sequence[tuple[str, str]], overlap_chars: int) -> int:
    """How many trailing elements of `window` the next window repeats.

    The fewest whose markup reaches `overlap_chars`, capped at half the window's
    *characters*. A step back as large as the window collapses the advance to one element
    and turns a 30-element document into 27 windows — every element re-sent a dozen times,
    at a dozen times the cost, for no more context than two views would have given. Half
    the characters bounds the repeated text at 2x the unique text however the elements are
    sized; the element count it replaces bounded nothing in characters, and sent 1.5578
    characters for every unique one over 1,000 documents against 1.1594 here.

    Never the whole window: `window[0]` is what the caller advances past, and giving it
    away would stand the walk still. Within that, the first element back is taken before
    the cap applies, so a window whose text is nearly all one large element carries
    something across instead of rounding down to nothing. A window of a *single* element
    has nothing to spare and the window after it opens cold — 306 of the 14,114 seams this
    rule produces, 2.2%, and the same 306 under
    this rule and the element count alike, since repeating an element that alone overflows
    `max_chars` would not fit the window it is meant to open.

    `overlap_chars` of 0 means no overlap and is honoured as written.
    """
    half = sum(len(html) for _, html in window) // 2
    taken = 0
    carried = 0
    for _, html in reversed(window[1:]):
        if carried >= overlap_chars or (taken and carried + len(html) > half):
            break
        carried += len(html)
        taken += 1
    return taken


def chunk_simplified_html(simplified: str, max_chars: int, overlap_chars: int = 2000) -> list[tuple[str, list[str]]]:
    """
    Split a simplified DOM into overlapping windows that fit a small context.

    The whole corpus averages ~46,000 prompt tokens, so a 32k-context model — including
    the 0.5B checkpoint this pipeline was built around — cannot see most documents at
    all. Scoring it anyway would produce a ranking that looks like extraction quality
    and is really context length.

    Windows are cut at element boundaries and never inside one: a half-element is
    unlabellable, and its id would go missing from the answer. Each seam repeats the
    fewest trailing elements that carry `overlap_chars` characters, so an element near a
    boundary is judged at least once with text on both sides of it — an element that
    opens a window has no preceding context and its label is the least trustworthy thing
    in the answer.

    The budget is characters, not a count of elements, because a count of elements does
    not bound text: element size decides, and a window holds few elements precisely when
    they are large. Measured over 1,000 documents, 8 elements meant anything from 0 to
    tens of thousands of characters, median 7,930 — far more context than a boundary
    element needs, and paid for in prefill, which is 104:1 of output on this corpus. Two
    thousand characters instead sends 1.1594 characters per unique one where 8 elements
    sent 1.5578, in 25% fewer windows, and holds the tenth percentile at 2,032.

    Returns `[(chunk_html, [item_id, ...]), ...]`, ids as strings, in document order.
    """
    if not simplified.strip():
        return []

    root = html_to_element(simplified)
    nodes = root.xpath(f"//*[@{ITEM_ID_ATTR}]")
    if not nodes:
        return []

    pieces = [(str(node.get(ITEM_ID_ATTR)), lxml_html.tostring(node, encoding="unicode")) for node in nodes]

    chunks: list[tuple[str, list[str]]] = []
    start = 0
    while start < len(pieces):
        size = 0
        end = start
        while end < len(pieces) and (size == 0 or size + len(pieces[end][1]) <= max_chars):
            size += len(pieces[end][1])
            end += 1
        window = pieces[start:end]
        chunks.append(("\n".join(html for _, html in window), [item for item, _ in window]))
        if end >= len(pieces):
            break
        # `_overlap_step` never returns the whole window, so `start + 1` is slack rather
        # than an active clamp. It stays because the loop's termination rests on it: an
        # element larger than `max_chars` pins the window at one element, and a step rule
        # that stopped honouring the cap would hang the walk rather than mis-size a chunk.
        start = max(end - _overlap_step(window, overlap_chars), start + 1)
    return chunks


def compact_answer_regex_for(item_ids: Sequence[str]) -> str:
    """Constrain an answer to exactly these ids, in this order.

    :func:`compact_answer_regex` assumes the ids are 1..n, which stops being true the
    moment a document is chunked — window three might hold ids 240 to 310. Without this
    the grammar would demand ids the window never showed.
    """
    pattern = "".join(f"{item}(main|other)" for item in item_ids)
    return rf"<answer>{_PAD}{pattern}{_PAD}</answer>"


def merge_chunk_labels(
    chunk_labels: Sequence[dict[str, str]], chunk_ids: Sequence[Sequence[str]]
) -> tuple[dict[str, str], int]:
    """
    Fold each window's labels into one, and count where the windows disagreed.

    An element in an overlap is labelled twice. The winner is the window that saw it
    with the most text around it — measured as distance to the nearest edge of that
    window — because that is the whole reason the overlap exists. Recency or first-wins
    would both discard the better-informed of the two views half the time.

    The disagreement count is returned rather than hidden: it is the cheapest available
    measure of how stable the labelling is, and a corpus where the two views of an
    element differ often is one where a single view should not be trusted either.
    """
    best: dict[str, tuple[int, str]] = {}
    disagreements = 0
    for labels, ids in zip(chunk_labels, chunk_ids, strict=True):
        for position, item in enumerate(ids):
            label = labels.get(str(item))
            if label is None:
                continue
            centrality = min(position, len(ids) - 1 - position)
            previous = best.get(str(item))
            if previous is not None and previous[1] != label:
                disagreements += 1
            if previous is None or centrality > previous[0]:
                best[str(item)] = (centrality, label)
    return {item: label for item, (_, label) in best.items()}, disagreements


def chat_response_budget(n_items: int) -> int:
    """Token ceiling for a compact answer from a model whose tokenizer is not MinerU's.

    :func:`compact_response_budget` allows 4 tokens per element, from a measured ~2.1 on
    the MinerU checkpoint. A hosted model tokenizes the same string differently, and
    measured against claude-opus-5 the real cost is 3.5 to 6.6 tokens per element — the
    ratio is worst on short documents, where the fixed `<answer>` overhead dominates.

    Two documents of the first five hit the ceiling and stopped mid-answer. One of them
    needed 843 tokens and was given 828: a fifteen-token shortfall silently deleted
    everything after the 96th of 191 elements, and the document still reported `ok`.

    So this is deliberately loose — 12 per element plus 1024. `max_tokens` is a ceiling,
    not a reservation: an answer that needs 843 tokens costs 843 whatever the cap says,
    so the only thing a tight bound buys is truncation.
    """
    return max(1024, int(n_items) * 12 + 1024)


def decode_html_cell(cell: str | bytes | None) -> str:
    """Return a raw HTML cell as ``str``; missing or undecodable values become ``""``.

    Missing covers ``None``, ``NaN`` and ``pd.NA`` -- a reader using
    ``dtype_backend="numpy_nullable"`` can produce any of the three.

    Bytes are tried as UTF-8 and then fall back to charset detection, matching
    :func:`nemo_curator.stages.text.download.utils.decode_html`. A plain
    ``decode("utf-8", errors="replace")`` turns every non-UTF-8 page into
    replacement characters, which the model then labels as garbage while the row
    keeps ``status == "ok"`` -- silent quality loss on the non-English slice that
    no throughput or extraction-rate metric reveals.

    This calls ``charset_normalizer`` directly rather than importing
    ``download.utils``: that module lives in a package whose ``__init__`` pulls in
    the whole download subpackage, which costs ~1.2s per worker and, worse, needs
    ``pycld2`` -- declared in the ``text_cpu`` extra, not ``mineru_html``. Reaching
    for it made ``pip install nemo_curator[mineru_html]`` raise ModuleNotFoundError
    on the first bytes-valued page.
    """
    if isinstance(cell, (bytes, bytearray)):
        raw = bytes(cell)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            from charset_normalizer import detect

            encoding = detect(raw)["encoding"]
            if not encoding or encoding == "utf-8":
                return ""
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                return ""
    # Not `cell or ""`: bool(pd.NA) raises instead of returning False.
    return "" if cell is None or pd.isna(cell) else cell


class FallbackExtractor:
    """Recover a document's content when the model could not label it.

    Mirrors the three handlers in ``mineru_html.process.map_to_main`` without
    importing them: importing any ``mineru_html`` submodule executes its package
    ``__init__``, which pulls in the transformers and vLLM inference backends --
    seconds of startup and hundreds of MB per worker, for code a CPU-only stage
    never runs. ``trafilatura`` is already a Curator dependency; ``mineru_html``
    is not.

    Build this in ``setup()``; the trafilatura import happens in ``__init__``.
    """

    def __init__(self, mode: FallbackMode = "trafilatura"):
        self.mode = mode
        if mode == "trafilatura":
            from trafilatura import extract
            from trafilatura.settings import Extractor

            self._extract = extract
            self._options = Extractor(output_format="html", comments=False)

    def __call__(self, cell: str | bytes | None) -> str:
        if self.mode == "empty" or cell is None:
            return ""
        html_str = decode_html_cell(cell)
        if self.mode == "bypass":
            return html_str
        try:
            result = self._extract(html_str, options=self._options)
        except Exception as e:  # noqa: BLE001 - third-party parsers raise broadly
            logger.debug(f"fallback extraction failed: {e}")
            return ""
        return result if result is not None else ""


def load_prompt_template(path: str) -> tuple[str, str]:
    """A prompt template and the sha256 of its bytes.

    The hash is not decoration. A prompt is the variable under study here, and its
    filename is a name someone can reuse — two runs claiming the same prompt while the
    file changed underneath them is exactly the kind of difference that shows up as a
    mysterious quality regression. Recording both makes it detectable.

    The file is a plain template with one placeholder, ``{simplified_html}``; a literal
    brace must be doubled.
    """
    raw = pathlib.Path(path).read_bytes()
    text = raw.decode("utf-8")
    if "{simplified_html}" not in text:
        msg = f"prompt template {path} must contain the placeholder {{simplified_html}}"
        raise ValueError(msg)
    return text, hashlib.sha256(raw).hexdigest()


def count_item_ids(text: str) -> int:
    """Number of ``_item_id`` attributes in a simplified-HTML string."""
    return len(_ITEM_ID_RE.findall(text))


def parse_compact_response(response: str) -> dict[str, str]:
    """Parse a compact ``1main2other3other`` response into ``{item_id: label}``.

    The model wraps its answer in ``<answer>...</answer>``; the regex ignores the
    wrapper, and does not require the closing tag — a truncated answer is still a
    hundred good labels, and refusing to read them throws away work already paid for.
    Later duplicates of an id win, matching upstream's dict comprehension.

    Salvage is the point. Without constrained decoding an answer arrives complete,
    truncated, punctuated, or not at all, and only the last of those is worth nothing.
    What must not happen is a partial answer being mistaken for a complete one — that is
    what :func:`label_coverage` is for, and why the extract stage records it.
    """
    if not response:
        return {}
    # A model told to emit JSON usually will. Tried first because `{"12": "main"}` also
    # matches the regex below, and agreeing with itself is not the same as being right.
    text = response.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(body, dict):
                return {
                    str(k): str(v).lower()
                    for k, v in body.items()
                    if str(k).isdigit() and str(v).lower() in (MAIN_LABEL, "other")
                }

    return {m.group(1): m.group(2).lower() for m in _COMPACT_PAIR_RE.finditer(response)}


def label_coverage(labels: dict[str, str], n_items: int) -> dict[str, float | int]:
    """How much of the document the answer actually spoke to.

    A label set is only a partition of the document if it covers it. 96 labels for 191
    elements prunes everything after the 96th and produces a document that reads as
    complete — the failure that is hardest to see and easiest to act on, because nothing
    about the output says it happened.

    `unknown` counts ids the answer named that the document does not contain, which is
    the clearest sign an unconstrained model is inventing rather than reading.
    """
    if n_items <= 0:
        return {"labelled": 0, "expected": 0, "coverage": 1.0, "unknown": 0}
    in_range = {k for k in labels if k.isdigit() and 1 <= int(k) <= n_items}
    return {
        "labelled": len(in_range),
        "expected": int(n_items),
        "coverage": len(in_range) / n_items,
        "unknown": len(labels) - len(in_range),
    }


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
