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

"""Chunking: a small-context model must still see every element, exactly once at the end."""

from __future__ import annotations

import itertools

import pytest

from nemo_curator.stages.text.html_extraction.mineru_utils import (
    chunk_simplified_html,
    compact_answer_regex_for,
    merge_chunk_labels,
)


def table(rows: int, item_id: str = "2", wrap_in_tbody: bool = False) -> str:
    """A LaTeXML-shaped table: the class soup is the payload, not the text.

    The largest element of 1608.00232 is one of these -- 192,762 serialised characters
    carrying 9,563 of text, across 1,740 `<td>` each wearing
    `class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r"`.
    """
    body = "".join(
        f'<tr><td class="ltx_td ltx_align_right ltx_border_r">cell {i}</td>'
        f'<td class="ltx_td ltx_align_left">{i}</td></tr>'
        for i in range(rows)
    )
    inner = f"<tbody>{body}</tbody>" if wrap_in_tbody else body
    return f'<table _item_id="{item_id}">{inner}</table>'


def page(*elements: str) -> str:
    return '<html><head><meta charset="utf-8"></head><body>' + "".join(elements) + "</body></html>"


def document(n: int) -> str:
    body = "".join(f'<p _item_id="{i}">Paragraph {i} with a handful of words in it.</p>' for i in range(1, n + 1))
    return f"<div>{body}</div>"


def lumpy_document(n: int) -> str:
    """Paragraphs with a table-sized element every fifth: the shape the two units differ on.

    A window holds few elements exactly when its elements are large, so the same 8
    elements are a paragraph of context here and a page of it there.
    """
    body = "".join(
        f'<p _item_id="{i}">{"table cell " * 400 if i % 5 == 0 else f"Paragraph {i} with a few words in it."}</p>'
        for i in range(1, n + 1)
    )
    return f"<div>{body}</div>"


def carried_chars(before: tuple[str, list[str]], after: tuple[str, list[str]]) -> int:
    """Characters of `before` that `after` opens by repeating.

    The windows join elements with a newline and no test element contains one, so a
    window's lines are its elements.
    """
    shared = set(before[1]) & set(after[1])
    return sum(len(line) for line, item in zip(after[0].split("\n"), after[1], strict=True) if item in shared)


def test_every_element_appears_in_some_window():
    # The whole point. An element no window contains is an element no model ever sees,
    # and it would be silently dropped from the document with nothing to say so.
    chunks = chunk_simplified_html(document(120), max_chars=1500, overlap_chars=500)
    covered = {i for _, ids in chunks for i in ids}
    assert covered == {str(i) for i in range(1, 121)}


def test_windows_are_in_document_order_and_contiguous():
    chunks = chunk_simplified_html(document(120), max_chars=1500, overlap_chars=500)
    for _, ids in chunks:
        numbers = [int(i) for i in ids]
        assert numbers == sorted(numbers)
        assert numbers == list(range(numbers[0], numbers[-1] + 1))


def test_a_document_that_fits_is_one_window_with_no_duplicates():
    # Chunking has to be a no-op when it is not needed, or every run pays for overlap
    # it did not require.
    chunks = chunk_simplified_html(document(30), max_chars=10**9, overlap_chars=500)
    assert len(chunks) == 1
    assert chunks[0][1] == [str(i) for i in range(1, 31)]


def test_overlap_repeats_elements_at_the_seams():
    chunks = chunk_simplified_html(document(120), max_chars=1500, overlap_chars=500)
    total = sum(len(ids) for _, ids in chunks)
    assert total > 120  # something is repeated
    for before, after in itertools.pairwise(chunks):
        assert set(before[1]) & set(after[1])  # consecutive windows share elements


def test_a_window_opens_on_enough_text_to_judge_its_first_element_by():
    # The property the budget buys and an element count could not: 8 elements meant
    # anything from 0 to tens of thousands of characters, median 7,930 over 1,000
    # documents, because element size decides. An element that opens a window with too
    # little before it is the least trustworthy label in the answer.
    chunks = chunk_simplified_html(document(400), max_chars=6000, overlap_chars=1000)
    assert len(chunks) > 2
    for before, after in itertools.pairwise(chunks):
        assert carried_chars(before, after) >= 1000


def test_a_window_of_large_elements_carries_text_across_without_re_sending_the_table():
    # Large elements are where the two units diverge: an element count re-sent whatever
    # those elements happened to weigh, in either direction. Every window here still opens
    # on text it has seen, and the character cap is what keeps that text from being the
    # whole table again — prefill is 104:1 of output on this corpus, so repeated
    # characters are the bill.
    chunks = chunk_simplified_html(lumpy_document(60), max_chars=6000, overlap_chars=1000)
    assert len(chunks) > 2
    for before, after in itertools.pairwise(chunks):
        assert carried_chars(before, after) > 0
    whole = chunk_simplified_html(lumpy_document(60), max_chars=10**9)[0][0]
    assert sum(len(html) for html, _ in chunks) < 1.1 * len(whole)  # measured 1.05x


def test_an_overlap_budget_larger_than_the_document_cannot_collapse_the_step():
    # Half the window's characters is what stops a budget nobody sized turning into a
    # step of one element: 30 elements became 27 windows the last time it was unbounded,
    # every element re-sent a dozen times for no more context than two views give.
    chunks = chunk_simplified_html(document(30), max_chars=300, overlap_chars=10**6)
    assert len(chunks) < 20
    slots = sum(len(ids) for _, ids in chunks)
    assert slots <= 2 * 30  # half the characters bounds the duplication at 2x


def test_an_oversized_element_does_not_leave_the_next_window_blind():
    # A window that is nearly all one element has less than the budget to spare, and a
    # cap that rounded that down to no elements would open the next window cold — the way
    # 306 of the 14,114 seams this rule produces open cold behind a window of a single
    # oversized element,
    # which has nothing to give. One element is the floor wherever there is one to give.
    big = "y" * 9000
    body = f'<p _item_id="1">tiny</p><p _item_id="2">{big}</p>' + "".join(
        f'<p _item_id="{i}">Paragraph {i} with a handful of words in it.</p>' for i in (3, 4, 5)
    )
    chunks = chunk_simplified_html(f"<div>{body}</div>", max_chars=9100, overlap_chars=2000)
    assert {i for _, ids in chunks for i in ids} == {"1", "2", "3", "4", "5"}
    for before, after in itertools.pairwise(chunks):
        if len(before[1]) > 1:
            assert carried_chars(before, after) > 0


def test_an_element_larger_than_the_window_still_advances():
    huge = '<div><p _item_id="1">' + ("x" * 5000) + '</p><p _item_id="2">small</p></div>'
    chunks = chunk_simplified_html(huge, max_chars=100, overlap_chars=2000)
    assert {i for _, ids in chunks for i in ids} == {"1", "2"}


@pytest.mark.parametrize("text", ["", "   ", "<div>no ids here</div>"])
def test_nothing_to_chunk_is_no_windows(text: str):
    assert chunk_simplified_html(text, max_chars=1000, overlap_chars=2000) == []


def test_the_grammar_names_the_window_s_own_ids():
    # compact_answer_regex assumes 1..n, which stops being true the moment a document is
    # chunked — window three might hold ids 240 to 242.
    pattern = compact_answer_regex_for(["240", "241", "242"])
    assert "240(main|other)241(main|other)242(main|other)" in pattern


def test_merge_prefers_the_window_that_saw_more_around_the_element():
    # Element 5 is one from the edge in the first window and three from the edge in the
    # second, so the second is better informed and wins.
    labels, _ = merge_chunk_labels(
        [{"5": "other"}, {"5": "main"}],
        [["4", "5", "6"], ["2", "3", "4", "5", "6", "7", "8"]],
    )
    assert labels["5"] == "main"


def test_merge_counts_disagreements_rather_than_hiding_them():
    _, disagreements = merge_chunk_labels(
        [{"1": "main", "2": "main"}, {"2": "other", "3": "main"}],
        [["1", "2"], ["2", "3"]],
    )
    assert disagreements == 1


def test_merge_covers_every_labelled_element():
    labels, _ = merge_chunk_labels(
        [{"1": "main", "2": "other"}, {"2": "other", "3": "main"}], [["1", "2"], ["2", "3"]]
    )
    assert set(labels) == {"1", "2", "3"}


def test_an_unanswered_window_does_not_erase_the_others():
    # One failed request among many must cost its own elements, not the document.
    labels, _ = merge_chunk_labels([{"1": "main"}, {}], [["1", "2"], ["2", "3"]])
    assert labels == {"1": "main"}


def test_the_composite_hands_chunking_down_to_the_stage_that_does_it():
    # The settings are declared on the composite but only ever act inside the simplify
    # stage. A run that accepts the flags and silently sends whole documents anyway looks
    # like a working run until the token counts come back wrong, so assert the handoff.
    from nemo_curator.stages.text.html_extraction import MinerUHtmlExtractor

    simplify = MinerUHtmlExtractor(
        base_url="http://localhost:8000", chunk_max_chars=24_000, chunk_overlap_chars=1_500
    ).decompose()[0]
    assert (simplify.chunk_max_chars, simplify.chunk_overlap_chars) == (24_000, 1_500)


def test_chunking_is_off_unless_it_is_asked_for():
    from nemo_curator.stages.text.html_extraction import MinerUHtmlExtractor

    simplify = MinerUHtmlExtractor(base_url="http://localhost:8000").decompose()[0]
    assert simplify.chunk_max_chars == 0


def test_the_driver_exposes_the_chunking_flags_it_forwards():
    # The composite's parameters are only reachable from a run if argparse accepts them;
    # this pair drifted apart once and cost a GPU job.
    import importlib.util
    import pathlib

    driver = pathlib.Path(__file__).parents[4] / "tutorials/text/mineru-html-extraction/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("mineru_run_pipeline", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(["--input", "in.parquet", "--output", "out", "--chunk-max-chars", "24000"])
    assert (args.chunk_max_chars, args.chunk_overlap_chars) == (24_000, 2_000)


def test_the_driver_reads_no_argument_it_does_not_declare():
    # A log line referring to a removed flag crashed a GPU job after the model had loaded.
    # argparse cannot catch that — the attribute is read long after parsing — so compare
    # the two sets directly.
    import ast
    import pathlib

    driver = pathlib.Path(__file__).parents[4] / "tutorials/text/mineru-html-extraction/run_pipeline.py"
    tree = ast.parse(driver.read_text())
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            declared |= {
                a.value[2:].replace("-", "_")
                for a in node.args
                if isinstance(a, ast.Constant) and str(a.value).startswith("--")
            }
            declared |= {k.value.value for k in node.keywords if k.arg == "dest" and isinstance(k.value, ast.Constant)}
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args"
    }
    assert not used - declared


def test_the_grammar_does_not_let_a_model_stall_in_whitespace():
    # An unbounded \s* either side of the body means whitespace is a legal next token
    # forever. A 0.5B model takes that offer: it opens <answer> and emits blank lines
    # until the token budget runs out, which surfaces as an empty answer rather than an
    # error. Three of the first chunked run's 25 windows were lost this way.
    import re

    from nemo_curator.stages.text.html_extraction.mineru_utils import (
        compact_answer_regex,
        compact_answer_regex_for,
    )

    for pattern in (compact_answer_regex(2), compact_answer_regex_for(["1", "2"])):
        stalled = "<answer>" + "\n   " * 20 + "1main2main</answer>"
        assert re.fullmatch(pattern, stalled) is None
        assert re.fullmatch(pattern, "<answer>\n1main2main\n</answer>") is not None
