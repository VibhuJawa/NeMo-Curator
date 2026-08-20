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


def document(n: int) -> str:
    body = "".join(f'<p _item_id="{i}">Paragraph {i} with a handful of words in it.</p>' for i in range(1, n + 1))
    return f"<div>{body}</div>"


def test_every_element_appears_in_some_window():
    # The whole point. An element no window contains is an element no model ever sees,
    # and it would be silently dropped from the document with nothing to say so.
    chunks = chunk_simplified_html(document(120), max_chars=1500, overlap=8)
    covered = {i for _, ids in chunks for i in ids}
    assert covered == {str(i) for i in range(1, 121)}


def test_windows_are_in_document_order_and_contiguous():
    chunks = chunk_simplified_html(document(120), max_chars=1500, overlap=8)
    for _, ids in chunks:
        numbers = [int(i) for i in ids]
        assert numbers == sorted(numbers)
        assert numbers == list(range(numbers[0], numbers[-1] + 1))


def test_a_document_that_fits_is_one_window_with_no_duplicates():
    # Chunking has to be a no-op when it is not needed, or every run pays for overlap
    # it did not require.
    chunks = chunk_simplified_html(document(30), max_chars=10**9, overlap=8)
    assert len(chunks) == 1
    assert chunks[0][1] == [str(i) for i in range(1, 31)]


def test_overlap_repeats_elements_at_the_seams():
    chunks = chunk_simplified_html(document(120), max_chars=1500, overlap=8)
    total = sum(len(ids) for _, ids in chunks)
    assert total > 120  # something is repeated
    for before, after in itertools.pairwise(chunks):
        assert set(before[1]) & set(after[1])  # consecutive windows share elements


def test_overlap_cannot_collapse_the_step():
    # Overlap larger than the window steps forward one element at a time: 30 elements
    # became 27 windows, every element re-sent a dozen times for no more context.
    chunks = chunk_simplified_html(document(30), max_chars=300, overlap=100)
    assert len(chunks) < 20


def test_an_element_larger_than_the_window_still_advances():
    huge = '<div><p _item_id="1">' + ("x" * 5000) + '</p><p _item_id="2">small</p></div>'
    chunks = chunk_simplified_html(huge, max_chars=100, overlap=4)
    assert {i for _, ids in chunks for i in ids} == {"1", "2"}


@pytest.mark.parametrize("text", ["", "   ", "<div>no ids here</div>"])
def test_nothing_to_chunk_is_no_windows(text: str):
    assert chunk_simplified_html(text, max_chars=1000, overlap=4) == []


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
        base_url="http://localhost:8000", chunk_max_chars=24_000, chunk_overlap=6
    ).decompose()[0]
    assert (simplify.chunk_max_chars, simplify.chunk_overlap) == (24_000, 6)


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
    assert (args.chunk_max_chars, args.chunk_overlap) == (24_000, 8)


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
