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

import subprocess
import sys
import types

import pandas as pd
import pytest

from nemo_curator.stages.text.html_extraction.mineru_html import (
    MAP_HTML_FIELD,
    N_ITEMS_FIELD,
    RESPONSE_FIELD,
    STATUS_FIELD,
    TOKENS_FIELD,
    MinerUHtmlExtractor,
    MinerUHtmlExtractStage,
    MinerUHtmlSimplifyStage,
    compact_response_budget,
)
from nemo_curator.stages.text.html_extraction.mineru_server import MinerUHtmlServerInferenceStage
from nemo_curator.tasks import DocumentBatch

pytest.importorskip("mineru_html", reason="mineru_html is not installed")

SERVER_URL = "http://localhost:8000"

PAGE = """<!DOCTYPE html><html><head><title>T</title></head><body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<article><h1>How to bake bread</h1>
<p>Start by mixing flour, water, salt and yeast in a large bowl until combined.</p>
<p>Let the dough rest for twenty minutes, then fold it over on itself four times.</p>
</article>
<footer>Copyright 2026 Example Corp</footer></body></html>"""


def make_batch(pages: list[str]) -> DocumentBatch:
    return DocumentBatch(
        dataset_name="t",
        data=pd.DataFrame({"content": pages, "url": [f"https://example.com/{i}" for i in range(len(pages))]}),
    )


def label_all(simplified: pd.DataFrame, label: str) -> str:
    """A compact model answer giving every element of a simplified row the same label."""
    n = int(simplified[N_ITEMS_FIELD].iloc[0])
    return "<answer>" + "".join(f"{i}{label}" for i in range(1, n + 1)) + "</answer>"


class TestCompactResponseBudget:
    def test_scales_with_items(self) -> None:
        assert compact_response_budget(100) > compact_response_budget(10)

    def test_has_a_floor(self) -> None:
        assert compact_response_budget(0) == 64


class TestSimplifyStage:
    @pytest.fixture
    def stage(self) -> MinerUHtmlSimplifyStage:
        s = MinerUHtmlSimplifyStage()
        s.setup()
        return s

    def test_produces_tokens_and_map_html(self, stage: MinerUHtmlSimplifyStage) -> None:
        out = stage.process(make_batch([PAGE])).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "ok"
        assert out[N_ITEMS_FIELD].iloc[0] > 0
        assert "_item_id" in out[MAP_HTML_FIELD].iloc[0]
        assert len(out[TOKENS_FIELD].iloc[0]) > 0

    def test_emitted_tokens_are_the_chat_wrapped_prompt(self, stage: MinerUHtmlSimplifyStage) -> None:
        # The completions route does not apply a chat template, so the stage must,
        # and what reaches the server must be exactly those tokens.
        out = stage.process(make_batch([PAGE])).to_pandas()
        prompt = stage._chat_wrap(stage._simplify_one(PAGE)[0])
        assert "How to bake bread" in prompt
        assert list(out[TOKENS_FIELD].iloc[0]) == stage._tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def test_prompt_is_chat_templated_once(self, stage: MinerUHtmlSimplifyStage) -> None:
        # Exactly one BOS marker: applying the template twice is the upstream bug.
        prompt = stage._chat_wrap("hello")
        assert prompt.count("hy_begin") == 1
        assert "hy_User" in prompt

    def test_upstream_double_template_mode(self) -> None:
        stage = MinerUHtmlSimplifyStage(chat_template_mode="upstream_double")
        stage.setup()
        assert stage._chat_wrap("hello").count("hy_begin") == 2

    def test_non_utf8_bytes_are_decoded_not_mangled(self, stage: MinerUHtmlSimplifyStage) -> None:
        # The bytes path is the Common Crawl path. decode("utf-8", errors="replace")
        # turned every non-UTF-8 page into replacement characters, which the model
        # then labelled as garbage while the row kept status "ok" -- invisible to
        # extraction_rate. Charset detection needs a realistic amount of text, so
        # this page is padded rather than a snippet.
        body = (
            "Beyonc\u00e9 \u2014 na\u00efve caf\u00e9. "
            + "L'\u00e9t\u00e9 dernier, o\u00f9 la f\u00eate \u00e9tait tr\u00e8s anim\u00e9e. " * 60
        )
        page = f"<html><body><article><h1>{body}</h1><p>{body}</p></article></body></html>"
        batch = DocumentBatch(dataset_name="t", data=pd.DataFrame({"content": [page.encode("windows-1252")]}))
        out = stage.process(batch).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "ok"
        assert "\ufffd" not in out[MAP_HTML_FIELD].iloc[0]
        assert "Beyonc\u00e9" in out[MAP_HTML_FIELD].iloc[0]

    def test_undecodable_bytes_are_flagged_not_crashed(self, stage: MinerUHtmlSimplifyStage) -> None:
        batch = DocumentBatch(dataset_name="t", data=pd.DataFrame({"content": [b"\xff\xfe\x00\x00"]}))
        assert stage.process(batch).to_pandas()[STATUS_FIELD].iloc[0] == "empty_input"

    def test_accepts_bytes(self, stage: MinerUHtmlSimplifyStage) -> None:
        batch = DocumentBatch(dataset_name="t", data=pd.DataFrame({"content": [PAGE.encode()]}))
        assert stage.process(batch).to_pandas()[STATUS_FIELD].iloc[0] == "ok"

    def test_empty_input_is_flagged(self, stage: MinerUHtmlSimplifyStage) -> None:
        out = stage.process(make_batch([""])).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "empty_input"

    def test_over_long_input_is_flagged(self) -> None:
        stage = MinerUHtmlSimplifyStage(max_model_len=16)
        stage.setup()
        out = stage.process(make_batch([PAGE])).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "too_long"

    def test_the_element_cap_shrinks_the_prompt_and_not_the_output(self) -> None:
        # simplify_html returns (simplified, map_html): the prompt is built from the
        # first and the output rebuilt from the second, which is what makes abridging
        # safe. A table the model labels main must still come out whole, so the cap has
        # to leave map_html alone -- and it must not change how many labels are asked
        # for, or the grammar would name an element the window never showed.
        rows = "".join(f"<tr><td>cell {i}</td><td>{i}</td></tr>" for i in range(500))
        big = f"<html><body><article><h1>Results</h1><table>{rows}</table></article></body></html>"

        whole = MinerUHtmlSimplifyStage()
        whole.setup()
        capped = MinerUHtmlSimplifyStage(element_max_chars=2_000)
        capped.setup()

        before, after = whole._simplify_one(big), capped._simplify_one(big)
        assert len(after[0]) < len(before[0]) / 2  # the prompt
        assert after[1] == before[1]  # map_html, untouched
        assert "cell 499" in after[1]
        assert after[2] == before[2]  # the same elements to label

    def test_the_element_cap_is_off_by_default(self, stage: MinerUHtmlSimplifyStage) -> None:
        assert stage.element_max_chars == 0

    def test_drop_html_field(self) -> None:
        stage = MinerUHtmlSimplifyStage(drop_html_field=True)
        stage.setup()
        assert "content" not in stage.process(make_batch([PAGE])).to_pandas().columns

    def test_pandas_null_html_does_not_kill_the_partition(self, stage: MinerUHtmlSimplifyStage) -> None:
        # dtype_backend="numpy_nullable" -- which both entry points use -- turns a
        # null in a string content column into pd.NA, and bool(pd.NA) raises. That
        # escaped process() and took every good row in the batch with it.
        batch = DocumentBatch(dataset_name="t", data=pd.DataFrame({"content": [PAGE, pd.NA, None, float("nan")]}))
        out = stage.process(batch).to_pandas()
        assert out[STATUS_FIELD].tolist() == ["ok", "empty_input", "empty_input", "empty_input"]

    def test_empty_batch_is_handled(self) -> None:
        # The fast tokenizer raises IndexError on an empty batch, so an empty
        # partition would otherwise kill the pipeline.
        s = MinerUHtmlSimplifyStage()
        s.setup()
        out = s.process(DocumentBatch(dataset_name="t", data=pd.DataFrame({"content": []}))).to_pandas()
        assert len(out) == 0

    def test_declared_outputs_are_present(self, stage: MinerUHtmlSimplifyStage) -> None:
        out = stage.process(make_batch([PAGE])).to_pandas()
        for col in stage.outputs()[1]:
            assert col in out.columns


class TestExtractStage:
    @pytest.fixture
    def simplified(self) -> pd.DataFrame:
        s = MinerUHtmlSimplifyStage()
        s.setup()
        return s.process(make_batch([PAGE])).to_pandas()

    def _run(self, df: pd.DataFrame, response: str, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df[RESPONSE_FIELD] = [response]
        stage = MinerUHtmlExtractStage(**kwargs)
        stage.setup()
        return stage.process(DocumentBatch(dataset_name="t", data=df)).to_pandas()

    def test_labels_all_main_keeps_body_text(self, simplified: pd.DataFrame) -> None:
        out = self._run(simplified, label_all(simplified, "main"))
        assert "bake bread" in out["text"].iloc[0]

    def test_labels_all_other_yields_blank_text(self, simplified: pd.DataFrame) -> None:
        # Pruning everything leaves an empty document shell, which the Markdown
        # converter renders as whitespace rather than the empty string.
        out = self._run(simplified, label_all(simplified, "other"))
        assert out["text"].iloc[0].strip() == ""

    def test_output_format_none_emits_pruned_html(self, simplified: pd.DataFrame) -> None:
        out = self._run(simplified, label_all(simplified, "main"), output_format="none")
        text = out["text"].iloc[0]
        assert text.lstrip().startswith("<")
        assert "bake bread" in text

    def test_internal_columns_are_dropped(self, simplified: pd.DataFrame) -> None:
        out = self._run(simplified, "<answer>1main</answer>")
        assert MAP_HTML_FIELD not in out.columns
        assert RESPONSE_FIELD not in out.columns

    def test_keep_internal_columns(self, simplified: pd.DataFrame) -> None:
        out = self._run(simplified, "<answer>1main</answer>", keep_internal_fields=True)
        assert MAP_HTML_FIELD in out.columns

    def test_main_html_field(self, simplified: pd.DataFrame) -> None:
        out = self._run(simplified, "<answer>1main</answer>", main_html_field="main_html")
        assert "main_html" in out.columns

    def test_failed_row_uses_fallback(self, simplified: pd.DataFrame) -> None:
        df = simplified.copy()
        df[STATUS_FIELD] = ["too_long"]
        out = self._run(df, "", fallback="trafilatura")
        assert out[STATUS_FIELD].iloc[0] == "too_long"
        assert "bake bread" in out["text"].iloc[0]

    def test_empty_fallback_yields_empty_text(self, simplified: pd.DataFrame) -> None:
        df = simplified.copy()
        df[STATUS_FIELD] = ["simplify_error"]
        out = self._run(df, "", fallback="empty")
        assert out["text"].iloc[0] == ""

    def test_bypass_fallback_returns_the_raw_html(self, simplified: pd.DataFrame) -> None:
        df = simplified.copy()
        df[STATUS_FIELD] = ["simplify_error"]
        out = self._run(df, "", fallback="bypass", output_format="none")
        assert out["text"].iloc[0] == PAGE

    def test_fallback_matches_upstream_handlers(self, simplified: pd.DataFrame) -> None:
        # The three handlers are reimplemented rather than imported from mineru_html
        # (see test_setup_does_not_import_mineru_html); they must stay byte-identical.
        from mineru_html.process.map_to_main import get_fallback_handler

        for fallback in ("trafilatura", "bypass", "empty"):
            stage = MinerUHtmlExtractStage(fallback=fallback)
            stage.setup()
            assert stage._fallback_html(PAGE) == get_fallback_handler(fallback).fallback_func(PAGE)

    def test_setup_does_not_import_mineru_html(self) -> None:
        # Importing any mineru_html submodule executes its __init__, which pulls in the
        # transformers and vLLM backends: ~33s and ~790 MB of GPU-idle ramp per actor,
        # for code this CPU-only stage never runs.
        code = (
            "import sys;"
            "from nemo_curator.stages.text.html_extraction import MinerUHtmlExtractStage;"
            "MinerUHtmlExtractStage().setup();"
            "assert 'mineru_html' not in sys.modules, 'extract setup imported mineru_html'"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)  # noqa: S603


class TestServerInferenceRouting:
    """Rows with nothing to label must never reach the server."""

    def _stage(self) -> MinerUHtmlServerInferenceStage:
        stage = MinerUHtmlServerInferenceStage(base_url=SERVER_URL)

        def _never_called(*_args, **_kwargs) -> None:
            msg = "the server must not be called for these rows"
            raise AssertionError(msg)

        stage._client = _never_called
        return stage

    def _batch(self, status: str, n_items: int) -> DocumentBatch:
        return DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame(
                {
                    STATUS_FIELD: [status],
                    N_ITEMS_FIELD: [n_items],
                    TOKENS_FIELD: [[1, 2, 3]],
                    MAP_HTML_FIELD: ["<html></html>"],
                }
            ),
        )

    def test_skips_rows_with_no_item_ids(self) -> None:
        out = self._stage().process(self._batch("ok", 0)).to_pandas()
        assert out[RESPONSE_FIELD].iloc[0] == ""

    def test_skips_rows_that_failed_earlier(self) -> None:
        out = self._stage().process(self._batch("too_long", 12)).to_pandas()
        assert out[RESPONSE_FIELD].iloc[0] == ""

    def test_base_url_v1_suffix_is_not_doubled(self) -> None:
        assert MinerUHtmlServerInferenceStage(base_url="http://h:8000/v1/").base_url == "http://h:8000"


class _FakeClient:
    """Stands in for AsyncOpenAI: `client.post(...)` and `close()`.

    The stage posts the raw body rather than calling `completions.create`, because
    the SDK's typed method walks every element of the token-id list (~47 ms per
    document against ~2 ms here).
    """

    def __init__(self, fail: bool = False, fail_prompts: set[str] | None = None, no_choices: bool = False):
        self.completions = self
        self.fail = fail
        self.fail_prompts = fail_prompts or set()
        self.no_choices = no_choices
        self.closed = False
        self.calls: list[dict] = []

    async def post(self, _path: str, *, cast_to=None, body: dict, **_kw) -> types.SimpleNamespace:  # noqa: ANN001
        return await self.create(**body)

    async def create(self, **kwargs) -> types.SimpleNamespace:
        if self.fail or kwargs["prompt"] in self.fail_prompts:
            msg = "server is down"
            raise RuntimeError(msg)
        self.calls.append(kwargs)
        if self.no_choices:
            return types.SimpleNamespace(choices=[])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(text=f"<answer>{kwargs['prompt']}</answer>")])

    async def close(self) -> None:
        self.closed = True


class TestServerInferenceRequests:
    """The HTTP path: one client and one event loop per worker, reused across batches."""

    def _stage(self, **client_kwargs) -> tuple[MinerUHtmlServerInferenceStage, list[_FakeClient]]:
        stage = MinerUHtmlServerInferenceStage(base_url=SERVER_URL)
        built: list[_FakeClient] = []

        def _factory() -> _FakeClient:
            built.append(_FakeClient(**client_kwargs))
            return built[-1]

        stage._client = _factory
        return stage, built

    def _batch(self, prompts: list[str]) -> DocumentBatch:
        return DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame(
                {
                    STATUS_FIELD: ["ok"] * len(prompts),
                    N_ITEMS_FIELD: [2] * len(prompts),
                    TOKENS_FIELD: prompts,
                }
            ),
        )

    def test_responses_land_on_their_own_rows(self) -> None:
        stage, _ = self._stage()
        out = stage.process(self._batch(["a", "b"])).to_pandas()
        assert out[RESPONSE_FIELD].tolist() == ["<answer>a</answer>", "<answer>b</answer>"]

    def test_client_is_reused_across_batches(self) -> None:
        # Rebuilding it per batch threw away the keep-alive pool, so every batch paid
        # up to max_concurrency TCP handshakes before its first request went out.
        stage, built = self._stage()
        stage.process(self._batch(["a"]))
        stage.process(self._batch(["b"]))
        assert len(built) == 1

    def test_teardown_closes_the_client(self) -> None:
        stage, built = self._stage()
        stage.process(self._batch(["a"]))
        stage.teardown()
        assert built[0].closed

    def test_prompt_column_is_dropped(self) -> None:
        stage, _ = self._stage()
        assert TOKENS_FIELD not in stage.process(self._batch(["a"])).to_pandas().columns

    def test_partial_failure_is_marked_so_the_row_falls_back(self) -> None:
        # The dangerous case, and the one the all-failed guard does NOT cover. A row
        # whose request was lost must not keep status "ok": downstream that parses
        # into an empty label map, prunes the whole document and emits blank text
        # that never reaches the fallback and still counts as a success.
        stage, _ = self._stage(fail_prompts={"b"})
        out = stage.process(self._batch(["a", "b", "c"])).to_pandas()
        assert out[STATUS_FIELD].tolist() == ["ok", "inference_error", "ok"]
        assert out[RESPONSE_FIELD].tolist() == ["<answer>a</answer>", "", "<answer>c</answer>"]

    def test_malformed_response_is_caught_per_row(self) -> None:
        # An empty `choices` list used to raise IndexError straight out of gather(),
        # bypassing every per-row guard. It is now caught like any other request
        # failure -- so with every response malformed it surfaces as the ordinary
        # all-requests-failed RuntimeError rather than a bare IndexError.
        stage, _ = self._stage(no_choices=True)
        with pytest.raises(RuntimeError, match="all 2 requests"):
            stage.process(self._batch(["a", "b"]))

    def test_wholesale_failure_raises(self) -> None:
        # A run where every request failed once reported a 2.2x "speedup", because
        # empty responses are indistinguishable downstream from a fast run.
        stage, _ = self._stage(fail=True)
        with pytest.raises(RuntimeError, match="ran no inference"):
            stage.process(self._batch(["a", "b"]))

    def test_max_tokens_is_sized_per_document(self) -> None:
        stage, built = self._stage()
        stage.process(self._batch(["a"]))
        assert built[0].calls[0]["max_tokens"] == compact_response_budget(2)

    def test_sampling_defaults_are_pinned(self) -> None:
        # The checkpoint's own generation_config collapses extraction_rate to 0.015.
        # They ride in the request body itself now that the stage posts raw.
        stage, built = self._stage()
        stage.process(self._batch(["a"]))
        body = built[0].calls[0]
        assert body["repetition_penalty"] == 1.0
        assert body["top_k"] == -1


class TestComposite:
    def test_decomposes_into_three_stages(self) -> None:
        stages = MinerUHtmlExtractor(base_url=SERVER_URL).decompose()
        assert [type(s) for s in stages] == [
            MinerUHtmlSimplifyStage,
            MinerUHtmlServerInferenceStage,
            MinerUHtmlExtractStage,
        ]

    def test_no_stage_requests_a_gpu(self) -> None:
        # The whole point of the server architecture: the pipeline needs no GPU.
        for stage in MinerUHtmlExtractor(base_url=SERVER_URL).decompose():
            assert stage.resources.gpus == 0

    def test_base_url_reaches_the_inference_stage(self) -> None:
        inference = MinerUHtmlExtractor(base_url=SERVER_URL).decompose()[1]
        assert inference.base_url == SERVER_URL

    def test_base_url_is_required(self) -> None:
        with pytest.raises(TypeError, match="base_url"):
            MinerUHtmlExtractor()

    def test_bypass_fallback_keeps_raw_html(self) -> None:
        # bypass returns the original document, so it needs the column just as much
        # as trafilatura does. Dropping it made bypass silently identical to empty.
        simplify = MinerUHtmlExtractor(base_url=SERVER_URL, fallback="bypass").decompose()[0]
        assert simplify.drop_html_field is False

    def test_non_trafilatura_fallback_drops_raw_html(self) -> None:
        simplify = MinerUHtmlExtractor(base_url=SERVER_URL, fallback="empty").decompose()[0]
        assert simplify.drop_html_field is True

    def test_trafilatura_fallback_keeps_raw_html(self) -> None:
        simplify = MinerUHtmlExtractor(base_url=SERVER_URL, fallback="trafilatura").decompose()[0]
        assert simplify.drop_html_field is False

    def test_worker_overrides_apply(self) -> None:
        simplify, inference, extract = MinerUHtmlExtractor(
            base_url=SERVER_URL, simplify_workers=8, inference_workers=2, extract_workers=4
        ).decompose()
        assert simplify.num_workers() == 8
        assert inference.num_workers() == 2
        assert extract.num_workers() == 4

    def test_workers_default_to_backend_autoscaling(self) -> None:
        for stage in MinerUHtmlExtractor(base_url=SERVER_URL).decompose():
            assert stage.num_workers() is None

    def test_answer_regex_pins_every_element_id(self) -> None:
        import re

        from nemo_curator.stages.text.html_extraction.mineru_utils import compact_answer_regex

        pattern = compact_answer_regex(2)
        assert re.fullmatch(pattern, "<answer>1main2other</answer>") is not None
        # every id, in order, exactly once: no skipping, no reordering, no stopping early
        for wrong in (
            "<answer>1main</answer>",
            "<answer>2other1main</answer>",
            "<answer>1main2other3main</answer>",
            "<answer>1maybe2other</answer>",
        ):
            assert re.fullmatch(pattern, wrong) is None
