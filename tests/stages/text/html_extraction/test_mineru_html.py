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

import pandas as pd
import pytest

from nemo_curator.stages.text.html_extraction.mineru_html import (
    MAP_HTML_FIELD,
    N_ITEMS_FIELD,
    PROMPT_FIELD,
    RESPONSE_FIELD,
    STATUS_FIELD,
    TOKENS_FIELD,
    MinerUHtmlExtractor,
    MinerUHtmlExtractStage,
    MinerUHtmlInferenceStage,
    MinerUHtmlSimplifyStage,
    compact_response_budget,
)
from nemo_curator.tasks import DocumentBatch

pytest.importorskip("mineru_html", reason="mineru_html is not installed")

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


class TestCompactResponseBudget:
    def test_scales_with_items(self) -> None:
        assert compact_response_budget(100) > compact_response_budget(10)

    def test_has_a_floor(self) -> None:
        assert compact_response_budget(0) == 64


class TestSimplifyStage:
    @pytest.fixture
    def stage(self) -> MinerUHtmlSimplifyStage:
        s = MinerUHtmlSimplifyStage(pretokenize=False)
        s.setup()
        return s

    def test_produces_prompt_and_map_html(self, stage: MinerUHtmlSimplifyStage) -> None:
        out = stage.process(make_batch([PAGE])).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "ok"
        assert out[N_ITEMS_FIELD].iloc[0] > 0
        assert "_item_id" in out[MAP_HTML_FIELD].iloc[0]
        assert "How to bake bread" in out[PROMPT_FIELD].iloc[0]

    def test_prompt_is_chat_templated_once(self, stage: MinerUHtmlSimplifyStage) -> None:
        # vLLM's generate() does not apply a chat template, so the stage must.
        # Exactly one BOS marker: applying the template twice is the upstream bug.
        prompt = stage.process(make_batch([PAGE])).to_pandas()[PROMPT_FIELD].iloc[0]
        assert prompt.count("hy_begin") == 1
        assert "hy_User" in prompt

    def test_upstream_double_template_mode(self) -> None:
        stage = MinerUHtmlSimplifyStage(pretokenize=False, chat_template_mode="upstream_double")
        stage.setup()
        prompt = stage.process(make_batch([PAGE])).to_pandas()[PROMPT_FIELD].iloc[0]
        assert prompt.count("hy_begin") == 2

    def test_pretokenized_prompts_match_the_text_prompt(self) -> None:
        text_stage = MinerUHtmlSimplifyStage(pretokenize=False)
        text_stage.setup()
        tok_stage = MinerUHtmlSimplifyStage(pretokenize=True)
        tok_stage.setup()

        prompt = text_stage.process(make_batch([PAGE])).to_pandas()[PROMPT_FIELD].iloc[0]
        ids = tok_stage.process(make_batch([PAGE])).to_pandas()[TOKENS_FIELD].iloc[0]
        assert list(ids) == tok_stage._tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def test_accepts_bytes(self, stage: MinerUHtmlSimplifyStage) -> None:
        batch = DocumentBatch(dataset_name="t", data=pd.DataFrame({"content": [PAGE.encode()]}))
        assert stage.process(batch).to_pandas()[STATUS_FIELD].iloc[0] == "ok"

    def test_empty_input_is_flagged(self, stage: MinerUHtmlSimplifyStage) -> None:
        out = stage.process(make_batch([""])).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "empty_input"

    def test_over_long_input_is_flagged(self) -> None:
        stage = MinerUHtmlSimplifyStage(pretokenize=False, max_model_len=16)
        stage.setup()
        out = stage.process(make_batch([PAGE])).to_pandas()
        assert out[STATUS_FIELD].iloc[0] == "too_long"

    def test_drop_html_field(self) -> None:
        stage = MinerUHtmlSimplifyStage(pretokenize=False, drop_html_field=True)
        stage.setup()
        assert "content" not in stage.process(make_batch([PAGE])).to_pandas().columns

    def test_declared_outputs_are_present(self, stage: MinerUHtmlSimplifyStage) -> None:
        out = stage.process(make_batch([PAGE])).to_pandas()
        for col in stage.outputs()[1]:
            assert col in out.columns


class TestExtractStage:
    @pytest.fixture
    def simplified(self) -> pd.DataFrame:
        s = MinerUHtmlSimplifyStage(pretokenize=False)
        s.setup()
        return s.process(make_batch([PAGE])).to_pandas()

    def _run(self, df: pd.DataFrame, response: str, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df[RESPONSE_FIELD] = [response]
        stage = MinerUHtmlExtractStage(**kwargs)
        stage.setup()
        return stage.process(DocumentBatch(dataset_name="t", data=df)).to_pandas()

    def test_labels_all_main_keeps_body_text(self, simplified: pd.DataFrame) -> None:
        n = int(simplified[N_ITEMS_FIELD].iloc[0])
        response = "<answer>" + "".join(f"{i}main" for i in range(1, n + 1)) + "</answer>"
        out = self._run(simplified, response)
        assert "bake bread" in out["text"].iloc[0]

    def test_labels_all_other_yields_blank_text(self, simplified: pd.DataFrame) -> None:
        # Pruning everything leaves an empty document shell, which the Markdown
        # converter renders as whitespace rather than the empty string.
        n = int(simplified[N_ITEMS_FIELD].iloc[0])
        response = "<answer>" + "".join(f"{i}other" for i in range(1, n + 1)) + "</answer>"
        out = self._run(simplified, response)
        assert out["text"].iloc[0].strip() == ""

    def test_output_format_none_emits_pruned_html(self, simplified: pd.DataFrame) -> None:
        n = int(simplified[N_ITEMS_FIELD].iloc[0])
        response = "<answer>" + "".join(f"{i}main" for i in range(1, n + 1)) + "</answer>"
        out = self._run(simplified, response, output_format="none")
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


class TestInferenceStageRouting:
    """The GPU stage must not call vLLM for rows that have nothing to label."""

    def _stage(self) -> MinerUHtmlInferenceStage:
        stage = MinerUHtmlInferenceStage()

        class _NeverCalled:
            def generate(self, *_args, **_kwargs) -> None:
                msg = "vLLM must not be called for these rows"
                raise AssertionError(msg)

        stage._llm = _NeverCalled()
        return stage

    def _batch(self, status: str, n_items: int) -> DocumentBatch:
        return DocumentBatch(
            dataset_name="t",
            data=pd.DataFrame(
                {
                    STATUS_FIELD: [status],
                    N_ITEMS_FIELD: [n_items],
                    PROMPT_FIELD: ["prompt"],
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


class TestComposite:
    def test_decomposes_into_three_stages(self) -> None:
        stages = MinerUHtmlExtractor().decompose()
        assert [type(s) for s in stages] == [
            MinerUHtmlSimplifyStage,
            MinerUHtmlInferenceStage,
            MinerUHtmlExtractStage,
        ]

    def test_gpu_only_on_the_inference_stage(self) -> None:
        simplify, inference, extract = MinerUHtmlExtractor().decompose()
        assert simplify.resources.gpus == 0
        assert inference.resources.gpus == 1
        assert extract.resources.gpus == 0

    def test_non_trafilatura_fallback_drops_raw_html(self) -> None:
        simplify = MinerUHtmlExtractor(fallback="empty").decompose()[0]
        assert simplify.drop_html_field is True

    def test_trafilatura_fallback_keeps_raw_html(self) -> None:
        simplify = MinerUHtmlExtractor(fallback="trafilatura").decompose()[0]
        assert simplify.drop_html_field is False

    def test_worker_overrides_apply(self) -> None:
        simplify, inference, extract = MinerUHtmlExtractor(
            simplify_workers=8, inference_workers=2, extract_workers=4
        ).decompose()
        assert simplify.num_workers() == 8
        assert inference.num_workers() == 2
        assert extract.num_workers() == 4

    def test_workers_default_to_backend_autoscaling(self) -> None:
        for stage in MinerUHtmlExtractor().decompose():
            assert stage.num_workers() is None

    def test_server_backend_swaps_the_inference_stage(self) -> None:
        simplify, inference, extract = MinerUHtmlExtractor(
            backend="server", base_url="http://localhost:8000"
        ).decompose()
        # Same CPU stages either way; only the middle stage changes, and it owns no GPU.
        assert simplify.name == "mineru_html_simplify"
        assert extract.name == "mineru_html_extract"
        assert inference.name == "mineru_html_server_inference"
        assert inference.resources.gpus == 0

    def test_server_backend_requires_a_base_url(self) -> None:
        with pytest.raises(ValueError, match="requires base_url"):
            MinerUHtmlExtractor(backend="server")

    def test_answer_regex_is_shared_by_both_backends(self) -> None:
        # The two inference paths must constrain output identically.
        from nemo_curator.stages.text.html_extraction.mineru_html import compact_answer_regex

        assert compact_answer_regex(2) == r"<answer>\s*1(main|other)2(main|other)\s*</answer>"
