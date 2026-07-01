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

"""Behavioral unit tests for Dripper stages."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.text.experimental.dripper import (
    DripperHTMLExtractionStage,
    DripperHTMLInferenceStage,
    DripperHTMLLayoutTemplateStage,
    DripperHTMLLayoutPropagationStage,
    DripperHTMLPreprocessStage,
)
from nemo_curator.stages.text.experimental.dripper import _mapping_serialization as mapping_ser
from nemo_curator.stages.text.experimental.dripper import _base_stages as base_mod
from nemo_curator.stages.text.experimental.dripper import layout_template as layout_mod
from nemo_curator.stages.text.experimental.dripper import propagation_stage as propagation_mod
from nemo_curator.stages.text.experimental.dripper._layout_planning import (
    _build_layout_group_plans,
    _select_validation_indexes,
)
from nemo_curator.stages.text.experimental.dripper import stage as stage_mod
from nemo_curator.tasks import DocumentBatch

# ---------------------------------------------------------------------------
# Fake types / helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeInput:
    raw_html: str
    url: str | None = None


@dataclass
class FakeOutput:
    main_html: str
    main_content: str | None = None


@dataclass
class FakeCase:
    input_data: FakeInput
    case_id: str = "fake-case"
    process_data: object = None
    generate_input: object = None
    generate_output: object = None
    parse_result: object = None
    output_data: object = None


class RecordingAsyncClient(AsyncLLMClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(max_concurrent_requests=8, max_retries=0, base_delay=0.0)
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.setup_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: object = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        self.calls.append({"messages": list(messages), "model": model, "generation_config": generation_config})
        return [self.responses.pop(0)]


def _make_mineru_bindings(label_aware: bool = False) -> stage_mod._MinerUHTMLBindings:
    def simplify_single_input(case: FakeCase) -> FakeCase:
        if "preprocess-fails" in case.input_data.raw_html:
            raise RuntimeError("preprocess failed")
        body = (
            "<main>No item ids</main>"
            if "no-items" in case.input_data.raw_html
            else f'<main _item_id="1">{case.input_data.raw_html}</main>'
        )
        case.process_data = SimpleNamespace(
            simpled_html=body, map_html=f"<html><body>{case.input_data.raw_html}</body></html>"
        )
        return case

    def parse_result(case: FakeCase) -> FakeCase:
        if case.generate_output.response == "bad-response":
            raise RuntimeError("parse failed")
        if label_aware:
            case.parse_result = SimpleNamespace(
                item_label=dict(re.findall(r"(\d+)(main|other)", case.generate_output.response))
            )
        else:
            case.parse_result = SimpleNamespace(item_label={"1": "main"})
        return case

    def extract_main_html_single(case: FakeCase) -> FakeCase:
        if label_aware:
            labels = getattr(case.parse_result, "item_label", {})
            main_ids = [iid for iid, lbl in labels.items() if lbl == "main"]
            case.output_data = FakeOutput(main_html="|".join(f"main:{iid}" for iid in main_ids))
        else:
            main_html = (
                "" if "empty-main" in case.input_data.raw_html else f"<article>{case.input_data.raw_html}</article>"
            )
            case.output_data = FakeOutput(main_html=main_html)
        return case

    def extract_main_html_fallback(case: FakeCase, fallback_handler: object) -> FakeCase:
        main_html = (
            "" if "empty-main" in case.input_data.raw_html else f"<fallback>{case.input_data.raw_html}</fallback>"
        )
        case.output_data = FakeOutput(main_html=main_html)
        return case

    def convert2content(case: FakeCase, output_format: str) -> FakeCase:
        if not case.output_data.main_html:
            raise RuntimeError("ExtractorChain base exception#Error during extraction: Document is empty")
        case.output_data.main_content = f"{output_format}:{case.output_data.main_html}"
        return case

    return stage_mod._MinerUHTMLBindings(
        input_cls=FakeInput,
        case_cls=FakeCase,
        output_cls=FakeOutput,
        process_data_cls=SimpleNamespace,
        generate_output_cls=lambda response: SimpleNamespace(response=response),
        simplify_single_input=simplify_single_input,
        build_prompt=lambda case, prompt_version=None: setattr(
            case, "generate_input", SimpleNamespace(full_prompt=f"{prompt_version}:{case.process_data.simpled_html}")
        )
        or case,
        parse_result=parse_result,
        extract_main_html_single=extract_main_html_single,
        extract_main_html_fallback=extract_main_html_fallback,
        convert2content=convert2content,
        get_fallback_handler=lambda fb: SimpleNamespace(name=fb),
    )


def _make_llm_web_kit_bindings(
    *, map_parser_cls=None, layout_parser_cls=None, get_feature=None, cluster_html_struct=None
) -> stage_mod._LLMWebKitBindings:
    class _DefaultMapParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, typical_data: dict) -> dict:
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": "<article>template</article>",
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class _DefaultLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
            }

    def _default_cluster(
        samples: list[dict[str, Any]], threshold: float = 0.95
    ) -> tuple[list[dict[str, Any]], list[int]]:
        for s in samples:
            s["layout_id"] = 0
        return samples, [0]

    return stage_mod._LLMWebKitBindings(
        get_feature=get_feature or (lambda html: {"tags": {1: ["body"], 2: [html]}}),
        cluster_html_struct=cluster_html_struct or _default_cluster,
        select_representative_html=lambda candidates: candidates[0] if candidates else None,
        map_parser_cls=map_parser_cls or _DefaultMapParser,
        layout_parser_cls=layout_parser_cls or _DefaultLayoutParser,
    )


def _batch(data: dict) -> DocumentBatch:
    return DocumentBatch(dataset_name="d", data=pd.DataFrame(data))


def _patch_mineru_bindings(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", factory)
    monkeypatch.setattr(base_mod, "_load_mineru_html_bindings", factory)
    monkeypatch.setattr(layout_mod, "_load_mineru_html_bindings", factory)
    monkeypatch.setattr(propagation_mod, "_load_mineru_html_bindings", factory)


def _patch_web_bindings(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    monkeypatch.setattr(stage_mod, "_load_llm_web_kit_bindings", factory)
    monkeypatch.setattr(layout_mod, "_load_llm_web_kit_bindings", factory)
    monkeypatch.setattr(propagation_mod, "_load_llm_web_kit_bindings", factory)


@pytest.fixture(autouse=True)
def patch_mineru_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mineru_bindings(monkeypatch, _make_mineru_bindings)


# ---------------------------------------------------------------------------
# DripperHTMLExtractionStage
# ---------------------------------------------------------------------------


def test_extraction_stage_runs_pipeline_with_async_client() -> None:
    client = RecordingAsyncClient(["1main"])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
        keep_intermediate=True,
        generation_config=GenerationConfig(max_tokens=2048),
    )
    out = stage.process(_batch({"url": ["https://example.test/a"], "html": ["<html>Hello</html>"]})).to_pandas()

    assert client.setup_calls == 1
    assert out["dripper_response"].tolist() == ["1main"]
    assert out["dripper_html"].tolist() == ["<article><html>Hello</html></article>"]
    assert out["dripper_simplified_html"].str.contains("_item_id").all()
    assert client.calls[0]["model"] == "dripper"


def test_extraction_stage_error_paths_use_fallback_and_warnings() -> None:
    def _run(html: str, responses: list[str]) -> pd.Series:
        client = RecordingAsyncClient(responses)
        stage = DripperHTMLExtractionStage(client=client, model_name="dripper", html_col="html", health_check=False)
        return stage.process(_batch({"html": [html]})).to_pandas().iloc[0]

    row = _run("<html>Fallback</html>", ["bad-response"])
    assert row["dripper_html"] == "<fallback><html>Fallback</html></fallback>"
    assert "parse failed" in row["dripper_warning"]

    row2 = _run("<html>no-items</html>", [])
    assert "no _item_id attributes" in row2["dripper_warning"]

    row3 = _run("", [])
    assert row3["dripper_warning"] == "empty HTML input"

    row4 = _run("<html>empty-main</html>", ["1main"])
    assert "Document is empty" in row4["dripper_warning"]
    assert row4["dripper_content"] == ""


def test_extraction_stage_decodes_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RecordingAsyncClient(["1main"])
    stage = DripperHTMLExtractionStage(client=client, model_name="dripper", html_col="html", health_check=False)
    out = stage.process(_batch({"html": [b"<html>Bad\xffByte</html>"]})).to_pandas()
    assert out.loc[0, "dripper_error"] == ""
    assert client.calls


def test_extraction_stage_missing_bindings_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mineru_bindings(
        monkeypatch, lambda: (_ for _ in ()).throw(RuntimeError("missing mineru"))
    )
    stage = DripperHTMLExtractionStage(
        client=RecordingAsyncClient(["1main"]), model_name="dripper", html_col="html", health_check=False
    )
    with pytest.raises(RuntimeError, match="missing mineru"):
        stage.setup()


# ---------------------------------------------------------------------------
# DripperHTMLInferenceStage
# ---------------------------------------------------------------------------


def test_inference_stage_deduplicates_identical_prompts() -> None:
    client = RecordingAsyncClient(["1main", "1other"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", generation_config=GenerationConfig(max_tokens=2048))
    inference = DripperHTMLInferenceStage(
        client=client, model_name="dripper", health_check=False, generation_config=GenerationConfig(max_tokens=2048)
    )
    batch = _batch({"html": ["<html>Same</html>", "<html>Same</html>", "<html>Different</html>"]})
    out = inference.process(preprocess.process(batch)).to_pandas()
    assert len(client.calls) == 2
    assert out["dripper_response"].tolist() == ["1main", "1main", "1other"]
    assert out["dripper_inference_time_s"].iloc[1] == 0.0


# ---------------------------------------------------------------------------
# DripperHTMLLayoutTemplateStage
# ---------------------------------------------------------------------------


def test_layout_stage_uses_precomputed_layout_id_column() -> None:
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        host_col="url_host_name",
        layout_id_col="dripper_layout_id",
    )
    stage._web_bindings = _make_llm_web_kit_bindings()
    df = pd.DataFrame(
        {
            "url": [f"https://a.example/{i}" for i in range(5)] + ["https://b.example/1", "https://b.example/2"],
            "url_host_name": ["a.example"] * 5 + ["b.example"] * 2,
            "dripper_layout_id": [
                "a.example_0",
                "a.example_0",
                "a.example_1",
                "a.example_1",
                "-1",
                "a.example_0",
                "a.example_0",
            ],
            "html": ["<p>x</p>"] * 7,
            stage_mod._DRIPPER_NEEDS_LLM_COL: [True] * 7,
        }
    )
    plans = _build_layout_group_plans(stage._planning_cfg, df)
    assert [(p.host_key, p.source, p.indexes) for p in plans] == [
        ("a.example", "precomputed_layout:a.example_0", [0, 1]),
        ("a.example", "precomputed_layout:a.example_1", [2, 3]),
        ("b.example", "precomputed_layout:a.example_0", [5, 6]),
    ]


def test_layout_stage_propagates_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_web_bindings(monkeypatch, _make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html", url_col="url", generation_config=GenerationConfig(max_tokens=2048)
    )
    layout = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(max_tokens=2048),
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
    )

    def _no_fallback(*_a, **_kw):
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(layout, "_fallback_row", _no_fallback)
    batch = _batch(
        {
            "url": ["https://example.test/a", "https://example.test/b", "https://example.test/c"],
            "html": ["<html>Rep</html>", "<html>Sib1</html>", "<html>Sib2</html>"],
        }
    )
    out = layout.process(preprocess.process(batch)).to_pandas()
    assert len(client.calls) == 1
    assert out["dripper_layout_representative"].tolist() == [True, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, True]
    assert out["dripper_layout_propagation_success"].tolist() == [False, True, True]


def test_layout_stage_deferred_mapping_uses_lossless_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_web_bindings(monkeypatch, _make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_defer_propagation=True,
        layout_template_require_success=True,
    )
    batch = _batch(
        {
            "url": ["https://example.test/a", "https://example.test/b"],
            "html": ["<html>Rep</html>", "<html>Sib</html>"],
        }
    )

    out = layout.process(preprocess.process(batch)).to_pandas()
    mapping_blob = out.loc[0, "dripper_layout_mapping_json"]
    mapping_data = mapping_ser.parse_mapping_data(mapping_blob)

    assert mapping_blob.startswith("pickle_b64:")
    assert mapping_data is not None
    assert mapping_data["html_element_dict"] == {"labels": {"item_id 1": 1}}
    assert out["dripper_layout_pending_propagation"].tolist() == [False, True]


def test_propagation_stage_decodes_pickle_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mineru_bindings(monkeypatch, lambda: _make_mineru_bindings())

    def _fake_lbp(params, html, mapping_data, dynamic, _parser_cache=None):  # noqa: ANN001, ARG001
        element_dict = mapping_data["html_element_dict"]
        tuple_key_ok = isinstance(next(iter(next(iter(element_dict.values())).keys())), tuple)
        return f"<article>{html}|tuple_key_ok={tuple_key_ok}</article>", ""

    monkeypatch.setattr(propagation_mod, "_run_lbp", _fake_lbp)
    mapping_blob = mapping_ser.serialize_mapping_data(
        {
            "html_element_dict": {1: {("article", "post", None, "id1", 1, 0): ("red", ("body", None, None))}},
            "typical_dict_html": "<html></html>",
        }
    )
    stage = DripperHTMLLayoutPropagationStage(
        layout_template_min_content_length_ratio=None,
        layout_template_max_content_length_ratio=None,
    )
    batch = _batch(
        {
            "url": ["https://example.test/rep", "https://example.test/sib"],
            "html": ["<html>Rep</html>", "<html>Sib</html>"],
            "dripper_layout_cluster": ["cluster-1", "cluster-1"],
            "dripper_layout_representative": [True, False],
            "dripper_layout_mapping_json": [mapping_blob, ""],
            "dripper_layout_pending_propagation": [False, True],
        }
    )

    out = stage.process(batch).to_pandas()

    assert bool(out.loc[1, "dripper_layout_propagation_success"]) is True
    assert "tuple_key_ok=True" in out.loc[1, "dripper_content"]


def test_layout_stage_validation_falls_back_to_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DivergingLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {"main_html_body": '<article _item_id="2">propagated sibling</article>', "main_html_success": True}

    class _LabelMapParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, typical_data: dict) -> dict:
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": '<article _item_id="1">template</article>',
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    _patch_mineru_bindings(monkeypatch, lambda: _make_mineru_bindings(label_aware=True))
    _patch_web_bindings(
        monkeypatch,
        lambda: _make_llm_web_kit_bindings(map_parser_cls=_LabelMapParser, layout_parser_cls=_DivergingLayoutParser),
    )
    client = RecordingAsyncClient(["1main", "1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_max_selected_item_ratio=1.0,
        layout_template_validation_rows=1,
        layout_template_validation_min_content_f1=0.98,
    )
    batch = _batch(
        {
            "url": [f"https://example.test/{c}" for c in "abc"],
            "html": [
                '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                '<p _item_id="1">Val main</p><p _item_id="2">Val nav</p>',
                '<p _item_id="1">Rem main</p><p _item_id="2">Rem nav</p>',
            ],
        }
    )
    out = layout.process(preprocess.process(batch)).to_pandas()
    assert len(client.calls) == 3
    assert out["dripper_layout_fallback_llm"].tolist() == [False, True, True]
    assert "layout template validation failed" in out.loc[1, "dripper_warning"]


def test_layout_stage_splits_by_url_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_web_bindings(monkeypatch, lambda: _make_llm_web_kit_bindings())
    client = RecordingAsyncClient(["1main", "1main"])
    layout = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_max_selected_item_ratio=1.0,
        layout_page_signature_mode="url_shape",
    )
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    batch = _batch(
        {
            "url": [
                "https://x.test/archive.html?start=10",
                "https://x.test/archive.html?start=20",
                "https://x.test/news/123.html",
                "https://x.test/news/456.html",
            ],
            "html": ["<p>Archive 1</p>", "<p>Archive 2</p>", "<p>Article 1</p>", "<p>Article 2</p>"],
        }
    )
    out = layout.process(preprocess.process(batch)).to_pandas()
    assert len(client.calls) == 2
    assert out["dripper_layout_cluster"].nunique() == 2


def test_layout_stage_uses_feature_hash_for_large_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    def _get_feature(html: str) -> dict:
        if "same" in html:
            return {"tags": {1: ["body"], 2: ["article", "nav"]}}
        return {"tags": {1: ["body"], 2: ["aside"]}}

    def _no_dbscan(samples: list, threshold: float = 0.95):
        raise AssertionError("feature_hash mode should not call exact DBSCAN")

    _patch_web_bindings(
        monkeypatch,
        lambda: _make_llm_web_kit_bindings(get_feature=_get_feature, cluster_html_struct=_no_dbscan),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    layout = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_max_exact_host_pages=2,
        layout_large_host_mode="feature_hash",
    )
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    batch = _batch(
        {
            "url": [f"https://x.test/{c}" for c in "abcd"],
            "html": [
                "<html>same rep</html>",
                "<html>same sib</html>",
                "<html>other lone</html>",
                "<html>same sib2</html>",
            ],
        }
    )
    out = layout.process(preprocess.process(batch)).to_pandas()
    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, False, False]
    assert out["dripper_layout_standalone_llm"].tolist() == [False, False, True, False]


def test_layout_stage_validation_indexes_cover_strata() -> None:
    df = pd.DataFrame({"url": [f"https://t.test/{i}" for i in range(10)], "dripper_item_count": list(range(10))})
    cols = ("url", "dripper_item_count")
    assert _select_validation_indexes(df, [], 2, cols) == []
    assert _select_validation_indexes(df, [1, 2, 3, 4], 2, cols) == [1, 4]
    assert _select_validation_indexes(df, list(range(10)), 4, cols) == [0, 3, 6, 9]
