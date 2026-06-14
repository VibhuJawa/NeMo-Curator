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

"""Unit tests for DripperHTMLExtractionStage."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.text.experimental.dripper import stage as stage_mod
from nemo_curator.stages.text.experimental.dripper.stage import (
    DripperHTMLExtractionStage,
    DripperHTMLInferenceStage,
    DripperHTMLLayoutTemplateStage,
    DripperHTMLPreprocessStage,
)
from nemo_curator.tasks import DocumentBatch


@dataclass
class FakeInput:
    raw_html: str
    url: str | None = None


@dataclass
class FakeGenerateOutput:
    response: str


@dataclass
class FakeOutput:
    main_html: str
    main_content: str | None = None


@dataclass
class FakeProcessData:
    simpled_html: str
    map_html: str


class FakeCase:
    def __init__(self, input_data: FakeInput) -> None:
        self.input_data = input_data
        self.case_id = "fake-case"
        self.process_data = None
        self.generate_input = None
        self.generate_output = None
        self.parse_result = None
        self.output_data = None


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
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "generation_config": generation_config,
            }
        )
        return [self.responses.pop(0)]


def make_bindings() -> stage_mod._MinerUHTMLBindings:
    def simplify_single_input(case: FakeCase) -> FakeCase:
        if "preprocess-fails" in case.input_data.raw_html:
            raise RuntimeError("preprocess failed")
        if "no-items" in case.input_data.raw_html:
            case.process_data = SimpleNamespace(
                simpled_html="<main>No item ids</main>",
                map_html="<html><body>No item ids</body></html>",
            )
            return case
        case.process_data = SimpleNamespace(
            simpled_html=f'<main _item_id="1">{case.input_data.raw_html}</main>',
            map_html=f"<html><body>{case.input_data.raw_html}</body></html>",
        )
        return case

    def build_prompt(case: FakeCase, prompt_version: str) -> FakeCase:
        case.generate_input = SimpleNamespace(full_prompt=f"{prompt_version}:{case.process_data.simpled_html}")
        return case

    def parse_result(case: FakeCase) -> FakeCase:
        if case.generate_output.response == "bad-response":
            raise RuntimeError("parse failed")
        case.parse_result = SimpleNamespace(item_label={"1": "main"})
        return case

    def extract_main_html_single(case: FakeCase) -> FakeCase:
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
        process_data_cls=FakeProcessData,
        generate_output_cls=FakeGenerateOutput,
        simplify_single_input=simplify_single_input,
        build_prompt=build_prompt,
        parse_result=parse_result,
        extract_main_html_single=extract_main_html_single,
        extract_main_html_fallback=extract_main_html_fallback,
        convert2content=convert2content,
        get_fallback_handler=lambda fallback: SimpleNamespace(name=fallback),
    )


def make_label_aware_bindings() -> stage_mod._MinerUHTMLBindings:
    base = make_bindings()

    def parse_result(case: FakeCase) -> FakeCase:
        matches = re.findall(r"(\d+)(main|other)", case.generate_output.response)
        case.parse_result = SimpleNamespace(item_label=dict(matches))
        return case

    def extract_main_html_single(case: FakeCase) -> FakeCase:
        labels = getattr(case.parse_result, "item_label", {})
        main_ids = [item_id for item_id, label in labels.items() if label == "main"]
        case.output_data = FakeOutput(main_html="|".join(f"main:{item_id}" for item_id in main_ids))
        return case

    return stage_mod._MinerUHTMLBindings(
        input_cls=base.input_cls,
        case_cls=base.case_cls,
        output_cls=base.output_cls,
        process_data_cls=base.process_data_cls,
        generate_output_cls=base.generate_output_cls,
        simplify_single_input=base.simplify_single_input,
        build_prompt=base.build_prompt,
        parse_result=parse_result,
        extract_main_html_single=extract_main_html_single,
        extract_main_html_fallback=base.extract_main_html_fallback,
        convert2content=base.convert2content,
        get_fallback_handler=base.get_fallback_handler,
    )


def make_llm_web_kit_bindings() -> stage_mod._LLMWebKitBindings:
    class FakeMapParser:
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

    class FakeLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
            }

    def cluster_html_struct(
        samples: list[dict[str, Any]], threshold: float = 0.95
    ) -> tuple[list[dict[str, Any]], list[int]]:
        for sample in samples:
            sample["layout_id"] = 0
        return samples, [0]

    def select_representative_html(candidates: list[dict[str, str]]) -> dict[str, str] | None:
        return candidates[0] if candidates else None

    return stage_mod._LLMWebKitBindings(
        get_feature=lambda html: {"tags": {1: ["body"], 2: [html]}},
        cluster_html_struct=cluster_html_struct,
        select_representative_html=select_representative_html,
        map_parser_cls=FakeMapParser,
        layout_parser_cls=FakeLayoutParser,
    )


@pytest.fixture(autouse=True)
def patch_mineru_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", make_bindings)


# ---------------------------------------------------------------------------
# Layout template helper tests
# ---------------------------------------------------------------------------


def test_layout_template_validation_indexes_spread_and_cover_strata() -> None:
    df = pd.DataFrame(
        {
            "url": [f"https://example.test/{idx}" for idx in range(10)],
            "dripper_item_count": list(range(10)),
        }
    )
    # Spread across cluster
    assert stage_mod._select_validation_indexes(df, [], 2, ("url", "dripper_item_count")) == []
    assert stage_mod._select_validation_indexes(df, [1, 2, 3, 4], 2, ("url", "dripper_item_count")) == [1, 4]
    assert stage_mod._select_validation_indexes(df, list(range(10)), 4, ("url", "dripper_item_count")) == [0, 3, 6, 9]

    # Cover query-value strata
    df2 = pd.DataFrame(
        {
            "url": [
                "https://example.test/page?id=a&context=1",
                "https://example.test/page?id=b&context=1",
                "https://example.test/page?id=c&context=0",
                "https://example.test/page?id=d&context=2",
                "https://example.test/page?id=e&context=0",
                "https://example.test/page?id=f&context=1",
            ],
            "dripper_item_count": [10] * 6,
        }
    )
    assert stage_mod._select_validation_indexes(df2, list(range(6)), 4, ("url", "dripper_item_count")) == [0, 2, 3, 5]


def test_layout_template_stage_uses_precomputed_layout_id_column() -> None:
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        host_col="url_host_name",
        layout_id_col="dripper_layout_id",
    )
    stage._web_bindings = make_llm_web_kit_bindings()
    df = pd.DataFrame(
        {
            "url": [
                "https://a.example/1",
                "https://a.example/2",
                "https://a.example/3",
                "https://a.example/4",
                "https://a.example/noise",
                "https://b.example/1",
                "https://b.example/2",
            ],
            "url_host_name": [
                "a.example",
                "a.example",
                "a.example",
                "a.example",
                "a.example",
                "b.example",
                "b.example",
            ],
            "dripper_layout_id": [
                "a.example_0",
                "a.example_0",
                "a.example_1",
                "a.example_1",
                "-1",
                "a.example_0",
                "a.example_0",
            ],
            "html": ["<p>a</p>", "<p>b</p>", "<p>c</p>", "<p>d</p>", "<p>noise</p>", "<p>e</p>", "<p>f</p>"],
            stage_mod._DRIPPER_NEEDS_LLM_COL: [True, True, True, True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [(plan.host_key, plan.source, plan.indexes) for plan in plans] == [
        ("a.example", "precomputed_layout:a.example_0", [0, 1]),
        ("a.example", "precomputed_layout:a.example_1", [2, 3]),
        ("b.example", "precomputed_layout:a.example_0", [5, 6]),
    ]


# ---------------------------------------------------------------------------
# Core extraction stage
# ---------------------------------------------------------------------------


def test_stage_reuses_mineru_pipeline_with_async_client() -> None:
    client = RecordingAsyncClient(["1main", "2main"])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
        keep_intermediate=True,
        generation_config=GenerationConfig(
            max_tokens=2048,
            extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": ["https://example.test/a", None],
                "html": ["<html>Hello</html>", b"<html>Bytes</html>"],
            }
        ),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert client.setup_calls == 1
    assert out["dripper_response"].tolist() == ["1main", "2main"]
    assert out["dripper_error"].tolist() == ["", ""]
    assert out["dripper_html"].tolist() == [
        "<article><html>Hello</html></article>",
        "<article><html>Bytes</html></article>",
    ]
    assert out["dripper_content"].tolist() == [
        "mm_md:<article><html>Hello</html></article>",
        "mm_md:<article><html>Bytes</html></article>",
    ]
    assert out["dripper_item_count"].tolist() == [1, 1]
    assert out["dripper_request_max_tokens"].tolist() == [2048, 2048]
    assert out["dripper_simplified_html"].str.contains("_item_id").all()
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == "dripper"
    assert client.calls[0]["generation_config"].extra_kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    assert client.calls[0]["messages"] == [
        {"role": "user", "content": 'short_compact:<main _item_id="1"><html>Hello</html></main>'}
    ]


# ---------------------------------------------------------------------------
# Layout template propagation
# ---------------------------------------------------------------------------


def test_layout_template_stage_infers_representative_and_propagates_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage_mod, "_load_llm_web_kit_bindings", make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        url_col="url",
        prompt_version="short_compact",
        generation_config=GenerationConfig(max_tokens=2048),
    )
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        generation_config=GenerationConfig(max_tokens=2048),
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
    )

    def fail_unused_fallback(_row: pd.Series, *, primary_error: str = "") -> stage_mod._LayoutTemplateRowResult:
        raise AssertionError("_fallback_row should not run when all layout rows produced results")

    monkeypatch.setattr(layout_stage, "_fallback_row", fail_unused_fallback)
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [
                    "https://example.test/a",
                    "https://example.test/b",
                    "https://example.test/c",
                ],
                "html": [
                    "<html>Rep</html>",
                    "<html>Sibling One</html>",
                    "<html>Sibling Two</html>",
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 1
    assert out["dripper_layout_representative"].tolist() == [True, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, True]
    assert out["dripper_layout_propagation_success"].tolist() == [False, True, True]
    assert out["dripper_html"].tolist() == [
        "<article><html>Rep</html></article>",
        "<propagated><html>Sibling One</html></propagated>",
        "<propagated><html>Sibling Two</html></propagated>",
    ]
    assert out["dripper_content"].tolist() == [
        "mm_md:<article><html>Rep</html></article>",
        "mm_md:<propagated><html>Sibling One</html></propagated>",
        "mm_md:<propagated><html>Sibling Two</html></propagated>",
    ]


def test_layout_template_stage_validates_cluster_before_propagating_remaining_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
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

    class DivergingLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": '<article _item_id="2">propagated sibling</article>',
                "main_html_success": True,
            }

    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FakeMapParser,
            layout_parser_cls=DivergingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_max_selected_item_ratio=1.0,
        layout_template_validation_rows=1,
        layout_template_validation_min_content_f1=0.98,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [
                    "https://example.test/a",
                    "https://example.test/b",
                    "https://example.test/c",
                ],
                "html": [
                    '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                    '<p _item_id="1">Validation main</p><p _item_id="2">Validation nav</p>',
                    '<p _item_id="1">Remaining main</p><p _item_id="2">Remaining nav</p>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 3
    assert out["dripper_layout_representative"].tolist() == [True, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, False, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [False, True, True]
    assert out.loc[1, "dripper_html"] == "main:1"
    assert "layout template validation failed" in out.loc[1, "dripper_warning"]
    assert out.loc[2, "dripper_html"] == "main:1"
    assert "layout template validation LLM" in out.loc[2, "dripper_warning"]


def test_layout_template_stage_splits_layout_groups_by_url_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()
    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: base_webkit_bindings,
    )
    client = RecordingAsyncClient(["1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_max_selected_item_ratio=1.0,
        layout_page_signature_mode="url_shape",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [
                    "https://example.test/archive.html?start=10",
                    "https://example.test/archive.html?start=20",
                    "https://example.test/news/123-first.html",
                    "https://example.test/news/456-second.html",
                ],
                "html": [
                    "<p>Archive page 1</p>",
                    "<p>Archive page 2</p>",
                    "<p>Article page 1</p>",
                    "<p>Article page 2</p>",
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, True, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, True]
    assert out["dripper_layout_cluster"].nunique() == 2


def test_layout_template_stage_uses_feature_hash_for_large_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    def get_feature(html: str) -> dict[str, dict[int, list[str]]]:
        if "same-layout" in html:
            return {"tags": {1: ["body"], 2: ["article", "nav"]}, "attrs": {2: ["content"]}}
        return {"tags": {1: ["body"], 2: ["aside"]}, "attrs": {2: ["sidebar"]}}

    def cluster_html_struct(
        samples: list[dict[str, Any]], threshold: float = 0.95
    ) -> tuple[list[dict[str, Any]], list[int]]:
        raise AssertionError("feature_hash large-host mode should not call exact DBSCAN")

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=get_feature,
            cluster_html_struct=cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=base_webkit_bindings.layout_parser_cls,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_max_exact_host_pages=2,
        layout_template_large_host_mode="feature_hash",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [
                    "https://example.test/a",
                    "https://example.test/b",
                    "https://example.test/c",
                    "https://example.test/d",
                ],
                "html": [
                    "<html>same-layout rep</html>",
                    "<html>same-layout sibling one</html>",
                    "<html>other-layout standalone</html>",
                    "<html>same-layout sibling two</html>",
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, True]
    assert out["dripper_layout_standalone_llm"].tolist() == [False, False, True, False]


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------


def test_layout_fingerprints() -> None:
    # feature fingerprint is order-insensitive
    assert stage_mod._layout_feature_fingerprint(
        {"tags": {1: ["body"], 2: ["article", "nav", "article"]}, "attrs": {2: ["content", "main"]}}
    ) == stage_mod._layout_feature_fingerprint(
        {"attrs": {2: ["main", "content"]}, "tags": {2: ["nav", "article", "article"], 1: ["body"]}}
    )
    # dom-path fingerprint preserves order, normalizes dynamic attrs
    assert stage_mod._layout_dom_path_fingerprint(
        '<html><body><main class="post-123"><h1>A</h1><p>B</p></main></body></html>'
    ) == stage_mod._layout_dom_path_fingerprint(
        '<html><body><main class="post-456"><h1>C</h1><p>D</p></main></body></html>'
    )
    assert stage_mod._layout_dom_path_fingerprint(
        '<html><body><main class="post-123"><h1>A</h1><p>B</p></main></body></html>'
    ) != stage_mod._layout_dom_path_fingerprint(
        '<html><body><main class="post-123"><p>B</p><h1>A</h1></main></body></html>'
    )


# ---------------------------------------------------------------------------
# Split / inference stage
# ---------------------------------------------------------------------------


def test_split_inference_stage_deduplicates_identical_prompts() -> None:
    client = RecordingAsyncClient(["1main", "1other"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        generation_config=GenerationConfig(max_tokens=2048),
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(max_tokens=2048),
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Same</html>", "<html>Same</html>", "<html>Different</html>"]}),
    )

    out = inference.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_response"].tolist() == ["1main", "1main", "1other"]
    assert out["dripper_inference_time_s"].iloc[1] == 0.0


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------


def test_stage_error_paths_use_fallback_and_warnings() -> None:
    # parse error -> fallback extraction path
    client = RecordingAsyncClient(["bad-response"])
    stage = DripperHTMLExtractionStage(client=client, model_name="dripper", html_col="html", health_check=False)
    out = stage.process(
        DocumentBatch(task_id="t", dataset_name="d", data=pd.DataFrame({"html": ["<html>Fallback</html>"]}))
    ).to_pandas()
    assert out.loc[0, "dripper_html"] == "<fallback><html>Fallback</html></fallback>"
    assert out.loc[0, "dripper_error"] == ""
    assert "parse failed" in out.loc[0, "dripper_warning"]

    # no item IDs -> skips LLM
    client2 = RecordingAsyncClient([])
    stage2 = DripperHTMLExtractionStage(client=client2, model_name="dripper", html_col="html", health_check=False)
    out2 = stage2.process(
        DocumentBatch(task_id="t", dataset_name="d", data=pd.DataFrame({"html": ["<html>no-items</html>"]}))
    ).to_pandas()
    assert client2.calls == []
    assert "no _item_id attributes" in out2.loc[0, "dripper_warning"]

    # empty HTML input -> warning, no content
    client3 = RecordingAsyncClient([])
    stage3 = DripperHTMLExtractionStage(client=client3, model_name="dripper", html_col="html", health_check=False)
    out3 = stage3.process(DocumentBatch(task_id="t", dataset_name="d", data=pd.DataFrame({"html": [""]}))).to_pandas()
    assert out3.loc[0, "dripper_warning"] == "empty HTML input"

    # empty-main document -> warning, no content
    client4 = RecordingAsyncClient(["1main"])
    stage4 = DripperHTMLExtractionStage(client=client4, model_name="dripper", html_col="html", health_check=False)
    out4 = stage4.process(
        DocumentBatch(task_id="t", dataset_name="d", data=pd.DataFrame({"html": ["<html>empty-main</html>"]}))
    ).to_pandas()
    assert "Document is empty" in out4.loc[0, "dripper_warning"]
    assert out4.loc[0, "dripper_content"] == ""


def test_stage_decodes_bytes_even_when_charset_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_mod, "_decode_html_bytes", lambda _html_bytes: None)
    client = RecordingAsyncClient(["1main"])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": [b"<html>Bad\xffByte</html>"]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert "Bad" in out.loc[0, "dripper_html"]
    assert client.calls


def test_setup_reports_missing_mineru_html(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_bindings() -> stage_mod._MinerUHTMLBindings:
        raise RuntimeError("missing mineru")

    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", missing_bindings)
    stage = DripperHTMLExtractionStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        html_col="html",
        health_check=False,
    )

    with pytest.raises(RuntimeError, match="missing mineru"):
        stage.setup()
