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

import asyncio
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
    DripperHTMLExtractionPipelineStage,
    DripperHTMLExtractionStage,
    DripperHTMLInferenceStage,
    DripperHTMLLayoutClusteringStage,
    DripperHTMLLayoutTemplateStage,
    DripperHTMLPostprocessStage,
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


class DelayedRecordingAsyncClient(RecordingAsyncClient):
    def __init__(self, responses: list[str], *, delay_s: float = 0.01) -> None:
        super().__init__(responses)
        self.delay_s = delay_s
        self.in_flight = 0
        self.max_in_flight = 0

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: object = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay_s)
            return await super()._query_model_impl(
                messages=messages,
                model=model,
                conversation_formatter=conversation_formatter,
                generation_config=generation_config,
            )
        finally:
            self.in_flight -= 1


class PromptAwareClient(RecordingAsyncClient):
    def __init__(self) -> None:
        super().__init__([])

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: object = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        message_list = list(messages)
        self.calls.append(
            {
                "messages": message_list,
                "model": model,
                "generation_config": generation_config,
            }
        )
        prompt = str(message_list[0].get("content", "")) if message_list else ""
        return ["2main1other" if ">B " in prompt else "1main2other"]


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
        main_html = "" if "empty-main" in case.input_data.raw_html else f"<article>{case.input_data.raw_html}</article>"
        case.output_data = FakeOutput(main_html=main_html)
        return case

    def extract_main_html_fallback(case: FakeCase, fallback_handler: object) -> FakeCase:  # noqa: ARG001
        main_html = "" if "empty-main" in case.input_data.raw_html else f"<fallback>{case.input_data.raw_html}</fallback>"
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
        case.parse_result = SimpleNamespace(item_label={item_id: label for item_id, label in matches})
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
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
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
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
            }

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
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


def test_layout_template_validation_indexes_are_spread_across_cluster() -> None:
    df = pd.DataFrame(
        {
            "url": [f"https://example.test/{idx}" for idx in range(10)],
            "dripper_item_count": list(range(10)),
        }
    )

    assert stage_mod._select_validation_indexes(df, [], 2, "url", "dripper_item_count") == []
    assert stage_mod._select_validation_indexes(df, [1, 2, 3, 4], 0, "url", "dripper_item_count") == []
    assert stage_mod._select_validation_indexes(df, [1, 2, 3, 4], 1, "url", "dripper_item_count") == [4]
    assert stage_mod._select_validation_indexes(df, [1, 2, 3, 4], 2, "url", "dripper_item_count") == [1, 4]
    assert stage_mod._select_validation_indexes(df, [1, 2, 3, 4], 3, "url", "dripper_item_count") == [1, 3, 4]
    assert stage_mod._select_validation_indexes(df, [1, 2], 5, "url", "dripper_item_count") == [1, 2]
    assert stage_mod._select_validation_indexes(df, list(range(10)), 4, "url", "dripper_item_count") == [
        0,
        3,
        6,
        9,
    ]


def test_layout_template_validation_indexes_cover_query_value_strata() -> None:
    df = pd.DataFrame(
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

    assert stage_mod._select_validation_indexes(df, list(range(6)), 4, "url", "dripper_item_count") == [
        0,
        2,
        3,
        5,
    ]


def test_layout_template_stage_uses_extra_validation_rows_for_large_clusters() -> None:
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        layout_template_validation_rows=2,
        layout_template_large_cluster_validation_rows=8,
        layout_template_large_cluster_min_size=64,
    )

    assert stage._effective_validation_rows(63) == 2
    assert stage._effective_validation_rows(64) == 8


def test_layout_template_stage_selects_spread_representative_candidates() -> None:
    webkit_bindings = make_llm_web_kit_bindings()
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        layout_template_representative_candidates=3,
    )
    stage._web_bindings = stage_mod._LLMWebKitBindings(
        get_feature=webkit_bindings.get_feature,
        cluster_html_struct=webkit_bindings.cluster_html_struct,
        select_representative_html=lambda candidates: candidates[2],
        map_parser_cls=webkit_bindings.map_parser_cls,
        layout_parser_cls=webkit_bindings.layout_parser_cls,
    )
    df = pd.DataFrame(
        {
            "url": [f"https://example.test/{idx}" for idx in range(5)],
            "html": [f"<html>{idx}</html>" for idx in range(5)],
            "dripper_item_count": list(range(5)),
        }
    )

    assert stage._select_representative_indexes(df, [0, 1, 2, 3, 4]) == [2, 0, 4]


def test_layout_template_stage_groups_by_manifest_host_column() -> None:
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        host_col="url_host_name",
    )
    stage._web_bindings = make_llm_web_kit_bindings()
    df = pd.DataFrame(
        {
            "url": [
                "https://shared.example/a",
                "https://shared.example/b",
                "https://shared.example/c",
                "https://shared.example/d",
            ],
            "url_host_name": ["www.example.com", "www.example.com", "blog.example.com", "blog.example.com"],
            "html": ["<p>a</p>", "<p>b</p>", "<p>c</p>", "<p>d</p>"],
            stage_mod._DRIPPER_NEEDS_LLM_COL: [True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [(plan.host_key, plan.indexes) for plan in plans] == [
        ("www.example.com", [0, 1]),
        ("blog.example.com", [2, 3]),
    ]


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
            "dripper_layout_id": ["a.example_0", "a.example_0", "a.example_1", "a.example_1", "-1", "a.example_0", "a.example_0"],
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


def test_layout_template_stage_can_leave_large_precomputed_layout_group_standalone() -> None:
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        host_col="url_host_name",
        layout_id_col="dripper_layout_id",
        layout_template_max_exact_host_pages=2,
        layout_template_large_host_mode="standalone",
    )
    stage._web_bindings = make_llm_web_kit_bindings()
    df = pd.DataFrame(
        {
            "url": [
                "https://a.example/1",
                "https://a.example/2",
                "https://a.example/3",
                "https://a.example/4",
                "https://a.example/5",
            ],
            "url_host_name": ["a.example"] * 5,
            "dripper_layout_id": [
                "a.example_0",
                "a.example_0",
                "a.example_0",
                "a.example_1",
                "a.example_1",
            ],
            "html": ["<p>a</p>", "<p>b</p>", "<p>c</p>", "<p>d</p>", "<p>e</p>"],
            stage_mod._DRIPPER_NEEDS_LLM_COL: [True, True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [(plan.source, plan.indexes) for plan in plans] == [
        ("precomputed_layout:a.example_1", [3, 4]),
    ]


def test_layout_template_stage_splits_large_precomputed_layout_group_by_dom_path_hash() -> None:
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
        host_col="url_host_name",
        layout_id_col="dripper_layout_id",
        layout_template_max_exact_host_pages=2,
        layout_template_large_host_mode="dom_path_hash",
    )
    stage._web_bindings = make_llm_web_kit_bindings()
    df = pd.DataFrame(
        {
            "url": [
                "https://a.example/1",
                "https://a.example/2",
                "https://a.example/3",
                "https://a.example/4",
            ],
            "url_host_name": ["a.example"] * 4,
            "dripper_layout_id": ["a.example_0"] * 4,
            "html": [
                '<html><body><main class="post-1"><h1>A</h1><p>rep</p></main></body></html>',
                '<html><body><main class="post-2"><h1>B</h1><p>sibling</p></main></body></html>',
                '<html><body><main class="post-3"><p>different</p><h1>C</h1></main></body></html>',
                '<html><body><main class="post-4"><p>other</p><h1>D</h1></main></body></html>',
            ],
            stage_mod._DRIPPER_NEEDS_LLM_COL: [True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [(plan.source, plan.indexes) for plan in plans] == [
        ("precomputed_layout:a.example_0", [0, 1]),
        ("precomputed_layout:a.example_0", [2, 3]),
    ]


def test_layout_clustering_stage_precomputes_host_bounded_layout_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage_mod, "_load_llm_web_kit_bindings", make_llm_web_kit_bindings)
    stage = DripperHTMLLayoutClusteringStage(
        host_col="url_host_name",
        layout_page_signature_mode="url_shape",
    )
    df = pd.DataFrame(
        {
            "url": [
                "https://a.example/article/1",
                "https://a.example/article/2",
                "https://a.example/profile/about",
                "https://b.example/article/1",
                "https://b.example/article/2",
            ],
            "url_host_name": ["a.example", "a.example", "a.example", "b.example", "b.example"],
            "html": [
                "<html><body>a one</body></html>",
                "<html><body>a two</body></html>",
                "<html><body>a singleton</body></html>",
                "<html><body>b one</body></html>",
                "<html><body>b two</body></html>",
            ],
        }
    )

    out = stage.process(DocumentBatch(task_id="task", dataset_name="test", data=df)).to_pandas()

    assert out.loc[0, "dripper_layout_id"]
    assert out.loc[0, "dripper_layout_id"] == out.loc[1, "dripper_layout_id"]
    assert out.loc[2, "dripper_layout_id"] == ""
    assert out.loc[3, "dripper_layout_id"]
    assert out.loc[3, "dripper_layout_id"] == out.loc[4, "dripper_layout_id"]
    assert out.loc[3, "dripper_layout_id"] != out.loc[0, "dripper_layout_id"]


def test_layout_template_stage_filters_dbscan_group_by_exemplar_similarity() -> None:
    webkit_bindings = make_llm_web_kit_bindings()
    stage = DripperHTMLLayoutTemplateStage(
        client=RecordingAsyncClient(["1main"]),
        model_name="dripper",
        health_check=False,
    )
    stage._web_bindings = stage_mod._LLMWebKitBindings(
        get_feature=webkit_bindings.get_feature,
        cluster_html_struct=webkit_bindings.cluster_html_struct,
        select_representative_html=webkit_bindings.select_representative_html,
        map_parser_cls=webkit_bindings.map_parser_cls,
        layout_parser_cls=webkit_bindings.layout_parser_cls,
        similarity=lambda left, right, _max_layer_n: 1.0 if left == right else 0.0,
    )
    df = pd.DataFrame(
        {
            "url": [f"https://example.test/{idx}" for idx in range(4)],
            "html": ["<p>a</p>", "<p>b</p>", "<p>c</p>", "<p>d</p>"],
            stage_mod._DRIPPER_NEEDS_LLM_COL: [True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [plan.indexes for plan in plans] == [[0, 1, 2]]


def test_layout_page_signature_key_splits_query_and_numeric_article_shapes() -> None:
    assert (
        stage_mod._layout_page_signature_key(
            "https://example.test/archive.html?start=10",
            42,
            "url_shape",
        )
        == "url=path=archive.html|q=start"
    )
    assert (
        stage_mod._layout_page_signature_key(
            "https://example.test/news/123-first.html",
            42,
            "url_shape",
        )
        == "url=path=news/#num.html|q="
    )
    assert stage_mod._layout_page_signature_key("https://example.test/a", 42, "item_count_bucket") == "items=33-64"
    assert (
        stage_mod._layout_page_signature_key(
            "https://example.test/news/123-first.html",
            42,
            "url_shape_item_count_bucket",
        )
        == "url=path=news/#num.html|q=|items=33-64"
    )


def test_layout_page_signature_key_semantic_shape_preserves_content_url_tokens() -> None:
    assert (
        stage_mod._layout_page_signature_key(
            "https://wits.worldbank.org/CountryProfile/en/Compare/Country/ABW/Indicator/MPRT-TRD-VL/"
            "partner/WLD/product/UNCTAD-SoP1/region/LCN/show/line",
            42,
            "url_semantic_shape",
        )
        != stage_mod._layout_page_signature_key(
            "https://wits.worldbank.org/CountryProfile/en/Compare/Country/ABW/Indicator/MPRT-TRD-VL/"
            "partner/WLD/product/UNCTAD-SoP3/region/LCN/show/line",
            42,
            "url_semantic_shape",
        )
    )
    assert (
        stage_mod._layout_page_signature_key(
            "https://source.android.com/?authuser=0&hl=es-419",
            42,
            "url_semantic_shape",
        )
        != stage_mod._layout_page_signature_key(
            "https://source.android.com/?authuser=0&hl=pl",
            42,
            "url_semantic_shape",
        )
    )
    assert (
        stage_mod._layout_page_signature_key(
            "https://example.test/news/123-first.html",
            42,
            "url_semantic_shape_item_count_bucket",
        )
        == "url=path=news/123-first.html|q=|items=33-64"
    )


def test_low_card_query_shape_preserves_repeated_query_values_only() -> None:
    urls = [
        f"https://publicpay.test/Reports/Cities/City.aspx?entityid={100 + idx}&year={2012 + idx % 2}&rpt={idx % 3}"
        for idx in range(20)
    ]
    low_card_keys = stage_mod._low_card_query_value_keys(urls)

    assert low_card_keys == {"rpt", "year"}

    signature = stage_mod._layout_page_signature_key_with_low_card_queries(
        urls[0],
        55,
        "url_low_card_query_shape_item_count_exact",
        low_card_keys,
    )

    assert signature == "url=path=reports/cities/city.aspx|q=entityid,rpt=0,year=2012|items=55"


def test_low_card_query_shape_uses_exact_values_when_all_query_values_are_high_card() -> None:
    urls = [f"https://scop.test/astral/jmolview?context={idx}&id={1000 + idx}&ver={idx}" for idx in range(20)]
    low_card_keys = stage_mod._low_card_query_value_keys(urls)

    assert low_card_keys == set()
    assert (
        stage_mod._layout_page_signature_key_with_low_card_queries(
            urls[0],
            55,
            "url_low_card_query_shape_item_count_exact",
            low_card_keys,
        )
        == "url=path=astral/jmolview|q=context=0,id=1000,ver=0|items=55"
    )


def test_low_card_query_shape_keeps_id_exact_when_other_query_keys_are_low_card() -> None:
    urls = [
        f"https://scop.test/astral/jmolview?context={idx % 2}&id=d{idx:04d}&ver={1 + idx % 2}.55"
        for idx in range(20)
    ]
    low_card_keys = stage_mod._low_card_query_value_keys(urls)

    assert low_card_keys == {"context", "ver"}
    assert (
        stage_mod._layout_page_signature_key_with_low_card_queries(
            urls[0],
            5,
            "url_low_card_query_shape_item_count_exact",
            low_card_keys,
        )
        == "url=path=astral/jmolview|q=context=0,id=d0000,ver=1.55|items=5"
    )


def test_failed_fallback_low_card_query_split_ignores_high_card_ids() -> None:
    stage = DripperHTMLLayoutTemplateStage(client=PromptAwareClient(), model_name="dripper", health_check=False)
    rows = []
    for idx in range(20):
        rows.append(
            {
                "url": (
                    "https://publicpay.test/Reports/Cities/City.aspx?"
                    f"entityid={100 + idx}&year={2012 + idx % 2}&rpt={idx % 2}"
                ),
                "dripper_item_count": 55,
            }
        )
    df = pd.DataFrame(rows)

    groups = stage._split_fallback_groups_by_signature(
        df,
        [list(range(20))],
        "url_low_card_query_shape_item_count_exact",
    )

    assert groups == [list(range(0, 20, 2)), list(range(1, 20, 2))]


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


def test_split_stages_match_mineru_pipeline_with_async_client() -> None:
    client = RecordingAsyncClient(["1main", "2main"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        prompt_version="short_compact",
        generation_config=GenerationConfig(max_tokens=2048),
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(max_tokens=2048),
    )
    postprocess = DripperHTMLPostprocessStage(
        html_col="html",
        output_format="mm_md",
        fallback="trafilatura",
        keep_intermediate=True,
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

    result = postprocess.process(inference.process(preprocess.process(batch)))
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


def test_composite_stage_decomposes_into_split_execution_stages() -> None:
    client = RecordingAsyncClient(["1main"])
    composite = DripperHTMLExtractionPipelineStage(
        client=client,
        model_name="dripper",
        generation_config=GenerationConfig(max_tokens=128),
        preprocess_worker_count=2,
        inference_worker_count=3,
        postprocess_worker_count=4,
    )

    stages = composite.decompose()

    assert [type(stage) for stage in stages] == [
        DripperHTMLPreprocessStage,
        DripperHTMLInferenceStage,
        DripperHTMLPostprocessStage,
    ]
    assert [stage.num_workers() for stage in stages] == [2, 3, 4]
    assert stages[1].client is client
    assert client.calls == []


def test_layout_template_defer_fallback_llm_uses_split_inference_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage_mod, "_load_llm_web_kit_bindings", make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main"])
    composite = DripperHTMLExtractionPipelineStage(
        client=client,
        model_name="dripper",
        generation_config=GenerationConfig(max_tokens=128),
        layout_template_mode=True,
        layout_template_defer_fallback_llm=True,
        preprocess_worker_count=2,
        inference_worker_count=3,
        postprocess_worker_count=4,
    )

    stages = composite.decompose()

    assert [type(stage) for stage in stages] == [
        DripperHTMLPreprocessStage,
        DripperHTMLLayoutTemplateStage,
        DripperHTMLInferenceStage,
        DripperHTMLPostprocessStage,
    ]
    assert [stage.num_workers() for stage in stages] == [2, 3, 3, 4]
    assert stages[1].client is client
    assert stages[2].client is client


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

    def fail_unused_fallback(_row: pd.Series, *, primary_error: str = "") -> stage_mod._LayoutTemplateRowResult:  # noqa: ARG001
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


def test_layout_template_stage_retries_representative_candidates_after_mapping_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class RetryMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            if "bad-rep" in typical_data["typical_raw_html"]:
                return {"typical_main_html_success": False}
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": "<article>template</article>",
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=RetryMapParser,
            layout_parser_cls=base_webkit_bindings.layout_parser_cls,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_representative_candidates=2,
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
                    "<html>bad-rep</html>",
                    "<html>Sibling One</html>",
                    "<html>Sibling Two</html>",
                    "<html>good-rep</html>",
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [False, False, False, True]
    assert out["dripper_layout_fallback_llm"].tolist() == [True, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, True, False]
    assert "typical_main_html_success=false" in out.loc[0, "dripper_warning"]


def test_layout_template_stage_fallback_llm_requests_are_concurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FailingMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:  # noqa: ARG002
            return {"typical_main_html_success": False}

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FailingMapParser,
            layout_parser_cls=base_webkit_bindings.layout_parser_cls,
        ),
    )
    client = DelayedRecordingAsyncClient(["1main", "1main", "1main", "1main"])
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
        max_concurrent_requests=4,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
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
                    "<html>Rep</html>",
                    "<html>Sibling One</html>",
                    "<html>Sibling Two</html>",
                    "<html>Sibling Three</html>",
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 4
    assert client.max_in_flight > 1
    assert out["dripper_layout_representative"].tolist() == [False, False, False, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [True, True, True, True]


def test_layout_template_stage_deduplicates_fallback_llm_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FailingMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:  # noqa: ARG002
            return {"typical_main_html_success": False}

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FailingMapParser,
            layout_parser_cls=base_webkit_bindings.layout_parser_cls,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
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
        max_concurrent_requests=4,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
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
                    "<html>Rep</html>",
                    "<html>Duplicate Sibling</html>",
                    "<html>Duplicate Sibling</html>",
                    "<html>Duplicate Sibling</html>",
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [False, False, False, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [True, True, True, True]
    fallback_times = out["dripper_inference_time_s"].tolist()
    assert sum(time_s == 0.0 for time_s in fallback_times) == 2


def test_layout_template_stage_converts_propagated_item_ids_through_mineru(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": '<article _item_id="2">template</article>',
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class FakeLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:  # noqa: ARG002
            return {
                "main_html_body": '<article _item_id="2">Sibling main</article>',
                "main_html_success": True,
            }

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
        for sample in samples:
            sample["layout_id"] = 0
        return samples, [0]

    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=lambda html: {"tags": {1: ["body"], 2: [html]}},
            cluster_html_struct=cluster_html_struct,
            select_representative_html=lambda candidates: candidates[0],
            map_parser_cls=FakeMapParser,
            layout_parser_cls=FakeLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_propagation_target="mapped_item_ids",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": ["https://example.test/a", "https://example.test/b"],
                "html": [
                    '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                    '<p _item_id="2">Sibling main</p><p _item_id="3">Sibling nav</p>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 1
    assert bool(out.loc[1, "dripper_layout_propagated"]) is True
    assert out.loc[1, "dripper_response"] == "2main3other"
    assert out.loc[1, "dripper_html"] == "main:2"
    assert out.loc[1, "dripper_content"] == "mm_md:main:2"


def test_layout_template_stage_uses_raw_html_for_layout_propagation_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()
    seen_html_sources: list[str] = []

    class RecordingLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            seen_html_sources.append(task_data["html_source"])
            return {
                "main_html_body": "<article>raw sibling main</article>",
                "main_html_success": True,
            }

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=RecordingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
    )
    rep_html = '<html><body><p _item_id="1">rep main</p></body></html>'
    sibling_html = '<html><body><p _item_id="2">sibling main</p></body></html>'
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": ["https://example.test/a", "https://example.test/b"],
                "html": [rep_html, sibling_html],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert seen_html_sources == [sibling_html]
    assert bool(out.loc[1, "dripper_layout_propagated"]) is True
    assert out.loc[1, "dripper_response"] == ""
    assert out.loc[1, "dripper_html"] == "<article>raw sibling main</article>"
    assert out.loc[1, "dripper_content"] == "mm_md:<article>raw sibling main</article>"


def test_layout_template_stage_falls_back_when_propagation_overselects_item_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": '<article _item_id="1">template</article>',
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class OverselectingLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:  # noqa: ARG002
            return {
                "main_html_body": '<main><p _item_id="2">body</p><p _item_id="3">metadata</p></main>',
                "main_html_success": True,
            }

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FakeMapParser,
            layout_parser_cls=OverselectingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_max_selected_item_ratio=0.5,
        layout_template_propagation_target="mapped_item_ids",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": ["https://example.test/a", "https://example.test/b"],
                "html": [
                    '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                    (
                        '<p _item_id="2">Sibling main</p>'
                        '<p _item_id="3">Sibling date</p>'
                        '<p _item_id="4">Sibling nav</p>'
                    ),
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert bool(out.loc[1, "dripper_layout_fallback_llm"]) is True
    assert bool(out.loc[1, "dripper_layout_propagated"]) is False
    assert "selected item ratio" in out.loc[1, "dripper_warning"]
    assert out.loc[1, "dripper_html"].startswith("<article>")


def test_layout_template_stage_validates_cluster_before_propagating_remaining_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
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
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:  # noqa: ARG002
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


def test_layout_template_stage_defers_validation_failure_fallback_to_inference_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
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
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": '<article _item_id="2">wrong sibling</article>',
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
        layout_template_defer_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_max_selected_item_ratio=1.0,
        layout_template_validation_rows=1,
        layout_template_validation_min_content_f1=0.98,
    )
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html", url_col="url")
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

    layout_out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert layout_out["dripper_layout_representative"].tolist() == [True, False, False]
    assert layout_out["dripper_layout_fallback_llm"].tolist() == [False, True, True]
    finalized = layout_out[stage_mod._DRIPPER_LAYOUT_FINALIZED_COL].tolist()
    needs_llm = layout_out[stage_mod._DRIPPER_NEEDS_LLM_COL].tolist()
    assert finalized[0]
    assert sum(finalized) == 2
    assert sum(needs_llm) == 1
    deferred_idx = finalized.index(False)
    validation_idx = next(idx for idx in [1, 2] if idx != deferred_idx)
    assert needs_llm[deferred_idx]
    assert not needs_llm[validation_idx]
    assert layout_out.loc[deferred_idx, "dripper_html"] == ""
    assert "layout template validation failed" in layout_out.loc[deferred_idx, stage_mod._DRIPPER_PRIMARY_ERROR_COL]
    assert "layout template validation LLM" in layout_out.loc[validation_idx, "dripper_warning"]

    final_out = postprocess.process(
        inference.process(DocumentBatch(task_id="task-2", dataset_name="test", data=layout_out))
    ).to_pandas()

    assert len(client.calls) == 3
    assert final_out["dripper_html"].tolist() == ["main:1", "main:1", "main:1"]
    assert final_out["dripper_layout_fallback_llm"].tolist() == [False, True, True]


def test_layout_template_stage_validates_spread_siblings_before_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": '<article _item_id="1">template</article>',
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class TailDivergingLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            item_id = "2" if "tail-drift" in task_data["html_source"] else "1"
            return {
                "main_html_body": f'<article _item_id="{item_id}">propagated sibling</article>',
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
            layout_parser_cls=TailDivergingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main", "1main", "1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_require_success=True,
        layout_template_max_selected_item_ratio=1.0,
        layout_template_validation_rows=2,
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
                    "https://example.test/d",
                    "https://example.test/e",
                ],
                "html": [
                    '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                    '<p _item_id="1">Validation main</p><p _item_id="2">Validation nav</p>',
                    '<p _item_id="1">Remaining main 1</p><p _item_id="2">Remaining nav 1</p>',
                    '<p _item_id="1">Remaining main 2</p><p _item_id="2">Remaining nav 2</p>',
                    '<p _item_id="1">tail-drift main</p><p _item_id="2">tail-drift nav</p>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 5
    assert out["dripper_layout_representative"].tolist() == [True, False, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, False, False, False, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [False, True, True, True, True]
    assert "layout template validation LLM" in out.loc[1, "dripper_warning"]
    assert "layout template validation LLM" in out.loc[4, "dripper_warning"]
    assert "layout template validation failed" in out.loc[2, "dripper_warning"]
    assert "layout template validation failed" in out.loc[3, "dripper_warning"]


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


def test_layout_template_min_main_html_sim_forces_fallback_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class LowSimilarityLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
                "main_html_sim": 0.70,
            }

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=LowSimilarityLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_fallback_llm=True,
        layout_template_max_selected_item_ratio=1.0,
        layout_template_min_main_html_sim=0.80,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": ["https://example.test/1", "https://example.test/2"],
                "html": ["<p>representative</p>", "<p>sibling</p>"],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [True, False]
    assert out["dripper_layout_propagated"].tolist() == [False, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [False, True]
    assert "main_html_sim 0.700 below 0.800" in out.loc[1, "dripper_warning"]


def test_layout_template_stage_can_try_one_template_for_whole_host_before_dbscan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
        for index, sample in enumerate(samples):
            sample["layout_id"] = index % 2
        return samples, [0, 1]

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=base_webkit_bindings.layout_parser_cls,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_max_selected_item_ratio=1.0,
        layout_template_host_single_cluster_min_pages=4,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [f"https://example.test/{idx}" for idx in range(4)],
                "html": [f"<html>page {idx}</html>" for idx in range(4)],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 1
    assert out["dripper_layout_cluster"].nunique() == 1
    assert out["dripper_layout_representative"].tolist() == [True, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, True, True]


def test_layout_template_host_single_cluster_validation_failure_uses_dbscan_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            return {
                "html_element_dict": {"labels": typical_data["llm_response"]},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": "main:1",
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class TailDivergingLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            item_id = "2" if "tail-drift" in task_data["html_source"] else "1"
            return {
                "main_html_body": f"main:{item_id}",
                "main_html_success": True,
            }

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
        for sample in samples:
            sample["layout_id"] = -1 if "tail-drift" in sample["html"] else 0
        return samples, [0, -1]

    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FakeMapParser,
            layout_parser_cls=TailDivergingLayoutParser,
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
        layout_template_host_single_cluster_min_pages=4,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [f"https://example.test/{idx}" for idx in range(4)],
                "html": [
                    '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                    '<p _item_id="1">Sibling main</p><p _item_id="2">Sibling nav</p>',
                    '<p _item_id="1">Validation main</p><p _item_id="2">Validation nav</p>',
                    '<p _item_id="1">tail-drift main</p><p _item_id="2">tail-drift nav</p>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 3
    assert out["dripper_layout_representative"].tolist() == [True, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, False]
    assert out["dripper_layout_standalone_llm"].tolist() == [False, False, False, True]
    assert out["dripper_layout_fallback_llm"].tolist() == [False, False, True, False]
    assert out.loc[1, "dripper_html"] == "main:1"
    assert out.loc[2, "dripper_warning"].count("layout template validation LLM") == 1


def test_failed_host_single_cluster_can_split_fallback_by_url_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            response = typical_data["llm_response"]
            main_id = "2" if response.get("item_id 2") == 1 else "1"
            return {
                "html_element_dict": {"labels": response},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": f"main:{main_id}",
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class TemplateLabelLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            labels = task_data.get("labels") or task_data.get("html_element_dict", {}).get("labels", {})
            main_id = "2" if labels.get("item_id 2") == 1 else "1"
            return {
                "main_html_body": f"main:{main_id}",
                "main_html_success": True,
            }

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
        for sample in samples:
            sample["layout_id"] = 0
        return samples, [0]

    monkeypatch.setattr(stage_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FakeMapParser,
            layout_parser_cls=TemplateLabelLayoutParser,
        ),
    )
    client = PromptAwareClient()
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
        layout_template_host_single_cluster_min_pages=6,
        layout_template_failed_host_fallback_signature_mode="url_shape",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [
                    "https://example.test/a/1",
                    "https://example.test/a/2",
                    "https://example.test/a/3",
                    "https://example.test/b/1",
                    "https://example.test/b/2",
                    "https://example.test/b/3",
                ],
                "html": [
                    '<p _item_id="1">A rep</p><p _item_id="2">A nav</p>',
                    '<p _item_id="1">A sibling</p><p _item_id="2">A nav</p>',
                    '<p _item_id="1">A validation</p><p _item_id="2">A nav</p>',
                    '<p _item_id="1">B nav</p><p _item_id="2">B rep</p>',
                    '<p _item_id="1">B nav</p><p _item_id="2">B sibling</p>',
                    '<p _item_id="1">B nav</p><p _item_id="2">B validation</p>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) <= 6
    assert out["dripper_layout_cluster"].nunique() == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, False, True, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, False, True, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [False, False, True, False, False, True]
    assert out.loc[1, "dripper_html"] == "main:1"
    assert out.loc[4, "dripper_html"] == "main:2"


def test_failed_dbscan_layout_can_split_fallback_by_url_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, typical_data: dict) -> dict:
            response = typical_data["llm_response"]
            main_id = "2" if response.get("item_id 2") == 1 else "1"
            return {
                "html_element_dict": {"labels": response},
                "typical_dict_html": typical_data["typical_raw_tag_html"],
                "typical_main_html": f"main:{main_id}",
                "similarity_layer": 3,
                "typical_main_html_success": True,
            }

    class TemplateLabelLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            labels = task_data.get("labels") or task_data.get("html_element_dict", {}).get("labels", {})
            main_id = "2" if labels.get("item_id 2") == 1 else "1"
            return {
                "main_html_body": f"main:{main_id}",
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
            layout_parser_cls=TemplateLabelLayoutParser,
        ),
    )
    client = PromptAwareClient()
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
        layout_template_failed_layout_fallback_signature_mode="url_shape",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": [
                    "https://example.test/a/1",
                    "https://example.test/a/2",
                    "https://example.test/a/3",
                    "https://example.test/b/1",
                    "https://example.test/b/2",
                    "https://example.test/b/3",
                ],
                "html": [
                    '<p _item_id="1">A rep</p><p _item_id="2">A nav</p>',
                    '<p _item_id="1">A sibling</p><p _item_id="2">A nav</p>',
                    '<p _item_id="1">A validation</p><p _item_id="2">A nav</p>',
                    '<p _item_id="1">B nav</p><p _item_id="2">B rep</p>',
                    '<p _item_id="1">B nav</p><p _item_id="2">B sibling</p>',
                    '<p _item_id="1">B nav</p><p _item_id="2">B validation</p>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) <= 6
    assert out["dripper_layout_cluster"].nunique() == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, False, True, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, False, True, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [False, False, True, False, False, True]
    assert out.loc[1, "dripper_html"] == "main:1"
    assert out.loc[4, "dripper_html"] == "main:2"


def test_layout_template_stage_uses_feature_hash_for_large_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    def get_feature(html: str) -> dict[str, dict[int, list[str]]]:
        if "same-layout" in html:
            return {"tags": {1: ["body"], 2: ["article", "nav"]}, "attrs": {2: ["content"]}}
        return {"tags": {1: ["body"], 2: ["aside"]}, "attrs": {2: ["sidebar"]}}

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
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


def test_layout_template_stage_uses_dom_path_hash_for_large_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    def cluster_html_struct(samples: list[dict[str, Any]], threshold: float = 0.95) -> tuple[list[dict[str, Any]], list[int]]:  # noqa: ARG001
        raise AssertionError("dom_path_hash large-host mode should not call exact DBSCAN")

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=lambda html: {"tags": {1: ["body"], 2: ["main"]}},
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
        layout_template_large_host_mode="dom_path_hash",
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
                    '<html><body><main class="post-123"><h1>A</h1><p>rep</p></main></body></html>',
                    '<html><body><main class="post-456"><h1>B</h1><p>sibling one</p></main></body></html>',
                    '<html><body><main class="post-789"><p>different order</p><h1>C</h1></main></body></html>',
                    '<html><body><main class="post-999"><h1>D</h1><p>sibling two</p></main></body></html>',
                ],
            }
        ),
    )

    out = layout_stage.process(preprocess.process(batch)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, True]
    assert out["dripper_layout_standalone_llm"].tolist() == [False, False, True, False]


def test_layout_feature_fingerprint_is_order_insensitive() -> None:
    assert stage_mod._layout_feature_fingerprint(
        {"tags": {1: ["body"], 2: ["article", "nav", "article"]}, "attrs": {2: ["content", "main"]}}
    ) == stage_mod._layout_feature_fingerprint(
        {"attrs": {2: ["main", "content"]}, "tags": {2: ["nav", "article", "article"], 1: ["body"]}}
    )


def test_layout_dom_path_fingerprint_preserves_order_and_normalizes_dynamic_attrs() -> None:
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


def test_layout_template_stage_passes_more_noise_setting_to_layout_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()
    seen_more_noise: list[bool] = []

    class RecordingLayoutParser:
        def __init__(self, template_data: dict) -> None:  # noqa: ARG002
            pass

        def parse(self, task_data: dict) -> dict:
            seen_more_noise.append(bool(task_data["more_noise_enable"]))
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
            }

    monkeypatch.setattr(
        stage_mod,
        "_load_llm_web_kit_bindings",
        lambda: stage_mod._LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=RecordingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url")
    layout_stage = DripperHTMLLayoutTemplateStage(
        client=client,
        model_name="dripper",
        health_check=False,
        layout_template_more_noise_enable=True,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame(
            {
                "url": ["https://example.test/a", "https://example.test/b"],
                "html": ["<html>Rep</html>", "<html>Sibling</html>"],
            }
        ),
    )

    layout_stage.process(preprocess.process(batch))

    assert seen_more_noise == [True]


def test_stage_can_cap_request_max_tokens_from_item_count() -> None:
    client = RecordingAsyncClient(["1main"])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
        generation_config=GenerationConfig(max_tokens=2048, temperature=0.0, top_p=1.0),
        dynamic_max_tokens=True,
        dynamic_max_token_padding=12,
        dynamic_max_tokens_per_item=5,
        dynamic_min_max_tokens=32,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Hello</html>"]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert out.loc[0, "dripper_item_count"] == 1
    assert out.loc[0, "dripper_request_max_tokens"] == 32
    assert client.calls[0]["generation_config"].max_tokens == 32


def test_split_stage_applies_dynamic_request_max_tokens() -> None:
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        generation_config=GenerationConfig(max_tokens=2048, temperature=0.0, top_p=1.0),
        dynamic_max_tokens=True,
        dynamic_max_token_padding=12,
        dynamic_max_tokens_per_item=5,
        dynamic_min_max_tokens=32,
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(max_tokens=2048, temperature=0.0, top_p=1.0),
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Hello</html>"]}),
    )

    out = inference.process(preprocess.process(batch)).to_pandas()

    assert out.loc[0, "dripper_request_max_tokens"] == 32
    assert client.calls[0]["generation_config"].max_tokens == 32


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


def test_stage_adds_structured_output_regex_without_dropping_existing_extra_body() -> None:
    client = RecordingAsyncClient(["<answer>1main</answer>"])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
        generation_config=GenerationConfig(
            max_tokens=2048,
            extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
        structured_output_mode="structured_outputs",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Hello</html>"]}),
    )

    out = stage.process(batch).to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert client.calls[0]["generation_config"].extra_kwargs == {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"regex": r"<answer>\s*1(main|other)\s*</answer>"},
        }
    }


def test_split_inference_stage_adds_guided_regex_from_prompt_item_ids() -> None:
    client = RecordingAsyncClient(["<answer>1main</answer>"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        generation_config=GenerationConfig(max_tokens=2048),
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(max_tokens=2048),
        structured_output_mode="guided_regex",
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Hello</html>"]}),
    )

    out = inference.process(preprocess.process(batch)).to_pandas()

    assert out.loc[0, "dripper_response"] == "<answer>1main</answer>"
    assert client.calls[0]["generation_config"].extra_kwargs == {
        "extra_body": {"guided_regex": r"<answer>\s*1(main|other)\s*</answer>"}
    }


def test_stage_applies_mineru_fallback_after_parse_error() -> None:
    client = RecordingAsyncClient(["bad-response"])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Fallback</html>"]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert out.loc[0, "dripper_response"] == "bad-response"
    assert out.loc[0, "dripper_html"] == "<fallback><html>Fallback</html></fallback>"
    assert out.loc[0, "dripper_content"] == "mm_md:<fallback><html>Fallback</html></fallback>"
    assert out.loc[0, "dripper_error"] == ""
    assert "parse failed" in out.loc[0, "dripper_warning"]


def test_stage_skips_llm_when_simplified_html_has_no_item_ids() -> None:
    client = RecordingAsyncClient([])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>no-items</html>"]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert client.calls == []
    assert out.loc[0, "dripper_response"] == ""
    assert out.loc[0, "dripper_html"] == "<fallback><html>no-items</html></fallback>"
    assert out.loc[0, "dripper_content"] == "mm_md:<fallback><html>no-items</html></fallback>"
    assert out.loc[0, "dripper_inference_time_s"] == 0.0
    assert out.loc[0, "dripper_error"] == ""
    assert "no _item_id attributes" in out.loc[0, "dripper_warning"]


def test_stage_strips_xml_invalid_characters_before_conversion() -> None:
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
        data=pd.DataFrame({"html": ["<html>Bad\x00Char</html>"]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert "\x00" not in out.loc[0, "dripper_html"]
    assert out.loc[0, "dripper_html"] == "<article><html>BadChar</html></article>"


def test_stage_treats_empty_document_conversion_as_warning() -> None:
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
        data=pd.DataFrame({"html": ["<html>empty-main</html>"]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert "Document is empty" in out.loc[0, "dripper_warning"]
    assert out.loc[0, "dripper_content"] == ""


def test_stage_treats_empty_html_input_as_warning() -> None:
    client = RecordingAsyncClient([])
    stage = DripperHTMLExtractionStage(
        client=client,
        model_name="dripper",
        html_col="html",
        health_check=False,
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": [""]}),
    )

    result = stage.process(batch)
    out = result.to_pandas()

    assert client.calls == []
    assert out.loc[0, "dripper_error"] == ""
    assert out.loc[0, "dripper_warning"] == "empty HTML input"
    assert out.loc[0, "dripper_content"] == ""


def test_stage_decodes_bytes_even_when_charset_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_mod, "_decode_html_bytes", lambda html_bytes: None)
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
