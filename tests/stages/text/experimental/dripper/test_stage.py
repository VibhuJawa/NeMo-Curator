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

"""Unit tests for the Dripper HTML extraction stages.

NOTE: This file was rewritten for the 2-phase (clustering -> plan -> inference ->
finalize -> inference -> postprocess) Dripper architecture after the monolithic
``stage.py`` was split into ``.../dripper/stages/*.py``.  It is verified to
``py_compile`` and ``ruff`` clean, but the host it was edited on does not have the
heavy runtime deps (torch/cuml/pandas/loguru/mineru_html/llm_web_kit), so the
assertions still need a ``pytest`` run in a full-deps environment to confirm.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.text.experimental.dripper.stages import (
    clustering as clustering_mod,
)
from nemo_curator.stages.text.experimental.dripper.stages import (
    layout_finalize as layout_finalize_mod,
)
from nemo_curator.stages.text.experimental.dripper.stages import (
    postprocess as postprocess_mod,
)
from nemo_curator.stages.text.experimental.dripper.stages import (
    preprocess as preprocess_mod,
)
from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
    _LLMWebKitBindings,
    _MinerUHTMLBindings,
)
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import (
    _json_safe_layout_mapping_data,
    _layout_page_signature_key,
    _layout_page_signature_key_with_low_card_queries,
    _low_card_query_value_keys,
    _select_validation_indexes,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
)
from nemo_curator.stages.text.experimental.dripper.stages.clustering import (
    DripperHTMLLayoutClusteringStage,
)
from nemo_curator.stages.text.experimental.dripper.stages.inference import (
    DripperHTMLInferenceStage,
)
from nemo_curator.stages.text.experimental.dripper.stages.layout_finalize import (
    DripperHTMLLayoutFinalizeStage,
)
from nemo_curator.stages.text.experimental.dripper.stages.layout_plan import (
    DripperHTMLLayoutPlanStage,
)
from nemo_curator.stages.text.experimental.dripper.stages.postprocess import (
    DripperHTMLPostprocessStage,
)
from nemo_curator.stages.text.experimental.dripper.stages.preprocess import (
    DripperHTMLPreprocessStage,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


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


def make_bindings() -> _MinerUHTMLBindings:  # noqa: C901
    def simplify_single_input(case: FakeCase) -> FakeCase:
        if "preprocess-fails" in case.input_data.raw_html:
            msg = "preprocess failed"
            raise RuntimeError(msg)
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
            msg = "parse failed"
            raise RuntimeError(msg)
        case.parse_result = SimpleNamespace(item_label={"1": "main"})
        return case

    def extract_main_html_single(case: FakeCase) -> FakeCase:
        main_html = (
            "" if "empty-main" in case.input_data.raw_html else f"<article>{case.input_data.raw_html}</article>"
        )
        case.output_data = FakeOutput(main_html=main_html)
        return case

    def extract_main_html_fallback(case: FakeCase, fallback_handler: object) -> FakeCase:  # noqa: ARG001
        main_html = (
            "" if "empty-main" in case.input_data.raw_html else f"<fallback>{case.input_data.raw_html}</fallback>"
        )
        case.output_data = FakeOutput(main_html=main_html)
        return case

    def convert2content(case: FakeCase, output_format: str) -> FakeCase:
        if not case.output_data.main_html:
            msg = "ExtractorChain base exception#Error during extraction: Document is empty"
            raise RuntimeError(msg)
        case.output_data.main_content = f"{output_format}:{case.output_data.main_html}"
        return case

    return _MinerUHTMLBindings(
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


def make_label_aware_bindings() -> _MinerUHTMLBindings:
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

    return _MinerUHTMLBindings(
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


def make_llm_web_kit_bindings() -> _LLMWebKitBindings:
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
        samples: list[dict[str, Any]],
        threshold: float = 0.95,  # noqa: ARG001
    ) -> tuple[list[dict[str, Any]], list[int]]:
        for sample in samples:
            sample["layout_id"] = 0
        return samples, [0]

    def select_representative_html(candidates: list[dict[str, str]]) -> dict[str, str] | None:
        return candidates[0] if candidates else None

    return _LLMWebKitBindings(
        get_feature=lambda html: {"tags": {1: ["body"], 2: [html]}},
        cluster_html_struct=cluster_html_struct,
        select_representative_html=select_representative_html,
        map_parser_cls=FakeMapParser,
        layout_parser_cls=FakeLayoutParser,
    )


# ---------------------------------------------------------------------------
# 2-phase helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_mineru_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the MinerU loader in every module that imports it.

    ``_load_mineru_html_bindings`` is imported into each consuming stage module's
    namespace, so the patch must target each one rather than a single shared module.
    """
    monkeypatch.setattr(preprocess_mod, "_load_mineru_html_bindings", make_bindings)
    monkeypatch.setattr(postprocess_mod, "_load_mineru_html_bindings", make_bindings)
    monkeypatch.setattr(layout_finalize_mod, "_load_mineru_html_bindings", make_bindings)


def _patch_web_bindings(monkeypatch: pytest.MonkeyPatch, factory: Callable[[], _LLMWebKitBindings]) -> None:
    """Patch the llm-webkit loader in every module that imports it."""
    monkeypatch.setattr(clustering_mod, "_load_llm_web_kit_bindings", factory)
    monkeypatch.setattr(layout_finalize_mod, "_load_llm_web_kit_bindings", factory)


def _add_precomputed_layout(df: pd.DataFrame, layout_id_col: str = "dripper_layout_id") -> pd.DataFrame:
    """Mimic DripperHTMLLayoutClusteringStage output: one layout per host.

    The 2-phase plan/finalize stages NEVER re-cluster; they consume a precomputed
    layout-id column.  These tests bypass the GPU clustering stage by writing a
    single stable layout id per (host) so the whole batch forms one layout group.
    """
    out = df.copy()
    if "url_host_name" in out.columns:
        hosts = out["url_host_name"].astype(str)
    elif "url" in out.columns:
        hosts = out["url"].astype(str).str.replace(r"^https?://", "", regex=True).str.split("/").str[0]
    else:
        hosts = pd.Series(["host"] * len(out), index=out.index)
    out[layout_id_col] = [f"layout-{host}-0" for host in hosts]
    return out


def _run_two_phase(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    client: AsyncLLMClient,
    plan_kwargs: dict[str, Any] | None = None,
    finalize_kwargs: dict[str, Any] | None = None,
    preprocess_kwargs: dict[str, Any] | None = None,
    layout_id_col: str = "dripper_layout_id",
    run_postprocess: bool = True,
    second_inference: bool = True,
) -> pd.DataFrame:
    """Drive preprocess -> [precompute layout] -> plan -> inference -> finalize -> inference -> postprocess."""
    generation_config = GenerationConfig(max_tokens=2048)
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        url_col="url",
        prompt_version="short_compact",
        generation_config=generation_config,
        **(preprocess_kwargs or {}),
    )
    plan = DripperHTMLLayoutPlanStage(
        generation_config=generation_config,
        layout_id_col=layout_id_col,
        **(plan_kwargs or {}),
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        generation_config=generation_config,
        health_check=False,
    )
    finalize = DripperHTMLLayoutFinalizeStage(
        generation_config=generation_config,
        layout_id_col=layout_id_col,
        **(finalize_kwargs or {}),
    )
    batch = DocumentBatch(task_id="task-1", dataset_name="test", data=df)
    preprocessed = preprocess.process(batch).to_pandas()
    preprocessed = _add_precomputed_layout(preprocessed, layout_id_col)
    planned = plan.process(DocumentBatch(task_id="task-1", dataset_name="test", data=preprocessed)).to_pandas()
    first = inference.process(DocumentBatch(task_id="task-1", dataset_name="test", data=planned)).to_pandas()
    finalized = finalize.process(DocumentBatch(task_id="task-1", dataset_name="test", data=first)).to_pandas()
    current = finalized
    if second_inference:
        current = inference.process(DocumentBatch(task_id="task-1", dataset_name="test", data=current)).to_pandas()
    if run_postprocess:
        postprocess = DripperHTMLPostprocessStage(html_col="html", url_col="url")
        current = postprocess.process(DocumentBatch(task_id="task-1", dataset_name="test", data=current)).to_pandas()
    return current


# ---------------------------------------------------------------------------
# Module-level helper tests (KEPT: just retargeted imports)
# ---------------------------------------------------------------------------


def test_layout_template_propagation_concurrency_defaults_to_single_worker() -> None:
    assert DripperHTMLLayoutPlanStage().layout_template_propagation_concurrency == 1
    assert DripperHTMLLayoutFinalizeStage().layout_template_propagation_concurrency == 1


def test_layout_mapping_data_stringifies_tuple_keys_for_ray_boundary() -> None:
    mapping_data = {
        "html_element_dict": {
            0: {
                ("html", None, None, "id0", 0, 0): (
                    "red",
                    ("root", None, None),
                    "/html",
                    False,
                )
            }
        },
        "typical_dict_html": "<html></html>",
    }

    safe = _json_safe_layout_mapping_data(mapping_data)

    assert isinstance(safe["html_element_dict"], str)
    assert "('html', None, None, 'id0', 0, 0)" in safe["html_element_dict"]
    assert safe["typical_dict_html"] == "<html></html>"


def test_layout_template_validation_indexes_are_spread_across_cluster() -> None:
    df = pd.DataFrame(
        {
            "url": [f"https://example.test/{idx}" for idx in range(10)],
            "dripper_item_count": list(range(10)),
        }
    )

    assert _select_validation_indexes(df, [], 2, "url", "dripper_item_count") == []
    assert _select_validation_indexes(df, [1, 2, 3, 4], 0, "url", "dripper_item_count") == []
    assert _select_validation_indexes(df, [1, 2, 3, 4], 1, "url", "dripper_item_count") == [4]
    assert _select_validation_indexes(df, [1, 2, 3, 4], 2, "url", "dripper_item_count") == [1, 4]
    assert _select_validation_indexes(df, [1, 2, 3, 4], 3, "url", "dripper_item_count") == [1, 3, 4]
    assert _select_validation_indexes(df, [1, 2], 5, "url", "dripper_item_count") == [1, 2]
    assert _select_validation_indexes(df, list(range(10)), 4, "url", "dripper_item_count") == [
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

    assert _select_validation_indexes(df, list(range(6)), 4, "url", "dripper_item_count") == [
        0,
        2,
        3,
        5,
    ]


def test_layout_page_signature_key_splits_query_and_numeric_article_shapes() -> None:
    assert (
        _layout_page_signature_key(
            "https://example.test/archive.html?start=10",
            42,
            "url_shape",
        )
        == "url=path=archive.html|q=start"
    )
    assert (
        _layout_page_signature_key(
            "https://example.test/news/123-first.html",
            42,
            "url_shape",
        )
        == "url=path=news/#num.html|q="
    )
    assert _layout_page_signature_key("https://example.test/a", 42, "item_count_bucket") == "items=33-64"
    assert (
        _layout_page_signature_key(
            "https://example.test/news/123-first.html",
            42,
            "url_shape_item_count_bucket",
        )
        == "url=path=news/#num.html|q=|items=33-64"
    )


def test_layout_page_signature_key_semantic_shape_preserves_content_url_tokens() -> None:
    assert _layout_page_signature_key(
        "https://wits.worldbank.org/CountryProfile/en/Compare/Country/ABW/Indicator/MPRT-TRD-VL/"
        "partner/WLD/product/UNCTAD-SoP1/region/LCN/show/line",
        42,
        "url_semantic_shape",
    ) != _layout_page_signature_key(
        "https://wits.worldbank.org/CountryProfile/en/Compare/Country/ABW/Indicator/MPRT-TRD-VL/"
        "partner/WLD/product/UNCTAD-SoP3/region/LCN/show/line",
        42,
        "url_semantic_shape",
    )
    assert _layout_page_signature_key(
        "https://source.android.com/?authuser=0&hl=es-419",
        42,
        "url_semantic_shape",
    ) != _layout_page_signature_key(
        "https://source.android.com/?authuser=0&hl=pl",
        42,
        "url_semantic_shape",
    )
    assert (
        _layout_page_signature_key(
            "https://example.test/news/123-first.html",
            42,
            "url_semantic_shape_item_count_bucket",
        )
        == "url=path=news/123-first.html|q=|items=33-64"
    )


def test_semantic_exact_query_shape_preserves_configured_query_values() -> None:
    assert (
        _layout_page_signature_key(
            "https://publicpay.test/Reports/Cities/City.aspx?entityid=100&year=2012&rpt=3",
            55,
            "url_semantic_exact_query_shape_item_count_exact",
        )
        == "url=path=reports/cities/city.aspx|q=entityid=100,rpt,year|items=55"
    )
    assert _layout_page_signature_key(
        "https://publicpay.test/Reports/Cities/City.aspx?entityid=100&year=2012&rpt=3",
        55,
        "url_semantic_exact_query_shape_item_count_exact",
    ) != _layout_page_signature_key(
        "https://publicpay.test/Reports/Cities/City.aspx?entityid=101&year=2012&rpt=3",
        55,
        "url_semantic_exact_query_shape_item_count_exact",
    )
    assert _layout_page_signature_key(
        "https://source.android.com/?authuser=0&hl=es-419",
        42,
        "url_semantic_exact_query_shape_item_count_exact",
    ) != _layout_page_signature_key(
        "https://source.android.com/?authuser=0&hl=pl",
        42,
        "url_semantic_exact_query_shape_item_count_exact",
    )


def test_low_card_query_shape_preserves_repeated_query_values_only() -> None:
    urls = [
        f"https://publicpay.test/Reports/Cities/City.aspx?entityid={100 + idx}&year={2012 + idx % 2}&rpt={idx % 3}"
        for idx in range(20)
    ]
    low_card_keys = _low_card_query_value_keys(urls)

    assert low_card_keys == {"rpt", "year"}

    signature = _layout_page_signature_key_with_low_card_queries(
        urls[0],
        55,
        "url_low_card_query_shape_item_count_exact",
        low_card_keys,
    )

    assert signature == "url=path=reports/cities/city.aspx|q=entityid,rpt=0,year=2012|items=55"


def test_semantic_low_card_query_shape_preserves_low_card_and_exact_query_values() -> None:
    urls = [
        f"https://publicpay.test/Reports/Cities/City.aspx?entityid={100 + idx}&year={2012 + idx % 2}&rpt={idx % 3}"
        for idx in range(20)
    ]
    low_card_keys = _low_card_query_value_keys(urls)

    signature = _layout_page_signature_key_with_low_card_queries(
        urls[0],
        55,
        "url_semantic_low_card_query_shape_item_count_exact",
        low_card_keys,
        exact_query_value_keys={"entityid", "id"},
    )

    assert low_card_keys == {"rpt", "year"}
    assert signature == "url=path=reports/cities/city.aspx|q=entityid=100,rpt=0,year=2012|items=55"
    assert _layout_page_signature_key_with_low_card_queries(
        "https://www.ncbi.nlm.nih.gov/cdd/PF00474",
        55,
        "url_semantic_low_card_query_shape_item_count_exact",
        set(),
    ) != _layout_page_signature_key_with_low_card_queries(
        "https://www.ncbi.nlm.nih.gov/cdd/PF00802",
        55,
        "url_semantic_low_card_query_shape_item_count_exact",
        set(),
    )


def test_low_card_query_shape_uses_exact_values_when_all_query_values_are_high_card() -> None:
    urls = [f"https://scop.test/astral/jmolview?context={idx}&id={1000 + idx}&ver={idx}" for idx in range(20)]
    low_card_keys = _low_card_query_value_keys(urls)

    assert low_card_keys == set()
    assert (
        _layout_page_signature_key_with_low_card_queries(
            urls[0],
            55,
            "url_low_card_query_shape_item_count_exact",
            low_card_keys,
        )
        == "url=path=astral/jmolview|q=context=0,id=1000,ver=0|items=55"
    )


def test_low_card_query_shape_keeps_id_exact_when_other_query_keys_are_low_card() -> None:
    urls = [
        f"https://scop.test/astral/jmolview?context={idx % 2}&id=d{idx:04d}&ver={1 + idx % 2}.55" for idx in range(20)
    ]
    low_card_keys = _low_card_query_value_keys(urls)

    assert low_card_keys == {"context", "ver"}
    assert (
        _layout_page_signature_key_with_low_card_queries(
            urls[0],
            5,
            "url_low_card_query_shape_item_count_exact",
            low_card_keys,
        )
        == "url=path=astral/jmolview|q=context=0,id=d0000,ver=1.55|items=5"
    )


def test_exact_query_shape_keeps_id_like_query_values_only() -> None:
    assert (
        _layout_page_signature_key(
            "https://publicpay.test/Reports/Cities/City.aspx?entityid=100&year=2012&rpt=3",
            55,
            "url_exact_query_shape_item_count_exact",
        )
        == "url=path=reports/cities/city.aspx|q=entityid=100,rpt,year|items=55"
    )
    assert _layout_page_signature_key(
        "https://publicpay.test/Reports/Cities/City.aspx?entityid=100&year=2012&rpt=3",
        55,
        "url_exact_query_shape_item_count_exact",
    ) == _layout_page_signature_key(
        "https://publicpay.test/Reports/Cities/City.aspx?entityid=100&year=2020&rpt=9",
        55,
        "url_exact_query_shape_item_count_exact",
    )
    assert _layout_page_signature_key(
        "https://publicpay.test/Reports/Cities/City.aspx?entityid=100&year=2012&rpt=3",
        55,
        "url_exact_query_shape_item_count_exact",
    ) != _layout_page_signature_key(
        "https://publicpay.test/Reports/Cities/City.aspx?entityid=101&year=2012&rpt=3",
        55,
        "url_exact_query_shape_item_count_exact",
    )


def test_exact_query_shape_can_preserve_configured_language_keys() -> None:
    default_a = _layout_page_signature_key(
        "https://source.android.com/?authuser=0&hl=es",
        59,
        "url_exact_query_shape_item_count_exact",
    )
    default_b = _layout_page_signature_key(
        "https://source.android.com/?authuser=4&hl=pl",
        59,
        "url_exact_query_shape_item_count_exact",
    )
    configured_a = _layout_page_signature_key(
        "https://source.android.com/?authuser=0&hl=es",
        59,
        "url_exact_query_shape_item_count_exact",
        exact_query_value_keys={"authuser", "entityid", "hl", "id"},
    )
    configured_b = _layout_page_signature_key(
        "https://source.android.com/?authuser=4&hl=pl",
        59,
        "url_exact_query_shape_item_count_exact",
        exact_query_value_keys={"authuser", "entityid", "hl", "id"},
    )

    assert default_a == default_b
    assert configured_a == "url=path=|q=authuser=0,hl=es|items=59"
    assert configured_b == "url=path=|q=authuser=4,hl=pl|items=59"
    assert configured_a != configured_b


# ---------------------------------------------------------------------------
# Layout-template base-stage helper tests
# ---------------------------------------------------------------------------


def test_layout_template_stage_uses_extra_validation_rows_for_large_clusters() -> None:
    stage = DripperHTMLLayoutPlanStage(
        layout_template_validation_rows=2,
        layout_template_large_cluster_validation_rows=8,
        layout_template_large_cluster_min_size=64,
    )

    assert stage._effective_validation_rows(63) == 2
    assert stage._effective_validation_rows(64) == 8


def test_layout_template_stage_uses_precomputed_layout_id_column() -> None:
    stage = DripperHTMLLayoutPlanStage(
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
            _DRIPPER_NEEDS_LLM_COL: [True, True, True, True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [(plan.host_key, plan.source, plan.indexes) for plan in plans] == [
        ("a.example", "precomputed_layout:a.example_0", [0, 1]),
        ("a.example", "precomputed_layout:a.example_1", [2, 3]),
        ("b.example", "precomputed_layout:a.example_0", [5, 6]),
    ]


def test_layout_template_stage_skips_noise_layout_ids_in_precomputed_column() -> None:
    stage = DripperHTMLLayoutPlanStage(
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
                "https://a.example/5",
            ],
            "url_host_name": ["a.example"] * 5,
            "dripper_layout_id": [
                "a.example_-1",
                "a.example_-1",
                "a.example_-1",
                "a.example_1",
                "a.example_1",
            ],
            "html": ["<p>a</p>", "<p>b</p>", "<p>c</p>", "<p>d</p>", "<p>e</p>"],
            _DRIPPER_NEEDS_LLM_COL: [True, True, True, True, True],
        }
    )

    plans = stage._build_layout_group_plans(df)

    assert [(plan.source, plan.indexes) for plan in plans] == [
        ("precomputed_layout:a.example_1", [3, 4]),
    ]


def test_layout_template_stage_requires_precomputed_layout_column() -> None:
    stage = DripperHTMLLayoutPlanStage(host_col="url_host_name", layout_id_col="dripper_layout_id")
    stage._web_bindings = make_llm_web_kit_bindings()
    df = pd.DataFrame(
        {
            "url": ["https://a.example/1", "https://a.example/2"],
            "url_host_name": ["a.example", "a.example"],
            "html": ["<p>a</p>", "<p>b</p>"],
            _DRIPPER_NEEDS_LLM_COL: [True, True],
        }
    )

    with pytest.raises(RuntimeError, match="DripperHTMLLayoutClusteringStage"):
        stage._build_layout_group_plans(df)


# ---------------------------------------------------------------------------
# Clustering-stage tests
# ---------------------------------------------------------------------------


def test_layout_clustering_stage_precomputes_host_bounded_layout_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clustering_mod, "_load_llm_web_kit_bindings", make_llm_web_kit_bindings)
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


def test_layout_precompute_path_derives_simpled_features_from_raw_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clustering_mod, "_load_llm_web_kit_bindings", make_llm_web_kit_bindings)
    df = pd.DataFrame(
        {
            "url": [
                "https://a.example/article/1",
                "https://a.example/article/2",
                "https://b.example/article/1",
                "https://b.example/article/2",
            ],
            "html": ["<html>one</html>", "<html>two</html>", "<html>three</html>", "<html>four</html>"],
        }
    )

    batch = DripperHTMLPreprocessStage().process(DocumentBatch(task_id="task", dataset_name="test", data=df))
    out = (
        DripperHTMLLayoutClusteringStage(
            host_col=None,
            layout_feature_source="simpled_html",
        )
        .process(batch)
        .to_pandas()
    )

    assert "url_host_name" not in out
    assert out.loc[0, "dripper_layout_id"]
    assert out.loc[0, "dripper_layout_id"] == out.loc[1, "dripper_layout_id"]
    assert out.loc[2, "dripper_layout_id"]
    assert out.loc[2, "dripper_layout_id"] == out.loc[3, "dripper_layout_id"]
    assert out.loc[0, "dripper_layout_id"] != out.loc[2, "dripper_layout_id"]


def test_layout_clustering_stage_filters_dbscan_group_by_exemplar_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webkit_bindings = make_llm_web_kit_bindings()
    monkeypatch.setattr(
        clustering_mod,
        "_load_llm_web_kit_bindings",
        lambda: _LLMWebKitBindings(
            get_feature=webkit_bindings.get_feature,
            cluster_html_struct=webkit_bindings.cluster_html_struct,
            select_representative_html=webkit_bindings.select_representative_html,
            map_parser_cls=webkit_bindings.map_parser_cls,
            layout_parser_cls=webkit_bindings.layout_parser_cls,
            similarity=lambda left, right, _max_layer_n: 1.0 if left == right else 0.0,
        ),
    )
    stage = DripperHTMLLayoutClusteringStage(host_col=None)
    df = pd.DataFrame(
        {
            "url": [f"https://example.test/{idx}" for idx in range(4)],
            "html": ["<p>a</p>", "<p>a</p>", "<p>a</p>", "<p>d</p>"],
        }
    )

    out = stage.process(DocumentBatch(task_id="task", dataset_name="test", data=df)).to_pandas()

    # The exemplar-similarity reassignment keeps rows that match the exemplar feature
    # and drops the diverging tail (its layout id stays empty).
    assert out.loc[0, "dripper_layout_id"]
    assert out.loc[0, "dripper_layout_id"] == out.loc[1, "dripper_layout_id"] == out.loc[2, "dripper_layout_id"]
    assert out.loc[3, "dripper_layout_id"] == ""


# ---------------------------------------------------------------------------
# Single-stage inference / extraction tests (CONVERTED to split stages)
# ---------------------------------------------------------------------------


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
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == "dripper"
    assert client.calls[0]["messages"] == [
        {"role": "user", "content": 'short_compact:<main _item_id="1"><html>Hello</html></main>'}
    ]


def test_inference_stage_threads_generation_config_extra_kwargs() -> None:
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        generation_config=GenerationConfig(max_tokens=2048),
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(
            max_tokens=2048,
            extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Hello</html>"]}),
    )

    inference.process(preprocess.process(batch))

    assert client.calls[0]["generation_config"].extra_kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


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

    assert out.loc[0, "dripper_item_count"] == 1
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


def test_split_inference_stage_adds_structured_output_regex_without_dropping_extra_body() -> None:
    client = RecordingAsyncClient(["<answer>1main</answer>"])
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        generation_config=GenerationConfig(max_tokens=2048),
    )
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        health_check=False,
        generation_config=GenerationConfig(
            max_tokens=2048,
            extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
        structured_output_mode="structured_outputs",
    )
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Hello</html>"]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

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


def test_split_stages_apply_mineru_fallback_after_parse_error() -> None:
    client = RecordingAsyncClient(["bad-response"])
    preprocess = DripperHTMLPreprocessStage(html_col="html")
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Fallback</html>"]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

    assert out.loc[0, "dripper_response"] == "bad-response"
    assert out.loc[0, "dripper_html"] == "<fallback><html>Fallback</html></fallback>"
    assert out.loc[0, "dripper_content"] == "mm_md:<fallback><html>Fallback</html></fallback>"
    assert out.loc[0, "dripper_error"] == ""
    assert "parse failed" in out.loc[0, "dripper_warning"]


def test_split_stages_skip_llm_when_simplified_html_has_no_item_ids() -> None:
    client = RecordingAsyncClient([])
    preprocess = DripperHTMLPreprocessStage(html_col="html")
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>no-items</html>"]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

    assert client.calls == []
    assert out.loc[0, "dripper_response"] == ""
    assert out.loc[0, "dripper_html"] == "<fallback><html>no-items</html></fallback>"
    assert out.loc[0, "dripper_content"] == "mm_md:<fallback><html>no-items</html></fallback>"
    assert out.loc[0, "dripper_inference_time_s"] == 0.0
    assert out.loc[0, "dripper_error"] == ""
    assert "no _item_id attributes" in out.loc[0, "dripper_warning"]


def test_split_stages_strip_xml_invalid_characters_before_conversion() -> None:
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html")
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>Bad\x00Char</html>"]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert "\x00" not in out.loc[0, "dripper_html"]
    assert out.loc[0, "dripper_html"] == "<article><html>BadChar</html></article>"


def test_split_stages_treat_empty_document_conversion_as_warning() -> None:
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html")
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": ["<html>empty-main</html>"]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert "Document is empty" in out.loc[0, "dripper_warning"]
    assert out.loc[0, "dripper_content"] == ""


def test_split_stages_treat_empty_html_input_as_warning() -> None:
    client = RecordingAsyncClient([])
    preprocess = DripperHTMLPreprocessStage(html_col="html")
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": [""]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

    assert client.calls == []
    assert out.loc[0, "dripper_error"] == ""
    assert out.loc[0, "dripper_warning"] == "empty HTML input"
    assert out.loc[0, "dripper_content"] == ""


def test_split_stages_decode_bytes_even_when_charset_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preprocess_mod, "_decode_html_bytes", lambda _html_bytes: None)
    monkeypatch.setattr(postprocess_mod, "_decode_html_bytes", lambda _html_bytes: None, raising=False)
    client = RecordingAsyncClient(["1main"])
    preprocess = DripperHTMLPreprocessStage(html_col="html")
    inference = DripperHTMLInferenceStage(client=client, model_name="dripper", health_check=False)
    postprocess = DripperHTMLPostprocessStage(html_col="html")
    batch = DocumentBatch(
        task_id="task-1",
        dataset_name="test",
        data=pd.DataFrame({"html": [b"<html>Bad\xffByte</html>"]}),
    )

    out = postprocess.process(inference.process(preprocess.process(batch))).to_pandas()

    assert out.loc[0, "dripper_error"] == ""
    assert "Bad" in out.loc[0, "dripper_html"]
    assert client.calls


def test_preprocess_setup_reports_missing_mineru_html(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_bindings() -> _MinerUHTMLBindings:
        msg = "missing mineru"
        raise RuntimeError(msg)

    monkeypatch.setattr(preprocess_mod, "_load_mineru_html_bindings", missing_bindings)
    stage = DripperHTMLPreprocessStage(html_col="html")

    with pytest.raises(RuntimeError, match="missing mineru"):
        stage.setup()


# ---------------------------------------------------------------------------
# 2-phase layout-template tests (CONVERTED from single-stage process())
# ---------------------------------------------------------------------------


def test_layout_template_infers_representative_and_propagates_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_web_bindings(monkeypatch, make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main"])
    df = pd.DataFrame(
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
    )

    out = _run_two_phase(df, client=client)

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


def test_split_external_layout_template_pipeline_batches_representative_and_validation_then_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_web_bindings(monkeypatch, make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main", "1main"])
    generation_config = GenerationConfig(max_tokens=2048)
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        url_col="url",
        prompt_version="short_compact",
        generation_config=generation_config,
    )
    plan = DripperHTMLLayoutPlanStage(
        generation_config=generation_config,
        layout_template_validation_rows=1,
        layout_template_validation_min_content_f1=0.5,
    )
    first_inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        generation_config=generation_config,
        health_check=False,
    )
    finalize = DripperHTMLLayoutFinalizeStage(
        generation_config=generation_config,
        layout_template_validation_rows=1,
        layout_template_validation_min_content_f1=0.5,
    )
    second_inference = DripperHTMLInferenceStage(
        client=client,
        model_name="dripper",
        generation_config=generation_config,
        health_check=False,
    )
    postprocess = DripperHTMLPostprocessStage(html_col="html", url_col="url")
    raw = pd.DataFrame(
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
    )
    preprocessed = preprocess.process(DocumentBatch(task_id="task-1", dataset_name="test", data=raw)).to_pandas()
    preprocessed = _add_precomputed_layout(preprocessed)

    planned = plan.process(DocumentBatch(task_id="task-1", dataset_name="test", data=preprocessed)).to_pandas()
    assert planned["dripper_layout_representative"].sum() == 1
    assert planned["dripper_layout_validation_llm"].sum() == 1
    assert planned[_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL].sum() == 1
    assert planned[_DRIPPER_NEEDS_LLM_COL].sum() == 2

    first = first_inference.process(DocumentBatch(task_id="task-1", dataset_name="test", data=planned)).to_pandas()
    finalized = finalize.process(DocumentBatch(task_id="task-1", dataset_name="test", data=first)).to_pandas()

    assert finalized["dripper_layout_representative"].sum() == 1
    assert finalized["dripper_layout_validation_llm"].sum() == 1
    assert finalized["dripper_layout_propagation_success"].sum() == 1
    assert finalized[_DRIPPER_NEEDS_LLM_COL].sum() == 0
    assert finalized[_DRIPPER_LAYOUT_FINALIZED_COL].all()

    final = postprocess.process(
        second_inference.process(DocumentBatch(task_id="task-1", dataset_name="test", data=finalized))
    ).to_pandas()

    assert len(client.calls) == 2
    assert final["dripper_layout_representative"].sum() == 1
    assert final["dripper_layout_validation_llm"].sum() == 1
    assert final["dripper_layout_propagation_success"].sum() == 1
    assert sorted(final["dripper_html"].tolist()) == sorted(
        [
            "<article><html>Rep</html></article>",
            "<article><html>Sibling Two</html></article>",
            "<propagated><html>Sibling One</html></propagated>",
        ]
    )


def test_layout_template_retries_representative_candidates_after_mapping_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class RetryMapParser:
        def __init__(self, template_data: dict) -> None:
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

    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=RetryMapParser,
            layout_parser_cls=base_webkit_bindings.layout_parser_cls,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main", "1main", "1main"])
    df = pd.DataFrame(
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
    )

    # With a precomputed single-layout group, the representative (first index) maps to
    # a failed template; the finalize stage defers the pending siblings to the second
    # inference pass, where they are extracted directly via fallback.
    out = _run_two_phase(df, client=client)

    assert out["dripper_layout_representative"].tolist() == [True, False, False, False]
    assert out.loc[1, "dripper_html"]
    assert "typical_main_html_success=false" in out.loc[0, _DRIPPER_PRIMARY_ERROR_COL]


def test_layout_template_uses_prompt_dedup_fallback_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_web_bindings(monkeypatch, make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main", "1main", "1main", "1main"])
    df = pd.DataFrame(
        {
            "url": [
                "https://example.test/a",
                "https://example.test/b",
                "https://example.test/c",
                "https://example.test/d",
            ],
            "html": [
                "<html>Duplicate page</html>",
                "<html>Duplicate page</html>",
                "<html>Duplicate page</html>",
                "<html>Duplicate page</html>",
            ],
        }
    )

    out = _run_two_phase(
        df,
        client=client,
        plan_kwargs={"layout_template_prompt_dedup_fallback_min_fraction": 0.5},
        finalize_kwargs={"layout_template_prompt_dedup_fallback_min_fraction": 0.5},
    )

    # The plan-stage prompt-dedup guard routes every row straight to fallback LLM
    # instead of template propagation.
    assert out["dripper_layout_representative"].tolist() == [False, False, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, False, False, False]
    assert out["dripper_layout_fallback_llm"].tolist() == [True, True, True, True]


def test_layout_plan_uses_low_return_fallback_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_web_bindings(monkeypatch, make_llm_web_kit_bindings)
    df = pd.DataFrame(
        {
            "url": [
                "https://example.test/a",
                "https://example.test/b",
                "https://example.test/c",
            ],
            "html": [
                "<html>Page A</html>",
                "<html>Page B</html>",
                "<html>Page C</html>",
            ],
        }
    )
    generation_config = GenerationConfig(max_tokens=2048)
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url", generation_config=generation_config)
    plan = DripperHTMLLayoutPlanStage(
        generation_config=generation_config,
        layout_template_validation_rows=2,
        layout_template_min_saved_call_pages=1,
    )
    preprocessed = preprocess.process(DocumentBatch(task_id="task-1", dataset_name="test", data=df)).to_pandas()
    preprocessed = _add_precomputed_layout(preprocessed)
    planned = plan.process(DocumentBatch(task_id="task-1", dataset_name="test", data=preprocessed)).to_pandas()

    # With 3 rows and 2 validation rows, no propagation call is saved, so the plan
    # stage forces fallback LLM on every row and the warning identifies the guard.
    assert planned["dripper_layout_representative"].tolist() == [False, False, False]
    assert planned["dripper_layout_fallback_llm"].tolist() == [True, True, True]
    assert planned[_DRIPPER_NEEDS_LLM_COL].tolist() == [True, True, True]


def test_layout_template_converts_propagated_item_ids_through_mineru(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMapParser:
        def __init__(self, template_data: dict) -> None:
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
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": '<article _item_id="2">Sibling main</article>',
                "main_html_success": True,
            }

    monkeypatch.setattr(preprocess_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(postprocess_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(layout_finalize_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=lambda html: {"tags": {1: ["body"], 2: [html]}},
            cluster_html_struct=make_llm_web_kit_bindings().cluster_html_struct,
            select_representative_html=lambda candidates: candidates[0],
            map_parser_cls=FakeMapParser,
            layout_parser_cls=FakeLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    df = pd.DataFrame(
        {
            "url": ["https://example.test/a", "https://example.test/b"],
            "html": [
                '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                '<p _item_id="2">Sibling main</p><p _item_id="3">Sibling nav</p>',
            ],
        }
    )

    out = _run_two_phase(
        df,
        client=client,
        plan_kwargs={"layout_template_propagation_target": "mapped_item_ids"},
        finalize_kwargs={"layout_template_propagation_target": "mapped_item_ids"},
    )

    assert len(client.calls) == 1
    assert bool(out.loc[1, "dripper_layout_propagated"]) is True
    assert out.loc[1, "dripper_response"] == "2main3other"
    assert out.loc[1, "dripper_html"] == "main:2"
    assert out.loc[1, "dripper_content"] == "mm_md:main:2"


def test_layout_template_uses_raw_html_for_layout_propagation_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()
    seen_html_sources: list[str] = []

    class RecordingLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            seen_html_sources.append(task_data["html_source"])
            return {
                "main_html_body": "<article>raw sibling main</article>",
                "main_html_success": True,
            }

    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=RecordingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    rep_html = '<html><body><p _item_id="1">rep main</p></body></html>'
    sibling_html = '<html><body><p _item_id="2">sibling main</p></body></html>'
    df = pd.DataFrame(
        {
            "url": ["https://example.test/a", "https://example.test/b"],
            "html": [rep_html, sibling_html],
        }
    )

    out = _run_two_phase(df, client=client)

    assert seen_html_sources == [sibling_html]
    assert bool(out.loc[1, "dripper_layout_propagated"]) is True
    assert out.loc[1, "dripper_response"] == ""
    assert out.loc[1, "dripper_html"] == "<article>raw sibling main</article>"
    assert out.loc[1, "dripper_content"] == "mm_md:<article>raw sibling main</article>"


def test_layout_template_can_use_layout_text_content_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class LayoutTextParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": "<article>raw sibling main</article>",
                "main_html": "raw sibling main",
                "main_html_success": True,
            }

    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=LayoutTextParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    df = pd.DataFrame(
        {
            "url": ["https://example.test/a", "https://example.test/b"],
            "html": [
                '<html><body><p _item_id="1">rep main</p></body></html>',
                '<html><body><p _item_id="2">sibling main</p></body></html>',
            ],
        }
    )

    out = _run_two_phase(
        df,
        client=client,
        plan_kwargs={"layout_template_propagation_content_source": "layout_text"},
        finalize_kwargs={"layout_template_propagation_content_source": "layout_text"},
    )

    assert bool(out.loc[1, "dripper_layout_propagated"]) is True
    assert out.loc[1, "dripper_html"] == "<article>raw sibling main</article>"
    assert out.loc[1, "dripper_content"] == "raw sibling main"


def test_layout_template_falls_back_when_propagation_overselects_item_ids(
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

    class OverselectingLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": '<main><p _item_id="2">body</p><p _item_id="3">metadata</p></main>',
                "main_html_success": True,
            }

    monkeypatch.setattr(preprocess_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(postprocess_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(layout_finalize_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FakeMapParser,
            layout_parser_cls=OverselectingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    df = pd.DataFrame(
        {
            "url": ["https://example.test/a", "https://example.test/b"],
            "html": [
                '<p _item_id="1">Rep main</p><p _item_id="2">Rep nav</p>',
                ('<p _item_id="2">Sibling main</p><p _item_id="3">Sibling date</p><p _item_id="4">Sibling nav</p>'),
            ],
        }
    )

    out = _run_two_phase(
        df,
        client=client,
        plan_kwargs={
            "layout_template_max_selected_item_ratio": 0.5,
            "layout_template_propagation_target": "mapped_item_ids",
        },
        finalize_kwargs={
            "layout_template_max_selected_item_ratio": 0.5,
            "layout_template_propagation_target": "mapped_item_ids",
        },
    )

    # Over-selecting siblings fail propagation in finalize, get deferred, and are
    # extracted directly by the second inference pass.
    assert bool(out.loc[1, "dripper_layout_propagated"]) is False
    assert out.loc[1, "dripper_html"]


def test_layout_template_validates_cluster_before_propagating_remaining_siblings(
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

    monkeypatch.setattr(preprocess_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(postprocess_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    monkeypatch.setattr(layout_finalize_mod, "_load_mineru_html_bindings", make_label_aware_bindings)
    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=FakeMapParser,
            layout_parser_cls=DivergingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main", "1main"])
    df = pd.DataFrame(
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
    )

    out = _run_two_phase(
        df,
        client=client,
        plan_kwargs={
            "layout_template_max_selected_item_ratio": 1.0,
            "layout_template_validation_rows": 1,
            "layout_template_validation_min_content_f1": 0.98,
        },
        finalize_kwargs={
            "layout_template_max_selected_item_ratio": 1.0,
            "layout_template_validation_rows": 1,
            "layout_template_validation_min_content_f1": 0.98,
        },
    )

    # The validation row diverges from the template, so the remaining sibling is
    # deferred to the second inference pass instead of propagated.
    assert out["dripper_layout_representative"].tolist() == [True, False, False]
    assert out["dripper_layout_propagated"].tolist() == [False, False, False]
    assert out["dripper_layout_validation_llm"].tolist() == [False, True, False]
    assert out.loc[1, "dripper_html"] == "main:1"
    assert out.loc[2, "dripper_html"] == "main:1"


def test_layout_template_min_main_html_sim_forces_fallback_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()

    class LowSimilarityLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
                "main_html_sim": 0.70,
            }

    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=LowSimilarityLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main", "1main"])
    df = pd.DataFrame(
        {
            "url": ["https://example.test/1", "https://example.test/2"],
            "html": ["<p>representative</p>", "<p>sibling</p>"],
        }
    )

    out = _run_two_phase(
        df,
        client=client,
        plan_kwargs={
            "layout_template_max_selected_item_ratio": 1.0,
            "layout_template_min_main_html_sim": 0.80,
        },
        finalize_kwargs={
            "layout_template_max_selected_item_ratio": 1.0,
            "layout_template_min_main_html_sim": 0.80,
        },
    )

    # The sibling propagation similarity (0.70) is below the floor (0.80), so it is
    # deferred and extracted directly by the second inference pass.
    assert out["dripper_layout_representative"].tolist() == [True, False]
    assert out.loc[1, "dripper_html"]


def test_layout_template_splits_layout_groups_by_precomputed_layout_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_web_bindings(monkeypatch, make_llm_web_kit_bindings)
    client = RecordingAsyncClient(["1main", "1main"])
    generation_config = GenerationConfig(max_tokens=2048)
    preprocess = DripperHTMLPreprocessStage(
        html_col="html",
        url_col="url",
        generation_config=generation_config,
    )
    plan = DripperHTMLLayoutPlanStage(
        generation_config=generation_config,
        layout_template_max_selected_item_ratio=1.0,
    )
    inference = DripperHTMLInferenceStage(
        client=client, model_name="dripper", generation_config=generation_config, health_check=False
    )
    finalize = DripperHTMLLayoutFinalizeStage(
        generation_config=generation_config,
        layout_template_max_selected_item_ratio=1.0,
    )
    raw = pd.DataFrame(
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
    )
    preprocessed = preprocess.process(DocumentBatch(task_id="task-1", dataset_name="test", data=raw)).to_pandas()
    # Two distinct precomputed layout ids -> two layout groups, each with its own
    # representative + propagated sibling.
    preprocessed["dripper_layout_id"] = ["layout-archive", "layout-archive", "layout-news", "layout-news"]

    planned = plan.process(DocumentBatch(task_id="task-1", dataset_name="test", data=preprocessed)).to_pandas()
    first = inference.process(DocumentBatch(task_id="task-1", dataset_name="test", data=planned)).to_pandas()
    out = finalize.process(DocumentBatch(task_id="task-1", dataset_name="test", data=first)).to_pandas()

    assert len(client.calls) == 2
    assert out["dripper_layout_representative"].tolist() == [True, False, True, False]
    assert out["dripper_layout_propagated"].tolist() == [False, True, False, True]
    assert out["dripper_layout_cluster"].nunique() == 2


def test_layout_template_passes_more_noise_setting_to_layout_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_webkit_bindings = make_llm_web_kit_bindings()
    seen_more_noise: list[bool] = []

    class RecordingLayoutParser:
        def __init__(self, template_data: dict) -> None:
            pass

        def parse(self, task_data: dict) -> dict:
            seen_more_noise.append(bool(task_data["more_noise_enable"]))
            return {
                "main_html_body": f"<propagated>{task_data['html_source']}</propagated>",
                "main_html_success": True,
            }

    _patch_web_bindings(
        monkeypatch,
        lambda: _LLMWebKitBindings(
            get_feature=base_webkit_bindings.get_feature,
            cluster_html_struct=base_webkit_bindings.cluster_html_struct,
            select_representative_html=base_webkit_bindings.select_representative_html,
            map_parser_cls=base_webkit_bindings.map_parser_cls,
            layout_parser_cls=RecordingLayoutParser,
        ),
    )
    client = RecordingAsyncClient(["1main"])
    df = pd.DataFrame(
        {
            "url": ["https://example.test/a", "https://example.test/b"],
            "html": ["<html>Rep</html>", "<html>Sibling</html>"],
        }
    )

    generation_config = GenerationConfig(max_tokens=2048)
    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url", generation_config=generation_config)
    plan = DripperHTMLLayoutPlanStage(generation_config=generation_config)
    inference = DripperHTMLInferenceStage(
        client=client, model_name="dripper", generation_config=generation_config, health_check=False
    )
    finalize = DripperHTMLLayoutFinalizeStage(generation_config=generation_config)
    finalize.layout_template_more_noise_enable = True

    preprocessed = preprocess.process(DocumentBatch(task_id="task-1", dataset_name="test", data=df)).to_pandas()
    preprocessed = _add_precomputed_layout(preprocessed)
    planned = plan.process(DocumentBatch(task_id="task-1", dataset_name="test", data=preprocessed)).to_pandas()
    first = inference.process(DocumentBatch(task_id="task-1", dataset_name="test", data=planned)).to_pandas()
    finalize.process(DocumentBatch(task_id="task-1", dataset_name="test", data=first))

    assert seen_more_noise == [True]
