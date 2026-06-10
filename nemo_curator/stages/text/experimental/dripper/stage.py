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

"""Dripper HTML main-content extraction through Curator inference clients."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlparse

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient


@dataclass(frozen=True)
class _MinerUHTMLBindings:
    """Runtime bindings to MinerU-HTML objects and processing functions."""

    input_cls: type
    case_cls: type
    output_cls: type
    process_data_cls: type
    generate_output_cls: type
    simplify_single_input: Callable[[Any], Any]
    build_prompt: Callable[..., Any]
    parse_result: Callable[[Any], Any]
    extract_main_html_single: Callable[[Any], Any]
    extract_main_html_fallback: Callable[..., Any]
    convert2content: Callable[..., Any]
    get_fallback_handler: Callable[[str], Any]


def _always_similar(_left: Any, _right: Any, _max_layer_n: int) -> float:
    return 1.0


@dataclass(frozen=True)
class _LLMWebKitBindings:
    """Runtime bindings to ccprocessor/llm-webkit layout-template algorithms."""

    get_feature: Callable[[str], Any]
    cluster_html_struct: Callable[..., Any]
    select_representative_html: Callable[[list[dict[str, str]]], dict[str, str] | None]
    map_parser_cls: type
    layout_parser_cls: type
    similarity: Callable[..., float] = _always_similar


@dataclass(frozen=True)
class _DripperRowResult:
    """Per-row Dripper output."""

    main_html: str
    main_content: Any
    raw_response: str
    preprocess_time_s: float
    inference_time_s: float
    postprocess_time_s: float
    total_time_s: float
    error: str
    warning: str = ""
    simplified_html: str = ""
    mapped_html: str = ""
    item_count: int = 0
    prompt_chars: int = 0
    request_max_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class _DripperPrepResult:
    """Per-row output from Dripper preprocessing."""

    prompt: str = ""
    needs_llm: bool = False
    empty_input: bool = False
    preprocess_time_s: float = 0.0
    primary_error: str = ""
    warning: str = ""
    simplified_html: str = ""
    mapped_html: str = ""
    item_count: int = 0
    prompt_chars: int = 0
    request_max_tokens: int = 0


@dataclass(frozen=True)
class _DripperInferenceResult:
    """Per-row output from Dripper inference."""

    raw_response: str = ""
    inference_time_s: float = 0.0
    primary_error: str = ""
    warning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


_InferenceCache = dict[tuple[str, int], asyncio.Task[_DripperInferenceResult]]


@dataclass(frozen=True)
class _DripperPostResult:
    """Per-row output from Dripper postprocessing."""

    main_html: str = ""
    main_content: Any = ""
    postprocess_time_s: float = 0.0
    error: str = ""
    warning: str = ""


@dataclass(frozen=True)
class _LayoutTemplateRowResult:
    """Per-row output from layout-template extraction."""

    raw_response: str = ""
    inference_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    main_html: str = ""
    main_content: Any = ""
    postprocess_time_s: float = 0.0
    error: str = ""
    warning: str = ""
    primary_error: str = ""
    deferred_llm: bool = False
    layout_finalized: bool = True
    layout_cluster: str = ""
    layout_representative: bool = False
    layout_propagated: bool = False
    layout_propagation_success: bool = False
    layout_fallback_llm: bool = False
    layout_standalone_llm: bool = False
    layout_pending_propagation: bool = False
    layout_mapping_json: str = ""


@dataclass(frozen=True)
class _LayoutGroupPlan:
    """A layout group to try, plus safer fallback groups if the attempt fails."""

    indexes: list[int]
    host_key: str = ""
    source: str = "dom"
    fallback_groups: tuple[list[int], ...] = ()


@dataclass(frozen=True)
class _LayoutGroupOutcome:
    """Result of processing one layout group."""

    results: dict[int, _LayoutTemplateRowResult]
    accepted: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class _LayoutClusterAssignment:
    """Precomputed host-bounded DOM layout assignment."""

    row_index: int
    layout_id: str


_DRIPPER_PROMPT_COL = "_dripper_prompt"
_DRIPPER_NEEDS_LLM_COL = "_dripper_needs_llm"
_DRIPPER_PRIMARY_ERROR_COL = "_dripper_primary_error"
_DRIPPER_EMPTY_INPUT_COL = "_dripper_empty_input"
_DRIPPER_LAYOUT_FINALIZED_COL = "_dripper_layout_finalized"


def _load_mineru_html_bindings() -> _MinerUHTMLBindings:
    """Import MinerU-HTML lazily so Curator remains importable without it."""
    try:
        from mineru_html.base import (
            MinerUHTMLCase,
            MinerUHTMLGenerateOutput,
            MinerUHTMLInput,
            MinerUHTMLOutput,
            MinerUHTMLProcessData,
        )
        from mineru_html.process import (
            build_prompt,
            convert2content,
            extract_main_html_fallback,
            extract_main_html_single,
            get_fallback_handler,
            parse_result,
            simplify_single_input,
        )
    except ModuleNotFoundError as exc:
        msg = (
            "DripperHTMLExtractionStage requires the optional 'mineru_html' package. "
            "Install MinerU-HTML in the Curator environment before running this stage."
        )
        raise RuntimeError(msg) from exc

    return _MinerUHTMLBindings(
        input_cls=MinerUHTMLInput,
        case_cls=MinerUHTMLCase,
        output_cls=MinerUHTMLOutput,
        process_data_cls=MinerUHTMLProcessData,
        generate_output_cls=MinerUHTMLGenerateOutput,
        simplify_single_input=simplify_single_input,
        build_prompt=build_prompt,
        parse_result=parse_result,
        extract_main_html_single=extract_main_html_single,
        extract_main_html_fallback=extract_main_html_fallback,
        convert2content=convert2content,
        get_fallback_handler=get_fallback_handler,
    )


def _load_llm_web_kit_bindings() -> _LLMWebKitBindings:
    """Import ccprocessor/llm-webkit layout-template parser lazily."""
    try:
        from llm_web_kit.html_layout.html_layout_cosin import cluster_html_struct, get_feature, similarity
        from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser
        from llm_web_kit.main_html_parser.parser.tag_mapping import MapItemToHtmlTagsParser
        from llm_web_kit.main_html_parser.typical_html.typical_html import select_representative_html
    except ModuleNotFoundError as exc:
        msg = (
            "Dripper layout-template mode requires the optional 'llm_web_kit' package "
            "from https://github.com/ccprocessor/llm-webkit."
        )
        raise RuntimeError(msg) from exc

    return _LLMWebKitBindings(
        get_feature=get_feature,
        cluster_html_struct=cluster_html_struct,
        select_representative_html=select_representative_html,
        map_parser_cls=MapItemToHtmlTagsParser,
        layout_parser_cls=LayoutBatchParser,
        similarity=similarity,
    )


@dataclass(kw_only=True)
class DripperHTMLExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract main HTML/content with Dripper through a Curator LLM client.

    The stage reuses MinerU-HTML's simplification, prompt construction,
    response parsing, main-HTML extraction, fallback, and content conversion
    functions. Only the inference call is replaced with Curator's
    OpenAI-compatible ``AsyncLLMClient`` path, which can point at an
    ``InferenceServer`` endpoint.
    """

    name: str = "DripperHTMLExtractionStage"
    client: AsyncLLMClient | None
    model_name: str
    html_col: str = "html"
    url_col: str | None = "url"
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    prompt_chars_col: str = "dripper_prompt_chars"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    prompt_version: str = "short_compact"
    output_format: str = "mm_md"
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    generation_config: GenerationConfig | None = None
    dynamic_max_tokens: bool = False
    dynamic_max_token_padding: int = 16
    dynamic_max_tokens_per_item: int = 6
    dynamic_min_max_tokens: int = 32
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    health_check: bool = True
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.client is None:
            msg = "DripperHTMLExtractionStage requires a non-None 'client' (AsyncLLMClient)"
            raise ValueError(msg)
        self.model_name = self.model_name.strip()
        if not self.model_name:
            msg = "DripperHTMLExtractionStage requires a non-empty 'model_name'"
            raise ValueError(msg)
        if self.max_concurrent_requests <= 0:
            msg = "max_concurrent_requests must be positive"
            raise ValueError(msg)
        if self.dynamic_max_token_padding < 0:
            msg = "dynamic_max_token_padding must be non-negative"
            raise ValueError(msg)
        if self.dynamic_max_tokens_per_item <= 0:
            msg = "dynamic_max_tokens_per_item must be positive"
            raise ValueError(msg)
        if self.dynamic_min_max_tokens <= 0:
            msg = "dynamic_min_max_tokens must be positive"
            raise ValueError(msg)
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            msg = f"structured_output_mode must be one of {sorted(_STRUCTURED_OUTPUT_MODES)}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            self.output_html_col,
            self.output_content_col,
            self.raw_response_col,
            self.preprocess_time_col,
            self.inference_time_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            self.item_count_col,
            self.prompt_chars_col,
            self.request_max_tokens_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
        ]
        if self.keep_intermediate:
            columns.extend([self.simplified_html_col, self.mapped_html_col])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return

        self._bindings = _load_mineru_html_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self.client.setup()
        if self.health_check:
            self._run_health_check()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        html_values = df[self.html_col].tolist()
        if self.url_col is not None and self.url_col in df.columns:
            url_values = df[self.url_col].tolist()
        else:
            url_values = [None] * len(df)

        results = run_async_safe(lambda: self._extract_all_async(html_values, url_values))
        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.raw_response_col] = [r.raw_response for r in results]
        df[self.preprocess_time_col] = [r.preprocess_time_s for r in results]
        df[self.inference_time_col] = [r.inference_time_s for r in results]
        df[self.postprocess_time_col] = [r.postprocess_time_s for r in results]
        df[self.total_time_col] = [r.total_time_s for r in results]
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [r.warning for r in results]
        df[self.item_count_col] = [r.item_count for r in results]
        df[self.prompt_chars_col] = [r.prompt_chars for r in results]
        df[self.request_max_tokens_col] = [r.request_max_tokens for r in results]
        df[self.prompt_tokens_col] = [r.prompt_tokens for r in results]
        df[self.completion_tokens_col] = [r.completion_tokens for r in results]
        df[self.total_tokens_col] = [r.total_tokens for r in results]
        if self.keep_intermediate:
            df[self.simplified_html_col] = [r.simplified_html for r in results]
            df[self.mapped_html_col] = [r.mapped_html for r in results]

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _run_health_check(self) -> None:
        try:
            response = run_async_safe(self._query_health_check)
        except RuntimeError:
            raise
        except Exception as exc:
            msg = f"Dripper LLM health check failed: {exc}. Ensure the inference server is reachable."
            raise RuntimeError(msg) from exc
        if not response:
            msg = "Dripper LLM health check returned an empty response"
            raise RuntimeError(msg)
        logger.info("Dripper LLM health check passed")

    async def _query_health_check(self) -> str:
        extra_kwargs = self.generation_config.extra_kwargs if self.generation_config is not None else None
        generation_config = GenerationConfig(max_tokens=8, temperature=0.0, top_p=1.0, extra_kwargs=extra_kwargs)
        response = await self.client.query_model(  # type: ignore[union-attr]
            model=self.model_name,
            messages=[{"role": "user", "content": 'Return exactly: "1main"'}],
            generation_config=generation_config,
        )
        return response[0] if response else ""

    async def _extract_all_async(self, html_values: list[Any], url_values: list[Any]) -> list[_DripperRowResult]:
        sem = asyncio.Semaphore(self.max_concurrent_requests)

        async def _extract_one_throttled(html_value: Any, url_value: Any) -> _DripperRowResult:
            async with sem:
                return await self._extract_one_async(html_value, url_value)

        tasks = [_extract_one_throttled(html_value, url_value) for html_value, url_value in zip(html_values, url_values)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[_DripperRowResult] = []
        for idx, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error("Dripper extraction failed for row {}: {}", idx, result)
                results.append(
                    _DripperRowResult(
                        main_html="",
                        main_content="",
                        raw_response="",
                        preprocess_time_s=0.0,
                        inference_time_s=0.0,
                        postprocess_time_s=0.0,
                        total_time_s=0.0,
                        error=str(result),
                    )
                )
            else:
                results.append(result)
        return results

    async def _extract_one_async(self, html_value: Any, url_value: Any) -> _DripperRowResult:
        assert self._bindings is not None
        start_total = time.perf_counter()
        html = self._coerce_html(html_value)
        if not html.strip():
            return _DripperRowResult(
                main_html="",
                main_content="",
                raw_response="",
                preprocess_time_s=0.0,
                inference_time_s=0.0,
                postprocess_time_s=0.0,
                total_time_s=time.perf_counter() - start_total,
                error="",
                warning="empty HTML input",
            )

        url = self._coerce_optional_str(url_value)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        raw_response = ""
        preprocess_time_s = 0.0
        inference_time_s = 0.0
        postprocess_time_s = 0.0
        primary_error = ""
        warning = ""
        item_count = 0
        prompt_chars = 0
        request_max_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            start_preprocess = time.perf_counter()
            case = self._bindings.simplify_single_input(case)
            item_count = self._count_item_ids(case)
            if not self._case_has_item_ids(case):
                case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
                warning = "no _item_id attributes after simplification; used fallback without LLM"
                preprocess_time_s = time.perf_counter() - start_preprocess
            else:
                case = self._bindings.build_prompt(case, prompt_version=self.prompt_version)
                prompt = case.generate_input.full_prompt
                prompt_chars = len(prompt)
                generation_config = _with_structured_output_config(
                    self._generation_config_for_item_count(item_count),
                    prompt,
                    self.structured_output_mode,
                )
                request_max_tokens = generation_config.max_tokens or 0
                preprocess_time_s = time.perf_counter() - start_preprocess
                start_inference = time.perf_counter()
                raw_response, prompt_tokens, completion_tokens, total_tokens = await self._query_model_with_usage(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    generation_config=generation_config,
                )
                inference_time_s = time.perf_counter() - start_inference
                start_postprocess = time.perf_counter()
                case.generate_output = self._bindings.generate_output_cls(response=raw_response)
                case = self._bindings.parse_result(case)
                case = self._bindings.extract_main_html_single(case)
                postprocess_time_s += time.perf_counter() - start_postprocess
        except Exception as exc:  # noqa: BLE001
            if preprocess_time_s == 0.0:
                preprocess_time_s = time.perf_counter() - start_total
            primary_error = str(exc)
            logger.debug("Dripper primary extraction failed, applying {} fallback: {}", self.fallback, primary_error)
            try:
                start_fallback = time.perf_counter()
                case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
                postprocess_time_s += time.perf_counter() - start_fallback
                warning = primary_error
            except Exception as fallback_exc:  # noqa: BLE001
                error = f"{primary_error}; fallback failed: {fallback_exc}"
                return _DripperRowResult(
                    main_html="",
                    main_content="",
                    raw_response=raw_response,
                    preprocess_time_s=preprocess_time_s,
                    inference_time_s=inference_time_s,
                    postprocess_time_s=postprocess_time_s,
                    total_time_s=time.perf_counter() - start_total,
                    error=error,
                    warning=primary_error,
                    simplified_html=self._get_processed_attr(case, "simpled_html"),
                    mapped_html=self._get_processed_attr(case, "map_html"),
                    item_count=item_count,
                    prompt_chars=prompt_chars,
                    request_max_tokens=request_max_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

        conversion_error = ""
        try:
            start_conversion = time.perf_counter()
            self._sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
            postprocess_time_s += time.perf_counter() - start_conversion
        except Exception as exc:  # noqa: BLE001
            postprocess_time_s += time.perf_counter() - start_conversion
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)

        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        if main_content is None:
            main_content = ""
        error = ""
        if conversion_error:
            if self._is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error

        return _DripperRowResult(
            main_html=main_html,
            main_content=main_content,
            raw_response=raw_response,
            preprocess_time_s=preprocess_time_s,
            inference_time_s=inference_time_s,
            postprocess_time_s=postprocess_time_s,
            total_time_s=time.perf_counter() - start_total,
            error=error,
            warning=warning,
            simplified_html=self._get_processed_attr(case, "simpled_html"),
            mapped_html=self._get_processed_attr(case, "map_html"),
            item_count=item_count,
            prompt_chars=prompt_chars,
            request_max_tokens=request_max_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def _query_model_with_usage(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig,
    ) -> tuple[str, int, int, int]:
        assert self.client is not None
        query_model_with_usage = getattr(self.client, "query_model_with_usage", None)
        if callable(query_model_with_usage):
            response = await query_model_with_usage(
                model=model,
                messages=messages,
                generation_config=generation_config,
            )
            contents = getattr(response, "contents", [])
            return (
                contents[0] if contents else "",
                _coerce_usage_int(getattr(response, "prompt_tokens", None)),
                _coerce_usage_int(getattr(response, "completion_tokens", None)),
                _coerce_usage_int(getattr(response, "total_tokens", None)),
            )

        response = await self.client.query_model(
            model=model,
            messages=messages,
            generation_config=generation_config,
        )
        return response[0] if response else "", 0, 0, 0

    @staticmethod
    def _sanitize_case_output_html(case: Any) -> None:
        output_data = getattr(case, "output_data", None)
        if output_data is None:
            return
        main_html = getattr(output_data, "main_html", None)
        if isinstance(main_html, str):
            output_data.main_html = _strip_xml_incompatible_chars(main_html)

    @staticmethod
    def _get_processed_attr(case: Any, attr: str) -> str:
        process_data = getattr(case, "process_data", None)
        value = getattr(process_data, attr, "") if process_data is not None else ""
        return value if isinstance(value, str) else ""

    @classmethod
    def _case_has_item_ids(cls, case: Any) -> bool:
        return "_item_id" in cls._get_processed_attr(case, "simpled_html") or "_item_id" in cls._get_processed_attr(
            case,
            "map_html",
        )

    @classmethod
    def _count_item_ids(cls, case: Any) -> int:
        html = cls._get_processed_attr(case, "simpled_html") or cls._get_processed_attr(case, "map_html")
        return len(set(_ITEM_ID_RE.findall(html)))

    def _generation_config_for_item_count(self, item_count: int) -> GenerationConfig:
        base = self.generation_config or GenerationConfig()
        if not self.dynamic_max_tokens or base.max_tokens is None or item_count <= 0:
            return base

        dynamic_max_tokens = max(
            self.dynamic_min_max_tokens,
            item_count * self.dynamic_max_tokens_per_item + self.dynamic_max_token_padding,
        )
        return replace(base, max_tokens=min(base.max_tokens, dynamic_max_tokens))

    @staticmethod
    def _coerce_html(value: Any) -> str:
        if _is_missing(value):
            return ""
        if isinstance(value, bytes | bytearray):
            raw_bytes = bytes(value)
            decoded = _decode_html_bytes(raw_bytes)
            if decoded is None:
                decoded = raw_bytes.decode("utf-8", errors="replace")
            return _strip_xml_incompatible_chars(decoded or "")
        return _strip_xml_incompatible_chars(str(value))

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if _is_missing(value):
            return None
        text = str(value)
        return text if text else None

    @staticmethod
    def _is_empty_document_error(error: str) -> bool:
        normalized = error.lower()
        return (
            "document is empty" in normalized
            or "empty html tree" in normalized
            or "empty html input" in normalized
        )


@dataclass(kw_only=True)
class DripperHTMLPreprocessStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Simplify HTML and build Dripper prompts before model inference."""

    name: str = "DripperHTMLPreprocessStage"
    html_col: str = "html"
    url_col: str | None = "url"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    prompt_chars_col: str = "dripper_prompt_chars"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    prompt_version: str = "short_compact"
    generation_config: GenerationConfig | None = None
    dynamic_max_tokens: bool = False
    dynamic_max_token_padding: int = 16
    dynamic_max_tokens_per_item: int = 6
    dynamic_min_max_tokens: int = 32
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.dynamic_max_token_padding < 0:
            msg = "dynamic_max_token_padding must be non-negative"
            raise ValueError(msg)
        if self.dynamic_max_tokens_per_item <= 0:
            msg = "dynamic_max_tokens_per_item must be positive"
            raise ValueError(msg)
        if self.dynamic_min_max_tokens <= 0:
            msg = "dynamic_min_max_tokens must be positive"
            raise ValueError(msg)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.raw_response_col,
            self.preprocess_time_col,
            self.inference_time_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            self.item_count_col,
            self.prompt_chars_col,
            self.request_max_tokens_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        html_values = df[self.html_col].tolist()
        if self.url_col is not None and self.url_col in df.columns:
            url_values = df[self.url_col].tolist()
        else:
            url_values = [None] * len(df)

        results = [self._prepare_one(html_value, url_value) for html_value, url_value in zip(html_values, url_values)]

        df[self.raw_response_col] = ""
        df[self.preprocess_time_col] = [r.preprocess_time_s for r in results]
        df[self.inference_time_col] = 0.0
        df[self.postprocess_time_col] = 0.0
        df[self.total_time_col] = [r.preprocess_time_s for r in results]
        df[self.error_col] = ""
        df[self.warning_col] = [r.warning for r in results]
        df[self.item_count_col] = [r.item_count for r in results]
        df[self.prompt_chars_col] = [r.prompt_chars for r in results]
        df[self.request_max_tokens_col] = [r.request_max_tokens for r in results]
        df[self.prompt_tokens_col] = 0
        df[self.completion_tokens_col] = 0
        df[self.total_tokens_col] = 0
        df[self.simplified_html_col] = [r.simplified_html for r in results]
        df[self.mapped_html_col] = [r.mapped_html for r in results]
        df[_DRIPPER_PROMPT_COL] = [r.prompt for r in results]
        df[_DRIPPER_NEEDS_LLM_COL] = [r.needs_llm for r in results]
        df[_DRIPPER_PRIMARY_ERROR_COL] = [r.primary_error for r in results]
        df[_DRIPPER_EMPTY_INPUT_COL] = [r.empty_input for r in results]

        self._log_metrics(
            {
                "preprocess_rows": float(len(df)),
                "preprocess_llm_rows": float(sum(r.needs_llm for r in results)),
                "preprocess_fallback_rows": float(sum((not r.needs_llm) and (not r.empty_input) for r in results)),
            }
        )
        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _prepare_one(self, html_value: Any, url_value: Any) -> _DripperPrepResult:
        assert self._bindings is not None
        started = time.perf_counter()
        html = DripperHTMLExtractionStage._coerce_html(html_value)
        if not html.strip():
            return _DripperPrepResult(
                empty_input=True,
                preprocess_time_s=time.perf_counter() - started,
                warning="empty HTML input",
            )

        url = DripperHTMLExtractionStage._coerce_optional_str(url_value)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        simplified_html = ""
        mapped_html = ""
        item_count = 0
        try:
            case = self._bindings.simplify_single_input(case)
            simplified_html = DripperHTMLExtractionStage._get_processed_attr(case, "simpled_html")
            mapped_html = DripperHTMLExtractionStage._get_processed_attr(case, "map_html")
            item_count = DripperHTMLExtractionStage._count_item_ids(case)
            if not DripperHTMLExtractionStage._case_has_item_ids(case):
                return _DripperPrepResult(
                    needs_llm=False,
                    preprocess_time_s=time.perf_counter() - started,
                    warning="no _item_id attributes after simplification; used fallback without LLM",
                    simplified_html=simplified_html,
                    mapped_html=mapped_html,
                    item_count=item_count,
                )

            case = self._bindings.build_prompt(case, prompt_version=self.prompt_version)
            prompt = case.generate_input.full_prompt
            generation_config = self._generation_config_for_item_count(item_count)
            return _DripperPrepResult(
                prompt=prompt,
                needs_llm=True,
                preprocess_time_s=time.perf_counter() - started,
                simplified_html=simplified_html,
                mapped_html=mapped_html,
                item_count=item_count,
                prompt_chars=len(prompt),
                request_max_tokens=generation_config.max_tokens or 0,
            )
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper preprocessing failed; postprocess stage will apply fallback: {}", primary_error)
            return _DripperPrepResult(
                needs_llm=False,
                preprocess_time_s=time.perf_counter() - started,
                primary_error=primary_error,
                warning=primary_error,
                simplified_html=simplified_html,
                mapped_html=mapped_html,
                item_count=item_count,
            )

    def _generation_config_for_item_count(self, item_count: int) -> GenerationConfig:
        base = self.generation_config or GenerationConfig()
        if not self.dynamic_max_tokens or base.max_tokens is None or item_count <= 0:
            return base

        dynamic_max_tokens = max(
            self.dynamic_min_max_tokens,
            item_count * self.dynamic_max_tokens_per_item + self.dynamic_max_token_padding,
        )
        return replace(base, max_tokens=min(base.max_tokens, dynamic_max_tokens))


@dataclass(kw_only=True)
class DripperHTMLInferenceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Run only Dripper model inference against an OpenAI-compatible client."""

    name: str = "DripperHTMLInferenceStage"
    client: AsyncLLMClient | None
    model_name: str
    raw_response_col: str = "dripper_response"
    inference_time_col: str = "dripper_inference_time_s"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    generation_config: GenerationConfig | None = None
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    health_check: bool = False
    worker_count: int | None = None

    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.client is None:
            msg = "DripperHTMLInferenceStage requires a non-None 'client' (AsyncLLMClient)"
            raise ValueError(msg)
        self.model_name = self.model_name.strip()
        if not self.model_name:
            msg = "DripperHTMLInferenceStage requires a non-empty 'model_name'"
            raise ValueError(msg)
        if self.max_concurrent_requests <= 0:
            msg = "max_concurrent_requests must be positive"
            raise ValueError(msg)
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            msg = f"structured_output_mode must be one of {sorted(_STRUCTURED_OUTPUT_MODES)}"
            raise ValueError(msg)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [_DRIPPER_PROMPT_COL, _DRIPPER_NEEDS_LLM_COL, self.request_max_tokens_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.raw_response_col,
            self.inference_time_col,
            self.warning_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            _DRIPPER_PRIMARY_ERROR_COL,
        ]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self.client.setup()
        if self.health_check:
            self._run_health_check()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        results = run_async_safe(lambda: self._infer_all_async(df))

        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        existing_raw_responses = (
            df[self.raw_response_col].astype(str).tolist()
            if self.raw_response_col in df
            else [""] * len(df)
        )
        existing_inference_times = (
            pd.to_numeric(df[self.inference_time_col], errors="coerce").fillna(0.0).tolist()
            if self.inference_time_col in df
            else [0.0] * len(df)
        )
        existing_prompt_tokens = (
            pd.to_numeric(df[self.prompt_tokens_col], errors="coerce").fillna(0).astype(int).tolist()
            if self.prompt_tokens_col in df
            else [0] * len(df)
        )
        existing_completion_tokens = (
            pd.to_numeric(df[self.completion_tokens_col], errors="coerce").fillna(0).astype(int).tolist()
            if self.completion_tokens_col in df
            else [0] * len(df)
        )
        existing_total_tokens = (
            pd.to_numeric(df[self.total_tokens_col], errors="coerce").fillna(0).astype(int).tolist()
            if self.total_tokens_col in df
            else [0] * len(df)
        )
        existing_warnings = df[self.warning_col].astype(str) if self.warning_col in df else pd.Series([""] * len(df))
        existing_primary_errors = (
            df[_DRIPPER_PRIMARY_ERROR_COL].astype(str)
            if _DRIPPER_PRIMARY_ERROR_COL in df
            else pd.Series([""] * len(df))
        )
        df[self.raw_response_col] = [
            r.raw_response if should_query else existing_raw
            for r, should_query, existing_raw in zip(results, needs_llm, existing_raw_responses, strict=True)
        ]
        df[self.inference_time_col] = [
            r.inference_time_s if should_query else existing_time
            for r, should_query, existing_time in zip(results, needs_llm, existing_inference_times, strict=True)
        ]
        df[self.warning_col] = [
            _append_warning(existing_warning, result.warning)
            for existing_warning, result in zip(existing_warnings.tolist(), results, strict=True)
        ]
        df[_DRIPPER_PRIMARY_ERROR_COL] = [
            _append_warning(existing_error, result.primary_error)
            for existing_error, result in zip(existing_primary_errors.tolist(), results, strict=True)
        ]
        df[self.prompt_tokens_col] = [
            r.prompt_tokens if should_query else existing_tokens
            for r, should_query, existing_tokens in zip(results, needs_llm, existing_prompt_tokens, strict=True)
        ]
        df[self.completion_tokens_col] = [
            r.completion_tokens if should_query else existing_tokens
            for r, should_query, existing_tokens in zip(results, needs_llm, existing_completion_tokens, strict=True)
        ]
        df[self.total_tokens_col] = [
            r.total_tokens if should_query else existing_tokens
            for r, should_query, existing_tokens in zip(results, needs_llm, existing_total_tokens, strict=True)
        ]

        llm_prompts = [
            str(row.get(_DRIPPER_PROMPT_COL, "") or "")
            for _, row in df.iterrows()
            if bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
        ]
        non_empty_llm_prompts = [prompt for prompt in llm_prompts if prompt.strip()]
        unique_llm_prompts = len(set(non_empty_llm_prompts))
        self._log_metrics(
            {
                "inference_rows": float(len(df)),
                "inference_llm_rows": float(sum(bool(v) for v in df[_DRIPPER_NEEDS_LLM_COL].tolist())),
                "inference_unique_llm_prompts": float(unique_llm_prompts),
                "inference_dedup_saved_rows": float(len(non_empty_llm_prompts) - unique_llm_prompts),
                "inference_errors": float(sum(1 for r in results if r.primary_error)),
            }
        )
        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _run_health_check(self) -> None:
        try:
            response = run_async_safe(self._query_health_check)
        except RuntimeError:
            raise
        except Exception as exc:
            msg = f"Dripper LLM health check failed: {exc}. Ensure the inference server is reachable."
            raise RuntimeError(msg) from exc
        if not response:
            msg = "Dripper LLM health check returned an empty response"
            raise RuntimeError(msg)
        logger.info("Dripper LLM health check passed")

    async def _query_health_check(self) -> str:
        extra_kwargs = self.generation_config.extra_kwargs if self.generation_config is not None else None
        generation_config = GenerationConfig(max_tokens=8, temperature=0.0, top_p=1.0, extra_kwargs=extra_kwargs)
        response = await self.client.query_model(  # type: ignore[union-attr]
            model=self.model_name,
            messages=[{"role": "user", "content": 'Return exactly: "1main"'}],
            generation_config=generation_config,
        )
        return response[0] if response else ""

    async def _infer_all_async(self, df: pd.DataFrame) -> list[_DripperInferenceResult]:
        sem = asyncio.Semaphore(self.max_concurrent_requests)
        prompts = df[_DRIPPER_PROMPT_COL].astype(str).tolist()
        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        request_max_tokens = (
            pd.to_numeric(df[self.request_max_tokens_col], errors="coerce").fillna(0).astype(int).tolist()
            if self.request_max_tokens_col in df.columns
            else [0] * len(df)
        )

        async def _infer_one_throttled(
            prompt: str,
            row_max_tokens: int,
        ) -> _DripperInferenceResult:
            async with sem:
                return await self._infer_one_async(prompt, True, row_max_tokens)

        grouped_indexes: dict[tuple[str, int], list[int]] = defaultdict(list)
        results: list[_DripperInferenceResult | None] = [None] * len(df)
        for idx, (prompt, should_query, row_max_tokens) in enumerate(
            zip(prompts, needs_llm, request_max_tokens, strict=True)
        ):
            if not should_query:
                results[idx] = _DripperInferenceResult()
            elif not prompt.strip():
                results[idx] = _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")
            else:
                grouped_indexes[(prompt, row_max_tokens)].append(idx)

        tasks = {
            key: _infer_one_throttled(prompt=key[0], row_max_tokens=key[1])
            for key in grouped_indexes
        }
        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for (_key, indexes), result in zip(grouped_indexes.items(), raw_results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Dripper inference failed for prompt group {} rows: {}", len(indexes), result)
                error = str(result)
                first_result = _DripperInferenceResult(primary_error=error, warning=error)
            else:
                first_result = result
            first_idx = indexes[0]
            results[first_idx] = first_result
            for duplicate_idx in indexes[1:]:
                results[duplicate_idx] = replace(
                    first_result,
                    inference_time_s=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

        return [result if result is not None else _DripperInferenceResult() for result in results]

    async def _infer_one_async(self, prompt: str, should_query: bool, row_max_tokens: int) -> _DripperInferenceResult:
        if not should_query:
            return _DripperInferenceResult()
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")

        started = time.perf_counter()
        try:
            generation_config = self.generation_config or GenerationConfig()
            if row_max_tokens > 0 and generation_config.max_tokens != row_max_tokens:
                generation_config = replace(generation_config, max_tokens=row_max_tokens)
            generation_config = _with_structured_output_config(
                generation_config,
                prompt,
                self.structured_output_mode,
            )
            raw_response, prompt_tokens, completion_tokens, total_tokens = await self._query_model_with_usage(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                generation_config=generation_config,
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.debug("Dripper inference failed; postprocess stage will apply fallback: {}", error)
            return _DripperInferenceResult(
                inference_time_s=time.perf_counter() - started,
                primary_error=error,
                warning=error,
            )
        return _DripperInferenceResult(
            raw_response=raw_response,
            inference_time_s=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def _query_model_with_usage(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig,
    ) -> tuple[str, int, int, int]:
        assert self.client is not None
        query_model_with_usage = getattr(self.client, "query_model_with_usage", None)
        if callable(query_model_with_usage):
            response = await query_model_with_usage(
                model=model,
                messages=messages,
                generation_config=generation_config,
            )
            contents = getattr(response, "contents", [])
            return (
                contents[0] if contents else "",
                _coerce_usage_int(getattr(response, "prompt_tokens", None)),
                _coerce_usage_int(getattr(response, "completion_tokens", None)),
                _coerce_usage_int(getattr(response, "total_tokens", None)),
            )

        response = await self.client.query_model(
            model=model,
            messages=messages,
            generation_config=generation_config,
        )
        return response[0] if response else "", 0, 0, 0


@dataclass(kw_only=True)
class DripperHTMLPostprocessStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Parse Dripper responses, extract main HTML, and convert content."""

    name: str = "DripperHTMLPostprocessStage"
    html_col: str = "html"
    url_col: str | None = "url"
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.html_col,
            self.raw_response_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            self.output_html_col,
            self.output_content_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
        ]
        if self.keep_intermediate:
            columns.extend([self.simplified_html_col, self.mapped_html_col])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        html_values = df[self.html_col].tolist()
        if self.url_col is not None and self.url_col in df.columns:
            url_values = df[self.url_col].tolist()
        else:
            url_values = [None] * len(df)

        results = [
            self._postprocess_one(row, html_value, url_value)
            for (_, row), html_value, url_value in zip(df.iterrows(), html_values, url_values, strict=True)
        ]

        preprocess_times = _numeric_series_or_zero(df, self.preprocess_time_col)
        inference_times = _numeric_series_or_zero(df, self.inference_time_col)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)

        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.postprocess_time_col] = postprocess_times
        df[self.total_time_col] = preprocess_times + inference_times + postprocess_times
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [r.warning for r in results]

        drop_cols = [
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
            _DRIPPER_LAYOUT_FINALIZED_COL,
        ]
        if not self.keep_intermediate:
            drop_cols.extend([self.simplified_html_col, self.mapped_html_col])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        self._log_metrics(
            {
                "postprocess_rows": float(len(df)),
                "postprocess_errors": float(sum(1 for r in results if r.error)),
                "postprocess_warnings": float(sum(1 for r in results if r.warning)),
            }
        )
        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _postprocess_one(self, row: pd.Series, html_value: Any, url_value: Any) -> _DripperPostResult:
        assert self._bindings is not None
        started = time.perf_counter()
        warning = str(row.get(self.warning_col, "") or "")
        primary_error = str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")
        if bool(row.get(_DRIPPER_LAYOUT_FINALIZED_COL, False)):
            return _DripperPostResult(
                main_html=str(row.get(self.output_html_col, "") or ""),
                main_content=row.get(self.output_content_col, "") or "",
                postprocess_time_s=float(row.get(self.postprocess_time_col, 0.0) or 0.0),
                error=str(row.get(self.error_col, "") or ""),
                warning=warning,
            )
        html = DripperHTMLExtractionStage._coerce_html(html_value)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not html.strip():
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                warning=warning or "empty HTML input",
            )

        url = DripperHTMLExtractionStage._coerce_optional_str(url_value)
        case = self._build_case(
            html=html,
            url=url,
            simplified_html=str(row.get(self.simplified_html_col, "") or ""),
            mapped_html=str(row.get(self.mapped_html_col, "") or ""),
        )
        raw_response = str(row.get(self.raw_response_col, "") or "")
        needs_llm = bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))

        if needs_llm and raw_response:
            try:
                case.generate_output = self._bindings.generate_output_cls(response=raw_response)
                case = self._bindings.parse_result(case)
                case = self._bindings.extract_main_html_single(case)
            except Exception as exc:  # noqa: BLE001
                primary_error = _append_warning(primary_error, str(exc))
                logger.debug("Dripper parse/extract failed, applying {} fallback: {}", self.fallback, primary_error)
                fallback_result = self._apply_fallback(case, primary_error)
                case = fallback_result[0]
                warning = _append_warning(warning, fallback_result[1])
                if fallback_result[2]:
                    return _DripperPostResult(
                        postprocess_time_s=time.perf_counter() - started,
                        error=fallback_result[2],
                        warning=warning,
                    )
        else:
            if needs_llm and not primary_error:
                primary_error = "empty Dripper response"
            fallback_result = self._apply_fallback(case, primary_error)
            case = fallback_result[0]
            warning = _append_warning(warning, fallback_result[1])
            if fallback_result[2]:
                return _DripperPostResult(
                    postprocess_time_s=time.perf_counter() - started,
                    error=fallback_result[2],
                    warning=warning,
                )

        conversion_error = ""
        try:
            self._sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
        except Exception as exc:  # noqa: BLE001
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)

        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        if main_content is None:
            main_content = ""
        error = ""
        if conversion_error:
            if DripperHTMLExtractionStage._is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error

        return _DripperPostResult(
            main_html=main_html,
            main_content=main_content,
            postprocess_time_s=time.perf_counter() - started,
            error=error,
            warning=warning,
        )

    def _build_case(self, *, html: str, url: str | None, simplified_html: str, mapped_html: str) -> Any:
        assert self._bindings is not None
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _apply_fallback(self, case: Any, primary_error: str) -> tuple[Any, str, str]:
        assert self._bindings is not None
        try:
            case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
            return case, primary_error, ""
        except Exception as fallback_exc:  # noqa: BLE001
            if primary_error:
                return case, primary_error, f"{primary_error}; fallback failed: {fallback_exc}"
            return case, "", f"fallback failed: {fallback_exc}"

    @staticmethod
    def _sanitize_case_output_html(case: Any) -> None:
        DripperHTMLExtractionStage._sanitize_case_output_html(case)


@dataclass(kw_only=True)
class DripperHTMLLayoutClusteringStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Precompute host-bounded llm-webkit DOM layout IDs on CPU.

    Running this as a separate pass lets the downstream template stage use
    ``layout_id_col`` instead of rebuilding DBSCAN clusters inside every
    representative/propagation actor.
    """

    name: str = "DripperHTMLLayoutClusteringStage"
    html_col: str = "html"
    url_col: str | None = "url"
    host_col: str | None = None
    item_count_col: str = "dripper_item_count"
    layout_id_col: str = "dripper_layout_id"
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_page_signature_mode: str = "none"
    layout_template_max_exact_host_pages: int = 0
    layout_template_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    worker_count: int | None = None

    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.layout_cluster_threshold <= 1.0:
            msg = "layout_cluster_threshold must be in (0, 1]"
            raise ValueError(msg)
        if self.layout_template_min_cluster_size <= 1:
            msg = "layout_template_min_cluster_size must be greater than 1"
            raise ValueError(msg)
        if self.layout_page_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = f"layout_page_signature_mode must be one of {sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            raise ValueError(msg)
        if self.layout_template_max_exact_host_pages < 0:
            msg = "layout_template_max_exact_host_pages must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_host_mode not in _LAYOUT_TEMPLATE_LARGE_HOST_MODES:
            msg = (
                "layout_template_large_host_mode must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_LARGE_HOST_MODES)}"
            )
            raise ValueError(msg)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        columns = [self.html_col]
        if self.url_col:
            columns.append(self.url_col)
        if self.host_col:
            columns.append(self.host_col)
        return ["data"], columns

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.layout_id_col]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._web_bindings = _load_llm_web_kit_bindings()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        started = time.perf_counter()
        assignments = self._build_layout_assignments(df)
        layout_ids = [""] * len(df)
        for assignment in assignments:
            layout_ids[assignment.row_index] = assignment.layout_id
        df[self.layout_id_col] = layout_ids

        assigned_rows = sum(bool(layout_id) for layout_id in layout_ids)
        elapsed_s = time.perf_counter() - started
        self._log_metrics(
            {
                "layout_clustering_rows": float(len(df)),
                "layout_clustering_assigned_rows": float(assigned_rows),
                "layout_clustering_unassigned_rows": float(len(df) - assigned_rows),
                "layout_clustering_elapsed_s": elapsed_s,
            }
        )
        logger.info(
            "Dripper layout clustering assigned {}/{} row(s) to {} layout ID(s) in {:.3f}s",
            assigned_rows,
            len(df),
            len({layout_id for layout_id in layout_ids if layout_id}),
            elapsed_s,
        )
        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _build_layout_assignments(self, df: pd.DataFrame) -> list[_LayoutClusterAssignment]:
        assert self._web_bindings is not None
        samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in df.iterrows():
            if _DRIPPER_NEEDS_LLM_COL in df.columns and not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = DripperHTMLExtractionStage._coerce_html(row.get(self.html_col, ""))
            if not html_text.strip():
                continue
            try:
                feature = self._web_bindings.get_feature(html_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper pre-layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            samples_by_host[self._row_host_key(row)].append(
                {"track_id": str(idx), "html": html_text, "feature": feature}
            )

        assignments: list[_LayoutClusterAssignment] = []
        for host_key, samples in samples_by_host.items():
            assignments.extend(self._build_host_layout_assignments(df, host_key, samples))
        return assignments

    def _build_host_layout_assignments(
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
    ) -> list[_LayoutClusterAssignment]:
        assert self._web_bindings is not None
        if len(samples) < self.layout_template_min_cluster_size:
            return []

        grouped_samples: dict[str, list[int]] = defaultdict(list)
        if self.layout_template_max_exact_host_pages and len(samples) > self.layout_template_max_exact_host_pages:
            if self.layout_template_large_host_mode == "standalone":
                logger.debug(
                    "Dripper pre-layout host={} rows={} exceeds max_exact_host_pages={}; leaving unassigned",
                    host_key,
                    len(samples),
                    self.layout_template_max_exact_host_pages,
                )
                return []
            fingerprint_fn = (
                (lambda sample: _layout_feature_fingerprint(sample.get("feature")))
                if self.layout_template_large_host_mode == "feature_hash"
                else (lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")))
            )
            by_fingerprint: dict[str, list[int]] = defaultdict(list)
            for sample in samples:
                by_fingerprint[fingerprint_fn(sample)].append(int(sample["track_id"]))
            for fingerprint, indexes in by_fingerprint.items():
                self._add_signature_grouped_indexes(
                    df,
                    grouped_samples,
                    host_key=host_key,
                    layout_key="fingerprint",
                    fingerprint=fingerprint,
                    indexes=indexes,
                )
        else:
            try:
                clustered_samples, _layout_ids = self._web_bindings.cluster_html_struct(
                    samples,
                    threshold=self.layout_cluster_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper pre-layout clustering failed for host {}: {}", host_key, exc)
                return []
            if not clustered_samples:
                return []

            max_layer_n = int(
                next((s.get("max_layer_n") for s in clustered_samples if int(s.get("layout_id", -1)) >= 0), None)
                or 5
            )
            exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for sample in clustered_samples:
                layout_id = int(sample.get("layout_id", -1))
                if layout_id < 0:
                    continue
                if len(exemplars_by_layout[layout_id]) < 3:
                    exemplars_by_layout[layout_id].append(sample)

            for sample in clustered_samples:
                layout_id = self._assign_layout_by_exemplar_similarity(
                    sample.get("feature"),
                    exemplars_by_layout,
                    max_layer_n,
                )
                if layout_id < 0:
                    continue
                row_idx = int(sample["track_id"])
                grouped_samples[f"__pending_dom_{layout_id:06d}"].append(row_idx)

            pending_groups = [
                (key, indexes) for key, indexes in list(grouped_samples.items()) if key.startswith("__pending_dom_")
            ]
            grouped_samples.clear()
            for pending_key, indexes in pending_groups:
                self._add_signature_grouped_indexes(
                    df,
                    grouped_samples,
                    host_key=host_key,
                    layout_key=pending_key.removeprefix("__pending_"),
                    fingerprint="",
                    indexes=indexes,
                )

        assignments: list[_LayoutClusterAssignment] = []
        for layout_key, indexes in grouped_samples.items():
            if len(indexes) < self.layout_template_min_cluster_size:
                continue
            assignments.extend(_LayoutClusterAssignment(row_index=idx, layout_id=layout_key) for idx in indexes)
        return assignments

    def _assign_layout_by_exemplar_similarity(
        self,
        feature: Any,
        exemplars_by_layout: dict[int, list[dict[str, Any]]],
        max_layer_n: int,
    ) -> int:
        assert self._web_bindings is not None
        for layout_id, exemplars in sorted(exemplars_by_layout.items()):
            for exemplar in exemplars:
                try:
                    score = self._web_bindings.similarity(feature, exemplar.get("feature"), max_layer_n)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper pre-layout similarity failed for layout {}: {}", layout_id, exc)
                    continue
                if score is not None and score >= self.layout_cluster_threshold:
                    return layout_id
        return -2

    def _row_host_key(self, row: pd.Series) -> str:
        if self.host_col and self.host_col in row:
            host_key = _url_host_key(row.get(self.host_col))
            if host_key:
                return host_key
        return _url_host_key(row.get(self.url_col) if self.url_col else None)

    def _layout_page_signature_key(self, row: pd.Series) -> str:
        return _layout_page_signature_key(
            row.get(self.url_col) if self.url_col else None,
            row.get(self.item_count_col) if self.item_count_col in row else None,
            self.layout_page_signature_mode,
        )

    def _add_signature_grouped_indexes(
        self,
        df: pd.DataFrame,
        grouped_samples: dict[str, list[int]],
        *,
        host_key: str,
        layout_key: str,
        fingerprint: str,
        indexes: list[int],
    ) -> None:
        low_card_query_keys: set[str] = set()
        if "url_low_card_query_shape" in self.layout_page_signature_mode and self.url_col:
            low_card_query_keys = _low_card_query_value_keys(
                [df.iloc[row_idx].get(self.url_col) for row_idx in indexes]
            )
        for row_idx in indexes:
            row = df.iloc[row_idx]
            if "url_low_card_query_shape" in self.layout_page_signature_mode:
                signature_key = _layout_page_signature_key_with_low_card_queries(
                    row.get(self.url_col) if self.url_col else None,
                    row.get(self.item_count_col) if self.item_count_col in row else None,
                    self.layout_page_signature_mode,
                    low_card_query_keys,
                )
            else:
                signature_key = self._layout_page_signature_key(row)
            stable_layout_key = self._stable_layout_id(host_key, layout_key, fingerprint, signature_key)
            grouped_samples[stable_layout_key].append(row_idx)

    @staticmethod
    def _stable_layout_id(host_key: str, layout_key: str, fingerprint: str, signature_key: str) -> str:
        payload = "\n".join([host_key, layout_key, fingerprint, signature_key])
        digest = hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:20]
        return f"layout-{digest}"


@dataclass(kw_only=True)
class DripperHTMLLayoutTemplateStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Infer layout representatives, then propagate their template on CPU.

    This follows ccprocessor/llm-webkit's released batch parser path: pages are grouped
    by host, clustered by structural DOM features, one representative is sent
    through the Dripper LLM, and the representative's item labels are distilled
    into a structural template for sibling pages in the same layout cluster.
    """

    name: str = "DripperHTMLLayoutTemplateStage"
    client: AsyncLLMClient | None
    model_name: str
    html_col: str = "html"
    url_col: str | None = "url"
    host_col: str | None = None
    layout_id_col: str | None = None
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    generation_config: GenerationConfig | None = None
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_template_fallback_llm: bool = True
    layout_template_require_success: bool = True
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_more_noise_enable: bool = True
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_validation_signature_mode: str = "none"
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_representative_candidates: int = 1
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    layout_template_defer_fallback_llm: bool = False
    layout_template_defer_propagation: bool = False
    layout_page_signature_mode: str = "none"
    layout_template_failed_host_fallback_signature_mode: str = "none"
    layout_template_failed_layout_fallback_signature_mode: str = "none"
    layout_template_host_single_cluster_min_pages: int = 0
    layout_template_host_single_cluster_max_pages: int = 0
    layout_template_max_exact_host_pages: int = 0
    layout_template_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    layout_template_propagation_concurrency: int = 32
    dynamic_classid_similarity_threshold: float = 0.85
    health_check: bool = False
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.client is None:
            msg = "DripperHTMLLayoutTemplateStage requires a non-None 'client' (AsyncLLMClient)"
            raise ValueError(msg)
        self.model_name = self.model_name.strip()
        if not self.model_name:
            msg = "DripperHTMLLayoutTemplateStage requires a non-empty 'model_name'"
            raise ValueError(msg)
        if self.max_concurrent_requests <= 0:
            msg = "max_concurrent_requests must be positive"
            raise ValueError(msg)
        if not 0.0 < self.layout_cluster_threshold <= 1.0:
            msg = "layout_cluster_threshold must be in (0, 1]"
            raise ValueError(msg)
        if self.layout_template_min_cluster_size <= 1:
            msg = "layout_template_min_cluster_size must be greater than 1"
            raise ValueError(msg)
        if self.layout_template_max_selected_item_ratio is not None and not (
            0.0 < self.layout_template_max_selected_item_ratio <= 1.0
        ):
            msg = "layout_template_max_selected_item_ratio must be in (0, 1] when set"
            raise ValueError(msg)
        if self.layout_template_validation_rows < 0:
            msg = "layout_template_validation_rows must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_cluster_validation_rows < 0:
            msg = "layout_template_large_cluster_validation_rows must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_cluster_min_size < 0:
            msg = "layout_template_large_cluster_min_size must be non-negative"
            raise ValueError(msg)
        if self.layout_template_representative_candidates <= 0:
            msg = "layout_template_representative_candidates must be positive"
            raise ValueError(msg)
        if self.layout_template_propagation_target not in _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES:
            msg = (
                "layout_template_propagation_target must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_min_main_html_sim is not None and not (
            0.0 <= self.layout_template_min_main_html_sim <= 1.0
        ):
            msg = "layout_template_min_main_html_sim must be in [0, 1] when set"
            raise ValueError(msg)
        if not 0.0 <= self.layout_template_validation_min_content_f1 <= 1.0:
            msg = "layout_template_validation_min_content_f1 must be in [0, 1]"
            raise ValueError(msg)
        if self.layout_template_validation_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = (
                "layout_template_validation_signature_mode must be one of "
                f"{sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_min_content_length_ratio is not None and self.layout_template_min_content_length_ratio < 0:
            msg = "layout_template_min_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if self.layout_template_max_content_length_ratio is not None and self.layout_template_max_content_length_ratio < 0:
            msg = "layout_template_max_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if (
            self.layout_template_min_content_length_ratio is not None
            and self.layout_template_max_content_length_ratio is not None
            and self.layout_template_min_content_length_ratio > self.layout_template_max_content_length_ratio
        ):
            msg = "layout_template_min_content_length_ratio must be <= layout_template_max_content_length_ratio"
            raise ValueError(msg)
        if self.layout_page_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = f"layout_page_signature_mode must be one of {sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            raise ValueError(msg)
        if self.layout_template_failed_host_fallback_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = (
                "layout_template_failed_host_fallback_signature_mode must be one of "
                f"{sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_failed_layout_fallback_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = (
                "layout_template_failed_layout_fallback_signature_mode must be one of "
                f"{sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_host_single_cluster_min_pages < 0:
            msg = "layout_template_host_single_cluster_min_pages must be non-negative"
            raise ValueError(msg)
        if self.layout_template_host_single_cluster_max_pages < 0:
            msg = "layout_template_host_single_cluster_max_pages must be non-negative"
            raise ValueError(msg)
        if (
            self.layout_template_host_single_cluster_max_pages > 0
            and self.layout_template_host_single_cluster_min_pages > self.layout_template_host_single_cluster_max_pages
        ):
            msg = (
                "layout_template_host_single_cluster_min_pages must be less than or equal to "
                "layout_template_host_single_cluster_max_pages when the max is set"
            )
            raise ValueError(msg)
        if self.layout_template_max_exact_host_pages < 0:
            msg = "layout_template_max_exact_host_pages must be non-negative"
            raise ValueError(msg)
        if self.layout_template_large_host_mode not in _LAYOUT_TEMPLATE_LARGE_HOST_MODES:
            msg = (
                "layout_template_large_host_mode must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_LARGE_HOST_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_propagation_concurrency <= 0:
            msg = "layout_template_propagation_concurrency must be positive"
            raise ValueError(msg)
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            msg = f"structured_output_mode must be one of {sorted(_STRUCTURED_OUTPUT_MODES)}"
            raise ValueError(msg)
        if self.dynamic_classid_similarity_threshold <= 0:
            msg = "dynamic_classid_similarity_threshold must be positive"
            raise ValueError(msg)
        if self.worker_count is not None and self.worker_count <= 0:
            msg = "worker_count must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.worker_count

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.html_col,
            self.raw_response_col,
            self.preprocess_time_col,
            self.warning_col,
            self.item_count_col,
            self.request_max_tokens_col,
            self.simplified_html_col,
            self.mapped_html_col,
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            self.output_html_col,
            self.output_content_col,
            self.raw_response_col,
            self.inference_time_col,
            self.postprocess_time_col,
            self.total_time_col,
            self.error_col,
            self.warning_col,
            self.prompt_tokens_col,
            self.completion_tokens_col,
            self.total_tokens_col,
            "dripper_layout_cluster",
            "dripper_layout_representative",
            "dripper_layout_propagated",
            "dripper_layout_propagation_success",
            "dripper_layout_fallback_llm",
            "dripper_layout_standalone_llm",
            _DRIPPER_LAYOUT_FINALIZED_COL,
        ]
        if self.layout_template_defer_propagation:
            columns.extend(["dripper_layout_pending_propagation", "dripper_layout_mapping_json"])
        if self.layout_template_defer_fallback_llm:
            columns.extend(
                [
                    self.simplified_html_col,
                    self.mapped_html_col,
                    _DRIPPER_PROMPT_COL,
                    _DRIPPER_NEEDS_LLM_COL,
                    _DRIPPER_PRIMARY_ERROR_COL,
                    _DRIPPER_EMPTY_INPUT_COL,
                ]
            )
        if self.keep_intermediate and not self.layout_template_defer_fallback_llm:
            columns.extend([self.simplified_html_col, self.mapped_html_col])
        return ["data"], columns

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._initialized:
            return
        self._bindings = _load_mineru_html_bindings()
        self._web_bindings = _load_llm_web_kit_bindings()
        self._fallback_handler = self._bindings.get_fallback_handler(self.fallback)
        self.client.setup()  # type: ignore[union-attr]
        if self.health_check:
            self._run_health_check()
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        if self.html_col not in df.columns:
            msg = f"Input batch is missing required HTML column: {self.html_col!r}"
            raise ValueError(msg)

        results = run_async_safe(lambda: self._process_all_async(df))
        preprocess_times = _numeric_series_or_zero(df, self.preprocess_time_col)
        inference_times = pd.Series([r.inference_time_s for r in results], index=df.index)
        postprocess_times = pd.Series([r.postprocess_time_s for r in results], index=df.index)

        df[self.output_html_col] = [r.main_html for r in results]
        df[self.output_content_col] = [r.main_content for r in results]
        df[self.raw_response_col] = [r.raw_response for r in results]
        df[self.inference_time_col] = inference_times
        df[self.postprocess_time_col] = postprocess_times
        df[self.total_time_col] = preprocess_times + inference_times + postprocess_times
        df[self.error_col] = [r.error for r in results]
        df[self.warning_col] = [
            _append_warning(str(existing or ""), result.warning)
            for existing, result in zip(df.get(self.warning_col, pd.Series([""] * len(df))).tolist(), results, strict=True)
        ]
        df[self.prompt_tokens_col] = [r.prompt_tokens for r in results]
        df[self.completion_tokens_col] = [r.completion_tokens for r in results]
        df[self.total_tokens_col] = [r.total_tokens for r in results]
        df["dripper_layout_cluster"] = [r.layout_cluster for r in results]
        df["dripper_layout_representative"] = [r.layout_representative for r in results]
        df["dripper_layout_propagated"] = [r.layout_propagated for r in results]
        df["dripper_layout_propagation_success"] = [r.layout_propagation_success for r in results]
        df["dripper_layout_fallback_llm"] = [r.layout_fallback_llm for r in results]
        df["dripper_layout_standalone_llm"] = [r.layout_standalone_llm for r in results]
        df[_DRIPPER_LAYOUT_FINALIZED_COL] = [r.layout_finalized for r in results]

        if self.layout_template_defer_propagation:
            df["dripper_layout_pending_propagation"] = [r.layout_pending_propagation for r in results]
            df["dripper_layout_mapping_json"] = [r.layout_mapping_json for r in results]

        if self.layout_template_defer_fallback_llm:
            existing_primary_errors = df[_DRIPPER_PRIMARY_ERROR_COL].astype(str).tolist()
            df[_DRIPPER_NEEDS_LLM_COL] = [r.deferred_llm for r in results]
            df[_DRIPPER_PRIMARY_ERROR_COL] = [
                _append_warning(existing_error, result.primary_error)
                for existing_error, result in zip(existing_primary_errors, results, strict=True)
            ]

        drop_cols = [
            _DRIPPER_PROMPT_COL,
            _DRIPPER_NEEDS_LLM_COL,
            _DRIPPER_PRIMARY_ERROR_COL,
            _DRIPPER_EMPTY_INPUT_COL,
        ]
        if not self.layout_template_defer_fallback_llm:
            drop_cols.append(_DRIPPER_LAYOUT_FINALIZED_COL)
        else:
            drop_cols = []
        if not self.keep_intermediate and not self.layout_template_defer_fallback_llm:
            drop_cols.extend([self.simplified_html_col, self.mapped_html_col])
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        self._log_metrics(
            {
                "layout_template_rows": float(len(df)),
                "layout_template_representative_rows": float(sum(r.layout_representative for r in results)),
                "layout_template_propagated_rows": float(sum(r.layout_propagated for r in results)),
                "layout_template_success_rows": float(sum(r.layout_propagation_success for r in results)),
                "layout_template_fallback_llm_rows": float(sum(r.layout_fallback_llm for r in results)),
                "layout_template_standalone_llm_rows": float(sum(r.layout_standalone_llm for r in results)),
                "layout_template_deferred_llm_rows": float(sum(r.deferred_llm for r in results)),
                "layout_template_finalized_rows": float(sum(r.layout_finalized for r in results)),
            }
        )
        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _run_health_check(self) -> None:
        try:
            response = run_async_safe(self._query_health_check)
        except RuntimeError:
            raise
        except Exception as exc:
            msg = f"Dripper LLM health check failed: {exc}. Ensure the inference server is reachable."
            raise RuntimeError(msg) from exc
        if not response:
            msg = "Dripper LLM health check returned an empty response"
            raise RuntimeError(msg)
        logger.info("Dripper LLM health check passed")

    async def _query_health_check(self) -> str:
        extra_kwargs = self.generation_config.extra_kwargs if self.generation_config is not None else None
        generation_config = GenerationConfig(max_tokens=8, temperature=0.0, top_p=1.0, extra_kwargs=extra_kwargs)
        response = await self.client.query_model(  # type: ignore[union-attr]
            model=self.model_name,
            messages=[{"role": "user", "content": 'Return exactly: "1main"'}],
            generation_config=generation_config,
        )
        return response[0] if response else ""

    async def _process_all_async(self, df: pd.DataFrame) -> list[_LayoutTemplateRowResult]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        propagation_semaphore = asyncio.Semaphore(
            min(self.max_concurrent_requests, self.layout_template_propagation_concurrency)
        )
        inference_cache: _InferenceCache = {}
        inference_cache_lock = asyncio.Lock()
        build_started = time.perf_counter()
        layout_plans = self._build_layout_group_plans(df)
        build_elapsed_s = time.perf_counter() - build_started
        grouped_indexes = {idx for plan in layout_plans for idx in plan.indexes}
        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        logger.info(
            "Dripper layout-template built {} group plans covering {}/{} rows in {:.3f}s; standalone rows={}",
            len(layout_plans),
            len(grouped_indexes),
            len(df),
            build_elapsed_s,
            len(df) - len(grouped_indexes),
        )

        async def _handle_group_attempt(
            indexes: list[int],
            cluster_id: str,
            host_key: str,
            source: str,
            fallback_groups: tuple[list[int], ...],
            *,
            split_failed_host_fallback: bool,
        ) -> dict[int, _LayoutTemplateRowResult]:
            outcome = await self._process_layout_group_with_status(
                df,
                indexes,
                cluster_id,
                semaphore,
                propagation_semaphore,
                inference_cache,
                inference_cache_lock,
                emit_failure_fallback=not fallback_groups,
            )
            if outcome.accepted or not fallback_groups:
                return outcome.results

            logger.info(
                "Dripper layout attempt {} host={} source={} rows={} failed ({}); "
                "falling back to {} child groups",
                cluster_id,
                host_key,
                source,
                len(indexes),
                outcome.failure_reason,
                len(fallback_groups),
            )

            child_groups = list(fallback_groups)
            if split_failed_host_fallback and self.layout_template_failed_host_fallback_signature_mode != "none":
                child_groups = self._split_fallback_groups_by_signature(
                    df,
                    child_groups,
                    self.layout_template_failed_host_fallback_signature_mode,
                )
                logger.info(
                    "Dripper layout attempt {} host={} split fallback into {} groups by {}",
                    cluster_id,
                    host_key,
                    len(child_groups),
                    self.layout_template_failed_host_fallback_signature_mode,
                )

            fallback_results: dict[int, _LayoutTemplateRowResult] = {}
            fallback_grouped_indexes: set[int] = set()
            fallback_tasks = [
                _handle_group_attempt(
                    fallback_indexes,
                    f"{cluster_id}-fallback-{fallback_index:06d}",
                    host_key,
                    "fallback",
                    tuple(self._build_failed_layout_fallback_groups(df, fallback_indexes)),
                    split_failed_host_fallback=False,
                )
                for fallback_index, fallback_indexes in enumerate(child_groups)
            ]
            if fallback_tasks:
                for group_result in await asyncio.gather(*fallback_tasks):
                    fallback_results.update(group_result)
                fallback_grouped_indexes = {idx for group in child_groups for idx in group}

            standalone_tasks = [
                _handle_standalone(idx) for idx in indexes if idx not in fallback_grouped_indexes
            ]
            if standalone_tasks:
                for idx, result in await asyncio.gather(*standalone_tasks):
                    fallback_results[idx] = result
            return fallback_results

        async def _handle_plan(plan_index: int, plan: _LayoutGroupPlan) -> dict[int, _LayoutTemplateRowResult]:
            return await _handle_group_attempt(
                plan.indexes,
                f"layout-{plan_index:06d}",
                plan.host_key,
                plan.source,
                plan.fallback_groups,
                split_failed_host_fallback=True,
            )

        async def _handle_standalone(idx: int) -> tuple[int, _LayoutTemplateRowResult]:
            if self.layout_template_defer_fallback_llm:
                return idx, self._defer_row(
                    df.iloc[idx],
                    layout_standalone_llm=needs_llm[idx],
                    primary_error="layout template standalone row",
                )
            if needs_llm[idx]:
                result = await self._infer_and_postprocess_row(
                    df.iloc[idx],
                    semaphore,
                    inference_cache=inference_cache,
                    inference_cache_lock=inference_cache_lock,
                    layout_standalone_llm=True,
                )
            else:
                result = self._fallback_row(df.iloc[idx])
            return idx, result

        tasks: list[Any] = [_handle_plan(plan_index, plan) for plan_index, plan in enumerate(layout_plans)]
        tasks.extend(_handle_standalone(idx) for idx in range(len(df)) if idx not in grouped_indexes)
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results_by_index: dict[int, _LayoutTemplateRowResult] = {}
        for raw_result in raw_results:
            if isinstance(raw_result, BaseException):
                logger.error("Dripper layout-template task failed: {}", raw_result)
                continue
            if isinstance(raw_result, tuple):
                idx, result = raw_result
                results_by_index[idx] = result
            else:
                results_by_index.update(raw_result)

        return [
            results_by_index[idx] if idx in results_by_index else self._missing_layout_result(df.iloc[idx])
            for idx in range(len(df))
        ]

    def _missing_layout_result(self, row: pd.Series) -> _LayoutTemplateRowResult:
        primary_error = "layout template task produced no result"
        if self.layout_template_defer_fallback_llm:
            return self._defer_row(row, primary_error=primary_error, layout_fallback_llm=True)
        return self._fallback_row(row, primary_error=primary_error)

    def _build_layout_groups(self, df: pd.DataFrame) -> list[list[int]]:
        return [plan.indexes for plan in self._build_layout_group_plans(df)]

    def _build_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan]:
        assert self._web_bindings is not None
        if len(df) < self.layout_template_min_cluster_size:
            return []
        precomputed_plans = self._build_precomputed_layout_group_plans(df)
        if precomputed_plans is not None:
            return precomputed_plans

        samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in df.iterrows():
            if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = DripperHTMLExtractionStage._coerce_html(row.get(self.html_col, ""))
            if not html_text.strip():
                continue
            try:
                feature = self._web_bindings.get_feature(html_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            samples_by_host[self._row_host_key(row)].append(
                {"track_id": str(idx), "html": html_text, "feature": feature}
            )

        plans: list[_LayoutGroupPlan] = []
        for host_key, samples in samples_by_host.items():
            if len(samples) < self.layout_template_min_cluster_size:
                continue
            host_indexes = sorted(int(sample["track_id"]) for sample in samples)
            fallback_groups = self._build_layout_groups_for_host_samples(df, host_key, samples)
            if self._should_try_host_single_cluster(len(samples)):
                plans.append(
                    _LayoutGroupPlan(
                        indexes=host_indexes,
                        host_key=host_key,
                        source="host_single_cluster",
                        fallback_groups=tuple(fallback_groups),
                    )
                )
                logger.debug(
                    "Dripper layout host={} rows={} will try single-template host group with {} fallback groups",
                    host_key,
                    len(host_indexes),
                    len(fallback_groups),
                )
                continue
            for indexes in fallback_groups:
                plans.append(
                    _LayoutGroupPlan(
                        indexes=indexes,
                        host_key=host_key,
                        source="dom",
                        fallback_groups=tuple(self._build_failed_layout_fallback_groups(df, indexes)),
                    )
                )
        return plans

    def _build_precomputed_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan] | None:
        if not self.layout_id_col or self.layout_id_col not in df.columns:
            return None

        by_layout: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = DripperHTMLExtractionStage._coerce_html(row.get(self.html_col, ""))
            if not html_text.strip():
                continue
            layout_key = self._row_layout_id_key(row)
            if not layout_key:
                continue
            by_layout[(self._row_host_key(row), layout_key)].append(int(idx))

        plans: list[_LayoutGroupPlan] = []
        for (host_key, layout_key), indexes in sorted(by_layout.items(), key=lambda item: (min(item[1]), item[0])):
            sorted_indexes = sorted(indexes)
            if len(sorted_indexes) < self.layout_template_min_cluster_size:
                continue
            plan_groups = self._split_large_precomputed_layout_group(df, host_key, layout_key, sorted_indexes)
            for plan_indexes in plan_groups:
                if len(plan_indexes) < self.layout_template_min_cluster_size:
                    continue
                fallback_groups = self._build_failed_layout_fallback_groups(df, plan_indexes)
                plans.append(
                    _LayoutGroupPlan(
                        indexes=plan_indexes,
                        host_key=host_key,
                        source=f"precomputed_layout:{layout_key}",
                        fallback_groups=tuple(fallback_groups),
                    )
                )
        logger.info(
            "Dripper layout-template used precomputed layout column {} to build {} group plans",
            self.layout_id_col,
            len(plans),
        )
        return plans

    def _split_large_precomputed_layout_group(
        self,
        df: pd.DataFrame,
        host_key: str,
        layout_key: str,
        indexes: list[int],
    ) -> list[list[int]]:
        if not self.layout_template_max_exact_host_pages or len(indexes) <= self.layout_template_max_exact_host_pages:
            return [indexes]
        if self.layout_template_large_host_mode == "standalone":
            logger.debug(
                "Dripper precomputed layout group host={} layout={} rows={} exceeds max_exact_host_pages={}; "
                "leaving standalone",
                host_key,
                layout_key,
                len(indexes),
                self.layout_template_max_exact_host_pages,
            )
            return []

        samples: list[dict[str, Any]] = []
        for idx in indexes:
            html_text = DripperHTMLExtractionStage._coerce_html(df.iloc[idx].get(self.html_col, ""))
            if not html_text.strip():
                continue
            sample: dict[str, Any] = {"track_id": str(idx), "html": html_text}
            if self.layout_template_large_host_mode == "feature_hash":
                try:
                    feature = self._web_bindings.get_feature(html_text) if self._web_bindings else None
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Dripper precomputed layout feature extraction failed for row {}: {}",
                        idx,
                        exc,
                    )
                    continue
                if feature is None:
                    continue
                sample["feature"] = feature
            samples.append(sample)
        fingerprint_fn = (
            (lambda sample: _layout_feature_fingerprint(sample.get("feature")))
            if self.layout_template_large_host_mode == "feature_hash"
            else (lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")))
        )
        groups = self._build_fingerprint_groups(df, host_key, samples, fingerprint_fn=fingerprint_fn)
        logger.debug(
            "Dripper precomputed layout group host={} layout={} rows={} exceeded max_exact_host_pages={}; "
            "split into {} {} group(s)",
            host_key,
            layout_key,
            len(indexes),
            self.layout_template_max_exact_host_pages,
            len(groups),
            self.layout_template_large_host_mode,
        )
        return groups

    def _row_host_key(self, row: pd.Series) -> str:
        if self.host_col and self.host_col in row:
            host_key = _url_host_key(row.get(self.host_col))
            if host_key:
                return host_key
        return _url_host_key(row.get(self.url_col) if self.url_col else None)

    def _row_layout_id_key(self, row: pd.Series) -> str:
        if not self.layout_id_col:
            return ""
        value = row.get(self.layout_id_col)
        text = "" if _is_missing(value) else str(value).strip()
        if not text or text in {"-1", "-2"} or text.endswith("_-1") or text.endswith("_-2"):
            return ""
        return text

    def _should_try_host_single_cluster(self, host_pages: int) -> bool:
        if self.layout_template_host_single_cluster_min_pages <= 0:
            return False
        if host_pages < self.layout_template_host_single_cluster_min_pages:
            return False
        return not (
            self.layout_template_host_single_cluster_max_pages > 0
            and host_pages > self.layout_template_host_single_cluster_max_pages
        )

    def _build_layout_groups_for_host_samples(
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
    ) -> list[list[int]]:
        assert self._web_bindings is not None
        if len(samples) < self.layout_template_min_cluster_size:
            return []

        groups: list[list[int]] = []
        if self.layout_template_max_exact_host_pages and len(samples) > self.layout_template_max_exact_host_pages:
            if self.layout_template_large_host_mode == "feature_hash":
                groups.extend(
                    self._build_fingerprint_groups(
                        df,
                        host_key,
                        samples,
                        fingerprint_fn=lambda sample: _layout_feature_fingerprint(sample.get("feature")),
                    )
                )
            elif self.layout_template_large_host_mode == "dom_path_hash":
                groups.extend(
                    self._build_fingerprint_groups(
                        df,
                        host_key,
                        samples,
                        fingerprint_fn=lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")),
                    )
                )
            else:
                logger.debug(
                    "Dripper layout host={} rows={} exceeds max_exact_host_pages={}; leaving standalone",
                    host_key,
                    len(samples),
                    self.layout_template_max_exact_host_pages,
                )
            return groups

        try:
            clustered_samples, _layout_ids = self._web_bindings.cluster_html_struct(
                samples,
                threshold=self.layout_cluster_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper layout clustering failed for host {}: {}", host_key, exc)
            return groups

        if not clustered_samples:
            return groups

        max_layer_n = int(
            next((s.get("max_layer_n") for s in clustered_samples if int(s.get("layout_id", -1)) >= 0), None)
            or 5
        )
        exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = int(sample.get("layout_id", -1))
            if layout_id < 0:
                continue
            if len(exemplars_by_layout[layout_id]) < 3:
                exemplars_by_layout[layout_id].append(sample)

        by_layout: dict[tuple[int, str], list[int]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = self._assign_layout_by_exemplar_similarity(
                sample.get("feature"),
                exemplars_by_layout,
                max_layer_n,
            )
            if layout_id < 0:
                continue
            row_idx = int(sample["track_id"])
            signature_key = self._layout_page_signature_key(df.iloc[row_idx])
            by_layout[(layout_id, signature_key)].append(row_idx)
        for (layout_id, signature_key), indexes in sorted(by_layout.items()):
            if len(indexes) >= self.layout_template_min_cluster_size:
                groups.append(sorted(indexes))
                logger.debug(
                    "Dripper layout group host={} layout_id={} signature={} rows={}",
                    host_key,
                    layout_id,
                    signature_key,
                    len(indexes),
                )
        return groups

    def _build_failed_layout_fallback_groups(self, df: pd.DataFrame, indexes: list[int]) -> list[list[int]]:
        mode = self.layout_template_failed_layout_fallback_signature_mode
        if mode == "none" or len(indexes) < self.layout_template_min_cluster_size:
            return []

        children = self._split_fallback_groups_by_signature(df, [indexes], mode)
        parent_set = set(indexes)
        return [child for child in children if set(child) != parent_set]

    def _assign_layout_by_exemplar_similarity(
        self,
        feature: Any,
        exemplars_by_layout: dict[int, list[dict[str, Any]]],
        max_layer_n: int,
    ) -> int:
        assert self._web_bindings is not None
        for layout_id, exemplars in sorted(exemplars_by_layout.items()):
            for exemplar in exemplars:
                try:
                    score = self._web_bindings.similarity(feature, exemplar.get("feature"), max_layer_n)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Dripper layout similarity failed for layout {}: {}", layout_id, exc)
                    continue
                if score is not None and score >= self.layout_cluster_threshold:
                    return layout_id
        return -2

    def _build_fingerprint_groups(
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
        *,
        fingerprint_fn: Callable[[dict[str, Any]], str],
    ) -> list[list[int]]:
        by_fingerprint: dict[str, list[int]] = defaultdict(list)
        for sample in samples:
            by_fingerprint[fingerprint_fn(sample)].append(int(sample["track_id"]))

        groups: list[list[int]] = []
        for fingerprint, indexes in sorted(by_fingerprint.items(), key=lambda item: (min(item[1]), item[0])):
            by_signature: dict[str, list[int]] = defaultdict(list)
            for row_idx in indexes:
                signature_key = self._layout_page_signature_key(df.iloc[row_idx])
                by_signature[signature_key].append(row_idx)
            for signature_key, signature_indexes in sorted(by_signature.items()):
                if len(signature_indexes) < self.layout_template_min_cluster_size:
                    continue
                groups.append(sorted(signature_indexes))
                logger.debug(
                    "Dripper layout fingerprint group host={} signature={} rows={} fingerprint_chars={}",
                    host_key,
                    signature_key,
                    len(signature_indexes),
                    len(fingerprint),
                )
        return groups

    def _layout_page_signature_key(self, row: pd.Series) -> str:
        return _layout_page_signature_key(
            row.get(self.url_col) if self.url_col else None,
            row.get(self.item_count_col),
            self.layout_page_signature_mode,
        )

    def _split_fallback_groups_by_signature(
        self,
        df: pd.DataFrame,
        groups: list[list[int]],
        mode: str,
    ) -> list[list[int]]:
        split_groups: list[list[int]] = []
        for group in groups:
            low_card_query_keys: set[str] = set()
            if "url_low_card_query_shape" in mode and self.url_col:
                low_card_query_keys = _low_card_query_value_keys(
                    [df.iloc[row_idx].get(self.url_col) for row_idx in group]
                )
            by_signature: dict[str, list[int]] = defaultdict(list)
            for row_idx in group:
                row = df.iloc[row_idx]
                if "url_low_card_query_shape" in mode:
                    signature_key = _layout_page_signature_key_with_low_card_queries(
                        row.get(self.url_col) if self.url_col else None,
                        row.get(self.item_count_col),
                        mode,
                        low_card_query_keys,
                    )
                else:
                    signature_key = _layout_page_signature_key(
                        row.get(self.url_col) if self.url_col else None,
                        row.get(self.item_count_col),
                        mode,
                    )
                by_signature[signature_key].append(row_idx)
            for _signature, indexes in sorted(by_signature.items(), key=lambda item: (min(item[1]), item[0])):
                if len(indexes) >= self.layout_template_min_cluster_size:
                    split_groups.append(sorted(indexes))
        return split_groups

    async def _process_layout_group(
        self,
        df: pd.DataFrame,
        indexes: list[int],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
        propagation_semaphore: asyncio.Semaphore,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> dict[int, _LayoutTemplateRowResult]:
        outcome = await self._process_layout_group_with_status(
            df,
            indexes,
            cluster_id,
            semaphore,
            propagation_semaphore,
            inference_cache,
            inference_cache_lock,
            emit_failure_fallback=True,
        )
        return outcome.results

    async def _process_layout_group_with_status(
        self,
        df: pd.DataFrame,
        indexes: list[int],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
        propagation_semaphore: asyncio.Semaphore,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
        *,
        emit_failure_fallback: bool,
    ) -> _LayoutGroupOutcome:
        group_started = time.perf_counter()
        representative_indexes = self._select_representative_indexes(df, indexes)
        representative_idx: int | None = None
        representative_result: _LayoutTemplateRowResult | None = None
        mapping_data: dict[str, Any] | None = None
        candidate_results: dict[int, _LayoutTemplateRowResult] = {}
        mapping_failures: list[str] = []

        for candidate_idx in representative_indexes:
            candidate_result, candidate_mapping = await self._infer_representative_and_mapping(
                df.iloc[candidate_idx],
                semaphore,
                cluster_id,
                inference_cache,
                inference_cache_lock,
            )
            candidate_results[candidate_idx] = candidate_result
            if candidate_mapping is not None:
                representative_idx = candidate_idx
                representative_result = candidate_result
                mapping_data = candidate_mapping
                break
            mapping_failures.append(
                f"{candidate_idx}:{candidate_result.primary_error or candidate_result.warning or 'mapping failed'}"
            )

        results: dict[int, _LayoutTemplateRowResult] = {}
        for candidate_idx, candidate_result in candidate_results.items():
            is_representative = candidate_idx == representative_idx
            results[candidate_idx] = replace(
                candidate_result,
                layout_cluster=cluster_id,
                layout_representative=is_representative,
                layout_fallback_llm=not is_representative,
            )

        if mapping_data is None:
            warning = "layout template mapping failed"
            if mapping_failures:
                warning = f"{warning}: {'; '.join(mapping_failures[:3])}"
            if not emit_failure_fallback:
                return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=warning)
            fallback_indexes = [idx for idx in indexes if idx not in results]
            if self.layout_template_defer_fallback_llm:
                for idx in fallback_indexes:
                    results[idx] = self._defer_row(
                        df.iloc[idx],
                        primary_error=warning,
                        layout_cluster=cluster_id,
                        layout_fallback_llm=True,
                    )
            elif self.layout_template_fallback_llm:
                fallback_results = await asyncio.gather(
                    *(
                        self._infer_and_postprocess_row(
                            df.iloc[idx],
                            semaphore,
                            inference_cache=inference_cache,
                            inference_cache_lock=inference_cache_lock,
                            layout_cluster=cluster_id,
                            layout_fallback_llm=True,
                            primary_error=warning,
                        )
                        for idx in fallback_indexes
                    )
                )
                results.update(zip(fallback_indexes, fallback_results, strict=True))
            else:
                for idx in fallback_indexes:
                    results[idx] = replace(
                        self._fallback_row(df.iloc[idx], primary_error=warning),
                        layout_cluster=cluster_id,
                    )
            return _LayoutGroupOutcome(results=results, accepted=False, failure_reason=warning)

        fallback_tasks: list[Any] = []
        fallback_indexes: list[int] = []
        assert representative_idx is not None
        assert representative_result is not None
        sibling_indexes = [idx for idx in indexes if idx not in results]
        validation_rows = self._effective_validation_rows(len(indexes))
        validation_indexes = _select_validation_indexes(
            df,
            sibling_indexes,
            validation_rows,
            self.url_col,
            self.item_count_col,
            self.layout_template_validation_signature_mode,
        )
        validation_index_set = set(validation_indexes)
        remaining_indexes = [idx for idx in sibling_indexes if idx not in validation_index_set]
        validation_failed = False
        validation_error = ""
        if validation_indexes:
            validation_propagated_task = asyncio.gather(
                *(
                    self._propagate_layout_template_async(
                        df.iloc[idx],
                        mapping_data,
                        cluster_id,
                        propagation_semaphore,
                    )
                    for idx in validation_indexes
                )
            )
            validation_llm_task = asyncio.gather(
                *(
                    self._infer_and_postprocess_row(
                        df.iloc[idx],
                        semaphore,
                        inference_cache=inference_cache,
                        inference_cache_lock=inference_cache_lock,
                        layout_cluster=cluster_id,
                        layout_fallback_llm=True,
                        primary_error="layout template validation LLM",
                    )
                    for idx in validation_indexes
                )
            )
            validation_propagated, validation_llm_results = await asyncio.gather(
                validation_propagated_task,
                validation_llm_task,
            )
            for idx, propagated, llm_result in zip(
                validation_indexes,
                validation_propagated,
                validation_llm_results,
                strict=True,
            ):
                results[idx] = llm_result
                content_f1 = _token_f1(propagated.main_content, llm_result.main_content)
                failure_reasons = []
                if propagated.error:
                    failure_reasons.append(f"propagation_error={propagated.error[:160]}")
                if content_f1 < self.layout_template_validation_min_content_f1:
                    failure_reasons.append(f"content_f1={content_f1:.3f}")
                if failure_reasons:
                    validation_failed = True
                    validation_error = (
                        "layout template validation failed"
                        f": {' '.join(failure_reasons)}"
                        f" min={self.layout_template_validation_min_content_f1:.3f}"
                    )
            if validation_failed:
                logger.debug("Dripper layout validation failed for {}: {}", cluster_id, validation_error)
                if not emit_failure_fallback:
                    return _LayoutGroupOutcome(
                        results=results,
                        accepted=False,
                        failure_reason=validation_error,
                    )

        propagated_results = []
        if remaining_indexes and not validation_failed:
            if self.layout_template_defer_propagation:
                mapping_json = json.dumps(mapping_data, default=str)
                for idx in remaining_indexes:
                    results[idx] = _LayoutTemplateRowResult(
                        layout_cluster=cluster_id,
                        layout_pending_propagation=True,
                        layout_mapping_json=mapping_json,
                        layout_finalized=False,
                    )
                return _LayoutGroupOutcome(results=results)
            propagated_results = await asyncio.gather(
                *(
                    self._propagate_layout_template_async(
                        df.iloc[idx],
                        mapping_data,
                        cluster_id,
                        propagation_semaphore,
                    )
                    for idx in remaining_indexes
                )
            )

        for i, idx in enumerate(remaining_indexes):
            if validation_failed:
                if self.layout_template_defer_fallback_llm:
                    results[idx] = self._defer_row(
                        df.iloc[idx],
                        primary_error=validation_error,
                        layout_cluster=cluster_id,
                        layout_fallback_llm=True,
                    )
                elif self.layout_template_fallback_llm:
                    fallback_indexes.append(idx)
                    fallback_tasks.append(
                        self._infer_and_postprocess_row(
                            df.iloc[idx],
                            semaphore,
                            inference_cache=inference_cache,
                            inference_cache_lock=inference_cache_lock,
                            layout_cluster=cluster_id,
                            layout_fallback_llm=True,
                            primary_error=validation_error,
                        )
                    )
                else:
                    results[idx] = replace(
                        self._fallback_row(df.iloc[idx], primary_error=validation_error),
                        layout_cluster=cluster_id,
                )
                continue
            propagated = propagated_results[i]
            if propagated.error and self.layout_template_defer_fallback_llm:
                results[idx] = self._defer_row(
                    df.iloc[idx],
                    primary_error=propagated.error,
                    layout_cluster=cluster_id,
                    layout_fallback_llm=True,
                )
                continue
            if propagated.error and self.layout_template_fallback_llm:
                fallback_indexes.append(idx)
                fallback_tasks.append(
                    self._infer_and_postprocess_row(
                        df.iloc[idx],
                        semaphore,
                        inference_cache=inference_cache,
                        inference_cache_lock=inference_cache_lock,
                        layout_cluster=cluster_id,
                        layout_fallback_llm=True,
                        primary_error=propagated.error,
                    )
                )
                continue
            results[idx] = propagated
        if fallback_tasks:
            fallback_results = await asyncio.gather(*fallback_tasks)
            results.update(zip(fallback_indexes, fallback_results, strict=True))
        logger.info(
            "Dripper layout-template group {} rows={} representative={} propagated={} fallback_llm={} elapsed_s={:.3f}",
            cluster_id,
            len(indexes),
            representative_idx,
            sum(result.layout_propagated for result in results.values()),
            sum(result.layout_fallback_llm for result in results.values()),
            time.perf_counter() - group_started,
        )
        return _LayoutGroupOutcome(results=results)

    def _effective_validation_rows(self, cluster_size: int) -> int:
        rows = self.layout_template_validation_rows
        if (
            self.layout_template_large_cluster_validation_rows > 0
            and self.layout_template_large_cluster_min_size > 0
            and cluster_size >= self.layout_template_large_cluster_min_size
        ):
            rows = max(rows, self.layout_template_large_cluster_validation_rows)
        return rows

    async def _propagate_layout_template_async(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
    ) -> _LayoutTemplateRowResult:
        async with semaphore:
            return await asyncio.to_thread(self._propagate_layout_template, row, mapping_data, cluster_id)

    def _select_representative_indexes(self, df: pd.DataFrame, indexes: list[int]) -> list[int]:
        selected = self._select_representative_index(df, indexes)
        representative_indexes = [selected]
        if self.layout_template_representative_candidates <= 1:
            return representative_indexes

        remaining_indexes = [idx for idx in indexes if idx != selected]
        representative_indexes.extend(
            _select_validation_indexes(
                df,
                remaining_indexes,
                self.layout_template_representative_candidates - 1,
                self.url_col,
                self.item_count_col,
            )
        )
        return representative_indexes

    def _select_representative_index(self, df: pd.DataFrame, indexes: list[int]) -> int:
        assert self._web_bindings is not None
        candidates = [
            {
                "track_id": str(idx),
                "html": DripperHTMLExtractionStage._coerce_html(df.iloc[idx].get(self.html_col, "")),
            }
            for idx in indexes
        ]
        try:
            representative = self._web_bindings.select_representative_html(candidates)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper representative selection failed: {}", exc)
            representative = None
        if representative is None:
            return indexes[0]
        try:
            selected = int(representative["track_id"])
        except (KeyError, TypeError, ValueError):
            return indexes[0]
        return selected if selected in indexes else indexes[0]

    async def _infer_representative_and_mapping(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        cluster_id: str,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> tuple[_LayoutTemplateRowResult, dict[str, Any] | None]:
        assert self._bindings is not None
        assert self._web_bindings is not None
        inference_result = await self._infer_row_cached(row, semaphore, inference_cache, inference_cache_lock)
        started = time.perf_counter()
        if inference_result.primary_error:
            return self._postprocess_error_row(row, inference_result, cluster_id), None

        html_text = DripperHTMLExtractionStage._coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        case = self._build_case(row)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=inference_result.raw_response)
            case = self._bindings.parse_result(case)
            webkit_response = _labels_to_webkit_response(getattr(case.parse_result, "item_label", {}))
            case = self._bindings.extract_main_html_single(case)
            post_result = self._convert_case(case)
            mapping_data = self._web_bindings.map_parser_cls({}).parse(
                {
                    "typical_raw_tag_html": mapped_html,
                    "typical_raw_html": html_text,
                    "llm_response": webkit_response,
                }
            )
            mapping_failure_reason = ""
            if self.layout_template_require_success and mapping_data.get("typical_main_html_success") is False:
                mapping_failure_reason = "typical_main_html_success=false"
                mapping_data = None
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper representative mapping failed: {}", primary_error)
            fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
            return (
                _LayoutTemplateRowResult(
                    raw_response=inference_result.raw_response,
                    inference_time_s=inference_result.inference_time_s,
                    prompt_tokens=inference_result.prompt_tokens,
                    completion_tokens=inference_result.completion_tokens,
                    total_tokens=inference_result.total_tokens,
                    main_html=fallback_result.main_html,
                    main_content=fallback_result.main_content,
                    postprocess_time_s=time.perf_counter() - started,
                    error=fallback_result.error,
                    warning=fallback_result.warning,
                    primary_error=primary_error,
                    layout_cluster=cluster_id,
                ),
                None,
            )

        warning = post_result.warning
        if mapping_data is None:
            primary_error = f"layout template mapping failed: {mapping_failure_reason or 'template unusable'}"
            warning = _append_warning(warning, primary_error)
        else:
            primary_error = ""
            mapping_data = dict(mapping_data)
            mapping_data["_dripper_representative_content_len"] = len(str(post_result.main_content or ""))
        return (
            _LayoutTemplateRowResult(
                raw_response=inference_result.raw_response,
                inference_time_s=inference_result.inference_time_s,
                prompt_tokens=inference_result.prompt_tokens,
                completion_tokens=inference_result.completion_tokens,
                total_tokens=inference_result.total_tokens,
                main_html=post_result.main_html,
                main_content=post_result.main_content,
                postprocess_time_s=time.perf_counter() - started,
                error=post_result.error,
                warning=warning,
                primary_error=primary_error,
                layout_cluster=cluster_id,
            ),
            mapping_data,
        )

    def _propagate_layout_template(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
    ) -> _LayoutTemplateRowResult:
        assert self._bindings is not None
        assert self._web_bindings is not None
        started = time.perf_counter()
        html_text = DripperHTMLExtractionStage._coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        use_mapped_item_ids = (
            self.layout_template_propagation_target == "mapped_item_ids" and "_item_id" in mapped_html
        )
        html_source = mapped_html if use_mapped_item_ids else html_text
        try:
            task_data = dict(mapping_data)
            task_data.update(
                {
                    "html_source": html_source,
                    "dynamic_id_enable": True,
                    "dynamic_classid_enable": True,
                    "more_noise_enable": self.layout_template_more_noise_enable,
                    "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
                }
            )
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
            if self.layout_template_require_success and parts.get("main_html_success") is False:
                raise RuntimeError(
                    f"layout propagation similarity below threshold: {parts.get('main_html_sim')}"
                )
            if self.layout_template_min_main_html_sim is not None:
                main_html_sim = _coerce_optional_float(parts.get("main_html_sim"))
                if main_html_sim is not None and main_html_sim < self.layout_template_min_main_html_sim:
                    raise RuntimeError(
                        "layout propagation main_html_sim "
                        f"{main_html_sim:.3f} below {self.layout_template_min_main_html_sim:.3f}"
                    )
            main_html = str(parts.get("main_html_body") or "")
            raw_response = ""
            if use_mapped_item_ids:
                all_item_ids = _item_ids_in_html(mapped_html)
                main_item_ids = set(_item_ids_in_html(main_html))
                if not all_item_ids:
                    raise RuntimeError("layout propagation target mapped HTML has no item ids")
                if not main_item_ids:
                    raise RuntimeError("layout propagation produced no target item ids")
                selected_item_ratio = len(main_item_ids) / len(all_item_ids)
                if (
                    self.layout_template_max_selected_item_ratio is not None
                    and selected_item_ratio > self.layout_template_max_selected_item_ratio
                ):
                    raise RuntimeError(
                        "layout propagation selected item ratio "
                        f"{selected_item_ratio:.3f} exceeds "
                        f"{self.layout_template_max_selected_item_ratio:.3f}"
                    )
                raw_response = _item_id_response(all_item_ids, main_item_ids)
                post_result = self._postprocess_raw_response(row, raw_response)
            else:
                post_result = self._convert_main_html(row, main_html)
            content_ratio_error = self._propagated_content_length_ratio_error(
                post_result.main_content,
                mapping_data,
            )
            if content_ratio_error:
                raise RuntimeError(content_ratio_error)
            return _LayoutTemplateRowResult(
                raw_response=raw_response,
                main_html=post_result.main_html,
                main_content=post_result.main_content,
                postprocess_time_s=time.perf_counter() - started,
                error=post_result.error,
                warning=post_result.warning,
                layout_cluster=cluster_id,
                layout_propagated=True,
                layout_propagation_success=not bool(post_result.error),
            )
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper layout propagation failed: {}", primary_error)
            fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
            return _LayoutTemplateRowResult(
                main_html=fallback_result.main_html,
                main_content=fallback_result.main_content,
                postprocess_time_s=time.perf_counter() - started,
                error=fallback_result.error or primary_error,
                warning=fallback_result.warning,
                primary_error=primary_error,
                layout_cluster=cluster_id,
                layout_propagated=True,
            )

    def _propagated_content_length_ratio_error(
        self,
        propagated_content: Any,
        mapping_data: dict[str, Any],
    ) -> str:
        if self.layout_template_min_content_length_ratio is None and self.layout_template_max_content_length_ratio is None:
            return ""
        rep_len = _coerce_positive_int(mapping_data.get("_dripper_representative_content_len"))
        if rep_len <= 0:
            return ""
        content_len = len(str(propagated_content or ""))
        ratio = content_len / rep_len
        if (
            self.layout_template_min_content_length_ratio is not None
            and ratio < self.layout_template_min_content_length_ratio
        ):
            return (
                "layout propagation content length ratio "
                f"{ratio:.3f} below {self.layout_template_min_content_length_ratio:.3f}"
            )
        if (
            self.layout_template_max_content_length_ratio is not None
            and ratio > self.layout_template_max_content_length_ratio
        ):
            return (
                "layout propagation content length ratio "
                f"{ratio:.3f} exceeds {self.layout_template_max_content_length_ratio:.3f}"
            )
        return ""

    async def _infer_and_postprocess_row(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        *,
        inference_cache: _InferenceCache | None = None,
        inference_cache_lock: asyncio.Lock | None = None,
        layout_cluster: str = "",
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
        primary_error: str = "",
    ) -> _LayoutTemplateRowResult:
        if inference_cache is None or inference_cache_lock is None:
            inference_result = await self._infer_row(row, semaphore)
        else:
            inference_result = await self._infer_row_cached(
                row,
                semaphore,
                inference_cache,
                inference_cache_lock,
            )
        if inference_result.primary_error:
            return self._postprocess_error_row(
                row,
                inference_result,
                layout_cluster,
                layout_fallback_llm=layout_fallback_llm,
                layout_standalone_llm=layout_standalone_llm,
                primary_error=_append_warning(primary_error, inference_result.primary_error),
            )

        post_result = self._postprocess_raw_response(row, inference_result.raw_response)
        return _LayoutTemplateRowResult(
            raw_response=inference_result.raw_response,
            inference_time_s=inference_result.inference_time_s,
            prompt_tokens=inference_result.prompt_tokens,
            completion_tokens=inference_result.completion_tokens,
            total_tokens=inference_result.total_tokens,
            main_html=post_result.main_html,
            main_content=post_result.main_content,
            postprocess_time_s=post_result.postprocess_time_s,
            error=post_result.error,
            warning=_append_warning(primary_error, post_result.warning),
            layout_cluster=layout_cluster,
            layout_fallback_llm=layout_fallback_llm,
            layout_standalone_llm=layout_standalone_llm,
        )

    async def _infer_row(self, row: pd.Series, semaphore: asyncio.Semaphore) -> _DripperInferenceResult:
        prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
        row_max_tokens = _coerce_usage_int(row.get(self.request_max_tokens_col, 0))
        return await self._infer_prompt(prompt, row_max_tokens, semaphore)

    async def _infer_row_cached(
        self,
        row: pd.Series,
        semaphore: asyncio.Semaphore,
        inference_cache: _InferenceCache,
        inference_cache_lock: asyncio.Lock,
    ) -> _DripperInferenceResult:
        prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
        row_max_tokens = _coerce_usage_int(row.get(self.request_max_tokens_col, 0))
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")

        key = (prompt, row_max_tokens)
        async with inference_cache_lock:
            task = inference_cache.get(key)
            owns_request = task is None
            if task is None:
                task = asyncio.create_task(self._infer_prompt(prompt, row_max_tokens, semaphore))
                inference_cache[key] = task

        result = await task
        if owns_request:
            return result
        return replace(
            result,
            inference_time_s=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    async def _infer_prompt(
        self,
        prompt: str,
        row_max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> _DripperInferenceResult:
        if not prompt.strip():
            return _DripperInferenceResult(primary_error="empty Dripper prompt", warning="empty Dripper prompt")
        async with semaphore:
            started = time.perf_counter()
            try:
                generation_config = self.generation_config or GenerationConfig()
                if row_max_tokens > 0 and generation_config.max_tokens != row_max_tokens:
                    generation_config = replace(generation_config, max_tokens=row_max_tokens)
                generation_config = _with_structured_output_config(
                    generation_config,
                    prompt,
                    self.structured_output_mode,
                )
                raw_response, prompt_tokens, completion_tokens, total_tokens = await self._query_model_with_usage(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    generation_config=generation_config,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                logger.debug("Dripper inference failed; postprocess stage will apply fallback: {}", error)
                return _DripperInferenceResult(
                    inference_time_s=time.perf_counter() - started,
                    primary_error=error,
                    warning=error,
                )
            return _DripperInferenceResult(
                raw_response=raw_response,
                inference_time_s=time.perf_counter() - started,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    async def _query_model_with_usage(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig,
    ) -> tuple[str, int, int, int]:
        assert self.client is not None
        query_model_with_usage = getattr(self.client, "query_model_with_usage", None)
        if callable(query_model_with_usage):
            response = await query_model_with_usage(
                model=model,
                messages=messages,
                generation_config=generation_config,
            )
            contents = getattr(response, "contents", [])
            return (
                contents[0] if contents else "",
                _coerce_usage_int(getattr(response, "prompt_tokens", None)),
                _coerce_usage_int(getattr(response, "completion_tokens", None)),
                _coerce_usage_int(getattr(response, "total_tokens", None)),
            )

        response = await self.client.query_model(
            model=model,
            messages=messages,
            generation_config=generation_config,
        )
        return response[0] if response else "", 0, 0, 0

    def _postprocess_raw_response(self, row: pd.Series, raw_response: str) -> _DripperPostResult:
        assert self._bindings is not None
        started = time.perf_counter()
        case = self._build_case(row)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=raw_response)
            case = self._bindings.parse_result(case)
            case = self._bindings.extract_main_html_single(case)
            result = self._convert_case(case)
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            logger.debug("Dripper parse/extract failed, applying {} fallback: {}", self.fallback, primary_error)
            result = self._fallback_and_convert(row, primary_error=primary_error)
        return replace(result, postprocess_time_s=time.perf_counter() - started)

    def _postprocess_error_row(
        self,
        row: pd.Series,
        inference_result: _DripperInferenceResult,
        layout_cluster: str,
        *,
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
        primary_error: str = "",
    ) -> _LayoutTemplateRowResult:
        primary_error = _append_warning(primary_error, inference_result.primary_error)
        fallback_result = self._fallback_and_convert(row, primary_error=primary_error)
        return _LayoutTemplateRowResult(
            raw_response=inference_result.raw_response,
            inference_time_s=inference_result.inference_time_s,
            prompt_tokens=inference_result.prompt_tokens,
            completion_tokens=inference_result.completion_tokens,
            total_tokens=inference_result.total_tokens,
            main_html=fallback_result.main_html,
            main_content=fallback_result.main_content,
            postprocess_time_s=fallback_result.postprocess_time_s,
            error=fallback_result.error,
            warning=fallback_result.warning,
            primary_error=primary_error,
            layout_cluster=layout_cluster,
            layout_fallback_llm=layout_fallback_llm,
            layout_standalone_llm=layout_standalone_llm,
        )

    def _fallback_row(self, row: pd.Series, *, primary_error: str = "") -> _LayoutTemplateRowResult:
        result = self._fallback_and_convert(
            row,
            primary_error=_append_warning(primary_error, str(row.get(_DRIPPER_PRIMARY_ERROR_COL, "") or "")),
        )
        return _LayoutTemplateRowResult(
            main_html=result.main_html,
            main_content=result.main_content,
            postprocess_time_s=result.postprocess_time_s,
            error=result.error,
            warning=result.warning,
            primary_error=primary_error,
        )

    def _defer_row(
        self,
        row: pd.Series,
        *,
        primary_error: str = "",
        layout_cluster: str = "",
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
    ) -> _LayoutTemplateRowResult:
        needs_llm = bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
        return _LayoutTemplateRowResult(
            raw_response=str(row.get(self.raw_response_col, "") or ""),
            inference_time_s=float(row.get(self.inference_time_col, 0.0) or 0.0),
            prompt_tokens=_coerce_usage_int(row.get(self.prompt_tokens_col, 0)),
            completion_tokens=_coerce_usage_int(row.get(self.completion_tokens_col, 0)),
            total_tokens=_coerce_usage_int(row.get(self.total_tokens_col, 0)),
            error=str(row.get(self.error_col, "") or ""),
            warning=_append_warning(str(row.get(self.warning_col, "") or ""), primary_error),
            primary_error=primary_error,
            deferred_llm=needs_llm,
            layout_finalized=False,
            layout_cluster=layout_cluster,
            layout_fallback_llm=layout_fallback_llm and needs_llm,
            layout_standalone_llm=layout_standalone_llm and needs_llm,
        )

    def _build_case(self, row: pd.Series) -> Any:
        assert self._bindings is not None
        html_text = DripperHTMLExtractionStage._coerce_html(row.get(self.html_col, ""))
        url = DripperHTMLExtractionStage._coerce_optional_str(row.get(self.url_col) if self.url_col else None)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html_text, url=url))
        simplified_html = str(row.get(self.simplified_html_col, "") or "")
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _fallback_and_convert(self, row: pd.Series, *, primary_error: str = "") -> _DripperPostResult:
        started = time.perf_counter()
        case = self._build_case(row)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not DripperHTMLExtractionStage._coerce_html(
            row.get(self.html_col, "")
        ).strip():
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                warning=_append_warning(primary_error, "empty HTML input"),
            )
        fallback_result = self._apply_fallback(case, primary_error)
        case = fallback_result[0]
        if fallback_result[2]:
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                error=fallback_result[2],
                warning=fallback_result[1],
            )
        result = self._convert_case(case, warning=fallback_result[1])
        return replace(result, postprocess_time_s=time.perf_counter() - started)

    def _convert_main_html(self, row: pd.Series, main_html: str) -> _DripperPostResult:
        assert self._bindings is not None
        case = self._build_case(row)
        case.output_data = self._bindings.output_cls(main_html=main_html)
        return self._convert_case(case)

    def _convert_case(self, case: Any, *, warning: str = "") -> _DripperPostResult:
        assert self._bindings is not None
        conversion_error = ""
        try:
            self._sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
        except Exception as exc:  # noqa: BLE001
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)

        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        if main_content is None:
            main_content = ""
        error = ""
        if conversion_error:
            if DripperHTMLExtractionStage._is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error
        return _DripperPostResult(main_html=main_html, main_content=main_content, error=error, warning=warning)

    def _apply_fallback(self, case: Any, primary_error: str) -> tuple[Any, str, str]:
        assert self._bindings is not None
        try:
            case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
            return case, primary_error, ""
        except Exception as fallback_exc:  # noqa: BLE001
            if primary_error:
                return case, primary_error, f"{primary_error}; fallback failed: {fallback_exc}"
            return case, "", f"fallback failed: {fallback_exc}"

    @staticmethod
    def _sanitize_case_output_html(case: Any) -> None:
        DripperHTMLExtractionStage._sanitize_case_output_html(case)


@dataclass(kw_only=True)
class DripperHTMLExtractionPipelineStage(CompositeStage[DocumentBatch, DocumentBatch]):
    """Composite Dripper stage that decomposes into prep, inference, and postprocess."""

    name: str = "DripperHTMLExtractionPipelineStage"
    client: AsyncLLMClient | None
    model_name: str
    html_col: str = "html"
    url_col: str | None = "url"
    host_col: str | None = None
    layout_id_col: str | None = None
    output_html_col: str = "dripper_html"
    output_content_col: str = "dripper_content"
    raw_response_col: str = "dripper_response"
    preprocess_time_col: str = "dripper_preprocess_time_s"
    inference_time_col: str = "dripper_inference_time_s"
    postprocess_time_col: str = "dripper_postprocess_time_s"
    total_time_col: str = "dripper_time_s"
    error_col: str = "dripper_error"
    warning_col: str = "dripper_warning"
    item_count_col: str = "dripper_item_count"
    prompt_chars_col: str = "dripper_prompt_chars"
    request_max_tokens_col: str = "dripper_request_max_tokens"
    prompt_tokens_col: str = "dripper_prompt_tokens"
    completion_tokens_col: str = "dripper_completion_tokens"
    total_tokens_col: str = "dripper_total_tokens"
    prompt_version: str = "short_compact"
    output_format: str = "mm_md"
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    generation_config: GenerationConfig | None = None
    dynamic_max_tokens: bool = False
    dynamic_max_token_padding: int = 16
    dynamic_max_tokens_per_item: int = 6
    dynamic_min_max_tokens: int = 32
    structured_output_mode: Literal["none", "structured_outputs", "guided_regex"] = "none"
    max_concurrent_requests: int = 64
    health_check: bool = False
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    preprocess_worker_count: int | None = None
    inference_worker_count: int | None = None
    postprocess_worker_count: int | None = None
    layout_worker_count: int | None = None
    layout_template_mode: bool = False
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_template_fallback_llm: bool = True
    layout_template_require_success: bool = True
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_more_noise_enable: bool = True
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_validation_signature_mode: str = "none"
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_representative_candidates: int = 1
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    layout_template_defer_fallback_llm: bool = False
    layout_template_defer_propagation: bool = False
    layout_page_signature_mode: str = "none"
    layout_template_failed_host_fallback_signature_mode: str = "none"
    layout_template_failed_layout_fallback_signature_mode: str = "none"
    layout_template_host_single_cluster_min_pages: int = 0
    layout_template_host_single_cluster_max_pages: int = 0
    layout_template_max_exact_host_pages: int = 0
    layout_template_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    layout_template_propagation_concurrency: int = 32
    dynamic_classid_similarity_threshold: float = 0.85

    def __post_init__(self) -> None:
        super().__init__()
        if self.client is None:
            msg = "DripperHTMLExtractionPipelineStage requires a non-None 'client' (AsyncLLMClient)"
            raise ValueError(msg)
        self.model_name = self.model_name.strip()
        if not self.model_name:
            msg = "DripperHTMLExtractionPipelineStage requires a non-empty 'model_name'"
            raise ValueError(msg)
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            msg = f"structured_output_mode must be one of {sorted(_STRUCTURED_OUTPUT_MODES)}"
            raise ValueError(msg)
        if self.layout_template_propagation_concurrency <= 0:
            msg = "layout_template_propagation_concurrency must be positive"
            raise ValueError(msg)
        if self.layout_template_representative_candidates <= 0:
            msg = "layout_template_representative_candidates must be positive"
            raise ValueError(msg)
        if self.layout_template_propagation_target not in _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES:
            msg = (
                "layout_template_propagation_target must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_min_main_html_sim is not None and not (
            0.0 <= self.layout_template_min_main_html_sim <= 1.0
        ):
            msg = "layout_template_min_main_html_sim must be in [0, 1] when set"
            raise ValueError(msg)
        if self.layout_template_validation_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = (
                "layout_template_validation_signature_mode must be one of "
                f"{sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_min_content_length_ratio is not None and self.layout_template_min_content_length_ratio < 0:
            msg = "layout_template_min_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if self.layout_template_max_content_length_ratio is not None and self.layout_template_max_content_length_ratio < 0:
            msg = "layout_template_max_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if (
            self.layout_template_min_content_length_ratio is not None
            and self.layout_template_max_content_length_ratio is not None
            and self.layout_template_min_content_length_ratio > self.layout_template_max_content_length_ratio
        ):
            msg = "layout_template_min_content_length_ratio must be <= layout_template_max_content_length_ratio"
            raise ValueError(msg)
        if self.layout_template_failed_host_fallback_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = (
                "layout_template_failed_host_fallback_signature_mode must be one of "
                f"{sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_failed_layout_fallback_signature_mode not in _LAYOUT_PAGE_SIGNATURE_MODES:
            msg = (
                "layout_template_failed_layout_fallback_signature_mode must be one of "
                f"{sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_host_single_cluster_min_pages < 0:
            msg = "layout_template_host_single_cluster_min_pages must be non-negative"
            raise ValueError(msg)
        if self.layout_template_host_single_cluster_max_pages < 0:
            msg = "layout_template_host_single_cluster_max_pages must be non-negative"
            raise ValueError(msg)
        if (
            self.layout_template_host_single_cluster_max_pages > 0
            and self.layout_template_host_single_cluster_min_pages > self.layout_template_host_single_cluster_max_pages
        ):
            msg = (
                "layout_template_host_single_cluster_min_pages must be less than or equal to "
                "layout_template_host_single_cluster_max_pages when the max is set"
            )
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        preprocess_stage = DripperHTMLPreprocessStage(
            html_col=self.html_col,
            url_col=self.url_col,
            raw_response_col=self.raw_response_col,
            preprocess_time_col=self.preprocess_time_col,
            inference_time_col=self.inference_time_col,
            postprocess_time_col=self.postprocess_time_col,
            total_time_col=self.total_time_col,
            error_col=self.error_col,
            warning_col=self.warning_col,
            item_count_col=self.item_count_col,
            prompt_chars_col=self.prompt_chars_col,
            request_max_tokens_col=self.request_max_tokens_col,
            prompt_tokens_col=self.prompt_tokens_col,
            completion_tokens_col=self.completion_tokens_col,
            total_tokens_col=self.total_tokens_col,
            simplified_html_col=self.simplified_html_col,
            mapped_html_col=self.mapped_html_col,
            prompt_version=self.prompt_version,
            generation_config=self.generation_config,
            dynamic_max_tokens=self.dynamic_max_tokens,
            dynamic_max_token_padding=self.dynamic_max_token_padding,
            dynamic_max_tokens_per_item=self.dynamic_max_tokens_per_item,
            dynamic_min_max_tokens=self.dynamic_min_max_tokens,
            worker_count=self.preprocess_worker_count,
        )
        if self.layout_template_mode:
            layout_stage = DripperHTMLLayoutTemplateStage(
                client=self.client,
                model_name=self.model_name,
                html_col=self.html_col,
                url_col=self.url_col,
                host_col=self.host_col,
                layout_id_col=self.layout_id_col,
                output_html_col=self.output_html_col,
                output_content_col=self.output_content_col,
                raw_response_col=self.raw_response_col,
                preprocess_time_col=self.preprocess_time_col,
                inference_time_col=self.inference_time_col,
                postprocess_time_col=self.postprocess_time_col,
                total_time_col=self.total_time_col,
                error_col=self.error_col,
                warning_col=self.warning_col,
                item_count_col=self.item_count_col,
                request_max_tokens_col=self.request_max_tokens_col,
                prompt_tokens_col=self.prompt_tokens_col,
                completion_tokens_col=self.completion_tokens_col,
                total_tokens_col=self.total_tokens_col,
                generation_config=self.generation_config,
                structured_output_mode=self.structured_output_mode,
                max_concurrent_requests=self.max_concurrent_requests,
                fallback=self.fallback,
                output_format=self.output_format,
                keep_intermediate=self.keep_intermediate,
                simplified_html_col=self.simplified_html_col,
                mapped_html_col=self.mapped_html_col,
                layout_cluster_threshold=self.layout_cluster_threshold,
                layout_template_min_cluster_size=self.layout_template_min_cluster_size,
                layout_template_fallback_llm=self.layout_template_fallback_llm,
                layout_template_require_success=self.layout_template_require_success,
                layout_template_max_selected_item_ratio=self.layout_template_max_selected_item_ratio,
                layout_template_more_noise_enable=self.layout_template_more_noise_enable,
                layout_template_validation_rows=self.layout_template_validation_rows,
                layout_template_validation_min_content_f1=self.layout_template_validation_min_content_f1,
                layout_template_validation_signature_mode=self.layout_template_validation_signature_mode,
                layout_template_large_cluster_validation_rows=self.layout_template_large_cluster_validation_rows,
                layout_template_large_cluster_min_size=self.layout_template_large_cluster_min_size,
                layout_template_representative_candidates=self.layout_template_representative_candidates,
                layout_template_propagation_target=self.layout_template_propagation_target,
                layout_template_min_main_html_sim=self.layout_template_min_main_html_sim,
                layout_template_min_content_length_ratio=self.layout_template_min_content_length_ratio,
                layout_template_max_content_length_ratio=self.layout_template_max_content_length_ratio,
                layout_template_defer_fallback_llm=self.layout_template_defer_fallback_llm,
                layout_template_defer_propagation=self.layout_template_defer_propagation,
                layout_page_signature_mode=self.layout_page_signature_mode,
                layout_template_failed_host_fallback_signature_mode=(
                    self.layout_template_failed_host_fallback_signature_mode
                ),
                layout_template_failed_layout_fallback_signature_mode=(
                    self.layout_template_failed_layout_fallback_signature_mode
                ),
                layout_template_host_single_cluster_min_pages=self.layout_template_host_single_cluster_min_pages,
                layout_template_host_single_cluster_max_pages=self.layout_template_host_single_cluster_max_pages,
                layout_template_max_exact_host_pages=self.layout_template_max_exact_host_pages,
                layout_template_large_host_mode=self.layout_template_large_host_mode,
                layout_template_propagation_concurrency=self.layout_template_propagation_concurrency,
                dynamic_classid_similarity_threshold=self.dynamic_classid_similarity_threshold,
                health_check=self.health_check,
                worker_count=self.layout_worker_count or self.inference_worker_count,
            )
            if not self.layout_template_defer_fallback_llm:
                return [preprocess_stage, layout_stage]
            return [
                preprocess_stage,
                layout_stage,
                DripperHTMLInferenceStage(
                    client=self.client,
                    model_name=self.model_name,
                    raw_response_col=self.raw_response_col,
                    inference_time_col=self.inference_time_col,
                    warning_col=self.warning_col,
                    request_max_tokens_col=self.request_max_tokens_col,
                    prompt_tokens_col=self.prompt_tokens_col,
                    completion_tokens_col=self.completion_tokens_col,
                    total_tokens_col=self.total_tokens_col,
                    generation_config=self.generation_config,
                    structured_output_mode=self.structured_output_mode,
                    max_concurrent_requests=self.max_concurrent_requests,
                    health_check=False,
                    worker_count=self.inference_worker_count,
                ),
                DripperHTMLPostprocessStage(
                    html_col=self.html_col,
                    url_col=self.url_col,
                    output_html_col=self.output_html_col,
                    output_content_col=self.output_content_col,
                    raw_response_col=self.raw_response_col,
                    preprocess_time_col=self.preprocess_time_col,
                    inference_time_col=self.inference_time_col,
                    postprocess_time_col=self.postprocess_time_col,
                    total_time_col=self.total_time_col,
                    error_col=self.error_col,
                    warning_col=self.warning_col,
                    fallback=self.fallback,
                    output_format=self.output_format,
                    keep_intermediate=self.keep_intermediate,
                    simplified_html_col=self.simplified_html_col,
                    mapped_html_col=self.mapped_html_col,
                    worker_count=self.postprocess_worker_count,
                ),
            ]

        return [
            preprocess_stage,
            DripperHTMLInferenceStage(
                client=self.client,
                model_name=self.model_name,
                raw_response_col=self.raw_response_col,
                inference_time_col=self.inference_time_col,
                warning_col=self.warning_col,
                request_max_tokens_col=self.request_max_tokens_col,
                prompt_tokens_col=self.prompt_tokens_col,
                completion_tokens_col=self.completion_tokens_col,
                total_tokens_col=self.total_tokens_col,
                generation_config=self.generation_config,
                structured_output_mode=self.structured_output_mode,
                max_concurrent_requests=self.max_concurrent_requests,
                health_check=self.health_check,
                worker_count=self.inference_worker_count,
            ),
            DripperHTMLPostprocessStage(
                html_col=self.html_col,
                url_col=self.url_col,
                output_html_col=self.output_html_col,
                output_content_col=self.output_content_col,
                raw_response_col=self.raw_response_col,
                preprocess_time_col=self.preprocess_time_col,
                inference_time_col=self.inference_time_col,
                postprocess_time_col=self.postprocess_time_col,
                total_time_col=self.total_time_col,
                error_col=self.error_col,
                warning_col=self.warning_col,
                fallback=self.fallback,
                output_format=self.output_format,
                keep_intermediate=self.keep_intermediate,
                simplified_html_col=self.simplified_html_col,
                mapped_html_col=self.mapped_html_col,
                worker_count=self.postprocess_worker_count,
            ),
        ]


def _numeric_series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _strip_xml_incompatible_chars(value: str) -> str:
    """Remove characters that XML/HTML converters reject while preserving text."""

    def is_xml_char(char: str) -> bool:
        codepoint = ord(char)
        return (
            codepoint == 0x09
            or codepoint == 0x0A
            or codepoint == 0x0D
            or 0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        )

    return "".join(char for char in value if is_xml_char(char))


def _decode_html_bytes(html_bytes: bytes) -> str | None:
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    try:
        from charset_normalizer import detect as charset_normalizer_detect
    except ModuleNotFoundError:
        return None

    detected_encoding = charset_normalizer_detect(html_bytes)["encoding"]
    if not detected_encoding or detected_encoding == "utf-8":
        return None
    try:
        return html_bytes.decode(detected_encoding)
    except Exception:  # noqa: BLE001
        return None


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_warning(existing: str, new_warning: str) -> str:
    if not existing:
        return new_warning
    if not new_warning:
        return existing
    return f"{existing}; {new_warning}"


def _url_host_key(value: Any) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _layout_page_signature_key(url_value: Any, item_count_value: Any, mode: str) -> str:
    return _layout_page_signature_key_with_low_card_queries(url_value, item_count_value, mode, set())


def _layout_page_signature_key_with_low_card_queries(
    url_value: Any,
    item_count_value: Any,
    mode: str,
    low_card_query_keys: set[str],
) -> str:
    if not mode or mode == "none":
        return ""
    parts: list[str] = []
    if "url_low_card_query_shape" in mode:
        parts.append(f"url={_url_low_card_query_shape_key(url_value, low_card_query_keys)}")
    elif "url_semantic_shape" in mode:
        parts.append(f"url={_url_semantic_shape_key(url_value)}")
    elif "url_shape" in mode:
        parts.append(f"url={_url_shape_key(url_value)}")
    if "item_count_exact" in mode:
        parts.append(f"items={_coerce_item_count(item_count_value)}")
    elif "item_count_bucket" in mode:
        parts.append(f"items={_item_count_bucket(item_count_value)}")
    return "|".join(parts)


def _url_shape_key(value: Any) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def _url_low_card_query_shape_key(value: Any, low_card_query_keys: set[str]) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]

    include_all_query_values = bool(parsed.query) and not low_card_query_keys
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if include_all_query_values or lowered_key in low_card_query_keys or lowered_key in _LAYOUT_EXACT_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _url_semantic_shape_key(value: Any) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    normalized_segments = [_normalize_semantic_url_path_segment(segment) for segment in raw_segments]
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.lower()
        if lowered_key in _LAYOUT_SEMANTIC_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={_normalize_semantic_url_query_value(query_value)}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_semantic_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        stem, extension = segment.rsplit(".", 1)
        segment = stem
        suffix = f".{extension}"
    if (
        segment.isdigit()
        or _LAYOUT_RE_MD5.fullmatch(segment)
        or _LAYOUT_RE_SHA1.fullmatch(segment)
        or _LAYOUT_RE_UUID.fullmatch(segment)
        or _LAYOUT_RE_TIMESTAMP.fullmatch(segment)
    ):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _normalize_semantic_url_query_value(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if (
        text.isdigit()
        or _LAYOUT_RE_MD5.fullmatch(text)
        or _LAYOUT_RE_SHA1.fullmatch(text)
        or _LAYOUT_RE_UUID.fullmatch(text)
        or _LAYOUT_RE_TIMESTAMP.fullmatch(text)
    ):
        return "#num"
    return text


def _item_count_bucket(value: Any) -> str:
    count = _coerce_item_count(value)
    if count <= 0:
        return "0"
    if count <= 8:
        return str(count)
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    if count <= 128:
        return "65-128"
    return "129+"


def _coerce_item_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _coerce_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float) and value.is_integer():
        value = int(value)
        return value if value > 0 else 0
    try:
        coerced = int(float(str(value)))
    except (TypeError, ValueError):
        return 0
    return coerced if coerced > 0 else 0


def _labels_to_webkit_response(labels: Any) -> dict[str, int]:
    if not isinstance(labels, dict):
        return {}
    response: dict[str, int] = {}
    for item_id, label in labels.items():
        normalized = str(label).strip().lower()
        response[f"item_id {item_id}"] = 1 if normalized in {"main", "1", "true"} else 0
    return response


def _item_ids_in_html(html: str) -> list[str]:
    item_ids: list[str] = []
    seen: set[str] = set()
    for item_id in _ITEM_ID_RE.findall(html):
        if item_id in seen:
            continue
        seen.add(item_id)
        item_ids.append(item_id)
    return item_ids


def _item_id_response(all_item_ids: list[str], main_item_ids: set[str]) -> str:
    labels = {item_id: ("main" if item_id in main_item_ids else "other") for item_id in all_item_ids}
    if all(item_id.isdigit() for item_id in all_item_ids):
        return "".join(f"{item_id}{label}" for item_id, label in labels.items())
    return json.dumps(labels, ensure_ascii=False, separators=(",", ":"))


def _layout_feature_fingerprint(feature: Any) -> str:
    if not isinstance(feature, dict):
        return ""

    def normalize_part(part: str) -> dict[str, list[tuple[str, int]]]:
        raw_layers = feature.get(part, {})
        if not isinstance(raw_layers, dict):
            return {}
        normalized: dict[str, list[tuple[str, int]]] = {}
        for layer, values in raw_layers.items():
            if not isinstance(values, list):
                continue
            counts = Counter(str(value) for value in values)
            normalized[str(layer)] = sorted(counts.items())
        return normalized

    payload = {
        "tags": normalize_part("tags"),
        "attrs": normalize_part("attrs"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _layout_dom_path_fingerprint(html_text: str) -> str:
    try:
        from lxml.html import HTMLParser, fromstring
    except ModuleNotFoundError:
        return ""

    try:
        parser = HTMLParser(collect_ids=False, encoding="utf-8", remove_comments=True, remove_pis=True)
        root = fromstring(html_text.encode("utf-8", errors="ignore"), parser=parser)
        body_nodes = root.xpath("//body")
        root = body_nodes[0] if body_nodes else root
    except Exception:  # noqa: BLE001
        return ""

    def normalize_dynamic_attribute(value: str) -> str:
        lowered = value.strip().lower()
        if _LAYOUT_RE_MD5.fullmatch(lowered):
            return "[MD5]"
        if _LAYOUT_RE_SHA1.fullmatch(lowered):
            return "[SHA1]"
        if _LAYOUT_RE_UUID.fullmatch(lowered):
            return "[UUID]"
        if _LAYOUT_RE_TIMESTAMP.fullmatch(lowered):
            return "[TIMESTAMP]"
        return _LAYOUT_RE_NUM.sub("", lowered)

    def normalize_attr_tokens(value: str | None) -> str:
        if not value:
            return ""
        tokens = value.split()
        if len(tokens) > 1:
            normalized = [token.lower() for token in tokens if not _LAYOUT_RE_NUM.search(token)]
        else:
            normalized = [normalize_dynamic_attribute(tokens[0])] if tokens else []
        return " ".join(token for token in normalized if token)

    def walk(element: Any) -> Any:
        raw_tag = getattr(element, "tag", None)
        if not isinstance(raw_tag, str):
            return None
        tag = raw_tag.lower()
        if tag in _LAYOUT_TAGS_TO_IGNORE:
            return None
        attrs: list[tuple[str, str]] = []
        if tag not in _LAYOUT_TAGS_IGNORE_ATTR:
            class_attr = normalize_attr_tokens(element.get("class"))
            id_attr = normalize_attr_tokens(element.get("id"))
            if class_attr:
                attrs.append(("class", class_attr))
            if id_attr:
                attrs.append(("id", id_attr))
        children = [child for child in (walk(child) for child in element) if child is not None]
        return [tag, attrs, children]

    return json.dumps(walk(root), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _with_structured_output_config(
    generation_config: GenerationConfig,
    prompt: str,
    mode: str,
) -> GenerationConfig:
    if mode == "none":
        return generation_config
    item_ids = _item_ids_in_html(prompt)
    if not item_ids or not all(item_id.isdigit() for item_id in item_ids):
        return generation_config

    regex = _compact_response_regex(item_ids)
    extra_kwargs = dict(generation_config.extra_kwargs or {})
    raw_extra_body = extra_kwargs.get("extra_body")
    if raw_extra_body is None:
        extra_body: dict[str, Any] = {}
    elif isinstance(raw_extra_body, dict):
        extra_body = dict(raw_extra_body)
    else:
        logger.warning("Skipping Dripper structured output because extra_body is not a dict")
        return generation_config

    if mode == "structured_outputs":
        extra_body["structured_outputs"] = {"regex": regex}
    elif mode == "guided_regex":
        extra_body["guided_regex"] = regex
    else:
        return generation_config
    extra_kwargs["extra_body"] = extra_body
    return replace(generation_config, extra_kwargs=extra_kwargs)


def _compact_response_regex(item_ids: list[str]) -> str:
    item_pattern = "".join(f"{re.escape(item_id)}(main|other)" for item_id in item_ids)
    return f"<answer>\\s*{item_pattern}\\s*</answer>"


def _token_f1(candidate: Any, reference: Any) -> float:
    candidate_tokens = Counter(_TOKEN_RE.findall(str(candidate or "").lower()))
    reference_tokens = Counter(_TOKEN_RE.findall(str(reference or "").lower()))
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((candidate_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _select_validation_indexes(
    df: pd.DataFrame,
    indexes: list[int],
    count: int,
    url_col: str | None,
    item_count_col: str,
    signature_mode: str = "none",
) -> list[int]:
    if count <= 0 or not indexes:
        return []
    if count >= len(indexes):
        return list(indexes)
    if count == 1:
        return [indexes[-1]]

    selected: list[int] = []
    selected_set: set[int] = set()

    def add(idx: int) -> None:
        if len(selected) >= count or idx in selected_set:
            return
        selected.append(idx)
        selected_set.add(idx)

    if signature_mode and signature_mode != "none":
        low_card_query_keys: set[str] = set()
        if "url_low_card_query_shape" in signature_mode and url_col:
            low_card_query_keys = _low_card_query_value_keys([df.iloc[idx].get(url_col) for idx in indexes])
        by_signature: dict[str, list[int]] = defaultdict(list)
        for idx in indexes:
            row = df.iloc[idx]
            signature_key = _layout_page_signature_key_with_low_card_queries(
                row.get(url_col) if url_col else None,
                row.get(item_count_col) if item_count_col in row else None,
                signature_mode,
                low_card_query_keys,
            )
            by_signature[signature_key].append(idx)
        signature_groups = sorted(
            by_signature.values(),
            key=lambda group: (-len(group), _validation_sample_key(df.iloc[group[0]], group[0], url_col, item_count_col)),
        )
        for group in signature_groups:
            for idx in _select_validation_indexes(df, sorted(group), 1, url_col, item_count_col):
                add(idx)
                break
            if len(selected) >= count:
                return sorted(selected)

    add(indexes[0])
    add(indexes[-1])

    item_sorted = sorted(
        indexes,
        key=lambda idx: (_coerce_item_count(df.iloc[idx].get(item_count_col)), idx),
    )
    add(item_sorted[0])
    add(item_sorted[-1])

    if url_col:
        query_value_rows: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for idx in indexes:
            url_text = str(df.iloc[idx].get(url_col) or "")
            for key, value in _validation_query_values(url_text):
                query_value_rows[key].append((value, idx))
        for key in sorted(query_value_rows):
            entries = sorted(query_value_rows[key])
            query_positions = 4 if count >= 8 else 3
            for position in _spread_positions(len(entries), min(count, query_positions)):
                add(entries[position][1])
            if len(selected) >= count:
                return sorted(selected)

        url_sorted = sorted(indexes, key=lambda idx: (str(df.iloc[idx].get(url_col) or ""), idx))
        for position in _spread_positions(len(url_sorted), count):
            add(url_sorted[position])
            if len(selected) >= count:
                return sorted(selected)

    remaining = [idx for idx in indexes if idx not in selected_set]
    remaining.sort(key=lambda idx: _validation_sample_key(df.iloc[idx], idx, url_col, item_count_col))
    for idx in remaining:
        add(idx)
        if len(selected) >= count:
            break
    return sorted(selected)


def _spread_positions(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(slot * (length - 1) / (count - 1)) for slot in range(count)})


def _validation_query_values(url_text: str) -> list[tuple[str, str]]:
    if not url_text:
        return []
    parsed = urlparse(url_text)
    if not parsed.hostname and "://" not in url_text:
        parsed = urlparse(f"//{url_text}")
    values: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.strip().lower()
        if normalized_key:
            values.append((normalized_key, value.strip().lower()))
    return values


def _low_card_query_value_keys(url_values: list[Any], max_distinct: int = 16) -> set[str]:
    values_by_key: dict[str, set[str]] = defaultdict(set)
    for url_value in url_values:
        url_text = "" if _is_missing(url_value) else str(url_value)
        for key, value in _validation_query_values(url_text):
            values_by_key[key].add(value)
    return {key for key, values in values_by_key.items() if 1 < len(values) <= max_distinct}


def _validation_sample_key(
    row: pd.Series,
    row_index: int,
    url_col: str | None,
    item_count_col: str,
) -> tuple[int, int]:
    url_text = str(row.get(url_col) or "") if url_col else ""
    item_count = str(row.get(item_count_col) or "")
    payload = f"{url_text}\0{item_count}\0{row_index}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False), row_index


_ITEM_ID_RE = re.compile(r"""_item_id\s*=\s*["']?([^"'\s>]+)""")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LAYOUT_PAGE_SIGNATURE_MODES = {
    "none",
    "url_shape",
    "url_low_card_query_shape",
    "url_semantic_shape",
    "item_count_bucket",
    "item_count_exact",
    "url_shape_item_count_bucket",
    "url_shape_item_count_exact",
    "url_low_card_query_shape_item_count_bucket",
    "url_low_card_query_shape_item_count_exact",
    "url_semantic_shape_item_count_bucket",
    "url_semantic_shape_item_count_exact",
}
_LAYOUT_SEMANTIC_QUERY_VALUE_KEYS = {"hl", "lang", "language", "locale"}
_LAYOUT_EXACT_QUERY_VALUE_KEYS = {"id"}
_LAYOUT_TAGS_TO_IGNORE = {"script", "style", "meta", "link", "br", "noscript"}
_LAYOUT_TAGS_IGNORE_ATTR = {"a", "i", "b", "li", "tr", "td", "img", "p", "body"}
_LAYOUT_RE_MD5 = re.compile(r"^[0-9a-f]{32}$")
_LAYOUT_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LAYOUT_RE_UUID = re.compile(r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$")
_LAYOUT_RE_TIMESTAMP = re.compile(r"^\d{10,13}$")
_LAYOUT_RE_NUM = re.compile(r"\d+")
_LAYOUT_TEMPLATE_LARGE_HOST_MODES = {"standalone", "feature_hash", "dom_path_hash"}
_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES = {"raw_html", "mapped_item_ids"}
_STRUCTURED_OUTPUT_MODES = {"none", "structured_outputs", "guided_regex"}
