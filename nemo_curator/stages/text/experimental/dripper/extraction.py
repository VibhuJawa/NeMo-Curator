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

"""DripperHTMLExtractionStage — MinerU-HTML extraction through a Curator LLM client."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig  # noqa: TC001
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient

from nemo_curator.stages.text.experimental.dripper.stage import (
    _STRUCTURED_OUTPUT_MODES,
    _append_warning,
    _case_has_item_ids,
    _coerce_html,
    _coerce_optional_str,
    _count_item_ids,
    _DripperRowResult,
    _generation_config_for_item_count,
    _get_processed_attr,
    _is_empty_document_error,
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
    _query_dripper_model,
    _rebuild_batch,
    _run_dripper_health_check,
    _sanitize_case_output_html,
    _with_structured_output_config,
)


@dataclass(kw_only=True)
class DripperHTMLExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract main HTML/content with Dripper through a Curator LLM client."""

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

        return _rebuild_batch(batch, df)

    def _run_health_check(self) -> None:
        run_async_safe(lambda: _run_dripper_health_check(self.client, self.model_name, self.generation_config))

    async def _extract_all_async(self, html_values: list[object], url_values: list[object]) -> list[_DripperRowResult]:
        sem = asyncio.Semaphore(self.max_concurrent_requests)

        async def _extract_one_throttled(html_value: object, url_value: object) -> _DripperRowResult:
            async with sem:
                return await self._extract_one_async(html_value, url_value)

        tasks = [
            _extract_one_throttled(html_value, url_value)
            for html_value, url_value in zip(html_values, url_values, strict=False)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[_DripperRowResult] = []
        for idx, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error("Dripper extraction failed for row {}: {}", idx, result)
                results.append(_DripperRowResult(error=str(result)))
            else:
                results.append(result)
        return results

    def _preprocess_case(self, case: object) -> tuple[object, int, str, str, bool]:
        """Simplify HTML, count items, build prompt. Returns (case, item_count, prompt, warning, needs_llm)."""
        case = self._bindings.simplify_single_input(case)
        item_count = _count_item_ids(case)
        if not _case_has_item_ids(case):
            case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
            return (
                case,
                item_count,
                "",
                "no _item_id attributes after simplification; used fallback without LLM",
                False,
            )
        case = self._bindings.build_prompt(case, prompt_version=self.prompt_version)
        prompt = case.generate_input.full_prompt
        return case, item_count, prompt, "", True

    async def _run_inference_async(
        self, case: object, prompt: str, item_count: int
    ) -> tuple[object, str, int, int, int, int]:
        """Run inference and postprocess. Returns (case, raw_response, request_max_tokens, prompt_tokens, completion_tokens, total_tokens)."""
        generation_config = _with_structured_output_config(
            self._generation_config_for_item_count(item_count), prompt, self.structured_output_mode
        )
        request_max_tokens = generation_config.max_tokens or 0
        raw_response, prompt_tokens, completion_tokens, total_tokens = await _query_dripper_model(
            self.client, self.model_name, [{"role": "user", "content": prompt}], generation_config
        )
        case.generate_output = self._bindings.generate_output_cls(response=raw_response)
        case = self._bindings.parse_result(case)
        case = self._bindings.extract_main_html_single(case)
        return case, raw_response, request_max_tokens, prompt_tokens, completion_tokens, total_tokens

    async def _extract_one_async(self, html_value: object, url_value: object) -> _DripperRowResult:
        start_total = time.perf_counter()
        html = _coerce_html(html_value)
        if not html.strip():
            return _DripperRowResult(total_time_s=time.perf_counter() - start_total, warning="empty HTML input")

        url = _coerce_optional_str(url_value)
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
            case, item_count, prompt, warning, needs_llm = self._preprocess_case(case)
            preprocess_time_s = time.perf_counter() - start_preprocess
            if needs_llm:
                prompt_chars = len(prompt)
                start_inference = time.perf_counter()
                (
                    case,
                    raw_response,
                    request_max_tokens,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                ) = await self._run_inference_async(case, prompt, item_count)
                inference_time_s = time.perf_counter() - start_inference
                start_postprocess = time.perf_counter()
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
                    raw_response=raw_response,
                    preprocess_time_s=preprocess_time_s,
                    inference_time_s=inference_time_s,
                    postprocess_time_s=postprocess_time_s,
                    total_time_s=time.perf_counter() - start_total,
                    error=error,
                    warning=primary_error,
                    simplified_html=_get_processed_attr(case, "simpled_html"),
                    mapped_html=_get_processed_attr(case, "map_html"),
                    item_count=item_count,
                    prompt_chars=prompt_chars,
                    request_max_tokens=request_max_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

        conversion_error, postprocess_time_s = self._convert_extraction_output(case, postprocess_time_s)
        base = _DripperRowResult(
            raw_response=raw_response,
            preprocess_time_s=preprocess_time_s,
            inference_time_s=inference_time_s,
            postprocess_time_s=postprocess_time_s,
            total_time_s=time.perf_counter() - start_total,
            warning=warning,
            simplified_html=_get_processed_attr(case, "simpled_html"),
            mapped_html=_get_processed_attr(case, "map_html"),
            item_count=item_count,
            prompt_chars=prompt_chars,
            request_max_tokens=request_max_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return self._build_extraction_result(case, base, conversion_error=conversion_error)

    def _convert_extraction_output(self, case: object, postprocess_time_s: float) -> tuple[str, float]:
        conversion_error = ""
        start_conversion = time.perf_counter()
        try:
            _sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=self.output_format)
            postprocess_time_s += time.perf_counter() - start_conversion
        except Exception as exc:  # noqa: BLE001
            postprocess_time_s += time.perf_counter() - start_conversion
            conversion_error = str(exc)
            logger.debug("Dripper content conversion failed: {}", conversion_error)
        return conversion_error, postprocess_time_s

    def _build_extraction_result(
        self, case: object, base: _DripperRowResult, *, conversion_error: str
    ) -> _DripperRowResult:
        output_data = getattr(case, "output_data", None)
        main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
        main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
        if main_content is None:
            main_content = ""
        error = ""
        warning = base.warning
        if conversion_error:
            if _is_empty_document_error(conversion_error) and not str(main_html).strip():
                warning = _append_warning(warning, conversion_error)
            else:
                error = conversion_error
        return replace(base, main_html=main_html, main_content=main_content, error=error, warning=warning)

    def _generation_config_for_item_count(self, item_count: int) -> GenerationConfig:
        return _generation_config_for_item_count(self, item_count)
