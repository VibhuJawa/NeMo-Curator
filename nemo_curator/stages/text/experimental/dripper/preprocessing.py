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

"""DripperHTMLPreprocessStage and DripperHTMLPostprocessStage.

These stages split the Dripper pipeline into discrete steps:
  1. DripperHTMLPreprocessStage  — simplify HTML, build prompts
  2. DripperHTMLInferenceStage   — run LLM inference (see inference.py)
  3. DripperHTMLPostprocessStage — parse responses, extract main HTML
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig  # noqa: TC001
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _append_warning,
    _apply_fallback_extraction,
    _case_has_item_ids,
    _coerce_html,
    _coerce_optional_str,
    _count_item_ids,
    _DripperPostResult,
    _DripperPrepResult,
    _generation_config_for_item_count,
    _get_processed_attr,
    _is_empty_document_error,
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
    _numeric_series_or_zero,
    _rebuild_batch,
    _sanitize_case_output_html,
)


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

        results = [
            self._prepare_one(html_value, url_value)
            for html_value, url_value in zip(html_values, url_values, strict=False)
        ]

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
        return _rebuild_batch(batch, df)

    def _prepare_one(self, html_value: object, url_value: object) -> _DripperPrepResult:
        started = time.perf_counter()
        html = _coerce_html(html_value)
        if not html.strip():
            return _DripperPrepResult(
                empty_input=True,
                preprocess_time_s=time.perf_counter() - started,
                warning="empty HTML input",
            )

        url = _coerce_optional_str(url_value)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        simplified_html = ""
        mapped_html = ""
        item_count = 0
        try:
            case = self._bindings.simplify_single_input(case)
            simplified_html = _get_processed_attr(case, "simpled_html")
            mapped_html = _get_processed_attr(case, "map_html")
            item_count = _count_item_ids(case)
            if not _case_has_item_ids(case):
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
        return _generation_config_for_item_count(self, item_count)


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
        return _rebuild_batch(batch, df)

    def _postprocess_one(self, row: pd.Series, html_value: object, url_value: object) -> _DripperPostResult:
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
        html = _coerce_html(html_value)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not html.strip():
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                warning=warning or "empty HTML input",
            )

        url = _coerce_optional_str(url_value)
        case = self._build_case(
            html=html,
            url=url,
            simplified_html=str(row.get(self.simplified_html_col, "") or ""),
            mapped_html=str(row.get(self.mapped_html_col, "") or ""),
        )
        raw_response = str(row.get(self.raw_response_col, "") or "")
        needs_llm = bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))

        case, warning, fallback_error = self._postprocess_prepare_case(
            case,
            raw_response=raw_response,
            needs_llm=needs_llm,
            primary_error=primary_error,
            warning=warning,
        )
        if fallback_error:
            return _DripperPostResult(
                postprocess_time_s=time.perf_counter() - started,
                error=fallback_error,
                warning=warning,
            )

        conversion_error = ""
        try:
            _sanitize_case_output_html(case)
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
            if _is_empty_document_error(conversion_error) and not str(main_html).strip():
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

    def _postprocess_prepare_case(
        self,
        case: object,
        *,
        raw_response: str,
        needs_llm: bool,
        primary_error: str,
        warning: str,
    ) -> tuple[object, str, str]:
        """Parse the LLM response or apply fallback. Returns (case, warning, fallback_error)."""
        if needs_llm and raw_response:
            try:
                case.generate_output = self._bindings.generate_output_cls(response=raw_response)
                case = self._bindings.parse_result(case)
                case = self._bindings.extract_main_html_single(case)
            except Exception as exc:  # noqa: BLE001
                primary_error = _append_warning(primary_error, str(exc))
                logger.debug("Dripper parse/extract failed, applying {} fallback: {}", self.fallback, primary_error)
                fallback_result = self._apply_fallback(case, primary_error)
                warning = _append_warning(warning, fallback_result[1])
                return fallback_result[0], warning, fallback_result[2]
            return case, warning, ""
        if needs_llm and not primary_error:
            primary_error = "empty Dripper response"
        fallback_result = self._apply_fallback(case, primary_error)
        warning = _append_warning(warning, fallback_result[1])
        return fallback_result[0], warning, fallback_result[2]

    def _build_case(self, *, html: str, url: str | None, simplified_html: str, mapped_html: str) -> object:
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _apply_fallback(self, case: object, primary_error: str) -> tuple[object, str, str]:
        return _apply_fallback_extraction(self._bindings, self._fallback_handler, case, primary_error)
