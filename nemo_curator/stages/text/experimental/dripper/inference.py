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

"""DripperHTMLInferenceStage — run Dripper LLM inference against an OpenAI-compatible client."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient

from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _STRUCTURED_OUTPUT_MODES,
    _append_warning,
    _coerce_usage_int,
    _DripperInferenceResult,
    _rebuild_batch,
    _run_dripper_health_check,
    _with_structured_output_config,
)


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
            run_async_safe(lambda: _run_dripper_health_check(self.client, self.model_name, self.generation_config))
        self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        results = run_async_safe(lambda: self._infer_all_async(df))

        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        existing_raw_responses = (
            df[self.raw_response_col].astype(str).tolist() if self.raw_response_col in df else [""] * len(df)
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
        return _rebuild_batch(batch, df)

    async def _infer_all_async(self, df: pd.DataFrame) -> list[_DripperInferenceResult]:
        sem = asyncio.Semaphore(self.max_concurrent_requests)
        prompts = df[_DRIPPER_PROMPT_COL].astype(str).tolist()
        needs_llm = df[_DRIPPER_NEEDS_LLM_COL].astype(bool).tolist()
        request_max_tokens = (
            pd.to_numeric(df[self.request_max_tokens_col], errors="coerce").fillna(0).astype(int).tolist()
            if self.request_max_tokens_col in df.columns
            else [0] * len(df)
        )

        async def _infer_one_throttled(prompt: str, row_max_tokens: int) -> _DripperInferenceResult:
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
                results[idx] = _DripperInferenceResult(
                    primary_error="empty Dripper prompt", warning="empty Dripper prompt"
                )
            else:
                grouped_indexes[(prompt, row_max_tokens)].append(idx)

        tasks = {key: _infer_one_throttled(prompt=key[0], row_max_tokens=key[1]) for key in grouped_indexes}
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
            generation_config = _with_structured_output_config(generation_config, prompt, self.structured_output_mode)
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
