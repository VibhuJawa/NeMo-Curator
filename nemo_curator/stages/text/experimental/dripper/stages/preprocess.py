from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _DripperPrepResult,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import (
    _case_has_item_ids,
    _coerce_html,
    _coerce_optional_str,
    _count_item_ids,
    _get_processed_attr,
    compress_html,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


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
        logger.info("DripperHTMLPreprocessStage setup complete")

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        logger.debug("Preprocess: {} rows", len(df))
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
        # Store HTML columns COMPRESSED (zlib -> bytes / large_binary): shrinks the dataframe ~4x,
        # keeps the post-preprocess _raw + the head-node compaction read small (avoids the mega-host
        # object-store wedge AND the large_string 2GB-offset cast), and every consumer goes through
        # _coerce_html which transparently decompresses. `html` is re-stored compressed too since it
        # persists through clustering/postprocess. Output columns stay uncompressed (deliverable).
        df[self.html_col] = [compress_html(h) for h in html_values]
        df[self.simplified_html_col] = [compress_html(r.simplified_html) for r in results]
        df[self.mapped_html_col] = [compress_html(r.mapped_html) for r in results]
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
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _prepare_one(self, html_value: bytes | str | None, url_value: str | None) -> _DripperPrepResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
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
        base = self.generation_config or GenerationConfig()
        if not self.dynamic_max_tokens or base.max_tokens is None or item_count <= 0:
            return base

        dynamic_max_tokens = max(
            self.dynamic_min_max_tokens,
            item_count * self.dynamic_max_tokens_per_item + self.dynamic_max_token_padding,
        )
        return replace(base, max_tokens=min(base.max_tokens, dynamic_max_tokens))
