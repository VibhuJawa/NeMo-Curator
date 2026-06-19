from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
    _load_mineru_html_bindings,
    _MinerUHTMLBindings,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_LAYOUT_FINALIZED_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _DripperPostResult,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import (
    _append_warning,
    _coerce_html,
    _coerce_optional_str,
    _is_empty_document_error,
    _numeric_series_or_zero,
    _sanitize_case_output_html,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


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
        logger.info("DripperHTMLPostprocessStage setup complete")

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if not self._initialized:
            self.setup()

        df = batch.to_pandas().copy()
        logger.debug("Postprocess: {} rows", len(df))
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
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _postprocess_one(  # noqa: C901, PLR0912
        self, row: pd.Series, html_value: bytes | str | None, url_value: str | None
    ) -> _DripperPostResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
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
            simplified_html=_coerce_html(row.get(self.simplified_html_col, "")),
            mapped_html=_coerce_html(row.get(self.mapped_html_col, "")),
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

    def _build_case(self, *, html: str, url: str | None, simplified_html: str, mapped_html: str) -> object:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _apply_fallback(self, case: object, primary_error: str) -> tuple[object, str, str]:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        try:
            case = self._bindings.extract_main_html_fallback(case, fallback_handler=self._fallback_handler)
        except Exception as fallback_exc:  # noqa: BLE001
            if primary_error:
                return case, primary_error, f"{primary_error}; fallback failed: {fallback_exc}"
            return case, "", f"fallback failed: {fallback_exc}"
        else:
            return case, primary_error, ""

    @staticmethod
    def _sanitize_case_output_html(case: object) -> None:
        _sanitize_case_output_html(case)
