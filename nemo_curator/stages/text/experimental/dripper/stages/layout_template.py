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

"""Abstract base class shared by DripperHTMLLayoutPlanStage and DripperHTMLLayoutFinalizeStage."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import (
    _coerce_positive_int,
    _item_id_response,
    _item_ids_in_html,
    _json_safe_layout_mapping_data,
    _labels_to_webkit_response,
    _layout_page_signature_key,
    _normalize_query_value_keys,
    _select_validation_indexes,
    _selected_item_ratio_from_labels,
)
from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _DRIPPER_EMPTY_INPUT_COL,
    _DRIPPER_NEEDS_LLM_COL,
    _DRIPPER_PRIMARY_ERROR_COL,
    _DRIPPER_PROMPT_COL,
    _LAYOUT_PAGE_SIGNATURE_MODES,
    _LAYOUT_RE_MD5,
    _LAYOUT_RE_NUM,
    _LAYOUT_RE_SHA1,
    _LAYOUT_RE_TIMESTAMP,
    _LAYOUT_RE_UUID,
    _LAYOUT_TAGS_IGNORE_ATTR,
    _LAYOUT_TAGS_TO_IGNORE,
    _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES,
    _LAYOUT_TEMPLATE_LARGE_HOST_MODES,
    _LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES,
    _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES,
    _STRUCTURED_OUTPUT_MODES,
    _DripperInferenceResult,
    _DripperPostResult,
    _LayoutGroupPlan,
    _LayoutTemplateRowResult,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import (
    _append_warning,
    _coerce_html,
    _coerce_optional_float,
    _coerce_optional_str,
    _coerce_usage_int,
    _is_empty_document_error,
    _is_missing,
    _sanitize_case_output_html,
    _url_host_key,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import GenerationConfig
    from nemo_curator.stages.text.experimental.dripper.stages._bindings import (
        _LLMWebKitBindings,
        _MinerUHTMLBindings,
    )

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DRIPPER_LAYOUT_ACCEPTANCE_SIGNATURE_COL = "_dripper_layout_acceptance_signature"


# ---------------------------------------------------------------------------
# Module-level fingerprint helpers
# ---------------------------------------------------------------------------


def _layout_feature_fingerprint(feature: object) -> str:
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


def _layout_dom_path_fingerprint(html_text: str) -> str:  # noqa: C901
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

    def walk(element: object) -> object:
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


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DripperHTMLLayoutTemplateStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Abstract base class for DripperHTMLLayoutPlanStage and DripperHTMLLayoutFinalizeStage.

    This class contains all shared layout-template grouping, clustering, propagation,
    and postprocessing logic.  Subclasses must implement process(), setup(), and outputs().
    """

    name: str = "DripperHTMLLayoutTemplateStage"
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
    fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura"
    output_format: str = "mm_md"
    keep_intermediate: bool = False
    simplified_html_col: str = "dripper_simplified_html"
    mapped_html_col: str = "dripper_mapped_html"
    layout_cluster_threshold: float = 0.95
    layout_template_min_cluster_size: int = 2
    layout_template_max_selected_item_ratio: float | None = 0.50
    layout_template_max_representative_selected_item_ratio: float | None = None
    layout_template_validation_rows: int = 0
    layout_template_validation_min_content_f1: float = 0.98
    layout_template_validation_signature_mode: str = "none"
    layout_template_large_cluster_validation_rows: int = 0
    layout_template_large_cluster_min_size: int = 0
    layout_template_representative_candidates: int = 1
    layout_template_feature_source: Literal["raw_html", "simpled_html", "mapped_html"] = "raw_html"
    layout_template_propagation_target: Literal["raw_html", "mapped_item_ids"] = "raw_html"
    layout_template_propagation_content_source: Literal["converted", "layout_text"] = "converted"
    layout_template_min_main_html_sim: float | None = None
    layout_template_min_content_length_ratio: float | None = None
    layout_template_max_content_length_ratio: float | None = None
    layout_page_signature_mode: str = "none"
    layout_exact_query_value_keys: str | Iterable[str] | None = None
    layout_template_failed_host_fallback_signature_mode: str = "none"
    layout_template_failed_layout_fallback_signature_mode: str = "none"
    layout_template_host_single_cluster_min_pages: int = 0
    layout_template_host_single_cluster_max_pages: int = 0
    layout_template_max_exact_host_pages: int = 0
    layout_template_large_host_mode: Literal["standalone", "feature_hash", "dom_path_hash"] = "standalone"
    layout_template_prompt_dedup_fallback_min_fraction: float = 0.0
    layout_template_min_saved_call_pages: int = 0
    layout_template_propagation_concurrency: int = 1
    dynamic_classid_similarity_threshold: float = 0.85
    worker_count: int | None = None

    _bindings: _MinerUHTMLBindings | None = field(init=False, repr=False, default=None)
    _web_bindings: _LLMWebKitBindings | None = field(init=False, repr=False, default=None)
    _fallback_handler: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
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
        if self.layout_template_max_representative_selected_item_ratio is not None and not (
            0.0 < self.layout_template_max_representative_selected_item_ratio <= 1.0
        ):
            msg = "layout_template_max_representative_selected_item_ratio must be in (0, 1] when set"
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
        if self.layout_template_feature_source not in _LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES:
            msg = f"layout_template_feature_source must be one of {sorted(_LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES)}"
            raise ValueError(msg)
        if self.layout_template_propagation_target not in _LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES:
            msg = (
                "layout_template_propagation_target must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES)}"
            )
            raise ValueError(msg)
        if self.layout_template_propagation_content_source not in _LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES:
            msg = (
                "layout_template_propagation_content_source must be one of "
                f"{sorted(_LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES)}"
            )
            raise ValueError(msg)
        if (
            self.layout_template_propagation_target == "mapped_item_ids"
            and self.layout_template_propagation_content_source == "layout_text"
        ):
            msg = "layout_template_propagation_content_source='layout_text' requires raw_html propagation target"
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
            msg = f"layout_template_validation_signature_mode must be one of {sorted(_LAYOUT_PAGE_SIGNATURE_MODES)}"
            raise ValueError(msg)
        if (
            self.layout_template_min_content_length_ratio is not None
            and self.layout_template_min_content_length_ratio < 0
        ):
            msg = "layout_template_min_content_length_ratio must be non-negative when set"
            raise ValueError(msg)
        if (
            self.layout_template_max_content_length_ratio is not None
            and self.layout_template_max_content_length_ratio < 0
        ):
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
        self.layout_exact_query_value_keys = _normalize_query_value_keys(self.layout_exact_query_value_keys)
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
            msg = f"layout_template_large_host_mode must be one of {sorted(_LAYOUT_TEMPLATE_LARGE_HOST_MODES)}"
            raise ValueError(msg)
        if not 0.0 <= self.layout_template_prompt_dedup_fallback_min_fraction <= 1.0:
            msg = "layout_template_prompt_dedup_fallback_min_fraction must be in [0, 1]"
            raise ValueError(msg)
        if self.layout_template_min_saved_call_pages < 0:
            msg = "layout_template_min_saved_call_pages must be non-negative"
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
        raise NotImplementedError

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        raise NotImplementedError

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Layout group building
    # ------------------------------------------------------------------

    def _build_layout_groups(self, df: pd.DataFrame) -> list[list[int]]:
        return [plan.indexes for plan in self._build_layout_group_plans(df)]

    def _build_layout_group_plans(self, df: pd.DataFrame) -> list[_LayoutGroupPlan]:  # noqa: C901
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        if len(df) < self.layout_template_min_cluster_size:
            return []
        precomputed_plans = self._build_precomputed_layout_group_plans(df)
        if precomputed_plans is not None:
            return precomputed_plans

        samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in df.iterrows():
            if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
                continue
            html_text = self._row_feature_html(row)
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
            dom_cluster_groups = self._build_layout_groups_for_host_samples(df, host_key, samples)
            if self._should_try_host_single_cluster(len(samples)):
                plans.append(
                    _LayoutGroupPlan(
                        indexes=host_indexes,
                        host_key=host_key,
                        source="host_single_cluster",
                    )
                )
                logger.debug(
                    "Dripper layout host={} rows={} will try single-template host group",
                    host_key,
                    len(host_indexes),
                )
                continue
            for indexes in dom_cluster_groups:
                plans.append(
                    _LayoutGroupPlan(
                        indexes=indexes,
                        host_key=host_key,
                        source="dom",
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
            html_text = _coerce_html(row.get(self.html_col, ""))
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
                plans.append(
                    _LayoutGroupPlan(
                        indexes=plan_indexes,
                        host_key=host_key,
                        source=f"precomputed_layout:{layout_key}",
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
            html_text = self._row_feature_html(df.iloc[idx])
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
        if not text or text in {"-1", "-2"} or text.endswith(("_-1", "_-2")):
            return ""
        return text

    def _row_feature_html(self, row: pd.Series) -> str:
        if self.layout_template_feature_source == "simpled_html":
            return _coerce_html(row.get(self.simplified_html_col, ""))
        if self.layout_template_feature_source == "mapped_html":
            return _coerce_html(row.get(self.mapped_html_col, ""))
        return _coerce_html(row.get(self.html_col, ""))

    def _should_try_host_single_cluster(self, host_pages: int) -> bool:
        if self.layout_template_host_single_cluster_min_pages <= 0:
            return False
        if host_pages < self.layout_template_host_single_cluster_min_pages:
            return False
        return not (
            self.layout_template_host_single_cluster_max_pages > 0
            and host_pages > self.layout_template_host_single_cluster_max_pages
        )

    def _build_layout_groups_for_host_samples(  # noqa: C901, PLR0912
        self,
        df: pd.DataFrame,
        host_key: str,
        samples: list[dict[str, Any]],
    ) -> list[list[int]]:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
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

        max_layer_n = int(clustered_samples[0].get("max_layer_n") or 5)
        exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = int(sample.get("layout_id", -1))
            if layout_id < 0:
                continue
            if len(exemplars_by_layout[layout_id]) < 3:  # noqa: PLR2004
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

    def _assign_layout_by_exemplar_similarity(
        self,
        feature: object,
        exemplars_by_layout: dict[int, list[dict[str, Any]]],
        max_layer_n: int,
    ) -> int:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        for layout_id, exemplars in exemplars_by_layout.items():
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
            exact_query_value_keys=self.layout_exact_query_value_keys,
        )

    # ------------------------------------------------------------------
    # Fallback / low-return helpers
    # ------------------------------------------------------------------

    def _low_return_fallback_reason(self, indexes: list[int]) -> str:
        threshold = self.layout_template_min_saved_call_pages
        if threshold <= 0:
            return ""
        max_saved = self._max_saved_call_pages(len(indexes))
        if max_saved >= threshold:
            return ""
        return (
            "layout template low-return fallback: max_saved_call_pages="
            f"{max_saved} threshold={threshold} rows={len(indexes)} "
            f"validation_rows={min(self._effective_validation_rows(len(indexes)), max(0, len(indexes) - 1))}"
        )

    def _max_saved_call_pages(self, cluster_size: int) -> int:
        validation_rows = min(self._effective_validation_rows(cluster_size), max(0, cluster_size - 1))
        return max(0, cluster_size - 1 - validation_rows)

    def _prompt_dedup_fallback_reason(self, df: pd.DataFrame, indexes: list[int]) -> str:
        threshold = self.layout_template_prompt_dedup_fallback_min_fraction
        if threshold <= 0.0 or len(indexes) < self.layout_template_min_cluster_size:
            return ""

        prompt_keys: list[tuple[str, int]] = []
        for idx in indexes:
            row = df.iloc[idx]
            prompt = str(row.get(_DRIPPER_PROMPT_COL, "") or "")
            if not prompt.strip():
                continue
            row_max_tokens = _coerce_usage_int(row.get(self.request_max_tokens_col, 0))
            prompt_keys.append((prompt, row_max_tokens))
        if len(prompt_keys) < self.layout_template_min_cluster_size:
            return ""

        unique_prompts = len(set(prompt_keys))
        duplicate_fraction = (len(prompt_keys) - unique_prompts) / len(prompt_keys)
        if duplicate_fraction < threshold:
            return ""
        return (
            "layout template prompt dedup fallback: duplicate_prompt_fraction="
            f"{duplicate_fraction:.3f} unique_prompt_keys={unique_prompts}/{len(prompt_keys)} "
            f"threshold={threshold:.3f}"
        )

    def _effective_validation_rows(self, cluster_size: int) -> int:
        rows = self.layout_template_validation_rows
        if (
            self.layout_template_large_cluster_validation_rows > 0
            and self.layout_template_large_cluster_min_size > 0
            and cluster_size >= self.layout_template_large_cluster_min_size
        ):
            rows = max(rows, self.layout_template_large_cluster_validation_rows)
        return rows

    # ------------------------------------------------------------------
    # Representative selection
    # ------------------------------------------------------------------

    def _select_representative_index(self, df: pd.DataFrame, indexes: list[int]) -> int:
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        candidates = [
            {
                "track_id": str(idx),
                "html": self._row_feature_html(df.iloc[idx]),
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

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    async def _propagate_layout_template_async(
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
        semaphore: asyncio.Semaphore,
    ) -> _LayoutTemplateRowResult:
        async with semaphore:
            return await asyncio.to_thread(self._propagate_layout_template, row, mapping_data, cluster_id)

    def _propagate_layout_template(  # noqa: C901, PLR0912, PLR0915
        self,
        row: pd.Series,
        mapping_data: dict[str, Any],
        cluster_id: str,
    ) -> _LayoutTemplateRowResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        started = time.perf_counter()
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        use_mapped_item_ids = (
            self.layout_template_propagation_target == "mapped_item_ids" and "_item_id" in mapped_html
        )
        html_source = mapped_html if use_mapped_item_ids else html_text
        layout_template_more_noise_enable = getattr(self, "layout_template_more_noise_enable", False)
        try:
            if self.layout_template_max_representative_selected_item_ratio is not None:
                representative_selected_item_ratio = _coerce_optional_float(
                    mapping_data.get("_dripper_representative_selected_item_ratio")
                )
                if representative_selected_item_ratio is None:
                    _msg = "layout propagation representative selected item ratio unavailable"
                    raise RuntimeError(_msg)  # noqa: TRY301
                if representative_selected_item_ratio > self.layout_template_max_representative_selected_item_ratio:
                    _msg = (
                        "layout propagation representative selected item ratio "
                        f"{representative_selected_item_ratio:.3f} exceeds "
                        f"{self.layout_template_max_representative_selected_item_ratio:.3f}"
                    )
                    raise RuntimeError(_msg)  # noqa: TRY301
            task_data = dict(mapping_data)
            task_data.setdefault("labels", mapping_data.get("_dripper_webkit_response", {}))
            task_data.update(
                {
                    "html_source": html_source,
                    "dynamic_id_enable": True,
                    "dynamic_classid_enable": True,
                    "more_noise_enable": layout_template_more_noise_enable,
                    "dynamic_classid_similarity_threshold": self.dynamic_classid_similarity_threshold,
                }
            )
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
            layout_template_require_success = getattr(self, "layout_template_require_success", True)
            if layout_template_require_success and parts.get("main_html_success") is False:
                _msg = f"layout propagation similarity below threshold: {parts.get('main_html_sim')}"
                raise RuntimeError(_msg)  # noqa: TRY301
            if self.layout_template_min_main_html_sim is not None:
                main_html_sim = _coerce_optional_float(parts.get("main_html_sim"))
                if main_html_sim is not None and main_html_sim < self.layout_template_min_main_html_sim:
                    _msg = (
                        "layout propagation main_html_sim "
                        f"{main_html_sim:.3f} below {self.layout_template_min_main_html_sim:.3f}"
                    )
                    raise RuntimeError(_msg)  # noqa: TRY301
            main_html = str(parts.get("main_html_body") or "")
            layout_text = str(parts.get("main_html") or "")
            raw_response = ""
            if use_mapped_item_ids:
                all_item_ids = _item_ids_in_html(mapped_html)
                main_item_ids = set(_item_ids_in_html(main_html))
                if not all_item_ids:
                    _msg = "layout propagation target mapped HTML has no item ids"
                    raise RuntimeError(_msg)  # noqa: TRY301
                if not main_item_ids:
                    _msg = "layout propagation produced no target item ids"
                    raise RuntimeError(_msg)  # noqa: TRY301
                selected_item_ratio = len(main_item_ids) / len(all_item_ids)
                if (
                    self.layout_template_max_selected_item_ratio is not None
                    and selected_item_ratio > self.layout_template_max_selected_item_ratio
                ):
                    _msg = (
                        "layout propagation selected item ratio "
                        f"{selected_item_ratio:.3f} exceeds "
                        f"{self.layout_template_max_selected_item_ratio:.3f}"
                    )
                    raise RuntimeError(_msg)  # noqa: TRY301
                raw_response = _item_id_response(all_item_ids, main_item_ids)
                post_result = self._postprocess_raw_response(row, raw_response)
            elif self.layout_template_propagation_content_source == "layout_text":
                if not layout_text.strip() and main_html.strip():
                    _msg = "layout propagation produced empty layout text content"
                    raise RuntimeError(_msg)  # noqa: TRY301
                post_result = _DripperPostResult(main_html=main_html, main_content=layout_text)
            else:
                post_result = self._convert_main_html(row, main_html)
            content_ratio_error = self._propagated_content_length_ratio_error(
                post_result.main_content,
                mapping_data,
            )
            if content_ratio_error:
                _msg = content_ratio_error
                raise RuntimeError(_msg)  # noqa: TRY301
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
        propagated_content: object,
        mapping_data: dict[str, Any],
    ) -> str:
        if (
            self.layout_template_min_content_length_ratio is None
            and self.layout_template_max_content_length_ratio is None
        ):
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

    # ------------------------------------------------------------------
    # Postprocessing helpers
    # ------------------------------------------------------------------

    def _postprocess_raw_response(self, row: pd.Series, raw_response: str) -> _DripperPostResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
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

    def _postprocess_error_row(  # noqa: PLR0913
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

    def _defer_row(  # noqa: PLR0913
        self,
        row: pd.Series,
        *,
        primary_error: str = "",
        layout_cluster: str = "",
        layout_fallback_llm: bool = False,
        layout_standalone_llm: bool = False,
        force_needs_llm: bool = False,
    ) -> _LayoutTemplateRowResult:
        needs_llm = force_needs_llm or bool(row.get(_DRIPPER_NEEDS_LLM_COL, False))
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

    # ------------------------------------------------------------------
    # Representative inference and mapping
    # ------------------------------------------------------------------

    def _representative_mapping_from_inference(
        self,
        row: pd.Series,
        inference_result: _DripperInferenceResult,
        cluster_id: str,
    ) -> tuple[_LayoutTemplateRowResult, dict[str, Any] | None]:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        if self._web_bindings is None:
            _msg = "_web_bindings must be initialized"
            raise RuntimeError(_msg)
        started = time.perf_counter()
        if inference_result.primary_error:
            return self._postprocess_error_row(row, inference_result, cluster_id), None
        html_text = _coerce_html(row.get(self.html_col, ""))
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        case = self._build_case(row)
        layout_template_require_success = getattr(self, "layout_template_require_success", True)
        try:
            case.generate_output = self._bindings.generate_output_cls(response=inference_result.raw_response)
            case = self._bindings.parse_result(case)
            item_labels = getattr(case.parse_result, "item_label", {})
            representative_selected_item_ratio = _selected_item_ratio_from_labels(item_labels)
            webkit_response = _labels_to_webkit_response(item_labels)
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
            if layout_template_require_success and mapping_data.get("typical_main_html_success") is False:
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
            mapping_data = _json_safe_layout_mapping_data(mapping_data)
            mapping_data["_dripper_webkit_response"] = webkit_response
            mapping_data["_dripper_representative_content_len"] = len(str(post_result.main_content or ""))
            mapping_data["_dripper_representative_selected_item_ratio"] = representative_selected_item_ratio
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

    def _build_host_template_mapping(
        self,
        df: pd.DataFrame,  # noqa: ARG002
        indexes: list[int],  # noqa: ARG002
    ) -> dict[str, dict[str, Any]] | None:
        """Return per-host mapping_data built from representative inference (synchronous helper)."""
        return None

    # ------------------------------------------------------------------
    # Case / conversion helpers
    # ------------------------------------------------------------------

    def _build_case(self, row: pd.Series) -> object:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        html_text = _coerce_html(row.get(self.html_col, ""))
        url = _coerce_optional_str(row.get(self.url_col) if self.url_col else None)
        case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html_text, url=url))
        simplified_html = str(row.get(self.simplified_html_col, "") or "")
        mapped_html = str(row.get(self.mapped_html_col, "") or "")
        if simplified_html or mapped_html:
            case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        return case

    def _fallback_and_convert(self, row: pd.Series, *, primary_error: str = "") -> _DripperPostResult:
        started = time.perf_counter()
        case = self._build_case(row)
        if bool(row.get(_DRIPPER_EMPTY_INPUT_COL, False)) or not _coerce_html(row.get(self.html_col, "")).strip():
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
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
        case = self._build_case(row)
        case.output_data = self._bindings.output_cls(main_html=main_html)
        return self._convert_case(case)

    def _convert_case(self, case: object, *, warning: str = "") -> _DripperPostResult:
        if self._bindings is None:
            _msg = "_bindings must be initialized"
            raise RuntimeError(_msg)
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
        return _DripperPostResult(main_html=main_html, main_content=main_content, error=error, warning=warning)

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
