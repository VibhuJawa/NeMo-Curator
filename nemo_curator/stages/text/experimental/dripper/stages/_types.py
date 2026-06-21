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

"""Shared dataclasses, type aliases, and module-level constants for Dripper stages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ITEM_ID_RE = re.compile(r"""_item_id\s*=\s*["']?([^"'\s>]+)""")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LAYOUT_PAGE_SIGNATURE_MODES = {
    "none",
    "url_shape",
    "url_exact_query_shape",
    "url_low_card_query_shape",
    "url_semantic_shape",
    "url_semantic_exact_query_shape",
    "url_semantic_low_card_query_shape",
    "item_count_bucket",
    "item_count_exact",
    "url_shape_item_count_bucket",
    "url_shape_item_count_exact",
    "url_exact_query_shape_item_count_bucket",
    "url_exact_query_shape_item_count_exact",
    "url_low_card_query_shape_item_count_bucket",
    "url_low_card_query_shape_item_count_exact",
    "url_semantic_shape_item_count_bucket",
    "url_semantic_shape_item_count_exact",
    "url_semantic_exact_query_shape_item_count_bucket",
    "url_semantic_exact_query_shape_item_count_exact",
    "url_semantic_low_card_query_shape_item_count_bucket",
    "url_semantic_low_card_query_shape_item_count_exact",
}
_LAYOUT_SEMANTIC_QUERY_VALUE_KEYS = {"hl", "lang", "language", "locale"}
_LAYOUT_LOW_CARD_EXACT_QUERY_VALUE_KEYS = {"id"}
_LAYOUT_EXACT_QUERY_VALUE_KEYS = {"entityid", "id"}
_LAYOUT_RE_MD5 = re.compile(r"^[0-9a-f]{32}$")
_LAYOUT_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LAYOUT_RE_UUID = re.compile(r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$")
_LAYOUT_RE_TIMESTAMP = re.compile(r"^\d{10,13}$")
_LAYOUT_TEMPLATE_FEATURE_SOURCE_MODES = {"raw_html", "simpled_html", "mapped_html"}
_LAYOUT_TEMPLATE_PROPAGATION_TARGET_MODES = {"raw_html", "mapped_item_ids"}
_LAYOUT_TEMPLATE_PROPAGATION_CONTENT_SOURCE_MODES = {"converted", "layout_text"}
_STRUCTURED_OUTPUT_MODES = {"none", "structured_outputs", "guided_regex"}

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

_DRIPPER_PROMPT_COL = "_dripper_prompt"
_DRIPPER_NEEDS_LLM_COL = "_dripper_needs_llm"
_DRIPPER_PRIMARY_ERROR_COL = "_dripper_primary_error"
_DRIPPER_EMPTY_INPUT_COL = "_dripper_empty_input"
_DRIPPER_LAYOUT_FINALIZED_COL = "_dripper_layout_finalized"
_DRIPPER_LAYOUT_FINALIZED_PUBLIC_COL = "dripper_layout_finalized"
_DRIPPER_LAYOUT_DEFERRED_LLM_COL = "dripper_layout_deferred_llm"
_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL = "_dripper_layout_pending_propagation"
_DRIPPER_LAYOUT_SPLIT_PLANNED_COL = "_dripper_layout_split_planned"
# Per-representative-row template side-table column. Holds the JSON-serialized, JSON-safe
# `mapping_data` (the exact dict the finalize feeds to `_propagate_layout_template`) for clusters
# whose validation gate PASSED; "" is the defer sentinel (validation-failed / mapping-None). Emitted
# additively by the finalize so a CPU-only Phase 2b (DripperHTMLBroadcastPropagateStage) can replay
# template propagation off-GPU using the identical mapping_data.
_DRIPPER_LAYOUT_TEMPLATE_JSON_COL = "_dripper_layout_template_json"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
    layout_validation_llm: bool = False
    layout_standalone_llm: bool = False
    # JSON-serialized, JSON-safe `mapping_data` for a representative row whose cluster passed the
    # validation gate; "" elsewhere (non-reps, validation-failed reps, mapping-None reps). The
    # finalize writes this to `_DRIPPER_LAYOUT_TEMPLATE_JSON_COL` so Phase 2b can replay propagation.
    template_json: str = ""


@dataclass(frozen=True)
class _LayoutGroupPlan:
    """A layout group to process."""

    indexes: list[int]
    host_key: str = ""
    source: str = "dom"


@dataclass(frozen=True)
class _LayoutClusterAssignment:
    """Precomputed host-bounded DOM layout assignment."""

    row_index: int
    layout_id: str
