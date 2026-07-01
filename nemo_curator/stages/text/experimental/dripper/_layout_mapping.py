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

"""Build Dripper layout mapping templates from representative LLM responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemo_curator.stages.text.experimental.dripper._html_compression import get_html_from_row
from nemo_curator.stages.text.experimental.dripper._layout_planning import _labels_to_webkit_response
from nemo_curator.stages.text.experimental.dripper.stage import (
    _LLMWebKitBindings,
    _MinerUHTMLBindings,
    _coerce_html,
    _coerce_optional_str,
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
)


@dataclass(frozen=True)
class LayoutMappingBuildResult:
    mapping_data: dict[str, Any] | None = None
    error: str = ""


def build_layout_mapping_data(
    row: pd.Series | dict[str, Any],
    *,
    bindings: _MinerUHTMLBindings | None = None,
    web_bindings: _LLMWebKitBindings | None = None,
) -> LayoutMappingBuildResult:
    """Build LayoutBatchParser template data for one representative row."""

    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    raw_response = _first_nonempty(row_dict, ("dripper_response", "llm_response", "llm_output_raw"))
    if not raw_response:
        return LayoutMappingBuildResult(error="empty Dripper response")
    html_text = _coerce_html(get_html_from_row(row_dict))
    mapped_html = _first_nonempty(row_dict, ("dripper_mapped_html", "map_html", "mapped_html"))
    if not html_text.strip():
        return LayoutMappingBuildResult(error="empty HTML input")
    if not mapped_html.strip():
        return LayoutMappingBuildResult(error="missing mapped HTML")

    bindings = bindings or _load_mineru_html_bindings()
    web_bindings = web_bindings or _load_llm_web_kit_bindings()
    url = _coerce_optional_str(row_dict.get("url"))
    simplified_html = _first_nonempty(row_dict, ("dripper_simplified_html", "simp_html", "simplified_html"))

    try:
        case = bindings.case_cls(bindings.input_cls(raw_html=html_text, url=url))
        if simplified_html or mapped_html:
            case.process_data = bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
        case.generate_output = bindings.generate_output_cls(response=raw_response)
        case = bindings.parse_result(case)
        webkit_response = _labels_to_webkit_response(getattr(case.parse_result, "item_label", {}))
        mapping_data = web_bindings.map_parser_cls({}).parse(
            {
                "typical_raw_tag_html": mapped_html,
                "typical_raw_html": html_text,
                "llm_response": webkit_response,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return LayoutMappingBuildResult(error=str(exc))

    out = dict(mapping_data)
    content = _first_nonempty(row_dict, ("dripper_content", "main_content"))
    if content:
        out["_dripper_representative_content_len"] = len(content)
    return LayoutMappingBuildResult(mapping_data=out)


def _first_nonempty(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""
