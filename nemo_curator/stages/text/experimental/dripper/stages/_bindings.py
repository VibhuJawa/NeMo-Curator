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

"""Lazy runtime bindings to optional MinerU-HTML and llm-webkit libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _always_similar(_left: object, _right: object, _max_layer_n: int) -> float:
    return 1.0


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


@dataclass(frozen=True)
class _LLMWebKitBindings:
    """Runtime bindings to ccprocessor/llm-webkit layout-template algorithms."""

    get_feature: Callable[[str], Any]
    cluster_html_struct: Callable[..., Any]
    select_representative_html: Callable[[list[dict[str, str]]], dict[str, str] | None]
    map_parser_cls: type
    layout_parser_cls: type
    similarity: Callable[..., float] = _always_similar


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
            "Dripper stages require the optional 'mineru_html' package. "
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
