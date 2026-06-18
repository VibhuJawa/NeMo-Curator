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

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlparse

from loguru import logger

from nemo_curator.stages.text.experimental.dripper.stages._types import (
    _ITEM_ID_RE,
    _LAYOUT_EXACT_QUERY_VALUE_KEYS,
    _LAYOUT_LOW_CARD_EXACT_QUERY_VALUE_KEYS,
    _LAYOUT_RE_MD5,
    _LAYOUT_RE_SHA1,
    _LAYOUT_RE_TIMESTAMP,
    _LAYOUT_RE_UUID,
    _LAYOUT_SEMANTIC_QUERY_VALUE_KEYS,
    _TOKEN_RE,
)
from nemo_curator.stages.text.experimental.dripper.stages._utils import _is_missing

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from nemo_curator.models.client.llm_client import GenerationConfig


def _layout_page_signature_key(
    url_value: str | None,
    item_count_value: object,
    mode: str,
    exact_query_value_keys: set[str] | None = None,
) -> str:
    return _layout_page_signature_key_with_low_card_queries(
        url_value,
        item_count_value,
        mode,
        set(),
        exact_query_value_keys=exact_query_value_keys,
    )


def _layout_page_signature_key_with_low_card_queries(
    url_value: str | None,
    item_count_value: object,
    mode: str,
    low_card_query_keys: set[str],
    *,
    exact_query_value_keys: set[str] | None = None,
) -> str:
    if not mode or mode == "none":
        return ""
    parts: list[str] = []
    if "url_semantic_low_card_query_shape" in mode:
        parts.append(
            f"url={_url_semantic_low_card_query_shape_key(url_value, low_card_query_keys, exact_query_value_keys)}"
        )
    elif "url_semantic_exact_query_shape" in mode:
        parts.append(f"url={_url_semantic_exact_query_shape_key(url_value, exact_query_value_keys)}")
    elif "url_exact_query_shape" in mode:
        parts.append(f"url={_url_exact_query_shape_key(url_value, exact_query_value_keys)}")
    elif "url_low_card_query_shape" in mode:
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


def _url_shape_key(value: object) -> str:
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


def _url_low_card_query_shape_key(value: object, low_card_query_keys: set[str]) -> str:
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
        if (
            include_all_query_values
            or lowered_key in low_card_query_keys
            or lowered_key in _LAYOUT_LOW_CARD_EXACT_QUERY_VALUE_KEYS
        ):
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _url_semantic_low_card_query_shape_key(
    value: object,
    low_card_query_keys: set[str],
    exact_query_value_keys: set[str] | None = None,
) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    normalized_segments = [_normalize_semantic_url_path_segment(segment) for segment in raw_segments]

    include_all_query_values = bool(parsed.query) and not low_card_query_keys
    exact_keys = exact_query_value_keys or _LAYOUT_EXACT_QUERY_VALUE_KEYS
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if (
            include_all_query_values
            or lowered_key in low_card_query_keys
            or lowered_key in exact_keys
            or lowered_key in _LAYOUT_SEMANTIC_QUERY_VALUE_KEYS
        ):
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _url_exact_query_shape_key(value: object, exact_query_value_keys: set[str] | None = None) -> str:
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

    query_parts = []
    exact_keys = exact_query_value_keys or _LAYOUT_EXACT_QUERY_VALUE_KEYS
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if lowered_key in exact_keys:
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_query_value_keys(keys: str | Iterable[str] | None) -> set[str]:
    if keys is None:
        return set(_LAYOUT_EXACT_QUERY_VALUE_KEYS)
    if isinstance(keys, str):
        raw_keys = keys.split(",")
    else:
        raw_keys = []
        for key in keys:
            raw_keys.extend(str(key).split(","))
    normalized = {key.strip().lower() for key in raw_keys if key.strip()}
    return normalized or set(_LAYOUT_EXACT_QUERY_VALUE_KEYS)


def _normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _url_semantic_shape_key(value: object) -> str:
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


def _url_semantic_exact_query_shape_key(value: object, exact_query_value_keys: set[str] | None = None) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    normalized_segments = [_normalize_semantic_url_path_segment(segment) for segment in raw_segments]
    exact_keys = exact_query_value_keys or _LAYOUT_EXACT_QUERY_VALUE_KEYS
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if lowered_key in exact_keys or lowered_key in _LAYOUT_SEMANTIC_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
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


def _item_count_bucket(value: object) -> str:  # noqa: PLR0911
    count = _coerce_item_count(value)
    if count <= 0:
        return "0"
    if count <= 8:  # noqa: PLR2004
        return str(count)
    if count <= 16:  # noqa: PLR2004
        return "9-16"
    if count <= 32:  # noqa: PLR2004
        return "17-32"
    if count <= 64:  # noqa: PLR2004
        return "33-64"
    if count <= 128:  # noqa: PLR2004
        return "65-128"
    return "129+"


def _coerce_item_count(value: object) -> int:
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


def _coerce_positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
        return max(0, value)
    try:
        coerced = int(float(str(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)


def _labels_to_webkit_response(labels: object) -> dict[str, int]:
    if not isinstance(labels, dict):
        return {}
    response: dict[str, int] = {}
    for item_id, label in labels.items():
        normalized = str(label).strip().lower()
        response[f"item_id {item_id}"] = 1 if normalized in {"main", "1", "true"} else 0
    return response


def _selected_item_ratio_from_labels(labels: object) -> float | None:
    if not isinstance(labels, dict) or not labels:
        return None
    selected = 0
    for label in labels.values():
        normalized = str(label).strip().lower()
        if normalized in {"main", "1", "true"}:
            selected += 1
    return selected / len(labels)


def _json_safe_layout_mapping_data(mapping_data: dict[str, Any]) -> dict[str, Any]:
    """Convert llm-webkit template data to a Ray/JSON-safe shape.

    llm-webkit's in-memory ``html_element_dict`` uses tuple keys. Its
    ``LayoutBatchParser`` also accepts a JSON string with those tuple keys
    stringified, which is safer once the mapping data crosses Ray task
    boundaries.
    """
    element_dict = mapping_data.get("html_element_dict")
    if isinstance(element_dict, dict):
        mapping_data = dict(mapping_data)
        mapping_data["html_element_dict"] = json.dumps(
            _stringify_mapping_keys(element_dict),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return mapping_data


def _stringify_mapping_keys(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _stringify_mapping_keys(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_stringify_mapping_keys(child) for child in value]
    if isinstance(value, list):
        return [_stringify_mapping_keys(child) for child in value]
    return value


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


def _token_f1(candidate: object, reference: object) -> float:
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


def _select_validation_indexes(  # noqa: C901, PLR0911, PLR0912, PLR0913
    df: pd.DataFrame,
    indexes: list[int],
    count: int,
    url_col: str | None,
    item_count_col: str,
    signature_mode: str = "none",
    *,
    exact_query_value_keys: set[str] | None = None,
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
        if _uses_low_card_query_shape(signature_mode) and url_col:
            low_card_query_keys = _low_card_query_value_keys([df.iloc[idx].get(url_col) for idx in indexes])
        by_signature: dict[str, list[int]] = defaultdict(list)
        for idx in indexes:
            row = df.iloc[idx]
            signature_key = _layout_page_signature_key_with_low_card_queries(
                row.get(url_col) if url_col else None,
                row.get(item_count_col) if item_count_col in row else None,
                signature_mode,
                low_card_query_keys,
                exact_query_value_keys=exact_query_value_keys,
            )
            by_signature[signature_key].append(idx)
        signature_groups = sorted(
            by_signature.values(),
            key=lambda group: (
                -len(group),
                _validation_sample_key(df.iloc[group[0]], group[0], url_col, item_count_col),
            ),
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
            query_positions = 4 if count >= 8 else 3  # noqa: PLR2004
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


def _uses_low_card_query_shape(mode: str) -> bool:
    return "url_low_card_query_shape" in mode or "url_semantic_low_card_query_shape" in mode


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
