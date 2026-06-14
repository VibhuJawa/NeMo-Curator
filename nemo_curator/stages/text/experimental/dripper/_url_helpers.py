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

"""Pure stateless helpers for the Dripper layout pipeline.

Contains URL-parsing / page-signature helpers, DOM fingerprinting utilities,
and miscellaneous pure functions extracted from layout_template.py to keep
that module below 1 900 lines.  None of these functions reference layout
dataclasses or the DripperHTMLLayoutTemplateStage class.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import parse_qsl, urlparse

from nemo_curator.stages.text.experimental.dripper.stage import _is_missing

# ---------------------------------------------------------------------------
# Compiled regex patterns (shared by URL helpers and DOM helpers)
# ---------------------------------------------------------------------------

_LAYOUT_RE_MD5 = re.compile(r"^[0-9a-f]{32}$")
_LAYOUT_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LAYOUT_RE_UUID = re.compile(r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$")
_LAYOUT_RE_TIMESTAMP = re.compile(r"^\d{10,13}$")
_LAYOUT_RE_NUM = re.compile(r"\d+")

# ---------------------------------------------------------------------------
# Domain-knowledge constants
# ---------------------------------------------------------------------------

# Item count bucket thresholds: (upper_bound, label) where label=None means str(count)
_ITEM_COUNT_BUCKET_THRESHOLDS = [(8, None), (16, "9-16"), (32, "17-32"), (64, "33-64"), (128, "65-128")]

_LAYOUT_SEMANTIC_QUERY_VALUE_KEYS = {"hl", "lang", "language", "locale"}
_LAYOUT_EXACT_QUERY_VALUE_KEYS = {"id"}

_LAYOUT_PAGE_SIGNATURE_MODES = {
    "none",
    "url_shape",
    "url_low_card_query_shape",
    "url_semantic_shape",
    "item_count_bucket",
    "item_count_exact",
    "url_shape_item_count_bucket",
    "url_shape_item_count_exact",
    "url_low_card_query_shape_item_count_bucket",
    "url_low_card_query_shape_item_count_exact",
    "url_semantic_shape_item_count_bucket",
    "url_semantic_shape_item_count_exact",
}

# ---------------------------------------------------------------------------
# Low-level URL parsing
# ---------------------------------------------------------------------------


def _parse_url(value: object) -> tuple[str, object]:
    """Return (raw_text, ParseResult) for a URL column value, or ('', None) if missing/empty."""
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return "", None
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    return text, parsed


def _url_host_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


# ---------------------------------------------------------------------------
# URL shape keys
# ---------------------------------------------------------------------------


def _normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _url_shape_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_url_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def _url_low_card_query_shape_key(value: object, low_card_query_keys: set[str]) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
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
            or lowered_key in _LAYOUT_EXACT_QUERY_VALUE_KEYS
        ):
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


def _url_semantic_shape_key(value: object) -> str:
    _text, parsed = _parse_url(value)
    if parsed is None:
        return ""
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


# ---------------------------------------------------------------------------
# Item-count helpers
# ---------------------------------------------------------------------------


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
    return max(0, _coerce_item_count(value))


def _item_count_bucket(value: object) -> str:
    count = _coerce_item_count(value)
    if count <= 0:
        return "0"
    for threshold, label in _ITEM_COUNT_BUCKET_THRESHOLDS:
        if count <= threshold:
            return str(count) if label is None else label
    return "129+"


# ---------------------------------------------------------------------------
# Page-signature dispatcher
# ---------------------------------------------------------------------------


def _layout_page_signature_key(url_value: object, item_count_value: object, mode: str) -> str:
    return _layout_page_signature_key_with_low_card_queries(url_value, item_count_value, mode, set())


def _layout_page_signature_key_with_low_card_queries(
    url_value: object,
    item_count_value: object,
    mode: str,
    low_card_query_keys: set[str],
) -> str:
    if not mode or mode == "none":
        return ""
    parts: list[str] = []
    if "url_low_card_query_shape" in mode:
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


# ---------------------------------------------------------------------------
# Query-value helpers (used by selection logic in layout_template.py)
# ---------------------------------------------------------------------------


def _validation_query_values(url_text: str) -> list[tuple[str, str]]:
    _text, parsed = _parse_url(url_text)
    if parsed is None:
        return []
    return [
        (key.strip().lower(), value.strip().lower())
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.strip()
    ]


def _low_card_query_value_keys(url_values: list[Any], max_distinct: int = 16) -> set[str]:
    values_by_key: dict[str, set[str]] = defaultdict(set)
    for url_value in url_values:
        url_text = "" if _is_missing(url_value) else str(url_value)
        for key, value in _validation_query_values(url_text):
            values_by_key[key].add(value)
    return {key for key, values in values_by_key.items() if 1 < len(values) <= max_distinct}


# ---------------------------------------------------------------------------
# DOM-attribute normalization and fingerprinting
# ---------------------------------------------------------------------------

_LAYOUT_TAGS_TO_IGNORE = {"script", "style", "meta", "link", "br", "noscript"}
_LAYOUT_TAGS_IGNORE_ATTR = {"a", "i", "b", "li", "tr", "td", "img", "p", "body"}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_dynamic_attribute(value: str) -> str:
    lowered = value.strip().lower()
    for pattern, label in (
        (_LAYOUT_RE_MD5, "[MD5]"),
        (_LAYOUT_RE_SHA1, "[SHA1]"),
        (_LAYOUT_RE_UUID, "[UUID]"),
        (_LAYOUT_RE_TIMESTAMP, "[TIMESTAMP]"),
    ):
        if pattern.fullmatch(lowered):
            return label
    return _LAYOUT_RE_NUM.sub("", lowered)


def _normalize_attr_tokens(value: str | None) -> str:
    if not value:
        return ""
    tokens = value.split()
    if len(tokens) > 1:
        normalized = [token.lower() for token in tokens if not _LAYOUT_RE_NUM.search(token)]
    else:
        normalized = [_normalize_dynamic_attribute(tokens[0])] if tokens else []
    return " ".join(token for token in normalized if token)


def _walk_dom_element(element: object) -> object:
    raw_tag = getattr(element, "tag", None)
    if not isinstance(raw_tag, str):
        return None
    tag = raw_tag.lower()
    if tag in _LAYOUT_TAGS_TO_IGNORE:
        return None
    attrs: list[tuple[str, str]] = []
    if tag not in _LAYOUT_TAGS_IGNORE_ATTR:
        class_attr = _normalize_attr_tokens(element.get("class"))
        id_attr = _normalize_attr_tokens(element.get("id"))
        if class_attr:
            attrs.append(("class", class_attr))
        if id_attr:
            attrs.append(("id", id_attr))
    children = [child for child in (_walk_dom_element(child) for child in element) if child is not None]
    return [tag, attrs, children]


def _layout_dom_path_fingerprint(html_text: str) -> str:
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
    return json.dumps(_walk_dom_element(root), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _layout_feature_fingerprint(feature: object) -> str:
    if not isinstance(feature, dict):
        return ""

    def normalize_part(part: str) -> dict[str, list[tuple[str, int]]]:
        raw = feature.get(part, {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(layer): sorted(Counter(str(v) for v in vals).items())
            for layer, vals in raw.items()
            if isinstance(vals, list)
        }

    payload = {"tags": normalize_part("tags"), "attrs": normalize_part("attrs")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Miscellaneous pure helpers
# ---------------------------------------------------------------------------


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _labels_to_webkit_response(labels: object) -> dict[str, int]:
    if not isinstance(labels, dict):
        return {}
    response: dict[str, int] = {}
    for item_id, label in labels.items():
        normalized = str(label).strip().lower()
        response[f"item_id {item_id}"] = 1 if normalized in {"main", "1", "true"} else 0
    return response


def _item_id_response(all_item_ids: list[str], main_item_ids: set[str]) -> str:
    labels = {item_id: ("main" if item_id in main_item_ids else "other") for item_id in all_item_ids}
    if all(item_id.isdigit() for item_id in all_item_ids):
        return "".join(f"{item_id}{label}" for item_id, label in labels.items())
    return json.dumps(labels, ensure_ascii=False, separators=(",", ":"))


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
