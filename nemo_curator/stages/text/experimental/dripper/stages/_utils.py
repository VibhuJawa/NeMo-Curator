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

from dataclasses import fields
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from nemo_curator.stages.text.experimental.dripper.stages._types import _ITEM_ID_RE

# ---------------------------------------------------------------------------
# Coercion / HTML helpers
# ---------------------------------------------------------------------------


def _coerce_html(value: object) -> str:
    """Coerce any HTML value to a clean string."""
    if _is_missing(value):
        return ""
    if isinstance(value, bytes | bytearray):
        raw_bytes = bytes(value)
        decoded = _decode_html_bytes(raw_bytes)
        if decoded is None:
            decoded = raw_bytes.decode("utf-8", errors="replace")
        return _strip_xml_incompatible_chars(decoded or "")
    return _strip_xml_incompatible_chars(str(value))


def _coerce_optional_str(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    return text if text else None


def _sanitize_case_output_html(case: object) -> None:
    output_data = getattr(case, "output_data", None)
    if output_data is None:
        return
    main_html = getattr(output_data, "main_html", None)
    if isinstance(main_html, str):
        output_data.main_html = _strip_xml_incompatible_chars(main_html)


def _get_processed_attr(case: object, attr: str) -> str:
    process_data = getattr(case, "process_data", None)
    value = getattr(process_data, attr, "") if process_data is not None else ""
    return value if isinstance(value, str) else ""


def _is_empty_document_error(error: str) -> bool:
    normalized = error.lower()
    return "document is empty" in normalized or "empty html tree" in normalized or "empty html input" in normalized


def _case_has_item_ids(case: object) -> bool:
    return "_item_id" in _get_processed_attr(case, "simpled_html") or "_item_id" in _get_processed_attr(
        case,
        "map_html",
    )


def _count_item_ids(case: object) -> int:
    html = _get_processed_attr(case, "simpled_html") or _get_processed_attr(case, "map_html")
    return len(set(_ITEM_ID_RE.findall(html)))


# ---------------------------------------------------------------------------
# Module-level helpers (from stage.py lines 4918-5011)
# ---------------------------------------------------------------------------


def _numeric_series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _strip_xml_incompatible_chars(value: str) -> str:
    """Remove characters that XML/HTML converters reject while preserving text."""

    def is_xml_char(char: str) -> bool:
        codepoint = ord(char)
        return (
            codepoint in {9, 10, 13}
            or 32 <= codepoint <= 55295  # noqa: PLR2004
            or 57344 <= codepoint <= 65533  # noqa: PLR2004
            or 65536 <= codepoint <= 1114111  # noqa: PLR2004
        )

    return "".join(char for char in value if is_xml_char(char))


def _decode_html_bytes(html_bytes: bytes) -> str | None:
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    try:
        from charset_normalizer import detect as charset_normalizer_detect
    except ModuleNotFoundError:
        return None

    detected_encoding = charset_normalizer_detect(html_bytes)["encoding"]
    if not detected_encoding or detected_encoding == "utf-8":
        return None
    try:
        return html_bytes.decode(detected_encoding)
    except Exception:  # noqa: BLE001
        return None


def _coerce_usage_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_warning(existing: str, new_warning: str) -> str:
    if not existing:
        return new_warning
    if not new_warning:
        return existing
    return f"{existing}; {new_warning}"


def _url_host_key(value: object) -> str:
    text = "" if _is_missing(value) else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _dataclass_init_kwargs(instance: object, *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude or set()
    return {
        item.name: getattr(instance, item.name) for item in fields(instance) if item.init and item.name not in excluded
    }
