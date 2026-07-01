#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Per-row HTML compression helpers for CC-scale Dripper stages."""

from __future__ import annotations

import zlib
from typing import Any

HTML_COL = "html"
HTML_ZLIB_COL = "html_zlib"
HTML_COMPRESSED_COL = HTML_ZLIB_COL
HTML_CHARS_COL = "html_chars"
HTML_CODEC = "zlib"
ZLIB_LEVEL = 3


def _looks_like_zlib(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0x78 and ((data[0] << 8) + data[1]) % 31 == 0


def coerce_html_text(value: object) -> str:
    """Return HTML text from raw str/bytes or compressed bytes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        if _looks_like_zlib(value):
            try:
                return zlib.decompress(value).decode("utf-8", errors="replace")
            except zlib.error:
                pass
        return value.decode("utf-8", errors="replace")
    return str(value)


def compress_html_zlib(value: object) -> bytes:
    """Compress one HTML value as stdlib zlib bytes."""
    html = coerce_html_text(value)
    if not html:
        return b""
    return zlib.compress(html.encode("utf-8", errors="replace"), level=ZLIB_LEVEL)


def get_html_from_row(
    row: Any,
    html_col: str = HTML_COL,
    html_zlib_col: str = HTML_ZLIB_COL,
) -> str:
    """Read HTML from a row-like object, preferring compressed per-row payloads."""
    getter = row.get if hasattr(row, "get") else None
    if getter is None:
        return ""
    compressed = getter(html_zlib_col, None)
    if compressed is not None and compressed != b"":
        return coerce_html_text(compressed)
    return coerce_html_text(getter(html_col, ""))


def row_with_compressed_html(row: dict[str, Any], html_col: str = HTML_COL) -> dict[str, Any]:
    """Return a copy of row with html_zlib/html_chars and without raw html."""
    out = dict(row)
    html = get_html_from_row(out, html_col=html_col)
    out[HTML_ZLIB_COL] = compress_html_zlib(html)
    out[HTML_CHARS_COL] = len(html)
    out.pop(html_col, None)
    return out
