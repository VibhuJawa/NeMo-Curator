# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import gzip
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass(kw_only=True)
class WARCParseStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Parse HTTP response bytes from WARC into HTML text.

    Input: binary_content column (bytes) — HTTP response or bare body from a WARC
    record.  CC WARC records may store bare HTTP/0.9 bodies (no status line or
    headers) for some servers; those are decoded directly without header parsing.
    Output: html column (str) — decoded HTML text.
    Also adds: http_status column (int | None).
    """

    name: str = "WARCParseStage"
    binary_content_col: str = "binary_content"
    html_col: str = "html"
    http_status_col: str = "http_status"
    drop_failed: bool = False

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.binary_content_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_col, self.http_status_col]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        html_values: list[str] = []
        status_values: list[int | None] = []

        for _, row in df.iterrows():
            raw = row.get(self.binary_content_col)
            html, status = self._parse_http_response(raw)
            html_values.append(html)
            status_values.append(status)

        df[self.html_col] = html_values
        df[self.http_status_col] = status_values

        if self.drop_failed:
            df = df[df[self.html_col].str.strip().ne("")]

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    @staticmethod
    def _parse_http_response(raw: object) -> tuple[str, int | None]:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return "", None
        if not isinstance(raw, (bytes, bytearray)):
            return "", None
        raw_bytes = bytes(raw)
        try:
            # CC WARC records sometimes store bare HTTP/0.9 responses: the payload
            # is the HTML body with no status line or headers.  Detect this by
            # checking the HTTP/ prefix and skip header parsing entirely.
            if not raw_bytes.startswith(b"HTTP/"):
                return _decode_body(raw_bytes, None), None

            # Split HTTP headers from body
            # HTTP response format: status-line\r\nheaders\r\n\r\nbody
            if b"\r\n\r\n" in raw_bytes:
                header_part, body = raw_bytes.split(b"\r\n\r\n", 1)
            elif b"\n\n" in raw_bytes:
                header_part, body = raw_bytes.split(b"\n\n", 1)
            else:
                return "", None

            # Parse status code from first line
            first_line = header_part.split(b"\r\n" if b"\r\n" in header_part else b"\n", 1)[0]
            parts = first_line.decode("ascii", errors="replace").split(" ", 2)
            status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None  # noqa: PLR2004

            # Detect charset from Content-Type header
            charset = _detect_charset(header_part)

            # Decode body
            html = _decode_body(body, charset)
        except Exception as exc:  # noqa: BLE001
            logger.debug("WARCParseStage failed to parse HTTP response: {}", exc)
            return "", None
        else:
            return html, status


def _detect_charset(header_bytes: bytes) -> str | None:
    import re

    header_text = header_bytes.decode("ascii", errors="replace").lower()
    m = re.search(r"charset=([\w-]+)", header_text)
    return m.group(1).strip() if m else None


def _decode_body(body: bytes, charset: str | None) -> str:
    # Try gzip decompression first
    if body[:2] == b"\x1f\x8b":
        with contextlib.suppress(Exception):
            body = gzip.decompress(body)

    if charset:
        try:
            return body.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass

    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        pass

    with contextlib.suppress(ModuleNotFoundError, Exception):
        from charset_normalizer import detect as charset_normalizer_detect

        detected = charset_normalizer_detect(body)
        enc = detected.get("encoding")
        if enc and enc != "utf-8":
            return body.decode(enc, errors="replace")

    return body.decode("utf-8", errors="replace")
