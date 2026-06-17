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

"""Fetch individual WARC records by byte-range and return raw HTML bytes.

Two public objects:

CCWarcByteRangeFetcher
    Fetch-only helper: byte-range GET one WARC record, parse the WARC body,
    return raw HTML bytes.  No text extraction — that runs downstream in
    three independent HtmlExtractStage actors (one per extractor library).

CCWarcByteRangeFetchStage(ProcessingStage)
    I/O-only actor stage.  Calls fetcher.fetch_only() in a
    ThreadPoolExecutor (max_workers=16 by default, matching
    download.py:_read_warc_records_batch).
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

# CC public HTTPS endpoint — no auth required
_CC_BASE_URL = "https://data.commoncrawl.org"


def _snapshot_from_warc_filename(warc_filename: str) -> str:
    """Extract CC snapshot ID from a WARC path.

    'crawl-data/CC-MAIN-2025-26/segments/.../warc/....warc.gz' → 'CC-MAIN-2025-26'
    """
    parts = warc_filename.split("/")
    return next(
        (p for p in parts if p.startswith(("CC-MAIN-", "CC-NEWS-"))),
        parts[1] if len(parts) > 1 else warc_filename,
    )


def _parse_warc_body(raw: bytes) -> bytes | None:
    """Extract the HTTP response body from a raw WARC record byte payload."""
    with io.BytesIO(raw) as buf:
        for record in ArchiveIterator(buf):
            if record.rec_type in ("response", "conversion"):
                body = record.content_stream().read()
                return body if body else None
    return None


def _fetch_warc_bytes_http(  # noqa: PLR0913
    warc_filename: str,
    offset: int,
    length: int,
    base_url: str,
    timeout: int,
    max_retries: int,
) -> bytes | None:
    """HTTP byte-range GET a single WARC record; return the HTTP response body."""
    import requests

    url = f"{base_url}/{warc_filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return _parse_warc_body(resp.content)
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(1.5**attempt)
            else:
                logger.debug(f"HTTP fetch failed {warc_filename}@{offset}: {exc}")
    return None


def _fetch_warc_bytes_s3(  # noqa: PLR0913
    s3_client: Any,  # noqa: ANN401 — boto3 client has no public type
    bucket: str,
    warc_filename: str,
    offset: int,
    length: int,
    max_retries: int,
) -> bytes | None:
    """S3 byte-range GET a single WARC record; return the HTTP response body."""
    range_header = f"bytes={offset}-{offset + length - 1}"
    for attempt in range(max_retries):
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=warc_filename, Range=range_header)
            return _parse_warc_body(resp["Body"].read())
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(1.5**attempt)
            else:
                logger.debug(f"S3 fetch failed {warc_filename}@{offset}: {exc}")
    return None


class CCWarcByteRangeFetcher:
    """Fetch-only helper: byte-range GET one WARC record and return raw HTML bytes.

    Used by CCWarcByteRangeFetchStage.  No text extraction — extraction runs
    downstream in three independent HtmlExtractStage actors.

    Input:  url, warc_filename, warc_record_offset, warc_record_length
    Output: cc_url, cc_snapshot_id, cc_html_bytes,
            content_digest, url_host_name
    """

    def __init__(  # noqa: PLR0913
        self,
        use_s3: bool = False,
        s3_bucket: str = "crawl-data",
        s3_endpoint: str = "https://pdx.s8k.io",
        s3_key_id: str | None = None,
        s3_secret: str | None = None,
        max_retries: int = 2,
        timeout: int = 30,
        base_url: str = _CC_BASE_URL,
    ):
        self._use_s3 = use_s3
        self._s3_bucket = s3_bucket
        self._max_retries = max_retries
        self._timeout = timeout
        self._base_url = base_url

        # boto3 clients are NOT thread-safe. Each thread in the ThreadPoolExecutor
        # gets its own client via threading.local() so concurrent GETs don't race.
        # Explicit s3_key_id/s3_secret take priority over env vars — pass them from
        # the pipeline entry point rather than relying on env vars surviving Ray pickling.
        self._s3_endpoint = s3_endpoint if use_s3 else None
        self._s3_key_id = (s3_key_id or os.environ.get("CC_PBSS_ACCESS_KEY_ID", "")) if use_s3 else ""
        self._s3_secret = (s3_secret or os.environ.get("CC_PBSS_SECRET_ACCESS_KEY", "")) if use_s3 else ""
        self._s3_tls: threading.local = threading.local()

    # ------------------------------------------------------------------
    # Pickle support — threading.local() cannot be pickled (Ray serialises
    # stages before sending them to worker processes).  __getstate__ drops
    # _s3_tls; __setstate__ re-creates it so each thread builds its boto3
    # client on first use in the new process.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_s3_tls"] = None  # exclude non-picklable thread-local storage
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._s3_tls = threading.local()  # recreate on unpickle / deserialization

    def _thread_s3_client(self) -> Any:  # noqa: ANN401 — boto3 client has no public type
        """Return a per-thread boto3 S3 client (thread-safe via threading.local)."""
        if not hasattr(self._s3_tls, "client"):
            import boto3
            import botocore.config

            self._s3_tls.client = boto3.client(
                "s3",
                endpoint_url=self._s3_endpoint,
                aws_access_key_id=self._s3_key_id,
                aws_secret_access_key=self._s3_secret,
                region_name="us-east-1",
                config=botocore.config.Config(
                    s3={"addressing_style": "path"},
                    connect_timeout=10,
                    read_timeout=30,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
        return self._s3_tls.client

    def _fetch(self, warc_filename: str, offset: int, length: int) -> bytes | None:
        if self._use_s3:
            # warc_filename is already a clean S3 key — the bucket prefix
            # is stripped upstream in CCIndexParquetReaderStage.process().
            return _fetch_warc_bytes_s3(
                self._thread_s3_client(), self._s3_bucket, warc_filename, offset, length, self._max_retries
            )
        return _fetch_warc_bytes_http(warc_filename, offset, length, self._base_url, self._timeout, self._max_retries)

    def fetch_only(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Fetch one WARC record and return raw HTML bytes + WARC coords.

        Performs the S3/HTTP byte-range fetch and WARC parse, but does NOT run
        any text extractors.  CCWarcByteRangeFetchStage calls this method so that
        fetch and extraction can be split into separate pipeline stages.

        Input keys:  url, warc_filename, warc_record_offset, warc_record_length
        Output keys: cc_url, cc_snapshot_id, warc_filename, warc_record_offset,
                     warc_record_length, cc_html_bytes, content_digest, url_host_name
        """
        warc_filename = row["warc_filename"]
        offset = int(row["warc_record_offset"])
        length = int(row["warc_record_length"])

        html_bytes = self._fetch(warc_filename, offset, length) or b""

        return {
            "cc_url": row["url"],
            # Prefer upstream snapshot id already present in the CC index row;
            # fall back to derivation only for rows that pre-date the annotation.
            "cc_snapshot_id": row.get("cc_snapshot_id") or _snapshot_from_warc_filename(warc_filename),
            "warc_filename": warc_filename,
            "warc_record_offset": offset,
            "warc_record_length": length,
            "cc_html_bytes": html_bytes,
            "content_digest": row.get("content_digest", ""),
            "url_host_name": row.get("url_host_name", ""),
        }


@dataclass
class CCWarcByteRangeFetchStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """I/O-only actor stage: byte-range fetch each WARC record, return raw HTML bytes.

    Calls fetcher.fetch_only() in a ThreadPoolExecutor — no text extraction here.
    CCHtmlExtractStage reads cc_html_bytes downstream and runs all three extractors.

    Output columns: cc_url, cc_snapshot_id, warc_filename, warc_record_offset,
                    warc_record_length, cc_html_bytes, content_digest, url_host_name
    """

    fetcher: CCWarcByteRangeFetcher
    max_workers: int = 16  # matches Curator's existing WARC downloader default (download.py:134)
    name: str = "cc_warc_byte_range_fetch"

    # Class constant — ClassVar excludes it from __init__, __repr__, and pickling.
    _FETCH_ONLY_COLUMNS: ClassVar[tuple[str, ...]] = (
        "cc_url",
        "cc_snapshot_id",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
        "cc_html_bytes",
        "content_digest",
        "url_host_name",
    )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Called once per actor worker — marks this as an actor stage.

        No extractor initialisation needed; fetch_only() requires only the S3/HTTP
        client that is already created inside CCWarcByteRangeFetcher.__init__.
        """
        logger.info(
            f"CCWarcByteRangeFetchStage.setup: ready (s3={self.fetcher._use_s3}, max_workers={self.max_workers})"
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self._FETCH_ONLY_COLUMNS)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Fetch all rows in parallel using ThreadPoolExecutor (I/O bound).

        Mirrors download.py:_read_warc_records_batch (lines 378-405).
        """
        df = task.to_pandas()
        records_in = df.to_dict("records")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map preserves order; wrap each result so one bad row
            # does not abort the whole batch — log and treat as None.
            def _safe_fetch(rec: dict) -> dict | None:
                try:
                    return self.fetcher.fetch_only(rec)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"fetch_only failed for {rec.get('warc_filename', '?')}: {exc}")
                    return None

            records = [r for r in executor.map(_safe_fetch, records_in) if r is not None]

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pd.DataFrame(records) if records else pd.DataFrame(columns=list(self._FETCH_ONLY_COLUMNS)),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
