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

"""Storage helpers for reading PBSS-hosted CC Index parquet data."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from cc_index_url_list.config import DEFAULT_REGION, CCIndexUrlListConfig, build_crawl_uris

if TYPE_CHECKING:
    from pathlib import Path


def build_storage_options(config: CCIndexUrlListConfig, region_name: str = DEFAULT_REGION) -> dict[str, Any]:
    """Build fsspec/s3fs storage options for the PBSS Common Crawl namespace."""
    if not config.cc_index_base_uri.startswith("s3://"):
        return {}

    pbss_access_key = os.environ.get("PBSS_ACCESS_KEY_ID")
    pbss_secret_key = os.environ.get("PBSS_SECRET_ACCESS_KEY")
    if pbss_access_key and pbss_secret_key:
        access_key = pbss_access_key
        secret_key = pbss_secret_key
        session_token = None
    else:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        msg = "Set PBSS_ACCESS_KEY_ID and PBSS_SECRET_ACCESS_KEY before reading PBSS CC Index parquet."
        raise RuntimeError(msg)

    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    if session_token:
        os.environ["AWS_SESSION_TOKEN"] = session_token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)

    return {
        "client_kwargs": {
            "endpoint_url": config.endpoint_url,
            "region_name": region_name,
        },
    }


def list_crawl_parquet_files(
    crawl_uris: list[str],
    storage_options: dict[str, Any],
    max_files_per_crawl: int | None,
) -> list[str]:
    """Expand crawl directories to an explicit, path-sorted parquet file list."""
    from nemo_curator.utils.file_utils import get_all_file_paths_under

    file_paths: list[str] = []
    for crawl_uri in crawl_uris:
        files = sorted(
            get_all_file_paths_under(
                crawl_uri,
                keep_extensions=[".parquet"],
                storage_options=storage_options,
            )
        )
        if not files:
            msg = f"No parquet files found at {crawl_uri}"
            raise FileNotFoundError(msg)
        file_paths.extend(files[:max_files_per_crawl] if max_files_per_crawl is not None else files)
    return file_paths


def build_input_paths(
    config: CCIndexUrlListConfig,
    storage_options: dict[str, Any],
    max_files_per_crawl: int | None,
) -> list[str]:
    """Return an explicit source parquet file list in deterministic crawl/path order."""
    return list_crawl_parquet_files(build_crawl_uris(config), storage_options, max_files_per_crawl)


def parquet_summary(directory: Path) -> tuple[int, float]:
    """Return ``(file_count, total_megabytes)`` for local parquet files."""
    files = sorted(directory.rglob("*.parquet"))
    total_mb = sum(file.stat().st_size for file in files) / (1024 * 1024)
    return len(files), total_mb
