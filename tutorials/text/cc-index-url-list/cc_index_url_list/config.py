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

"""Configuration loading and crawl URI expansion for the CC Index URL tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "selected_crawls.yaml"
DEFAULT_REGION = "us-east-1"


@dataclass(frozen=True)
class CCIndexUrlListConfig:
    """Configuration loaded from ``selected_crawls.yaml``."""

    cc_index_base_uri: str
    endpoint_url: str
    included_crawls: list[str]
    output_name: str = "global_unique_urls"


def load_yaml_mapping(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        msg = f"Config must be a YAML mapping: {config_path}"
        raise TypeError(msg)
    return raw


def require_non_empty_string(raw: dict[str, Any], key: str) -> str:
    """Return a stripped non-empty string config value."""
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def validate_included_crawls(raw: dict[str, Any]) -> list[str]:
    """Return validated CC-MAIN crawl IDs from the config."""
    included_crawls = raw["included_crawls"]
    if not isinstance(included_crawls, list) or not included_crawls:
        msg = "included_crawls must be a non-empty YAML list."
        raise ValueError(msg)

    crawls = [crawl.strip() if isinstance(crawl, str) else crawl for crawl in included_crawls]
    if not all(isinstance(crawl, str) and crawl.startswith("CC-MAIN-") for crawl in crawls):
        msg = "Every included_crawls entry must be a CC-MAIN crawl ID string."
        raise ValueError(msg)
    if len(set(crawls)) != len(crawls):
        msg = "included_crawls contains duplicate crawl IDs."
        raise ValueError(msg)
    return crawls


def normalize_output_name(raw: dict[str, Any]) -> str:
    """Return the final output dataset directory name."""
    output_name = raw.get("output_name", "global_unique_urls")
    if not isinstance(output_name, str):
        msg = "output_name must be a non-empty string."
        raise TypeError(msg)
    output_name = output_name.strip().strip("/")
    if not output_name:
        msg = "output_name must be a non-empty string."
        raise ValueError(msg)
    if "/" in output_name or "\\" in output_name:
        msg = "output_name must be a directory name, not a path."
        raise ValueError(msg)
    return output_name


def load_config(config_path: str | Path) -> CCIndexUrlListConfig:
    """Load and validate the YAML tutorial config."""
    raw = load_yaml_mapping(config_path)
    if "excluded_crawls" in raw:
        msg = "Config does not support excluded_crawls; omit crawls from included_crawls instead."
        raise ValueError(msg)

    required = {"cc_index_base_uri", "endpoint_url", "included_crawls"}
    missing = required - raw.keys()
    if missing:
        msg = f"Config is missing required key(s): {sorted(missing)}"
        raise ValueError(msg)

    return CCIndexUrlListConfig(
        cc_index_base_uri=require_non_empty_string(raw, "cc_index_base_uri").rstrip("/"),
        endpoint_url=require_non_empty_string(raw, "endpoint_url"),
        included_crawls=validate_included_crawls(raw),
        output_name=normalize_output_name(raw),
    )


def build_crawl_uri(cc_index_base_uri: str, crawl: str) -> str:
    """Return the CC Index parquet directory for one crawl."""
    return f"{cc_index_base_uri.rstrip('/')}/crawl={crawl}/subset=warc/"


def build_crawl_uris(config: CCIndexUrlListConfig) -> list[str]:
    """Return CC Index parquet directories for the configured crawls."""
    return [build_crawl_uri(config.cc_index_base_uri, crawl) for crawl in config.included_crawls]
