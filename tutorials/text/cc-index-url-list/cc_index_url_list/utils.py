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

"""Shared helpers for the CC Index URL list tutorial scripts."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
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


@dataclass(frozen=True)
class ResolvedRunInputs:
    """Inputs shared by the identify and remove phases."""

    storage_options: dict[str, Any]
    input_paths: list[str]
    dedup_ids_dir: Path
    final_dir: Path


class RayClientProtocol(Protocol):
    """Common lifecycle methods for Curator Ray clients."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


def configure_logging() -> None:
    """Redact credential values from loguru output emitted by Curator internals."""
    secret_values = {
        value
        for value in (
            os.environ.get("PBSS_SECRET_ACCESS_KEY"),
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
            os.environ.get("AWS_SESSION_TOKEN"),
        )
        if value
    }

    logger.remove()

    def redacting_sink(message: object) -> None:
        text = str(message)
        for secret in secret_values:
            text = text.replace(secret, "<redacted>")
        sys.stderr.write(text)

    logger.add(redacting_sink, level=os.environ.get("CC_INDEX_URL_LIST_LOG_LEVEL", "INFO"))


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


def resolve_run_inputs(config: CCIndexUrlListConfig, args: argparse.Namespace) -> ResolvedRunInputs:
    """Resolve storage options, source paths, and output directories for one phase."""
    output_root = Path(args.output)
    dedup_ids_dir = output_root / "_dedup_ids"
    final_dir = output_root / config.output_name
    needs_storage = not args.dry_run or args.max_files_per_crawl is not None
    storage_options = build_storage_options(config, region_name=args.region_name) if needs_storage else {}
    input_paths = (
        build_input_paths(config, storage_options, args.max_files_per_crawl)
        if needs_storage
        else build_crawl_uris(config)
    )
    return ResolvedRunInputs(
        storage_options=storage_options,
        input_paths=input_paths,
        dedup_ids_dir=dedup_ids_dir,
        final_dir=final_dir,
    )


def default_ray_temp_dir() -> str:
    """Return a short local Ray temp directory."""
    return str(Path(tempfile.gettempdir()) / f"{os.environ.get('USER', 'user')}_nemo_curator_ray")


def create_ray_client(args: argparse.Namespace) -> RayClientProtocol:
    """Create the Ray client that owns the workflow cluster."""
    from nemo_curator.core.client import RayClient, SlurmRayClient

    client_kwargs = {
        "include_dashboard": not args.disable_ray_dashboard,
        "ray_temp_dir": args.ray_temp_dir or os.environ.get("RAY_TMPDIR") or default_ray_temp_dir(),
        "num_cpus": args.ray_num_cpus,
        "num_gpus": getattr(args, "ray_num_gpus", None),
    }
    if args.slurm_ray:
        broadcast_dir = args.ray_port_broadcast_dir or str(Path(args.output) / "_ray_port_broadcast")
        os.environ.setdefault("RAY_PORT_BROADCAST_DIR", broadcast_dir)
        return SlurmRayClient(
            **client_kwargs,
            worker_connect_timeout_s=args.ray_worker_connect_timeout,
        )
    return RayClient(**client_kwargs)


def positive_int(value: str) -> int:
    """Argparse type for positive integer options."""
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def add_common_args(parser: argparse.ArgumentParser, *, include_gpu_args: bool) -> None:
    """Add arguments shared by both phase entrypoints."""
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--max-files-per-crawl",
        type=positive_int,
        default=None,
        help="Use only the first N parquet files per crawl for smoke tests",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print configured inputs without running Curator")
    parser.add_argument("--region-name", default=DEFAULT_REGION, help=f"S3 region name (default: {DEFAULT_REGION})")
    parser.add_argument(
        "--slurm-ray",
        action="store_true",
        help="Use Curator SlurmRayClient; required for multi-node srun launches",
    )
    parser.add_argument(
        "--ray-temp-dir",
        default=None,
        help="Ray temporary directory",
    )
    parser.add_argument("--ray-num-cpus", type=positive_int, default=None, help="CPUs to expose to Ray per node")
    if include_gpu_args:
        parser.add_argument("--ray-num-gpus", type=positive_int, default=None, help="GPUs to expose to Ray per node")
    parser.add_argument(
        "--ray-worker-connect-timeout",
        type=positive_int,
        default=600,
        help="Seconds for SlurmRayClient head to wait for worker nodes",
    )
    parser.add_argument(
        "--ray-port-broadcast-dir",
        default=None,
        help="Shared directory for SlurmRayClient head-port handoff; defaults under --output",
    )
    parser.add_argument("--disable-ray-dashboard", action="store_true", help="Disable Ray dashboard startup")
