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

"""Shared helpers for creating a global unique URL list from CC Index snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("selected_crawls.yaml")
DEFAULT_REGION = "us-east-1"
FILEGROUP_SIGNATURE_FILENAME = "filegroup_signature.json"
CC_INDEX_FIELDS = [
    "url",
    "warc_filename",
]
FileGroupSignature = tuple[tuple[str, ...], ...]


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
    if not all(isinstance(crawl, str) and crawl.startswith("CC-MAIN-") for crawl in included_crawls):
        msg = "Every included_crawls entry must be a CC-MAIN crawl ID string."
        raise ValueError(msg)
    if len(set(included_crawls)) != len(included_crawls):
        msg = "included_crawls contains duplicate crawl IDs."
        raise ValueError(msg)
    return list(included_crawls)


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
    crawl_uris = build_crawl_uris(config)
    return list_crawl_parquet_files(crawl_uris, storage_options, max_files_per_crawl)


def parquet_summary(directory: Path) -> tuple[int, float]:
    """Return ``(file_count, total_megabytes)`` for local parquet files."""
    files = sorted(directory.rglob("*.parquet"))
    total_mb = sum(file.stat().st_size for file in files) / (1024 * 1024)
    return len(files), total_mb


def filegroup_signature(input_tasks: list[Any]) -> FileGroupSignature:
    """Return the ordered file paths for each Curator file group task."""
    return tuple(tuple(task.data) for task in input_tasks)


def filegroup_signature_digest(signature: FileGroupSignature) -> str:
    """Return a stable digest for comparing file group order across runs."""
    hasher = hashlib.sha256()
    for filegroup in signature:
        hasher.update(b"\x1e")
        for path in filegroup:
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\x00")
    return hasher.hexdigest()


def filegroup_signature_path(dedup_ids_dir: Path) -> Path:
    """Return the persisted filegroup signature path."""
    return dedup_ids_dir / FILEGROUP_SIGNATURE_FILENAME


def write_filegroup_signature(input_tasks: list[Any], dedup_ids_dir: Path) -> FileGroupSignature:
    """Persist the ordered source file groups used for exact deduplication."""
    signature = filegroup_signature(input_tasks)
    digest = filegroup_signature_digest(signature)
    path = filegroup_signature_path(dedup_ids_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sha256": digest,
        "filegroups": [list(filegroup) for filegroup in signature],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info(f"Locked file group order for {len(signature):,} task(s); sha256={digest}")
    logger.info(f"Wrote file group signature to {path}")
    return signature


def load_filegroup_signature(dedup_ids_dir: Path) -> FileGroupSignature:
    """Load the filegroup signature written by the exact-dedup phase."""
    path = filegroup_signature_path(dedup_ids_dir)
    if not path.exists():
        msg = f"Missing file group signature: {path}. Run identify_cc_index_url_duplicates.py first."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        msg = f"Invalid file group signature payload: {path}"
        raise TypeError(msg)
    filegroups = payload.get("filegroups")
    if not isinstance(filegroups, list):
        msg = f"Invalid file group signature payload: {path}"
        raise TypeError(msg)
    if not all(isinstance(filegroup, list) and all(isinstance(item, str) for item in filegroup) for filegroup in filegroups):
        msg = f"Invalid file group entries in signature payload: {path}"
        raise TypeError(msg)
    signature = tuple(tuple(filegroup) for filegroup in filegroups)
    digest = filegroup_signature_digest(signature)
    if payload.get("sha256") != digest:
        msg = f"File group signature digest mismatch in {path}"
        raise ValueError(msg)
    logger.info(f"Loaded file group signature for {len(signature):,} task(s); sha256={digest}")
    return signature


def validate_filegroup_signature(input_tasks: list[Any], expected: FileGroupSignature, phase: str) -> None:
    """Fail if the file group membership or ordering changed between workflow phases."""
    actual = filegroup_signature(input_tasks)
    if actual != expected:
        msg = (
            f"Curator file group order changed before {phase}. "
            "Exact deduplication and duplicate removal must receive the same ordered file groups "
            "so the ID generator can map assigned IDs back to the source parquet files."
        )
        raise RuntimeError(msg)
    logger.info(f"Validated file group order for {phase}; sha256={filegroup_signature_digest(actual)}")


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


def create_input_filegroups(input_paths: list[str]) -> list[Any]:
    """Create one Curator file group per source parquet shard."""
    from nemo_curator.tasks import FileGroupTask
    from nemo_curator.utils.file_utils import infer_dataset_name_from_path

    if not input_paths:
        msg = f"No parquet file groups were created from {input_paths}"
        raise RuntimeError(msg)

    logger.info(f"Phase 1: creating Curator file groups from {len(input_paths):,} input path(s)")
    dataset_name = infer_dataset_name_from_path(input_paths[0])
    tasks = [
        FileGroupTask(
            dataset_name=dataset_name,
            data=[path],
            _metadata={
                "partition_index": partition_index,
                "total_partitions": len(input_paths),
                "source_files": [path],
            },
            reader_config={},
        )
        for partition_index, path in enumerate(input_paths)
    ]
    logger.info(f"Phase 1 complete: {len(tasks):,} file group task(s)")
    return tasks


def run_exact_url_identification(
    input_tasks: list[Any],
    storage_options: dict[str, Any],
    dedup_ids_dir: Path,
) -> None:
    """Run Curator exact deduplication over the CC Index inputs."""
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

    read_kwargs = {"storage_options": storage_options} if storage_options else {}
    write_filegroup_signature(input_tasks, dedup_ids_dir)

    logger.info(f"Phase 2: identifying duplicate URLs from {len(input_tasks):,} file group task(s)")
    dedup_result = ExactDeduplicationWorkflow(
        input_path=None,
        output_path=str(dedup_ids_dir),
        text_field="url",
        input_filetype="parquet",
        read_kwargs=read_kwargs,
        assign_id=True,
    ).run(initial_tasks=input_tasks)
    logger.info(f"Phase 2 complete: {dedup_result.metadata.get('num_duplicates', 0):,} duplicate IDs")


def run_duplicate_removal(
    input_tasks: list[Any],
    storage_options: dict[str, Any],
    dedup_ids_dir: Path,
    final_dir: Path,
) -> None:
    """Run Curator duplicate removal using exact-dedup side outputs."""
    from nemo_curator.stages.deduplication.exact.identification import ExactDuplicateIdentification
    from nemo_curator.stages.deduplication.exact.workflow import ID_GENERATOR_OUTPUT_FILENAME
    from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
    from nemo_curator.stages.text.deduplication import TextDuplicatesRemovalWorkflow

    duplicate_ids_dir = dedup_ids_dir / ExactDuplicateIdentification.name
    id_generator_path = dedup_ids_dir / ID_GENERATOR_OUTPUT_FILENAME
    if not duplicate_ids_dir.exists():
        msg = f"Missing duplicate ID directory: {duplicate_ids_dir}. Run identify_cc_index_url_duplicates.py first."
        raise FileNotFoundError(msg)
    if not id_generator_path.exists():
        msg = f"Missing exact-dedup ID generator: {id_generator_path}. Run identify_cc_index_url_duplicates.py first."
        raise FileNotFoundError(msg)

    read_kwargs = {"storage_options": storage_options} if storage_options else {}
    expected_filegroups = load_filegroup_signature(dedup_ids_dir)
    validate_filegroup_signature(input_tasks, expected_filegroups, "duplicate removal")

    logger.info(f"Phase 3: writing global unique URLs to {final_dir}")
    removal_result = TextDuplicatesRemovalWorkflow(
        input_path=None,
        ids_to_remove_path=str(duplicate_ids_dir),
        output_path=str(final_dir),
        input_filetype="parquet",
        input_fields=list(CC_INDEX_FIELDS),
        input_kwargs=read_kwargs,
        output_fields=CC_INDEX_FIELDS,
        output_filetype="parquet",
        output_kwargs={"index": False},
        output_mode="overwrite",
        duplicate_id_field=CURATOR_DEDUP_ID_STR,
        id_generator_path=str(id_generator_path),
    ).run(initial_tasks=input_tasks)
    logger.info(f"Phase 3 complete: {removal_result.metadata.get('num_duplicates_removed', 0):,} rows removed")


def resolve_run_inputs(
    config: CCIndexUrlListConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], Path, Path]:
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
    return storage_options, input_paths, dedup_ids_dir, final_dir


def log_run_header(config: CCIndexUrlListConfig, input_paths: list[str], final_dir: Path, phase_name: str) -> None:
    """Log common phase metadata."""
    logger.info(f"Configured crawls: {', '.join(config.included_crawls)}")
    logger.info(f"Input paths: {len(input_paths):,}")
    logger.info(f"Output dataset: {final_dir}")
    logger.info(f"Phase: {phase_name}")


def run_gpu_identification(config: CCIndexUrlListConfig, args: argparse.Namespace) -> None:
    """Run the GPU exact-dedup phase and write duplicate-ID side outputs."""
    t0 = time.time()
    storage_options, input_paths, dedup_ids_dir, final_dir = resolve_run_inputs(config, args)
    log_run_header(config, input_paths, final_dir, "gpu-identify")
    if args.dry_run:
        return

    input_tasks = create_input_filegroups(input_paths)
    ray_client = create_ray_client(args)
    ray_client.start()
    try:
        run_exact_url_identification(input_tasks, storage_options, dedup_ids_dir)
    finally:
        ray_client.stop()

    logger.info("=" * 72)
    logger.info(f"Dedup IDs dir: {dedup_ids_dir}")
    logger.info(f"Signature:     {filegroup_signature_path(dedup_ids_dir)}")
    logger.info(f"Elapsed:      {time.time() - t0:.0f}s")
    logger.info("=" * 72)


def run_cpu_removal(config: CCIndexUrlListConfig, args: argparse.Namespace) -> None:
    """Run the CPU duplicate-removal phase and write the final URL dataset."""
    t0 = time.time()
    storage_options, input_paths, dedup_ids_dir, final_dir = resolve_run_inputs(config, args)
    log_run_header(config, input_paths, final_dir, "cpu-remove")
    if args.dry_run:
        return

    input_tasks = create_input_filegroups(input_paths)
    ray_client = create_ray_client(args)
    ray_client.start()
    try:
        run_duplicate_removal(input_tasks, storage_options, dedup_ids_dir, final_dir)
    finally:
        ray_client.stop()

    n_files, total_mb = parquet_summary(final_dir)
    logger.info("=" * 72)
    logger.info(f"Output files: {n_files:,}")
    logger.info(f"Output size:  {total_mb:,.1f} MiB")
    logger.info(f"Output dir:   {final_dir}")
    logger.info(f"Elapsed:      {time.time() - t0:.0f}s")
    logger.info("=" * 72)


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
