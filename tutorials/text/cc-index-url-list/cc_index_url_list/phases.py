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

"""Top-level GPU and CPU phases for the CC Index URL list tutorial."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from cc_index_url_list.config import CCIndexUrlListConfig, build_crawl_uris
from cc_index_url_list.filegroups import create_input_filegroups, filegroup_signature_path
from cc_index_url_list.ray_utils import create_ray_client
from cc_index_url_list.storage import build_input_paths, build_storage_options, parquet_summary
from cc_index_url_list.workflows import run_duplicate_removal, run_exact_url_identification

if TYPE_CHECKING:
    import argparse


@dataclass(frozen=True)
class ResolvedRunInputs:
    """Inputs shared by the identify and remove phases."""

    storage_options: dict[str, Any]
    input_paths: list[str]
    dedup_ids_dir: Path
    final_dir: Path


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


def log_run_header(config: CCIndexUrlListConfig, run_inputs: ResolvedRunInputs, phase_name: str) -> None:
    """Log common phase metadata."""
    logger.info(f"Configured crawls: {', '.join(config.included_crawls)}")
    logger.info(f"Input paths: {len(run_inputs.input_paths):,}")
    logger.info(f"Output dataset: {run_inputs.final_dir}")
    logger.info(f"Phase: {phase_name}")


def run_gpu_identification(config: CCIndexUrlListConfig, args: argparse.Namespace) -> None:
    """Run the GPU exact-dedup phase and write duplicate-ID side outputs."""
    t0 = time.time()
    run_inputs = resolve_run_inputs(config, args)
    log_run_header(config, run_inputs, "gpu-identify")
    if args.dry_run:
        return

    input_tasks = create_input_filegroups(run_inputs.input_paths)
    ray_client = create_ray_client(args)
    ray_client.start()
    try:
        run_exact_url_identification(input_tasks, run_inputs.storage_options, run_inputs.dedup_ids_dir)
    finally:
        ray_client.stop()

    logger.info("=" * 72)
    logger.info(f"Dedup IDs dir: {run_inputs.dedup_ids_dir}")
    logger.info(f"Signature:     {filegroup_signature_path(run_inputs.dedup_ids_dir)}")
    logger.info(f"Elapsed:      {time.time() - t0:.0f}s")
    logger.info("=" * 72)


def run_cpu_removal(config: CCIndexUrlListConfig, args: argparse.Namespace) -> None:
    """Run the CPU duplicate-removal phase and write the final URL dataset."""
    t0 = time.time()
    run_inputs = resolve_run_inputs(config, args)
    log_run_header(config, run_inputs, "cpu-remove")
    if args.dry_run:
        return

    input_tasks = create_input_filegroups(run_inputs.input_paths)
    ray_client = create_ray_client(args)
    ray_client.start()
    try:
        run_duplicate_removal(
            input_tasks=input_tasks,
            storage_options=run_inputs.storage_options,
            dedup_ids_dir=run_inputs.dedup_ids_dir,
            final_dir=run_inputs.final_dir,
        )
    finally:
        ray_client.stop()

    n_files, total_mb = parquet_summary(run_inputs.final_dir)
    logger.info("=" * 72)
    logger.info(f"Output files: {n_files:,}")
    logger.info(f"Output size:  {total_mb:,.1f} MiB")
    logger.info(f"Output dir:   {run_inputs.final_dir}")
    logger.info(f"Elapsed:      {time.time() - t0:.0f}s")
    logger.info("=" * 72)
