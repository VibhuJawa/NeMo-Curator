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

"""GPU phase: identify duplicate URLs in configured CC Index snapshots."""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Any

from cc_index_url_list.filegroup_utils import (
    create_input_filegroups,
    filegroup_signature_path,
    write_filegroup_signature,
)
from cc_index_url_list.utils import (
    CCIndexUrlListConfig,
    ResolvedRunInputs,
    add_common_args,
    configure_logging,
    create_ray_client,
    load_config,
    resolve_run_inputs,
)
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def log_identification_header(config: CCIndexUrlListConfig, run_inputs: ResolvedRunInputs) -> None:
    """Log identify-phase metadata."""
    logger.info(f"Configured crawls: {', '.join(config.included_crawls)}")
    logger.info(f"Input paths: {len(run_inputs.input_paths):,}")
    logger.info(f"Output dataset: {run_inputs.final_dir}")
    logger.info("Phase: gpu-identify")


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


def run_gpu_identification(config: CCIndexUrlListConfig, args: argparse.Namespace) -> None:
    """Run the GPU exact-dedup phase and write duplicate-ID side outputs."""
    t0 = time.time()
    run_inputs = resolve_run_inputs(config, args)
    log_identification_header(config, run_inputs)
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


def parse_args() -> argparse.Namespace:
    """Parse GPU exact-dedup phase arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Curator ExactDeduplicationWorkflow over configured CC Index snapshots. "
            "This phase requires GPU nodes and writes duplicate-ID side outputs under --output/_dedup_ids."
        ),
    )
    add_common_args(parser, include_gpu_args=True)
    return parser.parse_args()


def main() -> None:
    """Run the GPU exact-dedup phase."""
    configure_logging()
    args = parse_args()
    run_gpu_identification(load_config(args.config), args)


if __name__ == "__main__":
    main()
