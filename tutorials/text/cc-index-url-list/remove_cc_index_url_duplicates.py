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

"""CPU phase: remove duplicate URLs and write the final CC Index URL list."""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Any

from cc_index_url_list.utils import (
    CCIndexUrlListConfig,
    add_common_args,
    configure_logging,
    create_input_filegroups,
    create_ray_client,
    load_config,
    load_filegroup_signature,
    log_run_header,
    parquet_summary,
    resolve_run_inputs,
    validate_filegroup_signature,
)
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

CC_INDEX_FIELDS = [
    "url",
    "warc_filename",
]


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


def parse_args() -> argparse.Namespace:
    """Parse CPU duplicate-removal phase arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Curator TextDuplicatesRemovalWorkflow using exact-dedup side outputs produced by "
            "identify_cc_index_url_duplicates.py. This phase is CPU-only and writes the final parquet dataset."
        ),
    )
    add_common_args(parser, include_gpu_args=False)
    return parser.parse_args()


def main() -> None:
    """Run the CPU duplicate-removal phase."""
    configure_logging()
    args = parse_args()
    run_cpu_removal(load_config(args.config), args)


if __name__ == "__main__":
    main()
