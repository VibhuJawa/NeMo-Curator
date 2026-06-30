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

"""Curator workflow wrappers for the CC Index URL list tutorial."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from cc_index_url_list.filegroups import (
    load_filegroup_signature,
    validate_filegroup_signature,
    write_filegroup_signature,
)

if TYPE_CHECKING:
    from pathlib import Path

CC_INDEX_FIELDS = [
    "url",
    "warc_filename",
]


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
