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

"""JSONL reader composite stage."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from fsspec.utils import infer_storage_options
from loguru import logger

from ray_curator.backends.experimental.utils import RayStageSpecKeys
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask, _EmptyTask
from ray_curator.utils.file_utils import _split_files_as_per_blocksize, get_all_files_paths_under


@dataclass
class FilePartitioningStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Stage that partitions input file paths into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    limit: int | None = None
    _name: str = "file_partitioning"

    def __post_init__(self):
        """Initialize default values."""
        if self.file_extensions is None:
            self.file_extensions = [".jsonl", ".json", ".parquet"]
        if self.storage_options is None:
            self.storage_options = {}
        if self.blocksize is not None:
            self._blocksize = self._parse_size(self.blocksize)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(cpus=0.5)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
        }

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        """Process the initial task to create file group tasks.

        This stage expects a simple Task with file paths information
        and outputs multiple FileGroupTasks for parallel processing.
        """
        # Get list of files
        # Returns a tuple of (file_path, size) if return_sizes is True, otherwise just the file_path
        files = self._get_file_list(return_sizes=bool(self.blocksize))
        logger.info(f"Found {len(files)} files")
        if not files:
            logger.warning(f"No files found matching pattern: {self.file_paths}")
            return []
        # Partition files
        if self.files_per_partition:
            partitions = self._partition_by_count(files, self.files_per_partition)
        elif self.blocksize:
            partitions = self._partition_by_size(files, self._blocksize)
        else:
            # Default to one file per partition
            logger.info("No partitions specified, defaulting to one file per partition")
            partitions = self._partition_by_count(files, 1)

        # Create FileGroupTask for each partition
        tasks = []
        dataset_name = self._get_dataset_name(files)

        for i, file_group in enumerate(partitions):
            if self.limit is not None and len(tasks) >= self.limit:
                logger.info(f"Reached limit of {self.limit} file groups")
                break
            file_task = FileGroupTask(
                task_id=f"file_group_{i}",
                dataset_name=dataset_name,
                data=file_group,
                _metadata={
                    "partition_index": i,
                    "total_partitions": len(partitions),
                    "source_files": file_group,  # Add source files for deterministic naming during write stage
                },
                reader_config={},  # Empty - will be populated by reader stage
            )
            tasks.append(file_task)

        logger.info(f"Created {len(tasks)} file groups from {len(files)} files")
        return tasks

    def _get_file_list(self, return_sizes: bool = False) -> list[str] | list[tuple[str, int]]:
        """Get the list of files to process."""
        logger.info(f"Getting file list for {self.file_paths}")
        if isinstance(self.file_paths, str):
            # Directory: list contents (recursively) and filter extensions
            return get_all_files_paths_under(
                self.file_paths,
                recurse_subdirectories=True,
                keep_extensions=self.file_extensions,
                storage_options=self.storage_options,
                return_sizes=return_sizes,
            )
        elif isinstance(self.file_paths, list):
            if return_sizes:
                output_ls = []
                for path in self.file_paths:
                    output_ls.extend(
                        get_all_files_paths_under(
                            path,
                            recurse_subdirectories=False,
                            keep_extensions=self.file_extensions,
                            storage_options=self.storage_options,
                            return_sizes=return_sizes,
                        )
                    )
                return output_ls
            else:
                return self.file_paths
        else:
            msg = f"Invalid file paths: {self.file_paths}, must be a string or list of strings"
            raise TypeError(msg)

    def _get_dataset_name(self, files: list[str]) -> str:
        """Extract dataset name from file paths (fsspec-compatible)."""
        if not files:
            return "dataset"

        # If the files are a list of tuples (when return_sizes is True), we need to get the first element of the tuple
        # Otherwise, we can just use the first element
        first = files[0][0] if isinstance(files[0], tuple) else files[0]
        opts = infer_storage_options(first, inherit_storage_options=self.storage_options)
        protocol = opts.get("protocol")
        if protocol and protocol not in {"file", "local"}:
            path_part = opts.get("path", first)
            host = opts.get("host") or opts.get("netloc")
            normalized = f"{host}/{path_part.lstrip('/')}" if host else path_part
            # We do PurePosixPath because that is fsspec compliant
            p = PurePosixPath(normalized)
        else:
            p = Path(first)

        if p.parent.name and p.parent.name != ".":
            return p.parent.name
        return p.stem or "dataset"

    def _partition_by_count(self, files: list[str], count: int) -> list[list[str]]:
        """Partition files by count."""
        partitions = []
        for i in range(0, len(files), count):
            partitions.append(files[i : i + count])
        return partitions

    def _partition_by_size(self, files: list[tuple[str, int]], blocksize: int | str) -> list[list[str]]:
        """Partition files by target size.

        Note: This is a simplified implementation. A full implementation
        would check actual file sizes and create balanced partitions.
        """
        sorted_files = sorted(files, key=lambda x: x[1])
        return _split_files_as_per_blocksize(sorted_files, blocksize)

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '128MB' to bytes."""
        size_str = size_str.upper().strip()

        # Check units in order from longest to shortest to avoid partial matches
        units = [
            ("TB", 1024 * 1024 * 1024 * 1024),
            ("GB", 1024 * 1024 * 1024),
            ("MB", 1024 * 1024),
            ("KB", 1024),
            ("B", 1),
        ]

        for unit, multiplier in units:
            if size_str.endswith(unit):
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)

        # If no unit, assume bytes
        return int(size_str)
