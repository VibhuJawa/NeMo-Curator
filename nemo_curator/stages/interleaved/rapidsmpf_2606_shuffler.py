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

"""RAPIDS-MPF 26.06 actor base for the isolated GPU Lance extra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cudf
import rmm.mr
from rapidsmpf.integrations.cudf.partition import partition_and_pack
from rapidsmpf.integrations.ray import RapidsMPFActor
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource, LimitAvailableMemory
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

from nemo_curator.stages.deduplication.gpu_utils import align_down_to_256, get_device_free_memory

if TYPE_CHECKING:
    import pylibcudf as plc


class GpuLanceRapidsMPFShuffler(RapidsMPFActor):
    """Own the MPF 26.06 communicator, memory resources, and URL shuffle."""

    def __init__(  # noqa: PLR0913
        self,
        nranks: int,
        total_nparts: int,
        shuffle_on: list[str],
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        *,
        enable_statistics: bool = False,
    ) -> None:
        self.shuffle_on = shuffle_on
        self.total_nparts = total_nparts
        if isinstance(rmm_pool_size, int):
            self.rmm_pool_size = align_down_to_256(rmm_pool_size)
        elif rmm_pool_size == "auto":
            free_memory = get_device_free_memory()
            self.rmm_pool_size = align_down_to_256(int(free_memory * 0.9)) if free_memory is not None else None
        elif rmm_pool_size is None:
            self.rmm_pool_size = None
        else:
            msg = f"Invalid rmm_pool_size: {rmm_pool_size}"
            raise ValueError(msg)

        if isinstance(spill_memory_limit, int):
            self.spill_memory_limit = align_down_to_256(spill_memory_limit)
        elif spill_memory_limit == "auto":
            self.spill_memory_limit = (
                align_down_to_256(int(0.8 * self.rmm_pool_size)) if self.rmm_pool_size is not None else None
            )
        elif spill_memory_limit is None:
            self.spill_memory_limit = None
        else:
            msg = f"Invalid spill_memory_limit: {spill_memory_limit}"
            raise ValueError(msg)

        self.enable_statistics = enable_statistics
        statistics = Statistics(enable=enable_statistics)
        self.mr = RmmResourceAdaptor(
            rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(),
                initial_pool_size=self.rmm_pool_size,
                maximum_pool_size=None,
            )
        )
        rmm.mr.set_current_device_resource(self.mr)
        memory_available = (
            None
            if self.spill_memory_limit is None
            else {MemoryType.DEVICE: LimitAvailableMemory(self.mr, limit=self.spill_memory_limit)}
        )
        self.br = BufferResource(
            device_mr=self.mr,
            memory_available=memory_available,
            statistics=statistics,
        )
        super().__init__(nranks, statistics)
        self.shuffler: Shuffler | None = None

    def setup_worker(self, root_address_bytes: bytes) -> None:
        """Join the UCXX communicator and create operation zero."""
        super().setup_worker(root_address_bytes)
        self.shuffler = Shuffler(
            self.comm,
            0,
            total_num_partitions=self.total_nparts,
            br=self.br,
        )

    def _active_shuffler(self) -> Shuffler:
        if self.shuffler is None:
            msg = "RAPIDS-MPF shuffler is not initialized"
            raise RuntimeError(msg)
        return self.shuffler

    def insert_chunk(self, table: plc.Table | cudf.DataFrame, column_names: list[str]) -> None:
        """Hash partition and submit one table to operation zero."""
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        if isinstance(table, cudf.DataFrame):
            table = cudf_to_pylibcudf_table(table)
        columns_to_hash = tuple(column_names.index(column) for column in self.shuffle_on)
        packed_inputs = partition_and_pack(
            table=table,
            columns_to_hash=columns_to_hash,
            num_partitions=self.total_nparts,
            br=self.br,
            stream=DEFAULT_STREAM,
        )
        self._active_shuffler().insert_chunks(packed_inputs)

    def insert_finished(self) -> None:
        """Send the single global completion signal required by MPF 26.06."""
        self._active_shuffler().insert_finished()
        self.comm.logger.info("Insert finished")

    def cleanup(self) -> None:
        """Report statistics and shut down an initialized shuffle."""
        if self.enable_statistics and self.is_initialized():
            self.comm.logger.info(self.statistics.report(mr=self.mr))
        if self.shuffler is not None:
            self.shuffler.shutdown()
            self.shuffler = None
