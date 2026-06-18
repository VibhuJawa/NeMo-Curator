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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.grouping import split_into_n_chunks

if TYPE_CHECKING:
    import pandas as pd


@dataclass(kw_only=True)
class HostDomainGroupingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Bin-pack mixed-host batches into per-host-domain DocumentBatches.

    IS_FANOUT_STAGE: one input batch of N rows from multiple hosts becomes M output
    batches (one per host_domain with >= min_rows_per_batch rows).
    Small hosts are bin-packed together to meet min_rows_per_batch.

    max_rows_per_batch BALANCES the fan-out: without it a mega-host (e.g. tgcom24's
    20k rows) becomes ONE giant batch -> one Ray block -> one actor that processes it
    serially while the other actors idle. When set, a host with more than this many
    rows is split into balanced ~equal chunks so its (per-row) work spreads across
    actors. NOTE: this fragments a host across blocks, so it is for the per-row Phase A
    preprocess ONLY -- a downstream consolidation must re-group each host whole before
    layout clustering (which requires all of a host's pages together). Leave None
    (default) to keep every host in one batch (whole-host clustering).
    """

    name: str = "HostDomainGroupingStage"
    host_domain_col: str = "host_domain"
    min_rows_per_batch: int = 1000
    max_rows_per_batch: int | None = None

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.host_domain_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _balanced_chunks(self, idxs: list[int]) -> list[list[int]]:
        """Split a host's row indices into balanced ~equal chunks of <= max_rows_per_batch.

        Returns ``[idxs]`` unchanged when ``max_rows_per_batch`` is unset or the host
        already fits. Otherwise splits into ``ceil(n / max)`` chunks via the shared
        ``split_into_n_chunks`` helper, which sizes them evenly (differ by <= 1, no
        straggler), so every chunk is <= max_rows_per_batch.
        """
        n = len(idxs)
        if not self.max_rows_per_batch or n <= self.max_rows_per_batch:
            return [idxs]
        n_chunks = -(-n // self.max_rows_per_batch)  # ceil(n / max_rows_per_batch)
        return list(split_into_n_chunks(idxs, n_chunks))

    def process(self, batch: DocumentBatch) -> list[DocumentBatch]:
        df = batch.to_pandas()
        logger.debug("HostDomainGrouping: {} rows in", len(df))

        # Group rows by host_domain
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            host = str(row.get(self.host_domain_col) or "")
            groups[host].append(int(idx))

        # Separate large hosts (>= min_rows) from small hosts (bin-pack small hosts)
        large_host_dfs: list[pd.DataFrame] = []
        small_host_rows: list[int] = []

        for host, idxs in groups.items():  # noqa: B007
            if len(idxs) >= self.min_rows_per_batch:
                for chunk in self._balanced_chunks(idxs):
                    large_host_dfs.append(df.loc[chunk].reset_index(drop=True))
            else:
                small_host_rows.extend(idxs)

        # Bin-pack small hosts into batches of min_rows_per_batch
        packed_batches: list[pd.DataFrame] = []
        current_rows: list[int] = []
        for row_idx in small_host_rows:
            current_rows.append(row_idx)
            if len(current_rows) >= self.min_rows_per_batch:
                packed_batches.append(df.loc[current_rows].reset_index(drop=True))
                current_rows = []
        if current_rows:
            packed_batches.append(df.loc[current_rows].reset_index(drop=True))

        all_dfs = large_host_dfs + packed_batches
        if not all_dfs:
            return []

        return [
            DocumentBatch(
                dataset_name=batch.dataset_name,
                data=sub_df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )
            for sub_df in all_dfs
        ]
