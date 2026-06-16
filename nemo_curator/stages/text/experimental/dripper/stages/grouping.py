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

if TYPE_CHECKING:
    import pandas as pd


@dataclass(kw_only=True)
class HostDomainGroupingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Bin-pack mixed-host batches into per-host-domain DocumentBatches.

    IS_FANOUT_STAGE: one input batch of N rows from multiple hosts becomes M output
    batches (one per host_domain with >= min_rows_per_batch rows).
    Small hosts are bin-packed together to meet min_rows_per_batch.
    """

    name: str = "HostDomainGroupingStage"
    host_domain_col: str = "host_domain"
    min_rows_per_batch: int = 1000

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.host_domain_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

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
                large_host_dfs.append(df.loc[idxs].reset_index(drop=True))
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
