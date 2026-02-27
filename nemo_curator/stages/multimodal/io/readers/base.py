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

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


@dataclass
class BaseMultimodalReader(ProcessingStage[FileGroupTask, MultiBatchTask]):
    """Base contract for multimodal readers."""

    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "base_multimodal_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sample_id", "position", "modality"]

    @staticmethod
    def reconcile_schema(inferred: pa.Schema) -> pa.Schema:
        """Build a schema with canonical types for reserved columns and inferred types for passthrough.

        Avoids unsafe downcasts from large_string -> string or large_binary -> binary
        which would cause offset overflow on large tables.
        """
        large_compat: dict[tuple[pa.DataType, pa.DataType], pa.DataType] = {
            (pa.large_string(), pa.string()): pa.large_string(),
            (pa.large_binary(), pa.binary()): pa.large_binary(),
            (pa.large_binary(), pa.large_binary()): pa.large_binary(),
            (pa.large_string(), pa.large_string()): pa.large_string(),
        }
        canonical = {f.name: f for f in MULTIMODAL_SCHEMA}
        fields: list[pa.Field] = []
        for f in inferred:
            if f.name not in canonical:
                fields.append(f)
                continue
            target = canonical[f.name]
            resolved_type = large_compat.get((f.type, target.type), target.type)
            fields.append(pa.field(f.name, resolved_type, nullable=target.nullable))
        return pa.schema(fields)
