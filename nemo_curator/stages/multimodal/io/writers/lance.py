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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .base import BaseMultimodalWriter

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MultimodalLanceWriterStage(BaseMultimodalWriter):
    """Write multimodal rows to Lance format with optional binary materialization."""

    file_extension: str = "lance"
    name: str = "multimodal_lance_writer"

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        import lance

        write_kwargs.pop("index", None)
        table = pa.Table.from_pandas(df, preserve_index=False)
        with self._time_metric("lance_write_s"):
            lance.write_dataset(table, file_path, mode="overwrite", **write_kwargs)
