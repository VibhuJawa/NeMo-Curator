# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

def sort_multimodal_table(table: pa.Table) -> pa.Table:
    """Sort rows by ``sample_id``, ``position``, and ``modality``."""
    if table.num_rows == 0:
        return table
    return table.sort_by([("sample_id", "ascending"), ("position", "ascending"), ("modality", "ascending")])
