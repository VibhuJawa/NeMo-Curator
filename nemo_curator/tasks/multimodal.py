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

"""Multimodal task type and schema for row-wise multimodal records.

Schema columns fall into two categories:

**Reserved columns** (``RESERVED_COLUMNS``) -- managed by pipeline stages:

    ==================  =============  ===========  ===============================================
    Column              Type           Category     Description
    ==================  =============  ===========  ===============================================
    ``sample_id``       string (req)   Identity     Unique document/sample identifier
    ``position``        int32 (req)    Identity     Position within sample (-1 for metadata rows)
    ``modality``        string (req)   Identity     One of: ``text``, ``image``, ``metadata``
    ``content_type``    string         Content      MIME type (e.g. ``text/plain``, ``image/jpeg``)
    ``text_content``    string         Content      Text payload for text rows
    ``binary_content``  large_binary   Content      Image bytes (populated by materialization)
    ``source_ref``      string         Internal     JSON locator: path, member, byte_offset, byte_size
    ``metadata_json``   string         Internal     Full JSON payload for metadata rows
    ``materialize_error`` string       Internal     Error message if materialization failed
    ==================  =============  ===========  ===============================================

**User columns** (passthrough) -- extra fields from source data added via the
``fields`` parameter on the reader. These flow through the pipeline untouched.
"""

import json
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
from loguru import logger

from .tasks import Task

MULTIMODAL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=True),
        pa.field("text_content", pa.string(), nullable=True),
        pa.field("binary_content", pa.large_binary(), nullable=True),
        pa.field("source_ref", pa.string(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=True),
        pa.field("materialize_error", pa.string(), nullable=True),
    ]
)

RESERVED_COLUMNS: frozenset[str] = frozenset(MULTIMODAL_SCHEMA.names)


@dataclass
class MultiBatchTask(Task[pa.Table | pd.DataFrame]):
    """Task carrying row-wise multimodal records.

    See module docstring for the full schema reference (reserved vs user columns).
    """

    REQUIRED_COLUMNS: frozenset[str] = frozenset(
        name for name, f in zip(MULTIMODAL_SCHEMA.names, MULTIMODAL_SCHEMA, strict=True) if not f.nullable
    )

    data: pa.Table | pd.DataFrame = field(default_factory=lambda: pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))

    # -- conversion --

    def to_pyarrow(self) -> pa.Table:
        if isinstance(self.data, pa.Table):
            return self.data
        if isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data, preserve_index=False)
        msg = f"Cannot convert {type(self.data)} to PyArrow table"
        raise TypeError(msg)

    def to_pandas(self) -> pd.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            return self.data
        if isinstance(self.data, pa.Table):
            return self.data.to_pandas(types_mapper=pd.ArrowDtype)
        msg = f"Cannot convert {type(self.data)} to Pandas DataFrame"
        raise TypeError(msg)

    # -- introspection --

    @property
    def num_items(self) -> int:
        return len(self.data)

    def get_columns(self) -> list[str]:
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        if isinstance(self.data, pa.Table):
            return self.data.column_names
        msg = f"Unsupported data type: {type(self.data)}"
        raise TypeError(msg)

    def validate(self) -> bool:
        if self.num_items <= 0:
            logger.warning(f"Task {self.task_id} has no items")
            return False
        columns = set(self.get_columns())
        missing = sorted(self.REQUIRED_COLUMNS - columns)
        if missing:
            logger.warning(f"Task {self.task_id} missing required columns: {missing}")
            return False
        return True

    # -- source_ref helpers --

    @staticmethod
    def build_source_ref(
        path: str | None,
        member: str | None,
        byte_offset: int | None = None,
        byte_size: int | None = None,
    ) -> str:
        """Build a ``source_ref`` JSON locator string."""
        return json.dumps(
            {"path": path, "member": member, "byte_offset": byte_offset, "byte_size": byte_size},
            ensure_ascii=True,
        )

    @staticmethod
    def parse_source_ref(source_value: str | None) -> dict[str, str | int | None]:
        """Parse a ``source_ref`` JSON string into a locator dict.

        Supports soft migration from older ``content_path``/``content_key`` payloads.
        """
        if source_value is None or pd.isna(source_value) or source_value == "":
            return {"path": None, "member": None, "byte_offset": None, "byte_size": None}
        parsed = json.loads(source_value)
        if not isinstance(parsed, dict):
            msg = "source_ref must decode to a JSON object"
            raise TypeError(msg)

        path = parsed.get("path", parsed.get("content_path"))
        member = parsed.get("member", parsed.get("content_key"))
        byte_offset = parsed.get("byte_offset")
        byte_size = parsed.get("byte_size")

        return {
            "path": path if path is None else str(path),
            "member": member if member is None else str(member),
            "byte_offset": int(byte_offset) if byte_offset is not None else None,
            "byte_size": int(byte_size) if byte_size is not None else None,
        }

    def with_parsed_source_ref_columns(self, prefix: str = "_src_") -> pd.DataFrame:
        """Return a DataFrame copy with parsed ``source_ref`` columns added.

        Columns: ``{prefix}path``, ``{prefix}member``, ``{prefix}byte_offset``, ``{prefix}byte_size``.
        """
        df = self.to_pandas().copy()
        parsed = [self.parse_source_ref(value) for value in df["source_ref"].tolist()]
        parsed_df = pd.DataFrame.from_records(
            parsed,
            columns=["path", "member", "byte_offset", "byte_size"],
        )
        for col in parsed_df.columns:
            df[f"{prefix}{col}"] = parsed_df[col].to_numpy(copy=False)
        return df
