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

import os
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa


def s3_storage_options_from_env() -> dict[str, Any]:
    """Build S3/PBSS storage options from standard AWS environment variables.

    Automatically adds path-style and performance settings when a non-AWS
    S3-compatible endpoint is detected (``AWS_ENDPOINT_URL_S3`` / ``AWS_ENDPOINT_URL``).
    """
    opts: dict[str, Any] = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        opts["endpoint"] = endpoint
    if key_id := os.environ.get("AWS_ACCESS_KEY_ID"):
        opts["aws_access_key_id"] = key_id
    if secret := os.environ.get("AWS_SECRET_ACCESS_KEY"):
        opts["aws_secret_access_key"] = secret
    opts["aws_region"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    if endpoint:
        opts["virtual_hosted_style_request"] = "false"
        opts["new_table_data_storage_version"] = "stable"
        opts["new_table_enable_v2_manifest_paths"] = "true"
        opts["io_threads"] = "128"
    return opts


def df_to_typed_arrow(df: "pd.DataFrame", schema: "pa.Schema | None") -> "pa.Table":  # type: ignore[name-defined]  # noqa: F821
    """Convert a pandas DataFrame to a PyArrow table with schema-aware type casting.

    Handles ``large_binary`` and ``large_string`` columns explicitly, which
    ``pa.Table.from_pandas()`` can misidentify from object-dtype Series.
    """
    import pyarrow as pa

    if schema is None:
        return pa.Table.from_pandas(df)
    arrays: dict[str, pa.Array] = {}
    for fld in schema:
        if fld.name not in df.columns:
            arrays[fld.name] = pa.nulls(len(df), type=fld.type)
            continue
        col = df[fld.name]
        if pa.types.is_large_binary(fld.type) or pa.types.is_binary(fld.type):
            vals = [
                v.encode("utf-8") if isinstance(v, str) else (v if isinstance(v, (bytes, bytearray)) else b"")
                for v in col
            ]
            arrays[fld.name] = pa.array(vals, type=fld.type)
        elif pa.types.is_large_string(fld.type) or pa.types.is_string(fld.type):
            arrays[fld.name] = pa.array([str(v) if v is not None else "" for v in col], type=fld.type)
        else:
            arrays[fld.name] = pa.array(col.tolist(), type=fld.type)
    return pa.table(arrays, schema=schema)


def batched(iterable: Iterable[Any], n: int) -> Iterator[tuple[Any, ...]]:
    """
    Batch an iterable into lists of size n.

    Args:
      iterable (Iterable[Any]): The iterable to batch
      n (int): The size of the batch

    Returns:
        Iterator[tuple[...]]: An iterator of tuples, each containing n elements from the iterable
    """
    if n < 1:
        msg = "n must be at least one"
        raise ValueError(msg)
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
