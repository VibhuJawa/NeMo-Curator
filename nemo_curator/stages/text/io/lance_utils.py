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

import base64
import pickle
from typing import Any

import pyarrow as pa

LANCE_ROWADDR_COLUMN = "__lance_rowaddr"
LANCE_FRAGID_COLUMN = "__lance_fragid"


def object_to_base64(value: object) -> str:
    """Encode Lance/Ray checkpoint payloads the same way lance-ray does internally."""
    return base64.b64encode(pickle.dumps(value)).decode("ascii")


def object_from_base64(value: str) -> object:
    return pickle.loads(base64.b64decode(value))  # noqa: S301 - checkpoint payloads are produced by Curator.


def schema_to_json_value(schema: pa.Schema) -> dict[str, object]:
    from lance.schema import schema_to_json

    return schema_to_json(schema)


def schema_from_json_value(value: dict[str, object]) -> pa.Schema:
    from lance.schema import json_to_schema

    return json_to_schema(value)


def lance_dataset_kwargs(
    storage_options: dict[str, Any] | None = None,
    version: int | str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    if version is not None:
        kwargs["version"] = version
    return kwargs
