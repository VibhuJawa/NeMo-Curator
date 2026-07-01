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

"""Lossless serialization helpers for Dripper layout mapping templates."""

from __future__ import annotations

import base64
import binascii
import pickle
from typing import Any

_PICKLE_B64_PREFIX = "pickle_b64:"


def serialize_mapping_data(mapping_data: dict[str, Any] | None) -> str:
    """Serialize a mapping template without destroying tuple keys."""

    if not mapping_data:
        return ""
    payload = pickle.dumps(mapping_data, protocol=pickle.HIGHEST_PROTOCOL)
    return _PICKLE_B64_PREFIX + base64.b64encode(payload).decode("ascii")


def parse_mapping_data(raw: object) -> dict[str, Any] | None:
    """Parse strict pickle+base64 mapping blobs."""

    if raw is None:
        return None
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    text = text.strip()
    if not text:
        return None

    if not text.startswith(_PICKLE_B64_PREFIX):
        return None
    return _loads_pickle_b64(text[len(_PICKLE_B64_PREFIX) :])


def _loads_pickle_b64(payload: str) -> dict[str, Any] | None:
    try:
        decoded = base64.b64decode(payload.encode("ascii"), validate=True)
        value = pickle.loads(decoded)  # noqa: S301 - internal pipeline artifact
    except (binascii.Error, UnicodeEncodeError, pickle.UnpicklingError, EOFError, ValueError):
        return None
    return value if isinstance(value, dict) else None
