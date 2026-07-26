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

"""Shared cache for materialized image payloads."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class PayloadCache:
    """Content-addressed cache for image payloads on a shared filesystem.

    Interleaved corpora reference the same image many times -- on MINT-1T HTML,
    1.58B image occurrences resolve to 356M unique images, so the average image
    is fetched 4.4 times per pass. Pointing this cache at a shared filesystem
    turns those repeats into local reads.

    Keys are hashed into a two-level fan-out because a flat directory of
    millions of entries is slower to stat than the object store it replaces.
    Writes land in a sibling temporary file and are renamed into place, so a
    reader never observes a partial payload and concurrent writers of the same
    (immutable) payload are harmless.

    Cache faults are never fatal: a read error is a miss and a write error is
    skipped, both logged.

    Args:
        root: Cache directory, typically on Lustre/Weka.
    """

    root: Path

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / digest[:2] / digest[2:4] / digest

    def get(self, key: str) -> bytes | None:
        """Return the cached payload for ``key``, or ``None`` on a miss."""
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            logger.warning(f"payload cache read failed for {key!r}: {error}")
            return None

    def put(self, key: str, payload: bytes) -> None:
        """Store ``payload`` under ``key``, overwriting any existing entry."""
        path = self._path(key)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload)
            temporary.replace(path)
        except OSError as error:
            logger.warning(f"payload cache write failed for {key!r}: {error}")
            temporary.unlink(missing_ok=True)
