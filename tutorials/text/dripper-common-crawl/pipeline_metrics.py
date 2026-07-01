#!/usr/bin/env python3
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

"""Small metrics helper for the Dripper Common Crawl tutorial scripts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageMetrics:
    """Record coarse wall-clock metrics for a tutorial stage."""

    stage_name: str
    shard_index: int = 0
    num_shards: int = 1
    n_gpus: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    ended_at: float | None = None
    total_pages: int = 0
    errors: int = 0

    def start(self) -> None:
        self.started_at = time.time()
        self.ended_at = None

    def finish(self, *, total_pages: int = 0, errors: int = 0) -> None:
        self.ended_at = time.time()
        self.total_pages = int(total_pages)
        self.errors = int(errors)

    def to_dict(self) -> dict[str, Any]:
        start = self.started_at
        end = self.ended_at
        elapsed = None if start is None or end is None else max(0.0, end - start)
        return {
            "stage_name": self.stage_name,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "n_gpus": self.n_gpus,
            "started_at": start,
            "ended_at": end,
            "elapsed_s": elapsed,
            "total_pages": self.total_pages,
            "errors": self.errors,
            "pages_per_s": None if not elapsed else self.total_pages / elapsed,
            "extra": self.extra,
        }

    def save(self, output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.stage_name}_metrics_shard_{self.shard_index:04d}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path
