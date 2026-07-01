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

"""Config reader for Dripper Common Crawl tutorial scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DripperConfig:
    """Minimal typed view of the tutorial YAML config."""

    raw: dict[str, Any] = field(default_factory=dict)
    num_shards: int = 1
    gpu_pipeline_shards: int = 1
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DripperConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        sharding = data.get("sharding", {}) or {}
        resources = data.get("resources", {}) or {}
        return cls(
            raw=data,
            num_shards=int(sharding.get("num_shards", 1)),
            gpu_pipeline_shards=int(sharding.get("gpu_pipeline_shards", sharding.get("num_shards", 1))),
            resources=resources,
        )
