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

"""Ray client helpers for local and Slurm-backed tutorial runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import argparse


class RayClientProtocol(Protocol):
    """Common lifecycle methods for Curator Ray clients."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


def default_ray_temp_dir() -> str:
    """Return a short local Ray temp directory."""
    return str(Path(tempfile.gettempdir()) / f"{os.environ.get('USER', 'user')}_nemo_curator_ray")


def create_ray_client(args: argparse.Namespace) -> RayClientProtocol:
    """Create the Ray client that owns the workflow cluster."""
    from nemo_curator.core.client import RayClient, SlurmRayClient

    client_kwargs = {
        "include_dashboard": not args.disable_ray_dashboard,
        "ray_temp_dir": args.ray_temp_dir or os.environ.get("RAY_TMPDIR") or default_ray_temp_dir(),
        "num_cpus": args.ray_num_cpus,
        "num_gpus": getattr(args, "ray_num_gpus", None),
    }
    if args.slurm_ray:
        broadcast_dir = args.ray_port_broadcast_dir or str(Path(args.output) / "_ray_port_broadcast")
        os.environ.setdefault("RAY_PORT_BROADCAST_DIR", broadcast_dir)
        return SlurmRayClient(
            **client_kwargs,
            worker_connect_timeout_s=args.ray_worker_connect_timeout,
        )
    return RayClient(**client_kwargs)
