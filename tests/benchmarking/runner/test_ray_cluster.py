# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.ray_cluster import select_ray_client_class
from nemo_curator.core.client import RayClient, SlurmRayClient


def test_single_node_allocation_uses_ray_client(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_NNODES", "1")
    assert select_ray_client_class() is RayClient


def test_multi_node_allocation_uses_slurm_ray_client(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_NNODES", "2")
    assert select_ray_client_class() is SlurmRayClient
