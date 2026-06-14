"""DripperConfig — typed configuration for the Dripper CC pipeline.

Replaces the raw YAML dict with a validated dataclass that:
- Has typed fields with documented defaults
- Validates required fields in __post_init__
- Can load from YAML: DripperConfig.from_yaml("configs/template.yaml")

Usage::

    cfg = DripperConfig.from_yaml("configs/my_run.yaml")
    runner = PipelineRunner(cfg.to_raw_dict(), args)
    runner.run()
"""
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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResources:
    """Slurm resource allocation for one pipeline stage.

    Args:
        partition: Slurm partition name (e.g. ``"cpu_short"``, ``"batch"``).
        cpus: Number of CPUs per task.
        mem: Memory string accepted by Slurm (e.g. ``"230G"``).
        time: Wall-clock time limit in ``HH:MM:SS`` format.
        gpus_per_node: GPUs requested per node; ``0`` means no GPU allocation.
    """

    partition: str
    cpus: int = 8
    mem: str = "32G"
    time: str = "01:00:00"
    gpus_per_node: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageResources:
        """Build a ``StageResources`` from a raw YAML mapping.

        Unknown keys are silently ignored so that stage-specific extras
        (e.g. ``cpus_per_actor``, ``batch_size``) do not cause errors.

        Args:
            d: Raw dictionary (typically from ``resources.<stage>`` in the YAML).

        Returns:
            A ``StageResources`` populated from *d*.
        """
        return cls(
            partition=d["partition"],
            cpus=int(d.get("cpus", 8)),
            mem=str(d.get("mem", "32G")),
            time=str(d.get("time", "01:00:00")),
            gpus_per_node=int(d.get("gpus_per_node", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a plain dict compatible with ``_sbatch_header``."""
        return {
            "partition": self.partition,
            "cpus": self.cpus,
            "mem": self.mem,
            "time": self.time,
            "gpus_per_node": self.gpus_per_node,
        }


@dataclass
class DripperConfig:
    """Full configuration for the Dripper CC clustering pipeline.

    Load from YAML::

        cfg = DripperConfig.from_yaml("configs/template.yaml")

    This class is the single authoritative source of truth for all pipeline
    parameters.  The raw ``dict`` formerly produced by ``load_config()`` in
    ``run_pipeline.py`` can be obtained via :meth:`to_raw_dict` for backward
    compatibility with the existing ``PipelineRunner`` / ``build_snapshot_run``
    callsites until they are migrated to consume ``DripperConfig`` directly.

    Args:
        cluster: Cluster connection settings (login node, venv paths, etc.).
            Required keys: ``login_node``, ``dc_node``, ``account``, ``venv``,
            ``remote_repo``.
        output_base: Output directory template; ``{snapshot}`` and ``{ts}``
            (``YYYYMMDD_HHMMSS``) are expanded at runtime.
        snapshots: List of CC snapshot entries.  Each entry must have a ``name``
            and ``manifest`` key; ``validation_baseline`` is optional.
        sharding: Shard counts per stage.  Defaults: ``num_shards=80``,
            ``gpu_pipeline_shards=80``.
        validation: F1 validation settings.  See ``configs/template.yaml`` for
            the full set of keys.
        resources: Per-stage Slurm resource allocations, keyed by stage name.
            Values are raw dicts (passthrough to ``_sbatch_header``).
    """

    cluster: dict[str, str]
    output_base: str
    snapshots: list[dict[str, str]]
    sharding: dict[str, int] = field(
        default_factory=lambda: {
            "num_shards": 80,
            "gpu_pipeline_shards": 80,
        }
    )
    validation: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "f1_threshold": 0.85,
            "halt_on_failure": False,
            "sample_size": 10_000,
        }
    )
    resources: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        required_cluster_keys = {"login_node", "dc_node", "account", "venv", "remote_repo"}
        missing = required_cluster_keys - set(self.cluster)
        if missing:
            msg = f"Missing required cluster keys: {missing}"
            raise ValueError(msg)
        if not self.snapshots:
            msg = "At least one snapshot must be specified"
            raise ValueError(msg)
        for i, snap in enumerate(self.snapshots):
            for key in ("name", "manifest"):
                if key not in snap:
                    msg = f"snapshots[{i}] is missing required key '{key}'"
                    raise ValueError(msg)

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> DripperConfig:
        """Load config from a YAML file.

        Args:
            path: Path to the YAML configuration file
                  (e.g. ``"configs/template.yaml"``).

        Returns:
            A fully validated :class:`DripperConfig` instance.

        Raises:
            ImportError: If ``pyyaml`` is not installed.
            ValueError: If required cluster keys or snapshots are absent.
        """
        try:
            import yaml
        except ImportError as exc:
            msg = "pyyaml is required to load DripperConfig from YAML. Install with: pip install pyyaml"
            raise ImportError(msg) from exc

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        return cls(
            cluster=raw["cluster"],
            output_base=raw["output_base"],
            snapshots=raw["snapshots"],
            sharding=raw.get("sharding", {}),
            validation=raw.get("validation", {}),
            resources=raw.get("resources", {}),
        )

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def num_shards(self) -> int:
        """Total shard count for stage1a, stage1b, and stage3 arrays."""
        return int(self.sharding.get("num_shards", 80))

    @property
    def gpu_pipeline_shards(self) -> int:
        """Shard count for the GPU pipeline (stages 1c+2+2b)."""
        return int(self.sharding.get("gpu_pipeline_shards", 80))

    def stage_resources(self, stage: str) -> StageResources:
        """Return the typed :class:`StageResources` for *stage*.

        Falls back to a minimal default if the stage is not present in the
        ``resources`` section so that dry-run / test scenarios work without a
        complete YAML.

        Args:
            stage: Stage key as used in ``configs/template.yaml``
                   (e.g. ``"stage3"``, ``"gpu_pipeline"``).

        Returns:
            A :class:`StageResources` for the requested stage.
        """
        raw = self.resources.get(stage, {})
        if not raw or "partition" not in raw:
            # Sensible fallback so test/dry-run paths don't crash
            raw = {"partition": "cpu_short", **raw}
        return StageResources.from_dict(raw)

    # ------------------------------------------------------------------ #
    # Backward-compat serialisation                                        #
    # ------------------------------------------------------------------ #

    def to_raw_dict(self) -> dict[str, Any]:
        """Return the raw dict representation expected by ``PipelineRunner``.

        This is the same structure that ``load_config()`` in ``run_pipeline.py``
        produced, enabling incremental migration: callers that still expect the
        raw dict can call ``cfg.to_raw_dict()`` instead of ``load_config()``.

        Returns:
            Dict with keys ``cluster``, ``output_base``, ``snapshots``,
            ``sharding``, ``validation``, and ``resources``.
        """
        return {
            "cluster": self.cluster,
            "output_base": self.output_base,
            "snapshots": self.snapshots,
            "sharding": self.sharding,
            "validation": self.validation,
            "resources": self.resources,
        }
