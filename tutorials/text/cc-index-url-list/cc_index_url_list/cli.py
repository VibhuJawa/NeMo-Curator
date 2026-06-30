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

"""Shared command-line helpers for the CC Index URL list tutorial."""

from __future__ import annotations

import argparse
import os
import sys

from loguru import logger

from cc_index_url_list.config import DEFAULT_CONFIG, DEFAULT_REGION


def configure_logging() -> None:
    """Redact credential values from loguru output emitted by Curator internals."""
    secret_values = {
        value
        for value in (
            os.environ.get("PBSS_SECRET_ACCESS_KEY"),
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
            os.environ.get("AWS_SESSION_TOKEN"),
        )
        if value
    }

    logger.remove()

    def redacting_sink(message: object) -> None:
        text = str(message)
        for secret in secret_values:
            text = text.replace(secret, "<redacted>")
        sys.stderr.write(text)

    logger.add(redacting_sink, level=os.environ.get("CC_INDEX_URL_LIST_LOG_LEVEL", "INFO"))


def positive_int(value: str) -> int:
    """Argparse type for positive integer options."""
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def add_common_args(parser: argparse.ArgumentParser, *, include_gpu_args: bool) -> None:
    """Add arguments shared by both phase entrypoints."""
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--max-files-per-crawl",
        type=positive_int,
        default=None,
        help="Use only the first N parquet files per crawl for smoke tests",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print configured inputs without running Curator")
    parser.add_argument("--region-name", default=DEFAULT_REGION, help=f"S3 region name (default: {DEFAULT_REGION})")
    parser.add_argument(
        "--slurm-ray",
        action="store_true",
        help="Use Curator SlurmRayClient; required for multi-node srun launches",
    )
    parser.add_argument(
        "--ray-temp-dir",
        default=None,
        help="Ray temporary directory",
    )
    parser.add_argument("--ray-num-cpus", type=positive_int, default=None, help="CPUs to expose to Ray per node")
    if include_gpu_args:
        parser.add_argument("--ray-num-gpus", type=positive_int, default=None, help="GPUs to expose to Ray per node")
    parser.add_argument(
        "--ray-worker-connect-timeout",
        type=positive_int,
        default=600,
        help="Seconds for SlurmRayClient head to wait for worker nodes",
    )
    parser.add_argument(
        "--ray-port-broadcast-dir",
        default=None,
        help="Shared directory for SlurmRayClient head-port handoff; defaults under --output",
    )
    parser.add_argument("--disable-ray-dashboard", action="store_true", help="Disable Ray dashboard startup")
