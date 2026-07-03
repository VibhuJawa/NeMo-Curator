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

"""Mark MINT-1T HTML image URL presence with persistent GPU exact joins.

Pipeline::

    InterleavedParquetReader
        -> GpuExactKeyLookupStage
        -> InterleavedParquetWriterStage

Input MINT rows store the exact image URL in ``source_ref`` only for image
rows. Output preserves all rows and appends nullable boolean ``image_present``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved import GpuExactKeyLookupStage
from nemo_curator.stages.interleaved.io import InterleavedParquetReader, InterleavedParquetWriterStage


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        msg = "storage options must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _manifest_reference_files(reference_path: Path, manifest: dict[str, Any]) -> list[str]:
    if manifest.get("format") != "nemo_curator_exact_key_reference_v1":
        msg = f"Unsupported exact-key reference manifest: {reference_path / 'manifest.json'}"
        raise ValueError(msg)
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        msg = "Reference manifest must declare at least one segment"
        raise ValueError(msg)

    reference_files: list[str] = []
    for segment in segments:
        relative_path = Path(segment["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            msg = f"Reference manifest contains an unsafe segment path: {relative_path}"
            raise ValueError(msg)
        path = reference_path / relative_path
        if not path.is_file():
            msg = f"Reference segment does not exist: {path}"
            raise FileNotFoundError(msg)
        expected_bytes = int(segment["bytes"])
        if path.stat().st_size != expected_bytes:
            msg = f"Reference segment size changed: {path}"
            raise ValueError(msg)
        reference_files.append(str(path))

    discovered = {str(path) for path in reference_path.glob("part-*.parquet")}
    unexpected = discovered.difference(reference_files)
    if unexpected:
        msg = f"Reference directory contains segments absent from its manifest: {sorted(unexpected)}"
        raise ValueError(msg)
    return reference_files


def _reference_identity(reference_path: Path, expected_rows: int | None) -> tuple[list[str], int | None]:
    manifest_path = reference_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        reference_files = _manifest_reference_files(reference_path, manifest)
        manifest_rows = int(manifest["rows"])
        if expected_rows is not None and expected_rows != manifest_rows:
            msg = f"Expected {expected_rows} reference rows but manifest declares {manifest_rows}"
            raise ValueError(msg)
        expected_rows = manifest_rows
    else:
        reference_files = sorted(str(path) for path in reference_path.glob("part-*.parquet"))
        if not reference_files:
            msg = f"No part-*.parquet reference files found under {reference_path}"
            raise FileNotFoundError(msg)
    return reference_files, expected_rows


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", required=True, help="Normalized MINT interleaved Parquet path")
    parser.add_argument("--reference-path", required=True, help="Local exact-key sidecar directory")
    parser.add_argument("--output-path", required=True, help="Presence-enriched Parquet output path")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--expected-reference-rows", type=int, default=None)
    parser.add_argument("--num-cpus", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--task-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-mib", type=int, default=512)
    parser.add_argument("--object-store-gib", type=int, default=32)
    parser.add_argument("--storage-options-json", type=_json_object, default={})
    parser.add_argument("--mode", choices=["ignore", "overwrite", "error"], default="error")
    args = parser.parse_args()

    if args.num_gpus <= 0 or args.task_batch_size <= 0:
        msg = "num-gpus and task-batch-size must be greater than zero"
        raise ValueError(msg)
    workers = args.num_workers or args.num_gpus
    reference_files, expected_rows = _reference_identity(
        Path(args.reference_path),
        args.expected_reference_rows,
    )
    io_kwargs = {"storage_options": args.storage_options_json} if args.storage_options_json else {}

    pipeline = Pipeline(
        name="mint_html_gpu_image_presence",
        description="MINT interleaved Parquet -> exact GPU URL presence -> Parquet",
    )
    pipeline.add_stage(
        InterleavedParquetReader(
            file_paths=args.input_path,
            files_per_partition=1,
            max_batch_bytes=args.max_batch_mib * 1024**2,
            read_kwargs=io_kwargs,
        )
    )
    pipeline.add_stage(
        GpuExactKeyLookupStage(
            reference_files=reference_files,
            reference_key_column="url",
            input_key_column="source_ref",
            presence_column="image_present",
            expected_reference_rows=expected_rows,
        ).with_(num_workers=workers, batch_size=args.task_batch_size)
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_path,
            materialize_on_write=False,
            mode=args.mode,
            write_kwargs={"compression": "zstd", **io_kwargs},
        )
    )

    print(pipeline.describe())
    ray_client = RayClient(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        object_store_memory=args.object_store_gib * 1024**3,
        include_dashboard=False,
    )
    try:
        ray_client.start()
        pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint_path)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
