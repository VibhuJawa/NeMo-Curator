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

"""Multimodal IO benchmark for WebDataset and Parquet reader/writer stages."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger
from utils import write_benchmark_results

from nemo_curator.stages.multimodal.io.readers.parquet import ParquetMultimodalReaderStage
from nemo_curator.stages.multimodal.io.readers.webdataset import WebDatasetReaderStage
from nemo_curator.stages.multimodal.io.writers.multimodal import MultimodalWriterStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch


def _as_single_batch(result: MultimodalBatch | list[MultimodalBatch]) -> MultimodalBatch:
    if not isinstance(result, list):
        return result
    if len(result) == 1:
        return result[0]
    table = pa.concat_tables([batch.data for batch in result], promote_options="default")
    metadata_tables = [batch.metadata_index for batch in result if batch.metadata_index is not None]
    metadata = None
    if metadata_tables:
        metadata = pa.concat_tables(metadata_tables, promote_options="default") if len(metadata_tables) > 1 else metadata_tables[0]
    return MultimodalBatch(
        task_id=result[0].task_id,
        dataset_name=result[0].dataset_name,
        data=table,
        metadata_index=metadata,
    )


def _directory_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _discover_shards(input_dir: Path, num_shards: int) -> list[Path]:
    shards = sorted(input_dir.glob("*.tar"))
    if len(shards) < num_shards:
        msg = f"Requested {num_shards} shards but found {len(shards)} in {input_dir}"
        raise ValueError(msg)
    return shards[:num_shards]


def _expand_shards(shards: list[Path], num_tasks: int) -> list[Path]:
    """Return a task-length shard sequence, replaying shards when needed."""
    if num_tasks <= len(shards):
        return shards[:num_tasks]
    repeats, remainder = divmod(num_tasks, len(shards))
    return shards * repeats + shards[:remainder]


def _benchmark_webdataset_reader(
    shards: list[Path],
    load_binary: bool,
) -> tuple[dict[str, Any], list[MultimodalBatch]]:
    stage = WebDatasetReaderStage(load_binary=load_binary, sample_format="auto")
    elapsed = 0.0
    rows = 0
    image_rows = 0
    batches: list[MultimodalBatch] = []
    for idx, shard in enumerate(shards):
        task = FileGroupTask(task_id=f"reader-{idx:05d}", dataset_name="benchmark", data=[str(shard)])
        start = time.perf_counter()
        batch = _as_single_batch(stage.process(task))
        elapsed += time.perf_counter() - start
        rows += batch.data.num_rows
        image_rows += int(pc.sum(pc.cast(pc.equal(batch.data["modality"], "image"), "int64")).as_py() or 0)
        batches.append(batch)
    return (
        {
            "total_s": elapsed,
            "rows": rows,
            "image_rows": image_rows,
            "throughput_rows_per_s": rows / elapsed if elapsed > 0 else 0.0,
        },
        batches,
    )


def _benchmark_writers(
    batches: list[MultimodalBatch],
    artifacts_dir: Path,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    results: dict[str, Any] = {}
    parquet_pairs: list[tuple[str, str]] = []

    for output_format, suffix in (("webdataset", "tar"), ("parquet", "parquet"), ("arrow", "arrow")):
        stage = MultimodalWriterStage(
            output_path=str(artifacts_dir / output_format / f"out.{suffix}"),
            output_format=output_format,  # type: ignore[arg-type]
            image_payload_policy="preserve",
        )
        elapsed = 0.0
        for batch in batches:
            start = time.perf_counter()
            output_task = stage.process(batch)
            elapsed += time.perf_counter() - start
            if output_format == "parquet":
                parquet_pairs.append((output_task.data[0], output_task.data[1]))
        output_dir = artifacts_dir / output_format
        results[f"{output_format}_write_total_s"] = elapsed
        results[f"{output_format}_write_output_bytes"] = _directory_size_bytes(output_dir)
    return results, parquet_pairs


def _benchmark_parquet_reader(parquet_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    stage = ParquetMultimodalReaderStage()
    data_paths = [pair[0] for pair in parquet_pairs]
    metadata_paths = [pair[1] for pair in parquet_pairs]
    data_task = FileGroupTask(task_id="parquet-data", dataset_name="benchmark", data=data_paths)
    metadata_task = FileGroupTask(task_id="parquet-meta", dataset_name="benchmark", data=metadata_paths)
    start = time.perf_counter()
    batch = _as_single_batch(stage.process((data_task, metadata_task)))
    elapsed = time.perf_counter() - start
    return {
        "parquet_reader_total_s": elapsed,
        "parquet_reader_rows": batch.data.num_rows,
        "parquet_reader_throughput_rows_per_s": batch.data.num_rows / elapsed if elapsed > 0 else 0.0,
    }


def _benchmark_parquet_roundtrip_row_groups(
    batches: list[MultimodalBatch],
    artifacts_dir: Path,
) -> dict[str, Any]:
    default_dir = artifacts_dir / "parquet_roundtrip_default"
    one_rg_dir = artifacts_dir / "parquet_roundtrip_one_rg"
    default_dir.mkdir(parents=True, exist_ok=True)
    one_rg_dir.mkdir(parents=True, exist_ok=True)

    writer_stage = MultimodalWriterStage(
        output_path=str(artifacts_dir / "tmp.parquet"),
        output_format="parquet",
        image_payload_policy="preserve",
    )

    default_write_s = 0.0
    default_read_s = 0.0
    one_rg_write_s = 0.0
    one_rg_read_s = 0.0
    default_row_groups = 0
    one_rg_row_groups = 0

    for idx, batch in enumerate(batches):
        table = writer_stage._build_output_table(batch)

        default_path = default_dir / f"{idx:05d}.parquet"
        start = time.perf_counter()
        pq.write_table(table, default_path)
        default_write_s += time.perf_counter() - start
        default_row_groups += pq.ParquetFile(default_path).metadata.num_row_groups
        start = time.perf_counter()
        _ = pq.read_table(default_path)
        default_read_s += time.perf_counter() - start

        one_rg_path = one_rg_dir / f"{idx:05d}.parquet"
        start = time.perf_counter()
        pq.write_table(table, one_rg_path, row_group_size=table.num_rows)
        one_rg_write_s += time.perf_counter() - start
        one_rg_row_groups += pq.ParquetFile(one_rg_path).metadata.num_row_groups
        start = time.perf_counter()
        _ = pq.read_table(one_rg_path)
        one_rg_read_s += time.perf_counter() - start

    file_count = max(len(batches), 1)
    return {
        "parquet_roundtrip_default_write_s": default_write_s,
        "parquet_roundtrip_default_read_s": default_read_s,
        "parquet_roundtrip_default_total_s": default_write_s + default_read_s,
        "parquet_roundtrip_default_row_groups_total": default_row_groups,
        "parquet_roundtrip_default_row_groups_per_file": default_row_groups / file_count,
        "parquet_roundtrip_one_rg_write_s": one_rg_write_s,
        "parquet_roundtrip_one_rg_read_s": one_rg_read_s,
        "parquet_roundtrip_one_rg_total_s": one_rg_write_s + one_rg_read_s,
        "parquet_roundtrip_one_rg_row_groups_total": one_rg_row_groups,
        "parquet_roundtrip_one_rg_row_groups_per_file": one_rg_row_groups / file_count,
        "parquet_roundtrip_one_rg_vs_default_write_speedup_x": default_write_s / one_rg_write_s if one_rg_write_s else 0.0,
        "parquet_roundtrip_one_rg_vs_default_read_speedup_x": default_read_s / one_rg_read_s if one_rg_read_s else 0.0,
        "parquet_roundtrip_one_rg_vs_default_total_speedup_x": (default_write_s + default_read_s) / (one_rg_write_s + one_rg_read_s) if (one_rg_write_s + one_rg_read_s) else 0.0,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards <= 0:
        msg = f"--num-shards must be > 0, got {args.num_shards}"
        raise ValueError(msg)
    if args.num_tasks <= 0:
        msg = f"--num-tasks must be > 0, got {args.num_tasks}"
        raise ValueError(msg)
    input_dir = Path(args.input_dir).resolve()
    unique_shards = _discover_shards(input_dir, args.num_shards)
    shards = _expand_shards(unique_shards, args.num_tasks)
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running multimodal IO benchmark on {} tasks ({} unique shards) from {}",
        len(shards),
        len(unique_shards),
        input_dir,
    )
    logger.info("Artifacts output directory: {}", artifacts_dir)

    reader_no_bin_metrics, _ = _benchmark_webdataset_reader(shards, load_binary=False)
    reader_bin_metrics, binary_batches = _benchmark_webdataset_reader(shards, load_binary=True)
    writer_metrics, parquet_pairs = _benchmark_writers(binary_batches, artifacts_dir)
    parquet_reader_metrics = _benchmark_parquet_reader(parquet_pairs)
    roundtrip_metrics = _benchmark_parquet_roundtrip_row_groups(binary_batches, artifacts_dir)

    metrics: dict[str, Any] = {
        "is_success": True,
        "num_tasks": len(shards),
        "num_unique_shards": len(unique_shards),
        "num_replayed_tasks": max(0, len(shards) - len(unique_shards)),
        "total_input_bytes": sum(shard.stat().st_size for shard in shards),
    }
    metrics.update({f"webdataset_reader_no_binary_{k}": v for k, v in reader_no_bin_metrics.items()})
    metrics.update({f"webdataset_reader_binary_{k}": v for k, v in reader_bin_metrics.items()})
    metrics.update(writer_metrics)
    metrics.update(parquet_reader_metrics)
    metrics.update(roundtrip_metrics)

    return {
        "params": {
            "input_dir": str(input_dir),
            "artifacts_dir": str(artifacts_dir),
            "num_shards": args.num_shards,
            "num_tasks": args.num_tasks,
        },
        "metrics": metrics,
        "tasks": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multimodal reader/writer stages.")
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Directory where benchmark JSON outputs are written.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input WebDataset *.tar shards.")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory where benchmark output artifacts are written.")
    parser.add_argument("--num-shards", type=int, default=14, help="Number of shards to process from input-dir.")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=14,
        help="Number of reader/writer tasks to run; if greater than num-shards, shards are replayed from the start.",
    )
    args = parser.parse_args()

    results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
