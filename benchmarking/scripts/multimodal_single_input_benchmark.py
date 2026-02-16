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

"""Single-input multimodal benchmark with integrity checks.

Builds one combined WebDataset tar from selected shards, benchmarks reader/writer
paths on that single input artifact, and verifies output integrity:
- parquet exact row+metadata roundtrip equality
- webdataset semantic equality (text aggregate + image payload hashes)
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
from loguru import logger
from utils import write_benchmark_results

from nemo_curator.stages.multimodal.io.readers.parquet import ParquetMultimodalReaderStage
from nemo_curator.stages.multimodal.io.readers.webdataset import WebDatasetReaderStage
from nemo_curator.stages.multimodal.io.writers.multimodal import MultimodalWriterStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch


def _discover_shards(input_dir: Path, num_shards: int) -> list[Path]:
    shards = sorted(input_dir.glob("*.tar"))
    if len(shards) < num_shards:
        msg = f"Requested {num_shards} shards but found {len(shards)} in {input_dir}"
        raise ValueError(msg)
    return shards[:num_shards]


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


def _aggregate_text_by_sample(table: pa.Table) -> dict[str, str]:
    rows = sorted(table.to_pylist(), key=lambda row: (str(row["sample_id"]), int(row["position"])))
    by_sample: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["modality"] != "text":
            continue
        by_sample[str(row["sample_id"])].append(str(row["text_content"] or ""))
    return {sample_id: "\n".join(parts) for sample_id, parts in by_sample.items()}


def _image_hashes_by_sample(table: pa.Table) -> dict[str, list[str]]:
    rows = sorted(table.to_pylist(), key=lambda row: (str(row["sample_id"]), int(row["position"])))
    by_sample: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["modality"] != "image":
            continue
        payload = row["binary_content"]
        if payload is None:
            continue
        by_sample[str(row["sample_id"])].append(hashlib.sha256(bytes(payload)).hexdigest())
    return dict(by_sample)


def _build_combined_tar(shards: list[Path], combined_tar_path: Path) -> dict[str, Any]:
    source_member_count = 0
    source_payload_bytes = 0
    source_hash = hashlib.sha256()
    combined_member_count = 0
    combined_payload_bytes = 0
    combined_hash = hashlib.sha256()

    with tarfile.open(combined_tar_path, "w") as output_tar:
        for shard in shards:
            prefix = shard.stem
            with tarfile.open(shard, "r:*") as input_tar:
                for member in input_tar:
                    if not member.isfile():
                        continue
                    payload_file = input_tar.extractfile(member)
                    payload = payload_file.read() if payload_file else b""

                    source_member_count += 1
                    source_payload_bytes += len(payload)
                    source_hash.update(member.name.encode("utf-8"))
                    source_hash.update(payload)

                    merged_name = f"{prefix}--{member.name}"
                    output_info = tarfile.TarInfo(name=merged_name)
                    output_info.size = len(payload)
                    output_tar.addfile(output_info, BytesIO(payload))

                    combined_member_count += 1
                    combined_payload_bytes += len(payload)
                    combined_hash.update(merged_name.encode("utf-8"))
                    combined_hash.update(payload)

    return {
        "combined_input_tar": str(combined_tar_path),
        "combined_input_size_bytes": combined_tar_path.stat().st_size,
        "source_member_count": source_member_count,
        "source_payload_bytes": source_payload_bytes,
        "source_hash": source_hash.hexdigest(),
        "combined_member_count": combined_member_count,
        "combined_payload_bytes": combined_payload_bytes,
        "combined_hash": combined_hash.hexdigest(),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = _discover_shards(input_dir, args.num_shards)
    combined_tar_path = output_dir / "combined_input.tar"

    logger.info("Combining {} shards into single input tar {}", len(shards), combined_tar_path)
    combine_metrics = _build_combined_tar(shards, combined_tar_path)

    reader = WebDatasetReaderStage(load_binary=True, sample_format="auto")
    reader_task = FileGroupTask(task_id="single-input", dataset_name="benchmark", data=[str(combined_tar_path)])
    start = time.perf_counter()
    batch = _as_single_batch(reader.process(reader_task))
    reader_elapsed = time.perf_counter() - start

    writer_metrics: dict[str, Any] = {}
    writer_outputs: dict[str, dict[str, str]] = {}
    for output_format, suffix in (("webdataset", "tar"), ("parquet", "parquet"), ("arrow", "arrow")):
        stage = MultimodalWriterStage(
            output_path=str(output_dir / output_format / f"out.{suffix}"),
            output_format=output_format,  # type: ignore[arg-type]
            image_payload_policy="preserve",
        )
        start = time.perf_counter()
        output_task = stage.process(batch)
        elapsed = time.perf_counter() - start
        data_path, metadata_path = output_task.data
        writer_metrics[f"{output_format}_write_total_s"] = elapsed
        writer_metrics[f"{output_format}_data_size_bytes"] = Path(data_path).stat().st_size
        writer_metrics[f"{output_format}_metadata_size_bytes"] = Path(metadata_path).stat().st_size
        writer_outputs[output_format] = {"data_path": data_path, "metadata_path": metadata_path}

    parquet_reader = ParquetMultimodalReaderStage()
    parquet_roundtrip = parquet_reader.process(
        (
            FileGroupTask(task_id="parquet-data", dataset_name="benchmark", data=[writer_outputs["parquet"]["data_path"]]),
            FileGroupTask(task_id="parquet-meta", dataset_name="benchmark", data=[writer_outputs["parquet"]["metadata_path"]]),
        )
    )
    parquet_rows_equal = parquet_roundtrip.data.to_pylist() == batch.data.to_pylist()
    left_meta = batch.metadata_index.to_pylist() if batch.metadata_index is not None else []
    right_meta = parquet_roundtrip.metadata_index.to_pylist() if parquet_roundtrip.metadata_index is not None else []
    parquet_metadata_equal = left_meta == right_meta

    webdataset_roundtrip = _as_single_batch(
        reader.process(
            FileGroupTask(task_id="webdataset-rt", dataset_name="benchmark", data=[writer_outputs["webdataset"]["data_path"]])
        )
    )
    webdataset_text_equal = _aggregate_text_by_sample(webdataset_roundtrip.data) == _aggregate_text_by_sample(batch.data)
    webdataset_image_equal = _image_hashes_by_sample(webdataset_roundtrip.data) == _image_hashes_by_sample(batch.data)

    with tarfile.open(writer_outputs["webdataset"]["data_path"], "r:*") as output_tar:
        output_members = [member.name for member in output_tar.getmembers() if member.isfile()]

    metrics: dict[str, Any] = {
        "is_success": True,
        "num_shards_combined": len(shards),
        "single_file_reader_total_s": reader_elapsed,
        "single_file_reader_rows": batch.data.num_rows,
        "single_file_reader_metadata_rows": 0 if batch.metadata_index is None else batch.metadata_index.num_rows,
        "single_file_reader_throughput_rows_per_s": batch.data.num_rows / reader_elapsed if reader_elapsed > 0 else 0.0,
        "parquet_exact_rows_equal": parquet_rows_equal,
        "parquet_exact_metadata_equal": parquet_metadata_equal,
        "webdataset_text_semantic_equal": webdataset_text_equal,
        "webdataset_image_semantic_equal": webdataset_image_equal,
        "webdataset_output_member_count": len(output_members),
        "webdataset_output_first_10_members": output_members[:10],
    }
    metrics.update(combine_metrics)
    metrics.update(writer_metrics)

    return {
        "params": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "num_shards": args.num_shards,
        },
        "metrics": metrics,
        "tasks": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multimodal IO using one combined input tar.")
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Directory where benchmark JSON outputs are written.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing source WebDataset *.tar shards.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where combined input and writer outputs are written.")
    parser.add_argument("--num-shards", type=int, default=14, help="Number of input shards to combine into one tar.")
    args = parser.parse_args()

    if args.num_shards <= 0:
        msg = f"--num-shards must be > 0, got {args.num_shards}"
        raise ValueError(msg)

    results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
