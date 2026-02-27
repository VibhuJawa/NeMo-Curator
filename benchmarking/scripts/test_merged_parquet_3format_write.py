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

"""Test script: read merged parquet, filter to fully-matched samples, write 3 formats.

Reads from the merged parquet domain-bucket data, keeps only samples where
every image row has match_status == "matched", then writes the first N samples
as materialized Parquet, WebDataset, and Lance.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.multimodal.io.writers.lance import MultimodalLanceWriterStage
from nemo_curator.stages.multimodal.io.writers.tabular import MultimodalParquetWriterStage
from nemo_curator.stages.multimodal.io.writers.webdataset import MultimodalWebdatasetWriterStage
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

STORAGE_OPTIONS = {
    "key": "team-iva-cc-image-text-release",
    "secret": "e36b8dbaa53497da987baa9129acb3be",
    "client_kwargs": {
        "endpoint_url": "https://pdx.s8k.io",
        "region_name": "us-east-1",
    },
}


def find_parquet_files(bucket_dir: str) -> list[str]:
    return sorted(
        os.path.join(bucket_dir, f)
        for f in os.listdir(bucket_dir)
        if f.endswith(".parquet")
    )


def load_and_filter(parquet_path: str, num_samples: int) -> MultiBatchTask:
    """Load merged parquet, keep only fully-matched samples, return first N."""
    logger.info("Reading parquet: {}", parquet_path)
    t0 = time.perf_counter()
    table = pq.read_table(parquet_path)
    logger.info("Read {} rows in {:.1f}s", table.num_rows, time.perf_counter() - t0)

    df = table.to_pandas()

    image_df = df[df["modality"] == "image"]
    unmatched_sids = set(
        image_df.loc[image_df["match_status"] != "matched", "sample_id"].unique()
    )
    all_sids = set(df["sample_id"].unique())
    fully_matched_sids = all_sids - unmatched_sids
    # Exclude sample_ids that have zero image rows (metadata/text-only docs)
    sids_with_images = set(image_df["sample_id"].unique())
    fully_matched_sids = fully_matched_sids & sids_with_images

    logger.info(
        "Samples: {} total, {} with images, {} fully matched",
        len(all_sids), len(sids_with_images), len(fully_matched_sids),
    )

    selected_sids = sorted(fully_matched_sids)[:num_samples]
    logger.info("Selected first {} sample_ids", len(selected_sids))

    filtered = df[df["sample_id"].isin(selected_sids)].reset_index(drop=True)
    logger.info("Filtered to {} rows across {} samples", len(filtered), len(selected_sids))

    for col in MULTIMODAL_SCHEMA.names:
        if col not in filtered.columns:
            filtered[col] = None

    logger.info("Output columns: {} ({} total)", list(filtered.columns), len(filtered.columns))
    for sid in selected_sids:
        sample = filtered[filtered["sample_id"] == sid]
        imgs = sample[sample["modality"] == "image"]
        logger.info(
            "  sample {} -- {} rows ({} text, {} image, {} metadata)",
            sid[:80], len(sample),
            (sample["modality"] == "text").sum(),
            len(imgs),
            (sample["modality"] == "metadata").sum(),
        )

    return MultiBatchTask(
        task_id="test_3format",
        dataset_name="merged_parquet_bucket0",
        data=filtered,
        _metadata={"source_files": [parquet_path]},
    )


def write_parquet(task: MultiBatchTask, output_dir: str) -> str:
    path = os.path.join(output_dir, "parquet")
    os.makedirs(path, exist_ok=True)
    writer = MultimodalParquetWriterStage(
        path=path,
        materialize_on_write=True,
        write_kwargs={"storage_options": STORAGE_OPTIONS},
        on_materialize_error="warn",
        mode="overwrite",
    )
    logger.info("Writing Parquet to {}", path)
    t0 = time.perf_counter()
    result = writer.process(task)
    logger.info("Parquet write done in {:.1f}s -> {}", time.perf_counter() - t0, result.data)
    return result.data[0]


def write_webdataset(task: MultiBatchTask, output_dir: str) -> str:
    path = os.path.join(output_dir, "webdataset")
    os.makedirs(path, exist_ok=True)
    writer = MultimodalWebdatasetWriterStage(
        path=path,
        materialize_on_write=True,
        write_kwargs={"storage_options": STORAGE_OPTIONS},
        on_materialize_error="warn",
        mode="overwrite",
    )
    logger.info("Writing WebDataset to {}", path)
    t0 = time.perf_counter()
    result = writer.process(task)
    logger.info("WebDataset write done in {:.1f}s -> {}", time.perf_counter() - t0, result.data)
    return result.data[0]


def write_lance(task: MultiBatchTask, output_dir: str) -> str:
    path = os.path.join(output_dir, "lance")
    os.makedirs(path, exist_ok=True)
    writer = MultimodalLanceWriterStage(
        path=path,
        materialize_on_write=True,
        write_kwargs={"storage_options": STORAGE_OPTIONS},
        on_materialize_error="warn",
        mode="overwrite",
    )
    logger.info("Writing Lance to {}", path)
    t0 = time.perf_counter()
    result = writer.process(task)
    logger.info("Lance write done in {:.1f}s -> {}", time.perf_counter() - t0, result.data)
    return result.data[0]


def verify_outputs(output_dir: str) -> None:
    """Quick sanity check on the written outputs."""
    import lance
    import pandas as pd

    parquet_dir = os.path.join(output_dir, "parquet")
    pq_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if pq_files:
        df_pq = pd.read_parquet(pq_files[0])
        logger.info("[Verify] Parquet: {} rows, {} samples, columns={}", len(df_pq), df_pq["sample_id"].nunique(), list(df_pq.columns))
        img_rows = df_pq[df_pq["modality"] == "image"]
        has_binary = img_rows["binary_content"].notna().sum()
        logger.info("[Verify] Parquet image rows: {}, with binary: {}", len(img_rows), has_binary)

    wds_dir = os.path.join(output_dir, "webdataset")
    tar_files = [f for f in os.listdir(wds_dir) if f.endswith(".tar")]
    if tar_files:
        import tarfile
        tar_path = os.path.join(wds_dir, tar_files[0])
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getnames()
        logger.info("[Verify] WebDataset tar: {} members, first 10: {}", len(members), members[:10])

    lance_dir = os.path.join(output_dir, "lance")
    lance_files = [f for f in os.listdir(lance_dir) if f.endswith(".lance")]
    if lance_files:
        ds = lance.dataset(os.path.join(lance_dir, lance_files[0]))
        df_lance = ds.to_table().to_pandas()
        logger.info("[Verify] Lance: {} rows, {} samples, columns={}", len(df_lance), df_lance["sample_id"].nunique(), list(df_lance.columns))
        img_rows = df_lance[df_lance["modality"] == "image"]
        has_binary = img_rows["binary_content"].notna().sum()
        logger.info("[Verify] Lance image rows: {}, with binary: {}", len(img_rows), has_binary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test merged parquet -> 3-format materialized write")
    parser.add_argument(
        "--input-path", type=str,
        default="/datasets/vjawa/mint1t_normalized/merged_parquet_domain_buckets_0_to_63/domain_bucket=0/",
    )
    parser.add_argument("--output-path", type=str, default=".tmp_multimodal_runs/3format_test")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--skip-verify", action="store_true", default=False)
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.input_path)
    if not parquet_files:
        logger.error("No parquet files found in {}", args.input_path)
        return 1
    logger.info("Found {} parquet file(s) in {}", len(parquet_files), args.input_path)

    task = load_and_filter(parquet_files[0], num_samples=args.num_samples)

    output_dir = str(Path(args.output_path).absolute())
    os.makedirs(output_dir, exist_ok=True)

    write_parquet(task, output_dir)
    write_webdataset(task, output_dir)
    write_lance(task, output_dir)

    if not args.skip_verify:
        verify_outputs(output_dir)

    logger.info("All 3 formats written to {}", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
