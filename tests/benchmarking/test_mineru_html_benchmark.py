# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"))

from mineru_html_benchmark import build_parquet_pipeline, build_snapshot_pipeline, vllm_performance_metrics

from nemo_curator.stages.text.download.common_crawl.stage import (
    CommonCrawlWARCDownloadAndReadStage,
    CommonCrawlWARCManifestSourceStage,
)
from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats


def test_build_pipeline_can_preserve_every_input_field(tmp_path: Path) -> None:
    args = Namespace(
        preserve_input_fields=True,
        input_path=str(tmp_path / "input.parquet"),
        files_per_partition=1,
        html_field="html",
        url_field="url",
        server_concurrency=1,
        html_compression="none",
        text_field="text",
        boilerplate_text_field="boilerplate_text",
        llm_output_field="llm_output_labels",
        model="model",
        cutoff_length=250,
        max_model_len=1024,
        structured_outputs="none",
        output_format="mm_md",
        fallback="empty",
        simplify_workers=1,
        inference_workers=1,
        extract_workers=1,
        chat_template_mode="single",
        cache_dir=None,
        drop_html_field=False,
        server_mode="external",
        served_model_name="mineru",
    )

    pipeline = build_parquet_pipeline(args, tmp_path / "output", "http://server")
    reader = pipeline.stages[0].decompose()[1]

    assert isinstance(reader, ParquetReaderStage)
    assert reader.fields is None
    assert pipeline.stages[1].decompose()[0].cutoff_length == 250


def test_snapshot_pipeline_uses_native_source_and_fused_local_download(tmp_path: Path) -> None:
    args = Namespace(
        snapshot="2025-26",
        warc_manifest=str(tmp_path / "warc.paths"),
        download_dir=str(tmp_path / "raid"),
        cc_transport="s3",
        cc_s3_bucket="crawl-data",
        cc_s3_key_prefix="crawl-data/",
        cc_s3_endpoint_url="https://pdx.s8k.io",
        cc_s5cmd_concurrency=16,
        cc_s5cmd_part_size_mb=128,
        download_workers=4,
        warc_records_per_batch=2048,
        html_field="content",
        html_compression="zstd",
        url_field="url",
        text_field="text",
        boilerplate_text_field=None,
        llm_output_field=None,
        model="model",
        cutoff_length=500,
        max_model_len=32768,
        structured_outputs="per_request",
        output_format="mm_md",
        fallback="trafilatura",
        simplify_workers=1,
        inference_workers=1,
        extract_workers=1,
        chat_template_mode="single",
        cache_dir=None,
        drop_html_field=True,
        server_mode="external",
        served_model_name="mineru",
        server_concurrency=1,
    )

    pipeline = build_snapshot_pipeline(args, tmp_path / "output", "http://server")

    assert isinstance(pipeline.stages[0], CommonCrawlWARCManifestSourceStage)
    assert isinstance(pipeline.stages[1], CommonCrawlWARCDownloadAndReadStage)
    assert pipeline.stages[1].compression == "zstd"
    assert pipeline.stages[1].records_per_batch == 2048
    assert pipeline.stages[1].downloader.s3_bucket == "crawl-data"
    assert pipeline.stages[1].downloader.s3_endpoint_url == "https://pdx.s8k.io"
    assert pipeline.stages[2].decompose()[-1].drop_html_field is True


def test_vllm_performance_metrics_use_global_request_window() -> None:
    tasks = []
    for start, end, requests in ((100.0, 103.0, 300.0), (101.0, 104.0, 300.0)):
        task = FileGroupTask(dataset_name="test", data=[])
        task._stage_perf = [
            StagePerfStats(
                stage_name="mineru_html_server_inference",
                custom_metrics={
                    "requests": requests,
                    "request_window_start_s": start,
                    "request_window_end_s": end,
                },
            )
        ]
        tasks.append(task)

    metrics = vllm_performance_metrics(tasks)

    assert metrics == {
        "vllm_requests": 600,
        "vllm_inference_time_s": 4.0,
        "vllm_docs_per_sec": 150.0,
    }
