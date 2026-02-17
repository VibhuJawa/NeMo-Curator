from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import ray
from aiohttp import ClientTimeout

from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.multimodal import MultimodalWriterStage

try:
    from tutorials.multimodal.omnicorpus_custom_reader import OmniCorpusReader
except ModuleNotFoundError:
    # Allows direct script execution: `python tutorials/multimodal/omnicorpus_pipeline.py`
    from omnicorpus_custom_reader import OmniCorpusReader

if TYPE_CHECKING:
    from nemo_curator.tasks import FileGroupTask

DEFAULT_SHARD_PATH = os.getenv(
    "OMNICORPUS_SHARD_PATH",
    "/raid/vjawa/tmp_omnicorpus_subset/data/CC-MAIN-2016-26/shard_0.parquet",
)
DEFAULT_OUTPUT_PATH = os.getenv(
    "OMNICORPUS_OUTPUT_PATH",
    "/raid/vjawa/tmp_omnicorpus_subset/tutorial_output/omni_full.tar",
)


def summarize_stage_perf(output_tasks: list[FileGroupTask]) -> dict[str, dict[str, float]]:
    """Aggregate StagePerfStats across output tasks."""
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"seconds": 0.0, "items": 0.0, "calls": 0.0})
    for task in output_tasks:
        for perf in task._stage_perf:
            stage_name = str(perf.stage_name)
            summary[stage_name]["seconds"] += float(perf.process_time)
            summary[stage_name]["items"] += float(perf.num_items_processed)
            summary[stage_name]["calls"] += 1.0
    for stage_stats in summary.values():
        seconds = stage_stats["seconds"]
        stage_stats["items_per_second"] = stage_stats["items"] / seconds if seconds > 0 else 0.0
    return dict(summary)


def warn_on_suspicious_stage_perf(stage_perf_summary: dict[str, dict[str, float]]) -> None:
    """Warn when StagePerf looks implausible (for example zero seconds with processed rows)."""
    for stage_name, stats in stage_perf_summary.items():
        if stats["seconds"] == 0.0 and stats["items"] > 0.0:
            print(
                "warning: stage has zero seconds but processed rows, timing may be under-reported:",
                stage_name,
                stats,
            )


def build_omnicorpus_pipeline(shard: str, output_path: str, max_records: int | None = None) -> Pipeline:
    pipeline = Pipeline(
        name="omnicorpus_to_webdataset",
        description="Read OmniCorpus parquet rows and write WebDataset tar via multimodal pipeline",
    )
    pipeline.add_stage(
        OmniCorpusReader(
            file_paths=shard,
            modalities_to_load="all",
            max_records=max_records,
            max_batch_bytes=32 * 1024 * 1024,
        )
    )
    pipeline.add_stage(
        # Custom writer tutorial note:
        # To implement your own writer, subclass BaseMultimodalWriterStage and
        # implement `write_data(self, task, output_path)`.
        # Optional hooks:
        # - `configure(...)` for validation/derived config
        # - `prepare_task(...)` for materialize/filter policies
        MultimodalWriterStage(
            output_path=output_path,
            output_format="webdataset",
            image_payload_policy="preserve",
            materialize_failure_policy="drop_image",
            materialize_max_retries=4,
            materialize_retry_backoff_sec=0.1,
            storage_options={"client_kwargs": {"timeout": ClientTimeout(total=2)}},
            mode="overwrite",
        )
    )
    return pipeline


def demo_omnicorpus_to_webdataset(
    *,
    shard_path: str = DEFAULT_SHARD_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    max_records: int | None = None,
    ray_address: str | None = None,
) -> None:
    """End-to-end tutorial pipeline: OmniCorpus parquet -> WebDataset."""
    print("input_rows_in_shard:", pq.read_metadata(shard_path).num_rows)
    if ray_address:
        print("ray_address:", ray_address)
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

    results = build_omnicorpus_pipeline(
        shard=shard_path,
        output_path=output_path,
        max_records=max_records,
    ).run(executor=RayDataExecutor())
    output_tasks = results or []
    print("pipeline_output_tasks:", len(output_tasks))
    if output_tasks:
        print("writer_outputs:", output_tasks[0].data)
        stage_perf_summary = summarize_stage_perf(output_tasks)
        print("stage_perf_summary:", stage_perf_summary)
        warn_on_suspicious_stage_perf(stage_perf_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniCorpus tutorial pipeline.")
    parser.add_argument("--shard-path", default=DEFAULT_SHARD_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--ray-address", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    demo_omnicorpus_to_webdataset(
        shard_path=args.shard_path,
        output_path=args.output_path,
        max_records=args.max_records,
        ray_address=args.ray_address,
    )
