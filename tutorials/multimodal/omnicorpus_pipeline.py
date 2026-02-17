from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from aiohttp import ClientTimeout

from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.multimodal.io.writers.multimodal import MultimodalWriterStage
from tutorials.multimodal.omnicorpus_custom_reader import OmniCorpusReader

if TYPE_CHECKING:
    from nemo_curator.tasks import MultimodalBatch


def summarize_stage_perf(output_tasks: list[MultimodalBatch]) -> dict[str, dict[str, float]]:
    """Aggregate StagePerfStats across output tasks."""
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"seconds": 0.0, "items": 0.0, "calls": 0.0})
    for task in output_tasks:
        for perf in task._stage_perf:
            stage_name = str(perf.stage_name)
            summary[stage_name]["seconds"] += float(perf.process_time)
            summary[stage_name]["items"] += float(perf.num_items_processed)
            summary[stage_name]["calls"] += 1.0
    return dict(summary)


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


def demo_omnicorpus_to_webdataset() -> None:
    """End-to-end tutorial pipeline: OmniCorpus parquet -> WebDataset."""
    shard = "/raid/vjawa/tmp_omnicorpus_subset/data/CC-MAIN-2016-26/shard_0.parquet"
    output_path = "/raid/vjawa/tmp_omnicorpus_subset/tutorial_output/omni_full.tar"
    print("input_rows_in_shard:", pq.read_metadata(shard).num_rows)

    results = build_omnicorpus_pipeline(shard=shard, output_path=output_path).run(executor=RayDataExecutor())
    output_tasks = results or []
    print("pipeline_output_tasks:", len(output_tasks))
    if output_tasks:
        print("writer_outputs:", output_tasks[0].data)
        print("stage_perf_summary:", summarize_stage_perf(output_tasks))


if __name__ == "__main__":
    demo_omnicorpus_to_webdataset()
