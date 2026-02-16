from __future__ import annotations

import json
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from aiohttp import ClientTimeout

from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReaderStage, RowSource
from nemo_curator.stages.multimodal.io.writers.multimodal import MultimodalWriterStage
from nemo_curator.tasks import MultimodalBatch, _EmptyTask
from nemo_curator.tasks.multimodal import METADATA_SCHEMA
from nemo_curator.utils.webdataset_utils import content_type_from_name


@dataclass
class URLTextParquetReaderStage(BaseMultimodalReaderStage, ABC):
    """Tutorial base class for URL+text parquet formats.

    This class centralizes boilerplate so a new reader implementer only needs to
    map source records into:
    - sample id
    - text list
    - image URL list
    - metadata payload
    """

    modalities_to_load: Literal["all", "image", "text"] = "all"
    columns: list[str] | None = None
    include_metadata_payload: bool = True
    max_records: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.modalities_to_load not in {"all", "image", "text"}:
            msg = f"Unsupported modalities_to_load='{self.modalities_to_load}'. Expected one of: all, image, text"
            raise ValueError(msg)

    def read_source_tables(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
        _ = metadata_path
        source_table = self._read_parquet_table(data_path, columns=self.columns)
        if self.max_records is not None:
            source_table = source_table.slice(0, self.max_records)
        source_shard = Path(data_path).name
        rows: list[dict[str, object]] = []
        metadata_rows: list[dict[str, object]] = []
        for row_idx, record in enumerate(source_table.to_pylist()):
            sample_id = self.sample_id(record, source_shard, row_idx)
            if self.include_metadata_payload:
                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_type": None,
                        "metadata_json": json.dumps(self.metadata_payload(record), ensure_ascii=True),
                    }
                )
            rows.extend(self._rows_for_record(sample_id=sample_id, source_shard=source_shard, record=record))
        metadata_table = pa.Table.from_pylist(metadata_rows, schema=METADATA_SCHEMA)
        return self._rows_to_table(rows), metadata_table

    def _rows_for_record(self, sample_id: str, source_shard: str, record: dict[str, object]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        texts = self.text_items(record)
        urls = self.image_urls(record)
        source = RowSource(source_shard=source_shard, content_path="", source_id=sample_id)
        position = 0
        for idx in range(max(len(texts), len(urls))):
            if self._loads_modality("text") and idx < len(texts):
                text_value = texts[idx]
                if isinstance(text_value, str) and text_value:
                    text_element_metadata = self.text_element_metadata(record, idx, text_value)
                    rows.append(
                        self._text_row(
                            sid=sample_id,
                            position=position,
                            source_shard=source_shard,
                            content_type="text/plain",
                            text_content=text_value,
                            element_metadata_json=self._json_or_none(text_element_metadata),
                        )
                    )
                    position += 1
            if self._loads_modality("image") and idx < len(urls):
                image_url = urls[idx]
                if isinstance(image_url, str) and image_url:
                    source.content_path = image_url
                    image_element_metadata = self.image_element_metadata(record, idx, image_url)
                    rows.append(
                        self._image_row(
                            sid=sample_id,
                            position=position,
                            source=source,
                            content_key=None,
                            binary_content=None,
                            content_type=content_type_from_name(image_url),
                            element_metadata_json=self._json_or_none(image_element_metadata),
                        )
                    )
                    position += 1
        return rows

    def _loads_modality(self, modality: str) -> bool:
        return self.modalities_to_load in {"all", modality}

    @abstractmethod
    def sample_id(self, record: dict[str, object], source_shard: str, row_idx: int) -> str:
        """Return deterministic sample id for one source record."""

    @abstractmethod
    def text_items(self, record: dict[str, object]) -> list[object]:
        """Return ordered text list for one source record."""

    @abstractmethod
    def image_urls(self, record: dict[str, object]) -> list[object]:
        """Return ordered image URL list for one source record."""

    @abstractmethod
    def metadata_payload(self, record: dict[str, object]) -> dict[str, object]:
        """Return sample-level metadata payload for metadata sidecar row."""

    def text_element_metadata(
        self,
        record: dict[str, object],
        text_idx: int,
        text_value: str,
    ) -> dict[str, object] | None:
        """Return per-text-row metadata payload, if available."""
        _ = record, text_idx, text_value
        return None

    def image_element_metadata(
        self,
        record: dict[str, object],
        image_idx: int,
        image_url: str,
    ) -> dict[str, object] | None:
        """Return per-image-row metadata payload, if available."""
        _ = record, image_idx, image_url
        return None


@dataclass
class OmniCorpusReaderStage(URLTextParquetReaderStage):
    """Tutorial OmniCorpus reader stage.

    Maps ``OpenGVLab/OmniCorpus-CC-210M`` parquet records:
    - ``general_metadata`` -> sample id / metadata payload
    - ``texts`` -> text rows
    - ``images`` -> URL-backed image rows
    """

    columns: list[str] | None = field(default_factory=lambda: ["general_metadata", "images", "texts", "metadata"])
    max_records: int | None = None
    name: str = "omnicorpus_tutorial_reader"

    def sample_id(self, record: dict[str, object], source_shard: str, row_idx: int) -> str:
        metadata = record.get("general_metadata")
        if isinstance(metadata, dict):
            sample_id = metadata.get("id")
            if isinstance(sample_id, str) and sample_id:
                return sample_id
        return f"{source_shard}:{row_idx}"

    def text_items(self, record: dict[str, object]) -> list[object]:
        texts = record.get("texts")
        return texts if isinstance(texts, list) else []

    def image_urls(self, record: dict[str, object]) -> list[object]:
        images = record.get("images")
        return images if isinstance(images, list) else []

    def metadata_payload(self, record: dict[str, object]) -> dict[str, object]:
        return {
            "general_metadata": record.get("general_metadata"),
        }

    def text_element_metadata(
        self,
        record: dict[str, object],
        text_idx: int,
        text_value: str,
    ) -> dict[str, object] | None:
        metadata = record.get("metadata")
        if not isinstance(metadata, list) or text_idx >= len(metadata):
            return None
        item = metadata[text_idx]
        if not isinstance(item, dict):
            return None
        out = dict(item)
        out["modality"] = "text"
        out["index"] = text_idx
        out["text"] = text_value
        return out

    def image_element_metadata(
        self,
        record: dict[str, object],
        image_idx: int,
        image_url: str,
    ) -> dict[str, object] | None:
        metadata = record.get("metadata")
        if not isinstance(metadata, list) or image_idx >= len(metadata):
            return None
        item = metadata[image_idx]
        if not isinstance(item, dict):
            return None
        out = dict(item)
        out["modality"] = "image"
        out["index"] = image_idx
        out["url"] = image_url
        return out


@dataclass
class OmniCorpusReader(CompositeStage[_EmptyTask, MultimodalBatch]):
    """Tutorial composite reader for OmniCorpus parquet shards."""

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".parquet"])
    limit: int | None = None
    modalities_to_load: Literal["all", "image", "text"] = "all"
    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    name: str = "omnicorpus_tutorial_reader"

    def __post_init__(self) -> None:
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
                limit=self.limit,
            ),
            OmniCorpusReaderStage(
                modalities_to_load=self.modalities_to_load,
                max_batch_bytes=self.max_batch_bytes,
                storage_options=self.storage_options,
            ),
        ]


def summarize_stage_perf(output_tasks: list[MultimodalBatch]) -> dict[str, dict[str, float]]:
    """Aggregate StagePerfStats across output tasks.

    Note:
        Stage timing field is ``process_time`` (seconds), not ``time_taken``.
    """
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"seconds": 0.0, "items": 0.0, "calls": 0.0})
    for task in output_tasks:
        for perf in task._stage_perf:
            stage_name = str(perf.stage_name)
            summary[stage_name]["seconds"] += float(perf.process_time)
            summary[stage_name]["items"] += float(perf.num_items_processed)
            summary[stage_name]["calls"] += 1.0
    return dict(summary)


def demo_omnicorpus_to_webdataset() -> None:
    """End-to-end tutorial pipeline: OmniCorpus parquet -> WebDataset.

    Runs over the full downloaded local shard (no sample slicing).
    """
    shard = "/raid/vjawa/tmp_omnicorpus_subset/data/CC-MAIN-2016-26/shard_0.parquet"
    row_count = pq.read_metadata(shard).num_rows
    print("input_rows_in_shard:", row_count)

    pipeline = Pipeline(
        name="omnicorpus_to_webdataset",
        description="Read OmniCorpus parquet rows and write WebDataset tar via multimodal pipeline",
    )
    pipeline.add_stage(
        OmniCorpusReader(
            file_paths=shard,
            modalities_to_load="all",
            max_batch_bytes=32 * 1024 * 1024,
        )
    )
    pipeline.add_stage(
        MultimodalWriterStage(
            output_path="/raid/vjawa/tmp_omnicorpus_subset/tutorial_output/omni_full.tar",
            output_format="webdataset",
            image_payload_policy="preserve",
            materialize_failure_policy="drop_image",
            materialize_max_retries=4,
            materialize_retry_backoff_sec=0.1,
            storage_options={"client_kwargs": {"timeout": ClientTimeout(total=2)}},
            mode="overwrite",
        )
    )

    results = pipeline.run(executor=RayDataExecutor())
    output_tasks = results or []
    print("pipeline_output_tasks:", len(output_tasks))
    if output_tasks:
        first = output_tasks[0]
        print("writer_outputs:", first.data)
        print("stage_perf_summary:", summarize_stage_perf(output_tasks))


if __name__ == "__main__":
    demo_omnicorpus_to_webdataset()
