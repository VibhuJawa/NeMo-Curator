from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReaderStage, RowSource
from nemo_curator.tasks import MultimodalBatch, _EmptyTask
from nemo_curator.tasks.multimodal import METADATA_SCHEMA
from nemo_curator.utils.webdataset_utils import content_type_from_name


@dataclass
class OmniCorpusReaderStage(BaseMultimodalReaderStage):
    """Tutorial reader stage for OpenGVLab/OmniCorpus-CC-210M parquet shards.

    Custom reader note:
    - To implement your own reader stage, subclass ``BaseMultimodalReaderStage``
      and implement ``read_data(data_path, metadata_path)``.
    - In that function, return normalized data rows and metadata rows
      (using helpers like ``_text_row``, ``_image_row``, and ``_rows_to_table``).
    """

    modalities_to_load: Literal["all", "image", "text"] = "all"
    columns: list[str] | None = field(default_factory=lambda: ["general_metadata", "images", "texts", "metadata"])
    include_metadata_payload: bool = True
    max_records: int | None = None
    name: str = "omnicorpus_tutorial_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.modalities_to_load not in {"all", "image", "text"}:
            msg = f"Unsupported modalities_to_load='{self.modalities_to_load}'. Expected one of: all, image, text"
            raise ValueError(msg)

    def read_data(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
        _ = metadata_path
        source_table = self._read_parquet_table(data_path, columns=self.columns)
        if self.max_records is not None:
            source_table = source_table.slice(0, self.max_records)

        source_shard = Path(data_path).name
        rows: list[dict[str, object]] = []
        metadata_rows: list[dict[str, object]] = []
        load_text = self.modalities_to_load in {"all", "text"}
        load_image = self.modalities_to_load in {"all", "image"}

        for row_idx, record in enumerate(source_table.to_pylist()):
            sample_id = self._sample_id(record.get("general_metadata"), source_shard, row_idx)
            if self.include_metadata_payload:
                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_type": None,
                        "metadata_json": json.dumps({"general_metadata": record.get("general_metadata")}, ensure_ascii=True),
                    }
                )

            record_metadata = record.get("metadata")
            for position, (modality, idx, value) in enumerate(self._iter_entries(record, load_text, load_image)):
                if modality == "text":
                    rows.append(
                        self._text_row(
                            sid=sample_id,
                            position=position,
                            source_shard=source_shard,
                            content_type="text/plain",
                            text_content=value,
                            element_metadata_json=self._json_or_none(
                                self._element_metadata(record_metadata, idx, "text", "text", value)
                            ),
                        )
                    )
                else:
                    rows.append(
                        self._image_row(
                            sid=sample_id,
                            position=position,
                            source=RowSource(
                                source_shard=source_shard,
                                content_path=value,
                                source_id=sample_id,
                            ),
                            content_key=None,
                            binary_content=None,
                            content_type=content_type_from_name(value),
                            element_metadata_json=self._json_or_none(
                                self._element_metadata(record_metadata, idx, "image", "url", value)
                            ),
                        )
                    )

        return self._rows_to_table(rows), pa.Table.from_pylist(metadata_rows, schema=METADATA_SCHEMA)

    @staticmethod
    def _sample_id(general_metadata: object, source_shard: str, row_idx: int) -> str:
        if isinstance(general_metadata, dict):
            metadata = general_metadata
            sample_id = metadata.get("id")
            if isinstance(sample_id, str) and sample_id:
                return sample_id
        return f"{source_shard}:{row_idx}"

    @classmethod
    def _iter_entries(
        cls,
        record: dict[str, object],
        load_text: bool,
        load_image: bool,
    ) -> list[tuple[Literal["text", "image"], int, str]]:
        """Return ordered, non-empty text/image entries for one source record."""
        texts = record.get("texts")
        images = record.get("images")
        text_list = texts if isinstance(texts, list) else []
        image_list = images if isinstance(images, list) else []
        out: list[tuple[Literal["text", "image"], int, str]] = []
        for idx in range(max(len(text_list), len(image_list))):
            if load_text and idx < len(text_list):
                text_value = text_list[idx]
                if isinstance(text_value, str) and text_value:
                    out.append(("text", idx, text_value))
            if load_image and idx < len(image_list):
                image_url = image_list[idx]
                if isinstance(image_url, str) and image_url:
                    out.append(("image", idx, image_url))
        return out

    @staticmethod
    def _element_metadata(
        record_metadata: object,
        idx: int,
        modality: Literal["text", "image"],
        value_key: Literal["text", "url"],
        value: str,
    ) -> dict[str, object] | None:
        if not isinstance(record_metadata, list) or idx >= len(record_metadata):
            return None
        item = record_metadata[idx]
        if not isinstance(item, dict):
            return None
        out = dict(item)
        out["modality"] = modality
        out["index"] = idx
        out[value_key] = value
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
    max_records: int | None = None
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
                max_records=self.max_records,
                max_batch_bytes=self.max_batch_bytes,
                storage_options=self.storage_options,
            ),
        ]
