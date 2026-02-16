from __future__ import annotations

from pathlib import Path

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal import WebDatasetReader, WebDatasetReaderStage


def test_webdataset_reader_decomposes_like_text_readers() -> None:
    stage = WebDatasetReader(file_paths="data/shards", files_per_partition=4, sample_format="simple")
    decomposed = stage.decompose()
    assert len(decomposed) == 2
    assert isinstance(decomposed[0], FilePartitioningStage)
    assert isinstance(decomposed[1], WebDatasetReaderStage)
    assert decomposed[0].files_per_partition == 4
    assert decomposed[1].sample_format == "simple"


def test_webdataset_reader_default_file_extensions_include_tar_formats() -> None:
    stage = WebDatasetReader(file_paths=[str(Path("a.tar")), str(Path("b.tgz"))])
    assert stage.file_extensions == [".tar", ".tar.gz", ".tgz", ".tar.zst"]


def test_webdataset_reader_description_contains_core_hints() -> None:
    stage = WebDatasetReader(
        file_paths="s3://bucket/train",
        blocksize="256MB",
        sample_format="interleaved",
        modalities_to_load="text",
        limit=2,
    )
    desc = stage.get_description()
    assert "s3://bucket/train" in desc
    assert "target blocksize 256MB" in desc
    assert "sample_format=interleaved" in desc
    assert "modalities_to_load=text" in desc
    assert "limited to 2 partitions" in desc
