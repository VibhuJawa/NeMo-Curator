# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch


def _task() -> DocumentBatch:
    task = DocumentBatch(
        dataset_name="test",
        data=pd.DataFrame({"document_id": ["a"]}),
        _metadata={"source_files": ["source.parquet"]},
    )
    task._set_task_id("", "curator-owned-id")
    return task


def test_atomic_parquet_writer_publishes_complete_file(tmp_path: Path) -> None:
    writer = ParquetWriter(path=str(tmp_path), atomic_local=True)

    result = writer.process(_task())

    output = Path(result.data[0])
    assert pd.read_parquet(output).to_dict(orient="list") == {"document_id": ["a"]}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_parquet_writer_removes_temporary_file_after_failure(tmp_path: Path) -> None:
    writer = ParquetWriter(path=str(tmp_path), atomic_local=True)

    with patch.object(pd.DataFrame, "to_parquet", side_effect=RuntimeError("injected write failure")):
        with pytest.raises(RuntimeError, match="injected write failure"):
            writer.process(_task())

    assert list(tmp_path.iterdir()) == []
