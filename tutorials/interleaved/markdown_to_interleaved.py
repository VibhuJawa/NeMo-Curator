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

"""Convert 10,000 PIN-14M markdown documents to interleaved Parquet rows."""

import pandas as pd
from datasets import load_dataset

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved import MarkdownToInterleavedStage
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.tasks import DocumentBatch


def main(output_path: str = "pin14m_interleaved") -> None:
    dataset = load_dataset("m-a-p/PIN-14M", "pin", split="train", streaming=True)
    samples = list(dataset.take(10_000))
    documents = [
        DocumentBatch(
            dataset_name="PIN-14M",
            data=pd.DataFrame(samples[start : start + 250]),
            _metadata={"source_files": [f"hf://datasets/m-a-p/PIN-14M#rows={start}:{start + 250}"]},
        )
        for start in range(0, len(samples), 250)
    ]
    pipeline = Pipeline(
        name="markdown_to_interleaved",
        stages=[
            MarkdownToInterleavedStage(
                image_source_uri=(
                    "https://huggingface.co/datasets/m-a-p/PIN-14M/resolve/main/data/DocLayNet/content_image.tar.gz"
                )
            ),
            InterleavedParquetWriterStage(
                path=output_path,
                materialize_on_write=False,
                mode="overwrite",
            ),
        ],
    )

    with RayClient():
        outputs = pipeline.run(executor=RayDataExecutor(), initial_tasks=documents)
    print(f"Wrote {len(outputs)} Parquet files to {output_path}")


if __name__ == "__main__":
    main()
