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

"""Extract main content from raw Common Crawl HTML with MinerU-HTML.

The pipeline is CPU-only: labelling is submitted to an OpenAI-compatible vLLM
server that you start separately (see README.md) and point ``--server-url`` at.

The input is a Parquet dataset with a ``content`` column holding raw HTML
(``bytes`` or ``str``) and a ``url`` column. The output is the same rows plus a
``text`` column with the extracted main content as Markdown.

    python run_pipeline.py \
        --input /home/vjawa/bench-data/cc_main_2025_26_html_100k \
        --output ./mineru-out \
        --server-url http://127.0.0.1:8000 \
        --limit 2000
"""

import argparse
import time

import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.html_extraction import MinerUHtmlExtractor
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch


class HeadStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Truncate every batch to at most ``n`` rows, for quick benchmark runs."""

    def __init__(self, n: int):
        self.n = n
        self.name = "head"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=batch.to_pandas().head(self.n),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Parquet file or directory of raw HTML")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--html-field", default="content")
    ap.add_argument("--url-field", default="url")
    ap.add_argument("--limit", type=int, default=None, help="Keep at most this many documents per reader partition")

    ap.add_argument("--blocksize", default="256MB", help="Reader partition size")
    ap.add_argument("--files-per-partition", type=int, default=None)

    ap.add_argument(
        "--server-url",
        required=True,
        help="Root of the OpenAI-compatible vLLM endpoint, e.g. http://127.0.0.1:8000. "
        "Start it yourself; this script does not manage it (see README.md)",
    )
    ap.add_argument("--served-model-name", default="mineru", help="--served-model-name given to that server")
    ap.add_argument(
        "--server-concurrency",
        type=int,
        default=64,
        help="In-flight requests per inference worker; queue depth = this x --inference-workers",
    )

    ap.add_argument(
        "--model",
        default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact",
        help="Tokenizer only; the server holds the weights",
    )
    ap.add_argument("--max-model-len", type=int, default=32768, help="Must match the server's --max-model-len")
    ap.add_argument(
        "--structured-outputs",
        choices=["none", "per_request"],
        default="per_request",
        help="Grammar strategy for the compact answer format",
    )
    ap.add_argument("--fallback", default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    ap.add_argument("--output-format", default="mm_md", choices=["mm_md", "md", "json", "txt", "none"])
    ap.add_argument("--simplify-workers", type=int, default=None)
    ap.add_argument("--inference-workers", type=int, default=None)
    ap.add_argument("--extract-workers", type=int, default=None)
    ap.add_argument("--no-pretokenize", action="store_true")
    ap.add_argument(
        "--chat-template-mode",
        choices=["single", "upstream_double"],
        default="single",
        help="upstream_double reproduces the reference implementation's doubled chat template",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    reader = ParquetReader(
        file_paths=args.input,
        blocksize=args.blocksize if args.files_per_partition is None else None,
        files_per_partition=args.files_per_partition,
        fields=[args.html_field, args.url_field] if args.url_field else [args.html_field],
        # ParquetReader defaults to dtype_backend="pyarrow". Raw HTML columns are
        # large: 25k Common Crawl pages is >2 GB of bytes in one partition, and
        # pickling an Arrow-backed `binary` column that big overflows its 32-bit
        # offsets ("offset overflow while concatenating arrays") when the batch
        # is shipped to the next stage. object dtype has no such limit.
        read_kwargs={"dtype_backend": "numpy_nullable"},
    )

    extractor = MinerUHtmlExtractor(
        base_url=args.server_url,
        served_model_name=args.served_model_name,
        server_concurrency=args.server_concurrency,
        html_field=args.html_field,
        url_field=args.url_field,
        model_identifier=args.model,
        max_model_len=args.max_model_len,
        structured_outputs=args.structured_outputs,
        output_format=args.output_format,
        fallback=args.fallback,
        simplify_workers=args.simplify_workers,
        inference_workers=args.inference_workers,
        extract_workers=args.extract_workers,
        pretokenize=not args.no_pretokenize,
        chat_template_mode=args.chat_template_mode,
    )

    pipeline = Pipeline(name="mineru_html_extraction", description="MinerU-HTML main content extraction")
    pipeline.add_stage(reader)
    if args.limit:
        pipeline.add_stage(HeadStage(args.limit))
    pipeline.add_stage(extractor)
    pipeline.add_stage(ParquetWriter(path=args.output, mode="overwrite" if args.overwrite else "ignore"))

    logger.info(pipeline.describe())

    t0 = time.perf_counter()
    results = pipeline.run()
    elapsed = time.perf_counter() - t0

    # The writer emits FileGroupTasks, so num_items counts files, not rows.
    written = [path for task in (results or []) for path in task.data]
    n_docs = sum(pq.ParquetFile(path).metadata.num_rows for path in written)
    logger.info(
        f"Wrote {n_docs} documents across {len(written)} files in {elapsed:.1f}s "
        f"({n_docs / elapsed:.1f} docs/s end to end)"
    )


if __name__ == "__main__":
    main()
