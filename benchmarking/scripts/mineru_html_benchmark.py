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

"""MinerU-HTML main-content extraction benchmark for nightly benchmarking.

Runs the three-stage MinerU-HTML pipeline (simplify on CPU -> label on GPU ->
render on CPU) over a parquet dataset of raw HTML and writes params/metrics/tasks
to the benchmark results directory, compatible with the nightly driver.

Requires ``nemo_curator[vllm]`` and the ``mineru_html`` package.
"""

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.html_extraction import DEFAULT_MODEL, MinerUHtmlExtractor
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.tasks import DocumentBatch


class HeadStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Truncate every batch to at most ``n`` rows, for short benchmark runs."""

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


def create_mineru_html_pipeline(args: argparse.Namespace, output_dir: Path) -> Pipeline:
    pipeline = Pipeline(
        name="mineru_html_extraction",
        description="Extract main content from raw HTML with MinerU-HTML",
    )

    read_kwargs = {
        # ParquetReader defaults to the "pyarrow" dtype backend. An Arrow-backed
        # 32-bit-offset binary column holding more than 2 GB of HTML -- roughly one
        # 25k-row Common Crawl partition -- overflows when the batch is pickled for
        # the next stage. object dtype has no such limit. Harmless when the source
        # was written as large_binary; keep it so the benchmark does not depend on
        # how the input table happened to be written.
        "dtype_backend": "numpy_nullable",
    }
    reader_kwargs = {}
    if args.files_per_partition is not None:
        reader_kwargs["files_per_partition"] = args.files_per_partition
    else:
        reader_kwargs["blocksize"] = args.blocksize

    fields = [args.html_field]
    if args.url_field:
        fields.append(args.url_field)

    pipeline.add_stage(
        ParquetReader(
            file_paths=args.input_path,
            fields=fields,
            read_kwargs=read_kwargs,
            **reader_kwargs,
        )
    )

    if args.limit is not None:
        pipeline.add_stage(HeadStage(args.limit))

    pipeline.add_stage(
        MinerUHtmlExtractor(
            html_field=args.html_field,
            url_field=args.url_field or None,
            text_field=args.text_field,
            model_identifier=args.model,
            max_model_len=args.max_model_len,
            structured_outputs=args.structured_outputs,
            kv_cache_dtype=args.kv_cache_dtype,
            quantization=args.quantization,
            output_format=args.output_format,
            fallback=args.fallback,
            simplify_workers=args.simplify_workers,
            inference_workers=args.inference_workers,
            extract_workers=args.extract_workers,
            chat_template_mode=args.chat_template_mode,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        )
    )

    pipeline.add_stage(ParquetWriter(path=str(output_dir)))
    return pipeline


# A non-empty result is not the same as a successful extraction: Common Crawl
# carries sub-200-byte pages that yield a character or two of Markdown and would
# otherwise count as wins. Rates are reported against this floor as well as raw.
MIN_SUBSTANTIVE_CHARS = 200

STATUS_FIELD = "_mineru_status"


def summarize_output(output_dir: Path, text_field: str) -> dict:
    """Count how many rows actually came out with extracted content.

    Throughput alone cannot distinguish a fast run from a run that fell back on
    every document, so extraction quality is recorded next to it. Three signals,
    because they disagree in informative ways:
      - non-empty text            (loosest; inflated by tiny source pages)
      - text >= MIN_SUBSTANTIVE_CHARS
      - the pipeline's own _mineru_status == "ok"
    """
    empty = {
        "num_documents_written": 0,
        "num_documents_with_text": 0,
        "num_documents_substantive": 0,
        "num_status_ok": 0,
        "total_text_chars": 0,
        "status_counts": {},
    }
    shards = sorted(output_dir.rglob("*.parquet"))
    if not shards:
        return empty

    written = with_text = substantive = status_ok = total_chars = 0
    status_counts: dict[str, int] = {}
    for shard in shards:
        pf = pq.ParquetFile(shard)
        names = pf.schema_arrow.names
        if text_field not in names:
            written += pf.metadata.num_rows
            continue
        cols = [text_field] + ([STATUS_FIELD] if STATUS_FIELD in names else [])
        for batch in pf.iter_batches(batch_size=1000, columns=cols):
            d = batch.to_pydict()
            statuses = d.get(STATUS_FIELD)
            for i, value in enumerate(d[text_field]):
                written += 1
                if value:
                    with_text += 1
                    total_chars += len(value)
                    if len(value) >= MIN_SUBSTANTIVE_CHARS:
                        substantive += 1
                if statuses is not None:
                    st = statuses[i] or "unknown"
                    status_counts[st] = status_counts.get(st, 0) + 1
                    if st == "ok":
                        status_ok += 1
    return {
        "num_documents_written": written,
        "num_documents_with_text": with_text,
        "num_documents_substantive": substantive,
        "num_status_ok": status_ok,
        "total_text_chars": total_chars,
        "status_counts": status_counts,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    pipeline = create_mineru_html_pipeline(args, output_dir)
    executor = setup_executor(args.executor)

    logger.info("Starting MinerU-HTML extraction pipeline...")
    start = time.perf_counter()
    try:
        results = pipeline.run(executor, initial_tasks=None)
        success = True
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results = []
        success = False
    elapsed = time.perf_counter() - start

    out_stats = summarize_output(output_dir, args.text_field)

    # Count documents from the written output, NOT from `task.num_items` on the
    # returned tasks. The final stage is a ParquetWriter, whose FileGroupTasks carry
    # num_items == 1 per output *file* -- summing those counts files and silently
    # reports a throughput ~250x too low.
    total_documents = out_stats["num_documents_written"]

    metrics = {
        "is_success": success,
        "time_taken_s": elapsed,
        "num_output_tasks": len(results) if results else 0,
        "num_documents_processed": total_documents,
        "throughput_docs_per_sec": (total_documents / elapsed) if elapsed > 0 else 0.0,
        **out_stats,
    }
    written = out_stats["num_documents_written"]
    # extraction_rate is gated on MIN_SUBSTANTIVE_CHARS. A rate over merely-non-empty
    # text runs several points high because tiny source pages produce a character or
    # two of Markdown and still count.
    metrics["extraction_rate"] = (out_stats["num_documents_substantive"] / written) if written else 0.0
    metrics["nonempty_rate"] = (out_stats["num_documents_with_text"] / written) if written else 0.0
    metrics["status_ok_rate"] = (out_stats["num_status_ok"] / written) if written else 0.0
    metrics["mean_text_chars"] = (
        out_stats["total_text_chars"] / out_stats["num_documents_with_text"]
        if out_stats["num_documents_with_text"]
        else 0.0
    )

    return {
        "params": {
            "input_path": args.input_path,
            "output_path": str(output_dir),
            "html_field": args.html_field,
            "url_field": args.url_field,
            "text_field": args.text_field,
            "limit": args.limit,
            "blocksize": args.blocksize,
            "files_per_partition": args.files_per_partition,
            "model": args.model,
            "max_model_len": args.max_model_len,
            "structured_outputs": args.structured_outputs,
            "kv_cache_dtype": args.kv_cache_dtype,
            "quantization": args.quantization,
            "output_format": args.output_format,
            "fallback": args.fallback,
            "chat_template_mode": args.chat_template_mode,
            "simplify_workers": args.simplify_workers,
            "inference_workers": args.inference_workers,
            "extract_workers": args.extract_workers,
            "executor": args.executor,
        },
        "metrics": metrics,
        "tasks": results or [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="MinerU-HTML main-content extraction benchmark")
    # Contract arg for the nightly driver.
    p.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")
    # Input / output.
    p.add_argument("--input-path", type=str, required=True, help="Parquet file or directory of raw HTML")
    p.add_argument("--output-path", type=str, required=True)
    p.add_argument("--html-field", type=str, default="content")
    p.add_argument("--url-field", type=str, default="url")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--limit", type=int, default=None, help="Keep at most this many documents per reader partition")
    p.add_argument("--blocksize", type=str, default="256MB")
    p.add_argument("--files-per-partition", type=int, default=None)
    # Model / engine.
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--structured-outputs", type=str, default="per_request", choices=["none", "per_request"])
    p.add_argument("--kv-cache-dtype", type=str, default="auto", choices=["auto", "fp8"])
    p.add_argument("--quantization", type=str, default=None)
    p.add_argument("--output-format", type=str, default="mm_md", choices=["mm_md", "md", "json", "txt", "none"])
    p.add_argument("--fallback", type=str, default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    p.add_argument("--chat-template-mode", type=str, default="single", choices=["single", "upstream_double"])
    p.add_argument("--simplify-workers", type=int, default=None)
    p.add_argument(
        "--inference-workers",
        type=int,
        default=None,
        help="GPU actor count, normally one per GPU. Unset lets the backend autoscale from 1, "
        "which under-uses GPUs on short runs because each actor cold-starts a vLLM engine.",
    )
    p.add_argument("--extract-workers", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    # Executor selection.
    p.add_argument("--executor", type=str, default="xenna", choices=["xenna", "ray_data", "ray_actors"])

    args = p.parse_args()
    results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
