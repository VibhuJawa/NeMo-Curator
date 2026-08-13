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

"""Compare MinerU-HTML and jusText extraction output with an LLM judge."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.evaluation import JudgeCriterion, PairwiseLLMJudgeStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch

DEFAULT_INPUT = Path(
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/crawl_data/"
    "crawl_extraction_experiments/justext_vs_dripper_10m/output"
)
_OPTIONAL_COHORT_FIELDS = (
    "_eval_source_file",
    "_eval_source_row",
    "_eval_stable_priority",
    "mineru_html_chars",
    "justext_chars",
    "char_count_difference",
    "relative_char_count_difference",
    "parser_comparison_stratum",
)


@dataclass
class DeterministicSampleStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Take a reproducible per-file sample for an inexpensive smoke evaluation."""

    rows_per_partition: int
    seed: int = 17
    name: str = "deterministic_eval_sample"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        if self.rows_per_partition > 0 and len(df) > self.rows_per_partition:
            df = df.sample(n=self.rows_per_partition, random_state=self.seed).sort_index().reset_index(drop=True)
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", required=True, help="Judge model exposed by an OpenAI-compatible endpoint")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--num-files", type=int, default=1, help="Files to evaluate; 0 means all files")
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=100,
        help="Deterministic rows per input file; 0 means every row",
    )
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    parser.add_argument("--max-candidate-chars", type=int, default=12000)
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not request JSON mode from endpoints that lack response_format support",
    )
    return parser.parse_args()


def resolve_inputs(input_path: Path, num_files: int) -> list[str]:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.parquet"))
    if num_files < 0:
        msg = "--num-files must be non-negative"
        raise ValueError(msg)
    selected = files if num_files == 0 else files[:num_files]
    if not selected:
        msg = f"No parquet inputs found under {input_path}"
        raise FileNotFoundError(msg)
    return [str(path) for path in selected]


def build_pipeline(args: argparse.Namespace, files: list[str]) -> Pipeline:
    client_kwargs = {
        "api_key": args.api_key,
        "max_concurrent_requests": args.max_concurrent_requests,
    }
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = AsyncOpenAIClient(**client_kwargs)

    extra_kwargs = None if args.disable_response_format else {"response_format": {"type": "json_object"}}
    generation_config = GenerationConfig(
        max_tokens=768,
        seed=0,
        temperature=0.0,
        top_p=1.0,
        extra_kwargs=extra_kwargs,
    )
    criteria = [
        JudgeCriterion(
            "useful_content_coverage",
            "Retains the page's substantive, user-visible information without obvious omissions.",
            weight=2.0,
        ),
        JudgeCriterion(
            "boilerplate_precision",
            "Excludes navigation, cookie notices, repeated chrome, ads, and unrelated link lists.",
            weight=2.0,
        ),
        JudgeCriterion(
            "readability",
            "Produces coherent, correctly ordered text that can be read without the original HTML.",
        ),
        JudgeCriterion(
            "structure_preservation",
            "Preserves meaningful headings, lists, tables, links, and other document structure.",
        ),
    ]

    available_fields = set(pq.read_schema(files[0]).names)
    fields = ["url", "text", "justext_extracted_text"]
    fields.extend(field for field in _OPTIONAL_COHORT_FIELDS if field in available_fields)
    pipeline = Pipeline(
        name="html_parser_llm_judge",
        description="Pairwise LLM evaluation of MinerU-HTML and jusText extraction output",
    )
    pipeline.add_stage(
        ParquetReader(
            file_paths=files,
            files_per_partition=1,
            fields=fields,
        )
    )
    pipeline.add_stage(DeterministicSampleStage(rows_per_partition=args.rows_per_file))
    pipeline.add_stage(
        PairwiseLLMJudgeStage(
            client=client,
            model_name=args.model,
            left_field="text",
            right_field="justext_extracted_text",
            left_label="MinerU-HTML",
            right_label="jusText",
            context_fields=["url"],
            criteria=criteria,
            output_prefix="html_parser_judge",
            generation_config=generation_config,
            randomize_order=True,
            random_seed=17,
            max_candidate_chars=args.max_candidate_chars,
        ).with_(num_workers=args.judge_workers)
    )
    # Preserve completed partition outputs when checkpointing skips them on a rerun.
    pipeline.add_stage(ParquetWriter(path=str(args.output), mode="ignore"))
    return pipeline


def main() -> None:
    args = parse_args()
    if args.rows_per_file < 0:
        msg = "--rows-per-file must be non-negative"
        raise ValueError(msg)
    if args.judge_workers <= 0 or args.max_concurrent_requests <= 0:
        msg = "worker and request concurrency must be positive"
        raise ValueError(msg)

    files = resolve_inputs(args.input, args.num_files)
    checkpoint = args.checkpoint or args.output.with_name(f"{args.output.name}.checkpoint")
    pipeline = build_pipeline(args, files)
    print(pipeline.describe())

    ray_client = RayClient(include_dashboard=False)
    try:
        ray_client.start()
        pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
