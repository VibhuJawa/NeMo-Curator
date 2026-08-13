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

"""Classify Common Crawl documents for phase-2 pretraining readiness."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.models.client import AsyncOpenAIClient, OpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.evaluation import PretrainingReadinessLLMJudgeStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch

DEFAULT_INPUT = Path(
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/crawl_data/"
    "crawl_extraction_experiments/justext_vs_dripper_10m/output"
)


@dataclass(frozen=True)
class SamplingPlan:
    """Selected files and any exact per-source sampling allocation."""

    files: list[str]
    sample_rows_by_source: dict[str, int] | None = None
    source_rows_by_file: dict[str, int] | None = None


@dataclass
class DeterministicSampleStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Take a reproducible per-file sample before issuing model requests."""

    rows_per_partition: int
    seed: int = 17
    sample_rows_by_source: dict[str, int] | None = None
    source_rows_by_file: dict[str, int] | None = None
    name: str = "deterministic_pretraining_sample"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "_eval_source_file",
            "_eval_source_rows",
            "_eval_sample_rows",
            "_eval_inclusion_probability",
            "_eval_sample_weight",
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        source_file = self._source_file(batch)
        source_rows = len(df)
        target_rows = self._target_rows(source_file, source_rows)
        if source_rows > target_rows:
            df = df.sample(n=target_rows, random_state=self.seed).sort_index().reset_index(drop=True)
        sample_rows = len(df)
        inclusion_probability = sample_rows / source_rows
        df = df.copy()
        df["_eval_source_file"] = source_file
        df["_eval_source_rows"] = source_rows
        df["_eval_sample_rows"] = sample_rows
        df["_eval_inclusion_probability"] = inclusion_probability
        df["_eval_sample_weight"] = 1.0 / inclusion_probability
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _source_file(self, batch: DocumentBatch) -> str:
        source_files = (batch._metadata or {}).get("source_files", [])
        if len(source_files) != 1:
            msg = "DeterministicSampleStage requires exactly one source file per partition"
            raise ValueError(msg)
        return str(source_files[0])

    def _target_rows(self, source_file: str, observed_rows: int) -> int:
        if self.source_rows_by_file is not None and self.source_rows_by_file.get(source_file) != observed_rows:
            msg = f"Parquet row count changed for {source_file}; rebuild the sample plan"
            raise ValueError(msg)
        if self.sample_rows_by_source is not None:
            target = self.sample_rows_by_source.get(source_file)
            if target is None or target <= 0:
                msg = f"Missing positive sample allocation for {source_file}"
                raise ValueError(msg)
            return min(target, observed_rows)
        return observed_rows if self.rows_per_partition == 0 else min(self.rows_per_partition, observed_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", required=True, help="Judge model exposed by an OpenAI-compatible endpoint")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Optional OpenAI model alias; local serving defaults to pretraining-judge",
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--text-field",
        default="text",
        help="Extracted text column: text for MinerU-HTML or justext_extracted_text for jusText",
    )
    parser.add_argument("--output-prefix", default="pretrain_judge")
    parser.add_argument("--num-files", type=int, default=1, help="Files to classify; 0 means all files")
    parser.add_argument("--file-selection-seed", type=int, default=17)
    parser.add_argument(
        "--target-sample-rows",
        type=int,
        default=None,
        help="Allocate this many rows proportionally across selected parquet files; overrides --rows-per-file",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=100,
        help="Deterministic rows per input file; 0 means every row",
    )
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument("--row-sample-seed", type=int, default=17)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    parser.add_argument("--max-document-chars", type=int, default=24000)
    parser.add_argument(
        "--serve-model-locally",
        action="store_true",
        help="Own a local Dynamo/vLLM InferenceServer lifecycle instead of using --base-url",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs used by the local judge model")
    parser.add_argument("--local-max-model-len", type=int, default=32768)
    parser.add_argument("--local-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--server-health-timeout", type=int, default=600)
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not request JSON mode from endpoints that lack response_format support",
    )
    return parser.parse_args()


def resolve_inputs(input_path: Path, num_files: int, seed: int = 17) -> list[str]:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.parquet"))
    if num_files < 0:
        msg = "--num-files must be non-negative"
        raise ValueError(msg)
    ranked = sorted(files, key=lambda path: _stable_file_rank(path, seed))
    selected = ranked if num_files == 0 else ranked[:num_files]
    if not selected:
        msg = f"No parquet inputs found under {input_path}"
        raise FileNotFoundError(msg)
    return [str(path) for path in selected]


def _stable_file_rank(path: Path, seed: int) -> bytes:
    identity = f"{seed}:{path}".encode()
    return hashlib.blake2b(identity, digest_size=16).digest()


def allocate_proportional_sample(files: list[str], target_rows: int, seed: int) -> SamplingPlan:
    """Allocate an exact global sample approximately uniformly over source rows."""
    if target_rows <= 0:
        msg = "--target-sample-rows must be positive"
        raise ValueError(msg)
    source_rows = {path: pq.ParquetFile(path).metadata.num_rows for path in files}
    total_rows = sum(source_rows.values())
    if target_rows > total_rows:
        msg = f"--target-sample-rows={target_rows} exceeds the selected input's {total_rows} rows"
        raise ValueError(msg)

    exact = {path: target_rows * rows / total_rows for path, rows in source_rows.items()}
    allocation = {path: math.floor(value) for path, value in exact.items()}
    remaining = target_rows - sum(allocation.values())
    remainder_order = sorted(
        files,
        key=lambda path: (-(exact[path] - allocation[path]), _stable_file_rank(Path(path), seed)),
    )
    for path in remainder_order[:remaining]:
        allocation[path] += 1

    selected = [path for path in files if allocation[path] > 0]
    selected_allocations = {path: allocation[path] for path in selected}
    selected_source_rows = {path: source_rows[path] for path in selected}
    if sum(selected_allocations.values()) != target_rows:
        msg = "proportional sample allocation did not preserve the requested row count"
        raise RuntimeError(msg)
    return SamplingPlan(selected, selected_allocations, selected_source_rows)


def build_pipeline(
    args: argparse.Namespace,
    sampling_plan: SamplingPlan,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
) -> Pipeline:
    client_kwargs = {
        "api_key": args.api_key,
        "max_concurrent_requests": args.max_concurrent_requests,
    }
    resolved_base_url = base_url or args.base_url
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = AsyncOpenAIClient(**client_kwargs)
    extra_kwargs = None if args.disable_response_format else {"response_format": {"type": "json_object"}}
    generation_config = GenerationConfig(
        max_tokens=2048,
        seed=0,
        temperature=0.0,
        top_p=1.0,
        extra_kwargs=extra_kwargs,
    )

    pipeline = Pipeline(
        name="cc_pretraining_readiness_llm_judge",
        description="Versioned topic, quality, language, mixture, and review signals for Phase-2 pretraining",
    )
    pipeline.add_stage(
        ParquetReader(
            file_paths=sampling_plan.files,
            files_per_partition=1,
            fields=["url", args.text_field],
        )
    )
    pipeline.add_stage(
        DeterministicSampleStage(
            rows_per_partition=args.rows_per_file,
            seed=args.row_sample_seed,
            sample_rows_by_source=sampling_plan.sample_rows_by_source,
            source_rows_by_file=sampling_plan.source_rows_by_file,
        )
    )
    pipeline.add_stage(
        PretrainingReadinessLLMJudgeStage(
            client=client,
            model_name=model_name or args.served_model_name or args.model,
            text_field=args.text_field,
            context_fields=["url"],
            output_prefix=args.output_prefix,
            generation_config=generation_config,
            max_document_chars=args.max_document_chars,
        ).with_(num_workers=args.judge_workers)
    )
    # Preserve completed partition outputs when checkpointing skips them on a rerun.
    pipeline.add_stage(ParquetWriter(path=str(args.output), mode="ignore"))
    return pipeline


def smoke_test_endpoint(base_url: str, model_name: str, api_key: str) -> None:
    """Prove model discovery is followed by one successful generation."""
    client = OpenAIClient(base_url=base_url, api_key=api_key, timeout=120)
    response = client.query_model(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with exactly READY."}],
        generation_config=GenerationConfig(max_tokens=16, temperature=0.0, top_p=1.0),
    )
    if not response or not response[0].strip():
        msg = "Local inference server returned an empty smoke-test response"
        raise RuntimeError(msg)


def run_pipeline(
    args: argparse.Namespace,
    sampling_plan: SamplingPlan,
    checkpoint: Path,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
) -> None:
    pipeline = build_pipeline(
        args,
        sampling_plan,
        base_url=base_url,
        model_name=model_name,
    )
    print(pipeline.describe())
    pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint)


def main() -> None:
    args = parse_args()
    if args.rows_per_file < 0:
        msg = "--rows-per-file must be non-negative"
        raise ValueError(msg)
    if args.target_sample_rows is not None and args.target_sample_rows <= 0:
        msg = "--target-sample-rows must be positive"
        raise ValueError(msg)
    if args.judge_workers <= 0 or args.max_concurrent_requests <= 0:
        msg = "worker and request concurrency must be positive"
        raise ValueError(msg)
    if args.max_document_chars <= 0:
        msg = "--max-document-chars must be positive"
        raise ValueError(msg)
    if args.tensor_parallel_size <= 0 or args.local_max_model_len <= 0 or args.server_health_timeout <= 0:
        msg = "local tensor parallelism, model length, and health timeout must be positive"
        raise ValueError(msg)
    if not 0.0 < args.local_gpu_memory_utilization <= 1.0:
        msg = "--local-gpu-memory-utilization must be in (0, 1]"
        raise ValueError(msg)
    if not args.text_field.strip() or not args.output_prefix.strip():
        msg = "--text-field and --output-prefix must not be empty"
        raise ValueError(msg)

    files = resolve_inputs(args.input, args.num_files, args.file_selection_seed)
    sampling_plan = SamplingPlan(files)
    if args.target_sample_rows is not None:
        sampling_plan = allocate_proportional_sample(
            files,
            args.target_sample_rows,
            args.row_sample_seed,
        )
    checkpoint = args.checkpoint or args.output.with_name(f"{args.output.name}.checkpoint")
    ray_client = RayClient(
        include_dashboard=False,
        num_gpus=args.tensor_parallel_size if args.serve_model_locally else None,
    )
    try:
        ray_client.start()
        if args.serve_model_locally:
            from nemo_curator.core.serve import (
                DynamoServerConfig,
                DynamoVLLMModelConfig,
                InferenceServer,
            )

            served_model_name = args.served_model_name or "pretraining-judge"
            model_config = DynamoVLLMModelConfig(
                model_identifier=args.model,
                model_name=served_model_name,
                engine_kwargs={
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "max_model_len": args.local_max_model_len,
                    "gpu_memory_utilization": args.local_gpu_memory_utilization,
                },
            )
            with InferenceServer(
                models=[model_config],
                backend=DynamoServerConfig(),
                health_check_timeout_s=args.server_health_timeout,
            ) as server:
                smoke_test_endpoint(server.endpoint, served_model_name, args.api_key)
                run_pipeline(
                    args,
                    sampling_plan,
                    checkpoint,
                    base_url=server.endpoint,
                    model_name=served_model_name,
                )
        else:
            run_pipeline(args, sampling_plan, checkpoint)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
