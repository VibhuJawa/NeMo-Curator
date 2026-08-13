# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the bidirectional MinerU-HTML versus jusText judge."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import data_designer.config as dd

from eval.text.html_parser import create_html_parser_judge
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Parquet path or glob")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint ending in /v1")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--max-candidate-chars", type=int, default=12000)
    parser.add_argument("--max-parallel-requests", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--checkpoint")
    parser.add_argument("--ray-temp-dir", required=True)
    parser.add_argument("--ray-cpus", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = dd.ChatCompletionInferenceParams(
        max_parallel_requests=args.max_parallel_requests,
        max_tokens=args.max_tokens,
        temperature=0,
        top_p=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    model_configs = [
        dd.ModelConfig(alias="judge", model=args.model, provider=args.provider, inference_parameters=inference)
    ]
    providers = [
        dd.ModelProvider(
            name=args.provider,
            endpoint=args.endpoint,
            api_key=os.environ.get(args.api_key_env, "unused"),
        )
    ]
    pipeline = Pipeline(name="html_parser_judge")
    pipeline.add_stage(ParquetReader(file_paths=args.input))
    pipeline.add_stage(
        create_html_parser_judge(
            args.model,
            model_configs=model_configs,
            model_providers=providers,
            max_candidate_chars=args.max_candidate_chars,
        )
    )
    pipeline.add_stage(ParquetWriter(path=args.output, mode="ignore"))
    client = RayClient(
        num_cpus=args.ray_cpus,
        num_gpus=0,
        include_dashboard=False,
        ray_temp_dir=args.ray_temp_dir,
    )
    try:
        client.start()
        pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
