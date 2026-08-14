# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI wiring for Data Designer judge stages."""

import argparse
import os

import data_designer.config as dd

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def parser(description: str, provider: str = "local") -> argparse.ArgumentParser:
    """Create the common local/OpenAI-compatible judge CLI."""
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--input", required=True, help="Input Parquet path or glob")
    result.add_argument("--output", required=True, help="Output directory")
    result.add_argument("--model", required=True)
    result.add_argument("--provider", default=provider)
    result.add_argument("--endpoint", help="OpenAI-compatible endpoint")
    result.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    result.add_argument("--max-parallel-requests", type=int, default=128)
    result.add_argument("--max-tokens", type=int, default=1024)
    result.add_argument("--checkpoint")
    result.add_argument("--ray-temp-dir", required=True)
    result.add_argument("--ray-cpus", type=int, default=32)
    return result


def run(args: argparse.Namespace, stage: object, name: str) -> None:
    """Run a judge stage with Curator's Parquet, Ray, and checkpoint patterns."""
    pipeline = Pipeline(name=name)
    pipeline.add_stage(ParquetReader(file_paths=args.input))
    pipeline.add_stage(stage)
    pipeline.add_stage(ParquetWriter(path=args.output, mode="ignore"))
    client = RayClient(num_cpus=args.ray_cpus, num_gpus=0, include_dashboard=False, ray_temp_dir=args.ray_temp_dir)
    try:
        client.start()
        pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint)
    finally:
        client.stop()


def model(args: argparse.Namespace) -> tuple[list[dd.ModelConfig], list[dd.ModelProvider] | None]:
    """Build Data Designer model/provider configuration from common arguments."""
    inference = dd.ChatCompletionInferenceParams(
        max_parallel_requests=args.max_parallel_requests, max_tokens=args.max_tokens, temperature=0, top_p=1,
        extra_body={"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}})
    configs = [dd.ModelConfig(alias="judge", model=args.model, provider=args.provider, inference_parameters=inference)]
    providers = None if not args.endpoint else [dd.ModelProvider(
        name=args.provider, endpoint=args.endpoint, api_key=os.environ.get(args.api_key_env, "unused"))]
    return configs, providers
