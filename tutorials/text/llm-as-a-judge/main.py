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
"""Run the Phase-2 continued-pretraining judge over Parquet documents."""

import argparse
import os
import sys
from pathlib import Path

# Top-level eval code is intentionally outside the installed Curator package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import data_designer.config as dd
from pretraining_readiness import PretrainingReadinessLLMJudgeStage

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Parquet path or glob")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True, help="Judge model identifier")
    parser.add_argument("--provider", default="nvidia", help="Data Designer provider name")
    parser.add_argument("--endpoint", help="Optional OpenAI-compatible endpoint")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY", help="Environment variable containing its key")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-document-chars", type=int, default=24000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_configs = [dd.ModelConfig(alias="judge", model=args.model, provider=args.provider)]
    providers = None
    if args.endpoint:
        providers = [
            dd.ModelProvider(
                name=args.provider,
                endpoint=args.endpoint,
                api_key=os.environ.get(args.api_key_env),
            )
        ]
    pipeline = Pipeline(name="phase2_pretraining_readiness")
    pipeline.add_stage(ParquetReader(file_paths=args.input))
    pipeline.add_stage(
        PretrainingReadinessLLMJudgeStage(
            model_name=args.model,
            model_configs=model_configs,
            model_providers=providers,
            text_field=args.text_field,
            context_fields=("url",),
            output_prefix="pretrain",
            max_document_chars=args.max_document_chars,
        )
    )
    pipeline.add_stage(ParquetWriter(path=args.output, mode="overwrite"))
    pipeline.run(executor=RayDataExecutor())


if __name__ == "__main__":
    main()
