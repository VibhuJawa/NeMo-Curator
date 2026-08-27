# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the Phase-2 continued-pretraining judge over Parquet documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pretraining_readiness import PretrainingReadinessLLMJudgeStage

from eval.text.run import model, parser, run


def main() -> None:
    arguments = parser(__doc__, "nvidia")
    arguments.add_argument("--text-field", default="text")
    arguments.add_argument("--max-document-chars", type=int, default=24000)
    arguments.set_defaults(max_tokens=2048)
    args = arguments.parse_args()
    configs, providers = model(args)
    stage = PretrainingReadinessLLMJudgeStage(
        model_name=args.model, model_configs=configs, model_providers=providers, text_field=args.text_field,
        context_fields=("url",), output_prefix="pretrain", max_document_chars=args.max_document_chars)
    run(args, stage, "phase2_pretraining_readiness")


if __name__ == "__main__":
    main()
