# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the bidirectional MinerU-HTML versus jusText judge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.text.html_parser import create_html_parser_judge
from eval.text.run import model, parser, run


def main() -> None:
    arguments = parser(__doc__)
    arguments.add_argument("--max-candidate-chars", type=int, default=12000)
    arguments.set_defaults(max_tokens=768)
    args = arguments.parse_args()
    configs, providers = model(args)
    stage = create_html_parser_judge(args.model, model_configs=configs, model_providers=providers,
                                     max_candidate_chars=args.max_candidate_chars)
    run(args, stage, "html_parser_judge")


if __name__ == "__main__":
    main()
