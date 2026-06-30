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

"""CPU phase: remove duplicate URLs and write the final CC Index URL list."""

from __future__ import annotations

import argparse

from cc_index_url_list.cli import add_common_args, configure_logging
from cc_index_url_list.config import load_config
from cc_index_url_list.phases import run_cpu_removal


def parse_args() -> argparse.Namespace:
    """Parse CPU duplicate-removal phase arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Curator TextDuplicatesRemovalWorkflow using exact-dedup side outputs produced by "
            "identify_cc_index_url_duplicates.py. This phase is CPU-only and writes the final parquet dataset."
        ),
    )
    add_common_args(parser, include_gpu_args=False)
    return parser.parse_args()


def main() -> None:
    """Run the CPU duplicate-removal phase."""
    configure_logging()
    args = parse_args()
    run_cpu_removal(load_config(args.config), args)


if __name__ == "__main__":
    main()
