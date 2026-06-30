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

"""GPU phase: identify duplicate URLs in configured CC Index snapshots."""

from __future__ import annotations

import argparse

from cc_index_url_list_common import add_common_args, configure_logging, load_config, run_gpu_identification


def parse_args() -> argparse.Namespace:
    """Parse GPU exact-dedup phase arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Curator ExactDeduplicationWorkflow over configured CC Index snapshots. "
            "This phase requires GPU nodes and writes duplicate-ID side outputs under --output/_dedup_ids."
        ),
    )
    add_common_args(parser, include_gpu_args=True)
    return parser.parse_args()


def main() -> None:
    """Run the GPU exact-dedup phase."""
    configure_logging()
    args = parse_args()
    run_gpu_identification(load_config(args.config), args)


if __name__ == "__main__":
    main()
