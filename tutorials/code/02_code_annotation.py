# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Stage 2: Code Annotation with Quality Metrics.

Compute quality metrics for code files using NeMo Curator's CodeAnnotation
stage with a distributed Ray pipeline.

Usage:
    python 02_code_annotation.py --input_file <path> --output_dir <path>
"""

import argparse
from pathlib import Path

import pandas as pd

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.code import CodeAnnotation
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import ParquetWriter


def print_summary(output_dir: Path) -> None:
    """Print summary of annotation results."""
    result_files = list(output_dir.glob("02_annotated/*.parquet"))
    if not result_files:
        return

    df = pd.concat([pd.read_parquet(f) for f in result_files])
    print(f"\nProcessed {len(df)} files with {len(df.columns)} columns")
    print(f"Total bytes: {df['basic_num_bytes'].sum():,}")
    print(f"Total lines: {df['basic_num_lines'].sum():,}")
    print(f"Avg comment fraction: {df['ors_comment_lines_frac'].mean():.2%}")

    token_col = next(c for c in df.columns if c.startswith("num_tokens_"))
    print(f"Total tokens: {df[token_col].sum():,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Code Annotation")
    parser.add_argument(
        "--input_file",
        type=str,
        default="code_verification/annotation_verify/nemotron_code_metadata_part_000000_head1500_extracted_rows.jsonl",
    )
    parser.add_argument("--output_dir", type=str, default="tutorials/code/output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ray_client = RayClient()
    ray_client.start()

    pipeline = Pipeline(
        name="Code Annotation",
        stages=[
            JsonlReader(file_paths=args.input_file),
            # Note: software_metrics=True is slower but provides complexity metrics
            CodeAnnotation(
                content_column="content",
                filename_column="rel_path",
                detect_language=True,
                basic_stats=True,
                software_metrics=True,
                opencoder_metrics=True,
                tokenize=True,
            ),
            ParquetWriter(path=str(output_dir / "02_annotated")),
        ],
    )

    print(f"Processing: {args.input_file}")
    pipeline.run()

    ray_client.stop()

    print_summary(output_dir)
    print(f"\nResults saved to: {output_dir / '02_annotated'}")


if __name__ == "__main__":
    main()
