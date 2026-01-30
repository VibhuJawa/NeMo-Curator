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

"""Stage 4: License Detection for Code Files.

Detect software licenses in code files using NeMo Curator's
CodeLicenseDetection stage with a distributed Ray pipeline.

Requires: uv pip install scancode-toolkit

Usage:
    python 04_license_detection.py --input_file <path> --output_dir <path>
"""

import argparse
from pathlib import Path

import pandas as pd

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.code import CodeLicenseDetection
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import ParquetWriter


def print_summary(output_dir: Path) -> None:
    """Print summary of license detection results."""
    result_files = list(output_dir.glob("04_license_detected/*.parquet"))
    if not result_files:
        return

    df = pd.concat([pd.read_parquet(f) for f in result_files])
    with_license = df["has_license"].sum()
    print(f"\nProcessed {len(df)} files")
    print(f"Files with license: {with_license} ({100 * with_license / len(df):.1f}%)")

    all_licenses = {}
    for licenses in df["licenses"]:
        if isinstance(licenses, list):
            for lic in licenses:
                all_licenses[lic] = all_licenses.get(lic, 0) + 1

    if all_licenses:
        print("\nTop licenses:")
        for lic, count in sorted(all_licenses.items(), key=lambda x: -x[1])[:10]:
            print(f"  {lic}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: License Detection")
    parser.add_argument(
        "--input_file",
        type=str,
        default="code_verification/annotation_verify/nemotron_code_metadata_part_000000_head1500_extracted_rows.jsonl",
    )
    parser.add_argument("--output_dir", type=str, default="tutorials/code/output")
    parser.add_argument("--timeout", type=int, default=100, help="Detection timeout per file (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ray_client = RayClient()
    ray_client.start()

    pipeline = Pipeline(
        name="Code License Detection",
        stages=[
            JsonlReader(file_paths=args.input_file),
            CodeLicenseDetection(content_column="content", detection_timeout=args.timeout),
            ParquetWriter(path=str(output_dir / "04_license_detected")),
        ],
    )

    print(f"Processing: {args.input_file}")
    pipeline.run()

    ray_client.stop()

    print_summary(output_dir)
    print(f"\nResults saved to: {output_dir / '04_license_detected'}")


if __name__ == "__main__":
    main()
