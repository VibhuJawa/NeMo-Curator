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

"""Stream judged parquet partitions into an auditable pretraining-readiness report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.dataset as ds

from nemo_curator.stages.evaluation import PretrainingJudgeSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Judged parquet file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report")
    parser.add_argument(
        "--output-prefix", default="pretrain_judge", help="Prefix used by PretrainingReadinessLLMJudgeStage"
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    return parser.parse_args()


def resolve_inputs(input_path: Path) -> list[str]:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.parquet"))
    if not files:
        msg = f"No parquet inputs found under {input_path}"
        raise FileNotFoundError(msg)
    return [str(path) for path in files]


def summarize(files: list[str], output_prefix: str, batch_size: int) -> dict:
    if batch_size <= 0:
        msg = "--batch-size must be positive"
        raise ValueError(msg)
    dataset = ds.dataset(files, format="parquet")
    available = set(dataset.schema.names)
    weight_column = "_eval_sample_weight" if "_eval_sample_weight" in available else None
    summary = PretrainingJudgeSummary(output_prefix=output_prefix, weight_column=weight_column)
    missing = set(summary.required_columns()) - available
    if missing:
        msg = f"Judged parquet schema is missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    scanner = dataset.scanner(columns=summary.required_columns(), batch_size=batch_size, use_threads=True)
    for batch in scanner.to_batches():
        summary.update(batch.to_pandas())
    report = summary.as_dict()
    report["input_files"] = len(files)
    report["sample_weight_column"] = weight_column
    return report


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    files = resolve_inputs(args.input)
    report = summarize(files, args.output_prefix, args.batch_size)
    write_report(report, args.output)
    rows = report["rows"]
    print(
        f"Wrote {args.output}: {rows['successful']}/{rows['total']} rows succeeded "
        f"across {report['input_files']} parquet files"
    )


if __name__ == "__main__":
    main()
