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

"""Build an immutable exact-key Parquet sidecar from a pinned Lance table.

The builder uses only Lance's public scanner API. It projects one key column,
streams record batches, and rolls output files at a row-count boundary so a GPU
actor can load each reference segment without a full-size concatenation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.parquet as pq


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        msg = "storage options must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lance-uri", required=True, help="Pinned source Lance dataset URI")
    parser.add_argument("--version", required=True, type=int, help="Pinned Lance dataset version")
    parser.add_argument("--key-column", default="url", help="Exact reference key column")
    parser.add_argument("--output-dir", required=True, help="Empty local output directory")
    parser.add_argument("--rows-per-file", type=int, default=20_000_000)
    parser.add_argument("--scan-batch-size", type=int, default=262_144)
    parser.add_argument("--batch-readahead", type=int, default=16)
    parser.add_argument("--fragment-readahead", type=int, default=8)
    parser.add_argument("--storage-options-json", type=_json_object, default={})
    return parser.parse_args()


def _open_reference_reader(args: argparse.Namespace) -> tuple[pa.RecordBatchReader, pa.Field]:
    dataset = lance.dataset(
        args.lance_uri,
        version=args.version,
        storage_options=args.storage_options_json or None,
    )
    if args.key_column not in dataset.schema.names:
        msg = f"Lance key column {args.key_column!r} does not exist"
        raise ValueError(msg)
    key_field = dataset.schema.field(args.key_column)
    if not (pa.types.is_string(key_field.type) or pa.types.is_large_string(key_field.type)):
        msg = f"MINT image URL key must be a string, got {key_field.type}"
        raise TypeError(msg)

    reader = dataset.scanner(
        columns=[args.key_column],
        batch_size=args.scan_batch_size,
        batch_readahead=args.batch_readahead,
        fragment_readahead=args.fragment_readahead,
        scan_in_order=True,
    ).to_reader()
    return reader, key_field


def _write_segments(
    reader: pa.RecordBatchReader,
    key_field: pa.Field,
    output_dir: Path,
    rows_per_file: int,
) -> tuple[int, list[dict[str, Any]]]:
    writer: pq.ParquetWriter | None = None
    writer_path: Path | None = None
    segment_rows = 0
    total_rows = 0
    segments: list[dict[str, Any]] = []

    def close_segment() -> None:
        nonlocal writer, writer_path, segment_rows
        if writer is None or writer_path is None:
            return
        writer.close()
        segments.append(
            {
                "path": writer_path.name,
                "rows": segment_rows,
                "bytes": writer_path.stat().st_size,
            }
        )
        writer = None
        writer_path = None
        segment_rows = 0

    try:
        for batch in reader:
            key_array = batch.column(0)
            if key_array.null_count:
                msg = f"Reference key column contains {key_array.null_count} null values"
                raise ValueError(msg)
            if writer is None or segment_rows >= rows_per_file:
                close_segment()
                writer_path = output_dir / f"part-{len(segments):05d}.parquet"
                writer = pq.ParquetWriter(
                    writer_path,
                    pa.schema([key_field]),
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=False,
                    write_statistics=True,
                )
            writer.write_batch(batch)
            segment_rows += batch.num_rows
            total_rows += batch.num_rows
    finally:
        close_segment()
    return total_rows, segments


def main() -> None:
    args = _parse_args()

    if args.rows_per_file <= 0 or args.scan_batch_size <= 0:
        msg = "rows-per-file and scan-batch-size must be greater than zero"
        raise ValueError(msg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        msg = f"Output directory is not empty: {output_dir}"
        raise FileExistsError(msg)

    reader, key_field = _open_reference_reader(args)
    total_rows, segments = _write_segments(reader, key_field, output_dir, args.rows_per_file)

    if total_rows == 0:
        msg = "Pinned Lance table produced no reference keys"
        raise ValueError(msg)

    manifest = {
        "format": "nemo_curator_exact_key_reference_v1",
        "source": {
            "lance_uri": args.lance_uri,
            "version": args.version,
            "key_column": args.key_column,
        },
        "rows": total_rows,
        "segments": segments,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
