# MINT-1T HTML Image Presence on GPUs

This tutorial marks which image URLs in a row-wise MINT-1T HTML dataset are
present in a pinned image Lance table. It uses an immutable Parquet projection
of the Lance URL column and keeps that reference set resident on each GPU.

Use this workflow for a large, repeated **exact membership** operation. Use
`LanceColumnFetchStage` instead when the output needs image bytes, hashes,
dimensions, annotations, or another column from the matched Lance row.

## Pipeline

```text
pinned image Lance table
    -> build_reference.py (run once)
    -> immutable URL-only Parquet sidecar

normalized MINT-1T HTML InterleavedBatch Parquet
    -> InterleavedParquetReader
    -> GpuExactKeyLookupStage
    -> InterleavedParquetWriterStage
    -> same rows plus image_present
```

The sidecar is deliberately built through Lance's public scanner API. The
tutorial does not copy, parse, or depend on Lance's private `_indices` layout.

## Install

Create an environment with the interleaved and Lance dependencies, then add
RAPIDS 26.06:

```bash
pip install -e ".[interleaved_cpu,lance]"
pip install "cudf-cu12>=26.6,<26.7"
```

`GpuExactKeyLookupStage` uses the persistent `FilteredJoin` API introduced in
cuDF 26.06. Curator's `deduplication_cuda12` extra currently pins an older
RAPIDS release, so do not install that extra into this environment.

## Inputs

The lookup has two inputs with separate lifecycles.

### 1. MINT interleaved tasks

`main.py` reads normalized, row-wise Parquet through
`InterleavedParquetReader`. Every document element is one row and document
order is represented by `sample_id` and `position`.

| Column | Required value |
|--------|----------------|
| `sample_id` | Stable document identifier |
| `position` | Element position within the document |
| `modality` | `metadata`, `text`, or `image` |
| `source_ref` | Exact image URL on image rows; null on text and metadata rows |

Other columns, such as `text_content`, `document_url`, `image_hash`, and
`images_metadata`, pass through unchanged. The stage does not download images
and the input should not contain image bytes merely to perform presence lookup.

For example:

| sample_id | position | modality | source_ref |
|-----------|----------|----------|------------|
| `doc-a` | -1 | `metadata` | null |
| `doc-a` | 0 | `text` | null |
| `doc-a` | 1 | `image` | `https://example.org/a.jpg` |
| `doc-a` | 2 | `image` | `https://example.org/missing.png` |

Raw MINT-1T HTML stores document content in parallel `images` and `texts`
arrays. Normalize those arrays once into the row-wise interleaved schema before
running this pipeline; this tutorial starts at that persisted boundary so the
expensive presence benchmark is independently resumable and repeatable.

### 2. Immutable reference sidecar

The reference directory contains one non-null exact URL column in segmented
Parquet files and a manifest:

```text
mint-image-urls/
├── manifest.json
├── part-00000.parquet
├── part-00001.parquet
└── ...
```

Every file must use the same Arrow key type. Their union is the exact
membership set. Segmenting avoids creating a second full-size contiguous URL
column during GPU setup. `manifest.json` pins the source Lance URI, version,
key column, total row count, and segment sizes. `main.py` follows the manifest's
ordered file list and validates every file size before starting Ray.

Build the sidecar once from a pinned Lance version:

```bash
python tutorials/interleaved/mint_html_gpu_presence/build_reference.py \
    --lance-uri s3://my-bucket/lance_dbs/mint_images/dataset \
    --version 2 \
    --key-column url \
    --output-dir /local/mint-image-urls \
    --rows-per-file 20000000 \
    --storage-options-json '{"endpoint_url":"https://my-s3-endpoint"}'
```

The output directory must be empty. The builder projects only `url`, rejects
null or non-string keys, writes Zstd Parquet, and records the exact row count.
Copy the completed immutable directory to node-local storage before starting
GPU actors when the source copy lives on a shared or remote filesystem.

## Run

Run one persistent actor per GPU. `task-batch-size` controls how many Curator
tasks an actor coalesces into one GPU probe; it does not change output task
boundaries.

```bash
python tutorials/interleaved/mint_html_gpu_presence/main.py \
    --input-path /path/to/mint_interleaved_parquet \
    --reference-path /local/mint-image-urls \
    --output-path /path/to/mint_interleaved_with_presence \
    --checkpoint-path /path/to/checkpoints \
    --num-cpus 64 \
    --num-gpus 8 \
    --num-workers 8 \
    --task-batch-size 8 \
    --mode error
```

For S3 input or output, pass filesystem credentials through
`--storage-options-json`. The reference sidecar in this tutorial is local so
every actor sees stable, high-throughput setup reads.

## Output

The writer produces row-wise interleaved Parquet with all input rows, their
original order, metadata, and unrelated columns preserved. It appends one
nullable boolean column:

| Input `source_ref` | Output `image_present` |
|--------------------|------------------------|
| null or empty | null; no lookup |
| exact URL in any reference segment | `True` |
| non-empty URL absent from all segments | `False` |

For the example input above, the two image rows become `True` and `False`,
while the metadata and text rows remain null. Duplicate input URLs are looked
up independently and preserve all original rows. The stage fails if
`image_present` already exists, preventing accidental overwrite.

Inspect the result without loading unrelated columns:

```python
import pyarrow.dataset as ds

result = ds.dataset(
    "/path/to/mint_interleaved_with_presence",
    format="parquet",
).to_table(columns=["sample_id", "position", "modality", "source_ref", "image_present"])

print(result.slice(0, 10))
```

## Choosing the stage

| Need | Stage |
|------|-------|
| Mark whether hundreds of millions of exact keys exist | `GpuExactKeyLookupStage` |
| Fetch image bytes, dimensions, hashes, or annotations | `LanceColumnFetchStage` |
| Perform a small or occasional presence query without a GPU sidecar | `LanceColumnFetchStage(columns={})` |

The GPU stage is not a GPU implementation of Lance's B-tree traversal. It is a
persistent exact hash join over a version-pinned key projection. That keeps the
public Lance table authoritative while moving the repeated bulk membership
operation to GPU memory.

## MINT-1T scale notes

An indicative H100 run used 355,952,746 reference URLs and 4,197 normalized
MINT partitions (about 142 GiB). Eight GPU actors completed the presence pass
in 7 minutes 50 seconds. Each actor held about 47.1 GB of persistent GPU state
and left about 37 GB free on an 80 GB H100. These measurements describe that
dataset and hardware; use the emitted setup, transfer, probe, gather, and GPU
memory metrics to size a different run.

The full reference is replicated once per GPU. If it does not fit with useful
headroom, reduce reference scope only through a deterministic partitioning
scheme shared by inputs and references; do not randomly sample URLs when
building a canonical presence dataset.
