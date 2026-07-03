# Interleaved Pipeline

Row-wise interleaved multimodal ingestion and write path for WebDataset tar shards (MINT-1T style), with materialization support for local, remote, and tar-archived binary content.

## Architecture

```
WebDataset tar shards          Parquet files
        |                            |
        v                            v
┌──────────────────────────┐  ┌──────────────────────────┐
│ InterleavedWebdataset-   │  │ InterleavedParquetReader  │  Both are CompositeStages:
│ Reader (io/reader.py)    │  │ (io/reader.py)            │  FilePartitioningStage +
│                          │  │                           │  <modality>ReaderStage
└──────────┬───────────────┘  └────────────┬─────────────┘
           └──────────┬────────────────────┘
                      |  InterleavedBatch (Arrow/Pandas)
                      v
         ┌─────────────────────────┐
         │  Filter Stages          │  e.g. InterleavedAspectRatioFilterStage
         │  (stages.py)            │  Row-wise filtering with optional materialization
         └────────┬────────────────┘
                  |
        ┌─────────┴──────────┐
        v                    v
┌───────────────┐   ┌──────────────────────────┐
│ Interleaved-  │   │ InterleavedWebdataset-    │
│ ParquetWriter │   │ WriterStage               │
│ Stage         │   │ (io/writers/webdataset.py)│
│ (tabular.py)  │   │ MINT-1T-style tar shards  │
└───────────────┘   └──────────────────────────┘
```

## Schema (`INTERLEAVED_SCHEMA`)

Defined in `nemo_curator/tasks/interleaved.py`. Columns are split into **reserved** (managed by the pipeline) and **user** (passthrough from source data).

### Reserved columns (`RESERVED_COLUMNS`)

These are set and managed by pipeline stages. Users should not write to them directly.

| Column | Type | Category | Description |
|--------|------|----------|-------------|
| `sample_id` | string (required) | Identity | Unique document/sample identifier |
| `position` | int32 (required) | Identity | Position within sample (-1 for metadata rows) |
| `modality` | string (required) | Identity | Row modality: `text`, `image`, `metadata` built-in; extensible to `audio`, `table`, `generated_image`, etc. |
| `content_type` | string | Content | MIME type (e.g. `text/plain`, `image/jpeg`) |
| `text_content` | string | Content | Text payload for text rows |
| `binary_content` | large_binary | Content | Image bytes (populated by materialization) |
| `source_ref` | string | Internal | JSON locator `{path, member, byte_offset, byte_size, frame_index}`. `path` alone = direct/remote read; + `member` = tar extract; + `byte_offset/size` = range read (fastest). `path` accepts local or remote (`s3://`) URIs. |
| `materialize_error` | string | Internal | Error message if materialization failed |

### User columns (passthrough)

Extra fields from the source data flow through the pipeline as additional columns. Specify them with the `fields` parameter on the reader:

```python
reader = InterleavedWebdatasetReader(
    file_paths="/data/shards/",
    fields=("p_hash", "score", "aux"),  # These become extra columns
)
```

If `fields` is `None` (default), all non-reserved fields from the source JSON are passed through. If specified explicitly, only the listed fields are included -- and the reader validates they exist and don't collide with reserved names.

## Key Concepts

### InterleavedBatch

The task type for interleaved multimodal data (`nemo_curator/tasks/interleaved.py`). Wraps either a PyArrow Table or Pandas DataFrame.

Class attributes:
- `REQUIRED_COLUMNS` -- frozenset of columns that must always be present (non-nullable schema fields)

Key methods:
- `build_source_ref(path, member, byte_offset, byte_size, frame_index)` -- build a JSON locator string
- `parse_source_ref(value)` -- parse back with soft migration for older formats
- `with_parsed_source_ref_columns(prefix)` -- expand source_ref into DataFrame columns
- `to_pyarrow()` / `to_pandas()` -- conversion between formats

### source_ref

A JSON string embedded in each row that tracks where the original content lives:

```json
{
  "path": "/data/shard-00000.tar",
  "member": "abc123.jpg",
  "byte_offset": 1024,
  "byte_size": 45678,
  "frame_index": null
}
```

- `path` + `member` -- tar archive path and member name
- `path` alone (no member) -- direct file path
- `byte_offset` + `byte_size` -- enables range reads without opening the tar
- `frame_index` (optional) -- selects a single frame from a multi-frame TIFF during materialization

### Materialization

Binary content (images) can be loaded lazily. Three I/O strategies dispatch automatically based on `source_ref` content (`utils/materialization.py`):

| Strategy | When | How |
|----------|------|-----|
| **Range read** | `byte_offset` + `byte_size` present | `fs.cat_ranges()` -- batched HTTP range requests per path |
| **Tar extract** | `member` present, no byte range | Open tar once, `extractfile()` per member |
| **Direct read** | No `member` | Read entire file via `fsspec.open()` |

When `frame_index` is set in the `source_ref`, materialization extracts a single frame from a multi-frame TIFF and returns it as a standalone TIFF. Non-TIFF content is returned unchanged regardless of `frame_index`.

Materialization can happen at read time (`materialize_on_read=True`) or write time (`materialize_on_write=True`).

### Indexed Lance column fetches

`LanceColumnFetchStage` performs an exact-key lookup against a pinned, scalar-indexed
Lance dataset and adds a selected column projection to an `InterleavedBatch`. The
input key column is opaque to the stage; for URL-backed image tables it can be the
exact URL stored directly in `source_ref`.

```python
from nemo_curator.stages.interleaved import (
    LanceColumnFetchStage,
    LanceDatasetConfig,
    LanceIndexCacheConfig,
)

stage = LanceColumnFetchStage(
    dataset=LanceDatasetConfig(
        uri="s3://bucket/images/dataset",
        version=2,
        key_column="url",
        index_name="url_btree",
        storage_options={...},
    ),
    index_cache=LanceIndexCacheConfig(mirror_path="/lustre/cache/images/v2/dataset"),
    input_key_column="source_ref",
    columns={
        "image": "binary_content",
        "md5": "reference_md5",
        "width": "reference_width",
        "height": "reference_height",
    },
    presence_column="image_present",
    existing_column_policy="fill_null",
    fetch_batch_size=32,
)
```

Use `columns={}` with a `presence_column` for an index-only presence pass. The
default destination collision policy is `error`; `fill_null` and `overwrite`
must be requested explicitly. Missing keys can either be marked in the presence
column or fail the task. The stage preserves Arrow types and does not decode
binary columns.

`InterleavedLanceReader` reads fragment partitions from a Lance table directly
into validated `InterleavedBatch` tasks. Together, the two stages support:

```text
InterleavedLanceReader -> LanceColumnFetchStage -> annotator
```

## Usage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedWebdatasetReader, InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage

pipeline = Pipeline(name="mint1t_pipeline")
pipeline.add_stage(InterleavedWebdatasetReader(
    file_paths="/data/mint1t/shards/",
))
pipeline.add_stage(InterleavedAspectRatioFilterStage(drop_invalid_rows=True))
pipeline.add_stage(InterleavedParquetWriterStage(
    path="/output/parquet/",
    materialize_on_write=True,
    mode="overwrite",
))
pipeline.run()
```

## File Layout

```
stages/interleaved/
├── __init__.py                     # Exports filter/annotator stages
├── lance.py                        # Lance column fetch and interleaved Lance reader
├── stages.py                       # BaseInterleavedAnnotatorStage, BaseInterleavedFilterStage,
│                                   # InterleavedAspectRatioFilterStage
├── io/
│   ├── __init__.py                 # Exports InterleavedWebdatasetReader, InterleavedParquetReader,
│   │                               # InterleavedParquetWriterStage, InterleavedWebdatasetWriterStage
│   ├── reader.py                   # InterleavedWebdatasetReader, InterleavedParquetReader (CompositeStages)
│   ├── readers/
│   │   ├── base.py                 # BaseInterleavedReader
│   │   ├── parquet.py              # InterleavedParquetReaderStage (ProcessingStage)
│   │   └── webdataset.py           # InterleavedWebdatasetReaderStage (ProcessingStage)
│   └── writers/
│       ├── base.py                 # BaseInterleavedWriter (filesystem + materialization + process)
│       ├── tabular.py              # InterleavedParquetWriterStage
│       └── webdataset.py           # InterleavedWebdatasetWriterStage
└── utils/
    ├── constants.py                # Default file extensions
    ├── materialization.py          # Three-strategy materialization dispatch
    └── validation_utils.py         # Field validation, storage options resolution
```
