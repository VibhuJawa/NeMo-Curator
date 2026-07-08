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

### GPU Lance installation

From a source checkout, install the GPU Lance stack with:

```bash
uv sync --extra gpu_lance_cuda12
```

The checked-in `uv` configuration pins `lance-ray` to reviewed commit
`fc6d9b9bb85c9adea095f20c87f4c2f0cf760f00` and resolves the PyLance
prerelease from the Lance package index. The extra explicitly pins the RAPIDS
26.06 package family (`cudf-cu12==26.6.*` and
`rapidsmpf-cu12==26.6.*`). It conflicts with the 25.10 deduplication extra and
the image, text, math, and all extras that select that stack; install GPU Lance
in a separate environment from those extras. These source settings are not
embedded in built package metadata. Until `lance-ray==0.5.0` is published, a
package-only installation must provide the exact source, Lance prerelease
index, and NVIDIA package index:

```bash
python -m pip install \
  --extra-index-url https://pypi.fury.io/lance-format/ \
  --extra-index-url https://pypi.nvidia.com/ \
  "lance-ray[gpu] @ git+https://github.com/VibhuJawa/lance-ray.git@fc6d9b9bb85c9adea095f20c87f4c2f0cf760f00"
python -m pip install -e ".[gpu_lance_cuda12]" \
  --extra-index-url https://pypi.fury.io/lance-format/ \
  --extra-index-url https://pypi.nvidia.com/
```

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
    index_cache=LanceIndexCacheConfig(),
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

An index mirror is a strict, local, exact replica rather than a best-effort
cache. Generate its contract once during mirror publication with
`build_lance_index_mirror_contract(...)`, persist the returned fields with the
run configuration, and pass both fields together:

```python
from nemo_curator.stages.interleaved.lance import (
    LanceIndexCacheConfig,
    LanceIndexMirrorContract,
)

index_cache = LanceIndexCacheConfig(
    mirror_path="/lustre/cache/images/v2/dataset",
    mirror_contract=LanceIndexMirrorContract(
        remote_uri="s3://bucket/images/dataset",
        remote_version=2,
        remote_fragment_manifest_sha256="...",
        mirror_uri="/lustre/cache/images/v2/dataset",
        mirror_version=2,
        key_column="url",
        key_stable_ordinal_sha256="...",
        index_name="url_btree",
        index_artifacts_sha256="...",
    ),
)
```

The contract binds both URIs and versions, the ordered fragment metadata, the
Arrow key-to-stable-ordinal stream, and all `_indices` artifact bytes. Node-local
cache paths and ready markers include the full contract digest. `mirror_path`
without `mirror_contract` is rejected; existing callers must either remove
`mirror_path` to use the pinned remote index or publish and pass this contract.

`InterleavedLanceReader` reads fragment partitions from a Lance table directly
into validated `InterleavedBatch` tasks. Together, the two stages support:

```text
InterleavedLanceReader -> LanceColumnFetchStage -> annotator
```

### GPU exact-key presence lookup

For bulk presence-only workflows, `GpuExactKeyLookupStage` loads immutable
Parquet key segments into GPU memory and builds persistent
`pylibcudf.join.FilteredJoin` objects once per actor. It probes exact values
without reading payload columns or rebuilding the GPU hash tables for each
task. The `gpu-lance-cuda12` extra pins the implementation to
`cudf-cu12==26.6.*`, `rapidsmpf-cu12==26.6.*`, and their matching RAPIDS
26.06 dependency stack.

The stage has two inputs:

| Input | Required content |
|-------|------------------|
| Curator task | An `InterleavedBatch` with `input_key_column`; its Arrow type must match the reference key type (regular and large UTF-8 strings are compatible) |
| Reference set | One or more immutable Parquet files containing a non-null `reference_key_column`; the union of these columns is the exact membership set |

It returns one `InterleavedBatch` for every input task. Every input row and
column is preserved, and one nullable boolean column is appended:

| Input key | Output `image_present` |
|-----------|------------------------|
| Null or empty string | Null; no GPU lookup is performed |
| Exact key in any reference file | `True` |
| Non-empty key absent from every reference file | `False` |

If the destination presence column already exists, the task fails rather than
silently overwriting it. Duplicate input keys are allowed and preserve their
individual rows. The reference files are a set for membership purposes, so a
key appearing in more than one reference segment still produces one boolean
result per input row.

```python
from nemo_curator.stages.interleaved import GpuExactKeyLookupStage

stage = GpuExactKeyLookupStage(
    reference_files=[
        "/local/image-urls/segment-000.parquet",
        "/local/image-urls/segment-001.parquet",
    ],
    reference_key_column="url",
    input_key_column="source_ref",
    presence_column="image_present",
    expected_reference_rows=355_952_746,
).with_(num_workers=8, batch_size=8)
```

Each reference file is retained as an independent hash table. This avoids a
second full-size allocation to concatenate large string columns during actor
setup. `process_batch()` concatenates eligible input keys for one probe and
then restores the original task boundaries and row order. Null and empty
string keys are not queried and receive null presence; all other absent keys
receive `False`.

The stage consumes a stable Parquet key sidecar rather than Lance's private
on-disk scalar-index files. Build that sidecar from a pinned source table using
its normal public reader API. Use `LanceColumnFetchStage` instead when payload
columns or stable row IDs must be returned. See the
[MINT-1T HTML GPU presence tutorial](../../../tutorials/interleaved/mint_html_gpu_presence/)
for a complete reader → lookup → writer pipeline and a public Lance-to-Parquet
sidecar builder.

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
├── gpu_key_lookup.py               # Persistent GPU exact-key membership
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
