# Multimodal Tutorials

## OmniCorpus Custom Reader (Tutorial-Only)

Files:
- `tutorials/multimodal/omnicorpus_custom_reader.py` (reader abstractions)
- `tutorials/multimodal/omnicorpus_pipeline.py` (pipeline wiring + execution)

This tutorial adds a custom reader for `OpenGVLab/OmniCorpus-CC-210M` **without changing core**.

### Why this reader is custom

OmniCorpus parquet shards store:
- `texts`: list of strings
- `images`: list of URL strings (not embedded bytes)
- `general_metadata`: struct

This does not match the normalized multimodal parquet contract used by core readers, so we map it in a custom stage.

### Composability pattern

The reader module contains:
- `OmniCorpusReaderStage`: a single tutorial stage that maps OmniCorpus rows into normalized multimodal rows
- `OmniCorpusReader`: composite reader (`FilePartitioningStage` + stage)

This keeps tutorial burden low while still showing the core extension points.

### Metadata placement contract

To keep schemas composable and reader/writer behavior simple:
- Put sample-shared values directly in the main table as rows with
  `modality="metadata"` and `position=-1`.
- Put per-element values in data rows (`element_metadata_json` in `MULTIMODAL_SCHEMA`).

In the OmniCorpus tutorial:
- sample-level payload keeps `general_metadata`
- per-text / per-image payloads come from row-level `metadata` entries via
  `text_element_metadata(...)` and `image_element_metadata(...)`

### Quick start

1. Download a local subset:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="OpenGVLab/OmniCorpus-CC-210M",
    repo_type="dataset",
    allow_patterns=["data/CC-MAIN-2016-26/shard_0.parquet"],
    local_dir="/raid/vjawa/tmp_omnicorpus_subset",
)
PY
```

2. Run end-to-end tutorial (reader + WebDataset writer):

```bash
python tutorials/multimodal/omnicorpus_pipeline.py
```

This runs:
- a pipeline (`OmniCorpusReader` + `MultimodalWriterStage`) on one parquet shard
- writer-driven image materialization on full shard rows
- `MultimodalWriterStage(output_format="webdataset")`
- outputs written under:
  - `/raid/vjawa/tmp_omnicorpus_subset/tutorial_output/`

Note:
- For old crawl URLs, many links are dead.
- The tutorial uses writer materialization policies:
  - retries (`materialize_max_retries`)
  - backoff (`materialize_retry_backoff_sec`)
  - failure handling (`materialize_failure_policy="drop_image"`)

### URL materialization notes

- Image rows are URL-backed (`content_path=<http-url>`, `binary_content=None`).
- `batch.materialize(modality="image")` fetches bytes via fsspec.
- For web-crawl datasets, dead/blocked URLs are common. Use small batches and retry/skip policies around materialization.
