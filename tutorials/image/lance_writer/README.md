# Retry-attempt image tars to Lance

This workflow builds a URL-addressable Lance table from image/JSON pairs stored
in tar files. It is designed for sources that retry the same logical tar: every
physical attempt is processed together, and exactly one winner is selected per
exact URL before the image is written.

The source contract must keep every retry copy of a URL in the same stable
`(source_shard, tar_id)` FPP. Cross-tar or cross-shard URL deduplication is a
separate dataset-wide operation and is not performed by this ingestion recipe.

The example focuses on ingestion. Dataset-wide validation, compaction,
secondary-index construction, and benchmarking are intentionally outside its
scope.

## Pipeline

```text
immutable tar inventory
        │
        ▼
stable (source_shard, tar_id) FPPs
        │  group all attempts; pack within one source shard toward 1 GiB
        ▼
FppPackPartitioningStage
        │  deterministic task IDs + Slurm-array filtering
        ▼
FppPackMaterializationStage ──fused──> LanceWriter
        │                                │
        │ Arrow large_binary images      └─ uncommitted fragments + checkpoints
        ▼
commit_lance_checkpoint (once, after every logical shard is complete)
```

The materializer and writer both request one CPU, allowing Ray Data to fuse
them. This avoids retaining an additional image-heavy intermediate block.

## Input contracts

The inventory is a Parquet table with one row per physical tar:

| Column | Type | Meaning |
|---|---|---|
| `source_shard` | integer | Source shard containing the logical tar |
| `tar_id` | string | Stable logical tar ID |
| `attempt` | integer | Monotonically increasing retry attempt |
| `tar_uri` | string | Full `s3://` URI |
| `tar_size` | integer | Object size in bytes |
| `tar_etag` | string | Object ETag |
| `last_modified` | string | Stable timestamp from the inventory pass |

Each tar contains image and JSON members with the same member stem. JSON rows
must contain `url`; `width`, `height`, `sha256`, and `status` are consumed when
present. Pillow determines the actual encoded format, so JPEG, MPO, PNG, and
WebP payloads remain distinguishable even when the member suffix is `.jpg`.

For equal URLs, the winner policy is:

1. largest pixel area;
2. newest attempt;
3. lexicographically smallest source ID as a deterministic tie-breaker.

The output includes the encoded image as ordinary Arrow `large_binary`, MD5
and SHA-256 hashes, image format/MIME/dimensions, the original JSON, and only
the winning tar/member provenance.

## Environment

Install the image and Lance dependencies directly:

```bash
pip install -e '.[image_lance]'
```

Use the standard AWS environment variables for credentials. S3-compatible
endpoint settings are explicit JSON objects because fsspec and Lance use
different option shapes:

```bash
export SOURCE_STORAGE_OPTIONS='{"client_kwargs":{"endpoint_url":"https://s3.example"}}'
export LANCE_STORAGE_OPTIONS='{"endpoint":"https://s3.example","aws_region":"us-east-1","virtual_hosted_style_request":"false","client_max_retries":"20"}'
```

## Build the manifest once

Run this on a CPU node after the source inventory is frozen:

```bash
python -m tutorials.image.lance_writer.pipeline build-manifest \
  --inventory /shared/inventory/physical_tars.parquet \
  --manifest-dir /shared/manifests/images-v1 \
  --target-pack-bytes 1073741824
```

The snapshot ID hashes the inventory descriptors and pack target. The command
reuses an identical manifest and refuses to overwrite a different one.

## Run with Slurm arrays and resumability

Choose enough logical shards to give each array element many packs. Limit
concurrent nodes from measured object-store throughput rather than CPU count.

```bash
export MANIFEST_DIR=/shared/manifests/images-v1
export DATASET_URI=s3://output-bucket/lance/images-v1
export LANCE_COMMIT_PATH=/shared/checkpoints/images-v1/lance
export CHECKPOINT_PATH=/shared/checkpoints/images-v1/curator

sbatch --account=<account> --partition=<cpu-partition> \
  --array=0-499%60 --export=ALL \
  tutorials/image/lance_writer/submit_array.sh
```

Both checkpoint paths should be on shared storage. Dataset payload is written
only to Lance; source tars are read through fsspec and are never copied to an
intermediate object-store prefix.

After the array finishes, use Curator's Slurm retry helper with the same
`CHECKPOINT_PATH`. The `fields` format preserves the original logical shard
count when the physical retry array is sparse:

```bash
python tutorials/slurm/retry_array.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --format fields \
  --max-array-size 1001
```

Each output line contains `array_expression shard_index_offset
minimum_shard_index original_total_shards`. Resubmit only those indices while
preserving the logical values:

```bash
while read -r array offset minimum total; do
  sbatch --account=<account> --partition=<cpu-partition> \
    --array="${array}%60" \
    --export="ALL,SHARD_INDEX_OFFSET=${offset},MINIMUM_SHARD_INDEX=${minimum},TOTAL_SHARDS=${total}" \
    tutorials/image/lance_writer/submit_array.sh
done < <(
  python tutorials/slurm/retry_array.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --format fields \
    --max-array-size 1001
)
```

Repeat until the helper prints no lines, then atomically commit all
checkpointed fragments:

```bash
python -m tutorials.image.lance_writer.pipeline commit \
  --dataset-uri "${DATASET_URI}" \
  --lance-commit-path "${LANCE_COMMIT_PATH}" \
  --lance-storage-options "${LANCE_STORAGE_OPTIONS}"
```

Do not run the commit while array shards are still writing. `LanceWriter`
checkpoint records make individual packs idempotent; Curator's completion
manifests determine which logical array shards need another attempt.
