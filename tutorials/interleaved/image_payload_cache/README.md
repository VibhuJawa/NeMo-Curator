# Interleaved Image Payload Cache

This tutorial materializes image bytes for an interleaved corpus while serving
repeated images from a shared-filesystem cache instead of re-fetching them.

Use it when the same image is referenced by many documents and materialization
is dominated by object-store reads rather than by CPU work.

## Problem

An interleaved document stores images by reference: an image row carries a
`source_ref` and no bytes. Materialization resolves those references into
`binary_content`. Within one task, `materialize_task_binary_content` already
deduplicates identical byte ranges, but a corpus-wide pass spreads references
to the same image across thousands of tasks and many workers, so the same
object is fetched again and again.

`PayloadCache` closes that gap. It is a content-addressed directory keyed by
`source_ref`: a hit returns the payload without touching the source, a miss
falls through to normal I/O and stores the result. Keys are hashed into a
two-level fan-out so no single directory holds millions of entries, and writes
are renamed into place so a reader never sees a partial payload. Cache faults
are never fatal — a failed read is a miss and a failed write is skipped.

## Measured motivation

Reference counts from the MINT-1T HTML corpus:

| Quantity | Value |
|----------|-------|
| Image occurrences | 1,582,193,028 |
| Unique images | 356.0M |
| Mean references per image | 4.445 |
| Images with exactly one reference | 40.2% |
| Highest observed reference count | 1,584 |

A full materialization pass therefore reads the average image 4.4 times. The
distribution is heavily skewed: 40.2% of images are referenced once and can
never produce a hit, while a long tail of images is referenced hundreds of
times and produces almost all of the savings.

That skew sets the ceiling. Every unique image must be fetched at least once,
so at most `1 - 356.0M / 1,582,193,028` = 77.5% of fetches can be avoided, no
matter how large the cache is.

## Sizing

Measured hit rate for a single pass over MINT-1T HTML:

| Cache size | Hit rate | Notes |
|------------|----------|-------|
| 1 TB | 16.3% | Holds only the hottest images |
| 2 TB | 24.5% | Roughly linear returns so far |
| 5 TB | 45.4% | Best marginal return per TB |
| Unbounded | 77.5% | Ceiling set by 356.0M unique images |

Returns are sublinear: the fifth terabyte buys far less than the first. Size
the cache against the cost of the fetches it removes, not against the corpus.
Point `--cache-path` at a shared filesystem (Lustre, Weka) so every worker sees
the same entries; a node-local path only deduplicates within one node.

The cache is not bounded or evicted by this class. Size it by provisioning the
directory's filesystem quota, and clear the directory between corpora.

## Run

```bash
python tutorials/interleaved/image_payload_cache/main.py \
    --input-path /path/to/interleaved_parquet \
    --cache-path /lustre/shared/payload-cache \
    --output-path /path/to/interleaved_materialized \
    --checkpoint-path /path/to/checkpoints \
    --num-cpus 64 \
    --mode error
```

The input is row-wise interleaved Parquet where image rows carry a `source_ref`
JSON locator (`path`, `member`, `byte_offset`, `byte_size`). That locator string
is the cache key, so two rows resolve to the same entry only when they name the
same bytes. The output is the same rows with `binary_content` filled and
`materialize_error` set on rows that could not be resolved. The writer runs with
`materialize_on_write=False` so materialization happens once, in the cached
stage, rather than a second time at write.

Reruns of the same corpus are where the cache pays off most: the first pass
populates it and later passes read almost entirely from the shared filesystem.

## Pipeline

```text
interleaved Parquet
    -> InterleavedParquetReader
    -> CachedMaterializeStage   (materialize_task_binary_content(cache=...))
    -> InterleavedParquetWriterStage
```

`CachedMaterializeStage` is defined in `main.py` and is the only tutorial-local
code: it holds a `PayloadCache` and passes it to
`materialize_task_binary_content`. Any stage that materializes payloads can
accept the same `cache=` argument.
