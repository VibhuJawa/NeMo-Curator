# Interleaved Image Payload Cache

This tutorial materializes image bytes for an interleaved corpus while serving
repeated images from a shared-filesystem cache instead of re-fetching them.

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

## Two preconditions

**The key is the locator, not the image.** Entries are keyed by the `source_ref`
string, so the cache pays off exactly as often as the same locator recurs. That
is true for a deduplicated store that holds one copy per unique image and points
many documents at it. It is false for a WebDataset-style layout that writes a
separate copy of the image into every sample's shard: no locator ever repeats,
and the hit rate is zero however often the image itself recurs.

**The source must be slower than the cache.** Measured on WekaFS at a mean
payload of 140 kB, 16 workers: a `put` costs 10.5 ms and a hit `get` 2.0 ms.
Against a ~0.5 s object-store GET a hit is ~250x cheaper and the cache is a
large win. Against a byte-range read into a local packed tar — 1.3 ms for the
same payload — it is a net loss, because the cache replaces a few large batched
range reads with many small whole-file reads. Cache a remote source; do not
cache a fast local one.

## Reference counts

Measured on the MINT-1T HTML corpus:

| Quantity | Value |
|----------|-------|
| Image occurrences | 1,582,193,028 |
| Unique images | 356.0M |
| Mean references per image | 4.445 |
| Images with exactly one reference | 40.2% |
| Highest observed reference count | 1,584 |

Every unique image must be fetched at least once, so at most
`1 - 356.0M / 1,582,193,028` = 77.5% of fetches can be avoided, no matter how
large the cache is. The distribution is skewed: 40.2% of images are referenced
once and can never produce a hit, while a long tail referenced hundreds of times
produces almost all of the savings.

## The 4.4x is a serial limit

A reference count of 4.4 is what a *single* worker sees. It is not what a
concurrent pipeline gets, and this is the most important caveat in this
tutorial. Measured end to end on a corpus whose static-oracle ideal hit rate was
0.4720:

| Workers | Hit rate | Source reads |
|---------|----------|--------------|
| 1 | **0.4720** — exactly the ideal | 18,553 |
| 16 | **0.0906** | 32,993 |

The mechanism is not a defect in the cache: at one worker the measured hit rate
matched the oracle to the last digit. It is that a worker touching a key for the
first time cannot see a write another in-flight worker has not made yet. With
`k` partitions in flight, an occurrence only hits if its key first appeared in a
strictly earlier wave, so the single-pass hit rate is governed by
`partitions / workers` rather than by reference multiplicity.

Three things recover it:

- **Key-affinity routing** — send every occurrence of a key to the same worker,
  so repeats are serialized behind the first fetch. This is the fix that
  restores the full multiplicity in one pass.
- **More passes.** Across passes the benefit is intact: a second run over the
  same shards hit 100% with zero source reads. Multiple epochs, retries, and
  several jobs over one corpus all realise the full reference count.
- **Many more partitions than workers**, which shrinks each wave's share of
  first touches.

Without one of these, budget for a fraction of 4.4x on the first pass.

## Sizing

Static-oracle hit rate for a full pass over MINT-1T HTML, with exact per-image
sizes and greedy admission by reference count:

| Cache size | Hit rate |
|------------|----------|
| 1 TB | 20.2% |
| 2 TB | 31.0% |
| 3 TB | 38.6% |
| 5 TB | 48.9% |
| Unbounded (51 TB) | 77.5% |

Returns are sublinear — the fifth terabyte buys far less than the first — so
size the cache against the cost of the fetches it removes, not against the
corpus. Admitting by value density `(k-1)/size` instead of by reference count
raises the 2 TB figure to 46.4%, worth roughly a 2.2x larger cache.

`PayloadCache` implements neither admission nor eviction: it stores every miss
and grows toward the full single-copy corpus. Bound it by provisioning the
directory's filesystem quota, and clear the directory between corpora. Point
`--cache-path` at a shared filesystem so every worker sees the same entries; a
node-local path only deduplicates within one node.

## Run

```bash
python tutorials/interleaved/image_payload_cache/main.py \
    --input-path /path/to/interleaved_parquet \
    --cache-path /shared/fs/payload-cache \
    --output-path /path/to/interleaved_materialized \
    --checkpoint-path /path/to/checkpoints \
    --num-cpus 64 \
    --mode error
```

The input is row-wise interleaved Parquet where image rows carry a `source_ref`
JSON locator (`path`, `member`, `byte_offset`, `byte_size`). The output is the
same rows with `binary_content` filled and `materialize_error` set on rows that
could not be resolved.

## Pipeline

```text
interleaved Parquet
    -> InterleavedParquetReader
    -> InterleavedParquetWriterStage(payload_cache_root=...)
```

There is no tutorial-local code. The writer materializes image payloads on
write and takes `payload_cache_root`, so a repeated `source_ref` is read from
the cache instead of the source.

The root is a plain path string rather than a cache object because a stage is
pickled to reach its workers. Each worker builds its own handle in `setup()`,
so the cache is worker-local and never crosses the wire.

Filter stages take the same option. `BaseInterleavedFilterStage` accepts
`payload_cache_root` and forwards it, but note that a filter materializes into
a scratch task purely to compute its keep-mask and does not emit the bytes —
so in a Reader -> Filter -> Writer pipeline both stages should be given the
same root, and the filter's fetches then warm the cache for the writer.
