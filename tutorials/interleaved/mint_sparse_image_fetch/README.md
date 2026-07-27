# Sparse image fetch with consolidated fetch actors

Fetch image payloads from a Lance table for an interleaved document corpus, configured
the way that was measured fastest on the production MINT-1T table.

```text
interleaved Parquet
    -> InterleavedParquetReader
    -> LanceColumnFetchStage        (few, large actors)
    -> InterleavedParquetWriterStage
```

---

## The one thing this recipe encodes

**Sparse fetch is limited by file-opens, not bytes.**

Lance prefetches the repetition index of *every page* in a column the first time a
process opens a fragment. Reading one image from a file therefore costs almost as much
as reading fifty. Decomposed on the production table:

| Cost class | GETs/image | Share |
|---|---:|---:|
| Per-page repetition index | 2.227 | 63.3% |
| Column metadata + file footer | 0.471 | 13.4% |
| **Image bytes** | **0.822** | **23.3%** |

Only **23%** of requests carried image data. The control that settles it: files an actor
read *one* image from still averaged **9.73** repetition-index GETs — the same as files
it read fifteen images from.

**That overhead is per process.** Inside one process Lance's own cache removes it
completely (measured repeat factor 1.00). So the fix is neither a data rewrite nor a
different sort order — both were tried and measured *worse* — but simply running fewer,
longer-lived fetch actors.

---

## Measured

One node, 1,600,000 image occurrences from the production table, aggregate in-flight
requests held constant at 2,048:

| Actors/node | `io_threads` | GETs/image | img/s | Amplification |
|---:|---:|---:|---:|---:|
| 16 | 128 | 3.520 | 2,668 | 1.416 |
| **1** | **2048** | **1.065** | **4,375** | **0.997** |

**1.64× faster, 3.31× fewer requests**, and read amplification falls to 1.0 — the
consolidated configuration fetches essentially no wasted bytes.

### Why 1.64× and not 3.31×

Throughput is `request_rate ÷ GETs_per_image`. Consolidation improves the denominator
but costs the numerator: one process sustained **4,650 GET/s** against **9,390** for
sixteen. The two effects multiply:

```
3.31× fewer requests  ×  0.50× achievable request rate  =  1.64×
```

**So fewer actors is not unconditionally better**, and 1 is probably not optimal. The
optimum lies between 1 and 16 and has not been measured — the arm that would have found
it timed out. If you cannot measure your own workload, `--fetch-actors-per-node 2` is a
defensible default.

---

## The setting that must not be changed alone

`--fetch-actors-per-node` scales `io_threads` inversely, holding aggregate in-flight
requests at 2,048 per node. **Consolidating actors without raising `io_threads` narrows
the request stream and gives the entire gain back.** This was measured directly: an arm
that cut requests 32% at constant bytes ran *no faster*, because it fell off the
request-rate ceiling (552–628 GET/s against 1,045–1,278 for every other arm).

Two supporting settings matter:

- **`--metadata-cache-gib`** must hold the fragment working set. A long-lived actor whose
  cache is too small silently re-opens files and the saving disappears. 4 GiB is ample for
  ~47k fragments; the library default of 1 GiB is the minimum that worked.
- **`presence_column`** is set to `image_fetched`. Not every document reference resolves,
  and a corpus-scale run should not die on one unresolved key.

---

## Run

```bash
python tutorials/interleaved/mint_sparse_image_fetch/main.py \
    --input-path   /path/to/interleaved-parquet \
    --output-path  /path/to/output \
    --lance-uri    s3://bucket/lance_dbs/mint_1t_html_images/<snapshot>/stable_row_ids/dataset \
    --lance-version 4 \
    --storage-options '{"aws_region": "us-east-1", "endpoint": "https://..."}' \
    --fetch-actors-per-node 1 \
    --cpus-per-node 64
```

To find your own optimum, sweep `--fetch-actors-per-node` over 1, 2, 4, 8, 16 with
everything else fixed. Order the arms as a palindrome (16, 1, 4, 1, 16) so any monotone
drift cancels, and compare the *repeats of the same arm* before believing any difference
between arms.

---

## Verifying it worked

The stage emits `lance_gets_per_image` and `lance_images_per_file_open` per batch. Check
them rather than assuming consolidation took effect — if `gets_per_image` has not dropped
towards ~1, Ray placed more actors than intended and the CPU request needs adjusting.

---

## What this recipe does *not* do

- **It does not cache.** A payload cache is a separate tutorial. On this corpus its ceiling
  is bounded by reference multiplicity, and its measured hit rate collapses under
  concurrency without key-affinity routing.
- **It does not beat a dense scan.** On matched hardware a sequential scan reached
  ~982 MB/s against ~580 MB/s here. Sparse fetch is the right tool when you touch a
  *fraction* of the corpus; for a full pass, scan the table instead.
- **It has not been measured above one node.** Every figure here is single-node.
