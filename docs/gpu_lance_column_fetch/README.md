# GPU Lance column fetch for interleaved documents

This document defines the design, correctness contract, benchmark protocol, and
scale-reporting method for fetching image columns from one pinned Lance table
while streaming documents from another. The implementation uses cuDF for the
compact URL join and keeps image payloads as PyArrow data in host memory.

The immediate use case is MINT document reconstruction: every image row in an
interleaved document has a `source_ref` URL, and that URL must be joined to the
canonical image table's `url` before the `image` column is attached to the
document. This is a large exact-key join followed by sparse payload reads.

The top-level performance objective is to reach the storage-bandwidth
constraint with bounded memory. GPU lookup speed is useful only insofar as it
removes key resolution from the critical path and lets large, locality-aware
payload reads drive the remote S3-compatible store near its measured sequential
ceiling. Lustre and NVMe are optional diagnostics, not competing goals.

## Evidence status

Measurements, derived comparisons, and projections are deliberately separate.

| Item | Status | Meaning |
| --- | --- | --- |
| One-H100 64/512/1,024/4,096-row private-take sweep | Measured | Correctness-validated real-data runs; two timed repeats after one warmup |
| Persistent public `lance-ray` GPU API | Measured | Correctness-validated real-data run; two timed repeats after one warmup |
| Image-only / image+URL / full projection A/B | Measured | Exact 16,384-row manifest, persistent warm fetchers, two repeats per projection, identical payload digest |
| Ray Data cold-actor `lance-ray` API | Measured, setup-sensitive | Correct output, but the harness recreated the actor pool on every repeat; this is not persistent steady state |
| 262,144-row image-only coordinate queue | Measured | Corrected validator, one warmup and two correct real-data repeats with identical payload digest |
| 16-fragment amortization curve | Measured locality control | Physical-read curve is valid; shared-resource throughput is not a speedup denominator |
| Public `fragment.take` exploration | Measured negative | One correctness-valid warmup was decisively slower; the path is not retained for production or compatibility |
| Earlier 8-node run | Rejected legacy comparison | The run is retained as a measured row, but missing policy/sidecar identity plus placement and configuration mismatches make it ineligible for scaling ratios |
| Two-node CPU Curator baseline | Measured | Same 16,384-row manifest and full validation projection; two timed repeats after one warmup |
| Naive PyLance and public `lance-ray` DataSource comparisons | Pending | Must be rerun on the same manifest, projection, cache policy, and exclusive allocation |
| 6B, 20B, and 100B+ scenarios | Modeled | Capacity and runtime scenarios derived from measured inputs; never benchmark results |

## Dataset contract

Every run must pin both Lance versions. Resolving `latest` independently can
silently join sidecars, row IDs, and payloads from different
snapshots.

### Document Lance table

The maximum-coverage input is a caller-authorized, immutable document snapshot:

```text
$DOCUMENT_LANCE_URI@$DOCUMENT_LANCE_VERSION
```

The document schema supplies at least `sample_id`, `position`, `modality`, and
`source_ref`. Rows with `modality = 'image'` use `source_ref` as the exact image
URL. Two alternative pinned policies are available for different downstream
quality contracts:

| Policy | Pinned version | Scale recorded in the dataset inventory |
| --- | ---: | ---: |
| `retain_all` | 1 | 6.960B rows, 883,755,778 documents, 1,719 fragments |
| `drop_missing_images` | 3 | 5.992B rows, 1,482 fragments |
| `drop_incomplete_documents` | 1 | 4.153B rows, 1,031 fragments |

Choose one policy before a run and record its URI, version, schema, fragment
inventory, and task-manifest digest.

### Canonical image Lance table

The current benchmark pins an immutable image snapshot:

```text
$IMAGE_LANCE_URI@$IMAGE_LANCE_VERSION
```

The exact key is `url`. The fetched projection is normally `image`, with `md5`,
`width`, and `height` included for validation. The table must expose stable row
IDs. The sanitized run identity remains the authority for the version actually
tested; private inventory services are not referenced by checked-in evidence.

### Sidecar indexes

The sidecars are immutable products of the same pinned image snapshot:

| Path | Required columns | Partition contract |
| --- | --- | --- |
| Replicated cuDF lookup | `url`, `stable_row_id: uint64` | Sorted exact-key Parquet segments with no null or duplicate URLs |
| RAPIDS-MPF shuffle join | `url`, `stable_row_id: uint64` | Hash-sharded with the same hash function and partition count used by the shuffle |

Sidecar identity, source URI/version, schema, row count, partitioning
parameters, and content digest belong in the run identity. The MPF sidecar's
stable IDs must be the global ordinals accepted by private `_take_rows` for the
pinned image snapshot; no derived physical row-address column is stored.

Every GPU reader requires a caller-pinned canonical sidecar manifest and its
SHA-256. The manifest binds the Lance URI/version and exact fragment-order
digest to every Parquet file's partition, ordinal, row count, byte size, and
SHA-256. The v2 contract also scans the pinned Lance key stream and proves the
full key-to-stable-ordinal identity after sorting the sidecar by stable ID;
coverage alone is insufficient. Construction proves exact duplicate-free
stable-ID coverage of `[0, dataset.count_rows())`; actor setup revalidates the
manifest and file identities before building a GPU index or issuing payload
reads. Older v1 manifests must be rebuilt. The
deterministic builder is
[`build_gpu_lance_sidecar_manifest.py`](../../nemo_curator/stages/interleaved/build_gpu_lance_sidecar_manifest.py).
Contract publication is an offline node-memory operation: it loads the compact
key/ID sidecar and sorts by stable ID, which is valid under the stated
single-node index-fit assumption and never moves image payload bytes.

## Design goals

1. Preserve document row order and duplicate image references exactly.
2. Keep the public data boundary PyArrow-based. cuDF receives only keys, row
   IDs, and routing coordinates; image bytes remain in Arrow host memory.
3. Stream bounded batches instead of materializing the document or payload
   tables for the full corpus.
4. Make sparse-call reduction an explicit optimization goal and reported
   result, not an incidental implementation detail.
5. Demonstrate a storage-throughput plateau with CPU and GPU headroom instead
   of declaring success from a small latency benchmark.
6. Fail closed on stale snapshots, ambiguous row-ID layouts, missing keys under
   the strict policy, duplicates, and incomplete payload fetches.
7. Compare like-for-like naive Lance, CPU Curator, `lance-ray`, and Ray Data
   paths before claiming an algorithmic speedup.

### Sparse-read success criteria

Every benchmark must report all three layers of sparsity:

- API work: private `_take_rows` calls and unique image rows per call. Record
  `fragment_take_calls` only for the exploratory public path.
- Physical work: Lance `read_iops` counter per logical image.
- Byte work: Lance read bytes divided by returned payload bytes.

The production optimization target is the measured best medium-sized private
`_take_rows` batches per bounded streaming window, plus fewer physical I/O
operations per image. The IDs are deduplicated and sorted before each call so
pinned PyLance can group work internally. The current remote-data default is
1,024 IDs with at most 16 pending calls: a single 16,384-ID call did not finish
within eight minutes, and 4,096-ID calls regressed. A change does not meet this
goal if throughput rises only by increasing unbounded concurrency or memory. Correctness, peak
host/device memory, read amplification, and the complete repeat spread must
remain visible beside throughput.

The baseline averages about **26 KiB per physical read**
(`4,256,011,264.5 / 159,920.5` bytes) while issuing 9.76 reads per image. The
acceptance target is multi-MiB effective reads, a concurrency plateau, and
70-85% of a separately measured sequential ceiling for the same storage tier
and payload projection. Larger queues must stop improving throughput before a
run is called storage-saturated.

## Streaming architecture

### Arrow data boundary

The stages accept pinned `LanceReadTask` manifests or Arrow-backed
`InterleavedBatch` objects and return Arrow-backed `InterleavedBatch` objects.
The compact join path is:

```text
Arrow document batches
  -> image URL and origin-coordinate columns
  -> temporary cuDF exact-key merge
  -> stable row IDs
  -> bounded sorted private Dataset._take_rows batches
  -> Arrow payload tables
  -> Arrow document reconstruction in original order
```

This keeps downstream code consistent with PyArrow and prevents multi-megabyte
image payloads from consuming GPU or shuffle memory.

### Replicated cuDF path

`GpuLanceColumnFetchStage` is the simple path when the sorted URL sidecar fits
the available node and device-memory plan:

1. Optionally stage one immutable copy of the Parquet sidecars on each node.
2. Load a persistent cuDF URL-to-stable-row-ID index in each GPU actor.
3. Deduplicate each bounded Arrow key window and perform the exact lookup on
   the GPU.
4. Sort resolved row IDs once, issue measured medium-sized private
   `Dataset._take_rows` calls with bounded pending work, and rebuild the
   original task and row order.

There is no distributed shuffle in this path. Replication avoids network
coordination, while the payload backend and result stay Arrow-based. Larger
task windows improve the opportunity to group reads but retain more input and
payload state.

### RAPIDS-MPF two-shuffle path

`GpuLanceShuffleFetchStage` removes full-index replication and preserves
streaming when the URL index must be distributed:

1. Each origin rank scans only image URLs and document row addresses from its
   pinned document manifests.
2. The first RAPIDS-MPF shuffle hashes compact request records by URL. The rank
   owning that hash partition performs a local cuDF merge with its sidecar
   shard.
3. A second rank-directed shuffle returns only origin coordinates and stable
   row IDs to the origin rank.
4. The origin rank processes a bounded manifest window, rescans those document
   fragments, and deduplicates and sorts resolved stable image row IDs.
5. The origin issues bounded private `Dataset._take_rows` calls for the window.
   Arrow payload columns are checked against the returned keys and emitted in
   original document order.

Only compact key and coordinate records cross the network. Image payloads are
read directly by the rank producing the output. This follows RAPIDS-MPF's
[process-per-GPU streaming shuffle architecture](https://docs.rapids.ai/api/rapidsmpf/stable/background/shuffle-architecture/):
chunks can be inserted and extracted incrementally, and spillable buffers
allow an out-of-core execution plan.

### Pinned private payload hot path

The production reader intentionally pins private `Dataset._take_rows`; it does
not contain public-API compatibility dispatch or a fallback to
`fragment.take`. This is an explicit performance dependency recorded in the
run identity and guarded by integration tests.

The source audit used installed PyLance 9.0.0-beta.11 at commit
[`0b82051`](https://github.com/lance-format/lance/commit/0b82051) and compared
it with upstream beta.18 at commit
[`d581bb9`](https://github.com/lance-format/lance/commit/d581bb9). The audited
[`take.rs` comparison](https://github.com/lance-format/lance/compare/0b82051...d581bb9)
had no changes to that path. It groups sorted row addresses by fragment and
buffers object-store reads up to the configured I/O parallelism, which is
exactly the locality work this pipeline should reuse instead of reimplementing
with Python `fragment.take` calls.

The image column is `large_binary` with no blob metadata. PyLance's private
`read_blobs` API rejects it as "not a blob column", so `read_blobs` is not an
alternative hot path for this dataset.

The pinned Rust path explains why projection and locality are separate levers.
`_take_rows` maps stable IDs, groups addresses by fragment, and sends
non-consecutive offsets through fragment point-take. The decoder converts only
consecutive indices into ranges; the file scheduler can merge ranges submitted
together when their gap is within the 64-KiB cloud coalescing threshold.
`IOTracker.read_iops` increments once per completed physical read, so the
reported IOPS are payload-path reads rather than GPU lookup calls. Production
IOTracker does not retain object path or range details; its request recorder is
compiled only with PyLance's `test-util` feature. Exact path, range, and column
attribution therefore requires a controlled projection sweep or a scoped
instrumented build of the pinned commit.

A separate 1,000-image/100-fragment locality control makes projection pruning
the highest-confidence immediate optimization. It is not a final speedup
denominator because its key distribution and concurrency differ from the real
16,384-URL workload:

| Private `_take_rows` projection | Physical reads | Read MiB | Images/s |
| --- | ---: | ---: | ---: |
| `image` | 101 | 213.29 | 90.13 |
| `url`, `image_size_bytes`, `image` | 301 | 213.73 | 61.34 |

The two tiny columns added less than 0.5 MiB but tripled physical reads. The
production timed projection therefore reads image bytes only and reconstructs
fan-out from the already resolved stable IDs; URL/MD5/dimensions belong in an
explicit validation projection. The summarized control predates the sanitized
artifact allowlist and its host-local raw file is not a checked-in evidence
source.

The initial uniform-occupancy hypothesis predicted about 14,224 touched
fragments and 1.15 rows per fragment for the 16,377-row workload. The new exact
global-ordinal planner disproved that approximation: these IDs map to 10,591
fragments, or 1.55 requested rows per touched fragment. The distribution is
not uniform. In the image-only projection, the 44,465.5 median physical reads
equal 4.20 reads per touched fragment; path/range attribution still comes from
the pinned trace experiment below.

### Bounded pinned physical-I/O trace

A bounded diagnostic used exact Lance commit `0b82051`, PyLance `9.0.0b11`,
256 deterministic sorted stable IDs, the `image/md5/width/height` projection,
16 I/O threads, and three alternating baseline/trace pairs. It submitted no
Slurm job. The existing pinned scheduler event records the post-coalescing
object path and byte range, so no custom Rust build was required.

| Metric | Measured result |
| --- | ---: |
| Data objects | 147 |
| Physical reads | 2,881 |
| Physical bytes / Arrow bytes | 44,967,602 / 27,777,834 |
| Reads/image | 11.254 |
| Read amplification | 1.619x |
| IOTracker reconciliation | 100% calls and bytes |
| Read-size p50 / p90 / p95 / max | 4,100 / 16,378 / 65,536 / 2,247,126 B |

Exactly 90.0% of calls were at most 16 KiB and carried only 18.7% of bytes;
0.76% of calls exceeded 256 KiB and carried 39.3% of bytes. Dataset open was
separate: four reads and 41,635,290 bytes per process before payload counters
were reset. On the four hottest files, 98.36% of traced bytes mapped to page
buffers and 95.66% mapped to the image column; 171,998 bytes overlapped adjacent
unprojected source-member pages.

An offline cross-request merge model is explicitly not a runtime result. A
4-KiB gap removed 290 calls (10.07%) while reducing modeled bytes 2.23%; 16 KiB
removed 19.85% of calls for 2.42% more bytes, and 64 KiB removed 22.60% for
9.80% more bytes. The 4-KiB point is therefore the next coalescing experiment.
Paired trace overhead was +0.598%, +0.561%, and -7.921%; the negative pair is
remote/cache variance, not a trace speedup.

Production IOTracker remains insufficient for attribution because it exposes
only aggregate counters; test-util request records also omit ranges at this
commit. The table above is the checked-in summary; host-local trace paths are
intentionally omitted.

### Queue, byte, and memory controls

Bounded row counts alone are not a sufficient memory contract for variable-size
URLs and images. The control surface is:

| Control | Scope | Contract |
| --- | --- | --- |
| Curator stage batch size / benchmark `coalesce_tasks` | Replicated input | Bounds task count; increasing it creates more fragment-locality opportunity |
| `coalesce_target_bytes` with `estimated_row_bytes` | Public `lance-ray` Ray Data API | Soft byte target converted to Ray's row-based `batch_size` |
| `max_lookup_bytes` | Public `lance-ray` GPU actor | Hard cap for each Arrow-to-GPU lookup window |
| `scan_batch_size` | MPF document scans | Bounds URL and document rescan batches |
| `fetch_task_window` | MPF origin rank | Bounds retained manifests; larger windows can reduce sparse reads at the cost of host payload memory |
| `fetch_window_bytes` with `estimated_payload_bytes_per_row` | MPF rank window | No-deadline estimated payload cap; fetched bytes are checked again with duplicate fan-out and accepted profiles are exactly `256MiB`, `1GiB`, and `4GiB` |
| Coalesced unique-ID window / current `fetch_batch_size` | Private payload reader | Current measured remote default is 1,024 IDs/take with 16 pending takes; both smaller and larger batches remain benchmark dimensions |
| Pinned PyLance object-store I/O parallelism | Private payload reader | Bounds buffered reads inside the Rust take path; it is not a substitute for a larger locality window |
| `rmm_pool_size` and `spill_memory_limit` | MPF actor | Bound or spill device working memory |

RAPIDS-MPF uses bounded channels with backpressure in its
[streaming engine](https://docs.rapids.ai/api/rapidsmpf/stable/background/streaming-engine/)
and exposes [memory and spill configuration](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/).
Production runs must additionally record peak queued Arrow bytes, peak host
RSS, peak GPU bytes, spill bytes, and window occupancy. A hard payload-window
byte cap cannot be known before I/O from a two-column URL/ID sidecar because
image sizes vary. The MPF actor rejects an estimated oversize window before
I/O and rejects fetched-byte fan-out before building document outputs, while
reporting actual versus estimated bytes. A pre-I/O hard heterogeneous-payload
bound remains follow-up work requiring immutable size metadata.

## Stable global-ordinal invariant

There is one supported layout: sidecar stable IDs are global physical ordinals
in pinned manifest-fragment order and are passed to private
`Dataset._take_rows`. There is no layout switch, auto-detection, public
`fragment.take` compatibility path, or fallback. The invariant is validated at
startup so the reader cannot silently use a stale or differently ordered
snapshot.

Construction and fetch fail before returning data unless all conditions hold:

1. The image URI and version are pinned, the table exposes stable row IDs, and
   at least one fragment exists.
2. Every public fragment ID is a unique nonnegative integer.
3. Every fragment exposes a positive `physical_rows`; public and metadata row
   counts agree.
4. `num_deletions == 0`, and neither public nor metadata deletion files exist.
5. The prefix sum of all fragment physical rows equals both the metadata sum
   and `dataset.count_rows()`. This proves complete physical-row coverage.
6. The sidecar is tied to that exact snapshot, uses `uint64`, contains exactly
   `dataset.count_rows()` rows, and spans `[0, total_physical_rows)`.
7. Every requested ID lies in `[0, total_physical_rows)` and maps to a real
   fragment interval.
8. Every private take returns the requested number of rows. Returned keys must
   belong to the request, match the sidecar when payload-key validation is
   enabled, and be unique.

Any violation must fail closed. The MPF path carries only the pinned stable ID,
sorts and deduplicates those IDs per fetch window, and passes them directly to
private `_take_rows`.

## Benchmark methodology

### Comparable-run contract

A speedup row is valid only when both arms use the same:

- pinned image URI/version and sidecar identity;
- query-manifest digest, row count, row order, and key distribution;
- payload projection and correctness checks;
- task/window geometry, sparse-read mode, concurrency, cache and validation
  policy, warmup count, and at least two measured repeats;
- package versions and code identity when recorded; and
- node type, storage endpoint, and exclusive-node allocation; and
- setup-included or setup-excluded timing definition.

The harness separates cold setup from warm processing, rotates arm order with a
recorded seed, runs warmups before measured repeats, and reports median plus the
full repeat range. Multi-rank elapsed time is the maximum rank wall time;
payload bytes and I/O counters are summed across ranks only after the artifact
count exactly matches the expected rank count, rank IDs are unique and
contiguous, and every available Slurm identity agrees. Labels alone are not
rank evidence. Physical nodes must be exclusive to one Ray cluster because
CPU/GPU resource requests alone do not prevent interference.

Short parameter sweeps are packed inside one exclusive allocation. Do not use
one Slurm array element per sub-ten-minute point: loop over the points in one
job, preserve a result and failure record per point, and release the allocation
once the suite finishes. This avoids scheduler churn and defunct-job detector
alerts while retaining independent measurements.

### Storage-saturation protocol

Strong-scaling latency and storage saturation are different experiments. A
fixed global workload that is divided among more nodes can show lower latency
without placing enough steady-state work on any node to saturate storage. The
saturation sweep therefore holds the per-node workload constant as nodes are
added.

| Geometry | Required configuration |
| --- | --- |
| GPU per node | Eight persistent actors, one per H100; keep the cuDF index resident across all waves |
| Work per node | 64 left tables concurrently per wave; 512 tables and 131,072 images total per 8-H100 node |
| Waves | Eight 2,048-image calls per actor for the primary run; four waves remain a sensitivity |
| Eight-node total | 1,048,576 images, preserving the same 131,072-image workload on every node |
| CPU sweep | 1, 2, 4, and 8 persistent actors per node over the same total per-node input; bound aggregate I/O rather than multiplying it with actor count |

The currently available jobs that globally split 64 left tables across all
ranks do not satisfy this geometry and are not storage-saturation proof.

Run the geometry independently on each storage axis:

| Storage axis | Purpose |
| --- | --- |
| S3-compatible object storage | Primary end-to-end production path, including remote request latency and throttling |
| Lustre | Optional bounded shared-filesystem diagnostic with the same payload projection and logical image set |
| Node-local NVMe | Local-I/O upper bound after explicit staging; staging time remains setup, not steady state |

Remote object storage is the optimization target. The bounded Lustre mirror
attempt transferred zero objects before it was stopped, so no Lustre A/B result
exists and no full-table mirror is planned. It must not delay the remote
projection, locality, concurrency, or saturation experiments.

Measure a large sequential payload-read ceiling for each tier before the join
sweep. Keep schema, compression, projection, and bytes/image comparable. Report
setup separately: Ray startup, sidecar staging, cuDF load/hash construction,
Lance open, and warmup. For steady state, report every repeat's wall time,
payload rate, physical reads and bytes/read, private-call size, queue occupancy,
spill, CPU utilization, GPU utilization/memory, network/storage counters, and
peak RSS.

Every saturation run publishes one terminal `eligibility.json`. Eligibility
requires the benchmark report, allocation identity, repeat set, additive
I/O/ratio reconciliation, and allocation-wide telemetry validation to pass.
`telemetry_validation.json` describes telemetry only; its `passed` status never
implies that the benchmark is eligible. Live Slurm launches must also provide
`--minimum-remaining-slurm-seconds` and a numeric allocation end epoch before
the runner creates output, starts telemetry, or starts Ray. The caller-set
floor must cover setup plus all requested repeats.

A production path is storage-saturated only when all of the following hold:

1. Sustained payload throughput reaches at least 70% of the measured sequential
   ceiling for that storage tier, with 85% as the target.
2. Effective physical reads move from the 26-KiB baseline into multi-MiB sizes
   without incorrect rows or unbounded read amplification.
3. Increasing actor count, I/O parallelism, private-call window, or queue depth
   no longer improves throughput beyond the observed repeat spread.
4. CPU and GPU telemetry show headroom while storage or network counters explain
   the plateau.
5. Setup remains excluded from steady-state throughput but is reported so index
   replication and staging costs cannot disappear from end-to-end planning.

### Benchmark arms

| Arm | Question answered |
| --- | --- |
| `naive_pylance_scalar` | Cost of one scalar-index scanner and one sparse stable-ID take per unique key |
| `cpu_lance_column_fetch_stage` | Benefit of batched CPU scalar lookup, deduplication, sorting, and shared fetch logic |
| `gpu_lance_column_fetch_stage` | Benefit of the persistent replicated cuDF index with pinned private `_take_rows` payload reads |
| `lance_ray_datasource` | Ray Data baseline using public `lance_ray.read_lance` with batched `IN` filters |
| `ray_data_persistent_gpu_actor` | Ray Data `map_batches` baseline with persistent one-GPU actors and the cuDF index |
| `lance_ray_gpu_fetcher` | Public API using Arrow batches, a persistent cuDF sidecar, byte-bounded lookup windows, and private `_take_rows` payload reads |
| `lance_ray_gpu_actor` | The same public API inside a Ray Data persistent actor pool |
| `gpu_lance_shuffle_fetch` | Distributed RAPIDS-MPF URL join followed by origin-local private `_take_rows` payload reads |

All listed arms except the end-to-end MPF execution are implemented in the
benchmark harness. The MPF two-shuffle stage and actor are implemented and
covered by focused tests, but still require a matched remote-data run.

### Correctness and reported metrics

Every repeat must verify row count, original order, no unexpected missing
payload, and a stable payload digest. Full validation projections also verify
stored MD5 and decoded image dimensions. A failed repeat is excluded from
comparisons and reported as a failure, not silently dropped.

Report setup time, wall time, lookup and payload-fetch time, images/s, payload
MiB/s, payload bytes, Lance read bytes, Lance `read_iops`, read amplification,
lookup/fetch call counts, private `_take_rows` calls, IDs per private call,
coalesced duplicate requests, sparse calls avoided, peak memory, and spill
metrics. The MPF path additionally reports logical duplicate fan-out, average
physical read size, physical read bandwidth, and actual-to-estimated payload
bytes. Fragment counts and `fragment_take_calls` are reported only for the
exploratory public path.

## Measured results

### Matched speedup table

Final speedup is `candidate median images/s / 64-row anchor median images/s`.
Measured rows use the same pinned table and 16,384-row manifest. The
projection-pruned row intentionally changes the timed projection but proves the
same payload digest; other rows use the full validation projection.

| Arm | Median steady wall (s) | Images/s | Speedup vs 64-row anchor | Status |
| --- | ---: | ---: | ---: | --- |
| Curator GPU, 64 IDs/take | 89.3353 | 183.5487 | 1.0000x | Measured anchor |
| Curator GPU, 1,024 IDs/take | 66.3122 | 247.3493 | **1.3476x** | Measured tuning result |
| `lance_ray_gpu_fetcher`, image-only | 52.4118 | 312.6114 | **1.7031x** | Measured production projection |
| Curator GPU, 262,144-row image-only queue | 330.0664 | 794.3123 | **4.3275x** | Measured locality leader; two correct repeats |
| `lance_ray_gpu_fetcher`, full validation | 65.7135 | 249.6370 | **1.3601x** | Matched projection-control session |
| `lance_ray_gpu_actor`, 1,024 IDs/take | 76.8734 | 213.2125 | 1.1616x | Measured actor process span; end-to-end wall was 117.8131 s because the pool was rebuilt |
| CPU Curator, two nodes, 64 IDs/take | 911.8877 | 19.1044 | 0.1041x | Measured; 14.44-23.77 images/s spread |
| `naive_pylance_scalar` | Pending | Pending | Pending | Exclusive matched rerun required |
| `lance_ray_datasource` | Pending | Pending | Pending | Ray Data rerun required |
| `ray_data_persistent_gpu_actor` | Pending | Pending | Pending | Ray Data rerun required |
| `gpu_lance_shuffle_fetch` | Pending | Pending | Pending | Two-shuffle private-read run required |

### One H100, real MINT URLs, stable global ordinals

This anchor used image table version 4, 16,384 rows arranged as 64 Arrow tasks
of 256 rows, a 64-task coalescing window, `lookup_batch_size=2000`,
`fetch_batch_size=64`, and `io_threads=16`. There were 16,377 unique URLs. One
warmup preceded two measured repeats. Cold index setup was measured separately
at 32.5656 seconds and is excluded from the warm wall times below.

| Metric | Repeat 1 | Repeat 2 | Median |
| --- | ---: | ---: | ---: |
| Wall time (s) | 91.8877 | 86.7830 | 89.3353 |
| Payload fetch time (s) | 88.9252 | 83.6757 | 86.3005 |
| GPU lookup time (s) | 0.0397 | 0.0375 | 0.0386 |
| Images/s | 178.3047 | 188.7928 | **183.5487** |
| Payload MiB/s | 20.4383 | 21.6405 | **21.0394** |
| Lance `read_iops` counter | 163,915 | 155,926 | 159,920.5 |
| Lance I/O operations per image | 10.0046 | 9.5170 | **9.7608** |
| Average KiB per physical read | 25.2179 | 26.8007 | **25.9895** |
| Lance read-byte amplification | 2.1494x | 2.1730x | **2.1612x** |

All 16,384 rows were present and correct in both repeats. MD5 and decoded
dimensions were checked for every image, row order matched the manifest, and
both repeats produced the same output digest. The complete repeat records are
in the sanitized [measured-evidence manifest](../../benchmarking/results/gpu_lance_column_fetch/measured_evidence_v1.json).

The lookup is about 0.04 seconds while payload fetch spans 83.68-88.93 seconds.
The observed bottleneck is therefore sparse payload I/O, not the cuDF join. The
9.76 physical I/O operations per image, approximately 26 KiB per read, and
2.16x read amplification are the primary numbers the larger private-call
window must improve toward multi-MiB reads. With 16,377 unique URLs and
`fetch_batch_size=64`, this configuration partitions the sorted IDs into 256
private `_take_rows` calls. The artifact's top-level `fetch_calls=1` is one
stage window, not one private take and not one physical read. Current telemetry
reports stage windows, private takes, locality-strategy calls, and IOTracker
physical reads separately.

### Private-take sweep and public API

All rows below use the same real 16,384-image manifest and full validation
projection. Results are two-repeat medians; the
[measured-evidence manifest](../../benchmarking/results/gpu_lance_column_fetch/measured_evidence_v1.json)
retains both repeats rather than selecting the best one.

| IDs/private take; max pending | Wall s | Images/s | Physical reads/image | KiB/read | Read amplification |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64; 16 | 89.3353 | 183.5487 | 9.7608 | 25.99 | 2.1612x |
| 512; 32 | 67.7538 | 242.1095 | 9.4226 | 26.72 | 2.1453x |
| **1,024; 16** | **66.3122** | **247.3493** | **8.6288** | **28.61** | **2.1035x** |
| 4,096; 4 | 94.9103 | 172.6265 | 9.4178 | 26.56 | 2.1310x |

The 1,024/16 configuration is the measured default. It delivered
239.09-255.61 images/s and 27.41-29.30 payload MiB/s. A single 16,384-ID call
did not complete within eight minutes and was stopped; it is a negative
observation, not a timed result. Larger API calls are not automatically larger
physical reads because the random IDs still fan out across fragments and
column pages.

The persistent public `lance_ray_gpu_fetcher` arm used the same 1,024/16
configuration and completed at 257.69 and 265.82 images/s, a 261.76 median and
1.4261x the 64-row anchor. Its 16 private takes avoided 16,361 potential sparse
API calls, but it still averaged 8.99 physical reads/image and 27.91 KiB/read.
This earlier independent full-validation session is not storage-bandwidth
saturation. Its configuration, repeat spread, and correctness checks are in the
[measured-evidence manifest](../../benchmarking/results/gpu_lance_column_fetch/measured_evidence_v1.json).

### Matched projection A/B

The production implementation now associates fetched Arrow rows with the
requested stable ordinals out of band, so it can restore order and duplicate
fan-out without rereading the URL. The exact 16,384-row manifest was run with
one warmup and two measured repeats per projection, using the same 1,024/16
payload concurrency:

| Timed projection | Images/s spread | Median images/s | Reads/image | KiB/read | Amplification |
| --- | ---: | ---: | ---: | ---: | ---: |
| `image` | 310.81-314.41 | **312.61** | **2.7140** | **69.06** | **1.5968x** |
| `image`, validation `url` | 288.46-288.88 | 288.67 | 5.0572 | 44.44 | 1.9147x |
| `image`, `url`, `md5`, `width`, `height` | 240.81-258.47 | 249.64 | 8.7884 | 28.09 | 2.1033x |

Every repeat was correct and all six produced payload digest
`89331cb6e72819da40e3b1b74aab8d308fa0ac493c302a61f47924a4e7b166bf`.
The full projection also checked all stored MD5 values and decoded dimensions.
Image-only removed 69.1% of full-projection physical calls and was 1.252x
faster in the matched session; it is 1.703x the original 64-row anchor. URL
alone added about 38,392 median reads, while MD5/dimensions added another
61,132. All projection repeats are retained in the sanitized
[measured-evidence manifest](../../benchmarking/results/gpu_lance_column_fetch/measured_evidence_v1.json).

The Ray Data `lance_ray_gpu_actor` control produced the same digest and a
209.01-217.41 images/s actor-process spread. Its end-to-end repeats were
114.81-120.82 seconds because this harness run recreated and warmed the actor
pool each time. It is a cold-actor baseline, not the pending persistent
eight-actor saturation result.

### Persistent eight-H100 attempt: incomplete diagnostic

One persistent-pool saturation attempt used 512 left tables, 131,072 images,
eight actors on one eight-H100 node, and eight scheduling waves. All eight
actors became ready and the first repeat returned all rows in manifest order
with no missing payloads. The allocation did not have enough remaining time
for the required second repeat, so it was intentionally stopped before another
full remote read cycle. The benchmark therefore remains top-level `running`
with one of two repeats and is **ineligible for speedup, scaling, or saturation
claims**.

The single completed repeat is retained only as a diagnostic record. Its steady
actor span was 687.7312 seconds (190.59 images/s); the complete repeat wall was
708.2808 seconds (185.06 images/s). Additive Lance counters recorded
1,236,821 reads and 38,855,528,480 physical bytes for 16,310,323,982 useful
payload bytes. Those top-level counters arithmetically imply 9.436 reads/image,
30.68 KiB/read, and 2.382x amplification, but the artifact's own ratio fields
claim 1,965.61 KiB/read and 154.929x amplification.
The 128 private takes averaged 1,023.85 rows each and avoided 130,925 scalar
sparse calls. This is IOPS/latency behavior, not bandwidth saturation, and it
shows that splitting the global queue into 2,048-row actor windows discarded
most of the 262K queue's locality benefit.

The partial artifact predates a reducer fix. Its byte/read counters and derived
fields do not reconcile, so the new gate does not bless either side as benchmark
evidence. Its terminal manifest records `telemetry_validation_status=passed`
but `status=ineligible` because the benchmark is running/ready, has only one of
two repeats, has contradictory average-read-size, amplification, and
rows-per-call metrics, lacks the later complete policy identity, and predates
the allocation-time guard. The sanitized checked-in evidence is the
[`terminal eligibility verdict`](../../benchmarking/results/gpu_lance_column_fetch/saturation/402310_1n_8w_lance_ray_gpu_actor_contract_v3_persistent/eligibility.json);
raw logs, run identity, telemetry streams, and partial benchmark are not part of
the publishable evidence allowlist.

### Two CPU nodes

The CPU Curator baseline split the same 64 tables across two exclusive CPU
nodes, using `fetch_batch_size=64` and 64 configured I/O threads per rank.
Maximum-rank setup was 3,941.58 seconds, almost entirely Lance index prewarm.
After a correctness-valid warmup, global repeat walls were 1,134.37 and 689.40
seconds: 14.44 and 23.77 images/s, with a 19.10 median. The full-projection
64-row one-H100 anchor is therefore 9.61x faster by median throughput, while the
1,024-row GPU tuning result is 12.95x faster. The large repeat spread means
storage/cache state dominates this CPU configuration; neither the faster nor
slower repeat should be quoted alone. See the
[aggregate report](../../benchmarking/results/gpu_lance_column_fetch/scaling_report_cpu_2node.json).

### Large coordinate queue: measured locality leader

A corrected remote rerun accumulated 262,144 real unique URLs in one Arrow
queue, resolved and sorted their stable IDs with cuDF, then retained the same
1,024-row private takes and 16-pending bound. One warmup preceded two measured
repeats. End-to-end stage walls were 333.7031 and 326.4298 seconds, or 785.56
and 803.06 images/s; the median is **794.31 images/s**. Lance fetch spans were
234.6551 and 232.5753 seconds, corresponding to 1,117.15-1,127.14 images/s.
The run returned 35,932,361,839 useful payload bytes per repeat at
102.69-104.98 MiB/s end to end and 146.03-147.34 MiB/s during fetch.

| Metric | 16K queue | 262K queue | Change |
| --- | ---: | ---: | ---: |
| End-to-end images/s | 247.349 | **794.312** | **3.21x** |
| Physical reads/image | 8.6288 | **2.0180** | **76.6% lower** |
| Read amplification | 2.1035x | **1.2411x** | **41.0% lower** |
| Private takes | 16 | 256 | Same 1,024 rows/take |
| Average physical read | 28.61 KiB | **82.33 KiB** | 2.88x larger |
| Peak host RSS | 10.70 GB | 105.24-109.34 GiB | Explicit queue cost |

Both repeats returned all 262,144 rows in manifest order with no missing
payload and identical payload digest
`c83e219257a6b47f8adc2aea488f1c123c23f49ca4c15588049eb662f8da8ac3`.
One image exceeded Pillow's decompression-bomb safety threshold in both
repeats and was recorded as a safety skip, not a mismatch. The earlier full
validation run independently checked all 262,144 MD5 values and directly
verified that row's URL, 18,217 by 12,138 dimensions, and MD5. The sanitized
[measured-evidence manifest](../../benchmarking/results/gpu_lance_column_fetch/measured_evidence_v1.json)
preserves both corrected repeats and their source artifact digest.

Even this leader averages only 82.3 KiB per physical read. It is substantially
less sparse, but it does not yet meet the multi-MiB read-size or 70-85% remote
sequential-ceiling acceptance criteria.

The queue result is the strongest evidence for the user's sparse-call
hypothesis: the win came from a wider coordinate opportunity window, not a
bigger private take. The production queue should therefore drop URL strings
after resolution, retain fixed-width stable-ID and origin coordinates plus an
inverse map, fetch every unique ID once, and scatter payloads back to duplicate
document origins. Arrow/Ray owns accumulation and backpressure; cuDF owns
factorization, lookup, deduplication, and sorting.

A separate one-million-URL cuDF queue probe resolved one million distinct real
URLs and stable IDs correctly. Persistent index setup took 16.13 seconds and
49,926,897,664 GPU bytes. After setup, URL deduplication, mapping, stable-ID
sorting, ID deduplication, and Arrow export took 0.0513, 0.1250, 0.1291,
0.0077, and 0.0062 seconds respectively. The 108.4-MB Arrow input became only
12 MB of fixed-width output coordinates, with 34.3 GB HBM still free. This
independently confirms that the coordinate transform is not the throughput
bottleneck; payload locality and object-store reads are.

### Fragment fixed-cost amortization

A 16-fragment nested-contiguous control kept 16 private calls in flight while
increasing useful rows per fragment. Shared-node and network contention makes
its images/s unsuitable as a speedup denominator, but its physical-call curve
is direct locality evidence:

| Rows/fragment | Total images | Physical reads | Reads/image |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 80 | 5.0000 |
| 10 | 160 | 81 | 0.5063 |
| 25 | 400 | 81 | 0.2025 |
| 100 | 1,600 | 89 | 0.0556 |
| 1,000 | 16,000 | 264 | **0.0165** |

A 100x increase in useful rows raised physical calls only from 80 to 89. This
supports persistent fragment/range buckets and threshold-based flushing, while
the 1,000-row point demonstrates the larger limit: two correct repeats measured
711.95-712.96 images/s, 134.94-135.14 payload MiB/s, 11.65 MB/read, and 0.967x
physical/Arrow amplification. This is a contiguous fragment-local control, not
a random-URL speedup denominator or proof of the remote sequential ceiling. The
500-row point is invalid because a live source edit caused setup failure; the
old harness masked that error with exit zero, which is now fixed.
The scheduler incident and submission-shape correction are recorded in
[`SCHEDULER_INCIDENT_402755.md`](SCHEDULER_INCIDENT_402755.md).

### Public fragment path: measured negative

The exploratory `fragment_ordinal` path used public `fragment.take` on the same
16,384-image workload. Its correctness-valid warmup took **450.2234 seconds**,
or **36.3908 images/s**. All 16,384 images passed the correctness checks.

Measured repeats were stopped because the regression was already decisive.
Because this has only one warmup observation, it is not a speedup denominator
and does not satisfy the final benchmark protocol. It does answer the design
question: Python-managed public fragment reads are not retained as a production
path, fallback, or compatibility mode. The later sweep showed that one giant
private call is also not the answer; medium private takes inside a much wider
coordinate queue are the measured winner.

### Earlier 8-node observation: rejected comparison

An earlier non-exclusive 8-node/8-GPU run observed a 13.5019-second median wall
time. Dividing the one-H100 89.3353-second median by that value gives a raw
**6.6165x** elapsed-time ratio. This is not a final strong-scaling result and
not an algorithmic speedup: the allocation was not the required exclusive
placement, and its per-rank task window and I/O-thread configuration differ
from the one-H100 anchor. The sanitized
[scaling report](../../benchmarking/results/gpu_lance_column_fetch/scaling_report_gpu.json)
correctly rejects these runs as a compatible strong-scaling pair.

This legacy pair is now explicitly listed under `comparison_exclusions` in the
regenerated scaling report. A future exclusive, identity-complete,
configuration-matched rerun may create a new comparison; no replacement run is
currently in flight.

### Separate locality evidence: not a speedup denominator

A different CPU-only public-Lance experiment read 40 fragments with 100
contiguous images per fragment. Image-only `fragment.scanner` with 32 workers
observed about 225.7 images/s and 52.3 MiB/s over 4,000 images; the associated
planning band is 150-225 images/s because storage performance varied. A random
supported grouped-read baseline in that experiment was about 30.54 images/s
and 7.75 MiB/s. Raising concurrency to 64 CPU workers and 256 I/O workers
regressed rather than improved throughput, and stable physical layout produced
higher I/O activity.

This is evidence that fragment locality and bounded concurrency matter. It is
not comparable to the random URL-join benchmark: the projection, key-resolution
work, query distribution, row layout, and execution path differ. It must not be
used as the denominator for a GPU, `lance-ray`, or Ray Data speedup.

### Required measurement matrix

| TODO | Required result | Acceptance condition |
| --- | --- | --- |
| `TODO(exclusive-scaling)` | Submit a fresh exclusive 1/2/4/8-node latency sweep; prior jobs are closed and excluded | Identical payload projection, immutable sidecar, read/concurrency/cache/validation policy, package/code identity, global manifest partitioning, exact rank set, and common Slurm run identity |
| `TODO(storage-saturation)` | Rerun 64 left tables concurrently/node with 8 persistent actors for 8 waves; the first one-node attempt completed only one of two required repeats | Hold 131,072 images/node: 16,384/GPU, 131,072 on one node, and 1,048,576 on eight nodes; widen locality windows beyond 2,048 rows, allocate enough time for repeat spread, and keep S3 primary |
| `DONE(one-big-private-call-negative)` | One 16,384-ID private call ran for more than eight minutes and was stopped | Do not use a giant-call production path; retain 1,024/16 and widen the coordinate queue |
| `DONE(queue-rerun)` | Corrected 262,144-row image-only queue completed two repeats at 785.56-803.06 images/s | Preserved 1,024/16; both repeats correct with identical digest, 2.018 reads/image, and 1.241x amplification |
| `DONE(projection-ab)` | Image-only, image+URL, and full projection on the exact 16,384-row manifest | Two repeats each; identical payload digest; image-only removed 69.1% of full-projection reads |
| `DONE(pinned-io-trace)` | Traced 2,881 post-coalescing reads on pinned PyLance `0b82051` | 100% IOTracker reconciliation; 4-KiB cross-request merge remains an unmeasured runtime experiment |
| `TODO(cpu-baselines)` | Rerun naive PyLance and CPU Curator with 1/2/4/8 persistent actors per node | Hold total per-node rows and aggregate I/O bounds constant while actor count changes; report setup, steady state, telemetry, and spread |
| `TODO(fragment-local)` | Sweep bounded sorted private-call window sizes and run the MPF stable-ID return path | Do not restore public fragment compatibility; report IDs/private call, I/O operations/image, read amplification, throughput, and peak memory |
| `TODO(hybrid-density)` | Add immutable payload-size metadata or validate a density estimator for variable-size images | Enforce a true hard payload-byte cap and compare it with the current explicit estimated-byte profiles |
| `DONE(public-lance-ray)` | Persistent public API ran 257.69-265.82 images/s with the matching digest | Rerun image-only after projection pruning and retain full repeat spread |
| `TODO(ray-data-rerun)` | Rerun `lance_ray_datasource` and persistent GPU actor arms | Exclusive Ray cluster; use 8 persistent actors/node and 4-8 waves for saturation, report startup/setup separately, and validate cross-arm digest |

No CPU-vs-GPU, naive-vs-GPU, or Ray-vs-GPU speedup should be quoted until the
corresponding row passes these gates.

In particular, the current globally split 64-table jobs answer only a small
fixed-work latency question. They must not be cited as evidence that the
storage ceiling, actor count, or queue depth has saturated.

## 6B, 20B, and 100B+ reporting model

[`gpu_lance_scale_model.py`](../../benchmarking/scripts/gpu_lance_scale_model.py)
produces a machine-readable JSON report and Markdown tables. It treats 6B,
20B, and 100B+ as image-reference/probe counts. The 100B scenario is the lower
bound represented by exactly 100B references, not a claim about an upper bound.

### Keep three scales independent

The model must not equate references, unique keys, and payload reads:

1. **Unique reference keys** determine sidecar and resident cuDF-index capacity.
2. **Image references** determine probes and compact shuffle records. Duplicate
   references still require placement in output documents.
3. **Logical payload reads and payload bytes** determine Lance I/O. Deduplication
   and cache behavior can make this smaller than the probe count.

For target reference count `R`, the current-MINT ratios derive:

```text
target_unique_keys = ceil(R * current_unique_keys / current_references)
target_payload_reads = ceil(R * current_payload_reads / current_references)
```

Sidecar and resident GPU-index bytes scale linearly from measured bytes per
unique key. The minimum sharded GPU count is computed from the configured H100
memory and usable-memory fraction, then rounded to whole 8-GPU nodes. The model
reports both partitioned sidecar bytes per node and the much larger cluster
read cost of full per-node replication.

### Runtime composition

Model v3 accepts a benchmark artifact only when the root and requested arm are
complete, at least two repeats are present, every repeat is correct and
internally consistent, and the payload and output digests are stable. It keeps
the raw repeats in the input artifact. Running, partial, teardown-failed,
incorrect, or digest-unstable jobs cannot become scale anchors.

The v3 source contract also pins the payload projection, a legacy-v1 sidecar
manifest digest, a sanitized sidecar-file count/inventory digest, sparse-read
mode, concurrency, cache, validation, and package/code identity. The legacy
queue artifact did not record its sidecar manifest URI/digest, so v3 labels that
identity as a caller pin whose in-artifact cross-check was unavailable; it is
not silently presented as a native v3 benchmark. Raw internal sidecar paths are
not copied into the generated model artifacts.

The primary queue diagnostic scales the observed end-to-end stage rate. A
separate subsystem view reports fetch-only rate. Both distinguish logical
useful payload bytes, Lance projected bytes, physical storage-read bytes,
physical calls, and average read size. The `dominant_arithmetic_term` compares
correlated equations from the same repeats; it is not a causal bottleneck
claim and `storage_saturation_proven` remains false.

Index-capacity nodes and throughput-reader nodes are independent axes. The
model first reports the minimum whole-node index footprint, then evaluates
explicit one-, two-, four-, and eight-node reader profiles. It never chooses a
runtime node count merely because that count can hold the index. Multi-reader
rates remain idealized queue-diagnostic extrapolations until the matching
saturation sweep supplies a measured node-rate curve.

### Required sensitivities and exclusions

Every scale report includes:

- minimum index capacity separately from explicit throughput profiles;
- end-to-end and fetch-only minimum/median/maximum repeat ranges;
- logical, projected, and physical byte totals and rates;
- physical reads/payload, average physical read size, amplification, and peak
  memory; and
- field-level evidence labels for inventory, measured queue geometry, modeled
  assumptions, and extrapolations.

The model excludes scheduler startup, metadata scans, object-store throttling,
partition skew, failures, retries, output writes, and cost. Before using its
100B+ result for capacity commitments, replace every placeholder with dataset
inventory or exclusive-run measurements and add an explicit skew and storage
throttling scenario. Payload-rate anchors must come from the per-node saturation
geometry, not the current globally split 64-table jobs; otherwise linear node
scaling extrapolates a transient latency result rather than a storage-bound
steady state.

### Queue-diagnostic scenarios

The clean v3 model uses the two correct 262K image-only repeats: 785.56-803.06
images/s end to end, 1,117.15-1,127.14 images/s fetch-only, and 2.0177-2.0184
physical reads/payload. Payload reads conservatively equal image references;
this is an explicit no-cross-window-reuse upper bound, not measured corpus-wide
I/O.

| Scenario | Unique keys | Logical payload | Physical reads | Minimum index nodes | Modeled 8-node runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6B | 1.35B | 748 TiB | 928-929 TiB | 1 | 1.68-1.72 days |
| 20B | 4.50B | 2.43 PiB | 3.02 PiB | 2 | 5.61-5.73 days |
| 100B+ | 22.5B | 12.2 PiB | 15.1 PiB | 6 | 28.0-28.7 days |

Every runtime in this table is a `queue_diagnostic_extrapolation` that assumes
64 idealized readers across eight nodes. It is neither an SLA nor evidence of
linear node scaling. Sparse-call sensitivity reports call counts only; it does
not invent a runtime improvement from rates measured in the same run.

Artifacts: [v3 model inputs](../../benchmarking/results/gpu_lance_column_fetch/scale_model_inputs_queue_clean_v3.json)
and [v3 JSON model](../../benchmarking/results/gpu_lance_column_fetch/scale_model_queue_diagnostic_v3.json).

## Reproduce a matched benchmark

The harness reads credentials from the execution environment or node identity;
do not put secrets in the command or result JSON. A representative run is:

```bash
python benchmarking/scripts/gpu_lance_column_fetch_benchmark.py \
  --query-manifest /path/to/pinned-query-manifest.parquet \
  --image-lance-uri "$IMAGE_LANCE_URI" \
  --image-lance-version 4 \
  --storage-options-json '{}' \
  --reference-glob '/path/to/url-row-id-sidecar/*.parquet' \
  --reference-manifest-uri /path/to/sidecar-manifest-v2.json \
  --reference-manifest-sha256 '<caller-pinned lowercase SHA-256>' \
  --expected-reference-rows 355952746 \
  --task-rows 256 \
  --coalesce-tasks 64 \
  --lookup-batch-size 2000 \
  --fetch-batch-size 1024 \
  --io-threads 16 \
  --max-pending-fetch-batches 16 \
  --warmup-count 1 \
  --repeat-count 3 \
  --arm naive_pylance_scalar \
  --arm cpu_lance_column_fetch_stage \
  --arm gpu_lance_column_fetch_stage \
  --arm lance_ray_datasource \
  --arm lance_ray_gpu_fetcher \
  --arm lance_ray_gpu_actor \
  --arm ray_data_persistent_gpu_actor \
  --output /path/to/result.json
```

Here `fetch_batch_size=1024` and 16 pending takes reproduce the measured remote
default. Use 64 only to reproduce the original anchor. Widen locality with the
coordinate queue (`task_rows * coalesce_tasks`), not by replacing medium takes
with one giant private call, and report queue bytes plus peak host memory.

Generate the scale report only after assembling measured inputs:

```bash
python benchmarking/scripts/gpu_lance_scale_model.py \
  --input /path/to/measured-scale-inputs.json \
  --output /path/to/scale-model.json \
  --markdown /path/to/scale-model.md
```

## Implementation and source references

- [`GpuLanceColumnFetchStage`](../../nemo_curator/stages/interleaved/gpu_lance.py)
- [`GpuLanceShuffleFetchStage`](../../nemo_curator/stages/interleaved/gpu_lance_shuffle.py)
- [Benchmark harness](../../benchmarking/scripts/gpu_lance_column_fetch_benchmark.py)
- [Sidecar-manifest builder](../../nemo_curator/stages/interleaved/build_gpu_lance_sidecar_manifest.py)
- [Saturation runner](../../benchmarking/scripts/gpu_lance_saturation_runner.py)
- [Remote sequential-ceiling runner](../../benchmarking/scripts/gpu_lance_remote_sequential_ceiling.py)
- [Scaling report builder](../../benchmarking/scripts/gpu_lance_scaling_report.py)
- [Scale model](../../benchmarking/scripts/gpu_lance_scale_model.py)
- [RAPIDS-MPF shuffle architecture](https://docs.rapids.ai/api/rapidsmpf/stable/background/shuffle-architecture/)
- [RAPIDS-MPF streaming execution](https://docs.rapids.ai/api/rapidsmpf/stable/background/streaming-engine/)
- [RAPIDS-MPF configuration](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/)
- [cuDF `DataFrame.merge`](https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.dataframe.merge/)
