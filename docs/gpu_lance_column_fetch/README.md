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

### Requested-deliverable status

| Requested outcome | Current evidence | Status |
| --- | --- | --- |
| GPU `lance-ray` fetch with a resident cuDF index and Arrow boundaries | The order-preserving URL API has correctness-valid real-v4 repeats; the sidecar-free reader now published the exact full-fragment payload as a fully validated durable Arrow overlay | Per-plan remote payload boundary measured; the canary reused pre-resolved stable IDs, so it is not an end-to-end GPU-lookup rate |
| Images/s per eight-H100 GPU node | **452.24 driver images/s** at the primary four-wave point; the 635.12-images/s one-wave point is locality-only | One-node production-geometry evidence measured; storage saturation unproven |
| Large coordinate-queue locality | The exact full-fragment canary sustained **1,094.28 unique images/s** through remote fetch/scatter/fsync and **825.81 unique images/s** through fully validated durable publication | Measured one-observation payload boundary; not a matched speedup or repeat distribution |
| Exactly 64 left-interleaved task tables active per node | The production overlay stage now batches at most 64 plans and reserves 64 Ray CPUs by default, admitting one actor on a 64-CPU node. Synthetic tests prove global dedupe/fetch, positional `N -> N` scatter, partial-publication retry, and byte-bounded subgrouping | Implementation complete; a current-head remote-S3 rate for the exact 64-plan path remains unmeasured |
| Naive PyLance versus cuDF speedup | The failed attempt was serialized; the harness now schedules deterministic one-key operations in 64-left-table waves, but has no current-head payload repeat | **Unresolved required comparison; corrected harness unmeasured** |
| Ray Data comparison | A persistent-actor control measured 323.36 driver images/s; filtered public-DataSource planning now defers row counts instead of executing the predicate twice, but its payload run remains absent | Partial control only; corrected public DataSource remains a **required unresolved comparison** |
| Sparse-read reduction | The full-fragment canary used 217 private takes for 885,388 unique IDs, avoided 885,171 scalar calls, and measured 1.2888 reads/unique image with 1.0759x amplification | Measured exact workload; causation is not isolated |
| Remote-S3 bandwidth constraint | The full-fragment canary reached 154.89 physical MiB/s by payload-materialization wall with 112.46-KiB average reads, 7.13% of the sequential physical lower bound | Not achieved; still latency/IOPS constrained |
| CPU and GPU 1/2/4/8-node scaling | No compliant 2/4/8-node GPU weak-scaling family or CPU 1/2/4/8 actor sweep exists | **Unresolved required measurement** |
| 6B, 20B, and 100B+ planning | Reproducible queue-diagnostic and materializer-sensitivity models are checked in | Modeled hypothesis, not a scaling result or SLA |
| Real document payload workflow | Job `407257` published 885,388 unique payloads / 928,687 occurrences from one 3,998,698-row document fragment as a 118.86-GiB Arrow overlay in 1,072.14 seconds | **Durable payload boundary complete; full document reconstruction is intentionally separate** |

| Item | Status | Meaning |
| --- | --- | --- |
| One-H100 64/512/1,024/4,096-row private-take sweep | Measured | Correctness-validated real-data runs; two timed repeats after one warmup |
| Persistent public `lance-ray` GPU API | Measured | Correctness-validated real-data run; two timed repeats after one warmup |
| Public URL-key unique-payload Ray stream | Implemented, unmeasured | Keys are resolved once and stable-ID-sorted unique payload batches are yielded through a bounded ordered future window; row count and pending batches are bounded, but variable payload bytes are not |
| Sidecar-free stable-ID payload reader | Completion-driven remote boundary measured; final metrics lost to later timeout | One persistent in-process reader consumes pre-resolved increasing `uint64` ordinals, owns no cuDF sidecar, keeps a bounded running/ready window full behind the consumer, validates exact operation coverage, and entered patch reconstruction at least 1.230x sooner than the censored ordered run |
| Image-only / image+URL / full projection A/B | Measured | Exact 16,384-row manifest, persistent warm fetchers, two repeats per projection, identical payload digest |
| Ray Data cold-actor `lance-ray` API | Measured, setup-sensitive | Correct output, but the harness recreated the actor pool on every repeat; this is not persistent steady state |
| 262,144-row image-only coordinate queue | Measured | Corrected validator, one warmup and two correct real-data repeats with identical payload digest |
| Full-left-fragment CPU sorted fetch | Measured, non-isolated locality diagnostic | Job `404464` completed `0:0`; 885,388 unique real images and two payload-fetch repeats; correctness covers coordinate order, row/null counts, and repeated logical byte counts, not digests proving payload identity or output order |
| 16-fragment amortization curve | Measured locality control, prose-only | Physical-read curve is valid; shared-resource throughput is not a speedup denominator, and no sanitized machine-readable artifact is checked in |
| One-node remote-S3 eight-H100 sweep | Measured | 131,072 images, eight persistent actors, one warmup and two repeats at 8/4/2/1 waves; 4/8-wave points meet the scheduling-wave policy, but only eight waves gives exactly 64 harness tasks per wave |
| Remote sequential payload scan | Measured lower bound | Reader concurrency did not plateau through 128 readers; the highest all-repeat median is a lower bound, not a storage ceiling |
| One-node Ray Data four-wave control | Measured; offline-v3 eligibility reported by sanitized summary | Both repeats are correct with matching digests; the summary records a zero-payload-read offline revalidation, but the complete v3 terminal family is not checked in |
| Public `fragment.take` exploration | Measured negative | One correctness-valid warmup was decisively slower; the path is not retained for production or compatibility |
| Earlier 8-node run | Rejected legacy comparison | The run is retained as a measured row, but missing policy/sidecar identity plus placement and configuration mismatches make it ineligible for scaling ratios |
| Two-node CPU Curator baseline | Measured | Same 16,384-row manifest and full validation projection; two timed repeats after one warmup |
| File-backed full-fragment patch stage | Implemented; remote materializer measured, final patch synthetic | Five remote-v4 materializer observations cover unique image-only fetch, bounded IPC spool, exact stable-ID placement, RSS, and I/O; actual duplicate fan-out, final document reconstruction, payload identity, and durable patch publication remain synthetic |
| Durable payload overlay | Implemented and remotely measured | Job `407257` fully validated 3,720 Arrow parts before atomic rename; the manifest binds Lance versions, plan/sidecar/fragment identities, coordinates, counts, schema, part hashes, and reconciled I/O metrics |
| Public document materialization graph | Grouped overlay implemented; one-plan remote boundary measured | `GpuLanceDocumentMaterializer` now defaults to one 64-plan/64-CPU overlay actor per 64-CPU node, one global stable-ID union per byte-bounded subgroup, and independently checkpointable outputs; combined full-patch mode remains explicit compatibility behavior |
| RAPIDS-MPF 26.06 lifecycle gate | Measured, setup-only | Job `405351` completed a 17.202-second real two-rank/two-window MPF lifecycle on one H100, including empty input, both operation IDs, extraction, ID reuse, and cleanup |
| Public document graph canary | Failed before payload I/O | The same job passed Ray and document-partition setup, then rejected a one-partition `replicated_sorted` sidecar at the hash-layout contract. No image take, coordinate plan, patch, or throughput result exists |
| Public document graph sidecar-load canary | Failed before payload I/O | Job `405580` confirmed the one-rank replicated-layout fix and completed document partitioning, then RMM rejected a 31.821345-GiB allocation while cuDF attempted to decode the complete 16-file sidecar in one call. No image take, coordinate plan, patch, or throughput result exists |
| Public document graph segmented-sidecar canary | Setup envelope measured; completion inferred from pinned control flow | Job `406706` measured 52.979 seconds from actor-setup start through UCXX setup under a 64-GiB RMM pool; because pinned `setup_worker` loads and validates the sidecar before returning, the event sequence implies the segmented load completed. The first document batch then failed before payload I/O |
| Sidecar-free storage-identity canary | Failed before document partitioning completed | Job `407202` passed signed preflight, the RAPIDS-MPF smoke, and the 64-CPU/one-GPU Ray gate, then the first document-v1 manifest request returned 404 under the unbound ambient storage identity. No coordinate plan, payload take, patch, or throughput result exists |
| Ordered-reader full-fragment canary | Bounded timeout after coordinate publication | Job `407213` used the approved identity and published the exact 928,687-occurrence/885,388-unique coordinate plan, but its single payload/patch task remained unfinished after 16:28 and was terminated by the 20-minute controller. No images/s or payload correctness claim exists |
| Completion-driven full-fragment canary | Payload completed; patch reconstruction timed out | Job `407235` used the byte-identical coordinate plan and crossed into patch writing within 817.086 seconds of payload-stage entry, a conservative lower bound of 1,083.59 unique images/s; five temporary parts were insufficient for a final patch/digest claim |
| Durable full-fragment overlay canary | Completed | Job `407257` reused the same plan, published a fully validated overlay in 1,072.14 seconds, measured 1,094.28 unique payload images/s and 825.81 durable-boundary images/s, and exited `COMPLETED/0:0` inside the cap |
| Current remote-v4 readiness gate | Passed, metadata only | The approved PDX identity opened image v4 and document v1 and validated stable row IDs, schema, and the `url_btree` index; this did not read payloads or authorize a scaling claim |
| Naive PyLance and public `lance-ray` DataSource comparisons | Corrected harnesses, still unresolved after bounded timeout | Naive scalar operations now run in deterministic 64-table waves; filtered DataSource planning no longer performs a duplicate driver predicate count, and cache-disabled block/concurrency geometry is pinned. There are still zero current-head valid repeats and no speedup ratio |
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

The layouts are not generally interchangeable. Multi-rank MPF execution
requires `hash_partitioned` plus its exact libcudf/RAPIDS-MPF partitioning
descriptor. A canonical `replicated_sorted` manifest is accepted only for the
mathematically equivalent one-rank, one-partition case; replicated manifests
must omit `partitioning`, and hash manifests must include it. Every other
layout mismatch fails before sidecar loading or payload reads.

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
Invoke it as a module so the sibling Curator `lance.py` module cannot shadow
the installed PyLance package:

```bash
python -m nemo_curator.stages.interleaved.build_gpu_lance_sidecar_manifest --help
```

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
  `sparse_calls_avoided` is
  `unique_payload_rows - private_take_calls`, relative to a hypothetical
  scalar private-call path. It is not a physical-read count or a measured
  saving versus another batch size.
- Physical work: Lance `read_iops` counter per logical image.
- Byte work: Lance read bytes divided by returned payload bytes.

The production optimization target is the measured best medium-sized private
`_take_rows` batches per bounded streaming window, plus fewer physical I/O
operations per image. The IDs are deduplicated and sorted before each call so
pinned PyLance can group work internally. The current remote-data default is
1,024 IDs with at most 16 pending calls: a single 16,384-ID call did not finish
within eight minutes, and the 16K full-validation `4096/4` point regressed. The
full-fragment diagnostic later used `4096/16` successfully under a different
workload and projection. A change does not meet this goal if throughput rises
only by increasing unbounded concurrency or memory. Correctness, peak
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

An unmeasured implementation update now retains fetched payloads as Arrow
tables paired with fixed-width stable IDs and reconstructs fan-out with
`pyarrow.compute.index_in`/`take`. Image columns no longer pass through Python
byte objects, nested row dictionaries, or a second payload-sized Python list.
All measured rows in this report predate that change; reduced host RSS and
reconstruction time remain hypotheses until the same manifests are rerun.

The public `stream_unique_lance_columns_on_gpu` path is the bounded-payload
counterpart for full-fragment queues. It resolves and deduplicates one Arrow key
queue once, sorts by stable row ID, and yields one unique
`(stable_row_id, key, payload...)` Arrow table per sparse private-read batch.
The ordered sliding window retains at most `max_pending_fetch_batches`
ready/running results inside the actor and never concatenates all unique payload
batches or performs payload-sized duplicate fan-out. The caller retains compact
origin coordinates and scatters payloads later. This is implementation and CPU
contract-test evidence only. `fetch_batch_size` is a hard row bound, not a byte
bound; immutable per-row payload sizes are still required for hard byte-weighted
admission.

For coordinate plans that already contain resolved ordinals,
`LanceStableIdPayloadStreamer` is the lower-level production boundary. It opens
or reuses one pinned Lance dataset and never constructs the cuDF index or loads
sidecar shards. Curator now supplies its sorted, deduplicated `uint64` IDs to
one persistent reader per payload actor, validates every returned ID and Arrow
field, then performs duplicate fan-out into the actual-byte-bounded file spool.
Reader telemetry separates private-call sum, active-read union, first-to-last
read envelope, scheduler wall, and total consumer-inclusive stream wall. This
is locally tested implementation evidence only. The ordered iterator can still
head-of-line block behind one slow remote request and refills a freed slot only
after the consumer resumes; the bounded real canary must quantify that gap
before this path is described as the throughput winner.

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
4. The origin rank processes a bounded manifest window and deduplicates and
   sorts resolved stable image row IDs.
5. The payload actor feeds the returned stable IDs to one persistent,
   sidecar-free Lance-Ray reader. Its bounded private `Dataset._take_rows`
   calls yield unique Arrow payload batches into Curator's duplicate scatter
   and a durable position-bucketed overlay. Payload bytes remain outside the
   coordinate shuffle, and document association is retained explicitly.

The current actor separates the rank's coordinate opportunity
window from private-call geometry: stable IDs are globally sorted, split into
at most 1,024 IDs per call, and executed with at most 16 calls submitted or
running. It reports observed pending depth and physical read operations/s. This
fixes the prior one-giant-call implementation. Its real MPF lifecycle has run,
but the remote coordinate-to-payload path remains unmeasured.

`GpuLanceDocumentMaterializer` now assembles that actor with a public
one-fragment Lance source and the durable overlay stage. The pipeline fails
fast unless it receives `RayActorPoolExecutor`. Its default eight-rank,
eight-task window exposes 64 active left fragments per node while retaining the
measured 1,024-ID/16-pending private-read geometry. Patch actors reserve eight
Ray CPUs each by default, so a node advertising 64 CPUs admits at most eight
payload actors instead of multiplying 64 fragment tasks into 64 actor-local
16-read queues. An optional global worker cap is also available. The reported
per-actor reservation combines the configured fetch estimate and spool window;
it is not a hard bound for variable-size payloads. The payload actor no longer
reimplements the remote read scheduler or reloads a URL sidecar: it lazily
constructs the pinned Lance-Ray stable-ID reader from the already-open image
dataset. This remains implementation evidence until a real
coordinate-to-overlay run completes.

An opt-in, unmeasured coordinate-plan mode now stops after the return shuffle
and publishes one digest-bound Arrow/Parquet artifact per deletion-free
document fragment. Each plan retains only `document_rowaddr`, the physical
`document_position`, and nullable `stable_row_id`; it is ordered by document
position and binds both dataset versions plus the sidecar and image-fragment
manifest digests. Existing artifacts are adopted only after full content,
schema, count, and SHA-256 validation. No image payload is fetched or routed
through MPF in this mode.

`LanceCoordinatePayloadOverlayStage` consumes one such plan, fetches each unique
stable ID once with medium private takes, fans duplicates out in Arrow, and
writes position-bucketed Arrow IPC parts directly beneath a same-filesystem
attempt directory. Before atomic rename it validates every part hash, Arrow
schema, bucket, coordinate, row count, and the filtered plan digest. The outer
manifest is identity-bound and retains reconciled sparse-call, physical-I/O,
amplification, queue, and throughput metrics. A crash after rename is adopted
with zero additional image requests; a pre-publication failure removes the
attempt and leaves no successful artifact.

The compatibility `LanceCoordinatePayloadPatchStage` still performs the prior
combined remote fetch plus complete document rewrite. It does not consume an
overlay. Full reconstruction from a durable overlay remains a separately
required downstream stage and must not be included in the remote-fetch
throughput denominator.

For recovery, `LanceCoordinatePlanReader` validates a complete shared plan
inventory and emits fragment-sorted, content-identity source tasks. A separate
checkpointed reader-to-overlay pipeline reruns only unfinished fetches without
repeating the GPU coordinate shuffle. `LancePayloadOverlayReader` then validates
the finished overlay inventory and emits semantic source IDs independent of
physical paths or completion-order part layout.

Ray/Curator carries the compact coordinate task and checkpoint identity between
stages. Payload bytes are streamed directly into the durable Arrow overlay
instead of a large Ray object-store queue. This preserves backpressure without
making plasma hold a fragment's 100+ GiB payload.

Only compact key and coordinate records cross the network. Image payloads are
read directly by the rank producing the output. This follows RAPIDS-MPF's
[process-per-GPU streaming shuffle architecture](https://docs.rapids.ai/api/rapidsmpf/stable/background/shuffle-architecture/):
bounded insertion windows and spillable buffers allow an out-of-core execution
plan. RAPIDS-MPF 26.06 completes a window with `wait()` before iterating local
partitions; it does not retain the older `wait_any()` incremental extraction
behavior. The task window is therefore the explicit backpressure unit.

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
object path and byte range, so no custom Rust build was required. The retained
host-local trace is not published as a sanitized machine-readable artifact;
the measurements in this subsection are checked-in prose-only evidence.

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
commit. The table above is the checked-in prose summary, not an independently
replayable evidence artifact; host-local trace paths are intentionally omitted.

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
| `fetch_window_bytes` with `estimated_payload_bytes_per_row` | MPF payload/output rank window | No-deadline estimated payload cap; fetched bytes are checked again with duplicate fan-out and accepted profiles are exactly `256MiB`, `1GiB`, and `4GiB`. This is not a coordinate-queue cap. |
| `fetch_batch_size` and `max_pending_takes` | MPF and replicated private payload readers | Current measured remote default is 1,024 IDs/take with 16 pending takes; the larger coordinate opportunity window is split into these bounded calls |
| `coordinate_plan_output_path` | MPF coordinate-only mode | Shared absolute filesystem root for one atomic, digest-bound plan per deletion-free left Lance fragment; payload reads are skipped |
| Overlay `batch_size=64` and `coordinate_window_bytes` | Checkpointed cross-plan payload stage | One 64-CPU actor accepts 1-64 plans, releases plans after identity prevalidation, then reloads only one semantic-sorted subgroup under a conservative `256MiB`, `1GiB`, or `4GiB` retained-Arrow bound; default is `4GiB`. Arrow-kernel scratch is not observable through this bound and remains covered by process RSS |
| Grouped overlay shared spool coordinator | Cross-plan payload scatter | All member spools share one `256MiB`, `1GiB`, or `4GiB` active-Arrow budget instead of multiplying the target by the member count. Bounded peak and isolated oversized-row peak are reported separately |
| Payload spool `target_bytes`, `bucket_rows`, and sync mode | Attempt-local payload reconstruction | Hard bound for normal retained Arrow bytes, default 131,072-row position buckets, and one isolated oversized row; production uses explicit `attempt_local` sync while retaining close, atomic rename, and SHA-256 validation, and final Parquet publication remains durable |
| Pinned PyLance object-store I/O parallelism | Private payload reader | Bounds buffered reads inside the Rust take path; it is not a substitute for a larger locality window |
| `rmm_pool_size` and `spill_memory_limit` | MPF actor | Bound or spill device working memory |

RAPIDS-MPF uses bounded channels with backpressure in its
[streaming engine](https://docs.rapids.ai/api/rapidsmpf/stable/background/streaming-engine/)
and exposes [memory and spill configuration](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/).
Production runs must additionally record peak queued Arrow bytes, peak host
RSS, peak GPU bytes, spill bytes, and window occupancy. The file-backed stage
reports its hard spool bound, isolated oversized rows, coordinate bytes,
process peak RSS, and a separate estimate for in-flight private-take results.
The overlay estimate uses the reader's full `2 * max_pending + 1` retained-batch
contract, while the compatibility patch estimate retains its older running-call
definition. Both remain row-based rather than actual-byte bounds because image
sizes are unknown before I/O. A composite pre-I/O hard bound still
requires immutable size metadata; the report does not relabel the spool target
as a whole-process memory cap.

There is also a larger output-contract gap. One real left fragment contains
928,687 image occurrences and 113.7447 GiB of unique logical payload, while the
largest accepted MPF payload/output window is 4 GiB. The current single-stage
actor validates the whole task window, concatenates its fetched payloads, and
emits one payload-bearing output per input. It therefore cannot run that
representative fragment under any accepted profile. Full-fragment locality
now uses the fixed-width coordinate task to feed bounded payload patches and
deterministic contiguous child outputs. The local Arrow IPC spool enforces an
actual-Arrow-byte target, deterministic document-position buckets, per-file
SHA validation, explicit oversized-row reporting, and cleanup/reaping under a
per-artifact lock. It is attempt-local, not checkpoint evidence. The reusable
spool primitive defaults to `fsync`; the production patch stage selects
`attempt_local`, which skips only ephemeral spool file/manifest/directory
fsyncs and recomputes after failure. Streaming payload fetch, full document
reconstruction, and durable patch publication are connected and covered by
synthetic retry/correctness tests. The remote materializer canary below now
measures fetch-to-spool throughput and peak RSS, but not final patch throughput.

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
| Work per node | 512 benchmark Arrow tasks of 256 rows and 131,072 images total per 8-H100 node; these task tables are not production Lance fragments |
| Active task tables per wave | 64/128/256/512 at the 8/4/2/1-wave points; only the eight-wave point matches the exact 64-task-at-once requirement |
| Waves | Eight 2,048-row or four 4,096-row calls per actor for primary evidence; two and one waves are locality sensitivities only |
| Eight-node total | 1,048,576 images, preserving the same 131,072-image workload on every node |
| CPU sweep | 1, 2, 4, and 8 persistent actors per node over the same total per-node input; bound aggregate I/O rather than multiplying it with actor count |

The completed one-node sweep satisfies the per-node row volume, actor count,
and 4-8-wave policy. Only its eight-wave point satisfies exact concurrency of
64 benchmark task tables per node; the four-wave locality-efficient point uses
128 task tables per wave. Equivalence between one 256-row harness task and one
production left-interleaved table is not established. Older jobs that globally
split 64 task tables across all ranks do not satisfy the per-node requirement,
and no compliant 2/4/8-node weak-scaling result exists yet.

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
implies that the benchmark is eligible. Schema v3 additionally waits for one
atomic, digest-bound terminal marker per allocated node; the configurable wait
defaults to 180 seconds and cannot be set below 120 seconds. Eligibility-v1/v2
families are not primary anchors until
[`gpu_lance_revalidate_terminal.py`](../../benchmarking/scripts/gpu_lance_revalidate_terminal.py)
revalidates their immutable benchmark and raw telemetry into a new v3 family.
Live Slurm launches must also provide
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

All listed arms except the end-to-end MPF payload execution are implemented in
the benchmark harness. The MPF two-shuffle stage and actor are implemented,
covered by focused tests, and have passed a real no-data lifecycle gate, but
still require a matched remote-data payload run.

### Baseline readiness and interpretation

Implementation is not benchmark readiness. The naive PyLance scalar arm and
public `lance_ray_datasource` arm now have a bounded real-data attempt, but not
a complete, comparable payload result. The exact 1,024-key public `url,md5`
oracle required 6,085 seconds. Version 4 then proved complete `url_btree`
coverage: 355,952,746 indexed rows, zero unindexed rows, 56,696 indexed
fragments, and zero unindexed fragments. Even with correctness-gated
`fast_search=True`, the serialized naive `url,image` warmup did not complete
and was cancelled; the Ray Data payload warmup never started. Both arms have
zero valid repeats and remain speedup-ineligible.

The current harness corrects both structural mismatches without inventing a
rate. Naive PyLance still performs one scalar-index lookup and one stable-ID
take per unique URL, but schedules deterministic waves containing at most one
URL from each left table, with 64 concurrent operations by default. It reports
overlapping lookup/fetch task-seconds separately while driver wall remains the
only throughput denominator. For filtered public DataSource reads, Lance-Ray
now sets initial `BlockMetadata.num_rows=None` instead of running the remote
predicate once on the driver and again on workers. The benchmark pins
`override_num_blocks`, concurrency, and a disabled worker dataset cache before
allowing a comparison. These are locally tested harness changes, not measured
speedups.

Slow public baselines are now isolated one arm per process. Check at 10 minutes
and enforce a 20-minute hard cap. A non-completing 1,024-row phase at 20 minutes
has an arithmetic throughput upper bound of 0.853 images/s, not a measured
rate. Preserve the partial report, terminal accounting, and setup metrics, then
move on. Never keep a baseline allocation running for hours merely to obtain a
denominator.

The run identity must also bind public-DataSource filter batch size and
concurrency, prewarm and cache settings, warmup rows, Lance CPU/I/O thread
limits, key/index identity, storage-endpoint digest, package/code identity, and
whether actor-pool construction is inside the timed envelope. A partial raw
summary or a ratio calculated before every expected repeat and terminal
correctness gate completes is not evidence.

The completed four-wave Ray Data actor control is eligible as an observed
framework/lifecycle comparison, not a clean algorithmic speedup. Ray Data
rebuilt and warmed its actor pool inside each measured repeat while the direct
`lance-ray` actors persisted across the suite. The common driver-wall ratio
therefore includes that lifecycle asymmetry; the separately named actor-span
ratio is useful context but does not by itself make setup symmetric.

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

The current harness defines primary cross-arm throughput as rows or payload
bytes divided by the common `arm.run` wall envelope for every arm. Actor
process span and fetch span remain separately named diagnostics. Legacy 404060
families predate that timing-basis field and retain both rates, but new
comparisons cannot substitute actor span for primary wall throughput.
Physical read operations/s must name its denominator. New replicated and
`lance-ray` runs use a private-take execution span; the file-backed fragment
stage reports a private-take execution envelope separately from total
materialization. Legacy one-node actor results below divide by actor-process
span, while the CPU full-fragment diagnostic divides by payload-fetch span.
Those legacy rates are retained with their scope and are not silently mixed.

## Measured results

### Historical 16K anchor-relative throughput table

For matched rows, the throughput ratio is
`candidate median images/s / 64-row anchor median images/s`. All numeric
anchor-relative rows except the 262,144-row queue use the same pinned table and
16,384-row manifest. The larger queue's anchor-relative ratio is an unmatched
locality diagnostic, not a final speedup. The projection-pruned row
intentionally changes the timed projection but proves the same payload digest;
other 16,384-row rows use the full validation projection.

The 183.55- and 247.35-images/s rows are historical anchor and tuning points,
not current throughput leaders. The 262K row was the prior large-queue
locality leader. The later full-left-fragment CPU diagnostic measured a higher
payload-only rate, but it is intentionally excluded from this ratio table
because its workload, phase, projection, batch size, hardware, and correctness
scope are not matched.

| Arm | Median steady wall (s) | Images/s | Ratio vs 64-row anchor | Status |
| --- | ---: | ---: | ---: | --- |
| Curator GPU, 64 IDs/take | 89.3353 | 183.5487 | 1.0000x | Measured anchor |
| Curator GPU, 1,024 IDs/take | 66.3122 | 247.3493 | **1.3476x** | Measured tuning result |
| `lance_ray_gpu_fetcher`, image-only | 52.4118 | 312.6114 | **1.7031x** | Measured production projection |
| Curator GPU, 262,144-row image-only queue | 330.0664 | 794.3123 | **4.3275x** | Unmatched locality diagnostic; two correct repeats |
| `lance_ray_gpu_fetcher`, full validation | 65.7135 | 249.6370 | **1.3601x** | Matched projection-control session |
| `lance_ray_gpu_actor`, 1,024 IDs/take | 117.8131 | 139.16 | 0.7582x | Cold end-to-end diagnostic; actor span was 76.8734 s, so this row is comparison-ineligible |
| CPU Curator, two nodes, 64 IDs/take | 911.8877 | 19.1044 | 0.1041x | Measured but comparison-ineligible; unmatched raw ratio and 14.44-23.77 images/s spread |
| `naive_pylance_scalar` | N/A | N/A | N/A | Bounded timeout: 1,024-key fast-search warmup did not complete; conservative observed upper bound <0.438 images/s, with no correctness-valid repeat |
| `lance_ray_datasource` | N/A | N/A | N/A | Setup-only: 72.26 s including 11.56 s Ray startup; payload warmup did not start before the bounded run was stopped |
| `ray_data_persistent_gpu_actor` | N/A | N/A | N/A | Measured separately at 131,072 rows; a sanitized summary reports offline schema-v3 eligibility, but the full v3 family is not checked in and the run is not comparable to this 16,384-row table |
| `gpu_lance_shuffle_fetch` | Pending | Pending | Pending | Two-shuffle private-read run required |

### Bounded public-baseline result

One exclusive 64-core CPU node built an independent 1,024-row oracle from 64
real left tables with 16 unique URLs per table. Two exact public scalar-index
queries of 512 URLs projected `url,md5`; job `404944` completed `0:0`, and the
oracle step took 6,085 seconds with 11,765,220 KiB maximum RSS. All URLs and
expected MD5 values were present, and the query artifact SHA-256 is
`e53230133646e23b71781cd271758d49b9778305d327cdde57bacd6f403d1cc4`.
This is setup/oracle evidence, not image-payload throughput.

The follow-up job `405042` verified full pinned index coverage before enabling
`fast_search=True`. Naive setup took 58.043 seconds. Public Lance-Ray DataSource
setup took 72.257 seconds, including 11.564 seconds of Ray startup, with one
exact dataset/session retained per worker. The 1,024-key naive warmup then ran
for at least 39 minutes without completing. It was cancelled at the user's
bounded-stop direction; the 42m43s measurement step and parent allocation are
terminal, Ray was stopped, and the node-local runtime was removed.

There are no completed warmups, repeats, output digests, or payload correctness
records from this attempt. The conservative observed bound is less than 0.438
images/s, but it is not a throughput point. No naive-vs-GPU or Ray-Data-vs-GPU
speedup is permitted. The sanitized artifact retains the raw hashes, setup
metrics, coverage gate, terminal accounting, timeout bounds, and limitations:
[public baseline timeout evidence](../../benchmarking/results/gpu_lance_column_fetch/public_baseline_timeout_v1.json).

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

Projection and locality are independent measured dimensions. The A/B above
holds the exact manifest and payload concurrency fixed, so it isolates the
extra column/page I/O caused by `url`, MD5, and dimensions. The 262K and
full-left-fragment results instead widen the coordinate opportunity window and
change rows per touched fragment while keeping the timed projection image-only.
Those workload-scale gains cannot be multiplied by the projection ratio, and
the offline trace merge estimates are hypotheses until a runtime coalescer
reproduces them.

The Ray Data `lance_ray_gpu_actor` control produced the same digest and a
209.01-217.41 images/s actor-process spread. Its end-to-end repeats were
114.81-120.82 seconds because this harness run recreated and warmed the actor
pool each time. It is a cold-actor baseline, not the separate persistent
eight-actor workload reported below.

### One-node remote-S3 saturation and locality sweep

A fresh non-array interactive allocation ran continuously for 2:56:44 with
`Requeue=0`, one exclusive eight-H100 node, and eight persistent actors. The
workload held 131,072 real images in 512 benchmark Arrow task tables of 256
rows; these are harness scheduling units, not 512 production Lance fragments.
Every point used the same pinned Lance version-4 table, sidecar manifest, query
manifest, image-only projection, 1,024-row private takes, one warmup, and two
measured repeats. All repeats returned every row in manifest order with no
missing payload and identical output and payload digests.

The 8/4-wave points meet the primary one-node workload policy. The 2/1-wave
points are valid locality measurements, but their larger per-actor windows do
not meet the required 4-8 scheduling-wave policy and cannot anchor scaling or
speedup reports. Only the eight-wave point schedules exactly 64 harness task
tables per node per wave. The four-wave point schedules 128, so its primary
classification does not close the separate fixed-64-concurrency requirement.

| Waves; rows/actor call | Actor images/s repeats (median) | Driver images/s repeats (median) | Physical MiB/s repeats (median) | Reads/image median | KiB/read median | Amplification median | Max pending/actor | Evidence class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8; 2,048 | 211.83, 188.07 (**199.95**) | 204.00, 181.92 (**192.96**) | 43.24, 37.67 (**40.45**) | 4.6790 | 44.29 | 1.7040x | 2 | Primary; exactly 64 harness tasks/wave |
| 4; 4,096 | 492.58, 489.37 (**490.98**) | 453.02, 451.47 (**452.24**) | 98.25, 95.51 (**96.88**) | **3.6193** | **56.00** | **1.6628x** | 4 | Primary, locality-efficient; 128 harness tasks/wave |
| 2; 8,192 | 541.00, 513.82 (**527.41**) | 500.71, 477.35 (**489.03**) | 108.45, 101.73 (**105.09**) | 4.4654 | 45.71 | 1.6789x | 8 | Locality diagnostic only |
| 1; 16,384 | 689.69, 719.02 (**704.36**) | 623.16, 647.08 (**635.12**) | 136.93, 141.76 (**139.35**) | 4.6331 | 43.77 | 1.6674x | 16 | Highest throughput in this wave sweep; locality diagnostic only |

The corresponding median physical read rates are 937.5, 1,777.3, 2,356.8,
and 3,261.1 operations/s for the 8/4/2/1-wave points. These are derived from
the cumulative Lance read count divided by actor process span, not driver wall.
At the primary four-wave point, each repeat avoided 130,901 hypothetical
scalar private calls under the definition above.

The four-wave point is 2.455x faster by actor span than the eight-wave median,
while reducing physical reads/image by 22.7%, increasing average read size by
26.4%, and slightly lowering amplification. One wave raises actor throughput
another 43.5% over four waves, but reads/image regress by 28.0% and average read
size falls by 21.8%. The higher IOPS and worse locality are consistent with
that gain. The fixed-order sweep does not isolate causation.

A separate remote sequential payload scan did not plateau through 128 readers.
Its highest all-repeat median was 2,229.501 logical MiB/s and 2,173.283 physical
MiB/s, so these are lower bounds rather than a measured ceiling. The raw
128-reader logical repeats were 535.038, 2,338.081, 2,236.201, and 2,222.801
MiB/s; the unexplained low point is retained rather than discarded. Four-wave
physical throughput reaches only 4.46% of the sequential physical lower bound,
and the one-wave point reaches 6.41%. Average reads remain 44-56 KiB rather
than multi-MiB. The system is therefore still latency/IOPS constrained and
**storage-bandwidth saturation is not proven**.

Five-second phase-window telemetry also shows substantial compute headroom.
These windows start after setup but extend through post-fetch correctness
hashing, so they are broader than the timed driver/fetch spans and are not
strict steady-state samples.
Across the four `lance-ray` wave points, median node CPU busy was 1.11-2.23%
and p95 was at most 3.02%. Median and p95 GPU utilization were 0% because the
cuDF join is shorter than the sampling interval; sampled maxima were 0-15%,
while the resident index held 48,143 MiB per GPU. Four-wave `eth0` receive
rates were 74.93 and 77.05 MiB/s, and one-wave rates were 99.67 and 100.52
MiB/s. These counters support the IOPS/latency diagnosis; they do not replace
the Lance read-byte accounting or establish a storage ceiling.

The result identifies the next implementation target: resolve and deduplicate
coordinates once per node, bucket them by physical fragment/range across all
actors, and then route only compact coordinates to readers. Payload bytes stay
out of that shuffle. This preserves the wide node-level locality opportunity
without forcing one actor to issue all reads.

The sanitized per-repeat values, artifact hashes, workload identity, and
limitations are retained in the
[one-node evidence summary](../../benchmarking/results/gpu_lance_column_fetch/one_node_remote_saturation_v2.json).

### Workload-matched four-wave Ray Data control

The Ray Data persistent-GPU-actor control used the same 131,072-row manifest,
eight actors, four waves, projection, take size, and correctness digests. Its
actor-process repeats were 428.23 and 408.65 images/s (418.44 median), while
driver end-to-end repeats were 324.22 and 322.50 images/s (323.36 median).
Per-repeat actor setup took 73.75 and 68.07 seconds, and its internal warmup
took 12.11 and 11.33 seconds. Setup is reported separately rather than hidden.
The data geometry is matched; actor lifecycle is not.

The control averaged 9.2567 reads/image, 31.12 KiB/read, 3,873.5 physical read
operations/s, and 2.3707x read amplification. The observed common driver-wall
ratio is 1.399x in favor of the four-wave `lance-ray` actor path, with 60.9%
fewer reads/image and 29.9% lower amplification. Ray Data rebuilt its actor
pool inside each repeat while the direct actors persisted, so 1.399x is a
framework-plus-lifecycle ratio, not a clean algorithmic speedup. The 1.173x
actor-span ratio is a secondary diagnostic and does not remove that asymmetry.
The checked-in sanitized summary reports that immutable offline revalidation
produced a schema-v3-eligible family. The complete v3 benchmark, run identity,
telemetry, node marker, and eligibility family is not checked in, so that
eligibility is a summary-level offline claim rather than independently
revalidatable repository evidence. The measured repeat values and digests
remain in the summary. According to that summary, the original Ray Data
terminal gate incorrectly required the `lance-ray` arm's `payload_take_*`
counter names instead of Ray Data's `private_take_*` names; the corrected
family binds the unchanged artifacts by SHA-256 and performed zero payload
reads. None of this removes the actor-lifecycle asymmetry.

Ray Data's phase-window CPU busy median was 2.72% (6.21% p95); GPU utilization
was 0% at median and p95 with an 88% sampled maximum. Its two `eth0` receive
rates were 84.55 and 84.66 MiB/s. This window likewise includes post-fetch
correctness work; it supports compute-headroom diagnosis but is not strict
timed-fetch telemetry.

### Two CPU nodes

The CPU Curator baseline split the same 64 tables across two exclusive CPU
nodes, using `fetch_batch_size=64` and 64 configured I/O threads per rank.
Maximum-rank setup was 3,941.58 seconds, almost entirely Lance index prewarm.
After a correctness-valid warmup, global repeat walls were 1,134.37 and 689.40
seconds: 14.44 and 23.77 images/s, with a 19.10 median. Dividing the GPU medians
by that CPU median produces raw unmatched throughput ratios of 9.61x for the
full-projection 64-row anchor and 12.95x for the 1,024-row tuning point. The
aggregate report marks this comparison ineligible because required sidecar and
policy identities are absent, so neither ratio is a CPU-vs-GPU speedup. The
large repeat spread also means neither CPU repeat should be quoted alone. See the
[aggregate report](../../benchmarking/results/gpu_lance_column_fetch/scaling_report_cpu_2node.json).

### Large coordinate queue: prior locality leader

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
| Hypothetical scalar sparse calls avoided | 16,361 | **261,888** | Definition above; not physical reads avoided |
| Average physical read | 28.61 KiB | **82.33 KiB** | 2.88x larger |
| Physical reads, repeat spread | 140,318-142,429 | **528,923-529,100** | Cumulative Lance payload reads |
| Physical MiB/s, repeat spread | 61.14-65.22 | **181.12-183.00** | Legacy payload-fetch-span denominator |
| Physical operations/s, repeat spread | 2,175.1-2,347.9 | **2,254.8-2,274.2** | Legacy payload-fetch-span denominator |
| Peak host RSS | 9.72-9.97 GiB | 105.24-109.34 GiB | Explicit queue cost |

Both repeats returned all 262,144 rows in manifest order with no missing
payload and identical payload digest
`c83e219257a6b47f8adc2aea488f1c123c23f49ca4c15588049eb662f8da8ac3`.
One image exceeded Pillow's decompression-bomb safety threshold in both
repeats and was recorded as a safety skip, not a mismatch. The earlier
full-validation run used the old validator and is therefore only diagnostic
MD5 evidence; it checked all 262,144 MD5 values and directly
verified that row's URL, 18,217 by 12,138 dimensions, and MD5. The sanitized
[measured-evidence manifest](../../benchmarking/results/gpu_lance_column_fetch/measured_evidence_v1.json)
preserves both corrected repeats and their source artifact digest.

Even this prior leader averages only 82.3 KiB per physical read. It is substantially
less sparse, but it does not yet meet the multi-MiB read-size or 70-85% remote
sequential-ceiling acceptance criteria.

The queue result first supported the user's sparse-call hypothesis: the
improvement is consistent with a wider coordinate opportunity window rather
than a bigger private take, but the unmatched 16K/262K comparison does not
isolate causation.
The production queue should therefore drop URL strings
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
bottleneck; payload locality and object-store reads are. Its host-local result
is not published as a sanitized machine-readable artifact, so these values are
prose-only evidence.

### Full-left-fragment, fragment-major CPU fetch

A CPU-only remote-S3 locality diagnostic accumulated one complete real
production left fragment, resolved its URLs through the Arrow sidecar,
deduplicated 928,687 present image occurrences to 885,388 stable IDs, and
sorted those IDs into right-fragment/row-offset order. The left fragment
contained 3,998,698 rows; the selected images touched 42,442 of 56,696 right
fragments. The 3.85-MB coordinate Parquet has zero nulls, strictly increasing
stable IDs, nondecreasing fragment IDs, and strictly increasing offsets within
each fragment.

Slurm job `404464` completed successfully with state `COMPLETED`, exit code
`0:0`, and elapsed time 13m53s. Its exclusive `cpu_interactive` allocation had
64 CPU cores and no GPU. It used 32 Lance CPU threads, 64 Lance I/O threads,
4,096 IDs/private take, at most 16 pending takes, and 217 private calls per
repeat. That is 4,080.13 rows/call and 885,171 scalar sparse API calls avoided.
Both timed repeats used an image-only projection over pre-resolved,
fragment-major sorted coordinates. The 14.970-second left scan, 92.787-second
CPU sidecar join, and 256-row warmup were excluded from the payload-fetch
timing; this is payload-only evidence, not an end-to-end workflow result.

| Metric | Repeat 1 | Repeat 2 | Median |
| --- | ---: | ---: | ---: |
| Payload-fetch seconds | 302.708 | 276.763 | 289.735 |
| Payload-fetch images/s | 2,924.895 | 3,199.087 | **3,061.991** |
| Logical MiB/s | 384.776 | 420.846 | **402.811** |
| Lance read MiB/s | 413.989 | 450.329 | **432.159** |
| Physical reads/image | 1.28866 | 1.23208 | **1.26037** |
| Average bytes/read | 115,170 | 119,802 | **117,486** |
| Actual read operations/s | 3,769.20 | 3,941.53 | **3,855.36** |
| Read amplification | 1.075923x | 1.070056x | **1.072990x** |

Both repeats returned 885,388 rows, zero nulls, and the same
122,132,490,480 logical bytes. The sidecar join validated exact URL-set
coverage, and the coordinate audit validated stable-ID/fragment/offset order,
but this harness did not retain an output-order digest or validate payload
digests, ID-to-payload association, MD5, or decoded dimensions. Its source
field named `lance_read_iops` is a cumulative read count; the operations/s row
above divides that count by fetch time.

For a like-phase comparison, the earlier 262K queue fetched payloads at
1,117.146 and 1,127.136 images/s, so this run measured **2.729x higher payload
throughput** by median. Workload size, density, private-call size, hardware,
and isolation still differ, so this is not an algorithmic speedup. Dividing by
the 262K end-to-end 794.312 images/s produces 3.855x, while the corresponding
tuned/original 16K cross-phase ratios are 12.379x and 16.682x. Those additional
phase- and configuration-mismatched ratios are diagnostics, not speedups or
evidence that CPU is faster than GPU. In particular, this CPU-only run is not
a hardware comparison with the one-node eight-GPU geometry.

The median Lance read rate is 19.885% of the earlier 2,173.283-MiB/s remote
sequential lower bound. That ratio is a cross-node-class, non-isolated
diagnostic, not same-hardware storage utilization. Average reads are still
only about 115 KiB rather than multi-MiB, so storage-bandwidth saturation
remains unproven. Job `404464` overlapped the main remote sensitivity suite at
the user's direction; the throughput is measured but non-isolated, and the
direction of contention bias is unknown.

Eight evenly spaced real left fragments contained 871,022-1,175,735 present
image occurrences and 830,386-1,139,886 unique URLs. Their median values were
924,813.5 and 883,813. They touched 33,947-44,653 right fragments, or
59.88%-78.76% of the table; the medians were 41,450 and 73.11%. Their
sample-level **mean** occupancies ranged from 19.65 to 33.58 unique
images/touched fragment, with a median of 21.18. A within-fragment occupancy
median of 15 was recorded only for benchmark fragment 0, not as an
eight-fragment median.

At the sample median, 6B, 20B, and 100B references correspond to roughly
6,488, 21,626, and 108,130 left-fragment-equivalent queue units. That is a
planning conversion, not a measured corpus fragment count or runtime model.

This is now the strongest production-shaped measured locality signal. One full
left fragment is a coordinate scheduling unit, not a payload-output window:
this sample represented 113.7447 GiB of unique logical payload, and the harness
did not record peak RSS. Production should accumulate only compact Arrow
origin/stable-ID coordinates for the fragment, drop URL strings after
resolution, globally deduplicate and sort by `(fragment_id, row_offset)`, then
emit bounded image-only payload patches through spill/backpressure. Payload
bytes remain outside every shuffle.

Artifact provenance:

- Raw result: `/lustre/fsw/portfolios/llmservice/users/vjawa/interleaved-lance-benchmark/gpu_column_fetch/cpu_sorted_fetch/retain_all_v1_fragment_0000_take4096_pending16_r2_9f0316a.json`
- Sorted coordinates: `/lustre/fsw/portfolios/llmservice/users/vjawa/interleaved-lance-benchmark/gpu_column_fetch/cpu_sorted_fetch/retain_all_v1_fragment_0000_sorted_coords_9f0316a.parquet`
- Runtime script: `/home/vjawa/nemo-curator-mm-image-curation/.codex-runtime/cpu_sorted_lance_fetch_benchmark.py`
- Sanitized checked-in evidence: [CPU full-fragment evidence](../../benchmarking/results/gpu_lance_column_fetch/cpu_full_fragment_sorted_fetch_v1.json)

The checked-in evidence records the retained artifact paths as
non-authoritative locators alongside source hashes, repeats, correctness
scope, and limitations. The SHA-256 values, not path availability, bind the
evidence identity.

### Bounded full-fragment spool materializer

Exclusive non-array CPU allocation `404805` completed `0:0` after 1h51m and
ran the same 885,388 sorted unique stable IDs through the production
materializer. Each observation used image-only projection, 4,096 IDs/private
take, 16 pending takes, 217 private calls, a 1-GiB Arrow spool window, and
131,072-row position buckets. The two optimized local-fsync repeats are the
repeat-backed center of this table; the ordered baseline, tmpfs, and
`attempt_local` rows are single diagnostics.

| Implementation / spool | Materialize seconds | Images/s | Physical MiB/s | Reads/image | Average bytes/read | Amplification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordered scheduler, local ext3, fsync | 862.453 | 1,026.593 | 145.306 | 1.28885 | 115,154 | 1.07587x |
| Completion order, local ext3, fsync r1 | 699.121 | 1,266.431 | 179.255 | 1.28910 | 115,134 | 1.07589x |
| Completion order, local ext3, fsync r2 | 708.854 | 1,249.042 | 176.793 | 1.28898 | 115,144 | 1.07588x |
| Completion order, tmpfs, fsync | 655.374 | 1,350.967 | 191.219 | 1.28893 | 115,148 | 1.07588x |
| Completion order, local ext3, `attempt_local` | 651.383 | 1,359.243 | 192.391 | 1.28902 | 115,141 | 1.07588x |

The optimized local-fsync median is **1,257.736 images/s**, with a
1,249.042-1,266.431 spread, or 1.38% of the median. Relative to the one ordered
observation, materialize-time ratios span 1.217-1.234x and the median-time
ratio is 1.225x. The common private-take envelope improves 1.219x, from
850.175 seconds to a 697.628-second optimized median. The baseline has one
repeat and a slightly broader executor timing boundary, while commit
`49c19d80` combines completion-order scheduling with spool vectorization;
these are measured commit-level diagnostics, not a formal isolated speedup.

Remote geometry did not improve in this A/B: the optimized fsync repeats made
1,141,251-1,141,355 physical reads and transferred
131,407,705,704-131,408,706,311 bytes. The scheduler instead raised the
sum-of-call-duration/envelope occupancy proxy from 11.40 to 14.18-14.30 and
allowed 139-140 later calls to complete ahead of earlier pending calls. It
removed head-of-line idle time while preserving the 16-call retention bound.
Those measurements predate the first `781588bb` Lance-Ray stable-ID reader.
Job `407213` then exercised that ordered reader on the same 885,388 unique IDs
inside the full document graph and timed out before the payload/patch task
returned. Lance-Ray `9bb587be` restores bounded completion-driven production,
while Curator locates and validates each returned stable-ID interval before
deterministic scatter. Job `407235` reran the byte-identical coordinate
artifact and crossed into patch reconstruction within an 817.086-second
stage-entry upper bound. That confirms the scheduling direction, although the
driver did not receive final reader metrics because later patch reconstruction
timed out.

The local spool was an ext3 virtual block partition with rotational flag 1,
not NVMe. Tmpfs measured 1.074x the local-fsync median. Explicit
`attempt_local` measured 1.081x and matched tmpfs within 0.7%, supporting the
decision to omit durability fsyncs from a recomputable attempt-local spool.
Both are single, non-alternating observations under variable remote service,
so neither ratio is promoted to a speedup. The durable final Parquet artifact
still fsyncs and atomically publishes; spool hashes still detect live-attempt
corruption.

Every run returned all rows with zero null payloads and validated spool file
hashes. The optimized runs additionally prove complete unique synthetic
positions and exact stable-ID association despite out-of-order completion.
They do not prove payload identity: no payload digest, MD5, decode, actual
duplicate fan-out, real document-position distribution, or final document
patch was checked. Materialize timing also excludes final spool finish, the
separate 78.21-80.86-second validation pass, cleanup, and final patch output.
Peak process RSS was 16.14-16.16 GiB for local fsync, 20.14 GiB for tmpfs, and
20.64 GiB for `attempt_local`; tmpfs pages are additionally cgroup-charged
outside process RSS.

The fastest observation still reaches only 192.391 physical MiB/s, or 8.85%
of the 2,173.283-MiB/s sequential remote lower bound. Reads remain about
115 KiB rather than multi-MiB, and no concurrency plateau was measured.
Storage-bandwidth saturation therefore remains **unproven**; fragment/range
coalescing is still the primary remote-S3 objective.

The append-only [sanitized canary evidence](../../benchmarking/results/gpu_lance_column_fetch/full_fragment_spool_canary_evidence_v1.json)
binds all five raw artifact hashes, exact runner-source hashes, phase
boundaries, scheduler terminal state, correctness gaps, and the separate scale
supplement below.

### Public document graph setup canary

Job `405351` was one exclusive, non-array, non-requeue allocation with one
logical H100 and a shared 20-minute measurement cap. Its real RAPIDS-MPF 26.06
gate passed in **17.202 seconds**: two fractional actors formed ranks 0/1,
constructed operations 0/1, handled a nonempty plus empty rank and a globally
empty second window, returned the exact stable ID and document coordinate,
reused both operation IDs after shutdown, and cleaned up idempotently.

The subsequent public document graph passed the 64-CPU/one-GPU Ray gate and
created one document-fragment task, then failed during GPU actor setup. The
pinned sidecar was canonical `replicated_sorted` with one partition and no
`partitioning` field; the actor requested the multi-rank `hash_partitioned`
contract. Strict validation stopped before sidecar Parquet loading, document
row scanning, `_take_rows`, coordinate publication, spooling, or patch output.
The driver ran 52.411 seconds and the allocation ended `FAILED/1:0` after 98
seconds. Node-wide Ethernet moved only 14.02 MiB receive and 0.90 MiB transmit;
all sampled GPU SM and memory-bandwidth utilization was 0%. GPU 0 briefly
reserved 73,507 MiB, consistent with but not proof of RMM-pool initialization.

This is **setup-only evidence**. It provides no images/s, payload bandwidth,
read amplification, speedup, or storage-saturation result. The follow-up code
accepts replicated layout only for exactly one rank and one partition; all
multi-rank execution remains hash-only. No fifth short allocation was
submitted in that failed-job sequence. The checked-in
[setup evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_setup_canary_evidence_v1.json)
binds the terminal scheduler record, phase timings, telemetry, code/data
identity, and raw artifact hashes.

### Public document graph sidecar-load canary

The later job `405580` was a fresh exclusive, non-array, non-requeue
allocation pinned to Curator `63b22be`, Lance-Ray `ad763123`, the same image-v4
and document-v1 identities, and the canonical 355,952,746-row replicated
sidecar. Slurm accounted the exclusive physical node as 128 CPUs and eight
H100s while Ray was explicitly capped and verified at 64 CPUs and one logical
H100. The no-data RAPIDS-MPF gate passed in **16.645 seconds**, node setup
passed, and one document-partition task produced one output task. The actor
also accepted the one-rank/one-partition sidecar contract, closing the prior
layout failure.

The next actor-setup step called `cudf.read_parquet` over all 16 files in that
owned partition. Pylibcudf requested exactly **34,167,909,376 bytes
(31.821345 GiB)** and RMM rejected it because the maximum pool size was
exceeded. The driver failed after 100.085 seconds and Slurm recorded
`FAILED/1:0` after 145 seconds. The controller therefore exited before its
10-minute checkpoint: its cleanup record was written at 136 seconds with
`checkpoint_reached=false` and reason
`controller_exited_before_ten_minutes`. The 20-minute maximum-wall controller
was active but was not reached.

This failure occurred before document image-row scanning, `_take_rows`, a
coordinate plan, payload spooling, or patch publication. It has **no valid
images/s, payload-bandwidth, read-amplification, speedup, or storage-saturation
claim**. Actor teardown was not graceful: the setup exception left an
unfinished MPF shuffler, its shutdown aborted, and the executor force-killed
the actor. The outer Ray client still stopped, controller cleanup returned
zero, node-local Ray/spool removal returned zero, and neither plan nor output
root existed at cleanup.

The subsequent segmented loader replaced the monolithic decode. The next
canary measured completion of the broader actor-setup-to-UCXX envelope, while
pinned control flow implies that the resident index load returned inside that
envelope. That does not retroactively make this OOM attempt a performance
result. The checked-in
[sidecar OOM evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_sidecar_oom_evidence_v1.json)
binds the scheduler outcome, controller/checkpoint behavior, exact allocation,
passed gates, cleanup boundary, telemetry, and raw artifact hashes without
embedding credentials.

### Public document graph segmented-sidecar canary

Job `406706` was a fresh exclusive, non-array, non-requeue allocation pinned to
Curator `15005a04`, Lance-Ray `ad763123`, image v4, document v1, and the same
355,952,746-row, 16-file `replicated_sorted` sidecar. Ray was capped and
verified at 64 CPUs and one logical H100; the segmented loader used a 64-GiB
RMM pool and an 8-GiB MPF spill limit. The GPU shuffle stage started actor
setup at **00:44:56**, and the executor recorded **UCXX setup complete at
00:45:49**, a measured **52.979-second actor-setup-to-UCXX envelope**. Pinned
actor control flow loads and validates the segmented sidecar before
`setup_worker` returns, so the event sequence implies that the complete index
load returned inside that envelope. The load-completion statement is code-path
inference from the measured events, not an isolated sidecar-load timer. GPU 0
peaked at **78,959 MiB** framebuffer use.

The first document scan batch then failed while decoding its Lance row address:
cuDF 26.06 does not implement `Series >> 32`, so the expression raised
`TypeError` before any MPF chunk was inserted. The job therefore never reached
`insert_finished`, extraction, coordinate-plan publication, `_take_rows`,
payload spooling, or final patch publication. Exception teardown was not
graceful because RAPIDS-MPF destroyed the unfinished shuffler, but the outer
Ray client stopped, controller cleanup returned zero, and neither plan nor
output root existed. Slurm recorded `FAILED/1:0` after 2m47s. The controller
exited before its 10-minute checkpoint (`checkpoint_reached=false` at 159
seconds); the 20-minute cap was not reached.

This allocation has **no valid images/s, payload-bandwidth, read-amplification,
speedup, or storage-saturation claim**. The local follow-up decodes row
addresses with Arrow compute and an explicit uint64 shift scalar, including
high-bit fragment-ID coverage, and force-kills failed unfinished MPF actors
without invoking unsafe teardown. The 67 focused shuffle tests and eight
focused executor tests pass on CPU with zero failures. Those are code and
synthetic CPU test results only; job `406706` did not execute either fix and no
remote GPU runtime success is implied. The checked-in
[segmented-sidecar failure evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_segmented_sidecar_rowaddr_failure_evidence_v1.json)
binds the setup chronology, memory configuration, terminal scheduler record,
cleanup/checkpoint boundary, telemetry, local-fix test classification, and raw
hashes without embedding credentials.

### Current-head controller preflight attempts

Two fresh non-array allocations, `407157` and `407160`, pinned Curator
`437bea08` and Lance-Ray `ad763123` but stopped after nine seconds each during
placed-allocation validation. The first launcher used the wrong canonical
separator in Slurm's `TresPerNode` GPU field; the second compared Slurm's
canonical absolute `Command` with the runtime spelling of `BASH_SOURCE[0]`.
Both attempts passed the code, lock, driver, storage-options, and sidecar hash
gates, but neither created an application run root, started Ray or the MPF
smoke, scanned the document, or issued payload I/O. They therefore contain no
performance or correctness result and are not additional public-graph
canaries.

The local launcher now uses Slurm's observed `gres/gpu:1` spelling and the
reviewed absolute command. Shell syntax and a replay of every scheduler field
against allocation `407160` pass, but that correction has not run remotely.
No third allocation was submitted. The checked-in
[controller-preflight evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_controller_preflight_failures_v1.json)
records the exact boundary, terminal accounting, sanitized log hashes, and the
local-only correction classification.

### Sidecar-free storage-identity failure

Job `407202` was the first current-head application attempt after the
controller fixes. It was a fresh exclusive, non-array, non-requeue allocation
pinned to Curator `f3ad3128` and Lance-Ray `781588bb`, with a 22-minute Slurm
limit and a 20-minute controller cap. Slurm placed the 64-CPU/one-GPU request
on an exclusive 128-CPU/eight-GPU node; Ray was capped and verified at exactly
64 CPUs and one logical H100. The signed code, lock, nonsecret storage-options,
and sidecar hashes passed, as did the 16.835-second RAPIDS-MPF no-data smoke.

The first `LancePartitioningStage` actor then attempted to open document
version 1. Its manifest `HEAD` request returned 404, so the only partition task
produced no output. The driver failed after 33.665 seconds and Slurm recorded
`FAILED/1:0` after 77 seconds. Cleanup returned zero, and the controller's
early checkpoint confirms that neither the coordinate-plan root nor patch root
existed. The 10-minute checkpoint and 20-minute cap were not reached.

This is measured as an **unrelated ambient storage identity failure**, not as
proof that the pinned dataset is absent. The launcher established only that
ambient AWS variables existed; it did not bind that identity to the approved
PDX DataMover storage location. A read-only follow-up with the same ambient
identity and nonsecret storage options returned 404 for both document v1 and
image v4. A second read-only probe then applied the approved
`pdx-multimodal` identity and opened the exact document v1 dataset in 3.542
seconds and image v4 dataset in 35.556 seconds with PyLance
`9.0.0-beta.11`. It observed 6,960,284,974 document rows across 1,719
fragments and 355,952,746 image rows across 56,696 fragments, with stable row
IDs enabled on image v4. The 39.098-second metadata-only probe confirms the
job's manifest failure was caused by the unrelated ambient identity; it does
not prove that the later coordinate, payload, or patch phases will complete.

The allocation completed zero document partitions, zero coordinate plans,
zero private payload takes, and zero patches. Physical payload reads and bytes
were not measured because payload instrumentation was never reached. Job
`407202` therefore has **no valid images/s, payload-bandwidth,
read-amplification, speedup, or storage-saturation claim**. The checked-in
[storage-identity failure evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_storage_identity_failure_v1.json)
binds the requested and placed allocation, code hashes, passed gates, exact
failure boundary, terminal accounting, explicit non-results, and raw artifact
hashes without embedding credentials.

### Bounded ordered-reader full-fragment timeout

Job `407213` was the first application run with the approved
`pdx-multimodal` identity bound explicitly by the controller. It was one fresh
exclusive, non-array, non-requeue interactive allocation, requested 64 CPUs
and one H100, and advertised exactly those resources to Ray. The 22-minute
Slurm limit contained a 20-minute controller cap. Slurm ended `COMPLETED/0:0`
after 1,189 seconds because the controller handled its deadline; the
measurement itself exited `124` with terminal state `bounded_timeout`.
Cleanup returned zero and no job was left running.

The 15.256-second RAPIDS-MPF smoke passed. Document partitioning produced one
fragment task, with the task itself taking 4.87 seconds. GPU shuffle setup took
58.839 seconds and the coordinate window took 36.80 seconds. The retained
5,398,989-byte Parquet plan contains 928,687 non-null image occurrences,
885,388 unique stable IDs, 43,299 duplicate occurrences, zero missing IDs, and
928,687 unique document positions. This exactly matches the earlier
full-fragment coordinate diagnostic and proves that payload bytes stayed out
of the shuffle.

The single payload/patch task was still unfinished when SIGTERM arrived. Its
progress display had reached 16:28 and the stage had occupied about 1,005
seconds. `result.json` remained `running`; the reader never published final
metrics; no patch part or manifest was published. Therefore the run has no
valid images/s, completed payload bytes, private-call count, physical IOPS,
read amplification, payload digest, document correctness, or storage-ceiling
claim.

Allocation-wide telemetry observed 129.560 GiB received on `eth0`, 122.495
GiB maximum Slurm disk-read accounting, 119.723 GiB maximum disk-write
accounting, 165.888 GiB maximum RSS, and about 2.04 average CPU cores. Logical
GPU 0 averaged 0.273% SM utilization and retained its large framebuffer only
during the brief cuDF phase. These are node/step counters without Lance paths
or phase attribution. They are consistent with most or all payload bytes
moving, but they do not prove payload completion or storage bandwidth.

Code inspection found the ordered reader waited on futures by request index
and refilled only after Curator resumed its generator. Prior matched
materializer diagnostics measured a 1.219x private-take-envelope improvement
from completion-order scheduling with essentially unchanged I/O geometry.
That makes head-of-line removal the highest-confidence next change, not a
proven isolated cause for this timeout. Lance-Ray `9bb587be` now keeps a
bounded completion-driven producer and ready queue active; Curator validates
every returned interval and restores exact document order and duplicate
fan-out. The checked-in
[bounded-timeout evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_bounded_timeout_v1.json)
retains the measured phases, exact coordinate identity, terminal accounting,
resource-counter caveats, and raw hashes. The otherwise identical
completion-driven rerun is reported next.

### Completion-driven payload boundary and patch timeout

Job `407235` changed only the pinned Lance-Ray/Curator scheduling path for the
same production fragment: reads were produced in completion order behind a
bounded running/ready queue, and Curator validated each returned stable-ID
interval before deterministic fan-out. The coordinate Parquet file and its
canonical digest are byte-for-byte identical to job `407213`: 928,687 image
occurrences, 885,388 unique IDs, and 43,299 duplicate occurrences. The run
used the same one-node, 64-Ray-CPU/one-H100, 4,096-row, 16-pending,
image-only, one-GiB attempt-local-spool geometry and the same 20-minute
measurement cap.

Pinned control flow makes the observed boundary meaningful. A patch part can
be written only after `materialize_lance_payload_to_spool` has exhausted the
reader, validated exact unique-ID and operation coverage, finished the
occurrence spool, and returned to document reconstruction. The first patch
part appeared no later than 817.086 seconds after payload-stage entry. This is
a conservative lower bound of **1,083.59 unique images/s** and **1,136.58
logical occurrences/s**, because the denominator also includes actor setup
and initial reconstruction/part writing. The ordered job had not reached this
boundary after 1,005.383 seconds, so the completion-driven change improved
time-to-patch-entry by **at least 1.230x**. This is a one-observation censored
boundary comparison, not a repeat distribution or final end-to-end speedup.

The next bottleneck is reconstruction. Five temporary Parquet parts totaling
5,078,519,614 bytes were present at SIGTERM, but no final manifest or patch
artifact was published. The driver therefore could not return final Lance
IOTracker metrics or independently verify payload digests, complete document
row/order conservation, or output hashes. Allocation-wide telemetry observed
129.608 GiB `eth0` receive, 131.766 GiB Slurm read accounting, 123.488 GiB
write accounting, 179.942 GiB peak RSS, about 2.22 average CPU cores, and
0.209% mean logical-GPU SM utilization. These pathless counters do not prove
physical IOPS, average read size, amplification, or storage saturation.

The inner measurement timeout fired correctly, but the 1,180-second outer
watchdog then won a cleanup race, so Slurm recorded `FAILED/124:0` and the
controller did not publish cleanup/exit markers. The five-part remote attempt
and lock were removed manually after recording their metadata; node-local
cleanup could not be verified after allocation release. The local controller
now retains the sub-20-minute measurement deadline while moving its outer
cleanup watchdog to 1,300 seconds, inside the 1,320-second Slurm limit. That
controller-only fix was not rerun. The checked-in
[completion-driven evidence](../../benchmarking/results/gpu_lance_column_fetch/real_document_patch_completion_order_timeout_v1.json)
separates the proved payload boundary, conservative rate, incomplete patch,
resource caveats, cleanup race, and raw hashes. Per the bounded experiment
policy, the follow-on run changed the product boundary instead of retrying the
same full-document patch; that durable-overlay result is reported next.

### Durable full-fragment payload overlay

Job `407257` ended the workflow at the remote-fetch product boundary instead
of entering document reconstruction. It reused the byte-identical coordinate
plan from jobs `407213` and `407235`: 3,998,698 document rows, 928,687 image
occurrences, 885,388 unique stable IDs, and 43,299 duplicate occurrences. The
run projected only `image`, used 4,096 stable IDs per private take with at most
16 pending, retained a one-GiB actual-Arrow-byte spool target, requested one
H100 and 64 CPUs on one exclusive node, and had a 10-minute checkpoint plus a
20-minute measurement cap. Slurm completed `0:0` in 18:52 without an array,
restart, retry, or competing experiment from this work.

| Boundary | Seconds | Unique images/s | Logical images/s | Physical MiB/s |
| --- | ---: | ---: | ---: | ---: |
| Remote fetch + duplicate scatter + fsync Arrow writes + part hashes | 809.108 | **1,094.277** | **1,147.792** | **154.886** |
| Fully validated durable overlay process | 1,072.144 | **825.810** | **866.196** | N/A |

The second boundary additionally includes plan/document validation, outer
manifest publication, a full sequential local reread of every part, and the
atomic directory rename. That work added 263.037 seconds. The driver then
spent 19.024 seconds comparing six evenly spaced overlay rows against pinned
image v4; this sample check is outside both table denominators.

Lance recorded 131,406,638,504 physical bytes in 1,141,129 reads: **1.28885
reads per unique payload**, **112.46 KiB/read**, and **1.07587x** byte
amplification. The 217 private takes avoided 885,171 hypothetical scalar API
calls, or 99.9755%, while 191 batches completed out of request order. All
885,388 unique payload rows and all 928,687 logical occurrences were conserved
with zero nulls. The final artifact contains 3,720 fsynced Arrow parts,
127,623,021,855 Arrow bytes (118.858 GiB), no oversized rows, and the exact
coordinate-plan digest.

Publication correctness is stronger than the earlier boundary inference.
Before rename, the stage validated every part's size, SHA-256, Arrow schema,
bucket membership, row count, and coordinates, then recomputed the canonical
coordinate digest. The final manifest and payload manifest are independently
hashed. Source payload identity was checked for six sampled rows across three
evenly spaced parts; that is a sample, not a full source-side digest of all
images.

The result is still not storage-bandwidth constrained. Payload-wall physical
throughput is only 7.13% of the 2,173.283-MiB/s sequential physical lower
bound; read-envelope throughput is 171.474 MiB/s, or 7.89%. Average reads are
about 112 KiB rather than multi-MiB, and no concurrency plateau was measured.
All 42,442 touched right fragments remained on the sparse strategy at the
observed 0.249% global coordinate density.

Allocation telemetry must be interpreted carefully. Exclusive placement made
Slurm reserve the complete eight-H100 node even though the application
requested one GPU; this run reused pre-resolved stable IDs and sampled 0% GPU
SM utilization, so it is a payload-overlay result rather than an end-to-end
GPU-lookup rate. The payload phase reported 33.78-GiB process peak RSS, while
the complete Slurm step peaked at 138.96 GiB during the validation-inclusive
run. Slurm also recorded 238.10 GiB read and 119.72 GiB write, consistent with
writing the overlay and rereading it for validation. Reducing that local
validation memory/pass cost is useful, but it does not replace the primary
remote objective.

The implementation gap for cross-plan locality is now closed. The public graph
uses a batch-aware positional `N -> N` overlay stage: one 64-CPU actor accepts
up to 64 plans, adopts already-published members, semantic-sorts only pending
members, greedily partitions them under a conservative coordinate-workspace
bound, performs one global stable-ID dedupe/sort and completion-driven fetch per
subgroup, and scatters into independently checkpointable overlays under one
shared payload-spool budget. Physical I/O metrics live in one hash-bound fetch
group rather than being copied into every member artifact. Synthetic tests
cover cross-plan duplicates, completion-order output, all-null plans, partial
publication and retry, coordinate preflight, shared spool accounting, and
legacy singleton adoption. These are correctness results, not remote speedups.

The remaining product evidence gap begins with one current-head, capped
remote-S3 grouped-materializer canary. The frozen candidate workload is the
existing 262,144-row real manifest with 64 Arrow task tables of 4,096 rows,
file digest `61fcbb3942900c58dc40e3cf7e91cc1c73956dc96d30b6834345f848ba3b9f05`,
logical digest `2c97926f37a43349c9510aa7bee1cb42e771571c47424fda72e88ff8129fb1d3`,
and ordered length-framed stable-ID digest
`238cac2db2302f097220c69b1a6d6558c2a344fc69466fd52c5cc7ba547f2d13`.
Those blocks contain no cross-block duplicate IDs, so the run measures global
sorting/locality and 64-way scatter, not cross-plan dedupe savings. The existing
generic benchmark does not exercise the new overlay path, and the manifest is
not a set of document-bound coordinate artifacts. The checked-in
`gpu_lance_grouped_payload_canary.py` driver validates those pins before remote
setup, constructs the exact 64 synthetic coordinate plans, and streams each
spool through the historical payload oracle without concatenating the output.
The result must be labeled a grouped materializer canary, not a
production-overlay A/B. It gets a 10-minute progress check and 20-minute hard
cap. Any per-plan baseline is a separate capped run; noncompletion is a
censored bound, never an exact throughput or speedup. No Lustre mirror or
multi-node scaling run precedes this gate.

The first current-head allocation, job `407335`, did not reach this gate.
Slurm marked the exclusive interactive allocation running, but after 7 minutes
31 seconds it had created neither batch stdout/stderr nor the wrapper's signed
preflight run root. For comparison, the five preceding bounded canaries created
their run roots 8-24 seconds after scheduler start. The allocation was cancelled
to avoid idle full-node resource waste. It produced no checkpoint, spool,
driver result, remote payload read, throughput, or correctness evidence and was
not automatically resubmitted. The
[sanitized scheduler record](../../benchmarking/results/gpu_lance_column_fetch/grouped_payload_canary_attempt_407335.json)
is infrastructure evidence only; the candidate canary remains unmeasured.

The sanitized
[durable-overlay evidence](../../benchmarking/results/gpu_lance_column_fetch/real_payload_overlay_canary_v1.json)
contains the exact timing, I/O, correctness, scheduler, resource, and raw-file
hashes. Cross-workload ratios to the 16K and 262K diagnostics are retained
there only as explicitly ineligible context, not as speedups.

### Fragment fixed-cost amortization

A 16-fragment nested-contiguous control kept 16 private calls in flight while
increasing useful rows per fragment. Shared-node and network contention makes
its images/s unsuitable as a speedup denominator, but its physical-call curve
is direct locality evidence. The retained host-local results are not published
as a sanitized machine-readable artifact, so this subsection is prose-only:

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
| `DEFERRED(exclusive-scaling)` | Submit a fresh exclusive 1/2/4/8-node latency sweep only after one node reaches a storage/concurrency plateau | Identical payload projection, immutable sidecar, read/concurrency/cache/validation policy, package/code identity, global manifest partitioning, exact rank set, and common Slurm run identity |
| `DONE(one-node-remote-sweep)` | One exclusive eight-H100 node completed 8/4/2/1-wave points with two correct repeats each | 4/8 waves meet the primary scheduling policy; only eight waves has exactly 64 harness task tables per wave, 1/2 waves remain diagnostics, and no point reached the remote sequential lower bound or multi-MiB reads |
| `DEFERRED(storage-saturation-scaling)` | Extend compliant weak scaling only after the cross-plan one-node A/B | Hold per-node work constant, require exact actor/rank sets and repeat spread, and keep remote S3 primary; do not scale an IOPS-bound per-plan path |
| `DONE(one-big-private-call-negative)` | One 16,384-ID private call ran for more than eight minutes and was stopped | Do not use a giant-call production path; retain 1,024/16 and widen the coordinate queue |
| `DONE(queue-rerun)` | Corrected 262,144-row image-only queue completed two repeats at 785.56-803.06 images/s | Preserved 1,024/16; both repeats correct with identical digest, 2.018 reads/image, and 1.241x amplification |
| `DONE(full-left-fragment-locality)` | CPU-only fragment-major fetch completed two non-isolated payload repeats at 2,924.90-3,199.09 images/s | 885,388 returned rows, zero nulls, repeated logical byte count, and coordinate-order checks; no payload identity/order digest, MD5, decode, RSS, or hardware-speedup claim |
| `DONE(full-fragment-spool-materializer-canary)` | One ordered baseline, two optimized local-fsync repeats, one tmpfs ceiling, and one `attempt_local` observation completed in non-array job `404805` | Exact coordinate/stable-ID placement, zero nulls, spool hashes, I/O, RSS, sync mode, and phase boundaries are retained; synthetic positions and no payload/final-document digest keep this diagnostic-only |
| `DONE(arrow-native-payload-fanout)` | Replace replicated-stage Python payload dictionaries/lists with Arrow tables and stable-ID `index_in`/`take` | Behavior is covered synthetically; rerun the exact 262K manifest before claiming an RSS or throughput improvement |
| `DONE(mpf-bounded-private-takes)` | Split each MPF coordinate window into 1,024-row private takes with at most 16 pending | Preserves sorted stable-ID order and reports pending depth and physical operations/s; no real MPF speedup claim yet |
| `DONE(mpf-2606-no-data-lifecycle)` | Run the real two-rank MPF 26.06 communicator and both shufflers through nonempty/empty windows on one H100 | Job `405351` passed exact return, global finish, bulk wait/local extraction, operation-ID reuse, and idempotent cleanup in 17.202 seconds; setup-only, no payload rate |
| `DONE(mpf-coordinate-plan-contract)` | Publish one compact plan per deletion-free left Lance fragment after GPU resolution | Deterministic Arrow schema/order, exact dataset/sidecar/fragment identity, atomic no-overwrite publication, and fail-closed adoption are covered synthetically |
| `DONE(local-payload-spool-primitive)` | Buffer Arrow payload rows into deterministic node-local IPC buckets under an actual-byte target | Synthetic tests cover conservation, tamper rejection, bounded normal rows, isolated oversized rows, `fsync`/`attempt_local`, and explicit cleanup; remote-v4 materializer observations confirm the 1-GiB bound and 131-135-file geometry for synthetic contiguous positions |
| `DONE(durable-payload-overlay)` | End the remote fetch path at a checkpointable Arrow artifact keyed by document position | Full validation occurs before atomic rename; manifest identity binds pinned inputs, filtered coordinate digest, schema/layout, exact logical/unique/null counts, part hashes, and reconciled producer metrics. Local tests cover completion-order output, duplicate fanout, all-missing fragments, corruption, exact inventory, post-rename adoption with zero image requests, and stable source IDs |
| `DONE(file-backed-full-fragment-patch-stage)` | Consume one coordinate plan, fetch unique image-only payloads, reconstruct the complete document fragment, and publish bounded deterministic Parquet patches | Remote v4 covers the materializer/spool boundary; synthetic tests cover actual duplicate fan-out, row/sample order, full patch publication, failure cleanup, stale-attempt reaping, and exact retry adoption; a real final-document patch canary remains required |
| `DONE(public-document-materializer-graph)` | Export one runnable source -> GPU coordinate shuffle -> payload overlay composite and tutorial | Defaults to the durable overlay boundary, enforces one fragment per source task, consistent document/image identities and read geometry, coordinate-only MPF traffic, and one 64-plan/64-CPU grouped overlay actor per 64-CPU node; combined document-patch behavior remains an explicit compatibility mode |
| `DONE(checkpointed-coordinate-replay)` | Enumerate existing coordinate plans as deterministic source tasks for a second patch pipeline | The public replay CLI requires the exact expected fragment inventory and rejects missing/stray/duplicate fragments; the reader also validates exact artifact bytes and all optional dataset/sidecar pins before retry adoption |
| `DONE(checkpointed-overlay-replay)` | Enumerate durable overlays as deterministic source tasks | The reader validates exact inventory, optional dataset/config pins, hashes and coordinates by default, accepts valid zero-part overlays, and derives task IDs from semantic identity instead of physical paths or completion-order part layout |
| `DONE(segmented-sidecar-setup-envelope)` | Bound replicated-sidecar decode staging and measure the actor-setup envelope | Job `406706` measured 52.979 seconds from actor-setup start through UCXX setup with a 64-GiB RMM pool, 8-GiB spill limit, and 78,959-MiB peak GPU framebuffer; pinned control flow implies the segmented load returned inside that envelope, but this was not an isolated load timer and no payload rate exists |
| `DONE(completion-driven-stable-id-reader)` | Keep sparse Lance reads full behind a bounded ready queue while consuming results in completion order | Lance-Ray and Curator tests cover head-of-line avoidance, refill during consumer pauses, exact interval coverage, deterministic fan-out/order restoration, retention bounds, partial close, failure cleanup, and retry. Job `407235` crossed into patch writing within 817.086 seconds, at least 1.230x sooner than the censored ordered boundary; final reader metrics were not returned |
| `DONE(real-overlay-canary)` | Publish the exact full-fragment remote payload as the new durable overlay boundary | Job `407257` published 885,388 unique / 928,687 logical rows as 3,720 fully validated parts, persisted exact I/O metrics, exited `COMPLETED/0:0` in 18:52, and performed no document rewrite |
| `DONE(cross-plan-coordinate-window-implementation)` | Aggregate up to 64 coordinate plans before shared payload fetches | Exact positional `N -> N` outputs, pending-only global stable-ID dedupe/sort, deterministic byte-bounded subgroups, one shared spool budget, hash-bound group I/O metrics, and partial-publication retry are covered by local tests; no remote speedup claim yet |
| `ATTEMPTED(scheduler-startup-failure)` | Run the frozen 64 x 4,096 real-row manifest through one current-head grouped materializer | Job `407335` never opened batch logs or created the signed preflight root and was cancelled after 7:31 with no remote read or result. The checked-in driver still passes exact local preflight; do not infer throughput or automatically resubmit |
| `DEFERRED(cross-plan-matched-baseline)` | Run the same frozen workload without the global queue in a separate capped job | Report noncompletion as censored evidence and never divide it into a speedup; do not delay the grouped canary or submit a long combined A/B allocation |
| `DEFERRED(real-final-document-patch-canary)` | Reconstruct and publish the actual document from a previously durable overlay | The durable overlay already supplies page-position payload access. A separate consumer may be added later without another remote fetch; it is not on the remote-read critical path |
| `DONE(nested-scaling-manifest-tooling)` | Derive atomic 1/2/4/8-node task-prefix families from one validated eight-node master scan | Validate master and actor-shard hashes, exact prefix digests, modulo actor assignment, and fail without partial publication |
| `DONE(exact-cross-arm-digest-binding)` | Bind every derived comparison to ordered query-manifest and stable repeat-output digests | Same-label runs with different inputs or outputs must never produce a ratio |
| `DONE(projection-ab)` | Image-only, image+URL, and full projection on the exact 16,384-row manifest | Two repeats each; identical payload digest; image-only removed 69.1% of full-projection reads |
| `DONE(prose-only-pinned-io-trace)` | Traced 2,881 post-coalescing reads on pinned PyLance `0b82051` | The checked-in report retains the numeric summary, but no sanitized machine-readable trace artifact; 4-KiB cross-request merge remains an unmeasured runtime experiment |
| `UNRESOLVED(required-naive-baseline)` | Produce a correctness-valid naive PyLance payload rate for the requested cuDF speedup denominator | The corrected harness runs scalar lookup/take work in deterministic 64-left-table waves and restores manifest order, but has no current-head repeat. Retain the old timeout evidence, enforce the 10-minute check and 20-minute hard cap, and do not treat implementation as a completed comparison |
| `DEFERRED(cpu-baselines)` | Rerun CPU 1/2/4/8 actor sweeps only after the matched 1,024-row three-arm comparison | First match `url,image` across arms; hold total per-node rows and aggregate I/O bounds constant while actor count changes; report setup, steady state, telemetry, and spread |
| `TODO(fragment-local)` | Sweep bounded sorted private-call window sizes and run the MPF stable-ID return path | Do not restore public fragment compatibility; report IDs/private call, I/O operations/image, read amplification, throughput, and peak memory |
| `TODO(hybrid-density)` | Add immutable payload-size metadata or validate a density estimator for variable-size images | Enforce a true hard payload-byte cap and compare it with the current explicit estimated-byte profiles |
| `DONE(unique-payload-stream-api)` | Yield stable-ID-sorted unique payload batches without whole-queue concatenation or payload fan-out | Local tests prove deterministic order under out-of-order completion, exact dedupe/digest, bounded retained batch count, and Ray iterator semantics; remote throughput and actual-byte bounds remain unmeasured |
| `DONE(sidecar-free-stable-id-reader)` | Move pre-resolved stable-ID payload reads into Lance-Ray without constructing another cuDF index | The reader is pinned to exact dataset/version/row/fragment order, streams Arrow batches with bounded pending work, preserves schema metadata, rejects concurrent iterators, drains partial reads, and reports read-only versus consumer wall time; Curator integration has 56 local tests, while remote throughput and head-of-line impact remain unmeasured |
| `DONE(bounded-patch-actor-admission)` | Prevent 64 fragment tasks from creating 64 independent patch actor queues on one 64-CPU node | Patch actors reserve eight Ray CPUs by default, yielding at most eight actors/node; an optional global cap and estimated per-actor reservation are exposed, but variable payload bytes remain unbounded |
| `DONE(public-lance-ray)` | Persistent public API ran 257.69-265.82 images/s with the matching digest | Retain the completed **full-validation projection** run and full repeat spread as the public-API baseline; the separate image-only A/B session ran 310.81-314.41 images/s |
| `DONE(measured-ray-data-persistent-control)` | Retain the completed four-wave persistent GPU actor control and its offline-revalidation claim | The sanitized summary records arm-specific counter repair, exact cross-arm output/payload digests, and zero payload rereads; the complete schema-v3 terminal family is not checked in, so eligibility is not independently revalidated here |
| `UNRESOLVED(required-public-ray-data-source)` | Produce a correctness-valid public `lance_ray.read_lance` payload rate for the requested Ray Data comparison | Filtered planning no longer performs a duplicate driver-side predicate count, and the harness pins cache-disabled `override_num_blocks` plus concurrency. Its payload warmup still has not run; the required follow-up remains isolated and capped at a 10-minute check and 20-minute hard stop |

No CPU-vs-GPU or naive-vs-GPU speedup should be quoted until the corresponding
row passes these gates. The four-wave persistent-actor control is the only
currently complete and digest-bound Ray-vs-GPU comparison, but its 1.399x
driver-wall ratio includes asymmetric actor lifecycle and is not an
algorithmic speedup. The public DataSource payload rate remains a required
unresolved comparison after the bounded timeout; the failed attempt is retained
as timeout evidence rather than treated as a completed baseline.

In particular, the older globally split 64-table jobs answer only a small
fixed-work latency question. They must not be cited as evidence that the
storage ceiling, actor count, or queue depth has saturated.

## 6B, 20B, and 100B+ reporting model

[`gpu_lance_scale_model.py`](../../benchmarking/scripts/gpu_lance_scale_model.py)
produces a machine-readable JSON report and Markdown tables. It treats 6B,
20B, and 100B+ as image-reference/probe counts. The 100B scenario is the lower
bound represented by exactly 100B references, not a claim about an upper bound.

Evidence labels for the current report are explicit:

| Quantity | Evidence class | Permitted use |
| --- | --- | --- |
| 262K queue repeat rates and I/O counters | Measured, two correctness-valid remote repeats | Current queue-diagnostic model input |
| One full-left-fragment 3,061.991 images/s | Measured, non-isolated, payload-fetch-only diagnostic | Locality and scheduling evidence only; no CPU/GPU or corpus-runtime claim |
| Full-fragment spool 1,249.042-1,266.431 images/s | Measured, two remote materializer repeats with synthetic positions | Fetch-to-spool scheduling and resource sensitivity; no final-document or payload-identity claim |
| Eight-left-fragment occurrence and locality ranges | Measured sample geometry | Queue-sizing sensitivity; not proof of the full-corpus distribution |
| 6,488 / 21,626 / 108,130 fragment-equivalent queues | Derived planning conversion | Reference counts divided by the measured eight-fragment median; not measured fragment counts or runtimes |
| 6B / 20B / 100B+ bytes, node counts, and runtimes | Modeled extrapolation / hypothesis | Capacity exploration only until exclusive bandwidth-saturated scaling validates the rate curve |

Neither full-fragment rate replaces the 262K runtime-model input. The
3,061.991-images/s harness omitted payload identity, output-order digests, RSS,
and reconstruction and overlapped another suite. The spool canary records RSS
and reconstruction coordinates but still starts after lookup/plan creation,
uses synthetic positions, omits payload identity and duplicate fan-out, and
stops before final document reconstruction. Using either as a corpus-runtime
anchor would turn a locality signal into an unsupported scaling claim.

### Full-fragment materializer sensitivity model

The sanitized canary evidence includes a separate sensitivity model using only
the two optimized local-fsync repeats. It preserves the existing inventory
ratios, assumes one payload read per reference with no cross-window reuse, and
keeps two 12-byte coordinate passes separate from payload bytes.

| Scenario | References | Modeled unique keys | Coordinate bytes | Image value bytes | Physical read bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6B | 6.000B | 1.350B | 144 GB | 0.828 PB | 0.891 PB |
| 20B | 20.000B | 4.499B | 480 GB | 2.759 PB | 2.968 PB |
| 100B+ | 100.000B | 22.497B | 2.400 TB | 13.794 PB | 14.842 PB |

The next table is idealized min / median / max days across the two measured
repeats. Each node contributes exactly one materializer; node scaling is simple
division and has not been measured.

| Scenario | 1 node | 2 nodes | 4 nodes | 8 nodes |
| --- | ---: | ---: | ---: | ---: |
| 6B | 54.83 / 55.22 / 55.60 | 27.42 / 27.61 / 27.80 | 13.71 / 13.80 / 13.90 | 6.85 / 6.90 / 6.95 |
| 20B | 182.78 / 184.05 / 185.33 | 91.39 / 92.03 / 92.66 | 45.70 / 46.01 / 46.33 | 22.85 / 23.01 / 23.17 |
| 100B+ | 913.91 / 920.27 / 926.64 | 456.96 / 460.14 / 463.32 | 228.48 / 230.07 / 231.66 | 114.24 / 115.03 / 115.83 |

These runtimes omit lookup, plan construction, real duplicate fan-out, final
document writes, validation, retries, skew, and failures. Index capacity is a
separate constraint: the existing resident cuDF model requires at least 1, 2,
and 6 eight-H100 nodes for 6B, 20B, and 100B+ respectively. The table is a
hypothesis for planning one materializer per node, not a scaling result or SLA.

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
mode, concurrency, cache, validation, and package identity. The checked-in v3
artifacts contain an empty `source.runtime_identity.code` object, so exact code
commit identity is unavailable and the model remains diagnostic. The legacy
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
throttling scenario. Payload-rate anchors must ultimately come from a
bandwidth-saturated per-node geometry. The completed one-node wave sweep uses
the right geometry but is not storage-saturated, and the older globally split
64-table jobs do not use that geometry at all. Using either as a final linear
node-scaling anchor would extrapolate an IOPS-sensitive result rather than a
storage-bound steady state.

### Queue-diagnostic scenarios

The clean v3 model uses the two correct 262K image-only repeats: 785.56-803.06
images/s end to end, 1,117.15-1,127.14 images/s fetch-only, and 2.0177-2.0184
physical reads/payload. Payload reads conservatively equal image references;
this is an explicit no-cross-window-reuse upper bound, not measured corpus-wide
I/O.

| Scenario | Unique keys | Logical payload | Physical read bytes | Minimum index nodes | Modeled 8-node runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6B | 1.35B | 748 TiB | 928-929 TiB | 1 | 1.68-1.72 days |
| 20B | 4.50B | 2.43 PiB | 3.02 PiB | 2 | 5.61-5.73 days |
| 100B+ | 22.5B | 12.2 PiB | 15.1 PiB | 6 | 28.0-28.7 days |

Every runtime in this table is a `queue_diagnostic_extrapolation` derived from
one deterministic 262K manifest and its two-repeat execution spread; the range
does not include corpus-sampling uncertainty. It assumes 80-GiB H100s, 80%
usable index memory, eight GPUs per node, and 64 active readers with 0.8
marginal-reader efficiency, yielding a 51.4x modeled rate multiplier rather
than 64x. It is neither an SLA nor evidence of linear node scaling. Sparse-call
sensitivity reports call counts only; it does not invent a runtime improvement
from rates measured in the same run.

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
  --naive-concurrency 64 \
  --ray-concurrency 64 \
  --ray-override-num-blocks 64 \
  --ray-worker-dataset-cache-size 0 \
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

Do not use the all-arm example for an unbounded public-baseline probe. Run
`naive_pylance_scalar` and `lance_ray_datasource` as separate processes, check
at 10 minutes, and stop at 20 minutes. On a fully indexed pinned snapshot the
optimized public probe may opt into `--public-index-fast-search`; the harness
first verifies exact row and fragment coverage and fails closed otherwise.
The process boundary is the reliable timeout boundary for native PyLance calls:

```bash
timeout --signal=TERM --kill-after=120s 1200s \
  python benchmarking/scripts/gpu_lance_column_fetch_benchmark.py \
  ... \
  --public-index-fast-search \
  --naive-concurrency 64 \
  --warmup-count 1 \
  --repeat-count 2 \
  --arm naive_pylance_scalar
```

Replace the final arm with `lance_ray_datasource` for the isolated Ray Data
probe, add `--ray-concurrency 64 --ray-override-num-blocks 64
--ray-worker-dataset-cache-size 0`, and give it a unique `--ray-temp-dir` and
output path. A timeout is a terminal bound, not permission to derive a repeat
rate from partial work.

Here `fetch_batch_size=1024` and 16 pending takes reproduce the measured remote
default. Use 64 only to reproduce the original anchor. Widen locality with the
coordinate queue (`task_rows * coalesce_tasks`), not by replacing medium takes
with one giant private call, and report queue bytes plus peak host memory.

For 1/2/4/8-node scaling, scan the document source once for the eight-node
master, then derive an atomic task-prefix family. This keeps every smaller
workload an exact prefix instead of changing the fragment sample at each scale:

```bash
python benchmarking/scripts/generate_gpu_lance_saturation_manifest.py \
  --preset eight-node \
  --output-dir /path/to/eight-node-master \
  --document-uri "$DOCUMENT_LANCE_URI" \
  --document-version "$DOCUMENT_LANCE_VERSION" \
  --storage-options-json @/path/to/nonsecret-storage-options.json

python benchmarking/scripts/derive_gpu_lance_scaling_manifests.py \
  --master-manifest-dir /path/to/eight-node-master \
  --output-root /path/to/nested-scaling-family
```

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
- [Nested scaling-manifest derivation](../../benchmarking/scripts/derive_gpu_lance_scaling_manifests.py)
- [Offline terminal-family revalidation](../../benchmarking/scripts/gpu_lance_revalidate_terminal.py)
- [Remote sequential-ceiling runner](../../benchmarking/scripts/gpu_lance_remote_sequential_ceiling.py)
- [Scaling report builder](../../benchmarking/scripts/gpu_lance_scaling_report.py)
- [Scale model](../../benchmarking/scripts/gpu_lance_scale_model.py)
- [RAPIDS-MPF shuffle architecture](https://docs.rapids.ai/api/rapidsmpf/stable/background/shuffle-architecture/)
- [RAPIDS-MPF streaming execution](https://docs.rapids.ai/api/rapidsmpf/stable/background/streaming-engine/)
- [RAPIDS-MPF configuration](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/)
- [cuDF `DataFrame.merge`](https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.dataframe.merge/)
