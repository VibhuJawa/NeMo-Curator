# GPU Lance document image fetch

This tutorial resolves image URLs from a pinned interleaved document Lance
table against a local cuDF sidecar for a pinned image Lance table. RAPIDS-MPF
shuffles only URLs, stable row IDs, and document coordinates. Image payloads are
read directly from the remote image table into bounded Arrow IPC parts, then
published as identity-bound overlays keyed by document position.

Install the isolated GPU Lance environment described in
[`nemo_curator/stages/interleaved/README.md`](../../../nemo_curator/stages/interleaved/README.md),
provide object-store credentials through the environment or node identity, and
start with one deletion-free document fragment:

```bash
python tutorials/interleaved/gpu_lance_document_materialization/main.py \
  --document-uri s3://bucket/documents/dataset \
  --document-version 1 \
  --image-uri s3://bucket/images/dataset \
  --image-version 4 \
  --index-shard /local/image-index/partition-000.parquet \
  --index-manifest-uri /shared/image-index/manifest.json \
  --index-manifest-sha256 '<caller-pinned lowercase SHA-256>' \
  --coordinate-plan-output-path /shared/image-plans/canary \
  --output-root /shared/document-image-overlays/canary \
  --materialization-mode payload_overlay \
  --fragment-id 0 \
  --num-gpus 1
```

Use eight sidecar partitions and eight GPUs for one full GPU node. The default
`fetch_task_window=8` then schedules up to 64 left fragments per node while
retaining `1024` stable IDs per private Lance call and at most `16` pending
calls. Increase the coordinate window before increasing private-call size.

The durable-overlay stage reserves 64 Ray CPUs by default, so a node exposing
64 Ray CPUs admits one actor. That actor batches up to 64 coordinate plans,
globally sorts and deduplicates their stable IDs, and scatters one shared fetch
back into independent checkpoint artifacts. Its default controls are a 4-GiB
retained Arrow coordinate bound and a shared 1-GiB active payload-spool budget;
accepted profiles are `256MiB`, `1GiB`, and `4GiB`. Set
`--payload-overlay-workers` only when the pipeline needs an explicit
cluster-wide actor count; leaving it unset lets task count and resources size
the pool. Opaque Arrow-kernel scratch is covered by process RSS rather than the
coordinate bound. The retained private-take estimate uses the reader's full
`2 * pending + 1` batch contract but remains row based because variable-size
images are unknown before I/O.
Each payload actor keeps one sidecar-free Lance-Ray stable-ID reader for its
lifetime. The GPU URL index remains in the coordinate stage; payload actors do
not load it again, and image bytes move directly from bounded Arrow read batches
into the durable overlay.

All artifact roots must be absolute. Coordinate plans and overlays belong on
shared durable storage.
Never put credentials in command-line arguments, manifests, or result paths.
The full coordinate collective requires `RayActorPoolExecutor` and is not a
checkpointed phase. Existing coordinate plans can be replayed through
`LanceCoordinatePlanReader` and `LanceCoordinatePayloadOverlayStage` in a
separate checkpointed pipeline:

```bash
python tutorials/interleaved/gpu_lance_document_materialization/fetch_existing_plans.py \
  --plan-root /shared/image-plans/canary \
  --document-uri s3://bucket/documents/dataset \
  --document-version 1 \
  --image-uri s3://bucket/images/dataset \
  --image-version 4 \
  --sidecar-manifest-sha256 '<caller-pinned lowercase SHA-256>' \
  --fragment-manifest-sha256 '<caller-pinned lowercase SHA-256>' \
  --expected-fragment-id 0 \
  --output-root /shared/document-image-overlays/canary \
  --checkpoint-path /shared/checkpoints/document-image-fetch-canary \
  --coordinate-window-bytes 4GiB \
  --payload-actor-cpus 64
```

The reader validates the complete plan inventory before emitting deterministic
source tasks. A retry adopts a fully published overlay without another remote
payload read and reprocesses only unfinished fragment plans.

The compatibility full-document patch path remains available when a downstream
consumer requires rewritten document rows. It combines remote fetch and patch
publication and therefore still needs node-local spool space:

```bash
python tutorials/interleaved/gpu_lance_document_materialization/patch_existing_plans.py \
  --plan-root /shared/image-plans/canary \
  --document-uri s3://bucket/documents/dataset \
  --document-version 1 \
  --image-uri s3://bucket/images/dataset \
  --image-version 4 \
  --sidecar-manifest-sha256 '<caller-pinned lowercase SHA-256>' \
  --fragment-manifest-sha256 '<caller-pinned lowercase SHA-256>' \
  --output-root /shared/document-patches/canary \
  --node-local-spool-root /local/document-payload-spool/canary \
  --checkpoint-path /shared/checkpoints/document-patch-canary \
  --payload-actor-cpus 8
```

Completed patch artifacts are adopted on retry. This compatibility command does
not consume an existing overlay.
