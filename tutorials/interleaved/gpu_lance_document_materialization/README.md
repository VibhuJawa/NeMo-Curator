# GPU Lance document materialization

This tutorial resolves image URLs from a pinned interleaved document Lance
table against a local cuDF sidecar for a pinned image Lance table. RAPIDS-MPF
shuffles only URLs, stable row IDs, and document coordinates. Image payloads are
read directly from the remote image table into a bounded node-local Arrow spool,
then published as complete, ordered document-fragment patches.

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
  --output-root /shared/document-patches/canary \
  --node-local-spool-root /local/document-payload-spool/canary \
  --fragment-id 0 \
  --num-gpus 1
```

Use eight sidecar partitions and eight GPUs for one full GPU node. The default
`fetch_task_window=8` then schedules up to 64 left fragments per node while
retaining `1024` stable IDs per private Lance call and at most `16` pending
calls. Increase the coordinate window before increasing private-call size.

All artifact roots must be absolute. Coordinate plans and patch outputs belong
on shared durable storage; the payload spool belongs on node-local storage.
Never put credentials in command-line arguments, manifests, or result paths.
The full coordinate collective requires `RayActorPoolExecutor` and is not a
checkpointed phase. Existing coordinate plans can be replayed through
`LanceCoordinatePlanReader` and `LanceCoordinatePayloadPatchStage` in a separate
checkpointed pipeline:

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
  --checkpoint-path /shared/checkpoints/document-patch-canary
```

The reader validates the complete plan inventory before emitting deterministic
source tasks. A retry adopts completed patch artifacts without another remote
payload read and reprocesses only unfinished fragment plans.
