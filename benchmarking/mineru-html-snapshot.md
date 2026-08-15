# MinerU-HTML: one Common Crawl snapshot

This path streams a main Common Crawl snapshot through native NeMo Curator
stages. It does not materialize an intermediate HTML dataset or maintain a
second work-unit manifest.

## Data and ownership model

`CommonCrawlWARCManifestSourceStage` emits one deterministic source task per
WARC from the frozen snapshot manifest; it performs no URL discovery.
Curator's Slurm-array source filter assigns each complete WARC to exactly one
logical shard. A fused `CommonCrawlWARCDownloadAndReadStage` then:

1. downloads the whole WARC to allocation-local storage;
2. reads the WARC on that same Ray worker;
3. compresses each response body as an independent Zstandard frame;
4. emits bounded 1,024-record batches so simplify and conversion use all of the
   configured CPU workers;
5. removes the local WARC before the task leaves the worker.

The fused stage is deliberate. Separate download and iteration stages may land
on different Ray nodes, which is invalid when the download directory is local
RAID. The compressed HTML survives only long enough for MinerU fallback and is
dropped before the output Parquet is written.

Every array element has one four-hour, eight-GPU node and is launched with
`SlurmRayClient`. `Pipeline.run(checkpoint_path=...)` records completed source
WARCs, so a retry re-downloads only unfinished sources. The checkpoint and
output paths must be shared; the WARC and Ray temp paths must be local to the
allocation. The Slurm launcher prefers its job-specific directory under
`/raid/scratch/$USER` and falls back to `SLURM_TMPDIR` or `/tmp` only when local
RAID is unavailable. Ray uses a separate short `r<jobid>-<array-index>` temp
root on the same filesystem so its generated UNIX socket stays below Linux's
107-byte path limit.

The 128-CPU node baseline uses 2 whole-WARC download, 32 simplify, 32
inference-client, and 32 extraction actors. Each downloader accepts only one
in-flight WARC, preventing multi-GiB fan-out results from crowding CPU work out
of Ray's execution budget. This leaves 30 CPUs unreserved for vLLM, Dynamo,
Ray, writers, and the operating system. Its 256 GiB Ray object store leaves a
128 GiB Ray Data execution budget for overlapping compressed, simplified, and
inferred batches on the measured 1.5 TiB H100 node. Override these independently with
`MINERU_DOWNLOAD_WORKERS`, `MINERU_SIMPLIFY_WORKERS`,
`MINERU_INFERENCE_WORKERS`, and `MINERU_EXTRACT_WORKERS` after measuring a
representative canary on the target node type; override the object store with
`MINERU_OBJECT_STORE_SIZE` only when shared-memory capacity has been verified.

## Serving baseline

The production baseline is vLLM 0.26, Dynamo 1.4,
`FULL_AND_PIECEWISE` CUDA graphs, FP8 KV cache, per-request structured output,
and suffix speculation with 16 draft tokens through ArcticInference. Native
async scheduling remains disabled because this suffix path does not support it.
The Curator defaults are used for model context (`32768`) and per-element DOM
cutoff (`500`).

## Submit

Install the locked environment once at a shared path:

```bash
uv sync --frozen --extra mineru_html_inference --extra text_cpu
```

Then export the shared locations and submit:

```bash
export CURATOR_DIR=/shared/checkouts/Curator
export MINERU_RESULTS_ROOT=/shared/cc/CC-MAIN-2025-26/run-results
export MINERU_OUTPUT_PATH=/shared/cc/CC-MAIN-2025-26/mineru-output
export MINERU_CHECKPOINT_PATH=/shared/cc/CC-MAIN-2025-26/mineru-checkpoint
export MINERU_SNAPSHOT_SUCCESS_PATH=/shared/cc/CC-MAIN-2025-26/SNAPSHOT_SUCCESS.json
export MINERU_MODEL_CACHE=/shared/huggingface/hub
export MINERU_SNAPSHOT=2025-26
export CURATOR_DYNAMO_BIN_DIR=/shared/bin

MINERU_TOTAL_SHARDS=1400 \
MINERU_MAX_GPU_NODES=32 \
benchmarking/slurm/submit_mineru_cc_snapshot.sh
```

`1400` is a conservative initial split based on the measured 1M replay rate;
WARC record counts vary, so run a small canary cohort and adjust the shard count
before the full launch. A logical shard count is immutable once its checkpoint
path exists. Use a new checkpoint path if it changes.

The production transport is whole-object multipart download from the internal
PDX mirror through `s5cmd`; it does not contact `data.commoncrawl.org`.
`pdx-commoncrawl` in `~/.config/datamover/storage_locations` supplies credentials
and the `https://pdx.s8k.io` endpoint. Official manifest entries map from
`crawl-data/CC-MAIN-...` to bucket/key `s3://crawl-data/CC-MAIN-...`. Each of the
2 download actors transfers one WARC at a time with up to 8 concurrent 256 MiB
parts. The Slurm preflight resolves the Data Mover location without logging its
secret and verifies one exact manifest object before starting Ray or the model.

The storage measurements show that PDX is request-rate-bound rather than
bandwidth-bound. Whole WARC objects keep requests large and the default transfer
shape stays far below the measured per-key and store-wide request ceilings.
Override `MINERU_DM_STORAGE_LOCATION`, `MINERU_CC_S3_ENDPOINT_URL`,
`MINERU_CC_S3_BUCKET`, `MINERU_CC_S3_KEY_PREFIX`,
`MINERU_CC_S5CMD_CONCURRENCY`, or `MINERU_CC_S5CMD_PART_SIZE_MB` only for a
different verified mirror layout.

## Verification and retries

Each successful shard writes Curator's native completion manifest. The
dependent CPU job verifies:

- every logical shard has a completion manifest;
- output Parquet count is at least the snapshot WARC count (each WARC fans out
  into deterministic bounded chunks);
- every Parquet footer is readable and contains `url`, `text`, and
  `_mineru_status`;
- quality rates pass on an evenly distributed 1,024-file sample.

Only then is `SNAPSHOT_SUCCESS.json` written. This proves operational
completeness and checks gross quality drift; it does not replace the labelled
100k F1 canary when model, prompt, parser, or serving versions change.

If a shard fails, its completion manifest is absent. Resubmit the missing
logical shard indices against the same checkpoint and output paths, then submit
a new verifier dependency. Do not combine an external work-unit manifest with
native Slurm-array sharding.
