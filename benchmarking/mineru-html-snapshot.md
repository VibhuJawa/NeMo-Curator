# MinerU-HTML: one Common Crawl snapshot

This is the production path for a file-aligned Common Crawl HTML snapshot. One
Slurm array task owns one GPU node and one immutable manifest line. It launches
the Curator pipeline and its eight Dynamo/vLLM replicas through
`benchmarking/run.py`.

## Safety contract

Each work unit has one final directory. The pipeline writes to a unique sibling
attempt directory and publishes it with a same-filesystem rename only after all
of these checks pass:

- every input path and its byte size still match the frozen manifest;
- input and output row counts equal `expected_rows`;
- every output is readable Parquet with `url`, `text`, and `_mineru_status`;
- the input and output URL multisets match, including duplicate multiplicity;
- at least 95% of rows have status `ok` and non-empty text;
- at most 2% of rows have status `convert_error`.

The success record includes the manifest SHA-256, work-unit definition, counts,
status histogram, URL fingerprint, and Slurm identity. A valid published unit is
idempotently skipped on retry. An existing directory without a matching success
record is never overwritten.

The snapshot verifier is a separate CPU job. It reopens every published Parquet
footer, compares file and row counts with the unit success records, and writes
the snapshot success record only when every manifest unit is present and valid.
These checks establish completeness and operational quality; they do not replace
an F1 evaluation against labelled data. Run the 100k reference corpus before a
snapshot when model, prompt, conversion, or serving versions change.

## Environment

Use a shared checkout and model cache visible from every node:

```bash
uv sync --frozen --extra mineru_html_inference
```

The locked production serving baseline is vLLM 0.26, Dynamo 1.4,
`FULL_AND_PIECEWISE` CUDA graphs, FP8 KV cache, structured output, and suffix
speculation with 16 draft tokens backed by ArcticInference's continuation
cache. Native async scheduling is intentionally omitted because vLLM 0.26 does
not support it with suffix speculation. The managed actors reuse this
locked driver environment so Dynamo's own vLLM extra cannot silently select a
different engine version. The same environment must be installed at the same
path on every Slurm node (normally a shared filesystem checkout).
Managed Dynamo also needs `etcd` and `nats-server`; put their shared directory
in `CURATOR_DYNAMO_BIN_DIR` or make both commands available on `PATH`.

## 1. Freeze the work-unit manifest

Input Parquet files must already contain independently compressed HTML frames
when `--html-compression=zstd` is used. No uncompressed-size column is required;
the decoder reads the size from each frame and expands only the current cell.

```bash
python benchmarking/scripts/plan_mineru_snapshot.py \
  --input-path /shared/cc/CC-MAIN-2025-26/html-zstd \
  --output-root /shared/cc/CC-MAIN-2025-26/mineru-output \
  --manifest-path /shared/cc/CC-MAIN-2025-26/mineru-work-units.jsonl \
  --snapshot-id CC-MAIN-2025-26 \
  --html-field _html_zstd \
  --target-rows 1800000
```

The default 1.8M rows is intentionally below the theoretical 3.5-hour capacity.
At the measured 1M-corpus rate of 181.6 documents/s, model work is about 2.75
hours. The remaining time covers Dynamo startup, input preflight, output
validation, skew, and cleanup. Recalibrate `--target-rows` after the first
snapshot cohort; do not select a single fastest run.

The initial gates are also grounded in that 1M run: 98.01% `ok`, 99.69%
non-empty text, and 10 conversion errors. Treat a materially different snapshot
distribution as a reason to investigate, not a reason to weaken gates blindly.

The planner reads Parquet footers concurrently, validates the required schema,
sorts paths deterministically, and never splits a file. It rejects a file over
the row budget instead of silently creating a work unit that may breach the
wall time; repartition such inputs first. Freeze and retain this manifest;
changing it changes the digest and invalidates existing success records.

## 2. Submit the GPU array

```bash
export CURATOR_DIR=/shared/checkouts/Curator
export MINERU_RESULTS_ROOT=/shared/cc/CC-MAIN-2025-26/run-results
export MINERU_WORK_UNIT_MANIFEST=/shared/cc/CC-MAIN-2025-26/mineru-work-units.jsonl
export MINERU_MODEL_CACHE=/shared/huggingface/hub
export CURATOR_DYNAMO_BIN_DIR=/shared/bin  # contains etcd and nats-server
export MINERU_HTML_FIELD=_html_zstd
export MINERU_HTML_COMPRESSION=zstd
export MINERU_URL_FIELD=url
export MINERU_SNAPSHOT_SUCCESS_PATH=/shared/cc/CC-MAIN-2025-26/SNAPSHOT_SUCCESS.json

MINERU_MAX_GPU_NODES=32 benchmarking/slurm/submit_mineru_cc_snapshot.sh
```

Set `MINERU_MAX_GPU_NODES` to the allocation and storage budget. The launcher
reads Slurm's `MaxArraySize`, splits larger snapshots into offset arrays,
divides that node cap across them, and submits one verifier with an `afterok`
dependency on every array.
Each task has a four-hour Slurm allocation and a 12,600-second `run.py` timeout,
leaving 30 minutes for teardown before Slurm's hard limit.

Do not also enable Curator's native Slurm-array sharding. The manifest is the one
and only ownership layer; combining two sharding schemes risks omissions.

## 3. Snapshot verification

If an array index fails, the `afterok` verifier remains pending. Resubmit only
the failed indices; already published units safely skip. Submit a new dependent
verifier after the retry succeeds.

Output directories and success records are normalized to world-readable files
(`0644`) and traversable directories (`0755`). Parent directories must also be
readable and traversable by the intended users.

## Artifacts

For each array index, `MINERU_RESULTS_ROOT/mineru-<array>-<index>/` contains the
`run.py` params, metrics, task timings, GPU samples, stdout/stderr, and Ray logs.
The authoritative dataset is the manifest's `output_path`, not the benchmark
scratch directory. Failed or timed-out attempts remain hidden as
`.work-unit-*.attempt-*` siblings for diagnosis and can be removed after the
snapshot is verified.
