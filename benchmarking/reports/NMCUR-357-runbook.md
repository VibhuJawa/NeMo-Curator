# NMCUR-357 interactive runbook

All commands run from the `codex/nmcur-357-justext-cjk-boilerplate` worktree. Keep model/Ray temporary state
on node-local storage; keep staged data, checkpoints, outputs, results, logs, and analysis on shared scratch.

## 1. Stage the exact cohort

```bash
python benchmarking/scripts/nmcur357_cohort.py \
  --input-root "$NMCUR357_SOURCE_PATH" \
  --output-root "$NMCUR357_ROOT/input" \
  --canary-root "$NMCUR357_ROOT/canary/input" \
  --manifest-path "$NMCUR357_ROOT/manifests/cohort.json" \
  --english-size 100000 \
  --canary-per-language 5000
```

## 2. Start the interactive node

```bash
tmux new -s nmcur357
salloc --account=nemotron_n4_pre --partition=batch --nodes=1 --ntasks=1 \
  --cpus-per-task=128 --gpus-per-node=8 --mem=0 --exclusive --time=04:00:00
srun --ntasks=1 --pty bash -l
```

Record `SLURM_JOB_ID`, `nvidia-smi -L`, CPU affinity, `/dev/shm`, local RAID, shared-output permissions, model
cache, and pinned Curator/vLLM/Dynamo/Ray versions. Ray appends long session/socket names, so use a short local
path such as `RAY_TMPDIR=/raid/scratch/r${SLURM_JOB_ID}`. Put the pinned `etcd` 3.5.32 and `nats-server`
2.10.28 binaries on `PATH`; the repository's `docker/common/install_etcd_nats.sh` records these versions. Export
`USE_TORCH=0`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after verifying the model cache so every Ray
actor uses the pinned local artifacts instead of making Hugging Face metadata requests.

## 3. Run MinerU through the managed lifecycle

Set these explicitly for every trial: `NMCUR357_INPUT_PATH`, `NMCUR357_OUTPUT_PATH`,
`NMCUR357_CHECKPOINT_PATH`, `NMCUR357_RUN_RESULTS_PATH`, `NMCUR357_MODEL_CACHE`,
`NMCUR357_OBJECT_STORE_SIZE`, `NMCUR357_SERVER_CONCURRENCY`, `NMCUR357_INFERENCE_WORKERS`,
`NMCUR357_SIMPLIFY_WORKERS`, and `NMCUR357_EXTRACT_WORKERS`.

The single-node canary baseline is 32 simplify workers, 48 inference workers, 32 extraction workers, server
concurrency 512, and a 200 GiB Ray object store. The three long-lived Ray Data actor pools total 112 CPUs;
do not configure their sum above the node's 128-CPU allocation. Actor startup can take about a minute after
the model becomes ready, so distinguish pending actor initialization from a stable no-progress condition.

```bash
python benchmarking/run.py \
  --config benchmarking/nmcur357-mineru.yaml \
  --entries-exact nmcur357_mineru \
  --session-name "$NMCUR357_SESSION" \
  --strict-config-check \
  --reason NMCUR-357
```

Use an isolated checkpoint/output for the injected-failure resume test. Use one fixed 30,000-document canary
configuration for three trials, each with a clean output/checkpoint. Only start the full cohort when the three measured
rates, including their spread, project completion with 30% headroom. A replacement allocation reuses the identical full
command, output, and shared checkpoint paths.

## 4. Validate and analyze

```bash
python benchmarking/scripts/nmcur357_analysis.py \
  --input-path "$NMCUR357_ROOT/input" \
  --output-path "$NMCUR357_ROOT/mineru_output" \
  --checkpoint-path "$NMCUR357_ROOT/checkpoints/full" \
  --analysis-path "$NMCUR357_ROOT/analysis" \
  --success-path "$NMCUR357_ROOT/mineru_output/_SUCCESS.json" \
  --run-metadata-json "$NMCUR357_ROOT/manifests/run_metadata.json" \
  --gpt-neo-tokenizer "$NMCUR357_GPT_NEO_TOKENIZER" \
  --reuse-per-document-metrics \
  --bootstrap-replicates 1000 \
  --bootstrap-seed 357
```

The analysis command writes `_SUCCESS.json` last. It fails if exact language counts, unique/identical document IDs,
status thresholds, footers, deterministic Zstandard samples, completed-source count, or failed-task-marker checks fail.
Use `--reuse-per-document-metrics` only after a retry has left a finalized per-document Parquet artifact; the command
validates its footer and exact schema before reuse. A language whose source jusText field is entirely null contributes
50 deterministic availability-review examples with null ratio/decile fields instead of fabricated ratio-decile samples.
Population-weighted non-spaced metrics renormalize over languages with available source jusText and report the covered
share of primary non-spaced documents; missing strata are never imputed as zero.
