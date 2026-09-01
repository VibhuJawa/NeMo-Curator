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
cache, and pinned Curator/vLLM/Dynamo/Ray versions. Set `RAY_TMPDIR` to local RAID before running the benchmark.

## 3. Run MinerU through the managed lifecycle

Set these explicitly for every trial: `NMCUR357_INPUT_PATH`, `NMCUR357_OUTPUT_PATH`,
`NMCUR357_CHECKPOINT_PATH`, `NMCUR357_RUN_RESULTS_PATH`, `NMCUR357_MODEL_CACHE`,
`NMCUR357_OBJECT_STORE_SIZE`, `NMCUR357_SERVER_CONCURRENCY`, `NMCUR357_INFERENCE_WORKERS`,
`NMCUR357_SIMPLIFY_WORKERS`, and `NMCUR357_EXTRACT_WORKERS`.

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
  --bootstrap-replicates 1000 \
  --bootstrap-seed 357
```

The analysis command writes `_SUCCESS.json` last. It fails if exact language counts, unique/identical document IDs,
status thresholds, footers, deterministic Zstandard samples, completed-source count, or failed-task-marker checks fail.
