# Agent Preferences

## Environment Setup Before Python
- Always run: `source /home/nfs/vjawa/.bashrc`
- Then run: `cd /raid/vjawa/NeMo-Curator && source .venv/bin/activate`

## Running Tests
- Multimodal tests: `python -m pytest tests/stages/multimodal/ -v`
- Full test suite: `python -m pytest tests/ -q`
- Lint check: `python -m ruff check nemo_curator/ tests/`

## Benchmarking
- Single shard smoke test (no materialization):
  ```
  python benchmarking/scripts/multimodal_mint1t_benchmark.py \
    --benchmark-results-path .tmp_multimodal_runs/run_name \
    --input-path /datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/CC-MAIN-20240412101354-20240412131354-00000.tar \
    --output-path .tmp_multimodal_runs/run_name/output \
    --no-materialize-on-write --no-materialize-on-read --mode overwrite
  ```
- Full 10GB (90 shards): use `--input-path /datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/`

## Available Datasets
- Single shard (79MB): `/datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/CC-MAIN-20240412101354-20240412131354-00000.tar`
- 10GB MINT1T (90 tar shards): `/datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/`
- 20GB interleaved sample (17 tars): `/raid/vjawa/mint_interleaved_100mb_sample/webdataset/`
- 173GB parquet subset (997 files): `/datasets/vjawa/nvmint_mint1t_parquet_1k_subset/`
