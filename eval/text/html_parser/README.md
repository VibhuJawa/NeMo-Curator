# HTML parser evaluation

This benchmark compares the proper library names **MinerU-HTML** and
**jusText**. `judge.py` configures the generic pairwise judge for content
recall, content precision, structure preservation, and coherence. Each row is
judged in both presentation orders; only an order-consistent result is exposed
as the winner.

Build a balanced diagnostic cohort:

```bash
python eval/text/html_parser/build_cohort.py \
  --input /path/to/mineru-html-vs-justext \
  --mode stratified --rows 25 \
  --output /path/to/parser-diagnostic.parquet
```

Build an equal-probability population benchmark:

```bash
python eval/text/html_parser/build_cohort.py \
  --input /path/to/mineru-html-vs-justext \
  --mode population --rows 5000 --seed 17 \
  --output /path/to/parser-population.parquet
```

Population output includes inclusion probability and inverse-probability
weight. Use the balanced cohort for failure analysis and the population cohort
for representative aggregate estimates; do not combine their raw row counts.

The current checked experiment artifacts are cohorts, not completed model
runs. A judged output adds `html_parser_judge_winner`, directional and
per-criterion winners, reasoning, order consistency, raw response, error, and
context-window diagnostics.

Run against a local OpenAI-compatible endpoint with resumable source shards:

```bash
python eval/text/html_parser/main.py \
  --input '/path/to/cohort-shards/*.parquet' \
  --output /path/to/judged \
  --checkpoint /path/to/checkpoint \
  --ray-temp-dir /tmp/ray-html-parser-judge-$SLURM_JOB_ID \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8
```
