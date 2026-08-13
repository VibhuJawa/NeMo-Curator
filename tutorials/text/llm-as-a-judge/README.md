# LLM judge for crawl extraction

The implementation is a thin evaluation adapter over Curator's existing
`DataDesignerStage`. Install the `sdg_cpu` extra. NeMo Data Designer supplies
model/provider configuration, adaptive concurrency, strict structured output,
correction/restart behavior, trace capture, token statistics, and its built-in
multi-score `LLMJudgeColumnConfig` and judge-score profiler.

`PairwiseLLMJudgeStage` is explicitly bidirectional: every row is judged as
MinerU-HTML→jusText and jusText→MinerU-HTML using two Data Designer judge
columns. Labels are mapped back to their canonical extractor names. A winner
is emitted only when both directions agree; otherwise the result is
`order_sensitive`. Per-criterion winners and reasoning are retained.

`PretrainingReadinessLLMJudgeStage` annotates rather than filters. The
`pretraining_phase2_v2` contract includes 14 broad topic families and 65
detailed topics, content form, training-value signals, Phase-2 bucket,
language/script/BCP-47 observations, ten quality scores with a locally computed
aggregate, depth, reasoning density, temporal profile, review flags, and an
advisory action. Its JSON Schema is enforced by Data Designer; aggregate score
and tier are still computed locally. Actual LID, deduplication,
decontamination, PII/secret checks, and mixture weights remain deterministic or
corpus-level Curator work.

Both judges report `context_truncated` and `context_issue`. The configured
character window must conservatively fit the judge model after prompt/schema
tokens. Data Designer's model config does not declare context length, so an
overflow is reported as `model token limit not verified` rather than
misrepresented as an exact tokenizer measurement. A truncated view cannot
establish the absence of risks elsewhere. Failed Data Designer generations are
restored to the output cohort with a row-scoped error instead of silently
changing benchmark membership.

For a custom endpoint, pass Data Designer `ModelConfig` entries whose alias
matches the stage's `model_alias` (default `judge`), plus any required
`ModelProvider` objects through the inherited `model_providers` field. Add the
stage to a normal Curator `Pipeline`; its input/output remains `DocumentBatch`.

## Cohorts

Create a balanced diagnostic cohort (25 rows per observed behavior stratum):

```bash
python tutorials/text/llm-as-a-judge/build_html_parser_cohort.py \
  --input /path/to/mineru-html-vs-justext \
  --mode stratified --rows 25 \
  --output /path/to/parser-diagnostic.parquet
```

Create an equal-probability 5,000-row population benchmark:

```bash
python tutorials/text/llm-as-a-judge/build_html_parser_cohort.py \
  --input /path/to/mineru-html-vs-justext \
  --mode population --rows 5000 --seed 17 \
  --output /path/to/parser-population.parquet
```

The population artifact stores inclusion probability and inverse-probability
weight. The manifest records snapshot size, sample size, seed, and observed
strata. Use the balanced cohort for failure analysis and the population cohort
for representative aggregate estimates; do not combine their raw row counts.

The design follows the quality-focused late-pretraining curriculum in the
[Nemotron 3 Super report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
and the corpus-level mixture methodology in
[Nemotron-CLIMB](https://docs.nvidia.com/nemo/curator/curate-text/tutorials/nemotron-climb).
