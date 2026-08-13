# LLM-as-a-judge evaluation

This tutorial has two complementary workflows:

- `compare_html_parsers.py` compares MinerU-HTML (`text`) with the jusText
  library (`justext_extracted_text`) on the same source pages.
- `classify_cc_for_pretraining.py` independently labels either extraction for
  Phase-2 continued-pretraining readiness.

Both retain raw model responses, isolate failures in row-level error columns,
sample deterministically, and use partition checkpoints plus
`ParquetWriter(mode="ignore")` for recoverable runs.

## Compare MinerU-HTML and jusText

The pairwise judge returns a normalized winner, rubric scores, rationale,
confidence, and randomized presentation order. The rubric covers useful-content
coverage, boilerplate precision, readability, and structure preservation.

For a decision-quality comparison, first construct a bounded-memory cohort
stratified across empty/one-sided extractions, short paired results, similar
lengths, moderate disagreements, and large disagreements in either direction:

```bash
python tutorials/text/llm-as-a-judge/build_html_parser_cohort.py \
  --rows-per-stratum 25 \
  --output /path/to/mineru-html-vs-justext.cohort.parquet
```

The builder scans every parquet row by default and retains the lowest stable
hashes in each stratum, so results are independent of batch size and require
memory proportional only to the cohort. Length strata select coverage; they do
not determine which extraction is better.

Build a separate equal-probability sample for population-level estimates:

```bash
python tutorials/text/llm-as-a-judge/build_html_parser_cohort.py \
  --sampling-mode population \
  --target-rows 5000 \
  --seed 17 \
  --output /path/to/mineru-html-vs-justext.population.parquet
```

Every source row has the same inclusion probability. The artifact records that
probability and its inverse sample weight, and the adjacent manifest records
the full population size, seed, source files, and observed sample strata. Use
this cohort for aggregate winner rates and weighted quality; use the balanced
stratified cohort for failure analysis. Do not combine their raw row counts.

Start with one input parquet partition and 100 deterministic rows:

```bash
python tutorials/text/llm-as-a-judge/compare_html_parsers.py \
  --input /path/to/mineru-html-vs-justext.cohort.parquet \
  --rows-per-file 0 \
  --model <served-model-name> \
  --base-url http://<host>:<port>/v1 \
  --output /path/to/html-parser-judge-smoke
```

Inspect `html_parser_judge_error` and raw responses before increasing scale.
Pass `--num-files 0 --rows-per-file 0` only for an intentional full experiment.
The implementation works with any OpenAI-compatible endpoint; use
`--disable-response-format` when the service lacks JSON response mode.

## Classify Phase-2 pretraining readiness

NVIDIA's Nemotron 3 curriculum describes Phase 1 as broad and diverse and
Phase 2 as a late-pretraining shift toward predominantly high-quality data.
Accordingly, `pretraining_phase2_v2` judges next-token training value, not SFT
example transformability. Its independent axes are:

- a two-level subject hierarchy spanning 14 broad families and detailed
  subdomains such as pure/applied math, AI/ML, systems, databases, individual
  sciences, engineering branches, clinical/biomedical/public health, law,
  economics, education, humanities, languages, arts, and practical knowledge;
- content form and multi-label training value signals;
- a candidate Phase-2 source bucket;
- primary and additional ISO-639-3 languages, ISO-15924 scripts, BCP-47 tags,
  estimated shares, multilingual mode, register, and locale hint;
- ten 1–5 quality dimensions covering extraction integrity, coherence,
  epistemic quality, density, depth, educational/reasoning value, context
  independence, originality signal, and Phase-2 value;
- locally computed weighted quality score and quality tier;
- knowledge depth, reasoning density, and temporal profile;
- quality and review-risk flags;
- advisory `upweight | include | downweight | exclude` action and confidence;
- view coverage, including whether only a head/tail window was judged.

The model never supplies the aggregate score or sampling weight. Curator
validates every dimension and computes the aggregate locally. The action and
bucket are discovery annotations, not policy gates.

Start with 100 MinerU-HTML documents from one partition:

```bash
python tutorials/text/llm-as-a-judge/classify_cc_for_pretraining.py \
  --model <served-model-name> \
  --base-url http://<host>:<port>/v1 \
  --output /path/to/cc-pretraining-judge-smoke
```

To have the curation job own a local judge, run inside a GPU allocation with
Curator's inference-server extra:

```bash
python tutorials/text/llm-as-a-judge/classify_cc_for_pretraining.py \
  --serve-model-locally \
  --model /path/to/a/capable/instruction-model \
  --tensor-parallel-size 1 \
  --output /path/to/cc-pretraining-judge-smoke
```

This starts Ray first, serves the model through Curator's Dynamo/vLLM
`InferenceServer`, verifies model discovery with a generation smoke test, and
then runs the checkpointed pipeline.

To judge jusText on the same cohort, add
`--text-field justext_extracted_text --output-prefix justext_pretrain_judge` and
use a separate output/checkpoint directory. Phase-2 labeling should be run
independently on both extractions; the pairwise judge alone cannot expose how
parser omissions change topic or quality distributions.

For representative estimates, use a global target such as
`--num-files 0 --target-sample-rows 5000`. The allocator assigns an exact sample
approximately proportional to parquet row counts, selects rows reproducibly,
and writes inclusion probabilities and inverse-probability weights. It fails
if a source row count changes before execution, preventing silent resume drift.

Summarize completed partitions in bounded memory:

```bash
python tutorials/text/llm-as-a-judge/summarize_pretraining_judgments.py \
  --input /path/to/cc-pretraining-judge-smoke \
  --output /path/to/cc-pretraining-judge-smoke.summary.json
```

The report includes observed and, when weights are present, estimated
distributions for family/topic, form, bucket, language, quality tier, depth,
reasoning, temporal profile, action, training values, and flags. It also reports
quality dimensions, confidence, error rates, quality bands, and mean quality by
topic. Multi-label shares can sum above 100 percent.

## Operational boundary

The all-record judge is deliberately a core semantic pass. Treat language ID,
exact/fuzzy/semantic deduplication, benchmark decontamination, PII and secret
detection, malware scanning, provenance/license policy, tokenizer statistics,
and final mixture weights as deterministic or corpus-level work. Run specialized
sparse judge passes for ambiguous safety/actionability or provenance cases.
A truncated row view cannot establish the absence of risk elsewhere in a long
document.

Before trusting a snapshot distribution, calibrate against human-reviewed
examples stratified by language/script, topic, form, length, quality tier,
parser, and rare flags. Report per-axis agreement, confusion, malformed-output
rate, and repeat-run stability. Optimize actual mixture weights with proxy-model
training rather than treating judge labels as weights.

The design is grounded in the quality-focused Phase-2 mixtures in the
[NVIDIA Nemotron 3 Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf),
the semantic-mixture methodology in
[Nemotron-CLIMB](https://docs.nvidia.com/nemo/curator/curate-text/tutorials/nemotron-climb),
and the filtering and ablation evidence in
[Nemotron-CC](https://arxiv.org/abs/2412.02595),
[DCLM](https://arxiv.org/abs/2406.11794), and
[FineWeb](https://arxiv.org/abs/2406.17557).
