# Phase-2 continued-pretraining judge

This tutorial annotates Common Crawl text for a quality-focused Phase-2
continued-pretraining mixture. It is not an SFT-readiness classifier and does
not filter rows or choose final sampling weights.

The implementation reuses Curator's `Pipeline`, Parquet reader/writer,
`RayDataExecutor`, and `DataDesignerStage`. NeMo Data Designer supplies model
and provider configuration, adaptive concurrency, schema-constrained output,
trace capture, and token statistics. The tutorial adds the task taxonomy,
validation, local aggregate score, and row-level error isolation.

Install the `sdg_cpu` extra, configure the provider key, then run:

```bash
python tutorials/text/llm-as-a-judge/main.py \
  --input '/path/to/documents/*.parquet' \
  --output /path/to/judged \
  --ray-temp-dir /tmp/ray-phase2-judge-$SLURM_JOB_ID \
  --provider nvidia \
  --model meta/llama-3.3-70b-instruct
```

For an OpenAI-compatible service, also pass `--endpoint` and, when needed,
`--api-key-env`. Provider credentials stay in the environment.

The `pretraining_phase2_v2` result separates topic from page form and includes
training-value signals, a candidate Phase-2 bucket, language/script hints, ten
quality scores, a locally computed score/tier, depth, reasoning density,
temporal profile, review flags, and an advisory action. Actual language ID,
deduplication, decontamination, PII/secret checks, provenance policy, and final
mixture weights remain deterministic or corpus-level Curator work.

Every result includes `pretrain_context_truncated` and
`pretrain_context_issue`. The configured character budget is conservative but
is not the model tokenizer's exact context limit. When a document is shortened,
the issue records original and judged character counts and says the model token
limit was not verified. A partial view cannot establish that omitted content is
clean or safe.

The taxonomy follows the quality-focused late-pretraining curriculum in the
[Nemotron 3 Super report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf).
Final corpus weights should be calibrated through proxy training, such as the
[Nemotron-CLIMB workflow](https://docs.nvidia.com/nemo/curator/curate-text/tutorials/nemotron-climb).

The generic bidirectional judge and the MinerU-HTML versus jusText benchmark
live under `eval/text` and `eval/text/html_parser`, respectively.
