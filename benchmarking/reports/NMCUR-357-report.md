# NMCUR-357: jusText CJK Boilerplate Study

## Run provenance

- Curator commit/version: checked-out source commit 16832b65; installed package version 1.3.0+880c1cf1
- Allocation/job IDs: ['952734 (resume/canary node, completed)', '953606 (full run, application success; allocation TIMEOUT after 04:00:16)', '956686 (analysis GPU allocation, intentionally interrupted and released after CPU partition discovery)', '956738 (CPU analysis allocation)']
- MinerU command: `python benchmarking/run.py --config benchmarking/nmcur357-mineru.yaml --entries-exact nmcur357_mineru --session-name full-1 --strict-config-check --reason NMCUR-357`
- Analysis command: `python benchmarking/scripts/nmcur357_analysis.py --input-path $NMCUR357_ROOT/input --output-path $NMCUR357_ROOT/mineru_output --checkpoint-path $NMCUR357_ROOT/checkpoints/full --analysis-path $NMCUR357_ROOT/analysis --success-path $NMCUR357_ROOT/mineru_output/_SUCCESS.json --run-metadata-json $NMCUR357_ROOT/manifests/run_metadata.json --reuse-per-document-metrics --bootstrap-replicates 1000 --bootstrap-seed 357`
- Configuration: `{"account": "nemotron_n4_pre", "ai_dynamo": "1.4.0.dev20260807", "analysis_node_shape": "cpu_interactive: 1 exclusive node, 64 CPUs, 251 GiB RAM, 4 hours", "canary_full_run_projection": "Estimate: 3.44 h at the slowest 102.643 docs/s canary rate; 4.47 h with 30% contingency, exceeding a 4 h GPU allocation", "drop_html_field": true, "etcd": "3.5.32", "executor": "RayDataExecutor", "extract_workers": 32, "fallback": "trafilatura", "full_run_processing_s": 10747.560806794092, "full_run_server_startup_s": 162.7013752530329, "full_run_subprocess_s": 10939.56, "full_run_throughput_docs_per_s": 118.28935168213529, "full_run_vllm_requests_per_s": 150.33879863676603, "html_compression": "zstd", "inference_workers": 48, "input_documents": 1271322, "input_files": 572, "model_replicas": 8, "nats": "2.10.28", "node_shape": "1 node, 128 CPUs, 8 H100 GPUs, full memory, exclusive", "output_format": "txt", "partition": "batch", "preserve_input_fields": true, "ray": "2.57.0", "ray_data_execution_budget_gib": 100, "ray_object_store_requested_gib": 200, "resource_declaration_caveat": "Ray reported simplify actors at roughly 5-6 GiB resident each without a meaningful heap-memory request; no OOM, spill, or actor restart occurred on the 1.5 TiB node.", "scratch_root_deviation": "Used the group-writable crawl_extraction_experiments/mineru_html/nmcur_357_justext_cjk_boilerplate root because the originally requested sibling root was denied by ACL.", "server_concurrency": 512, "simplify_workers": 32, "slurm_incident_classification": "Initial controller errors were sandbox-only connectivity failures after escalated scontrol/squeue/sacct succeeded. Job 953606 later reached its allocation time limit after application success and durable managed cleanup.", "structured_outputs": "per_request", "torch": "2.11.0+cu129", "vllm": "0.26.0+cu129"}`
- Performance trials: `[{"convert_error_rate": 0.0192, "documents": 30000, "status_ok_rate": 0.968267, "throughput_docs_per_s": 105.7059448, "trial": 1, "vllm_requests_per_s": 190.6940515}, {"convert_error_rate": 0.019233, "documents": 30000, "status_ok_rate": 0.968233, "throughput_docs_per_s": 102.6433622, "trial": 2, "vllm_requests_per_s": 191.4505823}, {"convert_error_rate": 0.019167, "documents": 30000, "status_ok_rate": 0.9683, "throughput_docs_per_s": 105.2962656, "trial": 3, "vllm_requests_per_s": 189.7873758}]`
- Performance trial spread: `{"throughput_docs_per_s": 3.0625826, "throughput_percent_of_mean": 2.93, "vllm_percent_of_mean": 0.87, "vllm_requests_per_s": 1.6632065}`

## Validation

- Documents: 1,271,322
- `_mineru_status == ok`: 97.0045%
- Conversion errors: 1.7579%
- Deterministic Zstandard cells checked: 128
- Completed/pending checkpoint sources: 572/0
- Input/output IDs: unique and identical
- All Parquet footers: readable

### Status by group

| Group | Documents | Primary `ok` | `ok` rate | Conversion-error rate |
|---|---:|---:|---:|---:|
| CHINESE | 408,221 | 396,289 | 97.08% | 2.01% |
| CHINESET | 129,780 | 114,896 | 88.53% | 9.12% |
| JAPANESE | 508,462 | 499,856 | 98.31% | 0.41% |
| KOREAN | 77,511 | 76,964 | 99.29% | 0.17% |
| THAI | 47,348 | 46,841 | 98.93% | 0.01% |
| ENGLISH | 100,000 | 98,393 | 98.39% | 0.05% |
| NON_SPACED | 1,171,322 | 1,134,846 | 96.89% | 1.90% |
| ALL | 1,271,322 | 1,233,239 | 97.00% | 1.76% |

## Primary results

Only `_mineru_status == ok` documents contribute to the ratios below. Non-spaced headline values are population-weighted language micros over strata with available source jusText.

| Group | jusText boilerplate | jusText main | Ambiguous | Unmatched | MinerU-other retained | Excess explained |
|---|---:|---:|---:|---:|---:|---:|
| CHINESE | 33.73% | 34.64% | 1.35% | 30.29% | 60.73% | 55.10% |
| CHINESET | n/a | n/a | n/a | n/a | n/a | n/a |
| JAPANESE | 30.80% | 37.78% | 2.27% | 29.15% | 61.35% | 53.71% |
| KOREAN | 28.75% | 43.93% | 0.97% | 26.36% | 56.54% | 58.46% |
| THAI | 31.97% | 40.54% | 4.04% | 23.45% | 80.41% | 56.18% |
| ENGLISH | 35.40% | 54.40% | 2.28% | 7.92% | 12.37% | 85.23% |
| NON_SPACED | 31.84% | 37.15% | 1.90% | 29.12% | 61.62% | 54.72% |
| ALL | 32.04% | 38.44% | 2.02% | 27.51% | 52.42% | 56.10% |

### Detailed occurrence and document distributions

The micro value is occurrence-weighted. Document columns summarize per-document ratios; the 1,000-replicate document-cluster Poisson bootstrap intervals use seed 357 and apply to the occurrence micro.

| Group | Metric | Micro | Bootstrap 95% | p10 | Median | p90 |
|---|---|---:|---:|---:|---:|---:|
| CHINESE | justext_boilerplate_share | 33.73% | 33.36%-34.10% | 7.38% | 35.80% | 73.93% |
| CHINESE | justext_main_share | 34.64% | 34.37%-34.91% | 1.13% | 27.25% | 75.30% |
| CHINESE | justext_ambiguous_share | 1.35% | 1.33%-1.36% | 0.00% | 0.00% | 4.17% |
| CHINESE | justext_unmatched_share | 30.29% | 29.96%-30.64% | 4.30% | 20.21% | 64.39% |
| CHINESE | mineru_other_retention_by_justext | 60.73% | 59.70%-61.66% | 39.16% | 73.49% | 95.88% |
| CHINESE | excess_justext_length_explained_by_other | 55.10% | 54.57%-55.62% | 27.26% | 68.31% | 91.75% |
| CHINESET | justext_boilerplate_share | n/a | n/a | n/a | n/a | n/a |
| CHINESET | justext_main_share | n/a | n/a | n/a | n/a | n/a |
| CHINESET | justext_ambiguous_share | n/a | n/a | n/a | n/a | n/a |
| CHINESET | justext_unmatched_share | n/a | n/a | n/a | n/a | n/a |
| CHINESET | mineru_other_retention_by_justext | n/a | n/a | n/a | n/a | n/a |
| CHINESET | excess_justext_length_explained_by_other | n/a | n/a | n/a | n/a | n/a |
| JAPANESE | justext_boilerplate_share | 30.80% | 30.68%-30.91% | 8.97% | 31.27% | 60.64% |
| JAPANESE | justext_main_share | 37.78% | 37.65%-37.91% | 2.92% | 26.90% | 68.24% |
| JAPANESE | justext_ambiguous_share | 2.27% | 2.26%-2.29% | 0.00% | 0.68% | 6.10% |
| JAPANESE | justext_unmatched_share | 29.15% | 29.04%-29.27% | 9.59% | 30.99% | 60.51% |
| JAPANESE | mineru_other_retention_by_justext | 61.35% | 61.22%-61.48% | 32.39% | 63.47% | 89.64% |
| JAPANESE | excess_justext_length_explained_by_other | 53.71% | 53.53%-53.86% | 23.43% | 54.48% | 83.90% |
| KOREAN | justext_boilerplate_share | 28.75% | 28.32%-29.15% | 6.24% | 36.12% | 81.00% |
| KOREAN | justext_main_share | 43.93% | 43.29%-44.54% | 0.12% | 19.82% | 75.00% |
| KOREAN | justext_ambiguous_share | 0.97% | 0.94%-1.01% | 0.00% | 0.00% | 3.10% |
| KOREAN | justext_unmatched_share | 26.36% | 25.95%-26.74% | 9.77% | 22.55% | 59.23% |
| KOREAN | mineru_other_retention_by_justext | 56.54% | 56.13%-56.98% | 18.97% | 77.66% | 91.21% |
| KOREAN | excess_justext_length_explained_by_other | 58.46% | 57.84%-59.08% | 23.61% | 70.04% | 88.64% |
| THAI | justext_boilerplate_share | 31.97% | 31.52%-32.42% | 8.28% | 29.32% | 69.84% |
| THAI | justext_main_share | 40.54% | 40.01%-41.07% | 5.18% | 28.87% | 75.16% |
| THAI | justext_ambiguous_share | 4.04% | 3.89%-4.19% | 0.00% | 1.53% | 11.27% |
| THAI | justext_unmatched_share | 23.45% | 22.94%-23.95% | 5.57% | 17.29% | 44.85% |
| THAI | mineru_other_retention_by_justext | 80.41% | 80.14%-80.69% | 63.64% | 86.28% | 98.50% |
| THAI | excess_justext_length_explained_by_other | 56.18% | 55.49%-56.93% | 30.83% | 60.66% | 86.91% |
| ENGLISH | justext_boilerplate_share | 35.40% | 31.76%-38.71% | 0.00% | 0.56% | 72.60% |
| ENGLISH | justext_main_share | 54.40% | 51.49%-57.66% | 0.00% | 88.78% | 98.99% |
| ENGLISH | justext_ambiguous_share | 2.28% | 2.16%-2.41% | 0.00% | 1.41% | 7.76% |
| ENGLISH | justext_unmatched_share | 7.92% | 7.48%-8.41% | 0.00% | 1.66% | 49.36% |
| ENGLISH | mineru_other_retention_by_justext | 12.37% | 11.86%-12.90% | 0.00% | 0.58% | 30.64% |
| ENGLISH | excess_justext_length_explained_by_other | 85.23% | 83.17%-86.76% | 0.78% | 91.30% | 100.00% |
| NON_SPACED | justext_boilerplate_share | 31.79% | 31.64%-31.93% | 8.29% | 33.00% | 67.45% |
| NON_SPACED | justext_main_share | 37.25% | 37.12%-37.38% | 1.69% | 26.99% | 71.95% |
| NON_SPACED | justext_ambiguous_share | 2.00% | 1.99%-2.02% | 0.00% | 0.27% | 5.44% |
| NON_SPACED | justext_unmatched_share | 28.97% | 28.82%-29.10% | 6.64% | 25.85% | 60.83% |
| NON_SPACED | mineru_other_retention_by_justext | 56.60% | 56.25%-56.95% | 1.49% | 66.42% | 92.63% |
| NON_SPACED | excess_justext_length_explained_by_other | 54.63% | 54.41%-54.85% | 24.59% | 61.34% | 88.52% |
| ALL | justext_boilerplate_share | 32.04% | 31.75%-32.32% | 5.97% | 31.94% | 67.50% |
| ALL | justext_main_share | 38.44% | 38.24%-38.62% | 1.68% | 28.28% | 76.68% |
| ALL | justext_ambiguous_share | 2.02% | 2.01%-2.04% | 0.00% | 0.31% | 5.58% |
| ALL | justext_unmatched_share | 27.51% | 27.35%-27.68% | 5.06% | 24.78% | 60.67% |
| ALL | mineru_other_retention_by_justext | 52.42% | 52.10%-52.76% | 0.00% | 63.34% | 92.10% |
| ALL | excess_justext_length_explained_by_other | 56.10% | 55.75%-56.44% | 24.13% | 61.48% | 89.19% |

## Secondary dashboard comparison

GPT-Neo token totals were not computed because the exact pinned tokenizer ID remains unresolved.

## Manual review

The non-spaced population-weighted language metrics renormalize across strata with available source jusText. Coverage among primary non-spaced documents is 89.88%.

Fifty deterministic examples per language are written to `review_examples.jsonl` for human review. Languages with available jusText ratios contribute five examples per ratio decile. If a language has no source jusText text, its examples are deterministic availability-review samples with null ratio and decile fields.

## Measurements, estimates, and uncertainty

The cohort counts, status rates, overlap ratios, bootstrap intervals, and trial rates are measurements. Any full-run duration projected from the canary is an estimate recorded in run metadata.

Remaining uncertainties:

- Traditional Chinese source jusText is entirely null, so its overlap metrics are unavailable.
- Review whether ambiguous shingles should be apportioned instead of reported separately.
- MinerU labels are a model-based reference, not human boilerplate ground truth.

## Artifacts

- `per_document_metrics.parquet`
- `summary.json` and `summary.csv`
- `review_examples.jsonl`
- final `_SUCCESS.json` validation receipt

Anything still running or unresolved: Secondary dashboard GPT token totals await confirmation of the exact pinned tokenizer ID; the cache contains EleutherAI/gpt-neox-20b, but the runbook only says GPT-Neo. Traditional Chinese conversion errors are materially elevated and require follow-up diagnosis.
