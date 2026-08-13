# Text evaluation

`llm_judge.py` provides reusable NeMo Data Designer-backed evaluation stages.
The pairwise stage evaluates both A→B and B→A, maps results back to stable
candidate labels, and reports `order_sensitive` when the two directions differ.
It also preserves raw Data Designer traces, keeps failures row-scoped, and
reports when configured character windows truncate candidate or context text.

Task-specific judge criteria and benchmark construction belong in their own
subdirectory. See [`html_parser`](html_parser/) for the MinerU-HTML versus
jusText benchmark. Phase-2 continued-pretraining classification is a tutorial
under `tutorials/text/llm-as-a-judge` rather than a library stage.
