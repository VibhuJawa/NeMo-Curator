Prompt templates for `--prompt-id`. The id is the filename stem.

A template is plain text with one placeholder, `{simplified_html}` — the simplified DOM,
every labellable element carrying an `_item_id`. A literal brace must be doubled.

Leaving `--prompt-id` unset uses MinerU's own packaged prompt (`--prompt-version`,
default `short_compact`), which is what the checkpoint was trained against and the right
default for `mineru-local`. A hosted general model has seen none of that training, so it
usually wants the instructions spelled out — see `explicit-compact.txt`.

Whichever is used, the answer contract is the same and is what the rest of the pipeline
parses: `<answer>1main2other3main…</answer>`, one label per `_item_id`, in order.
