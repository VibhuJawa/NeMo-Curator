# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract main content from raw Common Crawl HTML with MinerU-HTML.

The pipeline is CPU-only: labelling is submitted to an OpenAI-compatible vLLM
server that you start separately (see README.md) and point ``--server-url`` at.

The input is a Parquet dataset with a ``content`` column holding raw HTML
(``bytes`` or ``str``) and a ``url`` column. The output is the same rows plus a
``text`` column with the extracted main content as Markdown.

    python run_pipeline.py \
        --input /home/vjawa/bench-data/cc_main_2025_26_html_100k \
        --output ./mineru-out \
        --server-url http://127.0.0.1:8000 \
        --limit 2000
"""

import argparse
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.html_extraction import (
    DEFAULT_MODEL,
    MinerUHtmlExtractor,
    MinerUHtmlInterleavedStage,
)
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch


class HeadStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Truncate every batch to at most ``n`` rows, for quick benchmark runs."""

    def __init__(self, n: int):
        self.n = n
        self.name = "head"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=batch.to_pandas().head(self.n),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def create_html_reader(
    input_path: str,
    html_field: str = "content",
    url_field: str | None = "url",
    blocksize: str | None = "256MB",
    files_per_partition: int | None = None,
) -> ParquetReader:
    """Read a raw-HTML parquet dataset for the MinerU-HTML pipeline.

    ``ParquetReader`` defaults to ``dtype_backend="pyarrow"``. Raw HTML columns are
    large: 25k Common Crawl pages is >2 GB of bytes in one partition, and pickling
    an Arrow-backed ``binary`` column that big overflows its 32-bit offsets
    ("offset overflow while concatenating arrays") when the batch is shipped to the
    next stage. object dtype has no such limit. Harmless when the source was written
    as large_binary; keep it so callers do not depend on how the input table happened
    to be written.
    """
    return ParquetReader(
        file_paths=input_path,
        blocksize=blocksize if files_per_partition is None else None,
        files_per_partition=files_per_partition,
        fields=[html_field, url_field] if url_field else [html_field],
        read_kwargs={"dtype_backend": "numpy_nullable"},
    )


def default_worker_counts() -> tuple[int, int, int]:
    """Worker counts for (simplify, inference, extract), sized from this machine.

    These ratios are the configuration that measured fastest on a 128-core node
    against an 8xH100 server: 8 simplify, 32 inference, 24 extract. They are
    expressed as fractions of the core count rather than hard-coded, because the
    absolute numbers would oversubscribe a smaller machine -- and an oversubscribed
    Ray Data actor pool does not fail, it hangs, since actors hold their CPU slot
    for the whole run.

    Simplify gets the fewest because it is the fastest stage per document (~20 ms
    against ~51 ms for extract), and inference the most because its workers spend
    almost all their time waiting on HTTP.
    """
    # sched_getaffinity, not cpu_count: under SLURM or any cgroup, cpu_count()
    # reports the machine's cores rather than the ones this process may use, which
    # would size the pools straight into the hang described above.
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    return (
        max(1, cores // 16),  # simplify
        max(2, cores // 4),  # inference
        max(1, cores * 3 // 16),  # extract
    )


HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = HERE / "models.yaml"
DEFAULT_PROMPTS = HERE / "prompts"


def endpoint_context(base_url: str, served_model: str, api_key: str) -> int | None:
    """The context length the endpoint advertises for a model, or `None`.

    `None` means NOT ADVERTISED, which is not the same as small — 78 of this endpoint's
    90 chat models say nothing — so the caller warns and falls back to the flag rather
    than inventing a number.
    """
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(  # noqa: S310
        base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        logger.warning(f"could not ask {base_url} what it serves: {exc}")
        return None

    for entry in body.get("data", []):
        if entry.get("id") == served_model:
            advertised = entry.get("max_input_tokens")
            return int(advertised) if advertised else None
    return None


def read_api_key(env_var: str, key_file: str) -> str:
    """Env first, then the file — a shell export does not survive into a Slurm job."""
    value = os.environ.get(env_var, "").strip() if env_var else ""
    if value:
        return value
    if key_file:
        raw = Path(key_file).expanduser().read_text().strip()
        first = raw.split("\n")[0]
        return first.split("=", 1)[1].strip() if "=" in first else raw
    return ""


def resolve_model(model_id: str, models_path: Path) -> dict:
    """One entry of models.yaml, or a served id asked for by name.

    An endpoint serves far more models than a file can sensibly name, and requiring a
    YAML block before one can be tried would make "try that model" a two-step job with
    an edit in the middle. So an id the file does not define, but which looks like a
    served model (it has a `/`), borrows the endpoint and credential of the first
    `chat` entry and is asked for by name. Borrowing copies *where to send the
    request*, never what to ask for: an id the endpoint does not know still fails at
    the first request, loudly.
    """
    import yaml

    body = yaml.safe_load(models_path.read_text()) or {}
    if model_id in body:
        return {"id": model_id, **body[model_id]}

    if "/" in model_id:
        for name, spec in body.items():
            if spec.get("api") == "chat":
                return {**spec, "id": model_id, "model": model_id, "borrowed_from": name}

    msg = f"no model {model_id!r} in {models_path}; have {sorted(body)}"
    raise SystemExit(msg)


def resolve_prompt(prompt_id: str | None, prompts_dir: Path) -> str | None:
    """`explicit-compact` -> `prompts/explicit-compact.txt`, or an absolute path as given.

    `None` keeps MinerU's own packaged prompt, which is what the checkpoint was trained
    against — the right default for the local model and the wrong one for a hosted
    general model that has seen none of that training.
    """
    if not prompt_id:
        return None
    candidate = Path(prompt_id)
    path = candidate if candidate.is_absolute() else prompts_dir / f"{prompt_id}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in prompts_dir.glob("*.txt"))
        msg = f"no prompt {prompt_id!r} at {path}; have {available}"
        raise SystemExit(msg)
    return str(path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # Not required=True: --list-models and --list-prompts are questions about the
    # configuration, not runs, and demanding an input path to ask one is a papercut.
    ap.add_argument("--input", help="Parquet file or directory of raw HTML")
    ap.add_argument("--output", help="Output directory")
    ap.add_argument("--html-field", default="content")
    ap.add_argument("--url-field", default="url")
    ap.add_argument("--text-field", default="text", help="Output column for the extracted markdown")
    ap.add_argument("--limit", type=int, default=None, help="Keep at most this many documents per reader partition")

    ap.add_argument("--blocksize", default="256MB", help="Reader partition size")
    ap.add_argument("--files-per-partition", type=int, default=None)

    ap.add_argument(
        "--server-url",
        help="Root of the OpenAI-compatible vLLM endpoint, e.g. http://127.0.0.1:8000. "
        "Start it yourself; this script does not manage it (see README.md)",
    )
    ap.add_argument(
        "--served-model-name",
        default="mineru",
        help="Model to ask for. --served-model-name for a vLLM server you run, or the "
        "endpoint's own id for a hosted one, e.g. us/aws/anthropic/eccn-claude-opus-5",
    )
    ap.add_argument(
        "--api",
        choices=["completions", "chat"],
        default="completions",
        help="completions posts token ids to a vLLM server (the original path). chat "
        "posts text to /v1/chat/completions, which is what a hosted endpoint serves — "
        "it turns tokenization off and drops the vLLM-only sampler and grammar options",
    )
    ap.add_argument("--api-key-env-var", default="", help="Variable holding the credential")
    ap.add_argument(
        "--api-key-file",
        default="",
        help="Read only when that variable is unset — a shell export does not survive into a Slurm job",
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="Prompt template with a {simplified_html} placeholder, instead of the packaged MinerU prompt",
    )
    ap.add_argument(
        "--interleaved",
        action="store_true",
        help="Write row-wise interleaved records (one row per text run or image) instead "
        "of one markdown blob per document",
    )
    ap.add_argument(
        "--server-concurrency",
        type=int,
        default=48,
        help="In-flight requests per inference worker; queue depth = this x --inference-workers",
    )

    ap.add_argument("--model", default=DEFAULT_MODEL, help="Tokenizer only; the server holds the weights")
    ap.add_argument("--max-model-len", type=int, default=32768, help="Must match the server's --max-model-len")
    ap.add_argument(
        "--structured-outputs",
        choices=["none", "per_request"],
        default="per_request",
        help="Grammar strategy for the compact answer format",
    )
    ap.add_argument(
        "--fallback",
        default="trafilatura",
        choices=["trafilatura", "bypass", "empty"],
        help="What a document the model could not label becomes. `empty` is no fallback "
        "at all — right when the output is a gold run, where a rule-extracted document "
        "mixed in with model-extracted ones measures the wrong thing",
    )
    ap.add_argument(
        "--unlabelled",
        default="main",
        choices=["main", "other"],
        help="What an element a partial answer never mentioned becomes. Defaults to "
        "keeping it: deleting text on a judgement that was never made is the more "
        "expensive mistake",
    )
    ap.add_argument(
        "--keep-internal-fields",
        action="store_true",
        help="Keep the _mineru_* columns, including label coverage, for analysis",
    )
    ap.add_argument("--output-format", default="mm_md", choices=["mm_md", "md", "json", "txt", "none"])
    simplify, inference, extract = default_worker_counts()
    ap.add_argument("--simplify-workers", type=int, default=simplify)
    ap.add_argument("--inference-workers", type=int, default=inference)
    ap.add_argument("--extract-workers", type=int, default=extract)
    ap.add_argument(
        "--chat-template-mode",
        choices=["single", "upstream_double"],
        default="single",
        help="upstream_double reproduces the reference implementation's doubled chat template",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory")
    ap.add_argument(
        "--model-id",
        default=None,
        help="An entry in models.yaml, or any id the endpoint serves. Fills --server-url, "
        "--served-model-name, --api and the credential flags from one name",
    )
    ap.add_argument("--models", default=str(DEFAULT_MODELS), help="Model registry")
    ap.add_argument(
        "--prompt-id",
        default=None,
        help="A template in prompts/ by filename stem. Unset keeps MinerU's packaged prompt",
    )
    ap.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--list-models", action="store_true", help="Print the registry and exit")
    ap.add_argument("--list-prompts", action="store_true", help="Print available prompts and exit")
    return ap.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = parse_args()

    if args.list_models:
        import yaml

        for name, spec in sorted((yaml.safe_load(Path(args.models).read_text()) or {}).items()):
            key = f"  key ${spec.get('api_key_env_var')}" if spec.get("api_key_env_var") else ""
            print(f"  {name:<22} {spec.get('api', 'completions'):<12} {spec.get('model', '')}{key}")
        return
    if args.list_prompts:
        print("  (unset)                MinerU's packaged prompt — what the checkpoint was trained on")
        for path in sorted(Path(args.prompts_dir).glob("*.txt")):
            print(f"  {path.stem:<22} {path}")
        return
    if not args.input or not args.output:
        msg = "--input and --output are required to run"
        raise SystemExit(msg)
    if not args.server_url and not args.model_id:
        msg = "give --server-url, or --model-id to take it from the registry"
        raise SystemExit(msg)

    # One name fills the four flags that have to agree with each other. Setting them
    # apart is how you end up posting token ids to an endpoint expecting messages.
    if args.model_id:
        spec = resolve_model(args.model_id, Path(args.models))
        args.server_url = spec.get("base_url", args.server_url)
        args.served_model_name = spec.get("model", args.served_model_name)
        args.api = spec.get("api", args.api)
        args.api_key_env_var = spec.get("api_key_env_var", "") or args.api_key_env_var
        args.api_key_file = os.path.expanduser(spec.get("api_key_file", "") or args.api_key_file)
        args.model = spec.get("tokenizer", args.model)
        declared = spec.get("max_model_len", args.max_model_len)
        if declared == "auto":
            advertised = endpoint_context(
                args.server_url, args.served_model_name, read_api_key(args.api_key_env_var, args.api_key_file)
            )
            if advertised:
                args.max_model_len = advertised
                logger.info(f"{args.served_model_name} advertises a {advertised:,}-token context")
            else:
                logger.warning(
                    f"{args.served_model_name} does not advertise a context; using --max-model-len "
                    f"{args.max_model_len:,}. Set max_model_len in {args.models} if that is wrong — "
                    f"too low silently sends documents to the fallback."
                )
        else:
            args.max_model_len = int(declared)
        if spec.get("borrowed_from"):
            logger.info(f"{args.model_id} is not in the registry; borrowing the endpoint of {spec['borrowed_from']}")

    args.prompt_file = resolve_prompt(args.prompt_id, Path(args.prompts_dir)) or args.prompt_file
    logger.info(
        f"labelling with {args.served_model_name} via {args.api} at {args.server_url}; "
        f"prompt: {args.prompt_file or 'MinerU packaged ' + args.prompt_version}"
    )

    reader = create_html_reader(
        input_path=args.input,
        html_field=args.html_field,
        url_field=args.url_field,
        blocksize=args.blocksize,
        files_per_partition=args.files_per_partition,
    )

    extractor = MinerUHtmlExtractor(
        base_url=args.server_url,
        served_model_name=args.served_model_name,
        server_concurrency=args.server_concurrency,
        html_field=args.html_field,
        url_field=args.url_field,
        model_identifier=args.model,
        max_model_len=args.max_model_len,
        structured_outputs=args.structured_outputs,
        output_format=args.output_format,
        fallback=args.fallback,
        simplify_workers=args.simplify_workers,
        inference_workers=args.inference_workers,
        extract_workers=args.extract_workers,
        chat_template_mode=args.chat_template_mode,
        prompt_path=args.prompt_file,
        api=args.api,
        api_key_env_var=args.api_key_env_var,
        api_key_file=args.api_key_file,
        unlabelled=args.unlabelled,
        keep_internal_fields=args.keep_internal_fields,
    )

    pipeline = Pipeline(name="mineru_html_extraction", description="MinerU-HTML main content extraction")
    pipeline.add_stage(reader)
    if args.limit:
        pipeline.add_stage(HeadStage(args.limit))
    pipeline.add_stage(extractor)
    mode = "overwrite" if args.overwrite else "ignore"
    if args.interleaved:
        # Imported here, not at module scope: nemo_curator.stages.interleaved pulls in
        # opencv through its image utils, and the default markdown path has no business
        # requiring it. Only asking for interleaved output asks for that dependency.
        try:
            from nemo_curator.stages.interleaved.io.writers import InterleavedParquetWriterStage
        except ImportError as exc:  # pragma: no cover - depends on which extras are installed
            msg = "--interleaved needs the interleaved extra: pip install 'nemo_curator[interleaved]'"
            raise SystemExit(msg) from exc

        # Rows, not a blob: one text run or one image per row, in document order, with
        # a source_ref locator on every image. Written by the interleaved writer rather
        # than the plain one so the reserved schema is enforced here and not assumed.
        pipeline.add_stage(MinerUHtmlInterleavedStage(text_field=args.text_field, url_field=args.url_field))
        pipeline.add_stage(InterleavedParquetWriterStage(path=args.output, mode=mode))
    else:
        pipeline.add_stage(ParquetWriter(path=args.output, mode=mode))

    logger.info(pipeline.describe())

    t0 = time.perf_counter()
    results = pipeline.run()
    elapsed = time.perf_counter() - t0

    # The writer emits FileGroupTasks, so num_items counts files, not rows.
    written = [path for task in (results or []) for path in task.data]
    n_docs = sum(pq.ParquetFile(path).metadata.num_rows for path in written)
    logger.info(
        f"Wrote {n_docs} documents across {len(written)} files in {elapsed:.1f}s "
        f"({n_docs / elapsed:.1f} docs/s end to end)"
    )


if __name__ == "__main__":
    main()
