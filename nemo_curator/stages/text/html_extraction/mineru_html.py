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

"""MinerU-HTML main-content extraction as NeMo Curator pipeline stages.

`MinerU-HTML <https://github.com/opendatalab/MinerU-HTML>`_ extracts the main
content of a web page by asking a small language model to label every element
of a simplified DOM as ``main`` or ``other``. The reference implementation runs
all six of its steps inside one process; that wastes GPUs, because only step 3
needs one.

This module splits the work along the CPU/GPU boundary, and then puts the model
behind an OpenAI-compatible HTTP endpoint you run yourself, so that **no
pipeline stage owns a GPU**:

=====================================  ========  ===================================
Stage                                  Hardware  Work
=====================================  ========  ===================================
:class:`MinerUHtmlSimplifyStage`       CPU       simplify DOM, build prompt, tokenize
``MinerUHtmlServerInferenceStage``     CPU       submit to a vLLM server over HTTP
:class:`MinerUHtmlExtractStage`        CPU       parse labels, prune DOM, to markdown
=====================================  ========  ===================================

The engines live in a ``vllm serve`` (see the tutorial README for the exact
command), which the pipeline never starts or stops. That is the point: engine
startup is paid once instead of every run, the engines scale and restart
independently of the pipeline, and a Curator job needs no GPU allocation at all.

Scale the CPU stages out until they keep the server saturated. On Common Crawl
the CPU side costs ~80 ms per document per core (20 ms simplify, 8 ms tokenize,
6 ms prune, 45 ms Markdown conversion), i.e. ~12.5 documents/s/core. Even an
H100 chewing through ~16 documents/s therefore needs under two cores of CPU
alongside it, so there is no reason to trade away conversion quality — budget
the cores and keep the default ``mm_md`` output, math and all.

Use :class:`MinerUHtmlExtractor` to add all three at once.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Literal

from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.html_extraction.assets import split_multi_image_figures
from nemo_curator.stages.text.html_extraction.mineru_server import MinerUHtmlServerInferenceStage
from nemo_curator.stages.text.html_extraction.mineru_utils import (
    CHUNK_IDS_FIELD,
    COVERAGE_FIELD,
    DEFAULT_MODEL,
    INTERNAL_FIELDS,
    MAP_HTML_FIELD,
    N_ITEMS_FIELD,
    PROMPT_FIELD,
    RESPONSE_FIELD,
    STATUS_FIELD,
    TOKENS_FIELD,
    FallbackExtractor,
    abridge_oversized_elements,
    chunk_simplified_html,
    compact_response_budget,
    count_item_ids,
    decode_html_cell,
    extract_main_html,
    label_coverage,
    load_prompt_template,
    parse_compact_response,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

# Characters per token when there is no tokenizer to ask. Deliberately low, so the
# estimate runs high: over-estimating marks a borderline document `too_long` and sends
# it to the fallback, while under-estimating sends a request the endpoint rejects and
# loses the document outright. It is an estimate and is only ever used for that gate —
# never reported as a token count.
_CHARS_PER_TOKEN = 3.0


def _estimate_tokens(prompt: str) -> int:
    return int(len(prompt) / _CHARS_PER_TOKEN)


_MINERU_INSTALL_HINT = (
    "mineru_html is required for the MinerU-HTML simplify stage. "
    "Install with: pip install 'nemo_curator[mineru_html]' (upstream's vllm extra "
    "is not needed; these stages only talk to a vLLM server over HTTP)."
)


class MinerUHtmlSimplifyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU stage: raw HTML -> model prompt + ``_item_id``-annotated HTML.

    Emits :data:`TOKENS_FIELD`, :data:`MAP_HTML_FIELD`, :data:`N_ITEMS_FIELD`
    and :data:`STATUS_FIELD`. Rows whose HTML cannot be parsed are kept with a
    non-``ok`` status so the extract stage can apply the configured fallback.

    Prompts are always tokenized here rather than at the server: the completions
    route accepts token ids directly, so this moves tokenization onto the CPU
    stage that already scales out, and it means the over-long pre-filter can
    count real tokens instead of estimating from character length.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        html_field: str = "content",
        model_identifier: str = DEFAULT_MODEL,
        prompt_version: str = "short_compact",
        cutoff_length: int = 500,
        max_model_len: int = 32768,
        drop_html_field: bool = False,
        chat_template_mode: Literal["single", "upstream_double"] = "single",
        cache_dir: str | None = None,
        prompt_path: str | None = None,
        tokenize: bool = True,
        chunk_max_chars: int = 0,
        chunk_overlap_chars: int = 2000,
        element_max_chars: int = 0,
    ):
        """
        Args:
            html_field: Column holding raw HTML as ``str`` or ``bytes``.
            model_identifier: HuggingFace id, used here only for the tokenizer.
            prompt_version: MinerU prompt template; ``short_compact`` matches
                the ``*-compact`` checkpoints.
            cutoff_length: Per-element text truncation in the simplified DOM.
            max_model_len: Context length the server was started with.
                Documents that cannot fit their prompt plus answer budget are
                marked ``too_long``.
            drop_html_field: Drop the raw HTML column once simplified. Halves
                the bytes shipped downstream, but disables the trafilatura and
                bypass fallbacks, which need the original document.
            chat_template_mode: ``single`` applies the chat template once;
                ``upstream_double`` reproduces the reference implementation's
                doubled template. See :meth:`_chat_wrap`.
            cache_dir: HuggingFace cache directory.
            prompt_path: A prompt template with a ``{simplified_html}`` placeholder,
                used instead of the packaged ``prompt_version``. The template's
                sha256 travels with the run — a prompt id is a name someone can
                reuse, so two runs claiming one prompt while the file changed
                underneath them is otherwise invisible.
            tokenize: Emit token ids for the completions route. Set ``False`` for a
                hosted chat endpoint, which wants messages and whose tokenizer we
                cannot run: the prompt is then written as text and the over-long
                gate falls back to a character estimate (see :meth:`process`).
            chunk_max_chars: Split the simplified DOM into windows of at most this many
                characters, so a small-context model can see all of it. ``0`` disables
                chunking and sends the whole document, which is the original behaviour.
                The corpus averages ~46,000 prompt tokens, so a 32k model — including the
                0.5B checkpoint this pipeline was built around — otherwise cannot attempt
                most documents, and scoring it anyway measures context length rather than
                extraction quality.
            chunk_overlap_chars: Characters of the preceding window repeated at each
                seam, rounded up to whole elements, so an element near a boundary is
                judged at least once with text on both sides of it. Elements were the
                wrong unit — they bound no text, since a window holds few elements
                exactly when they are large — and 8 of them repeated a median 7,930
                characters per seam, 1.5578 characters sent per unique one. Clamped to
                half the window's characters, which bounds the duplication at 2x however
                the elements are sized.
            element_max_chars: Cap the serialised size of any one element of the
                *prompt*, abridging what is over it. ``0`` disables the cap. Windows
                are cut at element boundaries, so one element larger than the window
                budget pins its window at whatever size it is: 16 of 5,000 documents
                were ``too_long`` on an element of 62,866 to 1,514,442 characters, all
                of them markup that ``cutoff_length`` exempts. Only the prompt shrinks
                — the output is rebuilt from the un-abridged ``map_html`` — so set this
                at or below ``chunk_max_chars`` to make that budget actually hold.
        """
        self.html_field = html_field
        self.model_identifier = model_identifier
        self.prompt_version = prompt_version
        self.cutoff_length = cutoff_length
        self.max_model_len = max_model_len
        self.drop_html_field = drop_html_field
        self.chat_template_mode = chat_template_mode
        self.cache_dir = cache_dir
        self.prompt_path = prompt_path
        self.tokenize = tokenize
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.element_max_chars = element_max_chars
        self.prompt_sha256 = ""
        self._template: str | None = None

        self.resources = Resources(cpus=1.0)
        self.name = "mineru_html_simplify"
        self._tokenizer = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        prompt_field = TOKENS_FIELD if self.tokenize else PROMPT_FIELD
        return ["data"], [MAP_HTML_FIELD, N_ITEMS_FIELD, STATUS_FIELD, CHUNK_IDS_FIELD, prompt_field]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        try:
            import mineru_html  # noqa: F401
        except ImportError as e:
            raise ImportError(_MINERU_INSTALL_HINT) from e

        if self.prompt_path:
            self._template, self.prompt_sha256 = load_prompt_template(self.prompt_path)

        if not self.tokenize:
            # No tokenizer at all: a hosted model's is not ours to run, and pulling
            # the MinerU checkpoint's just to count would be counting with the wrong
            # ruler while looking precise.
            self._tokenizer = None
            return

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_identifier, cache_dir=self.cache_dir)

    def _chat_wrap(self, prompt: str) -> str:
        """Wrap the prompt in the model's chat format.

        The reference implementation applies the template twice — once in
        ``InferenceBackend.process`` and again in
        ``VLLMInferenceBackend.generate`` — leaving a stray
        ``<|hy_begin_of_sentence|><|hy_User|>`` pair inside the user turn. The
        token cost is negligible (~7 per document), but it is not a cosmetic
        difference: on Common Crawl the two prompts disagree on the ``main``
        set for ~20% of documents, against a ~1% run-to-run noise floor. The
        aggregate label distribution is unchanged (29.7% vs 29.6% of elements
        labelled ``main``), so neither is systematically more aggressive.

        ``single`` is the default because it is what the tokenizer's chat
        template is meant to produce; ``upstream_double`` reproduces the
        reference implementation byte for byte, which is what you want when
        comparing against its published benchmark numbers.
        """
        if self.chat_template_mode == "upstream_double":
            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, enable_thinking=False
            )
        return self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )

    def _render_prompt(self, simplified: str, n_items: int) -> str:
        # Imported here rather than at module scope, like the rest of the mineru_html
        # surface: the package is an optional extra and importing it eagerly would make
        # the module unimportable without it.
        from mineru_html.process.build_prompt import get_full_prompt

        if self._template is not None:
            return self._template.format(simplified_html=simplified, n_items=n_items)
        return get_full_prompt(simplified, self.prompt_version)

    def _simplify_one(self, raw: str | bytes) -> tuple[str, str, int, str]:
        """Simplify one document into (prompt, map_html, n_items, status, chunk_ids).

        `prompt` is one string, or a JSON list of window prompts when chunking is on;
        `chunk_ids` is the matching JSON list of id lists, and empty otherwise.
        """
        from mineru_html.process.simplify_html import simplify_html

        html_str = decode_html_cell(raw)
        if not html_str:
            return "", "", 0, "empty_input", ""
        try:
            simplified, map_html = simplify_html(html_str, cutoff_length=self.cutoff_length)
        except Exception as e:  # noqa: BLE001 - upstream raises a wide range of parser errors
            logger.debug(f"simplify_html failed: {e}")
            return "", "", 0, "simplify_error", ""
        # Before `count_item_ids`, and it must not change what that counts: abridging
        # only ever drops descendants of a labelled element, never a labelled element.
        simplified = abridge_oversized_elements(simplified, self.element_max_chars)
        n_items = count_item_ids(simplified)
        # `n_items` is offered to the template as well as the document. Telling a model
        # how many labels are expected is the cheapest defence against the failure that
        # actually happens without constrained decoding — stopping half way — because it
        # gives the answer a length the model can check itself against.
        if self.chunk_max_chars > 0:
            windows = chunk_simplified_html(simplified, self.chunk_max_chars, self.chunk_overlap_chars)
            prompts = [self._render_prompt(html, len(ids)) for html, ids in windows]
            chunk_ids = [ids for _, ids in windows]
            return json.dumps(prompts), map_html, n_items, "ok", json.dumps(chunk_ids)
        return self._render_prompt(simplified, n_items), map_html, n_items, "ok", ""

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        metrics: dict[str, float] = {}

        t0 = time.perf_counter()
        rows = [self._simplify_one(raw) for raw in df[self.html_field]]
        prompts = [r[0] for r in rows]
        map_htmls = [r[1] for r in rows]
        n_items = [r[2] for r in rows]
        statuses = [r[3] for r in rows]
        chunk_ids = [r[4] for r in rows]
        metrics["simplify_time"] = time.perf_counter() - t0

        df = df.copy()
        df[MAP_HTML_FIELD] = map_htmls
        df[N_ITEMS_FIELD] = n_items
        df[STATUS_FIELD] = statuses
        df[CHUNK_IDS_FIELD] = chunk_ids

        budgets = [compact_response_budget(n) for n in n_items]

        if self.tokenize and self.chunk_max_chars > 0:
            # Each window is its own request, so each is tokenized on its own. Tokenizing
            # `prompts` directly here would tokenize the JSON list of windows as if it
            # were a document — the string `[\"You label the elements...` — and send that
            # to the model. The completions route is exactly the one the 0.5B checkpoint
            # uses, so this path is the one that matters most.
            t0 = time.perf_counter()
            per_row: list[list[list[int]]] = []
            for prompt_json, status in zip(prompts, statuses, strict=True):
                if status != "ok" or not prompt_json:
                    per_row.append([])
                    continue
                wrapped = [self._chat_wrap(window) for window in json.loads(prompt_json)]
                per_row.append(self._tokenizer(wrapped, add_special_tokens=False)["input_ids"])
            df[TOKENS_FIELD] = per_row
            metrics["tokenize_time"] = time.perf_counter() - t0
            # The longest window decides: one window that cannot fit is a document that
            # cannot be labelled whole, however comfortably the others sit.
            lengths = [max((len(ids) for ids in row), default=0) for row in per_row]
        elif self.tokenize:
            t0 = time.perf_counter()
            chat_prompts = [self._chat_wrap(p) if s == "ok" else "" for p, s in zip(prompts, statuses, strict=True)]
            # The fast tokenizer raises IndexError on an empty batch, which an
            # otherwise harmless empty partition would turn into a dead pipeline.
            token_ids = self._tokenizer(chat_prompts, add_special_tokens=False)["input_ids"] if chat_prompts else []
            df[TOKENS_FIELD] = token_ids
            metrics["tokenize_time"] = time.perf_counter() - t0
            lengths = [len(ids) for ids in token_ids]
        else:
            # The prompt goes as text, unwrapped: a hosted endpoint applies its own chat
            # template, and applying the MinerU checkpoint's first would nest one
            # model's turn markers inside another's.
            df[PROMPT_FIELD] = [p if s == "ok" else "" for p, s in zip(prompts, statuses, strict=True)]
            lengths = [_estimate_tokens(p) if s == "ok" else 0 for p, s in zip(prompts, statuses, strict=True)]
            metrics["tokenize_time"] = 0.0

        if self.chunk_max_chars > 0:
            # Budget is per window too: a window of 40 elements needs an answer for 40,
            # not for the whole document.
            budgets = [
                max((compact_response_budget(len(ids)) for ids in json.loads(raw or "[]")), default=64)
                for raw in chunk_ids
            ]

        too_long = [
            s == "ok" and (length + budget) > self.max_model_len
            for s, length, budget in zip(statuses, lengths, budgets, strict=True)
        ]
        if any(too_long):
            df.loc[too_long, STATUS_FIELD] = "too_long"
            metrics["too_long_frac"] = sum(too_long) / len(df)

        if self.drop_html_field and self.html_field in df.columns:
            df = df.drop(columns=[self.html_field])

        self._log_metrics(metrics)
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class MinerUHtmlExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU stage: model labels -> pruned HTML -> markdown/text."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        html_field: str = "content",
        url_field: str | None = "url",
        text_field: str = "text",
        main_html_field: str | None = None,
        output_format: str = "mm_md",
        fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura",
        keep_internal_fields: bool = False,
        unlabelled: Literal["main", "other"] = "main",
        split_figures: bool = True,
    ):
        """
        Args:
            html_field: Column with the raw HTML, used by the fallback.
            url_field: Column with the source URL, used to absolutise links.
            text_field: Output column for the extracted content.
            main_html_field: If set, also write the pruned HTML here.
            output_format: ``mm_md`` (default), ``md``, ``json``, ``txt``, or
                ``none`` to emit the pruned HTML without converting it.
                Conversion is the most expensive CPU step in the pipeline
                (~45 ms/document, 41% of it in pylatexenc detecting LaTeX), but
                that is what preserves maths and structured markup, and the
                absolute cost is small next to a GPU — use ``none`` only when
                downstream code consumes HTML directly, not to save CPU.
            fallback: What to do for rows the model could not handle.
            keep_internal_fields: Keep the ``_mineru_*`` scratch columns.
            unlabelled: What an element the answer never mentioned becomes. ``main``
                keeps it, ``other`` drops it. Defaults to keeping, because an
                unlabelled element is one the model did not judge, and deleting text
                on the strength of a judgement that was never made is the more
                expensive mistake — a truncated answer would otherwise silently
                delete the entire back half of a document. Whichever is chosen, the
                document's status says the answer was partial.
            split_figures: Give every image in a ``<figure>`` its own figure before
                converting. The converter keeps only one image per figure — 1 of 2
                and 1 of 3 measured against it — so this is on by default and is a
                no-op on documents whose figures hold one image each. See
                :func:`~nemo_curator.stages.text.html_extraction.assets.split_multi_image_figures`.
        """
        self.html_field = html_field
        self.url_field = url_field
        self.text_field = text_field
        self.main_html_field = main_html_field
        self.output_format = output_format
        self.fallback = fallback
        self.keep_internal_fields = keep_internal_fields
        self.unlabelled = unlabelled
        self.split_figures = split_figures

        self.resources = Resources(cpus=1.0)
        self.name = "mineru_html_extract"
        self._convert = None

    def inputs(self) -> tuple[list[str], list[str]]:
        cols = [MAP_HTML_FIELD, RESPONSE_FIELD, STATUS_FIELD, N_ITEMS_FIELD]
        # "trafilatura" and "bypass" both re-read the original document, so the
        # column is a hard requirement, not an optimisation. Declaring it makes
        # validate_input fail loudly instead of every fallback row silently
        # degrading to fallback="empty".
        if self.fallback != "empty":
            cols.append(self.html_field)
        return ["data"], cols

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = [self.text_field, STATUS_FIELD]
        if self.main_html_field:
            cols.append(self.main_html_field)
        return ["data"], cols

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._fallback_html = FallbackExtractor(self.fallback)

        # Hoisted out of _render: the converter costs ~0.75s to import, which belongs
        # in worker warmup rather than stalling the first batch through the stage.
        if self.output_format != "none":
            from webpage_converter.convert import convert_html_to_structured_data

            self._convert = convert_html_to_structured_data

    def _prune(  # noqa: PLR0913, PLR0917
        self,
        statuses: list[str],
        map_htmls: list[str],
        responses: list[str],
        raw_htmls: list[str | bytes | None],
        n_items: list[int],
        coverages: list[float],
    ) -> list[str]:
        """Turn model labels into main-content HTML, salvaging a partial answer.

        Without constrained decoding an answer arrives complete, truncated, or not at
        all. A truncated one is still mostly right, and throwing it away to re-extract
        with a rule would discard work already paid for — so what parsed is used, and
        what was never mentioned takes ``unlabelled``. Only an answer with nothing in
        it at all is treated as a failure.
        """
        main_htmls: list[str] = []
        for i, status in enumerate(statuses):
            if status != "ok":
                main_htmls.append(self._fallback_html(raw_htmls[i]))
                continue
            try:
                labels = parse_compact_response(responses[i])
                coverage = label_coverage(labels, n_items[i])
                coverages[i] = float(coverage["coverage"])
                if not labels:
                    # The request succeeded and the answer named no element. Pruning on
                    # an empty label map deletes the document and returns "\n" while the
                    # status still says "ok" — the blindness the inference stage guards
                    # against for *lost* requests, reached by a different route. Measured
                    # on 1 of the first 5 documents against a hosted endpoint.
                    statuses[i] = "no_labels"
                    main_htmls.append(self._fallback_html(raw_htmls[i]))
                    continue
                if coverage["coverage"] < 1.0:
                    # Salvaged, not discarded — but never passed off as complete. 96 of
                    # 191 elements labelled produced a document that read as whole and
                    # was missing everything after the 96th.
                    statuses[i] = "partial_labels"
                    labels = {
                        str(item): labels.get(str(item), self.unlabelled) for item in range(1, int(n_items[i]) + 1)
                    }
                main_htmls.append(extract_main_html(map_htmls[i], labels))
            except Exception as e:  # noqa: BLE001 - lxml raises broadly on malformed input
                logger.debug(f"extract_main_html failed: {e}")
                statuses[i] = "extract_error"
                main_htmls.append(self._fallback_html(raw_htmls[i]))
        return main_htmls

    def _render(self, main_htmls: list[str], urls: list[str | None], statuses: list[str]) -> list[str]:
        """Render main-content HTML into the requested output format.

        ``split_figures`` is applied here rather than in ``_prune`` because it is a
        concession to the converter and not part of the extraction: what
        ``main_html_field`` records stays the document as it was extracted.
        """
        if self.output_format == "none":
            return main_htmls

        texts: list[str] = []
        for i, main_html in enumerate(main_htmls):
            if not main_html:
                texts.append("")
                continue
            try:
                html = split_multi_image_figures(main_html) if self.split_figures else main_html
                texts.append(self._convert(main_html=html, url=urls[i], output_format=self.output_format))
            except Exception as e:  # noqa: BLE001 - the converter raises broadly
                logger.debug(f"convert_html_to_structured_data failed: {e}")
                statuses[i] = "convert_error"
                texts.append("")
        return texts

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        metrics: dict[str, float] = {}

        statuses = df[STATUS_FIELD].tolist()
        raw_htmls = df[self.html_field].tolist() if self.html_field in df.columns else [None] * len(df)
        # Normalise pandas nulls to None. The converter absolutises links against the
        # URL and handles None fine, but np.nan and pd.NA both raise inside it -- and
        # unlike _prune, _render has no fallback path, so one null URL turned an
        # otherwise good document into an empty convert_error.
        urls = (
            df[self.url_field].where(df[self.url_field].notna(), None).tolist()
            if self.url_field and self.url_field in df.columns
            else [None] * len(df)
        )

        n_items = df[N_ITEMS_FIELD].tolist() if N_ITEMS_FIELD in df.columns else [0] * len(df)
        coverages = [1.0] * len(df)

        t0 = time.perf_counter()
        main_htmls = self._prune(
            statuses, df[MAP_HTML_FIELD].tolist(), df[RESPONSE_FIELD].tolist(), raw_htmls, n_items, coverages
        )
        metrics["extract_time"] = time.perf_counter() - t0
        df[COVERAGE_FIELD] = coverages
        metrics["mean_label_coverage"] = float(sum(coverages) / len(coverages)) if coverages else 1.0

        t0 = time.perf_counter()
        texts = self._render(main_htmls, urls, statuses)
        metrics["convert_time"] = time.perf_counter() - t0

        df[self.text_field] = texts
        df[STATUS_FIELD] = statuses
        if self.main_html_field:
            df[self.main_html_field] = main_htmls

        if not self.keep_internal_fields:
            drop = [c for c in INTERNAL_FIELDS if c in df.columns]
            if drop:
                df = df.drop(columns=drop)

        metrics["ok_frac"] = float((df[STATUS_FIELD] == "ok").mean())
        self._log_metrics(metrics)

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class MinerUHtmlExtractor(CompositeStage[DocumentBatch, DocumentBatch]):
    """Full MinerU-HTML extraction: simplify -> label -> render, all on CPU.

    Labelling is submitted to an OpenAI-compatible vLLM server that you start and
    own; no stage in this pipeline requests a GPU. See the tutorial README for the
    ``vllm serve`` command, in particular ``--generation-config vllm``, without
    which the checkpoint's own sampling defaults wreck the answers.

    Example:
        >>> from nemo_curator.pipeline import Pipeline
        >>> from nemo_curator.stages.text.html_extraction import MinerUHtmlExtractor
        >>> pipeline = Pipeline(name="mineru")
        >>> pipeline.add_stage(MinerUHtmlExtractor(base_url="http://127.0.0.1:8000"))
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        base_url: str,
        html_field: str = "content",
        url_field: str | None = "url",
        text_field: str = "text",
        model_identifier: str = DEFAULT_MODEL,
        max_model_len: int = 32768,
        structured_outputs: Literal["none", "per_request"] = "per_request",
        output_format: str = "mm_md",
        fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura",
        main_html_field: str | None = None,
        served_model_name: str = "mineru",
        server_concurrency: int = 64,
        simplify_workers: int | None = None,
        inference_workers: int | None = None,
        extract_workers: int | None = None,
        chat_template_mode: Literal["single", "upstream_double"] = "single",
        cache_dir: str | None = None,
        prompt_path: str | None = None,
        api: Literal["completions", "chat"] = "completions",
        api_key_env_var: str = "",
        api_key_file: str = "",
        unlabelled: Literal["main", "other"] = "main",
        keep_internal_fields: bool = False,
        keep_html: bool = False,
        split_figures: bool = True,
        chunk_max_chars: int = 0,
        chunk_overlap_chars: int = 2000,
        element_max_chars: int = 0,
    ):
        """
        Args:
            base_url: Root of the OpenAI-compatible endpoint serving the model,
                with or without a trailing ``/v1``. Required: this pipeline has
                no in-process engine to fall back on.
            html_field: Column holding raw HTML (``str`` or ``bytes``).
            url_field: Column holding the source URL, or ``None``.
            text_field: Output column for extracted content.
            model_identifier: HuggingFace id of the MinerU-HTML checkpoint. Used
                here only for the tokenizer; the server holds the weights.
            max_model_len: Context length the server was started with. Documents
                that do not fit are routed to ``fallback`` without being sent.
            structured_outputs: ``per_request`` (default) constrains each answer
                with :func:`compact_answer_regex`, guaranteeing every element id
                appears exactly once; ``none`` is a few percent faster but lets
                ~7% of answers contain out-of-range ids (harmless -- they match
                no element).
            output_format: ``mm_md``, ``md``, ``json`` or ``txt``.
            fallback: Extraction strategy for rows the model cannot handle.
            main_html_field: If set, also emit the pruned HTML.
            served_model_name: ``--served-model-name`` given to that server.
            server_concurrency: In-flight requests per inference worker. Queue
                depth at the server is this times ``inference_workers``.
            simplify_workers: Worker count for the simplify stage. Size the CPU
                stages so they keep the server saturated.
            inference_workers: Worker count for the inference stage. These are
                CPU workers that do nothing but hold HTTP requests open, so the
                useful number is set by the server's capacity, not the node's:
                raise it (with ``server_concurrency``) until the engines are
                busy. Left unset, backends autoscale the pool from one worker,
                which under-feeds the server for the first part of a short run.
            extract_workers: Worker count for the extract stage.
            chat_template_mode: ``single`` (default) or ``upstream_double`` for
                byte-compatibility with the reference implementation.
            cache_dir: HuggingFace cache directory for the tokenizer.
            prompt_path: A prompt template with a ``{simplified_html}`` placeholder,
                instead of the packaged MinerU prompt.
            api: ``completions`` (default) is the original path — token ids to a vLLM
                server you run. ``chat`` posts text to ``/v1/chat/completions``, which
                is what a hosted endpoint serves; it turns tokenization off, since the
                hosted model's tokenizer is not ours to run, and drops the vLLM-only
                sampler and grammar options. Constrained decoding goes with them, so
                a few percent of answers name elements that do not exist — counted and
                dropped downstream, never silently accepted.
            api_key_env_var: Variable holding the credential for a hosted endpoint.
            api_key_file: Read only when that variable is unset, because a shell
                export does not survive into a Slurm job.
            unlabelled: What an element a partial answer never mentioned becomes.
            split_figures: Give every image in a ``<figure>`` its own figure before
                converting, working around a converter that keeps only one image per
                figure. Needed for any corpus whose figures hold several images.
            keep_internal_fields: Keep the ``_mineru_*`` columns, including the label
                coverage — what an analysis run wants and a production one does not.
            chunk_max_chars: Split each document into windows of at most this many
                characters so a small-context model can see all of it; ``0`` disables it.
            chunk_overlap_chars: Characters of preceding context repeated at each seam,
                to the nearest whole element.
            element_max_chars: Cap on the serialised size of any one element of the
                prompt; ``0`` disables it. An element larger than the window budget
                pins its window at its own size, which is what left 16 of 5,000
                documents ``too_long`` with chunking already on. Set it at or below
                ``chunk_max_chars``.
            keep_html: Keep the raw HTML column in the output. It is dropped by default
                when ``fallback="empty"``, which halves the bytes shipped downstream and
                is safe only because no fallback then needs the original. That is an
                optimisation, not a policy, and the two got tangled: asking for no
                fallback silently also threw away the input the output is meant to be
                compared against. Set this when the run is a gold reference.
        """
        super().__init__()
        self.base_url = base_url
        self.html_field = html_field
        self.url_field = url_field
        self.text_field = text_field
        self.model_identifier = model_identifier
        self.max_model_len = max_model_len
        self.structured_outputs = structured_outputs
        self.output_format = output_format
        self.fallback = fallback
        self.main_html_field = main_html_field
        self.served_model_name = served_model_name
        self.server_concurrency = server_concurrency
        self.simplify_workers = simplify_workers
        self.inference_workers = inference_workers
        self.extract_workers = extract_workers
        self.chat_template_mode = chat_template_mode
        self.cache_dir = cache_dir
        self.prompt_path = prompt_path
        self.api = api
        self.api_key_env_var = api_key_env_var
        self.api_key_file = api_key_file
        self.unlabelled = unlabelled
        self.keep_internal_fields = keep_internal_fields
        self.keep_html = keep_html
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.element_max_chars = element_max_chars
        self.split_figures = split_figures
        self.name = "mineru_html_extractor"

    def decompose(self) -> list[ProcessingStage]:
        simplify = MinerUHtmlSimplifyStage(
            html_field=self.html_field,
            model_identifier=self.model_identifier,
            max_model_len=self.max_model_len,
            chat_template_mode=self.chat_template_mode,
            # trafilatura is the only fallback that needs the original document.
            # Only "empty" can discard the raw HTML. "bypass" returns the original
            # document as its fallback, so dropping the column silently turned it
            # into "empty" -- _fallback_html short-circuits on raw is None.
            drop_html_field=self.fallback == "empty" and not self.keep_html,
            cache_dir=self.cache_dir,
            prompt_path=self.prompt_path,
            # One switch, not two: a text prompt and the chat route are the same
            # decision, and letting them be set apart only creates a pipeline that
            # sends token ids to an endpoint expecting messages.
            tokenize=self.api == "completions",
            chunk_max_chars=self.chunk_max_chars,
            chunk_overlap_chars=self.chunk_overlap_chars,
            element_max_chars=self.element_max_chars,
        )
        inference = MinerUHtmlServerInferenceStage(
            base_url=self.base_url,
            served_model_name=self.served_model_name,
            structured_outputs=self.structured_outputs,
            max_concurrency=self.server_concurrency,
            api=self.api,
            api_key_env_var=self.api_key_env_var,
            api_key_file=self.api_key_file,
        )
        extract = MinerUHtmlExtractStage(
            unlabelled=self.unlabelled,
            keep_internal_fields=self.keep_internal_fields,
            html_field=self.html_field,
            url_field=self.url_field,
            text_field=self.text_field,
            main_html_field=self.main_html_field,
            output_format=self.output_format,
            fallback=self.fallback,
            split_figures=self.split_figures,
        )
        if self.simplify_workers is not None:
            simplify = simplify.with_(num_workers=self.simplify_workers)
        if self.inference_workers is not None:
            inference = inference.with_(num_workers=self.inference_workers)
        if self.extract_workers is not None:
            extract = extract.with_(num_workers=self.extract_workers)
        return [simplify, inference, extract]
