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

This module splits the work along the CPU/GPU boundary:

======================================  ========  ==================================
Stage                                   Hardware  Work
======================================  ========  ==================================
:class:`MinerUHtmlSimplifyStage`        CPU       simplify DOM, build prompt, tokenize
:class:`MinerUHtmlInferenceStage`       GPU       vLLM batch generation
:class:`MinerUHtmlExtractStage`         CPU       parse labels, prune DOM, to markdown
======================================  ========  ==================================

Scale the CPU stages out until they keep the GPU stage saturated. On Common
Crawl the CPU side costs ~80 ms per document per core (20 ms simplify, 8 ms
tokenize, 6 ms prune, 45 ms Markdown conversion), i.e. ~12.5 documents/s/core.
Even an H100 chewing through ~16 documents/s therefore needs under two cores of
CPU alongside it, so there is no reason to trade away conversion quality —
budget the cores and keep the default ``mm_md`` output, math and all.

Use :class:`MinerUHtmlExtractor` to add all three at once.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.html_extraction.mineru_utils import (
    count_item_ids,
    extract_main_html,
    parse_compact_response,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

DEFAULT_MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"

# Column names shared between the three stages.
PROMPT_FIELD = "_mineru_prompt"
TOKENS_FIELD = "_mineru_prompt_tokens"
MAP_HTML_FIELD = "_mineru_map_html"
N_ITEMS_FIELD = "_mineru_n_items"
RESPONSE_FIELD = "_mineru_response"
STATUS_FIELD = "_mineru_status"

_INTERNAL_FIELDS = (PROMPT_FIELD, TOKENS_FIELD, MAP_HTML_FIELD, N_ITEMS_FIELD, RESPONSE_FIELD)

_MINERU_INSTALL_HINT = (
    "mineru_html is required for the MinerU-HTML stages. "
    "Install with: pip install 'mineru_html' (the vllm extra is not needed; "
    "Curator drives vLLM itself)."
)
_VLLM_INSTALL_HINT = "vLLM is required for MinerUHtmlInferenceStage. Install with: pip install nemo_curator[vllm]"


def compact_response_budget(n_items: int) -> int:
    """Token budget for a compact ``{id}{label}`` answer over ``n_items`` elements.

    Measured on Common Crawl, a compact answer costs ~2.1 tokens per element;
    4 plus a fixed 64-token slack is a comfortable ceiling. Sizing each request
    individually (instead of the reference implementation's flat 16k) is what
    lets a 32k-context engine accept documents longer than 16k tokens at all.
    """
    return max(64, n_items * 4 + 64)


class MinerUHtmlSimplifyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU stage: raw HTML -> model prompt + ``_item_id``-annotated HTML.

    Emits :data:`PROMPT_FIELD` (or :data:`TOKENS_FIELD` when ``pretokenize``),
    :data:`MAP_HTML_FIELD`, :data:`N_ITEMS_FIELD` and :data:`STATUS_FIELD`.
    Rows whose HTML cannot be parsed are kept with a non-``ok`` status so the
    extract stage can apply the configured fallback.
    """

    def __init__(  # noqa: PLR0913
        self,
        html_field: str = "content",
        model_identifier: str = DEFAULT_MODEL,
        prompt_version: str = "short_compact",
        cutoff_length: int = 500,
        max_model_len: int = 32768,
        pretokenize: bool = True,
        drop_html_field: bool = False,
        chat_template_mode: Literal["single", "upstream_double"] = "single",
        cache_dir: str | None = None,
    ):
        """
        Args:
            html_field: Column holding raw HTML as ``str`` or ``bytes``.
            model_identifier: HuggingFace id, used here only for the tokenizer.
            prompt_version: MinerU prompt template; ``short_compact`` matches
                the ``*-compact`` checkpoints.
            cutoff_length: Per-element text truncation in the simplified DOM.
            max_model_len: Engine context length. Documents that cannot fit
                their prompt plus answer budget are marked ``too_long``.
            pretokenize: Tokenize here rather than inside the GPU actor. Keeps
                the single vLLM process off the critical path.
            drop_html_field: Drop the raw HTML column once simplified. Halves
                the bytes shipped downstream, but disables the trafilatura
                fallback, which needs the original document.
            chat_template_mode: ``single`` applies the chat template once;
                ``upstream_double`` reproduces the reference implementation's
                doubled template. See :meth:`_chat_wrap`.
            cache_dir: HuggingFace cache directory.
        """
        self.html_field = html_field
        self.model_identifier = model_identifier
        self.prompt_version = prompt_version
        self.cutoff_length = cutoff_length
        self.max_model_len = max_model_len
        self.pretokenize = pretokenize
        self.drop_html_field = drop_html_field
        self.chat_template_mode = chat_template_mode
        self.cache_dir = cache_dir

        self.resources = Resources(cpus=1.0)
        self.name = "mineru_html_simplify"
        self._tokenizer = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.html_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = [MAP_HTML_FIELD, N_ITEMS_FIELD, STATUS_FIELD]
        cols.append(TOKENS_FIELD if self.pretokenize else PROMPT_FIELD)
        return ["data"], cols

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        try:
            import mineru_html  # noqa: F401
        except ImportError as e:
            raise ImportError(_MINERU_INSTALL_HINT) from e

        # The tokenizer is needed even when not pre-tokenizing: vLLM's
        # generate() does not apply a chat template, so the prompt has to
        # arrive already wrapped.
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

    def _simplify_one(self, raw: str | bytes) -> tuple[str, str, int, str]:
        """Simplify one document into (prompt, map_html, n_items, status)."""
        from mineru_html.process.build_prompt import get_full_prompt
        from mineru_html.process.simplify_html import simplify_html

        html_str = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        if not html_str:
            return "", "", 0, "empty_input"
        try:
            simplified, map_html = simplify_html(html_str, cutoff_length=self.cutoff_length)
        except Exception as e:  # noqa: BLE001 - upstream raises a wide range of parser errors
            logger.debug(f"simplify_html failed: {e}")
            return "", "", 0, "simplify_error"
        return get_full_prompt(simplified, self.prompt_version), map_html, count_item_ids(simplified), "ok"

    def _simplify_column(self, raw_html: Iterable[str | bytes]) -> tuple[list[str], list[str], list[int], list[str]]:
        """Simplify each document, returning (prompts, map_htmls, n_items, statuses)."""
        rows = [self._simplify_one(raw) for raw in raw_html]
        prompts, map_htmls, n_items, statuses = (list(col) for col in zip(*rows, strict=True)) if rows else ([],) * 4
        return prompts, map_htmls, n_items, statuses

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        metrics: dict[str, float] = {}

        t0 = time.perf_counter()
        prompts, map_htmls, n_items, statuses = self._simplify_column(df[self.html_field])
        metrics["simplify_time"] = time.perf_counter() - t0

        df = df.copy()
        df[MAP_HTML_FIELD] = map_htmls
        df[N_ITEMS_FIELD] = n_items
        df[STATUS_FIELD] = statuses

        t0 = time.perf_counter()
        chat_prompts = [self._chat_wrap(p) if s == "ok" else "" for p, s in zip(prompts, statuses, strict=True)]
        if self.pretokenize:
            token_ids = self._tokenizer(chat_prompts, add_special_tokens=False)["input_ids"]
            df[TOKENS_FIELD] = token_ids
            lengths = [len(t) for t in token_ids]
        else:
            df[PROMPT_FIELD] = chat_prompts
            # ~3.2 characters per token on this corpus; dividing by 2 keeps the
            # over-long pre-filter conservative so nothing that fits is dropped.
            lengths = [len(p) // 2 for p in chat_prompts]
        metrics["tokenize_time"] = time.perf_counter() - t0

        budgets = [compact_response_budget(n) for n in n_items]
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


class MinerUHtmlInferenceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """GPU stage: label every ``_item_id`` as ``main`` or ``other`` with vLLM."""

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str = DEFAULT_MODEL,
        structured_outputs: Literal["none", "per_request"] = "per_request",
        max_model_len: int = 32768,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 256,
        gpu_memory_utilization: float = 0.90,
        kv_cache_dtype: str = "auto",
        quantization: str | None = None,
        vllm_init_kwargs: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            model_identifier: HuggingFace id of the MinerU-HTML checkpoint.
            structured_outputs: ``per_request`` (default) applies the
                reference implementation's ``1(main|other)2(main|other)...``
                regex, guaranteeing every id appears exactly once; ``none``
                is a few percent faster but lets ~7% of answers contain
                out-of-range ids (harmless — they match no element).
            max_model_len: Engine context length. 32k covers ~99% of Common
                Crawl documents; longer ones are routed to the fallback.
                Attention is quadratic and this workload's cost is concentrated
                in a long tail, so lowering this is the biggest single knob on
                the stage — see BENCHMARKS.md.
            max_num_batched_tokens: Prefill budget per engine step.
            max_num_seqs: Maximum concurrent sequences.
            gpu_memory_utilization: Fraction of GPU memory for the engine.
            kv_cache_dtype: ``auto`` or ``fp8``. ``fp8`` is the largest GPU
                win measured (1.56x) and perturbs labels less than quantizing
                weights. Not the default because Ada needs FlashInfer's
                prebuilt kernels first (``pip install --extra-index-url
                https://flashinfer.ai/whl/<cuda>/ flashinfer-jit-cache==<same
                version as flashinfer-python>``), and a default that fails to
                start is worse than one that is slower.
            quantization: vLLM weight quantization, e.g. ``"fp8"``. Bounded
                by the ~28% of prefill FLOPs that are linear; stacked on FP8 KV
                it bought 4% for a 5-point drop in label agreement, so it is
                not recommended.
            vllm_init_kwargs: Extra keyword arguments forwarded to ``LLM``.
            cache_dir: HuggingFace cache directory.
            hf_token: HuggingFace token.
            verbose: Show vLLM progress bars and stats.
        """
        self.model_identifier = model_identifier
        self.structured_outputs = structured_outputs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_dtype = kv_cache_dtype
        self.quantization = quantization
        self.vllm_init_kwargs = vllm_init_kwargs or {}
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.verbose = verbose

        # vLLM v1 runs a separate EngineCore process alongside this one, and the
        # structured-output workers compile grammars on the CPU. Reserving two
        # cores keeps the executor from oversubscribing the node and starving
        # the engine.
        self.resources = Resources(cpus=2.0, gpus=1.0)
        self.name = "mineru_html_inference"
        self._llm = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [N_ITEMS_FIELD, STATUS_FIELD]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [RESPONSE_FIELD]

    def _init_llm(self, local_files_only: bool) -> None:
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(_VLLM_INSTALL_HINT) from e
        from huggingface_hub import snapshot_download

        # vLLM does not thread download_dir through config resolution, so hand
        # it a resolved snapshot path rather than a repo id.
        model_path = snapshot_download(
            self.model_identifier,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            local_files_only=local_files_only,
        )

        kwargs: dict[str, Any] = {
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            "disable_log_stats": not self.verbose,
        }
        if self.quantization:
            kwargs["quantization"] = self.quantization
        kwargs.update(self.vllm_init_kwargs)
        self._llm = LLM(model=model_path, **kwargs)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Download weights and prime the torch.compile cache once per node.

        This runs as a Ray task that finishes before any worker actor starts,
        so building the engine here costs an extra load but keeps concurrent
        workers on the same node from racing on the compile cache. Same
        approach as the other vLLM-backed stages in Curator.
        """
        if not self.verbose:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        self._init_llm(local_files_only=False)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._llm is None:
            self._init_llm(local_files_only=True)

    def teardown(self) -> None:
        import torch

        del self._llm
        self._llm = None
        gc.collect()
        torch.cuda.empty_cache()

    def _sampling_params(self, n_items: list[int]) -> list[Any]:
        from vllm import SamplingParams

        params = []
        for n in n_items:
            extra: dict[str, Any] = {}
            if self.structured_outputs == "per_request":
                from vllm.sampling_params import StructuredOutputsParams

                pattern = "".join(f"{i}(main|other)" for i in range(1, int(n) + 1))
                extra["structured_outputs"] = StructuredOutputsParams(regex=rf"<answer>\s*{pattern}\s*</answer>")
            params.append(SamplingParams(temperature=0.0, max_tokens=compact_response_budget(int(n)), **extra))
        return params

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        # A simplified DOM with no _item_id has nothing to label. Leaving the
        # response empty gives the extract stage the same empty label map that
        # the model's answer would have resolved to, without spending a request
        # slot. On Common Crawl this is 6.4% of documents (they are short, so
        # only 0.3% of prefill tokens — the saving is in scheduling, not FLOPs).
        runnable = (df[STATUS_FIELD] == "ok") & (df[N_ITEMS_FIELD] > 0)
        df[RESPONSE_FIELD] = ""

        if not runnable.any():
            return DocumentBatch(
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        sub = df.loc[runnable]
        if TOKENS_FIELD in df.columns:
            from vllm.inputs import TokensPrompt

            prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in sub[TOKENS_FIELD]]
        else:
            prompts = sub[PROMPT_FIELD].tolist()

        sampling = self._sampling_params(sub[N_ITEMS_FIELD].tolist())

        t0 = time.perf_counter()
        outputs = self._llm.generate(prompts, sampling_params=sampling, use_tqdm=self.verbose)
        elapsed = time.perf_counter() - t0

        df.loc[runnable, RESPONSE_FIELD] = [o.outputs[0].text if o.outputs else "" for o in outputs]

        prompt_tokens = sum(len(o.prompt_token_ids or ()) for o in outputs)
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
        self._log_metrics(
            {
                "vllm_generate_time": elapsed,
                "prompt_tokens": float(prompt_tokens),
                "generated_tokens": float(gen_tokens),
                "prefill_tokens_per_s": prompt_tokens / elapsed if elapsed else 0.0,
            }
        )

        if TOKENS_FIELD in df.columns:
            df = df.drop(columns=[TOKENS_FIELD])
        elif PROMPT_FIELD in df.columns:
            df = df.drop(columns=[PROMPT_FIELD])

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class MinerUHtmlExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU stage: model labels -> pruned HTML -> markdown/text."""

    def __init__(  # noqa: PLR0913
        self,
        html_field: str = "content",
        url_field: str | None = "url",
        text_field: str = "text",
        main_html_field: str | None = None,
        output_format: str = "mm_md",
        fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura",
        keep_internal_fields: bool = False,
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
        """
        self.html_field = html_field
        self.url_field = url_field
        self.text_field = text_field
        self.main_html_field = main_html_field
        self.output_format = output_format
        self.fallback = fallback
        self.keep_internal_fields = keep_internal_fields

        self.resources = Resources(cpus=1.0)
        self.name = "mineru_html_extract"
        self._fallback_handler = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [MAP_HTML_FIELD, RESPONSE_FIELD, STATUS_FIELD]

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = [self.text_field, STATUS_FIELD]
        if self.main_html_field:
            cols.append(self.main_html_field)
        return ["data"], cols

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        try:
            from mineru_html.process.map_to_main import get_fallback_handler
        except ImportError as e:
            raise ImportError(_MINERU_INSTALL_HINT) from e
        self._fallback_handler = get_fallback_handler(self.fallback)

    def _fallback_html(self, raw: str | bytes | None) -> str:
        if self.fallback == "empty" or raw is None:
            return ""
        html_str = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        try:
            return self._fallback_handler.fallback_func(html_str)
        except Exception as e:  # noqa: BLE001 - trafilatura raises broadly on malformed input
            logger.debug(f"fallback extraction failed: {e}")
            return ""

    def _prune(
        self, statuses: list[str], map_htmls: list[str], responses: list[str], raw_htmls: list[str | bytes | None]
    ) -> list[str]:
        """Turn model labels into main-content HTML, falling back where needed."""
        main_htmls: list[str] = []
        for i, status in enumerate(statuses):
            if status != "ok":
                main_htmls.append(self._fallback_html(raw_htmls[i]))
                continue
            try:
                main_htmls.append(extract_main_html(map_htmls[i], parse_compact_response(responses[i])))
            except Exception as e:  # noqa: BLE001 - lxml raises broadly on malformed input
                logger.debug(f"extract_main_html failed: {e}")
                statuses[i] = "extract_error"
                main_htmls.append(self._fallback_html(raw_htmls[i]))
        return main_htmls

    def _render(self, main_htmls: list[str], urls: list[str | None], statuses: list[str]) -> list[str]:
        """Render main-content HTML into the requested output format."""
        if self.output_format == "none":
            return main_htmls

        from webpage_converter.convert import convert_html_to_structured_data

        texts: list[str] = []
        for i, main_html in enumerate(main_htmls):
            if not main_html:
                texts.append("")
                continue
            try:
                texts.append(
                    convert_html_to_structured_data(main_html=main_html, url=urls[i], output_format=self.output_format)
                )
            except Exception as e:  # noqa: BLE001 - converter raises broadly on malformed input
                logger.debug(f"convert_html_to_structured_data failed: {e}")
                statuses[i] = "convert_error"
                texts.append("")
        return texts

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        metrics: dict[str, float] = {}

        statuses = df[STATUS_FIELD].tolist()
        raw_htmls = df[self.html_field].tolist() if self.html_field in df.columns else [None] * len(df)
        urls = df[self.url_field].tolist() if self.url_field and self.url_field in df.columns else [None] * len(df)

        t0 = time.perf_counter()
        main_htmls = self._prune(statuses, df[MAP_HTML_FIELD].tolist(), df[RESPONSE_FIELD].tolist(), raw_htmls)
        metrics["extract_time"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        texts = self._render(main_htmls, urls, statuses)
        metrics["convert_time"] = time.perf_counter() - t0

        df[self.text_field] = texts
        df[STATUS_FIELD] = statuses
        if self.main_html_field:
            df[self.main_html_field] = main_htmls

        if not self.keep_internal_fields:
            drop = [c for c in _INTERNAL_FIELDS if c in df.columns]
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
    """Full MinerU-HTML extraction: simplify (CPU) -> label (GPU) -> render (CPU).

    Example:
        >>> from nemo_curator.pipeline import Pipeline
        >>> from nemo_curator.stages.text.html_extraction import MinerUHtmlExtractor
        >>> pipeline = Pipeline(name="mineru")
        >>> pipeline.add_stage(MinerUHtmlExtractor(html_field="content"))
    """

    def __init__(  # noqa: PLR0913
        self,
        html_field: str = "content",
        url_field: str | None = "url",
        text_field: str = "text",
        model_identifier: str = DEFAULT_MODEL,
        max_model_len: int = 32768,
        structured_outputs: Literal["none", "per_request"] = "per_request",
        kv_cache_dtype: str = "auto",
        quantization: str | None = None,
        output_format: str = "mm_md",
        fallback: Literal["trafilatura", "bypass", "empty"] = "trafilatura",
        main_html_field: str | None = None,
        simplify_workers: int | None = None,
        extract_workers: int | None = None,
        pretokenize: bool = True,
        chat_template_mode: Literal["single", "upstream_double"] = "single",
        vllm_init_kwargs: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            html_field: Column holding raw HTML (``str`` or ``bytes``).
            url_field: Column holding the source URL, or ``None``.
            text_field: Output column for extracted content.
            model_identifier: HuggingFace id of the MinerU-HTML checkpoint.
            max_model_len: vLLM context length.
            structured_outputs: See :class:`MinerUHtmlInferenceStage`.
            kv_cache_dtype: ``auto`` or ``fp8``.
            quantization: vLLM weight quantization, e.g. ``"fp8"``.
            output_format: ``mm_md``, ``md``, ``json`` or ``txt``.
            fallback: Extraction strategy for rows the model cannot handle.
            main_html_field: If set, also emit the pruned HTML.
            simplify_workers: Worker count for the simplify stage. Size this
                so the CPU stages keep the GPU busy.
            extract_workers: Worker count for the extract stage.
            pretokenize: Tokenize on the CPU workers instead of the GPU actor.
            chat_template_mode: ``single`` (default) or ``upstream_double`` for
                byte-compatibility with the reference implementation.
            vllm_init_kwargs: Extra keyword arguments forwarded to ``LLM``.
            cache_dir: HuggingFace cache directory.
            verbose: Show vLLM progress bars and stats.
        """
        super().__init__()
        self.html_field = html_field
        self.url_field = url_field
        self.text_field = text_field
        self.model_identifier = model_identifier
        self.max_model_len = max_model_len
        self.structured_outputs = structured_outputs
        self.kv_cache_dtype = kv_cache_dtype
        self.quantization = quantization
        self.output_format = output_format
        self.fallback = fallback
        self.main_html_field = main_html_field
        self.simplify_workers = simplify_workers
        self.extract_workers = extract_workers
        self.pretokenize = pretokenize
        self.chat_template_mode = chat_template_mode
        self.vllm_init_kwargs = vllm_init_kwargs
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.name = "mineru_html_extractor"

    def decompose(self) -> list[ProcessingStage]:
        simplify = MinerUHtmlSimplifyStage(
            html_field=self.html_field,
            model_identifier=self.model_identifier,
            max_model_len=self.max_model_len,
            pretokenize=self.pretokenize,
            chat_template_mode=self.chat_template_mode,
            # trafilatura is the only fallback that needs the original document.
            drop_html_field=self.fallback != "trafilatura",
            cache_dir=self.cache_dir,
        )
        inference = MinerUHtmlInferenceStage(
            model_identifier=self.model_identifier,
            structured_outputs=self.structured_outputs,
            max_model_len=self.max_model_len,
            kv_cache_dtype=self.kv_cache_dtype,
            quantization=self.quantization,
            vllm_init_kwargs=self.vllm_init_kwargs,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )
        extract = MinerUHtmlExtractStage(
            html_field=self.html_field,
            url_field=self.url_field,
            text_field=self.text_field,
            main_html_field=self.main_html_field,
            output_format=self.output_format,
            fallback=self.fallback,
        )
        if self.simplify_workers is not None:
            simplify = simplify.with_(num_workers=self.simplify_workers)
        if self.extract_workers is not None:
            extract = extract.with_(num_workers=self.extract_workers)
        return [simplify, inference, extract]
