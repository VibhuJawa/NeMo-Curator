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

"""CPU-only MinerU-HTML inference against a persistent vLLM server.

The in-process :class:`MinerUHtmlInferenceStage` calls ``LLM.generate()`` once per
partition and blocks until that partition's *slowest* document finishes. Answer
lengths span ~37x, so the engine drains to a batch of ~1 at every partition
boundary. A persistent endpoint has no partition boundaries: every CPU worker
submits into one continuously-batched queue, so a straggler overlaps with fresh
work instead of stalling a GPU.

Measured on 8x H100 over 10k Common Crawl documents: 55.7 docs/s here versus 33.6
for the best in-process configuration, at identical extraction quality (0.809).

This stage owns no GPU. Host the model with ``vllm serve`` (``--data-parallel-size``
up to the GPU count) or Curator's ``InferenceServer``, and point ``base_url`` at it.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.html_extraction.mineru_html import (
    N_ITEMS_FIELD,
    PROMPT_FIELD,
    RESPONSE_FIELD,
    STATUS_FIELD,
    TOKENS_FIELD,
    compact_answer_regex,
    compact_response_budget,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from openai import AsyncOpenAI


# A prompt is either raw text or pre-tokenized ids (the pipeline pre-tokenizes on CPU).
PromptT = list[int] | str


class MinerUHtmlServerInferenceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Label DOM items by calling a vLLM OpenAI-compatible server. No GPU."""

    def __init__(  # noqa: PLR0913
        self,
        base_url: str = "http://127.0.0.1:8000",
        served_model_name: str = "mineru",
        structured_outputs: Literal["none", "per_request"] = "per_request",
        max_concurrency: int = 64,
        request_timeout_s: float = 600.0,
        max_retries: int = 3,
        cpus: float = 2.0,
    ):
        """
        Args:
            base_url: Server root, with or without a trailing ``/v1``.
            served_model_name: ``--served-model-name`` given to the server.
            structured_outputs: ``per_request`` applies the same regex the in-process
                stage uses (:func:`compact_answer_regex`); ``none`` disables it.
            max_concurrency: In-flight requests per worker. Queue depth at the server is
                this times the worker count -- that product is what keeps the engine
                saturated, so size it against the worker count.
            request_timeout_s: Per-request timeout. Generous by default: under load a
                request can sit queued behind a long prefill, and a spurious timeout
                silently costs a document.
            max_retries: Retries per request, handled by the OpenAI client.
            cpus: CPU reservation per worker.
        """
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.served_model_name = served_model_name
        self.structured_outputs = structured_outputs
        self.max_concurrency = max_concurrency
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries

        self.resources = Resources(cpus=cpus)
        self.name = "mineru_html_server_inference"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [N_ITEMS_FIELD, STATUS_FIELD]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [RESPONSE_FIELD]

    def _client(self) -> AsyncOpenAI:
        from openai import AsyncOpenAI

        # The SDK owns retry with exponential backoff, so this stage does not.
        return AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="unused",  # pragma: allowlist secret
            max_retries=self.max_retries,
            timeout=self.request_timeout_s,
        )

    def _extra_body(self, n: int) -> dict[str, Any]:
        # The checkpoint ships generation_config.json with temperature 0.7 / top_k 20 /
        # repetition_penalty 1.05, and an OpenAI server applies those as request defaults
        # -- the in-process path never sees them because it builds SamplingParams directly.
        # repetition_penalty is the damaging one: the answer is deliberately repetitive
        # ("1main2other3main..."), so penalising repeats degrades exactly the tokens the
        # model must emit, and the longer the document the worse it gets. Measured with
        # the checkpoint defaults applied: extraction_rate 0.015 vs 0.809.
        # Pinning per request means correctness does not depend on how the server was
        # launched (`--generation-config vllm` also fixes it, but is easy to omit).
        body: dict[str, Any] = {"repetition_penalty": 1.0, "top_k": -1}
        if self.structured_outputs == "per_request":
            body["structured_outputs"] = {"regex": compact_answer_regex(n)}
        return body

    async def _one(
        self, client: AsyncOpenAI, sem: asyncio.Semaphore, prompt: PromptT, n: int
    ) -> tuple[str, str | None]:
        """Return ``(text, error)``. Empty text with an error means the request was lost."""
        async with sem:
            try:
                resp = await client.completions.create(
                    model=self.served_model_name,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=compact_response_budget(int(n)),
                    extra_body=self._extra_body(int(n)),
                )
            except Exception as exc:  # noqa: BLE001 - one lost request degrades one row
                # An empty response is the same empty label map the extract stage already
                # handles. Never swallow the reason: a run where every request failed once
                # reported a 2.2x "speedup" because it had silently stopped doing inference.
                logger.warning(f"[mineru-server] request failed after {self.max_retries} retries: {exc!r}")
                return "", repr(exc)
            return resp.choices[0].text, None

    async def _run_all(self, prompts: list[PromptT], n_items: list[int]) -> list[tuple[str, str | None]]:
        sem = asyncio.Semaphore(self.max_concurrency)
        client = self._client()
        try:
            return await asyncio.gather(*[self._one(client, sem, p, n) for p, n in zip(prompts, n_items, strict=True)])
        finally:
            await client.close()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        # A simplified DOM with no _item_id has nothing to label, so it never reaches the
        # server. On Common Crawl that is 6.4% of documents (short ones, so only 0.3% of
        # prefill tokens -- the saving is in scheduling, not FLOPs).
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
        # The completions route takes token ids directly, so CPU-side pre-tokenization is
        # preserved and the server does not re-tokenize.
        if TOKENS_FIELD in df.columns:
            # .tolist() hands back the existing lists; list(map(int, ...)) was ~62 ms of
            # identity calls per partition, all of it before the first request goes out.
            prompts: list[PromptT] = sub[TOKENS_FIELD].tolist()
        else:
            prompts = sub[PROMPT_FIELD].tolist()
        n_items = [int(n) for n in sub[N_ITEMS_FIELD]]

        t0 = time.perf_counter()
        results = asyncio.run(self._run_all(prompts, n_items))
        elapsed = time.perf_counter() - t0

        texts = [t for t, _ in results]
        errors = [e for _, e in results if e]
        df.loc[runnable, RESPONSE_FIELD] = texts

        # A wholesale failure looks downstream like "every document fell back", which is
        # indistinguishable from a fast run unless it is raised here.
        if len(errors) == len(texts):
            msg = (
                f"[mineru-server] all {len(texts)} requests to {self.base_url} failed "
                f"(last error: {errors[-1]}). Refusing to report throughput for a batch "
                "that ran no inference."
            )
            raise RuntimeError(msg)

        self._log_metrics(
            {
                "server_request_time": elapsed,
                "requests": float(len(texts)),
                "requests_per_s": len(texts) / elapsed if elapsed else 0.0,
                "failed_requests": float(len(errors)),
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
