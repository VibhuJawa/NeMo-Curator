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

This stage owns no GPU: the engines live in a separate ``vllm serve`` (or Curator's
``InferenceServer``) and CPU workers submit to its OpenAI-compatible endpoint.

Curator briefly also shipped an in-process vLLM stage, which is why the choice was
measured rather than assumed. On 8x H100 over 10k Common Crawl documents
(``benchmarking/mineru-html-benchmark.yaml``), at identical extraction quality
(0.810), the two were at parity on inference wall time -- 78.1s in-process against
75.5s here. Per document this stage is in fact ~2x slower (120.8ms vs 62.5ms) from
HTTP and serialization overhead, and only keeps up because that latency is spread
across twice as many workers, which are cheap because they are CPU-only.

The reason to run this way is therefore operational, not speed: ~78s of vLLM engine
startup is paid once outside the run instead of inside every run, the engines scale,
restart and are shared independently of the pipeline, and a Curator job needs no GPU
allocation at all.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.html_extraction.mineru_utils import (
    N_ITEMS_FIELD,
    RESPONSE_FIELD,
    STATUS_FIELD,
    TOKENS_FIELD,
    compact_answer_regex,
    compact_response_budget,
)
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from nemo_curator.backends.base import WorkerMetadata


class MinerUHtmlServerInferenceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Label DOM items by calling a vLLM OpenAI-compatible server. No GPU."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        base_url: str = "http://127.0.0.1:8000",
        served_model_name: str = "mineru",
        structured_outputs: Literal["none", "per_request"] = "per_request",
        max_concurrency: int = 64,
        request_timeout_s: float = 600.0,
        max_retries: int = 3,
        cpus: float = 1.0,
    ):
        """
        Args:
            base_url: Server root, with or without a trailing ``/v1``.
            served_model_name: ``--served-model-name`` given to the server.
            structured_outputs: ``per_request`` constrains each answer with
                :func:`compact_answer_regex`; ``none`` disables it.
            max_concurrency: In-flight requests per worker. Queue depth at the server is
                this times the worker count -- that product is what keeps the engine
                saturated, so size it against the worker count.
            request_timeout_s: Per-request timeout. Generous by default: under load a
                request can sit queued behind a long prefill, and a spurious timeout
                silently costs a document.
            max_retries: Retries per request, handled by the OpenAI client.
            cpus: CPU reservation per worker. This stage is HTTP-bound -- it holds
                requests open and JSON-encodes token ids -- so one core is ample.
                It is not a soft hint: Ray Data charges this against the cluster
                total for the pool's whole lifetime, so concurrent workers are
                capped at ``(cores - simplify - extract) / cpus``. At the previous
                default of 2.0, asking for 80 inference workers silently ran 48.
        """
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.served_model_name = served_model_name
        self.structured_outputs = structured_outputs
        self.max_concurrency = max_concurrency
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries

        self.resources = Resources(cpus=cpus)
        self.name = "mineru_html_server_inference"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session_client: AsyncOpenAI | None = None
        self._openai_cls = None
        self._completion_cls = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [N_ITEMS_FIELD, STATUS_FIELD, TOKENS_FIELD]

    def outputs(self) -> tuple[list[str], list[str]]:
        # STATUS_FIELD is an output too: a row whose request was lost is marked
        # "inference_error" here so the extract stage falls back for it.
        return ["data"], [RESPONSE_FIELD, STATUS_FIELD]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        # Importing openai costs ~2.7s, which belongs in worker warmup rather than
        # inside the first batch while the server sits idle. Building the client
        # stays lazy -- see _session for why.
        self._load_sdk()

    def _load_sdk(self) -> None:
        if self._completion_cls is None:
            from openai import AsyncOpenAI
            from openai.types import Completion

            self._openai_cls = AsyncOpenAI
            self._completion_cls = Completion

    def _client(self) -> AsyncOpenAI:
        self._load_sdk()
        # The SDK owns retry with exponential backoff, so this stage does not.
        return self._openai_cls(
            base_url=f"{self.base_url}/v1",
            api_key="unused",  # pragma: allowlist secret
            max_retries=self.max_retries,
            timeout=self.request_timeout_s,
        )

    def _session(self) -> tuple[asyncio.AbstractEventLoop, AsyncOpenAI]:
        """The loop and client this worker owns for its whole lifetime.

        Built on first use rather than in ``setup()`` so a worker that only ever
        sees skippable rows opens no connection at all. Both have to outlive the
        batch: ``asyncio.run`` closes its loop on the way out, which invalidates
        any client bound to it, so rebuilding the client per batch was the only
        option -- and that threw away the keep-alive pool, making every batch pay
        up to ``max_concurrency`` TCP handshakes before its first request.
        """
        if self._session_client is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._session_client = self._client()
        return self._loop, self._session_client

    def teardown(self) -> None:
        if self._session_client is not None:
            self._loop.run_until_complete(self._session_client.close())
            self._loop.close()
            asyncio.set_event_loop(None)
            self._session_client = None
            self._loop = None

    def _extra_body(self, n: int) -> dict[str, Any]:
        # The checkpoint ships generation_config.json with temperature 0.7 / top_k 20 /
        # repetition_penalty 1.05, and an OpenAI server applies those as request defaults
        # unless it was started with `--generation-config vllm`.
        # repetition_penalty is the damaging one: the answer is deliberately repetitive
        # ("1main2other3main..."), so penalising repeats degrades exactly the tokens the
        # model must emit, and the longer the document the worse it gets. Measured with
        # the checkpoint defaults applied: extraction_rate 0.015 vs 0.809.
        # Pinning per request means correctness does not depend on how the server was
        # launched (`--generation-config vllm` also fixes it, but is easy to omit).
        body: dict[str, Any] = {"repetition_penalty": 1.0, "top_k": -1}
        if self.structured_outputs == "per_request":
            # Dynamo's OpenAI frontend translates ``guided_regex`` to the
            # backend's StructuredOutputsParams.
            body["guided_regex"] = compact_answer_regex(n)
        return body

    async def _one(
        self, client: AsyncOpenAI, sem: asyncio.Semaphore, prompt: list[int], n: int
    ) -> tuple[str, str | None]:
        """Return ``(text, error)``. Empty text with an error means the request was lost."""
        async with sem:
            try:
                # client.post, not client.completions.create: the typed method runs the
                # body through the SDK's param transform, which recurses per list
                # element -- ~2 calls per token, so ~47 ms of CPU for a 5k-token prompt
                # against ~2 ms here (measured on 200 real documents). That is ~37% of
                # the pipeline's entire CPU budget, and worse, the transform never
                # awaits, so it runs on this worker's event loop and caps it near 20
                # requests/s no matter what max_concurrency says.
                # post() still goes through _base_client._request, so retries, timeout
                # and the keep-alive pool are unchanged.
                resp = await client.post(
                    "/completions",
                    cast_to=self._completion_cls,
                    body={
                        "model": self.served_model_name,
                        "prompt": prompt,
                        "temperature": 0.0,
                        "max_tokens": compact_response_budget(n),
                        **self._extra_body(n),
                    },
                )
                # Inside the try: a malformed response with an empty `choices` list would
                # otherwise raise IndexError straight out of gather() and take the whole
                # partition down, where every other failure only degrades one row.
                return resp.choices[0].text, None
            except Exception as exc:  # noqa: BLE001 - one lost request degrades one row
                # Never swallow the reason: a run where every request failed once reported
                # a 2.2x "speedup" because it had silently stopped doing inference. The
                # caller also marks this row's status so it falls back rather than passing
                # an empty response off as a document with no main content.
                logger.warning(f"[mineru-server] request failed after {self.max_retries} retries: {exc!r}")
                return "", repr(exc)

    async def _run_all(
        self, client: AsyncOpenAI, prompts: list[list[int]], n_items: list[int]
    ) -> list[tuple[str, str | None]]:
        sem = asyncio.Semaphore(self.max_concurrency)
        return await asyncio.gather(*[self._one(client, sem, p, n) for p, n in zip(prompts, n_items, strict=True)])

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        # A simplified DOM with no _item_id has nothing to label, so it never reaches the
        # server. On Common Crawl that is 6.4% of documents (short ones, so only 0.3% of
        # prefill tokens -- the saving is in scheduling, not FLOPs).
        runnable = (df[STATUS_FIELD] == "ok") & (df[N_ITEMS_FIELD] > 0)
        df[RESPONSE_FIELD] = ""

        if runnable.any():
            sub = df.loc[runnable]
            # .tolist() hands back the existing lists; list(map(int, ...)) was ~62 ms of
            # identity calls per partition, all of it before the first request goes out.
            prompts: list[list[int]] = sub[TOKENS_FIELD].tolist()
            n_items: list[int] = sub[N_ITEMS_FIELD].tolist()

            loop, client = self._session()
            t0 = time.perf_counter()
            results = loop.run_until_complete(self._run_all(client, prompts, n_items))
            elapsed = time.perf_counter() - t0

            texts = [t for t, _ in results]
            errors = [e for _, e in results if e]
            df.loc[runnable, RESPONSE_FIELD] = texts

            # Mark the rows whose request was lost. Without this a PARTIAL failure is
            # invisible: the row keeps status "ok", the extract stage parses the empty
            # response into an empty label map, prunes the whole document and emits "\n"
            # -- a blank result that never triggers the fallback and still scores as a
            # success in status_ok_rate and nonempty_rate. One replica restarting
            # mid-run is enough to lose thousands of documents this way. Assigning
            # through the boolean mask keeps this positional, so it is correct even
            # when the frame carries duplicate index labels.
            if errors:
                failed = runnable.copy()
                failed[runnable] = [e is not None for _, e in results]
                df.loc[failed, STATUS_FIELD] = "inference_error"

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

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df.drop(columns=[TOKENS_FIELD], errors="ignore"),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
