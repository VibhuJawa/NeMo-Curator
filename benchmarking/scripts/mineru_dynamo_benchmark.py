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

"""MinerU-HTML against a Curator-managed Dynamo server.

The standalone-``vllm serve`` variant (``mineru_html_benchmark.py``) needs an
endpoint started and torn down by hand. This one hands that to Curator: an
:class:`InferenceServer` with a :class:`DynamoVLLMModelConfig` brings up
``num_replicas`` engines inside the same Ray cluster the pipeline runs on, and
the pipeline talks to ``server.endpoint``.

Two knobs are what this script exists to sweep:

* ``M`` -- ``--inference-workers``, the CPU actors that call the server.
* ``B`` -- ``--server-concurrency``, in-flight requests each of them holds.

``M x B`` is the total in flight, and ``M x B / num_replicas`` is the per-replica
queue depth that determines whether the engines stay saturated. A reasonable
starting point is 512-1024 per replica; ``M = 4 x num_replicas`` is a sane
default for M because these workers are HTTP-bound and cost almost no CPU each.

Note B is in-flight requests per worker, NOT documents per request. Each document
carries its own ``structured_outputs`` regex, built from its own element count,
and that field is request-level in the OpenAI schema -- so N documents are always
N requests, however they are scheduled.
"""

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.core.serve import (
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
)
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.html_extraction import DEFAULT_MODEL, MinerUHtmlExtractor
from nemo_curator.stages.text.io.writer import ParquetWriter

MIN_SUBSTANTIVE_CHARS = 200
STATUS_FIELD = "_mineru_status"


def build_server(args: argparse.Namespace) -> InferenceServer:
    """One Dynamo model, ``--num-replicas`` engines, one GPU each."""
    engine_kwargs = {
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "generation_config": "vllm",
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.kv_cache_dtype != "auto":
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    # Ray builds the Dynamo actor a uv venv from scratch, so nothing installed in
    # the driver venv is visible to the engine. Anything an engine flag needs has
    # to be declared here; merge_runtime_envs unions this with Dynamo's own pin
    # rather than replacing it.
    extra_packages: list[str] = []
    if args.speculative_tokens > 0:
        # Suffix speculative decoding lives in arctic-inference. Without it the
        # engine dies at startup with "Arctic Inference is required for suffix
        # decoding".
        #
        # Leave this UNPINNED. vLLM 0.23's own error message recommends ==0.1.1,
        # and following that is what fails: every arctic-inference release is
        # sdist-only, so it always builds from source, and 0.1.1's build requires
        # torch==2.7.0 -- unsatisfiable against the torch vLLM 0.23 pulls in, which
        # fails the entire runtime_env setup rather than just this package.
        # Unpinned resolves to 0.2.0 and builds fine; its own vllm==0.18.0 pin sits
        # behind an extra we do not request, so it never fights the actor's vLLM.
        extra_packages.append(args.arctic_inference_spec)
        engine_kwargs["speculative_config"] = {
            "method": "suffix",
            "num_speculative_tokens": args.speculative_tokens,
            "suffix_decoding_max_spec_factor": 2.0,
            "suffix_decoding_max_cached_requests": 10000,
        }
    # vLLM >=0.23 routes sampling through FlashInfer and JIT-compiles the kernel
    # during profile_run, which needs nvcc. The Dynamo actor's venv has no CUDA
    # toolkit, so the engine dies with "Could not find nvcc and default
    # cuda_home='/usr/local/cuda' doesn't exist" -- during _dummy_sampler_run,
    # not over the KV cache, so turning off fp8 does not help. Use vLLM's own
    # non-FlashInfer sampler instead; installing flashinfer-jit-cache would also
    # work but needs a version-matched wheel from a custom index.
    env_vars = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    runtime_env: dict = {"env_vars": env_vars}
    if extra_packages:
        runtime_env["uv"] = {"packages": extra_packages}

    return InferenceServer(
        models=[
            DynamoVLLMModelConfig(
                model_identifier=args.model,
                num_replicas=args.num_replicas,
                engine_kwargs=engine_kwargs,
                runtime_env=runtime_env,
            )
        ],
        backend=DynamoServerConfig(),
        health_check_timeout_s=args.server_timeout_s,
    )


def build_pipeline(args: argparse.Namespace, base_url: str, output_dir: Path) -> Pipeline:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tutorials/text/mineru-html-extraction"))
    from run_pipeline import create_html_reader

    pipeline = Pipeline(
        name="mineru_html_dynamo",
        description="MinerU-HTML against a Curator-managed Dynamo server",
    )
    pipeline.add_stage(
        create_html_reader(
            input_path=args.input_path,
            html_field=args.html_field,
            url_field=args.url_field,
            blocksize=None,
            files_per_partition=args.files_per_partition,
        )
    )
    pipeline.add_stage(
        MinerUHtmlExtractor(
            base_url=base_url,
            served_model_name=args.model,
            server_concurrency=args.server_concurrency,
            html_field=args.html_field,
            url_field=args.url_field or None,
            model_identifier=args.model,
            max_model_len=args.max_model_len,
            structured_outputs=args.structured_outputs,
            simplify_workers=args.simplify_workers,
            inference_workers=args.inference_workers,
            extract_workers=args.extract_workers,
        )
    )
    pipeline.add_stage(ParquetWriter(path=str(output_dir)))
    return pipeline


def summarize_output(output_dir: Path, text_field: str) -> dict:
    written = with_text = substantive = total_chars = 0
    status_counts: dict[str, int] = {}
    for shard in sorted(output_dir.rglob("*.parquet")):
        pf = pq.ParquetFile(shard)
        names = pf.schema_arrow.names
        if text_field not in names:
            written += pf.metadata.num_rows
            continue
        cols = [text_field] + ([STATUS_FIELD] if STATUS_FIELD in names else [])
        for batch in pf.iter_batches(batch_size=1000, columns=cols):
            d = batch.to_pydict()
            statuses = d.get(STATUS_FIELD)
            for i, value in enumerate(d[text_field]):
                written += 1
                if value:
                    with_text += 1
                    total_chars += len(value)
                    if len(value) >= MIN_SUBSTANTIVE_CHARS:
                        substantive += 1
                if statuses is not None:
                    st = statuses[i] or "unknown"
                    status_counts[st] = status_counts.get(st, 0) + 1
    return {
        "num_documents_written": written,
        "num_documents_with_text": with_text,
        "num_documents_substantive": substantive,
        "num_status_ok": status_counts.get("ok", 0),
        "total_text_chars": total_chars,
        "status_counts": status_counts,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    server = build_server(args)
    logger.info(f"Starting Dynamo server: {args.num_replicas} replicas of {args.model}")
    t_server = time.perf_counter()
    server.start()
    server_startup_s = time.perf_counter() - t_server
    logger.info(f"Server healthy after {server_startup_s:.1f}s at {server.endpoint}")

    # The stage appends /v1 itself.
    base_url = server.endpoint.removesuffix("/v1")
    in_flight = args.inference_workers * args.server_concurrency
    logger.info(
        f"M={args.inference_workers} workers x B={args.server_concurrency} in flight "
        f"= {in_flight} total, {in_flight / args.num_replicas:.0f} per replica"
    )

    try:
        pipeline = build_pipeline(args, base_url, output_dir)
        executor = setup_executor(args.executor)
        start = time.perf_counter()
        try:
            results = pipeline.run(executor, initial_tasks=None)
            success = True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results, success = [], False
        elapsed = time.perf_counter() - start
    finally:
        server.stop()

    out_stats = summarize_output(output_dir, args.text_field)
    written = out_stats["num_documents_written"]
    metrics = {
        "is_success": success,
        "time_taken_s": elapsed,
        "server_startup_s": server_startup_s,
        "num_output_tasks": len(results) if results else 0,
        "num_documents_processed": written,
        "throughput_docs_per_sec": (written / elapsed) if elapsed > 0 else 0.0,
        "requests_in_flight": in_flight,
        "requests_per_replica": in_flight / args.num_replicas,
        **out_stats,
    }
    metrics["extraction_rate"] = (out_stats["num_documents_substantive"] / written) if written else 0.0
    metrics["nonempty_rate"] = (out_stats["num_documents_with_text"] / written) if written else 0.0
    metrics["status_ok_rate"] = (out_stats["num_status_ok"] / written) if written else 0.0
    metrics["mean_text_chars"] = (
        out_stats["total_text_chars"] / out_stats["num_documents_with_text"]
        if out_stats["num_documents_with_text"]
        else 0.0
    )
    return {"params": vars(args) | {"output_path": str(output_dir)}, "metrics": metrics, "tasks": results or []}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-results-path", required=True)
    p.add_argument("--input-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--html-field", default="content")
    p.add_argument("--url-field", default="url")
    p.add_argument("--text-field", default="text")
    p.add_argument("--files-per-partition", type=int, default=1)
    # Server.
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-replicas", type=int, default=8, help="Engines, one GPU each (N)")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    # fp8 KV is the largest single engine-side win measured on this workload (1.57x).
    p.add_argument("--kv-cache-dtype", default="fp8", choices=["auto", "fp8"])
    p.add_argument("--speculative-tokens", type=int, default=16, help="0 disables speculative decoding")
    p.add_argument(
        "--arctic-inference-spec",
        default="arctic-inference",
        help="requirement string for the suffix proposer; unpinned by default (see build_server)",
    )
    p.add_argument("--server-timeout-s", type=int, default=2400)
    # The two knobs under study. B=256 (1024 requests/replica at M=32) measured best
    # of 32/64/128/256; the curve is flat within ~5% above 512/replica and loses a
    # few percent below 256. Dynamo wants a far deeper queue than a standalone
    # `vllm serve`, whose optimum was 192/replica -- the frontend adds per-request
    # latency that only more in-flight work covers.
    p.add_argument("--inference-workers", type=int, default=32, help="M: CPU actors calling the server")
    p.add_argument("--server-concurrency", type=int, default=256, help="B: in-flight requests per worker")
    p.add_argument("--simplify-workers", type=int, default=8)
    p.add_argument("--extract-workers", type=int, default=24)
    # Must be "none" on Dynamo: its frontend validates request bodies strictly and
    # rejects the `structured_outputs` extra body that `vllm serve` accepts, with
    #   400 Validation: Unsupported parameter(s): `structured_outputs`
    # so every request fails. Leaving the model unconstrained costs nothing measurable
    # at scale -- extraction over 40k documents is 0.8043-0.8047 unconstrained against
    # 0.8026-0.8032 for a grammar-constrained standalone server, i.e. marginally
    # better. (A 628-document smoke test suggested a large loss; that was small-sample
    # noise and did not survive.) Malformed answers carry element ids that match no
    # element, and the extract stage already ignores those.
    p.add_argument("--structured-outputs", default="none", choices=["none", "per_request"])
    p.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data", "ray_actors"])
    args = p.parse_args()

    results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
