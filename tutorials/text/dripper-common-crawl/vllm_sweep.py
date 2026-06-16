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

"""Run a vLLM serving sweep for Dripper prompts through Curator InferenceServer.

This is deliberately separate from ``main.py``:

* ``main.py`` measures end-to-end Dripper extraction quality and cost.
* this script measures server-level throughput across vLLM scheduling knobs.

The benchmark dataset is still realistic: it streams Common Crawl pages, applies
MinerU-HTML simplification and prompt construction, and gives those exact prompts
to ``vllm bench serve --dataset-name custom``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

from loguru import logger

from nemo_curator.stages.text.experimental.dripper.stages._bindings import _load_mineru_html_bindings
from nemo_curator.stages.text.experimental.dripper.stages._utils import _coerce_html, _count_item_ids

if TYPE_CHECKING:
    from types import ModuleType

    from nemo_curator.core.serve import InferenceServer


@dataclass(frozen=True)
class EngineSweepCase:
    """One vLLM engine configuration to test."""

    label: str
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enable_chunked_prefill: bool | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None


def parse_args() -> argparse.Namespace:  # noqa: PLR0915
    common = load_common_crawl_module()
    parser = argparse.ArgumentParser(description="Sweep vLLM serving knobs for Dripper prompts")

    parser.add_argument("--warc-paths-uri", default=common.DEFAULT_WARC_PATHS)
    parser.add_argument("--output-dir", default="outputs/dripper_cc_main_2025_26_vllm_sweep")
    parser.add_argument("--max-pages", type=int, default=320)
    parser.add_argument("--max-warcs", type=int, default=4)
    parser.add_argument("--html-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-html-bytes", type=int, default=1)
    parser.add_argument(
        "--s3-endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    )
    parser.add_argument("--s3-region", default=os.environ.get("AWS_REGION", "us-east-1"))

    parser.add_argument("--model-identifier", default=common.DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="dripper")
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float", "float16", "float32", "half"], default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "bfloat16", "float16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"],
        default=None,
    )
    parser.add_argument("--calculate-kv-scales", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--generation-config", default=None)
    parser.add_argument("--load-format", default=None)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=["lazy", "eager", "prefetch", "torchao"],
        default=None,
    )
    parser.add_argument("--performance-mode", choices=["balanced", "interactivity", "throughput"], default=None)
    parser.add_argument(
        "--distributed-executor-backend", choices=["ray", "mp", "uni", "external_launcher"], default=None
    )
    parser.add_argument(
        "--attention-backend", choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "XFORMERS"], default=None
    )
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-dbo", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dbo-decode-token-threshold", type=int, default=None)
    parser.add_argument("--dbo-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--max-num-partial-prefills", type=int, default=None)
    parser.add_argument("--max-long-partial-prefills", type=int, default=None)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--prompt-version", default="short_compact")
    parser.add_argument("--dynamic-max-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-max-token-padding", type=int, default=16)
    parser.add_argument("--dynamic-max-tokens-per-item", type=int, default=6)
    parser.add_argument("--dynamic-min-max-tokens", type=int, default=32)
    parser.add_argument("--h100-count", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--health-check-timeout-s", type=int, default=1800)
    parser.add_argument("--client-ready-timeout-s", type=int, default=120)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-verbose", action="store_true")
    parser.add_argument("--inference-backend", choices=["ray_serve", "dynamo"], default="ray_serve")
    parser.add_argument("--dynamo-mode", choices=["aggregated", "disagg"], default="aggregated")
    parser.add_argument("--dynamo-prefill-replicas", type=int, default=1)
    parser.add_argument("--dynamo-decode-replicas", type=int, default=1)
    parser.add_argument(
        "--dynamo-router-mode",
        choices=[
            "auto",
            "round-robin",
            "round_robin",
            "random",
            "power-of-two",
            "kv",
            "direct",
            "least-loaded",
            "device-aware-weighted",
        ],
        default="auto",
    )
    parser.add_argument("--dynamo-router-kv-events", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamo-etcd-endpoint", default=None)
    parser.add_argument("--dynamo-nats-url", default=None)

    parser.add_argument("--concurrency-values", default="16,32,64,128")
    parser.add_argument("--gpu-memory-utilization-values", default="0.9")
    parser.add_argument("--prefix-caching-values", default="true")
    parser.add_argument("--chunked-prefill-values", default="true")
    parser.add_argument("--max-num-seqs-values", default="64,128")
    parser.add_argument("--max-num-batched-tokens-values", default="16384,32768")
    parser.add_argument("--max-sweep-cases", type=int, default=0)

    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument(
        "--num-warmups",
        default="concurrency",
        help="Integer warmup request count, or 'concurrency' to use the active max concurrency.",
    )
    parser.add_argument("--bench-timeout-s", type=int, default=1800)
    parser.add_argument("--sleep-after-server-stop-s", type=int, default=10)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-prompts-by-max-model-len", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ray-temp-dir", default=os.environ.get("RAY_TMPDIR", "/tmp/ray_dripper_sweep"))  # noqa: S108
    parser.add_argument("--ray-port", type=int, default=None)
    parser.add_argument("--ray-dashboard-port", type=int, default=None)
    parser.add_argument("--ray-client-server-port", type=int, default=None)
    parser.add_argument("--ray-metrics-port", type=int, default=None)
    parser.add_argument("--ray-min-worker-port", type=int, default=None)
    parser.add_argument("--ray-max-worker-port", type=int, default=None)
    parser.add_argument("--ray-dashboard-host", default=os.environ.get("RAY_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    parser.add_argument("--ray-num-gpus", type=int, default=None)
    parser.add_argument("--ray-object-store-memory-gb", type=float, default=None)
    parser.add_argument("--ray-worker-connect-timeout-s", type=int, default=600)
    parser.add_argument("--ray-cleanup-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ray-include-dashboard-metrics", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:  # noqa: PLR0915
    started = time.perf_counter()
    args = parse_args()
    common = load_common_crawl_module()
    validate_args(args)

    output_dir = Path(args.output_dir).resolve()
    bench_result_dir = output_dir / "bench_results"
    bench_log_dir = output_dir / "bench_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_result_dir.mkdir(parents=True, exist_ok=True)
    bench_log_dir.mkdir(parents=True, exist_ok=True)

    log_environment(args)
    page_load_started = time.perf_counter()
    pages, warc_paths, load_stats = common.load_common_crawl_pages(args)
    page_load_s = time.perf_counter() - page_load_started
    dataset_path, dataset_stats = write_custom_prompt_dataset(args, pages, output_dir)
    if dataset_stats["prompt_rows"] <= 0:
        msg = "No Dripper prompts were generated for the vLLM sweep"
        raise RuntimeError(msg)
    bench_output_len = choose_bench_output_len(args, dataset_stats)

    sweep_cases = build_sweep_cases(args)
    concurrency_values = parse_int_csv(args.concurrency_values, "--concurrency-values")
    prompt_count = min(args.num_prompts, dataset_stats["prompt_rows"])
    if prompt_count <= 0:
        msg = "--num-prompts must be positive"
        raise ValueError(msg)

    ray_client = common.build_ray_client(args)
    ray_client.start()
    ray_start_s = time.perf_counter() - started
    summaries: list[dict[str, Any]] = []

    try:
        for sweep_case in sweep_cases:
            server = build_case_server(common, args, sweep_case)
            server_started = time.perf_counter()
            try:
                logger.info("Starting sweep case {}", sweep_case.label)
                server.start()
                server_start_s = time.perf_counter() - server_started
                client_endpoint = common.normalize_loopback_endpoint(server.endpoint)
                common.wait_for_openai_models(client_endpoint, args.client_ready_timeout_s)
                bench_base_url = endpoint_without_v1(client_endpoint)

                for concurrency in concurrency_values:
                    summary = run_vllm_bench(
                        args=args,
                        sweep_case=sweep_case,
                        base_url=bench_base_url,
                        dataset_path=dataset_path,
                        prompt_count=prompt_count,
                        concurrency=concurrency,
                        output_len=bench_output_len,
                        result_dir=bench_result_dir,
                        log_dir=bench_log_dir,
                    )
                    summary["server_start_s"] = server_start_s
                    summaries.append(summary)
                    write_summaries(output_dir, summaries)
            finally:
                try:
                    server.stop()
                finally:
                    if args.sleep_after_server_stop_s > 0:
                        time.sleep(args.sleep_after_server_stop_s)
    finally:
        ray_client.stop()

    metadata = {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "model_identifier": args.model_identifier,
        "served_model_name": args.served_model_name,
        "server_port": args.server_port,
        "inference_backend": args.inference_backend,
        "dynamo_mode": args.dynamo_mode,
        "dynamo_prefill_replicas": args.dynamo_prefill_replicas,
        "dynamo_decode_replicas": args.dynamo_decode_replicas,
        "dynamo_router_mode": args.dynamo_router_mode,
        "dynamo_router_kv_events": args.dynamo_router_kv_events,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "calculate_kv_scales": args.calculate_kv_scales,
        "generation_config": args.generation_config,
        "load_format": args.load_format,
        "safetensors_load_strategy": args.safetensors_load_strategy,
        "performance_mode": args.performance_mode,
        "distributed_executor_backend": args.distributed_executor_backend,
        "attention_backend": args.attention_backend,
        "async_scheduling": args.async_scheduling,
        "enable_dbo": args.enable_dbo,
        "dbo_decode_token_threshold": args.dbo_decode_token_threshold,
        "dbo_prefill_token_threshold": args.dbo_prefill_token_threshold,
        "max_num_partial_prefills": args.max_num_partial_prefills,
        "max_long_partial_prefills": args.max_long_partial_prefills,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "server_verbose": args.server_verbose,
        "dataset_path": str(dataset_path),
        "dataset_stats": dataset_stats,
        "bench_output_len": bench_output_len,
        "warc_paths_uri": args.warc_paths_uri,
        "warc_paths_sampled": warc_paths,
        "input_load_stats": load_stats,
        "timings_s": {
            "page_load_s": page_load_s,
            "ray_start_s": ray_start_s,
            "python_end_to_end_s": time.perf_counter() - started,
        },
        "h100_count": args.h100_count,
        "sweep_cases": [case.__dict__ for case in sweep_cases],
        "concurrency_values": concurrency_values,
        "num_prompts": prompt_count,
    }
    (output_dir / "sweep_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if args.plot:
        write_plot(output_dir, summaries)

    logger.info("Wrote sweep outputs under {}", output_dir)
    return 0


def load_common_crawl_module() -> ModuleType:
    module_name = "_dripper_common_crawl_main"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load Common Crawl helpers from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_args(args: argparse.Namespace) -> None:  # noqa: C901
    if args.max_pages <= 0:
        msg = "--max-pages must be positive"
        raise ValueError(msg)
    if args.max_warcs <= 0:
        msg = "--max-warcs must be positive"
        raise ValueError(msg)
    if args.replicas <= 0:
        msg = "--replicas must be positive"
        raise ValueError(msg)
    if args.num_prompts <= 0:
        msg = "--num-prompts must be positive"
        raise ValueError(msg)
    if args.max_tokens <= 0:
        msg = "--max-tokens must be positive"
        raise ValueError(msg)
    if args.max_model_len <= 0:
        msg = "--max-model-len must be positive"
        raise ValueError(msg)
    if args.dynamic_max_token_padding < 0:
        msg = "--dynamic-max-token-padding must be non-negative"
        raise ValueError(msg)
    if args.dynamic_max_tokens_per_item <= 0:
        msg = "--dynamic-max-tokens-per-item must be positive"
        raise ValueError(msg)
    if args.dynamic_min_max_tokens <= 0:
        msg = "--dynamic-min-max-tokens must be positive"
        raise ValueError(msg)
    if args.dynamo_prefill_replicas <= 0:
        msg = "--dynamo-prefill-replicas must be positive"
        raise ValueError(msg)
    if args.dynamo_decode_replicas <= 0:
        msg = "--dynamo-decode-replicas must be positive"
        raise ValueError(msg)
    parse_int_csv(args.concurrency_values, "--concurrency-values")
    parse_float_csv(args.gpu_memory_utilization_values, "--gpu-memory-utilization-values")
    parse_bool_csv(args.prefix_caching_values, "--prefix-caching-values", allow_auto=False)
    parse_bool_csv(args.chunked_prefill_values, "--chunked-prefill-values", allow_auto=True)
    parse_optional_int_csv(args.max_num_seqs_values, "--max-num-seqs-values")
    parse_optional_int_csv(args.max_num_batched_tokens_values, "--max-num-batched-tokens-values")
    parse_warmups(args.num_warmups, 1)


def log_environment(args: argparse.Namespace) -> None:
    logger.info("HOST={}", socket.gethostname())
    logger.info("SLURM_JOB_ID={}", os.environ.get("SLURM_JOB_ID", ""))
    logger.info("SLURM_JOB_NODELIST={}", os.environ.get("SLURM_JOB_NODELIST", ""))
    logger.info("COMMAND={}", " ".join(sys.argv))
    logger.info("PYTHON={}", sys.version.replace("\n", " "))
    logger.info("CUDA_VISIBLE_DEVICES={}", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logger.info("RAY_TMPDIR={}", args.ray_temp_dir)
    logger.info("MODEL={}", args.model_identifier)


def write_custom_prompt_dataset(
    args: argparse.Namespace,
    pages: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    bindings = _load_mineru_html_bindings()
    tokenizer = load_tokenizer(args) if args.filter_prompts_by_max_model_len else None
    dataset_path = output_dir / "dripper_vllm_custom_prompts.jsonl"
    stats = {
        "pages_seen": len(pages),
        "prompt_rows": 0,
        "empty_html_skipped": 0,
        "prompt_build_errors": 0,
        "prompt_len_skipped": 0,
        "no_item_ids_skipped": 0,
        "min_prompt_tokens": None,
        "max_prompt_tokens": None,
        "dynamic_max_tokens": args.dynamic_max_tokens,
        "dynamic_max_token_padding": args.dynamic_max_token_padding,
        "dynamic_max_tokens_per_item": args.dynamic_max_tokens_per_item,
        "dynamic_min_max_tokens": args.dynamic_min_max_tokens,
    }
    item_counts: list[int] = []
    prompt_token_counts: list[int] = []
    expected_output_tokens_values: list[int] = []

    with dataset_path.open("w", encoding="utf-8") as output:
        for page in pages:
            html = _coerce_html(page.get("html", ""))
            if not html.strip():
                stats["empty_html_skipped"] += 1
                continue
            try:
                case = bindings.case_cls(bindings.input_cls(raw_html=html, url=page.get("url")))
                case = bindings.simplify_single_input(case)
                item_count = _count_item_ids(case)
                if item_count <= 0:
                    stats["no_item_ids_skipped"] += 1
                    continue
                case = bindings.build_prompt(case, prompt_version=args.prompt_version)
                prompt = case.generate_input.full_prompt
            except Exception as exc:  # noqa: BLE001
                stats["prompt_build_errors"] += 1
                logger.debug("Failed to build Dripper prompt for {}: {}", page.get("url", ""), exc)
                continue

            expected_output_tokens = expected_output_tokens_for_item_count(args, item_count)
            prompt_tokens = count_prompt_tokens(tokenizer, prompt)
            if (
                args.filter_prompts_by_max_model_len
                and prompt_tokens is not None
                and prompt_tokens + expected_output_tokens > args.max_model_len
            ):
                stats["prompt_len_skipped"] += 1
                continue

            row = {
                "prompt": prompt,
                "output_tokens": expected_output_tokens,
                "item_count": item_count,
                "url": page.get("url") or "",
                "warc_id": page.get("warc_id") or "",
                "prompt_tokens": prompt_tokens,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["prompt_rows"] += 1
            item_counts.append(item_count)
            expected_output_tokens_values.append(expected_output_tokens)
            if prompt_tokens is not None:
                prompt_token_counts.append(prompt_tokens)
                min_tokens = stats["min_prompt_tokens"]
                max_tokens = stats["max_prompt_tokens"]
                stats["min_prompt_tokens"] = prompt_tokens if min_tokens is None else min(min_tokens, prompt_tokens)
                stats["max_prompt_tokens"] = prompt_tokens if max_tokens is None else max(max_tokens, prompt_tokens)

    stats.update(describe_values("item_count", item_counts))
    stats.update(describe_values("prompt_tokens", prompt_token_counts))
    stats.update(describe_values("expected_output_tokens", expected_output_tokens_values))
    logger.info("Wrote {} Dripper prompts to {}", stats["prompt_rows"], dataset_path)
    return dataset_path, stats


def expected_output_tokens_for_item_count(args: argparse.Namespace, item_count: int) -> int:
    if not args.dynamic_max_tokens:
        return args.max_tokens
    dynamic_max_tokens = max(
        args.dynamic_min_max_tokens,
        item_count * args.dynamic_max_tokens_per_item + args.dynamic_max_token_padding,
    )
    return min(args.max_tokens, dynamic_max_tokens)


def choose_bench_output_len(args: argparse.Namespace, dataset_stats: dict[str, Any]) -> int:
    if not args.dynamic_max_tokens:
        return args.max_tokens
    # vLLM bench serve's custom dataset path is version-sensitive; using a
    # single p95 output length keeps the benchmark conservative while matching
    # compact Dripper far better than a 2048-token synthetic decode.
    value = dataset_stats.get("p95_expected_output_tokens")
    if isinstance(value, int | float) and value > 0:
        return min(args.max_tokens, max(1, int(value)))
    return args.max_tokens


def describe_values(prefix: str, values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            f"min_{prefix}": None,
            f"mean_{prefix}": 0.0,
            f"p50_{prefix}": 0.0,
            f"p95_{prefix}": 0.0,
            f"max_{prefix}": None,
        }
    sorted_values = sorted(values)
    return {
        f"min_{prefix}": sorted_values[0],
        f"mean_{prefix}": sum(sorted_values) / len(sorted_values),
        f"p50_{prefix}": percentile(sorted_values, 0.50),
        f"p95_{prefix}": percentile(sorted_values, 0.95),
        f"max_{prefix}": sorted_values[-1],
    }


def percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def load_tokenizer(args: argparse.Namespace) -> Any | None:  # noqa: ANN401
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(args.model_identifier, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to load tokenizer for prompt length filtering: {}", exc)
        return None


def count_prompt_tokens(tokenizer: Any | None, prompt: str) -> int | None:  # noqa: ANN401
    if tokenizer is None:
        return None
    try:
        return len(tokenizer(prompt).input_ids)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to count prompt tokens: {}", exc)
        return None


def build_sweep_cases(args: argparse.Namespace) -> list[EngineSweepCase]:
    gpu_values = parse_float_csv(args.gpu_memory_utilization_values, "--gpu-memory-utilization-values")
    prefix_values = parse_bool_csv(args.prefix_caching_values, "--prefix-caching-values", allow_auto=False)
    chunked_values = parse_bool_csv(args.chunked_prefill_values, "--chunked-prefill-values", allow_auto=True)
    max_seq_values = parse_optional_int_csv(args.max_num_seqs_values, "--max-num-seqs-values")
    batched_token_values = parse_optional_int_csv(
        args.max_num_batched_tokens_values,
        "--max-num-batched-tokens-values",
    )

    cases: list[EngineSweepCase] = []
    for gpu, prefix, chunked, max_seqs, batched_tokens in itertools.product(
        gpu_values,
        prefix_values,
        chunked_values,
        max_seq_values,
        batched_token_values,
    ):
        if chunked is not True and batched_tokens is not None and batched_tokens <= args.max_model_len:
            logger.warning(
                "Skipping risky vLLM case: chunked prefill is not explicitly enabled and max_num_batched_tokens={} <= max_model_len={}",
                batched_tokens,
                args.max_model_len,
            )
            continue
        label = "_".join(
            [
                f"gpu{format_value(gpu)}",
                f"prefix{format_value(prefix)}",
                f"chunk{format_value(chunked)}",
                f"seqs{format_value(max_seqs)}",
                f"btok{format_value(batched_tokens)}",
            ]
        )
        cases.append(
            EngineSweepCase(
                label=label,
                gpu_memory_utilization=gpu,
                enable_prefix_caching=bool(prefix),
                enable_chunked_prefill=chunked,
                max_num_seqs=max_seqs,
                max_num_batched_tokens=batched_tokens,
            )
        )
    if args.max_sweep_cases > 0:
        cases = cases[: args.max_sweep_cases]
    if not cases:
        msg = "Sweep grid produced no valid vLLM engine cases"
        raise ValueError(msg)
    return cases


def build_case_server(common: ModuleType, args: argparse.Namespace, sweep_case: EngineSweepCase) -> InferenceServer:
    case_args = argparse.Namespace(**vars(args))
    case_args.gpu_memory_utilization = sweep_case.gpu_memory_utilization
    case_args.enable_prefix_caching = sweep_case.enable_prefix_caching
    case_args.enable_chunked_prefill = sweep_case.enable_chunked_prefill
    case_args.max_num_seqs = sweep_case.max_num_seqs
    case_args.max_num_batched_tokens = sweep_case.max_num_batched_tokens
    return common.build_inference_server(case_args)


def run_vllm_bench(  # noqa: PLR0913
    *,
    args: argparse.Namespace,
    sweep_case: EngineSweepCase,
    base_url: str,
    dataset_path: Path,
    prompt_count: int,
    concurrency: int,
    output_len: int,
    result_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    result_filename = f"{sweep_case.label}_conc{concurrency}.json"
    result_path = result_dir / result_filename
    log_path = log_dir / f"{sweep_case.label}_conc{concurrency}.log"
    warmups = parse_warmups(args.num_warmups, concurrency)

    cmd = [
        require_vllm_cli(),
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.model_identifier,
        "--trust-remote-code",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--custom-output-len",
        str(output_len),
        "--num-prompts",
        str(prompt_count),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--num-warmups",
        str(warmups),
        "--temperature",
        "0.0",
        "--top-p",
        str(args.top_p),
        "--extra-body",
        json.dumps({"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}),
        "--skip-chat-template",
        "--no-oversample",
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--metadata",
        f"sweep_case={sweep_case.label}",
        f"gpu_memory_utilization={sweep_case.gpu_memory_utilization}",
        f"enable_prefix_caching={sweep_case.enable_prefix_caching}",
        f"enable_chunked_prefill={sweep_case.enable_chunked_prefill}",
        f"max_num_seqs={sweep_case.max_num_seqs}",
        f"max_num_batched_tokens={sweep_case.max_num_batched_tokens}",
        f"bench_output_len={output_len}",
        f"dynamic_max_tokens={args.dynamic_max_tokens}",
        f"inference_backend={args.inference_backend}",
        f"dynamo_mode={args.dynamo_mode}",
        f"dtype={args.dtype}",
        f"quantization={args.quantization}",
        f"kv_cache_dtype={args.kv_cache_dtype}",
        f"calculate_kv_scales={args.calculate_kv_scales}",
        f"generation_config={args.generation_config}",
        f"load_format={args.load_format}",
        f"safetensors_load_strategy={args.safetensors_load_strategy}",
        f"performance_mode={args.performance_mode}",
        f"distributed_executor_backend={args.distributed_executor_backend}",
        f"attention_backend={args.attention_backend}",
        f"async_scheduling={args.async_scheduling}",
        f"enable_dbo={args.enable_dbo}",
    ]
    logger.info("Running vLLM bench case={} concurrency={}", sweep_case.label, concurrency)

    env = os.environ.copy()
    env["NO_PROXY"] = append_no_proxy(env.get("NO_PROXY", ""))
    env["no_proxy"] = append_no_proxy(env.get("no_proxy", ""))
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(  # noqa: S603
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.bench_timeout_s,
            check=False,
            env=env,
        )
    elapsed_s = time.perf_counter() - start

    summary: dict[str, Any] = {
        "sweep_case": sweep_case.label,
        "concurrency": concurrency,
        "num_warmups": warmups,
        "num_prompts": prompt_count,
        "bench_output_len": output_len,
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
        "elapsed_s": elapsed_s,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "gpu_memory_utilization": sweep_case.gpu_memory_utilization,
        "enable_prefix_caching": sweep_case.enable_prefix_caching,
        "enable_chunked_prefill": sweep_case.enable_chunked_prefill,
        "max_num_seqs": sweep_case.max_num_seqs,
        "max_num_batched_tokens": sweep_case.max_num_batched_tokens,
        "dynamic_max_tokens": args.dynamic_max_tokens,
        "inference_backend": args.inference_backend,
        "dynamo_mode": args.dynamo_mode,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "calculate_kv_scales": args.calculate_kv_scales,
        "generation_config": args.generation_config,
        "load_format": args.load_format,
        "safetensors_load_strategy": args.safetensors_load_strategy,
        "performance_mode": args.performance_mode,
        "distributed_executor_backend": args.distributed_executor_backend,
        "attention_backend": args.attention_backend,
        "async_scheduling": args.async_scheduling,
        "enable_dbo": args.enable_dbo,
        "dbo_decode_token_threshold": args.dbo_decode_token_threshold,
        "dbo_prefill_token_threshold": args.dbo_prefill_token_threshold,
        "max_num_partial_prefills": args.max_num_partial_prefills,
        "max_long_partial_prefills": args.max_long_partial_prefills,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "server_verbose": args.server_verbose,
    }
    if result_path.exists():
        try:
            result_json = json.loads(result_path.read_text(encoding="utf-8"))
            flatten_bench_result(summary, result_json)
            add_cost_metrics(args, summary)
        except Exception as exc:  # noqa: BLE001
            summary["result_parse_error"] = str(exc)
    return summary


def add_cost_metrics(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    request_throughput = summary.get("bench_request_throughput")
    if isinstance(request_throughput, int | float) and request_throughput > 0:
        h100_hours_per_page = args.h100_count / (3600 * request_throughput)
        summary["model_only_h100_hours_per_page"] = h100_hours_per_page
        summary["model_only_pages_per_h100_hour"] = 1 / h100_hours_per_page


def flatten_bench_result(summary: dict[str, Any], result_json: dict[str, Any]) -> None:
    for key, value in result_json.items():
        if isinstance(value, int | float | str | bool) or value is None:
            summary[f"bench_{key}"] = value


def require_vllm_cli() -> str:
    cli = shutil.which("vllm")
    if cli is None:
        msg = "Unable to find the 'vllm' CLI in PATH"
        raise RuntimeError(msg)
    return cli


def endpoint_without_v1(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def append_no_proxy(value: str) -> str:
    items = [item for item in value.split(",") if item]
    for required in ("localhost", "127.0.0.1", "::1"):
        if required not in items:
            items.append(required)
    return ",".join(items)


def write_summaries(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    (output_dir / "sweep_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "sweep_summary.csv"
    if not summaries:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in summaries for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_plot(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to SVG plot because matplotlib is unavailable: {}", exc)
        write_svg_plot(output_dir, summaries)
        return

    rows = [
        row
        for row in summaries
        if row.get("status") == "completed" and isinstance(row.get("bench_request_throughput"), int | float)
    ]
    if not rows:
        logger.warning("Skipping plot because no completed request throughput rows are available")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sweep_case"]), []).append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group_rows in sorted(grouped.items()):
        group_rows = sorted(group_rows, key=lambda row: int(row["concurrency"]))  # noqa: PLW2901
        ax.plot(
            [int(row["concurrency"]) for row in group_rows],
            [float(row["bench_request_throughput"]) for row in group_rows],
            marker="o",
            label=label,
        )
    ax.set_xlabel("max concurrency")
    ax.set_ylabel("requests/s")
    ax.set_title("Dripper vLLM sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "concurrency_vs_req_s.png", dpi=160)
    plt.close(fig)


def write_svg_plot(output_dir: Path, summaries: list[dict[str, Any]]) -> None:  # noqa: C901
    rows = [
        row
        for row in summaries
        if row.get("status") == "completed" and isinstance(row.get("bench_request_throughput"), int | float)
    ]
    if not rows:
        logger.warning("Skipping SVG plot because no completed request throughput rows are available")
        return

    width = 900
    height = 560
    margin_left = 72
    margin_right = 24
    margin_top = 40
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    conc_values = [int(row["concurrency"]) for row in rows]
    throughput_values = [float(row["bench_request_throughput"]) for row in rows]
    min_x = min(conc_values)
    max_x = max(conc_values)
    max_y = max(throughput_values)
    if min_x == max_x:
        min_x = 0
    if max_y <= 0:
        max_y = 1.0

    def x_scale(value: int) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * plot_width if max_x != min_x else margin_left

    def y_scale(value: float) -> float:
        return margin_top + plot_height - (value / max_y) * plot_height

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sweep_case"]), []).append(row)
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4d7c0f"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Dripper vLLM sweep</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827"/>',
    ]
    for idx in range(6):
        y_value = max_y * idx / 5
        y = y_scale(y_value)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        svg.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{y_value:.1f}</text>'
        )
    for x_value in sorted(set(conc_values)):
        x = x_scale(x_value)
        svg.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 5}" stroke="#111827"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="12">{x_value}</text>'
        )
    svg.append(
        f'<text x="{margin_left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14">max concurrency</text>'
    )
    svg.append(
        f'<text x="18" y="{margin_top + plot_height / 2}" transform="rotate(-90 18 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="14">requests/s</text>'
    )

    for index, (label, group_rows) in enumerate(sorted(grouped.items())):
        color = colors[index % len(colors)]
        group_rows = sorted(group_rows, key=lambda row: int(row["concurrency"]))  # noqa: PLW2901
        points = " ".join(
            f"{x_scale(int(row['concurrency'])):.2f},{y_scale(float(row['bench_request_throughput'])):.2f}"
            for row in group_rows
        )
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for row in group_rows:
            x = x_scale(int(row["concurrency"]))
            y = y_scale(float(row["bench_request_throughput"]))
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        legend_y = margin_top + 18 + index * 18
        svg.append(
            f'<line x1="{margin_left + plot_width - 210}" y1="{legend_y}" x2="{margin_left + plot_width - 190}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{margin_left + plot_width - 184}" y="{legend_y + 4}" font-family="Arial" font-size="11">{escape_svg(label[:46])}</text>'
        )
    svg.append("</svg>")
    (output_dir / "concurrency_vs_req_s.svg").write_text("\n".join(svg), encoding="utf-8")


def escape_svg(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_warmups(value: str, concurrency: int) -> int:
    normalized = str(value).strip().lower()
    if normalized == "concurrency":
        return concurrency
    try:
        warmups = int(normalized)
    except ValueError as exc:
        msg = "--num-warmups must be an integer or 'concurrency'"
        raise ValueError(msg) from exc
    if warmups < 0:
        msg = "--num-warmups must be non-negative"
        raise ValueError(msg)
    return warmups


def parse_int_csv(value: str, flag_name: str) -> list[int]:
    values = []
    for raw in split_csv(value):
        try:
            parsed = int(raw)
        except ValueError as exc:
            msg = f"{flag_name} contains a non-integer value: {raw!r}"
            raise ValueError(msg) from exc
        if parsed <= 0:
            msg = f"{flag_name} values must be positive"
            raise ValueError(msg)
        values.append(parsed)
    if not values:
        msg = f"{flag_name} must contain at least one value"
        raise ValueError(msg)
    return values


def parse_optional_int_csv(value: str, flag_name: str) -> list[int | None]:
    values: list[int | None] = []
    for raw in split_csv(value):
        normalized = raw.lower()
        if normalized in {"", "auto", "none", "null"}:
            values.append(None)
            continue
        try:
            parsed = int(raw)
        except ValueError as exc:
            msg = f"{flag_name} contains a non-integer value: {raw!r}"
            raise ValueError(msg) from exc
        if parsed <= 0:
            msg = f"{flag_name} values must be positive"
            raise ValueError(msg)
        values.append(parsed)
    return values or [None]


def parse_float_csv(value: str, flag_name: str) -> list[float]:
    values = []
    for raw in split_csv(value):
        try:
            parsed = float(raw)
        except ValueError as exc:
            msg = f"{flag_name} contains a non-float value: {raw!r}"
            raise ValueError(msg) from exc
        if parsed <= 0 or parsed >= 1:
            msg = f"{flag_name} values must be in the open interval (0, 1)"
            raise ValueError(msg)
        values.append(parsed)
    if not values:
        msg = f"{flag_name} must contain at least one value"
        raise ValueError(msg)
    return values


def parse_bool_csv(value: str, flag_name: str, *, allow_auto: bool) -> list[bool | None]:
    values: list[bool | None] = []
    for raw in split_csv(value):
        normalized = raw.lower()
        if normalized in {"true", "1", "yes", "on"}:
            values.append(True)
        elif normalized in {"false", "0", "no", "off"}:
            values.append(False)
        elif allow_auto and normalized in {"auto", "none", "null"}:
            values.append(None)
        else:
            msg = f"{flag_name} contains an invalid boolean value: {raw!r}"
            raise ValueError(msg)
    if not values:
        msg = f"{flag_name} must contain at least one value"
        raise ValueError(msg)
    return values


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def format_value(value: object) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
