#!/usr/bin/env python3
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

"""Dripper HTML content extraction — quickstart.

Demonstrates the full Dripper pipeline on a small synthetic dataset
without requiring a GPU cluster.

The script is self-contained: it writes a small parquet manifest, builds a
``DripperHTMLWorkflow``, and runs it with ``XennaExecutor`` (CPU-only,
no Ray cluster required for small data).

A real LLM inference server (OpenAI-compatible) is expected on
``--server-url`` (default ``http://localhost:8000/v1``).  If no server is
running, pass ``--dry-run`` to skip actual inference and only exercise the
preprocessing / postprocessing stages.

Usage
-----
Dry-run (no LLM server needed, exercises pre/post stages only)::

    python quickstart.py --dry-run

Full run against a local vLLM server::

    python quickstart.py \\
        --server-url http://localhost:8000/v1 \\
        --model-name opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact

Requirements
------------
::

    pip install "nemo-curator[dripper]"
    # Also installs: mineru-html>=1.1, llm-web-kit>=4.1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred so the script still imports cleanly when
# dependencies are not installed.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dripper quickstart — exercises DripperHTMLWorkflow on synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs.  Defaults to a temporary directory.",
    )
    p.add_argument(
        "--n-pages",
        type=int,
        default=20,
        help="Number of synthetic HTML pages to generate.",
    )
    p.add_argument(
        "--server-url",
        default="http://localhost:8000/v1",
        help="Base URL of an OpenAI-compatible inference server.",
    )
    p.add_argument(
        "--model-name",
        default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact",
        help="Model ID served at --server-url.",
    )
    p.add_argument(
        "--layout-cluster-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for layout-template clustering.",
    )
    p.add_argument(
        "--no-layout-clustering",
        action="store_true",
        help="Skip the layout clustering stage (faster, fewer LLM savings).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip LLM inference entirely — only the preprocess and postprocess stages run. "
            "Useful to verify the pipeline wiring without a server."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Log per-stage progress and timing.",
    )
    return p


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

_HTML_TEMPLATES = [
    # News article
    "<html><head><title>{title}</title></head><body>"
    "<nav><a href='/'>Home</a><a href='/news'>News</a></nav>"
    "<article><h1>{title}</h1><p>Published by staff writer.</p>"
    "<p>{body}</p></article>"
    "<footer>Copyright 2026 Example Media.</footer></body></html>",
    # Product page
    "<html><head><title>{title} — Shop</title></head><body>"
    "<header><h1>ExampleShop</h1></header>"
    "<main><h2>{title}</h2><p class='desc'>{body}</p>"
    "<button>Add to cart</button></main></body></html>",
    # Blog post
    "<html><body><header class='site-header'><a href='/'>Blog</a></header>"
    "<div class='post'><h2>{title}</h2><div class='content'><p>{body}</p></div>"
    "<div class='comments'><p>No comments yet.</p></div></div></body></html>",
    # Wikipedia-style
    "<html><body><div id='mw-content-text'><h1>{title}</h1><p>{body}</p>"
    "<div class='reflist'><ol><li>Reference 1.</li></ol></div></div></body></html>",
    # Forum post
    "<html><body><div class='forum'><div class='post'>"
    "<span class='author'>user42</span><p>{body}</p></div></div></body></html>",
]

_BODIES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Scientists discovered a new method to improve efficiency by 30 percent.",
    "Local community gathers to celebrate the annual harvest festival.",
    "New research suggests that regular exercise improves cognitive function.",
    "The stock market closed higher on strong earnings reports this quarter.",
]


def _make_synthetic_dataset(output_dir: Path, n_pages: int) -> str:
    """Write a small synthetic HTML parquet manifest and return its path."""
    records = []
    for i in range(n_pages):
        template = _HTML_TEMPLATES[i % len(_HTML_TEMPLATES)]
        body = _BODIES[i % len(_BODIES)]
        title = f"Article {i}: {body[:30]}..."
        host = f"example{i % 5}.com"
        records.append(
            {
                "url": f"https://{host}/page-{i:04d}",
                "url_host_name": host,
                "html": template.format(title=title, body=body),
            }
        )
    df = pd.DataFrame(records)
    out_path = output_dir / "synthetic_pages.parquet"
    df.to_parquet(str(out_path), index=False)
    logger.info("Wrote {:,} synthetic pages → {}", n_pages, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Dry-run stub client (no LLM queries)
# ---------------------------------------------------------------------------


def _make_dry_run_client() -> object:
    """Return a minimal AsyncLLMClient that returns empty responses synchronously."""
    try:
        from collections.abc import Iterable

        from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig

        class _DryRunClient(AsyncLLMClient):
            """Stub client: returns an empty string for every inference call."""

            def __init__(self) -> None:
                super().__init__(max_concurrent_requests=1, max_retries=0, base_delay=0.0)

            def setup(self) -> None:
                pass

            async def _query_model_impl(
                self,
                *,
                messages: Iterable,
                model: str,
                conversation_formatter: object = None,
                generation_config: GenerationConfig | dict | None = None,
            ) -> list[str]:
                return [""]

        return _DryRunClient()
    except ImportError as exc:
        logger.error("Could not import AsyncLLMClient: {}", exc)
        raise


def _make_openai_client(server_url: str, model_name: str) -> object:
    """Return a configured OpenAI-compatible LLM client."""
    try:
        from nemo_curator.models.client.openai_client import OpenAIClient

        return OpenAIClient(
            model=model_name,
            base_url=server_url,
            api_key="EMPTY",
        )
    except ImportError as exc:
        logger.error(
            "Could not import OpenAIClient.  Install nemo-curator[dripper] and ensure "
            "the package is on PYTHONPATH: {}",
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_arg_parser().parse_args()

    try:
        from nemo_curator.backends.xenna import XennaExecutor
        from nemo_curator.stages.text.experimental.dripper import DripperHTMLWorkflow
    except ImportError as exc:
        logger.error("Required imports missing.  Run: pip install 'nemo-curator[dripper]'\n  {}", exc)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as _tmp:
        output_dir = Path(args.output_dir or _tmp)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # 1. Create synthetic dataset
        # ------------------------------------------------------------------ #
        manifest_path = _make_synthetic_dataset(output_dir, args.n_pages)

        # ------------------------------------------------------------------ #
        # 2. Build the client
        # ------------------------------------------------------------------ #
        if args.dry_run:
            logger.info("Dry-run mode: using stub LLM client (no inference server needed).")
            client = _make_dry_run_client()
        else:
            logger.info("Using OpenAI-compatible client at {}", args.server_url)
            client = _make_openai_client(args.server_url, args.model_name)

        # ------------------------------------------------------------------ #
        # 3. Construct the workflow — matches SemanticDedup usage pattern
        # ------------------------------------------------------------------ #
        workflow = DripperHTMLWorkflow(
            client=client,
            model_name=args.model_name,
            perform_layout_clustering=(not args.no_layout_clustering),
            layout_cluster_threshold=args.layout_cluster_threshold,
            fallback="trafilatura",
            output_format="mm_md",
            verbose=args.verbose,
        )

        logger.info(
            "DripperHTMLWorkflow configured: layout_clustering={}, threshold={:.2f}",
            not args.no_layout_clustering,
            args.layout_cluster_threshold,
        )

        # ------------------------------------------------------------------ #
        # 4. Load the synthetic dataset into DocumentBatch tasks
        # ------------------------------------------------------------------ #
        try:
            from nemo_curator.tasks import DocumentBatch

            df = pd.read_parquet(manifest_path)
            initial_tasks = [
                DocumentBatch(
                    task_id=f"quickstart-{i}",
                    dataset_name="quickstart_synthetic",
                    data=chunk,
                )
                for i, (_, chunk) in enumerate(df.groupby(df.index // max(1, len(df) // 4)))
            ]
            logger.info("Prepared {:,} DocumentBatch tasks from {:,} pages.", len(initial_tasks), len(df))
        except ImportError as exc:
            logger.error("Could not import DocumentBatch: {}", exc)
            sys.exit(1)

        # ------------------------------------------------------------------ #
        # 5. Run the pipeline
        # ------------------------------------------------------------------ #
        t0 = time.time()
        logger.info("Running DripperHTMLWorkflow on {:,} synthetic pages...", args.n_pages)

        result = workflow.run(executor=XennaExecutor(), initial_tasks=initial_tasks)

        elapsed = time.time() - t0
        output_tasks = result.get("output_tasks") or []
        total_pages = sum(len(t.to_pandas()) for t in output_tasks if hasattr(t, "to_pandas"))

        logger.info(
            "Done in {:.1f}s — {:,} pages processed ({:.1f} p/s).",
            elapsed,
            total_pages,
            total_pages / elapsed if elapsed > 0 else 0.0,
        )

        # ------------------------------------------------------------------ #
        # 6. Show a sample of results
        # ------------------------------------------------------------------ #
        if output_tasks:
            first_df = output_tasks[0].to_pandas()
            sample_cols = [
                c for c in ["url", "dripper_content", "dripper_error", "dripper_time_s"] if c in first_df.columns
            ]
            logger.info(
                "Sample output (first task, columns: {}):\n{}", sample_cols, first_df[sample_cols].head(3).to_string()
            )
        else:
            logger.warning("No output tasks returned — check the pipeline configuration.")


if __name__ == "__main__":
    main()
