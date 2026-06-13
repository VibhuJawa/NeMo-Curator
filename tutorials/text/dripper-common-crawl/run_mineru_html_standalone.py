#!/usr/bin/env python3
"""
run_mineru_html_standalone.py

Pure MinerU-HTML baseline — runs the upstream library directly on pages from
a manifest parquet, with no NeMo Curator infrastructure.

This is the true "Dripper standalone" baseline:
  - Reads pages from a manifest (url, html columns)
  - Optionally fetches HTML from WARCs if html column is missing
  - Batches pages and calls MinerUHTML.process() directly
  - Writes results to a parquet + metrics JSON

Usage (Slurm):
  python run_mineru_html_standalone.py \
    --input   /lustre/.../layout_precompute_manifest.parquet \
    --output  /lustre/.../mineru_standalone_output \
    --max-pages 2000 \
    --batch-size 64 \
    --model opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact

Stage 2 usage (representatives-only, GPU inference):
  python run_mineru_html_standalone.py \
    --input   /lustre/.../cluster_assignments/ \
    --output  /lustre/.../gpu_results \
    --representatives-only \
    --shard-index 3 \
    --num-shards  64 \
    --batch-size  64 \
    --model opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact

  The --representatives-only flag:
    - Reads clustered_manifest.parquet (or a directory of cluster_assignments/)
    - Filters to rows where is_representative=True OR is_noise=True
    - Skips HTML > 500 KB (logged as "too_long" in dripper_error)
    - Outputs inference_results/shard_NNNN_of_MMMM.parquet with columns:
        url, url_host_name, layout_cluster_id, cluster_role, host_bucket,
        dripper_content, dripper_html, dripper_error, dripper_time_s,
        xpath_rules, template_html, inference_time_s
    - Writes metrics_shard_NNNN.json alongside
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _detect_gpus() -> int:
    """Return number of GPUs visible to this process."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "NoDevFiles":
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        r = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=5)
        return max(1, len([l for l in r.stdout.strip().splitlines() if l.startswith("GPU")]))
    except Exception:
        return 1


def _run_dp_parallel(args) -> None:
    """DP=N: spawn one subprocess per GPU, each handling 1/N of the pages.

    Each child gets CUDA_VISIBLE_DEVICES=i, --dp-gpus 1 (to avoid recursion),
    and --shard-index / --num-shards scaled by N so outputs don't collide.
    """
    n = args.dp_gpus
    print(f"[mineru_stage2] DP={n}: launching {n} parallel workers across {n} GPUs", flush=True)
    procs = []
    for gpu_id in range(n):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        child_shard = args.shard_index * n + gpu_id
        child_nshards = args.num_shards * n
        cmd = [
            sys.executable,
            __file__,
            "--input",
            args.input,
            "--output",
            args.output,
            "--representatives-only",
            "--shard-index",
            str(child_shard),
            "--num-shards",
            str(child_nshards),
            "--batch-size",
            str(args.batch_size),
            "--model",
            args.model,
            "--hf-cache",
            args.hf_cache,
            "--dp-gpus",
            "1",  # prevent recursive fan-out
        ]
        if args.max_pages:
            cmd += ["--max-pages", str(args.max_pages)]
        log = Path(args.output) / f"dp_worker_{gpu_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as lf:
            procs.append((gpu_id, subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)))
        print(f"  GPU {gpu_id}: shard {child_shard}/{child_nshards}  log={log}", flush=True)

    failed = 0
    for gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failed += 1
            print(f"  GPU {gpu_id}: FAILED (rc={rc})", file=sys.stderr, flush=True)
        else:
            print(f"  GPU {gpu_id}: done", flush=True)

    if failed:
        sys.exit(f"[mineru_stage2] {failed}/{n} DP workers failed")


# ── HTML size guard ───────────────────────────────────────────────────────────
# Pages larger than this skip LLM inference to avoid 180-240s stall batches.
# The real max_context_window is 32768 tokens ≈ 100-150 KB of HTML in practice;
# 500 KB is a generous guard that still eliminates the worst offenders.
HTML_SIZE_LIMIT_BYTES = 500 * 1024  # 500 KB


def read_parquet(path):
    return pq.ParquetFile(str(path)).read().to_pandas()


def read_parquet_with_filter(path, filters=None):
    """Read parquet file or directory with optional PyArrow predicate filters."""
    p = Path(path)
    if p.is_dir():
        dataset = pq.ParquetDataset(str(p), filters=filters)
        return dataset.read().to_pandas()
    else:
        # Single file — apply filter after read (PyArrow filters work on datasets)
        dataset = pq.ParquetDataset(str(p), filters=filters)
        return dataset.read().to_pandas()


def coerce_html(raw):
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def html_byte_len(raw):
    """Return byte length of raw HTML (bytes or str)."""
    if isinstance(raw, bytes):
        return len(raw)
    return len((raw or "").encode("utf-8", errors="replace"))


def _extract_xpath_rules(result):
    """Extract pre-serialized xpath_rules JSON from a MinerUHTMLGeneric result.

    The rules are built from map_parser_cls() immediately after inference so
    Stage 3 can evaluate them with lxml directly without re-running the heavy
    _preprocess_template_data() call per sibling.

    Returns a JSON string, or an empty string if unavailable.
    """
    if result is None:
        return ""
    try:
        # Attempt to access the structured parser output which holds XPath rules.
        output_data = result.output_data
        # MinerUHTML stores CSS/XPath selectors in the parsed content map.
        # Try common attribute paths used by the library.
        for attr in ("xpath_rules", "css_rules", "content_map", "selectors"):
            val = getattr(output_data, attr, None)
            if val is not None:
                return json.dumps(val, ensure_ascii=False)
    except Exception:
        pass
    return ""


def _extract_template_html(result):
    """Extract simplified template HTML with _item_id labels if available."""
    if result is None:
        return ""
    try:
        output_data = result.output_data
        for attr in ("template_html", "labeled_html", "simplified_html"):
            val = getattr(output_data, attr, None)
            if val:
                return str(val)
    except Exception:
        pass
    return ""


# ── Representatives-only (Stage 2) logic ─────────────────────────────────────


def load_representatives(input_path, max_pages):
    """Load cluster_assignments and filter to representative + noise pages.

    Accepts either:
      - A single clustered_manifest.parquet with columns including
        is_representative (bool) and optionally is_noise (bool).
      - A directory of shard_NNNN.parquet files produced by Stage 1.
        Must contain cluster_role column with values:
        'representative' | 'sibling' | 'singleton'.

    Only rows with actual HTML content are kept (the html column must be
    non-null — Stage 1 writes html only for representative/noise pages).
    """
    p = Path(input_path)

    # Try predicate pushdown for directories (much faster for large datasets)
    try:
        if p.is_dir():
            # Stage 1 output: cluster_role column
            filters = [
                [("cluster_role", "in", ["representative", "singleton"])],
            ]
            df = read_parquet_with_filter(input_path, filters=filters)
        else:
            # Single parquet — read all, filter below
            df = read_parquet(input_path)
    except Exception as exc:
        print(f"[mineru_stage2] WARNING: predicate pushdown failed ({exc}), reading full dataset", file=sys.stderr)
        import glob as _glob

        import pyarrow as _pa

        if Path(input_path).is_dir():
            files = sorted(_glob.glob(str(Path(input_path) / "shard_*.parquet")))
            if not files:
                files = sorted(_glob.glob(str(Path(input_path) / "*.parquet")))
            tables = [pq.ParquetFile(f).read() for f in files]
            df = _pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame()
        else:
            df = pq.ParquetFile(str(input_path)).read().to_pandas()

    n_before = len(df)

    # Normalise to a consistent boolean mask regardless of schema variant
    if "cluster_role" in df.columns:
        # Stage 1 canonical schema
        mask = df["cluster_role"].isin(["representative", "singleton"])
        df = df[mask].copy()
        # Derive is_noise flag for singletons (treated as standalone LLM pages)
        df["is_representative"] = df["cluster_role"] == "representative"
        df["is_noise"] = df["cluster_role"] == "singleton"
    elif "is_representative" in df.columns:
        # Legacy schema
        rep_mask = df["is_representative"].astype(bool)
        noise_mask = df.get("is_noise", pd.Series(False, index=df.index)).astype(bool)
        df = df[rep_mask | noise_mask].copy()
    else:
        raise ValueError(
            "Input manifest has neither 'cluster_role' nor 'is_representative' column. "
            "Cannot determine which pages need GPU inference."
        )

    # Normalise cluster id column
    for cid_col in ("layout_cluster_id", "cluster_id", "dripper_layout_id"):
        if cid_col in df.columns:
            if cid_col != "layout_cluster_id":
                df = df.rename(columns={cid_col: "layout_cluster_id"})
            break
    if "layout_cluster_id" not in df.columns:
        df["layout_cluster_id"] = None

    # Only keep rows that actually have HTML (Stage 1 embeds html for reps only)
    if "html" in df.columns:
        has_html = df["html"].notna() & (df["html"] != b"") & (df["html"] != "")
        missing_html = (~has_html).sum()
        if missing_html:
            print(
                f"[mineru_stage2] WARNING: {missing_html:,} representative rows have no html — dropping",
                file=sys.stderr,
            )
        df = df[has_html].reset_index(drop=True)
    else:
        raise ValueError(
            "Input manifest is missing 'html' column. "
            "Stage 1 must embed html for representative pages before Stage 2 can run."
        )

    print(f"[mineru_stage2] filtered {n_before:,} → {len(df):,} representative/noise pages (have HTML)")
    if max_pages > 0:
        df = df.head(max_pages)
        print(f"[mineru_stage2] capped to {len(df):,} pages (--max-pages {max_pages})")
    return df


def run_representatives_only(args):
    """Stage 2 entry point: GPU inference on representatives only."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    print("[mineru_stage2] === Stage 2: GPU inference on representatives only ===")
    print(f"[mineru_stage2] input:        {args.input}")
    print(f"[mineru_stage2] output:       {args.output}")
    print(f"[mineru_stage2] max_pages:    {args.max_pages or 'all'}")
    print(f"[mineru_stage2] batch_size:   {args.batch_size}")
    print(f"[mineru_stage2] model:        {args.model}")
    print(f"[mineru_stage2] html_limit:   {HTML_SIZE_LIMIT_BYTES // 1024} KB")
    print(f"[mineru_stage2] shard:        {args.shard_index}/{args.num_shards}")
    print()

    # ── Load and filter ───────────────────────────────────────────────────────
    df = load_representatives(args.input, args.max_pages)

    # Shard: each GPU array task handles a slice
    if args.num_shards > 1:
        total = len(df)
        shard_start = total * args.shard_index // args.num_shards
        shard_end = total * (args.shard_index + 1) // args.num_shards
        df = df.iloc[shard_start:shard_end].reset_index(drop=True)
        print(
            f"[mineru_stage2] shard {args.shard_index}/{args.num_shards}: "
            f"rows {shard_start}–{shard_end - 1}  ({len(df):,} pages)"
        )

    # Checkpoint: skip if output shard already complete
    if args.num_shards > 1:
        out_parquet = output_dir / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.parquet"
    else:
        out_parquet = output_dir / "inference_results.parquet"

    if out_parquet.exists():
        try:
            existing = pq.ParquetFile(str(out_parquet)).metadata.num_rows
            if existing == len(df):
                print(f"[mineru_stage2] shard already complete ({existing:,} rows) — skipping")
                return
            else:
                print(f"[mineru_stage2] shard exists but row count mismatch ({existing} vs {len(df)}) — reprocessing")
        except Exception:
            pass

    if len(df) == 0:
        print("[mineru_stage2] no pages to process in this shard — writing empty output")
        _write_stage2_outputs(output_dir, out_parquet, pd.DataFrame(), args, t_start, t_start, 0)
        return

    # ── Load MinerU-HTML ──────────────────────────────────────────────────────
    print("[mineru_stage2] loading MinerUHTML extractor...", flush=True)
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    from mineru_html.api import MinerUHTMLConfig, MinerUHTMLGeneric
    from mineru_html.inference.factory import create_vllm_backend

    n_gpus = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    print(f"[mineru_stage2] tensor_parallel_size={n_gpus}", flush=True)

    config = MinerUHTMLConfig(prompt_version="short_compact", response_format="compact")
    llm = create_vllm_backend(
        model_path=args.model,
        response_format=config.response_format,
        # CRITICAL FIX: was 256*1024 — caused 180-240s stall batches on long HTML.
        # 32768 tokens is the actual model max and eliminates pathological batches.
        max_context_window=32768,
        model_init_kwargs={
            "tensor_parallel_size": n_gpus,
            "gpu_memory_utilization": 0.85,
            "enable_prefix_caching": True,
        },
    )
    extractor = MinerUHTMLGeneric(llm, config)

    t_load = time.perf_counter()
    print(f"[mineru_stage2] extractor ready in {t_load - t_start:.1f}s", flush=True)

    # ── Run inference in batches ──────────────────────────────────────────────
    rows = df.to_dict("records")
    results = []
    errors = 0
    too_long_count = 0

    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]

        # Pre-filter: skip pages exceeding the HTML size limit
        runnable = []
        skipped_too_long = []
        for r in batch:
            raw = r.get("html", "")
            if html_byte_len(raw) > HTML_SIZE_LIMIT_BYTES:
                skipped_too_long.append(r)
            else:
                runnable.append(r)

        too_long_count += len(skipped_too_long)
        for r in skipped_too_long:
            results.append(
                {
                    "url": r.get("url", ""),
                    "url_host_name": r.get("url_host_name", ""),
                    "layout_cluster_id": r.get("layout_cluster_id"),
                    "cluster_role": r.get("cluster_role", ""),
                    "host_bucket": r.get("host_bucket"),
                    "dripper_content": "",
                    "dripper_html": "",
                    "dripper_error": "too_long",
                    "dripper_time_s": 0.0,
                    "xpath_rules": "",
                    "template_html": "",
                    "inference_time_s": 0.0,
                }
            )

        if not runnable:
            done = min(batch_start + args.batch_size, len(rows))
            print(
                f"[mineru_stage2] {done:>6}/{len(rows)} pages  (batch all too_long, {len(skipped_too_long)} skipped)"
            )
            continue

        html_list = [coerce_html(r.get("html", "")) for r in runnable]

        t0 = time.perf_counter()
        try:
            batch_results = extractor.process(html_list)
        except Exception as e:
            print(
                f"[mineru_stage2] batch {batch_start // args.batch_size} ERROR: {e}",
                file=sys.stderr,
            )
            batch_results = [None] * len(runnable)
            errors += len(runnable)

        elapsed = time.perf_counter() - t0
        per_page_s = elapsed / len(runnable)

        for r, result in zip(runnable, batch_results):
            if result is not None:
                try:
                    main_content = str(result.output_data.main_content or "")
                    main_html = str(getattr(result.output_data, "main_html", "") or "")
                    error = ""
                except Exception as e:
                    main_content = ""
                    main_html = ""
                    error = str(e)[:200]
                    errors += 1
            else:
                main_content = ""
                main_html = ""
                error = "batch_failed"

            xpath_rules = _extract_xpath_rules(result)
            template_html = _extract_template_html(result)

            results.append(
                {
                    "url": r.get("url", ""),
                    "url_host_name": r.get("url_host_name", ""),
                    "layout_cluster_id": r.get("layout_cluster_id"),
                    "cluster_role": r.get("cluster_role", ""),
                    "host_bucket": r.get("host_bucket"),
                    "dripper_content": main_content,
                    "dripper_html": main_html,
                    "dripper_error": error,
                    "dripper_time_s": per_page_s,
                    "xpath_rules": xpath_rules,
                    "template_html": template_html,
                    "inference_time_s": per_page_s,
                }
            )

        done = min(batch_start + args.batch_size, len(rows))
        rate = done / (time.perf_counter() - t_load) if (time.perf_counter() - t_load) > 0 else 0
        print(
            f"[mineru_stage2] {done:>6}/{len(rows)} pages  "
            f"{rate:.1f} pages/s  batch={elapsed:.1f}s  "
            f"(runnable={len(runnable)}, too_long={len(skipped_too_long)})"
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    result_df = pd.DataFrame(results)
    _write_stage2_outputs(output_dir, out_parquet, result_df, args, t_start, t_load, errors, too_long_count)


def _write_stage2_outputs(output_dir, out_parquet, result_df, args, t_start, t_load, errors, too_long_count=0):
    t_end = time.perf_counter()
    total_pages = len(result_df)
    pages_s = total_pages / max(t_end - t_load, 1e-3)

    # Atomic write: write to .tmp then rename to avoid partial reads
    tmp_parquet = out_parquet.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp_parquet), index=False, compression="snappy")
    tmp_parquet.rename(out_parquet)

    total_s = t_end - t_start
    metrics = {
        "extractor": "MinerU-HTML-stage2-representatives",
        "model": args.model,
        "input_path": str(args.input),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_pages": total_pages,
        "successful_pages": total_pages - errors - too_long_count,
        "error_pages": errors,
        "too_long_pages": too_long_count,
        "html_size_limit_bytes": HTML_SIZE_LIMIT_BYTES,
        "elapsed_s": total_s,
        "load_s": t_load - t_start,
        "inference_s": t_end - t_load,
        "throughput_pages_per_s": pages_s,
        "batch_size": args.batch_size,
        "output_parquet": str(out_parquet),
    }

    if args.num_shards > 1:
        out_metrics = output_dir / f"metrics_shard_{args.shard_index:04d}.json"
    else:
        out_metrics = output_dir / "metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("[mineru_stage2] DONE")
    print(f"  pages:      {total_pages:,}  ({errors} errors, {too_long_count} too_long)")
    print(f"  elapsed:    {total_s:.1f}s  (load={metrics['load_s']:.1f}s  inference={metrics['inference_s']:.1f}s)")
    print(f"  throughput: {pages_s:.1f} pages/s")
    print(f"  output:     {out_parquet}")
    print(f"  metrics:    {out_metrics}")


# ── Original standalone (baseline) logic ─────────────────────────────────────


def run_standalone(args):
    """Original per-page standalone mode (Run B / Run C baseline)."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    print(f"[mineru_standalone] input:       {args.input}")
    print(f"[mineru_standalone] output:      {args.output}")
    print(f"[mineru_standalone] max_pages:   {args.max_pages or 'all'}")
    print(f"[mineru_standalone] batch_size:  {args.batch_size}")
    print(f"[mineru_standalone] model:       {args.model}")
    print(f"[mineru_standalone] hf_cache:    {args.hf_cache}")
    print(f"[mineru_standalone] shard:       {args.shard_index}/{args.num_shards}")
    print()

    # ── Load input ────────────────────────────────────────────────────────────
    print("[mineru_standalone] loading manifest...")
    df = read_parquet(args.input)
    if args.max_pages > 0:
        df = df.head(args.max_pages)

    # Shard: slice rows by task index
    if args.num_shards > 1:
        total = len(df)
        shard_start = total * args.shard_index // args.num_shards
        shard_end = total * (args.shard_index + 1) // args.num_shards
        df = df.iloc[shard_start:shard_end].reset_index(drop=True)
        print(f"[mineru_standalone] shard {args.shard_index}/{args.num_shards}: rows {shard_start}–{shard_end - 1}")

    print(f"[mineru_standalone] {len(df):,} pages to process")

    if "html" not in df.columns:
        print("[mineru_standalone] ERROR: manifest missing 'html' column. Need WARC fetch first.", file=sys.stderr)
        sys.exit(1)

    # ── Load MinerU-HTML ──────────────────────────────────────────────────────
    print("[mineru_standalone] loading MinerUHTML extractor...")
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    # Use create_vllm_backend directly so we can set tensor_parallel_size=8
    # MinerUHTML() hardcodes tensor_parallel_size=1 — bypass it
    from mineru_html.api import MinerUHTMLConfig, MinerUHTMLGeneric
    from mineru_html.inference.factory import create_vllm_backend

    n_gpus = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    print(f"[mineru_standalone] tensor_parallel_size={n_gpus}", flush=True)

    config = MinerUHTMLConfig(prompt_version="short_compact", response_format="compact")
    llm = create_vllm_backend(
        model_path=args.model,
        response_format=config.response_format,
        # CRITICAL FIX: was 256*1024 — caused 180-240s stall batches on long HTML.
        # 32768 tokens is the actual model max and eliminates pathological batches.
        max_context_window=32768,
        model_init_kwargs={
            "tensor_parallel_size": n_gpus,
            "gpu_memory_utilization": 0.85,
        },
    )
    extractor = MinerUHTMLGeneric(llm, config)

    t_load = time.perf_counter()
    print(f"[mineru_standalone] extractor ready in {t_load - t_start:.1f}s")

    # ── Run inference in batches ──────────────────────────────────────────────
    rows = df.to_dict("records")
    results = []
    errors = 0

    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        html_list = [coerce_html(r.get("html", "")) for r in batch]

        t0 = time.perf_counter()
        try:
            batch_results = extractor.process(html_list)
        except Exception as e:
            print(f"[mineru_standalone] batch {batch_start // args.batch_size} ERROR: {e}", file=sys.stderr)
            batch_results = [None] * len(batch)
            errors += len(batch)

        elapsed = time.perf_counter() - t0

        for row, result in zip(batch, batch_results):
            if result is not None:
                try:
                    main_content = str(result.output_data.main_content or "")
                    main_html = str(getattr(result.output_data, "main_html", "") or "")
                    error = ""
                except Exception as e:
                    main_content = ""
                    main_html = ""
                    error = str(e)[:200]
                    errors += 1
            else:
                main_content = ""
                main_html = ""
                error = "batch_failed"

            results.append(
                {
                    "url": row.get("url", ""),
                    "url_host_name": row.get("url_host_name", ""),
                    "dripper_layout_id": row.get("dripper_layout_id", ""),
                    "dripper_content": main_content,
                    "dripper_html": main_html,
                    "dripper_error": error,
                    "dripper_time_s": elapsed / len(batch),
                }
            )

        done = min(batch_start + args.batch_size, len(rows))
        rate = done / (time.perf_counter() - t_load) if time.perf_counter() > t_load else 0
        print(f"[mineru_standalone] {done:>6}/{len(rows)} pages  {rate:.1f} pages/s  batch={elapsed:.1f}s")

    # ── Write outputs ─────────────────────────────────────────────────────────
    t_end = time.perf_counter()
    result_df = pd.DataFrame(results)
    if args.num_shards > 1:
        out_parquet = output_dir / f"shard_{args.shard_index:04d}_of_{args.num_shards:04d}.parquet"
    else:
        out_parquet = output_dir / "dripper_results.parquet"
    result_df.to_parquet(str(out_parquet), index=False, compression="snappy")

    total_s = t_end - t_start
    pages_s = len(rows) / max(t_end - t_load, 1)
    metrics = {
        "extractor": "MinerU-HTML-standalone",
        "model": args.model,
        "input_manifest_path": str(args.input),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_pages": len(rows),
        "successful_pages": len(rows) - errors,
        "error_pages": errors,
        "elapsed_s": total_s,
        "load_s": t_load - t_start,
        "inference_s": t_end - t_load,
        "throughput_pages_per_s": pages_s,
        "batch_size": args.batch_size,
        "output_parquet": str(out_parquet),
    }

    if args.num_shards > 1:
        out_metrics = output_dir / f"metrics_shard_{args.shard_index:04d}.json"
    else:
        out_metrics = output_dir / "metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("[mineru_standalone] DONE")
    print(f"  pages:      {len(rows):,}  ({errors} errors)")
    print(f"  elapsed:    {total_s:.1f}s  (load={metrics['load_s']:.1f}s  inference={metrics['inference_s']:.1f}s)")
    print(f"  throughput: {pages_s:.1f} pages/s")
    print(f"  output:     {out_parquet}")
    print(f"  metrics:    {out_metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input manifest parquet (must have url + html columns)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    parser.add_argument("--batch-size", type=int, default=32, help="Pages per MinerUHTML batch")
    parser.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="0-based shard index (default: SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards; 1 = no sharding")
    # ── Stage 2 flag ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--representatives-only",
        action="store_true",
        default=False,
        help=(
            "Stage 2 mode: read clustered_manifest.parquet (or cluster_assignments/ dir), "
            "filter to is_representative=True/is_noise=True, run GPU inference, "
            "and write inference_results/shard_NNNN_of_MMMM.parquet with "
            "url, layout_cluster_id, dripper_content, dripper_html, dripper_error, "
            "xpath_rules, template_html columns. "
            "Pages with HTML > 500 KB are written with dripper_error='too_long'."
        ),
    )
    args = parser.parse_args()

    if args.representatives_only:
        run_representatives_only(args)
    else:
        run_standalone(args)


if __name__ == "__main__":
    main()
