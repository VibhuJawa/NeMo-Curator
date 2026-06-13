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

"""
stage1b_gpu_dbscan.py — GPU-only DBSCAN clustering on pre-computed DOM features.

INPUT:  stage1a output parquet (url, url_host_name, dom_feature JSON, html)
OUTPUT: cluster assignments parquet per shard:
          url, url_host_name, html, cluster_id, cluster_role,
          layout_cluster_id, is_representative, cluster_size

One spawn process per GPU; each owns its CUDA_VISIBLE_DEVICES and runs
cuML DBSCAN (cuBLAS matmul cosine sim) on its assigned host groups.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _singleton_row(url, host, html, warc_src: dict) -> dict:
    return {
        "url": url,
        "url_host_name": host,
        "html": html,
        "cluster_id": "",
        "cluster_role": "singleton",
        "layout_cluster_id": "",
        "is_representative": False,
        "cluster_size": 1,
        "warc_filename": warc_src.get("warc_filename"),
        "warc_record_offset": warc_src.get("warc_record_offset"),
        "warc_record_length": warc_src.get("warc_record_length"),
    }


def _detect_gpus() -> int:
    n = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS_PER_NODE", "")
    if n:
        try:
            return int(n.split(":")[-1])
        except ValueError:
            pass
    try:
        r = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=5)
        return max(1, sum(1 for line in r.stdout.splitlines() if line.startswith("GPU")))
    except Exception:
        return 1


def _cluster_one_gpu(
    gpu_id: int,
    hosts: list[tuple[str, list[dict]]],
    threshold: float,
    min_cluster_size: int,
    gpu_min_size: int,
    result_file: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering import (
            _gpu_available,
            cluster_html_struct_gpu,
        )
        from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings

        web = _load_llm_web_kit_bindings()
        has_gpu = _gpu_available()
    except Exception as e:
        print(f"[stage1b GPU {gpu_id}] WARNING: cuML unavailable ({e}), using sklearn", flush=True)
        cluster_html_struct_gpu = None
        web = None
        has_gpu = False

    def _run_clustering(chunk, ci=None):
        try:
            if cluster_html_struct_gpu and has_gpu and len(chunk) >= gpu_min_size:
                cc, _ = cluster_html_struct_gpu(chunk, threshold=threshold, gpu_min_size=gpu_min_size)
            elif web:
                cc, _ = web.cluster_html_struct(chunk, threshold=threshold)
            else:
                cc = chunk
                for i, s in enumerate(cc):
                    s["layout_id"] = 0 if i == 0 else -1
            if ci is not None:
                for s in cc:
                    lid = s.get("layout_id", -1)
                    if lid >= 0:
                        s["layout_id"] = ci * 100000 + lid
        except Exception as exc:
            label = f"chunk {ci}" if ci is not None else "DBSCAN"
            print(f"[stage1b GPU {gpu_id}] {label} failed for chunk: {exc}", flush=True)
            cc = chunk
        return cc

    all_assignments = []
    max_host = int(os.environ.get("STAGE1B_MAX_HOST_SIZE", "3000"))

    for host, samples in hosts:
        if not samples:
            continue

        if len(samples) > max_host:
            print(
                f"[stage1b GPU {gpu_id}] {host}: {len(samples)} pages > max_host_size={max_host}, chunking",
                flush=True,
            )
            chunk_results = []
            for ci, chunk_start in enumerate(range(0, len(samples), max_host)):
                chunk_results.extend(_run_clustering(samples[chunk_start : chunk_start + max_host], ci=ci))
            clustered = chunk_results
        else:
            clustered = _run_clustering(samples)

        by_lid: dict[int, list] = defaultdict(list)
        for s in clustered:
            by_lid[int(s.get("layout_id", -1))].append(s)

        for lid, members in by_lid.items():
            if lid < 0 or len(members) < min_cluster_size:
                for m in members:
                    all_assignments.append(_singleton_row(m["url"], host, m.get("html"), m))
                continue

            cid = f"{host}:cluster_{lid}"
            try:
                rep_candidates = [{"track_id": m["url"], "html": m.get("html", "")} for m in members]
                rep_url = web.select_representative_html(rep_candidates)["track_id"] if web else members[0]["url"]
            except Exception:
                rep_url = members[0]["url"]

            for m in members:
                is_rep = m["url"] == rep_url
                all_assignments.append(
                    {
                        "url": m["url"],
                        "url_host_name": host,
                        "html": m.get("html"),
                        "cluster_id": cid,
                        "cluster_role": "representative" if is_rep else "sibling",
                        "layout_cluster_id": cid,
                        "is_representative": is_rep,
                        "cluster_size": len(members),
                        "warc_filename": m.get("warc_filename"),
                        "warc_record_offset": m.get("warc_record_offset"),
                        "warc_record_length": m.get("warc_record_length"),
                    }
                )

    df = pd.DataFrame(all_assignments)
    df.to_parquet(result_file, index=False, compression="snappy")
    print(f"[stage1b GPU {gpu_id}] done: {len(df)} rows → {result_file}", flush=True)


def run(args):
    import multiprocessing as mp

    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        if exact.exists():
            inp = exact
        else:
            candidates = sorted(inp.glob("shard_*.parquet"))
            if not candidates:
                raise FileNotFoundError(f"No shard parquets found in {args.input}")
            inp = candidates[0]
    pf = pq.ParquetFile(str(inp))
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards

    need = ["url", "url_host_name", "dom_feature", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    cols = [c for c in need if c in pf.schema_arrow.names]

    rows_seen, parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=cols):
        df = batch.to_pandas()
        lo = max(0, start - rows_seen)
        hi = min(len(df), end - rows_seen)
        rows_seen += len(df)
        if lo < hi:
            parts.append(df.iloc[lo:hi])
        if rows_seen >= end:
            break

    shard_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    n_gpus = _detect_gpus()
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline_metrics import StageMetrics

    tracker = StageMetrics("stage1b", shard_index=args.shard_index, num_shards=args.num_shards, n_gpus=n_gpus)
    tracker.start()
    print(f"[stage1b] shard {args.shard_index}/{args.num_shards}: {len(shard_df):,} pages, {n_gpus} GPUs")

    if len(shard_df) == 0:
        return

    by_host: dict[str, list] = defaultdict(list)
    singleton_rows = []
    for rec in shard_df.to_dict("records"):
        feat_json = rec.get("dom_feature", "")
        if not feat_json:
            singleton_rows.append(_singleton_row(rec["url"], rec.get("url_host_name", ""), rec.get("html"), rec))
            continue
        try:
            feat = json.loads(feat_json)
        except Exception:
            feat = None
        if feat is None:
            continue
        host = str(rec.get("url_host_name") or "")
        by_host[host].append(
            {
                "track_id": rec["url"],
                "url": rec["url"],
                "html": rec.get("html", ""),
                "feature": feat,
                "warc_filename": rec.get("warc_filename"),
                "warc_record_offset": rec.get("warc_record_offset"),
                "warc_record_length": rec.get("warc_record_length"),
            }
        )

    sorted_hosts = sorted(by_host.items(), key=lambda kv: -len(kv[1]))
    gpu_assignments: list[list] = [[] for _ in range(n_gpus)]
    for i, (host, samples) in enumerate(sorted_hosts):
        gpu_assignments[i % n_gpus].append((host, samples))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_files = [str(out_dir / f"gpu_{gpu_id}_tmp.parquet") for gpu_id in range(n_gpus)]

    ctx = mp.get_context("spawn")
    procs = []
    t0 = time.perf_counter()
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_cluster_one_gpu,
            args=(
                gpu_id,
                gpu_assignments[gpu_id],
                args.threshold,
                args.min_cluster_size,
                args.gpu_min_size,
                tmp_files[gpu_id],
            ),
            name=f"dbscan-gpu{gpu_id}",
        )
        p.start()
        procs.append(p)

    failed = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed += 1
            print(f"[stage1b] WARNING: {p.name} exited with code {p.exitcode}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[stage1b] GPU DBSCAN done in {elapsed:.1f}s", flush=True)

    out_path = out_dir / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    import pyarrow as pa

    writer = None
    total_rows = 0
    for f in tmp_files:
        if not Path(f).exists():
            continue
        pf_tmp = pq.ParquetFile(f)
        for batch in pf_tmp.iter_batches(batch_size=8192):
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), batch.schema, compression="snappy")
            writer.write_batch(batch)
            total_rows += batch.num_rows
        Path(f).unlink()

    if singleton_rows:
        sing_table = pa.Table.from_pandas(pd.DataFrame(singleton_rows))
        if writer is None:
            writer = pq.ParquetWriter(str(tmp), sing_table.schema, compression="snappy")
        writer.write_table(sing_table)
        total_rows += len(singleton_rows)

    if writer:
        writer.close()
        tmp.rename(out_path)
    else:
        pd.DataFrame().to_parquet(str(out_path), index=False)

    print(f"[stage1b] merged {total_rows:,} rows → {out_path}", flush=True)
    result_df = pq.read_table(str(out_path), columns=["cluster_role"]).to_pandas()

    n_reps = int((result_df["cluster_role"] == "representative").sum())
    n_sing = int((result_df["cluster_role"] == "singleton").sum())
    call_reduction = 1.0 - (n_reps + n_sing) / max(len(result_df), 1)

    tracker.finish(total_pages=len(result_df), errors=failed)
    tracker.extra = {
        "representative_pages": n_reps,
        "singleton_pages": n_sing,
        "call_reduction_fraction": round(call_reduction, 4),
        "dbscan_elapsed_s": round(elapsed, 2),
        "output": str(out_path),
    }
    tracker.save(str(out_path.parent))
    tracker.checkpoint(len(result_df), label="final")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="stage1a output dir")
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--gpu-min-size", type=int, default=200)
    run(p.parse_args())


if __name__ == "__main__":
    main()
