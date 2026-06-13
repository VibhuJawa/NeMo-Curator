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
stage2b_cpu_postprocess.py — CPU-only template building from LLM responses.

RUNS ON: cpu_short partition (no GPU needed).

Reads Stage 2 output (url, cluster_id, llm_response, simp_html, map_html, html),
runs map_parser_cls to build the propagation template, then convert2content for
the representative's final extracted text.

Output adds: mapping_json, dripper_content, dripper_html
Stage 3 uses mapping_json for LayoutBatchParser propagation to siblings.
"""
import argparse, base64, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_metrics import StageMetrics

_BINDINGS_W = None
_BINDINGS_M = None
_STRIP_XML = None
_LABELS_TO_WEBKIT = None
_FALLBACK_HANDLER = None

def _init_worker():
    global _BINDINGS_W, _BINDINGS_M, _STRIP_XML, _LABELS_TO_WEBKIT, _FALLBACK_HANDLER
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    try:
        from nemo_curator.stages.text.experimental.dripper.stage import (
            _load_llm_web_kit_bindings, _load_mineru_html_bindings,
            _strip_xml_incompatible_chars, _labels_to_webkit_response,
        )
        _BINDINGS_W = _load_llm_web_kit_bindings()
        _BINDINGS_M = _load_mineru_html_bindings()
        _STRIP_XML = _strip_xml_incompatible_chars
        _LABELS_TO_WEBKIT = _labels_to_webkit_response
        try:
            _FALLBACK_HANDLER = _BINDINGS_M.get_fallback_handler("trafilatura")
        except Exception:
            _FALLBACK_HANDLER = None
    except Exception as e:
        print(f"[stage2b] WARNING: bindings unavailable: {e}", flush=True)


def _trafilatura_content(raw_html: str, url: str) -> str:
    """Last-resort content via the trafilatura fallback handler (matches the
    standalone baseline's --fallback trafilatura). Recovers pages the LLM left
    empty so they score against the baseline instead of F1=0."""
    if _FALLBACK_HANDLER is None or _BINDINGS_M is None or not raw_html.strip():
        return ""
    try:
        M = _BINDINGS_M
        case = M.case_cls(M.input_cls(raw_html=raw_html, url=url))
        case = M.extract_main_html_fallback(case, fallback_handler=_FALLBACK_HANDLER)
        od = getattr(case, "output_data", None)
        if od is not None and _STRIP_XML is not None and isinstance(getattr(od, "main_html", None), str):
            od.main_html = _STRIP_XML(od.main_html)
        case = M.convert2content(case, output_format="mm_md")
        od = getattr(case, "output_data", None)
        return str(getattr(od, "main_content", "") or "") if od is not None else ""
    except Exception:
        return ""


def _postprocess_one(rec: dict) -> dict:
    url          = rec.get("url", "")
    raw_html     = rec.get("html", "") or ""
    simp_html    = rec.get("simp_html", "") or ""
    map_html     = rec.get("map_html", "") or ""
    llm_response = rec.get("llm_response", "") or ""

    out = {
        "url":           url,
        "url_host_name": rec.get("url_host_name", ""),
        "cluster_id":    rec.get("cluster_id", ""),
        "cluster_role":  rec.get("cluster_role", ""),
        "mapping_json":  "",
        "dripper_content": "",
        "dripper_html":  "",
        "dripper_error": rec.get("dripper_error", "") or "",
        "inference_time_s": rec.get("inference_time_s", 0.0),
    }

    if not _BINDINGS_W or not _BINDINGS_M or not llm_response:
        if not llm_response:
            out["dripper_error"] = out["dripper_error"] or "no_llm_response"
            out["dripper_content"] = _trafilatura_content(raw_html, url)  # baseline parity
        return out

    role = str(rec.get("cluster_role", "") or "")
    M = _BINDINGS_M

    try:
        # Representative/singleton content comes from the SAME path the standalone
        # Dripper uses: parse_result → extract_main_html_single → convert2content.
        # The chat-templated compact model emits the verbose "<answer>1other2main…"
        # response that parse_result expects.
        case = M.case_cls(M.input_cls(raw_html=raw_html, url=url))
        if simp_html or map_html:
            case.process_data = M.process_data_cls(simpled_html=simp_html, map_html=map_html)
        case.generate_output = M.generate_output_cls(response=llm_response)

        webkit_response = {}
        try:
            case = M.parse_result(case)
            if _LABELS_TO_WEBKIT is not None:
                webkit_response = _LABELS_TO_WEBKIT(getattr(case.parse_result, "item_label", {}))
            case = M.extract_main_html_single(case)
        except Exception as exc:
            out["dripper_error"] = f"primary_failed:{type(exc).__name__}:{str(exc)[:70]}"
            if _FALLBACK_HANDLER is not None:
                try:
                    case = M.extract_main_html_fallback(case, fallback_handler=_FALLBACK_HANDLER)
                except Exception as fexc:
                    out["dripper_error"] += f"; fb:{str(fexc)[:50]}"

        od = getattr(case, "output_data", None)
        if od is not None and _STRIP_XML is not None and isinstance(getattr(od, "main_html", None), str):
            od.main_html = _STRIP_XML(od.main_html)
        try:
            case = M.convert2content(case, output_format="mm_md")
        except Exception as exc:
            out["dripper_error"] = out["dripper_error"] or f"convert:{type(exc).__name__}:{str(exc)[:70]}"
        od = getattr(case, "output_data", None)
        out["dripper_html"]    = str(getattr(od, "main_html", "") or "") if od is not None else ""
        out["dripper_content"] = str(getattr(od, "main_content", "") or "") if od is not None else ""
        # Recover empty extractions via trafilatura (baseline parity) so they don't score F1=0.
        if not out["dripper_content"].strip():
            out["dripper_content"] = _trafilatura_content(raw_html, url)

        # Propagation template (representatives only) — built with the parsed
        # webkit_response, exactly as the standalone layout-template stage does.
        if role == "representative" and _BINDINGS_W is not None:
            try:
                template = _BINDINGS_W.map_parser_cls({}).parse({
                    "typical_raw_html":     raw_html,
                    "typical_raw_tag_html": map_html or simp_html,
                    "llm_response":         webkit_response,
                })
                # Serialize LOSSLESSLY via pickle+base64. The template's
                # html_element_dict has tuple keys; a JSON round-trip stringifies
                # them and breaks LayoutBatchParser propagation in Stage 3.
                out["mapping_json"] = base64.b64encode(pickle.dumps(template)).decode("ascii")
            except Exception as exc:
                out["dripper_error"] = out["dripper_error"] or \
                    f"map_parser:{type(exc).__name__}:{str(exc)[:70]}"
    except Exception as e:
        out["dripper_error"] = f"postprocess:{type(e).__name__}:{str(e)[:150]}"

    return out


def run(args):
    tracker = StageMetrics("stage2b", shard_index=args.shard_index,
                           num_shards=args.num_shards, n_workers=args.workers)
    tracker.start()

    inp = Path(args.input)
    if inp.is_dir():
        import glob as _g
        files = sorted(_g.glob(str(inp / f"shard_{args.shard_index:04d}.parquet")))
        if not files:
            files = sorted(_g.glob(str(inp / "*.parquet")))
        inp = Path(files[0]) if files else inp

    df = pq.ParquetFile(str(inp)).read().to_pandas()
    print(f"[stage2b] {len(df):,} pages to postprocess ({args.workers} workers)", flush=True)

    records = df.to_dict("records")
    results = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_postprocess_one, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 500 == 0:
                ok_so_far = sum(1 for r in results if r.get("mapping_json"))
                tracker.checkpoint(pages_done=done,
                                   label=f"mapping_ok={ok_so_far}")

    result_df = pd.DataFrame(results)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet"
                      if args.num_shards > 1 else "postprocess_results.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    mapping_ok  = int((result_df["mapping_json"].astype(str).str.len() > 5).sum())
    content_ok  = int((result_df["dripper_content"].astype(str).str.len() > 5).sum())
    errors      = int((result_df["dripper_error"].astype(str).str.len() > 2).sum())
    tracker.finish(total_pages=len(result_df), errors=errors)
    tracker.extra = {"mapping_ok": mapping_ok, "content_ok": content_ok}
    print(f"[stage2b] content_ok={content_ok}/{len(result_df)}  "
          f"mapping_ok(reps)={mapping_ok}  errors={errors}", flush=True)
    tracker.save(args.output)
    print(f"[stage2b] output → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, help="Stage 2 output dir")
    p.add_argument("--output",      required=True, help="Output dir")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards",  type=int, default=1)
    p.add_argument("--workers",     type=int, default=max(1, (os.cpu_count() or 4) - 2))
    run(p.parse_args())


if __name__ == "__main__":
    main()
