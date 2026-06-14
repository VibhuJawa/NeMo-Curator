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

"""Stage 3 (Ray variant): CPU template propagation via ProcessingStage + RayDataExecutor.

Drop-in replacement for stage3_cpu_propagation.py that uses NeMo Curator's
RayDataExecutor actor pool instead of multiprocessing.ProcessPoolExecutor.

Key differences from the ProcessPoolExecutor variant:
  1. Bindings (llm_web_kit + mineru_html) are loaded once per Ray actor in
     setup(), not re-imported on every chunk restart.
  2. _cluster_static_ok memo is instance state (self._cluster_static_ok) so it
     persists for the actor's lifetime and is not accidentally shared across actors.
  3. Slurm/Ray workers are spawned processes too — no fork-safety regression vs
     multiprocessing.get_context("spawn").
  4. content-length ratio guard is applied (invariant 8 — parity with upstream
     DripperHTMLLayoutPropagationStage._run_propagation lines 201-212).

WHEN TO USE THIS vs stage3_cpu_propagation.py:
  - Use this when running on a Ray cluster (multi-node Slurm + ray start --head/worker).
  - Use the ProcessPoolExecutor variant for simple single-node Slurm array jobs where
    Ray is not already running.

Slurm: --partition=cpu_long  --cpus-per-task=64  --mem=235G  --time=06:00:00
       (no --array needed; shard_index comes from --shard-index / SLURM_ARRAY_TASK_ID)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "dripper_time_s",
    "propagation_success",
    "propagation_method",
]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Pure helper functions (picklable, no global state — safe to call from actors)
# ---------------------------------------------------------------------------


def _coerce_html(raw: object) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return "" if raw is None else str(raw)


def _parse_xpath_rules(raw: object) -> list[dict[str, Any]] | None:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass  # malformed JSON — return None below
    return None


def _parse_mapping_json(raw: object) -> dict[str, Any] | None:
    """Deserialise Stage-2b template: pickle+base64 first, then JSON fallback."""
    import base64
    import pickle

    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            obj = pickle.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            logger.debug("pickle.loads from bytes failed; trying string decode")
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        for loader in (
            lambda s: pickle.loads(base64.b64decode(s)),  # own pipeline output (trusted source)
            lambda s: json.loads(s),
        ):
            try:
                obj = loader(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                logger.debug("loader failed; trying next")
    return None


def _token_f1(a: str, b: str) -> float:
    """Token-multiset F1 between two texts."""
    from collections import Counter

    ca = Counter(_TOKEN_RE.findall(a.lower())) if a else Counter()
    cb = Counter(_TOKEN_RE.findall(b.lower())) if b else Counter()
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p = common / sum(ca.values())
    r = common / sum(cb.values())
    return 2 * p * r / (p + r)


def _load_cluster_manifest_shard(path: str) -> pd.DataFrame:
    meta_cols = [
        "url",
        "url_host_name",
        "cluster_id",
        "cluster_role",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
    ]
    schema_names = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in meta_cols if c in schema_names]).to_pandas()
    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "cluster_role" not in df.columns:
        df["cluster_role"] = "singleton"
    if "html" in schema_names:
        sibling_mask = df["cluster_role"] == "sibling"
        if sibling_mask.any():
            html_df = pq.read_table(path, columns=["url", "html"]).to_pandas()
            html_df = html_df.drop_duplicates(subset="url", keep="first")
            df["html"] = df["url"].map(html_df.set_index("url")["html"])
            df.loc[~sibling_mask, "html"] = None
        else:
            df["html"] = None
    else:
        df["html"] = None
    return df


def _load_inference_results(path: str) -> pd.DataFrame:
    cols_needed = [
        "cluster_id",
        "layout_cluster_id",
        "url",
        "llm_output_raw",
        "xpath_rules",
        "template_html",
        "inference_time_s",
        "error",
        "dripper_error",
        "dripper_content",
        "dripper_html",
        "mapping_json",
    ]
    schema_names = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in cols_needed if c in schema_names]).to_pandas()
    if "cluster_id" not in df.columns and "layout_cluster_id" in df.columns:
        df = df.rename(columns={"layout_cluster_id": "cluster_id"})
    if "error" not in df.columns and "dripper_error" in df.columns:
        df = df.rename(columns={"dripper_error": "error"})
    return df


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(tmp_path), compression="snappy")
    tmp_path.rename(out_path)


# ---------------------------------------------------------------------------
# ProcessingStage for Stage 3 — one DocumentBatch = one cluster task
# ---------------------------------------------------------------------------


@dataclass
class _StageConfig:
    """Groups LBP/content hyperparameters for Stage3PropagationStage.build()."""

    dynamic_classid_similarity_threshold: float = 0.70
    more_noise_enable: bool = True
    min_content_length_ratio: float = 0.25
    max_content_length_ratio: float = 4.0
    static_validation_min_f1: float = 0.97
    worker_count: int | None = None


@dataclass(kw_only=True)
class Stage3PropagationStage:
    """NeMo Curator ProcessingStage that processes one cluster task per DocumentBatch.

    Each Ray actor loads llm_web_kit and mineru_html once in setup().
    The _cluster_static_ok dict is per-actor-instance, not module-level, so it
    survives across DocumentBatch calls within the same actor lifetime without
    cross-actor contamination.

    Usage
    -----
    Build the stage (lazy import pattern keeps the module importable without Curator):

        stage = Stage3PropagationStage.build(
            dynamic_classid_similarity_threshold=0.70,
            more_noise_enable=True,
            min_content_length_ratio=0.25,
            max_content_length_ratio=4.0,
            static_validation_min_f1=0.97,
            worker_count=64,
        )

    Then pass it to RayDataExecutor.execute() alongside DocumentBatch tasks whose
    _metadata["cluster_task"] is a dict matching the shape produced by
    _build_cluster_tasks().
    """

    dynamic_classid_similarity_threshold: float = 0.70
    more_noise_enable: bool = True
    min_content_length_ratio: float = 0.25
    max_content_length_ratio: float = 4.0
    static_validation_min_f1: float = 0.97
    worker_count: int | None = None

    # Instance-level state — set in setup(), NOT module-level globals
    _lbp_bindings: object = field(init=False, repr=False, default=None)
    _mineru_bindings: object = field(init=False, repr=False, default=None)
    _cluster_static_ok: dict[str, bool] = field(init=False, repr=False, default_factory=dict)
    _initialized: bool = field(init=False, repr=False, default=False)

    # Filled by build() — kept as None here so the dataclass stays importable
    # without nemo_curator on PYTHONPATH.
    _stage_base_cls: object = None
    _resources_cls: object = None
    _document_batch_cls: object = None

    @classmethod
    def build(cls, cfg: _StageConfig | None = None, **kwargs: object) -> type:
        """Return a concrete ProcessingStage subclass ready for RayDataExecutor.

        Pass a ``_StageConfig`` instance, or keyword args that match its fields.
        Imports nemo_curator lazily so the file stays importable without it.
        """
        if cfg is None:
            cfg = _StageConfig(**{k: v for k, v in kwargs.items() if hasattr(_StageConfig, k)})  # type: ignore[arg-type]
        return _build_stage3_impl(cfg)


# ---------------------------------------------------------------------------
# Module-level factory used by Stage3PropagationStage.build() to construct the
# concrete ProcessingStage subclass without embedding a 400-line class body
# inside a classmethod (which triggers C901 complexity violations).
# ---------------------------------------------------------------------------


def _build_stage3_impl(cfg: _StageConfig) -> type:
    """Build and return the concrete ProcessingStage subclass closed over cfg."""
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch

    _dct = cfg.dynamic_classid_similarity_threshold
    _nme = cfg.more_noise_enable
    _min = cfg.min_content_length_ratio
    _max = cfg.max_content_length_ratio
    _f1 = cfg.static_validation_min_f1
    _wc = cfg.worker_count

    class _Stage3PropagationStageImpl(ProcessingStage[DocumentBatch, DocumentBatch]):
        """Concrete ProcessingStage for Stage 3 CPU propagation.

        Each actor has its own _cluster_static_ok dict (instance state, not
        module-level), so the static/dynamic LBP validation memo is per-actor
        and does not leak across actors or between runs.

        Because setup() is overridden, is_actor_stage() returns True automatically
        and RayDataExecutor wraps this as a persistent actor pool.
        """

        name: str = "stage3_cpu_propagation"
        resources = Resources(cpus=1.0)  # 1 CPU core per actor; tune via worker_count
        batch_size = 1  # one cluster task (DocumentBatch) per call

        def num_workers(self) -> int | None:
            return _wc

        def setup(self, _worker_metadata: object = None) -> None:
            """Load heavy bindings once per actor.  Called by RayDataStageActorAdapter.__init__."""
            if self._initialized:
                return
            self._lbp_bindings = self._load_lbp_bindings()
            self._mineru_bindings = self._load_mineru_bindings()
            self._cluster_static_ok: dict[str, bool] = {}
            self._initialized = True

        def _load_lbp_bindings(self) -> object:
            try:
                from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser

                class _B:
                    pass

                b = _B()
                b.layout_parser_cls = LayoutBatchParser
            except ImportError as exc:
                logger.warning("llm_web_kit unavailable in actor: %s", exc)
                return None
            else:
                return b

        def _load_mineru_bindings(self) -> object:
            try:
                from mineru_html.base import MinerUHTMLCase, MinerUHTMLInput, MinerUHTMLOutput
                from mineru_html.process import convert2content

                class _MB:
                    pass

                mb = _MB()
                mb.convert2content = convert2content
                mb.output_cls = MinerUHTMLOutput
                mb.case_cls = MinerUHTMLCase
                mb.input_cls = MinerUHTMLInput
                try:
                    from nemo_curator.stages.text.experimental.dripper.stage import (
                        _strip_xml_incompatible_chars,
                    )

                    mb.strip_xml = _strip_xml_incompatible_chars
                except ImportError:
                    mb.strip_xml = None  # optional helper — absence is safe
            except ImportError as exc:
                logger.warning("mineru_html unavailable in actor: %s", exc)
                return None
            else:
                return mb

        def process(self, task: DocumentBatch) -> DocumentBatch:
            if not self._initialized:
                self.setup()

            cluster_task: dict[str, Any] = task._metadata.get("cluster_task", {})
            if not cluster_task:
                df = task.to_pandas()
                results = [
                    self._make_fallback_row(r, str(r.get("cluster_role", "singleton")), "missing_cluster_task")
                    for r in df.to_dict("records")
                ]
                return DocumentBatch(
                    dataset_name=task.dataset_name,
                    data=pd.DataFrame(results, columns=OUTPUT_COLUMNS),
                    _metadata=task._metadata,
                    _stage_perf=task._stage_perf,
                )

            results = self._process_cluster_task(cluster_task)
            return DocumentBatch(
                dataset_name=task.dataset_name,
                data=pd.DataFrame(results, columns=OUTPUT_COLUMNS),
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )

        def _process_cluster_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
            manifest_rows = task["manifest_rows"]
            gpu_row = task.get("gpu_row")
            mapping_data = task.get("mapping_data")
            sib_rows = [r for r in manifest_rows if str(r.get("cluster_role", "")) == "sibling"]
            use_static = bool(
                sib_rows
                and mapping_data is not None
                and self._cluster_static_trustworthy(task.get("cluster_id"), sib_rows, mapping_data)
            )
            return self._dispatch_rows(manifest_rows, gpu_row, mapping_data, use_static)

        def _dispatch_rows(
            self,
            manifest_rows: list[dict[str, Any]],
            gpu_row: dict[str, Any] | None,
            mapping_data: dict[str, Any] | None,
            use_static: bool,
        ) -> list[dict[str, Any]]:
            """Dispatch each row to the appropriate handler."""
            results = []
            for row in manifest_rows:
                role = str(row.get("cluster_role", "singleton"))
                if role in ("representative", "singleton"):
                    if gpu_row is not None:
                        merged = dict(row)
                        merged.update(
                            {
                                "dripper_content": gpu_row.get("dripper_content", ""),
                                "dripper_html": gpu_row.get("dripper_html", gpu_row.get("llm_output_raw", "")),
                                "dripper_error": gpu_row.get("error", ""),
                                "inference_time_s": gpu_row.get("inference_time_s", 0.0),
                            }
                        )
                        fn = (
                            self._process_representative_row
                            if role == "representative"
                            else self._process_singleton_row
                        )
                        results.append(fn(merged))
                    else:
                        results.append(self._make_fallback_row(row, role, f"missing_gpu_result_for_{role}"))
                elif role == "sibling":
                    results.append(self._process_sibling_row(row, mapping_data, use_static))
                else:
                    results.append(self._make_fallback_row(row, role, f"unknown_cluster_role={role}"))
            return results

        def _cluster_static_trustworthy(
            self,
            cluster_id: object,
            sample_rows: list[dict[str, Any]],
            mapping_data: dict[str, Any] | None,
        ) -> bool:
            """Return True if static LBP reproduces dynamic LBP on K sample siblings."""
            if mapping_data is None:
                return False
            key = str(cluster_id)
            if key in self._cluster_static_ok:
                return self._cluster_static_ok[key]

            k = 3
            f1s: list[float] = []
            for row in sample_rows[:k]:
                html = _coerce_html(row.get("html", ""))
                if not html.strip():
                    continue
                sh, se = self._lbp_propagate(html, mapping_data, dynamic=False)
                dh, de = self._lbp_propagate(html, mapping_data, dynamic=True)
                if not dh or de:
                    continue
                if not sh or se:
                    f1s.append(0.0)
                    continue
                url = row.get("url", "")
                sc, _ = self._convert_to_content(sh, url)
                dc, _ = self._convert_to_content(dh, url)
                f1s.append(_token_f1(sc, dc))

            ok = bool(f1s) and (sum(f1s) / len(f1s) >= _f1)
            self._cluster_static_ok[key] = ok
            return ok

        def _lbp_propagate(self, html: str, mapping_data: dict[str, Any], dynamic: bool = True) -> tuple[str, str]:
            """Run LayoutBatchParser propagation. Returns (main_html, error)."""
            if self._lbp_bindings is None:
                return "", "llm_web_kit_not_available"
            html_source = html.strip()
            if not html_source:
                return "", "empty_html"
            try:
                task_data = dict(mapping_data)
                task_data.update(
                    {
                        "html_source": html_source,
                        "dynamic_id_enable": dynamic,
                        "dynamic_classid_enable": dynamic,
                        "more_noise_enable": _nme,
                        "dynamic_classid_similarity_threshold": _dct,
                    }
                )
                parts = self._lbp_bindings.layout_parser_cls({}).parse(task_data)
            except Exception as exc:
                return "", f"layout_parser_error={exc!s:.200}"
            if parts.get("main_html_success") is False:
                return "", f"main_html_success_false sim={parts.get('main_html_sim', 'n/a')}"
            main_html = str(parts.get("main_html_body") or "")
            if not main_html.strip():
                return "", "layout_parser_empty_output"
            return main_html, ""

        def _convert_to_content(self, main_html: str, url: str) -> tuple[str, str]:
            """Convert main_html to text via MinerU-HTML. Returns (content, error)."""
            mb = self._mineru_bindings
            if mb is None:
                try:
                    import lxml.html

                    return lxml.html.fromstring(main_html).text_content().strip(), ""
                except Exception as exc:
                    return "", f"lxml_text_fallback_error={exc!s:.100}"
            try:
                case = mb.case_cls(mb.input_cls(raw_html="", url=url))
                case.output_data = mb.output_cls(main_html=main_html)
                if getattr(mb, "strip_xml", None) is not None and isinstance(case.output_data.main_html, str):
                    case.output_data.main_html = mb.strip_xml(case.output_data.main_html)
                result = mb.convert2content(case, output_format="mm_md")
                output = getattr(result, "output_data", None)
                content = getattr(output, "main_content", "") if output is not None else ""
                return str(content or ""), ""
            except Exception as exc:
                return "", f"content_conversion_error={exc!s:.150}"

        def _apply_ratio_guard(
            self, candidate_html: str, candidate_content: str, mapping_data: dict[str, Any]
        ) -> tuple[str, str, str]:
            """Content-length ratio guard. Returns (accepted_html, accepted_content, error_if_rejected)."""
            rep_len = mapping_data.get("_dripper_representative_content_len")
            if not rep_len or rep_len <= 0:
                return candidate_html, candidate_content, ""
            ratio = len(candidate_content) / rep_len
            if ratio < _min:
                return "", "", f"content_length_ratio_low={ratio:.3f}"
            if ratio > _max:
                return "", "", f"content_length_ratio_high={ratio:.3f}"
            return candidate_html, candidate_content, ""

        def _process_sibling_row(
            self, row: dict[str, Any], mapping_data: dict[str, Any] | None, use_static: bool = False
        ) -> dict[str, Any]:
            url = row.get("url", "")
            url_host_name = row.get("url_host_name", "")
            cluster_id = row.get("cluster_id")
            html = _coerce_html(row.get("html", ""))
            t0 = time.perf_counter()
            method, main_html, content, error = "fallback", "", "", ""

            if mapping_data is not None:
                main_html, content, error, method = self._try_static_then_dynamic(
                    html, url, mapping_data, use_static, error
                )

            if not main_html:
                method = "fallback"
                if not error:
                    error = "no_template_available"

            return {
                "url": url,
                "url_host_name": url_host_name,
                "cluster_id": cluster_id,
                "cluster_role": "sibling",
                "dripper_content": content,
                "dripper_html": main_html,
                "dripper_error": error,
                "dripper_time_s": time.perf_counter() - t0,
                "propagation_success": bool(main_html and not error),
                "propagation_method": method,
            }

        def _try_static_then_dynamic(
            self, html: str, url: str, mapping_data: dict[str, Any], use_static: bool, prev_error: str
        ) -> tuple[str, str, str, str]:
            """Try static LBP, then dynamic LBP. Returns (main_html, content, error, method)."""
            main_html, content, error, method = "", "", prev_error, "fallback"

            if use_static:
                lbp_html, lbp_err = self._lbp_propagate(html, mapping_data, dynamic=False)
                if lbp_html and not lbp_err:
                    raw_content, conv_err = self._convert_to_content(lbp_html, url)
                    if not conv_err:
                        ah, ac, re = self._apply_ratio_guard(lbp_html, raw_content, mapping_data)
                        if ah:
                            return ah, ac, "", "lbp_static"
                        error = re
                    else:
                        error = conv_err
                else:
                    error = lbp_err

            if not main_html:
                dyn_html, dyn_err = self._lbp_propagate(html, mapping_data, dynamic=True)
                if dyn_html and not dyn_err:
                    raw_content, conv_err = self._convert_to_content(dyn_html, url)
                    if not conv_err:
                        ah, ac, re = self._apply_ratio_guard(dyn_html, raw_content, mapping_data)
                        if ah:
                            return ah, ac, "", "layout_batch_parser"
                        error = re
                    else:
                        error = conv_err or dyn_err
                elif dyn_err:
                    error = f"static_failed({error}); dynamic_failed({dyn_err})" if error else dyn_err

            return main_html, content, error, method

        @staticmethod
        def _process_representative_row(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "url": row.get("url", ""),
                "url_host_name": row.get("url_host_name", ""),
                "cluster_id": row.get("cluster_id"),
                "cluster_role": "representative",
                "dripper_content": row.get("dripper_content", ""),
                "dripper_html": row.get("dripper_html", ""),
                "dripper_error": row.get("dripper_error", ""),
                "dripper_time_s": row.get("inference_time_s", 0.0),
                "propagation_success": not bool(row.get("dripper_error", "")),
                "propagation_method": "representative",
            }

        @staticmethod
        def _process_singleton_row(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "url": row.get("url", ""),
                "url_host_name": row.get("url_host_name", ""),
                "cluster_id": None,
                "cluster_role": "singleton",
                "dripper_content": row.get("dripper_content", ""),
                "dripper_html": row.get("dripper_html", ""),
                "dripper_error": row.get("dripper_error", ""),
                "dripper_time_s": row.get("inference_time_s", 0.0),
                "propagation_success": not bool(row.get("dripper_error", "")),
                "propagation_method": "singleton",
            }

        @staticmethod
        def _make_fallback_row(row: dict[str, Any], role: str, error: str) -> dict[str, Any]:
            return {
                "url": row.get("url", ""),
                "url_host_name": row.get("url_host_name", ""),
                "cluster_id": row.get("cluster_id") if role != "singleton" else None,
                "cluster_role": role,
                "dripper_content": "",
                "dripper_html": "",
                "dripper_error": error,
                "dripper_time_s": 0.0,
                "propagation_success": False,
                "propagation_method": "fallback",
            }

    return _Stage3PropagationStageImpl


# ---------------------------------------------------------------------------
# Task builder: manifest + GPU results → list[DocumentBatch]
# Each DocumentBatch = one cluster task; cluster_task dict lives in _metadata.
# ---------------------------------------------------------------------------

PAGES_PER_TASK = 300


def _build_gpu_lookups(gpu_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build cluster-id and url lookup dicts from GPU results DataFrame."""
    cluster_gpu_lookup: dict[str, dict[str, Any]] = {}
    for row in gpu_df.to_dict("records"):
        cid = row.get("cluster_id")
        if cid is not None and str(cid) not in cluster_gpu_lookup:
            cluster_gpu_lookup[str(cid)] = row

    singleton_gpu_lookup: dict[str, dict[str, Any]] = {}
    for row in gpu_df.to_dict("records"):
        cid = row.get("cluster_id")
        url = str(row.get("url") or "")
        if (cid is None or str(cid).lower() in ("none", "null", "nan", "")) and url:
            singleton_gpu_lookup[url] = row

    return cluster_gpu_lookup, singleton_gpu_lookup


def _group_manifest_by_cluster(
    manifest_df: pd.DataFrame,
) -> dict[str | None, list[dict[str, Any]]]:
    """Group manifest rows by cluster_id key."""
    cluster_groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        cid_key: str | None = (
            str(cid) if (cid is not None and str(cid).lower() not in ("none", "null", "nan", "")) else None
        )
        cluster_groups[cid_key].append(row)
    return cluster_groups


def build_cluster_tasks(
    manifest_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
) -> list[Any]:
    """Build a list of DocumentBatch objects, one per cluster task.

    Imported lazily inside process_shard to keep the module importable
    without nemo_curator.
    """
    from nemo_curator.tasks import DocumentBatch

    cluster_gpu_lookup, singleton_gpu_lookup = _build_gpu_lookups(gpu_df)
    cluster_groups = _group_manifest_by_cluster(manifest_df)

    tasks: list[dict[str, Any]] = []
    for cid_key, rows in cluster_groups.items():
        if cid_key is None:
            for row in rows:
                tasks.append(
                    {
                        "cluster_id": None,
                        "manifest_rows": [row],
                        "gpu_row": singleton_gpu_lookup.get(str(row.get("url", ""))),
                        "mapping_data": None,
                    }
                )
        else:
            gpu_row = cluster_gpu_lookup.get(cid_key)
            mapping_data = (
                _parse_mapping_json(gpu_row.get("mapping_json") or gpu_row.get("llm_output_raw"))
                if gpu_row is not None
                else None
            )
            non_sib = [r for r in rows if str(r.get("cluster_role", "")) != "sibling"]
            sib = [r for r in rows if str(r.get("cluster_role", "")) == "sibling"]
            tasks.append(
                {
                    "cluster_id": cid_key,
                    "manifest_rows": non_sib + sib[:PAGES_PER_TASK],
                    "gpu_row": gpu_row,
                    "mapping_data": mapping_data,
                }
            )
            for i in range(PAGES_PER_TASK, len(sib), PAGES_PER_TASK):
                tasks.append(
                    {
                        "cluster_id": cid_key,
                        "manifest_rows": sib[i : i + PAGES_PER_TASK],
                        "gpu_row": None,
                        "mapping_data": mapping_data,
                    }
                )

    # Wrap each task dict as a DocumentBatch with an empty DataFrame for data
    # (the actual rows are in _metadata["cluster_task"])
    doc_batches = []
    for t in tasks:
        # Use the first row's columns as schema; actors read from _metadata, not data.
        placeholder_df = pd.DataFrame(
            [{"url": r.get("url", ""), "cluster_role": r.get("cluster_role", "")} for r in t["manifest_rows"][:1]]
        )
        db = DocumentBatch(dataset_name="stage3", data=placeholder_df)
        db._metadata["cluster_task"] = t
        doc_batches.append(db)
    return doc_batches


# ---------------------------------------------------------------------------
# process_shard — mirrors stage3_cpu_propagation.process_shard
# ---------------------------------------------------------------------------


@dataclass
class _ShardSpec:
    """Groups shard routing args to reduce positional-arg count."""

    cluster_manifest_dir: str
    inference_results_dir: str
    output_dir: str
    shard_index: int
    num_shards: int


@dataclass
class _ShardContext:
    """Groups shard timing/counting args for _write_and_report."""

    shard_index: int
    num_shards: int
    my_files: list
    t_start: float


def _load_gpu_frames(
    gpu_dir: Path,
    shard_index: int,
    manifest_cluster_ids: set[str],
    manifest_urls: set[str],
) -> list[pd.DataFrame]:
    """Load and filter GPU result frames relevant to this shard's manifest."""
    exact_gpu = gpu_dir / f"shard_{shard_index:04d}.parquet"
    gpu_files = (
        [exact_gpu]
        if exact_gpu.exists()
        else (sorted(gpu_dir.glob("shard_*.parquet")) or sorted(gpu_dir.glob("*.parquet")))
    )
    if not gpu_files:
        msg = f"No GPU inference result files found in {gpu_dir}"
        raise FileNotFoundError(msg)

    frames = []
    for f in gpu_files:
        try:
            shard_df = _load_inference_results(str(f))
            if len(shard_df) == 0:
                continue
            mask = pd.Series(False, index=shard_df.index)
            if "cluster_id" in shard_df.columns and manifest_cluster_ids:
                mask |= shard_df["cluster_id"].astype(str).isin(manifest_cluster_ids)
            if "url" in shard_df.columns and manifest_urls:
                null_cid = shard_df["cluster_id"].isna() | shard_df["cluster_id"].astype(str).isin(
                    ("none", "null", "nan", "")
                )
                mask |= null_cid & shard_df["url"].astype(str).isin(manifest_urls)
            filtered = shard_df[mask]
            if len(filtered) > 0:
                frames.append(filtered)
        except OSError as exc:
            print(f"[stage3-ray] WARNING: could not read GPU shard {f}: {exc}", flush=True)
    return frames


def _collect_manifest_ids(manifest_df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Extract cluster-id set and URL set from manifest for GPU lookup filtering."""
    manifest_cluster_ids: set[str] = set()
    manifest_urls: set[str] = set()
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        if cid is not None and str(cid).lower() not in ("none", "null", "nan", ""):
            manifest_cluster_ids.add(str(cid))
        manifest_urls.add(str(row.get("url", "")))
    return manifest_cluster_ids, manifest_urls


def _load_and_build_tasks(manifest_df: pd.DataFrame, gpu_dir: Path, shard_index: int) -> list:
    """Load GPU results and build cluster DocumentBatch tasks. Returns list[DocumentBatch]."""
    manifest_cluster_ids, manifest_urls = _collect_manifest_ids(manifest_df)
    gpu_frames = _load_gpu_frames(gpu_dir, shard_index, manifest_cluster_ids, manifest_urls)
    gpu_df = pd.concat(gpu_frames, ignore_index=True) if gpu_frames else pd.DataFrame()
    del gpu_frames
    print(f"[stage3-ray] {len(gpu_df):,} relevant GPU result rows loaded", flush=True)
    print("[stage3-ray] building DocumentBatch tasks (one per cluster)...", flush=True)
    return build_cluster_tasks(manifest_df, gpu_df)


def process_shard(spec: _ShardSpec, num_workers: int, stage_cfg: _StageConfig | None = None) -> dict[str, Any]:
    """Process one shard of cluster tasks via RayDataExecutor actor pool."""
    from nemo_curator.backends.ray_data.executor import RayDataExecutor

    if stage_cfg is None:
        stage_cfg = _StageConfig(worker_count=num_workers)
    else:
        stage_cfg = _StageConfig(
            dynamic_classid_similarity_threshold=stage_cfg.dynamic_classid_similarity_threshold,
            more_noise_enable=stage_cfg.more_noise_enable,
            min_content_length_ratio=stage_cfg.min_content_length_ratio,
            max_content_length_ratio=stage_cfg.max_content_length_ratio,
            static_validation_min_f1=stage_cfg.static_validation_min_f1,
            worker_count=num_workers,
        )

    shard_index = spec.shard_index
    num_shards = spec.num_shards
    t_start = time.perf_counter()
    output_dir_path = Path(spec.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"shard_{shard_index:04d}.parquet"

    if out_path.exists():
        try:
            meta = pq.read_metadata(str(out_path))
            if meta.num_rows > 0:
                print(f"[stage3-ray] SKIP shard {shard_index} — already exists ({meta.num_rows:,} rows)", flush=True)
                return {"status": "skipped", "shard": shard_index, "rows": meta.num_rows}
            out_path.unlink(missing_ok=True)
        except OSError:
            out_path.unlink(missing_ok=True)  # corrupt file — remove and reprocess

    manifest_dir, gpu_dir = Path(spec.cluster_manifest_dir), Path(spec.inference_results_dir)
    manifest_files = sorted(manifest_dir.glob("shard_*.parquet")) or sorted(manifest_dir.glob("*.parquet"))
    if not manifest_files:
        msg = f"No manifest shards found in {manifest_dir}"
        raise FileNotFoundError(msg)

    total_files = len(manifest_files)
    my_files = manifest_files[total_files * shard_index // num_shards : total_files * (shard_index + 1) // num_shards]
    if not my_files:
        print(f"[stage3-ray] shard {shard_index}: no manifest files — writing empty shard", flush=True)
        _atomic_write_parquet(pd.DataFrame(columns=OUTPUT_COLUMNS), out_path)
        return {"status": "empty", "shard": shard_index, "rows": 0}

    print(f"[stage3-ray] shard {shard_index}/{num_shards}: loading {len(my_files)} manifest file(s)...", flush=True)
    manifest_df = pd.concat([_load_cluster_manifest_shard(str(f)) for f in my_files], ignore_index=True)
    print(f"[stage3-ray] {len(manifest_df):,} manifest rows loaded", flush=True)

    doc_tasks = _load_and_build_tasks(manifest_df, gpu_dir, shard_index)
    del manifest_df
    total_tasks = len(doc_tasks)
    print(f"[stage3-ray] shard {shard_index}: {total_tasks:,} cluster tasks", flush=True)

    stage_cls = Stage3PropagationStage.build(stage_cfg)

    executor = RayDataExecutor()
    print(f"[stage3-ray] executing via RayDataExecutor with {num_workers} actors...", flush=True)
    t_exec = time.perf_counter()
    output_tasks = executor.execute([stage_cls()], initial_tasks=doc_tasks)
    exec_elapsed = time.perf_counter() - t_exec
    print(f"[stage3-ray] execution done in {exec_elapsed:.1f}s, collecting results...", flush=True)

    result_df = _collect_results(output_tasks)
    shard_ctx = _ShardContext(shard_index=shard_index, num_shards=num_shards, my_files=my_files, t_start=t_start)
    return _write_and_report(result_df, out_path, output_dir_path, shard_ctx)


def _collect_results(output_tasks: list) -> pd.DataFrame:
    """Collect and align output DocumentBatch tasks into a single DataFrame."""
    all_frames = []
    for t in output_tasks:
        df = t.to_pandas()
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        all_frames.append(df[OUTPUT_COLUMNS])
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


def _write_and_report(
    result_df: pd.DataFrame,
    out_path: Path,
    output_dir_path: Path,
    ctx: _ShardContext,
) -> dict[str, Any]:
    """Write parquet output and return metrics dict."""
    _atomic_write_parquet(result_df, out_path)

    n_success = int(result_df["propagation_success"].fillna(False).sum())
    n_fallback = len(result_df) - n_success
    n_lbp = int((result_df["propagation_method"] == "layout_batch_parser").sum())
    n_lbp_static = int((result_df["propagation_method"] == "lbp_static").sum())
    n_rep = int((result_df["propagation_method"] == "representative").sum())
    n_singleton = int((result_df["propagation_method"] == "singleton").sum())
    total_pages = len(result_df)

    elapsed_total = time.perf_counter() - ctx.t_start
    pages_per_s = total_pages / max(elapsed_total, 0.001)
    metrics = {
        "shard_index": ctx.shard_index,
        "num_shards": ctx.num_shards,
        "manifest_files": len(ctx.my_files),
        "total_pages": total_pages,
        "success_pages": n_success,
        "fallback_pages": n_fallback,
        "lbp_pages": n_lbp,
        "lbp_static_pages": n_lbp_static,
        "representative_pages": n_rep,
        "singleton_pages": n_singleton,
        "elapsed_s": elapsed_total,
        "pages_per_s": pages_per_s,
        "output_path": str(out_path),
    }
    (output_dir_path / f"metrics_shard_{ctx.shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))

    print(f"[stage3-ray] shard {ctx.shard_index} DONE", flush=True)
    print(f"  pages:   {total_pages:,}  (success={n_success} fallback={n_fallback})", flush=True)
    print(f"  lbp_static={n_lbp_static}  lbp={n_lbp}  rep={n_rep}  singleton={n_singleton}", flush=True)
    print(f"  elapsed: {elapsed_total:.1f}s  ({pages_per_s:.1f} pages/s)", flush=True)
    print(f"  output:  {out_path}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 (Ray): CPU template propagation via RayDataExecutor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cluster-manifest", required=True)
    p.add_argument("--inference-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    p.add_argument("--num-shards", type=int, default=80)
    p.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "64")),
        help="Number of Ray actors (= num_workers() passed to the stage)",
    )
    p.add_argument("--dynamic-classid-similarity-threshold", type=float, default=0.70)
    p.add_argument(
        "--more-noise-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--min-content-length-ratio", type=float, default=0.25)
    p.add_argument("--max-content-length-ratio", type=float, default=4.0)
    p.add_argument(
        "--static-validation-min-f1",
        type=float,
        default=0.97,
        help=(
            "Minimum token-F1 for static LBP validation on K=3 sample siblings. Passed as _f1 to the stage closure."
        ),
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    print("=" * 70, flush=True)
    print("  Stage 3 (Ray): CPU Template Propagation via RayDataExecutor", flush=True)
    print("=" * 70, flush=True)
    print(f"  cluster_manifest:  {args.cluster_manifest}", flush=True)
    print(f"  inference_results: {args.inference_results}", flush=True)
    print(f"  output_dir:        {args.output_dir}", flush=True)
    print(f"  shard:             {args.shard_index}/{args.num_shards}", flush=True)
    print(f"  num_workers:       {args.num_workers}", flush=True)
    print(f"  classid_threshold: {args.dynamic_classid_similarity_threshold}", flush=True)
    print(f"  content_ratio:     [{args.min_content_length_ratio}, {args.max_content_length_ratio}]", flush=True)
    print(f"  static_val_f1:     {args.static_validation_min_f1}", flush=True)
    print("=" * 70, flush=True)

    shard_spec = _ShardSpec(
        cluster_manifest_dir=args.cluster_manifest,
        inference_results_dir=args.inference_results,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    stage_cfg = _StageConfig(
        dynamic_classid_similarity_threshold=args.dynamic_classid_similarity_threshold,
        more_noise_enable=args.more_noise_enable,
        min_content_length_ratio=args.min_content_length_ratio,
        max_content_length_ratio=args.max_content_length_ratio,
        static_validation_min_f1=args.static_validation_min_f1,
        worker_count=args.num_workers,
    )
    metrics = process_shard(shard_spec, args.num_workers, stage_cfg)

    status = metrics.get("status", "done")
    if status == "skipped":
        print(f"[stage3-ray] Shard {args.shard_index} already complete — skipped.", flush=True)
    elif status == "empty":
        print(f"[stage3-ray] Shard {args.shard_index} had no input — wrote empty shard.", flush=True)
    else:
        print(f"[stage3-ray] Shard {args.shard_index} complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
