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

"""Tests for Dripper Common Crawl manifest input helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DRIPPER_CC_DIR = REPO_ROOT / "tutorials" / "text" / "dripper-common-crawl"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dripper_cc_module(name: str, filename: str):
    sys.path.insert(0, str(DRIPPER_CC_DIR))
    try:
        return load_module(name, DRIPPER_CC_DIR / filename)
    finally:
        sys.path.remove(str(DRIPPER_CC_DIR))


def test_host_clustered_manifest_builder_filters_and_sorts(tmp_path: Path, monkeypatch) -> None:
    builder = load_module("dripper_manifest_builder", DRIPPER_CC_DIR / "build_host_clustered_manifest.py")
    monkeypatch.setattr(builder, "xxhash_host_bucket", lambda host, modulus: len(host) % modulus)

    index_path = tmp_path / "index.parquet"
    output_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            make_index_row("https://b.example/1", "b.example", 200, "text/html", 10, 11),
            make_index_row("https://a.example/1", "a.example", 200, "text/html", 20, 12),
            make_index_row("https://a.example/2", "a.example", 200, "text/html", 30, 13),
            make_index_row("https://a.example/3", "a.example", 200, "text/html", 40, 14),
            make_index_row("https://b.example/2", "b.example", 200, "text/html", 50, 15),
            make_index_row("https://c.example/1", "c.example", 200, "application/json", 60, 16),
            make_index_row("https://d.example/1", "d.example", 404, "text/html", 70, 17),
        ]
    ).to_parquet(index_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_host_clustered_manifest.py",
            "--cc-index-path",
            str(index_path),
            "--output",
            str(output_path),
            "--max-pages",
            "4",
            "--min-host-pages",
            "2",
            "--max-pages-per-host",
            "2",
        ],
    )
    assert builder.main() == 0

    out = pd.read_parquet(output_path)
    assert out["url_host_name"].tolist() == ["a.example", "a.example", "b.example", "b.example"]
    assert out["warc_record_offset"].tolist() == [20, 30, 10, 50]
    assert out["warc_record_length"].tolist() == [12, 13, 11, 15]
    assert (output_path.with_suffix(output_path.suffix + ".metrics.json")).exists()


def test_xxhash_host_bucket_matches_llm_webkit_formula() -> None:
    import xxhash

    builder = load_module("dripper_manifest_builder_xxhash", DRIPPER_CC_DIR / "build_host_clustered_manifest.py")
    host = "www.example.com"

    assert builder.xxhash_host_bucket(host, 10000) == xxhash.xxh64_intdigest(host) % 10000


def test_dripper_main_loads_manifest_html(tmp_path: Path) -> None:
    main_mod = load_module("dripper_cc_main", DRIPPER_CC_DIR / "main.py")
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {"url": "https://a.example/1", "html": "<html>one</html>", "content_type": "text/html"},
            {"url": "https://a.example/2", "html": "<html>two</html>", "content_type": "text/html"},
            {"url": "https://a.example/json", "html": "{}", "content_type": "application/json"},
        ]
    ).to_parquet(manifest_path, index=False)

    args = SimpleNamespace(
        input_manifest_path=str(manifest_path),
        max_pages=0,
        min_html_bytes=1,
        html_only=True,
        manifest_fetch_workers=2,
        manifest_warc_bucket="crawl-data",
    )
    pages, sampled, stats = main_mod.load_manifest_pages(args)

    assert sampled == [str(manifest_path)]
    assert [page["url"] for page in pages] == ["https://a.example/1", "https://a.example/2"]
    assert [page["html"] for page in pages] == ["<html>one</html>", "<html>two</html>"]
    assert stats["manifest_html_rows_loaded"] == 2
    assert stats["manifest_rows_skipped_non_html"] == 1


def test_s3_client_pool_matches_manifest_fetch_workers(monkeypatch) -> None:
    main_mod = load_module("dripper_cc_main_s3_pool", DRIPPER_CC_DIR / "main.py")
    calls: dict[str, object] = {}

    class FakeBotoConfig:
        def __init__(self, **kwargs) -> None:
            calls["config_kwargs"] = kwargs

    fake_boto3 = ModuleType("boto3")

    def fake_client(**kwargs):
        calls["client_kwargs"] = kwargs
        return object()

    fake_boto3.client = lambda *args, **kwargs: fake_client(service=args[0], **kwargs)  # type: ignore[attr-defined]
    fake_botocore = ModuleType("botocore")
    fake_botocore_config = ModuleType("botocore.config")
    fake_botocore_config.Config = FakeBotoConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)

    args = SimpleNamespace(
        s3_endpoint_url="https://example.invalid",
        s3_region="us-east-1",
        manifest_fetch_workers=128,
    )

    main_mod.make_s3_client(args)

    assert calls["client_kwargs"]["service"] == "s3"
    assert calls["config_kwargs"]["max_pool_connections"] == 128


def test_host_bucketed_index_shard_builder_writes_partitioned_shards(tmp_path: Path, monkeypatch) -> None:
    builder = load_dripper_cc_module("host_bucketed_index_shards", "build_host_bucketed_index_shards.py")
    clustered_builder = sys.modules.get("build_host_clustered_manifest")
    assert clustered_builder is not None
    monkeypatch.setattr(clustered_builder, "xxhash_host_bucket", lambda host, modulus: len(host) % modulus)

    index_path = tmp_path / "index.parquet"
    output_dir = tmp_path / "bucketed"
    pd.DataFrame(
        [
            make_index_row("https://a.example/1", "a.example", 200, "text/html", 20, 12),
            make_index_row("https://a.example/2", "a.example", 200, "text/html", 30, 13),
            make_index_row("https://b.example/1", "b.example", 200, "text/html", 10, 11),
            make_index_row("https://json.example/1", "json.example", 200, "application/json", 40, 14),
        ]
    ).to_parquet(index_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_host_bucketed_index_shards.py",
            "--cc-index-path",
            str(index_path),
            "--output-dir",
            str(output_dir),
            "--source-id",
            "part-test",
            "--host-bucket-group-size",
            "10",
        ],
    )
    assert builder.main() == 0

    shard_files = sorted(output_dir.rglob("*.parquet"))
    assert len(shard_files) == 1
    out = pd.concat([pd.read_parquet(path) for path in shard_files], ignore_index=True)
    assert sorted(out["url"].tolist()) == [
        "https://a.example/1",
        "https://a.example/2",
        "https://b.example/1",
    ]
    assert (output_dir / "part-test.metrics.json").exists()


def test_host_clustered_manifest_reducer_selects_top_hosts(tmp_path: Path, monkeypatch) -> None:
    reducer = load_dripper_cc_module("host_clustered_manifest_from_shards", "build_host_clustered_manifest_from_shards.py")
    shard_dir = tmp_path / "shards" / "host_bucket_group=0"
    shard_dir.mkdir(parents=True)
    output_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            make_index_row("https://a.example/3", "a.example", 200, "text/html", 30, 13),
            make_index_row("https://a.example/1", "a.example", 200, "text/html", 10, 11),
            make_index_row("https://a.example/2", "a.example", 200, "text/html", 20, 12),
            make_index_row("https://b.example/2", "b.example", 200, "text/html", 50, 15),
            make_index_row("https://b.example/1", "b.example", 200, "text/html", 40, 14),
            make_index_row("https://c.example/1", "c.example", 200, "text/html", 60, 16),
        ]
    ).assign(host_bucket=0).to_parquet(shard_dir / "part-test.parquet", index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_host_clustered_manifest_from_shards.py",
            "--input-shards",
            str(tmp_path / "shards"),
            "--output",
            str(output_path),
            "--max-pages",
            "4",
            "--min-host-pages",
            "2",
            "--max-pages-per-host",
            "2",
        ],
    )
    assert reducer.main() == 0

    out = pd.read_parquet(output_path)
    assert out["url_host_name"].tolist() == ["a.example", "a.example", "b.example", "b.example"]
    assert out["url"].tolist() == [
        "https://a.example/1",
        "https://a.example/2",
        "https://b.example/1",
        "https://b.example/2",
    ]
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    assert metrics_path.exists()


def test_prompt_dedup_estimator_selects_top_host_rows(tmp_path: Path) -> None:
    estimator = load_dripper_cc_module("prompt_dedup_estimator", "estimate_prompt_dedup_call_reduction.py")
    shard_dir = tmp_path / "shards" / "host_bucket_group=7"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "part.parquet"
    pd.DataFrame(
        [
            make_index_row("https://b.example/1", "b.example", 200, "text/html", 10, 11),
            make_index_row("https://a.example/1", "a.example", 200, "text/html", 20, 12),
            make_index_row("https://a.example/2", "a.example", 200, "text/html", 30, 13),
            make_index_row("https://a.example/3", "a.example", 200, "text/html", 40, 14),
            make_index_row("https://b.example/2", "b.example", 200, "text/html", 50, 15),
            make_index_row("https://c.example/1", "c.example", 200, "text/html", 60, 16),
        ]
    ).to_parquet(shard_path, index=False)

    files = estimator.resolve_manifest_files(str(tmp_path / "shards"), {7})
    host_counts, rows_seen = estimator.count_hosts(files, batch_size=2, max_rows=0)
    selected_hosts = estimator.select_top_hosts(host_counts, top_hosts=2, min_host_pages=2)
    selected, stats = estimator.select_manifest_rows(
        files,
        selected_hosts=[host for host, _count in selected_hosts],
        batch_size=2,
        max_pages=3,
        max_pages_per_host=2,
        max_rows=0,
    )

    assert rows_seen == 6
    assert selected_hosts == [("a.example", 3), ("b.example", 2)]
    assert selected["url"].tolist() == [
        "https://b.example/1",
        "https://a.example/1",
        "https://a.example/2",
    ]
    assert stats["selected_by_host"] == {"b.example": 1, "a.example": 2}
    assert stats["stopped_by_max_pages"] is True


def test_prompt_dedup_sample_manifest_builder_replays_estimate_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    builder = load_dripper_cc_module(
        "prompt_dedup_sample_manifest_builder",
        "build_prompt_dedup_sample_manifest.py",
    )
    shard_dir = tmp_path / "shards" / "host_bucket_group=7"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            make_index_row("https://b.example/1", "b.example", 200, "text/html", 10, 11),
            make_index_row("https://a.example/1", "a.example", 200, "text/html", 20, 12),
            make_index_row("https://a.example/2", "a.example", 200, "text/html", 30, 13),
            make_index_row("https://a.example/3", "a.example", 200, "text/html", 40, 14),
            make_index_row("https://c.example/1", "c.example", 200, "text/html", 50, 15),
        ]
    ).to_parquet(shard_dir / "part.parquet", index=False)
    estimate_path = tmp_path / "prompt_dedup_estimate.json"
    output_path = tmp_path / "prompt_dedup_manifest_rows.parquet"
    estimate_path.write_text(
        json_dump(
            {
                "input": str(tmp_path / "shards"),
                "candidate_rows": 3,
                "selected_hosts": [{"host": "a.example", "count": 3}, {"host": "b.example", "count": 1}],
                "args": {
                    "batch_size": 2,
                    "host_bucket_groups": "7",
                    "max_files": 0,
                    "max_pages": 3,
                    "max_pages_per_host": 2,
                    "select_max_rows": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_prompt_dedup_sample_manifest.py",
            "--estimate-json",
            str(estimate_path),
            "--output",
            str(output_path),
        ],
    )
    assert builder.main() == 0

    out = pd.read_parquet(output_path)
    assert out["url"].tolist() == ["https://b.example/1", "https://a.example/1", "https://a.example/2"]
    assert {"warc_filename", "warc_record_offset", "warc_record_length"}.issubset(out.columns)
    assert output_path.with_suffix(output_path.suffix + ".metrics.json").exists()


def test_prompt_dedup_estimator_hash_metrics_do_not_need_prompt_text(monkeypatch) -> None:
    estimator = load_dripper_cc_module("prompt_dedup_estimator_metrics", "estimate_prompt_dedup_call_reduction.py")
    args = SimpleNamespace(
        top_prompt_groups=10,
        max_tokens=2048,
        top_p=1.0,
        prompt_version="short_compact",
        dynamic_max_tokens=False,
        dynamic_max_token_padding=16,
        dynamic_max_tokens_per_item=6,
        dynamic_min_max_tokens=32,
        preprocess_batch_size=64,
    )
    pages = [
        {"url": "https://a.example/1", "url_host_name": "a.example", "html": "<html>a</html>"},
        {"url": "https://a.example/2", "url_host_name": "a.example", "html": "<html>a</html>"},
        {"url": "https://b.example/1", "url_host_name": "b.example", "html": "<html>b</html>"},
    ]

    class FakeStage:
        def setup(self) -> None:
            return None

        def process(self, batch):
            df = batch.to_pandas().copy()
            df[estimator.PROMPT_COL] = ["same prompt", "same prompt", "other prompt"]
            df[estimator.NEEDS_LLM_COL] = [True, True, True]
            df[estimator.EMPTY_INPUT_COL] = [False, False, False]
            df[estimator.PRIMARY_ERROR_COL] = ["", "", ""]
            df["dripper_warning"] = ["", "", ""]
            df["dripper_item_count"] = [3, 3, 4]
            df["dripper_prompt_chars"] = [11, 11, 12]
            df["dripper_request_max_tokens"] = [128, 128, 128]
            return SimpleNamespace(to_pandas=lambda: df)

    fake_dripper_module = ModuleType("nemo_curator.stages.text.experimental.dripper")
    fake_dripper_module.DripperHTMLPreprocessStage = lambda **_kwargs: FakeStage()  # type: ignore[attr-defined]
    fake_llm_module = ModuleType("nemo_curator.models.client.llm_client")
    fake_llm_module.GenerationConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    fake_tasks_module = ModuleType("nemo_curator.tasks")

    class FakeDocumentBatch:
        def __init__(self, *, data, **_kwargs) -> None:
            self._data = data

        def to_pandas(self):
            return self._data

    fake_tasks_module.DocumentBatch = FakeDocumentBatch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_curator.stages.text.experimental.dripper", fake_dripper_module)
    monkeypatch.setitem(sys.modules, "nemo_curator.models.client.llm_client", fake_llm_module)
    monkeypatch.setitem(sys.modules, "nemo_curator.tasks", fake_tasks_module)

    row_df, metrics = estimator.preprocess_and_hash_pages(pages, args=args)

    assert metrics["needs_llm_pages"] == 3
    assert metrics["unique_prompt_requests"] == 2
    assert metrics["exact_prompt_saved_pages"] == 1
    assert metrics["exact_prompt_reduction_factor"] == 1.5
    assert "same prompt" not in row_df.to_json()
    assert row_df["prompt_hash"].str.len().tolist() == [64, 64, 64]


def test_prompt_dedup_sample_output_is_runnable_manifest_without_prompt_text() -> None:
    estimator = load_dripper_cc_module("prompt_dedup_estimator_sample_output", "estimate_prompt_dedup_call_reduction.py")
    processed_df = pd.DataFrame(
        [
            {
                "url": "https://a.example/1",
                "url_host_name": "a.example",
                "warc_filename": "crawl-data/CC-MAIN-2025-26/example.warc.gz",
                "warc_record_offset": 10,
                "warc_record_length": 20,
                "html": b"<html>one</html>",
                estimator.PROMPT_COL: "do not persist this prompt",
                "dripper_prompt_chars": 26,
            }
        ]
    )
    row_df = pd.DataFrame(
        [
            {
                "row_index": 0,
                "url": "https://a.example/1",
                "url_host_name": "a.example",
                "needs_llm": True,
                "prompt_hash": "a" * 64,
                "request_key": f"{'a' * 64}:128",
            }
        ]
    )

    sample_df = estimator.build_sample_output_dataframe(processed_df, row_df)

    assert "html" in sample_df.columns
    assert {"warc_filename", "warc_record_offset", "warc_record_length"}.issubset(sample_df.columns)
    assert estimator.PROMPT_COL not in sample_df.columns
    assert "do not persist this prompt" not in sample_df.to_json()
    assert sample_df["prompt_hash"].tolist() == ["a" * 64]
    assert sample_df["prompt_dedup_url"].tolist() == ["https://a.example/1"]


def test_prompt_dedup_estimator_layout_call_reduction(monkeypatch) -> None:
    estimator = load_dripper_cc_module("prompt_dedup_estimator_layout", "estimate_prompt_dedup_call_reduction.py")

    html_layout_module = ModuleType("llm_web_kit.html_layout.html_layout_cosin")
    typical_module = ModuleType("llm_web_kit.main_html_parser.typical_html.typical_html")

    def fake_get_feature(html):
        text = html.decode("utf-8") if isinstance(html, bytes) else str(html)
        return {"layout": text.split(":", 1)[0]}

    def fake_cluster_html_struct(samples, threshold):
        by_layout: dict[str, list[dict[str, object]]] = {}
        for sample in samples:
            by_layout.setdefault(sample["feature"]["layout"], []).append(sample)
        layout_ids = {
            layout: layout_index
            for layout_index, (layout, members) in enumerate(sorted(by_layout.items()))
            if len(members) >= 2
        }
        out = []
        for sample in samples:
            copied = dict(sample)
            copied["layout_id"] = layout_ids.get(sample["feature"]["layout"], -1)
            out.append(copied)
        return out, sorted(set(layout_ids.values()))

    def fake_select_representative_html(candidates):
        return sorted(candidates, key=lambda item: item["track_id"])[0]

    html_layout_module.get_feature = fake_get_feature  # type: ignore[attr-defined]
    html_layout_module.cluster_html_struct = fake_cluster_html_struct  # type: ignore[attr-defined]
    typical_module.select_representative_html = fake_select_representative_html  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "llm_web_kit", ModuleType("llm_web_kit"))
    monkeypatch.setitem(sys.modules, "llm_web_kit.html_layout", ModuleType("llm_web_kit.html_layout"))
    monkeypatch.setitem(sys.modules, "llm_web_kit.html_layout.html_layout_cosin", html_layout_module)
    monkeypatch.setitem(sys.modules, "llm_web_kit.main_html_parser", ModuleType("llm_web_kit.main_html_parser"))
    monkeypatch.setitem(
        sys.modules,
        "llm_web_kit.main_html_parser.typical_html",
        ModuleType("llm_web_kit.main_html_parser.typical_html"),
    )
    monkeypatch.setitem(sys.modules, "llm_web_kit.main_html_parser.typical_html.typical_html", typical_module)

    processed_df = pd.DataFrame(
        [
            {"url": "https://a.example/1", "url_host_name": "a.example", "html": "blog:one"},
            {"url": "https://a.example/2", "url_host_name": "a.example", "html": "blog:two"},
            {"url": "https://a.example/3", "url_host_name": "a.example", "html": "single:three"},
            {"url": "https://b.example/1", "url_host_name": "b.example", "html": "profile:one"},
            {"url": "https://b.example/2", "url_host_name": "b.example", "html": "profile:two"},
        ]
    )
    row_df = pd.DataFrame(
        [
            {"row_index": 0, "needs_llm": True, "request_key": "p0:128"},
            {"row_index": 1, "needs_llm": True, "request_key": "p1:128"},
            {"row_index": 2, "needs_llm": True, "request_key": "p2:128"},
            {"row_index": 3, "needs_llm": True, "request_key": "q:128"},
            {"row_index": 4, "needs_llm": True, "request_key": "q:128"},
        ]
    )
    args = SimpleNamespace(
        layout_cluster_threshold=0.95,
        layout_min_cluster_size=2,
        layout_max_exact_host_pages=100,
        top_layout_clusters=10,
    )

    metrics = estimator.estimate_layout_cluster_calls(processed_df, row_df, args=args)

    assert metrics["needs_llm_pages"] == 5
    assert metrics["feature_ok_pages"] == 5
    assert metrics["layout_cluster_count"] == 2
    assert metrics["layout_clustered_pages"] == 4
    assert metrics["layout_representative_pages"] == 2
    assert metrics["unique_prompt_requests"] == 4
    assert metrics["estimated_llm_requests_with_layout"] == 3
    assert metrics["layout_additional_saved_vs_exact_prompt_requests"] == 1


def make_index_row(
    url: str,
    host: str,
    status: int,
    mime_type: str,
    offset: int,
    length: int,
) -> dict[str, object]:
    return {
        "url": url,
        "url_host_name": host,
        "fetch_status": status,
        "content_mime_type": mime_type,
        "content_mime_detected": mime_type,
        "content_languages": "eng",
        "warc_filename": "crawl-data/CC-MAIN-2025-26/example.warc.gz",
        "warc_record_offset": offset,
        "warc_record_length": length,
    }


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
