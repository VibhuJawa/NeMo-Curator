# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"))

import nmcur357_analysis as analysis


def test_normalize_text_handles_entities_compatibility_case_and_whitespace() -> None:
    assert analysis.normalize_text("\uff21 &amp; B\n") == "a&b"


def test_document_metrics_attributes_every_occurrence() -> None:
    result = analysis.document_metrics(
        {
            "document_id": "doc",
            "url": "https://example.test",
            "html_cld2_lang": "JAPANESE",
            "_mineru_status": "ok",
            "justext_extracted_text": "abcde",
            "mineru_main_text": "abc",
            "mineru_other_text": "bcd",
        },
        shingle_width=2,
    )

    assert result["justext_shingle_occurrences"] == 4
    assert result["justext_main_occurrences"] == 1
    assert result["justext_ambiguous_occurrences"] == 1
    assert result["justext_other_occurrences"] == 1
    assert result["justext_unmatched_occurrences"] == 1
    assert result["mineru_other_retention_by_justext"] == 1.0
    assert result["excess_justext_length_explained_by_other"] == 0.5


def test_non_ok_rows_are_excluded_from_primary_metrics() -> None:
    result = analysis.document_metrics(
        {
            "document_id": "doc",
            "url": "",
            "html_cld2_lang": "THAI",
            "_mineru_status": "inference_error",
        }
    )

    assert result["justext_boilerplate_share"] is None
    assert result["justext_shingle_occurrences"] is None


def test_poisson_bootstrap_is_deterministic() -> None:
    rows = []
    for index in range(20):
        row = {
            "_mineru_status": "ok",
            **{
                column: index + offset + 1
                for offset, column in enumerate({c for p in analysis.METRIC_PAIRS.values() for c in p})
            },
        }
        rows.append(row)
    frame = pd.DataFrame(rows)

    first = analysis.poisson_bootstrap_intervals(frame, replicates=25, seed=357, chunk_size=7)
    second = analysis.poisson_bootstrap_intervals(frame, replicates=25, seed=357, chunk_size=7)

    assert first == second
    assert all(values["low"] <= values["high"] for values in first.values())


def test_population_weighting_excludes_language_without_source_justext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "NON_SPACED_LANGUAGES", ("CHINESE", "CHINESET"))
    groups = {
        "CHINESE": {
            "primary_documents": 75,
            "metrics": {
                "justext_boilerplate_share": {"micro": 0.4, "denominator": 100},
            },
        },
        "CHINESET": {
            "primary_documents": 25,
            "metrics": {
                "justext_boilerplate_share": {"micro": None, "denominator": 0},
            },
        },
    }

    weighted, coverage = analysis.population_weighted_language_metric(groups, "justext_boilerplate_share")

    assert weighted == 0.4
    assert coverage == 0.75


def test_summarize_group_marks_all_metrics_unavailable_without_source_justext() -> None:
    row = {
        "_mineru_status": "ok",
        **{column: 0 for pair in analysis.METRIC_PAIRS.values() for column in pair},
        **dict.fromkeys(analysis.METRIC_PAIRS, 0.0),
    }
    row["mineru_other_shingle_occurrences"] = 10

    result = analysis.summarize_group(pd.DataFrame([row]), bootstrap_replicates=5, seed=357)

    assert all(values["micro"] is None for values in result["metrics"].values())
    assert all(values["document_median"] is None for values in result["metrics"].values())


def _write_tiny_input(path: Path) -> None:
    compressor = zstd.ZstdCompressor()
    raw = b"<html>ok</html>"
    pq.write_table(
        pa.table(
            {
                "_html_zstd": [compressor.compress(raw)],
                "_html_uncompressed_bytes": [len(raw)],
                "url": ["https://example.test"],
                "justext_extracted_text": ["ok"],
                "html_cld2_lang": ["JAPANESE"],
                "source_file": ["source.parquet"],
                "source_row_index": [0],
                "document_id": ["source.parquet:0"],
            }
        ),
        path,
    )


def _write_tiny_output(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "document_id": ["source.parquet:0"],
                "url": ["https://example.test"],
                "justext_extracted_text": ["ok"],
                "html_cld2_lang": ["JAPANESE"],
                "mineru_main_text": ["ok"],
                "mineru_other_text": [""],
                "mineru_labels": ['{"1":"main"}'],
                "_mineru_status": ["ok"],
            }
        ),
        path,
    )


def test_validate_run_checks_ids_counts_status_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.mkdir()
    output_path.mkdir()
    checkpoint_path.mkdir()
    _write_tiny_input(input_path / "source.parquet")
    _write_tiny_output(output_path / "result.parquet")
    monkeypatch.setattr(analysis, "EXPECTED_LANGUAGE_COUNTS", {"JAPANESE": 1})
    monkeypatch.setattr(analysis, "EXPECTED_TOTAL", 1)
    monkeypatch.setattr(analysis, "checkpoint_completed_sources", lambda _: {"source"})
    monkeypatch.setattr(analysis, "validate_zstandard_sample", lambda _: 1)

    validation = analysis.validate_run(input_path, output_path, checkpoint_path)

    assert validation["document_id_sets_identical"] is True
    assert validation["status_ok_rate"] == 1.0
    assert validation["pending_checkpoint_sources"] == 0


def test_write_per_document_metrics_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_tiny_output(output / "result.parquet")
    destination = tmp_path / "analysis" / "per_document.parquet"

    assert analysis.write_per_document_metrics(output, destination, batch_size=1) == 1
    assert pd.read_parquet(destination)["justext_main_share"].iloc[0] == 1.0
    assert not list(destination.parent.glob(".*.tmp"))


def test_validated_per_document_rows_accepts_durable_artifact(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_tiny_output(output / "result.parquet")
    destination = tmp_path / "per_document.parquet"
    analysis.write_per_document_metrics(output, destination, batch_size=1)

    assert analysis.validated_per_document_rows(destination) == 1


def test_review_selection_handles_language_without_justext_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "LANGUAGES", ("CHINESET", "JAPANESE"))
    frame = pd.DataFrame(
        [
            {
                "document_id": f"chineset-{index}",
                "html_cld2_lang": "CHINESET",
                "_mineru_status": "ok",
                "justext_boilerplate_share": None,
            }
            for index in range(60)
        ]
        + [
            {
                "document_id": f"japanese-{index}",
                "html_cld2_lang": "JAPANESE",
                "_mineru_status": "ok",
                "justext_boilerplate_share": index / 59,
            }
            for index in range(60)
        ]
    )

    selected = analysis._review_selection(frame)
    chineset = [value for value in selected.values() if value[0] == "CHINESET"]
    japanese = [value for value in selected.values() if value[0] == "JAPANESE"]

    assert len(chineset) == 50
    assert all(decile is None and ratio is None for _, decile, ratio in chineset)
    assert len(japanese) == 50
    assert sorted(decile for _, decile, _ in japanese) == [decile for decile in range(10) for _ in range(5)]


def test_per_document_schema_survives_null_only_first_batch(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    pq.write_table(
        pa.table(
            {
                "document_id": ["failed", "ok"],
                "url": ["", ""],
                "justext_extracted_text": ["", "ok"],
                "html_cld2_lang": ["JAPANESE", "JAPANESE"],
                "mineru_main_text": ["", "ok"],
                "mineru_other_text": ["", ""],
                "mineru_labels": ["{}", "{}"],
                "_mineru_status": ["inference_error", "ok"],
            }
        ),
        output / "result.parquet",
    )
    destination = tmp_path / "per_document.parquet"

    assert analysis.write_per_document_metrics(output, destination, batch_size=1) == 2
    result = pd.read_parquet(destination)
    assert pd.isna(result.loc[0, "justext_main_share"])
    assert result.loc[1, "justext_main_share"] == 1.0
