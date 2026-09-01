# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarking.scripts.nmcur357_cohort import (
    DOCUMENT_ID_FIELD,
    ENGLISH,
    LANGUAGE_FIELD,
    NON_SPACED_LANGUAGES,
    OUTPUT_FIELDS,
    document_hash,
    document_id,
    run,
)


def _write_source(path: Path, languages: list[str]) -> None:
    rows = len(languages)
    pq.write_table(
        pa.table(
            {
                "_html_zstd": [f"html-{i}".encode() for i in range(rows)],
                "_html_uncompressed_bytes": [100 + i for i in range(rows)],
                "url": [f"https://example.com/{i}" for i in range(rows)],
                "justext_extracted_text": [f"text-{i}" for i in range(rows)],
                LANGUAGE_FIELD: languages,
                "unrelated": list(range(rows)),
            }
        ),
        path,
    )


def test_materializes_exact_deterministic_cohort_and_canary(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    canary = tmp_path / "canary"
    source.mkdir()
    languages = list(NON_SPACED_LANGUAGES) * 2 + [ENGLISH] * 5 + ["FRENCH"]
    _write_source(source / "a.parquet", languages)
    _write_source(source / "b.parquet", list(reversed(languages)))

    manifest_path = tmp_path / "manifest.json"
    manifest = run(
        Namespace(
            input_root=source,
            output_root=output,
            canary_root=canary,
            manifest_path=manifest_path,
            english_size=3,
            canary_per_language=1,
            resume=True,
        )
    )

    full = pa.concat_tables([pq.read_table(path) for path in sorted(output.glob("*.parquet"))])
    canary_table = pa.concat_tables([pq.read_table(path) for path in sorted(canary.glob("*.parquet"))])
    assert full.column_names == list(OUTPUT_FIELDS)
    assert len(full) == len(NON_SPACED_LANGUAGES) * 4 + 3
    assert len(canary_table) == len(NON_SPACED_LANGUAGES) + 1
    assert set(full[LANGUAGE_FIELD].to_pylist()) == {*NON_SPACED_LANGUAGES, ENGLISH}
    assert len(set(full[DOCUMENT_ID_FIELD].to_pylist())) == len(full)
    assert manifest["full"]["rows"] == len(full)
    assert (output / "_SUCCESS.json").is_file()

    all_english = [
        document_id(path.name, i)
        for path in sorted(source.glob("*.parquet"))
        for i, language in enumerate(pq.read_table(path, columns=[LANGUAGE_FIELD]).column(0).to_pylist())
        if language == ENGLISH
    ]
    expected = set(sorted(all_english, key=document_hash)[:3])
    actual = {
        doc_id
        for doc_id, language in zip(full[DOCUMENT_ID_FIELD].to_pylist(), full[LANGUAGE_FIELD].to_pylist(), strict=True)
        if language == ENGLISH
    }
    assert actual == expected


def test_resume_keeps_valid_output_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    canary = tmp_path / "canary"
    source.mkdir()
    _write_source(source / "a.parquet", [*NON_SPACED_LANGUAGES, ENGLISH, ENGLISH])
    args = Namespace(
        input_root=source,
        output_root=output,
        canary_root=canary,
        manifest_path=tmp_path / "manifest.json",
        english_size=1,
        canary_per_language=1,
        resume=True,
    )
    run(args)
    target = output / "a.parquet"
    first_mtime = target.stat().st_mtime_ns
    run(args)
    assert target.stat().st_mtime_ns == first_mtime
