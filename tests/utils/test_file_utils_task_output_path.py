from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_curator.utils.file_utils import resolve_sidecar_output_path, resolve_task_scoped_output_path

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_task_scoped_output_path_local_file_base(tmp_path: Path) -> None:
    base = str(tmp_path / "out.parquet")
    out = resolve_task_scoped_output_path(base, "task/0", "parquet")
    assert out == str(tmp_path / "out.task_0.parquet")


def test_resolve_task_scoped_output_path_local_dir_base(tmp_path: Path) -> None:
    base = str(tmp_path / "out_dir")
    out = resolve_task_scoped_output_path(base, "task/1", "arrow")
    assert out == str(tmp_path / "out_dir" / "task_1.arrow")


def test_resolve_task_scoped_output_path_remote_file_base() -> None:
    out = resolve_task_scoped_output_path("memory://bucket/out.parquet", "task/2", "parquet")
    assert out == "memory://bucket/out.task_2.parquet"


def test_resolve_task_scoped_output_path_remote_dir_base() -> None:
    out = resolve_task_scoped_output_path("memory://bucket/prefix/", "task/3", "tar")
    assert out == "memory://bucket/prefix/task_3.tar"


def test_resolve_sidecar_output_path_local() -> None:
    out = resolve_sidecar_output_path("out.task_0.parquet", "metadata", "parquet")
    assert out == "out.task_0.metadata.parquet"


def test_resolve_sidecar_output_path_remote() -> None:
    out = resolve_sidecar_output_path("memory://bucket/out.task_0.arrow", "metadata", "arrow")
    assert out == "memory://bucket/out.task_0.metadata.arrow"
