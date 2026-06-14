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

"""run_pipeline.py — Single-command Dripper CC clustering pipeline orchestrator.

Usage:
    python run_pipeline.py --config configs/template.yaml
    python run_pipeline.py --config configs/template.yaml --dry-run
    python run_pipeline.py --config configs/template.yaml --resume
    python run_pipeline.py --config configs/template.yaml --snapshots CC-MAIN-2025-26

Pipeline stages (per shard, streaming via aftercorr):
    Stage 1a  CPU  DOM feature extraction   (RayActorPoolExecutor, 64 workers)
    Stage 1b  GPU  DBSCAN clustering        (cuML, HostDBSCANStage)
    GPU        GPU  vLLM inference 1c+2+2b  (kv-fp8, 8×H100)
    Stage 3   CPU  LBP propagation          (PPT=16, HTML-size sort)

Post-processing (afterok on all stage-3 shards):
    Validation   CPU  F1 sample check against reference baseline
    Stage 3b     GPU  Fallback GPU inference for over-extracted siblings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # fallback for environments without PyYAML
    yaml = None  # type: ignore[assignment]

from configs.dripper_config import DripperConfig  # typed config dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_STAGES = ("stage1a", "stage1b", "gpu_pipeline", "stage3", "stage3b_build", "stage3b_gpu", "stage3b_merge")


@dataclass
class ClusterConfig:
    login_node: str
    dc_node: str
    account: str
    venv: str
    cached_venv: str
    hf_cache: str
    remote_repo: str

    @property
    def script_dir(self) -> str:
        return f"{self.remote_repo}/tutorials/text/dripper-common-crawl"

    @property
    def curator_root(self) -> str:
        return self.remote_repo

    @property
    def python_cpu(self) -> str:
        return f"{self.venv}/bin/python3"

    @property
    def python_gpu(self) -> str:
        return f"{self.venv}/bin/python3"


@dataclass
class SnapshotRun:
    name: str
    manifest: str
    validation_baseline: str
    output_base: str  # fully expanded output root
    cluster: ClusterConfig
    sharding: dict[str, int]
    resources: dict[str, Any]
    validation: dict[str, Any]

    @property
    def stage1a_dir(self) -> str:
        return f"{self.output_base}/stage1a"

    @property
    def stage1b_dir(self) -> str:
        return f"{self.output_base}/stage1b"

    @property
    def gpu_dir(self) -> str:
        return f"{self.output_base}/stage2b"

    @property
    def stage3_dir(self) -> str:
        return f"{self.output_base}/stage3"

    @property
    def stage3b_dir(self) -> str:
        return f"{self.output_base}/stage3b"

    @property
    def logs_dir(self) -> str:
        return f"{self.output_base}/logs"

    @property
    def sbatch_dir(self) -> str:
        return f"{self.output_base}/sbatch"

    @property
    def num_shards(self) -> int:
        return self.sharding["num_shards"]

    @property
    def gpu_shards(self) -> int:
        return self.sharding["gpu_pipeline_shards"]


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = f.read()
    if yaml is not None:
        return yaml.safe_load(raw)
    # Minimal YAML subset parser for environments without PyYAML (dry-run on Mac)

    def _parse_yaml_minimal(_text: str) -> dict:
        msg = "PyYAML not available. Install with: pip install pyyaml"
        raise RuntimeError(msg)

    return _parse_yaml_minimal(raw)


def build_snapshot_run(snap_entry: dict, cfg: dict, ts: str) -> SnapshotRun:
    name = snap_entry["name"]
    output_base = cfg["output_base"].format(snapshot=name.replace("-", "_").lower(), ts=ts)
    return SnapshotRun(
        name=name,
        manifest=snap_entry["manifest"],
        validation_baseline=snap_entry.get("validation_baseline", ""),
        output_base=output_base,
        cluster=ClusterConfig(**cfg["cluster"]),
        sharding=cfg["sharding"],
        resources=cfg["resources"],
        validation=cfg["validation"],
    )


# ---------------------------------------------------------------------------
# SSH / remote helpers
# ---------------------------------------------------------------------------

_SSH_OPTS = ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/.ssh_ctl_%h_%p_%r", "-o", "ControlPersist=60s"]


def _ssh(node: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *_SSH_OPTS, node, cmd], capture_output=True, text=True, check=check)


def _rsync(local: str, remote_node: str, remote_path: str) -> None:
    subprocess.run(["rsync", "-av", local, f"{remote_node}:{remote_path}"], check=True)


def _remote_mkdir(node: str, *paths: str) -> None:
    _ssh(node, "mkdir -p " + " ".join(f'"{p}"' for p in paths))


def _remote_file_nonempty(node: str, path: str) -> bool:
    """Return True if a parquet file exists on the remote node with >0 rows."""
    cmd = (
        f'python3 -c "import pyarrow.parquet as pq, sys; '
        f"m=pq.read_metadata('{path}'); sys.exit(0 if m.num_rows>0 else 1)\" 2>/dev/null"
    )
    return _ssh(node, cmd, check=False).returncode == 0


def _remote_write(_node: str, dc_node: str, content: str, remote_path: str) -> None:
    """Write text content to a remote file via a temp file + rsync."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(content)
        local_tmp = f.name
    try:
        _rsync(local_tmp, dc_node, remote_path)
    finally:
        os.unlink(local_tmp)


# ---------------------------------------------------------------------------
# Resume checker
# ---------------------------------------------------------------------------


class ResumeChecker:
    def __init__(self, snap: SnapshotRun) -> None:
        self.snap = snap
        self._cache: dict[tuple, bool] = {}

    def shard_done(self, stage: str, shard: int) -> bool:
        key = (stage, shard)
        if key not in self._cache:
            outdir = getattr(self.snap, f"{stage}_dir", None) or self.snap.stage3b_dir
            path = f"{outdir}/shard_{shard:04d}.parquet"
            self._cache[key] = _remote_file_nonempty(self.snap.cluster.login_node, path)
        return self._cache[key]

    def all_shards_done(self, stage: str, n: int) -> bool:
        with ThreadPoolExecutor(max_workers=min(32, n)) as ex:
            futs = {ex.submit(self.shard_done, stage, s): s for s in range(n)}
            return all(f.result() for f in as_completed(futs))

    def global_done(self, sentinel_file: str) -> bool:
        return _remote_file_nonempty(self.snap.cluster.login_node, sentinel_file)


# ---------------------------------------------------------------------------
# sbatch script builders
# ---------------------------------------------------------------------------


def _sbatch_header(job_name: str, res: dict, array: str | None, logs_dir: str, account: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={res['partition']}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={res.get('cpus', 8)}",
        f"#SBATCH --mem={res.get('mem', '32G')}",
        f"#SBATCH --time={res.get('time', '01:00:00')}",
    ]
    if res.get("gpus_per_node"):
        lines.append(f"#SBATCH --gpus-per-node={res['gpus_per_node']}")
    if array:
        lines += [
            f"#SBATCH --array={array}",
            f"#SBATCH --output={logs_dir}/{job_name}_%04a_%j.out",
            f"#SBATCH --error={logs_dir}/{job_name}_%04a_%j.err",
        ]
    else:
        lines += [
            f"#SBATCH --output={logs_dir}/{job_name}_%j.out",
            f"#SBATCH --error={logs_dir}/{job_name}_%j.err",
        ]
    return "\n".join(lines)


def _env_setup(snap: SnapshotRun, gpu: bool = False) -> str:
    c = snap.cluster
    env = textwrap.dedent(f"""
        set -eu
        export PYTHONPATH='{c.script_dir}:{c.curator_root}:${{PYTHONPATH:-}}'
        export RAY_TMPDIR=/tmp
        export HF_HOME='{c.hf_cache}'
        export TRANSFORMERS_CACHE='{c.hf_cache}'
    """).strip()
    if gpu:
        env += textwrap.dedent(f"""
            for _d in '{c.cached_venv}'/lib/python3.12/site-packages/nvidia/*/lib \\
                      '{c.cached_venv}'/lib/python3.12/site-packages/cuml/*/lib; do
              [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${{LD_LIBRARY_PATH:-}}"
            done
        """).strip()
    return env


def sbatch_stage1a(snap: SnapshotRun) -> str:
    c, r = snap.cluster, snap.resources["stage1a"]
    last = snap.num_shards - 1
    header = _sbatch_header("s1a", r, f"0-{last}", snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap)
        + f"""
echo "=== Stage1a shard ${{SLURM_ARRAY_TASK_ID}}/{last} ==="
{c.python_cpu} '{c.script_dir}/stage1a_feature_extraction.py' \\
  --manifest-dir  '{snap.manifest}' \\
  --output-dir    '{snap.stage1a_dir}' \\
  --shard-index   ${{SLURM_ARRAY_TASK_ID}} \\
  --num-shards    {snap.num_shards} \\
  --cpus-per-actor {r.get("cpus_per_actor", 1)}
"""
    )


def sbatch_stage1b(snap: SnapshotRun) -> str:
    c, r = snap.cluster, snap.resources["stage1b"]
    last = snap.num_shards - 1
    header = _sbatch_header("s1b", r, f"0-{last}", snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap, gpu=True)
        + f"""
echo "=== Stage1b shard ${{SLURM_ARRAY_TASK_ID}}/{last} ==="
{c.python_gpu} '{c.script_dir}/stage1b_gpu_dbscan.py' \\
  --input-dir     '{snap.stage1a_dir}' \\
  --output-dir    '{snap.stage1b_dir}' \\
  --shard-index   ${{SLURM_ARRAY_TASK_ID}} \\
  --num-shards    {snap.num_shards} \\
  --batch-size    {r.get("batch_size", 16)} \\
  --gpu-min-size  {r.get("gpu_min_size", 5)}
"""
    )


def sbatch_gpu_pipeline(snap: SnapshotRun) -> str:
    c, r = snap.cluster, snap.resources["gpu_pipeline"]
    last = snap.gpu_shards - 1
    header = _sbatch_header("s-gpu", r, f"0-{last}", snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap, gpu=True)
        + f"""
echo "=== GPU pipeline shard ${{SLURM_ARRAY_TASK_ID}}/{last} ==="
{c.python_gpu} '{c.script_dir}/stage_gpu_pipeline.py' \\
  --input      '{snap.stage1b_dir}' \\
  --output     '{snap.gpu_dir}' \\
  --shard-index ${{SLURM_ARRAY_TASK_ID}} \\
  --num-shards {snap.gpu_shards} \\
  --model      '{r["model"]}' \\
  --hf-cache   '{c.hf_cache}' \\
  --kv-cache-dtype {r.get("kv_cache_dtype", "fp8")} \\
  --max-tokens {r.get("max_tokens", 2048)} \\
  --gpu-mem-util {r.get("gpu_mem_util", 0.90)} \\
  --max-model-len {r.get("max_model_len", 32768)} \\
  --max-num-seqs {r.get("max_num_seqs", 512)} \\
  --max-num-batched-tokens {r.get("max_num_batched_tokens", 16384)}
"""
    )


def sbatch_stage3(snap: SnapshotRun) -> str:
    c, r = snap.cluster, snap.resources["stage3"]
    last = snap.num_shards - 1
    header = _sbatch_header("s3", r, f"0-{last}", snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap)
        + f"""
echo "=== Stage3 shard ${{SLURM_ARRAY_TASK_ID}}/{last} ==="
{c.python_cpu} '{c.script_dir}/stage3_cpu_propagation.py' \\
  --cluster-manifest  '{snap.stage1b_dir}' \\
  --inference-results '{snap.gpu_dir}' \\
  --output-dir        '{snap.stage3_dir}' \\
  --shard-index       ${{SLURM_ARRAY_TASK_ID}} \\
  --num-shards        {snap.num_shards} \\
  --num-workers       {r.get("num_workers", 64)}
"""
    )


def sbatch_stage3b_build(snap: SnapshotRun) -> str:
    c, r = snap.cluster, snap.resources["stage3b_build"]
    header = _sbatch_header("s3b-build", r, None, snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap)
        + f"""
echo "=== Stage3b build ==="
{c.python_cpu} '{c.script_dir}/stage3b_fallback_llm.py' \\
  --mode    build \\
  --stage3  '{snap.stage3_dir}' \\
  --stage1b '{snap.stage1b_dir}' \\
  --output  '{snap.stage3b_dir}/build_output'
"""
    )


def sbatch_stage3b_gpu(snap: SnapshotRun) -> str:
    c, r = snap.cluster, snap.resources["stage3b_gpu"]
    header = _sbatch_header("s3b-gpu", r, None, snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap, gpu=True)
        + f"""
echo "=== Stage3b GPU inference ==="
{c.python_gpu} '{c.script_dir}/stage_gpu_pipeline.py' \\
  --input     '{snap.stage3b_dir}/build_output/shard_0000.parquet' \\
  --output    '{snap.stage3b_dir}/gpu_output' \\
  --model     '{r.get("model", snap.resources["gpu_pipeline"]["model"])}' \\
  --hf-cache  '{c.hf_cache}' \\
  --kv-cache-dtype {snap.resources["gpu_pipeline"].get("kv_cache_dtype", "fp8")}
"""
    )


def sbatch_stage3b_merge(snap: SnapshotRun, final_f1_script: str) -> str:
    c, r = snap.cluster, snap.resources["stage3b_merge"]
    header = _sbatch_header("s3b-merge", r, None, snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap)
        + f"""
echo "=== Stage3b merge ==="
{c.python_cpu} '{c.script_dir}/stage3b_fallback_llm.py' \\
  --mode             merge \\
  --stage3           '{snap.stage3_dir}' \\
  --fallback-stage2b '{snap.stage3b_dir}/gpu_output' \\
  --output           '{snap.stage3b_dir}/merged'
{final_f1_script}
"""
    )


def sbatch_validation(snap: SnapshotRun, downstream_job_ids: list[str]) -> str:
    c, r = snap.cluster, snap.resources["validation"]
    cfg = snap.validation
    baseline = snap.validation_baseline
    pipeline = snap.stage3_dir
    threshold = cfg["f1_threshold"]
    sample_size = cfg.get("sample_size", 10000)
    halt = str(cfg.get("halt_on_failure", False)).lower()
    downstream_str = " ".join(downstream_job_ids)
    header = _sbatch_header("s-validate", r, None, snap.logs_dir, c.account)
    return (
        header
        + "\n"
        + _env_setup(snap)
        + f"""
echo "=== Validation: F1 sample check ==="
{c.python_cpu} - << 'PYEOF'
import re, sys, pathlib, subprocess
import pyarrow.parquet as pq, pandas as pd, glob, random

# --- sample {sample_size} common URLs ---
bl = pq.read_table('{baseline}', columns=['url']).to_pandas()
s3_files = sorted(glob.glob('{pipeline}/shard_*.parquet'))
if not s3_files:
    print("No stage3 parquets found, skipping validation")
    sys.exit(0)
pipe = pd.concat([pq.read_table(f, columns=['url']).to_pandas() for f in s3_files[:10]])
common = list(set(bl['url']) & set(pipe['url']))
sample_urls = set(random.sample(common, min({sample_size}, len(common))))

# --- write sampled parquet ---
sample_dir = pathlib.Path('{snap.stage3b_dir}/val_sample')
sample_dir.mkdir(parents=True, exist_ok=True)
sample_path = str(sample_dir / 'sample.parquet')
s3_full = pd.concat([pq.read_table(f).to_pandas() for f in s3_files])
s3_full[s3_full['url'].isin(sample_urls)].to_parquet(sample_path, index=False)
print(f"Validation sample: {{len(sample_urls)}} URLs written to {{sample_path}}", flush=True)
PYEOF

{c.python_cpu} '{c.script_dir}/compare_f1.py' \\
  --pipeline  '{snap.stage3b_dir}/val_sample' \\
  --baseline  '{baseline}' \\
  --baseline-col dripper_content \\
  --pipeline-col dripper_content 2>&1 | tee '{snap.logs_dir}/f1_validation.txt'

{c.python_cpu} - << 'PYEOF'
import re, sys, pathlib, subprocess
report = pathlib.Path('{snap.logs_dir}/f1_validation.txt').read_text()
m = re.search(r"mean F1:[\\s]+([\\d.]+)", report)
if not m:
    print("[validate] could not parse F1 - skipping threshold check")
    sys.exit(0)
mean_f1 = float(m.group(1))
threshold = {threshold}
passed = mean_f1 >= threshold
print(f"[validate] mean F1={{mean_f1:.4f}}  threshold={{threshold}}  passed={{passed}}", flush=True)
pathlib.Path('{snap.logs_dir}/f1_result.json').write_text(
    f'{{"mean_f1": {{mean_f1}}, "threshold": {{threshold}}, "passed": {{str(passed).lower()}}}}'
)
if not passed and {halt}:
    print(f"[validate] HALTING downstream jobs: {downstream_str}", flush=True)
    subprocess.run(['scancel'] + '{downstream_str}'.split(), check=False)
    sys.exit(1)
sys.exit(0)
PYEOF
"""
    )


def _final_f1_script(snap: SnapshotRun) -> str:
    """Inline F1 compare after stage3b merge, if validation_baseline is set."""
    if not snap.validation_baseline:
        return ""
    c = snap.cluster
    return f"""
echo "=== Final F1: merged output vs baseline ==="
{c.python_cpu} '{c.script_dir}/compare_f1.py' \\
  --pipeline  '{snap.stage3b_dir}/merged' \\
  --baseline  '{snap.validation_baseline}' \\
  --baseline-col dripper_content --pipeline-col dripper_content
"""


# ---------------------------------------------------------------------------
# Slurm submitter
# ---------------------------------------------------------------------------


class SlurmSubmitter:
    def __init__(self, snap: SnapshotRun, dry_run: bool) -> None:
        self.snap = snap
        self.dry_run = dry_run
        self._counter = 0

    def submit(self, script_content: str, script_name: str, dependency: str | None = None) -> str | None:
        remote_path = f"{self.snap.sbatch_dir}/{script_name}"
        if not self.dry_run:
            _remote_write(
                self.snap.cluster.login_node,
                self.snap.cluster.dc_node,
                script_content,
                remote_path,
            )
            dep_flag = f"--dependency={dependency}" if dependency else ""
            cmd = f"sbatch --parsable {dep_flag} '{remote_path}'"
            result = _ssh(self.snap.cluster.login_node, cmd)
            job_id = result.stdout.strip()
            logger.info("[submit] %s → job %s  dep=%s", script_name, job_id, dependency or "none")
            return job_id
        else:
            self._counter += 1
            fake_id = f"DRY{self._counter:04d}"
            logger.info("[dry-run] %s → %s  dep=%s", script_name, fake_id, dependency or "none")
            return fake_id


# ---------------------------------------------------------------------------
# Resume-aware DAG builder
# ---------------------------------------------------------------------------


def _dep(*job_ids: str | None, mode: str = "aftercorr") -> str | None:
    """Build Slurm dependency string; None entries (already-done) are ignored."""
    valid = [j for j in job_ids if j is not None]
    if not valid:
        return None
    return f"{mode}:" + ":".join(valid)


def build_and_submit_dag(snap: SnapshotRun, submitter: SlurmSubmitter, resume: ResumeChecker) -> dict:
    """Submit all Slurm jobs for one snapshot. Returns map stage→job_id."""
    n, g = snap.num_shards, snap.gpu_shards

    def _skip_if_done(stage: str, n_shards: int) -> bool:
        if resume.all_shards_done(stage, n_shards):
            logger.info("[resume] %s: all %d shards complete, skipping", stage, n_shards)
            return True
        return False

    ids: dict[str, str | None] = {}

    # Stage 1a
    ids["stage1a"] = None if _skip_if_done("stage1a", n) else submitter.submit(sbatch_stage1a(snap), "stage1a.sh")

    # Stage 1b — aftercorr on stage1a (shard-level streaming)
    ids["stage1b"] = (
        None
        if _skip_if_done("stage1b", n)
        else submitter.submit(sbatch_stage1b(snap), "stage1b.sh", _dep(ids["stage1a"]))
    )

    # GPU pipeline — aftercorr on stage1b (different shard count; afterok for robustness)
    ids["gpu"] = (
        None
        if _skip_if_done("gpu_pipeline", g)
        else submitter.submit(sbatch_gpu_pipeline(snap), "gpu_pipeline.sh", _dep(ids["stage1b"], mode="afterok"))
    )

    # Stage 3 — aftercorr on stage1b (per-shard) + afterok on GPU (all shards needed)
    # Use the stricter afterok:stage1b:gpu when both still running;
    # if either is already done, use only the live one.
    s3_dep = _dep(ids["stage1b"]) if ids["gpu"] is None else _dep(ids["stage1b"], ids["gpu"], mode="afterok")
    ids["stage3"] = None if _skip_if_done("stage3", n) else submitter.submit(sbatch_stage3(snap), "stage3.sh", s3_dep)

    # Stage 3b build — afterok on ALL of stage3
    ids["s3b_build"] = submitter.submit(
        sbatch_stage3b_build(snap),
        "stage3b_build.sh",
        _dep(ids["stage3"], mode="afterok"),
    )

    # Stage 3b GPU — afterok on build
    ids["s3b_gpu"] = submitter.submit(
        sbatch_stage3b_gpu(snap),
        "stage3b_gpu.sh",
        _dep(ids["s3b_build"], mode="afterok"),
    )

    # Stage 3b merge — afterok on GPU (includes final F1 compare if baseline set)
    downstream = [v for k, v in ids.items() if v and k.startswith("s3b")]
    ids["s3b_merge"] = submitter.submit(
        sbatch_stage3b_merge(snap, _final_f1_script(snap)),
        "stage3b_merge.sh",
        _dep(ids["s3b_gpu"], mode="afterok"),
    )

    # Validation — afterok on ALL of stage3, parallel with stage3b
    if snap.validation["enabled"] and snap.validation_baseline:
        ids["validation"] = submitter.submit(
            sbatch_validation(snap, [v for v in downstream if v]),
            "validation.sh",
            _dep(ids["stage3"], mode="afterok"),
        )

    return ids


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


class PipelineRunner:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.ts = datetime.now(tz=None).strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005

    def run(self) -> None:
        snapshots = self.cfg["snapshots"]
        if self.args.snapshots:
            names = {s.strip() for s in self.args.snapshots.split(",")}
            snapshots = [s for s in snapshots if s["name"] in names]
        for entry in snapshots:
            snap = build_snapshot_run(entry, self.cfg, self.ts)
            self._run_snapshot(snap)

    def _run_snapshot(self, snap: SnapshotRun) -> None:
        logger.info("=== Snapshot: %s → %s ===", snap.name, snap.output_base)
        if not self.args.dry_run:
            self._prepare_remote(snap)
        resume = ResumeChecker(snap) if self.args.resume else _NullResumeChecker()
        submitter = SlurmSubmitter(snap, dry_run=self.args.dry_run)
        job_ids = build_and_submit_dag(snap, submitter, resume)
        if not self.args.dry_run:
            _ssh(
                snap.cluster.login_node,
                f"cat > '{snap.sbatch_dir}/job_ids.json' << 'EOF'\n{json.dumps(job_ids, indent=2)}\nEOF",
            )
        logger.info("Job IDs: %s", json.dumps(job_ids, indent=2))

    def _prepare_remote(self, snap: SnapshotRun) -> None:
        c = snap.cluster
        _remote_mkdir(
            c.login_node,
            snap.stage1a_dir,
            snap.stage1b_dir,
            snap.gpu_dir,
            snap.stage3_dir,
            snap.stage3b_dir,
            snap.logs_dir,
            snap.sbatch_dir,
        )
        # Sync latest stage scripts to cluster
        tutorial_dir = Path(__file__).parent
        for py_file in tutorial_dir.glob("stage*.py"):
            _rsync(str(py_file), c.dc_node, c.script_dir + "/" + py_file.name)
        _rsync(str(tutorial_dir / "compare_f1.py"), c.dc_node, c.script_dir + "/compare_f1.py")


class _NullResumeChecker:
    """No-op resume checker — always says nothing is complete."""

    def shard_done(self, *_a) -> bool:
        return False

    def all_shards_done(self, *_a) -> bool:
        return False

    def global_done(self, *_a) -> bool:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Dripper CC clustering pipeline.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    p.add_argument("--resume", action="store_true", help="Skip stages whose output already exists.")
    p.add_argument("--snapshots", default="", help="Comma-separated snapshot names to run (default: all).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    # DripperConfig.from_yaml validates required fields and provides typed access.
    # to_raw_dict() returns the same dict structure PipelineRunner has always expected,
    # so the migration is backward-compatible.
    dripper_cfg = DripperConfig.from_yaml(args.config)
    PipelineRunner(dripper_cfg.to_raw_dict(), args).run()


if __name__ == "__main__":
    main()
