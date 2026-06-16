#!/usr/bin/env bash
# Poll Nebius Slurm jobs and print metrics when done.
#
# Usage: bash poll_nebius_jobs.sh HOST JOB_ID,JOB_ID2 OUTPUT_DIR1 OUTPUT_DIR2
#
# Example:
#   bash poll_nebius_jobs.sh vjawa@nb-hel-cs-001-vscode-01.nvidia.com \
#     346402,346403 \
#     /lustre/.../dripper_streaming_smoke \
#     /lustre/.../dripper_streaming_shard0001

set -euo pipefail

HOST="${1:?Usage: $0 <host> <job_ids_csv> <output_dir1> [output_dir2]}"
JOB_IDS="${2:?}"
OUTPUT_DIR1="${3:?}"
OUTPUT_DIR2="${4:-}"

ssh "${HOST}" bash <<REMOTE
set -euo pipefail

echo "=== Job status ==="
squeue -j ${JOB_IDS} --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "(no jobs in queue — may be complete)"
echo ""
echo "=== Partition availability ==="
sinfo -p batch --summarize 2>/dev/null | head -6 || true

echo ""
echo "=== metrics: ${OUTPUT_DIR1} ==="
cat "${OUTPUT_DIR1}/metrics.json" 2>/dev/null || echo "not-done-yet"

if [ -n "${OUTPUT_DIR2}" ]; then
  echo ""
  echo "=== metrics: ${OUTPUT_DIR2} ==="
  cat "${OUTPUT_DIR2}/metrics.json" 2>/dev/null || echo "not-done-yet"
fi

echo ""
echo "=== Output row counts ==="
python3 -c "
import glob, sys
for d in ['${OUTPUT_DIR1}', '${OUTPUT_DIR2}']:
    if not d: continue
    try:
        import pandas as pd
        files = glob.glob(d + '/*.parquet')
        if not files:
            print(f'{d}: no parquet files yet')
            continue
        n = sum(len(pd.read_parquet(f, columns=['url'])) for f in files if not f.endswith('_raw'))
        print(f'{d}: {n:,} rows in {len(files)} shards')
    except Exception as e:
        print(f'{d}: error - {e}')
" 2>/dev/null || echo "(row count unavailable)"

echo ""
echo "=== Recent log tail: job ${JOB_IDS%%,*} ==="
_LOG_DIR="/lustre/fsw/portfolios/llmservice/users/\$(whoami)/nemo_curator_shared/logs"
ls -t "\${_LOG_DIR}"/dripper_streaming_{${JOB_IDS}}.log 2>/dev/null | head -2 | while read f; do
  echo "--- \$f ---"
  tail -20 "\$f" 2>/dev/null || true
done
REMOTE
