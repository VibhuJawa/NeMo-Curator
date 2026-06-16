#!/usr/bin/env bash
# Find parquet files with WARC manifest schema (host_domain, warc_filename columns).
#
# Usage:
#   bash find_warc_manifest.sh HOST [SEARCH_ROOT]

set -euo pipefail
HOST="${1:?Usage: $0 <host> [search_root]}"
ROOT="${2:-/lustre/fsw/portfolios/llmservice/users/vjawa}"

ssh "${HOST}" bash <<REMOTE
echo "=== Searching for parquet files with warc_filename column under ${ROOT} ==="
source /lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_shared/.venv/bin/activate 2>/dev/null || true

python3 - <<'PYEOF' 2>/dev/null
import subprocess, pyarrow.parquet as pq, sys
result = subprocess.run(
    ['find', '${ROOT}', '-maxdepth', '4', '-name', '*.parquet'],
    capture_output=True, text=True
)
for f in result.stdout.splitlines()[:100]:
    try:
        cols = pq.read_schema(f).names
        if 'warc_filename' in cols or 'host_domain' in cols:
            print(f'MATCH: {f} | cols: {cols[:8]}')
    except Exception:
        pass
PYEOF
REMOTE
