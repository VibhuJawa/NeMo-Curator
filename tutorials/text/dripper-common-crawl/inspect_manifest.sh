#!/usr/bin/env bash
# Inspect a manifest directory: list files, show parquet schemas.
#
# Usage:
#   bash inspect_manifest.sh HOST MANIFEST_PATH

set -euo pipefail
HOST="${1:?Usage: $0 <host> <manifest_path>}"
MANIFEST="${2:?}"

ssh "${HOST}" bash <<REMOTE
echo "=== Directory listing ==="
ls -lh "${MANIFEST}/" 2>/dev/null || echo "not a directory"
echo ""
echo "=== Parquet files (recursive) ==="
find "${MANIFEST}" -name "*.parquet" 2>/dev/null | head -20
echo ""
echo "=== Schema of first parquet file found ==="
FIRST=\$(find "${MANIFEST}" -name "*.parquet" 2>/dev/null | head -1)
if [[ -n "\${FIRST}" ]]; then
    echo "File: \${FIRST}"
    source /lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_shared/.venv/bin/activate 2>/dev/null || true
    python3 -c "
import pyarrow.parquet as pq
import sys
t = pq.read_table('\${FIRST}', columns=None)
print('Columns:', t.schema.names)
print('Rows:', len(t))
print('Schema:')
print(t.schema)
"
fi
REMOTE
