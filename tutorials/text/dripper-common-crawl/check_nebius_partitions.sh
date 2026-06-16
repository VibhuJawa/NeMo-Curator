#!/usr/bin/env bash
# Show available Slurm partitions and idle nodes on Nebius.
#
# Usage:
#   bash check_nebius_partitions.sh HOST

set -euo pipefail
HOST="${1:?Usage: $0 <host>}"

ssh "${HOST}" bash <<'REMOTE'
echo "=== Partitions ==="
sinfo --format='%.15P %.5a %.10l %.6D %.6t' --sort=P 2>/dev/null

echo ""
echo "=== Idle / available nodes per partition ==="
sinfo --format='%.15P %.6t %.6D %N' --sort=P 2>/dev/null | grep -E 'idle|mix|alloc' | head -30

echo ""
echo "=== My pending jobs ==="
squeue -u "$(whoami)" --format='%.8i %.20j %.8T %.10M %.10L %R' 2>/dev/null
REMOTE
