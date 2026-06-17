#!/usr/bin/env bash
# Run PBSS connectivity benchmark on the compute node.
# Submit: sbatch slurm/bench_pbss.sh
# Watch:  tail -f /lustre/fsw/portfolios/llmservice/users/vjawa/bench_pbss.log

#SBATCH --job-name=pbss-bench
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_dataprocessing
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/vjawa/bench_pbss.log
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/vjawa/bench_pbss.log

set -euo pipefail

VENV=/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cached_venv
REPO=/home/vjawa/nemo-curator-adlr-mm

echo "=============================="
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Start : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================="

cd $REPO/tutorials/text/cc-lancedb
$VENV/bin/python -u benchmark_pbss.py

echo "=============================="
echo "End   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================="
