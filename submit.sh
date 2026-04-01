#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00

set -euo pipefail

# SLURM/SSH jobs get a minimal PATH; ensure uv is available
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
source "$HOME/.bashrc" 2>/dev/null || source "$HOME/.profile" 2>/dev/null || true

# ── Args: NAME NOTEBOOK [-- marimo cli args...] ──────────────────────────────
NAME="${1:?Usage: submit.sh NAME NOTEBOOK [-- --key value ...]}"
NOTEBOOK="${2:?Usage: submit.sh NAME NOTEBOOK [-- --key value ...]}"
shift 2

echo "══════════════════════════════════════════════════════"
echo "Experiment : $NAME"
echo "Notebook   : $NOTEBOOK"
echo "Args       : $*"
echo "Job started: $(date)"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "══════════════════════════════════════════════════════"

uv sync
PYTHONUNBUFFERED=1 uv run marimo export html "$NOTEBOOK" -o "/mnt/imaging.data/ppilip/${NAME}.html" -- --name "$NAME" "$@"

echo "Job finished: $(date)"
