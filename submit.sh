#!/bin/bash
#SBATCH --job-name=cvae-erk
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=cvae_%j.log
#SBATCH --error=cvae_%j.log

# ── Resource rationale ──────────────────────────────────────────────────────
# Model: ~150K parameters, batch=64, ~4K training windows
# 300 epochs × ~63 batches/epoch = ~19K optimizer steps
# GPU memory: <100 MB (any GPU will do)
# Time: 1 hour is generous — expect <10 min on any GPU
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Environment setup (adjust to your cluster) ─────────────────────────────
# module load python/3.11 cuda/12.1
uv sync

# ── Run notebook ────────────────────────────────────────────────────────────
NOTEBOOK="${1:-notebooks/notebook.ipynb}"
OUT_NOTEBOOK="~/results/${NOTEBOOK%.ipynb}_executed_$(date +%Y%m%d_%H%M%S).ipynb"
mkdir -p "$(dirname "$OUT_NOTEBOOK")"
uv run python -m ipykernel install --user --name masters
uv run papermill "$NOTEBOOK" "$OUT_NOTEBOOK" --no-progress-bar -k masters

echo "Job finished: $(date)"
echo "Executed notebook: $OUT_NOTEBOOK"
