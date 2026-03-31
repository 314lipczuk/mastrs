#!/usr/bin/env bash
# watch_jobs.sh — tail SLURM job logs in a tmux split layout
# Usage: ./watch_jobs.sh [--user USERNAME]

set -euo pipefail

USER="${SLURM_USER:-$USER}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) USER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# --- Collect log files from active jobs (snapshot at startup) ---
mapfile -t LOG_FILES < <(
  squeue -u "$USER" -o "%o" 2>/dev/null \
    | tail -n +2 \
    | grep -v '^N/A$' \
    | grep -v '^$' \
    | sort -u
)

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
  echo "No active SLURM jobs with log files found for user: $USER"
  exit 1
fi

echo "Found ${#LOG_FILES[@]} log file(s):"
printf '  %s\n' "${LOG_FILES[@]}"

if ! command -v tmux &>/dev/null; then
  echo "tmux is required but not found."
  exit 1
fi

SESSION="slurm_watch_$$"

watch_cmd() {
  echo "watch -n 5 'echo === $1 ===; echo; tail -n 40 \"$1\" 2>/dev/null || echo file gone'"
}

# Create session with first pane running the watch command directly
tmux new-session -d -s "$SESSION" -x "$(tput cols)" -y "$(tput lines)" "$(watch_cmd "${LOG_FILES[0]}")"

# Each split-window call targets the session (active pane) and runs the command directly
for i in "${!LOG_FILES[@]}"; do
  [[ $i -eq 0 ]] && continue
  tmux split-window -t "$SESSION" "$(watch_cmd "${LOG_FILES[$i]}")"
  tmux select-layout -t "$SESSION" tiled
done

tmux select-pane -t "$SESSION:0.0"

echo "Attaching to tmux session '$SESSION' (Ctrl+b d to detach)"
tmux attach-session -t "$SESSION"
