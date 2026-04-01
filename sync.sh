#!/bin/bash
# Sync project to research clusters

# ── Cluster destinations ────────────────────────────────────────────────────
IZB_DEST="izb:/home/ppilip/masters"
IBU_DEST="ibucluster:/home/ppilipczuk/data/masters"

# ── Files/dirs to exclude ───────────────────────────────────────────────────
EXCLUDES=(
    .claude/
    .venv/
    __pycache__/
    .DS_Store
    .python-version
    .vscode/
    materials/
    docs/
    extra/
    "*.pyc"
    "*.egg-info/"
    ".git/"
    cluster_results/
    results/
)

# ── Build exclude flags ──────────────────────────────────────────────────────
exclude_flags=()
for pattern in "${EXCLUDES[@]}"; do
    exclude_flags+=(--exclude="$pattern")
done

# ── Parse args ───────────────────────────────────────────────────────────────
TARGETS=("izb")
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

# ── Sync function ─────────────────────────────────────────────────────────────
sync_to() {
    local name="$1"
    local dest="$2"
    echo ">>> Syncing to $name ($dest)"
    rsync -avz --delete "${exclude_flags[@]}" \
        /Users/polya/workshop/masters/ "$dest/"
    echo ">>> Done: $name"
}

# ── Run ───────────────────────────────────────────────────────────────────────
for target in "${TARGETS[@]}"; do
    case "$target" in
        izb)         sync_to "izb" "$IZB_DEST" ;;
        ibucluster)  sync_to "ibucluster" "$IBU_DEST" ;;
        *)           echo "Unknown target: $target (valid: izb, ibucluster)" ;;
    esac
done
