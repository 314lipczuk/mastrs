"""Write a training bundle carrying the real optoRTK mCitrine measurement.

    uv run python experiments/build_mcitrine_dataset.py \
        --in dataset_all.parquet --out dataset_all_mcitrine.parquet

Reads the source bundle from ``materials/``, joins per-cell mCitrine from each
acquisition's own ``exp_data.parquet`` on the imaging mount (see
``optoerk.data.mcitrine``), and writes a NEW bundle beside it. The input is never
modified — the C0-surrogate bundle stays exactly as it was, so the existing
checkpoint remains reproducible.

Needs the imaging mount, so this runs on the cluster (or locally with the
Kingston mount attached).
"""
from __future__ import annotations

import argparse
import json

import polars as pl

from optoerk.core.utils import imaging_root, materials_path
from optoerk.data.mcitrine import MCITRINE_COL, join_mcitrine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="dataset_all.parquet")
    ap.add_argument("--out", dest="dst", default="dataset_all_mcitrine.parquet")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing output bundle")
    args = ap.parse_args()

    src, dst = materials_path(args.src), materials_path(args.dst)
    if not src.exists():
        raise SystemExit(f"source bundle not found: {src}")
    if dst.exists() and not args.overwrite:
        raise SystemExit(f"{dst} exists; pass --overwrite to replace it")
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to write the output over the input bundle")

    root = imaging_root()
    if not root.exists():
        raise SystemExit(
            f"imaging mount not found at {root}. This needs the raw acquisition "
            f"folders; run on the cluster or attach the Kingston mount."
        )
    print(f"[mcitrine] {src.name} -> {dst.name}   (imaging root {root})")

    df = pl.read_parquet(src)
    out, coverage = join_mcitrine(df, root=root)

    print(coverage)
    n_cells = out.select("uid").n_unique()
    n_hit = out.filter(pl.col(MCITRINE_COL).is_not_null()).select("uid").n_unique()
    print(f"[mcitrine] {n_hit:,}/{n_cells:,} cells ({n_hit / n_cells * 100:.1f}%) "
          f"carry a real per-cell mCitrine value")

    out.write_parquet(dst)
    # The coverage table is provenance: which acquisition each experiment's
    # expression came from, and how much of it resolved. Kept next to the bundle
    # so a run months later can tell what it was trained on.
    side = dst.with_suffix(".coverage.json")
    side.write_text(json.dumps({
        "source_bundle": src.name,
        "imaging_root": str(root),
        "cells_total": n_cells,
        "cells_with_mcitrine": n_hit,
        "per_experiment": coverage.to_dicts(),
    }, indent=2))
    print(f"[mcitrine] wrote {dst}\n[mcitrine] wrote {side}")


if __name__ == "__main__":
    main()
