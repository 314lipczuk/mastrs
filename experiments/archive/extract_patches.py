import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import os

    from utils import results_write_path
    from notebooks.experiment.extract_patches import (
        extract_patches,
        load_meta,
    )

    results_base = results_write_path()


@app.cell
def _():
    import pandas as pd

    args = mo.cli_args()
    EXPERIMENT_NAME = args.get("name", "extract_patches_all")
    DATASET = args.get(
        "dataset",
        str(Path(__file__).resolve().parent.parent / "dataset.parquet"),
    )
    HALF = int(args.get("half", 32))
    CHANNELS = tuple(int(c) for c in args.get("channels", "0,1").split(","))
    COMPRESS = args.get("compress", "none")
    OUT_DIR = args.get("results-dir", f"{results_base}/{EXPERIMENT_NAME}")
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_H5 = str(Path(OUT_DIR) / "patches.h5")

    EXP_PATHS = [
        "/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals",
        "/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-10-12_DoseResponse",
        "/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-02_Sustained_1min",
        "/mnt/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-09-04_RampReverse",
    ]

    mo.md(f"""
    # Patch extraction (all experiments) — `{EXPERIMENT_NAME}`

    | param | value |
    |-------|-------|
    | dataset | `{DATASET}` |
    | half | {HALF} |
    | channels | {CHANNELS} |
    | compress | {COMPRESS} |
    | out | `{OUT_H5}` |
    """)
    return CHANNELS, COMPRESS, DATASET, EXP_PATHS, EXPERIMENT_NAME, HALF, OUT_H5, pd


@app.cell
def _(DATASET, EXP_PATHS, pd):
    df = pd.read_parquet(DATASET)
    print(f"loaded {DATASET}: {len(df):,} rows  uids: {df['uid'].nunique():,}")
    print(f"patterns in df: {sorted(df['ramp_pattern_name'].unique())}")

    EXP_DIR_MAP = {}
    for ep in EXP_PATHS:
        ep_path = Path(ep)
        # use whichever per-experiment parquet exists to discover ramp_pattern_name(s)
        for fname in ("data_filtered.parquet", "exp_data.parquet"):
            pq = ep_path / fname
            if pq.exists():
                pats = pd.read_parquet(pq, columns=["ramp_pattern_name"])[
                    "ramp_pattern_name"
                ].unique()
                for p in pats:
                    EXP_DIR_MAP[p] = ep_path
                break
        else:
            print(f"WARN: no parquet found in {ep_path}")

    # sanity: every pattern in df must map to an exp_dir
    _missing = set(df["ramp_pattern_name"].unique()) - set(EXP_DIR_MAP.keys())
    if _missing:
        raise RuntimeError(f"unmapped ramp_pattern_name(s): {_missing}")

    print(f"map covers {len(EXP_DIR_MAP)} pattern(s):")
    for k, v in EXP_DIR_MAP.items():
        print(f"  {k}  ->  {v}")
    return EXP_DIR_MAP, df


@app.cell
def _(CHANNELS, COMPRESS, EXP_DIR_MAP, HALF, OUT_H5, df):
    _compression = None if COMPRESS == "none" else COMPRESS
    _h5 = extract_patches(
        df, EXP_DIR_MAP, OUT_H5,
        half=HALF, channels=CHANNELS,
        compression=_compression,
        progress=True,
    )
    print(f"DONE  size_MB: {Path(_h5).stat().st_size / 1e6:.1f}")
    return


@app.cell
def _(OUT_H5):
    _meta = load_meta(OUT_H5)
    print(f"patches: {len(_meta):,}")
    print(f"unique uids in h5: {_meta['uid'].nunique():,}")
    _meta.head(10)
    return


if __name__ == "__main__":
    app.run()
