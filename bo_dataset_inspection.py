import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np

    return alt, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # BO dataset inspection

    Compare `BO_v1.parquet` / `BO_v2.parquet` vs main `dataset.parquet`.
    Goal: visualize, check schema compatibility with existing pipeline.
    """)
    return


@app.cell
def _(pl):
    bo1 = pl.read_parquet("BO_v1.parquet")
    bo2 = pl.read_parquet("BO_v2.parquet")
    ref = pl.read_parquet("dataset.parquet")
    return bo1, bo2, ref


@app.cell
def _(bo1, bo2, mo, ref):
    shape_tbl = [
        {"file": "BO_v1", "rows": bo1.height, "cols": bo1.width},
        {"file": "BO_v2", "rows": bo2.height, "cols": bo2.width},
        {"file": "dataset (ref)", "rows": ref.height, "cols": ref.width},
    ]
    mo.md(f"## Shape\n\n" + "\n".join(f"- **{r['file']}**: {r['rows']:,} rows × {r['cols']} cols" for r in shape_tbl))
    return


@app.cell
def _(bo1, mo, pl, ref):
    bo_cols = set(bo1.columns)
    ref_cols = set(ref.columns)
    shared = sorted(bo_cols & ref_cols)
    bo_only = sorted(bo_cols - ref_cols)
    ref_only = sorted(ref_cols - bo_cols)

    def _type_row(c):
        bt = str(bo1.schema[c])
        rt = str(ref.schema[c])
        return {"col": c, "BO": bt, "ref": rt, "match": bt == rt}

    compat_df = pl.DataFrame([_type_row(c) for c in shared])
    mo.md(
        f"## Schema overlap\n\n"
        f"- shared: **{len(shared)}**\n"
        f"- BO-only: **{len(bo_only)}** → {bo_only}\n"
        f"- ref-only: **{len(ref_only)}** → {ref_only}\n"
    )
    return (compat_df,)


@app.cell
def _(compat_df, mo, pl):
    mo.md("### Type compatibility on shared columns")
    mismatches = compat_df.filter(~pl.col("match")) if False else compat_df
    compat_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Compatibility verdict

    **Core signal columns match** (`mean_intensity_C0_nuc`, `mean_intensity_C1_nuc`, `cnr`, `particle`, `fov`, `timestep`, `time`, `stim`, `stim_exposure`, `label`, `x`, `y`).

    **Differences to handle before feeding into existing pipeline:**
    - `cnr` is `Float32` in BO vs `Float64` in ref → cast.
    - `area_nuc` (BO) vs `area` (ref) → rename.
    - BO has `phase_name` / `phase_id` / `condition_idx` / `ramp` (BO-specific experiment structure).
    - BO missing downstream-derived cols: `uid`, `cnr_mean_norm`, `cnr_median_norm`, `energy_uJ`, `fluence_mJ_cm2`, `u_t`, `m_t`, `recency`, `burst_pos`, `ewma_fast/slow`, `s_cum` → recomputable via preprocessing.
    - BO missing `stim_timestep`, `stim_power` (null), `stim_channel_*`, `ramp_pattern_name`, `optocheck*`.
    - `channels` is `List[String]` in BO vs `List[Struct]` in ref.

    → **Usable** with a thin adapter: rename `area_nuc`→`area`, cast `cnr`, run existing normalization to fill derived columns.
    """)
    return


@app.cell
def _(bo1, bo2, mo, ref):
    def _track_stats(df, name):
        tl = df.group_by(["fov", "particle"]).len()
        return {
            "file": name,
            "n_particles": df["particle"].n_unique(),
            "n_fovs": df["fov"].n_unique(),
            "n_timesteps": df["timestep"].n_unique(),
            "t_min": float(df["time"].min()),
            "t_max": float(df["time"].max()),
            "track_len_mean": float(tl["len"].mean()),
            "track_len_min": int(tl["len"].min()),
            "track_len_max": int(tl["len"].max()),
        }

    stats = [_track_stats(bo1, "BO_v1"), _track_stats(bo2, "BO_v2"), _track_stats(ref, "dataset")]
    mo.md("## Track & time structure")
    stats
    return


@app.cell
def _(mo):
    source_sel = mo.ui.dropdown(
        options=["BO_v1", "BO_v2", "dataset"], value="BO_v1", label="Source"
    )
    n_tracks = mo.ui.slider(5, 50, value=20, label="Tracks to plot")
    mo.hstack([source_sel, n_tracks])
    return n_tracks, source_sel


@app.cell
def _(bo1, bo2, ref, source_sel):
    src_map = {"BO_v1": bo1, "BO_v2": bo2, "dataset": ref}
    current = src_map[source_sel.value]
    return (current,)


@app.cell
def _(alt, current, n_tracks, pl):
    sample_keys = (
        current.select(["fov", "particle"]).unique().sample(n=min(n_tracks.value, current.select(["fov", "particle"]).unique().height), seed=0)
    )
    sub = current.join(sample_keys, on=["fov", "particle"], how="inner").with_columns(
        (pl.col("fov").cast(pl.Utf8) + "_" + pl.col("particle").cast(pl.Utf8)).alias("track")
    )

    ycol = "cnr" if "cnr" in sub.columns else "mean_intensity_C0_nuc"

    chart = (
        alt.Chart(sub.select(["time", ycol, "track", "stim"]).to_pandas())
        .mark_line(opacity=0.6)
        .encode(
            x=alt.X("time:Q", title="time (s)"),
            y=alt.Y(f"{ycol}:Q"),
            color=alt.Color("track:N", legend=None),
        )
        .properties(width=700, height=300, title=f"Sample tracks — {ycol}")
    )
    chart
    return


@app.cell
def _(alt, bo1, bo2, pl, ref):
    import numpy as _np

    edges = _np.linspace(-1.0, 3.0, 81)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _hist(df, name):
        v = df["cnr"].cast(pl.Float64).drop_nulls().to_numpy()
        v = v[(v >= edges[0]) & (v <= edges[-1])]
        counts, _ = _np.histogram(v, bins=edges)
        return pl.DataFrame({"cnr": centers, "count": counts, "source": name})

    dist = pl.concat([_hist(bo1, "BO_v1"), _hist(bo2, "BO_v2"), _hist(ref, "dataset")])

    hist = (
        alt.Chart(dist.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("cnr:Q", title="cnr"),
            y=alt.Y("count:Q"),
            color="source:N",
        )
        .properties(width=700, height=300, title="CNR distribution across sources")
    )
    hist
    return


@app.cell
def _(alt, bo1, bo2, pl):
    bo_phases = pl.concat(
        [
            bo1.group_by(["phase_name", "stim_exposure", "ramp"]).len().with_columns(pl.lit("BO_v1").alias("src")),
            bo2.group_by(["phase_name", "stim_exposure", "ramp"]).len().with_columns(pl.lit("BO_v2").alias("src")),
        ]
    )

    phase_chart = (
        alt.Chart(bo_phases.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("stim_exposure:Q", title="stim exposure"),
            y=alt.Y("ramp:Q", title="ramp"),
            color=alt.Color("len:Q", scale=alt.Scale(scheme="viridis")),
            facet=alt.Facet("src:N", columns=2),
        )
        .properties(width=300, height=250, title="BO phase coverage (exposure × ramp)")
    )
    phase_chart
    return


@app.cell
def _(alt, current, pl):
    stim_time = (
        current.group_by("timestep")
        .agg(pl.col("stim").mean().alias("frac_stim"), pl.col("time").first())
        .sort("timestep")
    )
    stim_chart = (
        alt.Chart(stim_time.to_pandas())
        .mark_line()
        .encode(x="time:Q", y=alt.Y("frac_stim:Q", title="fraction of cells stimulated"))
        .properties(width=700, height=200, title="Stimulation fraction over time (current source)")
    )
    stim_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Apply existing preprocessing pipeline

    Adapt BO schema (rename `area_nuc`→`area`, alias `phase_name`→`ramp_pattern_name`,
    fill missing `cell_line`, `stim_power`=100), then run `load_and_clean` + `augment`.
    """)
    return


@app.cell
def _():
    import sys
    import tempfile
    import os
    import pandas as pd

    sys.path.insert(0, "notebooks/experiment")
    import preprocessing as P

    def adapt_and_clean(src_path, stim_power_default=10.0):
        df = pd.read_parquet(src_path)
        df = df.rename(columns={"area_nuc": "area"})
        df["ramp_pattern_name"] = df["phase_name"]
        df["cell_line"] = "EGFR"
        df["stim_power"] = stim_power_default
        df["cnr"] = df["cnr"].astype("float64")
        fd, tmp = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        df.to_parquet(tmp)
        try:
            out = P.load_and_clean(tmp, baseline_cnr_max=None, cell_line="EGFR")
            out = P.augment(out)
        finally:
            os.unlink(tmp)
        return out

    bo1_clean = adapt_and_clean("BO_v1.parquet")
    bo2_clean = adapt_and_clean("BO_v2.parquet")
    return bo1_clean, bo2_clean, pd


@app.cell
def _(ref):
    import sys as _s2
    if "notebooks/experiment" not in _s2.path:
        _s2.path.insert(0, "notebooks/experiment")
    import preprocessing as _P2

    ref_pd = ref.to_pandas() if hasattr(ref, "to_pandas") else ref
    ref_aug = _P2.augment(ref_pd)
    return (ref_aug,)


@app.cell
def _(bo1_clean, bo2_clean, mo, pd, ref_aug):
    def _summary(df, name):
        return {
            "src": name,
            "rows": len(df),
            "cells": df["uid"].nunique(),
            "cnr_norm_mean": round(float(df["cnr_median_norm"].mean()), 3),
            "cnr_norm_std": round(float(df["cnr_median_norm"].std()), 3),
            "amplitude_mean": round(float(df["amplitude"].mean()), 3),
            "total_fluence_mean": round(float(df["total_fluence"].mean()), 2),
            "responder_frac": round(float(df.groupby("uid")["responder"].first().mean()), 3),
        }

    compare = pd.DataFrame([
        _summary(bo1_clean, "BO_v1"),
        _summary(bo2_clean, "BO_v2"),
        _summary(ref_aug, "dataset (ref)"),
    ])
    mo.md("### Cleaned + augmented summary — BO vs reference")
    compare
    return


@app.cell
def _(alt, bo1_clean, bo2_clean, pd, ref_aug):
    import numpy as _np2

    edges2 = _np2.linspace(0.0, 4.0, 81)
    centers2 = 0.5 * (edges2[:-1] + edges2[1:])

    def _nhist(s, name):
        v = s.dropna().to_numpy()
        v = v[(v >= edges2[0]) & (v <= edges2[-1])]
        c, _ = _np2.histogram(v, bins=edges2)
        return pd.DataFrame({"cnr_median_norm": centers2, "count": c, "source": name})

    dist_norm = pd.concat([
        _nhist(bo1_clean["cnr_median_norm"], "BO_v1"),
        _nhist(bo2_clean["cnr_median_norm"], "BO_v2"),
        _nhist(ref_aug["cnr_median_norm"], "dataset (ref)"),
    ])

    chart_norm = (
        alt.Chart(dist_norm)
        .mark_line()
        .encode(x="cnr_median_norm:Q", y="count:Q", color="source:N")
        .properties(width=700, height=250, title="Normalized CNR (post-pipeline)")
    )
    chart_norm
    return


@app.cell
def _(alt, bo1_clean, bo2_clean, pd, ref_aug):
    def _per_cell(df, name):
        agg = df.groupby("uid").agg(
            total_fluence=("total_fluence", "first"),
            amplitude=("amplitude", "first"),
            responder=("responder", "first"),
        ).reset_index()
        agg["source"] = name
        return agg

    cells = pd.concat([
        _per_cell(bo1_clean, "BO_v1"),
        _per_cell(bo2_clean, "BO_v2"),
        _per_cell(ref_aug, "dataset (ref)"),
    ])

    dose_chart = (
        alt.Chart(cells)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("total_fluence:Q", title="total fluence (mJ/cm²)"),
            y=alt.Y("amplitude:Q", title="peak CNR - 1"),
            color="source:N",
            shape=alt.Shape("responder:N"),
        )
        .properties(width=700, height=350, title="Dose-response (per cell)")
    )
    dose_chart
    return


@app.cell
def _(alt, bo1_clean, bo2_clean, pd, ref_aug):
    def _area_over_time(df, name):
        col = "area_nuc" if "area_nuc" in df.columns else "area"
        g = df.groupby("frame")[col].agg(["median", "mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        g.columns = ["median", "mean", "q25", "q75"]
        g = g.reset_index()
        g["time_min"] = g["frame"] * 1.0
        g["source"] = name
        return g

    ref_parts = [
        _area_over_time(sub, f"ref:{pat}")
        for pat, sub in ref_aug.groupby("ramp_pattern_name")
    ]
    area_df = pd.concat([
        _area_over_time(bo1_clean, "BO_v1"),
        _area_over_time(bo2_clean, "BO_v2"),
        *ref_parts,
    ])

    band = (
        alt.Chart(area_df)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("time_min:Q", title="frame (min)"),
            y=alt.Y("q25:Q", title="nuclear area (px)"),
            y2="q75:Q",
            color="source:N",
        )
    )
    line = (
        alt.Chart(area_df)
        .mark_line()
        .encode(x="time_min:Q", y="median:Q", color="source:N")
    )
    (band + line).properties(width=750, height=320, title="Nuclear area over time (median ± IQR)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    - Both BO files: 90 timesteps, ~89 min total, ~60s cadence. Reference dataset: up to 510 timesteps (much longer runs).
    - BO structured as BO iterations × conditions (`phase_name` = `BO_iter_{i}_cond_{j}`); varies `stim_exposure` and `ramp`.
    - Core intensity / CNR columns compatible — needs minor renaming + dtype cast to run through existing preprocessing.
    - BO lacks derived features (norm cnr, EWMA, burst_pos, etc.); regenerate via `notebooks/experiment/preprocessing.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
