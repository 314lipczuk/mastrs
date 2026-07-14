import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import polars as pl
    import altair as alt
    import matplotlib.pyplot as plt
    import marimo as mo
    from hastyplot import qplot

    from optoerk.core.utils import materials_path
    from optoerk.data.preprocessing import (
        CALIBRATIONS,
        EXPERIMENT_INSTRUMENT,
    )

    alt.data_transformers.disable_max_rows()
    return (
        CALIBRATIONS,
        EXPERIMENT_INSTRUMENT,
        materials_path,
        mo,
        np,
        pd,
        pl,
        plt,
        qplot,
    )


@app.cell
def _(mo):
    mo.md("""
    # Stimulation dose metrics across the `all` bundle

    Which number represents "how much light a cell got"? We have several, and
    they do **not** agree on how far out-of-distribution the Niesen run is:

    | metric | units | area-dependent? | what it captures |
    |---|---|---|---|
    | `stim_power` | % | no | DMD/LED setpoint (nominal) |
    | optical power `P_uW` | µW | **no** | measured power out of the objective |
    | `energy_uJ` | µJ | **no** | `P_uW × exposure` per pulse |
    | irradiance | mW/cm² | **yes** | power ÷ illuminated field area |
    | `fluence_mJ_cm2` = `u_t` | mJ/cm² | **yes** | `irradiance × exposure` — the model's dose channel |

    The two `NIESEN_TOCHECK.md` open items map cleanly onto this split:

    - **ITEM 1 (area guess).** Niesen's field area is *assumed* equal to jungfrau's
      900 µm. Area only enters the **area-dependent** row (irradiance / fluence).
      The **area-invariant** metrics (µW, µJ) are unaffected — so comparing the two
      families isolates how much of the Niesen gap is a real measurement vs. the
      area assumption.
    - **ITEM 2 (no-ND 70×).** Niesen ran without the ND5 filter jungfrau used, so its
      µW is ~70× higher at the same %. This shows up in **every** metric and is real
      physics, not the area guess.

    This notebook quantifies both and shows what the 70× does to frozen
    standardization (`compute_norm_stats`) — the training-skew concern in ITEM 2.
    """)
    return


@app.cell
def _(CALIBRATIONS, EXPERIMENT_INSTRUMENT, materials_path, np, pd):
    df = pd.read_parquet(materials_path("dataset_all.parquet"))

    # Recover the two metrics the bundle doesn't store: instantaneous optical
    # power (µW, area-INVARIANT) and irradiance (mW/cm², area-DEPENDENT). Both
    # come straight off the per-experiment calibration curve, so they need the
    # instrument each row was acquired on (recoverable from the experiment name).
    df["instrument"] = df["original_experiment_name"].map(EXPERIMENT_INSTRUMENT)

    df["power_uW"] = np.nan
    df["irradiance_mW_cm2"] = np.nan
    for _inst, _cal in CALIBRATIONS.items():
        _m = df["instrument"] == _inst
        df.loc[_m, "power_uW"] = np.interp(df.loc[_m, "stim_power"], _cal["pct"], _cal["uW"])
        df.loc[_m, "irradiance_mW_cm2"] = np.interp(
            df.loc[_m, "stim_power"], _cal["pct"], _cal["mW_cm2"]
        )

    # Per-pulse (m_t==1) rows carry the dose; per-cell cumulative totals from cumsum.
    pulses = df[df["m_t"] == 1].copy()
    print(f"{len(df):,} frames  |  {df['uid'].nunique():,} cells  |  {len(pulses):,} pulses")
    df["original_experiment_name"].value_counts()
    return df, pulses


@app.cell
def _(mo):
    mo.md("""
    ## 1. Per-experiment dose, every metric side by side

    Per-**pulse** means over stimulated frames, plus the per-**cell** cumulative
    dose. Sorted by fluence. Watch how Niesen ranks vs. how it ranks on the
    area-invariant columns.
    """)
    return


@app.cell
def _(pd, pl, pulses):
    _g = pulses.groupby("original_experiment_name")
    summary = (
        pl.from_pandas(
            pd.DataFrame(
                {
                    "stim_power_%": _g["stim_power"].mean(),
                    "exposure_ms": _g["stim_exposure"].mean(),
                    "power_uW": _g["power_uW"].mean(),
                    "energy_uJ/pulse": _g["energy_uJ"].mean(),
                    "irradiance_mW_cm2": _g["irradiance_mW_cm2"].mean(),
                    "fluence_mJ_cm2/pulse": _g["fluence_mJ_cm2"].mean(),
                    "fluence_max": _g["fluence_mJ_cm2"].max(),
                }
            ).reset_index()
        )
        .rename({"original_experiment_name": "experiment"})
        .sort("fluence_mJ_cm2/pulse", descending=True)
    )
    summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Per-pulse dose distributions (log scale)

    Every experiment's per-pulse fluence, log10. The jungfrau experiments pile
    up around 1–20 mJ/cm²; Niesen sits ~1–2 decades to the right — a sparse,
    detached tail the frozen norm stats have to span.
    """)
    return


@app.cell
def _(np, plt, pulses):
    # Matplotlib (static raster) rather than Vega: 1.27M raw pulses ship fine to a
    # PNG but would blow up an interactive Altair payload (~350 MB).
    _fig, _ax = plt.subplots(figsize=(11, 4.5))
    _lf_all = np.log10(pulses["fluence_mJ_cm2"].to_numpy())
    _bins = np.linspace(np.nanmin(_lf_all[np.isfinite(_lf_all)]),
                        np.nanmax(_lf_all[np.isfinite(_lf_all)]), 61)
    for _exp, _g in pulses.groupby("original_experiment_name"):
        _lf = np.log10(_g["fluence_mJ_cm2"].to_numpy())
        _lf = _lf[np.isfinite(_lf)]
        _ax.hist(_lf, bins=_bins, histtype="step", linewidth=1.6,
                 label=_exp, alpha=0.85)
    _ax.set_xlabel("log10 fluence (mJ/cm²) per pulse")
    _ax.set_ylabel("pulse count")
    _ax.set_title("Per-pulse fluence by experiment — Niesen is the detached right tail")
    _ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Two axes of the Niesen gap — 70× or 7×?

    Ratio of the **Niesen** median to the **jungfrau-family** median for each metric.
    The gap is **not one number** — it splits along two independent axes:

    - **Area-invariant (µW, µJ) vs. area-dependent (mW/cm², mJ/cm²) — this is ITEM 1.**
      Within a matched integration convention the two are *exactly equal*
      (power µW ratio == irradiance ratio; energy ratio == fluence ratio), because the
      **same** 0.636 mm² area constant divides both scopes. So the area assumption
      contributes **nothing** to the gap. Fixing ITEM 1 rescales Niesen fluence but
      does not change any of these ratios.
    - **Instantaneous vs. time-integrated (× exposure) — this is the 70× / 7× split.**
      Instantaneous metrics (µW, mW/cm²) are **70×** — the pure no-ND scope difference.
      Per-pulse metrics (µJ, mJ/cm²) are only **~7×**, because
      `fluence = irradiance × exposure` and Niesen ran ~10× **shorter** exposures
      (78 ms vs. the 750 ms jungfrau-family median, which is BO-dominated).
      70× irradiance × 0.1× exposure ≈ 7× fluence.

    **Why it matters:** the model's dose channel `u_t` **is** fluence, so the OOD factor
    the model actually sees per pulse is **~7×**, not 70×. The 70× only reappears when
    exposure is matched (vs. `freepattern_v2` at 74 ms, fluence ratio is back to ~74×).
    """)
    return


@app.cell
def _(pl, pulses, qplot):
    # metric -> (label, area axis, integration axis). power/irradiance are the
    # same instantaneous quantity (÷area); energy/fluence are those × exposure.
    _metrics = {
        "power_uW":          ("power µW",         "area-invariant", "instantaneous"),
        "irradiance_mW_cm2": ("irradiance mW/cm²", "area-dependent", "instantaneous"),
        "energy_uJ":         ("energy µJ",         "area-invariant", "per-pulse (×exposure)"),
        "fluence_mJ_cm2":    ("fluence mJ/cm² = u_t", "area-dependent", "per-pulse (×exposure)"),
        "stim_exposure":     ("exposure ms",       "—",              "exposure alone"),
    }
    _pp = pulses.copy()
    _pp["family"] = _pp["instrument"].where(
        _pp["instrument"] == "niesen", "jungfrau"
    )
    _rows = []
    for _col, (_label, _area, _integ) in _metrics.items():
        _med = _pp.groupby("family")[_col].median()
        _rows.append(
            {
                "metric": _label,
                "area_axis": _area,
                "integration": _integ,
                "ratio_niesen_over_jungfrau": float(_med["niesen"] / _med["jungfrau"]),
            }
        )
    ratio_df = pl.DataFrame(_rows)
    qplot(
        ratio_df,
        x="ratio_niesen_over_jungfrau",
        y="metric",
        color="integration",
        mark="bar",
        title="Niesen ÷ jungfrau median, by metric",
        subtitle="Instantaneous=70×, per-pulse=7× (exposure 0.1×). area-inv==area-dep within each ⇒ area (ITEM 1) is irrelevant.",
        width=680,
        height=230,
    )
    return (ratio_df,)


@app.cell
def _(ratio_df):
    ratio_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. What the 70× does to frozen standardization (ITEM 2's training risk)

    `compute_norm_stats` takes a single mean/std of `u_t` over **all** train
    frames, then every experiment is standardized by that one constant. Below,
    per-experiment spread (p75−p25 over stimulated frames) of `u_t` after three
    transforms:

    - **linear z** — the current scheme. Niesen's huge values inflate the global
      std, so every jungfrau experiment's dose spread collapses toward zero.
    - **log1p → z** — compresses the tail; restores per-experiment spread.
    - **global rank (pct)** — scale-free; every experiment keeps a usable range.
    """)
    return


@app.cell
def _(df, np, pl):
    _stim = df[df["m_t"] == 1].copy()
    _u = df["u_t"].to_numpy()  # global stats over ALL frames, mirroring compute_norm_stats

    _mu, _sd = _u.mean(), max(_u.std(), 1e-6)
    _stim["z_linear"] = (_stim["u_t"] - _mu) / _sd

    _ul = np.log1p(_u)
    _mul, _sdl = _ul.mean(), max(_ul.std(), 1e-6)
    _stim["z_log1p"] = (np.log1p(_stim["u_t"]) - _mul) / _sdl

    _stim["rank_pct"] = df["u_t"].rank(pct=True).loc[_stim.index].to_numpy()

    _long = []
    for _transform in ["z_linear", "z_log1p", "rank_pct"]:
        _q = _stim.groupby("original_experiment_name")[_transform].quantile([0.25, 0.75]).unstack()
        for _exp, _r in _q.iterrows():
            _long.append(
                {
                    "experiment": _exp,
                    "transform": _transform,
                    "iqr_spread": float(_r[0.75] - _r[0.25]),
                    "is_niesen": "Niesen" in _exp,
                }
            )
    skew_df = pl.DataFrame(_long)
    return (skew_df,)


@app.cell
def _(qplot, skew_df):
    qplot(
        skew_df,
        x="iqr_spread",
        y="experiment",
        color="transform",
        mark="bar",
        opacity=0.75,
        facet_col="transform",
        title="Per-experiment usable dose spread (IQR over stim frames), by transform",
        subtitle="linear: jungfrau bars vanish next to Niesen. log1p / rank: spread restored.",
        width=260,
        height=340,
    )
    return


@app.cell
def _(skew_df):
    skew_df.sort(["transform", "iqr_spread"])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaways

    1. **The gap is 70× instantaneous but ~7× in fluence — mind which one you quote.**
       At matched 10% power Niesen is 70× the optical power/irradiance (the no-ND scope
       difference), but its per-pulse **fluence** — the model's `u_t` channel — is only
       ~7×, because Niesen ran ~10× shorter exposures. The model sees the ~7×; the 70×
       reappears only when exposure is matched.
    2. **Area (ITEM 1) contributes nothing to either number.** Area-invariant (µW, µJ)
       and area-dependent (mW/cm², mJ/cm²) ratios are identical within each integration
       convention, because the same 0.636 mm² area divides both scopes. Fixing ITEM 1
       rescales Niesen fluence but changes none of these ratios.
    3. **Under linear standardization the tail is still real.** Niesen's per-pulse
       fluence sits ~1 decade above the jungfrau bulk (panel 2), and a single global std
       flattens every jungfrau experiment's dose signal (panel 4, `z_linear`) — even at
       7×, not 70×.
    4. **If Niesen stays in the training mix**, a `log1p` or rank transform on `u_t`
       restores per-experiment dose spread — matching the rank-normalization already
       used for `optortk_expr`. Otherwise, move Niesen to eval-only (drop it from
       `BUNDLES["all"]`).
    """)
    return


if __name__ == "__main__":
    app.run()
