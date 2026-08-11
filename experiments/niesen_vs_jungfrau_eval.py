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

    from optoerk.core.experiment import load_experiment
    from optoerk.core.utils import materials_path, results_write_path
    from optoerk.data.history_data import load_history_tracks
    from optoerk.data.history_dataset import make_split
    from optoerk.eval.history_predict import predict_many

    alt.data_transformers.disable_max_rows()
    return (
        load_experiment,
        load_history_tracks,
        make_split,
        materials_path,
        mo,
        np,
        pl,
        predict_many,
        qplot,
        results_write_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Niesen vs jungfrau — per-scope model evaluation

    Does the full-history model trained on the **combined** (`all`) bundle do
    equally well on jungfrau and Niesen cells, or does it favour one? And how do
    the `all`-trained variants compare to a model trained on **Niesen only**?

    Evaluated on **held-out** cells (seed-0 val split — the same split training
    used), scored separately for each scope. Metrics per cell, averaged over
    prediction points (`t = 10 … T−F`, full-history context each time):

    - **MAE** — mean abs error, absolute `cnr_median_norm` units.
    - **nMAE** — MAE ÷ that cell's CNR std. Scope-fair: Niesen's larger response
      amplitude inflates raw MAE, so nMAE is the honest cross-scope skill number.
    - **skill vs persistence** — `1 − MSE_model / MSE_persist` (persistence =
      hold last observed CNR flat over the horizon). >0 beats the naive baseline.
    - **NLL** — Gaussian NLL from the predicted (mean, σ); catches mis-calibration.

    **Overfit read:** compare **train vs val** within each scope. A model that
    overfits one scope shows a large train↔val gap *there* and not elsewhere.
    """)
    return


@app.cell
def _(results_write_path):
    # The three usable 2026-07-14 bundles. Two trained on the combined `all` bundle
    # (niesen+jungfrau), one on Niesen alone. Edit this list to add older variants.
    MODEL_DIR = results_write_path()
    MODELS = [
        ("all (fixed-F)", "seq2scal_history_optortk_2026-07-14_09.48.21"),
        ("all (multilen)", "seq2scal_history_optortk_multilen_2026-07-14_09.48.21"),
        ("niesen-only", "seq2scal_history_niesen_only_2026-07-14_09.48.21"),
    ]

    # Eval budget (kept modest so headless export runs; raise for tighter numbers).
    CELLS_PER_GROUP = 150   # random cells per (scope × split)
    POINT_STEP = 20         # frames between prediction points within a cell
    MAX_POINTS = 8          # cap prediction points per cell
    SEED = 0
    return CELLS_PER_GROUP, MAX_POINTS, MODELS, MODEL_DIR, POINT_STEP, SEED


@app.cell
def _(SEED, load_history_tracks, make_split, materials_path, np):
    # Load once: per-cell tracks + the deterministic split + scope labels.
    cnr, feats, cond, meta_raw = load_history_tracks(materials_path("dataset_all.parquet"))
    split = make_split(cond, seed=SEED)
    which = np.empty(len(cond), dtype=object)
    for _k, _idx in split.items():
        which[_idx] = _k

    meta_df = meta_raw.copy()
    meta_df["split"] = which
    meta_df["scope"] = np.where(
        meta_df["original_experiment_name"].str.contains("Niesen"), "niesen", "jungfrau"
    )
    meta_df["cnr_std"] = [float(np.std(np.asarray(cnr[i], np.float32))) for i in range(len(cnr))]
    print("cells:", len(meta_df))
    print(meta_df.groupby(["scope", "split"]).size().unstack(fill_value=0))
    return cnr, feats, meta_df


@app.cell
def _(CELLS_PER_GROUP, SEED, meta_df, np):
    # Subsampled cell-index lists per (scope × split). train + val only (test skipped).
    _rng = np.random.default_rng(SEED)
    GROUPS = {}
    for _scope in ("jungfrau", "niesen"):
        for _sp in ("train", "val"):
            _idx = np.array(meta_df.index[(meta_df["scope"] == _scope) & (meta_df["split"] == _sp)])
            _rng.shuffle(_idx)
            GROUPS[(_scope, _sp)] = _idx[:CELLS_PER_GROUP]
    {f"{s}/{sp}": len(v) for (s, sp), v in GROUPS.items()}
    return (GROUPS,)


@app.cell
def _(MAX_POINTS, POINT_STEP, cnr, feats, meta_df, np, predict_many):
    def eval_cells(model, idx_list, *, t_min=10, step=POINT_STEP, max_points=MAX_POINTS):
        """Per-cell error rows for a trained model over a list of cell indices."""
        F = model.cfg.future_len
        rows = []
        for i in idx_list:
            c = np.asarray(cnr[i], np.float32)
            f = np.asarray(feats[i], np.float32)  # rows = HISTORY_FEATURES
            T = len(c)
            if T < t_min + F + 1:
                continue
            ts = list(range(t_min, T - F + 1, step))[:max_points]
            if not ts:
                continue
            m, s = predict_many(model, c, f[0], ts, fov=f[1], n200=f[2], expr=f[3], device="cpu")
            A = np.stack([c[t : t + F] for t in ts])            # (P, F) actual future
            P = np.stack([np.full(F, c[t - 1]) for t in ts])    # persistence baseline
            err = np.abs(m - A)
            se = (m - A) ** 2
            pse = (P - A) ** 2
            s = np.maximum(s, 1e-3)
            nll = 0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * ((A - m) / s) ** 2
            cstd = float(meta_df.loc[int(i), "cnr_std"]) or 1e-6
            late = slice(F // 2, F)  # long-horizon window where forecasting (not echo) matters
            rows.append(
                {
                    "cell": int(i),
                    "mae": float(err.mean()),
                    "nmae": float(err.mean() / cstd),
                    "nll": float(nll.mean()),
                    "skill": float(1 - se.mean() / max(pse.mean(), 1e-9)),
                    "skill_late": float(1 - se[:, late].mean() / max(pse[:, late].mean(), 1e-9)),
                    "mae_by_h": [float(x) for x in err.mean(0)],
                    "pers_by_h": [float(x) for x in np.abs(P - A).mean(0)],
                }
            )
        return rows

    return (eval_cells,)


@app.cell
def _(GROUPS, MODELS, MODEL_DIR, eval_cells, load_experiment, mo, pl):
    import os as _os

    _missing = [n for _, n in MODELS if not _os.path.isdir(_os.path.join(MODEL_DIR, n))]
    mo.stop(
        bool(_missing),
        mo.md(f"**Model bundle(s) not found under `{MODEL_DIR}`:** {_missing}. "
              f"Is the results mount attached?"),
    )

    _rows = []
    for _label, _name in MODELS:
        _model = load_experiment(_os.path.join(MODEL_DIR, _name)).reconstruct_model()
        _model.to("cpu").eval()
        for (_scope, _sp), _idx in GROUPS.items():
            for _r in eval_cells(_model, _idx):
                _r.update({"model": _label, "scope": _scope, "split": _sp})
                _rows.append(_r)
        print(f"done: {_label}")
    res = pl.DataFrame(_rows)
    print(f"{len(res)} per-cell rows")
    return (res,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Held-out (val) skill by scope and model
    """)
    return


@app.cell
def _(pl, res):
    summary_val = (
        res.filter(pl.col("split") == "val")
        .group_by("model", "scope")
        .agg(
            n=pl.len(),
            mae=pl.col("mae").mean().round(4),
            nmae=pl.col("nmae").mean().round(3),
            skill_persist_all_h=pl.col("skill").mean().round(3),
            skill_persist_late_h=pl.col("skill_late").mean().round(3),
            nll=pl.col("nll").mean().round(3),
        )
        .sort("model", "scope")
    )
    summary_val
    return (summary_val,)


@app.cell
def _(qplot, summary_val):
    qplot(
        summary_val,
        x="nmae",
        y="model",
        color="scope",
        mark="bar",
        opacity=0.85,
        title="Held-out nMAE by scope (lower = better; scope-fair)",
        subtitle="Equal-height pairs ⇒ the model is even-handed across scopes",
        width=640,
        height=220,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Overfit check — train vs. val gap, per scope

    For each model×scope, mean MAE on training cells vs. held-out val cells. A
    big gap that appears on **one** scope only = that scope is being memorized /
    overfit. A gap present on both is generic (capacity/regularization), not
    scope-specific.
    """)
    return


@app.cell
def _(pl, res):
    overfit = (
        res.group_by("model", "scope", "split")
        .agg(mae=pl.col("mae").mean())
        .pivot(values="mae", index=["model", "scope"], on="split")
        .with_columns((pl.col("val") - pl.col("train")).alias("gap"))
        .with_columns(
            (pl.col("gap") / pl.col("train")).round(3).alias("gap_frac"),
            pl.col("train").round(4),
            pl.col("val").round(4),
            pl.col("gap").round(4),
        )
        .sort("model", "scope")
    )
    overfit
    return (overfit,)


@app.cell
def _(overfit, pl, qplot):
    _long = overfit.unpivot(
        on=["train", "val"], index=["model", "scope"], variable_name="split", value_name="mae"
    ).with_columns((pl.col("model") + " · " + pl.col("scope")).alias("model_scope"))
    qplot(
        _long,
        x="mae",
        y="model_scope",
        color="split",
        mark="bar",
        opacity=0.8,
        title="Train vs. val MAE (gap = overfitting)",
        subtitle="A one-sided train↔val gap flags scope-specific overfit",
        width=660,
        height=320,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Error growth over the prediction horizon
    """)
    return


@app.cell
def _(pl, qplot, res):
    _val = res.filter(pl.col("split") == "val")
    _model_h = (
        _val.explode("mae_by_h")
        .with_columns(h=(pl.col("mae_by_h").cum_count().over(["model", "scope", "cell"])).cast(pl.Int32))
        .group_by("model", "scope", "h")
        .agg(mae=pl.col("mae_by_h").mean())
    )
    # Persistence is model-independent: build it from one model's rows so it appears
    # once per scope as a reference line the model curves cross over.
    _first = res.get_column("model")[0]
    _pers_h = (
        _val.filter(pl.col("model") == _first)
        .explode("pers_by_h")
        .with_columns(h=(pl.col("pers_by_h").cum_count().over(["scope", "cell"])).cast(pl.Int32))
        .group_by("scope", "h")
        .agg(mae=pl.col("pers_by_h").mean())
        .with_columns(model=pl.lit("persistence (baseline)"))
    )
    horizon = pl.concat([_model_h, _pers_h.select(_model_h.columns)]).sort("model", "scope", "h")
    qplot(
        horizon,
        x="h",
        y="mae",
        color="model",
        mark="line",
        facet_col="scope",
        title="Val MAE vs. horizon step (1…F), by scope",
        subtitle="Persistence wins early; the model earns value where its curve dips below the baseline",
        width=360,
        height=300,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Finer breakdown — per experiment (val nMAE)

    Scope is coarse; this splits jungfrau into its constituent experiments so a
    scope-level average isn't hiding one bad protocol (e.g. is it *Niesen*, or is
    it also the high-dose BO runs?).
    """)
    return


@app.cell
def _(meta_df, pl, qplot, res):
    _exp = pl.from_pandas(
        meta_df.reset_index()[["index", "original_experiment_name"]].rename(
            columns={"index": "cell"}
        )
    )
    per_exp = (
        res.filter(pl.col("split") == "val")
        .join(_exp, on="cell", how="left")
        .group_by("model", "original_experiment_name")
        .agg(n=pl.len(), nmae=pl.col("nmae").mean().round(3))
        .sort("original_experiment_name", "model")
    )
    qplot(
        per_exp,
        x="nmae",
        y="original_experiment_name",
        color="model",
        mark="bar",
        opacity=0.8,
        title="Val nMAE per experiment, by model",
        width=680,
        height=420,
    )
    return (per_exp,)


@app.cell
def _(per_exp):
    per_exp
    return


@app.cell
def _(mo):
    mo.md("""
    ## How to read it

    - **`all` models even-handed?** Compare the jungfrau vs Niesen **nMAE** bars
      (panel 1) for `all (fixed-F)` / `all (multilen)`. Similar heights ⇒ the
      combined model treats both scopes comparably; a tall Niesen bar ⇒ it's
      paying for the OOD dose regime.
    - **Overfit** (panel 2): a train↔val gap that shows up on **one** scope only
      is scope-specific memorization. Watch the small-N Niesen train set.
    - **`niesen-only` is the control:** it should be strong on Niesen but poor on
      jungfrau (never saw it). If an `all` model matches `niesen-only` *on Niesen*
      while also handling jungfrau, the combined training isn't costing Niesen
      skill — the key question for keeping Niesen in the mix.
    - **Persistence is a strong short-horizon baseline.** None of these models beat
      naive persistence on aggregate MSE (`skill_persist_all_h < 0`) because holding
      the last CNR is near-unbeatable at h=1–2 on autocorrelated signals. The fair
      read is `skill_persist_late_h` (long horizon, where forecasting — not echo —
      matters) and panel 3, where the model curve dips below the persistence line at
      larger h. This is a model/metric property, not scope-specific.
    """)
    return


if __name__ == "__main__":
    app.run()
