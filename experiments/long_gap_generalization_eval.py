import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
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
    from optoerk.eval.history_predict import predict_many

    alt.data_transformers.disable_max_rows()
    return (
        alt,
        load_experiment,
        load_history_tracks,
        materials_path,
        mo,
        np,
        os,
        pl,
        plt,
        predict_many,
        qplot,
        results_write_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Long-gap generalization — can these models predict across real long gaps?

    The augmentation-ablation models (baseline, `prepend`, `wrap`, `both`) are trained
    on `all` with trajectories **≤200 frames** and same-cell self-wrap breaks **≤60
    frames**. The **`long`** experiment (48 h, ~2845 frames/cell, sparse stims) is a
    held-out probe with **genuinely long inter-stim gaps** — never in any training
    bundle, and far beyond the training trajectory length.

    The question isn't just "low error" (deep in a gap the cell sits at baseline and is
    trivially flat). It's: **when a stim arrives after a long quiescent gap, does the
    model still predict the ERK response** — and do the prepend/wrap augmentations
    help it do so? So we score error split by (i) frames since last stim and (ii)
    whether a stim lands in the prediction horizon (a "response" point).

    Context is **capped** at a fixed window (the models are full-history but `long` is
    ~14× their training length; a bounded window is both tractable and closer to the
    `wrap` regime). Baseline model works today; the three ablation bundles are picked
    up automatically once their SLURM runs finish.
    """)
    return


@app.cell
def _(os, results_write_path):
    # Eval budget + context cap (keep headless export tractable; raise for tighter numbers).
    CELLS_LONG = 60      # random `long` cells to score per model
    CONTEXT_CAP = 250    # bounded encoder window (frames) — long is ~2845/cell
    POINT_STRIDE = 40    # grid spacing of prediction points within a cell
    MAX_POINTS = 40      # cap prediction points per cell
    SEED = 0

    # Model bundles to compare, by name prefix. The latest matching run is used, so
    # the three ablation jobs are picked up automatically once trained. The digit
    # guard keeps `..._prepend` from matching `..._prepend_wrap`, and `optortk` from
    # matching `optortk_multilen`.
    _MODEL_PREFIXES = [
        ("baseline (0.5 wrap)", "seq2scal_history_optortk"),
        ("prepend", "seq2scal_history_all_prepend"),
        ("wrap (p=1)", "seq2scal_history_all_wrap"),
        ("both", "seq2scal_history_all_prepend_wrap"),
    ]

    def _find_latest(prefix):
        rp = results_write_path()
        if not os.path.isdir(rp):
            return None
        cands = []
        for d in os.listdir(rp):
            full = os.path.join(rp, d)
            loadable = os.path.exists(os.path.join(full, "bundle.pt")) or os.path.exists(
                os.path.join(full, "checkpoints", "bundle.pt")
            )
            if not (os.path.isdir(full) and loadable):
                continue
            if d.startswith(prefix + "_") and len(d) > len(prefix) + 1 and d[len(prefix) + 1].isdigit():
                cands.append(d)
        return os.path.join(rp, sorted(cands)[-1]) if cands else None

    MODELS = [(label, _find_latest(pfx)) for label, pfx in _MODEL_PREFIXES]
    MODELS_FOUND = [(l, p) for l, p in MODELS if p is not None]
    return (
        CELLS_LONG,
        CONTEXT_CAP,
        MAX_POINTS,
        MODELS,
        MODELS_FOUND,
        POINT_STRIDE,
        SEED,
    )


@app.cell
def _(MODELS, MODELS_FOUND, mo, os):
    mo.stop(
        not MODELS_FOUND,
        mo.md("**No model bundles found** under the results mount. Is it attached?"),
    )
    _lines = [
        f"- **{l}** → `{os.path.basename(p)}`" if p else f"- **{l}** → _not trained yet (skipped)_"
        for l, p in MODELS
    ]
    mo.md("### Models found\n" + "\n".join(_lines))
    return


@app.cell
def _(load_history_tracks, materials_path, mo, np):
    _long_path = materials_path("dataset_long.parquet")
    mo.stop(
        not _long_path.exists(),
        mo.md(
            f"**`{_long_path.name}` not built.** Run "
            "`uv run python -m optoerk.data.preprocessing long` first."
        ),
    )
    long_cnr, long_feats, long_cond, long_meta = load_history_tracks(_long_path)
    _T = np.array([len(c) for c in long_cnr])
    print(f"long: {len(long_cnr)} cells | T median {int(np.median(_T))} min {int(_T.min())} max {int(_T.max())}")
    return long_cnr, long_feats


@app.cell
def _(mo):
    mo.md("""
    ## 1. How long are the gaps? (why this is the hard case)
    """)
    return


@app.cell
def _(long_cnr, long_feats, np, pl, qplot):
    # Quiet-period length = run of consecutive unstimulated frames (u_t==0). Stims
    # come in ramp bursts, so raw inter-stim diffs are mostly 1; the meaningful gap
    # is the quiet stretch BETWEEN bursts. Pooled over cells.
    _gaps = []
    for i in range(len(long_cnr)):
        m = np.asarray(long_feats[i], np.float32)[0] > 0  # stim mask
        pad = np.concatenate([[True], m, [True]])
        edges = np.diff(pad.astype(int))
        starts = np.where(edges == -1)[0]  # stim -> quiet
        ends = np.where(edges == 1)[0]     # quiet -> stim
        _gaps.extend((ends - starts).tolist())
    _gaps = np.asarray(_gaps)
    gap_df = pl.DataFrame({"gap_frames": _gaps})
    print(
        f"quiet gaps: n={len(_gaps)}  median={int(np.median(_gaps))}  "
        f"p90={int(np.quantile(_gaps,0.9))}  max={int(_gaps.max())}  "
        f"|  {float((_gaps > 60).mean())*100:.0f}% exceed the 60-frame training break"
    )
    qplot(
        gap_df.filter((pl.col("gap_frames") > 1) & (pl.col("gap_frames") <= 400)),
        x="gap_frames",
        mark="hist",
        bins=60,
        title="Quiet-gap length between stims in `long` (frames = minutes)",
        subtitle="Training self-wrap breaks maxed at 60; a large share of long's quiet gaps run well past that",
        width=680,
        height=280,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Scoring each model on `long`
    """)
    return


@app.cell
def _(CONTEXT_CAP, MAX_POINTS, POINT_STRIDE, np, predict_many):
    def eval_long(model, cnr, feats, idx_list, *, t_min=10, cap=CONTEXT_CAP):
        """Per-prediction-point rows on `long`, with gap context recorded.

        For each point t: MAE over the horizon, `recency` (frames since last stim),
        and `stim_future` (a stim lands in [t, t+F)). Context is capped at `cap`.
        """
        F = model.cfg.future_len
        rows = []
        for i in idx_list:
            c = np.asarray(cnr[i], np.float32)
            f = np.asarray(feats[i], np.float32)  # [u_t, fov, n200, expr]
            T = len(c)
            if T < t_min + F + 1:
                continue
            stim_idx = np.where(f[0] > 0)[0]
            stim_set = set(stim_idx.tolist())
            # Burst onsets: first stim after a quiet stretch — predicting AT the onset
            # tests "can it predict the response to a stim arriving after a long gap".
            onsets = [int(s) for s in stim_idx if (s - 1) not in stim_set and t_min <= s <= T - F]
            grid = list(range(t_min, T - F + 1, POINT_STRIDE))
            if len(grid) > MAX_POINTS:
                grid = list(np.asarray(grid)[np.linspace(0, len(grid) - 1, MAX_POINTS).astype(int)])
            onset_set = set(onsets)
            ts = sorted(set(grid) | onset_set)  # onsets always kept
            m, s = predict_many(model, c, f[0], ts, fov=f[1], n200=f[2], expr=f[3],
                                device="cpu", cap=cap)
            cstd = float(np.std(c)) or 1e-6
            for k, t in enumerate(ts):
                A = c[t : t + F]
                prev = stim_idx[stim_idx < t]
                recency = int(t - prev.max()) if len(prev) else int(t)
                rows.append(
                    {
                        "cell": int(i),
                        "t": int(t),
                        "mae": float(np.abs(m[k] - A).mean()),
                        "nmae": float(np.abs(m[k] - A).mean() / cstd),
                        "recency": recency,          # frames since last stim (= gap at an onset)
                        "is_onset": int(t) in onset_set,  # post-gap burst onset
                        "stim_future": bool((f[0][t : t + F] > 0).any()),
                    }
                )
        return rows

    return (eval_long,)


@app.cell
def _(
    CELLS_LONG,
    MODELS_FOUND,
    SEED,
    eval_long,
    load_experiment,
    long_cnr,
    long_feats,
    np,
    pl,
):
    _rng = np.random.default_rng(SEED)
    _idx = np.arange(len(long_cnr))
    _rng.shuffle(_idx)
    _idx = _idx[:CELLS_LONG]

    _rows = []
    for _label, _path in MODELS_FOUND:
        _model = load_experiment(_path).reconstruct_model()
        _model.to("cpu").eval()
        for _r in eval_long(_model, long_cnr, long_feats, _idx):
            _r["model"] = _label
            _rows.append(_r)
        print(f"done: {_label}")
    res = pl.DataFrame(_rows)
    print(f"{len(res)} prediction-point rows on long")
    return (res,)


@app.cell
def _(pl, res):
    summary = (
        res.group_by("model")
        .agg(
            n_points=pl.len(),
            mae=pl.col("mae").mean().round(4),
            nmae=pl.col("nmae").mean().round(3),
            mae_onset=pl.col("mae").filter(pl.col("is_onset")).mean().round(4),
            n_onset=pl.col("is_onset").sum(),
            mae_quiescent=pl.col("mae").filter(~pl.col("stim_future")).mean().round(4),
        )
        .sort("mae_onset")
    )
    summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. The key test — burst-onset accuracy vs. the preceding gap

    **Burst-onset** points (the first stim after a quiet stretch), MAE binned by the
    length of the quiet gap that preceded them. Predicting *at* the onset means the
    context is all quiescent baseline and the future fluence carries the incoming
    stim — so this is exactly "predict the ERK response to a stim arriving after a gap
    of length X." If a model generalizes, MAE stays **flat** as the preceding gap grows
    past the 60-frame training limit; a rising curve = it degrades once gaps exceed
    what it trained on.
    """)
    return


@app.cell
def _(alt, pl, res):
    _bins = [0, 10, 30, 60, 120, 300, 100000]
    _labels = ["0-10", "10-30", "30-60", "60-120", "120-300", "300+"]
    resp = (
        res.filter(pl.col("is_onset"))
        .with_columns(gap_bin=pl.col("recency").cut(breaks=_bins[1:-1], labels=_labels))
        .group_by("model", "gap_bin")
        .agg(mae=pl.col("mae").mean(), n=pl.len())
    )
    # cut() gives string labels; pass the numeric order to Altair (else it sorts
    # the nominal axis lexicographically: "3-10" after "120-300").
    alt.Chart(
        resp,
        title=alt.TitleParams(
            "Burst-onset MAE vs. preceding quiet-gap length (frames)",
            subtitle="Onset error is LOWER at long gaps (clean baseline) than short-gap re-stims",
        ),
    ).mark_line(point=True).encode(
        x=alt.X("gap_bin:N", sort=_labels, title="preceding gap (frames)"),
        y=alt.Y("mae:Q", title="MAE"),
        color=alt.Color("model:N", title="model"),
    ).properties(width=680, height=320)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Error vs. time since last stim (all points)
    """)
    return


@app.cell
def _(alt, pl, res):
    _bins = [0, 3, 10, 30, 60, 120, 300, 100000]
    _labels = ["0-3", "3-10", "10-30", "30-60", "60-120", "120-300", "300+"]
    recency_df = (
        res.with_columns(recency_bin=pl.col("recency").cut(breaks=_bins[1:-1], labels=_labels))
        .group_by("model", "recency_bin")
        .agg(mae=pl.col("mae").mean(), n=pl.len())
    )
    # Explicit numeric bin order — Altair would otherwise sort the nominal x axis
    # lexicographically ("3-10" landing between "120-300" and "30-60").
    alt.Chart(
        recency_df,
        title=alt.TitleParams(
            "MAE vs. frames since last stim (all prediction points)",
            subtitle="Deep in a gap the cell is flat baseline (easy); the spike at 3-10 is the rising response",
        ),
    ).mark_line(point=True).encode(
        x=alt.X("recency_bin:N", sort=_labels, title="frames since last stim"),
        y=alt.Y("mae:Q", title="MAE"),
        color=alt.Color("model:N", title="model"),
    ).properties(width=680, height=320)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Example: actual vs. predicted across a long gap
    """)
    return


@app.cell
def _(
    CONTEXT_CAP,
    MODELS_FOUND,
    SEED,
    load_experiment,
    long_cnr,
    long_feats,
    np,
    plt,
    predict_many,
):
    # Pick a long cell with real dynamics; overlay each model's 1-step-ahead track.
    _rng = np.random.default_rng(SEED + 7)
    _elig = [i for i in range(len(long_cnr)) if len(long_cnr[i]) > 400]
    _cell = int(max(_elig, key=lambda i: np.std(long_cnr[i]))) if _elig else int(np.argmax([np.std(c) for c in long_cnr]))
    _c = np.asarray(long_cnr[_cell], np.float32)
    _f = np.asarray(long_feats[_cell], np.float32)
    _T = len(_c)
    _ts = list(range(10, _T - 10, 3))

    _fig, _ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[3, 1])
    _ax[0].plot(np.arange(_T), _c, color="black", lw=1.2, label="actual CNR", alpha=0.8)
    for _label, _path in MODELS_FOUND:
        _model = load_experiment(_path).reconstruct_model().eval()
        _m, _ = predict_many(_model, _c, _f[0], _ts, fov=_f[1], n200=_f[2], expr=_f[3],
                             device="cpu", cap=CONTEXT_CAP)
        _ax[0].plot([t for t in _ts], _m[:, 0], lw=1.0, alpha=0.85, label=_label)
    _ax[0].set_ylabel("CNR (norm)")
    _ax[0].legend(fontsize=8, ncol=len(MODELS_FOUND) + 1)
    _ax[0].set_title(f"long cell {_cell} (T={_T}) — actual vs 1-step prediction, capped context={CONTEXT_CAP}")
    _stim = np.where(_f[0] > 0)[0]
    _ax[1].vlines(_stim, 0, 1, color="tab:red", lw=0.6)
    _ax[1].set_ylabel("stim")
    _ax[1].set_xlabel("frame (min)")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## How to read it

    - **Panel 3 is the answer to the question.** Flat response-MAE as the preceding
      gap grows past 60 frames ⇒ the model predicts post-long-gap responses as well
      as short-gap ones (generalizes). A rising curve ⇒ it degrades exactly where the
      gaps exceed the training self-wrap break (≤60).
    - **prepend vs baseline**: prepend teaches onset-from-rest, which is what a
      post-long-gap stim *is* (cell back at baseline, fresh rise). Expect it to help
      most in the ≥60-frame gap bins if the hypothesis holds.
    - **wrap (p=1) vs baseline**: pure self-wrap trains every sample on the bounded
      sliding window ending after a break — closer to this capped-context eval. Expect
      steadier behavior at long recency.
    - Cross-check panel 3 against the `mae_onset` column (table): that's the headline
      "can it predict the post-gap response" number; `mae_quiescent` should be small
      for everyone (flat baseline is easy) and isn't the discriminator.
    - Panel 5 is the sanity check: does the predicted track actually rise with the
      stims after long quiet stretches, or does it flatline / lag?
    """)
    return


if __name__ == "__main__":
    app.run()
