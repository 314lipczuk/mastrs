import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from hastyplot import qplot 

    alt.data_transformers.disable_max_rows()

    from optoerk.core.utils import REPO_ROOT, materials_path, results_write_path

    return (
        Path,
        REPO_ROOT,
        alt,
        materials_path,
        mo,
        mpl,
        np,
        pl,
        plt,
        results_write_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Thesis figures

    One notebook, one figure per section. Every figure is written to
    `FIG_DIR` as both PDF (for LaTeX) and PNG (for quick viewing), under a
    stable `figNN_slug` name so the thesis source can reference it directly.
    """)
    return


@app.cell
def _(REPO_ROOT, mpl, plt):
    FIG_DIR = REPO_ROOT / "thesis" / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Single place to define the look of every figure in the thesis.
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    def save_fig(fig, name: str) -> str:
        """Write `fig` to FIG_DIR as PDF + PNG. Returns the PDF path."""
        pdf = FIG_DIR / f"{name}.pdf"
        fig.savefig(pdf)
        fig.savefig(FIG_DIR / f"{name}.png")
        return str(pdf)

    mpl.rcParams["pdf.fonttype"] = 42  # embed real fonts, not paths
    str(FIG_DIR)
    return (save_fig,)


@app.cell
def _(mo):
    mo.md("""
    ## Figure 1 — dataset overview

    The full training bundle (`dataset_all.parquet`): every tracked cell from
    every stimulation experiment, in long format (one row per cell per frame).
    """)
    return


@app.cell(hide_code=True)
def _(materials_path, pl):
    DATASET = materials_path("dataset_all.parquet")

    # Long format: one row per cell per frame. Only the columns the figures need,
    # so the 6.6M-row bundle stays comfortable in memory.
    df = pl.read_parquet(
        DATASET,
        columns=[
            "original_experiment_name", "stim_condition", "cell_line",
            "uid", "fov", "particle", "frame", "time_min",
            "cnr_median_norm", "u_t", "m_t", "stim_power", "stim_exposure",
        ],
    )
    df.shape
    return (df,)


@app.cell(hide_code=True)
def _(df, pl):
    # Per-experiment composition, the numbers that go in the figure and its caption.
    dataset_summary = (
        df.group_by("original_experiment_name")
        .agg(
            pl.col("uid").n_unique().alias("cells"),
            pl.col("stim_condition").n_unique().alias("conditions"),
            (pl.col("frame").max() + 1).alias("frames"),
            pl.col("m_t").mean().alias("frac_stimulated"),
            pl.col("cnr_median_norm").median().alias("median_cnr"),
            pl.len().alias("rows"),
        )
        .sort("cells", descending=True)
        .rename({"original_experiment_name": "experiment"})
    )
    dataset_summary
    return (dataset_summary,)


@app.cell(hide_code=True)
def _():
    # Categorical hues, assigned in fixed order and never cycled. Validated
    # colourblind-safe (worst all-pairs CVD dE 9.2, normal-vision 24.0) against a
    # light print surface, which is the only surface a thesis figure ever has.
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
    INK = "#0b0b0b"
    MUTED = "#52514e"
    GRID = "#d9d8d4"
    STIM_BLUE = "#9ec5f0"  # light pulses, kept recessive behind the signal
    SERIES
    return GRID, INK, MUTED, SERIES, STIM_BLUE


@app.cell(hide_code=True)
def _(df, pl):
    # Deterministic choices so the figure is stable across reruns.
    # Three conditions spanning the stimulation range the dataset covers, from a
    # single pulse to near-continuous light.
    EXAMPLE_CONDITIONS = ["Single", "bo_osc_v8_c1", "Sustained"]

    # A representative cell: from the middle of the response-amplitude distribution
    # of one condition, so it is typical rather than cherry-picked.
    _amp = (
        df.filter(pl.col("stim_condition") == "bo_osc_v8_c1")
        .group_by("uid")
        .agg((pl.col("cnr_median_norm").max() - pl.col("cnr_median_norm").min()).alias("amp"))
        .sort("amp")
    )
    EXAMPLE_UID = _amp["uid"][_amp.height // 2]
    EXAMPLE_UID
    return EXAMPLE_CONDITIONS, EXAMPLE_UID


@app.cell(hide_code=True)
def _(
    EXAMPLE_CONDITIONS,
    EXAMPLE_UID,
    GRID,
    INK,
    MUTED,
    SERIES,
    STIM_BLUE,
    dataset_summary,
    df,
    np,
    pl,
    plt,
    save_fig,
):
    def _panel_composition(ax):
        """(a) How much data, and where it came from."""
        s = dataset_summary.sort("cells")
        y = np.arange(s.height)
        ax.barh(y, s["cells"], color=SERIES[0], height=0.68)
        ax.set_yticks(y, s["experiment"], fontsize=7)
        for yi, n in zip(y, s["cells"]):
            ax.text(n + 400, yi, f"{n:,}", va="center", fontsize=6.5, color=MUTED)
        ax.set_xlabel("tracked cells")
        ax.set_xlim(0, s["cells"].max() * 1.18)
        ax.set_title("a  Dataset composition", loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_example(ax):
        """(b) What a single training example actually looks like."""
        cell = (
            df.filter(pl.col("uid") == EXAMPLE_UID)
            .sort("frame")
            .select("frame", "cnr_median_norm", "m_t")
        )
        t = cell["frame"].to_numpy()
        cnr = cell["cnr_median_norm"].to_numpy()

        # One span per lit frame. `fill_between(..., where=)` would collapse these
        # to zero width, since most pulses are a single isolated frame.
        for _t in t[cell["m_t"].to_numpy() > 0]:
            ax.axvspan(_t - 0.5, _t + 0.5, color=STIM_BLUE, lw=0, zorder=0)

        ax.plot(t, cnr, color=INK, lw=1.4, zorder=2)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=":", zorder=1)

        ax.set_xlabel("time (min)")
        ax.set_ylabel("CNR (baseline-normalised)")
        ax.set_xlim(t.min(), t.max())
        ax.set_title("b  One cell, one trajectory", loc="left", fontweight="bold")
        ax.text(0.985, 0.06, "shaded = light pulse", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=MUTED)


    def _panel_distribution(ax):
        """(c) The dynamic range the model has to cover."""
        v = df["cnr_median_norm"].to_numpy()
        clip_hi = 4.0
        kept = v[v <= clip_hi]  # excluded, not piled onto the last bin
        ax.hist(kept, bins=120, color=SERIES[0], lw=0)
        med = float(np.median(v))
        ax.axvline(med, color=SERIES[1], lw=1.6)
        ax.text(med + 0.08, ax.get_ylim()[1] * 0.92, f"median {med:.2f}",
                fontsize=7, color=SERIES[1])
        ax.set_xlabel("CNR (baseline-normalised)")
        ax.set_ylabel("frames")
        ax.set_xlim(0, clip_hi)
        ax.set_title("c  Response distribution", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.985, 0.72, f"{(v > clip_hi).mean():.2%} above {clip_hi:g}, not shown",
                transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)


    def _panel_conditions(ax):
        """(d) Different light patterns drive visibly different population responses."""
        for colour, cond in zip(SERIES, EXAMPLE_CONDITIONS):
            g = (
                df.filter(pl.col("stim_condition") == cond)
                .group_by("frame")
                .agg(
                    pl.col("cnr_median_norm").median().alias("med"),
                    pl.col("cnr_median_norm").quantile(0.25).alias("lo"),
                    pl.col("cnr_median_norm").quantile(0.75).alias("hi"),
                    pl.col("m_t").mean().alias("duty"),
                )
                .sort("frame")
            )
            t = g["frame"].to_numpy()
            duty = float(g["duty"].mean())
            ax.fill_between(t, g["lo"], g["hi"], color=colour, alpha=0.15, lw=0)
            ax.plot(t, g["med"], color=colour, lw=1.8,
                    label=f"{cond}  ({duty:.0%} of frames lit)")

        ax.set_xlabel("time (min)")
        ax.set_ylabel("CNR (median, IQR shaded)")
        ax.set_title("d  Population response by condition", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig1, _axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    _panel_composition(_axes[0, 0])
    _panel_example(_axes[0, 1])
    _panel_distribution(_axes[1, 0])
    _panel_conditions(_axes[1, 1])
    fig1.tight_layout(pad=1.4, w_pad=2.4, h_pad=2.2)
    save_fig(fig1, "fig01_dataset_overview")
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Figure 8 — does the model predict?

    The headline claim: given a cell's entire past and a candidate future light
    dose, how well does the model forecast its CNR over the horizon the
    controller plans across?

    Evaluated on the **test split** — cells no run trained or early-stopped on —
    rebuilt with the same seed the training notebook used, so these numbers are
    the same ones stamped into the bundle.
    """)
    return


@app.cell(hide_code=True)
def _(Path, results_write_path):
    from torch.utils.data import DataLoader

    from optoerk.core.experiment import load_experiment
    from optoerk.data.history_data import load_history_tracks
    from optoerk.data.history_dataset import HistoryDataset, NormStats, collate_history, make_split
    from optoerk.eval.encoding_metrics import evaluate
    from optoerk.eval.history_predict import predict_many

    # The run the encoding study endorsed: expression as a plain encoder channel,
    # trained on the real mCitrine measurement. `raw` cnr_mode, so everything below
    # is in absolute cnr_median units (a resting cell sits near 0.75, not 1.0).
    BUNDLE_DIR = Path(results_write_path()) / "enc_a_channel_2026-08-06_18.04.20"

    bundle = load_experiment(str(BUNDLE_DIR))
    model = bundle.reconstruct_model()
    mcfg = model.cfg
    bundle.warnings, mcfg.cnr_mode, mcfg.future_len, list(mcfg.norm_channels)
    return (
        DataLoader,
        HistoryDataset,
        NormStats,
        bundle,
        collate_history,
        evaluate,
        load_experiment,
        load_history_tracks,
        make_split,
        mcfg,
        model,
        predict_many,
    )


@app.cell(hide_code=True)
def _(load_history_tracks, make_split, materials_path, mcfg):
    # Same load and split as training (seed 0), so `split["test"]` is the same set
    # of cells the bundle's stored metrics were computed on. ~1 min.
    FEATURES = [c for c in mcfg.norm_channels if c != "cnr"]

    hist_cnr, hist_feats, hist_cond, hist_meta = load_history_tracks(
        materials_path("dataset_all_mcitrine.parquet"),
        cnr_mode=mcfg.cnr_mode,
        features=FEATURES,
    )
    split = make_split(hist_cond, seed=0)
    {k: len(v) for k, v in split.items()}
    return FEATURES, hist_cnr, hist_cond, hist_feats, hist_meta, split


@app.cell(hide_code=True)
def _(
    DataLoader,
    HistoryDataset,
    NormStats,
    bundle,
    collate_history,
    evaluate,
    hist_cnr,
    hist_feats,
    mcfg,
    model,
    np,
    split,
):
    import torch

    norm_stats = NormStats(channels=list(mcfg.norm_channels),
                           mean=list(mcfg.norm_mean), std=list(mcfg.norm_std))

    _te = split["test"]
    test_ds = HistoryDataset(
        hist_cnr[_te], hist_feats[_te], np.arange(len(_te)), norm_stats,
        F=mcfg.future_len, t_min=10, p_concat=0.0,
        future_channels=mcfg.future_channels, seed=0,
    )
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_history)


    @torch.no_grad()
    def collect_predictions(model, loader):
        """Point forecast, truth, and last observed CNR per sample, in absolute CNR.

        `evaluate` already returns the aggregate metrics; this keeps the raw arrays so
        the figure can show the predicted-vs-observed cloud and score a persistence
        reference on exactly the same samples.
        """
        ci = list(model.cfg.norm_channels).index("cnr")
        cnr_mean, cnr_std = float(model.cfg.norm_mean[ci]), float(model.cfg.norm_std[ci])
        model.eval()
        pred, true, last = [], [], []
        for batch in loader:
            ctx_, lens = batch["ctx"], batch["lengths"]
            pi, mu, _sigma = model(ctx_, lens, batch["fut_flu"])
            point = (pi * mu).sum(-1)                       # (B, F) standardized
            pred.append(point.numpy())
            true.append(batch["tgt"].numpy())
            # the last CNR the encoder actually saw — the persistence forecast
            rows = torch.arange(ctx_.shape[0])
            last.append(ctx_[rows, (lens - 1).clamp(min=0), ci].numpy())
        to_abs = lambda a: np.concatenate(a) * cnr_std + cnr_mean
        return to_abs(pred), to_abs(true), to_abs(last)


    # HistoryDataset draws a fresh prediction origin per pass (its generator advances
    # so repeated epochs see different windows). Re-seeding before each pass pins both
    # passes to the SAME prediction points, so the point forecasts and the calibration
    # metrics describe one eval set rather than two draws — and the figure is stable
    # across notebook reruns.
    test_ds.reseed(0)
    pred_abs, true_abs, last_abs = collect_predictions(model, test_dl)
    test_ds.reseed(0)
    test_metrics = evaluate(model, test_dl)

    # Persistence: "the cell stays where it is" — the reference an RMSE in CNR units
    # is meaningless without.
    persist_rmse = np.sqrt(((last_abs[:, None] - true_abs) ** 2).mean(axis=0))
    model_rmse = np.sqrt(((pred_abs - true_abs) ** 2).mean(axis=0))

    # The bundle's own number came from a different draw of origins, so it agrees to
    # sampling noise rather than exactly.
    {
        "n_samples": len(pred_abs),
        "rmse": float(test_metrics["rmse"]),
        "rmse_recomputed": float(np.sqrt(((pred_abs - true_abs) ** 2).mean())),
        "rmse_in_bundle": bundle.metrics["encoding"]["test_rmse"],
        "rmse_step1": float(model_rmse[0]),
        "rmse_step30": float(model_rmse[-1]),
        "persistence_step1": float(persist_rmse[0]),
        "persistence_step30": float(persist_rmse[-1]),
    }
    return (
        last_abs,
        model_rmse,
        persist_rmse,
        pred_abs,
        test_dl,
        test_ds,
        test_metrics,
        torch,
        true_abs,
    )


@app.cell(hide_code=True)
def _(hist_cnr, np, pred_abs, split, test_ds, true_abs):
    # The cell shown in panel (a), chosen by a stated rule rather than by eye.
    # The DataLoader is unshuffled, so sample i of `pred_abs` is cell `test_ds.idx[i]`.
    _err_per_sample = np.abs(pred_abs - true_abs).mean(axis=1)
    _cell_of_sample = np.asarray(test_ds.idx)

    _te_cnr = hist_cnr[split["test"]]
    _len = np.array([len(_te_cnr[j]) for j in _cell_of_sample])
    _amp = np.array([float(np.ptp(_te_cnr[j])) for j in _cell_of_sample])

    # Long enough to show several forecast origins, and a mid-range response so the
    # panel shows the model tracking real dynamics rather than a flat cell.
    _lo, _hi = np.quantile(_amp, [1 / 3, 2 / 3])
    _ok = np.where((_len >= 80) & (_amp >= _lo) & (_amp <= _hi))[0]
    # ...and, among those, the one with the median prediction error: typical, not best.

    _OFFSET = 14
    EXAMPLE_SAMPLE = int(_ok[np.argsort(_err_per_sample[_ok])[len(_ok) // 2]]) - _OFFSET
    EXAMPLE_CELL = int(_cell_of_sample[EXAMPLE_SAMPLE])

    {"n_eligible": len(_ok), "cell": EXAMPLE_CELL,
     "T": int(_len[EXAMPLE_SAMPLE]), "amplitude": float(_amp[EXAMPLE_SAMPLE]),
     "mae": float(_err_per_sample[EXAMPLE_SAMPLE])}
    return (EXAMPLE_CELL,)


@app.cell(hide_code=True)
def _(
    EXAMPLE_CELL,
    FEATURES,
    GRID,
    INK,
    MUTED,
    SERIES,
    STIM_BLUE,
    hist_cnr,
    hist_feats,
    mcfg,
    model,
    model_rmse,
    np,
    persist_rmse,
    plt,
    pred_abs,
    predict_many,
    save_fig,
    split,
    true_abs,
):
    CONTROL_HORIZON = 8   # what the live controller plans over; F=30 is what the model offers


    def _panel_forecasts(ax, ax_light):
        """(a) The model doing its job: forecast a horizon ahead, from real context."""
        j = EXAMPLE_CELL
        cnr = np.asarray(hist_cnr[split["test"]][j], np.float32)
        feats = np.asarray(hist_feats[split["test"]][j], np.float32)   # (K, T), FEATURES order
        chans = {name: feats[k] for k, name in enumerate(FEATURES)}
        u = chans["u_t"]
        T, F = len(cnr), mcfg.future_len

        origins = np.linspace(25, T - F, 3).astype(int)
        means, sigmas = predict_many(model, cnr, u, origins, channels=chans)

        ax.plot(np.arange(T), cnr, color=INK, lw=1.4, zorder=3, label="observed")
        for colour, t0, m, s in zip(SERIES, origins, means, sigmas):
            h = np.arange(t0, t0 + F)
            ax.fill_between(h, m - s, m + s, color=colour, alpha=0.22, lw=0, zorder=1)
            ax.plot(h, m, color=colour, lw=1.9, zorder=2)
            ax.plot([t0], [cnr[t0 - 1]], marker="o", ms=4.5, color=colour, zorder=4)
            ax_light.axvline(t0, color=colour, lw=0.9, alpha=0.9)

        ax.set_ylabel("CNR (absolute)")
        ax.set_xlim(0, T)
        ax.tick_params(labelbottom=False)
        ax.set_title("a  Forecasts from three origins on one held-out cell",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=7)
        ax.text(0.995, 0.03, "dot = forecast origin · band = ±1σ · horizon 30 min",
                transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)

        # The light input as a dose, not a binary: this protocol shortens its
        # inter-pulse interval over the run, which a full-height on/off shading turns
        # into an unreadable solid block.
        ax_light.bar(np.arange(T), u, width=1.0, color=STIM_BLUE, lw=0)
        ax_light.set_xlim(0, T)
        ax_light.set_ylim(0, float(u.max()) * 1.1)
        ax_light.set_ylabel("light\n(mJ/cm²)", fontsize=7)
        ax_light.set_xlabel("time (min)")
        ax_light.tick_params(labelsize=7)
        ax_light.spines["left"].set_visible(True)


    def _panel_horizon(ax):
        """(b) How the error grows with lead time, against a do-nothing reference."""
        steps = np.arange(1, len(model_rmse) + 1)
        ax.plot(steps, persist_rmse, color=SERIES[1], lw=1.8, label="persistence")
        ax.plot(steps, model_rmse, color=SERIES[0], lw=2.0, label="model")
        ax.axvline(CONTROL_HORIZON, color=MUTED, lw=0.8, ls="--")
        ax.annotate("control horizon", xy=(CONTROL_HORIZON, persist_rmse.max()),
                    xytext=(CONTROL_HORIZON + 1.2, persist_rmse.max() * 0.97),
                    fontsize=7, color=MUTED, va="top")

        ax.set_xlabel("forecast lead time (min)")
        ax.set_ylabel("RMSE (absolute CNR)")
        ax.set_xlim(1, len(model_rmse))
        ax.set_ylim(0, None)
        ax.set_title("b  Error vs lead time", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_scatter(ax):
        """(c) Predicted against observed at the lead time the controller acts on."""
        k = CONTROL_HORIZON - 1
        x, y = true_abs[:, k], pred_abs[:, k]
        lim = (0.4, float(np.quantile(np.concatenate([x, y]), 0.999)))
        ax.hexbin(x, y, gridsize=55, extent=(*lim, *lim), bins="log",
                  cmap="Blues", mincnt=1, linewidths=0)
        ax.plot(lim, lim, color=INK, lw=0.9, ls="--")

        r2 = 1 - ((y - x) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        ax.text(0.04, 0.94, f"$R^2$ = {r2:.3f}\nRMSE = {model_rmse[k]:.3f}",
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
        ax.set_xlabel("observed CNR")
        ax.set_ylabel("predicted CNR")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(f"c  {CONTROL_HORIZON} min ahead, {len(x):,} held-out cells",
                     loc="left", fontweight="bold")


    fig8 = plt.figure(figsize=(9.2, 6.6))
    # Nested: the trace and its light strip are one unit (tight), well separated from
    # the two summary panels below.
    _gs = fig8.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42,
                            left=0.085, right=0.98, top=0.94, bottom=0.09)
    _top = _gs[0].subgridspec(2, 1, height_ratios=[1.0, 0.26], hspace=0.08)
    _bot = _gs[1].subgridspec(1, 2, wspace=0.26)
    _ax_a = fig8.add_subplot(_top[0])
    _panel_forecasts(_ax_a, fig8.add_subplot(_top[1], sharex=_ax_a))
    _panel_horizon(fig8.add_subplot(_bot[0]))
    _panel_scatter(fig8.add_subplot(_bot[1]))
    save_fig(fig8, "fig08_model_performance")
    fig8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Figure 8b — success and failure, side by side

    A variant of panel (a) that splits the two behaviours into two cells instead
    of hoping one cell shows both.

    The failure the model has is specific: it cannot anticipate **endogenous ERK
    pulses**, which by construction are not in the light input. So the selector
    scores forecast error separately over horizon frames where the light is
    **off** and where it is **on**.

    "Off" is defined per cell as `u_t` at that cell's own minimum — not `u_t == 0`.
    Only 1,905 of 7,237 test cells ever reach exact zero; the rest sit on a
    nonzero fluence floor, so a global zero test would call most cells
    permanently lit.
    """)
    return


@app.cell(hide_code=True)
def _(
    FEATURES,
    hist_cnr,
    hist_feats,
    mcfg,
    model,
    np,
    pl,
    predict_many,
    split,
):
    ORIGIN_STRIDE = 5   # origins scanned per cell; 1 would be 5x the compute for no new shape


    def dark_mask(u):
        """Frames with no commanded light, per cell.

        Not `u == 0`: most experiments carry a nonzero fluence floor between pulses,
        so an absolute test marks those cells as lit for their whole run.
        """
        return u <= u.min() * 1.001 + 1e-9


    def cell_arrays(cell_idx):
        """(cnr, channel dict) for one test cell."""
        cnr = np.asarray(hist_cnr[split["test"]][cell_idx], np.float32)
        feats = np.asarray(hist_feats[split["test"]][cell_idx], np.float32)
        return cnr, {name: feats[k] for k, name in enumerate(FEATURES)}


    def origin_errors(cell_idx, origins):
        """Forecast, ±1σ, absolute error and darkness, for each origin of one cell."""
        cnr, chans = cell_arrays(cell_idx)
        F = mcfg.future_len
        means, sigmas = predict_many(model, cnr, chans["u_t"], origins, channels=chans)
        truth = np.stack([cnr[t:t + F] for t in origins])
        dark = np.stack([dark_mask(chans["u_t"])[t:t + F] for t in origins])
        return means, sigmas, np.abs(means - truth), dark


    def valid_origins(T, stride=ORIGIN_STRIDE, t_min=15):
        return np.arange(t_min, T - mcfg.future_len + 1, stride)


    def score_forecast_split(cell_idx):
        """Error over dark vs lit horizon frames, scanned across the whole track.

        Scanning rather than sampling three fixed origins: whether a cell fails
        depends on *when* the forecast is launched, and three arbitrary launch points
        can miss the excursion entirely.
        """
        cnr, _chans = cell_arrays(cell_idx)
        origins = valid_origins(len(cnr))
        _m, _s, err, dark = origin_errors(cell_idx, origins)
        per_origin = err.mean(axis=1)
        return {
            "cell": cell_idx,
            "T": len(cnr),
            "amp": float(np.ptp(cnr)),
            "n_origins": len(origins),
            "err": float(err.mean()),
            "err_dark": float(err[dark].mean()) if dark.any() else np.nan,
            "err_lit": float(err[~dark].mean()) if (~dark).any() else np.nan,
            "n_dark": int(dark.sum()),
            "n_lit": int((~dark).sum()),
            "worst_origin": int(origins[per_origin.argmax()]),
            "worst_origin_err": float(per_origin.max()),
        }


    # Every test cell long enough to fit three forecast origins (~3 min).
    _cells = [j for j in range(len(hist_cnr[split["test"]]))
              if len(hist_cnr[split["test"]][j]) >= 80]
    forecast_scores = pl.DataFrame([score_forecast_split(j) for j in _cells])
    forecast_scores.describe()
    return (
        cell_arrays,
        dark_mask,
        forecast_scores,
        origin_errors,
        valid_origins,
    )


@app.cell(hide_code=True)
def _(forecast_scores, pl):
    # Two cells, two stated rules. Over the population the split barely separates
    # anything — median err_dark 0.058 vs err_lit 0.059 — so the failure cell is
    # explicitly a tail case, not a representative one, and is labelled as such.
    _elig = forecast_scores.filter((pl.col("n_dark") >= 10) & (pl.col("n_lit") >= 10))

    # Typical: median overall error among cells that actually respond (amplitude
    # above the population median), so the panel is not a flat cell predicted flat.
    _resp = _elig.filter(pl.col("amp") >= _elig["amp"].median()).sort("err")
    CELL_TYPICAL = int(_resp["cell"][_resp.height // 2])

    # Characteristic failure: tracks normally under light (err_lit at or below the
    # median) but diverges in the dark. Taken at the 95th percentile of err_dark
    # rather than the maximum, so it is a representative tail case and not one
    # pathological track.
    _fail = (
        _elig.filter(pl.col("err_lit") <= _elig["err_lit"].median())
        .sort("err_dark")
    )
    CELL_FAILURE = int(_fail["cell"][int(0.95 * (_fail.height - 1))])

    pl.concat([
        _elig.filter(pl.col("cell") == CELL_TYPICAL).with_columns(pl.lit("typical").alias("role")),
        _elig.filter(pl.col("cell") == CELL_FAILURE).with_columns(pl.lit("failure").alias("role")),
    ])
    return CELL_FAILURE, CELL_TYPICAL


@app.cell(hide_code=True)
def _(
    CELL_FAILURE,
    CELL_TYPICAL,
    GRID,
    INK,
    MUTED,
    SERIES,
    STIM_BLUE,
    cell_arrays,
    dark_mask,
    forecast_scores,
    mcfg,
    model,
    mpl,
    np,
    origin_errors,
    pl,
    plt,
    predict_many,
    save_fig,
    valid_origins,
):
    def spread_origins(T, k=2):
        """Evenly spaced launch points — what a reader would pick by default.

        Kept non-overlapping and to the same count as the failure panel, so the two
        panels differ in *which* origins are shown and nothing else. Three fans over
        a 90-frame track cannot avoid overlapping at a 30-min horizon, and the
        overlap is what made the panel unreadable.
        """
        F = mcfg.future_len
        while k > 1 and (T - F - 25) / (k - 1) < F:
            k -= 1
        return np.linspace(25, T - F, k).astype(int)


    def worst_origins(cell_idx, k=3):
        """The k highest-error launch points, kept non-overlapping.

        Greedy: take the worst, exclude everything within one horizon of it, repeat.
        Without the separation the "worst" origins are the same excursion several
        times, one frame apart. Fewer than k come back when the track is too short to
        hold k non-overlapping horizons — a 90-frame track fits two 30-min fans, not
        three — and that is left as-is rather than padded with overlapping ones.
        """
        cnr, _ = cell_arrays(cell_idx)
        origins = valid_origins(len(cnr))
        _m, _s, err, _d = origin_errors(cell_idx, origins)
        per_origin = err.mean(axis=1)

        chosen, order = [], np.argsort(-per_origin)
        for i in order:
            if len(chosen) == k:
                break
            if all(abs(origins[i] - c) >= mcfg.future_len for c in chosen):
                chosen.append(int(origins[i]))
        return np.array(sorted(chosen))


    def forecast_pair_panel(ax, ax_light, cell_idx, origins, title, subtitle):
        """One cell: observed CNR with forecast fans at `origins`, and its light input."""
        cnr, chans = cell_arrays(cell_idx)
        u, T = chans["u_t"], len(cnr)
        means, sigmas = predict_many(model, cnr, u, origins, channels=chans)

        ax.plot(np.arange(T), cnr, color=INK, lw=1.4, zorder=3)
        for colour, t0, m, s in zip(SERIES, origins, means, sigmas):
            h = np.arange(t0, t0 + mcfg.future_len)
            ax.fill_between(h, m - s, m + s, color=colour, alpha=0.22, lw=0, zorder=1)
            ax.plot(h, m, color=colour, lw=1.9, zorder=2)
            ax.plot([t0], [cnr[t0 - 1]], marker="o", ms=4.5, color=colour, zorder=4)

        ax.set_ylabel("CNR (absolute)")
        ax.set_xlim(0, T)
        ax.tick_params(labelbottom=False)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.97, 0.06, subtitle, transform=ax.transAxes, ha="right",
                fontsize=7.5, color=MUTED)

        # Dark frames shaded on the input strip, so "off" is visible as the rule
        # defines it — the cell's own fluence floor, which is often not zero.
        ax_light.fill_between(np.arange(T), 0, 1, where=dark_mask(u), step="mid", lw=0,
                              color=GRID, alpha=0.6, zorder=0,
                              transform=mpl.transforms.blended_transform_factory(
                                  ax_light.transData, ax_light.transAxes))
        ax_light.bar(np.arange(T), u, width=1.0, color=STIM_BLUE, lw=0, zorder=1)
        ax_light.set_xlim(0, T)
        ax_light.set_ylim(0, float(u.max()) * 1.1)
        ax_light.set_ylabel("light\n(mJ/cm²)", fontsize=7)
        ax_light.set_xlabel("time (min)")
        ax_light.tick_params(labelsize=7)


    _row = lambda c: forecast_scores.filter(pl.col("cell") == c).row(0, named=True)
    _t, _f = _row(CELL_TYPICAL), _row(CELL_FAILURE)

    fig8b = plt.figure(figsize=(9.4, 4.4))
    _gs8b = fig8b.add_gridspec(2, 2, height_ratios=[1.0, 0.3], hspace=0.1, wspace=0.22,
                               left=0.075, right=0.985, top=0.88, bottom=0.16)
    _a0 = fig8b.add_subplot(_gs8b[0, 0])
    forecast_pair_panel(
        _a0, fig8b.add_subplot(_gs8b[1, 0], sharex=_a0), CELL_TYPICAL,
        spread_origins(_t["T"]),
        "a  Typical cell, evenly spaced origins",
        f"error  dark {_t['err_dark']:.3f} · lit {_t['err_lit']:.3f}",
    )
    _a1 = fig8b.add_subplot(_gs8b[0, 1])
    forecast_pair_panel(
        _a1, fig8b.add_subplot(_gs8b[1, 1], sharex=_a1), CELL_FAILURE,
        worst_origins(CELL_FAILURE),
        "b  Failure case, its worst origins",
        f"error  dark {_f['err_dark']:.3f} · lit {_f['err_lit']:.3f}",
    )
    fig8b.text(0.5, 0.008, "grey = light off (this cell's own fluence floor) · "
               "dot = forecast origin · band = ±1σ · horizon 30 min",
               ha="center", fontsize=7, color=MUTED)
    save_fig(fig8b, "fig08b_forecast_examples")
    fig8b
    return


@app.cell(hide_code=True)
def _(forecast_scores, last_abs, np, pl, pred_abs, test_metrics, true_abs):
    from scipy.stats import spearmanr

    # Two claims about the failure mode, each reduced to a number.
    #
    # 1. dark vs lit: is the model worse when it cannot see the cause of a change?
    _diff = forecast_scores["err_lit"] - forecast_scores["err_dark"]
    darklit = {
        "err_dark": float(forecast_scores["err_dark"].median()),
        "err_lit": float(forecast_scores["err_lit"].median()),
        "frac_cells_lit_worse": float((_diff > 0).mean()),
        "n_cells": forecast_scores.height,
        "n_forecasts": int(forecast_scores["n_origins"].sum()),
    }

    # 2. shrinkage: does the model pull every cell toward the population response?
    #    Signed error against the cell's level BEFORE the forecast window. Using the
    #    past (an encoder input) rather than the target avoids the regression
    #    artefact that would appear if the same variable sat on both sides.
    _signed = (pred_abs - true_abs).mean(axis=1)
    _past = last_abs
    _rho = spearmanr(_past, _signed)

    _q = np.quantile(_past, np.linspace(0, 1, 11))
    _q[-1] += 1e-9
    _dec = np.clip(np.digitize(_past, _q[1:-1]), 0, 9)
    bias_by_past_decile = pl.DataFrame({
        "decile": np.arange(1, 11),
        "past_cnr": [float(_past[_dec == i].mean()) for i in range(10)],
        "signed_err": [float(_signed[_dec == i].mean()) for i in range(10)],
        "actual_change": [float((true_abs.mean(axis=1) - _past)[_dec == i].mean()) for i in range(10)],
        "predicted_change": [float((pred_abs.mean(axis=1) - _past)[_dec == i].mean()) for i in range(10)],
    })
    shrinkage = {
        "spearman": float(_rho.statistic),
        "p": float(_rho.pvalue),
        "max_abs_bias": float(bias_by_past_decile["signed_err"].abs().max()),
        "rmse": float(test_metrics["rmse"]),
    }
    bias_by_past_decile
    return darklit, shrinkage, spearmanr


@app.cell(hide_code=True)
def _(darklit, mo, shrinkage):
    mo.md(f"""
    ### What panel (b) is, and is not, evidence for

    **The hypothesis the selector was built on.** The model's inputs are the cell's
    own past and the commanded light. Endogenous ERK pulses — the "idling motor"
    that fires without stimulation — are in neither. So the expectation was that
    forecast error concentrates in frames where the light is **off**, and the
    selector scores dark and lit horizon frames separately.

    **It does not.** Over {darklit['n_cells']:,} test cells and
    {darklit['n_forecasts']:,} forecasts, median error is
    **{darklit['err_dark']:.4f}** in the dark against **{darklit['err_lit']:.4f}**
    under light — and the lit half is the worse one in
    **{darklit['frac_cells_lit_worse']:.1%}** of cells. The split separates nothing,
    and what separation exists points the wrong way. Two consequences:

    * "the model cannot anticipate endogenous pulses" is not a claim this data
      supports, and should not be written as one;
    * panel (b) is therefore a **tail case selected by a near-null criterion** — the
      95th percentile of `err_dark` among cells whose `err_lit` is at or below the
      median. It is a real cell and a real failure, but it is not evidence that
      darkness is what causes failures.

    **The other reading, and how far it actually goes.** Panel (b) *looks* like
    shrinkage toward the population: the model undershoots this cell's plateau and
    over-predicts its decay once light stops, as though forecasting a more typical
    cell than this one. That reading comes from looking at the panel, so it was
    tested directly — signed error against the cell's CNR **before** the forecast
    window (an encoder input, not the target, so the same variable does not sit on
    both sides and manufacture the correlation).

    The direction holds and the population effect is negligible: Spearman
    **{shrinkage['spearman']:+.3f}** (p = {shrinkage['p']:.0e}, significant only
    because n is large), and the largest per-decile bias is
    **{shrinkage['max_abs_bias']:.4f}** against an RMSE of
    **{shrinkage['rmse']:.3f}** — about
    {100 * shrinkage['max_abs_bias'] / shrinkage['rmse']:.0f}% of a typical error.
    The table above shows why: predicted change tracks actual change closely in
    every decile, over-reverting the top decile by 0.009 CNR and over-rising the
    bottom by 0.006.

    So the model is **very nearly unbiased across response levels**, and the ~0.09
    CNR plateau mismatch in panel (b) is roughly ten times the population-level
    bias. Shrinkage is a per-cell effect in the tail, not a systematic pull.

    **What this leaves.** Panel (b) is honest as an illustration that individual
    cells fail badly while the population statistics look fine — which is the same
    heterogeneity story as the expression-decile result, where error climbs ~30%
    from the lowest to the highest decile even though the covariate is real,
    correctly measured and used. Naming a single mechanism for panel (b) is not yet
    warranted by anything measured here.
    """)
    return


@app.cell(hide_code=True)
def _(cell_arrays, dark_mask, hist_cnr, np, pl, split):
    QUIET_LAG = 20   # min after a pulse before a dark frame counts as "not still responding"
    EXCURSION = 1.2  # x resting baseline; what counts as an activation


    def spontaneous_activity(cell_idx):
        """Activity in dark frames, per cell, relative to that cell's resting baseline.

        Two windows, because they are not equally clean:
          * pre-stim — every frame before the cell's first pulse. Uncontaminated, but
            short (median 10 min), so it bounds the *rate* of spontaneous pulses only
            weakly: a 15-30 min ERK pulse can fall outside a 10 min window.
          * post-pulse quiet — dark frames at least QUIET_LAG after the last pulse.
            Longer, but ERK relaxes over 15-30 min, so it still contains stimulation
            decay and reads as an upper bound rather than a spontaneous rate.
        """
        cnr, chans = cell_arrays(cell_idx)
        u = chans["u_t"]
        lit = ~dark_mask(u)
        if not lit.any():
            return None
        first = int(np.argmax(lit))
        if first < 5:
            return None                       # no usable pre-stim window

        base = float(np.median(cnr[:first]))
        since = np.arange(len(cnr)) - np.maximum.accumulate(np.where(lit, np.arange(len(cnr)), -10**6))
        quiet = (~lit) & (since >= QUIET_LAG)

        return {
            "cell": cell_idx,
            "n_prestim": first,
            "n_quiet": int(quiet.sum()),
            "prestim_rel_sd": float(np.std(cnr[:first] / base)),
            "prestim_frac_active": float((cnr[:first] / base >= EXCURSION).mean()),
            "quiet_frac_active": float((cnr[quiet] / base >= EXCURSION).mean()) if quiet.any() else np.nan,
        }


    spontaneous = pl.DataFrame([
        r for r in (spontaneous_activity(j) for j in range(len(hist_cnr[split["test"]])))
        if r is not None
    ])
    starvation = {
        "n_cells": spontaneous.height,
        "median_prestim_frames": float(spontaneous["n_prestim"].median()),
        "prestim_rel_sd": float(spontaneous["prestim_rel_sd"].median()),
        "prestim_frac_active": float(spontaneous["prestim_frac_active"].mean()),
        "quiet_frac_active": float(spontaneous["quiet_frac_active"].median()),
    }
    spontaneous.describe()
    return QUIET_LAG, starvation


@app.cell(hide_code=True)
def _(QUIET_LAG, mo, starvation):
    mo.md(f"""
    #### Why the dark/lit split finds nothing: the cells were starved

    The cells in every one of these experiments were held in **serum starvation**,
    specifically to suppress spontaneous ERK activations. If that worked, dark
    frames contain almost nothing for the model to miss — and the null above is
    explained by the experimental design rather than by anything about the model.

    Measured on {starvation['n_cells']:,} test cells, relative to each cell's own
    resting baseline:

    | window | quantity | value |
    |---|---|---|
    | pre-stimulation | relative SD of CNR | **{starvation['prestim_rel_sd']:.3f}** |
    | pre-stimulation | frames ≥ 1.2× baseline | **{starvation['prestim_frac_active']:.2%}** |
    | ≥ {QUIET_LAG} min after last pulse | frames ≥ 1.2× baseline | {starvation['quiet_frac_active']:.1%} |

    Before any light reaches them, cells sit within **±2.4%** of their own baseline
    and cross 1.2× on well under one percent of frames. That is a flat line, not an
    idling motor. The starvation did what it was for.

    Two caveats on how far this goes:

    * the pre-stimulation window is short — median
      {starvation['median_prestim_frames']:.0f} min — and an ERK pulse lasts 15–30
      min, so this bounds the *amplitude* of spontaneous activity well and its
      *rate* only weakly;
    * the post-pulse figure is **not** a spontaneous rate. ERK relaxes over 15–30
      min, so frames {QUIET_LAG} min after a pulse still carry stimulation decay; it
      is an upper bound, and its being ~7× the pre-stimulation figure is what one
      would expect from decay alone.

    This resolves the earlier null and changes what it means. The dark/lit split did
    not fail to find the endogenous-pulse failure mode — **there were almost no
    endogenous pulses in this data to find**. The disturbance the MPC framing treats
    as central was suppressed by design in every training experiment, so nothing
    here tests the model against it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Figure 9 — is the uncertainty honest?

    The controller consumes σ, not just the mean: an MDN whose σ means nothing is
    a point predictor with extra steps. Two separate properties matter, and a
    model can have either without the other:

    * **calibration** — do the stated intervals cover at their nominal rate?
    * **sharpness / discrimination** — does σ rise *where the errors actually are*?
      A model can be perfectly calibrated by emitting the same wide σ everywhere,
      which is useless to a controller choosing between dose plans.
    """)
    return


@app.cell(hide_code=True)
def _(bundle, model, np, test_dl, test_ds, torch):
    from scipy.stats import norm


    @torch.no_grad()
    def collect_calibration(model, loader):
        """PIT, predictive sd, point forecast and target — per sample, per horizon step.

        The head is a 3-component mixture, so calibration is measured with the **exact
        mixture CDF** rather than a Gaussian approximation of it: PIT = F(y) should be
        Uniform(0,1) for a calibrated predictive distribution, whatever its shape. The
        `cov68`/`cov95` stored in the bundle instead z-score against the mixture's
        total variance, which is a Gaussian stand-in for a distribution free to be
        skewed or bimodal.
        """
        ci = list(model.cfg.norm_channels).index("cnr")
        cnr_std = float(model.cfg.norm_std[ci])
        model.eval()
        pits, sds, points, tgts = [], [], [], []
        for batch in loader:
            pi, mu, sigma = model(batch["ctx"], batch["lengths"], batch["fut_flu"])
            tgt = batch["tgt"]
            point = (pi * mu).sum(-1)
            var = (pi * (sigma ** 2 + (mu - point.unsqueeze(-1)) ** 2)).sum(-1)

            # F(y) = sum_k pi_k * Phi((y - mu_k) / sigma_k)
            z = (tgt.unsqueeze(-1) - mu) / sigma.clamp_min(1e-6)
            pits.append((pi * torch.from_numpy(norm.cdf(z.numpy()))).sum(-1).numpy())
            sds.append((var.clamp_min(1e-12).sqrt() * cnr_std).numpy())
            points.append(point.numpy())
            tgts.append(tgt.numpy())
        return {
            "pit": np.concatenate(pits),                     # (N, F) in [0, 1]
            "sd": np.concatenate(sds),                       # (N, F) absolute CNR
            "err": np.abs(np.concatenate(points) - np.concatenate(tgts)) * cnr_std,
        }


    test_ds.reseed(0)
    calib = collect_calibration(model, test_dl)

    # Central-interval coverage implied by the PIT, at every level rather than two.
    CAL_LEVELS = np.linspace(0.02, 0.98, 49)
    pit_coverage = np.array([
        float((np.abs(calib["pit"] - 0.5) <= lvl / 2).mean()) for lvl in CAL_LEVELS
    ])
    {
        "cov68_pit": float(np.interp(0.68, CAL_LEVELS, pit_coverage)),
        "cov95_pit": float(np.interp(0.95, CAL_LEVELS, pit_coverage)),
        "cov68_gaussian_in_bundle": bundle.metrics["encoding"]["cov68"],
        "cov95_gaussian_in_bundle": bundle.metrics["encoding"]["cov95"],
    }
    return CAL_LEVELS, calib, norm, pit_coverage


@app.cell(hide_code=True)
def _(
    CAL_LEVELS,
    GRID,
    INK,
    MUTED,
    SERIES,
    calib,
    norm,
    np,
    pit_coverage,
    plt,
    save_fig,
    spearmanr,
):
    def _panel_reliability(ax):
        """(a) Do the stated intervals cover, and does the Gaussian stand-in mislead?"""
        z_abs = calib["err"] / np.maximum(calib["sd"], 1e-9)
        gauss_cov = np.array([
            float((z_abs <= norm.ppf((1 + lvl) / 2)).mean()) for lvl in CAL_LEVELS
        ])

        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.plot(CAL_LEVELS, gauss_cov, color=SERIES[1], lw=1.8,
                label="Gaussian z-score")
        ax.plot(CAL_LEVELS, pit_coverage, color=SERIES[0], lw=2.2,
                label="exact mixture CDF")
        for lvl, colour in ((0.68, SERIES[2]), (0.95, SERIES[2])):
            ax.plot([lvl], [np.interp(lvl, CAL_LEVELS, pit_coverage)], marker="o",
                    ms=5, color=colour, zorder=4)

        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("a  Reliability", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=7)
        ax.text(0.97, 0.06, "above the line = under-confident",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


    def _panel_pit(ax):
        """(b) The same question without a reference distribution: is PIT uniform?"""
        ax.hist(calib["pit"].ravel(), bins=40, range=(0, 1), color=SERIES[0], lw=0,
                density=True)
        ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
        ax.set_xlabel("PIT  =  F(observed)")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1)
        # Zoomed: a calibrated PIT sits at 1.0 everywhere, so the whole diagnostic
        # lives in a narrow band and a 0-based axis would show a flat blue block.
        ax.set_ylim(0.6, 1.2)
        ax.set_title("b  Probability integral transform", loc="left", fontweight="bold")
        ax.text(0.5, 0.055, "flat = calibrated · ∪ = over-confident · ∩ = under-confident",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.text(0.5, 0.94, "note: y-axis zoomed on 1.0", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.5, color=MUTED)


    def _panel_coverage_horizon(ax):
        """(c) Calibration is a per-lead-time property, not one number."""
        steps = np.arange(1, calib["pit"].shape[1] + 1)
        for lvl, colour in ((0.68, SERIES[0]), (0.95, SERIES[1])):
            cov = (np.abs(calib["pit"] - 0.5) <= lvl / 2).mean(axis=0)
            ax.plot(steps, cov, color=colour, lw=1.9, label=f"{lvl:.0%} interval")
            ax.axhline(lvl, color=colour, lw=0.8, ls="--", alpha=0.6)

        ax.set_xlabel("forecast lead time (min)")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(1, len(steps))
        ax.set_ylim(0.5, 1.0)
        ax.set_title("c  Coverage vs lead time", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="center right", fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_sharpness(ax):
        """(d) Does σ know where the errors are? The property MPC actually uses."""
        sd, err = calib["sd"].ravel(), calib["err"].ravel()
        q = np.quantile(sd, np.linspace(0, 1, 11))
        q[-1] += 1e-9
        b = np.clip(np.digitize(sd, q[1:-1]), 0, 9)
        mean_sd = np.array([sd[b == i].mean() for i in range(10)])
        rmse_bin = np.array([np.sqrt((err[b == i] ** 2).mean()) for i in range(10)])

        lim = (0, max(mean_sd.max(), rmse_bin.max()) * 1.08)
        ax.plot(lim, lim, color=MUTED, lw=0.9, ls="--")
        ax.plot(mean_sd, rmse_bin, color=SERIES[0], lw=1.8, marker="o", ms=5)

        rho = spearmanr(sd, err).statistic
        ax.text(0.05, 0.93, f"Spearman(σ, |error|) = {rho:.3f}", transform=ax.transAxes,
                fontsize=7.5, color=INK, va="top")
        ax.set_xlabel("predicted σ  (decile mean)")
        ax.set_ylabel("realized RMSE")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title("d  Sharpness: does σ track the error?", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig9, _ax9 = plt.subplots(2, 2, figsize=(8.8, 7.0))
    _panel_reliability(_ax9[0, 0])
    _panel_pit(_ax9[0, 1])
    _panel_coverage_horizon(_ax9[1, 0])
    _panel_sharpness(_ax9[1, 1])
    fig9.tight_layout(pad=1.3, w_pad=2.6, h_pad=2.4)
    save_fig(fig9, "fig09_calibration")
    fig9
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Figure 10 — how much of the past does the model use?

    One model, one eval path, one thing changed: how many frames of the cell's
    own past the encoder is allowed to see. `predict_many(..., cap=k)` keeps the
    last **k** frames and discards everything before, so "full true history" and
    "fresh start, no memory" are two points on a single curve rather than two
    different models.

    This deliberately replaces an old-model-vs-new-model comparison. Those two
    differ in architecture, horizon (10 vs 30), `cnr_mode` and training bundle at
    once, so a difference between them localizes to nothing.

    Two caveats that belong in the caption:

    * `cap = 1` is **not** "no information" — the encoder still sees the current
      CNR, which is the single strongest predictor. The curve therefore measures
      what the past adds *beyond the present*, which is the question worth asking.
    * training used `t_min = 10`, so contexts shorter than 10 frames were never
      seen during training. Points below that are an extrapolation and are marked
      as such.
    """)
    return


@app.cell(hide_code=True)
def _(cell_arrays, hist_cnr, mcfg, model, np, pl, predict_many, split):
    MEMORY_CAPS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60]
    MEMORY_TMIN = 10          # training floor; caps below this are out of distribution
    MEMORY_STRIDE = 10        # every Nth eligible cell, for a smooth curve in ~1 min


    def memory_sweep_rows(cell_idx):
        """One forecast per context cap, from the latest valid origin of one cell.

        The origin is fixed at `T - F` so the cell has the deepest possible history
        available; the only thing varying across rows is how much of it the encoder
        is permitted to read.
        """
        cnr, chans = cell_arrays(cell_idx)
        T, F = len(cnr), mcfg.future_len
        origin = T - F
        truth = cnr[origin:origin + F]
        last_obs = float(cnr[origin - 1])

        out = []
        for cap in MEMORY_CAPS:
            means, _sig = predict_many(model, cnr, chans["u_t"], [origin],
                                       channels=chans, cap=cap)
            out.append({
                "cell": cell_idx,
                "cap": cap,
                "err": float(np.abs(means[0] - truth).mean()),
                "pred_mean": float(means[0].mean()),
                "true_mean": float(truth.mean()),
                "last_obs": last_obs,
            })
        return out


    _elig = [j for j in range(len(hist_cnr[split["test"]]))
             if len(hist_cnr[split["test"]][j]) >= max(MEMORY_CAPS) + mcfg.future_len]
    _sample = _elig[::MEMORY_STRIDE]
    memory_scores = pl.DataFrame([r for j in _sample for r in memory_sweep_rows(j)])

    # Per cap: forecast error, and how well the model tracks each cell's *change*
    # from its last observed value. The change is where cell identity lives — at
    # cap=1 the model knows the current level and little else about this cell.
    memory_curve = (
        memory_scores
        .with_columns(
            (pl.col("pred_mean") - pl.col("last_obs")).alias("pred_delta"),
            (pl.col("true_mean") - pl.col("last_obs")).alias("true_delta"),
        )
        .group_by("cap")
        .agg(
            pl.col("err").mean().alias("mae"),
            pl.corr(pl.col("pred_delta"), pl.col("true_delta"), method="spearman").alias("delta_rho"),
            pl.len().alias("n"),
        )
        .sort("cap")
    )
    memory_curve
    return MEMORY_CAPS, MEMORY_TMIN, memory_curve


@app.cell(hide_code=True)
def _(
    CELL_TYPICAL,
    GRID,
    INK,
    MEMORY_CAPS,
    MEMORY_TMIN,
    MUTED,
    SERIES,
    STIM_BLUE,
    cell_arrays,
    dark_mask,
    mcfg,
    memory_curve,
    model,
    mpl,
    np,
    plt,
    predict_many,
    save_fig,
):
    SHOW_CAPS = [1, 10, 60]   # fresh start · training floor · full history


    def _panel_memory_example(ax, ax_light):
        """(a) The same cell, the same origin, three context budgets."""
        cnr, chans = cell_arrays(CELL_TYPICAL)
        u, T, F = chans["u_t"], len(cnr), mcfg.future_len
        origin = T - F

        ax.plot(np.arange(T), cnr, color=INK, lw=1.4, zorder=3, label="observed")
        for colour, cap in zip(SERIES, SHOW_CAPS):
            m, s = predict_many(model, cnr, u, [origin], channels=chans, cap=cap)
            h = np.arange(origin, origin + F)
            ax.fill_between(h, m[0] - s[0], m[0] + s[0], color=colour, alpha=0.2, lw=0, zorder=1)
            ax.plot(h, m[0], color=colour, lw=1.9, zorder=2,
                    label=f"cap {cap}" + (" (full)" if cap == max(SHOW_CAPS) else ""))
            # The span of past this forecast was allowed to read, at a fixed height in
            # axes coordinates — `get_ylim()` shifts as the panel fills, which stacked
            # these at three accidental heights.
            _y = 0.03 + 0.045 * SHOW_CAPS.index(cap)
            ax.plot([origin - cap, origin], [_y, _y], color=colour, lw=2.5,
                    solid_capstyle="butt", alpha=0.9, zorder=4,
                    transform=mpl.transforms.blended_transform_factory(
                        ax.transData, ax.transAxes))

        ax.axvline(origin, color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel("CNR (absolute)")
        ax.set_xlim(0, T)
        ax.tick_params(labelbottom=False)
        ax.set_title("a  One cell, one origin, three context budgets",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
        # Placed in the empty band between the resting baseline and the oscillation
        # troughs — the only region of this panel nothing is drawn in.
        ax.text(0.40, 0.24, "bars = how much past each forecast could read\n"
                "future light is given to all three",
                transform=ax.transAxes, ha="center", fontsize=6.5, color=MUTED)

        ax_light.fill_between(np.arange(T), 0, 1, where=dark_mask(u), step="mid", lw=0,
                              color=GRID, alpha=0.6, zorder=0,
                              transform=mpl.transforms.blended_transform_factory(
                                  ax_light.transData, ax_light.transAxes))
        ax_light.bar(np.arange(T), u, width=1.0, color=STIM_BLUE, lw=0, zorder=1)
        ax_light.set_xlim(0, T)
        ax_light.set_ylim(0, float(u.max()) * 1.1)
        ax_light.set_ylabel("light\n(mJ/cm²)", fontsize=7)
        ax_light.set_xlabel("time (min)")
        ax_light.tick_params(labelsize=7)


    def _ood_band(ax):
        ax.axvspan(0.9, MEMORY_TMIN, color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.text(MEMORY_TMIN * 1.15, ax.get_ylim()[1], "shorter than any\ncontext seen in training",
                fontsize=6.5, color=MUTED, va="top")


    def _panel_memory_error(ax):
        """(b) Forecast error against how much past the encoder may read."""
        c = memory_curve
        ax.plot(c["cap"], c["mae"], color=SERIES[0], lw=2.0, marker="o", ms=4)
        ax.set_xscale("log")
        ax.set_xticks(MEMORY_CAPS, [str(k) for k in MEMORY_CAPS], fontsize=7)
        ax.set_xlabel("context available to the encoder (min)")
        ax.set_ylabel("forecast MAE (absolute CNR)")
        ax.set_ylim(0, float(c["mae"].max()) * 1.12)
        ax.set_title("b  Error vs available history", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _ood_band(ax)
        _lo, _hi = float(c["mae"].min()), float(c["mae"].max())
        ax.annotate(f"{_hi / _lo:.1f}x", xy=(60, _lo), xytext=(26, _hi * 0.62),
                    fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))


    def _panel_memory_identity(ax):
        """(c) Does the past tell the model which cell this is?

        Correlation between predicted and actual *change* from the last observed
        value. The level is already known at cap=1; the change is where the cell's
        own responsiveness has to come from.
        """
        c = memory_curve
        ax.plot(c["cap"], c["delta_rho"], color=SERIES[2], lw=2.0, marker="o", ms=4)
        ax.set_xscale("log")
        ax.set_xticks(MEMORY_CAPS, [str(k) for k in MEMORY_CAPS], fontsize=7)
        ax.set_xlabel("context available to the encoder (min)")
        ax.set_ylabel("Spearman(predicted Δ, actual Δ)")
        ax.set_ylim(0.6, 1.0)
        ax.set_title("c  Tracking each cell's own change", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _ood_band(ax)


    fig10 = plt.figure(figsize=(9.4, 6.6))
    _gs10 = fig10.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.42,
                               left=0.085, right=0.98, top=0.93, bottom=0.09)
    _t10 = _gs10[0].subgridspec(2, 1, height_ratios=[1.0, 0.26], hspace=0.08)
    _b10 = _gs10[1].subgridspec(1, 2, wspace=0.28)
    _a10 = fig10.add_subplot(_t10[0])
    _panel_memory_example(_a10, fig10.add_subplot(_t10[1], sharex=_a10))
    _panel_memory_error(fig10.add_subplot(_b10[0]))
    _panel_memory_identity(fig10.add_subplot(_b10[1]))
    save_fig(fig10, "fig10_memory_ablation")
    fig10
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Figure 7 — same light, different cells

    The central difficulty of the whole project, and the thing that justifies
    per-cell control at all. Every cell in `bo_osc_v8_c1` received an **identical**
    light pattern — twelve pulses, five minutes apart, frames 10 to 65 — so any
    spread in the response is the cells, not the input.

    The question that decides whether per-cell control can help is not *how large*
    the spread is but *whether it is stable*. Spread that is measurement noise
    cannot be controlled against; spread that is a persistent property of each cell
    can be inferred from its past and acted on — which is exactly what the encoder
    does in Figure 10.
    """)
    return


@app.cell(hide_code=True)
def _(FEATURES, hist_cnr, hist_cond, hist_feats, hist_meta, np, pl, spearmanr):
    HET_CONDITION = "bo_osc_v8_c1"
    HET_PULSES = np.arange(10, 70, 5)     # the fixed stimulation frames of this protocol
    HET_BASELINE = 10                     # frames before the first pulse
    HET_WINDOW = 5                        # frames scored after each pulse
    HET_LEN = 90                          # common track length for this protocol
    HET_MIN_PER_FOV = 50                  # cells needed before a field is scored
    HET_NONRESPONDER = 0.10               # sustained elevation below this = no response


    def heterogeneity_arrays(condition):
        """Baseline-normalised traces, per-pulse responses and expression, per cell.

        Baseline-normalised because a cell's resting CNR offset is not what is being
        asked about: the question is how much the same light *moves* each cell.
        """
        idx = np.where(hist_cond == condition)[0]
        # Full-length tracks only: a common time axis is needed for the heatmap and
        # for pulse-by-pulse comparison across cells.
        idx = np.array([i for i in idx if len(hist_cnr[i]) >= HET_LEN])
        traces = np.stack([np.asarray(hist_cnr[i], np.float32)[:HET_LEN] for i in idx])
        base = np.median(traces[:, :HET_BASELINE], axis=1, keepdims=True)
        base = np.where(np.abs(base) < 1e-6, 1.0, base)
        norm = traces / base

        # Rise attributable to each pulse: peak in the window after it, minus the
        # value just before it. Per pulse, so within-cell repeatability is measurable.
        rises = np.stack([
            norm[:, p:p + HET_WINDOW].max(axis=1) - norm[:, p - 1]
            for p in HET_PULSES
        ], axis=1)                                   # (n_cells, n_pulses)

        expr = np.array([np.asarray(hist_feats[i], np.float32)[FEATURES.index("optortk_expr")][0]
                         for i in idx])
        return {
            "idx": idx, "norm": norm, "rises": rises, "expr": expr,
            "sustained": norm[:, HET_PULSES[0]:HET_PULSES[-1] + HET_WINDOW].mean(axis=1) - 1.0,
        }


    het = heterogeneity_arrays(HET_CONDITION)



    # How much of the spread does the measured covariate account for?
    het_expr_rho = float(spearmanr(het["expr"], het["sustained"]).statistic)
    _q = np.quantile(het["expr"], np.linspace(0, 1, 11))
    _q[-1] += 1e-9
    het_expr_bin = np.clip(np.digitize(het["expr"], _q[1:-1]), 0, 9)
    het_expr_var_explained = float(
        1 - np.mean([het["sustained"][het_expr_bin == i].var() for i in range(10)])
        / het["sustained"].var()
    )

    # Expression rank comes from `add_optortk_expression`, which ranks WITHIN an
    # imaging session (`original_experiment_name`), not within a field. So the
    # relationship has to be checked within FOV: a pooled correlation could otherwise
    # be a plate effect — fields differing in both expression and delivered light.
    het_fov = hist_meta["fov"].to_numpy()[het["idx"]]

    _by_fov = []
    for _f in np.unique(het_fov):
        _m = het_fov == _f
        if _m.sum() < HET_MIN_PER_FOV:
            continue
        _e, _s = het["expr"][_m], het["sustained"][_m]
        _q = np.quantile(_e, np.linspace(0, 1, 11))
        _q[-1] += 1e-9
        _b = np.clip(np.digitize(_e, _q[1:-1]), 0, 9)
        _by_fov.append({
            "fov": int(_f), "n": int(_m.sum()),
            "rho": float(spearmanr(_e, _s).statistic),
            "mean_expr": float(_e.mean()), "mean_resp": float(_s.mean()),
            "decile_median": [float(np.median(_s[_b == i])) if (_b == i).any() else np.nan
                              for i in range(10)],
        })

    het_fov_df = pl.DataFrame(_by_fov)
    het_rho_pooled = float(spearmanr(het["expr"], het["sustained"]).statistic)
    het_rho_within = float(het_fov_df["rho"].mean())
    het_rho_between = float(spearmanr(het_fov_df["mean_expr"], het_fov_df["mean_resp"]).statistic)

    # Deciles on the SESSION-level rank, which is the interpretable axis: every cell
    # here comes from one experiment (`bo_v8`), so the ranks are mutually comparable.
    # Re-ranking inside each field would instead plot "bright for this field", and
    # decile 10 of a dim field is not the same receptor level as decile 10 of a bright
    # one. The field-level check stays as a correlation, not as the axis.
    _qs = np.quantile(het["expr"], np.linspace(0, 1, 11))
    _qs[-1] += 1e-9
    het_session_bin = np.clip(np.digitize(het["expr"], _qs[1:-1]), 0, 9)
    het_session_decile = np.array([
        [np.median(het["sustained"][het_session_bin == i]),
         np.quantile(het["sustained"][het_session_bin == i], 0.25),
         np.quantile(het["sustained"][het_session_bin == i], 0.75)]
        for i in range(10)
    ])

    # Same binning applied field by field, so each field's own trend is visible on the
    # same axis. Deciles a field does not populate are left as gaps rather than filled.
    _fov_curves = []
    for _f in np.unique(het_fov):
        _m = het_fov == _f
        if _m.sum() < HET_MIN_PER_FOV:
            continue
        _row = [np.median(het["sustained"][_m & (het_session_bin == i)])
                if (_m & (het_session_bin == i)).sum() >= 5 else np.nan
                for i in range(10)]
        _fov_curves.append(_row)
    het_fov_curves = np.array(_fov_curves)

    # --- Absolute CNR, not fold change -------------------------------------------
    # Expression predicts the BASELINE too (rho -0.24; median baseline falls 0.593 to
    # 0.467 across deciles), so a fold change divides by a term that is itself moving
    # with the x axis. Absolute CNR avoids that, and it is the quantity the controller
    # actually targets: this run's live setpoint was 1.2 in the same units.
    HET_SETPOINT = 1.2

    _raw = np.stack([np.asarray(hist_cnr[i], np.float32)[:HET_LEN] for i in het["idx"]])
    het_baseline = np.median(_raw[:, :HET_BASELINE], axis=1)
    het_achieved = _raw[:, HET_PULSES[0]:HET_PULSES[-1] + HET_WINDOW].mean(axis=1)

    het_achieved_decile = np.array([
        [np.median(het_achieved[het_session_bin == i]),
         np.quantile(het_achieved[het_session_bin == i], 0.25),
         np.quantile(het_achieved[het_session_bin == i], 0.75),
         float((het_achieved[het_session_bin == i] >= HET_SETPOINT).mean())]
        for i in range(10)
    ])
    het_reach_frac = float((het_achieved >= HET_SETPOINT).mean())
    het_rho_baseline = float(spearmanr(het["expr"], het_baseline).statistic)
    het_rho_achieved = float(spearmanr(het["expr"], het_achieved).statistic)


    # --- Does expression add anything beyond simply watching the cell? ------------
    # The data-side version of a model ablation. If a couple of minutes of observed
    # response already contains what expression would have told you, then neither the
    # model nor the controller needs the covariate -- it needs to look.
    HET_LATE = (50, 70)          # the response being predicted
    HET_OBS_LENGTHS = [3, 5, 8, 10, 15, 20, 25, 30]


    def _r2(y, *cols):
        """R^2 of y on the given predictors, all rank-transformed (monotone, robust)."""
        from scipy.stats import rankdata
        X = np.column_stack([rankdata(c) for c in cols] + [np.ones(len(y))])
        yr = rankdata(y)
        beta, *_ = np.linalg.lstsq(X, yr, rcond=None)
        resid = yr - X @ beta
        return float(1 - resid.var() / yr.var())


    _norm = het["norm"]
    _late = _norm[:, HET_LATE[0]:HET_LATE[1]].mean(axis=1)
    het_incremental = []
    for _k in HET_OBS_LENGTHS:
        _obs = _norm[:, HET_PULSES[0]:HET_PULSES[0] + _k].mean(axis=1)
        het_incremental.append({
            "obs_min": _k,
            "r2_watch": _r2(_late, _obs),
            "r2_watch_plus_expr": _r2(_late, _obs, het["expr"]),
            "r2_expr_only": _r2(_late, het["expr"]),
        })
    het_incremental = pl.DataFrame(het_incremental)

    # Cells that barely move: a distinct mode in panel (b), not a tail.
    het_nonresponder = het["sustained"] < HET_NONRESPONDER
    het_nonresp_frac = float(het_nonresponder.mean())
    het_nonresp_expr = float(np.median(het["expr"][het_nonresponder]))
    het_resp_expr = float(np.median(het["expr"][~het_nonresponder]))

    {
        "n_cells": len(het["idx"]),
        "fovs_scored": het_fov_df.height,
        "sustained_p10": float(np.quantile(het["sustained"], 0.10)),
        "sustained_median": float(np.median(het["sustained"])),
        "sustained_p90": float(np.quantile(het["sustained"], 0.90)),
        "rho_pooled": het_rho_pooled,
        "rho_within_fov": het_rho_within,
        "rho_between_fov": het_rho_between,
        "nonresponder_frac": het_nonresp_frac,
    }
    return (
        HET_NONRESPONDER,
        HET_PULSES,
        het,
        het_incremental,
        het_nonresp_expr,
        het_nonresp_frac,
        het_resp_expr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Work in progress
    receptor level plot is not clear; Should be per experiment (or perhaps FOV?). The Y axis choice is also debatable.

    What is better?
    - max cnr?
    - median baseline
    - some kind of sensitivity metric? `stimulation seconds` * `time_activated`?
    - Or loss of sensitivity over time? Q1 median rise from 1s of exposure to Q4 ... ?

    Receptor level check in theory should answer 'how much light do you need', and in turn also 'how quickly can you climb', but not necessarily 'how high can you go' (debatable, cause it's possible that 'how high can you go' is mostly limited by the feedbacks - then quick rise might help us here as well).

    The sad part is that no matter what I do here, it will be confouded by the experiment.
    But taking FOV-level is no better, since then I will have little datapoints in each group - and so group can easily not reperesent diversity i want.
    Yet another thing is the possibility of confoundign factor at highly populated FOVS - the network effect - that cells that are in close proximity will be overall more resiliant / less prone to stimulation.
    """)
    return


@app.cell(hide_code=True)
def _(
    GRID,
    HET_NONRESPONDER,
    HET_PULSES,
    INK,
    MUTED,
    SERIES,
    STIM_BLUE,
    het,
    het_incremental,
    het_nonresp_expr,
    het_nonresp_frac,
    het_resp_expr,
    np,
    plt,
    save_fig,
):
    def _panel_het_heatmap(ax):
        """(a) Identical input, seven thousand different outputs."""
        order = np.argsort(het["sustained"])
        img = het["norm"][order]
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                       vmin=0.6, vmax=3.2, interpolation="nearest",
                       extent=(0, img.shape[1], 0, img.shape[0]))
        for p in HET_PULSES:
            ax.plot([p], [img.shape[0] * 1.012], marker="v", ms=3.2,
                    color=STIM_BLUE, clip_on=False)
        ax.set_xlabel("time (min)          ▼ = light pulse")
        ax.set_ylabel(f"cells, sorted by response  (n = {img.shape[0]:,})")
        ax.set_yticks([])
        ax.set_title("a  One condition, one light pattern", loc="left", fontweight="bold")
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("CNR / own baseline", fontsize=7)
        cb.ax.tick_params(labelsize=7)


    def _panel_het_distribution(ax):
        """(b) The spread, and the non-responding mode inside it."""
        s = het["sustained"]
        lo_x, hi_x = -0.5, 3.5
        kept = s[(s >= lo_x) & (s <= hi_x)]        # excluded, not clipped
        ax.hist(kept, bins=90, color=SERIES[0], lw=0)
        ax.axvspan(lo_x, HET_NONRESPONDER, color=MUTED, alpha=0.12, lw=0)
        for q, style in ((0.10, ":"), (0.50, "-"), (0.90, ":")):
            ax.axvline(np.quantile(s, q), color=SERIES[1], lw=1.4, ls=style)

        ax.text(0.97, 0.94,
                f"p10  {np.quantile(s, 0.10):.2f}\nmedian  {np.median(s):.2f}\n"
                f"p90  {np.quantile(s, 0.90):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
        # Anchored to the shaded band but written out over the empty right-hand tail,
        # where nothing is drawn.
        ax.annotate(f"non-responders \u2014 {het_nonresp_frac:.0%} of cells\n"
                    f"median expression rank {het_nonresp_expr:.2f},\n"
                    f"against {het_resp_expr:.2f} for the rest",
                    xy=(HET_NONRESPONDER, ax.get_ylim()[1] * 0.55),
                    xytext=(1.45, ax.get_ylim()[1] * 0.72),
                    fontsize=7, color=MUTED, va="center",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.set_xlabel("sustained elevation over baseline")
        ax.set_ylabel("cells")
        ax.set_xlim(lo_x, hi_x)
        ax.set_title("b  Spread of response", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.97, 0.55, f"{(s > hi_x).mean():.1%} above {hi_x:g}, not shown",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


    def _panel_het_incremental(ax):
        """(d) Does knowing the receptor level beat simply watching the cell?

        The data-side counterpart of a model ablation: predict each cell's later
        response from its own observed early response, then ask what expression adds
        on top. No model involved — this is a property of the data.
        """
        d = het_incremental
        k = d["obs_min"].to_numpy()
        watch = d["r2_watch"].to_numpy()
        both = d["r2_watch_plus_expr"].to_numpy()
        expr_only = float(d["r2_expr_only"][0])

        ax.fill_between(k, watch, both, color=SERIES[1], alpha=0.35, lw=0,
                        label="what expression adds")
        ax.plot(k, both, color=SERIES[1], lw=1.4)
        ax.plot(k, watch, color=SERIES[0], lw=2.2, marker="o", ms=4,
                label="watching the cell")
        ax.axhline(expr_only, color=MUTED, lw=1.2, ls="--")
        ax.text(k[-1], expr_only - 0.045, "expression alone", ha="right",
                fontsize=7, color=MUTED)

        _cross = k[np.argmax(watch > expr_only)]
        ax.annotate(f"{_cross} min of watching\nis already worth more",
                    xy=(_cross, expr_only), xytext=(_cross + 1.5, 0.30),
                    fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.text(0.97, 0.28, f"at 10 min, expression adds {both[3] - watch[3]:+.3f}",
                transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)

        ax.set_xlabel("minutes of the cell's response observed")
        ax.set_ylabel("R² of the later response (frames 50–70)")
        ax.set_xlim(k.min(), k.max())
        ax.set_ylim(0, 1)
        ax.set_title("d  Watching beats knowing", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower right", fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig7 = plt.figure(figsize=(9.8, 6.6))
    _gs7 = fig7.add_gridspec(2, 2, width_ratios=[1.05, 1.0], hspace=0.5, wspace=0.28,
                             left=0.075, right=0.97, top=0.93, bottom=0.1)
    _panel_het_heatmap(fig7.add_subplot(_gs7[:, 0]))
    _panel_het_distribution(fig7.add_subplot(_gs7[0, 1]))
    _panel_het_incremental(fig7.add_subplot(_gs7[1, 1]))
    save_fig(fig7, "fig07_heterogeneity")
    fig7
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Methods figure — why optoRTK expression is ranked, not rescaled

    optoRTK expression is measured as **mCitrine**, one value per cell from a short
    reference acquisition. Raw values are not comparable between imaging sessions,
    so the pipeline converts them to a **percentile rank within each session**
    (`add_optortk_expression`, cohort = `original_experiment_name`).

    A rank is a strong choice: it discards magnitude, so rank 0.9 in one session
    is not the same number of receptors as rank 0.9 in another. This figure is the
    justification — it shows that the cheaper fixes cannot work, because the
    sessions differ in the **shape** of the distribution and not only its scale.
    """)
    return


@app.cell(hide_code=True)
def _(materials_path, np, pl):
    # One mCitrine value per cell, straight from the bundle the model trains on.
    _mc = (
        pl.read_parquet(materials_path("dataset_all_mcitrine.parquet"),
                        columns=["original_experiment_name", "uid", "mcitrine"])
        .group_by(["original_experiment_name", "uid"])
        .agg(pl.col("mcitrine").first())
        .drop_nulls()
        .filter(pl.col("mcitrine") > 0)
    )
    MC_MIN_CELLS = 150

    mcitrine_by_exp = {
        exp[0]: g["mcitrine"].to_numpy()
        for exp, g in _mc.group_by("original_experiment_name")
        if g.height >= MC_MIN_CELLS
    }
    # Largest session is the reference the others are compared against.
    MC_REFERENCE = max(mcitrine_by_exp, key=lambda k: len(mcitrine_by_exp[k]))

    mcitrine_summary = pl.DataFrame([
        {"experiment": k, "n": len(v), "median": float(np.median(v)),
         "p90_over_p10": float(np.quantile(v, 0.9) / np.quantile(v, 0.1)),
         "skew": float(((v - v.mean()) ** 3).mean() / v.std() ** 3)}
        for k, v in mcitrine_by_exp.items()
    ]).sort("median", descending=True)
    mcitrine_summary
    return MC_REFERENCE, mcitrine_by_exp, mcitrine_summary


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MC_REFERENCE,
    MUTED,
    SERIES,
    mcitrine_by_exp,
    mcitrine_summary,
    np,
    plt,
    save_fig,
):
    MC_Q = np.linspace(0.01, 0.99, 99)


    def _panel_median_scaled(ax):
        """(a) If sessions differed only in scale, dividing by the median would collapse them."""
        for name, v in mcitrine_by_exp.items():
            s = np.sort(v / np.median(v))
            y = np.arange(1, len(s) + 1) / len(s)
            ref = name == MC_REFERENCE
            ax.plot(s, y, lw=2.0 if ref else 1.0,
                    color=INK if ref else SERIES[0],
                    alpha=1.0 if ref else 0.45, zorder=3 if ref else 2)

        ax.set_xscale("log")
        ax.set_xlim(0.2, 8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mCitrine ÷ that session's own median")
        ax.set_ylabel("cumulative fraction of cells")
        ax.set_title("a  Rescaling does not align them", loc="left", fontweight="bold")
        _lo = mcitrine_summary["p90_over_p10"].min()
        _hi = mcitrine_summary["p90_over_p10"].max()
        ax.text(0.03, 0.99,
                f"each session \u00f7 its own median\np90/p10 still ranges {_lo:.2f} to {_hi:.2f}",
                transform=ax.transAxes, va="top", fontsize=7, color=INK)
        ax.text(0.97, 0.06, f"bold = {MC_REFERENCE} (largest session)",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_qq(ax):
        """(b) Quantile-quantile against the reference session.

        A straight line means some affine transform (scale, or scale after a log)
        could map that session onto the reference. Curvature means no such transform
        exists and only an order-preserving map — a rank — aligns them.
        """
        ref = np.quantile(mcitrine_by_exp[MC_REFERENCE], MC_Q)
        for name, v in mcitrine_by_exp.items():
            if name == MC_REFERENCE:
                continue
            q = np.quantile(v, MC_Q)
            # scale-matched at the median, so the comparison is about shape not offset
            ax.plot(ref, q * (np.median(ref) / np.median(q)), lw=1.1,
                    color=SERIES[0], alpha=0.55)
        ax.plot(ref, ref, color=INK, lw=1.4, ls="--", zorder=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"quantiles of {MC_REFERENCE} (mCitrine)")
        ax.set_ylabel("quantiles of each other session,\nmedian-matched")
        ax.set_title("b  Shape mismatch, session by session", loc="left", fontweight="bold")
        ax.text(0.03, 0.95,
                "dashed = perfect agreement after rescaling.\n"
                "Curves bending away from it cannot be fixed\n"
                "by any scale factor, in raw units or in log.",
                transform=ax.transAxes, va="top", fontsize=7, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig_norm, _axn = plt.subplots(1, 2, figsize=(9.4, 4.3))
    _panel_median_scaled(_axn[0])
    _panel_qq(_axn[1])
    fig_norm.tight_layout(pad=1.3, w_pad=2.8)
    save_fig(fig_norm, "figM1_expression_normalisation")
    fig_norm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Figure — the relationship replicates, even though the values do not compare

    Expression ranks cannot be compared between sessions (see the normalisation
    figure), so no plot can put two sessions' expression values on one axis. What
    *can* be compared is the **relationship measured separately inside each
    session**: a correlation is scale-free, so eleven independent estimates of it
    are directly comparable even when the underlying units are not.

    That turns the incomparability from a limitation into a replication argument.
    """)
    return


@app.cell(hide_code=True)
def _(FEATURES, hist_cnr, hist_feats, hist_meta, np, pl, spearmanr):
    from scipy.stats import rankdata
    REPL_MIN_CELLS = 120
    REPL_BOOT = 400


    def protocol_agnostic_response(cell_idx):
        """Response of one cell, defined without reference to a specific protocol.

        Baseline is the cell's own pre-stimulation frames; the response window runs
        from its first lit frame to its last. Every experiment here uses a different
        pattern and duration, so any fixed window would measure a different thing in
        each one.
        """
        cnr = np.asarray(hist_cnr[cell_idx], np.float32)
        u = np.asarray(hist_feats[cell_idx], np.float32)[FEATURES.index("u_t")]
        lit = u > u.min() * 1.001 + 1e-9
        if not lit.any():
            return np.nan
        first, last = int(np.argmax(lit)), int(len(lit) - 1 - np.argmax(lit[::-1]))
        if first < 5:
            return np.nan                       # no usable pre-stimulation baseline
        base = float(np.median(cnr[:first]))
        if abs(base) < 1e-6:
            return np.nan
        return float(cnr[first:last + 1].mean() / base - 1.0)


    _resp = np.array([protocol_agnostic_response(i) for i in range(len(hist_cnr))])
    _expr = np.array([np.asarray(hist_feats[i], np.float32)[FEATURES.index("optortk_expr")][0]
                      for i in range(len(hist_cnr))])
    _exp_name = hist_meta["original_experiment_name"].to_numpy()

    _rng = np.random.default_rng(0)
    _rows, _curves = [], {}
    for name in sorted(set(_exp_name.tolist())):
        m = (_exp_name == name) & np.isfinite(_resp)
        if m.sum() < REPL_MIN_CELLS:
            continue
        e, r = _expr[m], _resp[m]
        rho = float(spearmanr(e, r).statistic)
        boot = np.array([
            spearmanr(*(lambda s: (e[s], r[s]))(_rng.integers(0, len(e), len(e)))).statistic
            for _ in range(REPL_BOOT)
        ])
        _rows.append({"experiment": name, "n": int(m.sum()), "rho": rho,
                      "lo": float(np.quantile(boot, 0.025)),
                      "hi": float(np.quantile(boot, 0.975))})

        # Decile curve, standardised within the session so eleven different response
        # scales can share one axis. z-scored on ranks: only the SHAPE is compared.
        q = np.quantile(e, np.linspace(0, 1, 11)); q[-1] += 1e-9
        b = np.clip(np.digitize(e, q[1:-1]), 0, 9)
        rz = (rankdata(r) - rankdata(r).mean()) / rankdata(r).std()
        _curves[name] = np.array([rz[b == i].mean() if (b == i).any() else np.nan
                                  for i in range(10)])

    replication = pl.DataFrame(_rows).sort("rho", descending=True)
    replication_curves = _curves
    replication
    return replication, replication_curves


@app.cell(hide_code=True)
def _(FEATURES, hist_feats, hist_meta, np, pl, replication, spearmanr):
    # How hard was each session driven? Duty cycle is the fraction of frames lit;
    # mean lit dose is the fluence when it was on. They are not the same axis —
    # Niesen ran ~70x the dose of everything else at a low duty.
    _ui = FEATURES.index("u_t")
    _exp = hist_meta["original_experiment_name"].to_numpy()
    _drive = []
    for _name in replication["experiment"].to_list():
        _m = np.where(_exp == _name)[0][:400]
        _u = np.concatenate([np.asarray(hist_feats[i], np.float32)[_ui] for i in _m])
        _lit = _u > _u.min() * 1.001 + 1e-9
        _drive.append({"experiment": _name,
                       "duty": float(_lit.mean()),
                       "mean_lit_dose": float(_u[_lit].mean()) if _lit.any() else 0.0})

    replication_drive = replication.join(pl.DataFrame(_drive), on="experiment")
    replication_meta_rho = float(spearmanr(replication_drive["duty"],
                                           replication_drive["rho"]).statistic)
    # Uncertainty on the meta-level correlation. Only 11 points, and they are not
    # fully independent (4 bo_* sessions, 3 freepattern_*), so the effective n is
    # smaller than 11 and the interval is wide.
    _rng2 = np.random.default_rng(0)
    _duty = replication_drive["duty"].to_numpy()
    _rho = replication_drive["rho"].to_numpy()
    _boot = [spearmanr(_duty[s], _rho[s]).statistic
             for s in (_rng2.integers(0, len(_duty), len(_duty)) for _ in range(2000))]
    _boot = np.array([b for b in _boot if np.isfinite(b)])
    replication_meta_ci = (float(np.quantile(_boot, 0.025)), float(np.quantile(_boot, 0.975)))
    replication_meta_ci
    return replication_drive, replication_meta_ci, replication_meta_rho


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    mpl,
    np,
    pl,
    plt,
    replication_curves,
    replication_drive,
    replication_meta_ci,
    replication_meta_rho,
    save_fig,
):
    def _panel_forest(ax):
        """(a) Eleven independent estimates of the same relationship."""
        d = replication_drive.sort("rho")
        y = np.arange(d.height)
        ax.hlines(y, d["lo"], d["hi"], color=SERIES[0], lw=2.0)
        ax.plot(d["rho"], y, "o", ms=5, color=SERIES[0])
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.set_yticks(y, [f"{e}  (n={n:,})" for e, n in zip(d["experiment"], d["n"])],
                      fontsize=7)
        ax.set_xlabel("Spearman ρ,  expression rank vs response")
        ax.set_title("a  Measured separately in each session", loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.98, 0.04, "bars = 95% bootstrap CI", transform=ax.transAxes,
                ha="right", fontsize=6.5, color=MUTED)


    def _panel_curves(ax):
        """(b) The shape of the relationship, standardised so sessions share an axis.

        Coloured by duty cycle rather than split at an arbitrary rho threshold: duty
        is the variable panel (c) shows the strength tracks, so the ordering of the
        lines here and the trend there are the same fact seen twice.
        """
        x = np.arange(1, 11)
        duty_of = dict(zip(replication_drive["experiment"], replication_drive["duty"]))
        cmap = plt.cm.Blues
        for name, curve in replication_curves.items():
            d = duty_of.get(name, 0.0)
            ax.plot(x, curve, lw=1.6, color=cmap(0.25 + 0.7 * d), alpha=0.95)
        ax.axhline(0, color=MUTED, lw=0.9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
        cb = ax.figure.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("duty cycle", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        ax.set_xlabel("expression decile, within session")
        ax.set_ylabel("response, standardised within session")
        ax.set_xticks(x)
        ax.set_title("b  Same shape where it is present", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    def _panel_drive(ax):
        """(c) Where it is absent, the session was barely stimulated — or saturated."""
        d = replication_drive
        ax.errorbar(d["duty"], d["rho"],
                    yerr=[d["rho"] - d["lo"], d["hi"] - d["rho"]],
                    fmt="o", ms=5, color=SERIES[0], ecolor=SERIES[0],
                    elinewidth=1.2, capsize=0, alpha=0.85)
        ax.axhline(0, color=MUTED, lw=1.0)

        # Placed on opposite sides: both points sit near zero and their labels would
        # otherwise overlap each other.
        _notes = {
            "DoseResponse": ("one pulse, 2% duty", (0.10, 0.17)),
            "freepattern_Niesen_EGFR_v1": ("saturating dose\n(~70x the rest)", (0.30, -0.16)),
        }
        for name, (note, xytext) in _notes.items():
            r = d.filter(pl.col("experiment") == name)
            if r.height:
                ax.annotate(note, xy=(r["duty"][0], r["rho"][0]), xytext=xytext,
                            fontsize=7, color=MUTED,
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.set_ylim(-0.25, 0.75)

        ax.set_xlabel("duty cycle  (fraction of frames lit)")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("c  Strength tracks how hard the session was driven",
                     loc="left", fontweight="bold")
        # Two different Spearmans live in this figure and the distinction matters:
        # each POINT is a within-session correlation over cells (panel a); this
        # annotation is a correlation over the 11 SESSION-level points.
        ax.text(0.03, 0.97,
                f"Spearman(duty, per-session \u03c1)\n"
                f"over {replication_drive.height} sessions = {replication_meta_rho:+.2f}\n"
                f"95% CI [{replication_meta_ci[0]:+.2f}, {replication_meta_ci[1]:+.2f}]",
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig_repl = plt.figure(figsize=(9.8, 6.6))
    _gsr = fig_repl.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.48, wspace=0.42,
                                 left=0.24, right=0.97, top=0.93, bottom=0.09)
    _panel_forest(fig_repl.add_subplot(_gsr[0, :]))
    _panel_curves(fig_repl.add_subplot(_gsr[1, 0]))
    _panel_drive(fig_repl.add_subplot(_gsr[1, 1]))
    save_fig(fig_repl, "fig11_expression_replication")
    fig_repl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The same relationship against delivered light, not just how often

    Duty cycle counts frames, not photons — a session can be lit on every frame at
    a weak dose, or rarely at a strong one. Replacing it with **time-averaged
    fluence** (mean mJ/cm² per frame above each cell's dark floor) folds pulse
    strength in, and changes the picture: the trend is not monotone but an
    **inverted U**. Expression predicts response only in a middle band of drive.

    That is why the duty-cycle version reported ρ = +0.89 — duty happened to be
    monotone because the saturating session also ran at low duty. On the fluence
    axis a rank correlation is simply the wrong summary, and the shape is the
    result.
    """)
    return


@app.cell(hide_code=True)
def _(
    FEATURES,
    hist_feats,
    hist_meta,
    np,
    pl,
    replication,
    replication_drive,
    spearmanr,
):
    # Time-averaged delivered fluence per session: mean over frames of the excess
    # above each cell's own dark floor, so the nonzero baseline u_t in most
    # experiments does not count as delivered light.
    _ui2 = FEATURES.index("u_t")
    _exp2 = hist_meta["original_experiment_name"].to_numpy()
    _fl = []
    for _name in replication["experiment"].to_list():
        _m = np.where(_exp2 == _name)[0][:400]
        _vals = []
        for _i in _m:
            _u = np.asarray(hist_feats[_i], np.float32)[_ui2]
            _vals.append(float((_u - _u.min()).mean()))
        _fl.append({"experiment": _name, "mean_fluence": float(np.mean(_vals))})

    replication_fluence = replication_drive.join(pl.DataFrame(_fl), on="experiment")

    _SAT = "freepattern_Niesen_EGFR_v1"     # ~70x the dose of every other session
    _x = replication_fluence["mean_fluence"].to_numpy()
    _r = replication_fluence["rho"].to_numpy()
    _keep = replication_fluence["experiment"].to_numpy() != _SAT

    fluence_rho_all = float(spearmanr(_x, _r).statistic)
    fluence_rho_rising = float(spearmanr(_x[_keep], _r[_keep]).statistic)
    _c = np.polyfit(np.log10(_x), _r, 2)
    fluence_peak = float(10 ** (-_c[1] / (2 * _c[0])))
    fluence_quad = _c

    {"rho_all_11": fluence_rho_all, "rho_excluding_saturating": fluence_rho_rising,
     "peak_mJ_cm2_per_frame": fluence_peak}
    return (
        fluence_peak,
        fluence_quad,
        fluence_rho_all,
        fluence_rho_rising,
        replication_fluence,
    )


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    fluence_peak,
    fluence_quad,
    fluence_rho_all,
    fluence_rho_rising,
    mpl,
    np,
    pl,
    plt,
    replication_curves,
    replication_fluence,
    save_fig,
):
    def _panel_forest_fl(ax):
        """(a) The eleven per-session correlations, as in the previous figure."""
        d = replication_fluence.sort("rho")
        y = np.arange(d.height)
        ax.hlines(y, d["lo"], d["hi"], color=SERIES[0], lw=2.0)
        ax.plot(d["rho"], y, "o", ms=5, color=SERIES[0])
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.set_yticks(y, [f"{e}  (n={n:,})" for e, n in zip(d["experiment"], d["n"])],
                      fontsize=7)
        ax.set_xlabel("Spearman ρ,  expression rank vs response")
        ax.set_title("a  Measured separately in each session", loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.98, 0.04, "bars = 95% bootstrap CI", transform=ax.transAxes,
                ha="right", fontsize=6.5, color=MUTED)


    def _panel_curves_fluence(ax):
        """(b) Same standardised decile curves, coloured by delivered fluence."""
        x = np.arange(1, 11)
        fl = dict(zip(replication_fluence["experiment"], replication_fluence["mean_fluence"]))
        norm = mpl.colors.LogNorm(vmin=0.05, vmax=30)
        cmap = plt.cm.viridis
        for name, curve in replication_curves.items():
            ax.plot(x, curve, lw=1.6, color=cmap(norm(fl.get(name, 0.05))), alpha=0.95)
        ax.axhline(0, color=MUTED, lw=0.9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = ax.figure.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("mean fluence\n(mJ/cm² per frame)", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        ax.set_xlabel("expression decile, within session")
        ax.set_ylabel("response, standardised within session")
        ax.set_xticks(x)
        ax.set_title("b  Shape, coloured by delivered light", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_fluence(ax):
        """(c) An inverted U: too little light and too much both erase the effect."""
        d = replication_fluence
        x, y = d["mean_fluence"].to_numpy(), d["rho"].to_numpy()

        _g = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        ax.plot(_g, np.polyval(fluence_quad, np.log10(_g)), color=SERIES[2], lw=1.4,
                alpha=0.7, zorder=1)
        ax.errorbar(x, y, yerr=[y - d["lo"], d["hi"] - y], fmt="o", ms=5,
                    color=SERIES[0], ecolor=SERIES[0], elinewidth=1.2, capsize=0, zorder=3)
        ax.axhline(0, color=MUTED, lw=1.0)
        ax.axvline(fluence_peak, color=SERIES[2], lw=0.9, ls=":")

        for name, note, off in (
            ("DoseResponse", "one pulse\ntoo little drive", (0.14, 0.20)),
            ("freepattern_Niesen_EGFR_v1", "~70x dose\nsaturated", (0.55, 0.20)),
        ):
            r = d.filter(pl.col("experiment") == name)
            if r.height:
                ax.annotate(note, xy=(r["mean_fluence"][0], r["rho"][0]),
                            xytext=(off[0], off[1]), textcoords="axes fraction",
                            fontsize=7, color=MUTED, ha="center",
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

        ax.set_xscale("log")
        ax.set_xlabel("time-averaged fluence  (mJ/cm² per frame)")
        ax.set_ylabel("Spearman ρ  (expression vs response)")
        ax.set_ylim(-0.25, 0.75)
        ax.set_title("c  A window, not a trend", loc="left", fontweight="bold")
        ax.text(0.03, 0.97,
                f"rising limb (10 sessions) ρ = {fluence_rho_rising:+.2f}\n"
                f"all 11 together ρ = {fluence_rho_all:+.2f}  — a rank\n"
                f"correlation cannot describe a peak\n"
                f"fitted peak ≈ {fluence_peak:.1f} mJ/cm² per frame",
                transform=ax.transAxes, va="top", fontsize=7, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig_fluence = plt.figure(figsize=(9.8, 6.6))
    _gsf = fig_fluence.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.48,
                                    wspace=0.42, left=0.24, right=0.97, top=0.93, bottom=0.09)
    _panel_forest_fl(fig_fluence.add_subplot(_gsf[0, :]))
    _panel_curves_fluence(fig_fluence.add_subplot(_gsf[1, 0]))
    _panel_fluence(fig_fluence.add_subplot(_gsf[1, 1]))
    save_fig(fig_fluence, "fig11b_expression_vs_fluence")
    fig_fluence
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So there seems to be an inverted U pattern. Not a strong finding, for the border values are from experiments that are somewhat of an outlier. Nevertheless, this could be replicated if i really need to.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Interactive version — hover a point for the session behind it

    Same data as the panel above. Not a thesis figure (matplotlib owns those);
    this is for inspecting which session sits where, since the interesting points
    are the two that fall off the curve at either end.
    """)
    return


@app.cell(hide_code=True)
def _(alt, fluence_quad, np, pl, replication_fluence):
    _fit_x = np.logspace(np.log10(replication_fluence["mean_fluence"].min()),
                         np.log10(replication_fluence["mean_fluence"].max()), 200)
    _fit_df = pl.DataFrame({"mean_fluence": _fit_x,
                            "rho": np.polyval(fluence_quad, np.log10(_fit_x))})

    _pts = replication_fluence.with_columns(
        pl.col("rho").round(3).alias("rho_r"),
        pl.col("lo").round(3).alias("lo_r"),
        pl.col("hi").round(3).alias("hi_r"),
        pl.col("duty").round(3).alias("duty_r"),
        pl.col("mean_fluence").round(3).alias("fluence_r"),
    )

    _x = alt.X("mean_fluence:Q", scale=alt.Scale(type="log"),
               title="time-averaged fluence (mJ/cm² per frame)")
    _tooltip = [
        alt.Tooltip("experiment:N", title="session"),
        alt.Tooltip("n:Q", title="cells", format=","),
        alt.Tooltip("rho_r:Q", title="ρ (expression vs response)"),
        alt.Tooltip("lo_r:Q", title="CI low"),
        alt.Tooltip("hi_r:Q", title="CI high"),
        alt.Tooltip("fluence_r:Q", title="mean fluence"),
        alt.Tooltip("duty_r:Q", title="duty cycle"),
    ]

    _curve = alt.Chart(_fit_df).mark_line(color="#1baf7a", opacity=0.7).encode(
        x=_x, y=alt.Y("rho:Q", title="Spearman ρ  (expression vs response)"))
    _bars = alt.Chart(_pts).mark_rule(color="#2a78d6", strokeWidth=2).encode(
        x=_x, y="lo:Q", y2="hi:Q", tooltip=_tooltip)
    _dots = alt.Chart(_pts).mark_circle(size=130, color="#2a78d6").encode(
        x=_x, y=alt.Y("rho:Q", scale=alt.Scale(domain=[-0.25, 0.75])), tooltip=_tooltip)
    _zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(
        color="#52514e", strokeDash=[4, 3]).encode(y="y:Q")

    fluence_chart = (_curve + _zero + _bars + _dots).properties(
        width=560, height=340,
        title="Expression → response strength against delivered light (hover for session)",
    ).interactive()
    fluence_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FIG:  What's the diversity of intervention for a single objective?

    How to measure?
    - take raw time series, plot agains each other, see if there are areas of higher-lower density?
    """)
    return


@app.cell(hide_code=True)
def _(CAL_RUN, np, parse_arm):
    # Arm 2 of v16: `sequence_mpc` holding one constant target (1.2) for every cell.
    # One objective, one model, one dose ladder — so any difference between what two
    # cells are commanded is the controller choosing differently for them.
    DIV_FOVS = (1, 6)
    DIV_MIN_FRAMES = 240      # cells followed long enough to have a trajectory

    div_df, DIV_T0, div_startup = parse_arm(CAL_RUN, DIV_FOVS)

    # The commanded dose, in delivered fluence rather than exposure time — the log
    # carries `fluence_out`, and it is a fixed multiple of the exposure. The rungs are
    # read off the data instead of hardcoded, so they cannot drift from the policy.
    DIV_LEVELS = np.array(sorted(div_df["fluence_out"].unique().to_list()))


    def div_matrix(df, n_frames):
        """(cells x frames) of commanded dose, for cells present from the first frame."""
        rows, keys = [], []
        for (fov, particle), g in df.sort("control_frame").group_by(
                ["fov", "particle"], maintain_order=True):
            cf = g["control_frame"].to_numpy()
            if cf[0] != 0 or len(cf) < n_frames or not np.all(np.diff(cf[:n_frames]) == 1):
                continue
            rows.append(g["fluence_out"].to_numpy()[:n_frames].astype(np.float32))
            keys.append((fov, particle))
        return np.array(rows), keys


    div_dose, div_keys = div_matrix(div_df, DIV_MIN_FRAMES)
    _idx = np.abs(div_dose[..., None] - DIV_LEVELS[None, None, :]).argmin(axis=-1)

    # How often is a cell given something other than what most cells get on that
    # frame? Zero would mean the per-cell machinery is doing nothing a single
    # population-wide command could not do.
    _mode = np.array([np.bincount(_idx[:, t], minlength=len(DIV_LEVELS)).argmax()
                      for t in range(div_dose.shape[1])])
    div_off_mode = (_idx != _mode[None, :]).mean(axis=0)

    # Spread of commands across cells at each frame, in bits. 0 = everyone identical.
    _p = np.stack([(_idx == k).mean(axis=0) for k in range(len(DIV_LEVELS))])
    div_entropy = -(np.where(_p > 0, _p * np.log2(np.maximum(_p, 1e-12)), 0.0)).sum(axis=0)
    div_level_frac = _p

    {"cells": div_dose.shape[0], "frames": div_dose.shape[1],
     "levels_mJ_cm2": [round(v, 2) for v in DIV_LEVELS],
     "mean_off_mode": float(div_off_mode.mean()),
     "mean_entropy_bits": float(div_entropy.mean()),
     "max_bits": float(np.log2(len(DIV_LEVELS))),
     "mean_dose": float(div_dose.mean())}
    return (
        DIV_LEVELS,
        div_dose,
        div_entropy,
        div_level_frac,
        div_matrix,
        div_off_mode,
    )


@app.cell(hide_code=True)
def _(Path, json, pl):
    # Every arm of the two runs that held 1.000 min/frame. Static, so the figure is
    # reproducible; the dropdown below only chooses which one to draw in detail.
    #
    # The open-loop arms are the null: one predetermined dose for every cell, so their
    # diversity is zero by construction and they set the floor the closed-loop arms are
    # read against.
    DIV_RUNS = {
        "v15": Path("/Volumes/imaging.data/mic01-imaging/314lipczuk/"
                    "2026-08-07_InferenceCNRhold_12h_v15/run15.jsonl"),
        "v16": Path("/Volumes/imaging.data/mic01-imaging/314lipczuk/"
                    "2026-08-07_InferenceCNRhold_12h_v16/run16.jsonl"),
    }


    def arm_table(path):
        """(arm, controller, reference, FOVs) for one run, from its startup record."""
        with open(path) as f:
            fp = json.loads(f.readline())["policies"]["fov"]
        by_arm = {}
        for k, v in fp.items():
            arm = (v.get("requested") or {}).get("arm")
            ref = (v.get("objective", {}).get("reference", {}) or {}).get("type")
            key = (arm, v["controller"]["type"], ref)
            by_arm.setdefault(key, []).append(int(k))
        return by_arm


    div_registry = []
    for run, path in DIV_RUNS.items():
        for (arm, controller, ref), fovs in sorted(arm_table(path).items()):
            div_registry.append({"run": run, "arm": arm, "controller": controller,
                                 "reference": ref, "fovs": tuple(sorted(fovs)),
                                 "label": f"{run} arm {arm} · {controller} · {ref}"})

    pl.DataFrame([{k: (str(v) if k == "fovs" else v) for k, v in r.items()}
                  for r in div_registry])
    return DIV_RUNS, div_registry


@app.cell(hide_code=True)
def _(DIV_RUNS, div_matrix, div_registry, np, parse_arm, pl):
    DIV_FRAMES = 240        # common window, so arms are compared over the same span


    def diversity_of(run, fovs, n_frames=DIV_FRAMES):
        """Dose matrix and diversity metrics for one arm."""
        df, _t0, _s = parse_arm(DIV_RUNS[run], fovs)
        dose, keys = div_matrix(df, n_frames)
        if len(dose) < 20:
            return None
        levels = np.array(sorted(df["fluence_out"].unique().to_list()))
        idx = np.abs(dose[..., None] - levels[None, None, :]).argmin(axis=-1)
        mode = np.array([np.bincount(idx[:, t], minlength=len(levels)).argmax()
                         for t in range(dose.shape[1])])
        off_mode = (idx != mode[None, :]).mean(axis=0)
        p = np.stack([(idx == k).mean(axis=0) for k in range(len(levels))])
        ent = -(np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0.0)).sum(axis=0)
        return {"dose": dose, "levels": levels, "level_frac": p,
                "off_mode": off_mode, "entropy": ent, "n": len(dose)}


    div_by_arm = {}
    for _r in div_registry:
        _res = diversity_of(_r["run"], _r["fovs"])
        if _res is not None:
            div_by_arm[_r["label"]] = {**_r, **_res}

    diversity_summary = pl.DataFrame([
        {"label": k, "run": v["run"], "controller": v["controller"],
         "reference": v["reference"], "n_cells": v["n"],
         "off_mode": float(v["off_mode"].mean()),
         "entropy_bits": float(v["entropy"].mean()),
         "top_rung_share": float(v["level_frac"][-1].mean()),
         "dose_spread": float(v["dose"].sum(axis=1).std() / max(v["dose"].sum(axis=1).mean(), 1e-9))}
        for k, v in div_by_arm.items()
    ]).sort(["controller", "reference", "run"])
    diversity_summary
    return div_by_arm, diversity_summary


@app.cell(hide_code=True)
def _(
    GRID,
    MUTED,
    SERIES,
    div_by_arm,
    diversity_summary,
    mpl,
    np,
    plt,
    save_fig,
):
    DIV_SHORT = {"constant": "hold a level", "schedule": "follow a schedule",
                 "frequency_staircase": "oscillate, period stepping down"}
    DIV_RUN_COLOUR = {"v15": SERIES[0], "v16": SERIES[1]}


    def _div_order():
        """Open loop first, as the zero reference, then the closed-loop objectives."""
        rows = diversity_summary.to_dicts()
        rows.sort(key=lambda r: (r["controller"] != "open_loop", r["reference"], r["run"]))
        return rows


    def _panel_div_bars(ax):
        """(a) How much the commands differ between cells, arm by arm."""
        rows = _div_order()
        y = np.arange(len(rows))
        ax.barh(y, [r["entropy_bits"] for r in rows],
                color=[DIV_RUN_COLOUR[r["run"]] for r in rows], height=0.7)
        labs = [f"{r['run']}  {'no feedback' if r['controller'] == 'open_loop' else DIV_SHORT.get(r['reference'], r['reference'])}"
                for r in rows]
        ax.set_yticks(y, labs, fontsize=7.5)
        for yi, r in zip(y, rows):
            ax.text(r["entropy_bits"] + 0.03, yi, f"n={r['n_cells']}", va="center",
                    fontsize=6.5, color=MUTED)
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.set_xlabel("spread of commands across cells (bits)")
        ax.set_xlim(0, 2.4)
        ax.set_title("a  One objective, how many different things does it do?",
                     loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.98, 0.06, "no feedback = one dose for every cell, so zero by construction",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


    DIV_MARKER = {"constant": "o", "schedule": "s", "frequency_staircase": "^"}


    def _panel_div_saturation(ax):
        """(b) Why some arms have less to choose from.

        Nothing to differentiate WITH once the controller is pinned at the top of the
        ladder: at the ceiling there is only one command available.
        """
        for r in diversity_summary.to_dicts():
            if r["controller"] == "open_loop":
                continue
            ax.plot(r["top_rung_share"], r["entropy_bits"],
                    DIV_MARKER.get(r["reference"], "o"), ms=9,
                    color=DIV_RUN_COLOUR[r["run"]])

        handles = ([mpl.lines.Line2D([], [], ls="", marker=m, ms=8, color=MUTED,
                                     label=DIV_SHORT[k]) for k, m in DIV_MARKER.items()]
                   + [mpl.lines.Line2D([], [], ls="", marker="o", ms=8,
                                       color=c, label=run)
                      for run, c in DIV_RUN_COLOUR.items()])
        ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower left", ncol=1)
        ax.set_xlabel("share of commands at the highest dose")
        ax.set_ylabel("spread of commands (bits)")
        ax.set_title("b  A saturated controller cannot differentiate",
                     loc="left", fontweight="bold")
        ax.set_xlim(0.2, 0.85)
        ax.set_ylim(1.15, 2.0)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    def _panel_div_time(ax):
        """(c) Diversity is not a constant of the objective.

        The one arm that collapses is the one that ends up pinned at the top rung; the
        oscillating arms rise and fall with their own reference, differentiating when
        it demands a change and converging while it holds.
        """
        for r in diversity_summary.to_dicts():
            if r["controller"] == "open_loop":
                continue
            v = div_by_arm[r["label"]]
            ax.plot(np.convolve(v["off_mode"], np.ones(9) / 9, mode="valid"),
                    lw=1.6, color=DIV_RUN_COLOUR[r["run"]],
                    ls={"constant": "-", "schedule": "--",
                        "frequency_staircase": ":"}[r["reference"]])
        ax.set_xlabel("control frame (min)")
        ax.set_ylabel("share of cells off the\nmajority command")
        ax.set_ylim(0, 0.75)
        ax.set_title("c  Not a constant of the objective", loc="left", fontweight="bold")
        ax.annotate("the arm that ends up\npinned at the top rung",
                    xy=(200, 0.15), xytext=(95, 0.045), fontsize=6.5, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.text(0.98, 0.97, "line style = objective, colour = run\n(legend in b)",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    fig_divsum = plt.figure(figsize=(9.8, 6.6))
    _gv = fig_divsum.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.5, wspace=0.3,
                                  left=0.26, right=0.97, top=0.93, bottom=0.1)
    _panel_div_bars(fig_divsum.add_subplot(_gv[0, :]))
    _panel_div_saturation(fig_divsum.add_subplot(_gv[1, 0]))
    _panel_div_time(fig_divsum.add_subplot(_gv[1, 1]))
    save_fig(fig_divsum, "fig14_intervention_diversity_by_objective")
    fig_divsum
    return


@app.cell(hide_code=True)
def _(div_by_arm, mo):
    div_choice = mo.ui.dropdown(
        options=list(div_by_arm),
        value=next(k for k in div_by_arm if "v16 arm 2" in k),
        label="Show in detail:",
        full_width=True,
    )
    div_choice
    return (div_choice,)


@app.cell(hide_code=True)
def _(INK, div_by_arm, div_choice, np, plt):
    _sel = div_by_arm[div_choice.value]
    _dose, _levels, _frac = _sel["dose"], _sel["levels"], _sel["level_frac"]

    _centred = _dose - _dose.mean(axis=0, keepdims=True)
    _uu, _ss, _ = np.linalg.svd(_centred, full_matrices=False)
    _pcs, _var = _uu[:, :2] * _ss[:2], (_ss ** 2 / (_ss ** 2).sum())[:2]


    def _panel_sel_heatmap(ax):
        order = np.argsort(_dose.sum(axis=1))
        im = ax.imshow(_dose[order], aspect="auto", origin="lower", cmap="magma",
                       interpolation="nearest",
                       extent=(0, _dose.shape[1], 0, _dose.shape[0]))
        ax.set_xlabel("control frame (min)")
        ax.set_ylabel(f"cells, sorted by total dose  (n = {_dose.shape[0]})")
        ax.set_yticks([])
        ax.set_title("a  What each cell was told to do", loc="left", fontweight="bold")
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("commanded dose (mJ/cm²)", fontsize=7)
        cb.ax.tick_params(labelsize=7)


    def _panel_sel_levels(ax):
        cmap = plt.cm.magma
        ax.stackplot(np.arange(_dose.shape[1]), _frac,
                     colors=[cmap(0.12 + 0.78 * i / max(len(_levels) - 1, 1))
                             for i in range(len(_levels))],
                     labels=[f"{v:.0f}" for v in _levels])
        ax.set_xlim(0, _dose.shape[1]); ax.set_ylim(0, 1)
        ax.set_xlabel("control frame (min)")
        ax.set_ylabel("share of cells")
        ax.set_title("b  Which rung the cells are on", loc="left", fontweight="bold")
        leg = ax.legend(frameon=False, fontsize=6.5, ncol=5, loc="upper center",
                        bbox_to_anchor=(0.5, -0.22), title="commanded dose (mJ/cm²)")
        leg.get_title().set_fontsize(6.5)


    def _panel_sel_density(ax):
        ax.hexbin(_pcs[:, 0], _pcs[:, 1], gridsize=22, cmap="Blues", mincnt=1, linewidths=0)
        ax.plot(_pcs[:, 0], _pcs[:, 1], "o", ms=2.5, color=INK, alpha=0.35)
        ax.set_xlabel(f"dose-sequence component 1  ({_var[0]:.0%} of variance)")
        ax.set_ylabel(f"component 2  ({_var[1]:.0%})")
        ax.set_title("c  Are there distinct strategies?", loc="left", fontweight="bold")
        ax.text(0.03, 0.96,
                f"{_sel['off_mode'].mean():.0%} of cells off the majority command\n"
                f"{_sel['entropy'].mean():.2f} of {np.log2(len(_levels)):.2f} bits of spread",
                transform=ax.transAxes, va="top", fontsize=7, color=INK)


    fig_divdetail = plt.figure(figsize=(9.8, 6.8))
    _gd = fig_divdetail.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.48,
                                     wspace=0.3, left=0.085, right=0.97, top=0.90,
                                     bottom=0.14)
    fig_divdetail.suptitle(div_choice.value, fontsize=9, y=0.97)
    _panel_sel_heatmap(fig_divdetail.add_subplot(_gd[0, :]))
    _panel_sel_levels(fig_divdetail.add_subplot(_gd[1, 0]))
    _panel_sel_density(fig_divdetail.add_subplot(_gd[1, 1]))
    fig_divdetail
    return


@app.cell(hide_code=True)
def _(
    DIV_LEVELS,
    INK,
    MUTED,
    div_dose,
    div_entropy,
    div_level_frac,
    div_off_mode,
    np,
    plt,
    save_fig,
):
    # Two dimensions of the dose sequences, to answer the density question directly:
    # if the controller had a few distinct strategies the cells would fall into
    # separate clumps; if it is really tuning per cell they lie on a continuum.
    _centred = div_dose - div_dose.mean(axis=0, keepdims=True)
    _u, _s, _vt = np.linalg.svd(_centred, full_matrices=False)
    div_pcs = _u[:, :2] * _s[:2]
    div_var = (_s ** 2 / (_s ** 2).sum())[:2]


    def _panel_div_heatmap(ax):
        """(a) Every cell's commanded dose, sorted by how much it received in total."""
        order = np.argsort(div_dose.sum(axis=1))
        im = ax.imshow(div_dose[order], aspect="auto", origin="lower", cmap="magma",
                       interpolation="nearest",
                       extent=(0, div_dose.shape[1], 0, div_dose.shape[0]))
        ax.set_xlabel("control frame (min)")
        ax.set_ylabel(f"cells, sorted by total dose  (n = {div_dose.shape[0]})")
        ax.set_yticks([])
        ax.set_title("a  What each cell was told to do", loc="left", fontweight="bold")
        ax.text(0.99, 0.03, "pale = the maximum rung", transform=ax.transAxes,
                ha="right", fontsize=6.5, color=MUTED)
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("commanded dose (mJ/cm²)", fontsize=7)
        cb.ax.tick_params(labelsize=7)


    def _panel_div_levels(ax):
        """(b) How the population is split across the ladder, frame by frame."""
        t = np.arange(div_dose.shape[1])
        cmap = plt.cm.magma
        ax.stackplot(t, div_level_frac,
                     colors=[cmap(0.12 + 0.78 * i / (len(DIV_LEVELS) - 1))
                             for i in range(len(DIV_LEVELS))],
                     labels=[f"{v:.0f}" for v in DIV_LEVELS])
        ax.set_xlim(0, div_dose.shape[1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("control frame (min)")
        ax.set_ylabel("share of cells")
        ax.set_title("b  Choice narrows as the run goes on", loc="left", fontweight="bold")
        leg = ax.legend(frameon=False, fontsize=6.5, ncol=5, loc="upper center",
                        bbox_to_anchor=(0.5, -0.22), title="commanded dose (mJ/cm²)")
        leg.get_title().set_fontsize(6.5)
        ax.text(0.98, 0.93,
                f"top rung carries {div_level_frac[-1].mean():.0%}\nof all commands",
                transform=ax.transAxes, va="top", ha="right", fontsize=7, color="white")


    def _panel_div_density(ax):
        """(c) Are there distinct strategies, or one continuum?"""
        ax.hexbin(div_pcs[:, 0], div_pcs[:, 1], gridsize=22, cmap="Blues", mincnt=1,
                  linewidths=0)
        ax.plot(div_pcs[:, 0], div_pcs[:, 1], "o", ms=2.5, color=INK, alpha=0.35)
        ax.set_xlabel(f"dose-sequence component 1  ({div_var[0]:.0%} of variance)")
        ax.set_ylabel(f"component 2  ({div_var[1]:.0%})")
        ax.set_title("c  A dense core and a thin tail", loc="left", fontweight="bold")
        ax.text(0.03, 0.96,
                f"at a typical frame {div_off_mode.mean():.0%} of cells are given\n"
                f"something other than the majority command\n"
                f"spread across cells: {div_entropy.mean():.2f} of "
                f"{np.log2(len(DIV_LEVELS)):.2f} bits",
                transform=ax.transAxes, va="top", fontsize=7, color=INK)


    fig_div = plt.figure(figsize=(9.8, 6.8))
    _gsd = fig_div.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.48, wspace=0.3,
                                left=0.085, right=0.97, top=0.93, bottom=0.14)
    _panel_div_heatmap(fig_div.add_subplot(_gsd[0, :]))
    _panel_div_levels(fig_div.add_subplot(_gsd[1, 0]))
    _panel_div_density(fig_div.add_subplot(_gsd[1, 1]))
    save_fig(fig_div, "fig14_intervention_diversity")
    fig_div
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _():


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Does giving every cell the same light let the model tell them apart?

    `v16` arm 1 (FOVs 0 and 7) delivered a fixed sequence of light steps —
    18 minutes each at 0, 300, 85, 600 and 150 ms exposure, repeating — the same
    for every cell, with no feedback. The server ran throughout, so the model saw
    each cell's response as it happened.

    Because every cell received identical light, anything the model predicts
    differently between cells must come from what it saw of that cell's own past.
    The future light it is given carries no information about which cell it is.

    Two starting points, each with the same amount of the past to read:

    | forecast starts at | the 18 minutes before it | what the model can know |
    |---|---|---|
    | control frame 18 | the light was off | this cell's resting level |
    | control frame 36 | the light was on at 300 ms | how strongly it responds |

    Matching the length of those two stretches is what makes the comparison mean
    something: otherwise simply having more of the past to read would explain any
    improvement, and this would only repeat Figure 10.
    """)
    return


@app.cell(hide_code=True)
def _(Path, pl):
    import json
    from collections import defaultdict

    CAL_RUN = Path("/Volumes/imaging.data/mic01-imaging/314lipczuk/"
                   "2026-08-07_InferenceCNRhold_12h_v16/run16.jsonl")
    CAL_FOVS = (0, 7)          # arm 1 — a fixed sequence of light steps, no feedback
    CAL_RUNG = 18              # frames per rung


    def parse_arm(path, fovs):
        """Per-cell-per-frame records for selected FOVs, plus the control-frame offset.

        `control_frame = timestep - first_controlled_timestep`. The waveform and the
        rung labels are clocked from the first controlled frame, not from faro's
        timestep, which starts wherever earlier acquisition phases left off.
        """
        keep = ("raw_cnr", "u_t_in", "fluence_out", "n_cells_200px",
                "optortk_expr", "nuc_area", "pred_cnr_h1")
        rows = defaultdict(list)
        with open(path) as f:
            startup = json.loads(f.readline())
            for line in f:
                if '"predict"' not in line:
                    continue
                r = json.loads(line)
                if r.get("event") != "predict" or r.get("fov") not in fovs:
                    continue
                for c in r.get("cells") or []:
                    rows["fov"].append(r["fov"])
                    rows["timestep"].append(r["timestep"])
                    rows["particle"].append(c["particle"])
                    for k in keep:
                        rows[k].append(c.get(k))
        df = pl.DataFrame(rows)
        t0 = int(df["timestep"].min())
        return df.with_columns((pl.col("timestep") - t0).alias("control_frame")), t0, startup


    cal_df, CAL_T0, cal_startup = parse_arm(CAL_RUN, CAL_FOVS)
    CAL_CHECKPOINT = cal_startup["policies"]["default"]["checkpoint_dir"]

    # Which light step is which, to confirm they land where the policy file says.
    _rungs = (
        cal_df.with_columns((pl.col("control_frame") // CAL_RUNG).alias("rung"))
        .group_by("rung").agg(pl.col("fluence_out").median().alias("fluence"),
                              pl.col("control_frame").min().alias("from"),
                              pl.col("control_frame").max().alias("to"))
        .sort("rung").head(6)
    )
    {"rows": cal_df.height, "cells": cal_df.select(["fov", "particle"]).n_unique(),
     "first_controlled_timestep": CAL_T0, "checkpoint": CAL_CHECKPOINT,
     "first_rungs": _rungs.to_dicts()}
    return CAL_CHECKPOINT, CAL_RUN, cal_df, json, parse_arm


@app.cell(hide_code=True)
def _(
    CAL_CHECKPOINT,
    Path,
    cal_df,
    load_experiment,
    np,
    pl,
    predict_many,
    results_write_path,
    spearmanr,
):
    cal_bundle = load_experiment(str(Path(results_write_path()) / CAL_CHECKPOINT))
    cal_model = cal_bundle.reconstruct_model()
    ccfg = cal_model.cfg
    CAL_FEATURES = [c for c in ccfg.norm_channels if c != "cnr"]

    CAL_START_AFTER_DARKNESS = 18      # context = the 0 ms rung
    CAL_START_AFTER_LIGHT = 36     # context = the 300 ms rung and its response
    CAL_CTX = 18              # equal-length stretches of the past: one rung each
    CAL_ORIGINS = [36, 54, 72, 90]   # starts of successive light steps
    CAL_LAST_ORIGIN = 90


    def cal_tracks():
        """Cells present and contiguous from control frame 0 through the second origin."""
        # Long enough for every origin tested below to have a FULL horizon. A short
        # tail silently shortens the forecast instead of failing, which would make
        # origins incomparable to each other.
        need = CAL_LAST_ORIGIN + ccfg.future_len
        out = []
        for (fov, particle), g in cal_df.sort("control_frame").group_by(
                ["fov", "particle"], maintain_order=True):
            cf = g["control_frame"].to_numpy()
            if cf[0] != 0 or len(cf) < need or not np.all(np.diff(cf[:need]) == 1):
                continue
            out.append({"fov": fov, "particle": particle,
                        "cnr": g["raw_cnr"].to_numpy()[:need].astype(np.float32),
                        **{k: g[{"u_t": "u_t_in"}.get(k, k)].to_numpy()[:need].astype(np.float32)
                           for k in CAL_FEATURES}})
        return out


    cal_cells = cal_tracks()


    def cal_forecast(origin, cap):
        """Predicted mean, sigma and truth over the horizon, for every cell, at one origin."""
        means, sigmas, truths = [], [], []
        for tr in cal_cells:
            chans = {k: tr[k] for k in CAL_FEATURES}
            m, s = predict_many(cal_model, tr["cnr"], chans["u_t"], [origin],
                                channels=chans, cap=cap)
            means.append(m[0]); sigmas.append(s[0])
            truths.append(tr["cnr"][origin:origin + ccfg.future_len])
        return np.array(means), np.array(sigmas), np.array(truths)


    def cal_metrics(origin, cap):
        """Sharpening, differentiation, and accuracy at one origin.

        Every cell in this arm receives the SAME dose, so the decoder's future input
        carries no cell-specific information: the spread of predicted means is
        produced entirely by the encoder's view of each cell's past. A model that
        cannot tell the cells apart would predict the same trajectory for all of them
        and `pred_spread` would be zero.
        """
        m, s, t = cal_forecast(origin, cap)
        pm, tm = m.mean(axis=1), t.mean(axis=1)
        return {
            "origin": origin, "cap": cap, "n": len(m),
            "sigma_mean": float(s.mean()),
            "pred_spread": float(pm.std()),
            "true_spread": float(tm.std()),
            "corr_pred_true": float(spearmanr(pm, tm).statistic),
            "mae": float(np.abs(m - t).mean()),
        }


    calibration = pl.DataFrame([
        cal_metrics(CAL_START_AFTER_DARKNESS, CAL_CTX),
        cal_metrics(CAL_START_AFTER_LIGHT, CAL_CTX),
    ])
    calibration
    return (
        CAL_CTX,
        CAL_ORIGINS,
        CAL_START_AFTER_LIGHT,
        cal_cells,
        cal_forecast,
        cal_metrics,
    )


@app.cell(hide_code=True)
def _(CAL_START_AFTER_LIGHT, cal_metrics, pl):
    # How many minutes of the light response does the model need?
    #
    # The two-origin comparison above has one weakness: the origins forecast
    # DIFFERENT future windows (the 300 ms rung vs the 85 ms rung), so part of the
    # improvement could be that the second window is easier to predict. Holding the
    # origin fixed at 36 and varying only how far back the encoder may read removes
    # that entirely — same cells, same future, same dose, one variable.
    #
    # `cap = k` keeps the last k frames before the origin, which is exactly the
    # operational question: calibrate for k minutes, then start controlling.
    CAL_CAPS = [1, 2, 3, 4, 6, 8, 11, 14, 18, 24, 30, 36]

    calibration_sweep = pl.DataFrame([
        cal_metrics(CAL_START_AFTER_LIGHT, k) for k in CAL_CAPS
    ])
    calibration_sweep
    return


@app.cell(hide_code=True)
def _(CAL_CTX, CAL_ORIGINS, cal_cells, cal_forecast, np, pl):
    # --- Does the model know anything about a cell beyond its current level? -----
    #
    # Swapping a random cell's past into the forecast would only re-prove that the
    # current CNR matters, which the last-observed-value predictor already shows.
    # The test that means something pairs each cell with a DONOR whose last observed
    # CNR is essentially identical, so both stretches of past contain a full light
    # response of the same length at the same dose and differ only in which cell they
    # came from. Whatever the forecast loses under that swap is cell identity; what it
    # keeps is the current level plus dynamics common to every cell.
    #
    # The light is identical across cells in this arm, so the future input the decoder
    # receives is the same no matter whose past is used — nothing has to be held fixed
    # by hand.

    SWAP_SEED = 0


    def swap_test(origin, cap=CAL_CTX):
        m, _s, t = cal_forecast(origin, cap)
        last = np.array([tr["cnr"][origin - 1] for tr in cal_cells])
        n = len(m)

        gap = np.abs(last[:, None] - last[None, :])
        np.fill_diagonal(gap, np.inf)
        donor = gap.argmin(axis=1)
        rng = np.random.default_rng(SWAP_SEED)
        shuffled = rng.permutation(n)

        per_cell = lambda pred: np.abs(pred - t).mean(axis=1)
        persistence = np.repeat(last[:, None], t.shape[1], axis=1)
        return {
            "origin": origin,
            "n": n,
            "own": per_cell(m),
            "matched": per_cell(m[donor]),
            "shuffled": per_cell(m[shuffled]),
            "last_value": per_cell(persistence),
            "donor_gap": np.abs(last - last[donor]),
            "last_sd": float(last.std()),
        }


    swap_results = {o: swap_test(o) for o in CAL_ORIGINS}

    swap_summary = pl.DataFrame([
        {"origin": o,
         "own": float(r["own"].mean()),
         "level_matched_donor": float(r["matched"].mean()),
         "shuffled_donor": float(r["shuffled"].mean()),
         "last_value_only": float(r["last_value"].mean()),
         "identity_cost": float(r["matched"].mean() - r["own"].mean()),
         "identity_share": float((r["matched"].mean() - r["own"].mean()) / r["own"].mean()),
         "median_donor_gap": float(np.median(r["donor_gap"]))}
        for o, r in swap_results.items()
    ])
    swap_summary
    return (swap_results,)


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, mpl, np, plt, save_fig, swap_results):
    SWAP_ORIGIN_SHOWN = 36
    SWAP_LABELS = [("own", "the cell's own past", INK),
                   ("matched", "a different cell at the same level", SERIES[0]),
                   ("last_value", "its current CNR, held", SERIES[2]),
                   ("shuffled", "a different cell, any level", SERIES[1])]


    def _panel_swap_bars(ax):
        """(a) What the forecast is actually built on."""
        origins = list(swap_results)
        w = 0.2
        for j, (key, lab, colour) in enumerate(SWAP_LABELS):
            vals = [swap_results[o][key].mean() for o in origins]
            err = [swap_results[o][key].std() / np.sqrt(swap_results[o]["n"]) for o in origins]
            ax.bar(np.arange(len(origins)) + (j - 1.5) * w, vals, w, yerr=err,
                   color=colour, label=lab, error_kw=dict(lw=0.9, ecolor=MUTED))

        ax.set_xticks(range(len(origins)), [f"min {o}" for o in origins])
        ax.set_xlabel("forecast starts at")
        ax.set_ylabel("forecast error (absolute CNR)")
        ax.set_title("a  Replace the cell's past and see what breaks",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
        ax.set_ylim(0, 0.42)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_swap_paired(ax):
        """(b) The same comparison cell by cell, not as an average."""
        r = swap_results[SWAP_ORIGIN_SHOWN]
        lim = (0, max(r["own"].max(), r["matched"].max()) * 1.05)
        ax.plot(r["own"], r["matched"], "o", ms=4, alpha=0.5, color=SERIES[0])
        ax.plot(lim, lim, color=INK, lw=1.0, ls="--")
        worse = float((r["matched"] > r["own"]).mean())
        ax.text(0.04, 0.95,
                f"donor's past is worse for {worse:.0%} of cells\n"
                f"median donor differs in current CNR\nby {np.median(r['donor_gap']):.3f}"
                f"  (population sd {r['last_sd']:.2f})",
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel("error using the cell's own past")
        ax.set_ylabel("error using a level-matched\ndifferent cell's past")
        ax.set_title(f"b  Cell by cell, from min {SWAP_ORIGIN_SHOWN}",
                     loc="left", fontweight="bold")


    def _panel_swap_identity(ax):
        """(c) How much of the forecast is genuinely about *this* cell."""
        origins = list(swap_results)
        share = [swap_results[o]["matched"].mean() / swap_results[o]["own"].mean() - 1
                 for o in origins]
        ax.plot(origins, share, color=SERIES[0], lw=2.2, marker="o", ms=6)
        ax.axhline(0, color=MUTED, lw=1.0)
        for o, s in zip(origins, share):
            ax.annotate(f"{s:.0%}", (o, s), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5, color=INK)

        ax.set_xlabel("forecast starts at (min into the run)")
        ax.set_ylabel("error added by using\nanother cell's past")
        ax.set_xticks(origins)
        ax.set_ylim(-0.02, 0.20)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_title("c  What is left that is about this cell", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig_swap = plt.figure(figsize=(9.8, 6.4))
    _gsw = fig_swap.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.5, wspace=0.32,
                                 left=0.09, right=0.98, top=0.92, bottom=0.1)
    _panel_swap_bars(fig_swap.add_subplot(_gsw[0, :]))
    _panel_swap_paired(fig_swap.add_subplot(_gsw[1, 0]))
    _panel_swap_identity(fig_swap.add_subplot(_gsw[1, 1]))
    save_fig(fig_swap, "fig13_history_swap")
    fig_swap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
