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

    # --- Page geometry ---------------------------------------------------------
    # A4 with 2.5 cm margins leaves a 16 cm text block. EVERY figure is sized in
    # these units, and that is what makes `font.size` below mean the same thing on
    # the page in every figure: a figure placed at 1:1 renders its 9 pt labels at
    # 9 pt.
    #
    # It was not true before. The set used 21 different widths between 8.8 and
    # 12.6 in, so each figure was scaled by a different factor when placed — 0.72
    # for the narrowest, 0.50 for the widest — and identical rcParams landed on the
    # page as anything from about 6.5 pt down to 4.5 pt, varying figure to figure.
    # Invisible in the notebook at figure.dpi 130; only visible in the compiled PDF,
    # which is why it survived this long.
    W_TEXT = 6.30   # 16 cm — the text block. The default width for every figure.
    W_WIDE = 9.40   # 24 cm — ONLY for a figure rotated onto the page (landscape
                    # float). Still placed at 1:1, so type stays 9 pt.

    # Single place to define the look of every figure in the thesis.
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        # NOT "tight". A tight bbox expands the saved canvas to contain anything
        # overflowing the axes, which silently turns W_TEXT into a suggestion —
        # dataset-overview came out 7.1 in from one long y-tick label,
        # forecast-examples 8.3 in from a
        # long caption. With a fixed bbox the saved width IS figsize, and
        # overflowing text CLIPS, which makes the problem visible here instead of
        # reaching the compiled PDF at the wrong scale.
        "savefig.bbox": None,
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
    return W_TEXT, save_fig


@app.cell
def _(mo):
    mo.md("""
    # dataset-overview — dataset overview
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

    # --- Arms are ORDINAL, and must not use SERIES -----------------------------
    # An arm is always a level of something ordered: free minutes 0/8/14/20, how
    # much feedback a controller is given, how much light a conditioning interval
    # delivered. Ordered data takes ONE HUE READ LIGHT-TO-DARK, so that "more" looks
    # like more and the reader can rank the arms without consulting the legend.
    #
    # This is also what keeps it colourblind-safe. Four unrelated categorical hues
    # cannot clear the all-pairs CVD floor — that is why SERIES stops at three — but
    # a monotone single-hue ramp does not have to: it separates on lightness, which
    # every form of colour vision preserves. Validated as an ordinal ramp: lightness
    # monotone, all adjacent gaps >= 0.06, light end 2.06:1 against the surface,
    # hue spread 3 degrees.
    #
    # Fixed by POSITION, so arm 1 is the palest in every figure of the thesis. Note
    # free_runup_v21.py still carries its own seaborn ARM_COLOR — that is the
    # inconsistency this replaces, not a second convention to keep in step.
    ARM_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

    # Magnitude — dose, rung occupancy, response density. One hue, light to dark.
    SEQ = "Blues"
    # Polarity — residuals above vs below the demand. Two poles, neutral middle.
    # Never a rainbow, and never a hue at the midpoint: the middle must read as
    # "no error", which only a neutral does.
    DIV = "RdBu_r"
    SERIES
    return ARM_RAMP, GRID, INK, MUTED, SERIES, STIM_BLUE


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
    return (EXAMPLE_UID,)


@app.cell(hide_code=True)
def _(
    EXAMPLE_UID,
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
    dataset_summary,
    df,
    mpl,
    np,
    pl,
    plt,
    save_fig,
):
    # Three families, and every panel is keyed to them by colour. Panel (a) says how
    # much data each family contributes, panel (d) says what that family's data looks
    # like — so the two are one argument rather than two unrelated summaries.
    _FAMILY = {
        "bo_v8": "BO", "bo_v10": "BO", "bo_v11_10s": "BO", "bo_v11_20s": "BO",
        "freepattern_v1": "FP", "freepattern_v2": "FP",
        "freepattern_Niesen_EGFR_v1": "FP",
        "Sustained_1min": "CH", "3-2-1minIntervals": "CH",
        "DoseResponse": "CH", "RampReverse": "CH",
    }
    _FAM_ORDER = ["BO", "FP", "CH"]
    _FAM_COLOR = dict(zip(_FAM_ORDER, SERIES))
    _FAM_LABEL = {"BO": "BO search", "FP": "Pattern sweep", "CH": "Characterisation"}
    # One experiment stands for each family in (d): the largest of that family, except
    # CH where Sustained is the clearest contrast to the other two.
    _REP = {"BO": "bo_v8", "FP": "freepattern_v2", "CH": "Sustained_1min"}


    def _panel_composition(ax):
        """(a) How much data, from which family."""
        s = (dataset_summary
             .with_columns(pl.col("experiment").replace_strict(_FAMILY).alias("family"))
             .with_columns(pl.col("family").replace_strict(
                 {f: i for i, f in enumerate(_FAM_ORDER)}).alias("_fo"))
             .sort(["_fo", "cells"], descending=[True, False]))
        y = np.arange(s.height)
        ax.barh(y, s["cells"], height=0.68,
                color=[_FAM_COLOR[f] for f in s["family"]])
        ax.set_yticks(y, s["experiment"], fontsize=6)
        for yi, n in zip(y, s["cells"]):
            ax.text(n + 500, yi, f"{n:,}", va="center", fontsize=6, color=MUTED)
        ax.set_xlabel("tracked cells")
        ax.set_xlim(0, s["cells"].max() * 1.22)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if v else "0"))
        ax.set_title("c  Data sources", loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(handles=[mpl.patches.Patch(color=_FAM_COLOR[f], label=_FAM_LABEL[f])
                           for f in _FAM_ORDER],
                  loc="lower right", frameon=False, fontsize=6.5, handlelength=1.2,
                  borderaxespad=0.4)
        ax.text(0.98, 0.30,
                f"{dataset_summary['cells'].sum():,} cells\n"
                f"{dataset_summary['rows'].sum() / 1e6:.1f}M frames",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=MUTED)


    def _panel_example(ax, ax_stim):
        """(b) One cell — and the fact that pulses differ in exposure, not just timing."""
        cell = (df.filter(pl.col("uid") == EXAMPLE_UID).sort("frame")
                  .select("frame", "cnr_median_norm", "m_t", "stim_exposure"))
        t = cell["frame"].to_numpy()
        ax.plot(t, cell["cnr_median_norm"].to_numpy(), color=INK, lw=1.4, zorder=2)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.set_ylabel("CNR\n(baseline-norm.)", fontsize=8)
        ax.set_xlim(t.min(), t.max())
        ax.tick_params(labelbottom=False)
        ax.set_title("b  One cell, one trajectory", loc="left", fontweight="bold")

        # Exposure as bar HEIGHT. The old panel shaded every lit frame identically,
        # which threw the information away: this cell alone spans 144 distinct
        # exposures from 280 to 2785 ms, and that variation is a property of the
        # dataset the model has to cover.
        e = cell["stim_exposure"].to_numpy()
        _lit = cell["m_t"].to_numpy() > 0
        ax_stim.bar(t[_lit], e[_lit], width=1.0, color=_FAM_COLOR["BO"], lw=0)
        ax_stim.set_xlim(t.min(), t.max())
        ax_stim.set_ylabel("exposure\n(ms)", fontsize=7)
        ax_stim.set_xlabel("time (min)")
        ax_stim.tick_params(labelsize=7)
        ax_stim.yaxis.grid(True, color=GRID, lw=0.5)
        ax_stim.set_axisbelow(True)


    def _panel_conditions(ax):
        """(d) What one experiment from each family looks like."""
        for f in _FAM_ORDER:
            exp = _REP[f]
            g = (df.filter(pl.col("original_experiment_name") == exp)
                   .group_by("frame")
                   .agg(pl.col("cnr_median_norm").median().alias("med"),
                        pl.col("cnr_median_norm").quantile(0.25).alias("lo"),
                        pl.col("cnr_median_norm").quantile(0.75).alias("hi"),
                        pl.col("m_t").mean().alias("duty"))
                   .sort("frame"))
            t = g["frame"].to_numpy()
            n_cond = int(dataset_summary.filter(
                pl.col("experiment") == exp)["conditions"][0])
            ax.fill_between(t, g["lo"], g["hi"], color=_FAM_COLOR[f], alpha=0.15, lw=0)
            ax.plot(t, g["med"], color=_FAM_COLOR[f], lw=1.8,
                    label=f"{exp}  ({n_cond} pattern{'s' if n_cond > 1 else ''}, "
                          f"{float(g['duty'].mean()):.0%} of frames lit)")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("CNR (median, IQR shaded)", fontsize=8)
        ax.set_title("a  What each family of experiments looks like", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=6.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    # The family comparison leads: it is the panel that says what this corpus IS.
    # Composition and the single trajectory support it underneath. Two gridspecs
    # rather than one, so the bottom-left panel can have the wide left margin its
    # experiment names need without indenting the full-width panel above it.
    fig_dataset = plt.figure(figsize=(W_TEXT, 5.6))
    _gs_top = fig_dataset.add_gridspec(1, 1, left=0.115, right=0.985, top=0.95, bottom=0.55)
    # Composition sits on the RIGHT so its long experiment names fall in the inner
    # gutter. On the left they forced a wide figure margin, which indented the whole
    # bottom row relative to the full-width panel above it.
    _gs_bot = fig_dataset.add_gridspec(2, 2, width_ratios=[1.0, 0.95],
                                height_ratios=[1.0, 0.30],
                                hspace=0.16, wspace=0.95,
                                left=0.115, right=0.985, top=0.42, bottom=0.085)
    _ax_a = fig_dataset.add_subplot(_gs_top[0, 0])
    _ax_b = fig_dataset.add_subplot(_gs_bot[0, 0])
    _ax_bs = fig_dataset.add_subplot(_gs_bot[1, 0], sharex=_ax_b)
    _ax_c = fig_dataset.add_subplot(_gs_bot[:, 1])
    _panel_conditions(_ax_a)
    _panel_example(_ax_b, _ax_bs)
    _panel_composition(_ax_c)

    save_fig(fig_dataset, "dataset-overview")
    fig_dataset
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    _rep = {"BO search": "bo_v8", "Pattern sweep": "freepattern_v2",
            "Characterisation": "Sustained_1min"}
    _rows = []
    for _fam, _exp in _rep.items():
        _g = (df.filter(pl.col("original_experiment_name") == _exp)
                .group_by("frame")
                .agg(pl.col("cnr_median_norm").median().alias("med"),
                     pl.col("cnr_median_norm").quantile(0.25).alias("lo"),
                     pl.col("cnr_median_norm").quantile(0.75).alias("hi"))
                .sort("frame")
                .filter((pl.col("frame") >= 30) & (pl.col("frame") <= 60)))
        _rows.append(dict(family=_fam,
                          med=float(_g["med"].mean()),
                          lo=float(_g["lo"].mean()),
                          hi=float(_g["hi"].mean()),
                          iqr=float((_g["hi"] - _g["lo"]).mean())))
    _t = {r["family"]: r for r in _rows}
    _meds = [r["med"] for r in _rows]
    _between = max(_meds) - min(_meds)
    _within = sum(r["iqr"] for r in _rows) / len(_rows)
    _bo, _ch = _t["BO search"], _t["Characterisation"]
    _overlap = min(_bo["hi"], _ch["hi"]) - max(_bo["lo"], _ch["lo"])

    mo.md(f"""
    ### What the overlapping bands in (a) are saying

    The three families separate cleanly **in the median** — {_t["BO search"]["med"]:.2f},
    {_t["Characterisation"]["med"]:.2f} and {_t["Pattern sweep"]["med"]:.2f} across the
    plateau — which is the reassuring, population-level reading: more light, more ERK.

    The bands say something else. The middle half of cells under one stimulation spans
    **{_within:.2f} CNR** on average, against **{_between:.2f} CNR** between the strongest
    and weakest condition — a ratio of **{_within / _between:.2f}**. The BO and
    characterisation bands overlap by **{_overlap:.2f} CNR**, which is
    **{100 * _overlap / min(_bo["iqr"], _ch["iqr"]):.0f}%** of the narrower of the two. Knowing
    which stimulation a cell received therefore narrows down where that individual cell
    ends up by very little: the spread *within* a condition is as large as the entire
    range *between* conditions.

    Two things sharpen this rather than explain it away. These traces are
    **baseline-normalised**, so each cell is already divided by its own resting level — the
    spread that remains is in the *response*, not in where cells started. And the pattern
    sweep's band is the narrow one ({_t["Pattern sweep"]["iqr"]:.2f}) *despite* pooling 88
    different stimulation patterns, because most of those patterns are weak; pooling
    conditions is not what widens the other two.

    This is the argument for modelling single cells rather than the population average, and
    it is the thing a mean-response model cannot represent. It is **not** yet evidence that
    the spread is *predictable* — that a cell's own history and covariates say where in the
    band it will land. That claim is heterogeneity's, and the expression results after it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## heterogeneity — same light, different cells

    The central difficulty of the whole project, and the thing that justifies
    per-cell control at all. Every cell in `bo_osc_v8_c1` received an **identical**
    light pattern — twelve pulses, five minutes apart, frames 10 to 65 — so any
    spread in the response is the cells, not the input.

    The question that decides whether per-cell control can help is not *how large*
    the spread is but *whether it is stable*. Spread that is measurement noise
    cannot be controlled against; spread that is a persistent property of each cell
    can be inferred from its past and acted on — which is exactly what the encoder
    does in encoder-needs.
    """)
    return


@app.cell(hide_code=True)
def _(FEATURES, hist_cnr, hist_cond, hist_feats, hist_meta, np, pl, spearmanr):
    HET_CONDITION = "bo_osc_v8_c1"
    # THIS PROTOCOL HAS NO PULSES. Frames 10-69 are ALL lit, with the exposure
    # ramping monotonically (720 ms rising ~35 ms/frame in one field; the BO
    # search swept the ramp between fields, 280-2785 ms overall). The previous
    # `HET_PULSES = np.arange(10, 70, 5)` described a pulse train that does not
    # exist. It was used only for the block bounds below and for a per-pulse
    # `rises` feature that nothing consumed.
    HET_STIM_ON, HET_STIM_OFF = 10, 70    # the illuminated block, [on, off)
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


        expr = np.array([np.asarray(hist_feats[i], np.float32)[FEATURES.index("optortk_expr")][0]
                         for i in idx])
        return {
            "idx": idx, "norm": norm, "expr": expr,
            "sustained": norm[:, HET_STIM_ON:HET_STIM_OFF].mean(axis=1) - 1.0,
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
    het_achieved = _raw[:, HET_STIM_ON:HET_STIM_OFF].mean(axis=1)

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


    # --- Does anything you can MEASURE still help, once you have watched? --------
    # The data-side version of a feature ablation, and the figure the outline asks
    # for under "feature relevance over time". For each input the model receives,
    # how much does knowing it add to predicting the cell's later response, on top
    # of simply having watched that cell for k minutes?
    #
    # Every predictor is summarised over the SAME window the watching predictor uses,
    # so the comparison is "everything available in those k minutes", not "a window
    # against a scalar".
    HET_LATE = (50, 70)          # the response being predicted
    HET_OBS_LENGTHS = [3, 5, 8, 10, 15, 20, 25, 30]

    # u_t is the control input, not a property of the cell. It is included because
    # the interesting claim is that even the delivered dose stops helping once the
    # encoder has watched long enough. REMOVE IT FROM THIS LIST to show only the
    # cell-level covariates; nothing else needs changing.
    HET_ABLATE = ["u_t", "optortk_expr", "n_cells_200px", "fov_density"]


    def _r2(y, *cols):
        """R^2 of y on the given predictors, all rank-transformed (monotone, robust)."""
        from scipy.stats import rankdata
        X = np.column_stack([rankdata(c) for c in cols] + [np.ones(len(y))])
        yr = rankdata(y)
        beta, *_ = np.linalg.lstsq(X, yr, rcond=None)
        resid = yr - X @ beta
        return float(1 - resid.var() / yr.var())


    _featarr = np.stack([
        np.stack([np.asarray(hist_feats[_i], np.float32)[FEATURES.index(_f)][:HET_LEN]
                  for _f in HET_ABLATE])
        for _i in het["idx"]
    ])                                        # (cells, features, frames)

    _norm = het["norm"]
    _late = _norm[:, HET_LATE[0]:HET_LATE[1]].mean(axis=1)
    het_incremental = []
    for _k in HET_OBS_LENGTHS:
        _sl = slice(HET_STIM_ON, HET_STIM_ON + _k)
        _obs = _norm[:, _sl].mean(axis=1)
        _row = {"obs_min": _k, "r2_watch": _r2(_late, _obs)}
        for _j, _f in enumerate(HET_ABLATE):
            _fv = _featarr[:, _j, _sl].mean(axis=1)
            _row[f"add_{_f}"] = _r2(_late, _obs, _fv) - _row["r2_watch"]
            _row[f"alone_{_f}"] = _r2(_late, _fv)
        het_incremental.append(_row)
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
        HET_ABLATE,
        HET_CONDITION,
        HET_MIN_PER_FOV,
        HET_NONRESPONDER,
        HET_STIM_OFF,
        HET_STIM_ON,
        het,
        het_fov,
        het_incremental,
        het_nonresp_frac,
    )


@app.cell(hide_code=True)
def _(
    GRID,
    HET_CONDITION,
    HET_MIN_PER_FOV,
    HET_NONRESPONDER,
    HET_STIM_OFF,
    HET_STIM_ON,
    INK,
    MUTED,
    SERIES,
    STIM_BLUE,
    W_TEXT,
    df,
    het,
    het_fov,
    het_nonresp_frac,
    hist_meta,
    np,
    pl,
    plt,
    save_fig,
):
    # Provenance and the dose caveat. bo_v8 was a Bayesian-optimisation sweep, so the
    # pulse TIMING is identical in every field but the per-pulse exposure is not.
    # Field accounts for ~19% of the variance in `sustained`, and the low-dose fields
    # carry most of the barely-moving cells. The claim here is about cells, so the
    # dose-controlled number -- the spread INSIDE one field -- is the one that has to
    # carry it, and the pooled figure is reported only for context.
    het_experiment = ", ".join(sorted(set(
        hist_meta["original_experiment_name"].to_numpy()[het["idx"]])))
    het_n_fields = int(len(np.unique(het_fov)))

    _fov_expo = (df.filter((pl.col("stim_condition") == HET_CONDITION)
                           & (pl.col("m_t") > 0))
                   .group_by("fov").agg(pl.col("stim_exposure").mean().alias("e")))
    het_expo_lo = float(_fov_expo["e"].min())
    het_expo_hi = float(_fov_expo["e"].max())

    het_within_spread = float(np.median([
        np.quantile(het["sustained"][het_fov == _f], 0.9)
        - np.quantile(het["sustained"][het_fov == _f], 0.1)
        for _f in np.unique(het_fov)
        if (het_fov == _f).sum() >= HET_MIN_PER_FOV
    ]))
    het_pooled_spread = float(np.quantile(het["sustained"], 0.9)
                              - np.quantile(het["sustained"], 0.1))


    def _panel_het_heatmap(ax):
        """(a) Identical input, seven thousand different outputs."""
        order = np.argsort(het["sustained"])
        img = het["norm"][order]
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                       vmin=0.6, vmax=3.2, interpolation="nearest",
                       extent=(0, img.shape[1], 0, img.shape[0]))
        # A bar, not pulse markers: every frame in this window is lit and the exposure
        # ramps through it. The twelve triangles this replaces implied a pulse train
        # the protocol does not have.
        ax.plot([HET_STIM_ON, HET_STIM_OFF], [img.shape[0] * 1.012] * 2,
                color=STIM_BLUE, lw=3.0, solid_capstyle="butt", clip_on=False)
        ax.set_xlabel("time (min)          ▬ = illuminated, ramping exposure")
        ax.set_ylabel(f"cells, sorted by response  (n = {img.shape[0]:,})")
        ax.set_yticks([])
        ax.set_title(f"a  One protocol, {img.shape[0]:,} cells", loc="left",
                     fontweight="bold")
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
        ax.annotate(f"{het_nonresp_frac:.0%} within\n10% of baseline",
                    xy=(HET_NONRESPONDER * 0.35, ax.get_ylim()[1] * 0.62),
                    xytext=(1.05, ax.get_ylim()[1] * 0.50),
                    fontsize=7.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.set_xlabel("sustained elevation  (fold − 1)")
        ax.set_ylabel("cells")
        ax.set_xlim(lo_x, hi_x)
        ax.set_title("b  Spread of response", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.98, 0.80, f"{(s > hi_x).mean():.1%} above {hi_x:g}, not shown",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color=MUTED)
        # Pooling across fields folds in the dose sweep. The dose-controlled
        # spread is what the cell-to-cell claim actually rests on.
        ax.text(0.98, 0.68,
                f"p90−p10 = {het_pooled_spread:.2f} pooled,\n"
                f"{het_within_spread:.2f} within one field",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color=MUTED, linespacing=1.6)


    fig_heterogeneity = plt.figure(figsize=(W_TEXT, 4.4))
    _gs7 = fig_heterogeneity.add_gridspec(1, 2, width_ratios=[1.0, 0.88], wspace=0.42,
                             left=0.085, right=0.965, top=0.90, bottom=0.125)
    _panel_het_heatmap(fig_heterogeneity.add_subplot(_gs7[0, 0]))
    _panel_het_distribution(fig_heterogeneity.add_subplot(_gs7[0, 1]))
    save_fig(fig_heterogeneity, "heterogeneity")
    fig_heterogeneity
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # model-accuracy — does the model predict?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
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
    W_TEXT,
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
        # Bottom right, not top left: both curves rise with lead time so the top-left
        # corner is empty — but the control-horizon rule stands at x=8, a quarter of
        # the way across, and ran straight through the legend box there.
        ax.legend(frameon=False, loc="lower right", fontsize=7)
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
        ax.text(0.04, 0.94,
                f"$R^2$ = {r2:.3f}\nRMSE = {model_rmse[k]:.3f}\n"
                f"n = {len(x):,} cells",
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK,
                linespacing=1.5)
        ax.set_xlabel("observed CNR")
        ax.set_ylabel("predicted CNR")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(f"c  {CONTROL_HORIZON} min ahead, held out",
                     loc="left", fontweight="bold")


    fig_accuracy = plt.figure(figsize=(W_TEXT, 5.5))
    # Nested: the trace and its light strip are one unit (tight), well separated from
    # the two summary panels below.
    _gs = fig_accuracy.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42,
                            left=0.10, right=0.985, top=0.94, bottom=0.09)
    _top = _gs[0].subgridspec(2, 1, height_ratios=[1.0, 0.26], hspace=0.08)
    _bot = _gs[1].subgridspec(1, 2, wspace=0.26)
    _ax_a = fig_accuracy.add_subplot(_top[0])
    _panel_forecasts(_ax_a, fig_accuracy.add_subplot(_top[1], sharex=_ax_a))
    _panel_horizon(fig_accuracy.add_subplot(_bot[0]))
    _panel_scatter(fig_accuracy.add_subplot(_bot[1]))
    save_fig(fig_accuracy, "model-accuracy")
    fig_accuracy
    return CONTROL_HORIZON, fig_accuracy


@app.cell(hide_code=True)
def _(CONTROL_HORIZON, mo, model_rmse, persist_rmse):
    mo.md(f"""
    ### What panel (b) is actually saying

    **The model's error stops growing; the do-nothing reference's does not.** RMSE at a
    one-minute lead is **{model_rmse[0]:.3f}** against persistence's
    **{persist_rmse[0]:.3f}**; at the {CONTROL_HORIZON}-minute control horizon,
    **{model_rmse[CONTROL_HORIZON - 1]:.3f}** against **{persist_rmse[CONTROL_HORIZON - 1]:.3f}**;
    at 30 minutes, **{model_rmse[-1]:.3f}** against **{persist_rmse[-1]:.3f}**. So from about
    the control horizon onward the model is flat — it loses almost nothing over the next
    twenty-two minutes — while persistence keeps degrading. That flatness is what makes a
    30-frame horizon usable for planning at all, and it is the panel's real claim: not that
    the model is accurate, but that its accuracy does not decay over the window the
    controller has to reason about.

    **It also resolves a live-experiment result that reads as a model failure and is not.**
    The v13 post-mortem recorded *no one-step skill* — `pred_cnr_h1` MAE at or marginally
    worse than persistence (0.043 vs 0.042, 0.046 vs 0.044). Offline, at the same lead, the
    model is **{model_rmse[0]:.3f}** against **{persist_rmse[0]:.3f}**, a factor of
    {persist_rmse[0] / model_rmse[0]:.1f}. The two cannot both describe the model, and the
    offline number is the one computed on held-out cells with a fixed protocol. The live
    figure was a scoring artefact — v13 also ran at 1.83 min/frame against a declared 1.0,
    so its "one-step" predictions were being scored against a step that was almost twice as
    long as the one they were made for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### forecast-examples — success and failure, side by side

    A variant of panel (a) that splits the two behaviours into two cells instead
    of hoping one cell shows both.

    Both cells are chosen by stated rules and both are tail cases. The typical cell is at
    the median overall error among cells that actually respond; the hard cell is at the
    **95th percentile of overall forecast error** — the percentile rather than the maximum,
    so it is a representative bad cell and not one pathological track.

    It used to be chosen differently. The failure the model was assumed to have is specific
    — it cannot anticipate **endogenous ERK pulses**, which by construction are not in the
    light input — so the hard cell was selected by scoring dark and lit horizon frames
    separately and taking a cell that tracked well under light and diverged in the dark.
    That split turned out to separate almost nothing (0.003 CNR of median difference across
    7,200 cells) and has been removed everywhere. See the starvation section below for why
    the premise was wrong: there were almost no endogenous pulses in this data to miss.

    The light strip still distinguishes lit from unlit frames, and "off" is defined per cell
    as `u_t` at that cell's own minimum — not `u_t == 0`. Only 1,905 of 7,237 test cells
    ever reach exact zero; the rest sit on a nonzero fluence floor, so a global zero test
    would call most cells permanently lit.
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


    def score_forecast(cell_idx):
        """Forecast error for one cell, scanned across the whole track.

        Scanning rather than sampling three fixed origins: whether a cell fails
        depends on *when* the forecast is launched, and three arbitrary launch points
        can miss the excursion entirely.

        This used to score dark and lit horizon frames separately. It does not any
        more: over 7,200 test cells the split moved the median error by 0.003 CNR
        against a typical error of 0.065 — real in direction, negligible in size, and
        at that n the p-value described the sample size rather than the effect. Two
        hand-picked example cells were the worst possible evidence for it, and they
        disagreed with each other. `dark_mask` survives because the light strip in
        forecast-examples still shades unlit frames.
        """
        cnr, _chans = cell_arrays(cell_idx)
        origins = valid_origins(len(cnr))
        _m, _s, err, _dark = origin_errors(cell_idx, origins)
        per_origin = err.mean(axis=1)
        return {
            "cell": cell_idx,
            "T": len(cnr),
            "amp": float(np.ptp(cnr)),
            "n_origins": len(origins),
            "err": float(err.mean()),
            "worst_origin": int(origins[per_origin.argmax()]),
            "worst_origin_err": float(per_origin.max()),
        }


    # Every test cell long enough to fit three forecast origins (~3 min).
    _cells = [j for j in range(len(hist_cnr[split["test"]]))
              if len(hist_cnr[split["test"]][j]) >= 80]
    forecast_scores = pl.DataFrame([score_forecast(j) for j in _cells])
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
    # Two cells, two stated rules. Both are tail cases by construction and are
    # labelled as such rather than presented as typical.
    _elig = forecast_scores.filter(pl.col("n_origins") >= 5)

    # Typical: median overall error among cells that actually respond (amplitude
    # above the population median), so the panel is not a flat cell predicted flat.
    _resp = _elig.filter(pl.col("amp") >= _elig["amp"].median()).sort("err")
    CELL_TYPICAL = int(_resp["cell"][_resp.height // 2])

    # Hard case: the 95th percentile of OVERALL forecast error, taken at the
    # percentile rather than the maximum so it is a representative bad cell and not
    # one pathological track. It was previously selected on the dark/lit split, which
    # is gone — that rule made the panel look like evidence about *why* cells fail,
    # which it never was.
    _fail = _elig.sort("err")
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
    W_TEXT,
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
        ax.set_title(title, loc="left", fontweight="bold", pad=31)
        ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", linespacing=1.45, fontsize=7.5, color=MUTED)

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

    fig_examples = plt.figure(figsize=(W_TEXT, 4.1))
    _gs8b = fig_examples.add_gridspec(2, 2, height_ratios=[1.0, 0.3], hspace=0.1, wspace=0.26,
                               left=0.085, right=0.99, top=0.775, bottom=0.16)
    _a0 = fig_examples.add_subplot(_gs8b[0, 0])
    forecast_pair_panel(
        _a0, fig_examples.add_subplot(_gs8b[1, 0], sharex=_a0), CELL_TYPICAL,
        spread_origins(_t["T"]),
        "a  A typical cell",
        f"forecasts started at evenly spaced frames\n"
        f"mean |error|  {_t['err']:.3f} CNR",
    )
    _a1 = fig_examples.add_subplot(_gs8b[0, 1])
    forecast_pair_panel(
        _a1, fig_examples.add_subplot(_gs8b[1, 1], sharex=_a1), CELL_FAILURE,
        worst_origins(CELL_FAILURE),
        "b  A failure mode",
        f"forecasts started at this cell's highest-error frames\n"
        f"mean |error|  {_f['err']:.3f} CNR",
    )
    save_fig(fig_examples, "forecast-examples")
    fig_examples
    return (fig_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dot: the forecast start. Band: ±1σ. Grey: light off, at this cell's own fluence floor.
    Mean |error| is the mean absolute deviation from observed CNR over the 30 min horizon.
    """)
    return


@app.cell(hide_code=True)
def _(forecast_scores, last_abs, np, pl, pred_abs, test_metrics, true_abs):
    from scipy.stats import spearmanr

    # How many cells and forecasts the panels below are drawn from.
    forecast_extent = {
        "n_cells": forecast_scores.height,
        "n_forecasts": int(forecast_scores["n_origins"].sum()),
    }

    # Shrinkage: does the model pull every cell toward the population response?
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
    return forecast_extent, shrinkage, spearmanr


@app.cell(hide_code=True)
def _(forecast_extent, mo, shrinkage):
    mo.md(f"""
    ### What panel (b) is, and is not, evidence for

    **It is a tail case, and it is labelled as one.** Panel (b) is the cell at the 95th
    percentile of overall forecast error among the {forecast_extent['n_cells']:,} test cells
    scored here ({forecast_extent['n_forecasts']:,} forecasts) — taken at the percentile
    rather than the maximum so it is a representative bad cell and not one pathological
    track. It is a real cell and a real failure. It is not evidence about **why** cells
    fail, and no mechanism should be read off it.

    *(This panel used to be selected by a dark-vs-lit error split — the hypothesis being
    that error concentrates where the model cannot see the cause of a change, since
    endogenous ERK pulses are in neither of its inputs. That split was dropped: across
    7,200 cells it moved the median error by 0.003 CNR against a typical error of 0.065,
    and the two example cells disagreed on its sign. A near-null criterion is a bad way to
    choose an illustration.)*

    **The other reading, and how far it actually goes.** Panel (b) *looks* like shrinkage
    toward the population: the model undershoots this cell's plateau and over-predicts its
    decay once light stops, as though forecasting a more typical cell than this one. That
    reading comes from looking at the panel, so it was tested directly — signed error
    against the cell's CNR **before** the forecast window (an encoder input, not the target,
    so the same variable does not sit on both sides and manufacture the correlation).

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
    #### The cells were starved, and almost never fired on their own

    The cells in every one of these experiments were held in **serum starvation**,
    specifically to suppress spontaneous ERK activations. Whether that worked decides
    what this whole dataset can and cannot say about the disturbance the controller is
    eventually meant to reject.

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

    This is a limitation on the whole programme, not a detail of one figure. The
    endogenous ERK pulses that the MPC framing treats as the **central disturbance** —
    the "idling motor" that fires without stimulation — were suppressed by design in
    every training experiment, and the live experiments use the same starvation
    protocol. So **there were almost no endogenous pulses in this data to find**,
    nothing here tests the model against them, and no claim about rejecting them is
    supported by anything measured so far.
    """)
    return


@app.cell
def _(fig_accuracy, fig_examples):
    fig_accuracy, fig_examples
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # uncertainty-calibration — is the uncertainty honest?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
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
    ARM_RAMP,
    CAL_LEVELS,
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
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
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)
        ax.text(0.97, 0.06, "above the line = under-confident",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


    def _panel_pit(ax):
        """(b) The asymmetry check, split by lead time.

        Panels (a) and (c) both score CENTRAL intervals, i.e. |PIT - 0.5|. That folds
        the two tails together, so a shortfall in one tail cancels against a surplus in
        the other and the curve still looks calibrated. PIT keeps the tails apart.

        Pooled over the horizon the lower tail looks light, which reads as a standing
        defect. It is not: it is a lead-time gradient. At one step the model is
        over-cautious and leaves far too much room below; by thirty steps both tails
        are on nominal. Splitting the histogram is the only way to see that the
        pooled number is an average over a trend, not a property of the model.
        """
        pit2 = calib["pit"]                       # (origins, lead times)
        pit = pit2.ravel()
        _bins = dict(bins=40, range=(0, 1), density=True)
        ax.hist(pit, color=GRID, lw=0, label="all lead times", **_bins)

        # Lead time is ordinal, so the two ends of the ramp rather than two hues.
        for _sl, _c, _lab in ((slice(0, 5), ARM_RAMP[0], "lead 1-5 min"),
                              (slice(20, 30), ARM_RAMP[3], "lead 21-30 min")):
            ax.hist(pit2[:, _sl].ravel(), histtype="step", color=_c, lw=1.6,
                    label=_lab, **_bins)
        ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")

        _lo_1 = float((pit2[:, 0] < 0.1).mean())
        _lo_30 = float((pit2[:, -1] < 0.1).mean())

        ax.set_xlabel("PIT  =  F(observed)")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1)
        # Zoomed: a calibrated PIT sits at 1.0 everywhere, so the whole diagnostic
        # lives in a narrow band and a 0-based axis would show a flat block.
        ax.set_ylim(0.6, 1.62)
        ax.set_title("b  Both tails, by lead time", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.3,
                  borderaxespad=0.2, labelspacing=0.25)
        # Short enough to sit beside the legend: a 40-character line and a legend
        # together need more than the 2.7 in this panel has.
        ax.text(0.98, 0.97,
                f"lower tail {_lo_1:.1%} → {_lo_30:.1%}\n"
                f"n = {pit2.shape[0]:,} × {pit2.shape[1]}\n"
                "y zoomed on 1.0",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                color=MUTED, linespacing=1.6)


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
        ax.set_title("d  Does σ track the error?", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    # Four panels, four different questions, and none of them substitutes for another:
    #   (a) do the stated intervals cover — and does the Gaussian shortcut mislead?
    #   (b) is the SHAPE right, including asymmetry that (a) and (c) cannot see?
    #   (c) does calibration hold across the whole horizon the controller plans over?
    #   (d) does sigma DISCRIMINATE, or is it one wide number emitted everywhere?
    # (b) was briefly cut as a duplicate of (a). It is not: (a) and (c) score central
    # intervals, |PIT - 0.5|, which folds the tails together.
    fig_calibration = plt.figure(figsize=(W_TEXT, 6.0))
    _gs9 = fig_calibration.add_gridspec(2, 2, hspace=0.62, wspace=0.44,
                             left=0.105, right=0.98, top=0.94, bottom=0.085)
    _panel_reliability(fig_calibration.add_subplot(_gs9[0, 0]))
    _panel_pit(fig_calibration.add_subplot(_gs9[0, 1]))
    _panel_coverage_horizon(fig_calibration.add_subplot(_gs9[1, 0]))
    _panel_sharpness(fig_calibration.add_subplot(_gs9[1, 1]))
    save_fig(fig_calibration, "uncertainty-calibration")
    fig_calibration
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## uncertainty-calibration-alt — the same coverage, scored as one gaussian instead of the mixture
    """)
    return


@app.cell(hide_code=True)
def _(
    CAL_LEVELS,
    CONTROL_HORIZON,
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
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
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)
        ax.text(0.97, 0.06, "above the line = under-confident",
                transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


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
        ax.set_title("c  σ against error", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    # Four panels, four different questions, and none of them substitutes for another:
    #   (a) do the stated intervals cover — and does the Gaussian shortcut mislead?
    #   (b) is the SHAPE right, including asymmetry that (a) and (c) cannot see?
    #   (c) does calibration hold across the whole horizon the controller plans over?
    #   (d) does sigma DISCRIMINATE, or is it one wide number emitted everywhere?
    # (b) was briefly cut as a duplicate of (a). It is not: (a) and (c) score central
    # intervals, |PIT - 0.5|, which folds the tails together.


    def _panel_tails_vs_lead(ax):
        """(b) Coverage error by tail, against lead time.

        Replaces the PIT histogram AND the folded coverage curve. Panels that score
        central intervals, |PIT - 0.5| <= level/2, cannot see one-sided error: a
        shortfall in one tail cancels a surplus in the other. Splitting the two tails
        and plotting the DEPARTURE from nominal shows the asymmetry directly, and
        shows that it is confined to short lead.
        """
        pit = calib["pit"]                       # (N, F) exact mixture CDF
        lead = np.arange(1, pit.shape[1] + 1)
        lo = 100 * ((pit < 0.10).mean(axis=0) - 0.10)
        hi = 100 * ((pit > 0.90).mean(axis=0) - 0.10)
        ax.axhline(0, color=INK, lw=1.0)
        ax.plot(lead, lo, color=SERIES[0], lw=2.0, marker="o", ms=3,
                label="lower tail (below the 10th)")
        ax.plot(lead, hi, color=SERIES[1], lw=2.0, marker="s", ms=3,
                label="upper tail (above the 90th)")
        ax.axvline(CONTROL_HORIZON, color=MUTED, lw=0.8, ls="--")
        ax.annotate("control\nhorizon", xy=(CONTROL_HORIZON, 0),
                    xytext=(CONTROL_HORIZON + 1.2, lo.min() * 0.55),
                    fontsize=6, color=MUTED, va="center")
        ax.set_xlabel("forecast lead time (min)")
        ax.set_ylabel("coverage error\n(percentage points)")
        ax.set_xlim(1, pit.shape[1])
        ax.set_ylim(lo.min() * 1.15, max(hi.max() * 1.15, 1.9))
        ax.set_title("b  Which tail, and when", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="lower right",
                  handlelength=1.4, borderaxespad=0.2)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.97,
                f"lower tail {lo[0]:+.1f} pp at one step,\n{lo[-1]:+.1f} by thirty",
                transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
                linespacing=1.5)


    # Three panels rather than four: (b) supersedes both the PIT histogram and the
    # folded coverage-vs-lead curve, which measured the same property with the tails
    # added together.
    fig_unc_alt = plt.figure(figsize=(W_TEXT, 3.5))
    _gua = fig_unc_alt.add_gridspec(1, 3, wspace=0.46,
                                    left=0.095, right=0.985, top=0.88, bottom=0.19)
    _panel_reliability(fig_unc_alt.add_subplot(_gua[0, 0]))
    _panel_tails_vs_lead(fig_unc_alt.add_subplot(_gua[0, 1]))
    _panel_sharpness(fig_unc_alt.add_subplot(_gua[0, 2]))
    save_fig(fig_unc_alt, "uncertainty-calibration-alt")
    fig_unc_alt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # encoder-needs — how much of the past does the model use?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
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
    HET_ABLATE,
    INK,
    MEMORY_CAPS,
    MEMORY_TMIN,
    MUTED,
    SERIES,
    STIM_BLUE,
    W_TEXT,
    cell_arrays,
    dark_mask,
    het_incremental,
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
        ax.set_title("a  One cell, three context budgets",
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
        ax.set_ylabel("forecast MAE (CNR)")
        ax.set_ylim(0, float(c["mae"].max()) * 1.12)
        ax.set_title("b  Error vs available history", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _ood_band(ax)
        _lo, _hi = float(c["mae"].min()), float(c["mae"].max())
        ax.annotate(f"{_hi / _lo:.1f}x", xy=(60, _lo), xytext=(26, _hi * 0.62),
                    fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))


    def _panel_watching(ax):
        """(c) Once you have watched the cell, is anything you can MEASURE still worth
        knowing? One curve per input the model receives.

        Plotted as the INCREMENT over watching alone, because the baseline is rising
        steeply across this axis and two quantities on one scale would hide it.
        """
        d = het_incremental
        k = d["obs_min"].to_numpy()

        # u_t is the control input rather than a property of the cell, so it is drawn
        # apart from the covariates — dashed and grey. Deleting these two lines and
        # its entry in HET_ABLATE removes it cleanly.
        style = {
            "u_t": dict(color=MUTED, lw=1.6, ls="--", label="delivered light"),
            "optortk_expr": dict(color=SERIES[0], lw=2.2, marker="o", ms=4,
                                 label="optoRTK expression"),
            "n_cells_200px": dict(color=SERIES[1], lw=1.8, marker="s", ms=3.5,
                                  label="local crowding"),
            "fov_density": dict(color=SERIES[2], lw=1.8, marker="^", ms=3.5,
                                label="field density"),
        }
        ax.axhline(0, color=INK, lw=0.8)
        for f in HET_ABLATE:
            ax.plot(k, d[f"add_{f}"].to_numpy(), **style[f])

        _w = d["r2_watch"].to_numpy()
        ax.text(0.97, 0.97,
                f"watching alone: R² {_w[0]:.2f} → {_w[-1]:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color=MUTED)

        ax.set_xlabel("minutes of response observed")
        ax.set_ylabel("R² added over watching alone")
        ax.set_xlim(k.min(), k.max())
        ax.set_title("c  What else to measure", loc="left",
                     fontweight="bold")
        ax.legend(frameon=False, loc="upper right", fontsize=6,
                  handlelength=1.6, labelspacing=0.3, borderaxespad=0.2,
                  bbox_to_anchor=(1.0, 0.88))
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    # Two questions with one answer: how much of a cell's own past the encoder needs,
    # and whether a measured covariate adds anything once it has that past. They were
    # two figures; the second was never independent of the first.
    fig_enc = plt.figure(figsize=(W_TEXT, 5.4))
    _gse = fig_enc.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.46,
                                left=0.10, right=0.975, top=0.93, bottom=0.085)
    _te = _gse[0].subgridspec(2, 1, height_ratios=[1.0, 0.26], hspace=0.08)
    _be = _gse[1].subgridspec(1, 2, wspace=0.40)
    _ae = fig_enc.add_subplot(_te[0])
    _panel_memory_example(_ae, fig_enc.add_subplot(_te[1], sharex=_ae))
    _panel_memory_error(fig_enc.add_subplot(_be[0]))
    _panel_watching(fig_enc.add_subplot(_be[1]))
    save_fig(fig_enc, "encoder-needs")
    fig_enc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## encoder-needs — five minutes of watching beats knowing the receptor level

    Split out of heterogeneity. It looks like part of the heterogeneity argument and is not:
    heterogeneity establishes that identical light produces a continuum of responses, which is
    a property of the **cells**. This asks what *predicts* where in that continuum a cell
    lands, which is the question the model architecture answers, and it belongs beside the
    memory ablation rather than in front of it.

    No model is involved — this is the data-side counterpart of an ablation. Each cell's
    later response (frames 50-70) is predicted from its own observed early response, and
    expression is then added on top. If a covariate measured once before the run carried
    the heterogeneity, conditioning on it would be enough and a full-history encoder would
    be unnecessary. It does not, and so it is not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## expression-normalisation — why optoRTK expression is ranked, not rescaled

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
    return MC_REFERENCE, mcitrine_by_exp, mcitrine_summary


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MC_REFERENCE,
    MUTED,
    SERIES,
    W_TEXT,
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


    fig_norm, _axn = plt.subplots(1, 2, figsize=(W_TEXT, 3.5))
    _panel_median_scaled(_axn[0])
    _panel_qq(_axn[1])
    fig_norm.tight_layout(pad=1.3, w_pad=2.8)
    save_fig(fig_norm, "expression-normalisation")
    fig_norm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## expression-replication — the relationship replicates, even though the values do not compare

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
    return replication_drive, replication_meta_ci, replication_meta_rho


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
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


    fig_repl = plt.figure(figsize=(W_TEXT, 5.3))
    _gsr = fig_repl.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.48, wspace=0.42,
                                 left=0.24, right=0.97, top=0.93, bottom=0.09)
    _panel_forest(fig_repl.add_subplot(_gsr[0, :]))
    _panel_curves(fig_repl.add_subplot(_gsr[1, 0]))
    _panel_drive(fig_repl.add_subplot(_gsr[1, 1]))
    save_fig(fig_repl, "expression-replication")
    fig_repl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### expression-vs-fluence — the same relationship against delivered light, not just how often

    Duty cycle counts frames, not photons — a session can be lit on every frame at
    a weak dose, or rarely at a strong one. Replacing it with **time-averaged
    fluence** (mean mJ/cm² per frame above each cell's dark floor) folds pulse
    strength in, and changes the picture: the trend is not monotone but an
    **inverted U**. Expression predicts response only in a middle band of drive.

    That is why the duty-cycle version reported ρ = +0.89 — duty happened to be
    monotone because the saturating session also ran at low duty. On the fluence
    axis a rank correlation is simply the wrong summary, and the shape is the
    result.

    The shape rests on its two end points, and both come from sessions that are
    outliers in other respects, so it is suggestive rather than established. It could
    be replicated if the claim needs to carry weight.
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
    W_TEXT,
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


    fig_fluence = plt.figure(figsize=(W_TEXT, 5.3))
    _gsf = fig_fluence.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.48,
                                    wspace=0.42, left=0.24, right=0.97, top=0.93, bottom=0.09)
    _panel_forest_fl(fig_fluence.add_subplot(_gsf[0, :]))
    _panel_curves_fluence(fig_fluence.add_subplot(_gsf[1, 0]))
    _panel_fluence(fig_fluence.add_subplot(_gsf[1, 1]))
    save_fig(fig_fluence, "expression-vs-fluence")
    fig_fluence
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diversity of stimulation for a single objective?

    The first version of this figure counted **commands**: at each frame, how spread out
    across the dose ladder are the rungs handed to different cells. On v15/v16 that
    number is unreadable, because those runs asked for a level the ladder could not
    reach and the controller sat at its top rung for most of the run. With most cells
    pinned to the same rung the entropy measures saturation, not choice.

    This version measures **responses**, and controls the objective by construction.
    The reference repeats: one cycle of the same demanded waveform can be cut out of the
    record for every cell, giving a fixed-length vector per (cell, cycle). Two sources of
    variation are then separable, because the design separates them:

    * **which cell** — the same cell across different cycles;
    * **which moment** — different cells in the same minute of the run.

    ### Why this run

    `2026-08-07_InferenceCNRhold_12h_v19` is the first run that can carry the analysis.
    720 controlled frames at a verified 1.000 min/frame with no degraded frames; a
    reference (0.85 → 1.10) the ladder can actually reach, so the actuator spends
    25-50 % of frames dark rather than railed; and four period arms, which turns the
    number of repeats per cell from v16's three or four into up to thirty-three.

    It also sets `n_phase_groups = 4`. Cells within one field are split into four groups
    offset by a quarter period, so **at any wall-clock minute four groups in the same
    dish sit at different points of the same cycle**. Cycle phase and elapsed time are
    decoupled by construction — which is what lets "a property of the moment" be tested
    rather than assumed.
    """)
    return


@app.cell(hide_code=True)
def _(Path, json, materials_path, parse_arm, pl):
    # The oscillation period sweep. Every figure below reads all eight FOVs, and
    # `parse_arm` re-reads the whole 147 MB log on each call, so parse once and cache.
    OSC_RUN = Path("/Volumes/imaging.data/mic01-imaging/314lipczuk/"
                   "2026-08-07_InferenceCNRhold_12h_v19/oscillation_run_v2.jsonl")
    OSC_CACHE = materials_path("serving_v19_cells.parquet")

    if OSC_CACHE.exists():
        osc_df = pl.read_parquet(OSC_CACHE)
        with open(OSC_RUN) as _f:
            osc_startup = json.loads(_f.readline())
    else:
        osc_df, _osc_t0, osc_startup = parse_arm(OSC_RUN, tuple(range(8)))
        osc_df.write_parquet(OSC_CACHE)

    # Periods come from the startup record, never from a hardcoded table: the policy is
    # the only thing that knows which field got which arm.
    OSC_ARMS = {
        int(_k): {
            "arm": _v["requested"]["arm"],
            **{_n: _v["objective"]["reference"][_n] for _n in
               ("period_min", "settle_min", "n_phase_groups", "low", "high",
                "t_low_min", "t_rise_min", "t_high_min", "t_fall_min")},
        }
        for _k, _v in osc_startup["policies"]["fov"].items()
    }
    OSC_SWING = OSC_ARMS[0]["high"] - OSC_ARMS[0]["low"]

    # The arm shown wherever one has to be picked. 60 min is the cleanest: real
    # modulation, and neither rail dominates (8 % of frames at the top rung against
    # 52 % for the 30 min arm, which is partly censored).
    OSC_FEATURED = 60

    pl.DataFrame([{"fov": _k, "arm": _v["arm"], "period_min": _v["period_min"],
                   "settle_min": _v["settle_min"], "phase_groups": _v["n_phase_groups"],
                   "low": _v["low"], "high": _v["high"]}
                  for _k, _v in sorted(OSC_ARMS.items())])
    return OSC_ARMS, OSC_FEATURED, OSC_SWING, osc_df


@app.cell(hide_code=True)
def _(OSC_ARMS, np, osc_df, pl):
    OSC_COLS = ("raw_cnr", "exposure_ms", "r_t", "plan_cost", "optortk_expr", "nuc_area")
    OSC_MIN_CYCLES = 6      # enough cycles to split a cell's own record early vs late


    def osc_windows(df, arms, min_cycles=OSC_MIN_CYCLES):
        """One fixed-length vector per (cell, cycle), cut at the logged `low_hold` onsets.

        Cutting on the server's own `segment` label rather than on arithmetic matters
        here: the phase offsets are period/4, which is half-integer for the 30 and 70 min
        arms (0/7.5/15/22.5 and 0/17.5/35/52.5) while frames are whole minutes. A window
        start computed as `settle + phase + k*period` misplaces those two groups.

        A cycle is kept only when its frames are contiguous and its last frame is labelled
        `fall`, which is what separates a whole cycle from the tail of the settle block.
        Cells contributing fewer than `min_cycles` are dropped entirely: a short track
        cannot support a within-cell early-vs-late contrast, and including it drags every
        per-cell median toward zero.
        """
        meta, arrs = [], {c: [] for c in OSC_COLS}
        for (fov, particle), g in df.sort("control_frame").group_by(
                ["fov", "particle"], maintain_order=True):
            a = arms[fov]
            P = int(a["period_min"])
            cf = g["control_frame"].to_numpy()
            seg = g["segment"].to_numpy()
            at = {int(f): i for i, f in enumerate(cf)}
            cols = {c: g[c].to_numpy() for c in OSC_COLS}
            onsets = [i for i in range(len(seg))
                      if seg[i] == "low_hold" and (i == 0 or seg[i - 1] != "low_hold")]
            rows, vecs = [], []
            for i in onsets:
                s = int(cf[i])
                idx = [at.get(f) for f in range(s, s + P)]
                if any(j is None for j in idx) or seg[idx[-1]] != "fall":
                    continue
                idx = np.array(idx)
                rows.append({
                    "fov": fov, "particle": particle, "cell": f"{fov}_{particle}",
                    "arm": a["arm"], "period": P,
                    "phase_group": int(round(g["phase_offset_min"][0]
                                             / (P / a["n_phase_groups"]))),
                    "cycle": len(rows), "start_frame": s, "start_h": s / 60.0,
                })
                vecs.append({c: cols[c][idx].astype(np.float32) for c in OSC_COLS})
            if len(rows) < min_cycles:
                continue
            meta += rows
            for v in vecs:
                for c in OSC_COLS:
                    arrs[c].append(v[c])
        return pl.DataFrame(meta), arrs


    osc_meta, osc_arr = osc_windows(osc_df, OSC_ARMS)


    def osc_stack(mask, key):
        """The (n_windows, period) matrix for a boolean window mask."""
        return np.stack([osc_arr[key][i] for i in np.flatnonzero(mask)]).astype(np.float64)


    osc_yield = osc_meta.group_by("period").agg(
        pl.len().alias("windows"), pl.col("cell").n_unique().alias("cells"),
        (pl.col("cycle").max() + 1).alias("max_cycles")).sort("period")
    return osc_arr, osc_meta, osc_stack


@app.cell(hide_code=True)
def _(np, osc_meta, osc_stack, pl):
    # Did the cut land where it should? Fold each phase group separately and compare the
    # references. If the windows are aligned the four groups lie on top of each other.
    #
    # They do for the 20 and 60 min arms. For 30 and 70 the quarter-period offset is a
    # half-integer number of minutes (7.5 and 17.5) against a whole-minute frame grid, so
    # groups 1 and 3 sample the ramps half a minute out of step with groups 0 and 2. The
    # holds are untouched; the residual below is confined to the rise and the fall. Every
    # feature is measured against a window's own reference, so nothing downstream depends
    # on the four groups being interchangeable.
    _rows = []
    for _P in sorted(set(osc_meta["period"].to_list())):
        _m = (osc_meta["period"] == _P).to_numpy()
        _ref = osc_stack(_m, "r_t")
        _g = osc_meta["phase_group"].to_numpy()[_m]
        _prof = np.stack([np.median(_ref[_g == q], axis=0) for q in sorted(set(_g.tolist()))])
        _d = np.abs(_prof - _prof[0]).max(axis=0)
        _rows.append({"period": _P, "phase_groups": _prof.shape[0],
                      "max_ref_residual": round(float(_d.max()), 4),
                      "frames_affected": int((_d > 1e-9).sum()), "of_frames": _P})
    pl.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def _(np, osc_arr, osc_meta, pl):
    def osc_features(meta, arr):
        """Scalars per window.

        Everything is measured against the window's OWN logged `r_t`, so the half-minute
        sampling difference between phase groups (see `osc_windows`) cannot leak into a
        feature: it changes which frames of a ramp a cell saw, not what that cell's own
        reference was on those frames.
        """
        out = []
        for i in range(meta.height):
            y, d, r = arr["raw_cnr"][i], arr["exposure_ms"][i], arr["r_t"][i]
            P = len(y)
            m_lo, m_hi = r <= r.min() + 1e-6, r >= r.max() - 1e-6
            th = 2 * np.pi * np.arange(P) / P
            lit = d > 0
            out.append({
                "cnr_mean": float(y.mean()),
                # What the objective actually asks for: the level in the high hold
                # against the level in the low hold. Under-reads when the response
                # peaks after the hold has ended, which is the fast arms' problem.
                "amp": float(np.median(y[m_hi]) - np.median(y[m_lo])),
                "rmse": float(np.sqrt(((y - r) ** 2).mean())),
                "bias": float((y - r).mean()),
                "dose_mean": float(d.mean()),
                "frac_dark": float((d == 0).mean()),
                "frac_top": float((d >= 150).mean()),
                # Where in the cycle the light is put, as a circular centre of mass.
                "dose_phase": (float((np.arctan2((d * np.sin(th)).sum(),
                                                 (d * np.cos(th)).sum()) % (2 * np.pi))
                                     / (2 * np.pi) * P) if d.sum() > 0 else np.nan),
                "n_switches": int((lit[1:] != lit[:-1]).sum()),
                "plan_cost": float(np.nanmean(arr["plan_cost"][i])),
                "expr": float(np.nanmedian(arr["optortk_expr"][i])),
                "area": float(np.nanmedian(arr["nuc_area"][i])),
            })
        return meta.hstack(pl.DataFrame(out))


    osc_feat = osc_features(osc_meta, osc_arr)

    # Headline numbers weight every cell once: the median over a cell's own cycles, then
    # the median over cells. Pooling frames instead would weight cells by how long they
    # were tracked, and pooling windows would weight them by how many cycles they saw.
    osc_percell = osc_feat.group_by(["period", "cell"]).agg(
        pl.col(["amp", "rmse", "bias", "dose_mean", "frac_dark",
                "frac_top", "n_switches", "dose_phase"]).median())

    osc_summary = osc_percell.group_by("period").agg(
        pl.len().alias("cells"),
        pl.col("amp").median().round(3), pl.col("rmse").median().round(3),
        pl.col("bias").median().round(3), pl.col("dose_mean").median().round(1),
        pl.col("frac_dark").median().round(2), pl.col("frac_top").median().round(2),
        pl.col("n_switches").median()).sort("period")
    return (osc_feat,)


@app.cell(hide_code=True)
def _(np, osc_feat, osc_stack):
    def osc_identity(period, key="raw_cnr", normalise="centre", n_perm=200, seed=0):
        """Is a window's shape a property of the cell, or of the moment it happened in?

        Nearest-neighbour counting is the obvious test and the wrong one — with hundreds
        of candidates the same-cell base rate is a couple of percent whatever the truth
        is. Mean pairwise distance uses every pair, and permuting the cell labels gives
        the null directly.

        `centre` compares shape at the window's own level; `unit` also divides out the
        magnitude, so only the pattern is left.
        """
        m = (osc_feat["period"] == period).to_numpy()
        V = osc_stack(m, key)
        V = V - V.mean(axis=1, keepdims=True)
        if normalise == "unit":
            V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)

        cell = np.array(osc_feat["cell"])[m]
        start = osc_feat["start_frame"].to_numpy()[m]
        pg = osc_feat["phase_group"].to_numpy()[m]

        d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
        np.fill_diagonal(d, 0.0)
        off = ~np.eye(len(V), dtype=bool)

        def mean_where(mask):
            mask = mask & off
            n = mask.sum()
            return float((d * mask).sum() / n) if n else np.nan

        same_cell = cell[:, None] == cell[None, :]
        same_min = start[:, None] == start[None, :]
        same_ph = pg[:, None] == pg[None, :]
        near = np.abs(start[:, None] - start[None, :]) <= period

        dist = {
            "same cell,\ndifferent time": mean_where(same_cell & ~same_min),
            "different cell,\nsame minute": mean_where(~same_cell & same_min),
            "different cell, same hour,\ndifferent cycle phase":
                mean_where(~same_cell & ~same_ph & near),
            "different cell,\ndifferent time": mean_where(~same_cell & ~same_min),
        }

        obs = dist["different cell,\nsame minute"] - dist["same cell,\ndifferent time"]
        rng = np.random.default_rng(seed)
        null = np.array([
            (lambda s: mean_where(~s & same_min) - mean_where(s & ~same_min))(
                (lambda p: p[:, None] == p[None, :])(rng.permutation(cell)))
            for _ in range(n_perm)])
        return dist, {"gap": obs, "gap_pct": 100 * obs / dist["different cell,\nsame minute"],
                      "z": float((obs - null.mean()) / null.std()), "n": len(V), "V": V}


    def osc_embed(V, k=2):
        """Two dimensions of a window set, for looking at rather than deciding with."""
        c = V - V.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(c, full_matrices=False)
        return u[:, :k] * s[:k], (s ** 2 / (s ** 2).sum())[:k]

    return osc_embed, osc_identity


@app.cell(hide_code=True)
def _(OSC_FEATURED, osc_embed, osc_identity, pl):
    div_dist, div_stat = osc_identity(OSC_FEATURED, "raw_cnr", "centre")
    div_dose_dist, div_dose_stat = osc_identity(OSC_FEATURED, "exposure_ms", "unit")

    div_cnr_pcs, div_cnr_var = osc_embed(div_stat["V"])
    div_dose_pcs, div_dose_var = osc_embed(div_dose_stat["V"])

    pl.DataFrame([
        {"quantity": "response shape", **{k.replace("\n", " "): round(v, 4)
                                          for k, v in div_dist.items()},
         "gap %": round(div_stat["gap_pct"], 1), "z": round(div_stat["z"], 1)},
        {"quantity": "dose pattern", **{k.replace("\n", " "): round(v, 4)
                                        for k, v in div_dose_dist.items()},
         "gap %": round(div_dose_stat["gap_pct"], 1), "z": round(div_dose_stat["z"], 1)},
    ])
    return (div_stat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## response-identity — window embeddings, and what each cell was given
    """)
    return


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
    materials_path,
    np,
    osc_feat,
    osc_identity,
    pl,
    plt,
    save_fig,
):
    # --- Identity, measured with no model and no controller ----------------------
    # The version this replaces measured identity on v19 and v21, both closed loop.
    # There the controller doses each cell on its own state, so part of the identity
    # being evidenced is manufactured by the thing it is evidence for. Everything in
    # (a)-(c) below comes from `bo_v8`: a fixed pulse train, byte-identical light
    # within a field, no model in the loop.
    _ol = np.load(materials_path("openloop_identity_traces.npz"))
    _ol_own, _ol_others = _ol["own"], _ol["others"]
    _ol_bars = pl.read_parquet(materials_path("openloop_identity_bars.parquet"))
    _ol_fields = pl.read_parquet(materials_path("openloop_identity_w5.parquet"))

    _OL_YLIM = (float(np.percentile(np.concatenate(
        [_ol_own.ravel(), _ol_others.ravel()]), 1.5)) * 1.15,
                float(np.percentile(np.concatenate(
        [_ol_own.ravel(), _ol_others.ravel()]), 98.5)) * 1.15)


    def _panel_ol_own(ax):
        """(a) One cell, its own windows. The shape it keeps returning to."""
        for v in _ol_own:
            ax.plot(np.arange(len(v)), v, color=SERIES[0], lw=1.1, alpha=0.55)
        ax.plot(np.arange(_ol_own.shape[1]), np.median(_ol_own, axis=0), color=INK,
                lw=2.0, ls="--", label="its median")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("minute of the pulse cycle")
        ax.set_ylabel("CNR, window mean removed")
        ax.set_ylim(*_OL_YLIM)
        ax.set_title(f"a  One cell, {_ol_own.shape[0]} of its own windows",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_ol_others(ax):
        """(b) Different cells, the same minute, the same light."""
        for v in _ol_others:
            ax.plot(np.arange(len(v)), v, color=SERIES[1], lw=1.1, alpha=0.55)
        ax.plot(np.arange(_ol_others.shape[1]), np.median(_ol_others, axis=0),
                color=INK, lw=2.0, ls="--", label="their median")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(*_OL_YLIM)
        ax.set_xlabel("minute of the pulse cycle")
        ax.set_title(f"b  {_ol_others.shape[0]} cells, one window",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_ol_bars(ax):
        """(c) The three distances the claim rests on.

        Read the pair that differs in ONE thing: same cell against different cell,
        both compared across different windows. That gap is identity. The same-window
        bar is lower than either, which under a FIXED pulse train it should be --
        every cell is receiving the same light at that minute.
        """
        lab = _ol_bars["label"].to_list()
        val = _ol_bars["value"].to_numpy()
        y = np.arange(len(lab))[::-1]
        col = [SERIES[0], MUTED, SERIES[1]]
        ax.barh(y, val, height=0.55, color=col)
        for yi, v in zip(y, val):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=6.5, color=MUTED)
        ax.set_yticks(y, lab, fontsize=6.5)
        ax.set_xlim(0.28, max(val) * 1.10)
        ax.set_xlabel("mean distance between response shapes")
        ax.set_title("c  Identity, without a controller", loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _gap = 100 * (val[2] - val[0]) / val[2]
        ax.text(0.97, 0.93, f"{_gap:.1f}% closer to its own",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color=INK)


    # --- Does it replicate outside v19? -----------------------------------------
    # The distances themselves are not comparable across runs — different waveforms,
    # different window lengths — but the GAP is a ratio and therefore scale-free, the
    # same argument expression-replication uses for correlations across sessions. So
    # every arm of v19 and every pattern of v21 contributes one comparable estimate.
    #
    # v22 is deliberately absent: its objective was mis-set, a third of its cells were
    # unreachable, and it cannot speak to whether cells differ.
    _v19_id = []
    for _p in sorted(set(osc_feat["period"].to_list())):
        _d, _s = osc_identity(_p, key="raw_cnr", normalise="centre", n_perm=200)
        _v19_id.append({"run": "v19", "group": f"{_p} min", "gap_pct": _s["gap_pct"],
                        "z": _s["z"], "n": _s["n"]})

    _v21_id = (pl.read_parquet(materials_path("v21_identity_summary.parquet"))
                 .filter(pl.col("arm") == "all")
                 .select(pl.lit("v21").alias("run"), pl.col("pattern").alias("group"),
                         "gap_pct", "z", "n")
                 .to_dicts())
    div_replication = pl.DataFrame(_v19_id + _v21_id)


    def _panel_div_forest(ax):
        """(d) Identity against the interval between drives, open loop and closed.

        CAVEAT, unresolved: the offline windows are one pulse period long while the
        live windows are whole cycles, so period and window length are confounded
        across the two halves. The per-field periods are also the NOMINAL condition
        periods, which a later check found are not what every field actually runs.
        """
        off = pl.read_parquet(materials_path("identity_across_conditions.parquet"))
        rng = np.random.default_rng(0)

        for regime, face in (("pulses", SERIES[0]), ("ramp", "white")):
            s = off.filter(pl.col("regime") == regime)
            x = s["period"].to_numpy() * np.exp(rng.uniform(-.045, .045, s.height))
            ax.plot(x, s["gap_pct"], "o", ms=2.6, alpha=0.45, mfc=face,
                    mec=SERIES[0], mew=0.6, ls="none")
        med = (off.filter(pl.col("regime") == "pulses")
                  .group_by("period").agg(pl.col("gap_pct").median()).sort("period"))
        ax.plot(med["period"], med["gap_pct"], "-", color=SERIES[0], lw=1.8, zorder=5)
        ax.plot(med["period"], med["gap_pct"], "o", ms=7, color=SERIES[0], zorder=6,
                label="open loop, per field")

        v19 = div_replication.filter(pl.col("run") == "v19")
        ax.plot([int(g.split()[0]) for g in v19["group"]], v19["gap_pct"], "s", ms=7,
                color=SERIES[1], zorder=6, label="v19, closed loop")

        v21 = div_replication.filter(pl.col("run") == "v21")
        ax.plot([120] * v21.height, v21["gap_pct"], "^", ms=6.5, color=SERIES[2],
                zorder=6, label="v21, no fixed period")

        ax.set_xscale("log")
        ax.set_xticks([3, 5, 7, 20, 30, 60, 120],
                      ["3", "5", "7", "20", "30", "60", "v21"], fontsize=6.5)
        ax.minorticks_off()
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xlabel("minutes between drives")
        ax.set_ylabel("closer to its own windows (%)")
        ax.set_title("d  Identity against timescale", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2,
                  borderaxespad=0.2, labelspacing=0.25)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.97, 0.11,
                f"{off.height} offline fields, {int((off['gap_pct'] > 0).sum())} positive",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=MUTED)


    fig_div = plt.figure(figsize=(W_TEXT, 5.4))
    _gd2 = fig_div.add_gridspec(2, 2, hspace=0.52, wspace=0.34,
                                left=0.115, right=0.975, top=0.93, bottom=0.095)
    _axa = fig_div.add_subplot(_gd2[0, 0])
    _panel_ol_own(_axa)
    _panel_ol_others(fig_div.add_subplot(_gd2[0, 1]))
    _panel_ol_bars(fig_div.add_subplot(_gd2[1, 0]))
    _panel_div_forest(fig_div.add_subplot(_gd2[1, 1]))
    save_fig(fig_div, "response-identity")
    fig_div
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # history-swap — does giving every cell the same light let the model tell them apart?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
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
    improvement, and this would only repeat encoder-needs.
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
        keep = ("raw_cnr", "cnr_norm", "u_t_in", "fluence_out", "exposure_ms",
                "n_cells_200px", "optortk_expr", "nuc_area", "pred_cnr_h1",
                "plan_cost", "r_t", "segment", "phase_offset_min",
                "n_frames_seen", "dark", "shared_dose")
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
    return CAL_CHECKPOINT, cal_df, json, parse_arm


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
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, pl, plt, save_fig):
    # --- history-swap, rebuilt on a swap that actually swaps ------------------------
    # The previous version never re-ran the model. It computed one forecast per cell
    # and re-indexed them (`m[donor]`), scoring a level-matched neighbour's PREDICTION
    # against this cell's future. That measures how alike two cells' futures are, not
    # what the encoder takes from a cell's own past.
    #
    # Here the encoder is genuinely re-run on the donor's history while the decoder is
    # given THIS cell's own future light, so the light is controlled by construction
    # and the comparison isolates identity. Computed in
    # analyses/true_swap.py over v19 and v21 (both on checkpoint
    # enc_e_area_lean_2026-08-07_02.05.26); v22 is excluded, its objective was mis-set.
    #
    # The per-origin panel is gone: it asserted "no decay" from a scatter of 8 points
    # per run whose spread was of the same order as the effect it was reading.
    swap_pc = pl.read_parquet(materials_path("swap_percell.parquet"))
    swap_ex = pl.read_parquet(materials_path("swap_example.parquet"))
    SWAP_SHOWN = "v19, min 270"


    def _panel_swap_example(ax):
        """(a) One cell, and what the model says when given a neighbour's past."""
        t = swap_ex["t"].to_numpy()
        ax.plot(t, swap_ex["own_past"], color=INK, lw=1.5, zorder=4, label="this cell's past")
        ax.plot(t, swap_ex["donor_past"], color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=3,
                label="the donor's past")
        ax.plot(t, swap_ex["truth"], color=INK, lw=1.5, zorder=4)
        ax.plot(t, swap_ex["pred_own"], color=SERIES[0], lw=2.0, zorder=5,
                label="forecast from its own past")
        ax.plot(t, swap_ex["pred_donor"], color=SERIES[1], lw=2.0, zorder=5,
                label="forecast from the donor's")
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("minutes from the forecast start")
        ax.set_ylabel("CNR (absolute)")
        ax.set_title("a  Same level, different history", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=1)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


    def _panel_swap_paired(ax):
        """(b) The same comparison cell by cell, not as an average."""
        x = swap_pc["err_own"].to_numpy()
        y = swap_pc["err_donor"].to_numpy()
        lim = (0, float(max(x.max(), y.max())) * 1.05)
        ax.plot(x, y, "o", ms=3.5, alpha=0.45, color=SERIES[0])
        ax.plot(lim, lim, color=INK, lw=1.0, ls="--")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("error from its own past")
        ax.set_ylabel("error from a level-matched\ndonor's past", fontsize=8)
        ax.set_title("b  Cell by cell", loc="left", fontweight="bold")
        ax.text(0.96, 0.06,
                f"{SWAP_SHOWN}\n"
                f"worse for {100 * float((y > x).mean()):.0f}% of cells\n"
                f"donors differ in current CNR\nby {swap_pc['donor_gap'].median():.3f} (median)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK)


    fig_swap = plt.figure(figsize=(W_TEXT, 3.0))
    _g13 = fig_swap.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.42,
                                 left=0.085, right=0.98, top=0.90, bottom=0.155)
    _panel_swap_example(fig_swap.add_subplot(_g13[0, 0]))
    _panel_swap_paired(fig_swap.add_subplot(_g13[0, 1]))
    save_fig(fig_swap, "history-swap")
    fig_swap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## sensitivity-decline — fold the cycles: does the same demand get the same answer later?

    The reference repeats, so a cell's whole trajectory can be folded onto a single
    cycle. Splitting a cell's cycles into its own first and second half then asks the
    question directly — **the same cell, the same demanded waveform, four hours apart.**

    Splitting each cell at its own median cycle, rather than cutting the run at a fixed
    hour, keeps the comparison paired. Cells enter and leave the field over twelve hours,
    and an unpaired early-versus-late contrast would partly be measuring which cells
    survived.

    Two things can change, and they mean different things:

    * **magnitude** — the controller pushing harder for the same demand, which is what a
      loss of sensitivity looks like from the actuator's side;
    * **shape** — the controller changing *when* in the cycle it puts the light, which
      would mean the strategy itself changed and not just its gain.

    The null for shape is the same cell's early cycles split in two: whatever agreement a
    cell shows with itself inside one epoch is the ceiling that early-versus-late has to
    be read against.
    """)
    return


@app.cell(hide_code=True)
def _(np, osc_arr, osc_feat, osc_meta, pl):
    def osc_epochs(period, min_cycles=6, seed=0):
        """Per cell: its own first half of cycles against its own second half.

        Returns one row per cell with the magnitude and the shape of both epochs, for the
        dose and for the response, plus the within-epoch shape agreement that any
        early-versus-late number has to be read against.
        """
        m = np.flatnonzero((osc_feat["period"] == period).to_numpy())
        cell = np.array(osc_feat["cell"])[m]
        cyc = osc_feat["cycle"].to_numpy()[m]
        hour = osc_feat["start_h"].to_numpy()[m]
        amp = osc_feat["amp"].to_numpy()[m]
        rmse = osc_feat["rmse"].to_numpy()[m]
        top = osc_feat["frac_top"].to_numpy()[m]
        Y = np.stack([osc_arr["raw_cnr"][i] for i in m]).astype(np.float64)
        D = np.stack([osc_arr["exposure_ms"][i] for i in m]).astype(np.float64)
        rng = np.random.default_rng(seed)

        def unit(v):
            v = v - v.mean()
            return v / max(np.linalg.norm(v), 1e-9)

        rows, prof = [], []
        for c in np.unique(cell):
            k = np.flatnonzero(cell == c)
            k = k[np.argsort(cyc[k])]
            if len(k) < min_cycles:
                continue
            h = len(k) // 2
            e, l = k[:h], k[-h:]
            # Null: two halves of the EARLY block, i.e. the same epoch against itself.
            p = rng.permutation(e)
            a, b = p[:len(p) // 2], p[len(p) // 2:]
            rows.append({
                "cell": c, "period": period, "n_cycles": len(k),
                "h_early": float(np.median(hour[e])), "h_late": float(np.median(hour[l])),
                "amp_early": float(np.median(amp[e])), "amp_late": float(np.median(amp[l])),
                "dose_early": float(D[e].mean()), "dose_late": float(D[l].mean()),
                "rmse_early": float(np.median(rmse[e])), "rmse_late": float(np.median(rmse[l])),
                "top_early": float(np.median(top[e])), "top_late": float(np.median(top[l])),
                "cos_dose": float(unit(D[e].mean(0)) @ unit(D[l].mean(0))),
                "cos_dose_null": float(unit(D[a].mean(0)) @ unit(D[b].mean(0))),
                "cos_cnr": float(unit(Y[e].mean(0)) @ unit(Y[l].mean(0))),
                "cos_cnr_null": float(unit(Y[a].mean(0)) @ unit(Y[b].mean(0))),
            })
            prof.append({"cnr_early": Y[e].mean(0), "cnr_late": Y[l].mean(0),
                         "dose_early": D[e].mean(0), "dose_late": D[l].mean(0)})
        return pl.DataFrame(rows), prof


    OSC_PERIODS = sorted(set(osc_meta["period"].to_list()))
    osc_epoch = {p: osc_epochs(p) for p in OSC_PERIODS}
    osc_drift = pl.concat([v[0] for v in osc_epoch.values()])

    osc_drift_summary = osc_drift.group_by("period").agg(
        pl.len().alias("cells"),
        (pl.col("amp_late") - pl.col("amp_early")).median().round(4).alias("d_amp"),
        (pl.col("dose_late") - pl.col("dose_early")).median().round(1).alias("d_dose_ms"),
        (100 * (pl.col("dose_late") / pl.col("dose_early") - 1)).median().round(0).alias("d_dose_pct"),
        pl.col("cos_dose").median().round(3), pl.col("cos_dose_null").median().round(3),
        pl.col("cos_cnr").median().round(3), pl.col("cos_cnr_null").median().round(3),
        pl.col("top_late").median().round(2)).sort("period")
    return OSC_PERIODS, osc_drift


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    OSC_PERIODS,
    SERIES,
    W_TEXT,
    materials_path,
    np,
    pl,
    plt,
    save_fig,
):
    # PERIOD_COLOUR is defined here and consumed by four other cells.
    PERIOD_COLOUR = dict(zip(OSC_PERIODS, [plt.cm.viridis(x) for x in (0.05, 0.35, 0.62, 0.85)]))

    # --- Sensitivity decline: the same demand, later in the run --------------------
    # Every run whose objective repeats contributes one comparison: the FIRST repeat
    # of a demand against the LAST repeat of the same demand, on the same cells.
    #
    # Two kinds of repeat, each on its own terms. v10, v19 and v21 drive a fixed
    # waveform, so a repeat is one period of the arm's own reference. v23 and v24 are
    # run-ups -- a repeating block whose demand climbs each time -- so their repeats
    # are read off the block label the server records, and first and last are matched
    # on the same demand LEVEL. Without that matching a run-up's later blocks ask for
    # more, and the extra light they need would be the design rather than a decline.
    #
    # Cells are paired: the same cell in both repeats. That is what stops the
    # comparison being about which cells survived to the end of the run.
    cyc_traj = pl.read_parquet(materials_path("cycle_traj.parquet"))
    cyc_dose = pl.read_parquet(materials_path("cycle_dose.parquet"))

    DEC_MPF = {"v10": 1.42}          # every other run held 1.00 min/frame
    DEC_C = {"first": SERIES[0], "last": SERIES[1]}
    DEC_ARMS = [a for a in cyc_traj["arm"].unique(maintain_order=True).to_list()]

    fig_decline = plt.figure(figsize=(W_TEXT, 6.6))
    _gd = fig_decline.add_gridspec(3, 3, hspace=0.62, wspace=0.34, left=0.095,
                                   right=0.975, top=0.945, bottom=0.075)

    for _i, _arm in enumerate(DEC_ARMS):
        _ax = fig_decline.add_subplot(_gd[_i // 3, _i % 3])
        _s = cyc_traj.filter(pl.col("arm") == _arm)
        _run = _s["run"][0]
        _mpf = DEC_MPF.get(_run, 1.0)
        for _w in ("first", "last"):
            _d = _s.filter(pl.col("which") == _w).sort("pos")
            _x = _d["pos"].to_numpy() * _mpf
            _m = _d["m"].to_numpy()
            _e = 1.96 * _d["se"].to_numpy()
            _ax.fill_between(_x, _m - _e, _m + _e, color=DEC_C[_w], alpha=0.20, lw=0)
            _ax.plot(_x, _m, color=DEC_C[_w], lw=1.8,
                     label=f"{_w} (n={int(_d['n'].median())})")
            _ax.plot(_x, _d["ref"].to_numpy(), color=DEC_C[_w], lw=0.9, ls="--",
                     alpha=0.75)
        _lo = min(float(_s["m"].min() - 1.96 * _s["se"].max()),
                  float(_s["ref"].min()))
        _hi = max(float(_s["m"].max() + 1.96 * _s["se"].max()),
                  float(_s["ref"].max()))
        _pad = (_hi - _lo) * 0.06
        # headroom at the top for the two-line key, so no curve reaches the title
        _ax.set_ylim(_lo - _pad, _hi + _pad + (_hi - _lo) * 0.34)
        _ax.set_title(_arm, loc="left", fontweight="bold", fontsize=7.5)
        _ax.set_xlabel("minutes into the repeat", fontsize=7)
        if _i % 3 == 0:
            _ax.set_ylabel("CNR")
        _ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2,
                   labelspacing=0.2, borderaxespad=0.2)
        _ax.yaxis.grid(True, color=GRID, lw=0.6)
        _ax.set_axisbelow(True)
        _ax.margins(x=0.02)

    # the light it took, first repeat against last, on the same paired cells
    # seven comparisons leave the last row short; the dose panel takes the two
    # free columns rather than leaving a hole beside it
    _axd = fig_decline.add_subplot(_gd[2, 1:])
    _y = np.arange(len(DEC_ARMS))[::-1]
    for _yi, _arm in zip(_y, DEC_ARMS):
        _s = cyc_dose.filter(pl.col("arm") == _arm)
        _a = _s.filter(pl.col("which") == "first").select("cell", "d")
        _b = _s.filter(pl.col("which") == "last").select("cell", "d")
        _j = _a.join(_b, on="cell", suffix="_l")
        _f, _l = float(_j["d"].median()), float(_j["d_l"].median())
        _axd.plot([_f, _l], [_yi, _yi], color=MUTED, lw=1.0, zorder=2)
        _axd.plot([_f], [_yi], "o", ms=5, color=DEC_C["first"], zorder=4)
        _axd.plot([_l], [_yi], "o", ms=5, color=DEC_C["last"], zorder=4)
        _axd.text(max(_f, _l) + 4, _yi, f"{_l / max(_f, 1e-9):.1f}\u00d7",
                  va="center", fontsize=5.8, color=INK)
    # the panel starts in the middle column, so long tick labels overflow into
    # the plot beside it
    # built from the arm names rather than a fixed table, so a change of
    # matched block does not silently fall through to the long label
    def _dec_short(a):
        run, _, rest = a.partition(" \u00b7 ")
        t = rest.replace("MPC, masked", "").replace("MPC", "").split()
        return (run + " " + " ".join(t[-1:])).strip()

    _axd.set_yticks(_y, [_dec_short(a) for a in DEC_ARMS], fontsize=6.5)
    _axd.tick_params(axis="y", length=0)
    _axd.set_ylim(-0.7, len(DEC_ARMS) - 0.3)
    _axd.set_xlim(0, 132)
    _axd.set_xlabel("median light per frame (ms)", fontsize=7)
    _axd.set_title("the light it took", loc="left", fontweight="bold", fontsize=7.5)
    _axd.xaxis.grid(True, color=GRID, lw=0.6)
    _axd.set_axisbelow(True)
    _axd.plot([], [], "o", ms=5, color=DEC_C["first"], label="first repeat")
    _axd.plot([], [], "o", ms=5, color=DEC_C["last"], label="last repeat")
    _axd.legend(frameon=False, fontsize=6, loc="upper right", handlelength=0.8,
                labelspacing=0.2, borderaxespad=0.3, ncol=2, columnspacing=0.9)

    save_fig(fig_decline, "sensitivity-decline")
    fig_decline
    return PERIOD_COLOUR, cyc_dose


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## bandwidth — how well is each objective met, and does that depend on how fast it is asked for?

    Four arms ran the same waveform at four periods. That is a gain-and-phase measurement
    of the whole loop — cells, model and controller together — and the two quantities it
    yields say different things.

    **Amplitude in the holds** is what the objective literally asks for: the level during
    the high hold against the level during the low hold. It is the honest score, and it
    collapses at short periods.

    **Gain of the fundamental** is how much the cell was modulated at all, regardless of
    *when*. It is insensitive to lag, and it does not collapse nearly as fast. The gap
    between the two is the part of the failure that is timing rather than strength.

    Both are computed from each cell's own average over its cycles, not from single
    cycles: one cycle of one cell carries a modulation of the same order as its noise.
    """)
    return


@app.cell(hide_code=True)
def _(OSC_PERIODS, np, osc_arr, osc_feat, pl):
    def osc_bode(period):
        """Per-cell gain and lag from the first harmonic of the folded cycle.

        A cross-correlation argmax is quantised to whole minutes and is pulled about by
        the waveform's asymmetry (a 2 min rise against a 10 min fall). The first Fourier
        coefficient is neither, and gain and phase are defined by it.

        Positive lag means the response trails its demand. Negative means it arrives
        early, which the controller can do: the MPC plans thirty frames ahead and starts
        driving before the step it can see coming.
        """
        m = np.flatnonzero((osc_feat["period"] == period).to_numpy())
        cell = np.array(osc_feat["cell"])[m]
        Y = np.stack([osc_arr["raw_cnr"][i] for i in m]).astype(np.float64)
        R = np.stack([osc_arr["r_t"][i] for i in m]).astype(np.float64)
        w = np.exp(-2j * np.pi * np.arange(period) / period)
        rows = []
        for c in np.unique(cell):
            k = cell == c
            zy, zr = (Y[k].mean(0) * w).sum(), (R[k].mean(0) * w).sum()
            dphi = (np.angle(zy) - np.angle(zr) + np.pi) % (2 * np.pi) - np.pi
            rows.append({"period": period, "cell": c,
                         "gain": float(abs(zy) / abs(zr)),
                         "lag_min": float(-dphi / (2 * np.pi) * period),
                         "lag_deg": float(-np.degrees(dphi))})
        return pl.DataFrame(rows)


    osc_bode_df = pl.concat([osc_bode(p) for p in OSC_PERIODS])

    # Median folded profile per arm, resampled onto a common phase axis so the four
    # periods can be drawn on one set of axes.
    OSC_PHASE = np.linspace(0, 1, 120, endpoint=False)
    osc_folded = {}
    for _P in OSC_PERIODS:
        _m = np.flatnonzero((osc_feat["period"] == _P).to_numpy())
        _cell = np.array(osc_feat["cell"])[_m]
        _Y = np.stack([osc_arr["raw_cnr"][i] for i in _m]).astype(np.float64)
        _R = np.stack([osc_arr["r_t"][i] for i in _m]).astype(np.float64)
        _bycell = np.stack([_Y[_cell == c].mean(0) for c in np.unique(_cell)])
        _t = np.arange(_P) / _P
        osc_folded[_P] = {
            "cnr": np.interp(OSC_PHASE, _t, np.median(_bycell, axis=0), period=1),
            "lo": np.interp(OSC_PHASE, _t, np.percentile(_bycell, 25, axis=0), period=1),
            "hi": np.interp(OSC_PHASE, _t, np.percentile(_bycell, 75, axis=0), period=1),
            "ref": np.interp(OSC_PHASE, _t, _R.mean(axis=0), period=1),
            "n": len(_bycell),
        }

    osc_bode_summary = osc_bode_df.group_by("period").agg(
        pl.len().alias("cells"), pl.col("gain").median().round(3),
        pl.col("lag_min").median().round(2), pl.col("lag_deg").median().round(1)).sort("period")
    return


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    OSC_PERIODS,
    PERIOD_COLOUR,
    W_TEXT,
    np,
    osc_drift,
    pl,
    plt,
    save_fig,
):
    # --- Effort rises; the tracking does not improve ------------------------------
    # What survives of the bandwidth figure. Each cell is compared with ITSELF: the
    # first half of its own cycles against the second half, so no cell is scored
    # against the population and the drifting baseline cannot manufacture the effect.
    #
    # Measured in ABSOLUTE units, not as a percentage of each cell's early value. The
    # percentage version is what made this unreadable: cells that started near zero
    # dose give ratios in the thousands, so a p90 of +7343% sat next to a median of
    # +20% and the axis had to be cut somewhere arbitrary.
    #
    # The claim is the DECOUPLING -- more light bought, no better tracking. The size
    # of the dose rise is not ordered by period and should not be read as if it were;
    # it tracks where each arm happened to start.
    from scipy.stats import wilcoxon

    _bw = osc_drift.with_columns(
        (pl.col("dose_late") - pl.col("dose_early")).alias("d_dose"),
        (pl.col("rmse_late") - pl.col("rmse_early")).alias("d_err"))

    _BW_RNG = np.random.default_rng(4)


    def _bw_panel(ax, col, unit, title, lo, hi, worse_up):
        """One paired-change distribution per period, against a null of no change."""
        _n_out = 0
        for _i, _P in enumerate(OSC_PERIODS):
            _v = _bw.filter(pl.col("period") == _P)[col].to_numpy()
            _v = _v[np.isfinite(_v)]
            _q1, _md, _q3 = np.percentile(_v, [25, 50, 75])
            # Clipping piles the tails into a bar at the axis limit that reads as a
            # mode. Drop them from the strip instead and say how many; the median and
            # the quartiles below are computed on the FULL sample either way.
            _in = _v[(_v >= lo) & (_v <= hi)]
            _n_out += len(_v) - len(_in)
            _x = _i + _BW_RNG.uniform(-0.17, 0.17, len(_in))
            ax.plot(_x, _in, "o", ms=2.4, alpha=0.35,
                    color=PERIOD_COLOUR[_P], mec="none", ls="none")
            ax.plot([_i - 0.30, _i + 0.30], [_md, _md], color=INK, lw=2.2, zorder=6)
            ax.plot([_i, _i], [_q1, _q3], color=INK, lw=0.9, zorder=5)
            _p = wilcoxon(_v).pvalue
            _up = 100 * float((_v > 0).mean())
            ax.text(_i, hi, f"{_up:.0f}%\n{'p<0.001' if _p < 1e-3 else f'p={_p:.2f}'}",
                    ha="center", va="top", fontsize=5.8, color=MUTED, linespacing=1.4)

        ax.axhline(0, color=INK, lw=1.0, zorder=4)
        ax.set_xticks(range(len(OSC_PERIODS)), [str(p) for p in OSC_PERIODS],
                      fontsize=7)
        ax.set_xlim(-0.55, len(OSC_PERIODS) - 0.45)
        ax.set_ylim(lo, hi * 1.30)
        ax.set_xlabel("period (min)")
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _note = worse_up + (f"   ({_n_out} cells outside the axis)" if _n_out else "")
        ax.text(0.5, 0.015, _note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=5.8, color=MUTED)


    fig_bode = plt.figure(figsize=(W_TEXT, 3.1))
    _gb = fig_bode.add_gridspec(1, 2, wspace=0.30, left=0.105, right=0.975,
                                top=0.87, bottom=0.185)
    _bw_panel(fig_bode.add_subplot(_gb[0, 0]), "d_dose",
              "change in light spent (ms/frame)",
              "a  It spends more", -60, 70,
              "% = cells spending more")
    _bw_panel(fig_bode.add_subplot(_gb[0, 1]), "d_err",
              "change in tracking error (CNR)",
              "b  It tracks no better", -0.10, 0.10,
              "% = cells tracking worse")
    save_fig(fig_bode, "bandwidth")
    fig_bode
    return (wilcoxon,)


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
    cyc_dose,
    np,
    pl,
    plt,
    save_fig,
    wilcoxon,
):
    # --- The same paired contrast, across every run with a repeating objective ----
    # `bandwidth` asks this question inside v19 alone, where the arms differ only by
    # period. This asks it of the whole admissible campaign: seven arms across four
    # runs, each cell still compared with ITSELF -- the first repeat of its own
    # demand against the last.
    #
    # The pairing and the demand matching are `cyc_dose`, built for
    # `sensitivity-decline`: a run-up's repeats are matched on the same demand LEVEL,
    # so the extra light a later block needs is the design and is not counted as a
    # decline. Absolute units, for the reason the v19 version gives -- cells starting
    # near zero dose give percentage changes in the thousands.
    #
    # v24 does not appear. Its demand sat above reach in eight of nine blocks, so it
    # has no pair of repeats at a common reachable level to compare.
    BWR_COL = {"v10": SERIES[0], "v19": SERIES[1], "v21": SERIES[2], "v23": "#8452a1"}
    BWR_LABEL = {"v10 · MPC": "v10\nMPC", "v19 · 20 min": "v19\n20 min",
                 "v19 · 30 min": "v19\n30 min", "v19 · 60 min": "v19\n60 min",
                 "v19 · 70 min": "v19\n70 min", "v21 · MPC, masked": "v21\nmasked",
                 "v23 · demand A": "v23\ndemand A"}

    bwr_cells = (cyc_dose.pivot(on="which", index=["run", "arm", "cell"], values=["d", "err"])
                 .drop_nulls()
                 .with_columns((pl.col("d_last") - pl.col("d_first")).alias("d_dose"),
                               (pl.col("err_last") - pl.col("err_first")).alias("d_err")))
    BWR_ARMS = bwr_cells.select("run", "arm").unique().sort("run", "arm")["arm"].to_list()

    _BWR_RNG = np.random.default_rng(4)


    def _bwr_panel(ax, col, unit, title, lo, hi, worse_up):
        """One paired-change distribution per arm, against a null of no change."""
        _n_out = 0
        for _i, _a in enumerate(BWR_ARMS):
            _s = bwr_cells.filter(pl.col("arm") == _a)
            _v = _s[col].to_numpy()
            _v = _v[np.isfinite(_v)]
            _q1, _md, _q3 = np.percentile(_v, [25, 50, 75])
            _in = _v[(_v >= lo) & (_v <= hi)]
            _n_out += len(_v) - len(_in)
            _x = _i + _BWR_RNG.uniform(-0.17, 0.17, len(_in))
            ax.plot(_x, _in, "o", ms=2.2, alpha=0.30, color=BWR_COL[_s["run"][0]],
                    mec="none", ls="none")
            ax.plot([_i - 0.30, _i + 0.30], [_md, _md], color=INK, lw=2.2, zorder=6)
            ax.plot([_i, _i], [_q1, _q3], color=INK, lw=0.9, zorder=5)
            _p = wilcoxon(_v).pvalue
            _up = 100 * float((_v > 0).mean())
            _ptxt = ("p<0.001" if _p < 1e-3 else
                     f"p={_p:.3f}" if _p < 0.01 else f"p={_p:.2f}")
            ax.text(_i, hi, f"{_up:.0f}%\n{_ptxt}",
                    ha="center", va="top", fontsize=5.4, color=MUTED, linespacing=1.4)

        ax.axhline(0, color=INK, lw=1.0, zorder=4)
        ax.set_xticks(range(len(BWR_ARMS)), [BWR_LABEL[a] for a in BWR_ARMS], fontsize=5.8)
        ax.set_xlim(-0.55, len(BWR_ARMS) - 0.45)
        ax.set_ylim(lo, hi * 1.30)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _note = worse_up + (f"   ({_n_out} cells outside the axis)" if _n_out else "")
        ax.text(0.5, 0.015, _note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.8, color=MUTED)


    fig_bwr = plt.figure(figsize=(W_TEXT, 5.4))
    _gbr = fig_bwr.add_gridspec(2, 1, hspace=0.42, left=0.115, right=0.975,
                                top=0.93, bottom=0.085)
    _bwr_panel(fig_bwr.add_subplot(_gbr[0]), "d_dose",
               "change in light spent (ms/frame)",
               "a  It spends more, in every run", -80, 130,
               "% = cells spending more")
    _bwr_panel(fig_bwr.add_subplot(_gbr[1]), "d_err",
               "change in tracking error (CNR)",
               "b  It tracks no better", -0.20, 0.20,
               "% = cells tracking worse")
    save_fig(fig_bwr, "bandwidth-runs")
    fig_bwr

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feedback ladder
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, pl, plt, save_fig):
    # --- E2 / v24: is it the control, or is it the light? ------------------------
    # Every tracking number before this run came from an arm that had feedback, so
    # none of them could separate "closed-loop single-cell control works" from
    # "shining light on cells raises CNR". Three arms on one plate, one reference:
    # per-cell closed loop, one constant dose at the closed arms' expected mean, and
    # zero stimulation light.
    #
    # SCOPE THE CLAIM TO LEVEL, NOT TRAJECTORY. The demand was set from a plate
    # estimate 0.11 too high and never re-anchored, so eight of nine blocks sat above
    # reach and no arm tracked a waveform. What this figure supports is that the loop
    # holds a reachable LEVEL better than constant light and better than nothing.
    _e2_b01 = pl.read_parquet(materials_path("e2_block01.parquet"))
    _e2_r01 = dict(pl.read_parquet(materials_path("e2_block01_rmse.parquet"))
                     .select("arm", "rmse").iter_rows())
    _e2_fld = pl.read_parquet(materials_path("e2_fields.parquet"))
    _e2_ind = pl.read_parquet(materials_path("e2_individuation.parquet"))
    E2_NAME = {1: "closed loop", 2: "constant 60 ms", 3: "dark"}
    E2_COL = {1: SERIES[0], 2: SERIES[1], 3: MUTED}


    def _panel_e2_block(ax, ax_light):
        """(a) The one demand block the loop reaches, and the light it took.

        The constant arm's population MEDIAN sits marginally closer (-0.024 against
        -0.034). Its per-cell error is worse. That gap is this thesis's own argument
        appearing inside one block: the average is closer while the cells are further
        off, and the average is not what is being controlled.
        """
        r = _e2_b01.filter(pl.col("arm") == 1)
        ax.plot(r["m"], r["r_t"], color=INK, lw=1.6, ls="--", label="demand 1.20",
                zorder=6)
        for a in (1, 2, 3):
            s = _e2_b01.filter(pl.col("arm") == a)
            if a == 1:
                ax.fill_between(s["m"], s["lo"], s["hi"], color=E2_COL[a], alpha=0.15, lw=0)
            ax.plot(s["m"], s["cnr"], color=E2_COL[a], lw=1.7, label=E2_NAME[a])
            ax_light.plot(s["m"], s["dose"], color=E2_COL[a], lw=1.3)
        ax.set_ylabel("CNR  (median, IQR)")
        ax.set_title("a  The loop holding a reachable demand", loc="left",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper left",
                  handlelength=1.4, columnspacing=1.1, borderaxespad=0.2)
        ax.set_ylim(0.95, 1.38)
        ax.tick_params(labelbottom=False)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax_light.text(0.985, 0.88,
                f"per-cell error {_e2_r01[1]:.3f} against {_e2_r01[2]:.3f} — "
                "closer per cell, on less light",
                transform=ax_light.transAxes, ha="right", va="top",
                fontsize=6.5, color=INK)
        ax_light.set_ylabel("light\n(ms/frame)", fontsize=7)
        ax_light.set_xlabel("minutes into the block")
        ax_light.set_ylim(0, 90)
        ax_light.tick_params(labelsize=7)


    def _panel_e2_ladder(ax):
        """(b) The ladder at the level inference is done: the field."""
        for a in (3, 2, 1):
            s = _e2_fld.filter(pl.col("arm") == a)
            y = 4 - a
            ax.plot(s["rmse"], [y] * s.height, "o", ms=5.5, color=E2_COL[a])
            ax.plot([s["rmse"].median()] * 2, [y - 0.22, y + 0.22], color=E2_COL[a], lw=2)
        _short = {1: "closed loop", 2: "constant", 3: "dark"}
        ax.set_yticks([3, 2, 1],
                      [f"{_short[a]}\n{_e2_fld.filter(pl.col('arm') == a)['dose'].mean():.0f} ms"
                       for a in (1, 2, 3)], fontsize=6.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_xlabel("tracking error (CNR), field median")
        ax.set_title("b  The ladder, by field", loc="left", fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.97, 0.72, "ρ = +0.93,  exact p = 0.0048\nover 420 relabellings",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color=INK, linespacing=1.5)


    def _panel_e2_individuation(ax):
        """(c) One dose for the field, or a dose per cell — paired inside the field."""
        for fov in sorted(set(_e2_ind["fov"].to_list())):
            s = _e2_ind.filter(pl.col("fov") == fov).sort("broadcast")
            a, b = float(s["rmse"][0]), float(s["rmse"][1])
            disc = b < a                       # the one field that goes the other way
            ax.plot([0, 1], [a, b], color=MUTED if disc else SERIES[0],
                    lw=1.5, ls=":" if disc else "-", zorder=2)
            ax.plot([0, 1], [a, b], "o", ms=5, mfc="white", mew=1.4,
                    color=MUTED if disc else SERIES[0], zorder=3)
        ax.set_xticks([0, 1], ["per-cell\ndose", "one broadcast\ndose"], fontsize=7)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel("tracking error (CNR)")
        ax.set_title("c  Individuation", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _lo, _hi = ax.get_ylim()
        ax.set_ylim(_lo, _hi + (_hi - _lo) * 0.45)
        ax.text(0.5, 0.97, "0.295 against 0.325, on 113 against 123 ms\n"
                           "3 of 4 fields — sign test p = 0.375, directional",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
                color=INK, linespacing=1.5)


    fig_e2 = plt.figure(figsize=(W_TEXT, 5.4))
    _ge2 = fig_e2.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.44,
                               left=0.115, right=0.975, top=0.93, bottom=0.095)
    _te2 = _ge2[0].subgridspec(2, 1, height_ratios=[1.0, 0.30], hspace=0.10)
    _be2 = _ge2[1].subgridspec(1, 2, wspace=0.42, width_ratios=[1.25, 1.0])
    _axe = fig_e2.add_subplot(_te2[0])
    _panel_e2_block(_axe, fig_e2.add_subplot(_te2[1], sharex=_axe))
    _panel_e2_ladder(fig_e2.add_subplot(_be2[0]))
    _panel_e2_individuation(fig_e2.add_subplot(_be2[1]))
    save_fig(fig_e2, "feedback-ladder")
    fig_e2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## rig-calibration — is the model calibrated for the experiment it is actually driving?

    The training set was recorded under **whole-field illumination**: every cell in the
    frame saw the same light. The live run illuminates **one nucleus at a time**. The same
    commanded exposure therefore does not have to mean the same delivered activation, and
    if it does not, the model will systematically expect the wrong thing from its own
    action.

    Two failures are worth separating, because they need different fixes.

    * A **cold start** — the encoder has almost no history for a cell that has just been
      picked up, so its first forecasts are made from nothing. That would fix itself, and
      the only question is how many frames it takes.
    * A **standing offset that depends on the dose** — the model expecting more response
      per millisecond than it gets. No amount of context removes that; it needs
      calibration.

    The measurement is the one-step forecast the server already logs. `pred_cnr_h1` at
    frame *t* is compared against that cell's own `raw_cnr` at *t+1*, on contiguous frames
    only. Persistence — "the cell stays where it is" — is carried alongside, because an
    error in CNR units means nothing without it, and because it uses the same noisy
    measurement: if a newly tracked cell were merely noisy, persistence would suffer too.
    """)
    return


@app.cell(hide_code=True)
def _(osc_df, pl):
    # One-step forecast error, on contiguous frames of the same cell. `dark` frames carry
    # no prediction (the server skips the model when it overrides to dark), so they drop
    # out with the null check.
    CAL_CTX_BINS = [(1, 3), (3, 6), (6, 12), (12, 25), (25, 60), (60, 150), (150, 10_000)]
    CAL_DEEP = 150            # "the encoder has all the history it is going to get"

    osc_h1 = (
        osc_df.sort(["fov", "particle", "control_frame"])
        .with_columns(
            pl.col("raw_cnr").shift(-1).over(["fov", "particle"]).alias("cnr_next"),
            pl.col("control_frame").shift(-1).over(["fov", "particle"]).alias("cf_next"))
        .filter((pl.col("cf_next") == pl.col("control_frame") + 1)
                & pl.col("pred_cnr_h1").is_not_null())
        .with_columns(
            (pl.col("cnr_next") - pl.col("pred_cnr_h1")).alias("err"),
            (pl.col("cnr_next") - pl.col("raw_cnr")).alias("err_persist"),
            (pl.col("control_frame") // 60).alias("hour"),
            (pl.col("exposure_ms") > 0).alias("lit"))
    )

    cal_by_ctx = pl.DataFrame([
        {"lo": lo, "hi": hi,
         "n": _k.height,
         "mid": float(_k["n_frames_seen"].median()),
         "mae": float(_k["err"].abs().mean()),
         "mae_persist": float(_k["err_persist"].abs().mean()),
         "bias": float(_k["err"].mean()),
         "bias_lit": float(_k.filter(pl.col("lit"))["err"].mean()),
         "bias_dark": float(_k.filter(~pl.col("lit"))["err"].mean())}
        for lo, hi in CAL_CTX_BINS
        if (_k := osc_h1.filter((pl.col("n_frames_seen") >= lo)
                                & (pl.col("n_frames_seen") < hi))).height > 200
    ])

    # The standing offset, measured only where context cannot be the explanation.
    cal_by_dose = (
        osc_h1.filter(pl.col("n_frames_seen") >= CAL_DEEP)
        .group_by("exposure_ms")
        .agg(pl.len().alias("n"), pl.col("err").mean().alias("bias"),
             (pl.col("err").std() / pl.len().sqrt()).alias("se"))
        .sort("exposure_ms")
    )

    # Control: does the same context effect appear at every hour of the run? If the cold
    # start were really the cells drifting, it would not.
    cal_by_hour = (
        osc_h1.with_columns(
            pl.when(pl.col("n_frames_seen") < 12).then(pl.lit("< 12 frames"))
            .otherwise(pl.lit("60+ frames")).alias("ctx_band"))
        .filter((pl.col("n_frames_seen") < 12) | (pl.col("n_frames_seen") >= 60))
        .group_by(["hour", "ctx_band"])
        .agg(pl.len().alias("n"), pl.col("err").mean().alias("bias"),
             pl.col("err").abs().mean().alias("mae"))
        .sort(["ctx_band", "hour"])
    )
    return CAL_DEEP, cal_by_ctx, cal_by_dose, osc_h1


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
    materials_path,
    model_rmse,
    np,
    pl,
    plt,
    save_fig,
):
    # --- Is the model calibrated for the rig it is actually driving? --------------
    # Trained on whole-field illumination, deployed on a rig that lights one nucleus
    # at a time. Two questions, one panel each: does the ACCURACY transfer, and does
    # the stated UNCERTAINTY transfer.
    #
    # Neither can be read off the serving log: it records a single one-step point
    # prediction and no spread at all. Both panels come from replaying the trained
    # model over the live traces, at every horizon it was trained for.
    #
    # The dose-gap panel is gone. It asked whether error grows as a cell's own dose
    # departs from its neighbours' -- the axis along which per-nucleus light differs
    # from whole-field -- and found nothing (|r| <= 0.07 in all four runs). A flat
    # line is worth a sentence, not a third of the figure.
    rc_rep = pl.read_parquet(materials_path("rigcal_replay.parquet"))
    rc_off_uq = pl.read_parquet(materials_path("offline_uq.parquet"))
    rc_live_uq = pl.read_parquet(materials_path("live_uq.parquet"))

    RC_H = np.arange(1, len(model_rmse) + 1)
    RC_RUNS = sorted(set(rc_rep["run"]))
    RC_C = [SERIES[0], SERIES[1], SERIES[2], "#8452a1"]

    _rc = (rc_rep.with_columns(((pl.col("pred") - pl.col("truth")) ** 2).alias("se"),
                               ((pl.col("persist") - pl.col("truth")) ** 2).alias("sp"))
                 .group_by(["run", "h"])
                 .agg(pl.col("se").mean().sqrt().alias("rmse"),
                      pl.col("sp").mean().sqrt().alias("rmse_p"))
                 .sort(["run", "h"]))

    fig_cal = plt.figure(figsize=(W_TEXT, 2.9))
    _gcal = fig_cal.add_gridspec(1, 2, wspace=0.28, left=0.095, right=0.975,
                                 top=0.88, bottom=0.20)

    # (a) accuracy: the model transfers, the baseline does not
    _ax = fig_cal.add_subplot(_gcal[0, 0])
    # Persistence is not drawn. It was the whole point of the earlier version of
    # this panel -- live persistence error is far lower than offline, which is why
    # the model's ADVANTAGE over doing nothing shrinks on the rig even though its own
    # error barely moves. But it doubles the number of lines, and the claim this
    # panel makes is about the model's error alone.
    _ax.plot(RC_H, model_rmse, color=INK, lw=2.6, zorder=6, label="offline held-out")
    for _i, _r in enumerate(RC_RUNS):
        _s = _rc.filter(pl.col("run") == _r).sort("h")
        _ax.plot(_s["h"], _s["rmse"], color=RC_C[_i], lw=1.6, label=_r)
    _ax.set_xlabel("forecast horizon (min)")
    _ax.set_ylabel("RMSE (CNR)")
    _ax.set_ylim(0, None)
    _ax.set_xticks([1, 8, 15, 22, 30])
    _ax.set_title("a  Accuracy transfers", loc="left", fontweight="bold")
    _ax.legend(frameon=False, fontsize=6.2, loc="lower right", ncol=2,
               handlelength=1.4, columnspacing=0.9, labelspacing=0.25,
               borderaxespad=0.4)
    _ax.yaxis.grid(True, color=GRID, lw=0.6)
    _ax.set_axisbelow(True)

    # (b) uncertainty: does the stated spread still mean what it claims?
    # Coverage of the central interval, from the EXACT mixture CDF rather than a
    # gaussian stand-in: PIT = sum_k pi_k Phi((y - mu_k)/sigma_k), and the interval
    # at level L covers when |PIT - 0.5| <= L/2.
    _axu = fig_cal.add_subplot(_gcal[0, 1])
    _lu = (rc_live_uq.with_columns(
               ((pl.col("pit") - 0.5).abs() <= 0.34).alias("c68"),
               ((pl.col("pit") - 0.5).abs() <= 0.475).alias("c95"))
           .group_by("h").agg(pl.col("c68").mean(), pl.col("c95").mean())
           .sort("h"))
    for _lvl, _col, _nom in ((("cov95", "c95"), SERIES[0], 0.95),
                             (("cov68", "c68"), SERIES[1], 0.68)):
        _o, _l = _lvl
        _axu.axhline(_nom, color=MUTED, lw=0.8, ls=":")
        _axu.plot(rc_off_uq["h"], rc_off_uq[_o], color=_col, lw=2.2,
                  label=f"{_nom:.0%} interval, offline")
        _axu.plot(_lu["h"], _lu[_l], color=_col, lw=1.6, ls="--",
                  label=f"{_nom:.0%} interval, live")
    _axu.set_xlabel("forecast horizon (min)")
    _axu.set_ylabel("share of outcomes inside")
    _axu.set_ylim(0.5, 1.02)
    _axu.set_xticks([1, 8, 15, 22, 30])
    _axu.set_title("b  So does its uncertainty", loc="left", fontweight="bold")
    _axu.legend(frameon=False, fontsize=5.6, loc="lower left", ncol=2,
                handlelength=1.6, columnspacing=0.9, labelspacing=0.25,
                borderaxespad=0.3)
    _axu.text(0.97, 0.97, "dotted: nominal", transform=_axu.transAxes, ha="right",
              va="top", fontsize=6, color=MUTED)
    _axu.yaxis.grid(True, color=GRID, lw=0.6)
    _axu.set_axisbelow(True)

    save_fig(fig_cal, "rig-calibration")
    fig_cal
    return


@app.cell(hide_code=True)
def _(CAL_DEEP, OSC_SWING, cal_by_ctx, cal_by_dose, mo, osc_h1, pl):
    _deep = osc_h1.filter(pl.col("n_frames_seen") >= CAL_DEEP)
    _cold = osc_h1.filter(pl.col("n_frames_seen") < 6)
    _top = _deep.filter(pl.col("exposure_ms") >= 150)
    _dark_bias = float(cal_by_dose.filter(pl.col("exposure_ms") == 0)["bias"][0])

    mo.md(f"""
    ### What this says about calibrating the model for the rig

    **The cold start is real and it is the model's, not the measurement's.** With one or
    two frames of history the forecast is
    {float(cal_by_ctx["mae"][0]) / float(cal_by_ctx["mae_persist"][0]):.1f}× worse than
    assuming the cell does not move — persistence reads the same noisy CNR and does fine,
    so this is the encoder having nothing to encode. It clears quickly: the model is ahead
    of persistence by six frames and at its asymptote by roughly fifteen. Over
    {_cold.height:,} frames of the run — {100 * _cold.height / osc_h1.height:.1f} % — the
    controller was acting on a forecast worse than doing nothing. The cheap fix is on the
    serving side, not in the weights: hold a cell dark, or fall back to persistence, until
    it has been seen half a dozen times.

    **The standing offset is real, and context does not touch it.** With
    {CAL_DEEP}+ frames of history the forecast is unbiased in the dark
    ({_dark_bias:+.4f} CNR) and biased by
    {float(_top["err"].mean()):+.4f} CNR per step at the top rung, on
    {_top.height:,} frames. The sign says the model expects more response per millisecond
    than the cells deliver, and the size scales with the dose — which is the shape the
    training-versus-serving mismatch predicts, whole-field illumination against a mask on
    one nucleus.

    Per step that is small against a {OSC_SWING:.2f} CNR demanded swing. It is not small over a
    rise: the loop carries three to five minutes of dead time and a rise of about four, so
    the controller spends eight or so consecutive steps expecting a response it will not
    get, and closes the gap by asking for more light — which is what
    the drift panels show it doing.

    **So: no retraining is implied, but a one-parameter correction is.** A gain on the
    dose channel fitted on this run would remove the offset without touching anything the
    model learned about the cells.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## response-modality — are there two kinds of cell, or one very broad kind?

    The response amplitudes are spread enormously — in the fastest arm the middle half of
    cells covers 0.007 to 0.219 CNR. A spread that wide invites the reading that there are
    two populations, entrained and not, and that would be a much more interesting claim
    than a wide continuum. It is testable rather than a matter of taste.

    The unit here is the **cell**, not the window: a bifurcation would be a property of
    cells, and window-level structure is dominated by the fact that windows from one cell
    resemble each other. Every cell's cycles are averaged into one profile first — in a
    single cycle of a single cell the modulation is about the size of the noise.

    Three quantities, each of which a split could show up in: the **amplitude** the cell
    achieved, how well its averaged cycle **follows the shape it was asked for**, and the
    **pattern alone** with level and amplitude both divided out.

    The test is Silverman's critical bandwidth. Smooth the values into a curve and keep
    smoothing until exactly one bump is left; then ask whether that much smoothing is
    unusual, by drawing imitation samples from the one-bump version and counting how often
    they need more. Panel (c) is the part that decides how much the answer is worth: the
    same test run on made-up data whose answer is known.
    """)
    return


@app.cell(hide_code=True)
def _(np, osc_arr, osc_feat, pl):
    def kde_modes(x, h, grid=256):
        """Number of bumps in a Gaussian kernel density estimate of bandwidth h."""
        t = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid)
        d = np.exp(-0.5 * ((t[:, None] - x[None, :]) / h) ** 2).sum(1)
        return int(((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])).sum())


    def critical_bw(x, k=1, iters=40):
        """The least smoothing that leaves at most k bumps."""
        lo, hi = 1e-4, float(x.max() - x.min())
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kde_modes(x, mid) <= k:
                hi = mid
            else:
                lo = mid
        return hi


    def silverman(x, k=1, n_boot=600, seed=0):
        """Chance of needing this much smoothing if there really were only k bumps.

        Resamples are drawn from the data smoothed at its own critical bandwidth and
        rescaled to the original variance, which is what stops the extra smoothing from
        making the imitation samples trivially easier to flatten.
        """
        x = np.asarray(x, float)
        n = len(x)
        rng = np.random.default_rng(seed)
        h = critical_bw(x, k)
        sd = x.std(ddof=1)
        hits = 0
        for _ in range(n_boot):
            y = x[rng.integers(0, n, n)] + h * rng.standard_normal(n)
            y = x.mean() + (y - y.mean()) / np.sqrt(1 + h ** 2 / sd ** 2)
            hits += critical_bw(y, k) > h
        return h, hits / n_boot


    def osc_cell_profiles(period):
        """One averaged cycle per cell, plus the scalars the split is tested on."""
        m = np.flatnonzero((osc_feat["period"] == period).to_numpy())
        cell = np.array(osc_feat["cell"])[m]
        Y = np.stack([osc_arr["raw_cnr"][i] for i in m]).astype(float)
        R = np.stack([osc_arr["r_t"][i] for i in m]).astype(float)
        D = np.stack([osc_arr["exposure_ms"][i] for i in m]).astype(float)
        uc = np.unique(cell)
        prof = np.stack([Y[cell == c].mean(0) for c in uc])
        ref = np.stack([R[cell == c].mean(0) for c in uc])
        dose = np.stack([D[cell == c].mean(0) for c in uc])
        lo, hi = ref[0] <= ref[0].min() + 1e-6, ref[0] >= ref[0].max() - 1e-6
        pc = prof - prof.mean(1, keepdims=True)
        rc = ref - ref.mean(1, keepdims=True)
        return {
            "cells": uc, "prof": prof, "ref": ref, "dose": dose, "lo": lo, "hi": hi,
            "amp": np.median(prof[:, hi], 1) - np.median(prof[:, lo], 1),
            "level_low": np.median(prof[:, lo], 1),
            "entrain": (pc * rc).sum(1) / (np.linalg.norm(pc, axis=1)
                                           * np.linalg.norm(rc, axis=1)),
            "dose_mean": dose.mean(1),
        }


    OSC_FLAT = 0.10           # a stated threshold for the tail panels, not a found boundary
    OSC_SHAPE_MIN_AMP = 0.05  # below this, dividing out amplitude just normalises noise

    osc_cellprof = {p: osc_cell_profiles(p) for p in (20, 30, 60, 70)}

    _rows = []
    for _P, _c in osc_cellprof.items():
        for _name, _v in (("amplitude", _c["amp"]), ("entrainment", _c["entrain"])):
            _h, _p = silverman(_v)
            _rows.append({"period": _P, "quantity": _name, "n": len(_v),
                          "median": round(float(np.median(_v)), 3),
                          "p25": round(float(np.percentile(_v, 25)), 3),
                          "p75": round(float(np.percentile(_v, 75)), 3),
                          "p_one_bump": _p})
    osc_modality = pl.DataFrame(_rows).sort(["quantity", "period"])
    return OSC_FLAT, OSC_SHAPE_MIN_AMP, osc_cellprof, osc_modality, silverman


@app.cell(hide_code=True)
def _(OSC_SHAPE_MIN_AMP, np, osc_cellprof, pl, silverman):
    # The same test on the pattern alone, and on made-up data whose answer is known.
    _rows = []
    for _P, _c in osc_cellprof.items():
        _keep = _c["amp"] > OSC_SHAPE_MIN_AMP
        _pc = _c["prof"][_keep] - _c["prof"][_keep].mean(1, keepdims=True)
        _V = _pc / np.linalg.norm(_pc, axis=1, keepdims=True)
        _u, _s, _ = np.linalg.svd(_V - _V.mean(0), full_matrices=False)
        _var = _s ** 2 / (_s ** 2).sum()
        for _k in range(3):
            _, _p = silverman(_u[:, _k] * _s[_k])
            _rows.append({"period": _P, "axis": f"PC{_k + 1}", "n": int(_keep.sum()),
                          "variance": round(float(_var[_k]), 3), "p_one_bump": _p})
    osc_shape_modality = pl.DataFrame(_rows)

    # What the test can and cannot see, at this sample size.
    _rng = np.random.default_rng(0)
    _n = 90
    OSC_POWER = {}
    for _name, _x in (
            ("one bell", _rng.standard_normal(_n)),
            ("one lopsided bell", _rng.lognormal(0, 0.8, _n)),
            ("two, 2 apart", np.r_[_rng.standard_normal(_n // 2),
                                   _rng.standard_normal(_n // 2) + 2]),
            ("two, 3 apart", np.r_[_rng.standard_normal(_n // 2),
                                   _rng.standard_normal(_n // 2) + 3]),
            ("two, 4 apart", np.r_[_rng.standard_normal(_n // 2),
                                   _rng.standard_normal(_n // 2) + 4]),
            ("80/20 split, 3 apart", np.r_[_rng.standard_normal(72),
                                           _rng.standard_normal(18) + 3])):
        OSC_POWER[_name] = (_x, silverman(_x)[1])

    pl.concat([osc_shape_modality,
               pl.DataFrame([{"period": -1, "axis": k, "n": 90, "variance": 0.0,
                              "p_one_bump": v[1]} for k, v in OSC_POWER.items()])])
    return (OSC_POWER,)


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    OSC_FLAT,
    OSC_POWER,
    PERIOD_COLOUR,
    SERIES,
    W_TEXT,
    np,
    osc_cellprof,
    osc_modality,
    pl,
    plt,
    save_fig,
):
    def _kde(x, grid=200, bw=None):
        x = np.asarray(x, float)
        h = bw or 0.9 * min(x.std(ddof=1),
                            (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) * len(x) ** -0.2
        t = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid)
        d = np.exp(-0.5 * ((t[:, None] - x[None, :]) / h) ** 2).sum(1)
        return t, d / (d.max() or 1)


    def _panel_dist(ax, key, xlabel, title):
        """(a, b) One curve per arm, with each cell drawn underneath it."""
        labels = []
        for i, (P, c) in enumerate(osc_cellprof.items()):
            v = c[key]
            t, d = _kde(v)
            ax.plot(t, d + i * 1.25, color=PERIOD_COLOUR[P], lw=1.8)
            ax.fill_between(t, i * 1.25, d + i * 1.25, color=PERIOD_COLOUR[P], alpha=0.18, lw=0)
            ax.plot(v, np.full(len(v), i * 1.25 - 0.12), "|", ms=4, alpha=0.5,
                    color=PERIOD_COLOUR[P])
            p = float(osc_modality.filter((pl.col("period") == P)
                                          & (pl.col("quantity") == ("amplitude" if key == "amp"
                                                                    else "entrainment")))["p_one_bump"][0])
            labels.append((i, f"{P} min   n={len(v)}   one bump: p={p:.2f}"))
        for i, txt in labels:
            ax.text(0.01, (i * 1.25 + 1.02) / (len(osc_cellprof) * 1.25 + 0.4), txt,
                    transform=ax.transAxes, fontsize=6.5, color=INK, ha="left", va="center")
        ax.set_ylim(-0.35, len(osc_cellprof) * 1.25 + 0.05)
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines["left"].set_visible(False)


    def _panel_power(ax):
        """(c) The same test on data whose answer is known, at the same sample size."""
        names = list(OSC_POWER)
        for i, name in enumerate(names):
            x, p = OSC_POWER[name]
            x = (x - x.mean()) / x.std(ddof=1)
            t, d = _kde(x)
            col = SERIES[1] if p < 0.05 else MUTED
            ax.plot(t, d * 0.72 + i, color=col, lw=1.6)
            ax.fill_between(t, i, d * 0.72 + i, color=col, alpha=0.18, lw=0)
            verdict = ("found" if p < 0.05
                       else "MISSED" if ("two" in name or "80/20" in name) else "correct")
            ax.text(-3.15, i + 0.80, f"{name}  →  p={p:.2f}  {verdict}",
                    fontsize=6.5, color=col, ha="left", va="center")
        ax.set_yticks([])
        ax.set_ylim(-1.05, len(names) + 0.55)
        ax.set_xlim(-3.2, 3.4)
        ax.set_xlabel("made-up data, rescaled to a common width")
        ax.set_title("c  What the test can and cannot see (n = 90)",
                     loc="left", fontweight="bold")
        ax.spines["left"].set_visible(False)
        ax.text(-3.15, -0.92,
                "it finds two equal groups three widths apart, and misses a one-in-five\n"
                "minority — so 'one bump' in (a) and (b) is weak evidence, not strong",
                fontsize=6.5, color=INK, va="bottom", ha="left")


    def _panel_tails(ax):
        """(d) The flat cells are two opposite failures, not one group."""
        for P, c in osc_cellprof.items():
            s = ax.scatter(c["level_low"], c["amp"], c=c["dose_mean"], cmap="magma",
                           vmin=0, vmax=150, s=16, alpha=0.85, edgecolors="none")
        _r = osc_cellprof[60]
        ax.axvline(float(_r["ref"][0].min()), color=INK, lw=1.0, ls="--")
        ax.axhline(OSC_FLAT, color=MUTED, lw=0.8, ls=":")
        ax.text(float(_r["ref"][0].min()) - 0.010, 0.46,
                "the level asked for\nin the low phase", fontsize=6.5, color=INK,
                va="top", ha="right")
        cb = ax.figure.colorbar(s, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("mean dose given (ms)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlabel("where the cell actually sat during the low phase (CNR)")
        ax.set_ylabel("amplitude achieved (CNR)")
        ax.set_title("d  Two opposite ways to end up flat", loc="left", fontweight="bold")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.annotate("never comes down, and light\ncannot push it down — so the\ncontroller withholds light",
                    xy=(1.16, 0.02), xytext=(1.02, 0.38), fontsize=6.5, color=INK,
                    ha="left", arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
        ax.annotate("never goes up, on\nnear-maximal light",
                    xy=(0.62, 0.015), xytext=(0.30, 0.20), fontsize=6.5, color=INK,
                    ha="left", arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))


    # Panel (d) removed: its scatter is the same one the control-envelope figure carried, and the control-envelope figure is cut.
    # Three panels now — the two distributions and the power check that qualifies them.
    fig_modes = plt.figure(figsize=(W_TEXT, 5.2))
    _gm = fig_modes.add_gridspec(2, 2, hspace=0.34, wspace=0.30,
                                 left=0.06, right=0.95, top=0.94, bottom=0.08)
    _panel_dist(fig_modes.add_subplot(_gm[0, 0]), "amp", "amplitude achieved (CNR)",
                "a  How much each cell moved")
    _panel_dist(fig_modes.add_subplot(_gm[0, 1]), "entrain",
                "agreement with the demanded shape",
                "b  How well each cell followed")
    _panel_power(fig_modes.add_subplot(_gm[1, :]))
    save_fig(fig_modes, "response-modality")
    fig_modes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dose-diversity — what the controller does, minute by minute of the cycle

    Every window is aligned to the same point of the same demanded waveform, so the
    stimulation can be read as a distribution rather than as an average. For each minute
    of the cycle: what share of cell-cycles were on each rung of the dose ladder?

    A controller with nothing per-cell to say would put a single bright band at each
    minute — everyone on the same rung at the same phase. What is actually there is a
    spread at every phase, and the spread is the diversity the section is about, now
    conditioned on the moment in the cycle instead of averaged over it.

    The number in each panel is the average across phases of how spread out the rung
    choice is, in bits. Zero would mean every cell is given the same thing at that moment;
    the ceiling for a five-rung ladder is 2.32 bits.
    """)
    return


@app.cell(hide_code=True)
def _(np, osc_feat, osc_stack, pl):
    OSC_RUNGS = np.array([0.0, 20.0, 45.0, 85.0, 150.0])


    def osc_dose_by_phase(period):
        """Share of aligned windows on each rung, minute by minute of the cycle."""
        m = (osc_feat["period"] == period).to_numpy()
        D = osc_stack(m, "exposure_ms")
        R = osc_stack(m, "r_t")
        Y = osc_stack(m, "raw_cnr")
        idx = np.abs(D[..., None] - OSC_RUNGS[None, None, :]).argmin(axis=-1)
        share = np.stack([(idx == k).mean(axis=0) for k in range(len(OSC_RUNGS))])
        ent = -(np.where(share > 0, share * np.log2(np.maximum(share, 1e-12)), 0.0)).sum(0)
        # Mean response of the windows sitting on each rung at each minute, so the same
        # axes can be read as "what did that choice produce" rather than only "how often".
        resp = np.full(share.shape, np.nan)
        for k in range(len(OSC_RUNGS)):
            for t in range(D.shape[1]):
                sel = idx[:, t] == k
                if sel.sum() >= 10:
                    resp[k, t] = Y[sel, t].mean()
        return {"share": share, "entropy": ent, "resp": resp, "ref": R.mean(0),
                "median_dose": np.median(D, axis=0), "n": D.shape[0]}


    osc_dose_phase = {p: osc_dose_by_phase(p) for p in (20, 30, 60, 70)}

    pl.DataFrame([{"period": p, "windows": v["n"],
                   "mean_spread_bits": round(float(v["entropy"].mean()), 2),
                   "min_bits": round(float(v["entropy"].min()), 2),
                   "max_bits": round(float(v["entropy"].max()), 2),
                   "ceiling_bits": round(float(np.log2(len(OSC_RUNGS))), 2)}
                  for p, v in osc_dose_phase.items()])
    return (OSC_RUNGS,)


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    OSC_PERIODS,
    OSC_RUNGS,
    PERIOD_COLOUR,
    W_TEXT,
    np,
    osc_feat,
    pl,
    plt,
    save_fig,
):
    # --- dose-diversity, rebuilt --------------------------------------------------
    # The previous version was a per-phase heatmap of the share of cell-cycles on each
    # exposure rung, captioned "a controller with nothing per-cell to say would show
    # one bright band". That is not true, and the figure could not have shown it: the
    # marginal distribution of rungs at a given minute is INVARIANT to shuffling doses
    # between cells. A controller drawing a uniformly random rung for every cell every
    # minute produces the same picture. The statistic could not separate a per-cell
    # strategy from noise, which is why the panel did not read.
    #
    # What does separate them: is a cell's dose repeatable across its OWN cycles, more
    # than a shuffle of cell labels would give? That is an intraclass correlation, and
    # it is decisive here -- ICC 0.70-0.81 against a null centred on zero.
    _dose_cell = (osc_feat.select("cell", "period", "cycle", "dose_mean").drop_nulls())


    def _icc(df):
        """Between-cell share of the total variance in dose. 0 = cells interchangeable."""
        g = df.group_by("cell").agg(pl.col("dose_mean").mean().alias("m"),
                                    pl.col("dose_mean").count().alias("k"))
        grand = df["dose_mean"].mean()
        ssb = float((g["k"] * (g["m"] - grand) ** 2).sum())
        j = df.join(g.select("cell", "m"), on="cell")
        ssw = float(((j["dose_mean"] - j["m"]) ** 2).sum())
        a, n, kbar = len(g), df.height, float(g["k"].mean())
        if a < 2 or n - a < 1 or kbar <= 1:
            return np.nan
        msb, msw = ssb / (a - 1), ssw / (n - a)
        return (msb - msw) / (msb + (kbar - 1) * msw)


    _dd_rng = np.random.default_rng(0)
    dose_icc = []
    for _P in OSC_PERIODS:
        _s = _dose_cell.filter(pl.col("period") == _P)
        _cells = _s["cell"].to_numpy()
        _null = [_icc(_s.with_columns(pl.Series("cell", _dd_rng.permutation(_cells))))
                 for _ in range(200)]
        _null = np.array([v for v in _null if np.isfinite(v)])
        _half = (_s.with_columns((pl.col("cycle") % 2).alias("half"))
                   .group_by("cell", "half").agg(pl.col("dose_mean").mean())
                   .pivot(index="cell", on="half", values="dose_mean").drop_nulls())
        dose_icc.append(dict(period=_P, icc=_icc(_s), null_p95=float(np.percentile(_null, 95)),
                             n_cells=_s["cell"].n_unique(),
                             cycles_per_cell=_s.height / _s["cell"].n_unique(),
                             r_split=float(np.corrcoef(_half["0"], _half["1"])[0, 1]),
                             odd=_half["0"].to_numpy(), even=_half["1"].to_numpy()))
    dose_icc = {d["period"]: d for d in dose_icc}


    def _p_repeat(ax):
        """(a) A cell's dose in its odd cycles against the same cell's even cycles.

        If the controller had no per-cell strategy the cloud would be a blob; the
        demand is identical for every cell in an arm, so anything on the diagonal is
        the controller distinguishing cells and doing so the same way each cycle.
        """
        for P in OSC_PERIODS:
            d = dose_icc[P]
            ax.plot(d["odd"], d["even"], "o", ms=3.0, alpha=0.5,
                    color=PERIOD_COLOUR[P], markeredgewidth=0, label=f"{P} min")
        ax.plot([0, 150], [0, 150], color=INK, lw=1.0, ls="--", zorder=5)
        ax.set_xlim(-6, 156)
        ax.set_ylim(-6, 156)
        ax.set_xticks(OSC_RUNGS, [f"{int(v)}" for v in OSC_RUNGS], fontsize=6.5)
        ax.set_yticks(OSC_RUNGS, [f"{int(v)}" for v in OSC_RUNGS], fontsize=6.5)
        ax.set_xlabel("mean dose in its odd cycles (ms)")
        ax.set_ylabel("mean dose in its even cycles (ms)", fontsize=7.5)
        ax.set_title("a  It keeps its own dose", loc="left",
                     fontweight="bold", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.2, loc="upper left", ncol=2,
                  handletextpad=0.2, borderaxespad=0.3)
        _r = np.median([dose_icc[P]["r_split"] for P in OSC_PERIODS])
        ax.text(0.97, 0.04, f"r = {_r:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


    def _p_spread(ax):
        """(b) How far apart the cells are held, against the ladder they are held on."""
        for v in OSC_RUNGS:
            ax.axhline(v, color=GRID, lw=0.8, zorder=0)
        for i, P in enumerate(OSC_PERIODS):
            m = (_dose_cell.filter(pl.col("period") == P)
                           .group_by("cell").agg(pl.col("dose_mean").mean())["dose_mean"]
                           .to_numpy())
            x = i + (np.random.default_rng(i).random(len(m)) - 0.5) * 0.42
            ax.plot(x, m, "o", ms=2.8, alpha=0.42, color=PERIOD_COLOUR[P],
                    markeredgewidth=0)
            ax.plot([i - 0.30, i + 0.30], [np.median(m)] * 2, color=INK, lw=2.4, zorder=6)
            ax.annotate(f"{np.percentile(m, 90) - np.percentile(m, 10):.0f}",
                        xy=(i, 152), fontsize=6.5, color=INK, ha="center", va="top")
        ax.set_xticks(range(len(OSC_PERIODS)), [f"{p}" for p in OSC_PERIODS], fontsize=7)
        ax.set_xlim(-0.6, len(OSC_PERIODS) - 0.4)
        ax.set_ylim(-8, 162)
        ax.set_yticks(OSC_RUNGS, [f"{int(v)}" for v in OSC_RUNGS], fontsize=6.5)
        ax.set_xlabel("period (min)")
        ax.set_ylabel("that cell's mean dose (ms)", fontsize=7.5)
        ax.set_title("b  Across the whole ladder", loc="left",
                     fontweight="bold", fontsize=8.5)
        ax.text(0.5, 0.985, "numbers: p10-p90 spread, ms", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.2, color=MUTED)
        ax.set_axisbelow(True)


    def _p_icc(ax):
        """(c) The same claim as a number, against a shuffled-label null."""
        x = np.arange(len(OSC_PERIODS))
        v = [dose_icc[P]["icc"] for P in OSC_PERIODS]
        ax.bar(x, v, width=0.55, color=[PERIOD_COLOUR[P] for P in OSC_PERIODS])
        for i, P in enumerate(OSC_PERIODS):
            ax.plot([i - 0.30, i + 0.30], [dose_icc[P]["null_p95"]] * 2,
                    color=INK, lw=1.6, zorder=6)
            ax.text(i, v[i] + 0.02, f"{v[i]:.2f}", ha="center", fontsize=6.8, color=INK)
        ax.set_xticks(x, [f"{p}" for p in OSC_PERIODS], fontsize=7)
        ax.set_xlim(-0.6, len(OSC_PERIODS) - 0.4)
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("period (min)")
        ax.set_ylabel("share of dose variance that is\nbetween cells, not within one",
                      fontsize=7.5)
        ax.set_title("c  Not an artefact", loc="left",
                     fontweight="bold", fontsize=8.5)
        ax.text(0.5, 0.99, "black rules: 95th centile of the\n"
                "shuffled-label null", transform=ax.transAxes, ha="center",
                va="top", fontsize=6.2, color=MUTED, linespacing=1.4)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    fig_dp = plt.figure(figsize=(W_TEXT, 2.95))
    _gp = fig_dp.add_gridspec(1, 3, wspace=0.46,
                              left=0.075, right=0.985, top=0.895, bottom=0.175)
    _p_repeat(fig_dp.add_subplot(_gp[0, 0]))
    _p_spread(fig_dp.add_subplot(_gp[0, 1]))
    _p_icc(fig_dp.add_subplot(_gp[0, 2]))
    save_fig(fig_dp, "dose-diversity")
    fig_dp
    return


@app.cell(hide_code=True)
def _(OSC_FEATURED, div_stat, np, osc_df, osc_feat, pl):
    # Example cells for the two trace panels, chosen by a stated rule rather than by eye.
    _m = (osc_feat["period"] == OSC_FEATURED).to_numpy()
    EX_CELL_IDS = np.array(osc_feat["cell"])[_m]
    EX_CYCLE = osc_feat["cycle"].to_numpy()[_m]
    _V = div_stat["V"]

    # The cell whose windows are a MEDIAN distance from each other: typical, not the
    # tightest cluster in the run, which would flatter the claim.
    _uc = np.unique(EX_CELL_IDS)
    _spread = []
    for _c in _uc:
        _k = np.flatnonzero(EX_CELL_IDS == _c)
        if len(_k) < 4:
            _spread.append(np.inf)
            continue
        _d = np.linalg.norm(_V[_k][:, None, :] - _V[_k][None, :, :], axis=-1)
        _spread.append(float(_d[np.triu_indices(len(_k), 1)].mean()))
    _spread = np.array(_spread)
    _ok = np.isfinite(_spread)
    EX_CELL = str(_uc[_ok][np.argsort(_spread[_ok])[len(_spread[_ok]) // 2]])

    # The cycle with the most cells present, so the companion panel is drawn from the
    # fullest moment rather than a thin one.
    _cyc, _cnt = np.unique(EX_CYCLE, return_counts=True)
    EX_CYCLE_ID = int(_cyc[np.argmax(_cnt)])

    # Phase groups on the wall clock, for the design panel: one cell from each group of
    # one field, showing that at any minute the four sit at different points of the cycle.
    EX_PHASE = []
    for _pg in range(4):
        _g = (osc_df.filter((pl.col("fov") == 3)
                            & (pl.col("phase_offset_min") == _pg * OSC_FEATURED / 4))
              .sort("control_frame").group_by("control_frame", maintain_order=True)
              .agg(pl.col("r_t").first()))
        EX_PHASE.append((_pg * OSC_FEATURED // 4,
                         _g["control_frame"].to_numpy(), _g["r_t"].to_numpy()))

    {"example cell": EX_CELL, "its windows": int((EX_CELL_IDS == EX_CELL).sum()),
     "fullest cycle": EX_CYCLE_ID,
     "cells in it": int((EX_CYCLE == EX_CYCLE_ID).sum())}
    return


@app.cell(hide_code=True)
def _(np, osc_feat, osc_stack, pl):
    CYC_PHASE = np.linspace(0, 1, 120, endpoint=False)
    CYC_MIN_WINDOWS = 20      # rows thinner than this are noise, and are not drawn


    def osc_cycle_folds(period):
        """Median folded response for every cycle index in one arm."""
        m = (osc_feat["period"] == period).to_numpy()
        Y = osc_stack(m, "raw_cnr")
        cyc = osc_feat["cycle"].to_numpy()[m]
        t = np.arange(period) / period
        rows, keep = [], []
        for c in range(cyc.max() + 1):
            k = cyc == c
            if k.sum() < CYC_MIN_WINDOWS:
                continue
            rows.append(np.interp(CYC_PHASE, t, np.median(Y[k], axis=0), period=1))
            keep.append(c)
        return np.array(rows), np.array(keep)


    osc_folds = {p: osc_cycle_folds(p) for p in (20, 30, 60, 70)}

    # Departure of each window's amplitude from its own cell's average, so a cycle index
    # cannot look weak merely because weak cells happened to be present for it.
    osc_dev = osc_feat.select(["period", "cell", "cycle", "start_h", "amp"]).with_columns(
        (pl.col("amp") - pl.col("amp").mean().over(["period", "cell"])).alias("dev"))


    def _agg(by):
        return (osc_dev.group_by(["period", by])
                .agg(pl.len().alias("n"), pl.col("dev").mean().alias("dev"),
                     (pl.col("dev").std() / pl.len().sqrt()).alias("se"))
                .filter(pl.col("n") >= CYC_MIN_WINDOWS).sort(["period", by]))


    osc_dev_cycle = _agg("cycle")
    osc_dev_hour = _agg("hour") if False else (
        osc_dev.with_columns(pl.col("start_h").floor().alias("hour"))
        .group_by(["period", "hour"])
        .agg(pl.len().alias("n"), pl.col("dev").mean().alias("dev"),
             (pl.col("dev").std() / pl.len().sqrt()).alias("se"))
        .filter(pl.col("n") >= CYC_MIN_WINDOWS).sort(["period", "hour"]))

    pl.DataFrame([{"period": p, "cycles_drawn": len(v[1]),
                   "windows_first_cycle": int((osc_feat["period"] == p).to_numpy().sum()
                                              and (osc_dev.filter((pl.col("period") == p)
                                                                  & (pl.col("cycle") == 0)).height)),
                   "cycles_per_hour": round(60 / p, 2)}
                  for p, v in osc_folds.items()])
    return


@app.cell(hide_code=True)
def _(OSC_PERIODS, np, osc_df, osc_feat, pl):
    # Running total of delivered fluence per cell, for cells tracked from frame 0 only.
    _d = (osc_df.sort(["fov", "particle", "control_frame"])
          .with_columns(pl.col("control_frame").min().over(["fov", "particle"]).alias("_f0"))
          .filter(pl.col("_f0") == 0)
          .with_columns(pl.col("fluence_out").cum_sum()
                        .over(["fov", "particle"]).alias("cum")))

    osc_cum_track = (_d.select(["fov", "particle", "control_frame", "cum"])
                     .join(osc_feat.select(["fov", "particle", "period"]).unique(),
                           on=["fov", "particle"], how="inner"))

    # A FIXED cohort: cells tracked from the first controlled frame and still present
    # near the end. Letting the cohort shrink puts steps in the median that are cell
    # turnover, not light.
    _late = (osc_cum_track.filter(pl.col("control_frame") == 660)
             .select(["fov", "particle"]).unique())
    osc_cum = (osc_cum_track.join(_late, on=["fov", "particle"], how="inner")
               .group_by(["period", "control_frame"])
               .agg(pl.len().alias("n"), pl.col("cum").median().alias("cum"))
               .filter(pl.col("control_frame") <= 660).sort(["period", "control_frame"]))

    # Windows whose cell has a complete running total, with the amplitude expressed as a
    # departure from that cell's own average.
    osc_flu = (osc_feat
               .join(osc_cum_track.select(["fov", "particle", "control_frame", "cum"])
                     .rename({"control_frame": "start_frame"}),
                     on=["fov", "particle", "start_frame"], how="inner")
               .with_columns(
                   (pl.col("amp") - pl.col("amp").mean().over(["period", "cell"])).alias("dev"),
                   pl.col("start_h").floor().alias("hour")))

    # Common fluence bins across arms, so the four can be read on one axis.
    FLU_EDGES = np.array([0, 2e3, 5e3, 9e3, 14e3, 20e3, 27e3, 40e3])
    osc_flu = osc_flu.with_columns(
        pl.col("cum").cut(FLU_EDGES[1:-1].tolist(), labels=[str(i) for i in range(7)])
        .cast(pl.Utf8).cast(pl.Int32).alias("fbin"))

    osc_dev_flu = (osc_flu.group_by(["period", "fbin"])
                   .agg(pl.len().alias("n"), pl.col("dev").mean().alias("dev"),
                        (pl.col("dev").std() / pl.len().sqrt()).alias("se"),
                        pl.col("cum").median().alias("cum"))
                   .filter(pl.col("n") >= 20).sort(["period", "fbin"]))

    # The endogeneity-aware control: within one arm AND one hour, does more light matter?
    osc_terc = (osc_flu.with_columns(
        ((pl.col("cum").rank("ordinal").over(["period", "hour"]) - 1)
         / pl.len().over(["period", "hour"]) * 3).floor().clip(0, 2).alias("terc"))
        .group_by("terc").agg(pl.len().alias("n"), pl.col("dev").mean().alias("dev"),
                              (pl.col("dev").std() / pl.len().sqrt()).alias("se"),
                              pl.col("cum").median().alias("cum")).sort("terc"))

    def osc_collapse(xcol, nbin=8, minn=25):
        """Between-arm spread of the deficit at matched x, against the size of the trend.

        If an axis is what the decline really runs on, the four arms should agree when
        lined up on it. A high trend-to-spread ratio means the axis explains a lot
        relative to what it leaves unexplained between arms.
        """
        lo = max(float(osc_flu.filter(pl.col("period") == P)[xcol].min())
                 for P in OSC_PERIODS)
        hi = min(float(osc_flu.filter(pl.col("period") == P)[xcol].max())
                 for P in OSC_PERIODS)
        edges = np.linspace(lo, hi, nbin + 1)
        spreads, means = [], []
        for i in range(nbin):
            vals = []
            for P in OSC_PERIODS:
                k = osc_flu.filter((pl.col("period") == P) & (pl.col(xcol) >= edges[i])
                                   & (pl.col(xcol) < edges[i + 1]))
                if k.height >= minn:
                    vals.append(float(k["dev"].mean()))
            if len(vals) == len(OSC_PERIODS):
                spreads.append(float(np.std(vals, ddof=1)))
                means.append(float(np.mean(vals)))
        if not means:
            return np.nan, np.nan, 0
        return float(np.mean(spreads)), float(max(means) - min(means)), len(means)


    OSC_AXES = {"hours into\nthe run": "start_h", "cumulative\nfluence": "cum",
                "number of\ndrives": "cycle"}
    osc_collapse_tbl = pl.DataFrame([
        {"axis": _k.replace("\n", " "), "bins": _n, "between_arm_sd": _sd,
         "trend_range": _tr, "trend_over_spread": _tr / _sd}
        for _k, _v in OSC_AXES.items() for _sd, _tr, _n in [osc_collapse(_v)]])

    # Elapsed time and delivered light are correlated here but not identical, so a joint
    # fit says which is carrying the effect. Standardised, so the two are comparable.
    # Cycle number is left out: within an arm it is exactly hours x 60 / period, so a fit
    # containing both is degenerate — it gets the binned comparison instead.
    _y = osc_flu["dev"].to_numpy()
    _raw = [osc_flu["start_h"].to_numpy().astype(float),
            osc_flu["cum"].to_numpy().astype(float)]
    _X = np.column_stack([np.ones(len(_y))] + [(c - c.mean()) / c.std() for c in _raw])
    _b = np.linalg.lstsq(_X, _y, rcond=None)[0]
    _se = np.sqrt(np.diag(np.linalg.inv(_X.T @ _X)
                          * ((_y - _X @ _b) ** 2).sum() / (len(_y) - _X.shape[1])))
    osc_joint = pl.DataFrame({"term": ["hours into\nthe run", "cumulative\nfluence"],
                              "beta": [float(_b[1]), float(_b[2])],
                              "se": [float(_se[1]), float(_se[2])]})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## reachability — what limits the loop: the geometry, not the model
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- What limits the loop: the geometry, not the model -----------------------
    # The population's resting spread is wider than the light can move any one cell,
    # and light only pushes up. That single fact explains why the anchor is nearly
    # unchoosable and why the plate drifting between runs was fatal rather than
    # inconvenient -- and it is the argument for per-cell control.
    #
    # Resting is measured on one definition in every run: the cell's own exposure is
    # 0 and has been 0 for at least the preceding five frames, so this is where a
    # cell sits without light, not where it is on the way down from the last pulse.
    _rest_all = pl.read_parquet(materials_path("reach_resting_all.parquet"))
    _dem_all = pl.read_parquet(materials_path("reach_demand_all.parquet"))
    _reach = pl.read_parquet(materials_path("reach_cells.parquet"))
    _anchors = dict(pl.read_parquet(materials_path("reach_anchors.parquet")).iter_rows())
    V23_ANCHOR = _anchors["v23"]
    V23_AMP = 0.14                     # the sine's half-swing about the anchor

    REACH_RUNS = ["v10", "v16", "v21", "v23", "v24"]


    def _panel_drift(ax):
        """(a) Where the cells rested, against what they were asked for.

        One row per run. The bar is the resting population from p10 to p90 with the
        median ticked; the marker is that run's demand. Light only pushes up, so a
        cell resting to the RIGHT of the marker cannot be brought to it at all.
        """
        _y = np.arange(len(REACH_RUNS))[::-1]
        for _yi, _run in zip(_y, REACH_RUNS):
            _v = _rest_all.filter(pl.col("run") == _run)["rest"].to_numpy()
            _d = _dem_all.filter(pl.col("run") == _run)
            _lo, _md, _hi = np.percentile(_v, [10, 50, 90])
            ax.plot([_lo, _hi], [_yi, _yi], color=SERIES[0], lw=7, alpha=0.30,
                    solid_capstyle="butt")
            ax.plot([np.percentile(_v, 25), np.percentile(_v, 75)], [_yi, _yi],
                    color=SERIES[0], lw=7, solid_capstyle="butt")
            ax.plot([_md], [_yi], "|", ms=11, mew=2.0, color=INK, zorder=6)
            # the demand: a band where it moved, a marker at its median
            _dl, _dh = float(_d["demand_lo"][0]), float(_d["demand_hi"][0])
            if _dh - _dl > 0.02:
                ax.plot([_dl, _dh], [_yi + 0.30, _yi + 0.30], color=SERIES[1],
                        lw=2.0, solid_capstyle="butt", alpha=0.55)
            ax.plot([float(_d["demand_p50"][0])], [_yi], "D", ms=5,
                    color=SERIES[1], zorder=7)
            ax.text(1.98, _yi, f"{100 * float(_d['frac_above'][0]):.0f}% above",
                    va="center", ha="right", fontsize=6, color=MUTED)

        ax.set_yticks(_y, [f"{r}  (n={_dem_all.filter(pl.col('run') == r)['cells'][0]})"
                           for r in REACH_RUNS], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(REACH_RUNS) - 0.3)
        ax.set_xlim(0.30, 2.0)
        ax.set_xlabel("CNR\nbars: resting p25\u2013p75, faint p10\u2013p90   \u25c6 the demand")
        ax.set_title("a  Rest against demand", loc="left", fontweight="bold")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


    def _panel_span(ax):
        """(b) Each cell's reachable range against what it was asked for."""
        r = _reach.sort("rest")
        lo = r["rest"].to_numpy()
        hi = lo + np.maximum(r["headroom"].to_numpy(), 0)
        y = np.arange(len(lo))
        ax.axvspan(V23_ANCHOR - V23_AMP, V23_ANCHOR + V23_AMP, color=SERIES[1],
                   alpha=0.16, lw=0)
        ax.hlines(y, lo, hi, color=MUTED, lw=0.5, alpha=0.55)
        ax.plot(lo, y, ".", ms=1.6, color=INK)
        ax.axvline(V23_ANCHOR, color=SERIES[1], lw=1.2)
        _above = float((lo > V23_ANCHOR).mean())
        _short = float((hi < V23_ANCHOR).mean())
        ax.set_xlabel("CNR")
        ax.set_ylabel("cells, sorted by resting")
        ax.set_yticks([])
        ax.set_title("b  Reach in v23", loc="left", fontweight="bold")
        ax.text(0.97, 0.06,
                f"{_above:.0%} start above the anchor\n{_short:.0%} cannot reach it",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color=INK, linespacing=1.5)


    fig_reach = plt.figure(figsize=(W_TEXT, 2.8))
    _gr = fig_reach.add_gridspec(1, 2, wspace=0.30, left=0.125, right=0.975,
                                 top=0.90, bottom=0.235)
    _panel_drift(fig_reach.add_subplot(_gr[0, 0]))
    _panel_span(fig_reach.add_subplot(_gr[0, 1]))
    save_fig(fig_reach, "reachability")
    fig_reach
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- Reach, in every admitted run --------------------------------------------
    # Panel (b) of `reachability` asks this of v23 alone. Here it is asked of all six
    # admitted runs on ONE definition, so the panels can be read against each other.
    #
    # Resting and demand are recomputed from the track parquets rather than read from
    # reach_resting_all / reach_demand_all, because those two carry no v11. The recipe
    # is the published one and reproduces it exactly -- 2731/2731 cells across the
    # five runs those files do cover:
    #   rest    median raw CNR over frames whose exposure, and that of the five
    #           frames before it, is 0; cells with fewer than 10 such frames dropped
    #   demand  p5 / median / p95 of the run's own r_t
    #
    # The top of a cell's reach is p95 of its own raw CNR over the run, not the
    # maximum: a single bright frame is noise, not reach. That makes this measure
    # CONSERVATIVE and it is not the number panel (b) reports -- on v23 it gives 19%
    # unable to reach the demand where panel (b) gives 8%.
    #
    # Light only pushes up, so a cell resting to the RIGHT of the demand line is
    # unreachable no matter what the controller does.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    RCH_RUNS = ["v10", "v11", "v16", "v21", "v23", "v24"]
    RCH_COL = {"v10": SERIES[0], "v11": "#c9a227", "v16": SERIES[1],
               "v21": SERIES[2], "v23": "#8452a1", "v24": "#0e7c86"}
    RCH_QUIET, RCH_MIN_REST, RCH_Q = 6, 10, 0.95

    _rch_rows, _rch_dem = [], {}
    for _r in RCH_RUNS:
        _t = (pl.read_parquet(materials_path(f"tracks_{_r}.parquet"))
                .sort(["fov", "particle", "timestep"]))
        _q = _t.with_columns(
            pl.col("exposure_ms").fill_null(0)
              .rolling_sum(RCH_QUIET, min_samples=RCH_QUIET)
              .over(["fov", "particle"]).alias("_roll"))
        _rest = (_q.filter(pl.col("_roll") == 0).group_by(["fov", "particle"])
                   .agg(pl.col("raw_cnr").median().alias("rest"), pl.len().alias("n"))
                   .filter(pl.col("n") >= RCH_MIN_REST))
        _top = (_t.group_by(["fov", "particle"])
                  .agg(pl.col("raw_cnr").quantile(RCH_Q).alias("top")))
        _rch_rows.append(_rest.join(_top, on=["fov", "particle"], how="inner")
                              .with_columns(pl.lit(_r).alias("run"),
                                            (pl.col("top") - pl.col("rest")).alias("headroom")))
        # p5/p95, not min/max: this is the published band and it reproduces
        # reach_demand_all exactly on the five runs that file covers.
        _rt = _t["r_t"].drop_nulls()
        _rch_dem[_r] = (float(_rt.quantile(0.05)), float(_rt.median()),
                        float(_rt.quantile(0.95)))

    rch_cells = pl.concat(_rch_rows)

    fig_rch = plt.figure(figsize=(W_TEXT, 4.9))
    _grc = fig_rch.add_gridspec(2, 3, wspace=0.17, hspace=0.42, left=0.085,
                                right=0.985, top=0.93, bottom=0.155)

    for _i, _run in enumerate(RCH_RUNS):
        _ax = fig_rch.add_subplot(_grc[_i // 3, _i % 3])
        _s = rch_cells.filter(pl.col("run") == _run).sort("rest")
        _lo = _s["rest"].to_numpy()
        _hi = _lo + np.maximum(_s["headroom"].to_numpy(), 0)
        _y = np.arange(len(_lo))
        _dl, _dm, _dh = _rch_dem[_run]

        if _dh - _dl > 0.02:
            _ax.axvspan(_dl, _dh, color=MUTED, alpha=0.13, lw=0)
        _ax.hlines(_y, _lo, _hi, color=RCH_COL[_run], lw=0.45, alpha=0.55)
        _ax.plot(_lo, _y, ".", ms=1.3, color=INK)
        _ax.axvline(_dm, color=SERIES[1], lw=1.2, zorder=6)

        _above = 100 * float((_lo > _dm).mean())
        _short = 100 * float((_hi < _dm).mean())
        _ax.set_title(f"{_run}  (n={len(_lo)})", loc="left", fontweight="bold",
                      fontsize=8)
        _ax.set_xlim(0.30, 2.0)
        _ax.set_ylim(-len(_lo) * 0.02, len(_lo) * 1.34)
        _ax.set_yticks([])
        _ax.set_xticks([0.5, 1.0, 1.5])
        _ax.tick_params(labelsize=7)
        _ax.xaxis.grid(True, color=GRID, lw=0.6)
        _ax.set_axisbelow(True)
        _ax.text(0.04, 0.99,
                 f"{_above:.0f}% rest above it\n{_short:.0f}% cannot reach it",
                 transform=_ax.transAxes, va="top", fontsize=6.4, color=INK,
                 linespacing=1.5)
        if _i % 3 == 0:
            _ax.set_ylabel("cells, sorted by\nresting CNR", fontsize=7.5)
        if _i // 3 == 1:
            _ax.set_xlabel("CNR", fontsize=8)

    # An explicit key. The bar takes the run's colour in each panel, so it is drawn
    # grey here -- the colour separates the runs and carries nothing else.
    _rch_key = [
        Line2D([], [], marker=".", ms=6, color=INK, ls="none",
               label="where the cell rests, unlit"),
        Line2D([], [], color=SERIES[1], lw=1.6, label="the demand"),
        Line2D([], [], color=MUTED, lw=2.4, alpha=0.55,
               label="how far light moves it (rest → p95 of its own CNR)"),
        Patch(facecolor=MUTED, alpha=0.13, lw=0,
              label="range the demand swept"),
    ]
    fig_rch.legend(handles=_rch_key, loc="lower center", ncol=2, frameon=False,
                   fontsize=6.6, handlelength=1.8, columnspacing=1.4,
                   labelspacing=0.35, bbox_to_anchor=(0.535, 0.0))
    save_fig(fig_rch, "reachability-runs")
    fig_rch

    return


@app.cell(hide_code=True)
def _(GRID, INK, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- Where the cells were, and how well the loop held them -------------------
    # Panel (a) is panel (a) of `reachability` extended to every admitted run.
    # Panel (b) puts tracking error on the SAME run axis, so the geometry a run was
    # dealt and the accuracy it achieved are read off one pair of rows.
    #
    # One visual idiom in both panels: faint bar p10-p90, solid bar p25-p75, tick at
    # the median.
    #
    # Resting is recomputed with the published recipe -- median raw CNR over frames
    # whose exposure, and that of the five before it, is 0, cells with fewer than 10
    # such frames dropped. It reproduces reach_resting_all exactly (2731/2731 cells)
    # and additionally covers v11 and v19, which that file does not.
    #
    # Tracking error comes from tracking_all, which was rebuilt by
    # experiments/rebuild_tracking_all.py to include v11. That rebuild reproduces
    # every one of the 4103 pre-existing rows exactly, so nothing already quoted
    # from this file has moved.
    RD_RUNS = ["v10", "v11", "v16", "v19", "v21", "v23", "v24"]
    # constant_dose is deliberately absent: in v10 and v11 that arm holds a fixed
    # dose against a demand sitting on the resting median, so it scores well on
    # RMSE by not moving. It is not a control comparison and is left out here.
    #
    # v24 is split into the four rungs it was designed as, not pooled by controller.
    # Its arm 1 runs `population_mpc` with share = "half", which splits each field:
    # cells with (particle // 4) % 2 == 1 take one broadcast dose for the whole
    # field, the rest are planned individually. Both halves sit in the same dish
    # under the same objective, so 1a-vs-1b is paired within the field. The rule is
    # a pure function of the particle id and is fixed for a cell's life --
    # `particle % 2` is a different and wrong split. Its two open-loop arms are
    # separated too: one delivers a constant 60 ms, the other no light at all.
    # Panel (b) shows the CLOSED-LOOP arms only. v16's open-loop field is a dose
    # probe on a wider ladder that ignores the cell, and v24's two open-loop arms are
    # the lower rungs of its ladder; none of the three is trying to track, so putting
    # them on a tracking-error axis beside the controllers invites the reading that
    # they lost a contest they were not entered in. They belong in the E2 figure,
    # where the ladder is the subject.
    RD_SUB = {"per-cell MPC": SERIES[0], "one dose for the field": SERIES[2]}
    RD_ORDER = list(RD_SUB)

    _rd_rest, _rd_dem = {}, {}
    for _r in RD_RUNS:
        _t = (pl.read_parquet(materials_path(f"tracks_{_r}.parquet"))
                .sort(["fov", "particle", "timestep"]))
        _q = _t.with_columns(pl.col("exposure_ms").fill_null(0)
                               .rolling_sum(6, min_samples=6)
                               .over(["fov", "particle"]).alias("_roll"))
        _rd_rest[_r] = (_q.filter(pl.col("_roll") == 0).group_by(["fov", "particle"])
                          .agg(pl.col("raw_cnr").median().alias("rest"),
                               pl.len().alias("n"))
                          .filter(pl.col("n") >= 10))["rest"].to_numpy()
        _rt = _t["r_t"].drop_nulls()
        _rd_dem[_r] = (float(_rt.quantile(0.05)), float(_rt.median()),
                       float(_rt.quantile(0.95)))

    rd_trk = pl.read_parquet(materials_path("tracking_all.parquet")).with_columns(
        pl.when(pl.col("arm") == "open loop, 0 ms")
          .then(pl.lit("dark, no light"))
        .when(pl.col("controller") == "open_loop")
          .then(pl.lit("open loop, fixed light"))
        .when((pl.col("controller") == "population_mpc")
              & ((pl.col("particle") // 4) % 2 == 1))
          .then(pl.lit("one dose for the field"))
        .when(pl.col("controller").is_in(["sequence_mpc", "population_mpc"]))
          .then(pl.lit("per-cell MPC"))
        .otherwise(pl.lit("drop")).alias("sub"))

    # Rows are ordered by panel (b): the run whose loop tracked best sits at the top.
    # The key pools a run's closed-loop arms, so v24 is placed by 1a and 1b together.
    _rd_rank = (rd_trk.filter(pl.col("run").is_in(RD_RUNS)
                              & pl.col("sub").is_in(RD_ORDER))
                .group_by("run").agg(pl.col("rmse").median().alias("m"))
                .sort("m")["run"].to_list())
    _rd_y = {_r: len(_rd_rank) - 1 - _i for _i, _r in enumerate(_rd_rank)}


    def _rd_bar(ax, v, y, colour, lw):
        """The shared idiom: p10-p90 faint, p25-p75 solid, median ticked."""
        _a, _b, _m, _c, _d = np.percentile(v, [10, 25, 50, 75, 90])
        ax.plot([_a, _d], [y, y], color=colour, lw=lw, alpha=0.30,
                solid_capstyle="butt")
        ax.plot([_b, _c], [y, y], color=colour, lw=lw, solid_capstyle="butt")
        ax.plot([_m], [y], "|", ms=1.6 * lw, mew=1.8, color=INK, zorder=6)


    fig_rd = plt.figure(figsize=(W_TEXT, 3.9))
    _grd = fig_rd.add_gridspec(1, 2, wspace=0.07, left=0.145, right=0.985,
                               top=0.87, bottom=0.31)

    _axa = fig_rd.add_subplot(_grd[0, 0])
    for _r in RD_RUNS:
        _y = _rd_y[_r]
        _rd_bar(_axa, _rd_rest[_r], _y, SERIES[0], 6.5)
        _dl, _dm, _dh = _rd_dem[_r]
        if _dh - _dl > 0.02:
            _axa.plot([_dl, _dh], [_y + 0.32, _y + 0.32], color=SERIES[1], lw=2.0,
                      alpha=0.55, solid_capstyle="butt")
        _axa.plot([_dm], [_y], "D", ms=5, color=SERIES[1], zorder=7)
    _axa.set_yticks([_rd_y[_r] for _r in RD_RUNS],
                    [f"{_r}  (n={len(_rd_rest[_r])})" for _r in RD_RUNS], fontsize=7)
    _axa.tick_params(axis="y", length=0)
    _axa.set_ylim(-0.75, len(RD_RUNS) - 0.25)
    _axa.set_xlim(0.30, 1.75)
    _axa.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    _axa.set_xlabel("resting CNR")
    _axa.set_title("a  Rest against demand", loc="left", fontweight="bold")
    _axa.xaxis.grid(True, color=GRID, lw=0.6)
    _axa.set_axisbelow(True)

    _axb = fig_rd.add_subplot(_grd[0, 1], sharey=_axa)
    for _r in RD_RUNS:
        _y = _rd_y[_r]
        _s = rd_trk.filter(pl.col("run") == _r)
        _present = [_c for _c in RD_ORDER
                    if _s.filter(pl.col("sub") == _c).height]
        _offs = (np.linspace(0.27, -0.27, len(_present)) if len(_present) > 1
                 else [0.0])
        for _c, _o in zip(_present, _offs):
            _v = _s.filter(pl.col("sub") == _c)["rmse"].to_numpy()
            _rd_bar(_axb, _v, _y + _o, RD_SUB[_c],
                    6.5 if len(_present) == 1 else 3.4)
    _axb.tick_params(labelleft=False, axis="y", length=0)
    _axb.set_xlim(0, 0.70)
    _axb.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    _axb.set_xticks([0.1, 0.3, 0.5])
    #_axb.set_xticks([0.2, 0.4, 0.6])
    _axb.set_xlabel("tracking error, RMSE (CNR)")
    _axb.set_title("b  corresponds to error", loc="left", fontweight="bold")
    _axb.xaxis.grid(True, color=GRID, lw=0.6)
    _axb.set_axisbelow(True)

    _rd_key = [
        plt.Line2D([], [], color=SERIES[0], lw=5,
                   label="resting population  ·  per-cell MPC"),
        plt.Line2D([], [], marker="D", ms=5, color=SERIES[1], ls="none",
                   label="the demand (bar above: range it swept)"),
        plt.Line2D([], [], color=SERIES[2], lw=5,
                   label="one dose for the whole field"),
    ]
    fig_rd.legend(handles=_rd_key, loc="lower center", ncol=2, frameon=False,
                  fontsize=6.5, handlelength=1.7, columnspacing=1.4,
                  labelspacing=0.35, bbox_to_anchor=(0.56, -0.005))
    save_fig(fig_rd, "reach-and-tracking")
    fig_rd

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where the cells sat, and how well the loop held them.** Panel (a) is the resting
    population of each run with the demand it was given; panel (b) is tracking error on
    the same rows, closed-loop arms only. Runs are ordered by median tracking error.
    Bars: p10–p90 faint, p25–p75 solid, tick at the median.
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- How much cells move, and how many of them move --------------------------
    # Neither question has an answer without a dose and without a null, so the figure
    # supplies both and drops the run/arm dimension entirely.
    #
    # THE NULL IS MEASURED. v24's arm 3 sat in the dark for twelve hours -- 229 cells,
    # zero stimulation light -- and still reads a median reach of 0.14 CNR. That is
    # drift and tracking noise. The per-frame measurement noise floor is NOT the
    # yardstick: those dark cells score ten times it, because this statistic spans
    # hours and the floor spans one frame.
    #
    # THE UNIT IS THE CELL, BINNED BY THE LIGHT IT ITSELF RECEIVED, not by the arm it
    # sat in. Under closed loop two cells in one field are dosed differently, so the
    # arm mean throws away most of the range; pooling by each cell's own mean dose
    # recovers it and puts every run on one axis.
    #
    # Reach per cell is p95 of its own raw CNR minus its own resting level, resting
    # being the median over frames whose exposure, and that of the five before, is 0.
    MOV_DARK_ARM = "open loop, 0 ms"
    MOV_RUNS = ["v10", "v11", "v16", "v19", "v21", "v23", "v24"]
    MOV_BINS = [(0.0, 20.0), (20.0, 35.0), (35.0, 50.0), (50.0, 75.0),
                (75.0, 150.0), (150.0, 1e9)]

    _mov_arms = (pl.read_parquet(materials_path("tracks_arms.parquet"))
                   .select("run", "fov", "arm_label"))


    def _mov_reach(run):
        """Per cell: where it rests, how far above that it ever got, and its dose."""
        _t = (pl.read_parquet(materials_path(f"tracks_{run}.parquet"))
                .sort(["fov", "particle", "timestep"])
                .join(_mov_arms.filter(pl.col("run") == run).drop("run"),
                      on="fov", how="left"))
        _q = _t.with_columns(pl.col("exposure_ms").fill_null(0)
                               .rolling_sum(6, min_samples=6)
                               .over(["fov", "particle"]).alias("_roll"))
        _rest = (_q.filter(pl.col("_roll") == 0)
                   .group_by(["fov", "particle", "arm_label"])
                   .agg(pl.col("raw_cnr").median().alias("rest"), pl.len().alias("nq"))
                   .filter(pl.col("nq") >= 10))
        _top = _t.group_by(["fov", "particle"]).agg(
            pl.col("raw_cnr").quantile(0.95).alias("t95"),
            pl.col("exposure_ms").mean().alias("dose"))
        return (_rest.join(_top, on=["fov", "particle"])
                     .with_columns((pl.col("t95") - pl.col("rest")).alias("reach"),
                                   pl.lit(run).alias("run")))


    mov_cells = pl.concat([_mov_reach(_r) for _r in MOV_RUNS])
    _mov_dark = mov_cells.filter(pl.col("arm_label") == MOV_DARK_ARM)["reach"].to_numpy()
    MOV_D50, MOV_D90 = np.percentile(_mov_dark, [50, 90])

    _mov_rows = [("no light at all", _mov_dark, INK)]
    _mov_lit = mov_cells.filter(pl.col("arm_label") != MOV_DARK_ARM)
    for _i, (_lo, _hi) in enumerate(MOV_BINS):
        _v = _mov_lit.filter((pl.col("dose") > _lo) & (pl.col("dose") <= _hi))["reach"].to_numpy()
        _lab = f"{_lo:.0f}–{_hi:.0f} ms" if _hi < 1e8 else f"over {_lo:.0f} ms"
        _mov_rows.append((_lab, _v, plt.cm.viridis(0.12 + 0.72 * _i / (len(MOV_BINS) - 1))))

    fig_mov = plt.figure(figsize=(W_TEXT, 3.9))
    _axm = fig_mov.add_axes([0.225, 0.285, 0.60, 0.60])
    _axm.axvspan(MOV_D50, MOV_D90, color=MUTED, alpha=0.14, lw=0, zorder=1)
    _axm.axvline(MOV_D90, color=INK, lw=1.0, ls="--", zorder=2)

    for _j, (_lab, _v, _col) in enumerate(_mov_rows):
        _y = len(_mov_rows) - 1 - _j
        _a, _b, _m, _c, _d = np.percentile(_v, [10, 25, 50, 75, 90])
        _axm.plot([_a, _d], [_y, _y], color=_col, lw=7, alpha=0.30,
                  solid_capstyle="butt", zorder=3)
        _axm.plot([_b, _c], [_y, _y], color=_col, lw=7, solid_capstyle="butt", zorder=4)
        _axm.plot([_m], [_y], "|", ms=11, mew=2.0, color=INK, zorder=6)
        _axm.text(1.035, _y, f"{100 * float((_v > MOV_D90).mean()):.0f}%",
                  transform=_axm.get_yaxis_transform(), va="center", fontsize=7.5,
                  color=INK, fontweight="bold" if _j else "normal")

    _axm.set_yticks(range(len(_mov_rows) - 1, -1, -1),
                    [f"{_l}\n{len(_v):,} cells" for _l, _v, _ in _mov_rows], fontsize=7)
    _axm.tick_params(axis="y", length=0)
    # room under the last row for the note on the dashed line
    _axm.set_ylim(-1.0, len(_mov_rows) - 0.3)
    _axm.set_xlim(0, 0.95)
    _axm.tick_params(labelsize=7.5)
    _axm.set_xlabel("how far the cell got above its own resting level (CNR)",
                    fontsize=8)
    _axm.xaxis.grid(True, color=GRID, lw=0.6)
    _axm.set_axisbelow(True)
    _axm.set_title("How cells react to light", loc="left",
                   fontweight="bold", fontsize=9.5)
    _axm.text(1.035, 1.045, "cells\nclearing\nthe null",
              transform=_axm.transAxes, va="bottom", fontsize=6.5, color=MUTED,
              linespacing=1.4)
    _axm.text(MOV_D90 + 0.012, -0.60,
              "dark p90 — what every lit row is judged against", fontsize=6.3,
              color=MUTED, va="center")

    save_fig(fig_mov, "how-cells-move")
    fig_mov

    return MOV_D50, MOV_D90, MOV_DARK_ARM, mov_cells


@app.cell(hide_code=True)
def _(MOV_D50, MOV_D90, MOV_DARK_ARM, mo, mov_cells, pl):
    mo.md(f"""
    **What light buys, and for how many cells.** Rows are cells pooled across all seven
    runs and binned by the mean light that cell itself received. The null is v24's dark
    arm — {mov_cells.filter(pl.col("arm_label") == MOV_DARK_ARM).height:,}
    cells held with no stimulation light for twelve hours, median apparent reach
    {MOV_D50:.2f} CNR and p90 {MOV_D90:.2f}. Bars: p10–p90 faint, p25–p75 solid, tick at
    the median.
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, pl, plt, save_fig):
        # --- E2 / v24: what the run was, and what the light actually did -------------
    # `feedback-ladder` is the RESULT. This is the design and its audit, and it exists
    # because two things about this run cannot be read off the result and change how
    # it must be worded.
    #
    # (a) The ladder is three rungs, not two arms, and the top rung is split INSIDE
    #     each field. `population_mpc` with share = "half" gives cells with
    #     (particle // 4) % 2 == 1 one broadcast dose for the whole field and plans
    #     the rest individually, so 1a-vs-1b is paired within a dish rather than
    #     compared across dishes. Fields 4/2/2 give 8!/(4!2!2!) = 420 relabellings,
    #     so the smallest attainable exact permutation p is 1/420 = 0.0024.
    #
    # (b) One reference for all eight fields, so the arms differ only in controller.
    #     Nine 60-min blocks cross three levels with three rates, and level is held
    #     constant across rates so a per-type difference cannot be "that type asked
    #     for more light". Nothing after the ceiling probe is scored.
    #
    # (c) THE DOSE MATCH FAILED. Arm 2 was set to 60 ms because v23's closed-loop
    #     arms had averaged 54-65 ms. On this plate the demand sat above reach and
    #     the controller spent 112.5 ms/frame -- 1.87x the arm it is being compared
    #     against, in the direction that flatters it. The policy names delivered
    #     light a primary reported quantity; this panel is that report.
    E2_ARMCOL = {"1a per-cell dose": SERIES[0], "1b broadcast dose": SERIES[2],
                 "open loop, 60 ms": MUTED, "open loop, 0 ms": "#a9a7a2"}
    E2_ORDER = list(E2_ARMCOL)
    E2_TARGET_MS = 60.0          # what arm 2 was set to, from v23's closed-loop means

    _e2d_arms = (pl.read_parquet(materials_path("tracks_arms.parquet"))
                   .filter(pl.col("run") == "v24").select("fov", "arm_label"))
    _e2d = (pl.read_parquet(materials_path("tracks_v24.parquet"))
              .join(_e2d_arms, on="fov", how="left")
              .with_columns(
                  pl.when(pl.col("arm_label") == "population MPC")
                    .then(pl.when((pl.col("particle") // 4) % 2 == 1)
                            .then(pl.lit("1b broadcast dose"))
                            .otherwise(pl.lit("1a per-cell dose")))
                    .otherwise(pl.col("arm_label")).alias("sub")))

    fig_e2d = plt.figure(figsize=(W_TEXT, 5.6))
    _ge2d = fig_e2d.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.55,
                                 wspace=0.34, left=0.095, right=0.975,
                                 top=0.93, bottom=0.235)

    # (a) the demand every field was given, and how it is built
    _axa = fig_e2d.add_subplot(_ge2d[0, :])
    _ref = (_e2d.group_by("timestep")
                .agg(pl.col("r_t").first(), pl.col("phase_label").first())
                .sort("timestep"))
    _axa.plot(_ref["timestep"], _ref["r_t"], color=INK, lw=1.3)
    for _row in (_ref.group_by("phase_label", maintain_order=True)
                     .agg(pl.col("timestep").min().alias("t0"),
                          pl.col("timestep").max().alias("t1")).iter_rows(named=True)):
        _lab = _row["phase_label"]
        if _lab.startswith("demand_"):
            _lvl, _typ = _lab.split("_")[2], _lab.split("_")[3]
            _c = {"H": SERIES[0], "O30": SERIES[2], "O15": SERIES[1]}[_typ]
            _axa.axvspan(_row["t0"], _row["t1"], color=_c, alpha=0.13, lw=0)
            _axa.text((_row["t0"] + _row["t1"]) / 2, 1.63, _typ, ha="center",
                      fontsize=5.6, color=_c, fontweight="bold")
            _axa.text((_row["t0"] + _row["t1"]) / 2, 1.55, _lvl, ha="center",
                      fontsize=5.6, color=MUTED)
        elif _lab.startswith("settle") or _lab.startswith("probe") or _lab == "gap_probe":
            _axa.axvspan(_row["t0"], _row["t1"], color=GRID, alpha=0.55, lw=0)
    _axa.text(46, 0.42, "settle", fontsize=6, color=MUTED, ha="center")
    _axa.text(727, 1.45, "ceiling probe\nnot scored", fontsize=5.8, color=MUTED,
              ha="right", va="top", linespacing=1.4)
    _axa.set_xlim(16, 736)
    _axa.set_ylim(0.28, 1.72)
    _axa.set_xlabel("minutes into the run", fontsize=8)
    _axa.set_ylabel("demanded CNR", fontsize=8)
    _axa.tick_params(labelsize=7)
    _axa.set_title("a  One reference, given to all eight fields", loc="left",
                   fontweight="bold", fontsize=9)
    _axa.yaxis.grid(True, color=GRID, lw=0.6)
    _axa.set_axisbelow(True)
    _axa.text(1.0, 1.015,
              "levels  L 1.20 · M 1.36 · H 1.52      rates  H hold · O30, O15 sine",
              transform=_axa.transAxes, ha="right", va="bottom", fontsize=6,
              color=MUTED)

    # (b) the ladder, and where each rung sat on the plate
    _axb = fig_e2d.add_subplot(_ge2d[1, 0])
    _fov_arm = dict(_e2d_arms.iter_rows())
    for _f in range(8):
        _a = _fov_arm[_f]
        if _a == "population MPC":
            _axb.add_patch(plt.Rectangle((_f - 0.42, 0.5), 0.84, 0.5,
                                         color=E2_ARMCOL["1a per-cell dose"]))
            _axb.add_patch(plt.Rectangle((_f - 0.42, 0.0), 0.84, 0.5,
                                         color=E2_ARMCOL["1b broadcast dose"]))
        else:
            _axb.add_patch(plt.Rectangle((_f - 0.42, 0.0), 0.84, 1.0,
                                         color=E2_ARMCOL[_a]))
    _axb.set_xlim(-0.6, 7.6)
    _axb.set_ylim(-0.15, 1.15)
    _axb.set_xticks(range(8), [str(_f) for _f in range(8)], fontsize=7)
    _axb.set_yticks([])
    _axb.set_xlabel("field of view", fontsize=8)
    _axb.set_title("b  The ladder on the plate", loc="left", fontweight="bold",
                   fontsize=9)
    for _s in ("top", "right", "left", "bottom"):
        _axb.spines[_s].set_visible(False)
    _axb.tick_params(axis="both", length=0)


    # (c) the ladder, drawn as a ladder. The panel has to carry three things the
    # reader cannot get from (a) or (b): that all four ran AT THE SAME TIME on one
    # plate, that they are ORDERED (each can do one thing less than the rung above),
    # and that the point of the ordering is that each adjacent PAIR isolates exactly
    # one capability. Arrow = the order, brackets = the pairs, swatches = back to (b).
    # Line widths are set by the panel: ~45 characters at 6.6 pt, ~30 at 7.4.
    _axc = fig_e2d.add_subplot(_ge2d[1, 1])
    _axc.set_axis_off()
    _axc.set_xlim(0, 1)
    _axc.set_ylim(0, 1)
    E2_RUNGS_TXT = [
        ("1a per-cell dose", "1a", "a dose chosen per cell"),
        ("1b broadcast dose", "1b", "one dose for the field"),
        ("open loop, 60 ms", "2", "the same dose, every frame"),
        ("open loop, 0 ms", "3", "no light at all"),
    ]
    E2_QUESTIONS = ["does a dose per cell beat one dose?",
                    "does reacting to the cells help?",
                    "does the light do anything?"]
    E2_RY = [0.60, 0.415, 0.23, 0.045]

    _axc.text(0.0, 1.00,
              "All four ran at once, on one plate. Each\n"
              "rung can do one thing less than the one\n"
              "above, so each pair asks one question.",
              color=INK, fontsize=6.6, va="top", linespacing=1.55)

    # the order, as an arrow down the left margin
    _axc.annotate("", xy=(0.03, E2_RY[-1] - 0.03), xytext=(0.03, E2_RY[0] + 0.03),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1,
                                  shrinkA=0, shrinkB=0))
    _axc.text(0.005, (E2_RY[0] + E2_RY[-1]) / 2, "can do less", rotation=90,
              ha="center", va="center", fontsize=6.0, color=MUTED)

    for _r, (_key, _arm, _txt) in enumerate(E2_RUNGS_TXT):
        _y = E2_RY[_r]
        _axc.add_patch(plt.Rectangle((0.070, _y - 0.026), 0.036, 0.052,
                                     color=E2_ARMCOL[_key], clip_on=False))
        _axc.text(0.125, _y, _arm, color=INK, fontsize=7.4, fontweight="bold",
                  va="center")
        _axc.text(0.205, _y, _txt, color=INK, fontsize=7.4, va="center")
        if _r < len(E2_QUESTIONS):
            _ym = (_y + E2_RY[_r + 1]) / 2
            # a bracket spanning the two rungs it compares
            _axc.plot([0.165, 0.145, 0.145, 0.165],
                      [_y - 0.050, _y - 0.050, E2_RY[_r + 1] + 0.050,
                       E2_RY[_r + 1] + 0.050],
                      color=MUTED, lw=0.8, clip_on=False)
            _axc.text(0.185, _ym, E2_QUESTIONS[_r], color=MUTED, fontsize=6.4,
                      style="italic", va="center")

    _axc.set_title("c  The comparison ladder", loc="left", fontweight="bold",
                   fontsize=9)

    _e2_key = [plt.Line2D([], [], color=E2_ARMCOL[_k], lw=6, label=_l)
               for _k, _l in (("1a per-cell dose", "arm 1a  a dose per cell"),
                              ("1b broadcast dose", "arm 1b  one dose for the field"),
                              ("open loop, 60 ms", "arm 2  pre-computed open loop"),
                              ("open loop, 0 ms", "arm 3  no light"))]
    _axb.legend(handles=_e2_key, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                frameon=False, fontsize=6.2, ncol=2, handlelength=1.5,
                columnspacing=1.2, labelspacing=0.3)

    save_fig(fig_e2d, "e2-design")
    fig_e2d

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The v24 design.** Arm 1's four fields are split inside the dish: `(particle // 4) % 2`
    decides which cells share one broadcast dose, so 1a against 1b is paired within a field
    rather than compared across dishes. Arm 2 delivers a flat 60 ms, set from v23's
    closed-loop arm means; arm 3 receives no stimulation light at all. The nine blocks are a
    complete 3 × 3 of levels against rates, each combination once, in a counterbalanced order.
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- E2 / v24: what the three lit rungs achieved -----------------------------
    # (a) is the run as it happened: the median cell of each rung against the one
    #     reference all of them were given. The dark arm is not drawn -- it never
    #     tries to follow anything, and its floor is in `feedback-ladder`.
    #
    # (b) is the contrast the split-field design was built for. Per-cell and
    #     broadcast cells sit in the SAME four dishes, so the comparison is paired at
    #     the field and every field-level confound cancels inside the pair. Scoring
    #     is per block type, because the design pre-registered a directional
    #     prediction about which rate the advantage should be largest at.
    #
    # THE PRE-REGISTERED PREDICTION IS NOT CONFIRMED. It expected the closed-loop
    # advantage largest on the holds and smallest on the fast sine, on the argument
    # that O15's 149 deg of lag makes reaction impossible. What is there instead is
    # the reverse ordering, with the largest gap at O30. At four fields per type this
    # is directional only and the write-up must say so rather than reading the order.
    E2R_COL = {"1a": SERIES[0], "1b": SERIES[2], "c60": MUTED}
    E2R_NAME = {"1a": "per-cell MPC", "1b": "one dose for the field",
                "c60": "constant 60 ms, no feedback"}
    E2R_TYPES = ["H", "O30", "O15"]
    E2R_TYPENAME = {"H": "hold", "O30": "sine, 30 min", "O15": "sine, 15 min"}

    _e2r_arms = (pl.read_parquet(materials_path("tracks_arms.parquet"))
                   .filter(pl.col("run") == "v24").select("fov", "arm_label"))
    _e2r = (pl.read_parquet(materials_path("tracks_v24.parquet"))
              .join(_e2r_arms, on="fov", how="left")
              .with_columns(
                  pl.when(pl.col("arm_label") == "population MPC")
                    .then(pl.when((pl.col("particle") // 4) % 2 == 1)
                            .then(pl.lit("1b")).otherwise(pl.lit("1a")))
                  .when(pl.col("arm_label") == "open loop, 60 ms").then(pl.lit("c60"))
                  .otherwise(pl.lit("dark")).alias("sub")))

    fig_e2r = plt.figure(figsize=(W_TEXT, 5.0))
    _ge2r = fig_e2r.add_gridspec(2, 1, height_ratios=[1.0, 0.78], hspace=0.46,
                                 left=0.105, right=0.925, top=0.93, bottom=0.155)

    # (a) the median cell of each rung, against the demand it was given
    _axa = fig_e2r.add_subplot(_ge2r[0])
    _ref = (_e2r.group_by("timestep").agg(pl.col("r_t").first())
                .sort("timestep").filter(pl.col("timestep") >= 76))
    _axa.plot(_ref["timestep"], _ref["r_t"], color=INK, lw=1.5, ls="--",
              label="the demand — identical in all eight fields", zorder=7)
    for _s in ("1a", "1b", "c60"):
        _g = (_e2r.filter((pl.col("sub") == _s) & (pl.col("timestep") >= 76))
                  .group_by("timestep")
                  .agg(pl.col("raw_cnr").median().alias("m"),
                       pl.col("raw_cnr").quantile(0.25).alias("lo"),
                       pl.col("raw_cnr").quantile(0.75).alias("hi"))
                  .sort("timestep"))
        if _s == "1a":
            _axa.fill_between(_g["timestep"], _g["lo"], _g["hi"],
                              color=E2R_COL[_s], alpha=0.13, lw=0)
        _axa.plot(_g["timestep"], _g["m"], color=E2R_COL[_s], lw=1.5,
                  label=E2R_NAME[_s])
    # The nine blocks cross three levels; the sines swing 0.16 peak-to-peak about
    # whichever level their block sits at, so these are the midlines the demand
    # returns to and not three separate references.
    for _lv, _nm in ((1.20, "L"), (1.36, "M"), (1.52, "H")):
        _axa.axhline(_lv, color=MUTED, lw=0.8, ls=":", zorder=1)
        _axa.text(742, _lv, f"{_nm}  {_lv:.2f}", va="center", ha="left",
                  fontsize=6.4, color=MUTED, clip_on=False)
    _axa.text(742, 1.615, "demand\nlevels", va="center", ha="left", fontsize=6,
              color=MUTED, linespacing=1.4, clip_on=False)
    _axa.set_xlim(76, 736)
    _axa.set_xlabel("minutes into the run", fontsize=8)
    _axa.set_ylabel("CNR", fontsize=8)
    _axa.tick_params(labelsize=7)
    _axa.set_title("a  The median cell of each rung, against the demand",
                   loc="left", fontweight="bold", fontsize=9)
    _axa.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower center",
                handlelength=1.5, columnspacing=1.3, borderaxespad=0.15)
    _axa.yaxis.grid(True, color=GRID, lw=0.6)
    _axa.set_axisbelow(True)
    _axa.text(0.995, 0.965, "band: IQR of the per-cell arm", transform=_axa.transAxes,
              ha="right", va="top", fontsize=6, color=MUTED)

    # (b) why the run could not be tracked: the demand against what a cell can do
    # Each cell's ceiling is p95 of its own CNR over the whole run -- what it actually
    # reached, under whatever light it was given, including the 300 ms top rung.
    #
    # THE ANCHOR WAS NOT THE MISTAKE. The 42-min dark settle measured this plate at a
    # resting median of 1.055, and the run's own rule (resting p50 + 0.12) puts the
    # anchor at 1.175 against the 1.20 used -- inside the 0.10 tolerance its abort
    # rule specified. What the design then did was stack two further levels 0.16 and
    # 0.32 ABOVE that anchor and spend two thirds of the scored run there.
    _axb = fig_e2r.add_subplot(_ge2r[1])
    _ceil = (_e2r.group_by(["fov", "particle"])
                 .agg(pl.col("raw_cnr").quantile(0.95).alias("ceil"), pl.len().alias("nn"))
                 .filter(pl.col("nn") >= 120))["ceil"].to_numpy()
    _grid = np.linspace(0.6, 1.9, 400)
    _frac = np.array([100 * (_ceil >= _g).mean() for _g in _grid])
    _axb.fill_between(_grid, 0, _frac, color=SERIES[0], alpha=0.13, lw=0)
    _axb.plot(_grid, _frac, color=SERIES[0], lw=2.0, zorder=4)

    for _lvl, _nm in ((1.20, "L"), (1.36, "M"), (1.52, "H")):
        _p = 100 * float((_ceil >= _lvl).mean())
        _axb.plot([_lvl, _lvl], [0, _p], color=INK, lw=1.0, ls="--", zorder=5)
        _axb.plot([_lvl], [_p], "o", ms=5, color=INK, zorder=6)
        _axb.text(_lvl + 0.035, _p + 5.5, f"{_nm}  {_lvl:.2f}\n{_p:.0f}% of cells",
                  ha="left", va="center", fontsize=6.4, color=INK,
                  linespacing=1.45)

    # where an untouched cell actually sat, at the start and at the end
    for _x, _lab, _dy in ((0.979, "resting, first hour", 62), (0.842, "resting, last hour", 78)):
        _axb.plot([_x, _x], [0, 100], color=MUTED, lw=0.9, ls=":", zorder=3)
        _axb.text(_x - 0.012, _dy, _lab, rotation=90, ha="right", va="center",
                  fontsize=5.8, color=MUTED)

    _axb.set_xlim(0.6, 1.9)
    _axb.set_ylim(0, 118)
    _axb.set_yticks([0, 25, 50, 75, 100])
    _axb.set_xlabel("CNR", fontsize=8)
    _axb.set_ylabel("cells that ever reached it (%)", fontsize=7.5)
    _axb.set_title("b  Why it could not be tracked: the demand against what a cell can do",
                   loc="left", fontweight="bold", fontsize=9)
    _axb.tick_params(labelsize=7)
    _axb.grid(True, color=GRID, lw=0.6)
    _axb.set_axisbelow(True)
    _axb.text(0.995, 0.98,
              "L is reachable for most cells. M and H are not —\n"
              "and six of the nine blocks, two thirds of the\n"
              "scored run, sit at M or H.",
              transform=_axb.transAxes, ha="right", va="top", fontsize=6.4,
              color=INK, linespacing=1.6)

    save_fig(fig_e2r, "e2-arms")
    fig_e2r

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the three lit rungs achieved.** A cell's ceiling in panel (b) is the 95th
    percentile of its own CNR over the run — what it actually reached, under whatever light
    it was given. The resting lines are the dark arm, which never received any light. Panel
    (a) shows the median cell of each rung against the one reference all eight fields were
    given, with the interquartile band on the per-cell arm.
    """)
    return


@app.cell(hide_code=True)
def _(materials_path, np, pl):
    # --- Free-window stimulation: extracting what the controller did with the room -
    # v21 and v23 are the same design: 50-min blocks of run-up then demand, with the
    # LAST 0 / 4 / 10 / 20 minutes of the run-up left unscored. Inside that window the
    # controller pays nothing, so whatever it does there is what it does when the
    # objective stops telling it what to do.
    #
    #   arm 1  fov 0, 7   0 free min      graded throughout
    #   arm 2  fov 2, 5   4 free min      shorter than the 3-5 min dead time
    #   arm 3  fov 1, 6  10 free min      clears the lead the demand needs
    #   arm 4  fov 3, 4  20 free min      clears it with room to excurse and return
    #
    # Every arm is read on the SAME 20-minute tail of the run-up, so the comparison is
    # between identical frames that differ only in whether they were scored. v21 has
    # no phase labels, so its blocks are recovered from the 50-min period; v23 labels
    # its run-ups directly.
    FW_FREE = {0: 0, 7: 0, 2: 4, 5: 4, 1: 10, 6: 10, 3: 20, 4: 20}
    FW_W = 20
    FW_RUNS = ["v21", "v23"]


    def _fw_extract(run):
        _t = (pl.read_parquet(materials_path(f"tracks_{run}.parquet"))
                .sort(["fov", "particle", "timestep"]))
        if run == "v23":
            _ends = (_t.filter(pl.col("phase_label").str.starts_with("runup_"))
                       .group_by("phase_label")
                       .agg(pl.col("timestep").max().alias("e")).sort("e"))["e"].to_list()
        else:
            # v21 carries no phase labels. Its demand ONSETS are every 50 min from
            # minute 101; the durations vary (6 to 26 min), so the run-up ends at the
            # last anchor frame before each onset, not at a fixed offset into a block.
            _ends = list(range(100, 601, 50))
        _rows, _vecs = [], []
        for _bi, _e in enumerate(_ends):
            _seg = _t.filter((pl.col("timestep") > _e - FW_W) & (pl.col("timestep") <= _e))
            for (_fov, _p), _s in _seg.group_by(["fov", "particle"], maintain_order=True):
                _d = _s.sort("timestep")["exposure_ms"].fill_null(0).to_numpy()
                if len(_d) < FW_W:
                    continue
                _tot = float(_d.sum())
                _rows.append(dict(
                    run=run, fov=_fov, particle=_p, block=_bi, free=FW_FREE[_fov],
                    total=_tot,
                    centroid=float((np.arange(FW_W) * _d).sum() / _tot / (FW_W - 1))
                             if _tot > 0 else float("nan"),
                    npulse=int(((_d > 0).astype(int)
                                * (1 - np.r_[0, (_d[:-1] > 0).astype(int)])).sum()),
                    cell=f"{run}_{_fov}_{_p}"))
                _vecs.append(_d)
        return pl.DataFrame(_rows), np.array(_vecs, float)


    _fw_parts = [_fw_extract(_r) for _r in FW_RUNS]
    fw_win = pl.concat([_p[0] for _p in _fw_parts])
    fw_vec = np.vstack([_p[1] for _p in _fw_parts])


    def fw_icc(vals, cells, min_windows=3):
        """Between-cell share of variance in a descriptor, one-way random effects.

        The question the whole section turns on: is the spread in what the controller
        does a property of the CELL, or of the moment? A high value says two windows
        from one cell resemble each other more than two windows from different cells.
        """
        _ok = np.isfinite(vals)
        vals, cells = vals[_ok], cells[_ok]
        _uq, _i = np.unique(cells, return_inverse=True)
        _keep = np.bincount(_i)[_i] >= min_windows
        vals, _i = vals[_keep], _i[_keep]
        if len(vals) < 30:
            return np.nan
        _uq, _i = np.unique(_i, return_inverse=True)
        _n = np.bincount(_i)
        if len(_n) < 5:
            return np.nan
        _means = np.bincount(_i, weights=vals) / _n
        _msb = (_n * (_means - vals.mean()) ** 2).sum() / (len(_n) - 1)
        _msw = ((vals - _means[_i]) ** 2).sum() / max(len(vals) - len(_n), 1)
        return max((_msb - _msw) / (_msb + (_n.mean() - 1) * _msw), 0.0)


    f"{fw_win.height:,} windows · {fw_win['cell'].n_unique():,} cells · vectors {fw_vec.shape}"

    return FW_FREE, FW_RUNS, fw_icc, fw_vec, fw_win


@app.cell(hide_code=True)
def _(
    FW_RUNS,
    GRID,
    MUTED,
    SERIES,
    W_TEXT,
    fw_icc,
    fw_win,
    np,
    pl,
    plt,
    save_fig,
):
    # --- Candidate A: the diversity is in HOW MUCH, and it belongs to the cell -----
    # One question, asked of three descriptors of a free window: how much light was
    # spent, WHEN in the window it was spent, and how many separate pulses it came in.
    # For each, what share of the spread is between cells rather than between moments?
    #
    # The null is a permutation that keeps everything except cell identity: within a
    # field and a block, the windows are re-dealt among the cells that were there. It
    # preserves the number of windows per cell, the block, the field and the marginal
    # distribution of every descriptor, and destroys only the pairing between a cell
    # and its own windows. Anything above that band is per-cell structure.
    FWA_METRIC = [("total", "how much light"), ("centroid", "when in the window"),
                  ("npulse", "how many pulses")]
    FWA_FREE = [0, 4, 10, 20]
    FWA_COL = dict(zip(FWA_FREE, [MUTED, "#a9a7a2", SERIES[2], SERIES[0]]))
    FWA_BOOT = 80

    _fwa_rng = np.random.default_rng(0)
    _fwa = []
    for _run in FW_RUNS:
        for _f in FWA_FREE:
            _sub = fw_win.filter((pl.col("run") == _run) & (pl.col("free") == _f))
            _cells = _sub["cell"].to_numpy()
            _strata = np.char.add(_sub["fov"].cast(str).to_numpy().astype(str),
                                  _sub["block"].cast(str).to_numpy().astype(str))
            for _m, _ in FWA_METRIC:
                _v = _sub[_m].to_numpy().astype(float)
                _null = []
                for _ in range(FWA_BOOT):
                    _perm = _cells.copy()
                    for _s in np.unique(_strata):
                        _mask = _strata == _s
                        _perm[_mask] = _fwa_rng.permutation(_perm[_mask])
                    _x = fw_icc(_v, _perm)
                    if np.isfinite(_x):
                        _null.append(_x)
                _fwa.append(dict(run=_run, free=_f, metric=_m,
                                 icc=fw_icc(_v, _cells),
                                 null=float(np.percentile(_null, 95)) if _null else np.nan,
                                 n=_sub.height))
    fwa = pl.DataFrame(_fwa)

    fig_fwa = plt.figure(figsize=(W_TEXT, 2.9))
    _gfa = fig_fwa.add_gridspec(1, 3, wspace=0.16, left=0.095, right=0.985,
                                top=0.80, bottom=0.30)
    for _j, (_m, _title) in enumerate(FWA_METRIC):
        _ax = fig_fwa.add_subplot(_gfa[0, _j])
        for _ri, _run in enumerate(FW_RUNS):
            _s = fwa.filter((pl.col("metric") == _m) & (pl.col("run") == _run)).sort("free")
            _x = np.arange(len(FWA_FREE)) + (_ri - 0.5) * 0.26
            _ax.plot(_x, _s["icc"], "o-", ms=5, lw=1.2, color=SERIES[_ri],
                     label=_run, mfc="white", mew=1.5)
        _nl = fwa.filter(pl.col("metric") == _m)["null"].max()
        _ax.axhspan(0, _nl, color=MUTED, alpha=0.18, lw=0)
        _ax.set_xticks(range(len(FWA_FREE)), [str(_f) for _f in FWA_FREE], fontsize=7)
        _ax.set_xlim(-0.5, len(FWA_FREE) - 0.5)
        _ax.set_ylim(0, 0.78)
        _ax.set_xlabel("free minutes", fontsize=7.5)
        _ax.set_title(f"{'abc'[_j]}  {_title}", loc="left", fontweight="bold", fontsize=8.5)
        _ax.yaxis.grid(True, color=GRID, lw=0.6)
        _ax.set_axisbelow(True)
        _ax.tick_params(labelsize=7)
        if _j == 0:
            _ax.set_ylabel("share of the spread that\nbelongs to the cell", fontsize=7.5)
            _ax.legend(frameon=False, fontsize=6.5, loc="upper right",
                       handlelength=1.4, borderaxespad=0.3)
            _ax.text(len(FWA_FREE) - 0.6, _nl + 0.025,
                     "what re-dealing the windows already gives",
                     fontsize=6, color=MUTED, va="bottom", ha="right")
        else:
            _ax.tick_params(labelleft=False)

    save_fig(fig_fwa, "freewindow-whose-diversity")
    fig_fwa

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Whose diversity is it.** Each point is one arm of one run, read on the same 20-minute
    tail of the run-up. Shaded: the 95th percentile of a null that re-deals windows among
    the cells of the same field and block, keeping the number of windows per cell, the
    field, the block and every marginal distribution, and destroying only the pairing
    between a cell and its own windows.
    """)
    return


@app.cell(hide_code=True)
def _(GRID, MUTED, SERIES, W_TEXT, fw_vec, fw_win, np, plt, save_fig):
    # --- Candidate B: are there kinds of stimulation, or one continuum? -----------
    # The tempting analysis is to embed each free window's dose sequence and look for
    # clusters. It does not work here, and the control is the point of the figure.
    #
    # Each window is divided by its own total, so only the SHAPE is left -- when the
    # light was spent, not how much. The control re-orders each window's own doses in
    # time: it keeps the exact set of rungs the controller chose and destroys the
    # order it chose them in. Anything the observed data shows that the control also
    # shows is a property of the six-rung ladder, not of strategy.
    from scipy.stats import gaussian_kde as _fwb_kde

    FWB_MIN_FREE = 10          # only arms with room to have a strategy at all

    _fwb_keep = ((fw_win["free"] >= FWB_MIN_FREE) & (fw_win["total"] > 0)).to_numpy()
    _fwb_X = fw_vec[_fwb_keep]
    _fwb_X = _fwb_X / _fwb_X.sum(1, keepdims=True)
    _fwb_rng = np.random.default_rng(0)
    _fwb_S = np.array([_fwb_rng.permutation(_r) for _r in _fwb_X])


    def _fwb_pca(X):
        _U, _s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
        return _U[:, :2] * _s[:2], (_s ** 2 / (_s ** 2).sum())


    def _fwb_silverman(x, n_boot=150, seed=0):
        """Critical bandwidth for one mode, then how unusual that much smoothing is."""
        _r = np.random.default_rng(seed)
        _lo, _hi = 1e-4, float(x.max() - x.min())
        for _ in range(50):
            _h = (_lo + _hi) / 2
            _g = np.linspace(x.min() - _h, x.max() + _h, 400)
            _d = _fwb_kde(x, bw_method=_h / x.std(ddof=1))(_g)
            if ((_d[1:-1] > _d[:-2]) & (_d[1:-1] > _d[2:])).sum() > 1:
                _lo = _h
            else:
                _hi = _h
        _hc, _b = _hi, 0
        for _ in range(n_boot):
            _xb = _r.choice(x, len(x), replace=True) + _hc * _r.standard_normal(len(x))
            _xb = x.mean() + (_xb - _xb.mean()) / np.sqrt(1 + _hc ** 2 / x.var())
            _g = np.linspace(_xb.min() - _hc, _xb.max() + _hc, 400)
            _d = _fwb_kde(_xb, bw_method=_hc / _xb.std(ddof=1))(_g)
            if ((_d[1:-1] > _d[:-2]) & (_d[1:-1] > _d[2:])).sum() > 1:
                _b += 1
        return _b / n_boot


    _pc_o, _ev_o = _fwb_pca(_fwb_X)
    _pc_s, _ev_s = _fwb_pca(_fwb_S)
    FWB_P_OBS = _fwb_silverman(_pc_o[:, 0])
    FWB_P_SHUF = _fwb_silverman(_pc_s[:, 0])

    fig_fwb = plt.figure(figsize=(W_TEXT, 3.0))
    _gfb = fig_fwb.add_gridspec(1, 3, wspace=0.42, left=0.085, right=0.985,
                                top=0.86, bottom=0.32, width_ratios=[1.0, 1.0, 0.85])

    _axfa = fig_fwb.add_subplot(_gfb[0, 0])
    _axfa.hexbin(_pc_o[:, 0], _pc_o[:, 1], gridsize=34, cmap="Blues", mincnt=1,
                 linewidths=0)
    _axfa.set_xlabel("first shape component", fontsize=7.5)
    _axfa.set_ylabel("second", fontsize=7.5)
    _axfa.set_title("a  The space of shapes", loc="left", fontweight="bold", fontsize=8.5)
    _axfa.tick_params(labelsize=7)
    _axfa.text(0.97, 0.96, f"{len(_fwb_X):,} windows", transform=_axfa.transAxes,
               ha="right", va="top", fontsize=6.2, color=MUTED)

    _axfb = fig_fwb.add_subplot(_gfb[0, 1])
    for _v, _c, _lab, _p in ((_pc_o[:, 0], SERIES[0], "what the controller did", FWB_P_OBS),
                             (_pc_s[:, 0], MUTED, "same doses, order destroyed", FWB_P_SHUF)):
        _g = np.linspace(np.percentile(_v, 0.5), np.percentile(_v, 99.5), 400)
        _axfb.plot(_g, _fwb_kde(_v)(_g), color=_c, lw=1.8,
                   label=f"{_lab}   p = {_p:.2f}")
    _axfb.set_xlabel("first shape component", fontsize=7.5)
    _axfb.set_ylabel("density", fontsize=7.5)
    _axfb.set_title("b  The control agrees", loc="left",
                    fontweight="bold", fontsize=8.5)
    _axfb.legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.4)
    _axfb.tick_params(labelsize=7)
    _axfb.set_yticks([])

    _axfc = fig_fwb.add_subplot(_gfb[0, 2])
    _kk = np.arange(1, 9)
    _axfc.plot(_kk, 100 * _ev_o[:8], "o-", ms=4, color=SERIES[0], lw=1.3,
               label="observed", mfc="white", mew=1.4)
    _axfc.plot(_kk, 100 * _ev_s[:8], "o-", ms=4, color=MUTED, lw=1.3,
               label="order destroyed", mfc="white", mew=1.4)
    _axfc.set_xlabel("component", fontsize=7.5)
    _axfc.set_ylabel("variance explained (%)", fontsize=7.5)
    _axfc.set_title("c  No dominant shape", loc="left", fontweight="bold", fontsize=8.5)
    _axfc.set_ylim(0, 14)
    _axfc.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.4)
    _axfc.yaxis.grid(True, color=GRID, lw=0.6)
    _axfc.set_axisbelow(True)
    _axfc.tick_params(labelsize=7)

    save_fig(fig_fwb, "freewindow-no-types")
    fig_fwb

    return FWB_P_OBS, FWB_P_SHUF


@app.cell(hide_code=True)
def _(FWB_P_OBS, FWB_P_SHUF, mo):
    mo.md(f"""
    **Are there kinds of stimulation, or one continuum?** Arms with 10 or 20 free minutes,
    both runs, each window divided by its own total so only timing remains. Silverman's
    test rejects one mode on the observed data (p = {FWB_P_OBS:.2f}) — and equally on a
    control that keeps each window's own dose rungs and destroys their order
    (p = {FWB_P_SHUF:.2f}). The modes are the six-rung ladder showing through the
    projection, not kinds of stimulation.
    """)
    return


@app.cell(hide_code=True)
def _(
    FW_FREE,
    FW_RUNS,
    GRID,
    MUTED,
    SERIES,
    W_TEXT,
    materials_path,
    np,
    pl,
    plt,
    save_fig,
):
    # --- Candidate C: what the free window actually buys --------------------------
    # The design's own primary question. Each 50-min block is a run-up followed by a
    # demand the controller IS graded on; the arms differ only in how many of the last
    # run-up minutes carry no score. So: does the controller use the room, and does
    # using it help?
    #
    # Three outcomes, all read on the graded demand half, which is identical across
    # arms. Arm 2's four free minutes are shorter than the loop's 3-5 min dead time,
    # so light commanded there lands after the demand has already opened -- it is free
    # on paper and constrained in practice, and it is what separates "freedom helped"
    # from "unscored frames helped".
    #
    # THE UNIT OF RANDOMISATION IS THE FIELD AND THERE ARE TWO PER ARM. The small dots
    # are those fields. Nothing here is a significance test and none is offered; the
    # claim rests on the direction agreeing across two runs and eight fields.
    FWC_OUT = [("free_dose", "light spent in the free window (ms)", None),
               ("arrival", "minutes before it first reaches the demand", None),
               ("dem_rmse", "tracking error over the demand (CNR)", None)]
    FWC_FREE = [0, 4, 10, 20]

    _fwc = []
    for _run in FW_RUNS:
        _t = (pl.read_parquet(materials_path(f"tracks_{_run}.parquet"))
                .sort(["fov", "particle", "timestep"]))
        if _run == "v23":
            _ru = (_t.filter(pl.col("phase_label").str.starts_with("runup_"))
                     .group_by("phase_label").agg(pl.col("timestep").min().alias("s"),
                                                  pl.col("timestep").max().alias("e"))
                     .sort("s"))
            _dm = (_t.filter(pl.col("phase_label").str.starts_with("demand_"))
                     .group_by("phase_label").agg(pl.col("timestep").min().alias("s"),
                                                  pl.col("timestep").max().alias("e"))
                     .sort("s"))
            _blocks = list(zip(_ru["s"], _ru["e"], _dm["s"], _dm["e"]))
        else:
            # run-up ends at the last anchor frame; the demand runs from the next
            # frame to wherever r_t drops back to the anchor
            _blocks = [(_b - 24, _b, _b + 1, _b + 26) for _b in range(100, 601, 50)]
        for _bi, (_rs, _re, _ds, _de) in enumerate(_blocks):
            _seg = _t.filter((pl.col("timestep") >= _rs) & (pl.col("timestep") <= _de))
            for (_fov, _p), _s in _seg.group_by(["fov", "particle"], maintain_order=True):
                _s = _s.sort("timestep")
                _f = FW_FREE[_fov]
                _ru_s = _s.filter((pl.col("timestep") >= _rs) & (pl.col("timestep") <= _re))
                _dm_s = _s.filter((pl.col("timestep") >= _ds) & (pl.col("timestep") <= _de))
                if _ru_s.height < 15 or _dm_s.height < 15:
                    continue
                _d = _ru_s["exposure_ms"].fill_null(0).to_numpy()
                _y = _dm_s["raw_cnr"].to_numpy()
                _r = _dm_s["r_t"].to_numpy()
                _hit = np.flatnonzero(_y >= _r)
                _fwc.append(dict(run=_run, fov=_fov, free=_f,
                                 free_dose=float(_d[-_f:].sum()) if _f else 0.0,
                                 dem_rmse=float(np.sqrt(np.mean((_y - _r) ** 2))),
                                 arrival=float(_hit[0]) if len(_hit) else np.nan,
                                 arrived=float(len(_hit) > 0)))
    fwc = pl.DataFrame(_fwc)

    fig_fwc = plt.figure(figsize=(W_TEXT, 2.9))
    _gfc = fig_fwc.add_gridspec(1, 3, wspace=0.42, left=0.095, right=0.985,
                                top=0.80, bottom=0.30)
    for _j, (_col, _lab, _) in enumerate(FWC_OUT):
        _ax = fig_fwc.add_subplot(_gfc[0, _j])
        for _ri, _run in enumerate(FW_RUNS):
            _byfov = (fwc.filter(pl.col("run") == _run)
                         .group_by(["free", "fov"])
                         .agg(pl.col(_col).median().alias("m")).sort("free"))
            _byarm = (fwc.filter(pl.col("run") == _run).group_by("free")
                         .agg(pl.col(_col).median().alias("m")).sort("free"))
            _xi = {_f: _i for _i, _f in enumerate(FWC_FREE)}
            _ax.plot([_xi[_f] for _f in _byfov["free"]], _byfov["m"], "o", ms=3,
                     color=SERIES[_ri], alpha=0.45, mew=0)
            _ax.plot([_xi[_f] for _f in _byarm["free"]], _byarm["m"], "-", lw=1.6,
                     color=SERIES[_ri], label=_run, zorder=4)
            _ax.plot([_xi[_f] for _f in _byarm["free"]], _byarm["m"], "o", ms=5,
                     color=SERIES[_ri], mfc="white", mew=1.5, zorder=5)
        _ax.axvspan(-0.5, 1.5, color=MUTED, alpha=0.10, lw=0)
        _ax.set_xticks(range(len(FWC_FREE)), [str(_f) for _f in FWC_FREE], fontsize=7)
        _ax.set_xlim(-0.5, len(FWC_FREE) - 0.5)
        _ax.set_xlabel("free minutes", fontsize=7.5)
        _ax.set_ylabel(_lab, fontsize=7)
        _ax.set_title(f"{'abc'[_j]}  {['It uses the room', 'It arrives sooner', 'It tracks no better'][_j]}",
                      loc="left", fontweight="bold", fontsize=8.5)
        _ax.yaxis.grid(True, color=GRID, lw=0.6)
        _ax.set_axisbelow(True)
        _ax.tick_params(labelsize=7)
        if _j == 0:
            _ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.4)
            _ax.text(0.5, 0.04, "shorter than\nthe dead time", transform=_ax.transAxes,
                     ha="center", va="bottom", fontsize=5.8, color=MUTED,
                     linespacing=1.4)

    save_fig(fig_fwc, "freewindow-what-it-buys")
    fig_fwc

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the free window buys.** Small dots are fields, two per arm, and they are the unit
    that was randomised; lines join the pooled medians. The shaded arms have 0 and 4 free
    minutes — four minutes is inside the loop's 3–5 min dead time, so light commanded there
    arrives only after the demand has already opened. Nothing here is a significance test.
    """)
    return


@app.cell(hide_code=True)
def _(
    INK,
    MUTED,
    SERIES,
    W_TEXT,
    fw_vec,
    fw_win,
    materials_path,
    np,
    pl,
    plt,
    save_fig,
):
    # --- Two windows from opposite ends of the shape space -----------------------
    # The space in `freewindow-no-types` says there are no kinds of stimulation, only
    # a continuum. This shows what living at two ends of that continuum looks like, so
    # "continuum" is not taken on faith.
    #
    # THE TWO ARE CHOSEN BY A STATED RULE, not by eye: the window nearest the 5th and
    # nearest the 95th percentile of the first shape component, both taken at the
    # median of the second, so each is typical of its end rather than an outlier of
    # it. Each window is divided by its own total before the space is built, so the
    # two differ in WHEN the light was spent and not in how much.
    FWX_LO_Q, FWX_HI_Q = 0.05, 0.95

    _fwx_mask = ((fw_win["free"] >= 10) & (fw_win["total"] > 0)).to_numpy()
    _fwx_meta = fw_win.filter(pl.Series(_fwx_mask)).with_row_index("row")
    _fwx_X = fw_vec[_fwx_mask]
    _fwx_X = _fwx_X / _fwx_X.sum(1, keepdims=True)
    _fwx_U, _fwx_S, _ = np.linalg.svd(_fwx_X - _fwx_X.mean(0), full_matrices=False)
    _fwx_pc = _fwx_U[:, :2] * _fwx_S[:2]

    _p2mid = np.median(_fwx_pc[:, 1])
    _targets = [(np.quantile(_fwx_pc[:, 0], FWX_LO_Q), _p2mid),
                (np.quantile(_fwx_pc[:, 0], FWX_HI_Q), _p2mid)]
    _sd = _fwx_pc.std(0)
    _pick = [int(np.argmin((((_fwx_pc - np.array(_t)) / _sd) ** 2).sum(1))) for _t in _targets]


    def _fwx_block_bounds(run, block):
        """Where this block's run-up and demand sit, in minutes into the run."""
        if run == "v23":
            _t = pl.read_parquet(materials_path("tracks_v23.parquet"))
            _ru = (_t.filter(pl.col("phase_label").str.starts_with("runup_"))
                     .group_by("phase_label").agg(pl.col("timestep").min().alias("s"),
                                                  pl.col("timestep").max().alias("e"))
                     .sort("s"))
            _dm = (_t.filter(pl.col("phase_label").str.starts_with("demand_"))
                     .group_by("phase_label").agg(pl.col("timestep").max().alias("e"))
                     .sort("e"))
            return int(_ru["s"][block]), int(_ru["e"][block]), int(_dm["e"][block])
        _b = 100 + 50 * block
        return _b, _b + 24, _b + 49


    fig_fwx = plt.figure(figsize=(W_TEXT, 3.6))
    _gfx = fig_fwx.add_gridspec(2, 2, width_ratios=[0.95, 1.15], wspace=0.34,
                                hspace=0.52, left=0.095, right=0.90,
                                top=0.90, bottom=0.155)

    _axs = fig_fwx.add_subplot(_gfx[:, 0])
    _axs.hexbin(_fwx_pc[:, 0], _fwx_pc[:, 1], gridsize=32, cmap="Blues", mincnt=1,
                linewidths=0)
    for _k, _i in enumerate(_pick):
        _axs.plot([_fwx_pc[_i, 0]], [_fwx_pc[_i, 1]], "o", ms=9, mfc="none",
                  mec=SERIES[1], mew=2.0, zorder=6)
        _axs.text(_fwx_pc[_i, 0], _fwx_pc[_i, 1] + 0.045, "AB"[_k], ha="center",
                  fontsize=8, fontweight="bold", color=SERIES[1], zorder=7)
    _axs.set_xlabel("first shape component", fontsize=7.5)
    _axs.set_ylabel("second shape component", fontsize=7.5)
    _axs.set_title("a  Two windows from opposite ends", loc="left",
                   fontweight="bold", fontsize=8.5)
    _axs.tick_params(labelsize=7)

    for _k, _i in enumerate(_pick):
        _m = _fwx_meta.filter(pl.col("row") == _i).to_dicts()[0]
        _rs, _re, _de = _fwx_block_bounds(_m["run"], _m["block"])
        _tr = (pl.read_parquet(materials_path(f"tracks_{_m['run']}.parquet"))
                 .filter((pl.col("fov") == _m["fov"]) & (pl.col("particle") == _m["particle"])
                         & (pl.col("timestep") >= _rs) & (pl.col("timestep") <= _de))
                 .sort("timestep"))
        _ax = fig_fwx.add_subplot(_gfx[_k, 1])
        _x = _tr["timestep"].to_numpy() - _rs
        _ax.axvspan(_re - _rs - _m["free"] + 1, _re - _rs, color=SERIES[2],
                    alpha=0.16, lw=0)
        _ax.axvline(_re - _rs + 0.5, color=MUTED, lw=0.9, ls="--")
        _ax.plot(_x, _tr["r_t"].to_numpy(), color=INK, lw=1.1, ls="--")
        _ax.plot(_x, _tr["raw_cnr"].to_numpy(), color=SERIES[0], lw=1.6)
        _ax.set_ylabel("CNR", fontsize=7.5)
        _ax.tick_params(labelsize=7)
        _ax.set_xlim(0, _de - _rs)
        _ax.set_title(f"{'bc'[_k]}  window {'AB'[_k]} — {_m['run']}, {_m['free']} free min",
                      loc="left", fontweight="bold", fontsize=8.5)
        _axd = _ax.twinx()
        _axd.bar(_x, _tr["exposure_ms"].fill_null(0).to_numpy(), width=0.9,
                 color=MUTED, alpha=0.45, lw=0)
        _axd.set_ylim(0, 330)
        _axd.set_ylabel("light (ms)", fontsize=7, color=MUTED)
        _axd.tick_params(labelsize=6.5, colors=MUTED)
        _axd.set_zorder(0)
        _ax.set_zorder(1)
        _ax.patch.set_visible(False)
        if _k == 1:
            _ax.set_xlabel("minutes into the block", fontsize=7.5)
        else:
            _ax.text(_re - _rs - _m["free"] + 1.5, _ax.get_ylim()[1],
                     "free window", fontsize=6, color=SERIES[2], va="top")
            _ax.text(_re - _rs + 1.5, _ax.get_ylim()[1], "graded demand", fontsize=6,
                     color=MUTED, va="top")

    save_fig(fig_fwx, "freewindow-examples")
    fig_fwx

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Two windows from opposite ends of the shape space.** Chosen by a stated rule: the
    window nearest the 5th and nearest the 95th percentile of the first shape component,
    both at the median of the second. Dashed line: the demand. Bars: light commanded. Green:
    the unscored window. Both cells are shown over one whole block.
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, fw_vec, fw_win, np, pl, plt, save_fig):
    # --- Every free window, as a picture -----------------------------------------
    # One row per window, one column per minute of the 20-minute tail, colour is the
    # rung the controller commanded. Rows are ordered by the first shape component --
    # the same axis the shape space is built on -- so the question "are there kinds of
    # stimulation, or one continuum?" is answered by looking: kinds would appear as
    # horizontal bands with edges between them, a continuum as a gradient.
    #
    # The colour scale is the LADDER ITSELF, six discrete rungs, not a continuous
    # ramp. That is deliberate: the discreteness is what breaks the clustering test in
    # `freewindow-no-types`, and here it is visible rather than argued.
    #
    # The strip on the left is how many minutes of that row's window were unscored;
    # the trace on the right is the total light in the row. Sorting is by SHAPE alone,
    # so any vertical structure in the total is a fact about the sort, not built in.
    from matplotlib.colors import BoundaryNorm, ListedColormap

    FWH_RUNGS = [0.0, 20.0, 45.0, 85.0, 150.0, 300.0]

    # All four arms, not only the two with real room. Arm 1 (0 free minutes) is the
    # control: its last twenty minutes are graded throughout, so its shape is what the
    # objective dictates. Arm 2's four free minutes are shorter than the loop's dead
    # time, so light commanded there lands after the demand has already opened.
    #
    # The shape axis is computed once over ALL windows, so the four blocks are ordered
    # on a common scale and can be read against each other.
    FWH_FREE = [0, 4, 10, 20]
    FWH_COL = [MUTED, "#a9a7a2", SERIES[2], SERIES[0]]

    _fwh_mask = (fw_win["total"] > 0).to_numpy()
    _fwh_meta = fw_win.filter(pl.Series(_fwh_mask))
    _fwh_D = fw_vec[_fwh_mask]
    _fwh_X = _fwh_D / _fwh_D.sum(1, keepdims=True)
    _fwh_U, _fwh_S, _ = np.linalg.svd(_fwh_X - _fwh_X.mean(0), full_matrices=False)

    # Arm first, shape second: the two blocks become four, and the gradient is redrawn
    # inside each. That is the comparison the panel is for -- does more free time
    # change the KIND of thing the controller does, or only give it more room?
    _pc1 = _fwh_U[:, 0] * _fwh_S[0]
    _free_all = _fwh_meta["free"].to_numpy()
    _order = np.lexsort((_pc1, _free_all))          # last key is the primary one
    _D = _fwh_D[_order]
    _free = _free_all[_order]
    _tot = _fwh_meta["total"].to_numpy()[_order]

    FWH_BIN = 40
    _nb = len(_D) // FWH_BIN
    _Db = _D[:_nb * FWH_BIN].reshape(_nb, FWH_BIN, _D.shape[1]).mean(1)
    _freeb = np.median(_free[:_nb * FWH_BIN].reshape(_nb, FWH_BIN), axis=1)
    _totb = _tot[:_nb * FWH_BIN].reshape(_nb, FWH_BIN).mean(1)
    # Block edges and centres, so each arm can be named on the axis instead of being
    # colour-coded with a legend to decode.
    _edges = [0] + [int(np.argmax(_freeb >= _f)) for _f in FWH_FREE[1:]] + [_nb]
    _centres = [(_edges[_i] + _edges[_i + 1]) / 2 for _i in range(len(FWH_FREE))]
    _splits = _edges[1:-1]

    fig_fwh = plt.figure(figsize=(W_TEXT, 4.0))
    _gfh = fig_fwh.add_gridspec(1, 2, width_ratios=[1.0, 0.30], wspace=0.07,
                                left=0.175, right=0.845, top=0.855, bottom=0.215)

    _axh = fig_fwh.add_subplot(_gfh[0, 0])
    _im = _axh.imshow(_Db, aspect="auto", cmap="viridis", interpolation="nearest",
                      extent=[-20, 0, _nb, 0], vmin=0, vmax=np.percentile(_Db, 99.5))
    for _s in _splits:
        _axh.axhline(_s, color="white", lw=1.2)
    _axh.set_yticks(_centres, [f"{_f} free min" for _f in FWH_FREE], fontsize=7)
    _axh.tick_params(axis="y", length=0)
    _axh.set_xticks([-20, -15, -10, -5, 0])
    _axh.tick_params(axis="x", labelsize=7)
    _axh.set_xlabel("minutes before the demand opens", fontsize=8)
    _axh.set_ylabel(f"{len(_D):,} windows", fontsize=7.5, labelpad=8)
    _axh.set_title("Every window, by arm and then by the shape of its stimulation",
                   loc="left", fontweight="bold", fontsize=9)

    _axt = fig_fwh.add_subplot(_gfh[0, 1])
    _axt.plot(_totb, np.arange(_nb), color=INK, lw=1.4)
    for _s in _splits:
        _axt.axhline(_s, color=GRID, lw=1.0)
    _axt.set_ylim(_nb, 0)
    _axt.set_xlim(0, 1900)
    _axt.set_yticks([])
    _axt.set_xticks([0, 1500])
    _axt.tick_params(labelsize=6.5)
    _axt.set_xlabel("total light (ms)", fontsize=7)
    _axt.xaxis.grid(True, color=GRID, lw=0.6)
    _axt.set_axisbelow(True)

    _cb = fig_fwh.colorbar(_im, ax=_axt, fraction=0.13, pad=0.55)
    _cb.ax.tick_params(labelsize=6.5)
    _cb.set_label("light in the minute (ms)", fontsize=6.2, color=MUTED,
                  rotation=270, labelpad=9)
    _cb.outline.set_visible(False)

    save_fig(fig_fwh, "freewindow-heatmap")
    fig_fwh

    return BoundaryNorm, ListedColormap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Every window, ordered by the shape of its stimulation.** Both runs, all four arms.
    Rows are grouped by arm, then ordered within the arm by the first shape component,
    computed once over every window; each row is the mean of 40 neighbouring windows. Kinds
    of stimulation would appear as bands with edges between them.
    """)
    return


@app.cell(hide_code=True)
def _(
    BoundaryNorm,
    FW_RUNS,
    GRID,
    INK,
    ListedColormap,
    MUTED,
    W_TEXT,
    fw_vec,
    fw_win,
    materials_path,
    np,
    pl,
    plt,
    save_fig,
):
    # --- Does the stimulation depend on the demand that is coming? ----------------
    # A copy of `freewindow-heatmap` with one thing changed: the strip on the left is
    # the DEMAND the block was preparing for, not the arm. Rows are still grouped by
    # arm and ordered within it by shape, so if the controller prepared differently
    # for different objectives, that would show as colour blocks lining up with the
    # shape ordering inside each arm.
    #
    # EIGHT CATEGORIES, NOT FOUR. The two runs do not share a demand vocabulary, so a
    # four-colour scale would make one colour stand for two unrelated patterns. v23's
    # four are named objectives that all open at the same level and differ in what
    # they ask for after arrival; v21's four differ in how long the demand holds and
    # whether it peaks to 1.15.
    #
    #   v23   hold  arrive then stay · release  then let go · climb  then find more
    #         re-arrive  arrive, let go, arrive again
    #   v21   10, 15 or 25 min at 1.05, or a block that peaks to 1.15
    #
    # THE TEST IS AT THE BLOCK, NOT THE WINDOW. Every window in a block shares that
    # block, and block position carries the run's drift, so treating windows as
    # independent inflates any demand effect enormously. Each demand appears in three
    # blocks, which is the replication the design actually has.
    from scipy.stats import f as f_dist

    FWD_V21 = ["10 min", "15 min", "25 min", "peaks to 1.15"]
    FWD_V23 = ["hold", "release", "climb", "re-arrive"]
    FWD_COL = [plt.cm.tab10(_i) for _i in (0, 1, 2, 3)] + \
              [plt.cm.tab10(_i) for _i in (4, 5, 6, 8)]
    FWD_FREE = [0, 4, 10, 20]
    FWD_BIN = 40


    def _fwd_v21_class(dur, peak):
        if peak > 1.10:
            return 3
        return 0 if dur <= 12 else (1 if dur <= 20 else 2)


    _t21 = pl.read_parquet(materials_path("tracks_v21.parquet"))
    _r21 = _t21.group_by("timestep").agg(pl.col("r_t").first()).sort("timestep")
    _v21v, _v21t = _r21["r_t"].to_numpy(), _r21["timestep"].to_numpy()
    _map21 = {}
    for _bi, _b in enumerate(range(100, 601, 50)):
        _seg = _v21v[(_v21t > _b) & (_v21t <= _b + 49)]
        _hi = _seg > 0.86
        _map21[_bi] = _fwd_v21_class(int(_hi.sum()),
                                     float(_seg[_hi].max()) if _hi.any() else 0.0)
    _map23 = {_i: 4 + "ABCD".index(_l.split("_")[2]) for _i, _l in enumerate(
        pl.read_parquet(materials_path("tracks_v23.parquet"))
          .filter(pl.col("phase_label").str.starts_with("demand_"))
          .group_by("phase_label").agg(pl.col("timestep").min().alias("s"))
          .sort("s")["phase_label"].to_list())}

    _fwd_mask = (fw_win["total"] > 0).to_numpy()
    _fwd_meta = fw_win.filter(pl.Series(_fwd_mask))
    _fwd_D0 = fw_vec[_fwd_mask]
    _fwd_X = _fwd_D0 / _fwd_D0.sum(1, keepdims=True)
    _fwd_U, _fwd_S, _ = np.linalg.svd(_fwd_X - _fwd_X.mean(0), full_matrices=False)
    _fwd_pc1 = _fwd_U[:, 0] * _fwd_S[0]
    _fwd_freeall = _fwd_meta["free"].to_numpy()
    _fwd_order = np.lexsort((_fwd_pc1, _fwd_freeall))
    _fwd_D = _fwd_D0[_fwd_order]
    _fwd_free = _fwd_freeall[_fwd_order]
    _fwd_tot = _fwd_meta["total"].to_numpy()[_fwd_order]
    _fwd_nb = len(_fwd_D) // FWD_BIN
    _Db = _fwd_D[:_fwd_nb * FWD_BIN].reshape(_fwd_nb, FWD_BIN, _fwd_D.shape[1]).mean(1)
    _freeb = np.median(_fwd_free[:_fwd_nb * FWD_BIN].reshape(_fwd_nb, FWD_BIN), axis=1)
    _totb = _fwd_tot[:_fwd_nb * FWD_BIN].reshape(_fwd_nb, FWD_BIN).mean(1)
    _edges = [0] + [int(np.argmax(_freeb >= _f)) for _f in FWD_FREE[1:]] + [_fwd_nb]
    _centres = [(_edges[_i] + _edges[_i + 1]) / 2 for _i in range(len(FWD_FREE))]
    _splits = _edges[1:-1]

    _dem = np.array([(_map21 if _r == "v21" else _map23)[_b] for _r, _b
                     in zip(_fwd_meta["run"].to_list(), _fwd_meta["block"].to_list())])
    _demo = _dem[_fwd_order]
    _demb = np.array([np.bincount(_demo[_i * FWD_BIN:(_i + 1) * FWD_BIN],
                                  minlength=8).argmax() for _i in range(_fwd_nb)])

    # Block-level test, run by run: is the spread between demands larger than the
    # spread from one block to the next within the same demand?
    fwd_F = {}
    for _run in FW_RUNS:
        _k = np.array(_fwd_meta["run"].to_list()) == _run
        _bl = _fwd_meta["block"].to_numpy()[_k]
        _pc = _fwd_pc1[_k]
        _dm = _dem[_k]
        _per = [(int(_dm[_bl == _b][0]), float(_pc[_bl == _b].mean()))
                for _b in np.unique(_bl)]
        _grp = [[_v for _d, _v in _per if _d == _t] for _t in sorted({_d for _d, _ in _per})]
        _grp = [_g for _g in _grp if len(_g) > 1]
        _between = np.var([np.mean(_g) for _g in _grp], ddof=1)
        _within = np.mean([np.var(_g, ddof=1) for _g in _grp])
        _df1, _df2 = len(_grp) - 1, sum(len(_g) - 1 for _g in _grp)
        fwd_F[_run] = (_between / _within, _df1, _df2,
                        float(1 - f_dist.cdf(_between / _within, _df1, _df2)))

    fig_fwd = plt.figure(figsize=(W_TEXT, 4.3))
    _gfd = fig_fwd.add_gridspec(1, 3, width_ratios=[0.05, 1.0, 0.30], wspace=0.07,
                                left=0.135, right=0.845, top=0.855, bottom=0.315)

    _axdm = fig_fwd.add_subplot(_gfd[0, 0])
    _axdm.imshow(_demb[:, None], aspect="auto", cmap=ListedColormap(FWD_COL),
                 norm=BoundaryNorm(np.arange(-0.5, 8.5), 8), interpolation="nearest")
    _axdm.set_xticks([])
    _axdm.set_yticks(_centres, [f"{_f} free" for _f in FWD_FREE], fontsize=6.8)
    _axdm.tick_params(axis="y", length=0)
    _axdm.set_title("demand\nahead", fontsize=6, color=MUTED, linespacing=1.3, pad=3)

    _axhd = fig_fwd.add_subplot(_gfd[0, 1])
    _imd = _axhd.imshow(_Db, aspect="auto", cmap="viridis", interpolation="nearest",
                        extent=[-20, 0, _fwd_nb, 0], vmin=0,
                        vmax=np.percentile(_Db, 99.5))
    for _s in _splits:
        _axhd.axhline(_s, color="white", lw=1.2)
    _axhd.set_yticks([])
    _axhd.set_xticks([-20, -15, -10, -5, 0])
    _axhd.tick_params(labelsize=7)
    _axhd.set_xlabel("minutes before the demand opens", fontsize=8)
    _axhd.set_title("The same windows, coloured by the demand they were preparing for",
                    loc="left", fontweight="bold", fontsize=9)
    _axhd.legend(handles=[plt.Line2D([], [], color=_c, lw=5, label=_l)
                          for _c, _l in zip(FWD_COL,
                                            [f"v21 {_x}" for _x in FWD_V21]
                                            + [f"v23 {_x}" for _x in FWD_V23])],
                 loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
                 frameon=False, fontsize=6.2, handlelength=1.2, columnspacing=1.1,
                 labelspacing=0.35)

    _axtd = fig_fwd.add_subplot(_gfd[0, 2])
    _axtd.plot(_totb, np.arange(_fwd_nb), color=INK, lw=1.4)
    for _s in _splits:
        _axtd.axhline(_s, color=GRID, lw=1.0)
    _axtd.set_ylim(_fwd_nb, 0)
    _axtd.set_xlim(0, 1900)
    _axtd.set_yticks([])
    _axtd.set_xticks([0, 1500])
    _axtd.tick_params(labelsize=6.5)
    _axtd.set_xlabel("total light (ms)", fontsize=7)
    _axtd.xaxis.grid(True, color=GRID, lw=0.6)
    _axtd.set_axisbelow(True)

    _cbd = fig_fwd.colorbar(_imd, ax=_axtd, fraction=0.13, pad=0.55)
    _cbd.ax.tick_params(labelsize=6.5)
    _cbd.set_label("light in the minute (ms)", fontsize=6.2, color=MUTED,
                   rotation=270, labelpad=9)
    _cbd.outline.set_visible(False)

    save_fig(fig_fwd, "freewindow-heatmap-by-demand")
    fig_fwd

    return (fwd_F,)


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- E2, alternative 1: four rungs from the first panel -----------------------
    # The published `feedback-ladder` pools 1a and 1b as "closed loop" until panel (c).
    # That hides the most interesting comparison in the run: in the one block whose
    # demand was reachable, BROADCAST FEEDBACK SCORES LIKE CONSTANT LIGHT, and only
    # per-cell dosing separates. Splitting the arm in panel (a) puts that in front.
    E2A_RUNGS = ["1a per-cell", "1b broadcast", "2 constant 60 ms", "3 dark"]
    E2A_COL = dict(zip(E2A_RUNGS, [SERIES[0], SERIES[2], MUTED, "#a9a7a2"]))
    E2A_SHORT = dict(zip(E2A_RUNGS, ["1a  per-cell", "1b  broadcast",
                                     "2  constant", "3  dark"]))
    E2A_BLOCK = "demand_01_L_H"

    _e2a_arms = (pl.read_parquet(materials_path("tracks_arms.parquet"))
                   .filter(pl.col("run") == "v24").select("fov", "arm_label"))
    e2a = (pl.read_parquet(materials_path("tracks_v24.parquet"))
             .join(_e2a_arms, on="fov", how="left")
             .with_columns(
                 pl.when(pl.col("arm_label") == "population MPC")
                   .then(pl.when((pl.col("particle") // 4) % 2 == 1)
                           .then(pl.lit("1b broadcast")).otherwise(pl.lit("1a per-cell")))
                 .when(pl.col("arm_label") == "open loop, 60 ms")
                   .then(pl.lit("2 constant 60 ms"))
                 .otherwise(pl.lit("3 dark")).alias("rung")))

    _b1 = e2a.filter(pl.col("phase_label") == E2A_BLOCK)
    _e2a_err = {_r: float(np.sqrt(((_s["raw_cnr"] - _s["r_t"]) ** 2).mean()))
                for _k, _s in _b1.group_by("rung") for _r in [_k[0]]}

    # field-level tracking error over everything the loop was scored on
    _cell = (e2a.filter(pl.col("timestep") >= 76)
                .group_by(["rung", "fov", "particle"])
                .agg(((pl.col("raw_cnr") - pl.col("r_t")) ** 2).mean().sqrt().alias("rmse"),
                     pl.col("exposure_ms").mean().alias("ms"), pl.len().alias("n"))
                .filter(pl.col("n") >= 120))
    e2a_fld = (_cell.group_by(["rung", "fov"])
                    .agg(pl.col("rmse").median(), pl.col("ms").mean()).sort("rung", "fov"))

    fig_e2a = plt.figure(figsize=(W_TEXT, 5.4))
    # nested, so the trace and its light strip stay coupled while the ladder below
    # gets a real gap rather than sharing one uniform hspace with them
    _ga = fig_e2a.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.40,
                               left=0.115, right=0.975, top=0.94, bottom=0.085)
    _gtop = _ga[0].subgridspec(2, 1, height_ratios=[1.25, 0.34], hspace=0.12)

    # (a) the one reachable block, with the closed loop split in two
    _axa = fig_e2a.add_subplot(_gtop[0])
    _axl = fig_e2a.add_subplot(_gtop[1], sharex=_axa)
    for _r in E2A_RUNGS:
        _g = (_b1.filter(pl.col("rung") == _r).group_by("timestep")
                 .agg(pl.col("raw_cnr").median().alias("m"),
                      pl.col("exposure_ms").mean().alias("ms")).sort("timestep"))
        _x = _g["timestep"].to_numpy() - _g["timestep"].min()
        _axa.plot(_x, _g["m"], color=E2A_COL[_r], lw=1.8,
                  label=f"{E2A_SHORT[_r]}   {_e2a_err[_r]:.3f}")
        _axl.plot(_x, _g["ms"], color=E2A_COL[_r], lw=1.2)
    _axa.axhline(1.20, color=INK, lw=1.4, ls="--", zorder=6)
    _axa.text(1, 1.203, "demand 1.20", fontsize=6.5, color=INK, va="bottom")
    _axa.set_ylabel("CNR (median)", fontsize=8)
    _axa.tick_params(labelsize=7, labelbottom=False)
    _axa.set_title("a  The one block the demand was reachable", loc="left",
                   fontweight="bold", fontsize=9)
    _axa.legend(frameon=False, fontsize=6.6, ncol=2, loc="center left",
                handlelength=1.4, columnspacing=1.4, labelspacing=0.3,
                borderaxespad=0.6,
                title="rung and its error in this block", title_fontsize=6.4)
    _axa.yaxis.grid(True, color=GRID, lw=0.6)
    _axa.set_axisbelow(True)

    _axl.set_ylabel("light\n(ms/frame)", fontsize=7, linespacing=1.3)
    _axl.set_xlabel("minutes into the block", fontsize=8)
    _axl.set_ylim(0, 90)
    _axl.tick_params(labelsize=6.5)

    # (b) the four-rung ladder, with 1a and 1b paired inside their shared dishes
    _axb = fig_e2a.add_subplot(_ga[1])
    _ypos = {_r: len(E2A_RUNGS) - 1 - _i for _i, _r in enumerate(E2A_RUNGS)}
    for _r in E2A_RUNGS:
        _s = e2a_fld.filter(pl.col("rung") == _r)
        _y = _ypos[_r]
        _axb.plot(_s["rmse"], [_y] * _s.height, "o", ms=6, color=E2A_COL[_r], zorder=4)
        _axb.plot([float(_s["rmse"].median())] * 2, [_y - 0.22, _y + 0.22],
                  color=E2A_COL[_r], lw=2.4, zorder=5)
    # the pairing: 1a and 1b are two halves of one dish, so join them field by field
    _pa = e2a_fld.filter(pl.col("rung") == "1a per-cell")
    _pb = e2a_fld.filter(pl.col("rung") == "1b broadcast")
    for _f in _pa["fov"]:
        _xa = float(_pa.filter(pl.col("fov") == _f)["rmse"][0])
        _xb = float(_pb.filter(pl.col("fov") == _f)["rmse"][0])
        _axb.plot([_xa, _xb], [_ypos["1a per-cell"], _ypos["1b broadcast"]],
                  color=MUTED, lw=0.9, alpha=0.7, zorder=2)
    _axb.set_yticks(list(_ypos.values()),
                    [f"{E2A_SHORT[_r]}\n"
                     f"{float(e2a_fld.filter(pl.col('rung') == _r)['ms'].mean()):.0f} ms"
                     for _r in E2A_RUNGS], fontsize=7)
    _axb.tick_params(axis="y", length=0)
    _axb.set_ylim(-0.6, len(E2A_RUNGS) - 0.4)
    _axb.set_xlabel("tracking error (CNR), field median", fontsize=8)
    _axb.tick_params(axis="x", labelsize=7)
    _axb.set_title("b  The ladder, by field, with the two halves of each dish joined",
                   loc="left", fontweight="bold", fontsize=9)
    _axb.xaxis.grid(True, color=GRID, lw=0.6)
    _axb.set_axisbelow(True)

    save_fig(fig_e2a, "feedback-ladder-alt1")
    fig_e2a

    return E2A_COL, E2A_SHORT, e2a


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **E2, alternative 1: the closed loop split from the first panel.** (a) The one block whose
    demand the cells could reach, with each rung's median CNR and the light it spent; the
    number beside each rung in the legend is its pooled error over that block. (b) The same
    four rungs at the level inference is done, the field, with the two halves of each dish
    joined: per-cell and broadcast cells sit in the same four dishes, so the comparison is
    paired and every field-level confound cancels inside the pair. Row labels carry the mean
    light each rung spent over the whole scored window.
    """)
    return


@app.cell(hide_code=True)
def _(
    E2A_COL,
    E2A_SHORT,
    GRID,
    INK,
    MUTED,
    W_TEXT,
    e2a,
    np,
    pl,
    plt,
    save_fig,
):
    # --- E2, alternative 2: the ladder as one axis, dose alongside ----------------
    # A different way of introducing the split early. Instead of a trace panel and a
    # separate individuation panel, every cell in the run is placed on one axis of
    # tracking error, four rungs deep, with the light each rung spent drawn beside it.
    # The 1a-1b comparison is then the top two rows rather than an afterthought, and
    # the dose panel makes the confound visible at the same moment as the result.
    E2B_ORDER = ["1a per-cell", "1b broadcast", "2 constant 60 ms", "3 dark"]

    _e2b_cell = (e2a.filter(pl.col("timestep") >= 76)
                   .group_by(["rung", "fov", "particle"])
                   .agg(((pl.col("raw_cnr") - pl.col("r_t")) ** 2).mean().sqrt().alias("rmse"),
                        pl.col("exposure_ms").mean().alias("ms"), pl.len().alias("n"))
                   .filter(pl.col("n") >= 120))

    fig_e2b = plt.figure(figsize=(W_TEXT, 3.5))
    _gb = fig_e2b.add_gridspec(1, 2, width_ratios=[1.0, 0.42], wspace=0.08,
                               left=0.155, right=0.975, top=0.86, bottom=0.24)
    _y = {_r: len(E2B_ORDER) - 1 - _i for _i, _r in enumerate(E2B_ORDER)}
    _rng = np.random.default_rng(0)

    # left: every cell, with the field medians on top of them
    _axe = fig_e2b.add_subplot(_gb[0, 0])
    for _r in E2B_ORDER:
        _s = _e2b_cell.filter(pl.col("rung") == _r)
        _v = _s["rmse"].to_numpy()
        _v = _v[_v <= 0.95]
        _axe.plot(_v, _y[_r] + (_rng.random(len(_v)) - 0.5) * 0.42, "o", ms=1.8,
                  alpha=0.18, color=E2A_COL[_r], mec="none")
        _fm = (_s.group_by("fov").agg(pl.col("rmse").median()).sort("fov"))["rmse"].to_numpy()
        _axe.plot(_fm, [_y[_r]] * len(_fm), "o", ms=6, color=E2A_COL[_r],
                  mfc="white", mew=1.8, zorder=5)
        _axe.plot([np.median(_fm)] * 2, [_y[_r] - 0.30, _y[_r] + 0.30], color=INK,
                  lw=2.2, zorder=6)

    # the two halves of one dish, joined
    _pa = (_e2b_cell.filter(pl.col("rung") == "1a per-cell").group_by("fov")
           .agg(pl.col("rmse").median()).sort("fov"))
    _pb = (_e2b_cell.filter(pl.col("rung") == "1b broadcast").group_by("fov")
           .agg(pl.col("rmse").median()).sort("fov"))
    for _xa, _xb in zip(_pa["rmse"], _pb["rmse"]):
        # dotted and muted: these say "same dish", they are not a measurement
        _axe.plot([_xa, _xb], [_y["1a per-cell"], _y["1b broadcast"]],
                  color=MUTED, lw=0.9, ls=":", alpha=0.65, zorder=4)

    _axe.set_yticks(list(_y.values()), [E2A_SHORT[_r] for _r in E2B_ORDER], fontsize=8)
    _axe.tick_params(axis="y", length=0)
    _axe.set_ylim(-0.65, len(E2B_ORDER) - 0.35)
    _axe.set_xlim(0, 0.95)
    _axe.set_xlabel("tracking error (CNR)", fontsize=8)
    _axe.tick_params(axis="x", labelsize=7)
    _axe.set_title("a  Every cell, every rung", loc="left", fontweight="bold", fontsize=9)
    _axe.xaxis.grid(True, color=GRID, lw=0.6)
    _axe.set_axisbelow(True)
    # right: what each rung spent to get there
    _axm = fig_e2b.add_subplot(_gb[0, 1], sharey=_axe)
    for _r in E2B_ORDER:
        _d = float(_e2b_cell.filter(pl.col("rung") == _r)["ms"].mean())
        _axm.barh([_y[_r]], [_d], height=0.55, color=E2A_COL[_r])
        _axm.text(_d + 4, _y[_r], f"{_d:.0f}", va="center", fontsize=7, color=INK)
    _axm.axvline(60, color=INK, lw=1.0, ls="--", zorder=5)
    _axm.tick_params(labelleft=False, axis="y", length=0)
    _axm.set_xlim(0, 165)
    _axm.set_xticks([0, 60, 120])
    _axm.tick_params(axis="x", labelsize=7)
    _axm.set_xlabel("light spent\n(ms/frame)", fontsize=8, linespacing=1.3)
    _axm.set_title("b  On what light", loc="left", fontweight="bold", fontsize=9)
    _axm.xaxis.grid(True, color=GRID, lw=0.6)
    _axm.set_axisbelow(True)
    _axm.text(62, len(E2B_ORDER) - 1 + 0.42, "arm 2's setting", fontsize=6,
              color=MUTED, va="bottom")

    save_fig(fig_e2b, "feedback-ladder-alt2")
    fig_e2b

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **E2, alternative 2: one axis, four rungs, and the light beside it.** (a) Every scored
    cell placed on tracking error, jittered within its rung; open rings are field medians and
    the black bar is the rung's median of those. Thin lines join the two halves of one dish,
    which is the paired 1a against 1b contrast. (b) What each rung spent to get there, against
    the 60 ms arm 2 was set to from v23's closed-loop means. The closed-loop rungs spent
    roughly twice that, so the step from rung 2 to rung 1 is not at a matched dose.
    """)
    return


@app.cell(hide_code=True)
def _(fwd_F, mo):
    mo.md(f"""
    **The same windows, coloured by the demand they were preparing for.** Scored at the
    block, which is the replication the design has: each demand appears in three blocks. The
    spread between demands is no larger than the spread between blocks of one demand —
    v21 F = {fwd_F['v21'][0]:.2f}, p = {fwd_F['v21'][3]:.2f} · v23 F = {fwd_F['v23'][0]:.2f},
    p = {fwd_F['v23'][3]:.2f}. Rows and binning as in the previous figure.
    """)
    return


@app.cell(hide_code=True)
def _(OSC_ARMS, OSC_PERIODS, np, osc_arr, osc_df, osc_feat, pl):
    # Everything below is read off the controller's own record: the dose it commanded and
    # `plan_cost`. `settle` frames are excluded throughout - the reference is flat there and
    # the loop has not yet been asked for anything.
    MPC_GROUP = ["fov", "control_frame", "phase_offset_min"]  # a field, a minute, one demand
    MPC_MIN_GROUP = 4          # cells needed before a group can speak to per-cell conditioning
    MPC_HORIZON = 30           # frames the controller planned over, from the startup record
    MPC_KS = (1, 2, 3, 5, 8, 10, 15)
    MPC_SHORT = 0.05           # CNR: "still materially below what was asked for"
    MPC_PLANT_LAG = (6.0, 7.0)  # min: the cells' own tau (~3) plus their dead time (~4), v16

    mpc_df = (
        osc_df.filter(pl.col("segment") != "settle")
        .with_columns(
            pl.col("fov").replace_strict({f: OSC_ARMS[f]["period_min"] for f in OSC_ARMS})
              .cast(pl.Int64).alias("period"),
            (pl.col("raw_cnr") - pl.col("r_t")).alias("e"))
        .sort(["fov", "particle", "control_frame"])
    )


    def mpc_boot(v, f=np.median, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        v = np.asarray(v, float)
        s = np.array([f(rng.choice(v, v.size, replace=True)) for _ in range(n)])
        return float(f(v)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


    # --- a, b: anticipation -------------------------------------------------------------
    # Folded on the same windows as every other v19 figure (cut at the logged `low_hold`
    # onsets), then re-centred so x=0 is the minute the demand steps up in each arm.
    mpc_fold = {}
    mpc_lead_rows = []
    for _P in OSC_PERIODS:
        _m = np.flatnonzero((osc_feat["period"] == _P).to_numpy())
        _cell = np.array(osc_feat["cell"])[_m]
        _U = np.stack([osc_arr["exposure_ms"][i] for i in _m]).astype(np.float64)
        _R = np.stack([osc_arr["r_t"][i] for i in _m]).astype(np.float64)
        _bycell = np.stack([_U[_cell == c].mean(0) for c in np.unique(_cell)])
        _tlow = OSC_ARMS[[f for f in OSC_ARMS if OSC_ARMS[f]["period_min"] == _P][0]]["t_low_min"]
        _x = ((np.arange(_P) - _tlow + _P / 2) % _P) - _P / 2
        _o = np.argsort(_x)
        mpc_fold[_P] = {
            "x": _x[_o],
            "dose": _bycell.mean(0)[_o],
            "lo": np.percentile(_bycell, 25, axis=0)[_o],
            "hi": np.percentile(_bycell, 75, axis=0)[_o],
            "ref": _R.mean(0)[_o],
            "n": len(_bycell),
        }
        # Per-cell lead from the first harmonic: + means the light arrives before the demand.
        _w = np.exp(-2j * np.pi * np.arange(_P) / _P)
        for _c in np.unique(_cell):
            _k = _cell == _c
            _zu, _zr = (_U[_k].mean(0) * _w).sum(), (_R[_k].mean(0) * _w).sum()
            _dphi = (np.angle(_zu) - np.angle(_zr) + np.pi) % (2 * np.pi) - np.pi
            mpc_lead_rows.append({"period": _P, "cell": _c,
                                  "lead_min": float(_dphi / (2 * np.pi) * _P)})
    mpc_lead = pl.DataFrame(mpc_lead_rows)

    # --- c: was the decision conditioned on the individual cell? ------------------------
    # Residuals against the cell's own (field, minute, demand) group. What survives is the
    # part of the dose that cannot be explained by the schedule.
    mpc_res = (
        mpc_df.filter(pl.len().over(MPC_GROUP) >= MPC_MIN_GROUP)
        .with_columns([(pl.col(c) - pl.col(c).mean().over(MPC_GROUP)).alias(f"r_{c}")
                       for c in ("exposure_ms", "e")])
    )
    mpc_cond_rows, mpc_cond_slope = [], {}
    for _P in OSC_PERIODS:
        _k = mpc_res.filter(pl.col("period") == _P)
        _x, _y = _k["r_e"].to_numpy(), _k["r_exposure_ms"].to_numpy()
        mpc_cond_slope[_P] = float(_x @ _y / (_x @ _x))
        _edges = np.percentile(_x, np.linspace(0, 100, 13))
        _b = np.clip(np.searchsorted(_edges, _x, side="right") - 1, 0, len(_edges) - 2)
        for _i in range(len(_edges) - 1):
            _s = _b == _i
            if _s.sum() < 200:
                continue
            mpc_cond_rows.append({"period": _P, "err": float(_x[_s].mean()),
                                  "dose": float(_y[_s].mean()),
                                  "se": float(_y[_s].std(ddof=1) / np.sqrt(_s.sum())),
                                  "n": int(_s.sum())})
    mpc_cond = pl.DataFrame(mpc_cond_rows)

    # Per-cell slopes, so the claim carries an error bar over cells rather than over frames.
    mpc_cond_cells = {}
    for _P in OSC_PERIODS:
        _k = mpc_res.filter(pl.col("period") == _P)
        _key = (_k["fov"] * 100000 + _k["particle"]).to_numpy()
        _x, _y = _k["r_e"].to_numpy(), _k["r_exposure_ms"].to_numpy()
        _sl = []
        for _c in np.unique(_key):
            _s = _key == _c
            if _s.sum() > 20 and (_x[_s] @ _x[_s]) > 0:
                _sl.append(float(_x[_s] @ _y[_s] / (_x[_s] @ _x[_s])))
        mpc_cond_cells[_P] = np.array(_sl)

    # --- d: how often was there anything left to choose? --------------------------------
    mpc_auth = mpc_df.group_by("period").agg(
        (pl.col("exposure_ms") == 0).mean().alias("floor"),
        ((pl.col("exposure_ms") > 0) & (pl.col("exposure_ms") < 150)).mean().alias("interior"),
        (pl.col("exposure_ms") == 150).mean().alias("ceiling"),
        ((pl.col("exposure_ms") == 150) & (pl.col("e") < -MPC_SHORT)).mean().alias("stuck"),
        pl.len().alias("frames")).sort("period")

    # --- e: the plan the optimizer chose against the cost it actually paid ---------------
    mpc_plan = (
        mpc_df.with_columns(
            (pl.col("e") ** 2).rolling_mean(MPC_HORIZON, min_samples=MPC_HORIZON)
            .shift(-(MPC_HORIZON - 1)).over(["fov", "particle"]).alias("realized"),
            pl.col("control_frame").shift(-(MPC_HORIZON - 1)).over(["fov", "particle"]).alias("cf_end"))
        .filter(pl.col("cf_end") == pl.col("control_frame") + MPC_HORIZON - 1)
        .drop_nulls(["plan_cost", "realized"])
    )
    mpc_plan_stats = {}
    for _P in OSC_PERIODS:
        _k = mpc_plan.filter(pl.col("period") == _P)
        _x, _y = _k["plan_cost"].to_numpy(), _k["realized"].to_numpy()
        mpc_plan_stats[_P] = {"rho": float(np.corrcoef(_x, _y)[0, 1]),
                              "slope": float(_x @ _y / (_x @ _x)), "n": len(_x)}
    mpc_plan_bins = []
    for _P in OSC_PERIODS:
        _k = mpc_plan.filter(pl.col("period") == _P)
        _x, _y = _k["plan_cost"].to_numpy(), _k["realized"].to_numpy()
        _edges = np.percentile(_x, np.linspace(2, 98, 11))
        for _i in range(len(_edges) - 1):
            _s = (_x >= _edges[_i]) & (_x < _edges[_i + 1])
            if _s.sum() < 100:
                continue
            mpc_plan_bins.append({"period": _P, "believed": float(np.median(_x[_s])),
                                  "realized": float(np.median(_y[_s])),
                                  "q1": float(np.percentile(_y[_s], 25)),
                                  "q3": float(np.percentile(_y[_s], 75))})
    mpc_plan_binned = pl.DataFrame(mpc_plan_bins)

    # --- f: what did the extra light buy? -----------------------------------------------
    # The same within-group residual regression as panel c, run forwards: response over K
    # frames against the differential dose, holding the differential error constant. Two
    # readings of "the dose" - the single frame commanded now, and the light actually
    # delivered across the window - and a sweep over K, because the answer depends on both.
    def mpc_gain(K, dose):
        d = mpc_df.with_columns(
            (pl.col("exposure_ms") if dose == "now" else
             pl.col("exposure_ms").rolling_mean(K, min_samples=K).shift(-(K - 1))
             .over(["fov", "particle"])).alias("u"),
            pl.col("raw_cnr").shift(-K).over(["fov", "particle"]).alias("cnr_k"),
            pl.col("control_frame").shift(-K).over(["fov", "particle"]).alias("cf_k"))
        d = (d.filter(pl.col("cf_k") == pl.col("control_frame") + K)
             .with_columns((pl.col("cnr_k") - pl.col("raw_cnr")).alias("dy"))
             .filter(pl.len().over(MPC_GROUP) >= MPC_MIN_GROUP))
        r = d.with_columns([(pl.col(c) - pl.col(c).mean().over(MPC_GROUP)).alias(f"r_{c}")
                            for c in ("u", "e", "dy")])
        rows = []
        for P in OSC_PERIODS:
            k = r.filter(pl.col("period") == P).drop_nulls(["r_dy", "r_u", "r_e"])
            X = np.column_stack([k["r_u"].to_numpy(), k["r_e"].to_numpy()])
            y = k["r_dy"].to_numpy()
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ b
            cov = np.linalg.inv(X.T @ X) * (resid @ resid) / (len(y) - 2)
            rows.append({"period": P, "K": K, "dose": dose, "gain100": b[0] * 100,
                         "ci": 1.96 * cov[0, 0] ** 0.5 * 100, "n": len(y)})
        return rows


    mpc_ident = pl.DataFrame([r for K in MPC_KS for d in ("now", "window")
                              for r in mpc_gain(K, d)])

    # --- the numbers, in one table ------------------------------------------------------
    mpc_summary = pl.DataFrame([
        {"quantity": "lead of light over demand (min)", "period": P,
         "value": round(mpc_boot(mpc_lead.filter(pl.col("period") == P)["lead_min"])[0], 2),
         "ci_lo": round(mpc_boot(mpc_lead.filter(pl.col("period") == P)["lead_min"])[1], 2),
         "ci_hi": round(mpc_boot(mpc_lead.filter(pl.col("period") == P)["lead_min"])[2], 2),
         "n": mpc_lead.filter(pl.col("period") == P).height, "unit": "cells"}
        for P in OSC_PERIODS
    ] + [
        {"quantity": "dose per unit CNR shortfall (ms, per cell)", "period": P,
         "value": round(-float(mpc_cond_cells[P].mean()), 1),
         "ci_lo": round(-float(mpc_cond_cells[P].mean())
                        - 1.96 * float(mpc_cond_cells[P].std(ddof=1))
                        / np.sqrt(len(mpc_cond_cells[P])), 1),
         "ci_hi": round(-float(mpc_cond_cells[P].mean())
                        + 1.96 * float(mpc_cond_cells[P].std(ddof=1))
                        / np.sqrt(len(mpc_cond_cells[P])), 1),
         "n": len(mpc_cond_cells[P]), "unit": "cells"}
        for P in OSC_PERIODS
    ] + [
        {"quantity": "frames pinned at a bound", "period": int(r["period"]),
         "value": round(r["floor"] + r["ceiling"], 3), "ci_lo": None, "ci_hi": None,
         "n": r["frames"], "unit": "frames"}
        for r in mpc_auth.iter_rows(named=True)
    ] + [
        {"quantity": "at the ceiling and still short", "period": int(r["period"]),
         "value": round(r["stuck"], 3), "ci_lo": None, "ci_hi": None,
         "n": r["frames"], "unit": "frames"}
        for r in mpc_auth.iter_rows(named=True)
    ] + [
        {"quantity": "plan cost: realized / believed", "period": P,
         "value": round(mpc_plan_stats[P]["slope"], 2),
         "ci_lo": round(mpc_plan_stats[P]["rho"], 3), "ci_hi": None,
         "n": mpc_plan_stats[P]["n"], "unit": "frames (ci_lo column = rank rho)"}
        for P in OSC_PERIODS
    ])
    return MPC_HORIZON, mpc_df, mpc_ident, mpc_plan_binned, mpc_plan_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## plan-cost — supplementary: the plan's own cost, and a confound warning
    """)
    return


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MPC_HORIZON,
    MUTED,
    OSC_PERIODS,
    PERIOD_COLOUR,
    W_TEXT,
    mpc_ident,
    mpc_plan_binned,
    mpc_plan_stats,
    np,
    pl,
    plt,
    save_fig,
):
    # mpc_e and mpc_f moved here when controller-behaviour was cut;
    # this figure is the only thing that draws them.
    def mpc_e(ax):
        """(e) The plan's own cost against the cost that was paid."""
        for P in OSC_PERIODS:
            k = mpc_plan_binned.filter(pl.col("period") == P)
            ax.errorbar(k["believed"], k["realized"],
                        yerr=[k["realized"] - k["q1"], k["q3"] - k["realized"]],
                        color=PERIOD_COLOUR[P], lw=1.6, marker="o", ms=3.2, capsize=2,
                        label=f"{P} min: {mpc_plan_stats[P]['slope']:.2f}x")
        lim = (2e-3, 3e-1)
        ax.plot(lim, lim, color=INK, lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("cost the winning plan predicted for itself")
        ax.set_ylabel(f"cost actually paid over the same\n{MPC_HORIZON} frames")
        ax.set_title("e  It knew what it was buying", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", title="realized / believed",
                  title_fontsize=6.5)
        rho = np.mean([mpc_plan_stats[P]["rho"] for P in OSC_PERIODS])
        ax.text(0.97, 0.03, f"dashed: believed = paid\nrank rho {rho:.2f} pooled over arms",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK)


    def mpc_f(ax):
        """(f) What the per-cell act bought - which this run cannot say."""
        for P in OSC_PERIODS:
            for dose, ls in (("now", "-"), ("window", "--")):
                k = mpc_ident.filter((pl.col("period") == P) & (pl.col("dose") == dose)).sort("K")
                ax.plot(k["K"], k["gain100"], ls=ls, color=PERIOD_COLOUR[P], lw=1.6,
                        marker="o" if dose == "now" else "^", ms=3)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xlabel("minutes over which the response is read")
        ax.set_ylabel("CNR bought per 100 ms\nof differential light")
        ax.set_title("f  And what did that buy? Not answerable here", loc="left",
                     fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(top=0.062)   # headroom, so the note never sits on top of a curve
        k1 = mpc_ident.filter((pl.col("K") == 1) & (pl.col("dose") == "now"))
        ax.plot([], [], "-o", color=MUTED, ms=3, label="dose = the frame commanded")
        ax.plot([], [], "--^", color=MUTED, ms=3, label="dose = light delivered over the window")
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
        ax.text(0.98, 0.98,
                f"at one minute: {k1['gain100'].min():+.3f} to {k1['gain100'].max():+.3f} CNR.\n"
                "Widen the window and it turns negative:\ndose was assigned on state, so the cells\n"
                "given more light are the cells already\nbehind. Only a yoked arm separates the two.",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK,
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.35", lw=0.6))


    # Supplementary to controller-behaviour. Both panels are real but neither carries the
    # controller claim: (a) is a self-knowledge check, (b) is a confound warning whose
    # conclusion is a sentence rather than a result.
    fig_mpc_s = plt.figure(figsize=(W_TEXT, 2.9))
    _gs = fig_mpc_s.add_gridspec(1, 2, wspace=0.42, left=0.115, right=0.965,
                                 top=0.88, bottom=0.19)
    mpc_e(fig_mpc_s.add_subplot(_gs[0, 0]))
    mpc_f(fig_mpc_s.add_subplot(_gs[0, 1]))
    save_fig(fig_mpc_s, "plan-cost")
    fig_mpc_s
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## tracking — how well was it tracked, and by how much of the population?

    controller-behaviour asked what the controller did. This one asks what it achieved, and the answer
    has to be given twice, because the two readings disagree and both are true.

    **Frame by frame**, against every controlled frame of every cell, the loop does not track
    the waveform. The error is compared against three things that make the number mean
    something: the swing it was asked for (0.25 CNR), the measurement noise floor of a single
    frame, and the error a controller would have had by not modulating at all — sitting at the
    mean of the reference and never moving. That last one is Nash-Sutcliffe efficiency, and a
    negative value says the modulation made the instantaneous error worse rather than better.

    **Cycle by cycle**, averaging each cell over its own cycles before scoring, a real
    modulation appears in about half the population. Averaging is not a trick here: one cycle
    of one cell carries a modulation of the same order as its own drift, so the per-cycle
    number is mostly noise, and the cycle average is the level at which the objective was
    posed in the first place.

    The population is then split by what each cell actually achieved, against a null built from
    that cell's own record: its averaged profile re-read at every other phase of the cycle. A
    cell whose demanded swing is no larger than the swings its own drift produces at unrelated
    phases is not evidence of control, however large its amplitude looks. The bands are cuts on
    a continuum, not kinds of cell - response-modality already tested for a bifurcation and did not
    find one - but the cuts are what the sample-size arithmetic for the next run needs.
    """)
    return


@app.cell(hide_code=True)
def _(OSC_FEATURED, OSC_PERIODS, mpc_df, np, osc_arr, osc_feat, pl, spearmanr):
    # Accuracy at two levels: every controlled frame, and each cell's own cycle average.
    TRK_SWING = 0.25            # CNR the objective asks for, high hold minus low hold
    TRK_MIN_FRAMES = 120        # a cell needs this many controlled frames to be scored
    TRK_CLOSE, TRK_PARTIAL = 0.5, 0.2   # share of the demanded swing: the two cuts
    TRK_TIERS = ("close", "partial", "weak", "not distinguishable", "against")
    TRK_TIER_LABEL = {
        "close": "followed it closely\n(>=50% of the swing)",
        "partial": "followed it partly\n(20-50%)",
        "weak": "moved with it, barely\n(<20%)",
        "not distinguishable": "not distinguishable from\nits own drift",
        "against": "moved against it"}

    # --- frame level -------------------------------------------------------------------
    # The measurement floor is read off each cell's own trace: the second difference of a
    # smooth signal is pure noise, and var(x[t] - 2x[t-1] + x[t-2]) = 6 sigma^2.
    trk_frame = (
        mpc_df.sort(["fov", "particle", "control_frame"])
        .with_columns(
            (pl.col("raw_cnr") - 2 * pl.col("raw_cnr").shift(1) + pl.col("raw_cnr").shift(2))
            .over(["fov", "particle"]).alias("dd"),
            (pl.col("control_frame") - pl.col("control_frame").shift(2))
            .over(["fov", "particle"]).alias("gap"))
        .group_by(["period", "fov", "particle"]).agg(
            (pl.col("e") ** 2).mean().sqrt().alias("rmse"),
            pl.col("e").mean().alias("bias"),
            ((pl.col("dd").filter(pl.col("gap") == 2) ** 2).mean() / 6).sqrt().alias("sigma"),
            pl.len().alias("frames"))
        .filter(pl.col("frames") >= TRK_MIN_FRAMES)
    )

    # Nash-Sutcliffe against the reference's own variance: 0 means "no better than never
    # moving", negative means the loop was further off than a flat line would have been.
    trk_arm = (
        mpc_df.group_by("period").agg(
            (pl.col("e") ** 2).mean().alias("mse"),
            pl.col("r_t").var().alias("ref_var"),
            pl.col("e").mean().alias("bias"),
            pl.len().alias("frames"))
        .with_columns((1 - pl.col("mse") / pl.col("ref_var")).alias("nse"),
                      pl.col("mse").sqrt().alias("rmse_pooled")).sort("period")
    )

    # --- cycle level: what each cell achieved, and what its own drift could fake ---------
    trk_rows = []
    for _P in OSC_PERIODS:
        _m = np.flatnonzero((osc_feat["period"] == _P).to_numpy())
        _cell = np.array(osc_feat["cell"])[_m]
        _Y = np.stack([osc_arr["raw_cnr"][i] for i in _m]).astype(np.float64)
        _ref = np.stack([osc_arr["r_t"][i] for i in _m]).astype(np.float64).mean(0)
        _hi, _lo = _ref >= 1.1 - 1e-6, _ref <= 0.85 + 1e-6
        # Every phase of the cycle except the ones that nearly reproduce the true alignment.
        _shifts = [s for s in range(_P) if _P // 8 <= s <= _P - _P // 8]
        _expr = np.array(osc_feat["expr"])[_m]
        _area = np.array(osc_feat["area"])[_m]
        _dose = np.array(osc_feat["dose_mean"])[_m]
        for _c in np.unique(_cell):
            _k = _cell == _c
            _y = _Y[_k].mean(0)
            _amp = _y[_hi].mean() - _y[_lo].mean()
            _null = np.array([np.roll(_y, s)[_hi].mean() - np.roll(_y, s)[_lo].mean()
                              for s in _shifts])
            trk_rows.append({
                "period": _P, "cell": _c, "cycles": int(_k.sum()), "amp": float(_amp),
                "frac": float(_amp / TRK_SWING),
                "rmse_cycle": float(np.sqrt(((_y - _ref) ** 2).mean())),
                "null_hi": float(np.percentile(_null, 95)),
                "null_lo": float(np.percentile(_null, 5)),
                "null_sd": float(_null.std()),
                "expr": float(np.nanmedian(_expr[_k])), "area": float(np.nanmedian(_area[_k])),
                "dose": float(np.nanmedian(_dose[_k]))})

    trk_cell = pl.DataFrame(trk_rows).with_columns(
        pl.when(pl.col("amp") < pl.col("null_lo")).then(pl.lit("against"))
          .when(pl.col("amp") <= pl.col("null_hi")).then(pl.lit("not distinguishable"))
          .when(pl.col("frac") >= TRK_CLOSE).then(pl.lit("close"))
          .when(pl.col("frac") >= TRK_PARTIAL).then(pl.lit("partial"))
          .otherwise(pl.lit("weak")).alias("tier"))

    trk_tiers = pl.DataFrame([
        {"period": P, **{t: float((k["tier"] == t).mean()) for t in TRK_TIERS},
         "cells": k.height}
        for P in OSC_PERIODS for k in [trk_cell.filter(pl.col("period") == P)]])

    # One representative cell per band, from the featured arm: the median achiever of its band.
    trk_ex = {}
    _mF = np.flatnonzero((osc_feat["period"] == OSC_FEATURED).to_numpy())
    _cellF = np.array(osc_feat["cell"])[_mF]
    trk_ex_ref = np.stack([osc_arr["r_t"][i] for i in _mF]).astype(np.float64).mean(0)
    for _t in TRK_TIERS:
        _k = trk_cell.filter((pl.col("period") == OSC_FEATURED) & (pl.col("tier") == _t))
        if _k.height == 0:
            continue
        _pick = _k.sort("frac")[_k.height // 2]
        _sel = _cellF == _pick["cell"][0]
        trk_ex[_t] = {"y": np.stack([osc_arr["raw_cnr"][i] for i in _mF[_sel]]).mean(0),
                      "frac": float(_pick["frac"][0]), "cell": _pick["cell"][0],
                      "n": int(_sel.sum())}

    # --- does anything about a cell say in advance how well it will be steered? ---------
    trk_predictors = pl.DataFrame([
        {"covariate": name, "period": P,
         "rho": float(spearmanr(k[col].to_numpy(), k["frac"].to_numpy()).statistic),
         "p": float(spearmanr(k[col].to_numpy(), k["frac"].to_numpy()).pvalue),
         "n": k.height}
        for name, col in (("optoRTK expression", "expr"), ("nuclear area", "area"),
                          ("light it was given", "dose"))
        for P in OSC_PERIODS
        for k in [trk_cell.filter(pl.col("period") == P).drop_nulls([col, "frac"])]])

    trk_summary = pl.DataFrame(
        [{"period": int(r["period"]), "level": "every frame",
          "rmse": round(r["rmse_pooled"], 3),
          "nse_vs_never_moving": round(r["nse"], 2),
          "median_share_of_swing": None, "n": r["frames"]}
         for r in trk_arm.iter_rows(named=True)]
        + [{"period": P, "level": "cycle average",
            "rmse": round(float(k["rmse_cycle"].median()), 3),
            "nse_vs_never_moving": None,
            "median_share_of_swing": round(float(k["frac"].median()), 2), "n": k.height}
           for P in OSC_PERIODS for k in [trk_cell.filter(pl.col("period") == P)]]
    ).sort(["period", "level"])
    return TRK_CLOSE, TRK_PARTIAL, TRK_SWING, TRK_TIERS, trk_cell, trk_frame


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- Tracking accuracy, across every live run ---------------------------------
    # The earlier version scored only the oscillation sweep. These three quantities
    # survive a change of objective, so the whole campaign goes on one axis:
    #
    #   rmse   root mean square of (CNR - demand) over the frames the loop controlled
    #   floor  that cell's OWN measurement noise, from the second difference of its
    #          trace (for white noise on a smooth signal the second difference has
    #          variance 6 sigma^2). It is the error a perfect controller would still
    #          show, so it is what rmse has to be judged against.
    #   NSE    Nash-Sutcliffe efficiency against the demand. This is the general form
    #          of "how much of the demand did it deliver": it needs a demand that
    #          moves but not a cyclic one, so run-ups and schedules are scored on the
    #          same axis as oscillations. NSE <= 0 means the cell did no better than
    #          sitting still at the average demand.
    #
    # Settle frames are excluded: the reference is flat there and nothing has been
    # asked of the loop yet. v16's open-loop ladder has a constant demand, so its NSE
    # is undefined and it is marked rather than scored.
    trk_all = pl.read_parquet(materials_path("tracking_all.parquet"))

    TRK_SUM = (trk_all.group_by(["run", "arm", "controller"], maintain_order=True)
                      .agg(pl.len().alias("cells"),
                           pl.col("rmse").median().alias("rmse"),
                           pl.col("floor").median().alias("floor"),
                           (pl.col("nse") > 0).mean().alias("beat"),
                           pl.col("nse").is_not_nan().mean().alias("scored"))
                      .sort(["run", "arm"], descending=[True, True]))


    def _trk_colour(ctrl):
        if ctrl == "population_mpc":
            return SERIES[2]
        if ctrl in ("open_loop", "constant_dose"):
            return MUTED
        return SERIES[0]


    fig_track = plt.figure(figsize=(W_TEXT, 4.4))
    _gt2 = fig_track.add_gridspec(1, 2, wspace=0.10, left=0.235, right=0.975,
                                  top=0.90, bottom=0.135, width_ratios=[1.15, 1])

    # (a) how far off, against what the measurement alone would give
    _a = fig_track.add_subplot(_gt2[0, 0])
    _y = np.arange(TRK_SUM.height)
    _col = [_trk_colour(c) for c in TRK_SUM["controller"]]
    _a.barh(_y, TRK_SUM["rmse"], height=0.62, color=_col)
    _a.plot(TRK_SUM["floor"], _y, "|", ms=9, mew=1.8, color=INK, ls="none",
            label="that cell's own noise floor")
    for _yi, _r, _b, _s in zip(_y, TRK_SUM["rmse"], TRK_SUM["beat"], TRK_SUM["scored"]):
        _a.text(_r + 0.008, _yi,
                "demand does not move" if _s < 0.5 else f"{_b:.0%} beat sitting still",
                va="center", fontsize=5.4, color=MUTED)
    _a.set_yticks(_y, [f"{r}  {a}" for r, a in zip(TRK_SUM["run"], TRK_SUM["arm"])],
                  fontsize=5.8)
    _a.tick_params(axis="y", length=0)
    _a.set_ylim(-0.7, TRK_SUM.height - 0.3)
    _a.set_xlim(0, float(TRK_SUM["rmse"].max()) * 1.62)
    _a.set_xlabel("tracking error, RMSE (CNR), median cell")
    _a.set_title("a  How far off", loc="left", fontweight="bold")
    _a.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.0,
              borderaxespad=0.6)
    _a.xaxis.grid(True, color=GRID, lw=0.6)
    _a.set_axisbelow(True)

    # (b) the whole distribution, run by run, against the pooled floor
    _b2 = fig_track.add_subplot(_gt2[0, 1])
    TRK_RUNC = [SERIES[0], SERIES[1], SERIES[2], "#8452a1", "#8c6d3f", "#c94f7c"]
    for _i, _run in enumerate(sorted(set(trk_all["run"]))):
        _v = np.sort(trk_all.filter(pl.col("run") == _run)["rmse"].to_numpy())
        _b2.plot(_v, np.linspace(0, 1, len(_v)), color=TRK_RUNC[_i], lw=1.7,
                 label=f"{_run} (n={len(_v)})")
    _f = np.sort(trk_all["floor"].to_numpy())
    _b2.plot(_f, np.linspace(0, 1, len(_f)), color=INK, lw=1.4, ls=":",
             label="the same cells' noise floor")
    _b2.set_xlim(0, 0.7)
    _b2.set_xlabel("that cell's own tracking error (CNR)")
    _b2.set_ylabel("share of cells at or below")
    _b2.set_title("b  Every cell, every run", loc="left", fontweight="bold")
    _b2.legend(frameon=False, fontsize=5.8, loc="center right", handlelength=1.2,
               labelspacing=0.25, borderaxespad=0.5)
    _b2.yaxis.grid(True, color=GRID, lw=0.6)
    _b2.set_axisbelow(True)
    _b2.text(0.03, 0.02,
             f"median error is {float(trk_all['rmse'].median()) / float(trk_all['floor'].median()):.0f}\u00d7 "
             "the measurement floor",
             transform=_b2.transAxes, va="bottom", fontsize=6, color=INK)

    save_fig(fig_track, "tracking")
    fig_track
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## tracking-controls — tracking against its controls
    """)
    return


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    OSC_PERIODS,
    PERIOD_COLOUR,
    SERIES,
    TRK_CLOSE,
    TRK_PARTIAL,
    TRK_SWING,
    TRK_TIERS,
    W_TEXT,
    np,
    pl,
    plt,
    save_fig,
    trk_cell,
    trk_frame,
):
    # TRK_COL and trk_f were lost when `tracking` was rebuilt to cover every run.
    # The tiers themselves survived -- they are a column of trk_cell -- so only the
    # drawing is reconstructed here. The five tiers are an ORDERED scale from "close"
    # to "against", so they take a graded ramp rather than five unrelated hues.
    # TRK_TIERS already exists in the data cell and is reused.
    TRK_COL = {"close": SERIES[2], "partial": "#8fce9c", "weak": GRID,
               "not distinguishable": MUTED, "against": SERIES[1]}


    def trk_f(ax):
        """(c) Every cell sorted into how much of the demand it actually delivered."""
        _x = np.arange(len(OSC_PERIODS))
        _bot = np.zeros(len(OSC_PERIODS))
        for _t in TRK_TIERS:
            _share = np.array([
                float((trk_cell.filter(pl.col("period") == P)["tier"] == _t).mean())
                for P in OSC_PERIODS])
            ax.bar(_x, _share, 0.62, bottom=_bot, color=TRK_COL[_t], label=_t)
            for _xi, _s, _b in zip(_x, _share, _bot):
                if _s > 0.09:
                    ax.text(_xi, _b + _s / 2, f"{_s:.0%}", ha="center", va="center",
                            fontsize=5.8,
                            color="white" if _t in ("close", "against") else INK)
            _bot += _share
        ax.set_xticks(_x, [f"{P} min" for P in OSC_PERIODS], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"], fontsize=6.5)
        ax.set_ylabel("share of cells")
        ax.set_title("c  How each cell ends up classed", loc="left",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=5.6, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=3, handlelength=1.0,
                  columnspacing=0.9)


    # trk_b and trk_c moved here when `tracking` was rebuilt to cover every run;
    # this supplementary is the only thing that draws them, and they are specific to
    # the oscillation sweep (they are indexed by OSC_PERIODS).
    def trk_b(ax):
        """(b) Per cell, and against that cell's own noise: the gap is not measurement."""
        for P in OSC_PERIODS:
            k = trk_frame.filter(pl.col("period") == P)
            v = np.sort(k["rmse"].to_numpy())
            ax.plot(v, np.linspace(0, 1, len(v)), color=PERIOD_COLOUR[P], lw=1.8,
                    label=f"{P} min  (n={len(v)})")
        s = np.sort(trk_frame["sigma"].to_numpy())
        ax.plot(s, np.linspace(0, 1, len(s)), color=MUTED, lw=1.6, ls=":",
                label="the same cells' noise floor")
        ax.axvline(TRK_SWING, color=INK, lw=1.0, ls="--")
        ax.set_xlim(0, 0.4)
        ax.set_xlabel("that cell's own tracking error (CNR)")
        ax.set_ylabel("share of cells at or below")
        ax.set_title("a  Every cell is far outside its noise", loc="left",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax.text(TRK_SWING - 0.008, 0.56, "demanded swing", rotation=90, ha="right",
                va="bottom", fontsize=6.5, color=INK)


    def trk_c(ax):
        """(c) The population, ranked by how much of the demand it actually delivered."""
        for P in OSC_PERIODS:
            k = trk_cell.filter(pl.col("period") == P)
            v = np.sort(k["frac"].to_numpy())
            ax.plot(v, np.linspace(0, 1, len(v)), color=PERIOD_COLOUR[P], lw=2.0,
                    label=f"{P} min: median {np.median(v):.2f}")
        band = float(trk_cell["null_hi"].median()) / TRK_SWING
        ax.axvspan(-band, band, color=MUTED, alpha=0.16, lw=0)
        ax.axvline(TRK_CLOSE, color=INK, lw=0.9, ls="--")
        ax.axvline(TRK_PARTIAL, color=INK, lw=0.9, ls=":")
        ax.set_xlim(-0.5, 2.0)
        ax.set_xlabel("swing achieved, as a share of the swing demanded")
        ax.set_ylabel("share of cells at or below")
        ax.set_title("b  How much of the demand each cell delivered", loc="left",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.text(0.98, 0.03,
                "grey: the swing the median cell's own drift\nfakes at unrelated phases"
                " - half the demand.\nEach cell is judged against its own null.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
                color=INK,
                bbox=dict(facecolor="white", edgecolor=GRID,
                          boxstyle="round,pad=0.3", lw=0.6))


    # Supplementary to tracking. All three defend the main claim rather than make
    # it: (a) rules out measurement noise, (b) shows the drift null the bands are cut
    # against, (c) reports that nothing about a cell predicts steerability — and that
    # the apparent light effect runs backwards, since dose is assigned on shortfall.
    fig_trk_s = plt.figure(figsize=(W_TEXT, 3.3))
    _gts = fig_trk_s.add_gridspec(1, 3, wspace=0.46, left=0.095, right=0.975,
                                  top=0.86, bottom=0.20)
    trk_b(fig_trk_s.add_subplot(_gts[0, 0]))
    trk_c(fig_trk_s.add_subplot(_gts[0, 1]))
    trk_f(fig_trk_s.add_subplot(_gts[0, 2]))
    save_fig(fig_trk_s, "tracking-controls")
    fig_trk_s
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## free-window-design — the free run-up: buying the controller room to have a strategy

    **A design schematic for a planned run, not a result.** No data appears in it. The block
    geometry and the arms are read out of `policies/policy_8fov_free_runup.toml`; the
    trajectories are the measured first-order description of the cells (dead time 4 min, rise
    constant 4 min, and v16's dose-to-steady-state ladder) driven by hand-chosen light, drawn
    to show what the design makes possible rather than what any controller did.

    Every live run so far graded the controller at every frame. That is the right way to score
    tracking, and it is the wrong way to find out what the controller *would* do, because a
    plan that leaves the target — even briefly, even usefully — is charged for leaving it. The
    question this run asks is what happens when some of that charge is lifted.

    The instrument is a **hole in the scoring**, not a change to the demand. Each 50-minute
    block is 25 minutes of run-up at an anchor of 0.85 followed by 25 minutes of demand at
    1.05, and the demand half is graded identically in all eight fields. What differs is how
    many of the last minutes of the run-up are left unscored: **0, 4, 10 or 20**, two fields
    each. Inside that window the controller can do anything at all and pay nothing for it, so
    long as it arrives.

    Two things follow, and they are the whole design.

    **The window has to be longer than the cells are slow, or it buys nothing.** Light takes
    3-5 minutes to act. A 4-minute window is spent before the first photon of it has any
    effect, so arm 2 is free on paper and constrained in practice — which is why it is in the
    run: it separates "freedom helped" from "the number of unscored frames helped".

    **Given real room, many different strategies satisfy the same demand equally well.** Panel
    c draws three that arrive within 0.01 CNR of each other while differing two-fold in total
    light and by 0.23 CNR in how high they go. A graded objective cannot tell them apart and
    never gets to see them; an unscored window is what lets the controller pick one, and which
    one it picks is the measurement.

    This is also why the dose penalty is not zero. With nothing separating equally-good plans
    the choice among them falls to sampling noise inside CEM and the run-up would not repeat
    block to block — and "does it repeat" is one of the things being asked.
    """)
    return


@app.cell(hide_code=True)
def _(REPO_ROOT, SERIES, np, pl):
    import tomllib

    # The arms are read from the policy file rather than restated, so this figure cannot
    # drift away from the run it describes.
    FRP_POLICY = REPO_ROOT / "policies" / "policy_8fov_free_runup.toml"
    FRP_BLOCK, FRP_RUNUP = 50, 25       # min: one block, and the run-up half of it
    FRP_ANCHOR, FRP_DEMAND = 0.85, 1.05
    FRP_DEAD, FRP_TAU = 4, 4.0          # min: measured 2026-08-19 from v16
    FRP_LADDER = ([0, 85, 150, 300], [0.745, 1.006, 1.123, 1.349])   # ms -> steady CNR

    with open(FRP_POLICY, "rb") as _f:
        frp_cfg = tomllib.load(_f)
    FRP_LEVELS = frp_cfg["default"]["levels_ms"]
    FRP_HORIZON = frp_cfg["default"]["control_horizon"]
    FRP_LAMBDA_DOSE = frp_cfg["default"]["lambda_dose"]
    _settle = int(frp_cfg["default"]["kernel"]["mask"]["spans"][0][1])   # the settle span ends where block 1 begins


    def frp_free_minutes(kernel):
        """Minutes at the end of the run-up that carry no score."""
        if kernel == "l2":
            return 0
        scored = np.zeros(FRP_RUNUP, bool)
        for _a, _b in kernel["mask"]["spans"]:
            _lo, _hi = max(_a, _settle), min(_b, _settle + FRP_RUNUP)
            if _hi > _lo:
                scored[_lo - _settle:_hi - _settle] = True
        return int((~scored).sum())


    frp_arms = {}
    for _fov, _spec in frp_cfg["fov"].items():
        _a = int(_spec["arm"])
        frp_arms.setdefault(_a, {"free": frp_free_minutes(_spec["kernel"]), "fovs": []})
        frp_arms[_a]["fovs"].append(int(_fov))
    for _a in frp_arms:
        frp_arms[_a]["fovs"].sort()
    FRP_ARM_ORDER = sorted(frp_arms, key=lambda a: frp_arms[a]["free"])


    def frp_ss(ms):
        """Steady-state CNR for a sustained dose, from the v16 ladder."""
        return np.interp(ms, *FRP_LADDER)


    def frp_sim(dose, n=None, x0=FRP_ANCHOR):
        """First-order response with dead time: what the cells do, not what a model predicts."""
        n = n or len(dose)
        x, out = x0, []
        for t in range(n):
            out.append(x)
            x = x + (frp_ss(dose[t - FRP_DEAD] if t - FRP_DEAD >= 0 else 0.0) - x) / FRP_TAU
        return np.array(out)


    # --- b: what a free window of a given length can buy --------------------------------
    def frp_reach(free_min, ms=300):
        """Best CNR reachable at the demand, driving flat out for the whole free window."""
        u = np.zeros(FRP_RUNUP + FRP_DEAD)
        u[FRP_RUNUP - free_min:FRP_RUNUP] = ms
        return float(frp_sim(u, n=FRP_RUNUP + 1)[FRP_RUNUP])


    frp_reach_curve = np.array([frp_reach(f) for f in range(FRP_RUNUP + 1)])
    # The run-up needs whichever of these the cells are actually offering on the day: the
    # ladder above is v16 early in a run, the policy file's 8.7 min is v20's reach at hour 12.
    FRP_LEAD_FAST = int(np.argmax(frp_reach_curve >= FRP_DEMAND))
    FRP_LEAD_SLOW = 8.7

    # --- c: strategies the graded objective cannot tell apart --------------------------
    # The target at the moment the window shuts is NOT the same in every pattern: `step`
    # opens at 1.05, the other three open at the anchor. So the window affords different
    # strategies depending on which demand follows it, and an excursion that has to unwind
    # needs more room than one that can arrive high.
    FRP_STRATEGIES = [
        ("drive late, arrive on the wire", [(15, 25, 150)], SERIES[0], FRP_DEMAND, "step"),
        ("overshoot early, coast into it", [(11, 19, 300)], SERIES[1], FRP_DEMAND, "step"),
        ("excurse and unwind by the time it shuts", [(11, 16, 300)], SERIES[2], FRP_ANCHOR,
         "ramp / peak / delayed"),
    ]
    frp_traces = []
    for _name, _spans, _col, _target, _serves in FRP_STRATEGIES:
        _u = np.zeros(FRP_BLOCK)
        for _a, _b, _v in _spans:
            _u[_a:_b] = _v
        _x = frp_sim(_u, n=FRP_RUNUP + 1)
        frp_traces.append({"name": _name, "dose": _u, "cnr": _x, "colour": _col,
                           "target": _target, "serves": _serves,
                           "arrival": float(_x[FRP_RUNUP]), "peak": float(_x.max()),
                           "light": float(_u[:FRP_RUNUP].sum()),
                           "needs": int(FRP_RUNUP - min(a for a, _, _ in _spans))})

    # --- d: the four demands, and the counterbalanced order they run in ------------------
    FRP_ORDER = ["step", "ramp", "peak", "delayed", "ramp", "peak",
                 "delayed", "step", "peak", "delayed", "step", "ramp"]
    # Only `step` opens at the demand. The other three open at the ANCHOR, and those
    # frames are scored, so on nine of the twelve blocks the window can only be used by
    # an excursion that has unwound before it shuts.
    FRP_PATTERN_NOTE = {
        "step": "opens at 1.05 the minute the window shuts:\npre-positioning pays directly",
        "ramp": "opens at the anchor and climbs for 14 min:\nposition buys nothing, only state can",
        "peak": "5 scored anchor min, then 5 min at 1.05 - and\nthe dead time lets it prepare"
                " inside those five",
        "delayed": "10 scored anchor min first: the window is 10 min\nfrom the event. The control.",
    }


    def frp_reference(pattern):
        """The demand half of a block, minute by minute, as the policy file writes it."""
        r = np.full(FRP_BLOCK, FRP_ANCHOR)
        if pattern == "step":
            r[25:50] = FRP_DEMAND
        elif pattern == "ramp":
            r[25:39] = np.linspace(FRP_ANCHOR, FRP_DEMAND, 14)
            r[39:50] = FRP_DEMAND
        elif pattern == "peak":
            r[30:35] = FRP_DEMAND
        elif pattern == "delayed":
            r[35:50] = FRP_DEMAND
        return r


    pl.DataFrame([
        {"arm": a, "free_min": frp_arms[a]["free"], "fovs": str(frp_arms[a]["fovs"]),
         "clears_the_lead": frp_arms[a]["free"] >= FRP_LEAD_FAST,
         "best_reachable_at_the_demand": round(frp_reach(frp_arms[a]["free"]), 3)}
        for a in FRP_ARM_ORDER])
    return (
        FRP_ANCHOR,
        FRP_ARM_ORDER,
        FRP_BLOCK,
        FRP_DEMAND,
        FRP_RUNUP,
        frp_arms,
        frp_sim,
    )


@app.cell(hide_code=True)
def _(
    FRP_ANCHOR,
    FRP_ARM_ORDER,
    FRP_BLOCK,
    FRP_DEMAND,
    FRP_RUNUP,
    frp_arms,
    frp_sim,
    np,
    pl,
):
    # PROPOSED demand patterns: every block opens at the demand the minute the free window
    # shuts, so the run-up is load-bearing in all twelve blocks. What differs is what the
    # controller has to do AFTER it arrives.
    PROP_CLIMB = 1.15          # v20 reached 1.14-1.20 late in a run; check before fixing it
    PROP_REST = 0.79           # measured resting, 0.745-0.823

    PROP_PATTERNS = {
        "hold": [(25, 50, FRP_DEMAND)],
        "release": [(25, 35, FRP_DEMAND)],
        "climb": [(25, 35, FRP_DEMAND), (35, 50, PROP_CLIMB)],
        "re-arrive": [(25, 30, FRP_DEMAND), (40, 50, FRP_DEMAND)],
    }
    PROP_ASKS = {
        "hold": "arrive, then stay",
        "release": "arrive, then let go",
        "climb": "arrive, then find more",
        "re-arrive": "arrive, let go, arrive again",
    }


    def prop_reference(name):
        r = np.full(FRP_BLOCK, FRP_ANCHOR)
        for _a, _b, _v in PROP_PATTERNS[name]:
            r[_a:_b] = _v
        return r


    def prop_costs(name):
        """Two l2 costs over the 25 scored demand frames, both for a cell that starts at rest.

        `inaction` is what a controller pays for doing nothing at all — the pressure the
        pattern applies. `arrival_only` is what it pays for arriving at 1.05 on time and then
        holding there — the part of the demand that preparation alone cannot buy.
        """
        r = prop_reference(name)[FRP_RUNUP:]
        return {"pattern": name, "asks": PROP_ASKS[name],
                "inaction": float(((r - PROP_REST) ** 2).sum()),
                "arrival_only": float(((r - FRP_DEMAND) ** 2).sum())}


    # Four ways through the free window that all arrive at the demand, chosen to be
    # strategies rather than variations: they differ in WHEN the light goes on and in
    # whether the cell is taken past the level and brought back. The scored objective is
    # indifferent between them, so which one the controller picks is the measurement.
    # A tuple level alternates frame by frame — the ladder has no rung that sits at 1.05
    # (85 ms settles to 1.01, 150 ms to 1.12), so holding the level means dithering.
    PROP_SOLUTIONS = [
        ("wait, then drive to the wire", [(18, 25, 300)]),
        ("up early, then hold the level", [(5, 9, 300), (9, 25, (150, 85))]),
        ("overshoot, fall back, settle", [(5, 13, 300), (15, 25, (150, 85))]),
        ("climb a rung at a time", [(7, 13, 45), (13, 18, 85), (18, 25, 150)]),
    ]
    # A noiseless first-order trace does not look like a cell. The noise here is the
    # measured amount: within-cell sd of successive CNR differences runs 0.014 dark to 0.020
    # at 150 ms (v16, 2026-08-19), which is also the per-frame floor read off v19. Split into
    # a correlated part, so the trace wanders the way a cell does, and an independent part,
    # which is the measurement. Seeded, so the figure is the same every time it is drawn.
    PROP_NOISE_WALK, PROP_NOISE_READ, PROP_NOISE_PHI = 0.012, 0.012, 0.6


    def prop_dose(spans):
        u = np.zeros(FRP_BLOCK)
        for a, b, v in spans:
            for t in range(a, b):
                u[t] = v[(t - a) % len(v)] if isinstance(v, tuple) else v
        return u


    prop_solutions = []
    for _j, (_name, _spans) in enumerate(PROP_SOLUTIONS):
        _x = frp_sim(prop_dose(_spans), n=FRP_RUNUP + 1)
        _rng = np.random.default_rng(100 + _j)
        _w = np.zeros(len(_x))
        for _t in range(1, len(_x)):
            _w[_t] = PROP_NOISE_PHI * _w[_t - 1] + _rng.normal(0, PROP_NOISE_WALK)
        prop_solutions.append({"name": _name, "arrives": float(_x[-1]), "peak": float(_x.max()),
                               # the free minutes a strategy needs is when its light first goes on
                               "needs": FRP_RUNUP - min(_a for _a, _b, _v in _spans),
                               "opens": min(_a for _a, _b, _v in _spans),
                               "cnr": _x + _w + _rng.normal(0, PROP_NOISE_READ, len(_x))})

    # What each window can actually produce, generated per arm rather than filtered from one
    # list: a 20-minute window and a 10-minute one do not afford the same moves. Arm 2 has no
    # entry with any shape to it, because it does not have one — see below.
    PROP_ARM_STRATEGIES = {
        20: PROP_SOLUTIONS,
        10: [("drive from the moment it opens", [(15, 25, 150)]),
             ("wait, then drive hard", [(18, 25, 300)]),
             ("hold just under the level", [(15, 25, (150, 85))])],
        4: [("anything at all — it lands too late", [(21, 25, 300)])],
        0: [],
    }
    prop_arm_traces = {}
    for _a in FRP_ARM_ORDER:
        _free = frp_arms[_a]["free"]
        _open = FRP_RUNUP - _free
        _out = []
        for _k, (_nm, _sp) in enumerate(PROP_ARM_STRATEGIES[_free]):
            _x = frp_sim(prop_dose(_sp), n=FRP_RUNUP + 1)
            _r = np.random.default_rng(200 + 10 * _a + _k)
            _wk = np.zeros(len(_x))
            for _t in range(1, len(_x)):
                _wk[_t] = PROP_NOISE_PHI * _wk[_t - 1] + _r.normal(0, PROP_NOISE_WALK)
            _out.append({"name": _nm,
                         "cnr": (_x + _wk + _r.normal(0, PROP_NOISE_READ, len(_x)))[_open:]})
        prop_arm_traces[_a] = _out

    # Which strategies each arm's window is long enough to express. The arms are read from
    # the policy file in the free-window-design cell; v2 keeps the same four.
    prop_fit = {a: [s for s in prop_solutions if s["needs"] <= frp_arms[a]["free"]]
                for a in FRP_ARM_ORDER}

    prop_table = pl.DataFrame([prop_costs(n) for n in PROP_PATTERNS]).with_columns(
        pl.col("inaction").round(2), pl.col("arrival_only").round(2))
    return (
        PROP_CLIMB,
        PROP_PATTERNS,
        prop_arm_traces,
        prop_reference,
        prop_solutions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## proposed-demands — the demands the runs were asked for
    """)
    return


@app.cell(hide_code=True)
def _(
    FRP_ANCHOR,
    FRP_ARM_ORDER,
    FRP_BLOCK,
    FRP_DEMAND,
    FRP_RUNUP,
    GRID,
    INK,
    MUTED,
    PROP_CLIMB,
    PROP_PATTERNS,
    SERIES,
    W_TEXT,
    frp_arms,
    np,
    plt,
    prop_arm_traces,
    prop_reference,
    save_fig,
):
    # Two panels, because the run is a 4x4 and the previous single panel had to pair
    # one arm with one pattern and then apologise for the pairing in its caption.
    #   (a) how much room the controller is given, at one fixed demand
    #   (b) what it is asked for, once the demand opens
    # Every field runs its own arm against all four patterns, so the two panels vary
    # one thing each and nothing has to be paired.
    PROP_ROWSTEP, PROP_GAIN = 1.25, 2.0
    PROP_LEFT, PROP_RIGHT = 25, 10
    PROP_PATCOL = dict(zip(PROP_PATTERNS, [SERIES[0], SERIES[1], SERIES[2], "#8452a1"]))

    fig_prop = plt.figure(figsize=(W_TEXT, 4.0))
    _gp = fig_prop.add_gridspec(1, 2, width_ratios=[1.30, 0.85], wspace=0.34,
                                left=0.125, right=0.975, top=0.88, bottom=0.135)

    # (a) the room: four arms, one demand, the dotted lines are what a window that
    # long can produce. Arm 2's four minutes sit inside the loop's dead time, so its
    # window admits only one shape however free it is on paper.
    _axp = fig_prop.add_subplot(_gp[0, 0])
    _ntop = 3 * PROP_ROWSTEP + 1.2
    for _i, _a in enumerate(FRP_ARM_ORDER[::-1]):
        _free = frp_arms[_a]["free"]
        _y = _i * PROP_ROWSTEP
        _cnr = lambda v: _y + (np.asarray(v) - FRP_ANCHOR) * PROP_GAIN

        _axp.plot([-PROP_LEFT, -_free], [_y, _y], color=MUTED, lw=1.8)
        if _free:
            _axp.axvspan(-_free, 0, ymin=(_y - 0.32 + 1.0) / (_ntop + 1.0),
                         ymax=(_y + 0.80 + 1.0) / (_ntop + 1.0),
                         color=SERIES[2], alpha=0.16, lw=0)
            for _tr in prop_arm_traces[_a]:
                _axp.plot(np.arange(-_free, 1), _cnr(_tr["cnr"]), color=INK, lw=1.0,
                          ls=":", alpha=0.85)
        _axp.plot(np.arange(PROP_RIGHT + 1),
                  _cnr(prop_reference("hold")[FRP_RUNUP:FRP_RUNUP + PROP_RIGHT + 1]),
                  color=SERIES[0], lw=2.0, drawstyle="steps-post")
        _axp.plot([0], [_cnr(FRP_DEMAND)], "o", ms=5, color=SERIES[0], zorder=6)
        _axp.text(-PROP_LEFT - 1.0, _y + 0.30, f"arm {_a}", ha="right", va="center",
                  fontsize=8.5, fontweight="bold")
        _axp.text(-PROP_LEFT - 1.0, _y + 0.05, f"{_free} free min", ha="right",
                  va="center", fontsize=6.6, color=MUTED)

    _axp.axvline(0, color=INK, lw=1.1, ls="--")
    _axp.set_xlim(-PROP_LEFT, PROP_RIGHT)
    _axp.set_ylim(-1.0, _ntop)
    _axp.set_yticks([])
    _axp.set_xticks([-20, -10, -4, 0, 10])
    _axp.tick_params(labelsize=7.5)
    _axp.set_xlabel("minutes from the moment the demand opens", fontsize=8)
    _axp.set_title("a  Sizes of unscored windows", loc="left",
                   fontweight="bold", fontsize=9)
    _axp.text(-0.8, _ntop - 0.25, "unscored", ha="right", va="center", fontsize=7,
              color=SERIES[2], fontweight="bold")
    _axp.text(0.8, _ntop - 0.25, "graded", ha="left", va="center", fontsize=7,
              color=INK, fontweight="bold")
    for _sp in ("left", "right", "top"):
        _axp.spines[_sp].set_visible(False)

    # (b) the goal: one small panel per demand, stacked. Overlaid on one axis they
    # sit on top of each other -- hold and re-arrive share 1.05 for most of the block,
    # release and re-arrive share the anchor -- so each gets its own row.
    _gq = _gp[0, 1].subgridspec(len(PROP_PATTERNS), 1, hspace=0.30)
    _xd = np.arange(FRP_BLOCK - FRP_RUNUP)
    for _i, (_pat, _col) in enumerate(PROP_PATCOL.items()):
        _axq = fig_prop.add_subplot(_gq[_i])
        _axq.axhline(FRP_ANCHOR, color=MUTED, lw=0.8, ls=":", zorder=1)
        _axq.axhline(FRP_DEMAND, color=GRID, lw=0.8, zorder=1)
        _axq.plot(_xd, prop_reference(_pat)[FRP_RUNUP:], color=_col, lw=2.0,
                  drawstyle="steps-post", zorder=4)
        _axq.plot([0], [FRP_DEMAND], "o", ms=4, color=_col, zorder=5)
        _axq.set_xlim(0, FRP_BLOCK - FRP_RUNUP - 1)
        _axq.set_ylim(FRP_ANCHOR - 0.04, PROP_CLIMB + 0.04)
        _axq.set_yticks([FRP_ANCHOR, FRP_DEMAND, PROP_CLIMB],
                        [f"{FRP_ANCHOR:.2f}", f"{FRP_DEMAND:.2f}", f"{PROP_CLIMB:.2f}"],
                        fontsize=6)
        _axq.tick_params(axis="y", length=2)
        # top left: every pattern opens at the demand and only climb reaches 1.15,
        # and it does so after minute 10, so this corner is clear in all four
        _axq.text(0.02, 0.94, _pat, transform=_axq.transAxes, ha="left", va="top",
                  fontsize=7.5, fontweight="bold", color=_col)
        for _sp in ("top", "right"):
            _axq.spines[_sp].set_visible(False)
        if _i == 0:
            _axq.set_title("b  Graded pattern", loc="left", fontweight="bold",
                           fontsize=9)
        if _i == len(PROP_PATTERNS) - 1:
            _axq.set_xlabel("minutes after the demand opens", fontsize=8)
            _axq.set_xticks([0, 10, 20])
            _axq.tick_params(axis="x", labelsize=7.5)
        else:
            _axq.set_xticks([0, 10, 20])
            _axq.tick_params(axis="x", labelbottom=False, length=2)
    fig_prop.text(0.605, 0.60, "demanded CNR", rotation=90, va="center", ha="center",
                  fontsize=8)

    save_fig(fig_prop, "proposed-demands")
    fig_prop

    return


@app.cell(hide_code=True)
def _(mo, prop_solutions):
    mo.md(f"""
    Dotted: what a window that long can produce —
    {"  ·  ".join(s["name"] for s in prop_solutions)}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Not for the manuscript

    Everything below this line is working material: open questions, an interactive
    inspector, a stale plan, and a mock-up built on invented numbers. None of it is a
    result and none of it should be read as one. It is kept because the reasoning in it
    is worth having, not because it is finished.

    **The mock-up in particular.** "Figure B — baseline experiment" is drawn from
    **invented data** and exists to decide how a result will be read before the run that
    produces it exists. It also loads `BL_PATH` from a path under `/private/tmp/`, which
    belongs to a session that has ended and will be deleted — so the cell will stop
    working, and when it does that is the path being wrong, not the analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure B — baseline experiment, how the result will be read  (MOCK DATA)

    Not results. The parquet behind this is **mock**: the cells are simulated from
    measured dynamics (tau 3 min, 4 min dead time, the v16 dose-to-steady-state curve
    and its decay across the run), driven by a lookahead bang-bang stand-in for the
    MPC, with the yoked arm playing real v19 per-cell dose recordings. It exists so
    the figure and the analysis are settled before the run, not after.

    The model could not be used to simulate the cells. Rolled forward on its own
    predictions for a few hundred steps it drifts to about 1.5 and never comes back
    down, so every arm saturates and the contrast collapses to "which arm withholds
    most light".

    Swap `BL_PATH` for the real log and every panel below is the analysis, unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(pl):

    BL_PATH = "/private/tmp/claude-501/-Users-polya-workshop-masters/8bd49575-8e15-459e-80f2-b463fca8471f/scratchpad/mock_baseline.parquet"
    BL_SETTLE, BL_PERIOD = 60, 60
    BL_ARMS = ["closed", "yoked", "population"]
    BL_LABEL = {"closed": "per-cell closed loop", "yoked": "yoked open loop",
                "population": "population open loop"}
    # v19 resolved paired within-arm contrasts of this size with 90-130 cells. An effect
    # inside the band is not resolvable at this n, whatever its sign.
    BL_FLOOR = (0.02, 0.05)

    bl_raw = pl.read_parquet(BL_PATH)
    # Fold on the CELL'S OWN cycle position, not a common clock: the four phase groups
    # are offset by a quarter period, so a shared clock scrambles the reference and
    # washes the oscillation out of the average.
    bl_run = bl_raw.filter(pl.col("t") >= BL_SETTLE).with_columns(
        ((pl.col("t") - BL_SETTLE + pl.col("pg") * BL_PERIOD // 4) % BL_PERIOD).alias("phase"),
        pl.when(pl.col("r_t") >= 1.10 - 1e-6).then(pl.lit("hi"))
          .when(pl.col("r_t") <= 0.85 + 1e-6).then(pl.lit("lo"))
          .otherwise(pl.lit("ramp")).alias("seg"))

    # Per cell: what the objective asks for, what it cost, and how well-matched the
    # open-loop prescription happened to be.
    _hold = (bl_run.filter(pl.col("seg") != "ramp")
             .group_by(["arm", "field", "cell", "seg"]).agg(pl.col("cnr").median().alias("m"))
             .pivot(on="seg", index=["arm", "field", "cell"], values="m")
             .with_columns((pl.col("hi") - pl.col("lo")).alias("amp")))
    _cost = (bl_run.with_columns(((pl.col("cnr") - pl.col("r_t")) ** 2).alias("e2"))
             .group_by(["arm", "field", "cell"]).agg(
                 pl.col("e2").mean().sqrt().alias("rmse"),
                 pl.col("ms").mean().alias("dose_ms"),
                 pl.col("ms").sum().alias("fluence_ms"),
                 pl.col("mismatch").first()))
    bl_cell = _hold.join(_cost, on=["arm", "field", "cell"])

    # The primary contrast is paired at the FIELD level: in the real run a cell is in one
    # arm only, so cells cannot be paired - fields can, because each field is split.
    bl_field = bl_cell.group_by(["arm", "field"]).agg(
        pl.col("rmse").median(), pl.col("amp").median(), pl.col("dose_ms").mean(),
        pl.len().alias("n_cells"))
    bl_paired = (bl_field.filter(pl.col("arm") == "closed")
                 .join(bl_field.filter(pl.col("arm") == "yoked"), on="field", suffix="_y")
                 .with_columns((pl.col("rmse") - pl.col("rmse_y")).alias("d_rmse"),
                               (pl.col("amp") - pl.col("amp_y")).alias("d_amp")))

    # folded response per arm
    bl_fold = (bl_run.group_by(["arm", "phase"]).agg(
        pl.col("cnr").median().alias("med"),
        pl.col("cnr").quantile(0.25).alias("lo"),
        pl.col("cnr").quantile(0.75).alias("hi"),
        pl.col("r_t").median().alias("ref")).sort(["arm", "phase"]))

    bl_cell.group_by("arm").agg(pl.len().alias("cells"), pl.col("amp").median().round(3),
        pl.col("rmse").median().round(3), pl.col("dose_ms").mean().round(1)).sort("arm")
    return BL_ARMS, BL_FLOOR, BL_LABEL, BL_PERIOD, bl_cell, bl_fold, bl_paired


@app.cell(hide_code=True)
def _(
    BL_ARMS,
    BL_FLOOR,
    BL_LABEL,
    BL_PERIOD,
    GRID,
    INK,
    MUTED,
    SERIES,
    bl_cell,
    bl_fold,
    bl_paired,
    np,
    pl,
    plt,
):
    BL_COL = {"closed": SERIES[0], "yoked": SERIES[1], "population": MUTED}


    def _bl_fold(ax):
        """(a) What happened: the three arms folded onto one cycle."""
        _r = bl_fold.filter(pl.col("arm") == "closed")
        ax.plot(_r["phase"], _r["ref"], color=INK, lw=1.2, ls="--", label="demanded", zorder=5)
        for a in BL_ARMS:
            k = bl_fold.filter(pl.col("arm") == a)
            ax.fill_between(k["phase"], k["lo"], k["hi"], color=BL_COL[a], alpha=0.13, lw=0)
            ax.plot(k["phase"], k["med"], color=BL_COL[a], lw=2.2, label=BL_LABEL[a])
        ax.set_xlabel(f"minutes into the {BL_PERIOD}-min cycle")
        ax.set_ylabel("CNR")
        ax.set_title("a  What happened", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2)


    def _bl_paired(ax):
        """(b) The primary contrast, paired at the field level."""
        rng = np.random.default_rng(0)
        for j, a in enumerate(("closed", "yoked")):
            v = bl_cell.filter(pl.col("arm") == a)["rmse"].to_numpy()
            ax.plot(j + (rng.random(len(v)) - 0.5) * 0.28, v, "o", ms=2.4, alpha=0.28,
                    color=BL_COL[a], markeredgewidth=0)
        for row in bl_paired.iter_rows(named=True):
            ax.plot([0, 1], [row["rmse"], row["rmse_y"]], "-o", color=INK, lw=1.1, ms=3.5)
        ax.set_xticks([0, 1], ["per-cell\nclosed loop", "yoked\nopen loop"], fontsize=7)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel("tracking error, RMSE (CNR)")
        ax.set_title("b  Primary contrast, paired by field", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        d = bl_paired["d_rmse"].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        ax.set_ylim(0, None)
        ax.text(0.5, 0.98,
                f"per field: {np.median(d):+.3f} CNR  (SE {se:.3f}, n={len(d)} fields)\n"
                "black lines pair the two halves of one field",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK)


    def _bl_floor(ax):
        """(c) Is the effect even resolvable at this n?"""
        d = bl_paired["d_rmse"].to_numpy()
        eff = abs(np.median(d))
        ax.axhspan(BL_FLOOR[0], BL_FLOOR[1], color=MUTED, alpha=0.22, lw=0)
        ax.bar([0], [eff], width=0.4, color=SERIES[0])
        ax.errorbar([0], [eff], yerr=[d.std(ddof=1) / np.sqrt(len(d))], color=INK, capsize=4)
        ax.set_xticks([0], ["closed vs yoked"], fontsize=7)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, max(eff * 2.0, BL_FLOOR[1] * 1.8))
        ax.set_ylabel("effect size (CNR)")
        ax.set_title("c  Against what v19 could resolve", loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.97, 0.97,
                f"grey band = {BL_FLOOR[0]:.2f}-{BL_FLOOR[1]:.2f} CNR, what v19's\n"
                "paired contrasts could resolve. An effect\n"
                "inside it is not readable at this n.",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK)


    def _bl_dose(ax):
        """(d) The confound check, as a panel and not a footnote."""
        tot = bl_cell.group_by("arm").agg(pl.col("fluence_ms").mean().alias("f")).sort("arm")
        labs = tot["arm"].to_list()
        vals = [v / 1000 for v in tot["f"]]
        ax.bar(range(len(labs)), vals, color=[BL_COL[a] for a in labs], width=0.6)
        ax.set_xticks(range(len(labs)), [BL_LABEL[a].replace(" ", "\n") for a in labs],
                      fontsize=6.5)
        ax.set_ylabel("total light delivered\nper cell (s of exposure)")
        ax.set_title("d  Did the arms get the same light?", loc="left", fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.5, 0.98,
                f"spread {max(vals) / max(min(vals), 1e-9):.2f}x. If the arms differ\n"
                "materially in light the contrast is\n"
                "contaminated and must be reported so.",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK)


    def _bl_match(ax):
        """(e) Does the advantage survive a well-matched prescription?"""
        ref = dict(zip(bl_paired["field"], bl_paired["rmse"]))
        y = (bl_cell.filter(pl.col("arm") == "yoked")
             .with_columns(pl.col("field").replace_strict(ref).alias("ref_rmse")))
        y = y.with_columns((pl.col("rmse") - pl.col("ref_rmse")).alias("deficit"))
        y = y.with_columns(y["mismatch"].qcut(3, labels=["0", "1", "2"])
                           .cast(pl.Utf8).cast(pl.Int32).alias("terc"))
        agg = y.group_by("terc").agg(
            pl.len().alias("n"), pl.col("deficit").mean(),
            (pl.col("deficit").std() / pl.len().sqrt()).alias("se"),
            pl.col("mismatch").median()).sort("terc")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.bar(agg["terc"], agg["deficit"], yerr=agg["se"], width=0.6, color=SERIES[1],
               capsize=3)
        ax.set_xticks(agg["terc"].to_list(), [f"{v:.2f}" for v in agg["mismatch"]], fontsize=7)
        ax.set_xlabel("how badly matched the donor was (CNR)")
        ax.set_ylabel("yoked cell's excess error\nover its field's closed loop")
        ax.set_title("e  Does it survive a good prescription?", loc="left", fontweight="bold")
        ax.set_ylim(0, float(agg["deficit"].max()) * 1.6)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.98,
                "if the left bar is near zero the advantage\nis about matching, not contingency",
                transform=ax.transAxes, va="top", fontsize=6.5, color=INK)


    fig_bl = plt.figure(figsize=(11.4, 7.2))
    _gbl = fig_bl.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.50, wspace=0.44,
                               left=0.075, right=0.97, top=0.90, bottom=0.10)
    _bl_fold(fig_bl.add_subplot(_gbl[0, :2]))
    _bl_paired(fig_bl.add_subplot(_gbl[0, 2]))
    _bl_floor(fig_bl.add_subplot(_gbl[1, 0]))
    _bl_dose(fig_bl.add_subplot(_gbl[1, 1]))
    _bl_match(fig_bl.add_subplot(_gbl[1, 2]))
    fig_bl.suptitle("MOCK DATA — the readout, settled before the run", x=0.075, ha="left",
                    fontsize=11, fontweight="bold", color=SERIES[1])
    fig_bl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # optoRTK expression — open questions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    # The three figures for the thesis

    Everything above is the working set. These three are what the chapter carries, one
    claim each, built from the same quantities.

    1. **One objective, many interventions** — the design, what the controller does with
       it minute by minute, and whether the differences between cells are a property of
       the cells or of the moment.
    2. **What the loop can and cannot do** — how much of the demand arrives at each
       timescale, the two ways a cell ends up flat, and what a repeat of the same demand
       four hours later looks like.
    3. **Is the model calibrated for the rig it is driving** — unchanged from the working
       version below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## run-ledger — every live run on the same two gates
    """)
    return


@app.cell(hide_code=True)
def _(
    GRID,
    INK,
    MUTED,
    SERIES,
    W_TEXT,
    json,
    materials_path,
    np,
    pl,
    plt,
    save_fig,
):
    # --- The campaign ledger -----------------------------------------------------
    # Every live run scored on the same two gates, so the choice to analyse only a
    # handful of them is a stated criterion rather than a preference.
    #
    #   cadence -- the model is trained on 1-minute sampling. A run whose median
    #              interval exceeds 1.05 min, or whose p90 exceeds 1.10, is feeding
    #              the encoder intervals it never saw.
    #   rail    -- the share of cell-frames sitting on the top exposure rung. Above
    #              20% the controller is not choosing a dose, it is asking for more
    #              light than the rig will give.
    #
    # NOT a manuscript figure. Built for the run-selection argument and for the
    # methods summary table; kept in the notebook only.
    led = (pl.read_parquet(materials_path("run_ledger.parquet"))
             .sort("order", descending=True))          # earliest at the top

    _VC = {"admissible": SERIES[2], "saturating": SERIES[1],
           "cadence slip": SERIES[0]}
    _col = [_VC.get(v, MUTED) for v in led["verdict"]]
    _y = np.arange(led.height)
    _names = [r if (t >= 700 or not f) else f"{r}  ({t}f)"
              for r, t, f in zip(led["run"], led["timesteps"], led["frames"])]
    _adm = [v == "admissible" for v in led["verdict"]]

    fig_ledger = plt.figure(figsize=(W_TEXT, 5.2))
    _gl = fig_ledger.add_gridspec(1, 2, wspace=0.08, left=0.105, right=0.975,
                                  top=0.90, bottom=0.155)

    # (a) cadence: median to p90, against the 1-minute target
    _axl = fig_ledger.add_subplot(_gl[0, 0])
    for _yi, _m, _p, _c in zip(_y, led["mpf"], led["p90"], _col):
        if _m is None:
            continue
        _axl.plot([_m, _p], [_yi, _yi], color=_c, lw=1.6, alpha=0.55,
                  solid_capstyle="round")
        _axl.plot([_m], [_yi], "o", ms=5, color=_c, zorder=5)
    _axl.axvline(1.0, color=INK, lw=1.0, ls="--")
    _axl.set_xscale("log")
    _axl.set_xticks([1, 2, 3, 5], ["1", "2", "3", "5"], fontsize=6.5)
    _axl.minorticks_off()
    _axl.set_xlim(0.85, 8)
    _axl.set_yticks(_y, _names, fontsize=6.5)
    for _t, _a in zip(_axl.get_yticklabels(), _adm):
        _t.set_color(INK if _a else MUTED)
        _t.set_fontweight("bold" if _a else "normal")
    _axl.set_ylim(-0.8, led.height - 0.2)
    _axl.set_xlabel("minutes per frame, median to p90\n"
                    "dashed: 1 min, as trained")
    _axl.set_title("a  Cadence", loc="left", fontweight="bold")
    _axl.xaxis.grid(True, color=GRID, lw=0.6)
    _axl.tick_params(axis="y", length=0)
    _axl.set_axisbelow(True)

    # (b) saturation, with the exposure ceiling that produced it
    _axr = fig_ledger.add_subplot(_gl[0, 1], sharey=_axl)
    _axr.barh(_y, [r if r is not None else 0 for r in led["rail_field"]],
              height=0.62, color=_col)
    # Annotate with the ladders actually issued. Runs that gave different fields
    # different ladders show both, which is where the run-wide number went wrong.
    for _yi, _r, _lad in zip(_y, led["rail_field"], led["ladders"]):
        if _lad is None:
            continue
        _tops = sorted({v for v in json.loads(_lad).values() if v})
        if not _tops:
            continue
        _axr.text((_r or 0) + 0.015, _yi,
                  "/".join(f"{v:.0f}" for v in _tops) + " ms",
                  va="center", fontsize=5.8, color=MUTED)
    _axr.axvline(0.20, color=INK, lw=1.0, ls="--")
    _axr.set_xlim(0, 1.06)
    _axr.set_xticks([0, 0.2, 0.5, 0.8], ["0", "20%", "50%", "80%"], fontsize=6.5)
    _axr.set_xlabel("closed-loop cell-frames on the top rung\n"
                    "of their own field's ladder")
    _axr.set_title("b  Saturation", loc="left", fontweight="bold")
    _axr.xaxis.grid(True, color=GRID, lw=0.6)
    _axr.tick_params(axis="y", length=0, labelleft=False)
    _axr.set_axisbelow(True)

    _hs = [plt.Line2D([], [], color=_VC[k], lw=4, label=k)
           for k in ("admissible", "saturating", "cadence slip")]
    _hs.append(plt.Line2D([], [], color=MUTED, lw=4,
                          label="not usable"))
    _axr.legend(handles=_hs, frameon=False, fontsize=6, loc="lower right",
                handlelength=1.1, labelspacing=0.3, borderaxespad=0.4)

    _n_adm = int(sum(_adm))
    _axl.text(0.0, 1.055, f"{led.height} live runs, {_n_adm} clear both gates",
              transform=_axl.transAxes, fontsize=6.5, color=MUTED)
    for _yi, _fr in zip(_y, led["frames"]):
        if not _fr:
            _axl.text(1.06, _yi, "never ran", fontsize=5.8, color=MUTED,
                      va="center", style="italic")

    save_fig(fig_ledger, "run-ledger")
    fig_ledger
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dose-ladder — the open-loop dose ladder, and why it does not replicate
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, pl, plt, save_fig):
    # --- The dose ladder does not replicate --------------------------------------
    # v14, v15 and v16 each ran the SAME open-loop ladder in fields 0 and 7: fixed
    # blocks of 0 / 300 / 85 / 600 / 150 ms, repeating, identical in every cell, no
    # controller and no model. It is the only direct actuator characterisation the
    # campaign contains -- and the three runs disagree with each other by more than
    # any dose differs from any other dose within a run.
    #
    # The effect plotted is the change in CNR ACROSS a block, per cell, measured from
    # that cell's own value at block entry, with entry level held fixed by
    # regression. The adjustment is necessary and not sufficient: the ladder cycles
    # in a fixed order with no washout, so dose is collinear with position in the
    # cycle and therefore with where the cell starts. Panel b shows that collinearity
    # directly. No run randomised the block order.
    #
    # NOT a manuscript figure.
    dose_fit = pl.read_parquet(materials_path("dose_ladder_fit.parquet"))
    dose_blocks = pl.read_parquet(materials_path("dose_ladder_blocks.parquet"))

    _DOSES = [0.0, 85.0, 150.0, 300.0, 600.0]
    _xpos = {d: i for i, d in enumerate(_DOSES)}
    # dose is a magnitude, so it gets one hue, light to dark
    _RAMP = ["#cfe0f5", "#94bdea", "#5b98dd", "#2a78d6", "#14508f"]

    fig_dose = plt.figure(figsize=(W_TEXT, 2.9))
    _gdz = fig_dose.add_gridspec(1, 2, wspace=0.28, left=0.095, right=0.975,
                                 top=0.87, bottom=0.185, width_ratios=[1.05, 1])

    # (a) three runs of one ladder, three different answers
    _axd = fig_dose.add_subplot(_gdz[0, 0])
    for _i, _r in enumerate(["v14", "v15", "v16"]):
        _s = dose_fit.filter(pl.col("run") == _r).sort("dose_ms")
        _x = [_xpos[d] for d in _s["dose_ms"]]
        _clean = bool(_s["cadence_clean"][0])
        _axd.errorbar(_x, _s["effect"], yerr=_s["se"], color=SERIES[_i],
                      lw=2.2 if _clean else 1.2, ls="-" if _clean else "--",
                      marker="o", ms=6 if _clean else 4, capsize=2, elinewidth=1,
                      zorder=5 if _clean else 3,
                      label=f"{_r}, cadence clean" if _clean else f"{_r}, cadence slipped")
    _axd.axhline(0, color=INK, lw=0.9)
    _axd.set_xticks(range(len(_DOSES)), [f"{d:.0f}" for d in _DOSES], fontsize=6.5)
    _axd.set_xlabel("exposure per frame (ms)")
    _axd.set_ylabel("change in CNR across a block")
    _axd.set_title("a  One ladder, three runs", loc="left", fontweight="bold")
    _axd.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.6,
                labelspacing=0.25, borderaxespad=0.5)
    _axd.yaxis.grid(True, color=GRID, lw=0.6)
    _axd.set_axisbelow(True)

    # (b) The confound, shown directly. Because the cycle order is fixed and nothing
    # washes out between blocks, each dose is systematically ENTERED from a different
    # CNR. The spread between those entry medians is the same size as the "effects"
    # in panel a, which is why the unadjusted contrast cannot be read at all.
    _axe = fig_dose.add_subplot(_gdz[0, 1])
    _v16 = dose_blocks.filter(pl.col("run") == "v16")
    _r = float(dose_fit.filter(pl.col("run") == "v16")["corr_entry_change"][0])
    _ent = (_v16.group_by("dose_ms")
                .agg(pl.col("c0").quantile(0.25).alias("q1"),
                     pl.col("c0").median().alias("md"),
                     pl.col("c0").quantile(0.75).alias("q3"))
                .sort("dose_ms"))
    for _row in _ent.iter_rows(named=True):
        _i = _xpos[_row["dose_ms"]]
        _axe.plot([_row["q1"], _row["q3"]], [_i, _i], color=_RAMP[_i], lw=8,
                  solid_capstyle="butt")
        _axe.plot([_row["md"]], [_i], "|", ms=12, mew=2.0, color=INK, zorder=5)
        _axe.text(_row["q3"] + 0.02, _i, f'{_row["md"]:.2f}', va="center",
                  fontsize=6.5, color=MUTED)
    _axe.set_yticks(range(len(_DOSES)), [f"{d:.0f}" for d in _DOSES], fontsize=6.5)
    _axe.set_ylim(-0.6, len(_DOSES) - 0.4)
    _axe.set_ylabel("exposure per frame (ms)")
    _axe.set_xlabel("CNR at block entry (IQR; tick = median)")
    _axe.set_title("b  Where each dose starts", loc="left",
                   fontweight="bold")
    _spread = float(_ent["md"].max() - _ent["md"].min())
    _axe.text(0.03, 0.96,
              f"entry medians span {_spread:.2f} CNR\nr(entry, change) = {_r:+.2f}",
              transform=_axe.transAxes, ha="left", va="top", fontsize=6,
              color=INK, linespacing=1.5)
    _axe.xaxis.grid(True, color=GRID, lw=0.6)
    _axe.tick_params(axis="y", length=0)
    _axe.set_axisbelow(True)

    save_fig(fig_dose, "dose-ladder")
    fig_dose
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## arm-tracks — what the five kept runs actually did, arm by arm
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## controller-choice \u2014 which MPC variant was chosen, and does it replicate?
    """)
    return


@app.cell(hide_code=True)
def _(materials_path, mo, np, pl):
    # --- Which controller variant was chosen, and does the answer replicate? ------
    # Computation only, no figure. v11 repeats v10's design exactly -- the same four
    # arms on the same field positions -- with a slower and taller demand (50 min
    # period and a 0.375 swing against 40 min and 0.300). The two runs are therefore
    # independent tests of the same question.
    #
    # Both scores are per cell and against that cell's OWN reference, so no phase
    # alignment is needed and no cell is scored against the population:
    #   rmse   root mean square of (CNR - demand) over the controlled frames
    #   frac   the cell's median CNR while its demand is high minus while it is low,
    #          as a fraction of the swing that was actually asked for
    #
    # Kruskal-Wallis across the four arms, then plain MPC against each of the others
    # by a one-sided Mann-Whitney with a Holm correction, because the question is
    # specifically whether plain MPC wins rather than whether the arms differ.
    from scipy.stats import kruskal, mannwhitneyu

    ARM_SETTLE = {"v10": 80, "v11": 100}     # settle_min from each run's reference
    ARM_SWING = {"v10": 0.30, "v11": 0.375}  # the demanded high-minus-low
    ARM_ORDER = ["MPC", "MPC, move 0.6", "MPC, band, move 0.6", "constant dose"]
    ARM_MINFRAMES = 200

    _arm_parts = []
    for _run in ("v10", "v11"):
        _d = (pl.read_parquet(materials_path(f"tracks_{_run}.parquet"))
                .join(pl.read_parquet(materials_path("tracks_arms.parquet"))
                        .filter(pl.col("run") == _run).select("fov", "arm_label"),
                      on="fov", how="left")
                .filter(pl.col("timestep") >= ARM_SETTLE[_run])
                .drop_nulls(["raw_cnr", "r_t"]))
        _lo, _hi = _d["r_t"].min(), _d["r_t"].max()
        _chi, _clo = _lo + 0.8 * (_hi - _lo), _lo + 0.2 * (_hi - _lo)
        _arm_parts.append(
            _d.group_by(["arm_label", "fov", "particle"])
              .agg(((pl.col("raw_cnr") - pl.col("r_t")) ** 2).mean().sqrt().alias("rmse"),
                   pl.col("raw_cnr").filter(pl.col("r_t") >= _chi).median().alias("top"),
                   pl.col("raw_cnr").filter(pl.col("r_t") <= _clo).median().alias("bot"),
                   pl.len().alias("frames"))
              .filter(pl.col("frames") >= ARM_MINFRAMES)
              .drop_nulls(["top", "bot"])
              .with_columns(pl.lit(_run).alias("run"),
                            (pl.col("top") - pl.col("bot")).alias("swing"))
              .with_columns((pl.col("swing") / ARM_SWING[_run]).alias("frac")))

    arm_cells = pl.concat(_arm_parts)

    arm_summary = (arm_cells.group_by(["run", "arm_label"])
                   .agg(pl.len().alias("cells"),
                        pl.col("rmse").median().round(4).alias("rmse"),
                        pl.col("swing").median().round(4).alias("swing"),
                        (100 * pl.col("frac").median()).round(1).alias("pct_of_demand"))
                   .sort(["run", "rmse"]))


    def _arm_holm(run, metric, better_high):
        """Plain MPC against each other arm, Holm-corrected within the run."""
        s = arm_cells.filter(pl.col("run") == run)
        base = s.filter(pl.col("arm_label") == "MPC")[metric].to_numpy()
        others = ARM_ORDER[1:]
        raw = [mannwhitneyu(base, s.filter(pl.col("arm_label") == a)[metric].to_numpy(),
                            alternative=("greater" if better_high else "less")).pvalue
               for a in others]
        order = np.argsort(raw)
        holm, running = np.empty(len(raw)), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, raw[i] * (len(raw) - rank)))
            holm[i] = running
        return [dict(run=run, metric=metric, comparison=f"MPC vs {a}",
                     median_gap=round(float(np.median(base)
                                            - np.median(s.filter(pl.col("arm_label") == a)[metric].to_numpy())), 4),
                     p_holm=float(p), wins=bool(p < 0.05))
                for a, p in zip(others, holm)]


    arm_tests = pl.DataFrame([r for run in ("v10", "v11")
                              for metric, hi in (("rmse", False), ("frac", True))
                              for r in _arm_holm(run, metric, hi)])

    arm_kw = pl.DataFrame([
        dict(run=run, metric=metric,
             p_kruskal=float(kruskal(*[
                 arm_cells.filter((pl.col("run") == run)
                                  & (pl.col("arm_label") == a))[metric].to_numpy()
                 for a in ARM_ORDER]).pvalue))
        for run in ("v10", "v11") for metric in ("rmse", "frac")])

    mo.vstack([
        mo.md("**Per-arm summary** — `rmse` lower is better, `pct_of_demand` higher"),
        arm_summary,
        mo.md("**Kruskal-Wallis across the four arms**"),
        arm_kw,
        mo.md("**Plain MPC against each other arm** — one-sided, Holm-corrected. "
              "`median_gap` is plain MPC minus the other arm."),
        arm_tests,
    ])
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- What the five kept runs actually did ------------------------------------
    # One row per run, one line per arm. The unit is an ARM, not a cell: each line is
    # the median CNR across every cell in that arm, and the dashed line is the demand
    # those cells were given. Individual traces are not drawn -- with 100-500 cells
    # per arm the spaghetti hides exactly the arm-to-arm contrast the runs were built
    # to make.
    #
    # An "arm" is a controller AND its objective. In v16 the same MPC ran four
    # different demands, so pooling by controller alone would average across
    # references that disagree.
    #
    # The x axis is real time, not frame index: these runs did not hold the same
    # cadence, and a frame index would silently rescale them against each other.
    armtrk = pl.read_parquet(materials_path("tracks_summary.parquet"))
    armtrk_led = (pl.read_parquet(materials_path("run_ledger.parquet"))
                 .select("run", "mpf", "ceiling", "verdict"))

    ARMTRK_RUNS = ["v10", "v11", "v16", "v21", "v23", "v24"]
    ARMTRK_NOTE = {
        "v10": "oscillation, 40 min period",
        "v11": "the same design, 50 min period",
        "v16": "four objectives at once",
        "v21": "schedule with free windows",
        "v23": "segmented run-up",
        "v24": "population MPC against open loop",
    }
    # Arms are nominal within a row, so they take a fixed categorical order and each
    # row carries its own key. Never cycled: no row here needs a fifth.
    ARMTRK_C = [SERIES[0], SERIES[1], SERIES[2], "#8452a1"]

    fig_armtrk = plt.figure(figsize=(W_TEXT, 9.0))
    _gt = fig_armtrk.add_gridspec(len(ARMTRK_RUNS), 1, hspace=0.62, left=0.105,
                               right=0.975, top=0.965, bottom=0.062)

    for _i, _run in enumerate(ARMTRK_RUNS):
        _ax = fig_armtrk.add_subplot(_gt[_i, 0])
        _s = armtrk.filter(pl.col("run") == _run)
        _arms = sorted(_s["arm_full"].unique().to_list())

        # the demand, drawn once per distinct reference trace
        _seen = set()
        for _a in _arms:
            _d = _s.filter(pl.col("arm_full") == _a).sort("hours")
            _key = np.round(_d["ref"].to_numpy(), 3).tobytes()
            if _key in _seen:
                continue
            _seen.add(_key)
            _ax.plot(_d["hours"], _d["ref"], color=INK, lw=1.0, ls="--",
                     alpha=0.75, zorder=2)

        for _j, _a in enumerate(_arms):
            _d = _s.filter(pl.col("arm_full") == _a).sort("hours")
            _c = ARMTRK_C[_j % len(ARMTRK_C)]
            _ax.fill_between(_d["hours"], _d["cnr_q1"], _d["cnr_q3"], color=_c,
                             alpha=0.13, lw=0, zorder=3)
            _ax.plot(_d["hours"], _d["cnr"], color=_c, lw=1.5, zorder=4,
                     label=f"{_a}  (n\u2248{int(_d['n'].median())})")

        _m = armtrk_led.filter(pl.col("run") == _run)
        _cad = float(_m["mpf"][0])
        _ax.set_title(f"{_run} \u2014 {ARMTRK_NOTE[_run]}   "
                      f"[{_cad:.2f} min/frame, {float(_m['ceiling'][0]):.0f} ms ceiling]",
                      loc="left", fontweight="bold", fontsize=7.5)
        _ax.set_ylabel("CNR")
        if _i == len(ARMTRK_RUNS) - 1:
            _ax.set_xlabel("hours into the run")
        # Headroom at the top so the key never sits on a trace.
        _lo = float(min(_s["cnr_q1"].min(), _s["ref"].min()))
        _hi = float(max(_s["cnr_q3"].max(), _s["ref"].max()))
        _pad = (_hi - _lo) * 0.06
        _ax.set_ylim(_lo - _pad,
                 _hi + _pad
                 + (_hi - _lo) * (0.95 if _run in ("v10", "v11") else 0.42))
        _ax.legend(frameon=False, fontsize=5.6,
                   ncol=(1 if _run in ("v10", "v11") else 2), loc="upper left",
                   handlelength=1.3, columnspacing=1.0, labelspacing=0.2,
                   borderaxespad=0.25)
        _ax.yaxis.grid(True, color=GRID, lw=0.6)
        _ax.set_axisbelow(True)
        _ax.margins(x=0.01)

        # v10 runs 25 cycles of a 40 min demand across 17 h. At full width the
        # cycles collapse into a band and the arm contrast -- the only place in this
        # run where anything modulates -- becomes invisible. The inset is two hours
        # of the same data at the same scale.
        # v10 and v11 both run many short cycles across a long experiment, so at
        # full width the cycles collapse into a band and the arm contrast is lost
        if _run in ("v10", "v11"):
            _in = _ax.inset_axes([0.56, 0.50, 0.42, 0.45], zorder=10)
            _in.set_facecolor("white")
            _in.patch.set_alpha(1.0)
            _w = (pl.col("hours") >= 3.0) & (pl.col("hours") <= 5.0)
            _z = _s.filter(_w)
            _zd = _z.filter(pl.col("arm_full") == _arms[0]).sort("hours")
            _in.plot(_zd["hours"], _zd["ref"], color=INK, lw=0.9, ls="--", alpha=0.75)
            for _j, _a in enumerate(_arms):
                _d = _z.filter(pl.col("arm_full") == _a).sort("hours")
                _in.plot(_d["hours"], _d["cnr"], color=ARMTRK_C[_j % len(ARMTRK_C)],
                         lw=1.2)
            _in.set_title("3\u20135 h", fontsize=6, color=MUTED, pad=2)
            _in.tick_params(labelsize=5.5, length=2, pad=1)
            _in.set_xticks([3, 4, 5])
            _in.yaxis.grid(True, color=GRID, lw=0.5)
            _in.set_axisbelow(True)
            for _sp in _in.spines.values():
                _sp.set_color(GRID)

    save_fig(fig_armtrk, "arm-tracks")
    fig_armtrk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## controller-across-runs — the controller's decisions, measured on every kept run
    """)
    return


@app.cell(hide_code=True)
def _(GRID, INK, MUTED, SERIES, W_TEXT, materials_path, np, pl, plt, save_fig):
    # --- The controller, across every run kept ------------------------------------
    # The v19 version of this figure asks four things of the controller. Two of them
    # transfer to every run and two do not, so this measures what transfers and says
    # plainly where the rest stops applying.
    #
    # (a) is the load-bearing panel. Cells that share a field, a frame and an
    # IDENTICAL demand are compared, so the schedule is held fixed by construction
    # and any spread in the dose they receive is per-cell decision-making. In v10 the
    # comparison is additionally conditioned on the phase group, because cells in one
    # v10 field are on four different demands. The open-loop arms are the negative
    # control and return exactly zero, which is what makes the positive numbers
    # readable as feedback rather than as noise.
    ctl = pl.read_parquet(materials_path("controller_behaviour_runs.parquet"))
    ctl = ctl.with_columns(
        (pl.col("run") + pl.lit("  ") + pl.col("arm")).alias("label"),
        pl.col("arm").str.contains("open loop").alias("is_ol"))
    ctl = ctl.sort(["run", "arm"], descending=[True, True])

    _y = np.arange(ctl.height)
    _lab = ctl["label"].to_list()
    _col = [MUTED if ol else (SERIES[2] if "population" in a else SERIES[0])
            for ol, a in zip(ctl["is_ol"], ctl["arm"])]

    fig_ctlx = plt.figure(figsize=(W_TEXT, 6.4))
    _gc = fig_ctlx.add_gridspec(2, 2, hspace=0.78, wspace=0.30, left=0.235,
                                right=0.965, top=0.94, bottom=0.10)


    def _ylab(ax, on):
        ax.set_yticks(_y, _lab if on else [""] * len(_y), fontsize=5.8)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, ctl.height - 0.3)


    # (a) does the dose depend on which cell it is?
    _a1 = fig_ctlx.add_subplot(_gc[0, 0])
    _a1.barh(_y, ctl["dose_spread_ms"], height=0.62, color=_col)
    for _yi, _v, _k in zip(_y, ctl["dose_spread_ms"], ctl["mean_distinct"]):
        _a1.text(_v + 1.5, _yi, f"{_k:.1f} rungs", va="center", fontsize=5.2,
                 color=MUTED)
    _a1.set_xlabel("spread of dose across cells sharing\na field, a frame and a demand (ms)")
    _a1.set_title("a  The dose is chosen per cell", loc="left", fontweight="bold")
    _a1.set_xlim(0, float(ctl["dose_spread_ms"].max()) * 1.42)
    _ylab(_a1, True)
    _a1.xaxis.grid(True, color=GRID, lw=0.6)
    _a1.set_axisbelow(True)

    # (b) was there anything left to choose?
    _a2 = fig_ctlx.add_subplot(_gc[0, 1])
    _bot = ctl["at_bottom"].to_numpy()
    _mid = ctl["headroom"].to_numpy()
    _top = ctl["at_top"].to_numpy()
    _a2.barh(_y, _bot, height=0.62, color=GRID, label="dark rung")
    _a2.barh(_y, _mid, left=_bot, height=0.62, color=SERIES[2], label="room to move")
    _a2.barh(_y, _top, left=_bot + _mid, height=0.62, color=SERIES[1],
             label="top rung")
    _a2.set_xlim(0, 1)
    _a2.set_xticks([0, 0.5, 1], ["0", "50%", "100%"], fontsize=6.5)
    _a2.set_xlabel("share of cell-frames")
    _a2.set_title("b  Room left to act", loc="left", fontweight="bold")
    for _x, _t in ((_bot[-1] / 2, "dark"),
                   (_bot[-1] + _mid[-1] / 2, "room to move"),
                   (_bot[-1] + _mid[-1] + _top[-1] / 2, "top")):
        if _x > 0.02:
            _a2.text(_x, _y[-1] + 0.62, _t, ha="center", va="bottom", fontsize=5.4,
                     color=MUTED)
    _ylab(_a2, False)

    # (c) did the light move before the demand did?
    _a3 = fig_ctlx.add_subplot(_gc[1, 0])
    _lead_min = ctl["lead_frames"].to_numpy() * np.array(
        [float(pl.read_parquet(materials_path("run_ledger.parquet"))
                 .filter(pl.col("run") == r)["mpf"][0]) for r in ctl["run"]])
    _a3.barh(_y, ctl["lead_corr"], height=0.62, color=_col)
    for _yi, _c_, _l in zip(_y, ctl["lead_corr"], _lead_min):
        if not np.isfinite(_c_) or not np.isfinite(_l):
            continue
        if abs(_c_) < 0.05:
            _a3.text(0.012, _yi, "no timing signal", va="center", ha="left",
                     fontsize=5.2, color=MUTED, style="italic")
            continue
        _a3.text(max(_c_, 0) + 0.012, _yi,
                 ("leads " if _l < 0 else "lags ") + f"{abs(_l):.0f} min",
                 va="center", ha="left", fontsize=5.2, color=MUTED)
    _a3.axvline(0, color=INK, lw=0.9)
    _a3.set_xlim(-0.14, 0.62)
    _a3.set_xlabel("correlation of dose with the demand's\nrate of change, at its best lag")
    _a3.set_title("c  Anticipation", loc="left", fontweight="bold")
    _ylab(_a3, True)
    _a3.xaxis.grid(True, color=GRID, lw=0.6)
    _a3.set_axisbelow(True)

    # (d) did the optimiser know what its plan would cost?
    _a4 = fig_ctlx.add_subplot(_gc[1, 1])
    _pc = ctl["plancost_corr"].to_numpy()
    _a4.barh(_y, np.nan_to_num(_pc), height=0.62, color=_col)
    for _yi, _v in zip(_y, _pc):
        if not np.isfinite(_v):
            _a4.text(0.02, _yi, "not recorded", va="center", fontsize=5.2,
                     color=MUTED, style="italic")
    _a4.set_xlim(0, 1.0)
    _a4.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"], fontsize=6.5)
    _a4.set_xlabel("corr(plan's own predicted cost,\ncost actually paid over 30 frames)")
    _a4.set_title("d  Cost, predicted vs paid", loc="left", fontweight="bold")
    _ylab(_a4, False)
    _a4.xaxis.grid(True, color=GRID, lw=0.6)
    _a4.set_axisbelow(True)

    save_fig(fig_ctlx, "controller-across-runs")
    fig_ctlx
    return


if __name__ == "__main__":
    app.run()
