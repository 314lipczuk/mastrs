import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import altair as alt
    import matplotlib.pyplot as plt
    from hastyplot import qplot

    alt.data_transformers.disable_max_rows()

    from optoerk.data.preprocessing import augment, load_and_clean

    return alt, augment, load_and_clean, mo, np, os, pd, pl, plt, qplot


@app.cell
def _(mo):
    mo.md("""
    # Single-experiment visualization

    Load **one** experiment `exp_data.parquet` through the canonical adapter and
    inspect it: raw schema, per-pattern CNR, single-cell trajectories, and a
    stratify-by-quartile pipeline. Everything downstream operates on the cleaned
    canonical frame `exp_df` (`stim_condition`, `frame`, baseline-normalized
    `cnr_median_norm`, per-cell `uid`). Cross-experiment comparison lives in
    `dataset_visualization.py` (the dataset/bundle tool).
    """)
    return


@app.cell
def _(mo):
    exp_path_input = mo.ui.text(
        value=(
            "/Volumes/imaging.data/PertzLab/Alex/FreePatternStimulation/2026-06-23_FreePatternStim_Jungfrau_v1/exp_data.parquet"
        ),
        label="Experiment parquet path",
        full_width=True,
    )
    exp_loader_picker = mo.ui.dropdown(
        options=["freepattern", "standard", "bo"],
        value="freepattern",
        label="Adapter family",
    )
    exp_bo_tag = mo.ui.text(value="v8", label="BO tag (bo family only)")
    exp_load_button = mo.ui.run_button(label="Load experiment")
    mo.vstack(
        [
            exp_path_input,
            mo.hstack([exp_loader_picker, exp_bo_tag, exp_load_button]),
        ]
    )
    return exp_bo_tag, exp_load_button, exp_loader_picker, exp_path_input


@app.cell
def _(
    exp_bo_tag,
    exp_load_button,
    exp_loader_picker,
    exp_path_input,
    load_and_clean,
    mo,
    os,
    pd,
    pl,
):
    # EXP_PARQUET forces a path and auto-loads (no click) so `marimo export html`
    # can render one experiment headlessly.
    _env_path = os.environ.get("EXP_PARQUET")
    mo.stop(
        not exp_load_button.value and not _env_path,
        mo.md("Enter a path and click **Load experiment**."),
    )

    exp_path = (_env_path or exp_path_input.value).strip()
    exp_name = exp_path.rstrip("/").split("/")[-2] if "/" in exp_path else exp_path

    # Raw is kept only for the schema table; every plot uses the canonical
    # `exp_df` from the adapter (family picked above).
    exp_raw = pd.read_parquet(exp_path)
    _family = exp_loader_picker.value
    _kw = {"bo_tag": exp_bo_tag.value} if _family == "bo" else {}
    exp_df = load_and_clean(
        exp_raw, experiment=_family, experiment_name=exp_name, **_kw
    )

    # Nested/object columns (channels, img_shape, ...) can't render in a table.
    _obj_cols = [c for c in exp_raw.columns if exp_raw[c].dtype == object]

    def _safe_nunique(_s):
        try:
            return int(_s.nunique(dropna=True))
        except TypeError:
            return None  # unhashable (nested arrays)

    _schema = pl.DataFrame(
        {
            "column": list(exp_raw.columns),
            "dtype": [str(exp_raw[c].dtype) for c in exp_raw.columns],
            "n_null": [int(exp_raw[c].isna().sum()) for c in exp_raw.columns],
            "n_unique": [_safe_nunique(exp_raw[c]) for c in exp_raw.columns],
            "example": [
                repr(exp_raw[c].dropna().iloc[0])[:60]
                if exp_raw[c].notna().any()
                else None
                for c in exp_raw.columns
            ],
        }
    )
    _preview = exp_raw.drop(columns=_obj_cols).head(500)

    mo.vstack(
        [
            mo.md(
                f"""
                ## `{exp_name}` — loaded via the **`{_family}`** adapter

                Raw shown for reference; canonical frame is in kernel as `exp_df`.

                - **Raw rows:** {len(exp_raw):,} · **cols:** {exp_raw.shape[1]}
                - **Canonical rows:** {len(exp_df):,} · **cells (uid):** {exp_df['uid'].nunique():,}
                  · **stim_conditions:** {exp_df['stim_condition'].nunique()}
                - **Nested/object cols (hidden in preview):** {_obj_cols}
                """
            ),
            mo.md("### Raw column schema"),
            mo.ui.table(_schema),
            mo.md("### Raw preview — first 500 rows"),
            mo.ui.table(_preview),
        ]
    )
    return exp_df, exp_name


@app.cell(hide_code=True)
def _(alt, exp_df, pl):
    # Mean baseline-normalized CNR per stim_condition over time (canonical).
    _pk = (
        pl.from_pandas(exp_df[["stim_condition", "frame", "cnr_median_norm", "uid"]])
        .group_by(["stim_condition", "frame"])
        .agg(
            pl.col("cnr_median_norm").mean().alias("cnr_mean"),
            pl.col("uid").n_unique().alias("n_cells"),
        )
        .with_columns(
            pl.col("stim_condition").str.extract(r"(\d+)$").cast(pl.Int32).alias("pat_i")
        )
        .sort(["pat_i", "frame"])
    )

    _heat = alt.Chart(_pk).mark_rect().encode(
        x=alt.X("frame:O", title="frame"),
        y=alt.Y("stim_condition:N", sort=alt.SortField("pat_i"), title="stim_condition"),
        color=alt.Color(
            "cnr_mean:Q", scale=alt.Scale(scheme="viridis"), title="mean CNR (norm)"
        ),
        tooltip=[
            "stim_condition",
            "frame",
            alt.Tooltip("cnr_mean:Q", format=".3f"),
            "n_cells",
        ],
    ).properties(
        title="Mean normalized CNR per stim_condition over time",
        width=760,
        height=1000,
    )
    _heat
    return


@app.cell
def _(exp_df, np, plt):
    # Single-cell CNR trajectories with per-pulse stimulation.
    # Cell = canonical `uid`. Orange stems = stim pulses; stem HEIGHT =
    # stim_exposure (ms). CNR (blue) is baseline-normalized cnr_median_norm.
    _rng = np.random.default_rng(1)
    _cols = ["uid", "frame", "cnr_median_norm", "stim", "stim_exposure"]
    _d = exp_df[_cols].copy()

    _sizes = _d.groupby("uid").size()
    _ok = _sizes[(_sizes >= 70) & (_sizes <= 95)].index.to_list()
    _pick = [_ok[i] for i in _rng.choice(len(_ok), size=min(9, len(_ok)), replace=False)]

    _emax = float(_d.loc[_d["stim"], "stim_exposure"].max())

    _fig, _axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True)
    for _i, (_ax, _uid) in enumerate(zip(_axes.ravel(), _pick)):
        _c = (
            _d[_d["uid"] == _uid]
            .groupby("frame", as_index=False)
            .agg(
                cnr=("cnr_median_norm", "mean"),
                stim=("stim", "max"),
                stim_exposure=("stim_exposure", "max"),
            )
            .sort_values("frame")
        )
        _t = _c["frame"].to_numpy()
        _cnr = _c["cnr"].to_numpy()
        _on = _c[_c["stim"]]
        _ot = _on["frame"].to_numpy()
        _oe = _on["stim_exposure"].to_numpy()

        _ax2 = _ax.twinx()
        _ax2.vlines(_ot, 0, _oe, color="#ffb703", lw=1.6, zorder=1)
        _ax2.scatter(_ot, _oe, color="#fb8500", s=12, zorder=2)
        _ax2.set_ylim(0, _emax * 1.1)
        _ax2.tick_params(axis="y", labelcolor="#fb8500", labelsize=7)
        if _i % 3 == 2:
            _ax2.set_ylabel("stim_exposure (ms)", color="#fb8500", fontsize=8)
        else:
            _ax2.set_yticklabels([])

        _ax.plot(_t, _cnr, color="#1f77b4", lw=1.4, zorder=3)
        _ax.axhline(1.0, color="grey", lw=0.6, ls="--", zorder=1)
        _ax.set_zorder(_ax2.get_zorder() + 1)
        _ax.patch.set_visible(False)
        _ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
        _ax.set_title(_uid.split("__", 1)[-1], fontsize=8, loc="left")
        _ax.margins(x=0)

    for _ax in _axes[-1]:
        _ax.set_xlabel("frame")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("CNR (norm)", color="#1f77b4")
    _fig.suptitle(
        "Single-cell CNR (blue) + per-pulse stim_exposure stems (orange)",
        fontsize=12,
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.gca()
    return


@app.cell
def _(exp_df, mo, pl):
    # Basic statistics off the canonical frame. A single-cell trajectory = `uid`.
    _b = pl.from_pandas(
        exp_df[["uid", "stim_condition", "fov", "frame", "cnr_median_norm", "stim"]]
    )

    _per_cell = _b.group_by("uid").agg(
        pl.col("stim_condition").first().alias("stim_condition"),
        pl.len().alias("n_frames"),
        pl.col("stim").sum().alias("n_pulses"),
        pl.col("cnr_median_norm").mean().alias("cnr_mean"),
        pl.col("cnr_median_norm").max().alias("cnr_max"),
    )

    _per_pattern = (
        _per_cell.group_by("stim_condition")
        .agg(
            pl.len().alias("n_cells"),
            pl.col("n_frames").median().alias("median_len"),
            pl.col("n_pulses").mean().round(1).alias("mean_pulses"),
            pl.col("cnr_mean").mean().round(3).alias("mean_cnr"),
            pl.col("cnr_max").mean().round(3).alias("mean_peak_cnr"),
        )
        .with_columns(
            pl.col("stim_condition").str.extract(r"(\d+)$").cast(pl.Int32).alias("pat_i")
        )
        .sort("pat_i")
        .drop("pat_i")
    )

    _n_cells = _per_cell.height
    _headline = mo.md(
        f"""
        ## Basic statistics

        - **Rows (frames):** {len(exp_df):,}
        - **Single-cell trajectories (uid):** {_n_cells:,}
        - **Stim patterns (stim_condition):** {_per_pattern.height}
        - **FOVs:** {exp_df['fov'].nunique()}
        - **Cells per pattern:** min {_per_cell.group_by('stim_condition').len()['len'].min()},
          median {int(_per_cell.group_by('stim_condition').len()['len'].median())},
          max {_per_cell.group_by('stim_condition').len()['len'].max()}
        - **Track length (frames/cell):** min {_per_cell['n_frames'].min()},
          median {int(_per_cell['n_frames'].median())},
          max {_per_cell['n_frames'].max()}
        - **Pulses per cell:** median {int(_per_cell['n_pulses'].median())},
          max {_per_cell['n_pulses'].max()}
        """
    )

    mo.vstack([_headline, mo.md("### Per-pattern breakdown"), mo.ui.table(_per_pattern)])
    return


@app.cell
def _(augment, exp_df, pl):
    # One row per cell (uid): categorical stratifiers + numeric metrics. augment()
    # adds per-cell amplitude / peak_cnr / total_fluence / responder.
    _aug = augment(exp_df)
    _per_cell = (
        _aug.groupby("uid")
        .agg(
            stim_condition=("stim_condition", "first"),
            fov=("fov", "first"),
            responder=("responder", "first"),
            n_frames=("frame", "count"),
            n_pulses=("m_t", "sum"),
            cnr_max=("cnr_median_norm", "max"),
            amplitude=("amplitude", "first"),
            peak_cnr=("peak_cnr", "first"),
            total_fluence=("total_fluence", "first"),
            median_cnr_0_9=("median_cnr_0_9", "first"),
        )
        .reset_index()
    )
    exp_cell_df = pl.from_pandas(_per_cell)
    exp_strat_cols = ["stim_condition", "fov", "responder"]
    exp_metric_cols = [
        "cnr_max", "amplitude", "peak_cnr", "n_pulses", "total_fluence", "median_cnr_0_9",
    ]
    exp_long_df = pl.from_pandas(
        exp_df[["uid", "frame", "cnr_median_norm"]].rename(
            columns={"frame": "t", "cnr_median_norm": "cnr"}
        )
    )
    return exp_cell_df, exp_long_df, exp_metric_cols, exp_strat_cols


@app.cell
def _(exp_cell_df, exp_name, mo):
    mo.md(
        f"""
        ## Stratified breakdown — `{exp_name}`

        - **Cells:** {len(exp_cell_df):,}
        - Pick a quartile metric and a stratification column below.
        """
    )
    exp_cell_df
    return


@app.cell
def _(exp_metric_cols, mo):
    quartile_metric = mo.ui.dropdown(
        options=exp_metric_cols,
        value=exp_metric_cols[0] if exp_metric_cols else None,
        label="Quartile metric",
    )
    quartile_metric
    return (quartile_metric,)


@app.cell
def _(exp_cell_df, exp_strat_cols, pl, quartile_metric):
    # Add a per-cell response-quartile label for the selected metric.
    _m = quartile_metric.value
    if _m is not None:
        exp_cell_strat = exp_cell_df.with_columns(
            pl.col(_m)
            .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"], allow_duplicates=True)
            .cast(pl.Utf8)
            .alias("quartile")
        )
        strat_options = exp_strat_cols + ["quartile"]
    else:
        exp_cell_strat = exp_cell_df
        strat_options = exp_strat_cols
    return exp_cell_strat, strat_options


@app.cell
def _(mo, strat_options):
    strat_picker = mo.ui.dropdown(
        options=strat_options,
        value=strat_options[0] if strat_options else None,
        label="Stratify by",
    )
    strat_picker
    return (strat_picker,)


@app.cell
def _(exp_cell_strat, pl, strat_picker):
    # One (uid -> stratum) label frame, reused by every plot below. Cast to str
    # so numeric params (fov, ...) colour as discrete groups.
    _sc = strat_picker.value
    exp_labels = exp_cell_strat.select(
        "uid", pl.col(_sc).cast(pl.Utf8).alias("stratum")
    )
    exp_cell_labelled = exp_cell_strat.with_columns(
        pl.col(_sc).cast(pl.Utf8).alias("stratum")
    )
    return exp_cell_labelled, exp_labels


@app.cell
def _(exp_cell_labelled, pl, qplot, strat_picker):
    exp_counts = (
        exp_cell_labelled.group_by("stratum")
        .agg(pl.len().alias("n_cells"))
        .sort("stratum")
    )
    qplot(
        exp_counts,
        x="n_cells",
        y="stratum",
        color="stratum",
        title=f"Cell count per {strat_picker.value}",
        height=max(200, 22 * len(exp_counts)),
    )
    return


@app.cell
def _(exp_labels, exp_long_df, pl, qplot, strat_picker):
    exp_traj = (
        exp_long_df.join(exp_labels, on="uid", how="inner")
        .group_by(["stratum", "t"])
        .agg(pl.col("cnr").mean().alias("cnr_mean"), pl.len().alias("n"))
        .sort(["stratum", "t"])
    )
    qplot(
        exp_traj,
        x="t",
        y="cnr_mean",
        color="stratum",
        mark="line",
        title=f"Mean CNR trajectory by {strat_picker.value}",
        height=350,
    )
    return


@app.cell
def _(alt, exp_cell_labelled, pl, quartile_metric, strat_picker):
    # Per-cell response boxplot per stratum (precomputed quantiles).
    _m = quartile_metric.value
    _q = (
        exp_cell_labelled.filter(pl.col(_m).is_not_null())
        .group_by("stratum")
        .agg(
            pl.col(_m).quantile(0.05).alias("lo"),
            pl.col(_m).quantile(0.25).alias("q1"),
            pl.col(_m).median().alias("median"),
            pl.col(_m).quantile(0.75).alias("q3"),
            pl.col(_m).quantile(0.95).alias("hi"),
        )
        .sort("stratum")
    )
    _base = alt.Chart(_q).encode(
        y=alt.Y("stratum:N", title=strat_picker.value),
        color=alt.Color("stratum:N", legend=None),
    )
    _rule = _base.mark_rule().encode(x="lo:Q", x2="hi:Q")
    _bar = _base.mark_bar(size=14).encode(x="q1:Q", x2="q3:Q")
    _tick = _base.mark_tick(color="white", size=14, thickness=2).encode(x="median:Q")
    (_rule + _bar + _tick).properties(
        title=f"{_m} per {strat_picker.value}", height=max(200, 22 * len(_q))
    )
    return


@app.cell
def _(alt, exp_labels, exp_long_df, pl, strat_picker):
    # CNR frame-value distribution per stratum.
    _df = exp_long_df.join(exp_labels, on="uid", how="inner")
    _n_bins = 60
    _vmin = _df["cnr"].min()
    _vmax = _df["cnr"].max()
    _width = (_vmax - _vmin) / _n_bins if _vmax > _vmin else 1.0
    _hist = (
        _df.with_columns(
            ((pl.col("cnr") - _vmin) / _width)
            .floor()
            .cast(pl.Int32)
            .clip(0, _n_bins - 1)
            .alias("_bin")
        )
        .group_by(["stratum", "_bin"])
        .agg(pl.len().alias("count"))
        .with_columns(((pl.col("_bin") + 0.5) * _width + _vmin).alias("cnr"))
    )
    alt.Chart(_hist).mark_bar(opacity=0.6).encode(
        x=alt.X("cnr:Q", title="cnr"),
        y=alt.Y("count:Q", title="count"),
        color=alt.Color("stratum:N", title=strat_picker.value),
    ).properties(title=f"CNR distribution by {strat_picker.value}", height=300)
    return


if __name__ == "__main__":
    app.run()
