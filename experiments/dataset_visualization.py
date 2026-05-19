import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    import matplotlib.pyplot as plt
    from hastyplot import qplot

    # Disable Altair's 20k row cap; we either send small frames or pre-aggregate.
    alt.data_transformers.disable_max_rows()

    from experiments.seq2seq_data import (
        load as load_dataset,
        AVAILABLE_DATASETS,
        STIM_COLS,
    )

    stim_col_names = list(STIM_COLS)
    return (
        AVAILABLE_DATASETS,
        alt,
        load_dataset,
        mo,
        np,
        pl,
        plt,
        qplot,
        stim_col_names,
    )


@app.cell
def _(mo):
    mo.md("""
    # Dataset Visualization

    Pick a dataset and inspect basic statistics, per-group counts,
    CNR distributions, mean trajectories, and stimulation feature
    distributions.
    """)
    return


@app.cell
def _(AVAILABLE_DATASETS, mo):
    dataset_picker = mo.ui.dropdown(
        options=list(AVAILABLE_DATASETS),
        value="real",
        label="Dataset",
    )
    load_button = mo.ui.run_button(label="Load dataset")
    mo.hstack([dataset_picker, load_button])
    return dataset_picker, load_button


@app.cell
def _(dataset_picker, load_button, load_dataset, mo, np, pl, stim_col_names):
    mo.stop(not load_button.value, mo.md("Pick a dataset and click **Load dataset**."))

    ds_name = dataset_picker.value
    cnr_raw, stim_raw, conditions = load_dataset(ds_name)

    # Normalize across uniform-2D and object-array (variable T) returns.
    is_object = cnr_raw.dtype == object

    n_items = len(cnr_raw)

    records = []
    for i in range(n_items):
        cnr_i = cnr_raw[i] if is_object else cnr_raw[i]
        stim_i = stim_raw[i] if is_object else stim_raw[i]  # (n_stim, T)
        T = len(cnr_i)
        row = {
            "idx": np.full(T, i, dtype=np.int32),
            "t": np.arange(T, dtype=np.int32),
            "condition": np.full(T, str(conditions[i]), dtype=object),
            "cnr": np.asarray(cnr_i, dtype=np.float32),
        }
        for c, name in enumerate(stim_col_names):
            row[name] = np.asarray(stim_i[c], dtype=np.float32)
        records.append(row)

    long_df = pl.DataFrame(
        {
            "idx": np.concatenate([r["idx"] for r in records]),
            "t": np.concatenate([r["t"] for r in records]),
            "condition": np.concatenate([r["condition"] for r in records]).astype(str),
            "cnr": np.concatenate([r["cnr"] for r in records]),
            **{
                name: np.concatenate([r[name] for r in records])
                for name in stim_col_names
            },
        }
    )

    # Per-track summary (one row per cell/window).
    per_track_df = pl.DataFrame(
        {
            "idx": np.arange(n_items, dtype=np.int32),
            "condition": np.asarray(conditions).astype(str),
            "length": np.asarray(
                [len(cnr_raw[i]) for i in range(n_items)], dtype=np.int32
            ),
            "cnr_mean": np.asarray(
                [float(np.mean(cnr_raw[i])) for i in range(n_items)], dtype=np.float32
            ),
            "cnr_std": np.asarray(
                [float(np.std(cnr_raw[i])) for i in range(n_items)], dtype=np.float32
            ),
            "cnr_min": np.asarray(
                [float(np.min(cnr_raw[i])) for i in range(n_items)], dtype=np.float32
            ),
            "cnr_max": np.asarray(
                [float(np.max(cnr_raw[i])) for i in range(n_items)], dtype=np.float32
            ),
        }
    )
    return ds_name, is_object, long_df, n_items, per_track_df


@app.cell
def _(ds_name, is_object, long_df, mo, n_items, per_track_df):
    n_conditions = per_track_df["condition"].n_unique()
    t_min, t_max = per_track_df["length"].min(), per_track_df["length"].max()
    layout = "variable-length object array" if is_object else "uniform 2D"
    mo.md(
        f"""
        ## Summary — `{ds_name}`

        - **Tracks / windows:** {n_items:,}
        - **Distinct conditions:** {n_conditions}
        - **Track length (T):** min={t_min}, max={t_max}
        - **Total frames:** {len(long_df):,}
        - **Storage layout:** {layout}
        """
    )
    return


@app.cell
def _(mo, per_track_df, pl):
    mo.md("## Counts by condition")
    counts_df = (
        per_track_df.group_by("condition")
        .agg(pl.len().alias("n_tracks"))
        .sort("n_tracks", descending=True)
    )
    counts_df
    return (counts_df,)


@app.cell
def _(counts_df, qplot):
    qplot(
        counts_df,
        x="n_tracks",
        y="condition",
        color="condition",
        title="Track count per condition",
        height=max(200, 22 * len(counts_df)),
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## CNR statistics per condition (per-track aggregates)
    """)
    return


@app.cell
def _(per_track_df, pl):
    stats_df = (
        per_track_df.group_by("condition")
        .agg(
            pl.len().alias("n"),
            pl.col("length").mean().round(1).alias("len_mean"),
            pl.col("cnr_mean").mean().round(3).alias("mean_of_means"),
            pl.col("cnr_mean").std().round(3).alias("std_of_means"),
            pl.col("cnr_min").min().round(3).alias("global_min"),
            pl.col("cnr_max").max().round(3).alias("global_max"),
        )
        .sort("n", descending=True)
    )
    stats_df
    return


@app.cell
def _(per_track_df, qplot):
    qplot(
        per_track_df,
        x="condition",
        y="cnr_mean",
        color="condition",
        mark="boxplot",
        title="Per-track mean CNR by condition",
        height=350,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Mean CNR trajectory per condition
    """)
    return


@app.cell
def _(long_df, pl, qplot):
    traj_df = (
        long_df.group_by(["condition", "t"])
        .agg(
            pl.col("cnr").mean().alias("cnr_mean"),
            pl.col("cnr").std().alias("cnr_std"),
            pl.len().alias("n"),
        )
        .sort(["condition", "t"])
    )
    qplot(
        traj_df,
        x="t",
        y="cnr_mean",
        color="condition",
        mark="line",
        title="Mean CNR trajectory (per timestep)",
        height=350,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## CNR value distribution (all frames)
    """)
    return


@app.cell
def _(alt, long_df, pl):
    def _hist(df, col, n_bins, group="condition"):
        vmin = df[col].min()
        vmax = df[col].max()
        width = (vmax - vmin) / n_bins if vmax > vmin else 1.0
        return (
            df.with_columns(
                ((pl.col(col) - vmin) / width)
                .floor()
                .cast(pl.Int32)
                .clip(0, n_bins - 1)
                .alias("_bin"),
            )
            .group_by([group, "_bin"])
            .agg(pl.len().alias("count"))
            .with_columns(((pl.col("_bin") + 0.5) * width + vmin).alias(col))
        )


    cnr_hist_df = _hist(long_df, "cnr", 60)
    alt.Chart(cnr_hist_df).mark_bar(opacity=0.6).encode(
        x=alt.X("cnr:Q", title="cnr"),
        y=alt.Y("count:Q", title="count"),
        color=alt.Color("condition:N"),
    ).properties(title="CNR distribution", height=300)
    return


@app.cell
def _(mo, stim_col_names):
    feature_picker = mo.ui.dropdown(
        options=stim_col_names,
        value=stim_col_names[0],
        label="Stim feature",
    )
    feature_picker
    return (feature_picker,)


@app.cell
def _(mo):
    mo.md("""
    ## Stim feature distribution by condition
    """)
    return


@app.cell
def _(alt, feature_picker, long_df, pl):
    def _hist2(df, col, n_bins, group="condition"):
        vmin = df[col].min()
        vmax = df[col].max()
        width = (vmax - vmin) / n_bins if vmax > vmin else 1.0
        return (
            df.with_columns(
                ((pl.col(col) - vmin) / width)
                .floor()
                .cast(pl.Int32)
                .clip(0, n_bins - 1)
                .alias("_bin"),
            )
            .group_by([group, "_bin"])
            .agg(pl.len().alias("count"))
            .with_columns(((pl.col("_bin") + 0.5) * width + vmin).alias(col))
        )


    _feat = feature_picker.value
    stim_hist_df = _hist2(long_df, _feat, 60)
    alt.Chart(stim_hist_df).mark_bar(opacity=0.6).encode(
        x=alt.X(f"{_feat}:Q", title=_feat),
        y=alt.Y("count:Q", title="count"),
        color=alt.Color("condition:N"),
    ).properties(title=f"{_feat} distribution", height=300)
    return


@app.cell
def _(feature_picker, long_df, pl, qplot):
    feat = feature_picker.value
    feat_traj = (
        long_df.group_by(["condition", "t"])
        .agg(pl.col(feat).mean().alias(f"{feat}_mean"))
        .sort(["condition", "t"])
    )
    qplot(
        feat_traj,
        x="t",
        y=f"{feat}_mean",
        color="condition",
        mark="line",
        title=f"Mean {feat} trajectory (per timestep)",
        height=300,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Track-length distribution
    """)
    return


@app.cell
def _(per_track_df, qplot):
    qplot(
        per_track_df,
        x="length",
        color="condition",
        bins=40,
        title="Track length distribution",
        height=280,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sampled trajectories — 5 tracks per condition
    """)
    return


@app.cell(hide_code=True)
def _(long_df, np, per_track_df, pl, plt):
    _rng = np.random.default_rng(0)
    _n_samples = 5

    _sampled_ids = (
        per_track_df.group_by("condition", maintain_order=True)
        .agg(pl.col("idx"))
        .with_columns(
            pl.col("idx").map_elements(
                lambda _ids: list(
                    _rng.choice(
                        _ids.to_numpy(),
                        size=min(_n_samples, len(_ids)),
                        replace=False,
                    )
                ),
                return_dtype=pl.List(pl.Int32),
            )
        )
        .explode("idx")
        .with_columns(sample_idx=pl.int_range(pl.len()).over("condition"))
    )

    _conditions_sorted = (
        _sampled_ids.select("condition")
        .unique(maintain_order=True)["condition"]
        .to_list()
    )
    _n_cols = len(_conditions_sorted)
    _n_rows = _n_samples

    _fig, _axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(2.4 * _n_cols, 1.6 * _n_rows),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    for _col_i, _cond in enumerate(_conditions_sorted):
        _cond_ids = _sampled_ids.filter(pl.col("condition") == _cond).sort(
            "sample_idx"
        )
        for _row_i, _r in enumerate(_cond_ids.iter_rows(named=True)):
            _ax = _axes[_row_i][_col_i]
            _track = long_df.filter(pl.col("idx") == _r["idx"]).sort("t")
            _t = _track["t"].to_numpy()
            _cnr = _track["cnr"].to_numpy()
            _u = _track["u_t"].to_numpy()

            _ax.plot(_t, _cnr, color="#1f77b4", linewidth=1.2)
            _ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
            _ax.tick_params(axis="x", labelsize=7)

            _ax2 = _ax.twinx()
            _ax2.fill_between(_t, 0, _u, color="#ff7f0e", alpha=0.35, linewidth=0)
            _ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=7)
            _ax2.set_ylim(bottom=0)

            if _row_i == 0:
                _ax.set_title(_cond, fontsize=9)
            if _col_i == 0:
                _ax.set_ylabel(f"#{_row_i}", fontsize=8)
            if _row_i == _n_rows - 1:
                _ax.set_xlabel("t", fontsize=8)

    _fig.suptitle("CNR (blue) + u_t (orange) per sampled track", fontsize=11)
    _fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
