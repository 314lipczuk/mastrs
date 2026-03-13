# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy==2.4.3",
#     "scikit-learn==1.8.0",
#     "wigglystuff",
#     "matplotlib==3.10.8",
#     "pandas==3.0.1",
#     "umap-learn==0.5.11",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    #from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from umap import UMAP
    from wigglystuff import ParallelCoordinates
    import matplotlib.pyplot as plt

    return PCA, ParallelCoordinates, UMAP, mo, np, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fashion MNIST — Parallel Coordinates

    This notebook loads the Fashion MNIST dataset, reduces the 784 pixel features
    down to a handful of PCA components, and visualizes them with an interactive
    parallel coordinates plot. Use the brushes on each axis to filter and explore
    how different clothing categories separate in PCA space.
    """)
    return


@app.cell
def _(pl):
    #mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    #images = mnist.data.astype(np.float32)
    #labels = mnist.target.astype(int)
    dta = pl.read_parquet('./dataset.parquet')
    #dta.group_by(['ramp_stim_pattern', 'frame'])
    dta['ramp_pattern_name'].unique()
    return (dta,)


@app.cell
def _(dta, np, pl):
    res = {}
    split_col = 'ramp_pattern_name'
    sort_col = 'frame'
    value_col = 'cnr_median_norm'
    tracks_col = 'uid'
    groups = dta.partition_by(split_col, maintain_order=True)
    splits = dta[split_col].unique().sort().to_list()

    for s, g in zip(splits, groups):
        vecs = [
            g.filter(pl.col(tracks_col) == t).sort(sort_col)[value_col].to_numpy()
            for t in g[tracks_col].unique().sort()
        ]
        from scipy.stats import mode
        N = mode([len(v) for v in vecs]).mode
        res[s] = np.stack([v for v in vecs if len(v) == N])
    images = res['3-2-1minIntervals']
    images.shape
    return (images,)


@app.cell
def _(dta):
    dta.group_by(["ramp_pattern_name", "uid"]).len().sort(["ramp_pattern_name", "len"])
    return


@app.cell
def _():
    return


@app.cell
def _(PCA, UMAP, checkbox, images, n_components_slider, np, pl):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(images), size=len(images), replace=False)

    if checkbox.value:
        pca = UMAP(n_components=n_components_slider.value)
    else:
        pca = PCA(n_components=n_components_slider.value)

    components = pca.fit_transform(images[idx])

    df = pl.DataFrame(
        {f"PC{i + 1}": components[:, i] for i in range(n_components_slider.value)}
    )
    return df, idx


@app.cell(hide_code=True)
def _(mo):
    n_samples_slider = mo.ui.slider(
        start=366, stop=5000, step=500, value=2000, label="Number of samples"
    )
    n_components_slider = mo.ui.slider(start=3, stop=15, step=1, value=8, label="Components")
    checkbox = mo.ui.checkbox(label="UMAP")
    [n_samples_slider, n_components_slider, checkbox]
    return checkbox, n_components_slider


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(ParallelCoordinates, df, mo):
    widget = mo.ui.anywidget(ParallelCoordinates(df, height=500, color_by="label"))
    widget
    return (widget,)


@app.cell(hide_code=True)
def _(idx, images, np, plt, widget):
    selected = [int(u) for u in widget.value.get("selected_uids", [])]
    sample_idx = selected[:10] if len(selected) >= 10 else selected

    fig, axes = plt.subplots(1, max(len(sample_idx), 1), figsize=(2 * max(len(sample_idx), 1), 2))
    axes = np.atleast_1d(axes)
    for _ax, _si in zip(axes, sample_idx):
        _ax.plot(images[idx[_si]])
        _ax.axis("on")
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
