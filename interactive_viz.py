"""
Conditional interactive visualizations for ExperimentBundle.

Each visualization is a (name, predicate, render_fn) tuple.
Predicates check bundle metadata; render functions return marimo elements.

Usage (in command_center.py or any marimo notebook):
    from interactive_viz import get_views
    views = get_views(bundle)  # dict[str, marimo element]
    mo.tabs(views)
"""

from __future__ import annotations

from typing import Any

import marimo as mo
import numpy as np
import polars as pl

from experiment import ExperimentBundle


# ---------------------------------------------------------------------------
# Predicates — operate on bundle metadata only (no model reconstruction)
# ---------------------------------------------------------------------------

def _get_latent_codes(b: ExperimentBundle) -> np.ndarray | None:
    for key in ("z_mu", "z"):
        v = b.metrics.get(key)
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    return None


def _has_latent_codes(b: ExperimentBundle) -> bool:
    return _get_latent_codes(b) is not None


def _wants_latent_2d(b: ExperimentBundle) -> bool:
    return _has_latent_codes(b) and b.model_config.get("latent_dim") == 2


def _wants_latent_3d(b: ExperimentBundle) -> bool:
    return _has_latent_codes(b) and b.model_config.get("latent_dim", 0) >= 3


def _has_kl(b: ExperimentBundle) -> bool:
    return "kl_per_dim" in b.metrics and isinstance(b.metrics["kl_per_dim"], np.ndarray)


def _has_loss_history(b: ExperimentBundle) -> bool:
    tr = b.training_results
    return "train_loss" in tr or "history" in tr


# ---------------------------------------------------------------------------
# Render functions — each returns a marimo-displayable element
# ---------------------------------------------------------------------------

def render_latent_2d(bundle: ExperimentBundle) -> Any:
    import altair as alt

    z_mu = _get_latent_codes(bundle)
    mse = bundle.metrics.get("mse_per_cell", bundle.metrics.get("mse_per_sample"))

    data = {"z0": z_mu[:, 0], "z1": z_mu[:, 1]}
    if mse is not None:
        data["mse"] = np.asarray(mse)

    df = pl.DataFrame(data)

    color_col = "mse" if "mse" in data else "z0"
    chart = (
        alt.Chart(df)
        .mark_circle(size=20, opacity=0.6)
        .encode(
            x=alt.X("z0:Q", title="Latent dim 0"),
            y=alt.Y("z1:Q", title="Latent dim 1"),
            color=alt.Color(f"{color_col}:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=list(data.keys()),
        )
        .properties(width=500, height=400, title="2D Latent Space")
        .interactive()
    )
    return mo.ui.altair_chart(chart)


def render_latent_3d(bundle: ExperimentBundle) -> Any:
    import plotly.express as px

    z_mu = _get_latent_codes(bundle)
    latent_dim = z_mu.shape[1]
    mse = bundle.metrics.get("mse_per_cell", bundle.metrics.get("mse_per_sample"))

    # Use first 3 dimensions by default
    data = {"z0": z_mu[:, 0], "z1": z_mu[:, 1], "z2": z_mu[:, 2]}
    if mse is not None:
        data["mse"] = np.asarray(mse)

    df = pl.DataFrame(data).to_pandas()

    color_col = "mse" if "mse" in data else None
    fig = px.scatter_3d(
        df,
        x="z0", y="z1", z="z2",
        color=color_col,
        color_continuous_scale="Viridis",
        opacity=0.5,
        labels={"z0": "Dim 0", "z1": "Dim 1", "z2": "Dim 2"},
        title=f"3D Latent Space (dims 0-2 of {latent_dim})",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
    return mo.ui.plotly(fig)


def render_kl_explorer(bundle: ExperimentBundle) -> Any:
    import altair as alt

    kl = bundle.metrics["kl_per_dim"]
    threshold = 0.05

    df = pl.DataFrame({
        "dimension": list(range(len(kl))),
        "kl": kl.astype(float),
        "active": [bool(v >= threshold) for v in kl],
    })

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("dimension:O", title="Latent Dimension"),
            y=alt.Y("kl:Q", title="KL Divergence"),
            color=alt.Color(
                "active:N",
                scale=alt.Scale(domain=[True, False], range=["#1f77b4", "#d3d3d3"]),
                legend=alt.Legend(title="Active"),
            ),
            tooltip=["dimension:O", alt.Tooltip("kl:Q", format=".4f"), "active:N"],
        )
        .properties(width=500, height=300, title=f"KL per Dimension (threshold={threshold})")
    )

    n_active = int(sum(kl >= threshold))
    summary = mo.md(
        f"**{n_active}** / {len(kl)} active dimensions "
        f"(KL >= {threshold}). Total KL = {kl.sum():.4f}"
    )
    return mo.vstack([chart, summary])


def render_loss_curves(bundle: ExperimentBundle) -> Any:
    import altair as alt

    tr = bundle.training_results
    history = tr.get("history", {})
    train_loss = history.get("train_loss", tr.get("train_loss"))
    val_loss = history.get("val_loss", tr.get("val_loss"))

    if train_loss is None:
        return mo.md("_No loss history available._")

    train_loss = np.asarray(train_loss)
    epochs = list(range(1, len(train_loss) + 1))
    rows = [{"epoch": e, "loss": float(v), "split": "train"} for e, v in zip(epochs, train_loss)]
    if val_loss is not None:
        val_loss = np.asarray(val_loss)
        rows += [{"epoch": e, "loss": float(v), "split": "val"} for e, v in zip(epochs, val_loss)]

    df = pl.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="Loss", scale=alt.Scale(type="log")),
            color="split:N",
            tooltip=["epoch:Q", alt.Tooltip("loss:Q", format=".6f"), "split:N"],
        )
        .properties(width=550, height=300, title="Training Loss")
        .interactive()
    )
    return chart


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------

INTERACTIVE_VIEWS: list[tuple[str, Any, Any]] = [
    ("2D Latent Space", _wants_latent_2d, render_latent_2d),
    ("3D Latent Space", _wants_latent_3d, render_latent_3d),
    ("KL Explorer",     _has_kl,          render_kl_explorer),
    ("Loss Curves",     _has_loss_history, render_loss_curves),
]


def get_views(bundle: ExperimentBundle) -> dict[str, Any]:
    """Return {name: marimo_element} for all applicable visualizations."""
    views = {}
    for name, predicate, render_fn in INTERACTIVE_VIEWS:
        if predicate(bundle):
            views[name] = render_fn(bundle)
    return views
