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


def _has_trajectory_data(b: ExperimentBundle) -> bool:
    from pathlib import Path

    if "AutoEncoder" not in b.model_type:
        return False
    if "Conditional" in b.model_type:
        return False
    if b.model_state_dict is None:
        return False
    if not Path("synthetic_EGFR_data.parquet").exists():
        return False
    return True


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


def render_trajectory_explorer(bundle: ExperimentBundle) -> Any:
    import altair as alt
    import torch
    from data import load_synthetic_states

    # Load test data from the same parquet used for training
    dataset = load_synthetic_states()
    test_split = dataset.test
    state_names = dataset.state_names
    traj_len = dataset.traj_len

    # Reconstruct model and run inference on test trajectories
    model = bundle.reconstruct_model()
    device = next(model.parameters()).device

    # Pick a sample of test trajectories for the UI
    n_test_trajs = len(test_split.traj_ids)
    max_trajs = min(n_test_trajs, 50)
    rng = np.random.default_rng(42)
    chosen_local = sorted(rng.choice(n_test_trajs, size=max_trajs, replace=False))

    # Run inference per trajectory and build long-form data
    rows = []
    with torch.no_grad():
        for local_idx in chosen_local:
            s = local_idx * traj_len
            e = s + traj_len
            orig = test_split.states[s:e]  # (traj_len, n_vars)
            light = test_split.light[s:e]  # (traj_len,)
            x = torch.tensor(orig, dtype=torch.float32).to(device)
            recon, z = model(x)
            recon = recon.cpu().numpy()
            z = z.cpu().numpy()
            traj_label = int(test_split.traj_ids[local_idx])

            for t in range(traj_len):
                row: dict[str, Any] = {"traj_id": traj_label, "timepoint": t, "light": float(light[t])}
                for d in range(z.shape[1]):
                    row[f"z{d}"] = float(z[t, d])
                for vi, name in enumerate(state_names):
                    row[name] = float(orig[t, vi])
                    row[f"{name}_recon"] = float(recon[t, vi])
                rows.append(row)

    df = pl.DataFrame(rows)
    latent_dim = z.shape[1]
    traj_options = sorted(int(test_split.traj_ids[i]) for i in chosen_local)

    # Trajectory selector via altair dropdown binding
    traj_select = alt.selection_point(
        name="traj_sel",
        fields=["traj_id"],
        bind=alt.binding_select(options=traj_options, name="Trajectory "),
        value=[{"traj_id": traj_options[0]}],
    )

    # Timepoint slider via altair binding
    time_select = alt.selection_point(
        name="time_sel",
        fields=["timepoint"],
        bind=alt.binding_range(min=0, max=traj_len - 1, step=1, name="Timepoint "),
        value=[{"timepoint": 0}],
    )

    base = alt.Chart(df).transform_filter(traj_select)

    chart_width = 400
    small_height = 60

    # --- Latent trajectory path (z_i vs z_j) ---
    from itertools import combinations
    latent_pairs = list(combinations(range(min(latent_dim, 4)), 2))[:3]
    path_charts = []
    for d1, d2 in latent_pairs:
        points = base.mark_circle(size=25, opacity=0.6).encode(
            x=alt.X(f"z{d1}:Q", title=f"z{d1}"),
            y=alt.Y(f"z{d2}:Q", title=f"z{d2}"),
            color=alt.Color("timepoint:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["timepoint:Q"] + [alt.Tooltip(f"z{d}:Q", format=".3f") for d in range(min(latent_dim, 4))],
        )
        path = base.mark_line(strokeWidth=0.8, opacity=0.3, color="gray").encode(
            x=f"z{d1}:Q", y=f"z{d2}:Q",
            order="timepoint:Q",
        )
        highlight = (
            base.mark_circle(size=100, color="red", stroke="black", strokeWidth=1)
            .encode(x=f"z{d1}:Q", y=f"z{d2}:Q")
            .transform_filter(time_select)
        )
        c = (path + points + highlight).properties(width=250, height=250, title=f"z{d1} vs z{d2}")
        path_charts.append(c)

    # --- Vertical rule for time-series charts ---
    rule = (
        base.mark_rule(color="red", strokeWidth=1.5)
        .encode(x="timepoint:Q")
        .transform_filter(time_select)
    )

    # --- Latent dims over time ---
    latent_time_charts = []
    for d in range(min(latent_dim, 5)):
        field = f"z{d}"
        line = base.mark_line(color="#1f77b4", strokeWidth=1.5).encode(
            x=alt.X("timepoint:Q", title=""),
            y=alt.Y(f"{field}:Q", title=field),
            tooltip=["timepoint:Q", alt.Tooltip(f"{field}:Q", format=".4f")],
        )
        latent_time_charts.append(
            (line + rule).properties(width=chart_width, height=small_height)
        )

    # --- Light stimulus pattern ---
    light_chart = (
        base.mark_line(color="#f5a623", strokeWidth=1.5).encode(
            x=alt.X("timepoint:Q", title=""),
            y=alt.Y("light:Q", title="light"),
            tooltip=["timepoint:Q", alt.Tooltip("light:Q", format=".2f")],
        ) + rule
    ).properties(width=chart_width, height=small_height, title="Light stimulus")

    # --- State variable plots: original (black) + reconstructed (red dashed) ---
    state_charts = []
    for name in state_names:
        orig_line = base.mark_line(color="black", strokeWidth=1.2).encode(
            x=alt.X("timepoint:Q", title=""),
            y=alt.Y(f"{name}:Q", title=name),
            tooltip=["timepoint:Q", alt.Tooltip(f"{name}:Q", format=".4f")],
        )
        rec_line = base.mark_line(color="red", strokeWidth=1.2, strokeDash=[4, 2]).encode(
            x=alt.X("timepoint:Q"),
            y=alt.Y(f"{name}_recon:Q"),
            tooltip=["timepoint:Q", alt.Tooltip(f"{name}_recon:Q", format=".4f")],
        )
        c = (orig_line + rec_line + rule).properties(width=chart_width, height=small_height)
        state_charts.append(c)

    # --- Layout: left column (latent path + latent time), right column (light + state space) ---
    left_column = alt.vconcat(
        alt.hconcat(*path_charts).properties(title="Latent trajectory"),
        *latent_time_charts,
    )
    right_column = alt.vconcat(light_chart, *state_charts).properties(
        title="State space: original (black) vs reconstructed (red dashed)"
    )

    combined = (
        alt.hconcat(left_column, right_column)
        .add_params(traj_select, time_select)
        .resolve_scale(y="independent")
    )

    return mo.ui.altair_chart(combined)


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------

INTERACTIVE_VIEWS: list[tuple[str, Any, Any]] = [
    ("Trajectory Explorer", _has_trajectory_data, render_trajectory_explorer),
    ("2D Latent Space",     _wants_latent_2d,     render_latent_2d),
    ("3D Latent Space",     _wants_latent_3d,     render_latent_3d),
    ("KL Explorer",         _has_kl,              render_kl_explorer),
    ("Loss Curves",         _has_loss_history,     render_loss_curves),
]


def get_views(bundle: ExperimentBundle) -> dict[str, Any]:
    """Return {name: marimo_element} for all applicable visualizations."""
    views = {}
    for name, predicate, render_fn in INTERACTIVE_VIEWS:
        if predicate(bundle):
            views[name] = render_fn(bundle)
    return views
