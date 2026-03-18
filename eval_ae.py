"""
Evaluation module for trained AutoEncoder models on pathway state data.

Usage:
    from eval_ae import evaluate

    result = evaluate(model, states, traj_lengths, state_names)
    result.summary()
    result.figures["latent_colored_by_state"]
    result.save("results/ae_states/")

To load a saved checkpoint and display all figures without retraining:
    from eval_ae import evaluate_checkpoint

    result = evaluate_checkpoint("results/ae_states/model_20260317_113736.pt")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AEResult:
    name: str
    timestamp: str
    metrics: dict[str, np.ndarray | float]
    figures: dict[str, Figure]

    def summary(self) -> None:
        print(f"=== {self.name} ({self.timestamp}) ===")
        m = self.metrics
        print(f"  Samples        : {m['n_samples']}")
        print(f"  Latent dim     : {m['latent_dim']}")
        print(f"  MSE overall    : {m['mse_overall']:.6f}")
        print(f"  MSE per state  :")
        for name, val in zip(m["state_names"], m["mse_per_state"]):
            print(f"    {name:>6s}: {val:.6f}")
        if "velocity_correlation" in m:
            print(f"  Velocity corr  : {m['velocity_correlation']:.4f}")
        if "knn_overlap" in m:
            for k, v in m["knn_overlap"].items():
                print(f"  k-NN overlap (k={k}): {v:.4f}")
        print()

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        fig_dir = d / "figures"
        fig_dir.mkdir(exist_ok=True)
        for name, fig in self.figures.items():
            fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        np.savez(
            d / "metrics.npz",
            mse_per_sample=self.metrics["mse_per_sample"],
            mse_per_state=self.metrics["mse_per_state"],
            z=self.metrics["z"],
            recon=self.metrics["recon"],
        )
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.summary()
        (d / "summary.txt").write_text(buf.getvalue())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _encode_and_reconstruct(
    model: nn.Module, states: np.ndarray, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    device = _get_device(model)
    x = torch.tensor(states, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    zs, recons = [], []
    with torch.no_grad():
        for (batch,) in loader:
            recon, z = model(batch.to(device))
            zs.append(z.cpu().numpy())
            recons.append(recon.cpu().numpy())
    return np.concatenate(zs), np.concatenate(recons)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_reconstruction_error(
    mse_per_sample: np.ndarray, mse_per_state: np.ndarray, state_names: list[str]
) -> Figure:
    """Distribution of reconstruction error across samples and state variables.

    Left panel: histogram of per-sample MSE with the mean marked as a
    dashed red line. A tight, low-mean distribution indicates consistent
    reconstruction quality. A long right tail reveals outlier samples the
    model struggles with.

    Right panel: bar chart of MSE broken down by state variable. Uneven
    bars highlight which biological variables are harder to reconstruct
    (e.g. fast-changing states like ERK vs slowly-varying ones like RAF).

    Args:
        mse_per_sample: (N,) MSE for each timepoint.
        mse_per_state: (n_vars,) mean MSE for each state variable.
        state_names: human-readable names for each state variable.

    Returns:
        Figure with the two panels described above.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(mse_per_sample, bins=80, edgecolor="black", alpha=0.7, linewidth=0.3)
    ax1.axvline(mse_per_sample.mean(), color="red", ls="--",
                label=f"mean={mse_per_sample.mean():.5f}")
    ax1.set_xlabel("MSE")
    ax1.set_ylabel("Count")
    ax1.set_title("Per-sample MSE")
    ax1.legend(fontsize=8)

    ax2.bar(state_names, mse_per_state, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE by state variable")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_reconstruction_scatter(
    states: np.ndarray, recon: np.ndarray, state_names: list[str]
) -> Figure:
    """True vs reconstructed value scatter for each state variable.

    Each subplot shows one state variable with the true value on the x-axis
    and the autoencoder's reconstruction on the y-axis. A red dashed diagonal
    marks perfect reconstruction. Points tightly clustered along the diagonal
    indicate accurate reconstruction; systematic deviations (curves, offsets,
    spread at certain value ranges) reveal where the model is biased or
    imprecise.

    Interpretation:
    - Tight band along diagonal = good reconstruction for that variable.
    - Spread widening at extremes = model struggles with rare/extreme values.
    - Curved deviation from diagonal = systematic nonlinear bias.
    - Points with equal-aspect axes so distortion is visually honest.

    Args:
        states: (N, n_vars) original state data.
        recon: (N, n_vars) reconstructed state data.
        state_names: human-readable names for each state variable.

    Returns:
        Figure with one scatter subplot per state variable.
    """
    n_vars = len(state_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 3.5))
    if n_vars == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(states), size=min(5000, len(states)), replace=False)
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.scatter(states[idx, i], recon[idx, i], s=1, alpha=0.3)
        lims = [min(states[idx, i].min(), recon[idx, i].min()),
                max(states[idx, i].max(), recon[idx, i].max())]
        ax.plot(lims, lims, "r--", lw=0.8)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"recon {name}")
        ax.set_aspect("equal")
        ax.set_title(name)
    fig.suptitle("Reconstruction: true vs predicted", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_latent_colored_by_state(
    z: np.ndarray, states: np.ndarray, state_names: list[str]
) -> Figure:
    """Latent space scatter plots colored by each original state variable.

    Each row corresponds to one state variable, each column to a pair of
    latent dimensions. Points are colored by the state variable's value
    using a viridis colormap.

    Interpretation:
    - Smooth color gradients across the latent space indicate that the
      latent representation captures that state variable's variation in
      an organized, continuous way — the AE has learned a meaningful
      embedding for that variable.
    - Patchy or random coloring means the latent space does not encode
      that variable's information in a geometrically structured way.
    - If two state variables show similar color patterns on the same
      latent-dimension pair, those variables are entangled in the latent
      representation (may or may not be desirable).

    Args:
        z: (N, latent_dim) latent codes.
        states: (N, n_vars) original state data.
        state_names: human-readable names for each state variable.

    Returns:
        Figure with (n_vars x n_latent_pairs) grid of scatter plots.
    """
    n_vars = len(state_names)
    latent_dim = z.shape[1]
    if latent_dim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 2 latent dims", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

    rng = np.random.default_rng(42)
    idx = rng.choice(len(z), size=min(8000, len(z)), replace=False)

    # use first 2 latent dims (or first 3 pairs if latent_dim >= 3)
    from itertools import combinations
    pairs = list(combinations(range(min(latent_dim, 4)), 2))[:3]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(n_vars, n_pairs, figsize=(4 * n_pairs, 3.2 * n_vars),
                             squeeze=False)
    for vi, name in enumerate(state_names):
        vals = states[idx, vi]
        for pi, (d1, d2) in enumerate(pairs):
            ax = axes[vi, pi]
            sc = ax.scatter(z[idx, d1], z[idx, d2], c=vals, s=1, alpha=0.4,
                            cmap="viridis", rasterized=True)
            ax.set_xlabel(f"z{d1}")
            ax.set_ylabel(f"z{d2}")
            if pi == 0:
                ax.set_ylabel(f"{name}\nz{d2}")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Latent space colored by state variables", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_latent_density(z: np.ndarray) -> Figure:
    """2D histogram of latent space occupancy for each pair of latent dims.

    Shows where the encoded data concentrates in latent space using a
    heatmap (inferno colormap, 80x80 bins).

    Interpretation:
    - A single tight cluster suggests the AE is using only a small region
      of the available latent space — the effective dimensionality may be
      lower than the nominal latent_dim.
    - Multiple separated clusters indicate the AE has discovered discrete
      modes in the data (e.g. different dynamical regimes).
    - A diffuse, roughly uniform cloud means the AE is spreading data
      across the full latent space, which is favorable for downstream
      interpolation and dynamics fitting.
    - Holes or voids in the density suggest regions of latent space that
      don't correspond to valid data — decoding from these regions may
      produce unrealistic states.

    Args:
        z: (N, latent_dim) latent codes.

    Returns:
        Figure with one 2D histogram per pair of latent dimensions.
    """
    from itertools import combinations
    latent_dim = z.shape[1]
    if latent_dim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 2 latent dims", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

    pairs = list(combinations(range(min(latent_dim, 4)), 2))[:3]
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 4), squeeze=False)

    for pi, (d1, d2) in enumerate(pairs):
        ax = axes[0, pi]
        ax.hist2d(z[:, d1], z[:, d2], bins=80, cmap="inferno")
        ax.set_xlabel(f"z{d1}")
        ax.set_ylabel(f"z{d2}")
        ax.set_title(f"z{d1} vs z{d2}")
    fig.suptitle("Latent density", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_latent_trajectories(
    z: np.ndarray, traj_lengths: np.ndarray, n_trajs: int = 12
) -> Figure:
    """Visualize how individual trajectories evolve through latent space.

    Left panel: time-series plot of latent dimension values over time for
    a random selection of trajectories. Each trajectory is a distinct color;
    all latent dims for that trajectory share the same color.

    Right panel: 2D path through the first two latent dimensions (z0 vs z1).
    Circle markers = trajectory start, X markers = trajectory end.

    Interpretation:
    - Smooth, non-erratic curves in the left panel indicate the latent
      representation varies continuously over time — desirable for fitting
      dynamics models (ODEs/SDEs) in latent space.
    - Jagged or noisy traces suggest the AE is encoding high-frequency
      noise rather than smooth dynamics.
    - In the right panel, trajectories that trace structured paths (loops,
      arcs) rather than random clouds indicate the latent space preserves
      temporal coherence. Trajectories that converge to the same region
      suggest a shared attractor in latent space.

    Args:
        z: (N, latent_dim) latent codes, trajectories concatenated.
        traj_lengths: (n_trajectories,) length of each trajectory.
        n_trajs: number of trajectories to plot.

    Returns:
        Figure with latent time-series (left) and 2D path (right).
    """
    latent_dim = z.shape[1]
    rng = np.random.default_rng(42)
    n_total = len(traj_lengths)
    chosen = rng.choice(n_total, size=min(n_trajs, n_total), replace=False)

    # compute trajectory start indices
    starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])

    if latent_dim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 2 latent dims", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

    # plot 1: latent dims over time; plot 2: 2D trajectory paths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab10

    for ci, ti in enumerate(chosen):
        s = starts[ti]
        e = s + traj_lengths[ti]
        zt = z[s:e]
        color = cmap(ci / max(1, len(chosen) - 1))
        for d in range(min(latent_dim, 3)):
            ax1.plot(zt[:, d], color=color, alpha=0.6, lw=0.8,
                     label=f"traj {ti}" if d == 0 else None)

    ax1.set_xlabel("Timepoint")
    ax1.set_ylabel("Latent value")
    ax1.set_title("Latent dims over time")

    for ci, ti in enumerate(chosen):
        s = starts[ti]
        e = s + traj_lengths[ti]
        zt = z[s:e]
        color = cmap(ci / max(1, len(chosen) - 1))
        ax2.plot(zt[:, 0], zt[:, 1], color=color, alpha=0.6, lw=0.8)
        ax2.scatter(zt[0, 0], zt[0, 1], color=color, s=20, zorder=5, marker="o")
        ax2.scatter(zt[-1, 0], zt[-1, 1], color=color, s=20, zorder=5, marker="x")

    ax2.set_xlabel("z0")
    ax2.set_ylabel("z1")
    ax2.set_title("Latent trajectories (o=start, x=end)")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_latent_marginals(z: np.ndarray) -> Figure:
    """Marginal (1D) distribution of each latent dimension.

    One histogram per latent dimension, with the standard deviation shown
    in the subplot title.

    Interpretation:
    - Roughly Gaussian marginals with similar standard deviations across
      dimensions suggest balanced use of the latent space — no single
      dimension dominates or collapses.
    - A near-zero std for a dimension means it has collapsed (carries no
      information) and the effective latent dimensionality is lower than
      nominal — consider reducing latent_dim.
    - Multimodal histograms indicate the AE has discovered discrete
      clusters along that dimension, which may correspond to distinct
      dynamical regimes in the data.
    - Very different scales across dimensions may cause problems for
      downstream models that assume isotropic distances in latent space.

    Args:
        z: (N, latent_dim) latent codes.

    Returns:
        Figure with one histogram per latent dimension.
    """
    latent_dim = z.shape[1]
    fig, axes = plt.subplots(1, latent_dim, figsize=(3.5 * latent_dim, 3))
    if latent_dim == 1:
        axes = [axes]
    for d in range(latent_dim):
        axes[d].hist(z[:, d], bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
        axes[d].set_title(f"z{d} (std={z[:, d].std():.3f})")
        axes[d].set_xlabel(f"z{d}")
    fig.suptitle("Latent marginal distributions", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_latent_colored_by_stimulus(
    z: np.ndarray,
    generator_labels: np.ndarray,
) -> Figure:
    """Scatter of latent space colored by stimulus generator type.

    Each point is colored by its generator label (e.g. "stochastic",
    "sequential", "functional", "smoothed"). Clean separation between
    colors indicates the autoencoder captures input-regime structure
    even though it never sees the stimulus directly.

    Args:
        z: (N, latent_dim) latent codes for all timepoints.
        generator_labels: (N,) string label for each timepoint indicating
            which stimulus generator produced the trajectory it belongs to.

    Returns:
        Figure with one subplot per latent-dimension pair, points colored
        by generator type.
    """
    from itertools import combinations

    latent_dim = z.shape[1]
    if latent_dim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 2 latent dims", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

    # Subsample for plotting speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(z), size=min(10000, len(z)), replace=False)

    # Determine unique generator types and assign a color to each
    unique_labels = np.unique(generator_labels)
    cmap = plt.cm.tab10
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    # Plot up to 3 pairs of latent dimensions
    pairs = list(combinations(range(min(latent_dim, 4)), 2))[:3]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5), squeeze=False)
    for pi, (d1, d2) in enumerate(pairs):
        ax = axes[0, pi]
        # Plot each generator type separately so it appears in the legend
        for lab in unique_labels:
            mask = generator_labels[idx] == lab
            ax.scatter(
                z[idx[mask], d1], z[idx[mask], d2],
                s=1, alpha=0.35, label=lab,
                color=label_to_color[lab], rasterized=True,
            )
        ax.set_xlabel(f"z{d1}")
        ax.set_ylabel(f"z{d2}")
        ax.set_title(f"z{d1} vs z{d2}")
        ax.legend(fontsize=7, markerscale=5)

    fig.suptitle("Latent space colored by stimulus generator", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _compute_knn_overlap(
    states: np.ndarray,
    z: np.ndarray,
    ks: tuple[int, ...] = (10, 50),
    n_subsample: int = 2000,
) -> dict[int, float]:
    """Compute k-nearest-neighbor overlap between state space and latent space.

    For a random subsample of points, find the k nearest neighbors in
    both the original state space and the latent space, then report the
    fraction of neighbors shared between the two. High overlap means
    the autoencoder preserves local topology — critical if downstream
    models (e.g. dynamics models) will operate in latent space.

    Args:
        states: (N, n_vars) original state-space data.
        z: (N, latent_dim) corresponding latent codes.
        ks: tuple of neighborhood sizes to evaluate (e.g. (10, 50)).
        n_subsample: number of query points to evaluate (random subset
            for computational tractability).

    Returns:
        Dictionary mapping each k to the mean fraction of shared neighbors
        (float in [0, 1]). E.g. {10: 0.72, 50: 0.65}.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(42)
    n = len(states)
    n_sub = min(n_subsample, n)
    query_idx = rng.choice(n, size=n_sub, replace=False)

    max_k = max(ks)

    # Fit kNN in both spaces (k+1 because the point itself is a neighbor)
    nn_state = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto")
    nn_latent = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto")
    nn_state.fit(states)
    nn_latent.fit(z)

    # Query only the subsample points
    state_neighbors = nn_state.kneighbors(states[query_idx], return_distance=False)
    latent_neighbors = nn_latent.kneighbors(z[query_idx], return_distance=False)

    # Column 0 is the point itself — drop it
    state_neighbors = state_neighbors[:, 1:]
    latent_neighbors = latent_neighbors[:, 1:]

    overlaps = {}
    for k in ks:
        # For each query point, compute |intersection| / k
        shared = np.array([
            len(np.intersect1d(state_neighbors[i, :k], latent_neighbors[i, :k]))
            for i in range(n_sub)
        ])
        overlaps[k] = float(shared.mean() / k)

    return overlaps


def _plot_trajectory_reconstruction(
    states: np.ndarray,
    recon: np.ndarray,
    traj_lengths: np.ndarray,
    state_names: list[str],
    n_trajs: int = 6,
) -> Figure:
    """Overlay original and reconstructed trajectories over time.

    Unlike the reconstruction scatter plot (which pools all timepoints and
    hides temporal structure), this plot shows each state variable as a
    time series. Systematic errors at transients vs. steady state become
    immediately visible.

    Args:
        states: (N, n_vars) original state data, trajectories concatenated.
        recon: (N, n_vars) reconstructed states from the autoencoder.
        traj_lengths: (n_trajectories,) length of each trajectory.
        state_names: human-readable names for each state variable.
        n_trajs: number of randomly chosen trajectories to display.

    Returns:
        Figure with one row per trajectory and one column per state variable.
        Solid lines = original, dashed lines = reconstruction.
    """
    rng = np.random.default_rng(42)
    n_total = len(traj_lengths)
    chosen = rng.choice(n_total, size=min(n_trajs, n_total), replace=False)

    # Precompute the start index of each trajectory in the flat array
    starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])

    n_vars = len(state_names)
    fig, axes = plt.subplots(
        len(chosen), n_vars,
        figsize=(3 * n_vars, 2.2 * len(chosen)),
        squeeze=False,
    )

    for ri, ti in enumerate(chosen):
        s = starts[ti]
        e = s + traj_lengths[ti]
        t = np.arange(traj_lengths[ti])

        for vi, name in enumerate(state_names):
            ax = axes[ri, vi]
            # Original trajectory as solid line
            ax.plot(t, states[s:e, vi], color="black", lw=1.0, label="original")
            # Reconstruction as dashed colored line
            ax.plot(t, recon[s:e, vi], color="tab:red", lw=1.0, ls="--",
                    label="recon")
            ax.set_ylabel(name, fontsize=8)
            if ri == 0:
                ax.set_title(name, fontsize=9)
            if ri == len(chosen) - 1:
                ax.set_xlabel("timepoint", fontsize=8)
            ax.tick_params(labelsize=7)

        # Label the row with trajectory index
        axes[ri, 0].annotate(
            f"traj {ti}", xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-40, 0), textcoords="offset points",
            fontsize=8, ha="right", va="center",
        )

    # Single legend from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Trajectory reconstruction overlay", fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    plt.close(fig)
    return fig


def _plot_latent_velocity_consistency(
    states: np.ndarray,
    z: np.ndarray,
    traj_lengths: np.ndarray,
) -> tuple[Figure, float]:
    """Compare finite-difference velocities in state space vs latent space.

    For each consecutive pair of timepoints within a trajectory, compute
    the velocity (finite difference) in both state space and latent space.
    If the autoencoder preserves dynamics, the speed in latent space should
    correlate with the speed in state space — fast state-space transitions
    should map to fast latent-space transitions.

    Left panel: scatter of ||dx/dt|| vs ||dz/dt|| with Pearson correlation.
    Right panel: 2D histogram (hexbin) of the same, for density visibility.

    Args:
        states: (N, n_vars) original state data, trajectories concatenated.
        z: (N, latent_dim) latent codes, same ordering as states.
        traj_lengths: (n_trajectories,) length of each trajectory.

    Returns:
        Figure with scatter and hexbin of state-space vs latent-space speeds.
    """
    # Compute finite differences within each trajectory, avoiding
    # differences that span trajectory boundaries.
    starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])

    # Build a mask of valid consecutive-pair indices (exclude last point
    # of each trajectory, since its forward difference crosses a boundary)
    valid = np.ones(len(states), dtype=bool)
    for s, length in zip(starts, traj_lengths):
        valid[s + length - 1] = False  # last timepoint of each trajectory

    # dx[i] = states[i+1] - states[i], only for valid i
    valid_idx = np.where(valid)[0]
    dx = states[valid_idx + 1] - states[valid_idx]  # (M, n_vars)
    dz = z[valid_idx + 1] - z[valid_idx]            # (M, latent_dim)

    # Compute speeds (L2 norm of velocity vectors)
    speed_state = np.linalg.norm(dx, axis=1)  # (M,)
    speed_latent = np.linalg.norm(dz, axis=1)  # (M,)

    # Pearson correlation between the two speed profiles
    corr = float(np.corrcoef(speed_state, speed_latent)[0, 1])

    # Subsample for scatter plot clarity
    rng = np.random.default_rng(42)
    idx = rng.choice(len(speed_state), size=min(10000, len(speed_state)), replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: scatter plot
    ax1.scatter(speed_state[idx], speed_latent[idx], s=1, alpha=0.2, rasterized=True)
    ax1.set_xlabel("||dx/dt|| (state space)")
    ax1.set_ylabel("||dz/dt|| (latent space)")
    ax1.set_title(f"Velocity consistency (r = {corr:.3f})")

    # Right: hexbin for density
    hb = ax2.hexbin(
        speed_state, speed_latent,
        gridsize=80, cmap="inferno", mincnt=1,
    )
    fig.colorbar(hb, ax=ax2, label="count")
    ax2.set_xlabel("||dx/dt|| (state space)")
    ax2.set_ylabel("||dz/dt|| (latent space)")
    ax2.set_title(f"Velocity density (r = {corr:.3f})")

    fig.suptitle("Latent velocity consistency", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig, corr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    states: np.ndarray,
    traj_lengths: np.ndarray,
    state_names: list[str],
    name: str = "ae_eval",
    batch_size: int = 1024,
    generator_labels: np.ndarray | None = None,
) -> AEResult:
    """
    Evaluate an AutoEncoder on pathway state data.

    Args:
        model: trained AutoEncoder
        states: (N, n_vars) array of all timepoints
        traj_lengths: (n_trajectories,) array — length of each trajectory
        state_names: list of state variable names (e.g. ["RAS", "RAF", ...])
        name: experiment name
        batch_size: for encoding
        generator_labels: (N,) string array — stimulus generator type per
            timepoint (e.g. "stochastic", "sequential"). If provided,
            produces a latent-space plot colored by generator and computes
            k-NN neighborhood overlap between state and latent spaces.
    """
    model.eval()
    z, recon = _encode_and_reconstruct(model, states, batch_size)

    # --- MSE ---
    mse_per_sample = np.mean((recon - states) ** 2, axis=1)
    mse_per_state = np.mean((recon - states) ** 2, axis=0)
    mse_overall = float(mse_per_sample.mean())

    # --- Figures ---
    figures = {}
    figures["reconstruction_error"] = _plot_reconstruction_error(
        mse_per_sample, mse_per_state, state_names
    )
    figures["reconstruction_scatter"] = _plot_reconstruction_scatter(
        states, recon, state_names
    )
    figures["latent_colored_by_state"] = _plot_latent_colored_by_state(
        z, states, state_names
    )
    figures["latent_density"] = _plot_latent_density(z)
    figures["latent_trajectories"] = _plot_latent_trajectories(z, traj_lengths)
    figures["latent_marginals"] = _plot_latent_marginals(z)
    figures["trajectory_reconstruction"] = _plot_trajectory_reconstruction(
        states, recon, traj_lengths, state_names
    )

    # Velocity consistency returns both the figure and the correlation scalar
    vel_fig, vel_corr = _plot_latent_velocity_consistency(z=z, states=states, traj_lengths=traj_lengths)
    figures["latent_velocity_consistency"] = vel_fig

    # --- Optional: stimulus-dependent plots ---
    if generator_labels is not None:
        figures["latent_colored_by_stimulus"] = _plot_latent_colored_by_stimulus(
            z, generator_labels
        )

    # --- k-NN overlap (topology preservation) ---
    knn_overlap = _compute_knn_overlap(states, z)

    metrics = {
        "n_samples": len(states),
        "latent_dim": z.shape[1],
        "mse_overall": mse_overall,
        "mse_per_sample": mse_per_sample,
        "mse_per_state": mse_per_state,
        "state_names": state_names,
        "z": z,
        "recon": recon,
        "velocity_correlation": vel_corr,
        "knn_overlap": knn_overlap,
    }

    return AEResult(
        name=name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        figures=figures,
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    parquet_path: str = "synthetic_EGFR_data.parquet",
    display_figures: bool = True,
) -> AEResult:
    """Load a saved model checkpoint and run full evaluation with all figures.

    This is a convenience function for inspecting a trained model without
    rerunning training. It loads the model from a .pt checkpoint (using the
    saved model_config), prepares the test set from the parquet data using
    the same train/test split, runs evaluate(), prints the summary, and
    optionally displays all figures (for use in Jupyter notebooks).

    Args:
        checkpoint_path: path to a .pt file saved by the training notebooks.
            Must contain "model_state_dict" and ideally "model_config".
        parquet_path: path to the synthetic data parquet file.
        display_figures: if True, call display() on each figure (requires
            IPython/Jupyter). If False, figures are still available in the
            returned AEResult.

    Returns:
        AEResult with all metrics and figures.

    Example:
        from eval_ae import evaluate_checkpoint
        result = evaluate_checkpoint("results/ae_states/model_20260317_113736.pt")
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from model.dl import AutoEncoder

    # --- Load checkpoint ---
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_config" in ckpt:
        cfg = ckpt["model_config"]
        model = AutoEncoder(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
        )
    else:
        # Fallback: infer architecture from state_dict layer shapes
        sd = ckpt["model_state_dict"]
        input_dim = sd["encoder.net.0.weight"].shape[1]
        encoder_weights = [k for k in sd if "encoder" in k and "weight" in k]
        hidden_dims = tuple(sd[k].shape[0] for k in encoder_weights[:-1])
        latent_dim = sd[encoder_weights[-1]].shape[0]
        model = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"])

    # --- Prepare test data (same split as training) ---
    STATE_COLS = ["RAS_s", "RAF_s", "MEK_s", "NFB_s", "ERK_s"]
    df = pd.read_parquet(parquet_path)
    all_states = np.stack(
        [np.concatenate(df[c].values) for c in STATE_COLS], axis=1
    ).astype(np.float32)

    traj_ids = np.arange(len(df))
    _, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)

    traj_len = len(df[STATE_COLS[0]].iloc[0])
    test_idx = np.concatenate(
        [np.arange(i * traj_len, (i + 1) * traj_len) for i in te_ids]
    )
    test_states = all_states[test_idx]
    test_traj_lengths = np.full(len(te_ids), traj_len)
    test_generator_labels = np.repeat(df["generator"].values[te_ids], traj_len)
    state_names = [c.replace("_s", "") for c in STATE_COLS]

    # --- Build label ---
    ts = ckpt.get("train_start", Path(checkpoint_path).stem)
    if "model_config" in ckpt:
        cfg = ckpt["model_config"]
        label = f"H{cfg['hidden_dims']}_L{cfg['latent_dim']} ({ts})"
    else:
        label = Path(checkpoint_path).stem

    # --- Evaluate ---
    result = evaluate(
        model=model,
        states=test_states,
        traj_lengths=test_traj_lengths,
        state_names=state_names,
        name=label,
        generator_labels=test_generator_labels,
    )

    # --- Display ---
    result.summary()
    if display_figures:
        try:
            from IPython.display import display
            for fig_name, fig in result.figures.items():
                print(f"--- {fig_name} ---")
                display(fig)
        except ImportError:
            print("IPython not available; access figures via result.figures dict")

    return result
