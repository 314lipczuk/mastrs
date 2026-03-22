"""
Evaluation module for trained MLP VAE models on pathway state data.

Usage:
    from eval_vae import evaluate

    result = evaluate(model, states, traj_lengths, state_names)
    result.summary()
    result.figures["reconstruction_scatter"]
    result.save("results/vae_exp/")

The VAE interface: model(x) -> (recon, mu, logvar)
                   model.encoder(x) -> (mu, logvar)
                   model.decoder(z) -> recon
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
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
class VAEResult:
    name: str
    timestamp: str
    metrics: dict[str, np.ndarray | float]
    figures: dict[str, Figure]

    def summary(self) -> None:
        print(f"=== {self.name} ({self.timestamp}) ===")
        m = self.metrics
        print(f"  Samples        : {m['n_samples']}")
        print(f"  Latent dim     : {m['latent_dim']}")
        print(f"  Beta           : {m.get('beta', '?')}")
        print(f"  MSE overall    : {m['mse_overall']:.6f}")
        print(f"  MSE per state  :")
        for name, val in zip(m["state_names"], m["mse_per_state"]):
            print(f"    {name:>6s}: {val:.6f}")
        kl = m["kl_per_dim"]
        print(f"  KL per dim     : {np.array2string(kl, precision=4)}")
        print(f"  Active dims    : {m['n_active_dims']} / {m['latent_dim']}")
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
            z_mu=self.metrics["z_mu"],
            z_logvar=self.metrics["z_logvar"],
            kl_per_dim=self.metrics["kl_per_dim"],
            recon=self.metrics["recon"],
        )
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.summary()
        (d / "summary.txt").write_text(buf.getvalue())

    def _repr_html_(self) -> str:
        import io, base64
        html = f"<h3>{self.name}</h3><pre>"
        buf = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf):
            self.summary()
        html += buf.getvalue() + "</pre>"
        for fig_name in ("reconstruction_scatter", "kl_per_dim", "latent_colored_by_state"):
            if fig_name in self.figures:
                img_buf = io.BytesIO()
                self.figures[fig_name].savefig(img_buf, format="png", dpi=100, bbox_inches="tight")
                img_buf.seek(0)
                b64 = base64.b64encode(img_buf.read()).decode()
                html += f'<img src="data:image/png;base64,{b64}" />'
        return html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _encode_and_reconstruct(
    model: nn.Module, states: np.ndarray, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode and reconstruct all samples.

    Returns (z_mu, z_logvar, recon, states).
    Uses encoder mean (no sampling) for deterministic evaluation.
    """
    device = _get_device(model)
    x = torch.tensor(states, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    mus, logvars, recons = [], [], []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            mu, logvar = model.encoder(batch)
            recon = model.decoder(mu)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            recons.append(recon.cpu().numpy())
    return np.concatenate(mus), np.concatenate(logvars), np.concatenate(recons), states


# ---------------------------------------------------------------------------
# Plots — reconstruction
# ---------------------------------------------------------------------------

def _plot_reconstruction_error(
    mse_per_sample: np.ndarray, mse_per_state: np.ndarray, state_names: list[str]
) -> Figure:
    """Histogram of per-sample MSE (left) and bar chart of MSE per state (right)."""
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
    """True vs reconstructed scatter for each state variable."""
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


# ---------------------------------------------------------------------------
# Plots — KL and encoder uncertainty
# ---------------------------------------------------------------------------

def _plot_kl_per_dimension(
    kl_per_dim: np.ndarray, threshold: float
) -> Figure:
    """Bar chart of KL per latent dimension with active/inactive coloring."""
    L = len(kl_per_dim)
    active = kl_per_dim >= threshold
    colours = ["tab:blue" if a else "lightgrey" for a in active]
    fig, ax = plt.subplots(figsize=(max(4, L * 0.6), 4))
    ax.bar(range(L), kl_per_dim, color=colours, edgecolor="black", linewidth=0.5)
    ax.axhline(threshold, color="red", ls="--", lw=0.8, label=f"threshold={threshold}")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean KL divergence")
    ax.set_title(f"KL per dim — {active.sum()}/{L} active")
    ax.set_xticks(range(L))
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_encoder_uncertainty(z_logvar: np.ndarray) -> Figure:
    """Distribution of encoder uncertainty (sigma) across samples."""
    sigma = np.exp(0.5 * z_logvar)  # (N, L)
    mean_sigma = sigma.mean(axis=1)  # (N,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(mean_sigma, bins=60, edgecolor="black", alpha=0.7, linewidth=0.3)
    ax1.set_xlabel("Mean sigma (across latent dims)")
    ax1.set_ylabel("Count")
    ax1.set_title("Encoder uncertainty distribution")

    # Per-dimension boxplot
    ax2.boxplot([sigma[:, d] for d in range(sigma.shape[1])],
                labels=[f"z{d}" for d in range(sigma.shape[1])])
    ax2.set_ylabel("Sigma")
    ax2.set_title("Uncertainty per latent dimension")

    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plots — latent space
# ---------------------------------------------------------------------------

def _plot_latent_colored_by_state(
    z: np.ndarray, states: np.ndarray, state_names: list[str]
) -> Figure:
    """Latent scatter plots colored by each state variable."""
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
    """2D histogram of latent space occupancy."""
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


def _plot_latent_marginals(z: np.ndarray, kl_per_dim: np.ndarray) -> Figure:
    """Marginal distribution of each latent dimension, annotated with KL."""
    latent_dim = z.shape[1]
    fig, axes = plt.subplots(1, latent_dim, figsize=(3.5 * latent_dim, 3))
    if latent_dim == 1:
        axes = [axes]
    for d in range(latent_dim):
        axes[d].hist(z[:, d], bins=80, edgecolor="black", linewidth=0.3, alpha=0.7)
        axes[d].set_title(f"z{d} (std={z[:, d].std():.3f}, KL={kl_per_dim[d]:.4f})")
        axes[d].set_xlabel(f"z{d}")
    fig.suptitle("Latent marginal distributions", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_latent_colored_by_stimulus(
    z: np.ndarray, generator_labels: np.ndarray
) -> Figure:
    """Latent scatter colored by stimulus generator type."""
    latent_dim = z.shape[1]
    if latent_dim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 2 latent dims", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

    rng = np.random.default_rng(42)
    idx = rng.choice(len(z), size=min(10000, len(z)), replace=False)

    unique_labels = np.unique(generator_labels)
    cmap = plt.cm.tab10
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    pairs = list(combinations(range(min(latent_dim, 4)), 2))[:3]
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5), squeeze=False)
    for pi, (d1, d2) in enumerate(pairs):
        ax = axes[0, pi]
        for lab in unique_labels:
            mask = generator_labels[idx] == lab
            ax.scatter(z[idx[mask], d1], z[idx[mask], d2],
                       s=1, alpha=0.35, label=lab,
                       color=label_to_color[lab], rasterized=True)
        ax.set_xlabel(f"z{d1}")
        ax.set_ylabel(f"z{d2}")
        ax.set_title(f"z{d1} vs z{d2}")
        ax.legend(fontsize=7, markerscale=5)

    fig.suptitle("Latent space colored by stimulus generator", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plots — trajectory-aware
# ---------------------------------------------------------------------------

def _plot_latent_trajectories(
    z: np.ndarray, traj_lengths: np.ndarray, n_trajs: int = 12
) -> Figure:
    """Individual trajectories through latent space: time-series and 2D path."""
    latent_dim = z.shape[1]
    rng = np.random.default_rng(42)
    n_total = len(traj_lengths)
    chosen = rng.choice(n_total, size=min(n_trajs, n_total), replace=False)
    starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])

    if latent_dim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 2 latent dims", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

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


def _plot_trajectory_reconstruction(
    states: np.ndarray, recon: np.ndarray,
    traj_lengths: np.ndarray, state_names: list[str], n_trajs: int = 6
) -> Figure:
    """Overlay original and reconstructed trajectories over time."""
    rng = np.random.default_rng(42)
    n_total = len(traj_lengths)
    chosen = rng.choice(n_total, size=min(n_trajs, n_total), replace=False)
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
            ax.plot(t, states[s:e, vi], color="black", lw=1.0, label="original")
            ax.plot(t, recon[s:e, vi], color="tab:red", lw=1.0, ls="--", label="recon")
            ax.set_ylabel(name, fontsize=8)
            if ri == 0:
                ax.set_title(name, fontsize=9)
            if ri == len(chosen) - 1:
                ax.set_xlabel("timepoint", fontsize=8)
            ax.tick_params(labelsize=7)

        axes[ri, 0].annotate(
            f"traj {ti}", xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-40, 0), textcoords="offset points",
            fontsize=8, ha="right", va="center",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Trajectory reconstruction overlay", fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    plt.close(fig)
    return fig


def _plot_latent_velocity_consistency(
    states: np.ndarray, z: np.ndarray, traj_lengths: np.ndarray
) -> tuple[Figure, float]:
    """Compare finite-difference velocities in state space vs latent space."""
    starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])

    valid = np.ones(len(states), dtype=bool)
    for s, length in zip(starts, traj_lengths):
        valid[s + length - 1] = False

    valid_idx = np.where(valid)[0]
    dx = states[valid_idx + 1] - states[valid_idx]
    dz = z[valid_idx + 1] - z[valid_idx]

    speed_state = np.linalg.norm(dx, axis=1)
    speed_latent = np.linalg.norm(dz, axis=1)

    corr = float(np.corrcoef(speed_state, speed_latent)[0, 1])

    rng = np.random.default_rng(42)
    idx = rng.choice(len(speed_state), size=min(10000, len(speed_state)), replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.scatter(speed_state[idx], speed_latent[idx], s=1, alpha=0.2, rasterized=True)
    ax1.set_xlabel("||dx/dt|| (state space)")
    ax1.set_ylabel("||dz/dt|| (latent space)")
    ax1.set_title(f"Velocity consistency (r = {corr:.3f})")

    hb = ax2.hexbin(speed_state, speed_latent, gridsize=80, cmap="inferno", mincnt=1)
    fig.colorbar(hb, ax=ax2, label="count")
    ax2.set_xlabel("||dx/dt|| (state space)")
    ax2.set_ylabel("||dz/dt|| (latent space)")
    ax2.set_title(f"Velocity density (r = {corr:.3f})")

    fig.suptitle("Latent velocity consistency", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig, corr


# ---------------------------------------------------------------------------
# Topology preservation
# ---------------------------------------------------------------------------

def _compute_knn_overlap(
    states: np.ndarray, z: np.ndarray,
    ks: tuple[int, ...] = (10, 50), n_subsample: int = 2000
) -> dict[int, float]:
    """k-NN overlap between state space and latent space."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(42)
    n = len(states)
    n_sub = min(n_subsample, n)
    query_idx = rng.choice(n, size=n_sub, replace=False)

    max_k = max(ks)

    nn_state = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto")
    nn_latent = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto")
    nn_state.fit(states)
    nn_latent.fit(z)

    state_neighbors = nn_state.kneighbors(states[query_idx], return_distance=False)[:, 1:]
    latent_neighbors = nn_latent.kneighbors(z[query_idx], return_distance=False)[:, 1:]

    overlaps = {}
    for k in ks:
        shared = np.array([
            len(np.intersect1d(state_neighbors[i, :k], latent_neighbors[i, :k]))
            for i in range(n_sub)
        ])
        overlaps[k] = float(shared.mean() / k)

    return overlaps


# ---------------------------------------------------------------------------
# Latent traversals (unconditional decoder)
# ---------------------------------------------------------------------------

def _compute_latent_traversals(
    model: nn.Module, z_mu: np.ndarray, active_dims: np.ndarray,
    n_steps: int = 7, range_val: float = 2.0,
) -> dict[int, np.ndarray]:
    """Decode traversals along each active latent dimension."""
    if not active_dims.any():
        return {}
    device = _get_device(model)
    centroid = z_mu.mean(axis=0)
    ref_idx = int(np.argmin(np.linalg.norm(z_mu - centroid, axis=1)))
    ref_z = torch.tensor(z_mu[ref_idx], dtype=torch.float32, device=device)

    active_indices = np.where(active_dims)[0]
    traversals = {}
    with torch.no_grad():
        for dim in active_indices:
            vals = np.linspace(-range_val, range_val, n_steps)
            z_batch = ref_z.unsqueeze(0).repeat(n_steps, 1)
            z_batch[:, dim] = torch.tensor(vals, dtype=torch.float32, device=device)
            decoded = model.decoder(z_batch)
            traversals[int(dim)] = decoded.cpu().numpy()
    return traversals


def _plot_latent_traversals(
    traversals: dict[int, np.ndarray], state_names: list[str], range_val: float
) -> Figure:
    """Plot decoded state vectors across latent traversals.

    For single-timepoint VAEs, each decoded output is a state vector, not a
    time series. Shows how each state variable changes as we sweep along
    a latent dimension.
    """
    n_dims = len(traversals)
    if n_dims == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No active dimensions", ha="center", va="center",
                transform=ax.transAxes)
        plt.close(fig)
        return fig

    ncols = min(4, n_dims)
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, (dim, decoded) in enumerate(sorted(traversals.items())):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        n_steps = decoded.shape[0]
        x_vals = np.linspace(-range_val, range_val, n_steps)
        for si, name in enumerate(state_names):
            ax.plot(x_vals, decoded[:, si], label=name, lw=1.2)
        ax.set_xlabel(f"z{dim} value")
        ax.set_ylabel("State value")
        ax.set_title(f"Traversal dim {dim}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(len(traversals), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)
    fig.suptitle("Latent traversals", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    states: np.ndarray,
    traj_lengths: np.ndarray,
    state_names: list[str],
    name: str = "vae_eval",
    batch_size: int = 1024,
    generator_labels: np.ndarray | None = None,
    kl_active_threshold: float = 0.05,
    traversal_range: float = 2.0,
    n_traversal_steps: int = 7,
) -> VAEResult:
    """
    Evaluate a VAE on pathway state data.

    Args:
        model: trained VAE with model(x) -> (recon, mu, logvar)
        states: (N, n_vars) array of all timepoints
        traj_lengths: (n_trajectories,) array — length of each trajectory
        state_names: list of state variable names (e.g. ["RAS", "RAF", ...])
        name: experiment name
        batch_size: for encoding
        generator_labels: (N,) string array — stimulus generator type per
            timepoint. If provided, produces latent-space plot colored by
            generator.
        kl_active_threshold: KL threshold for considering a dimension active
        traversal_range: range for latent traversals (+/- this value)
        n_traversal_steps: number of steps in latent traversals
    """
    model.eval()

    # --- encode & reconstruct ---
    z_mu, z_logvar, recon, states = _encode_and_reconstruct(model, states, batch_size)

    # --- MSE ---
    mse_per_sample = np.mean((recon - states) ** 2, axis=1)
    mse_per_state = np.mean((recon - states) ** 2, axis=0)
    mse_overall = float(mse_per_sample.mean())

    # --- KL ---
    kl_per_dim = -0.5 * np.mean(1 + z_logvar - z_mu**2 - np.exp(z_logvar), axis=0)
    active_dims = kl_per_dim >= kl_active_threshold

    # --- Traversals ---
    traversals = _compute_latent_traversals(
        model, z_mu, active_dims, n_traversal_steps, traversal_range
    )

    # --- Velocity consistency ---
    vel_fig, vel_corr = _plot_latent_velocity_consistency(z_mu, states, traj_lengths)

    # --- k-NN overlap ---
    knn_overlap = _compute_knn_overlap(states, z_mu)

    # --- Figures ---
    figures = {}
    figures["reconstruction_error"] = _plot_reconstruction_error(
        mse_per_sample, mse_per_state, state_names
    )
    figures["reconstruction_scatter"] = _plot_reconstruction_scatter(
        states, recon, state_names
    )
    figures["kl_per_dim"] = _plot_kl_per_dimension(kl_per_dim, kl_active_threshold)
    figures["encoder_uncertainty"] = _plot_encoder_uncertainty(z_logvar)
    figures["latent_colored_by_state"] = _plot_latent_colored_by_state(
        z_mu, states, state_names
    )
    figures["latent_density"] = _plot_latent_density(z_mu)
    figures["latent_marginals"] = _plot_latent_marginals(z_mu, kl_per_dim)
    figures["latent_trajectories"] = _plot_latent_trajectories(z_mu, traj_lengths)
    figures["trajectory_reconstruction"] = _plot_trajectory_reconstruction(
        states, recon, traj_lengths, state_names
    )
    figures["latent_velocity_consistency"] = vel_fig
    figures["latent_traversals"] = _plot_latent_traversals(
        traversals, state_names, traversal_range
    )

    if generator_labels is not None:
        figures["latent_colored_by_stimulus"] = _plot_latent_colored_by_stimulus(
            z_mu, generator_labels
        )

    # --- Metrics ---
    beta = float(model.beta) if hasattr(model, "beta") else None
    metrics = {
        "n_samples": len(states),
        "latent_dim": z_mu.shape[1],
        "beta": beta,
        "mse_overall": mse_overall,
        "mse_per_sample": mse_per_sample,
        "mse_per_state": mse_per_state,
        "state_names": state_names,
        "z_mu": z_mu,
        "z_logvar": z_logvar,
        "recon": recon,
        "kl_per_dim": kl_per_dim,
        "n_active_dims": int(active_dims.sum()),
        "active_dims": active_dims,
        "velocity_correlation": vel_corr,
        "knn_overlap": knn_overlap,
    }

    return VAEResult(
        name=name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        figures=figures,
    )
