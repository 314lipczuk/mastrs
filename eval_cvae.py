"""
Evaluation module for trained CVAE models.

Usage:
    from eval_cvae import evaluate, compare_experiments

    result = evaluate(model, dataset, metadata)
    result.summary()
    result.figures["reconstructions"]
    result.save("results/exp01/")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name: str
    timestamp: str
    model_config: dict
    metrics: dict[str, np.ndarray | float | int]
    figures: dict[str, Figure]

    def summary(self) -> None:
        print(f"=== {self.name} ({self.timestamp}) ===")
        cfg = self.model_config
        print(f"  latent_dim     : {cfg.get('latent_dim', '?')}")
        print(f"  beta           : {cfg.get('beta', '?')}")
        print(f"  total_params   : {cfg.get('total_params', '?'):,}")
        print(f"  trainable      : {cfg.get('trainable_params', '?'):,}")
        print()
        m = self.metrics
        mse = m["mse_per_cell"]
        print(f"  MSE  mean={mse.mean():.6f}  median={np.median(mse):.6f}  std={mse.std():.6f}")
        print(f"  Active dims    : {m['n_active_dims']} / {cfg.get('latent_dim', '?')}")
        print(f"  KL per dim     : {np.array2string(m['kl_per_dim'], precision=4)}")
        if "condition_clf_acc" in m:
            print(f"  Condition clf  : acc={m['condition_clf_acc']:.3f}  chance={m['condition_clf_chance']:.3f}")
        print()

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        # figures
        fig_dir = d / "figures"
        fig_dir.mkdir(exist_ok=True)
        for name, fig in self.figures.items():
            fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        # metrics
        arrays = {k: v for k, v in self.metrics.items() if isinstance(v, np.ndarray)}
        scalars = {k: v for k, v in self.metrics.items() if not isinstance(v, np.ndarray)}
        np.savez(d / "metrics.npz", **arrays)
        pd.Series(scalars).to_json(d / "scalars.json")
        # summary
        import io
        buf = io.StringIO()
        import contextlib
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
        for name in ("reconstructions", "kl_per_dim", "latent_space"):
            if name in self.figures:
                img_buf = io.BytesIO()
                self.figures[name].savefig(img_buf, format="png", dpi=100, bbox_inches="tight")
                img_buf.seek(0)
                b64 = base64.b64encode(img_buf.read()).decode()
                html += f'<img src="data:image/png;base64,{b64}" />'
        return html


# ---------------------------------------------------------------------------
# Model config discovery
# ---------------------------------------------------------------------------

def _discover_model_config(model: nn.Module) -> dict:
    config: dict = {}
    for name, param in model.named_parameters():
        if "fc_mu.weight" in name:
            config["latent_dim"] = param.shape[0]
        if "fc_logvar.weight" in name:
            config["latent_dim_check"] = param.shape[0]
    if hasattr(model, "beta"):
        config["beta"] = float(model.beta)
    config["total_params"] = sum(p.numel() for p in model.parameters())
    config["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return config


def _get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Encoding / reconstruction helpers
# ---------------------------------------------------------------------------

def _encode_all(
    model: nn.Module, dataset, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    device = _get_device(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mus, logvars = [], []
    with torch.no_grad():
        for encoder_input, stim_cond, target in loader:
            encoder_input = encoder_input.float().to(device)
            mu, logvar = model.encoder(encoder_input)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    return np.concatenate(mus), np.concatenate(logvars)


def _reconstruct_all(
    model: nn.Module, dataset, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    device = _get_device(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    recons, targets = [], []
    with torch.no_grad():
        for encoder_input, stim_cond, target in loader:
            encoder_input = encoder_input.float().to(device)
            stim_cond = stim_cond.float().to(device)
            mu, logvar = model.encoder(encoder_input)
            recon = model.decoder(mu, stim_cond)  # use mean (no sampling)
            recons.append(recon.cpu().numpy())
            targets.append(target.cpu().numpy())
    return np.concatenate(recons), np.concatenate(targets)


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def _compute_mse_per_cell(recons: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean((recons - targets) ** 2, axis=(1, 2))  # (n_cells,)


def _plot_mse_distribution(
    mse_per_cell: np.ndarray, metadata: pd.DataFrame, condition_col: str
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(mse_per_cell, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(mse_per_cell.mean(), color="red", ls="--", label=f"mean={mse_per_cell.mean():.5f}")
    ax1.axvline(np.median(mse_per_cell), color="orange", ls="--", label=f"median={np.median(mse_per_cell):.5f}")
    ax1.set_xlabel("MSE")
    ax1.set_ylabel("Count")
    ax1.set_title("Per-cell MSE distribution")
    ax1.legend(fontsize=8)

    conditions = metadata[condition_col].values[: len(mse_per_cell)]
    unique_conds = sorted(set(conditions))
    data = [mse_per_cell[conditions == c] for c in unique_conds]
    ax2.boxplot(data, labels=unique_conds)
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE by condition")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Reconstructions plot
# ---------------------------------------------------------------------------

def _plot_reconstructions(
    recons: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    condition_col: str,
    n_examples: int,
    mse_per_cell: np.ndarray,
) -> Figure:
    conditions = metadata[condition_col].values[: len(recons)]
    unique_conds = sorted(set(conditions))
    # sample roughly equally from each condition
    indices = []
    per_cond = max(1, n_examples // len(unique_conds))
    rng = np.random.default_rng(42)
    for c in unique_conds:
        cond_idx = np.where(conditions == c)[0]
        chosen = rng.choice(cond_idx, size=min(per_cond, len(cond_idx)), replace=False)
        indices.extend(chosen)
    indices = indices[:n_examples]

    ncols = min(4, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for i, idx in enumerate(indices):
        r, c_idx = divmod(i, ncols)
        ax = axes[r, c_idx]
        t = targets[idx, 0]
        rec = recons[idx, 0]
        ax.plot(t, color="black", lw=1, label="original")
        ax.plot(rec, color="red", lw=1, alpha=0.8, label="recon")
        cond = conditions[idx]
        ax.set_title(f"{cond} | MSE={mse_per_cell[idx]:.5f}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    # hide unused
    for i in range(len(indices), nrows * ncols):
        r, c_idx = divmod(i, ncols)
        axes[r, c_idx].set_visible(False)
    fig.suptitle("Reconstruction examples", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------

def _compute_frequency_analysis(
    recons: np.ndarray, targets: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = targets.shape[-1]
    freqs = rfftfreq(T, d=dt)
    orig_psd = np.abs(rfft(targets[:, 0, :], axis=-1)) ** 2
    recon_psd = np.abs(rfft(recons[:, 0, :], axis=-1)) ** 2
    return orig_psd, recon_psd, freqs


def _plot_power_spectra(
    orig_psd: np.ndarray, recon_psd: np.ndarray, freqs: np.ndarray
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    mean_orig = orig_psd.mean(axis=0)
    mean_recon = recon_psd.mean(axis=0)
    ax1.semilogy(freqs, mean_orig, label="Original", color="black")
    ax1.semilogy(freqs, mean_recon, label="Reconstruction", color="red", alpha=0.8)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Power (log)")
    ax1.set_title("Mean power spectral density")
    ax1.legend(fontsize=8)

    eps = 1e-12
    ratio = recon_psd / (orig_psd + eps)
    mean_ratio = ratio.mean(axis=0)
    std_ratio = ratio.std(axis=0)
    ax2.plot(freqs, mean_ratio, color="blue")
    ax2.fill_between(freqs, mean_ratio - std_ratio, mean_ratio + std_ratio, alpha=0.2)
    ax2.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Power ratio (recon / orig)")
    ax2.set_title("Spectral fidelity")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# KL per dimension
# ---------------------------------------------------------------------------

def _compute_kl_per_dimension(
    z_mu: np.ndarray, z_logvar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # KL(q || p) = -0.5 * (1 + logvar - mu^2 - exp(logvar))  per cell per dim
    kl_per_cell = -0.5 * (1 + z_logvar - z_mu ** 2 - np.exp(z_logvar))  # (n, L)
    kl_per_dim = kl_per_cell.mean(axis=0)  # (L,)
    return kl_per_dim, kl_per_cell


def _plot_kl_per_dimension(
    kl_per_dim: np.ndarray, threshold: float
) -> Figure:
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


# ---------------------------------------------------------------------------
# Latent traversals
# ---------------------------------------------------------------------------

def _compute_latent_traversals(
    model: nn.Module,
    dataset,
    z_mu: np.ndarray,
    active_dims: np.ndarray,
    n_steps: int,
    range_val: float,
) -> dict[int, np.ndarray]:
    if not active_dims.any():
        return {}
    device = _get_device(model)
    # pick reference cell near centroid
    centroid = z_mu.mean(axis=0)
    ref_idx = int(np.argmin(np.linalg.norm(z_mu - centroid, axis=1)))
    _, stim_cond, _ = dataset[ref_idx]
    stim_cond = stim_cond.float().unsqueeze(0).to(device)
    ref_z = torch.tensor(z_mu[ref_idx], dtype=torch.float32, device=device)

    traversals = {}
    active_indices = np.where(active_dims)[0]
    with torch.no_grad():
        for dim in active_indices:
            vals = np.linspace(-range_val, range_val, n_steps)
            z_batch = ref_z.unsqueeze(0).repeat(n_steps, 1)
            z_batch[:, dim] = torch.tensor(vals, dtype=torch.float32, device=device)
            stim_batch = stim_cond.repeat(n_steps, 1, 1)
            decoded = model.decoder(z_batch, stim_batch)
            traversals[int(dim)] = decoded.cpu().numpy()
    return traversals


def _plot_latent_traversals(
    traversals: dict[int, np.ndarray], active_dims: np.ndarray, range_val: float
) -> Figure:
    n_dims = len(traversals)
    if n_dims == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No active dimensions", ha="center", va="center", transform=ax.transAxes)
        plt.close(fig)
        return fig

    ncols = min(4, n_dims)
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows), squeeze=False)
    cmap = plt.cm.viridis

    all_vals = np.concatenate([v[:, 0, :] for v in traversals.values()])
    ymin, ymax = all_vals.min(), all_vals.max()
    margin = (ymax - ymin) * 0.05

    for i, (dim, decoded) in enumerate(sorted(traversals.items())):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        n_steps = decoded.shape[0]
        for s in range(n_steps):
            colour = cmap(s / max(1, n_steps - 1))
            ax.plot(decoded[s, 0], color=colour, lw=0.8)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(f"Dim {dim}", fontsize=9)
    for i in range(len(traversals), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)
    fig.suptitle("Latent traversals", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Stimulus invariance (condition classifier)
# ---------------------------------------------------------------------------

def _compute_stimulus_invariance(
    z_mu: np.ndarray, metadata: pd.DataFrame, condition_col: str
) -> tuple[float, float, np.ndarray]:
    labels = metadata[condition_col].values[: len(z_mu)]
    unique = np.unique(labels)
    chance = 1.0 / len(unique)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    scores = cross_val_score(clf, z_mu, labels, cv=min(5, len(unique)), scoring="accuracy")
    return float(scores.mean()), float(chance), scores


def _plot_stimulus_invariance(
    z_mu: np.ndarray,
    metadata: pd.DataFrame,
    condition_col: str,
    accuracy: float,
    chance: float,
    active_dims: np.ndarray,
) -> Figure:
    labels = metadata[condition_col].values[: len(z_mu)]
    active_idx = np.where(active_dims)[0]
    if len(active_idx) < 2:
        active_idx = np.arange(min(2, z_mu.shape[1]))
    pairs = list(combinations(active_idx, 2))[:3]
    n_plots = max(1, len(pairs))
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), squeeze=False)
    unique_labels = sorted(set(labels))
    cmap = plt.cm.tab10
    for p_i, (d1, d2) in enumerate(pairs):
        ax = axes[0, p_i]
        for li, lab in enumerate(unique_labels):
            mask = labels == lab
            ax.scatter(z_mu[mask, d1], z_mu[mask, d2], s=3, alpha=0.4,
                       color=cmap(li / max(1, len(unique_labels) - 1)), label=lab)
        ax.set_xlabel(f"z{d1}")
        ax.set_ylabel(f"z{d2}")
        if p_i == 0:
            ax.legend(fontsize=6, markerscale=3)
    fig.suptitle(f"Latent space — clf acc={accuracy:.3f} (chance={chance:.3f})", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Within-condition distributions
# ---------------------------------------------------------------------------

def _plot_within_condition(
    z_mu: np.ndarray, metadata: pd.DataFrame, condition_col: str, active_dims: np.ndarray
) -> Figure:
    labels = metadata[condition_col].values[: len(z_mu)]
    unique_conds = sorted(set(labels))
    active_idx = np.where(active_dims)[0]
    if len(active_idx) == 0:
        active_idx = np.arange(min(2, z_mu.shape[1]))
    n_dims = len(active_idx)
    n_conds = len(unique_conds)
    fig, axes = plt.subplots(n_conds, n_dims, figsize=(3 * n_dims, 2.2 * n_conds), squeeze=False)
    for ci, cond in enumerate(unique_conds):
        mask = labels == cond
        for di, dim in enumerate(active_idx):
            ax = axes[ci, di]
            ax.hist(z_mu[mask, dim], bins=30, alpha=0.7, edgecolor="black", linewidth=0.3)
            if ci == 0:
                ax.set_title(f"z{dim}", fontsize=9)
            if di == 0:
                ax.set_ylabel(cond, fontsize=8)
    fig.suptitle("Within-condition latent distributions", fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Encoder uncertainty
# ---------------------------------------------------------------------------

def _plot_encoder_uncertainty(
    z_logvar: np.ndarray, metadata: pd.DataFrame, condition_col: str
) -> Figure:
    sigma = np.exp(0.5 * z_logvar)
    mean_sigma = sigma.mean(axis=1)  # (n_cells,)
    conditions = metadata[condition_col].values[: len(mean_sigma)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(mean_sigma, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Mean sigma (across latent dims)")
    ax1.set_ylabel("Count")
    ax1.set_title("Encoder uncertainty distribution")

    unique_conds = sorted(set(conditions))
    data = [mean_sigma[conditions == c] for c in unique_conds]
    ax2.boxplot(data, labels=unique_conds)
    ax2.set_ylabel("Mean sigma")
    ax2.set_title("Uncertainty by condition")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataset,
    metadata: pd.DataFrame,
    condition_col: str = "condition",
    dt: float = 1.0,
    n_recon_examples: int = 12,
    n_traversal_steps: int = 7,
    traversal_range: float = 2.0,
    kl_active_threshold: float = 0.05,
    name: str | None = None,
) -> ExperimentResult:
    if len(dataset) != len(metadata):
        warnings.warn(
            f"Dataset has {len(dataset)} samples but metadata has "
            f"{len(metadata)} rows. If using windowed data, ensure "
            f"evaluation uses one canonical window per cell."
        )

    model.eval()
    config = _discover_model_config(model)
    latent_dim = config.get("latent_dim", None)

    # --- encode & reconstruct ---
    z_mu, z_logvar = _encode_all(model, dataset)
    recons, targets = _reconstruct_all(model, dataset)
    if latent_dim is None:
        config["latent_dim"] = z_mu.shape[1]

    # --- MSE ---
    mse_per_cell = _compute_mse_per_cell(recons, targets)

    # --- KL ---
    kl_per_dim, kl_per_cell = _compute_kl_per_dimension(z_mu, z_logvar)
    active_dims = kl_per_dim >= kl_active_threshold

    # --- Frequency ---
    orig_psd, recon_psd, freqs = _compute_frequency_analysis(recons, targets, dt)

    # --- Stimulus invariance ---
    accuracy, chance, clf_scores = _compute_stimulus_invariance(z_mu, metadata, condition_col)

    # --- Traversals ---
    traversals = _compute_latent_traversals(
        model, dataset, z_mu, active_dims, n_traversal_steps, traversal_range
    )

    # --- Figures ---
    figures = {}
    figures["reconstructions"] = _plot_reconstructions(
        recons, targets, metadata, condition_col, n_recon_examples, mse_per_cell
    )
    figures["mse_distribution"] = _plot_mse_distribution(mse_per_cell, metadata, condition_col)
    figures["power_spectra"] = _plot_power_spectra(orig_psd, recon_psd, freqs)
    figures["kl_per_dim"] = _plot_kl_per_dimension(kl_per_dim, kl_active_threshold)
    figures["latent_traversals"] = _plot_latent_traversals(traversals, active_dims, traversal_range)
    figures["stimulus_invariance"] = _plot_stimulus_invariance(
        z_mu, metadata, condition_col, accuracy, chance, active_dims
    )
    figures["latent_space"] = _plot_stimulus_invariance(
        z_mu, metadata, condition_col, accuracy, chance, active_dims
    )
    figures["within_condition"] = _plot_within_condition(z_mu, metadata, condition_col, active_dims)
    figures["encoder_uncertainty"] = _plot_encoder_uncertainty(z_logvar, metadata, condition_col)

    # --- Assemble ---
    metrics = {
        "mse_per_cell": mse_per_cell,
        "z_mu": z_mu,
        "z_logvar": z_logvar,
        "kl_per_dim": kl_per_dim,
        "kl_per_cell": kl_per_cell,
        "active_dims": active_dims,
        "n_active_dims": int(active_dims.sum()),
        "orig_psd": orig_psd,
        "recon_psd": recon_psd,
        "psd_freqs": freqs,
        "condition_clf_acc": accuracy,
        "condition_clf_chance": chance,
    }

    return ExperimentResult(
        name=name or "experiment",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_config=config,
        metrics=metrics,
        figures=figures,
    )


# ---------------------------------------------------------------------------
# Experiment comparison
# ---------------------------------------------------------------------------

def compare_experiments(
    results: dict[str, ExperimentResult],
    metrics: list[str] | None = None,
) -> Figure:
    if metrics is None:
        metrics = ["mse", "kl", "invariance", "spectral"]

    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    names = list(results.keys())
    colours = [plt.cm.Set2(i / max(1, len(names) - 1)) for i in range(len(names))]

    for pi, metric in enumerate(metrics):
        ax = axes[pi]
        if metric == "mse":
            data = [results[n].metrics["mse_per_cell"] for n in names]
            ax.boxplot(data, labels=names)
            ax.set_ylabel("MSE")
            ax.set_title("Reconstruction MSE")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        elif metric == "kl":
            max_L = max(len(results[n].metrics["kl_per_dim"]) for n in names)
            x = np.arange(max_L)
            width = 0.8 / len(names)
            for ni, n in enumerate(names):
                kl = results[n].metrics["kl_per_dim"]
                ax.bar(x[: len(kl)] + ni * width, kl, width, label=n, color=colours[ni])
            ax.set_xlabel("Latent dimension")
            ax.set_ylabel("Mean KL")
            ax.set_title("KL per dimension")
            ax.legend(fontsize=7)

        elif metric == "invariance":
            accs = [results[n].metrics["condition_clf_acc"] for n in names]
            chance = results[names[0]].metrics["condition_clf_chance"]
            ax.bar(range(len(names)), accs, color=colours)
            ax.axhline(chance, color="grey", ls="--", lw=0.8, label=f"chance={chance:.2f}")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Accuracy")
            ax.set_title("Condition classification")
            ax.legend(fontsize=7)

        elif metric == "spectral":
            for ni, n in enumerate(names):
                m = results[n].metrics
                freqs = m["psd_freqs"]
                ratio = (m["recon_psd"] / (m["orig_psd"] + 1e-12)).mean(axis=0)
                ax.plot(freqs, ratio, label=n, color=colours[ni])
            ax.axhline(1.0, color="grey", ls="--", lw=0.8)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Power ratio")
            ax.set_title("Spectral fidelity")
            ax.legend(fontsize=7)

    fig.suptitle("Experiment comparison", fontsize=13)
    fig.tight_layout()
    plt.close(fig)
    return fig
