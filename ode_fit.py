"""
Single-cell ODE fitting pipeline for ERK signaling dynamics.

Fits a driven FitzHugh-Nagumo model to per-cell ERK time series
under known light stimulation, extracts multi-dimensional state
trajectories, and packages them for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellFitResult:
    """Result of fitting the FHN model to a single cell."""

    cell_idx: int
    params: dict[str, float]          # {a, b, epsilon, gain, tau_stim}
    loss: float                       # best-fit MSE
    u_trajectory: np.ndarray          # (T,) fitted fast variable
    w_trajectory: np.ndarray          # (T,) fitted slow variable
    I_drive: np.ndarray               # (T,) convolved stimulus drive
    converged: bool                   # did optimiser converge


# ---------------------------------------------------------------------------
# ODE helpers
# ---------------------------------------------------------------------------

PARAM_NAMES: list[str] = ["a", "b", "epsilon", "gain", "tau_stim"]

PARAM_BOUNDS: list[tuple[float, float]] = [
    (-2.0, 2.0),   # a
    (0.1, 5.0),    # b
    (0.01, 1.0),   # epsilon
    (0.0, 10.0),   # gain
    (0.5, 30.0),   # tau_stim
]


def _convolve_stimulus(stim_1d: np.ndarray, tau_stim: float, dt: float) -> np.ndarray:
    """Convolve a stimulus trace with an exponential kernel."""
    T = len(stim_1d)
    # Build causal exponential kernel, truncated at 5*tau_stim
    kernel_len = min(T, max(1, int(5 * tau_stim / dt)))
    t_kern = np.arange(kernel_len) * dt
    kernel = np.exp(-t_kern / tau_stim) / tau_stim
    kernel *= dt  # discretisation factor so integral ≈ 1
    convolved = np.convolve(stim_1d, kernel, mode="full")[:T]
    return convolved


def _fhn_rhs(
    t: float,
    y: np.ndarray,
    a: float,
    b: float,
    epsilon: float,
    I_interp,  # callable(t) -> float
) -> list[float]:
    """Right-hand side of the FitzHugh-Nagumo system."""
    u, w = y
    du = u - u**3 / 3.0 - w + I_interp(t)
    dw = epsilon * (u + a - b * w)
    return [du, dw]


def _integrate_fhn(
    params_dict: dict[str, float],
    stim_1d: np.ndarray,
    u0: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Integrate the FHN ODE for one cell. Returns (u, w, I_drive, success)."""
    a = params_dict["a"]
    b = params_dict["b"]
    epsilon = params_dict["epsilon"]
    gain = params_dict["gain"]
    tau_stim = params_dict["tau_stim"]

    T = len(stim_1d)
    t_span = (0.0, (T - 1) * dt)
    t_eval = np.arange(T) * dt

    # Convolved stimulus drive
    I_raw = _convolve_stimulus(stim_1d, tau_stim, dt)
    I_drive = gain * I_raw

    # Linear interpolation of I_drive for the integrator
    def I_interp(t: float) -> float:
        idx = t / dt
        i0 = int(idx)
        if i0 >= T - 1:
            return float(I_drive[-1])
        frac = idx - i0
        return float(I_drive[i0] * (1.0 - frac) + I_drive[i0 + 1] * frac)

    # Steady-state w(0) from du/dt=0, dw/dt=0 is complicated;
    # use w0 = u0 - u0^3/3 (from du/dt=0 with I=0)
    w0 = u0 - u0**3 / 3.0

    try:
        sol = solve_ivp(
            _fhn_rhs,
            t_span,
            [u0, w0],
            method="LSODA",
            t_eval=t_eval,
            args=(a, b, epsilon, I_interp),
            max_step=dt,
            rtol=1e-6,
            atol=1e-8,
        )
        if sol.success and sol.y.shape[1] == T:
            return sol.y[0], sol.y[1], I_drive, True
        else:
            return np.full(T, np.nan), np.full(T, np.nan), I_drive, False
    except Exception:
        return np.full(T, np.nan), np.full(T, np.nan), I_drive, False


# ---------------------------------------------------------------------------
# Single-cell fitting
# ---------------------------------------------------------------------------


def _objective(
    x: np.ndarray,
    stim_1d: np.ndarray,
    erk_1d: np.ndarray,
    u0: float,
    dt: float,
) -> float:
    """MSE between observed ERK and fitted u(t)."""
    params_dict = dict(zip(PARAM_NAMES, x))
    u, _w, _I, success = _integrate_fhn(params_dict, stim_1d, u0, dt)
    if not success or np.any(np.isnan(u)):
        return 1e6
    return float(np.mean((u - erk_1d) ** 2))


def _sample_initial_params(rng: np.random.Generator) -> np.ndarray:
    """Sample one set of initial parameters uniformly within bounds."""
    return np.array([
        rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS
    ])


def fit_single_cell(
    cell_idx: int,
    erk_1d: np.ndarray,
    stim_1d: np.ndarray,
    dt: float = 1.0,
    n_restarts: int = 3,
) -> CellFitResult:
    """Fit the FHN model to a single cell's ERK trace."""
    T = len(erk_1d)
    u0 = float(erk_1d[0])
    rng = np.random.default_rng(seed=cell_idx)

    best_loss = np.inf
    best_x: np.ndarray | None = None
    best_converged = False

    for _ in range(n_restarts):
        x0 = _sample_initial_params(rng)
        try:
            res = minimize(
                _objective,
                x0,
                args=(stim_1d, erk_1d, u0, dt),
                method="L-BFGS-B",
                bounds=PARAM_BOUNDS,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
                best_converged = res.success
        except Exception:
            continue

    if best_x is None:
        # All restarts failed
        return CellFitResult(
            cell_idx=cell_idx,
            params={k: np.nan for k in PARAM_NAMES},
            loss=np.inf,
            u_trajectory=np.full(T, np.nan),
            w_trajectory=np.full(T, np.nan),
            I_drive=np.full(T, np.nan),
            converged=False,
        )

    params_dict = dict(zip(PARAM_NAMES, best_x))
    u, w, I_drive, integrate_ok = _integrate_fhn(params_dict, stim_1d, u0, dt)

    return CellFitResult(
        cell_idx=cell_idx,
        params=params_dict,
        loss=float(best_loss),
        u_trajectory=u,
        w_trajectory=w,
        I_drive=I_drive,
        converged=best_converged and integrate_ok,
    )


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------


def fit_all_cells(
    erk: np.ndarray,
    stim: np.ndarray,
    dt: float = 1.0,
    n_restarts: int = 3,
    n_jobs: int = -1,
) -> list[CellFitResult]:
    """
    Fit the FHN model to every cell in parallel.

    Parameters
    ----------
    erk : (n_cells, T) observed ERK fluorescence
    stim : (n_cells, T) light stimulation signal
    dt : timestep in minutes
    n_restarts : number of random initial parameter sets per cell
    n_jobs : parallel workers (-1 = all cores)

    Returns
    -------
    List of CellFitResult, one per cell.
    """
    n_cells = erk.shape[0]

    results: list[CellFitResult] = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_cell)(i, erk[i], stim[i], dt, n_restarts)
        for i in tqdm(range(n_cells), desc="Fitting cells")
    )
    return results


# ---------------------------------------------------------------------------
# State matrix extraction
# ---------------------------------------------------------------------------


def extract_state_matrix(results: list[CellFitResult]) -> np.ndarray:
    """
    Stack fitted trajectories into a single array.

    Returns
    -------
    np.ndarray of shape (n_cells, T, 3) with channels [u, w, I_drive].
    """
    return np.stack(
        [
            np.column_stack([r.u_trajectory, r.w_trajectory, r.I_drive])
            for r in results
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Diagnostic plotting
# ---------------------------------------------------------------------------


def plot_fits(
    results: list[CellFitResult],
    erk: np.ndarray,
    n_examples: int = 12,
) -> Figure:
    """
    Grid of subplots: original ERK (black), fitted u (red),
    fitted w (blue, secondary y-axis) for random cells.
    """
    rng = np.random.default_rng(0)
    indices = rng.choice(len(results), size=min(n_examples, len(results)), replace=False)
    indices = np.sort(indices)

    n = len(indices)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax_idx, cell_pick in enumerate(indices):
        r = results[cell_pick]
        row, col = divmod(ax_idx, ncols)
        ax = axes[row, col]
        T = len(r.u_trajectory)
        t = np.arange(T)

        ax.plot(t, erk[r.cell_idx], color="black", lw=0.8, label="ERK")
        ax.plot(t, r.u_trajectory, color="red", lw=1.0, label="u (fit)")
        ax.set_ylabel("ERK / u", fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(t, r.w_trajectory, color="blue", lw=0.8, alpha=0.7, label="w")
        ax2.set_ylabel("w", fontsize=8, color="blue")
        ax2.tick_params(axis="y", labelcolor="blue", labelsize=7)

        ax.set_title(
            f"cell {r.cell_idx}  MSE={r.loss:.4f}\n"
            f"a={r.params['a']:.2f} b={r.params['b']:.2f} "
            f"ε={r.params['epsilon']:.3f}",
            fontsize=7,
        )
        ax.tick_params(labelsize=7)
        if row == nrows - 1:
            ax.set_xlabel("time (min)", fontsize=8)

    # Hide empty subplots
    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig


def plot_parameter_distributions(results: list[CellFitResult]) -> Figure:
    """Histograms of each fitted parameter across all cells."""
    converged = [r for r in results if r.converged]
    n_params = len(PARAM_NAMES)

    fig, axes = plt.subplots(1, n_params, figsize=(3.5 * n_params, 3))
    if n_params == 1:
        axes = [axes]

    for ax, pname in zip(axes, PARAM_NAMES):
        vals = [r.params[pname] for r in converged]
        ax.hist(vals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel(pname, fontsize=10)
        ax.set_ylabel("count", fontsize=10)
        ax.set_title(f"{pname}  (n={len(vals)})", fontsize=10)

    fig.suptitle("Parameter distributions (converged cells)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_fit_quality(results: list[CellFitResult]) -> Figure:
    """
    Left: histogram of per-cell MSE.
    Right panels: scatter of MSE vs each parameter.
    """
    converged = [r for r in results if r.converged]
    mse_vals = np.array([r.loss for r in converged])

    n_params = len(PARAM_NAMES)
    fig, axes = plt.subplots(1, 1 + n_params, figsize=(3.5 * (1 + n_params), 3))

    # MSE histogram
    axes[0].hist(mse_vals, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("MSE", fontsize=10)
    axes[0].set_ylabel("count", fontsize=10)
    axes[0].set_title("Fit quality (MSE)", fontsize=10)

    # MSE vs parameter scatters
    for ax, pname in zip(axes[1:], PARAM_NAMES):
        pvals = [r.params[pname] for r in converged]
        ax.scatter(pvals, mse_vals, s=8, alpha=0.5)
        ax.set_xlabel(pname, fontsize=10)
        ax.set_ylabel("MSE", fontsize=10)
        ax.set_title(f"MSE vs {pname}", fontsize=10)

    fig.suptitle("Fit quality diagnostics", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Synthetic test / parameter recovery
# ---------------------------------------------------------------------------


def _generate_synthetic_data(
    n_cells: int = 20,
    T: int = 200,
    dt: float = 1.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    Generate synthetic ERK traces from known FHN parameters.

    Returns (erk, stim, true_params_list).
    """
    rng = np.random.default_rng(seed)

    # Random stimulus: a few brief pulses per cell
    stim = np.zeros((n_cells, T), dtype=np.float32)
    for i in range(n_cells):
        n_pulses = rng.integers(2, 5)
        pulse_times = rng.choice(T - 10, size=n_pulses, replace=False)
        for pt in pulse_times:
            width = rng.integers(1, 4)
            stim[i, pt : pt + width] = 1.0

    # Ground truth parameters — modest variation
    true_params_list: list[dict[str, float]] = []
    erk = np.zeros((n_cells, T), dtype=np.float32)

    for i in range(n_cells):
        p = {
            "a": rng.uniform(0.5, 1.0),
            "b": rng.uniform(0.5, 1.5),
            "epsilon": rng.uniform(0.05, 0.2),
            "gain": rng.uniform(1.0, 4.0),
            "tau_stim": rng.uniform(2.0, 8.0),
        }
        true_params_list.append(p)

        u0 = rng.uniform(-0.5, 0.5)
        u, w, I_drive, ok = _integrate_fhn(p, stim[i], u0, dt)
        if ok:
            erk[i] = u + rng.normal(0, noise_std, size=T).astype(np.float32)
        else:
            erk[i] = rng.normal(0, noise_std, size=T).astype(np.float32)

    return erk, stim, true_params_list


def test_parameter_recovery(
    n_cells: int = 20,
    noise_std: float = 0.05,
) -> bool:
    """
    Generate synthetic cells with known parameters + noise.
    Fit them. Report parameter recovery accuracy.

    Returns True if median relative error < 20% per parameter.
    """
    erk, stim, true_params_list = _generate_synthetic_data(
        n_cells=n_cells, noise_std=noise_std
    )

    results = fit_all_cells(erk, stim, dt=1.0, n_restarts=5, n_jobs=-1)

    all_pass = True
    print(f"\n{'param':>10}  {'med_rel_err':>12}  {'pass':>5}")
    print("-" * 35)

    for pname in PARAM_NAMES:
        rel_errors = []
        for r, tp in zip(results, true_params_list):
            if r.converged and np.isfinite(r.params[pname]):
                true_val = tp[pname]
                fit_val = r.params[pname]
                if abs(true_val) > 1e-8:
                    rel_errors.append(abs(fit_val - true_val) / abs(true_val))
        if rel_errors:
            median_re = float(np.median(rel_errors))
            ok = median_re < 0.20
            print(f"{pname:>10}  {median_re:>12.4f}  {'✓' if ok else '✗':>5}")
            if not ok:
                all_pass = False
        else:
            print(f"{pname:>10}  {'N/A':>12}  {'✗':>5}")
            all_pass = False

    n_converged = sum(r.converged for r in results)
    print(f"\nConverged: {n_converged}/{n_cells}")
    print(f"Overall pass: {all_pass}")
    return all_pass


# ---------------------------------------------------------------------------
# CLI entry point (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running parameter recovery test...")
    test_parameter_recovery()
