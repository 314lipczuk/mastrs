"""
Shared data loading for LSTM seq2seq experiments.

Provides a unified interface for both synthetic (stochastic simulator)
and real (microscopy parquet) data. Both loaders return the same format:
    cnr:        (N, T) float32 — baseline-normalized CNR signal
    stim:       (N, n_stim, T) float32 — stimulation feature channels
    conditions: (N,) str — label per trajectory (generator type or ramp pattern)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STIM_COLS = [
    "u_t", "m_t", "recency", "ewma_fast", "ewma_slow",
    "n_5", "slope_5", "burst_pos", "s_cum",
]


def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized EWMA along axis=1 for a 2D array (N, T)."""
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
    return out


def _stim_features(light: np.ndarray, tau: float = 5.0, window: int = 5) -> np.ndarray:
    """Derive all 9 stim feature channels from a (N, T) light array.

    Returns stim array of shape (N, 9, T) with channels matching STIM_COLS:
        u_t, m_t, recency, ewma_fast, ewma_slow, n_5, slope_5, burst_pos, s_cum
    """
    N, T = light.shape
    m = (light > 0).astype(np.float32)

    u_t      = light
    m_t      = m
    ewma_fast = _ewma(u_t, alpha=0.5)
    ewma_slow = _ewma(u_t, alpha=0.1)
    s_cum    = np.cumsum(u_t, axis=1)

    recency   = np.zeros((N, T), dtype=np.float32)
    burst_pos = np.zeros((N, T), dtype=np.float32)
    n_5       = np.zeros((N, T), dtype=np.float32)
    slope_5   = np.zeros((N, T), dtype=np.float32)

    last_pulse = np.full(N, -1, dtype=int)

    for t in range(T):
        stimmed = m[:, t] > 0

        # recency: exp(-dt/tau) since last pulse, 0 if never stimulated
        last_pulse = np.where(stimmed, t, last_pulse)
        dt = np.where(last_pulse >= 0, t - last_pulse, np.inf).astype(float)
        recency[:, t] = np.where(last_pulse >= 0, np.exp(-dt / tau), 0.0)

        # burst_pos: 1-indexed position within consecutive on-burst, 0 when off
        prev_bp = burst_pos[:, t - 1] if t > 0 else np.zeros(N)
        prev_m  = m[:, t - 1] > 0     if t > 0 else np.zeros(N, dtype=bool)
        burst_pos[:, t] = np.where(stimmed, np.where(prev_m, prev_bp + 1, 1), 0)

        # n_5: pulse count in last `window` frames (inclusive)
        start = max(0, t - window + 1)
        n_5[:, t] = m[:, start : t + 1].sum(axis=1)

        # slope_5: OLS slope of u_t over last `window` frames
        w = u_t[:, start : t + 1]        # (N, w_len)
        w_len = w.shape[1]
        if w_len >= 2:
            x = np.arange(w_len, dtype=float)
            x_c = x - x.mean()
            ss = (x_c ** 2).sum()
            if ss > 0:
                slope_5[:, t] = ((x_c * (w - w.mean(axis=1, keepdims=True))).sum(axis=1) / ss)

    return np.stack(
        [u_t, m_t, recency, ewma_fast, ewma_slow, n_5, slope_5, burst_pos, s_cum],
        axis=1,
    ).astype(np.float32)


def load_synthetic(
    path: str = "stochastic_sim_output.parquet",
    baseline_frames: int = 10,
    cnr_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load stochastic simulator output and derive all 9 stim features from light array.

    Returns
    -------
    cnr : (N, T) baseline-normalized CNR
    stim : (N, 9, T) stimulus features matching STIM_COLS
    conditions : (N,) generator labels
    """
    df = pd.read_parquet(path)

    cnr_raw = np.stack(df["cnr"].values).astype(np.float32)
    light   = np.stack(df["light"].values).astype(np.float32)

    valid   = np.abs(cnr_raw).max(axis=1) < cnr_max
    cnr_raw = cnr_raw[valid]
    light   = light[valid]
    conditions = df["generator"].values[valid]

    baseline = np.median(cnr_raw[:, :baseline_frames], axis=1, keepdims=True)
    baseline = np.where(np.abs(baseline) < 1e-6, 1.0, baseline)
    cnr = cnr_raw / baseline

    stim = _stim_features(light)

    return cnr, stim, conditions


def load_synthetic_v2(
    path: str = "stochastic_sim_v2_output.parquet",
    baseline_frames: int = 10,
    cnr_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load v2 stochastic simulator output (k12 heterogeneity + wider biosensor gain).

    Same schema as v1 plus a `k12` column. Returns the same (cnr, stim, conditions)
    triple as `load_synthetic`.
    """
    return load_synthetic(path=path, baseline_frames=baseline_frames, cnr_max=cnr_max)


def load_real(
    path: str = "dataset.parquet",
    window_size: int = 20,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real microscopy data via preprocessing pipeline with all 9 stim features.

    Returns
    -------
    cnr : (N_windows, window_size) baseline-normalized CNR
    stim : (N_windows, 9, window_size) stimulus features matching STIM_COLS
    conditions : (N_windows,) ramp pattern labels
    """
    from notebooks.experiment.preprocessing import load_and_clean, make_windows, DEFAULT_STIM_COLS

    df = load_and_clean(path, baseline_cnr_max=None)

    cnr, stim_all, meta = make_windows(
        df,
        window_size=window_size,
        stride=stride,
        value_col="cnr_median_norm",
        stim_cols=DEFAULT_STIM_COLS,
    )

    conditions = meta["ramp_pattern_name"].values

    return cnr, stim_all, conditions


AVAILABLE_DATASETS = ("synthetic", "synthetic_v2", "real")


def load(
    ds_name: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the loader for a named dataset.

    Single source of truth for "which datasets exist". Adding a new dataset
    means touching this function (and `AVAILABLE_DATASETS`) — notebooks don't
    need to know the catalog.

    Returns (cnr, stim, conditions) — same contract as the underlying loaders.
    """
    if ds_name == "synthetic":
        return load_synthetic(**kwargs)
    if ds_name == "synthetic_v2":
        return load_synthetic_v2(**kwargs)
    if ds_name == "real":
        return load_real(**kwargs)
    raise ValueError(
        f"Unknown dataset {ds_name!r}. Available: {list(AVAILABLE_DATASETS)}"
    )
