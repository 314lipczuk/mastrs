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

STIM_COLS = ["u_t", "m_t", "ewma_slow", "s_cum"]


def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized EWMA along axis=1 for a 2D array."""
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
    return out


def load_synthetic(
    path: str = "stochastic_sim_output.parquet",
    baseline_frames: int = 10,
    cnr_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load stochastic simulator output and derive stim features from light array.

    Returns
    -------
    cnr : (N, T) baseline-normalized CNR
    stim : (N, 4, T) stimulus features [u_t, m_t, ewma_slow, s_cum]
    conditions : (N,) generator labels
    """
    df = pd.read_parquet(path)

    cnr_raw = np.stack(df["cnr"].values).astype(np.float32)
    light = np.stack(df["light"].values).astype(np.float32)

    # Filter outlier trajectories
    valid = np.abs(cnr_raw).max(axis=1) < cnr_max
    cnr_raw = cnr_raw[valid]
    light = light[valid]
    conditions = df["generator"].values[valid]

    # Baseline-normalize: divide by median of first N frames
    baseline = np.median(cnr_raw[:, :baseline_frames], axis=1, keepdims=True)
    baseline = np.where(np.abs(baseline) < 1e-6, 1.0, baseline)
    cnr = cnr_raw / baseline

    # Derive stim features directly from light array (N, T)
    u_t = light
    m_t = (light > 0).astype(np.float32)
    ewma_slow = _ewma(u_t, alpha=0.1)
    s_cum = np.cumsum(u_t, axis=1)

    stim = np.stack([u_t, m_t, ewma_slow, s_cum], axis=1)  # (N, 4, T)

    return cnr, stim, conditions


def load_real(
    path: str = "dataset.parquet",
    window_size: int = 20,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real microscopy data via preprocessing pipeline.

    Returns
    -------
    cnr : (N_windows, window_size) baseline-normalized CNR
    stim : (N_windows, 4, window_size) stimulus features
    conditions : (N_windows,) ramp pattern labels
    """
    from notebooks.experiment.preprocessing import load_and_clean, make_windows

    df = load_and_clean(path, baseline_cnr_max=None)

    cnr, stim_all, meta = make_windows(
        df,
        window_size=window_size,
        stride=stride,
        value_col="cnr_median_norm",
        stim_cols=STIM_COLS,
    )

    conditions = meta["ramp_pattern_name"].values

    return cnr, stim_all, conditions
