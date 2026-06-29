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

from baseline_prepend import prepend_channels  # noqa: E402

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


def _prepend_baseline_tracks(
    cnr: np.ndarray,
    stim: np.ndarray,
    n_prepend: int,
    *,
    seed: int = 42,
    block_len: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend ``n_prepend`` baseline-like frames to per-cell tracks.

    For each cell: block-bootstrap the CNR baseline, prepend zero light, then
    re-derive all 9 stim features over the full (prepended+real) sequence so the
    EWMA / recency / cumulative channels enter the real region correctly (no
    prior stimulation). ``cnr``/``stim`` are object arrays of per-cell tracks.
    """
    if n_prepend <= 0:
        return cnr, stim
    rng = np.random.default_rng(seed)
    cnr_out = np.empty(len(cnr), dtype=object)
    stim_out = np.empty(len(stim), dtype=object)
    for i in range(len(cnr)):
        ci = np.asarray(cnr[i], dtype=np.float32)
        light = np.asarray(stim[i][0], dtype=np.float32)  # u_t channel
        new_cnr, new_light = prepend_channels(
            [ci, light], n_prepend, zero_channels={1}, block_len=block_len, rng=rng
        )
        cnr_out[i] = new_cnr.astype(np.float32)
        stim_out[i] = _stim_features(new_light[None, :])[0].astype(np.float32)
    return cnr_out, stim_out


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
    path: str = "dataset.parquet.v0",
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


def load_real_uncertain(
    path: str = "dataset_real_uncertain.parquet",
    window_size: int = 20,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined real dataset (dataset.parquet + BO_v1 + BO_v2).

    BO_v1/v2 stim_power metadata is unconfirmed — treat results as exploratory.
    Rows are already preprocessed (load_and_clean output schema); skip re-cleaning
    and window directly.
    """
    from notebooks.experiment.preprocessing import make_windows, DEFAULT_STIM_COLS

    df = pd.read_parquet(path)

    cnr, stim_all, meta = make_windows(
        df,
        window_size=window_size,
        stride=stride,
        value_col="cnr_median_norm",
        stim_cols=DEFAULT_STIM_COLS,
    )

    conditions = meta["ramp_pattern_name"].values

    return cnr, stim_all, conditions


def load_real_tracks(
    path: str = "dataset.parquet.v0",
    baseline_prepend: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real microscopy data as per-cell full trajectories (no windowing).

    Returns object arrays so callers can consume per-cell tracks uniformly;
    variable T across cells is preserved. Downstream ``Seq2SeqDataset``-style
    classes handle windowing.

    Returns
    -------
    cnr : np.ndarray(dtype=object), shape (n_cells,)
        Each element is a 1D float32 CNR trajectory.
    stim : np.ndarray(dtype=object), shape (n_cells,)
        Each element is a 2D float32 array of shape ``(9, T_cell)``.
    conditions : np.ndarray, shape (n_cells,)
        ramp_pattern_name per cell.
    """
    from notebooks.experiment.preprocessing import (
        DEFAULT_STIM_COLS,
        load_and_clean,
        make_tracks,
    )

    df = load_and_clean(path, baseline_cnr_max=None)
    cnr, stim, meta = make_tracks(
        df, value_col="cnr_median_norm", stim_cols=DEFAULT_STIM_COLS
    )
    conditions = meta["ramp_pattern_name"].to_numpy()
    cnr, stim = _prepend_baseline_tracks(cnr, stim, baseline_prepend)
    return cnr, stim, conditions


def load_real_uncertain_tracks(
    path: str = "dataset_real_uncertain.parquet",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same contract as :func:`load_real_tracks` for the combined dataset."""
    from notebooks.experiment.preprocessing import DEFAULT_STIM_COLS, make_tracks

    df = pd.read_parquet(path)
    cnr, stim, meta = make_tracks(
        df, value_col="cnr_median_norm", stim_cols=DEFAULT_STIM_COLS
    )
    conditions = meta["ramp_pattern_name"].to_numpy()
    return cnr, stim, conditions


def load_real_plus_bo_tracks(
    path: str = "dataset.parquet",
    baseline_prepend: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-BO real dataset (`dataset.parquet`); pre-BO snapshot is `real`."""
    from notebooks.experiment.preprocessing import DEFAULT_STIM_COLS, make_tracks

    df = pd.read_parquet(path)
    cnr, stim, meta = make_tracks(
        df, value_col="cnr_median_norm", stim_cols=DEFAULT_STIM_COLS
    )
    conditions = meta["ramp_pattern_name"].to_numpy()
    cnr, stim = _prepend_baseline_tracks(cnr, stim, baseline_prepend)
    return cnr, stim, conditions


def load_real_plus_bo(
    path: str = "dataset.parquet",
    window_size: int = 20,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed post-BO real dataset — counterpart to :func:`load_real`."""
    from notebooks.experiment.preprocessing import DEFAULT_STIM_COLS, make_windows

    df = pd.read_parquet(path)
    cnr, stim_all, meta = make_windows(
        df,
        window_size=window_size,
        stride=stride,
        value_col="cnr_median_norm",
        stim_cols=DEFAULT_STIM_COLS,
    )
    conditions = meta["ramp_pattern_name"].values
    return cnr, stim_all, conditions


AVAILABLE_DATASETS = (
    "synthetic", "synthetic_v2", "real", "real_uncertain", "real_plus_bo",
)


# `real` is the pre-BO snapshot (preserves the semantics of past experiments
# that recorded data_source="real"). `real_plus_bo` is the post-BO file
# (includes BO oscillation v8, v10, v11_10s, v11_20s; ramp_pattern_name carries
# the per-experiment tag, e.g. `bo_osc_v10_c<idx>`).
REAL_DATASET_PATHS = {
    "real": "dataset.parquet.v0",
    "real_plus_bo": "dataset.parquet",
}


def load(
    ds_name: str,
    *,
    baseline_prepend: int = 0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the loader for a named dataset.

    **Default contract (new)**: ``load("real")`` / ``load("real_uncertain")``
    return **per-cell full trajectories** as object arrays — suitable for
    frame-by-frame navigation and in-notebook `Seq2SeqDataset` windowing.

    **Legacy contract (back-compat)**: passing ``window_size=`` (and/or
    ``stride=``) routes to the pre-windowed real loaders used by
    ``ensemble_seq2scal.py`` / ``compare_seq2seq.py`` — these return
    ``(N_windows, window_size)`` 2D arrays as before.

    Synthetic loaders are unchanged (already return full tracks as uniform
    2D arrays).

    ``baseline_prepend > 0`` prepends that many block-bootstrapped baseline
    frames (zero light) to each per-cell track so the stimulation onset becomes
    predictable. Only supported for the ``real`` / ``real_plus_bo`` track
    contract (raises otherwise).
    """
    if baseline_prepend and (
        ds_name not in ("real", "real_plus_bo")
        or "window_size" in kwargs
        or "stride" in kwargs
    ):
        raise ValueError(
            "baseline_prepend is only supported for the 'real'/'real_plus_bo' "
            "per-cell track loaders (no window_size/stride)."
        )
    if ds_name == "synthetic":
        return load_synthetic(**kwargs)
    if ds_name == "synthetic_v2":
        return load_synthetic_v2(**kwargs)
    if ds_name == "real":
        if "window_size" in kwargs or "stride" in kwargs:
            return load_real(**kwargs)
        return load_real_tracks(baseline_prepend=baseline_prepend, **kwargs)
    if ds_name == "real_uncertain":
        if "window_size" in kwargs or "stride" in kwargs:
            return load_real_uncertain(**kwargs)
        return load_real_uncertain_tracks(**kwargs)
    if ds_name == "real_plus_bo":
        if "window_size" in kwargs or "stride" in kwargs:
            return load_real_plus_bo(**kwargs)
        return load_real_plus_bo_tracks(baseline_prepend=baseline_prepend, **kwargs)
    raise ValueError(
        f"Unknown dataset {ds_name!r}. Available: {list(AVAILABLE_DATASETS)}"
    )
