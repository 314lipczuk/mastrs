"""Chronos-2 zero-shot baseline on the seq2scal test windows (handoff Task 5).

External foundation-model baseline. No fine-tuning — Chronos-2 forecasts the
held-out windows zero-shot. Two protocols:

- **Protocol A** — per-window univariate: each (history, future) window is its
  own series; future fluence is a known covariate. Honest worst case.
- **Protocol B** — cross-learning: windows from the same stim protocol are
  grouped into one predict call so Chronos-2 conditions in-context on related
  trajectories.

To keep the comparison exact, Chronos-2 forecasts the **delta-CNR** series —
the same target the LSTM-MDN models are scored on. History is the diff of the
H-value CNR window; the F future deltas are the target.

Dependency
----------
Requires AutoGluon TimeSeries. AutoGluon pins ``torch<2.10`` while this repo
runs ``torch>=2.10``, so the two CANNOT share an environment — run this script
in a **separate venv**::

    uv venv /tmp/chronos-env --python 3.12
    /tmp/chronos-env/bin/pip install "autogluon.timeseries" pandas pyarrow h5py scikit-learn
    /tmp/chronos-env/bin/python experiments/chronos2_baseline.py --source real_plus_bo

This script is intentionally NOT a ``launcher.py`` Job for the same reason.

The Chronos-2 model identifier is passed as ``CHRONOS_MODEL`` below; adjust if
AutoGluon exposes it under a different ``model_path``.

Usage
-----
    uv run python experiments/chronos2_baseline.py --source real_plus_bo \\
        --dry_run false [--result-path DIR] [--extended-history]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment import ExperimentTracker  # noqa: E402
from experiments.seq2scal_models import Seq2SeqDataset, prepare_data  # noqa: E402
from experiments.seq2seq_data import load as load_dataset  # noqa: E402
from utils import results_write_path  # noqa: E402

CHRONOS_MODEL = "chronos-2"
QUANTILE_LEVELS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95 (19)


# ---------------------------------------------------------------------------
# quantile-based metrics  (fully self-contained; testable without autogluon)
# ---------------------------------------------------------------------------


def pinball_crps(q_levels, q_vals, y):
    """CRPS via the quantile decomposition: CRPS = 2 * mean_tau pinball_tau.

    q_levels: (L,)  q_vals: (N, F, L)  y: (N, F)  ->  (N, F)
    """
    tau = np.asarray(q_levels)[None, None, :]
    yk = y[..., None]
    diff = yk - q_vals
    pinball = np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
    return 2.0 * pinball.mean(axis=-1)


def quantile_mae(q_levels, q_vals, y):
    """MAE from the median (0.5) quantile (nearest level)."""
    mid = int(np.argmin(np.abs(np.asarray(q_levels) - 0.5)))
    return np.abs(q_vals[..., mid] - y)


def quantile_nll(q_levels, q_vals, y):
    """NLL from a piecewise-linear CDF fitted through the predicted quantiles.

    Density between adjacent quantiles q_i, q_{i+1} is constant
    ``(tau_{i+1} - tau_i) / (q_{i+1} - q_i)``. Outside the quantile range the
    density decays exponentially with the nearest-edge slope. y: (N, F).
    """
    levels = np.asarray(q_levels)
    n, f, L = q_vals.shape
    out = np.empty((n, f), dtype=np.float64)
    flat_q = q_vals.reshape(-1, L)
    flat_y = y.reshape(-1)
    for j in range(flat_q.shape[0]):
        qs = np.sort(flat_q[j])
        yj = flat_y[j]
        dq = np.diff(qs)
        dtau = np.diff(levels)
        dens = np.where(dq > 1e-9, dtau / np.maximum(dq, 1e-9), 1e3)
        if yj <= qs[0]:
            d = dens[0] * np.exp(-(qs[0] - yj) / max(qs[1] - qs[0], 1e-6))
        elif yj >= qs[-1]:
            d = dens[-1] * np.exp(-(yj - qs[-1]) / max(qs[-1] - qs[-2], 1e-6))
        else:
            k = int(np.searchsorted(qs, yj) - 1)
            d = dens[k]
        out.reshape(-1)[j] = -np.log(max(d, 1e-12))
    return out


# ---------------------------------------------------------------------------
# window construction  (reuses the exact LSTM test windows)
# ---------------------------------------------------------------------------


def build_windows(source, history_len, future_len, *, dry_run, test_stride=10):
    """Return the held-out test windows as delta-CNR series + fluence covariate.

    history : (N, H-1) delta-CNR context
    future  : (N, F)   delta-CNR target
    fluence : (N, F)   future fluence (known covariate)
    cond    : (N,)     stim-protocol label per window (for Protocol B grouping)
    """
    prep = prepare_data(source, dry_run=dry_run)
    ds = Seq2SeqDataset(prep.test, history_len, future_len, stride=test_stride)
    _, _, conditions = load_dataset(source)

    history, future, fluence = [], [], []
    for enc_in, dec_stim, dec_target in ds:
        cnr_hist = enc_in[:, 0].numpy()
        history.append(np.diff(cnr_hist))           # (H-1,) delta-CNR context
        future.append(dec_target.numpy())          # (F,) delta-CNR target
        fluence.append(dec_stim[:, 0].numpy())      # (F,) future fluence
    # condition per window: Seq2SeqDataset does not retain cell idx, so group
    # by a coarse hash of the test split order — replaced below if needed.
    cond = np.array(["all"] * len(history))
    return (
        np.asarray(history, dtype=np.float32),
        np.asarray(future, dtype=np.float32),
        np.asarray(fluence, dtype=np.float32),
        cond,
        conditions,
    )


# ---------------------------------------------------------------------------
# Chronos-2 inference
# ---------------------------------------------------------------------------


def _to_ts_frame(history, fluence_ctx, fluence_fut):
    """Build a long-format TimeSeriesDataFrame: one item per window.

    Each item has H-1 context steps plus F future steps; ``fluence`` is the
    known covariate populated over the whole horizon.
    """
    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame

    rows = []
    n, hist_len = history.shape
    f = fluence_fut.shape[1]
    for i in range(n):
        for t in range(hist_len):
            rows.append((i, t, history[i, t], fluence_ctx[i, t]))
        for t in range(f):
            rows.append((i, hist_len + t, np.nan, fluence_fut[i, t]))
    df = pd.DataFrame(rows, columns=["item_id", "timestamp", "target", "fluence"])
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")


def chronos_predict(history, fluence, future_len, *, quantile_levels):
    """Zero-shot Chronos-2 forecast. Returns quantile array (N, F, L).

    ``fluence`` is the future fluence covariate (N, F). Context fluence is
    taken as zeros (history is delta-CNR; pulses inside the H window are not
    reconstructed — Protocol A is deliberately the honest worst case).
    """
    from autogluon.timeseries import TimeSeriesPredictor

    n, hist_len = history.shape
    fluence_ctx = np.zeros((n, hist_len), dtype=np.float32)
    ts = _to_ts_frame(history, fluence_ctx, fluence)
    known = ts.drop(columns=["target"]).loc[
        ts.index.get_level_values("timestamp") >= hist_len
    ]
    predictor = TimeSeriesPredictor(
        prediction_length=future_len,
        known_covariates_names=["fluence"],
        quantile_levels=quantile_levels,
        verbosity=1,
    )
    # zero-shot: fit only the Chronos-2 foundation model (no parameter updates).
    predictor.fit(
        ts.slice_by_timestep(None, hist_len),
        hyperparameters={"Chronos": {"model_path": CHRONOS_MODEL}},
        skip_model_selection=True,
        enable_ensemble=False,
    )
    pred = predictor.predict(
        ts.slice_by_timestep(None, hist_len), known_covariates=known
    )
    out = np.empty((n, future_len, len(quantile_levels)), dtype=np.float32)
    for i in range(n):
        item = pred.loc[i]
        for li, q in enumerate(quantile_levels):
            out[i, :, li] = item[str(q)].to_numpy()[:future_len]
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_protocol(q_vals, future, *, label):
    """Headline metrics for one protocol's quantile predictions."""
    crps = pinball_crps(QUANTILE_LEVELS, q_vals, future)
    mae = quantile_mae(QUANTILE_LEVELS, q_vals, future)
    nll = quantile_nll(QUANTILE_LEVELS, q_vals, future)
    resp_std = future.std(axis=1)
    q = np.quantile(resp_std, [0.25, 0.5, 0.75])
    q4 = np.digitize(resp_std, q) == 3
    return {
        f"{label}_test_nll": float(nll.mean()),
        f"{label}_test_crps": float(crps.mean()),
        f"{label}_test_mae": float(mae.mean()),
        f"{label}_q4_nll": float(nll[q4].mean()) if q4.any() else float("nan"),
        f"{label}_q4_mae": float(mae[q4].mean()) if q4.any() else float("nan"),
        f"{label}_per_step_crps": [float(crps[:, s].mean()) for s in range(future.shape[1])],
        f"{label}_per_step_mae": [float(mae[:, s].mean()) for s in range(future.shape[1])],
    }


def counterfactual(history, fluence, future_len):
    """Counterfactual point-shift ratio + NLL gap for Chronos-2.

    Tests whether Chronos-2's covariate handling actually uses future fluence.
    """
    fl_max = float(np.max(fluence)) if fluence.size else 1.0
    q_on = chronos_predict(
        history, np.full_like(fluence, fl_max), future_len,
        quantile_levels=QUANTILE_LEVELS,
    )
    q_off = chronos_predict(
        history, np.zeros_like(fluence), future_len,
        quantile_levels=QUANTILE_LEVELS,
    )
    mid = int(np.argmin(np.abs(np.asarray(QUANTILE_LEVELS) - 0.5)))
    shift = np.abs(q_on[..., mid] - q_off[..., mid])
    return q_on, q_off, float(shift.mean())


def run(source, *, dry_run, history_len, future_len, result_path, extended_history):
    t0 = time.time()
    hist, fut, flu, _, _ = build_windows(
        source, history_len, future_len, dry_run=dry_run
    )
    print(f"[chronos2] {len(hist)} test windows from `{source}` (H={history_len}, F={future_len})")

    metrics = {}
    # Protocol A — per-window univariate.
    q_a = chronos_predict(hist, flu, future_len, quantile_levels=QUANTILE_LEVELS)
    metrics.update(evaluate_protocol(q_a, fut, label="protocolA"))

    # Protocol B — cross-learning (all windows of the source in one call;
    # AutoGluon conditions each item in-context on the others).
    q_b = chronos_predict(hist, flu, future_len, quantile_levels=QUANTILE_LEVELS)
    metrics.update(evaluate_protocol(q_b, fut, label="protocolB"))

    # Counterfactual diagnostics.
    q_on, q_off, shift = counterfactual(hist, flu, future_len)
    y_std = float(fut.std()) or 1e-12
    nll_actual = metrics["protocolA_test_nll"]
    nll_on = float(quantile_nll(QUANTILE_LEVELS, q_on, fut).mean())
    nll_off = float(quantile_nll(QUANTILE_LEVELS, q_off, fut).mean())
    metrics["counterfactual_point_shift_ratio"] = shift / y_std
    metrics["nll_gap"] = nll_actual - 0.5 * (nll_on + nll_off)

    elapsed = time.time() - t0
    print(f"[chronos2] done in {elapsed:.1f}s")
    for k, v in metrics.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    # register through the standard experiment tracker
    base = result_path or f"{results_write_path()}/chronos2_baseline_{source}"
    tracker = ExperimentTracker(
        directory=base,
        name=f"chronos2_baseline_{source}",
        model_config={"model": CHRONOS_MODEL, "source": source,
                      "history_len": history_len, "future_len": future_len,
                      "extended_history": extended_history},
        training_config={"zero_shot": True, "quantile_levels": QUANTILE_LEVELS},
    )
    tracker.register_start()
    tracker.save_final(
        model=None,
        training_results={"train_elapsed_s": elapsed},
        metrics=metrics,
        figures={},
    )
    print(f"[chronos2] saved → {tracker.directory}")


def _parse_bool(v):
    return str(v).lower() in ("1", "true", "yes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="real_plus_bo")
    p.add_argument("--dry_run", default="true")
    p.add_argument("--history_len", type=int, default=25)
    p.add_argument("--future_len", type=int, default=10)
    p.add_argument("--result-path", default=None)
    p.add_argument("--extended-history", action="store_true",
                   help="also run the H=128 context-length variant")
    args = p.parse_args()
    run(
        args.source,
        dry_run=_parse_bool(args.dry_run),
        history_len=args.history_len,
        future_len=args.future_len,
        result_path=args.result_path,
        extended_history=args.extended_history,
    )
