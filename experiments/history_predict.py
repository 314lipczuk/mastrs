"""Prediction adapters for the full-history model (``Seq2ScalarHistory``).

Two entry points over one core, so the memory ladder and cell_video both drive
the new model:

  * :func:`predict_many` — the memory-ladder interface
    (``cross_stitch_responder.py``): absolute-CNR ``(means, sigmas)`` for a list
    of prediction points, using the **full past** as context.
  * :func:`predict_history_cell` — cell_video's ``PredictFn`` signature
    ``(model, cell, t, future_len, history_len, device) -> (mean (F,), sigma (F,))``.

The model runs in standardized space; inputs are standardized with the frozen
stats carried on ``model.cfg`` (``norm_mean``/``norm_std``, channel order
``[cnr, u_t, fov_density, n_cells_200px]``) and outputs denormalized to absolute
CNR. Crowding channels default to the population mean (→ 0 after standardizing)
when unavailable (e.g. the ladder's synthetic probes).
"""
from __future__ import annotations

import numpy as np
import torch


def _stats(model):
    cfg = model.cfg
    return np.asarray(cfg.norm_mean, np.float32), np.asarray(cfg.norm_std, np.float32)


def predict_many(model, cnr, flu, ts, *, fov=None, n200=None, device=None, cap=None):
    """Absolute-CNR ``(means, sigmas)`` each ``(len(ts), F)`` for prediction
    points ``ts``, with full-history context ``[max(0, t-cap) : t]``.

    ``fov`` / ``n200`` are per-frame crowding arrays ``(T,)`` or None (→ neutral
    population mean). ``cap`` bounds the context length (None = unbounded).
    """
    cfg = model.cfg
    mean, std = _stats(model)
    cnr = np.asarray(cnr, np.float32)
    flu = np.asarray(flu, np.float32)
    T = len(cnr)
    fov = np.full(T, mean[2], np.float32) if fov is None else np.asarray(fov, np.float32)
    n200 = np.full(T, mean[3], np.float32) if n200 is None else np.asarray(n200, np.float32)

    X = np.stack([cnr, flu, fov, n200], axis=-1)            # (T, 4) raw
    Xs = (X - mean) / std                                   # standardized
    F = cfg.future_len
    cap = T if cap is None else cap
    device = device or next(model.parameters()).device

    ctxs, lens, futs = [], [], []
    for t in ts:
        s = max(0, t - cap)
        ctxs.append(Xs[s:t])
        lens.append(t - s)
        futs.append(Xs[t:t + F, 1])                         # standardized fluence future
    Lmax = max(lens)
    ctx_b = np.zeros((len(ts), Lmax, 4), np.float32)
    for i, c in enumerate(ctxs):
        ctx_b[i, : len(c)] = c
    fut_b = np.stack(futs)[..., None]                       # (B, F, 1)

    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(
            torch.from_numpy(ctx_b).to(device),
            torch.tensor(lens),
            torch.from_numpy(fut_b).to(device),
        )
        m_std = (pi * mu).sum(-1).cpu().numpy()             # (B, F) standardized
        s_std = model.pred_std(pi, mu, sigma).cpu().numpy()
    cnr_mean, cnr_std = mean[0], std[0]
    return (m_std * cnr_std + cnr_mean).astype(np.float32), (s_std * cnr_std).astype(np.float32)


def predict_history_cell(model, cell, t, future_len, history_len, device):
    """cell_video ``PredictFn``: single prediction point, full-history context.

    ``cell.stim`` carries the history features ``[u_t, fov_density,
    n_cells_200px]`` (rows). ``future_len``/``history_len`` are ignored — the
    horizon comes from ``model.cfg`` and the context is the full past.
    """
    stim = np.asarray(cell.stim, np.float32)
    flu = stim[0]
    fov = stim[1] if stim.shape[0] > 1 else None
    n200 = stim[2] if stim.shape[0] > 2 else None
    m, s = predict_many(model, cell.cnr, flu, [t], fov=fov, n200=n200, device=device)
    return m[0], s[0]
