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
``model.cfg.norm_channels`` = ``[cnr, u_t, fov_density, n_cells_200px,
optortk_expr]``) and outputs denormalized to absolute CNR. Input channels are
assembled **by name in the model's channel order**, so this stays correct as
``CHANNELS`` grows. Channels not supplied default to the population mean (→ 0
after standardizing): crowding for the ladder's synthetic probes, optoRTK
expression when a cell's value is absent.
"""
from __future__ import annotations

import numpy as np
import torch


def _stats(model):
    cfg = model.cfg
    return np.asarray(cfg.norm_mean, np.float32), np.asarray(cfg.norm_std, np.float32)


def predict_many(model, cnr, flu, ts, *, fov=None, n200=None, expr=None, device=None, cap=None):
    """Absolute-CNR ``(means, sigmas)`` each ``(len(ts), F)`` for prediction
    points ``ts``, with full-history context ``[max(0, t-cap) : t]``.

    ``fov`` / ``n200`` are per-frame crowding arrays ``(T,)``; ``expr`` is the
    per-cell optoRTK expression channel ``(T,)`` (constant per cell). Any of these
    left None default to the neutral population mean. ``cap`` bounds the context
    length (None = unbounded).
    """
    cfg = model.cfg
    mean, std = _stats(model)
    cnr = np.asarray(cnr, np.float32)
    flu = np.asarray(flu, np.float32)
    T = len(cnr)

    # Assemble input channels by NAME in the model's channel order, so this stays
    # correct as CHANNELS grows/reorders. Unsupplied channels → population mean.
    chans = list(cfg.norm_channels)
    supplied = {
        "cnr": cnr,
        "u_t": flu,
        "fov_density": None if fov is None else np.asarray(fov, np.float32),
        "n_cells_200px": None if n200 is None else np.asarray(n200, np.float32),
        "optortk_expr": None if expr is None else np.asarray(expr, np.float32),
    }
    cols = []
    for i, name in enumerate(chans):
        if name not in supplied:
            raise ValueError(
                f"predict_many has no input array for channel {name!r} "
                f"(model channels={chans}); add it to `supplied`."
            )
        v = supplied[name]
        cols.append(np.full(T, mean[i], np.float32) if v is None else v)
    X = np.stack(cols, axis=-1)                             # (T, C) raw
    Xs = (X - mean) / std                                   # standardized

    C = len(chans)
    flu_idx = chans.index("u_t")
    cnr_idx = chans.index("cnr")
    F = cfg.future_len
    cap = T if cap is None else cap
    device = device or next(model.parameters()).device

    ctxs, lens, futs = [], [], []
    for t in ts:
        s = max(0, t - cap)
        ctxs.append(Xs[s:t])
        lens.append(t - s)
        futs.append(Xs[t:t + F, flu_idx])                  # standardized fluence future
    Lmax = max(lens)
    ctx_b = np.zeros((len(ts), Lmax, C), np.float32)
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
    cnr_mean, cnr_std = mean[cnr_idx], std[cnr_idx]
    return (m_std * cnr_std + cnr_mean).astype(np.float32), (s_std * cnr_std).astype(np.float32)


def predict_history_cell(model, cell, t, future_len, history_len, device):
    """cell_video ``PredictFn``: single prediction point, full-history context.

    ``cell.stim`` carries the history features ``[u_t, fov_density,
    n_cells_200px, optortk_expr]`` (rows, = HISTORY_FEATURES). ``future_len``/
    ``history_len`` are ignored — the horizon comes from ``model.cfg`` and the
    context is the full past.
    """
    stim = np.asarray(cell.stim, np.float32)
    flu = stim[0]
    fov = stim[1] if stim.shape[0] > 1 else None
    n200 = stim[2] if stim.shape[0] > 2 else None
    expr = stim[3] if stim.shape[0] > 3 else None
    m, s = predict_many(model, cell.cnr, flu, [t], fov=fov, n200=n200, expr=expr, device=device)
    return m[0], s[0]
