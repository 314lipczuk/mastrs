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


def predict_many(model, cnr, flu, ts, *, fov=None, n200=None, expr=None,
                 channels=None, device=None, cap=None):
    """Absolute-CNR ``(means, sigmas)`` each ``(len(ts), F)`` for prediction
    points ``ts``, with full-history context ``[max(0, t-cap) : t]``.

    ``fov`` / ``n200`` are per-frame crowding arrays ``(T,)``; ``expr`` is the
    per-cell optoRTK expression channel ``(T,)`` (constant per cell). Any of these
    left None default to the neutral population mean. ``cap`` bounds the context
    length (None = unbounded).

    ``channels`` is the general form: ``{name: (T,) array}`` for any channel the
    caller can supply, merged over the three keywords above. Prefer it whenever
    the feature set is not the historical four — a caller that knows only
    ``fov``/``n200``/``expr`` cannot express ``nuc_area``, and one that packs its
    arrays positionally will silently misalign the moment a channel is added or
    removed (which is exactly what the `area_lean` feature set does).
    """
    cfg = model.cfg
    mean, std = _stats(model)
    cnr = np.asarray(cnr, np.float32)
    flu = np.asarray(flu, np.float32)
    T = len(cnr)

    # Assemble input channels by NAME in the model's channel order, so this stays
    # correct as the feature set grows/reorders. Unsupplied → population mean.
    chans = list(cfg.norm_channels)
    supplied = {
        "cnr": cnr,
        "u_t": flu,
        "fov_density": None if fov is None else np.asarray(fov, np.float32),
        "n_cells_200px": None if n200 is None else np.asarray(n200, np.float32),
        "optortk_expr": None if expr is None else np.asarray(expr, np.float32),
    }
    for name, arr in (channels or {}).items():
        if arr is not None:
            supplied[name] = np.asarray(arr, np.float32)
    # The interaction is a product, so it is reconstructed exactly whenever both
    # of its inputs are known. Feeding its population mean alongside a real
    # fluence would describe a cell that cannot exist.
    if "u_t_x_expr" in chans and supplied.get("u_t_x_expr") is None:
        e = supplied.get("optortk_expr")
        supplied["u_t_x_expr"] = None if e is None else flu * e

    cols = []
    for i, name in enumerate(chans):
        v = supplied.get(name)
        cols.append(np.full(T, mean[i], np.float32) if v is None else v)
    X = np.stack(cols, axis=-1)                             # (T, C) raw
    Xs = (X - mean) / std                                   # standardized

    C = len(chans)
    cnr_idx = chans.index("cnr")
    # The decoder's known-future inputs, from the model's own config — NOT always
    # just fluence. A stim_dim=2 model (the interaction variant) needs both, and
    # hardcoding u_t would hand it a half-width tensor.
    fut_names = list(getattr(cfg, "future_channels", None) or ["u_t"])
    fut_idx = [chans.index(c) for c in fut_names]
    F = cfg.future_len
    cap = T if cap is None else cap
    device = device or next(model.parameters()).device

    ctxs, lens, futs = [], [], []
    for t in ts:
        s = max(0, t - cap)
        ctxs.append(Xs[s:t])
        lens.append(t - s)
        futs.append(Xs[t:t + F][:, fut_idx])               # (F, n_future) standardized
    Lmax = max(lens)
    ctx_b = np.zeros((len(ts), Lmax, C), np.float32)
    for i, c in enumerate(ctxs):
        ctx_b[i, : len(c)] = c
    fut_b = np.stack(futs)                                  # (B, F, n_future)

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

    ``cell.stim`` carries the model's feature rows in the model's own channel
    order — i.e. ``model.cfg.norm_channels`` minus the leading ``cnr``. The caller
    (``cell_video``) loads the tracks with exactly that feature set, which is what
    makes the zip below exact.

    This used to unpack ``stim`` by fixed row index (``stim[1]`` is fov_density,
    ``stim[3]`` is expression, ...), hardcoded to the historical four features.
    That silently misaligns the moment a feature set differs: under ``area_lean``,
    which drops ``fov_density``, the model would have been fed crowding as
    density, expression as crowding and cell area as expression, and nothing
    anywhere would have said so. Naming the rows removes the whole class of bug.

    ``future_len`` / ``history_len`` are ignored — the horizon comes from
    ``model.cfg`` and the context is the full past.
    """
    stim = np.asarray(cell.stim, np.float32)
    names = list(model.cfg.norm_channels)[1:]               # drop the cnr channel
    if stim.shape[0] != len(names):
        raise ValueError(
            f"cell.stim has {stim.shape[0]} feature rows but the model expects "
            f"{len(names)} ({names}). Load the tracks with the bundle's own "
            f"feature set — see cell_video's history branch."
        )
    channels = dict(zip(names, stim))
    m, s = predict_many(model, cell.cnr, channels["u_t"], [t],
                        channels=channels, device=device)
    return m[0], s[0]
