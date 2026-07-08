"""Shared helpers for stimulation ablation and counterfactual analysis.

Used by `lstm_experiment_review.py` (single model) and `ensemble_review.py`
(grouped ensemble). Keep numerics matching the in-place code that lived in
those notebooks so results stay comparable across refactors.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from experiments.seq2seq_data import STIM_COLS, _stim_features

_S = {name: i for i, name in enumerate(STIM_COLS)}


def window_samples(cnr: np.ndarray, stim: np.ndarray, H: int, F: int, stride: int = 2) -> list[dict]:
    """Slide an H+F window over each track.

    cnr:  (N_tracks, T)
    stim: (N_tracks, n_stim, T)

    Each sample dict contains:
        enc_in      (H, 1+n_stim)  — encoder input (cnr + stim)
        dec_stim    (F, n_stim)    — ACTUAL future stim features (baseline)
        dec_target  (F,)           — future delta targets
        full_window (H+F,)         — absolute cnr across full window
        light       (H+F,)         — u_t channel across full window
        enc_stim    (n_stim, H)    — encoder-side stim (for patching)
        hist_light  (H,)           — encoder-side u_t (for counterfactual rebuild)
        cell_idx, t_start          — bookkeeping
    """
    total = H + F
    samples: list[dict] = []
    u_idx = _S["u_t"]
    for i in range(len(cnr)):
        t = 0
        while t + total <= cnr.shape[1]:
            enc_cnr = cnr[i, t : t + H]
            enc_stim = stim[i, :, t : t + H]
            dec_stim = stim[i, :, t + H : t + total]
            full_win = cnr[i, t : t + total]
            dec_tgt = np.diff(full_win)[H - 1 : H - 1 + F]
            enc_in = np.concatenate([enc_cnr[:, np.newaxis], enc_stim.T], axis=-1)
            light = stim[i, u_idx, t : t + total]
            samples.append(dict(
                enc_in=enc_in.astype(np.float32),
                dec_stim=dec_stim.T.astype(np.float32),
                dec_target=dec_tgt.astype(np.float32),
                full_window=full_win.astype(np.float32),
                light=light.astype(np.float32),
                enc_stim=enc_stim.astype(np.float32),
                hist_light=stim[i, u_idx, t : t + H].astype(np.float32),
                cell_idx=i,
                t_start=t,
            ))
            t += stride
    return samples


def build_zeroed_decoder_stim(enc_stim: np.ndarray, F: int) -> np.ndarray:
    """Decoder stim features for 'future light = 0'.

    Stateful channels (s_cum, ewma_fast, ewma_slow, recency) decay from the
    encoder boundary value; window-local channels (m_t, u_t, n_5, slope_5,
    burst_pos) are zero because no pulses fall inside the future window.
    """
    H = enc_stim.shape[1]
    full_light = np.concatenate([np.zeros(H), np.zeros(F)])[np.newaxis, :]
    full_feats = _stim_features(full_light)
    dec = full_feats[0, :, H:].T.copy()

    dec[:, _S["s_cum"]] = enc_stim[_S["s_cum"], -1]
    dec[:, _S["ewma_fast"]] = enc_stim[_S["ewma_fast"], -1] * (0.5 ** np.arange(1, F + 1))
    dec[:, _S["ewma_slow"]] = enc_stim[_S["ewma_slow"], -1] * (0.9 ** np.arange(1, F + 1))
    dec[:, _S["recency"]] = enc_stim[_S["recency"], -1] * (np.exp(-1.0 / 5.0) ** np.arange(1, F + 1))
    return dec.astype(np.float32)


def build_counterfactual_decoder_stim(
    enc_stim: np.ndarray,
    hist_light: np.ndarray,
    future_light: np.ndarray,
) -> np.ndarray:
    """Decoder stim for an arbitrary hypothetical future light pattern.

    Rebuilds all 9 channels by running ``_stim_features`` over
    ``[hist_light | future_light]`` and slicing the future portion. The
    cumulative channels (s_cum, ewma_fast, ewma_slow) are then patched to
    start from the encoder-boundary state, because hist_light alone does not
    reproduce the trajectory-wide cumulatives from the dataset.
    """
    F = len(future_light)
    H = enc_stim.shape[1]
    full_light = np.concatenate([hist_light, future_light])[np.newaxis, :]
    full_feats = _stim_features(full_light)
    dec = full_feats[0, :, H:].T.copy()

    dec[:, _S["s_cum"]] = enc_stim[_S["s_cum"], -1] + np.cumsum(future_light)

    for alpha, ch in [(0.5, "ewma_fast"), (0.1, "ewma_slow")]:
        decay = (1 - alpha) ** np.arange(1, F + 1)
        conv = np.array([
            sum((1 - alpha) ** j * future_light[k - j] for j in range(k + 1))
            for k in range(F)
        ])
        dec[:, _S[ch]] = enc_stim[_S[ch], -1] * decay + alpha * conv
    return dec.astype(np.float32)


ScenarioBuilder = Callable[[dict], np.ndarray]


def default_scenarios(F: int) -> dict[str, ScenarioBuilder]:
    """Standard ablation + counterfactual scenarios keyed by name."""
    always_off = np.zeros(F, dtype=np.float32)
    always_on = np.ones(F, dtype=np.float32)
    pulse = np.array([1.0 if (k // 3) % 2 == 0 else 0.0 for k in range(F)], dtype=np.float32)

    return {
        "true": lambda s: s["dec_stim"],
        "zeroed": lambda s: build_zeroed_decoder_stim(s["enc_stim"], F),
        "always_on": lambda s: build_counterfactual_decoder_stim(s["enc_stim"], s["hist_light"], always_on),
        "always_off": lambda s: build_counterfactual_decoder_stim(s["enc_stim"], s["hist_light"], always_off),
        "pulse_3on3off": lambda s: build_counterfactual_decoder_stim(s["enc_stim"], s["hist_light"], pulse),
    }


def ensemble_predict_scenarios(
    models: list,
    samples: list[dict],
    scenarios: Mapping[str, ScenarioBuilder],
    device,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Run every model across every scenario on the same sample set.

    Returns a dict mapping scenario name to an array of shape (M, N, F) of
    predicted deltas, where M = len(models), N = len(samples), F = horizon.
    """
    import torch

    N = len(samples)
    enc_batch = np.stack([s["enc_in"] for s in samples])
    dec_by_scen = {name: np.stack([builder(s) for s in samples]) for name, builder in scenarios.items()}

    out: dict[str, list[np.ndarray]] = {name: [] for name in scenarios}
    for model in models:
        model.eval()
        preds_by_scen: dict[str, list[np.ndarray]] = {name: [] for name in scenarios}
        with torch.no_grad():
            for i in range(0, N, batch_size):
                enc_t = torch.tensor(enc_batch[i : i + batch_size]).to(device)
                for name in scenarios:
                    dec_t = torch.tensor(dec_by_scen[name][i : i + batch_size]).to(device)
                    pred = model(enc_t, dec_t).cpu().numpy()
                    preds_by_scen[name].append(pred)
        for name in scenarios:
            out[name].append(np.concatenate(preds_by_scen[name], axis=0))
    return {name: np.stack(out[name], axis=0) for name in scenarios}
