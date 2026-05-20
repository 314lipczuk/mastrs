"""Shared evaluation battery for :mod:`experiments.seq2scal_models` forecasters.

Head-agnostic: the Gaussian head is the K=1 special case of the MDN, so every
metric is computed on ``(pi, mu, sigma)`` arrays of shape ``(N, F, K)`` and
works for both. Mixture-only metrics (mode usage, pi-flip fraction) report
``None`` when ``K == 1``.

The full battery is computed for the run log; :func:`summary_table` pulls the
10 headline metrics into a fixed comparison block (handoff "Reporting").

Public API
----------
- :func:`evaluate`          -> :class:`EvalResult` (full battery).
- :func:`step0_diagnostics` -> dict (Task 2 per-step / step-0 analysis).
- :func:`summary_table`     -> str (the 10 headline metrics, fixed format).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from experiments.seq2scal_models import Seq2SeqDataset

IDEAL_COVERAGE = {1: 0.6826895, 2: 0.9544997, 3: 0.9973002}

# ---------------------------------------------------------------------------
# pointwise metric primitives  (all operate on (N, F, K) / (N, F) numpy arrays)
# ---------------------------------------------------------------------------


def _nll_per(pi, mu, sigma, y):
    yk = y[..., None]
    log_p = np.log(pi + 1e-12)
    log_g = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((yk - mu) / sigma) ** 2
    return -logsumexp(log_p + log_g, axis=-1)  # (N, F)


def _A(u, s):
    z = u / (s + 1e-12)
    return u * (2 * norm.cdf(z) - 1) + 2 * s * norm.pdf(z)


def _crps_per(pi, mu, sigma, y):
    """Closed-form mixture CRPS (Grimit et al. 2006); Gaussian CRPS at K=1."""
    yk = y[..., None]
    term1 = (pi * _A(yk - mu, sigma)).sum(-1)
    mu_j, mu_k = mu[..., :, None], mu[..., None, :]
    s_j, s_k = sigma[..., :, None], sigma[..., None, :]
    pi_jk = pi[..., :, None] * pi[..., None, :]
    term2 = 0.5 * (pi_jk * _A(mu_j - mu_k, np.sqrt(s_j ** 2 + s_k ** 2))).sum((-1, -2))
    return term1 - term2  # (N, F)


def _point(pi, mu):
    return (pi * mu).sum(-1)


def _std(pi, mu, sigma):
    mean = _point(pi, mu)[..., None]
    return np.sqrt((pi * (sigma ** 2 + (mu - mean) ** 2)).sum(-1)).clip(1e-12)


def _pit(pi, mu, sigma, y):
    z = (y[..., None] - mu) / sigma
    return (pi * norm.cdf(z)).sum(-1)


# ---------------------------------------------------------------------------
# model output collection
# ---------------------------------------------------------------------------


@dataclass
class _Outputs:
    pi: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    target: np.ndarray   # (N, F) delta-CNR
    last: np.ndarray     # (N,)   last observed CNR
    resp_std: np.ndarray  # (N,)  full-window response std
    boundary_slope: np.ndarray  # (N,) OLS slope of last-5 history CNR


def _ols_slope(window):
    n = window.shape[-1]
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    ss = (xc ** 2).sum()
    return (window - window.mean(-1, keepdims=True)).dot(xc) / ss if ss > 0 else 0.0


@torch.no_grad()
def _collect(model, ds, device, *, batch=512, enc_zero_fluence=False, dec_stim_value=None):
    """Run the model over a dataset, optionally ablating fluence inputs.

    ``enc_zero_fluence``: zero encoder fluence channel (idx 1).
    ``dec_stim_value``: if not None, fill the decoder stim with this constant.
    """
    model.eval()
    pis, mus, sigs, tgts, lasts, rstd, bslope = [], [], [], [], [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
    for enc_in, dec_stim, dec_target in loader:
        enc = enc_in.clone().to(device)
        stim = dec_stim.clone().to(device)
        if enc_zero_fluence:
            enc[:, :, 1] = 0.0
        if dec_stim_value is not None:
            stim[:] = dec_stim_value
        pi, mu, sigma = model(enc, stim)
        pis.append(pi.cpu().numpy())
        mus.append(mu.cpu().numpy())
        sigs.append(sigma.cpu().numpy())
        tgts.append(dec_target.numpy())
        lasts.append(enc_in[:, -1, 0].numpy())
        bslope.append(_ols_slope(enc_in[:, -5:, 0].numpy()))
    pi = np.concatenate(pis)
    out = _Outputs(
        pi=pi, mu=np.concatenate(mus), sigma=np.concatenate(sigs),
        target=np.concatenate(tgts), last=np.concatenate(lasts),
        resp_std=ds.resp_std.copy(), boundary_slope=np.concatenate(bslope),
    )
    return out


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    headline: dict = field(default_factory=dict)   # the 10 summary metrics
    full: dict = field(default_factory=dict)       # demoted / appendix metrics
    per_step: dict = field(default_factory=dict)   # F-length arrays

    @property
    def is_mixture(self):
        return self.headline.get("pi_flip_fraction") is not None


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------


def evaluate(model, test_arrays, mcfg, *, device, test_stride=10, fluence_on=None):
    """Run the full eval battery on the held-out test cells.

    ``test_arrays`` is the 5-tuple from :func:`seq2scal_models.prepare_data`'s
    ``.test`` field. ``fluence_on`` is the constant used for the all-on
    counterfactual (defaults to the max observed test fluence).
    """
    ds = Seq2SeqDataset(test_arrays, mcfg.history_len, mcfg.future_len, stride=test_stride)
    base = _collect(model, ds, device)
    K = base.pi.shape[-1]
    is_mix = K > 1

    if fluence_on is None:
        fluence_on = float(max((np.max(f) for f in test_arrays[1]), default=1.0))

    # --- accuracy ----------------------------------------------------------
    nll = _nll_per(base.pi, base.mu, base.sigma, base.target)        # (N, F)
    crps = _crps_per(base.pi, base.mu, base.sigma, base.target)
    point = _point(base.pi, base.mu)
    std = _std(base.pi, base.mu, base.sigma)
    abs_resid = np.abs(point - base.target)

    # response-magnitude quartiles
    q = np.quantile(base.resp_std, [0.25, 0.5, 0.75])
    qbin = np.digitize(base.resp_std, q)  # 0..3
    q4 = qbin == 3

    # --- calibration -------------------------------------------------------
    cov_table = {}
    for k in (1, 2, 3):
        cov_table[k] = [
            float(np.mean(abs_resid[:, s] <= k * std[:, s]))
            for s in range(mcfg.future_len)
        ]
    mean_1sig = float(np.mean(cov_table[1]))
    dev_from_ideal = float(np.mean([
        np.mean(cov_table[k]) - IDEAL_COVERAGE[k] for k in (1, 2, 3)
    ]))

    # --- MPC / counterfactual ---------------------------------------------
    on = _collect(model, ds, device, dec_stim_value=fluence_on)
    off = _collect(model, ds, device, dec_stim_value=0.0)
    pt_on, pt_off = _point(on.pi, on.mu), _point(off.pi, off.mu)
    point_shift = np.abs(pt_on - pt_off)
    y_std = float(base.target.std()) or 1e-12
    cf_ratio = float(point_shift.mean() / y_std)

    nll_actual = float(nll.mean())
    nll_on = float(_nll_per(on.pi, on.mu, on.sigma, base.target).mean())
    nll_off = float(_nll_per(off.pi, off.mu, off.sigma, base.target).mean())
    nll_gap = nll_actual - 0.5 * (nll_on + nll_off)

    # all_stim_zero: zero encoder fluence AND decoder stim
    zero = _collect(model, ds, device, enc_zero_fluence=True, dec_stim_value=0.0)
    nll_zero = float(_nll_per(zero.pi, zero.mu, zero.sigma, base.target).mean())
    all_stim_zero_dnll = nll_zero - nll_actual

    # cnr_only_zero (demoted): zero encoder cnr channel only
    # mixture-only: pi-flip fraction under on/off counterfactual
    if is_mix:
        flip = float(np.mean(on.pi.argmax(-1) != off.pi.argmax(-1)))
        pi_entropy = float(np.mean(-(base.pi * np.log(base.pi + 1e-12)).sum(-1)))
        frac_single = float(np.mean(base.pi.max(-1) > 0.95))
    else:
        flip = pi_entropy = frac_single = None

    headline = {
        "test_nll": nll_actual,
        "test_crps": float(crps.mean()),
        "q4_nll": float(nll[q4].mean()) if q4.any() else float("nan"),
        "q4_mae": float(abs_resid[q4].mean()) if q4.any() else float("nan"),
        "mean_1sigma_coverage": mean_1sig,
        "mean_deviation_from_ideal": dev_from_ideal,
        "counterfactual_point_shift_ratio": cf_ratio,
        "all_stim_zero_dnll": all_stim_zero_dnll,
        "nll_gap": nll_gap,
        "pi_flip_fraction": flip,
    }

    # --- appendix / demoted metrics ---------------------------------------
    pit = _pit(base.pi, base.mu, base.sigma, base.target).ravel()
    pit_sorted = np.sort(pit)
    ks = float(np.max(np.abs(pit_sorted - np.linspace(0, 1, len(pit_sorted)))))

    # sharpness vs accuracy deciles
    sig_flat, res_flat = std.ravel(), abs_resid.ravel()
    edges = np.quantile(sig_flat, np.linspace(0, 1, 11))
    dbin = np.clip(np.digitize(sig_flat, edges[1:-1]), 0, 9)
    sharpness = [
        {
            "decile": d,
            "sigma_mean": float(sig_flat[dbin == d].mean()),
            "abs_resid_mean": float(res_flat[dbin == d].mean()),
            "ideal_abs_resid": float(0.7979 * sig_flat[dbin == d].mean()),
        }
        for d in range(10) if (dbin == d).any()
    ]

    strat = []
    for b in range(4):
        m = qbin == b
        if m.any():
            strat.append({
                "quartile": b + 1, "n": int(m.sum()),
                "nll": float(nll[m].mean()), "crps": float(crps[m].mean()),
                "mae": float(abs_resid[m].mean()), "sigma": float(std[m].mean()),
            })

    full = {
        "test_mae": float(abs_resid.mean()),
        "test_mean_sigma": float(std.mean()),
        "pit_ks": ks,
        "calibration_coverage": {f"{k}sigma": cov_table[k] for k in (1, 2, 3)},
        "sharpness_deciles": sharpness,
        "stratified_by_response": strat,
        "counterfactual": {
            "fluence_on": fluence_on,
            "nll_actual": nll_actual, "nll_all_on": nll_on, "nll_all_off": nll_off,
            "mean_abs_point_shift": float(point_shift.mean()), "target_std": y_std,
        },
        "all_stim_zero": {"nll": nll_zero, "dnll": all_stim_zero_dnll},
        "mode_usage": (
            None if not is_mix else
            {"mean_pi_entropy": pi_entropy, "frac_single_mode": frac_single}
        ),
        "n_test_windows": len(ds),
    }

    per_step = {
        "nll": [float(nll[:, s].mean()) for s in range(mcfg.future_len)],
        "crps": [float(crps[:, s].mean()) for s in range(mcfg.future_len)],
        "mae": [float(abs_resid[:, s].mean()) for s in range(mcfg.future_len)],
        "coverage_1sigma": cov_table[1],
    }
    return EvalResult(headline=headline, full=full, per_step=per_step)


def step0_diagnostics(model, test_arrays, mcfg, *, device, test_stride=10):
    """Task 2: per-step and step-0 forecast-error diagnostics.

    Identifies whether the step-0 error is prediction offset, variance
    explosion, slope extrapolation, or sigma_0 miscalibration.
    """
    ds = Seq2SeqDataset(test_arrays, mcfg.history_len, mcfg.future_len, stride=test_stride)
    o = _collect(model, ds, device)
    point = _point(o.pi, o.mu)
    std = _std(o.pi, o.mu, o.sigma)
    nll = _nll_per(o.pi, o.mu, o.sigma, o.target)
    resid = point - o.target  # signed (N, F)

    fl = mcfg.future_len
    per_step = {
        "mae": [float(np.abs(resid[:, s]).mean()) for s in range(fl)],
        "nll": [float(nll[:, s].mean()) for s in range(fl)],
        "mean_residual": [float(resid[:, s].mean()) for s in range(fl)],
        "residual_std": [float(resid[:, s].std()) for s in range(fl)],
        "sigma_mean": [float(std[:, s].mean()) for s in range(fl)],
    }

    # step-0 |residual| stratified by |boundary slope| quartile
    bslope = np.abs(o.boundary_slope)
    qb = np.digitize(bslope, np.quantile(bslope, [0.25, 0.5, 0.75]))
    slope_strat = [
        {
            "slope_quartile": b + 1,
            "n": int((qb == b).sum()),
            "step0_abs_resid": float(np.abs(resid[qb == b, 0]).mean()),
        }
        for b in range(4) if (qb == b).any()
    ]

    return {
        "per_step": per_step,
        "step0_mean_residual": per_step["mean_residual"][0],
        "step0_residual_std": per_step["residual_std"][0],
        "steps1plus_residual_std": float(resid[:, 1:].std()),
        "step0_by_boundary_slope": slope_strat,
    }


def summary_table(result: EvalResult) -> str:
    """The 10 headline metrics in a fixed markdown block (handoff Reporting)."""
    h = result.headline
    flip = "N/A" if h["pi_flip_fraction"] is None else f"{h['pi_flip_fraction']:.4f}"
    rows = [
        ("accuracy", "test NLL", f"{h['test_nll']:.4f}"),
        ("accuracy", "test CRPS", f"{h['test_crps']:.4f}"),
        ("accuracy", "Q4 NLL", f"{h['q4_nll']:.4f}"),
        ("accuracy", "Q4 MAE", f"{h['q4_mae']:.4f}"),
        ("calibration", "mean 1sigma coverage", f"{h['mean_1sigma_coverage']:.4f}"),
        ("calibration", "mean deviation from ideal", f"{h['mean_deviation_from_ideal']:+.4f}"),
        ("MPC", "counterfactual point-shift ratio", f"{h['counterfactual_point_shift_ratio']:.4f}"),
        ("MPC", "all_stim_zero dNLL", f"{h['all_stim_zero_dnll']:+.4f}"),
        ("MPC", "NLL gap", f"{h['nll_gap']:+.4f}"),
        ("MPC (MDN only)", "pi flip fraction", flip),
    ]
    lines = ["", "=== SUMMARY TABLE (10 headline metrics) ===",
             f"{'group':<16}{'metric':<34}{'value':>12}"]
    lines += [f"{g:<16}{m:<34}{v:>12}" for g, m, v in rows]
    lines.append("=" * 62)
    return "\n".join(lines)
