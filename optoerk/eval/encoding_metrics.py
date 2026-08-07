"""Metrics for comparing optoRTK-expression **encodings** head to head.

Four training runs vary how the model is allowed to use a per-cell expression
rank, and one of them also adds ``nuc_area``. Held-out NLL alone cannot separate
them: a covariate can improve the average loss slightly while being used for the
wrong thing, or be genuinely useful only in the tail of the population that a
mean hides. So this module reports, on a **fixed** eval set shared by every run:

  * :func:`evaluate` — per-sample NLL, per-horizon-step MAE/RMSE, and calibration
    (does the predictive interval actually cover?). Per-sample, so the comparison
    notebook can do *paired* tests rather than comparing two noisy means.
  * :func:`permutation_importance` — shuffle a channel across cells and measure
    how much NLL degrades. The direct answer to "is the model using this at all",
    and the one that decides whether ``nuc_area`` earns its place.
  * :func:`dose_effect` — the model's believed effect of light on each cell,
    obtained by running it twice (real fluence vs fluence forced to zero) and
    differencing. Correlating that with the cell's expression rank is the direct
    test of whether the model learned expression **as a gain**, which is what it
    physically is and the entire reason the encoding variants exist.
  * :func:`stratified_error` — error by expression decile. A covariate that helps
    on average but leaves the high expressers just as wrong has not done its job.

Channel groups matter throughout. ``u_t_x_expr`` is a deterministic function of
``u_t`` and ``optortk_expr``, so permuting expression without permuting the
interaction — or zeroing fluence without zeroing the interaction — feeds the
model a state that cannot occur, and the number that comes back means nothing.
:func:`linked_channels` encodes those dependencies once.
"""
from __future__ import annotations

import numpy as np
import torch


def linked_channels(name: str, channels: list[str]) -> list[str]:
    """Every channel that must move together with ``name`` to stay self-consistent.

    ``optortk_expr`` drags ``u_t_x_expr`` with it, and vice versa: the interaction
    is their product, so changing one alone describes a cell that does not exist.
    """
    linked = {name}
    if name in ("optortk_expr", "u_t", "u_t_x_expr") and "u_t_x_expr" in channels:
        if name in ("optortk_expr", "u_t"):
            linked.add("u_t_x_expr")
    return [c for c in channels if c in linked]


def _std_value(model, channel: str, raw: float) -> float:
    """A raw value in the model's standardized units for one channel."""
    cfg = model.cfg
    i = list(cfg.norm_channels).index(channel)
    return (raw - float(cfg.norm_mean[i])) / float(cfg.norm_std[i])


def _cnr_scale(model) -> tuple[float, float]:
    i = list(model.cfg.norm_channels).index("cnr")
    return float(model.cfg.norm_mean[i]), float(model.cfg.norm_std[i])


@torch.no_grad()
def evaluate(model, loader, device=None) -> dict:
    """Per-sample NLL plus per-horizon error and calibration, in absolute CNR.

    Returns arrays rather than scalars: the runs are compared on the *same* eval
    samples in the same order, so a paired difference is available and a
    difference of means is not the only thing on offer.
    """
    device = device or next(model.parameters()).device
    model.eval()
    cnr_mean, cnr_std = _cnr_scale(model)

    nlls, abs_err, sq_err, z_scores = [], [], [], []
    for batch in loader:
        ctx = batch["ctx"].to(device)
        lens = batch["lengths"].to(device)
        fut = batch["fut_flu"].to(device)
        tgt = batch["tgt"].to(device)
        pi, mu, sigma = model(ctx, lens, fut)
        # NLL per sample (mean over the horizon), the training objective itself.
        nll_bf = -torch.logsumexp(
            torch.log(pi.clamp_min(1e-12))
            - 0.5 * ((tgt.unsqueeze(-1) - mu) / sigma.clamp_min(1e-6)) ** 2
            - torch.log(sigma.clamp_min(1e-6))
            - 0.5 * float(np.log(2 * np.pi)),
            dim=-1,
        )                                                    # (B, F)
        point = (pi * mu).sum(-1)                            # (B, F) standardized
        var = (pi * (sigma ** 2 + (mu - point.unsqueeze(-1)) ** 2)).sum(-1)
        sd = var.clamp_min(1e-12).sqrt()

        err = (point - tgt) * cnr_std                        # absolute CNR units
        nlls.append(nll_bf.mean(dim=1).cpu().numpy())
        abs_err.append(err.abs().cpu().numpy())
        sq_err.append((err ** 2).cpu().numpy())
        # standardized residual: calibration is whether these are ~N(0,1)
        z_scores.append(((point - tgt) / sd.clamp_min(1e-6)).cpu().numpy())

    abs_err = np.concatenate(abs_err)                        # (N, F)
    sq_err = np.concatenate(sq_err)
    z = np.concatenate(z_scores)
    return {
        "nll": np.concatenate(nlls),                         # (N,)
        "mae_per_step": abs_err.mean(axis=0),                # (F,)
        "rmse_per_step": np.sqrt(sq_err.mean(axis=0)),       # (F,)
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "abs_err_per_sample": abs_err.mean(axis=1),          # (N,)
        # Coverage of the nominal central intervals. A well-calibrated predictive
        # distribution puts ~68% / ~95% of residuals inside 1 / 2 sd; the MDN is
        # free to be over-confident, and that shows up here and nowhere else.
        "cov68": float((np.abs(z) <= 1.0).mean()),
        "cov95": float((np.abs(z) <= 1.959964).mean()),
        "z_std": float(z.std()),
    }


@torch.no_grad()
def permutation_importance(model, loader, channel: str, *, device=None,
                           seed: int = 0, repeats: int = 3,
                           linked: bool = True) -> dict:
    """NLL increase when ``channel`` is shuffled across cells within each batch.

    Breaks the channel's association with the target while leaving its marginal
    distribution untouched, so any degradation is attributable to *information
    the model was using*. A channel whose permutation costs nothing is a channel
    the model ignores — which is the question ``nuc_area`` is here to answer.

    Linked channels move with it (see :func:`linked_channels`), and the shuffle is
    applied to the decoder's future inputs too when the channel appears there;
    otherwise the model still sees the true value during the rollout and the
    importance is understated.

    ``linked=False`` shuffles this channel ALONE. The result is not a valid input
    state — permuting expression without its interaction describes a cell that
    cannot exist — so it is not an importance in the causal sense. It is a
    decomposition: it separates what a channel contributes on its own from what
    its linked partners contribute, and the two numbers together are what makes a
    linked importance interpretable.

    On the interaction variant that distinction decides the whole comparison.
    Linked, ``optortk_expr`` scores ~0.53 there against ~0.06 in every other run,
    which reads as "the interaction made the model use expression nine times
    harder". Decomposed, expression alone is ~0.02 and ``u_t_x_expr`` alone is
    ~0.48 against ~0.47 for ``u_t`` — the model turned the interaction into a
    second dose channel and leaned on the expression channel proper LESS than the
    baseline did.
    """
    device = device or next(model.parameters()).device
    model.eval()
    channels = list(model.cfg.norm_channels)
    if channel not in channels:
        raise ValueError(f"{channel!r} not in {channels}")
    group = linked_channels(channel, channels) if linked else [channel]
    idx = [channels.index(c) for c in group]
    # Which of the permuted channels are also decoder inputs, and where.
    fut_names = _future_channel_names(model, loader)
    fut_pos = [(fut_names.index(c), channels.index(c)) for c in group if c in fut_names]

    base, permuted = [], []
    rng = np.random.default_rng(seed)
    for batch in loader:
        ctx = batch["ctx"].to(device)
        lens = batch["lengths"].to(device)
        fut = batch["fut_flu"].to(device)
        tgt = batch["tgt"].to(device)
        base.append(_nll(model, ctx, lens, fut, tgt))
        B = ctx.shape[0]
        for _ in range(repeats):
            perm = torch.as_tensor(rng.permutation(B), device=device)
            ctx_p = ctx.clone()
            ctx_p[:, :, idx] = ctx[perm][:, :, idx]
            fut_p = fut.clone()
            for j, _c in fut_pos:
                fut_p[:, :, j] = fut[perm][:, :, j]
            permuted.append(_nll(model, ctx_p, lens, fut_p, tgt))

    b = float(np.mean(np.concatenate(base)))
    p = float(np.mean(np.concatenate(permuted)))
    return {"channel": channel, "group": group, "linked": linked, "nll": b,
            "nll_permuted": p, "delta_nll": p - b}


@torch.no_grad()
def _nll(model, ctx, lens, fut, tgt) -> np.ndarray:
    """Per-sample NLL. Decorated in its own right, not only via its callers — it
    is the natural building block for ad-hoc diagnostics, and outside a no_grad
    context the ``.numpy()`` at the end raises on a grad-tracking tensor."""
    pi, mu, sigma = model(ctx, lens, fut)
    nll_bf = -torch.logsumexp(
        torch.log(pi.clamp_min(1e-12))
        - 0.5 * ((tgt.unsqueeze(-1) - mu) / sigma.clamp_min(1e-6)) ** 2
        - torch.log(sigma.clamp_min(1e-6))
        - 0.5 * float(np.log(2 * np.pi)),
        dim=-1,
    )
    return nll_bf.mean(dim=1).detach().cpu().numpy()


def _future_channel_names(model, loader) -> list[str]:
    """Names of the decoder's future inputs, from the dataset that built them."""
    ds = getattr(loader, "dataset", None)
    names = getattr(ds, "future_channels", None)
    if names:
        return list(names)
    # stim_dim == 1 can only be fluence.
    return ["u_t"]


@torch.no_grad()
def dose_effect(model, loader, *, device=None) -> dict:
    """The model's believed effect of light, per sample, and its link to expression.

    Runs each sample twice — once with the real commanded fluence over the
    horizon, once with fluence forced to raw zero — and differences the predicted
    mean CNR. That difference is what the model thinks the light *did* for this
    cell.

    Correlating it with the cell's expression rank is the direct test of the whole
    exercise: optoRTK expression is a gain on the dose-response, so a model that
    has actually learned it should predict a larger effect for high expressers.
    A model that merely reduced its average loss will show no such relationship.

    Zeroing fluence also zeroes any interaction channel, in raw units before
    standardizing — the interaction is a product, so raw 0 is the consistent
    counterfactual and standardized 0 is not.
    """
    device = device or next(model.parameters()).device
    model.eval()
    channels = list(model.cfg.norm_channels)
    expr_i = channels.index("optortk_expr")
    cnr_mean, cnr_std = _cnr_scale(model)
    fut_names = _future_channel_names(model, loader)
    zeros = [_std_value(model, c, 0.0) for c in fut_names]

    effects, exprs, doses = [], [], []
    for batch in loader:
        ctx = batch["ctx"].to(device)
        lens = batch["lengths"].to(device)
        fut = batch["fut_flu"].to(device)
        pi, mu, _ = model(ctx, lens, fut)
        lit = (pi * mu).sum(-1).mean(dim=1)
        fut0 = fut.clone()
        for j, z in enumerate(zeros):
            fut0[:, :, j] = z
        pi0, mu0, _ = model(ctx, lens, fut0)
        dark = (pi0 * mu0).sum(-1).mean(dim=1)
        effects.append(((lit - dark) * cnr_std).cpu().numpy())
        rows = torch.arange(ctx.shape[0], device=device)
        last = (lens.to(device) - 1).clamp(min=0)
        exprs.append(ctx[rows, last, expr_i].cpu().numpy())
        # mean commanded fluence over the horizon, standardized -> raw
        j = fut_names.index("u_t")
        i = channels.index("u_t")
        raw = fut[:, :, j].mean(dim=1).cpu().numpy() * float(model.cfg.norm_std[i]) \
            + float(model.cfg.norm_mean[i])
        doses.append(raw)

    eff = np.concatenate(effects)
    ex = np.concatenate(exprs)
    dose = np.concatenate(doses)
    # Only cells that actually got light can show a gain.
    lit_mask = dose > np.percentile(dose, 50)
    from scipy.stats import spearmanr
    rho = float(spearmanr(ex[lit_mask], eff[lit_mask]).statistic) if lit_mask.sum() > 20 else float("nan")
    return {"effect": eff, "expr_std": ex, "dose": dose,
            "gain_spearman": rho, "n_lit": int(lit_mask.sum()),
            "mean_effect": float(eff.mean())}


def stratified_error(abs_err_per_sample: np.ndarray, expr_std: np.ndarray,
                     *, n_bins: int = 10) -> dict:
    """Mean absolute error by expression decile.

    A covariate that lowers the average while leaving the extremes untouched has
    not done the job a gain covariate exists to do.
    """
    q = np.quantile(expr_std, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-9
    bins = np.clip(np.digitize(expr_std, q[1:-1]), 0, n_bins - 1)
    per_bin = np.array([
        abs_err_per_sample[bins == b].mean() if (bins == b).any() else np.nan
        for b in range(n_bins)
    ])
    return {"decile_mae": per_bin,
            "spread": float(np.nanmax(per_bin) - np.nanmin(per_bin)),
            "counts": np.bincount(bins, minlength=n_bins)}
