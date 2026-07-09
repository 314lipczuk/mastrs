"""Absolute-output seq2scalar LSTM forecaster for ERK-CNR experiments.

Variant of ``seq2scal_models.py``: the model predicts the **absolute CNR
value** at each future step directly (head output = CNR, per-step sigma is the
band around that absolute value), rather than the per-step delta-CNR. Two spots
differ from the delta module — the dataset target (:class:`Seq2SeqDataset`) and
the decoder's autoregressive feedback (:meth:`Seq2ScalarSeq.forward`); both are
flagged inline. Everything else (heads, configs, FiLM, stim-init, fit) is the
same so :mod:`experiments.eval_seq2scal_abs` can reuse the same plumbing.

A single model class, :class:`Seq2ScalarSeq`, with explicit architecture flags
on :class:`ModelConfig` so the handoff variants compose:

- ``head_type``      : ``"mdn"`` (K-component mixture) or ``"gaussian"`` (K=1).
- ``decoder_type``   : ``"encdec"`` (separate decoder LSTM initialised from the
                       encoder's final state) or ``"continuous"`` (one LSTM run
                       through history+future, no boundary).
- ``stim_init``      : bias the LSTM initial state on the future stim sequence.
- ``film``           : ``"none"`` / ``"output"`` (4a) / ``"hidden"`` (4b).
- ``sigma_step_bias``: a learnable per-forecast-step bias added to log-sigma.

The Gaussian head is the **K=1** special case of the mixture, so the eval
battery in :mod:`experiments.eval_seq2scal` treats both heads identically
(``pi`` is all-ones for Gaussian); mixture-only metrics are reported N/A there.

Architecture note: unlike the legacy sliding-window re-encode model in
``lstm_seq2scal_mdn.py``, this module uses a genuine encoder->decoder so that
stim-gated init (Task 3) and FiLM (Task 4) are well defined and compose. The
in-table "baseline MDN" should therefore be a ``head_type=mdn`` run of *this*
module (job ``mdn_baseline_bo``), not the legacy run, so comparisons are
apples-to-apples.

Encoder feature layout (minfeats, no images), 5 channels:
    [cnr, fluence, baseline, ewma_slow(cnr), ewma_fast(cnr)]
Decoder stim is fluence only (stim_dim=1). Targets are absolute CNR values.
"""
from __future__ import annotations

import math
import os
import random
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from optoerk.data.seq2seq_data import load as load_dataset

# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def _init_forget_bias(lstm: nn.LSTM) -> None:
    """Initialise LSTM forget-gate bias to 1.0 (standard stability trick)."""
    for name, param in lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(1.0)


class MDNHead(nn.Module):
    """K-component mixture-density head: pi (softmax), mu, sigma (exp)."""

    def __init__(self, in_feat: int, n_gaussians: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.pi_linear = nn.Linear(in_feat, n_gaussians)
        self.mu = nn.Linear(in_feat, n_gaussians)
        self.log_sigma = nn.Linear(in_feat, n_gaussians)

    def forward(self, x, sigma_bias=0.0):
        pi = torch.softmax(self.pi_linear(x), dim=-1)
        mu = self.mu(x)
        sigma = torch.exp(self.log_sigma(x) + sigma_bias).clamp(min=1e-3)
        return pi, mu, sigma


class GaussianHead(nn.Module):
    """Single heteroscedastic Gaussian head — the K=1 mixture special case.

    Returns ``pi`` of all-ones shape (B, 1) so downstream code is head-agnostic.
    """

    def __init__(self, in_feat: int):
        super().__init__()
        self.n_gaussians = 1
        self.mu = nn.Linear(in_feat, 1)
        self.raw_sigma = nn.Linear(in_feat, 1)

    def forward(self, x, sigma_bias=0.0):
        mu = self.mu(x)
        sigma = F_nn.softplus(self.raw_sigma(x) + sigma_bias) + 1e-3
        pi = torch.ones_like(mu)
        return pi, mu, sigma


class FiLMLayer(nn.Module):
    """FiLM modulation: stim vector -> (gamma, beta), each hidden_dim wide.

    Initialised so gamma == 1 and beta == 0 (identity), so a fresh model starts
    as the no-FiLM baseline and learns to modulate gradually.
    """

    def __init__(self, stim_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(stim_dim, hidden_dim)
        self.beta = nn.Linear(stim_dim, hidden_dim)
        for layer in (self.gamma, self.beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, stim):
        return 1.0 + self.gamma(stim), self.beta(stim)


class StimInitProj(nn.Module):
    """Project the flattened future-stim sequence to LSTM (h0, c0).

    Initialised near zero so the model starts at the standard zero-init
    baseline and learns to use the future-stim prior gradually.
    """

    def __init__(self, future_len: int, stim_dim: int, num_layers: int, hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(future_len * stim_dim, num_layers * hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, future_stim):
        b = future_stim.shape[0]
        flat = future_stim.reshape(b, -1)
        hc = self.proj(flat).view(b, self.num_layers, self.hidden_dim, 2)
        h0 = hc[..., 0].permute(1, 0, 2).contiguous()
        c0 = hc[..., 1].permute(1, 0, 2).contiguous()
        return h0, c0


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def nll(pi, mu, sigma, target):
    """Mixture NLL; reduces to the Gaussian NLL when K=1 (one component)."""
    y = target.unsqueeze(-1)
    log_gauss = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(sigma)
        - 0.5 * ((y - mu) / sigma) ** 2
    )
    log_mix = torch.log(pi + 1e-12) + log_gauss
    return -torch.logsumexp(log_mix, dim=-1).mean()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder_dim: int = Field(..., ge=1)
    stim_dim: int = Field(..., ge=1)
    hidden_dim: int = Field(64, ge=1)
    num_layers: int = Field(2, ge=1)
    n_gaussians: int = Field(3, ge=1)
    n_mlp_layers: int = Field(5, ge=1)
    mlp_hidden: int | None = None
    dropout: float = Field(0.1, ge=0.0, le=0.9)
    history_len: int = Field(25, ge=1)
    future_len: int = Field(10, ge=1)
    data_source: str = "real_plus_bo"
    variant: str = "seq2scalar_seq"
    ewma_slow_alpha: float = Field(0.05, gt=0.0, lt=1.0)
    ewma_fast_alpha: float = Field(0.30, gt=0.0, lt=1.0)
    # Prepend `history_len` block-bootstrapped baseline frames (zero light) to
    # each track so the stimulation onset becomes predictable (real data only).
    prepend_baseline: bool = False
    # "random" or "condition_held_out" — see seq2scal_models.prepare_data.
    split_regime: str = "random"
    # architecture flags
    head_type: str = "mdn"
    decoder_type: str = "encdec"
    stim_init: bool = False
    film: str = "none"
    sigma_step_bias: bool = False

    @model_validator(mode="after")
    def _validate(self):
        if self.mlp_hidden is None:
            object.__setattr__(self, "mlp_hidden", self.hidden_dim)
        if self.head_type not in ("mdn", "gaussian"):
            raise ValueError(f"head_type must be mdn|gaussian, got {self.head_type!r}")
        if self.decoder_type not in ("encdec", "continuous"):
            raise ValueError(f"decoder_type must be encdec|continuous, got {self.decoder_type!r}")
        if self.film not in ("none", "output", "hidden"):
            raise ValueError(f"film must be none|output|hidden, got {self.film!r}")
        if self.split_regime not in ("random", "condition_held_out"):
            raise ValueError(
                f"split_regime must be random|condition_held_out, got {self.split_regime!r}"
            )
        if self.head_type == "gaussian":
            object.__setattr__(self, "n_gaussians", 1)
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 900
    batch_size: int = 900
    patience: int = 100
    tf_ratio_start: float = 1.0
    tf_ratio_end: float = 0.0
    tf_anneal_frac: float = 120.0 / 900.0
    tf_hold_frac: float = 0.0
    grad_clip: float = 1.0
    train_stride: int = Field(5, ge=1)
    test_stride: int = Field(10, ge=1)
    use_stratified_sampler: bool = True
    n_strata: int = Field(3, ge=1)
    # "balanced"          → 1/count per stratum (default; equal mass per bin).
    # "quartile_weighted" → use `quartile_weights` directly; requires n_strata==len(weights).
    sampler_type: str = "balanced"
    quartile_weights: list[float] | None = None
    seed: int = 42

    @field_validator("quartile_weights", mode="before")
    @classmethod
    def _parse_quartile_weights(cls, v):
        # CLI args arrive as strings like "[0.1,0.2,0.3,0.4]"; parse before list[float] coercion.
        if v in (None, "", "None"):
            return None
        if isinstance(v, str):
            import json
            s = v.strip()
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                s = s.strip("[](){} ")
                return [float(x) for x in s.split(",") if x.strip()]
        return v

    @model_validator(mode="after")
    def _check_sampler(self):
        if self.sampler_type not in ("balanced", "quartile_weighted"):
            raise ValueError(
                f"sampler_type must be 'balanced' or 'quartile_weighted', got {self.sampler_type!r}"
            )
        if self.sampler_type == "quartile_weighted":
            w = self.quartile_weights
            if w is None or len(w) != self.n_strata:
                raise ValueError(
                    f"quartile_weighted requires quartile_weights of length n_strata={self.n_strata}, got {w}"
                )
            if any(x <= 0 for x in w):
                raise ValueError(f"quartile_weights must all be > 0, got {w}")
            if abs(sum(w) - 1.0) > 1e-6:
                raise ValueError(f"quartile_weights must sum to 1.0, got {sum(w):.6f}")
        return self


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _ewma_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


BO_OSC_PREFIX = "bo_osc_"
OOD_CONDITIONS = ("Sustained", "ramp1", "3-2-1minIntervals")


@dataclass
class PreparedData:
    """Per-cell minfeats arrays split into train/val/test (+ optional OOD).

    Each split is a 5-tuple of object arrays:
        (cnr, fluence, baseline, ewma_slow, ewma_fast)

    See ``seq2scal_models.PreparedData`` — same shape; ``test_ood`` is populated
    when ``split_regime="condition_held_out"``.
    """

    train: tuple
    val: tuple
    test: tuple
    n_cells: int
    data_source: str
    test_ood: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)


def prepare_data(
    data_source: str,
    *,
    ewma_slow_alpha: float = 0.05,
    ewma_fast_alpha: float = 0.30,
    dry_run: bool = False,
    seed: int = 42,
    baseline_prepend: int = 0,
    split_regime: str = "random",
) -> PreparedData:
    """Same contract as :func:`seq2scal_models.prepare_data`.

    ``split_regime="condition_held_out"`` partitions ``bo_osc_*`` cells 70/10/20
    and reserves Sustained / ramp1 / 3-2-1minIntervals entirely for per-condition
    OOD test sets.
    """
    cnr_all, stim_all, conditions = load_dataset(
        data_source, baseline_prepend=baseline_prepend
    )
    n = len(cnr_all)
    conditions = np.asarray(conditions)

    cnr = np.empty(n, dtype=object)
    fluence = np.empty(n, dtype=object)
    baseline = np.empty(n, dtype=object)
    ewma_slow = np.empty(n, dtype=object)
    ewma_fast = np.empty(n, dtype=object)
    for i in range(n):
        ci = np.asarray(cnr_all[i], dtype=np.float32)
        cnr[i] = ci
        fluence[i] = np.asarray(stim_all[i][0], dtype=np.float32)  # u_t channel
        base = float(np.median(ci[: min(10, len(ci))]))
        baseline[i] = np.full(len(ci), base, dtype=np.float32)
        ewma_slow[i] = _ewma_1d(ci, ewma_slow_alpha)
        ewma_fast[i] = _ewma_1d(ci, ewma_fast_alpha)

    def _pick(idx):
        return tuple(arr[idx] for arr in (cnr, fluence, baseline, ewma_slow, ewma_fast))

    test_ood_dict: dict = {}
    if split_regime == "condition_held_out":
        bo_idx = np.where(
            np.array([str(c).startswith(BO_OSC_PREFIX) for c in conditions])
        )[0]
        if bo_idx.size == 0:
            raise ValueError(
                "condition_held_out: no bo_osc_* cells found in dataset "
                f"{data_source!r} (have: {sorted(set(conditions))})"
            )
        tr, te = train_test_split(bo_idx, test_size=0.2, random_state=seed)
        tr, va = train_test_split(tr, test_size=0.125, random_state=seed)
        ood_index_lists: dict = {}
        for cond in OOD_CONDITIONS:
            cond_idx = np.where(conditions == cond)[0]
            if cond_idx.size > 0:
                test_ood_dict[cond] = _pick(cond_idx)
                ood_index_lists[cond] = [int(i) for i in cond_idx]
        if dry_run:
            tr, va, te = tr[:800], va[:200], te[:200]
            test_ood_dict = {
                k: tuple(arr[:50] for arr in v) for k, v in test_ood_dict.items()
            }
        splits = {
            "regime": "condition_held_out",
            "train": [int(i) for i in tr],
            "val": [int(i) for i in va],
            "test_indist": [int(i) for i in te],
            "test_ood": ood_index_lists,
        }
    else:
        ids = np.arange(n)
        tr, te = train_test_split(ids, test_size=0.2, random_state=seed)
        tr, va = train_test_split(tr, test_size=0.125, random_state=seed)
        if dry_run:
            tr, va, te = tr[:800], va[:200], te[:200]
        splits = {
            "regime": "random",
            "train": [int(i) for i in tr],
            "val": [int(i) for i in va],
            "test_indist": [int(i) for i in te],
            "test_ood": {},
        }

    return PreparedData(
        _pick(tr), _pick(va), _pick(te), n, data_source,
        test_ood=test_ood_dict, splits=splits,
    )


class Seq2SeqDataset(Dataset):
    """Slide an H+F window over each cell; minfeats 5-channel encoder input.

    Each sample: (enc_in (H,5), dec_stim (F,1), dec_target (F,)). The full
    window's response std is stored for stratified sampling.
    """

    def __init__(self, arrays, history_len, future_len, stride=5):
        cnr, fluence, baseline, ewma_slow, ewma_fast = arrays
        self.samples = []
        self.resp_std = []
        self.t_starts = []  # window start within its (possibly prepended) track
        total = history_len + future_len
        for i in range(len(cnr)):
            ci = cnr[i]
            fi = fluence[i]
            bi = baseline[i]
            esi = ewma_slow[i]
            efi = ewma_fast[i]
            T = len(ci)
            t = 0
            while t + total <= T:
                enc_in = np.stack(
                    [
                        ci[t : t + history_len],
                        fi[t : t + history_len],
                        bi[t : t + history_len],
                        esi[t : t + history_len],
                        efi[t : t + history_len],
                    ],
                    axis=-1,
                ).astype(np.float32)
                full = ci[t : t + total]
                # absolute-output variant: target is the future CNR itself, not
                # the per-step delta. dec_target[i] = CNR at step H+i.
                dec_target = full[history_len : history_len + future_len]
                dec_stim = fi[t + history_len : t + total][:, None].astype(np.float32)
                self.samples.append((enc_in, dec_stim, dec_target.astype(np.float32)))
                self.resp_std.append(float(np.std(full)))
                self.t_starts.append(t)
                t += stride
        self.resp_std = np.asarray(self.resp_std, dtype=np.float32)
        self.t_starts = np.asarray(self.t_starts, dtype=int)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc_in, dec_stim, dec_target = self.samples[idx]
        return (
            torch.tensor(enc_in),
            torch.tensor(dec_stim),
            torch.tensor(dec_target),
        )


def stratified_sampler(
    dataset: Seq2SeqDataset,
    n_strata: int,
    stratum_weights: list[float] | None = None,
) -> WeightedRandomSampler:
    """Sample across ``n_strata`` response-magnitude bins.

    Default (``stratum_weights=None``): inverse-count per stratum → uniform mass
    per bin. With ``stratum_weights`` (length ``n_strata``, summing to 1) each
    bin's per-sample weight is ``w_b / count_b`` so the *expected fraction* of
    samples drawn from bin b equals ``w_b`` — use e.g. ``[0.1, 0.2, 0.3, 0.4]``
    to over-sample high-response (Q4) cells.
    """
    rs = dataset.resp_std
    if len(rs) == 0 or n_strata <= 1:
        return WeightedRandomSampler(torch.ones(len(rs)), len(rs), replacement=True)
    edges = np.quantile(rs, np.linspace(0.0, 1.0, n_strata + 1))
    bins = np.clip(np.digitize(rs, edges[1:-1]), 0, n_strata - 1)
    counts = np.bincount(bins, minlength=n_strata).astype(np.float64)
    counts[counts == 0] = 1.0
    if stratum_weights is None:
        per_bin = 1.0 / counts
    else:
        per_bin = np.asarray(stratum_weights, dtype=np.float64) / counts
    weights = per_bin[bins]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(rs), replacement=True
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def _tf_schedule_linear(tcfg: TrainingConfig, total_epochs: int):
    start, end = tcfg.tf_ratio_start, tcfg.tf_ratio_end
    frac, hold = tcfg.tf_anneal_frac, tcfg.tf_hold_frac

    def schedule(epoch):
        hold_epochs = int(total_epochs * hold)
        anneal_epochs = max(int(total_epochs * frac) - 1, 1)
        if epoch < hold_epochs:
            return start
        p = min((epoch - hold_epochs) / anneal_epochs, 1.0)
        return start + (end - start) * p

    return schedule


class Seq2ScalarSeq(nn.Module):
    """Encoder-decoder LSTM with a configurable distributional head.

    ``forward(encoder_input, future_stim, targets=None, tf_ratio=0.0)`` returns
    ``(pi, mu, sigma)``, each ``(B, F, K)``. K=3 for the MDN head, K=1 for the
    Gaussian head.
    """

    Config = ModelConfig
    TrainingConfigCls = TrainingConfig

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = kwargs
        if isinstance(cfg, dict):
            cfg = ModelConfig.model_validate(cfg)
        self.cfg = cfg
        self.n_gaussians = cfg.n_gaussians

        self.encoder = nn.LSTM(
            cfg.encoder_dim, cfg.hidden_dim, cfg.num_layers,
            batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        _init_forget_bias(self.encoder)
        if cfg.decoder_type == "encdec":
            self.decoder = nn.LSTM(
                cfg.encoder_dim, cfg.hidden_dim, cfg.num_layers,
                batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            _init_forget_bias(self.decoder)
        else:
            self.decoder = None  # continuous: reuse the encoder LSTM

        layers = [
            nn.Linear(cfg.hidden_dim + cfg.stim_dim, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(cfg.n_mlp_layers - 1):
            layers += [
                nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
        self.trunk = nn.Sequential(*layers)

        if cfg.head_type == "gaussian":
            self.head = GaussianHead(cfg.mlp_hidden)
        else:
            self.head = MDNHead(cfg.mlp_hidden, cfg.n_gaussians)

        self.stim_init_proj = (
            StimInitProj(cfg.future_len, cfg.stim_dim, cfg.num_layers, cfg.hidden_dim)
            if cfg.stim_init else None
        )
        self.film_layer = (
            FiLMLayer(cfg.stim_dim, cfg.hidden_dim) if cfg.film != "none" else None
        )
        self.sigma_step_bias_param = (
            nn.Parameter(torch.zeros(cfg.future_len)) if cfg.sigma_step_bias else None
        )

    # -- helpers ------------------------------------------------------------

    def _zero_state(self, batch, device):
        shape = (self.cfg.num_layers, batch, self.cfg.hidden_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def point_pred(self, pi, mu):
        return (pi * mu).sum(dim=-1)

    def pred_std(self, pi, mu, sigma):
        mean = (pi * mu).sum(dim=-1, keepdim=True)
        var = (pi * (sigma ** 2 + (mu - mean) ** 2)).sum(dim=-1)
        return torch.sqrt(var.clamp(min=1e-12))

    def loss(self, preds, target):
        pi, mu, sigma = preds
        return nll(pi, mu, sigma, target)

    # -- forward ------------------------------------------------------------

    def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
        cfg = self.cfg
        b, _, _ = encoder_input.shape
        n_f = future_stim.shape[1]
        device = encoder_input.device

        h0, c0 = self._zero_state(b, device)
        if self.stim_init_proj is not None:
            dh0, dc0 = self.stim_init_proj(future_stim)
            h0, c0 = h0 + dh0, c0 + dc0

        _, (h_n, c_n) = self.encoder(encoder_input, (h0, c0))
        dec_lstm = self.encoder if cfg.decoder_type == "continuous" else self.decoder
        dh, dc = h_n, c_n

        baseline = encoder_input[:, -1, 2:3]
        dec_input = encoder_input[:, -1:, :]  # frame at time H-1
        pis, mus, sigmas = [], [], []
        for i in range(n_f):
            out, (dh, dc) = dec_lstm(dec_input, (dh, dc))
            h_step = out[:, -1, :]
            stim_i = future_stim[:, i, :]

            if self.film_layer is not None:
                gamma, beta = self.film_layer(stim_i)
                if cfg.film == "output":
                    h_step = gamma * h_step + beta
                else:  # hidden
                    dh = gamma.unsqueeze(0) * dh + beta.unsqueeze(0)
                    h_step = dh[-1]

            sigma_bias = (
                self.sigma_step_bias_param[i]
                if self.sigma_step_bias_param is not None else 0.0
            )
            feats = self.trunk(torch.cat([h_step, stim_i], dim=-1))
            pi, mu, sigma = self.head(feats, sigma_bias=sigma_bias)
            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

            if i < n_f - 1:
                last_frame = dec_input[:, -1, :]
                # absolute-output variant: the prediction IS the next CNR; no
                # delta accumulation. Teacher forcing feeds the absolute target.
                use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                if use_teacher:
                    next_cnr = targets[:, i : i + 1]
                else:
                    next_cnr = (pi * mu).sum(dim=-1, keepdim=True)
                a_s, a_f = cfg.ewma_slow_alpha, cfg.ewma_fast_alpha
                next_es = a_s * next_cnr + (1.0 - a_s) * last_frame[:, 3:4]
                next_ef = a_f * next_cnr + (1.0 - a_f) * last_frame[:, 4:5]
                dec_input = torch.cat(
                    [next_cnr, stim_i, baseline, next_es, next_ef], dim=-1
                ).unsqueeze(1)

        return (
            torch.stack(pis, dim=1),
            torch.stack(mus, dim=1),
            torch.stack(sigmas, dim=1),
        )

    # -- training -----------------------------------------------------------

    @staticmethod
    def fit(dataset, ctx):
        """Self-contained trainer. ``dataset`` is ``{"train": arrays, "val": arrays}``
        where ``arrays`` is the 5-tuple from :func:`prepare_data`.
        """
        mcfg = ctx.model_config
        tcfg = ctx.training_config

        random.seed(tcfg.seed)
        np.random.seed(tcfg.seed)
        torch.manual_seed(tcfg.seed)

        train_ds = Seq2SeqDataset(
            dataset["train"], mcfg.history_len, mcfg.future_len, stride=tcfg.train_stride
        )
        val_ds = Seq2SeqDataset(
            dataset["val"], mcfg.history_len, mcfg.future_len, stride=tcfg.train_stride
        )
        if tcfg.use_stratified_sampler:
            _sw = tcfg.quartile_weights if tcfg.sampler_type == "quartile_weighted" else None
            train_loader = DataLoader(
                train_ds, batch_size=tcfg.batch_size,
                sampler=stratified_sampler(train_ds, tcfg.n_strata, stratum_weights=_sw),
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False)

        model = Seq2ScalarSeq(mcfg).to(ctx.device)
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=tcfg.epochs, eta_min=1e-5
        )
        tf_fn = _tf_schedule_linear(tcfg, tcfg.epochs)

        hist = {"train_loss": [], "val_loss": [], "tf_ratio": []}
        ckpt_fd, ckpt = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd)
        best, wait = float("inf"), 0

        for ep in range(tcfg.epochs):
            tf_r = tf_fn(ep)
            t = _run_epoch(model, train_loader, ctx.device, opt, tcfg.grad_clip, tf_r, True)
            v = _run_epoch(model, val_loader, ctx.device, opt, tcfg.grad_clip, 0.0, False)
            hist["train_loss"].append(t)
            hist["val_loss"].append(v)
            hist["tf_ratio"].append(tf_r)
            sched.step()

            if v < best:
                best, wait = v, 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= tcfg.patience:
                    print(f"Early stopping at epoch {ep}")
                    break
            if ep % ctx.print_every == 0:
                print(f"Epoch {ep:3d} | tf={tf_r:.2f} T:{t:.5f} V:{v:.5f}")
            if ctx.progress_cb is not None:
                ctx.progress_cb(ep, tcfg.epochs, {"train": t, "val": v, "tf": tf_r})
            if ctx.tracker is not None:
                _cur = {k: w.clone() for k, w in model.state_dict().items()}
                model.load_state_dict(torch.load(ckpt, weights_only=True))
                ctx.tracker.checkpoint(model, training_results={"history": hist})
                model.load_state_dict(_cur)

        model.load_state_dict(torch.load(ckpt, weights_only=True))
        os.remove(ckpt)
        return model, hist


def _run_epoch(model, loader, device, optimizer, grad_clip, tf_ratio, is_train):
    model.train() if is_train else model.eval()
    losses = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for enc_in, dec_stim, dec_target in loader:
            enc_in = enc_in.to(device)
            dec_stim = dec_stim.to(device)
            dec_target = dec_target.to(device)
            targets = dec_target if is_train else None
            preds = model(enc_in, dec_stim, targets=targets, tf_ratio=tf_ratio)
            loss = model.loss(preds, dec_target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))
