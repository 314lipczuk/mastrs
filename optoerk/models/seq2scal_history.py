"""Full-history seq2scalar model for the long-gap / two-pattern task.

Differences from ``seq2scal_models_abs``:
  * the encoder consumes the **entire variable-length past** (packed, no fixed
    history window) behind a **swappable ``Encoder`` interface** (LSTM base now;
    SSM/Transformer can drop in later);
  * minimal **raw** inputs ``[cnr, fluence, fov_density, n_cells_200px]`` — no
    EWMA/baseline minfeats (Step 2 of the memory ladder showed those unused);
  * the decoder rolls over **future fluence only** (commanded; future crowding
    is unknown live), feeding back its own predicted CNR.

Everything is in **standardized** space (inputs and target standardized with the
frozen train-population stats carried on the config); denormalize the head's
``mu``/``sigma`` with the cnr stats to recover absolute CNR. The distributional
head, FiLM and NLL are reused from ``seq2scal_models_abs``.
"""
from __future__ import annotations

import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler

from optoerk.models.seq2scal_abs import (
    FiLMLayer,
    GaussianHead,
    MDNHead,
    _init_forget_bias,
    _tf_schedule_linear,
    nll,
)

# Channel index of cnr within the input vector (CHANNELS = [cnr, u_t, ...]).
CNR_CHANNEL = 0


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class HistoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_dim: int = Field(5, ge=1)          # [cnr, u_t, fov_density, n_cells_200px, optortk_expr]
    stim_dim: int = Field(1, ge=1)           # decoder future input = fluence
    hidden_dim: int = Field(64, ge=1)
    num_layers: int = Field(2, ge=1)
    n_gaussians: int = Field(3, ge=1)
    n_mlp_layers: int = Field(3, ge=1)
    mlp_hidden: int | None = None
    dropout: float = Field(0.1, ge=0.0, le=0.9)
    future_len: int = Field(10, ge=1)
    encoder_type: str = "lstm"               # swappable: "lstm" (SSM/transformer later)
    head_type: str = "mdn"                   # "mdn" | "gaussian"
    film: str = "none"                       # "none" | "output" | "hidden"
    sigma_step_bias: bool = False            # learnable per-forecast-step log-sigma bias
    data_source: str = "real_plus_bo"
    variant: str = "seq2scalar_history"
    # frozen normalization (channel order matches history_dataset.CHANNELS),
    # carried so the model is self-contained for eval / deployment.
    norm_channels: list[str] = Field(default_factory=list)
    norm_mean: list[float] = Field(default_factory=list)
    norm_std: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self):
        if self.mlp_hidden is None:
            object.__setattr__(self, "mlp_hidden", self.hidden_dim)
        if self.head_type not in ("mdn", "gaussian"):
            raise ValueError(f"head_type must be mdn|gaussian, got {self.head_type!r}")
        if self.film not in ("none", "output", "hidden"):
            raise ValueError(f"film must be none|output|hidden, got {self.film!r}")
        if self.encoder_type not in ("lstm",):
            raise ValueError(f"encoder_type must be lstm (for now), got {self.encoder_type!r}")
        if self.head_type == "gaussian":
            object.__setattr__(self, "n_gaussians", 1)
        return self


class HistoryTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 256
    patience: int = 40
    grad_clip: float = 1.0
    tf_ratio_start: float = 1.0
    tf_ratio_end: float = 0.0
    tf_anneal_frac: float = 0.3
    tf_hold_frac: float = 0.0
    # self-concat augmentation (train split only)
    p_concat: float = 0.5
    break_min: int = 0
    break_max: int = 60
    # multi-length training: sample horizon F ~ U[future_len_min, model.future_len]
    # per batch. None → fixed horizon = model.future_len.
    future_len_min: int | None = None
    # response-magnitude stratified sampler over cells
    use_stratified_sampler: bool = True
    n_strata: int = Field(3, ge=1)
    num_workers: int = 0
    seed: int = 42


# ---------------------------------------------------------------------------
# swappable encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Interface: (ctx (B,L,C), lengths (B,)) -> (h_n, c_n) each (num_layers,B,H).

    Encodes the full variable-length past into a state that initializes the
    decoder. Subclasses swap the sequence backbone; the (h_n, c_n) contract
    keeps the decoder unchanged.
    """

    def forward(self, ctx: torch.Tensor, lengths: torch.Tensor):  # pragma: no cover
        raise NotImplementedError


class LSTMEncoder(Encoder):
    def __init__(self, cfg: HistoryConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            cfg.input_dim, cfg.hidden_dim, cfg.num_layers,
            batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        _init_forget_bias(self.lstm)

    def forward(self, ctx, lengths):
        packed = pack_padded_sequence(
            ctx, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.lstm(packed)
        return h_n, c_n


def build_encoder(cfg: HistoryConfig) -> Encoder:
    if cfg.encoder_type == "lstm":
        return LSTMEncoder(cfg)
    raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class Seq2ScalarHistory(nn.Module):
    """Full-history encoder + future-fluence decoder, absolute-CNR MDN head.

    ``forward(ctx, lengths, future_flu, targets=None, tf_ratio=0.0)`` returns
    ``(pi, mu, sigma)`` each ``(B, F, K)`` in standardized cnr units.
    """

    Config = HistoryConfig

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = kwargs
        if isinstance(cfg, dict):
            cfg = HistoryConfig.model_validate(cfg)
        self.cfg = cfg
        self.n_gaussians = cfg.n_gaussians

        self.encoder = build_encoder(cfg)
        # decoder input per future step = [cnr_feedback, fluence]
        self.decoder = nn.LSTM(
            cfg.stim_dim + 1, cfg.hidden_dim, cfg.num_layers,
            batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        _init_forget_bias(self.decoder)

        layers = [
            nn.Linear(cfg.hidden_dim + cfg.stim_dim, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(cfg.n_mlp_layers - 1):
            layers += [nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
        self.trunk = nn.Sequential(*layers)

        self.head = (
            GaussianHead(cfg.mlp_hidden) if cfg.head_type == "gaussian"
            else MDNHead(cfg.mlp_hidden, cfg.n_gaussians)
        )
        self.film_layer = FiLMLayer(cfg.stim_dim, cfg.hidden_dim) if cfg.film != "none" else None
        # per-step sigma bias sized to the max horizon (future_len); valid for any
        # F <= future_len used at train (multi-length) or eval.
        self.sigma_step_bias_param = (
            nn.Parameter(torch.zeros(cfg.future_len)) if cfg.sigma_step_bias else None
        )

    # -- helpers ------------------------------------------------------------

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

    def forward(self, ctx, lengths, future_flu, targets=None, tf_ratio=0.0):
        cfg = self.cfg
        B = ctx.shape[0]
        n_f = future_flu.shape[1]

        h_n, c_n = self.encoder(ctx, lengths)
        dh, dc = h_n, c_n

        # initial feedback = cnr at the last real context frame (standardized)
        last_idx = (lengths - 1).clamp(min=0)
        cnr_fb = ctx[torch.arange(B, device=ctx.device), last_idx, CNR_CHANNEL:CNR_CHANNEL + 1]

        pis, mus, sigmas = [], [], []
        for i in range(n_f):
            flu_i = future_flu[:, i, :]                       # (B, stim_dim)
            dec_in = torch.cat([cnr_fb, flu_i], dim=-1).unsqueeze(1)
            out, (dh, dc) = self.decoder(dec_in, (dh, dc))
            h_step = out[:, -1, :]

            if self.film_layer is not None:
                gamma, beta = self.film_layer(flu_i)
                if cfg.film == "output":
                    h_step = gamma * h_step + beta
                else:  # hidden
                    dh = gamma.unsqueeze(0) * dh + beta.unsqueeze(0)
                    h_step = dh[-1]

            sigma_bias = (
                self.sigma_step_bias_param[i]
                if self.sigma_step_bias_param is not None else 0.0
            )
            feats = self.trunk(torch.cat([h_step, flu_i], dim=-1))
            pi, mu, sigma = self.head(feats, sigma_bias=sigma_bias)
            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

            if i < n_f - 1:
                pred = (pi * mu).sum(dim=-1, keepdim=True)
                if targets is not None and torch.rand(1).item() < tf_ratio:
                    cnr_fb = targets[:, i : i + 1]
                else:
                    cnr_fb = pred

        return (
            torch.stack(pis, dim=1),
            torch.stack(mus, dim=1),
            torch.stack(sigmas, dim=1),
        )

    # -- training -----------------------------------------------------------

    @staticmethod
    def fit(dataset, ctx):
        """Trainer matching the scaffold contract.

        ``dataset`` = ``{"train": (cnr, feats), "val": (cnr, feats),
        "stats": NormStats}`` where cnr/feats are per-cell object arrays.
        """
        from optoerk.data.history_dataset import (
            HistoryDataset,
            collate_history,
            make_history_collate,
        )

        mcfg, tcfg = ctx.model_config, ctx.training_config
        random.seed(tcfg.seed)
        np.random.seed(tcfg.seed)
        torch.manual_seed(tcfg.seed)

        cnr_tr, feats_tr = dataset["train"]
        cnr_va, feats_va = dataset["val"]
        stats = dataset["stats"]

        train_ds = HistoryDataset(
            cnr_tr, feats_tr, np.arange(len(cnr_tr)), stats,
            F=mcfg.future_len, p_concat=tcfg.p_concat,
            break_min=tcfg.break_min, break_max=tcfg.break_max, seed=tcfg.seed,
        )
        val_ds = HistoryDataset(  # clean val: no concat augmentation
            cnr_va, feats_va, np.arange(len(cnr_va)), stats,
            F=mcfg.future_len, p_concat=0.0, seed=tcfg.seed + 1,
        )

        def _winit(wid):
            info = torch.utils.data.get_worker_info()
            info.dataset.reseed(tcfg.seed + 1000 * (wid + 1))

        # multi-length: sample horizon per batch in [f_min, future_len]; val is
        # fixed at the full horizon for a stable validation signal.
        f_max = mcfg.future_len
        f_min = tcfg.future_len_min if tcfg.future_len_min is not None else f_max
        train_collate = make_history_collate(f_min, f_max, seed=tcfg.seed)

        if tcfg.use_stratified_sampler:
            resp = np.array([float(np.std(np.asarray(c))) for c in cnr_tr])
            train_loader = DataLoader(
                train_ds, batch_size=tcfg.batch_size,
                sampler=_response_sampler(resp, tcfg.n_strata),
                collate_fn=train_collate, num_workers=tcfg.num_workers,
                worker_init_fn=_winit if tcfg.num_workers > 0 else None,
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=tcfg.batch_size, shuffle=True,
                collate_fn=train_collate, num_workers=tcfg.num_workers,
                worker_init_fn=_winit if tcfg.num_workers > 0 else None,
            )
        val_loader = DataLoader(
            val_ds, batch_size=tcfg.batch_size, shuffle=False,
            collate_fn=collate_history, num_workers=tcfg.num_workers,
        )

        model = Seq2ScalarHistory(mcfg).to(ctx.device)
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs, eta_min=1e-5)
        tf_fn = _tf_schedule_linear(tcfg, tcfg.epochs)

        hist = {"train_loss": [], "val_loss": [], "tf_ratio": []}
        ckpt_fd, ckpt = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd)
        best, wait = float("inf"), 0

        for ep in range(tcfg.epochs):
            tf_r = tf_fn(ep)
            t = _run_epoch_history(model, train_loader, ctx.device, opt, tcfg.grad_clip, tf_r, True)
            v = _run_epoch_history(model, val_loader, ctx.device, opt, tcfg.grad_clip, 0.0, False)
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


def _response_sampler(resp_std: np.ndarray, n_strata: int) -> WeightedRandomSampler:
    """Inverse-count weighting over response-magnitude strata (uniform mass/bin)."""
    n = len(resp_std)
    if n == 0 or n_strata <= 1:
        return WeightedRandomSampler(torch.ones(max(n, 1)), n, replacement=True)
    edges = np.quantile(resp_std, np.linspace(0.0, 1.0, n_strata + 1))
    bins = np.clip(np.digitize(resp_std, edges[1:-1]), 0, n_strata - 1)
    counts = np.bincount(bins, minlength=n_strata).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts)[bins]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), n, replacement=True)


def _run_epoch_history(model, loader, device, optimizer, grad_clip, tf_ratio, is_train):
    model.train() if is_train else model.eval()
    losses = []
    cm = torch.enable_grad() if is_train else torch.no_grad()
    with cm:
        for b in loader:
            ctx_t = b["ctx"].to(device)
            lengths = b["lengths"]  # stays on CPU for pack_padded
            fut = b["fut_flu"].to(device)
            tgt = b["tgt"].to(device)
            preds = model(ctx_t, lengths, fut, targets=(tgt if is_train else None), tf_ratio=tf_ratio)
            loss = model.loss(preds, tgt)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))
