"""PyTorch Lightning rewrite of lstm_seq2scal_mdn_minfeats.

Extracts model, dataset, and sampler components with zero behavioural
change, then wraps training/validation in a LightningModule + DataModule
so the Trainer handles device placement, checkpointing, early stopping,
gradient clipping, LR scheduling, and logging.

Usage:
    # Script mode (loads real data, trains, saves):
    uv run python experiments/lstm_seq2scal_mdn_lightning.py

    # Import components:
    from experiments.lstm_seq2scal_mdn_lightning import (
        LitSeq2ScalarMDN, Seq2ScalarDataModule, ModelConfig, TrainingConfig,
    )
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import lightning as L
import numpy as np
import pandas as _pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
#                                  HELPERS
# ---------------------------------------------------------------------------


def _init_forget_bias(lstm):
    for name, param in lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(1.0)


# ---------------------------------------------------------------------------
#                              MODEL COMPONENTS
# ---------------------------------------------------------------------------


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        _init_forget_bias(self.lstm)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class MDNHead(nn.Module):
    def __init__(self, in_feat, n_gaussians):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.pi_linear = nn.Linear(in_feat, n_gaussians)
        self.mu = nn.Linear(in_feat, n_gaussians)
        self.log_sigma = nn.Linear(in_feat, n_gaussians)

    def forward(self, x):
        pi = torch.softmax(self.pi_linear(x), dim=-1)
        mu = self.mu(x)
        sigma = torch.exp(self.log_sigma(x)).clamp(min=1e-3)
        return pi, mu, sigma


def mdn_nll(pi, mu, sigma, target):
    y = target.unsqueeze(-1)
    log_gauss = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(sigma)
        - 0.5 * ((y - mu) / sigma) ** 2
    )
    log_mix = torch.log(pi + 1e-12) + log_gauss
    return -torch.logsumexp(log_mix, dim=-1).mean()


# ---------------------------------------------------------------------------
#                                CONFIGURATION
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
    history_len: int = Field(30, ge=1)
    future_len: int = Field(5, ge=1)
    data_source: str = "synthetic"
    variant: str = "seq2scalar_mdn_ar_tf"

    @model_validator(mode="after")
    def _fill_mlp_hidden(self):
        if self.mlp_hidden is None:
            object.__setattr__(self, "mlp_hidden", self.hidden_dim)
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 400
    batch_size: int = 64
    patience: int = 100
    tf_ratio_start: float = 1.0
    tf_ratio_end: float = 0.0
    tf_anneal_frac: float = 0.3
    tf_hold_frac: float = 0.0
    grad_clip: float = 1.0
    train_stride: int = Field(5, ge=1)
    test_stride: int = Field(10, ge=1)
    use_stratified_sampler: bool = True
    n_strata: int = Field(3, ge=2)
    sampler_type: str = "balanced"
    quartile_weights: list[float] = Field(
        default_factory=lambda: [0.10, 0.20, 0.30, 0.40]
    )
    seed: int = 42

    @field_validator("quartile_weights", mode="before")
    @classmethod
    def _parse_quartile_weights(cls, v):
        if isinstance(v, str):
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
                f"sampler_type must be 'balanced' or 'quartile_weighted', "
                f"got {self.sampler_type!r}"
            )
        if self.sampler_type == "quartile_weighted":
            w = self.quartile_weights
            if len(w) != 4:
                raise ValueError(f"quartile_weights must have 4 entries, got {len(w)}")
            if any(x <= 0 for x in w):
                raise ValueError(f"quartile_weights must all be > 0, got {w}")
            if abs(sum(w) - 1.0) > 1e-6:
                raise ValueError(
                    f"quartile_weights must sum to 1.0, got {sum(w):.6f}"
                )
        return self


# ---------------------------------------------------------------------------
#          DATASET
# ---------------------------------------------------------------------------


class Seq2SeqDataset(Dataset):
    """Encoder sees CNR + per-step fluence + per-cell baseline.
    Decoder sees fluence only.

    Inputs are per-cell object arrays:
        cnr       : (n_cells,) of (T_i,) CNR trajectories
        fluence   : (n_cells,) of (T_i,) fluence_mJ_cm2 trajectories
        baseline  : (n_cells,) of float — per-cell median_cnr_0_9
    """

    def __init__(self, cnr, fluence, baseline, history_len, future_len, stride=5):
        self.samples = []
        total = history_len + future_len
        for i in range(len(cnr)):
            cnr_i = np.asarray(cnr[i], dtype=np.float32)
            flu_i = np.asarray(fluence[i], dtype=np.float32)
            base_i = float(baseline[i])
            T = len(cnr_i)
            t = 0
            while t + total <= T:
                enc_cnr = cnr_i[t : t + history_len]
                enc_flu = flu_i[t : t + history_len]
                enc_base = np.full(history_len, base_i, dtype=np.float32)
                dec_flu = flu_i[t + history_len : t + total]
                full_window = cnr_i[t : t + total]
                dec_target = np.diff(full_window)[
                    history_len - 1 : history_len - 1 + future_len
                ]
                enc_in = np.stack(
                    [enc_cnr, enc_flu, enc_base], axis=-1
                )  # (history_len, 3)
                dec_stim = dec_flu[:, np.newaxis]  # (future_len, 1)
                self.samples.append((enc_in, dec_stim, dec_target))
                t += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc_in, dec_stim, dec_target = self.samples[idx]
        return (
            torch.tensor(enc_in, dtype=torch.float32),
            torch.tensor(dec_stim, dtype=torch.float32),
            torch.tensor(dec_target, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
#          SAMPLERS
# ---------------------------------------------------------------------------


def compute_response_scores(dataset):
    """Per-window dynamism score: std(Δfuture) + |mean(Δfuture)|."""
    scores = np.empty(len(dataset), dtype=np.float32)
    for i, (_, _, dec_target) in enumerate(dataset.samples):
        d = np.asarray(dec_target, dtype=np.float32)
        scores[i] = float(d.std() + abs(d.mean()))
    return scores


def stratify_by_quantile(scores, n_strata):
    edges = np.quantile(scores, np.linspace(0.0, 1.0, n_strata + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(scores, edges[1:-1])
    return [np.where(bin_idx == k)[0] for k in range(n_strata)]


class StratifiedSampler(torch.utils.data.Sampler):
    def __init__(self, stratum_indices, batch_size, generator=None):
        self.stratum_indices = [np.asarray(s, dtype=np.int64) for s in stratum_indices]
        self.n_strata = len(self.stratum_indices)
        if batch_size % self.n_strata != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"n_strata ({self.n_strata})"
            )
        self.per_stratum = batch_size // self.n_strata
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        rng = np.random.default_rng()
        shuffled = [rng.permutation(s) for s in self.stratum_indices]
        n_batches = len(self)
        for b in range(n_batches):
            batch = []
            start = b * self.per_stratum
            end = start + self.per_stratum
            for s in shuffled:
                batch.extend(s[start:end].tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        min_size = min(len(s) for s in self.stratum_indices)
        return min_size // self.per_stratum


def compute_window_resp_stds(dataset):
    out = np.empty(len(dataset), dtype=np.float32)
    for i, (enc_in, _, dec_target) in enumerate(dataset.samples):
        hist_cnr = enc_in[:, 0]
        last = float(hist_cnr[-1])
        future = last + np.cumsum(np.asarray(dec_target, dtype=np.float32))
        full = np.concatenate([hist_cnr, future])
        out[i] = float(full.std())
    return out


class QuartileWeightedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        window_resp_stds,
        batch_size,
        weights=(0.10, 0.20, 0.30, 0.40),
        batches_per_epoch=None,
        generator=None,
    ):
        window_resp_stds = np.asarray(window_resp_stds, dtype=np.float64)
        if window_resp_stds.ndim != 1:
            raise ValueError(
                f"window_resp_stds must be 1-D, got shape {window_resp_stds.shape}"
            )
        if len(weights) != 4:
            raise ValueError(f"weights must have 4 entries (Q1..Q4), got {len(weights)}")
        if any(w <= 0 for w in weights):
            raise ValueError(f"all weights must be > 0, got {weights}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(weights):.6f}")

        edges = np.quantile(window_resp_stds, [0.25, 0.5, 0.75])
        bin_idx = np.digitize(window_resp_stds, edges)
        bin_indices = [np.where(bin_idx == k)[0].astype(np.int64) for k in range(4)]
        empty = [k for k, idx in enumerate(bin_indices) if len(idx) == 0]
        if empty:
            raise ValueError(
                f"empty bins {empty} — too few windows or degenerate "
                "resp_std distribution"
            )

        if batches_per_epoch is None:
            batches_per_epoch = max(1, len(window_resp_stds) // batch_size)

        total = batches_per_epoch * batch_size
        counts = [int(round(total * w)) for w in weights]
        counts[-1] = total - sum(counts[:-1])
        if counts[-1] < 0:
            raise ValueError(f"final-bin correction produced negative count: {counts}")

        self.window_resp_stds = window_resp_stds
        self.weights = tuple(float(w) for w in weights)
        self.edges = edges
        self.bin_indices = bin_indices
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.samples_per_bin = counts
        self.generator = generator

    def __iter__(self):
        rng = np.random.default_rng()
        drawn = []
        for idx, n in zip(self.bin_indices, self.samples_per_bin):
            if n == 0:
                continue
            drawn.append(rng.choice(idx, size=n, replace=True))
        all_idx = np.concatenate(drawn)
        rng.shuffle(all_idx)
        for b in range(self.batches_per_epoch):
            yield all_idx[
                b * self.batch_size : (b + 1) * self.batch_size
            ].tolist()

    def __len__(self):
        return self.batches_per_epoch


# ---------------------------------------------------------------------------
#     TEACHER FORCING SCHEDULE
# ---------------------------------------------------------------------------


def _tf_schedule_linear(tcfg, total_epochs):
    start, end = tcfg.tf_ratio_start, tcfg.tf_ratio_end
    frac, hold = tcfg.tf_anneal_frac, tcfg.tf_hold_frac

    def schedule(epoch):
        hold_epochs = int(total_epochs * hold)
        anneal_epochs = max(int(total_epochs * frac) - 1, 1)
        if epoch < hold_epochs:
            p = 0.0
        else:
            p = min((epoch - hold_epochs) / anneal_epochs, 1.0)
        return start + (end - start) * p

    return schedule


# ---------------------------------------------------------------------------
#      RAW NN MODULE
# (forward pass identical to original Seq2ScalarMDN)
# ---------------------------------------------------------------------------


class Seq2ScalarMDN(nn.Module):
    """Autoregressive sliding-window encoder + MLP trunk + MDN head.

    Kept as a standalone nn.Module so the forward pass is identical
    to the original. LitSeq2ScalarMDN wraps this for Lightning.
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
        self.encoder = LSTMEncoder(
            cfg.encoder_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout
        )
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
        self.head = MDNHead(cfg.mlp_hidden, cfg.n_gaussians)

    def _step(self, h_top, stim_i):
        feats = self.trunk(torch.cat([h_top, stim_i], dim=-1))
        return self.head(feats)

    def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
        F = future_stim.shape[1]
        current_window = encoder_input
        pis, mus, sigmas = [], [], []
        for i in range(F):
            h, _ = self.encoder(current_window)
            pi, mu, sigma = self._step(h[-1], future_stim[:, i, :])
            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)
            if i < F - 1:
                last_abs = current_window[:, -1, 0:1]
                last_baseline = current_window[:, -1, 2:3]
                use_teacher = (
                    targets is not None and torch.rand(1).item() < tf_ratio
                )
                if use_teacher:
                    delta = targets[:, i : i + 1]
                else:
                    delta = (pi * mu).sum(dim=-1, keepdim=True)
                next_cnr_abs = last_abs + delta
                next_input = torch.cat(
                    [next_cnr_abs, future_stim[:, i, :], last_baseline], dim=-1
                ).unsqueeze(1)
                current_window = torch.cat(
                    [current_window[:, 1:, :], next_input], dim=1
                )
        return (
            torch.stack(pis, dim=1),
            torch.stack(mus, dim=1),
            torch.stack(sigmas, dim=1),
        )

    def point_pred(self, pi, mu):
        return (pi * mu).sum(dim=-1)

    def pred_std(self, pi, mu, sigma):
        mean = (pi * mu).sum(dim=-1, keepdim=True)
        var = (pi * (sigma**2 + (mu - mean) ** 2)).sum(dim=-1)
        return torch.sqrt(var.clamp(min=1e-12))


# ---------------------------------------------------------------------------
#                    LIGHTNING MODULE
# ---------------------------------------------------------------------------


class LitSeq2ScalarMDN(L.LightningModule):
    """LightningModule wrapping Seq2ScalarMDN with training/validation
    steps, teacher forcing schedule, gradient clipping, and optimiser
    + LR scheduler."""

    def __init__(self, model_cfg: ModelConfig, training_cfg: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(
            {
                "model_cfg": model_cfg.model_dump(),
                "training_cfg": training_cfg.model_dump(),
            }
        )
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.model = Seq2ScalarMDN(model_cfg)
        self.tf_fn = _tf_schedule_linear(training_cfg, training_cfg.epochs)
        self._current_tf_ratio = training_cfg.tf_ratio_start
        self._val_losses: list[float] = []
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
        return self.model(encoder_input, future_stim, targets, tf_ratio)

    def training_step(self, batch, batch_idx):
        enc_in, dec_stim, dec_target = batch
        preds = self(enc_in, dec_stim, targets=dec_target, tf_ratio=self._current_tf_ratio)
        loss = mdn_nll(*preds, dec_target)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        enc_in, dec_stim, dec_target = batch
        val_preds = self(enc_in, dec_stim, targets=None, tf_ratio=0.0)
        loss = mdn_nll(*val_preds, dec_target)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Track best val loss for the custom early-stopping logic
        (only count patience when tf_ratio < 0.5, mirroring the original)."""
        current_epoch = self.current_epoch
        if self.tf_fn(current_epoch) < 0.5:
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                v = float(val_loss)
                if v < self._best_val_loss:
                    self._best_val_loss = v
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                if self._patience_counter >= self.training_cfg.patience:
                    if self.trainer.should_stop:
                        return
                    import lightning.fabric.utilities.rank_zero as _r0
                    _r0.rank_zero_info(
                        f"Early stopping triggered at epoch {current_epoch} "
                        f"(val_loss={v:.5f}, best={self._best_val_loss:.5f})"
                    )
                    self.trainer.should_stop = True

    def on_train_epoch_start(self):
        self._current_tf_ratio = self.tf_fn(self.current_epoch)
        self.log("tf_ratio", self._current_tf_ratio, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_cfg.lr,
            weight_decay=self.training_cfg.weight_decay,
        )
        sched = ReduceLROnPlateau(opt, patience=10, factor=0.5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
#      DATA MODULE
# ---------------------------------------------------------------------------


class Seq2ScalarDataModule(L.LightningDataModule):
    """Handles dataset construction and optional stratified sampling."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        dataset_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        dry_run: bool = False,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.dataset_dict = dataset_dict
        self.dry_run = dry_run
        self._train_ds: Seq2SeqDataset | None = None
        self._val_ds: Seq2SeqDataset | None = None

    def setup(self, stage: str | None = None):
        cnr_tr, flu_tr, base_tr = self.dataset_dict["train"]
        cnr_va, flu_va, base_va = self.dataset_dict["val"]
        self._train_ds = Seq2SeqDataset(
            cnr_tr, flu_tr, base_tr,
            self.model_cfg.history_len, self.model_cfg.future_len,
            stride=self.training_cfg.train_stride,
        )
        self._val_ds = Seq2SeqDataset(
            cnr_va, flu_va, base_va,
            self.model_cfg.history_len, self.model_cfg.future_len,
            stride=self.training_cfg.train_stride,
        )

    def train_dataloader(self):
        _sampler_type = getattr(self.training_cfg, "sampler_type", "balanced")
        if self.training_cfg.use_stratified_sampler and _sampler_type == "quartile_weighted":
            _resp_stds = compute_window_resp_stds(self._train_ds)
            _sampler = QuartileWeightedSampler(
                _resp_stds,
                batch_size=self.training_cfg.batch_size,
                weights=tuple(self.training_cfg.quartile_weights),
            )
            _sizes = ", ".join(str(len(s)) for s in _sampler.bin_indices)
            print(
                f"QuartileWeightedSampler: edges={_sampler.edges.round(4).tolist()} "
                f"bin_sizes=[{_sizes}] weights={list(_sampler.weights)} "
                f"samples_per_bin={_sampler.samples_per_bin} "
                f"batches/epoch={len(_sampler)}"
            )
            return DataLoader(self._train_ds, batch_sampler=_sampler)
        elif self.training_cfg.use_stratified_sampler:
            _scores = compute_response_scores(self._train_ds)
            _strata = stratify_by_quantile(_scores, self.training_cfg.n_strata)
            _sampler = StratifiedSampler(_strata, self.training_cfg.batch_size)
            _sizes = ", ".join(str(len(s)) for s in _strata)
            _q = np.quantile(_scores, np.linspace(0.0, 1.0, self.training_cfg.n_strata + 1))
            print(
                f"StratifiedSampler: n_strata={self.training_cfg.n_strata} "
                f"sizes=[{_sizes}] "
                f"per_stratum={_sampler.per_stratum} batches/epoch={len(_sampler)} "
                f"edges={_q.round(4).tolist()}"
            )
            return DataLoader(self._train_ds, batch_sampler=_sampler)
        else:
            return DataLoader(
                self._train_ds,
                batch_size=self.training_cfg.batch_size,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
        )


# ---------------------------------------------------------------------------
# LIGHTNING CALLBACK — TF LOG
# ---------------------------------------------------------------------------


class TFLogCallback(Callback):
    """Logs the teacher-forcing ratio at the start of each epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        tf_val = pl_module._current_tf_ratio
        pl_module.log("tf_ratio", tf_val, logger=True)
        if (trainer.current_epoch + 1) % 20 == 0:
            print(
                f"Epoch {trainer.current_epoch:3d} | tf={tf_val:.2f} "
                f"train_loss={trainer.callback_metrics.get('train_loss', float('nan')):.5f} "
                f"val_loss={trainer.callback_metrics.get('val_loss', float('nan')):.5f}"
            )


# ---------------------------------------------------------------------------
# DATA LOADING UTILITY
# Mirrors the marimo cell logic in lstm_seq2scal_mdn_minfeats.
# ---------------------------------------------------------------------------


def load_real_data(dry_run: bool = False):
    """Load the real dataset.parquet and return splits matching the
    original notebook cell.

    Returns
    -------
    dataset_dict : {"train": (cnr, fluence, baseline),
                    "val": (cnr, fluence, baseline)}
    te_tuple    : (cnr_te, fluence_te, baseline_te)  — hold-out test split
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from notebooks.experiment.preprocessing import make_tracks

    _df = _pd.read_parquet("dataset.parquet")
    cnr_all, _stim_all, _meta = make_tracks(
        _df,
        value_col="cnr_median_norm",
        stim_cols=["fluence_mJ_cm2"],
    )
    fluence_all = np.empty(len(_stim_all), dtype=object)
    for _i, _s in enumerate(_stim_all):
        fluence_all[_i] = _s[0]

    _baseline_lookup = _df.groupby("uid")["median_cnr_0_9"].first()
    baseline_all = np.array(
        [float(_baseline_lookup[u]) for u in _meta["uid"]], dtype=np.float32
    )

    n_traj = len(cnr_all)
    _traj_ids = np.arange(n_traj)
    _tr_ids, _te_ids = train_test_split(_traj_ids, test_size=0.2, random_state=42)
    _tr_ids, _va_ids = train_test_split(_tr_ids, test_size=0.125, random_state=42)

    if dry_run:
        _tr_ids = _tr_ids[: min(len(_tr_ids), 800)]
        _va_ids = _va_ids[: min(len(_va_ids), 200)]
        _te_ids = _te_ids[: min(len(_te_ids), 200)]

    dataset_dict = {
        "train": (cnr_all[_tr_ids], fluence_all[_tr_ids], baseline_all[_tr_ids]),
        "val": (cnr_all[_va_ids], fluence_all[_va_ids], baseline_all[_va_ids]),
    }
    te_tuple = (cnr_all[_te_ids], fluence_all[_te_ids], baseline_all[_te_ids])
    return dataset_dict, te_tuple


# ---------------------------------------------------------------------------
#     CLI ENTRY POINT
# ---------------------------------------------------------------------------


def main():
    """Script entry point: parse CLI args, build configs, train, save.

    Usage:
        uv run python experiments/lstm_seq2scal_mdn_lightning.py \\
            [--model_cfg.hidden_dim 128] [--epochs 200] [--dry_run True]
    """
    import argparse

    # Default configs
    mcfg = ModelConfig(encoder_dim=3, stim_dim=1, data_source="real")
    tcfg = TrainingConfig()

    # Minimal CLI override
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", default=False)
    args, remaining = parser.parse_known_args()

    print("=" * 60)
    print("LitSeq2ScalarMDN — PyTorch Lightning training")
    print("=" * 60)
    print(f"Model config:   {mcfg.model_dump()}")
    print(f"Training config: {tcfg.model_dump()}")
    print(f"Dry run:        {args.dry_run}")

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)

    print("\nLoading data ...")
    dataset_dict, _ = load_real_data(dry_run=args.dry_run)
    n_train = len(dataset_dict["train"][0])
    n_val = len(dataset_dict["val"][0])
    print(f"  Train: {n_train} trajectories, Val: {n_val} trajectories")

    datamodule = Seq2ScalarDataModule(mcfg, tcfg, dataset_dict, dry_run=args.dry_run)

    model = LitSeq2ScalarMDN(mcfg, tcfg)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="lit-seq2scal-mdn-{epoch:03d}-{val_loss:.5f}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=tcfg.patience,
        mode="min",
        min_delta=0.0,
        verbose=True,
    )
    tf_log_cb = TFLogCallback()

    trainer = L.Trainer(
        max_epochs=tcfg.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=tcfg.grad_clip,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_cb, early_stop_cb, tf_log_cb],
        logger=TensorBoardLogger("lightning_logs", name="lstm_seq2scal_mdn"),
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
    )

    print("\nStarting training ...")
    t0 = time.time()
    trainer.fit(model, datamodule=datamodule)
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_loss:   {checkpoint_cb.best_model_score:.5f}")

    # Save final model state for downstream use
    out_dir = Path("lightning_logs") / "lstm_seq2scal_mdn" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), out_dir / "model_state.pt")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(mcfg.model_dump(), f, indent=2)
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(tcfg.model_dump(), f, indent=2)
    print(f"Final model saved to {out_dir / 'model_state.pt'}")


if __name__ == "__main__":
    main()
