"""Seq2Scalar LSTM encoder + MLP head model.

Extracted from experiments/ensemble_seq2scal.py so the class is importable at
module level — required for ExperimentBundle.reconstruct_model() to resolve
``experiments.seq2scal_model.Seq2Scalar``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _init_forget_bias(lstm: nn.LSTM) -> None:
    for name, param in lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(1.0)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        _init_forget_bias(self.lstm)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class Seq2Scalar(nn.Module):
    """LSTM encoder + MLP head, autoregressive with teacher forcing."""

    def __init__(
        self,
        encoder_dim,
        stim_dim,
        hidden_dim,
        num_layers,
        mlp_hidden=None,
        n_mlp_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = hidden_dim
        self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers, dropout)
        layers = [nn.Linear(hidden_dim + stim_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_mlp_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(mlp_hidden, 1)]
        self.head = nn.Sequential(*layers)

    def _predict_step(self, h_top, stim_i):
        return self.head(torch.cat([h_top, stim_i], dim=-1)).squeeze(-1)

    def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
        F = future_stim.shape[1]
        current_window = encoder_input
        predictions = []
        for i in range(F):
            h, _ = self.encoder(current_window)
            pred = self._predict_step(h[-1], future_stim[:, i, :])
            predictions.append(pred)
            if i < F - 1:
                last_abs = current_window[:, -1, 0:1]
                use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                delta = targets[:, i:i+1] if use_teacher else pred.unsqueeze(-1)
                next_cnr_abs = last_abs + delta
                next_input = torch.cat([next_cnr_abs, future_stim[:, i, :]], dim=-1).unsqueeze(1)
                current_window = torch.cat([current_window[:, 1:, :], next_input], dim=1)
        return torch.stack(predictions, dim=1)

    def loss(self, predictions, target):
        return nn.functional.mse_loss(predictions, target)
