"""Shared data pipeline for Seq2Scalar training (single-seed + ensemble).

Keeps Seq2SeqDataset and the train/val/test split in one place so every
ensemble member — and the aggregator — sees the exact same windows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from experiments.seq2seq_data import AVAILABLE_DATASETS, Seq2SeqDataset, load

__all__ = ["Seq2SeqDataset", "build_loaders"]


def build_loaders(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Build train/val/test DataLoaders deterministically from config.

    Same split (random_state=42) regardless of model seed, so every ensemble
    member trains and is evaluated on the same windows.

    Returns a dict with:
        train_loader, val_loader, test_loader  — DataLoader
        test_conditions                         — np.ndarray aligned with test_loader order
        n_traj, traj_len                        — for logging
    """
    H = config["history_len"]
    F_ = config["future_len"]
    total_window = H + F_
    data_source = config["data_source"]
    assert data_source in AVAILABLE_DATASETS, f"Unknown source {data_source!r}."

    if data_source in ("real", "real_uncertain"):
        cnr_all, stim_all, conditions_all = load(
            data_source, window_size=total_window, stride=max(1, total_window // 4),
        )
    else:
        cnr_all, stim_all, conditions_all = load(data_source)

    n_traj = len(cnr_all)
    traj_len = cnr_all.shape[1]

    traj_ids = np.arange(n_traj)
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    stride = 15 if data_source not in ("real", "real_uncertain") else max(1, total_window // 4)
    train_ds = Seq2SeqDataset(cnr_all[tr_ids], stim_all[tr_ids], conditions_all[tr_ids], H, F_, stride=stride)
    val_ds = Seq2SeqDataset(cnr_all[va_ids], stim_all[va_ids], conditions_all[va_ids], H, F_, stride=stride)
    test_ds_full = Seq2SeqDataset(cnr_all[te_ids], stim_all[te_ids], conditions_all[te_ids], H, F_, stride=stride)
    test_conditions = np.array(test_ds_full.sample_conditions)

    if dry_run:
        n_dry = 2000
        train_ds = Subset(train_ds, range(min(n_dry, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_dry, len(val_ds)) // 4))
        test_ds = Subset(test_ds_full, range(min(n_dry, len(test_ds_full))))
        test_conditions = test_conditions[: len(test_ds)]
    else:
        test_ds = test_ds_full

    BS = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    return dict(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_conditions=test_conditions,
        n_traj=n_traj,
        traj_len=traj_len,
    )
