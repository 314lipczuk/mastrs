"""
Shared data loading for synthetic EGFR state-space experiments.

Provides deterministic train/val/test splits at the trajectory level.
Both training scripts and evaluation/viewing code use this module
to ensure consistent data access.

Usage:
    from data import load_synthetic_states
    dataset = load_synthetic_states()
    dataset.test.states   # (N_test_timepoints, 5)
    dataset.test.traj_lengths  # (n_test_trajectories,)
    dataset.state_names   # ["RAS", "RAF", "MEK", "NFB", "ERK"]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


STATE_COLS = ["RAS_s", "RAF_s", "MEK_s", "NFB_s", "ERK_s"]
STATE_NAMES = [c.replace("_s", "") for c in STATE_COLS]

DEFAULT_PARQUET = "synthetic_EGFR_data.parquet"


@dataclass
class StateSplit:
    """One split (train/val/test) of the state-space dataset."""
    states: np.ndarray       # (N_timepoints, n_vars) — all timepoints concatenated
    light: np.ndarray        # (N_timepoints,) — light stimulus per timepoint
    traj_lengths: np.ndarray  # (n_trajectories,) — length of each trajectory
    traj_ids: np.ndarray      # (n_trajectories,) — original row indices in the parquet


@dataclass
class SyntheticStateData:
    """Complete train/val/test split of synthetic EGFR state data."""
    train: StateSplit
    val: StateSplit
    test: StateSplit
    state_names: list[str]
    traj_len: int


def load_synthetic_states(
    parquet_path: str = DEFAULT_PARQUET,
) -> SyntheticStateData:
    """Load synthetic EGFR data and split into train/val/test.

    Splits at the trajectory level with deterministic seeds matching
    the existing training pipeline: 70% train, 10% val, 20% test.
    """
    df = pd.read_parquet(parquet_path)

    all_states = np.stack(
        [np.concatenate(df[c].values) for c in STATE_COLS], axis=1
    ).astype(np.float32)
    all_light = np.concatenate(df["light"].values).astype(np.float32)

    traj_len = len(df[STATE_COLS[0]].iloc[0])

    traj_ids = np.arange(len(df))
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    def _make_split(ids: np.ndarray) -> StateSplit:
        idx = np.concatenate(
            [np.arange(i * traj_len, (i + 1) * traj_len) for i in ids]
        )
        return StateSplit(
            states=all_states[idx],
            light=all_light[idx],
            traj_lengths=np.full(len(ids), traj_len),
            traj_ids=ids,
        )

    return SyntheticStateData(
        train=_make_split(tr_ids),
        val=_make_split(va_ids),
        test=_make_split(te_ids),
        state_names=list(STATE_NAMES),
        traj_len=traj_len,
    )
