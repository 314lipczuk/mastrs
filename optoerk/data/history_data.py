"""Per-cell full-trajectory loader for the full-history (long-gap) model.

Minimal, non-hand-engineered features only: the model input per frame is
``[cnr, fluence, fov_density, n_cells_200px]`` (cnr is both target and an input
channel). No EWMA / recency / baseline minfeats — Step 2 of the memory ladder
showed those are unused. See ``memory_characterization_plan.md``.

Returns per-cell object arrays (variable T preserved):
    cnr        : (n_cells,)  each (T,) float32  — baseline-normalized CNR
    feats      : (n_cells,)  each (K, T) float32 — rows = HISTORY_FEATURES
    conditions : (n_cells,)  str — stim_condition (protocol) per cell
    meta       : pd.DataFrame — one row per cell (uid, condition, fov, T)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Feature channels supplied alongside cnr (which is the value/target column).
# Full model input = [cnr] + HISTORY_FEATURES.
HISTORY_FEATURES = ["u_t", "fov_density", "n_cells_200px"]


def load_history_tracks(
    path: str = "dataset.parquet",
    *,
    radius: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load real microscopy data as per-cell tracks with minimal raw features.

    ``dataset.parquet`` is already preprocessed (has ``cnr_median_norm``,
    ``u_t``, ``uid`` …), so we skip ``load_and_clean`` — exactly like
    ``seq2seq_data.load_real_plus_bo_tracks``. Calling ``load_and_clean`` here
    would re-apply a global tracking-length threshold that drops every shorter
    protocol (e.g. the 90-frame bo_osc runs). Crowding features are computed on
    all detected rows, then carried through ``make_tracks``.
    """
    from optoerk.data.preprocessing import add_crowding_features, make_tracks

    df = pd.read_parquet(path)
    df = add_crowding_features(df, radius=radius)
    cnr, feats, meta = make_tracks(
        df, value_col="cnr_median_norm", stim_cols=HISTORY_FEATURES
    )
    conditions = meta["stim_condition"].to_numpy()
    return cnr, feats, conditions, meta
