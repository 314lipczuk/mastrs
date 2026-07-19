"""Per-cell full-trajectory loader for the full-history (long-gap) model.

Minimal, non-hand-engineered features only: the model input per frame is
``[cnr, fluence, fov_density, n_cells_200px, optortk_expr]`` (cnr is both target
and an input channel). No EWMA / recency / baseline minfeats — Step 2 of the
memory ladder showed those are unused. See ``memory_characterization_plan.md``.
``optortk_expr`` is the per-cell optoRTK expression level (session-relative C0
rank; a static per-cell covariate for response gain).

Returns per-cell object arrays (variable T preserved):
    cnr        : (n_cells,)  each (T,) float32  — CNR (baseline-normalized or
                 raw absolute, per ``cnr_mode``; see ``CNR_MODE_COLUMNS``)
    feats      : (n_cells,)  each (K, T) float32 — rows = HISTORY_FEATURES
    conditions : (n_cells,)  str — stim_condition (protocol) per cell
    meta       : pd.DataFrame — one row per cell (uid, condition, fov, T)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from optoerk.core.utils import materials_path

# Feature channels supplied alongside cnr (which is the value/target column).
# Full model input = [cnr] + HISTORY_FEATURES.
HISTORY_FEATURES = ["u_t", "fov_density", "n_cells_200px", "optortk_expr"]

# The cnr channel can be either baseline-normalized (per-cell divide by resting
# median, the historical default) or raw absolute CNR. This is the single source
# of truth mapping the mode name -> parquet column; ``cnr_mode`` threads through
# the model config and the serving path so a raw-CNR model is never fed (or
# online-normalized) as if it were a norm-CNR model. See ``history_dataset`` and
# ``optoerk.serving``.
CNR_MODES = ("norm", "raw")
CNR_MODE_COLUMNS = {"norm": "cnr_median_norm", "raw": "cnr_median"}


def load_history_tracks(
    path: str | Path = materials_path("dataset_all.parquet"),
    *,
    radius: float = 200.0,
    cnr_mode: str = "norm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load real microscopy data as per-cell tracks with minimal raw features.

    ``dataset.parquet`` is already preprocessed (has ``cnr_median_norm``,
    ``u_t``, ``uid`` …), so we skip ``load_and_clean`` — exactly like
    ``seq2seq_data.load_real_plus_bo_tracks``. Calling ``load_and_clean`` here
    would re-apply a global tracking-length threshold that drops every shorter
    protocol (e.g. the 90-frame bo_osc runs). Crowding features are computed on
    all detected rows, then carried through ``make_tracks``.

    ``cnr_mode`` selects the cnr column: ``"norm"`` -> ``cnr_median_norm``
    (baseline-normalized, default) or ``"raw"`` -> ``cnr_median`` (absolute).
    """
    if cnr_mode not in CNR_MODE_COLUMNS:
        raise ValueError(f"cnr_mode must be one of {CNR_MODES}, got {cnr_mode!r}")
    from optoerk.data.preprocessing import (
        add_crowding_features,
        add_optortk_expression,
        make_tracks,
    )

    df = pd.read_parquet(path)
    df = add_crowding_features(df, radius=radius)
    df = add_optortk_expression(df)
    cnr, feats, meta = make_tracks(
        df, value_col=CNR_MODE_COLUMNS[cnr_mode], stim_cols=HISTORY_FEATURES
    )
    conditions = meta["stim_condition"].to_numpy()
    return cnr, feats, conditions, meta
