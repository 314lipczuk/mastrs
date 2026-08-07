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
# Full model input = [cnr] + features.
HISTORY_FEATURES = ["u_t", "fov_density", "n_cells_200px", "optortk_expr"]

# Derived channels: computed here rather than read from the parquet, so the
# formula lives in one place and the serving path can reproduce it by name.
U_X_EXPR = "u_t_x_expr"          # fluence * expression — the gain interaction
DERIVED = {U_X_EXPR: ("u_t", "optortk_expr")}

# Named feature sets, one per training variant. The point of the comparison is
# that optoRTK expression is physically a GAIN on the dose-response, and the
# baseline set gives the model no structural way to express that: `optortk_expr`
# enters as a constant channel repeated at every frame, and the decoder's only
# future input is fluence, so during a 30-step rollout the interaction has to be
# carried entirely in the LSTM hidden state.
#
#   base        the current channels, now fed the REAL mCitrine rank
#   interaction adds u_t * expr, which is also passed to the decoder as a second
#               future/stim channel (see FUTURE_CHANNELS) — encoder-only would be
#               invisible during the rollout, which is the regime that matters
#   area        base + nuc_area; tests whether cell size earns a channel at all
#   area_lean   the CANDIDATE: `area` minus `fov_density`. Everything the
#               2026-08-06 comparison actually justified, and nothing else.
#
# Why `area_lean` drops fov_density, and only that. Permutation importance over
# the four comparison runs, in Δ NLL when the channel is shuffled across cells:
#
#     u_t            0.92-0.97     the dose; everything else is a rounding error
#     optortk_expr   0.059         real, and now the true mCitrine measurement
#     nuc_area       0.0095        modest but the only significant NLL gain
#     n_cells_200px  0.0067        marginal — but genuinely per-cell, so kept
#     fov_density    0.0046        lowest of any channel, and it CANNOT be a
#                                  per-cell feature: it is the same number for
#                                  every cell in a frame, a frame-level covariate
#                                  wearing a per-cell costume
#
# A separate check ruled out the possibility that fov_density was merely a poor
# proxy for something real nearby: the local neighbour-CNR field adds ~0.001 R2
# to a cell's own CNR change at any horizon (peaking at H=5-15 min, the right
# timescale for neighbour-to-neighbour ERK propagation, so the effect is real but
# ~50x too small to matter). Better spatial features would refine that 0.1%; they
# would not create the missing signal, because no training experiment ever
# stimulated cells within a frame differently — neighbour DOSE has zero variance
# in all 72,441 cells. That is an experiment to run, not a channel to add.
FEATURE_SETS: dict[str, list[str]] = {
    "base": HISTORY_FEATURES,
    "interaction": [*HISTORY_FEATURES, U_X_EXPR],
    "area": [*HISTORY_FEATURES, "nuc_area"],
    "area_lean": [c for c in HISTORY_FEATURES if c != "fov_density"] + ["nuc_area"],
}

# Which channels are fed to the decoder as known-future inputs. Fluence always
# (it is commanded, so it is known); the interaction only when it exists, since
# it is a deterministic function of the commanded fluence and a static per-cell
# value and is therefore equally knowable ahead of time.
FUTURE_CHANNELS: dict[str, list[str]] = {
    "base": ["u_t"],
    "interaction": ["u_t", U_X_EXPR],
    "area": ["u_t"],
    "area_lean": ["u_t"],
}


def resolve_feature_set(name: str) -> tuple[list[str], list[str]]:
    """``(features, future_channels)`` for a named set; raises on an unknown name."""
    if name not in FEATURE_SETS:
        raise ValueError(
            f"unknown feature_set {name!r}; known: {sorted(FEATURE_SETS)}"
        )
    return list(FEATURE_SETS[name]), list(FUTURE_CHANNELS[name])

# The cnr channel can be either baseline-normalized (per-cell divide by resting
# median, the historical default) or raw absolute CNR. This is the single source
# of truth mapping the mode name -> parquet column; ``cnr_mode`` threads through
# the model config and the serving path so a raw-CNR model is never fed (or
# online-normalized) as if it were a norm-CNR model. See ``history_dataset`` and
# ``optoerk.serving``.
CNR_MODES = ("norm", "raw")
CNR_MODE_COLUMNS = {"norm": "cnr_median_norm", "raw": "cnr_median"}


def load_history_tracks(
    path: str | Path = materials_path("dataset_all_mcitrine.parquet"),
    *,
    radius: float = 200.0,
    cnr_mode: str = "norm",
    features: list[str] | None = None,
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

    features = list(features) if features is not None else list(HISTORY_FEATURES)
    df = pd.read_parquet(path)
    df = add_crowding_features(df, radius=radius)
    df = add_optortk_expression(df)
    # Derived channels are built after their inputs exist, so a feature set can
    # ask for `u_t_x_expr` without the parquet ever carrying it.
    for name in features:
        if name in DERIVED:
            a, b = DERIVED[name]
            df[name] = (df[a].astype("float32") * df[b].astype("float32"))
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"features {missing} are not columns of {path}")
    cnr, feats, meta = make_tracks(
        df, value_col=CNR_MODE_COLUMNS[cnr_mode], stim_cols=features
    )
    conditions = meta["stim_condition"].to_numpy()
    return cnr, feats, conditions, meta
