"""Turn a faro ``/predict`` payload into the model's per-frame channels.

The model input per frame is the standardized 5-vector
``[cnr, u_t, fov_density, n_cells_200px, optortk_expr]``
(``history_dataset.CHANNELS``):

  * ``cnr``            — a norm-mode model was trained on ``cnr_median_norm`` =
                        per-cell baseline-normalized *median* CNR, so the server
                        normalizes online (see :mod:`optoerk.serving.state`); a
                        raw-mode model was trained on absolute ``cnr_median`` and
                        the raw scalar is fed directly. faro sends raw ``cnr``
                        (and usually ``cnr_median``); this module only extracts
                        the raw CNR-like scalar either way.
  * ``u_t``            — commanded fluence (mJ/cm2); supplied by per-cell state
                        (the last commanded dose), not by the payload.
  * ``fov_density``    — number of cells detected in this FOV at this frame.
  * ``n_cells_200px``  — number of *other* cells within ``radius`` px.
  * ``optortk_expr``   — the cell's optoRTK expression as a session-relative
                        percentile rank. Derived from the mCitrine measurement
                        this module extracts, but the ranking needs the whole
                        session's cohort and so lives in
                        :mod:`optoerk.serving.expression`.

The two crowding channels are derived server-side from all cells' (x, y) in the
payload, replicating ``preprocessing.add_crowding_features`` exactly (it also
counts every detected cell and excludes self).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def extract_raw_cnr(cell: dict[str, Any]) -> float | None:
    """The raw (un-normalized) CNR-like scalar the model's cnr channel uses.

    Prefers ``cnr_median`` (matches training's ``cnr_median`` = median ring /
    median nuc), then median intensities if present, then plain ``cnr``.
    Returns None if nothing usable is found.
    """
    v = cell.get("cnr_median")
    if v is not None and np.isfinite(v):
        return float(v)
    ring = cell.get("median_intensity_C1_ring")
    nuc = cell.get("median_intensity_C1_nuc")
    if ring is not None and nuc not in (None, 0) and np.isfinite(ring) and np.isfinite(nuc):
        return float(ring) / float(nuc)
    v = cell.get("cnr")
    if v is not None and np.isfinite(v):
        return float(v)
    return None


#: Payload keys carrying the raw optoRTK expression measurement, in preference
#: order. faro's ``RefFE`` writes ``ref_mean_intensity``; the older Cedric-side
#: export of the same quantity is ``optocheck_mean_intensity``.
OPTORTK_KEYS = ("ref_mean_intensity", "optocheck_mean_intensity")


def extract_optortk_value(cell: dict[str, Any]) -> float | None:
    """The raw optoRTK expression measurement for this cell, pre-ranking.

    This is **mCitrine**, imaged in its own short optocheck/reference acquisition
    — a different fluorescence channel from the timelapse's ``miRFP`` (C0) and
    ``mScarlet3`` (C1, which ``cnr_median`` comes from). It is what
    ``preprocessing.add_optortk_expression`` ranks offline, so it is what serving
    must feed for train and serve to agree.

    **Returns None on most frames, and that is normal.** The optocheck runs once
    or twice per experiment, not every frame, so the value arrives only on those
    frames and the cell carries it thereafter. A caller must not treat a single
    absent frame as an error; see ``InferenceService._check_optortk_coverage`` for
    where the real contract failure is detected.

    It must never be reconstructed from the C0 channels. That surrogate — whole-
    cell miRFP — was what the pipeline used before the measurement was wired
    through, and measured against mCitrine on the same cells it reaches only
    Spearman 0.60-0.71 and misplaces 27-30% of cells across a high/low split.
    """
    for key in OPTORTK_KEYS:
        v = cell.get(key)
        if v is not None and np.isfinite(v) and float(v) > 0:
            return float(v)
    return None


#: Payload keys carrying the nuclear area, in preference order. faro's
#: ``FE_ErkKtr`` renames skimage's ``area`` to ``area_nuc``; the training bundles
#: call the same quantity ``nuc_area``. Both spellings are accepted so neither
#: side has to be renamed to make a model that uses the channel servable.
AREA_KEYS = ("area_nuc", "nuc_area", "area")


def extract_nuc_area(cell: dict[str, Any]) -> float | None:
    """The cell's nuclear area, the model's ``nuc_area`` channel.

    Named differently on the two sides of the wire — ``area_nuc`` in the payload,
    ``nuc_area`` in the model's channel list — which is exactly the sort of
    mismatch that ends with a used channel being fed its population mean and
    nothing saying so.
    """
    for key in AREA_KEYS:
        v = cell.get(key)
        if v is not None and np.isfinite(v) and float(v) > 0:
            return float(v)
    return None


def compute_crowding(
    cells: list[dict[str, Any]], radius: float = 200.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (fov_density, n_cells_200px) from the frame's (x, y) positions.

    Mirrors ``preprocessing.add_crowding_features`` for a single (fov, frame)
    group: ``fov_density`` = number of cells in the frame (same for all), and
    ``n_cells_200px`` = neighbours within ``radius`` px excluding self.
    Cells with missing coordinates get density = len(cells), neighbours = 0.
    """
    n = len(cells)
    fov_density = np.full(n, float(n), dtype=np.float32)
    n_in_radius = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return fov_density, n_in_radius

    xy = np.array(
        [[c.get("x", np.nan), c.get("y", np.nan)] for c in cells], dtype=float
    )
    valid = np.isfinite(xy).all(axis=1)
    if valid.sum() > 1:
        from scipy.spatial import cKDTree

        pts = xy[valid]
        counts = cKDTree(pts).query_ball_point(pts, r=radius, return_length=True)
        n_in_radius[valid] = (counts - 1).astype(np.float32)  # exclude self
    return fov_density, n_in_radius
