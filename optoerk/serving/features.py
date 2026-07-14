"""Turn a faro ``/predict`` payload into the model's per-frame channels.

The model input per frame is the standardized 4-vector
``[cnr, u_t, fov_density, n_cells_200px]`` (``history_dataset.CHANNELS``):

  * ``cnr``            — the model was trained on ``cnr_median_norm`` = per-cell
                        baseline-normalized *median* CNR. faro sends raw ``cnr``
                        (and usually ``cnr_median``); the server normalizes online
                        (see :mod:`optoerk.serving.state`). This module only
                        extracts the raw CNR-like scalar.
  * ``u_t``            — commanded fluence (mJ/cm2); supplied by per-cell state
                        (the last commanded dose), not by the payload.
  * ``fov_density``    — number of cells detected in this FOV at this frame.
  * ``n_cells_200px``  — number of *other* cells within ``radius`` px.

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
