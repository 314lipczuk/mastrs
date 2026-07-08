"""Shared fixtures for the data-layer refactor tests.

Provides ``make_raw`` — a factory that builds tiny in-memory raw-schema
DataFrames for each experiment family (standard / bo / freepattern), so the
adapter + clean/derive tests need no cluster mounts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make repo-root modules (notebooks/, experiments/) importable from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_raw(kind, cells=None, n_frames=15, seed=0):
    """Build a minimal raw-schema frame for one experiment family.

    ``cells`` is a list of ``(label, fov, particle)`` triples (one cell each).
    Intensities are set so the baseline CNR (~0.5) stays under the default
    ``baseline_cnr_max`` so cells survive cleaning; a small bump after frame 10
    gives a response to normalize against.
    """
    if cells is None:
        if kind == "freepattern":
            cells = [("pattern_0", 0, 0), ("pattern_0", 0, 1),
                     ("pattern_1", 1, 0), ("pattern_1", 1, 1)]
        else:
            cells = [("Sustained", 0, 0), ("Sustained", 0, 1)]

    _ = np.random.default_rng(seed)  # reserved for future noise
    rows = []
    for ci, (label, fov, particle) in enumerate(cells):
        for t in range(n_frames):
            stim_on = t in (3, 4, 5)
            ring = 1.0 + (0.2 if t > 10 else 0.0)
            row = dict(
                fov=fov, particle=particle, timestep=t, time=float(t),
                x=float(fov * 10 + particle), y=float(t),
                mean_intensity_C1_nuc=2.0, mean_intensity_C1_ring=ring,
                median_intensity_C1_nuc=2.0, median_intensity_C1_ring=ring,
                mean_intensity_C0_nuc=1.0, mean_intensity_C0_ring=1.0,
                median_intensity_C0_nuc=1.0, median_intensity_C0_ring=1.0,
                stim=stim_on, stim_power=10.0,
                stim_exposure=(float(t) if stim_on else 0.0),
            )
            if kind == "standard":
                row["ramp_pattern_name"] = label
                row["cell_line"] = "EGFR"
                row["area"] = 100.0
            elif kind == "bo":
                row["condition_idx"] = fov
                row["area_nuc"] = 100.0
                row["channels"] = [1, 2]          # object col -> dropped
            elif kind == "freepattern":
                row["treatment_name"] = label
                row["uid"] = ci                   # raw INT uid (treatment id)
                row["area_nuc"] = 100.0
                row["channels"] = [1, 2]
            else:
                raise ValueError(kind)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def make_raw():
    return _build_raw
