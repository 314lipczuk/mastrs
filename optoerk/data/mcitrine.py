"""Join the real optoRTK expression measurement into the training bundle.

**What the measurement is.** optoRTK expression is imaged in its own channel,
**mCitrine**, in a dedicated short acquisition (an "optocheck" / reference frame)
run once or twice per experiment — not in the timelapse, whose channels are
``miRFP`` (C0) and ``mScarlet3`` (C1, the ERK-KTR that ``cnr_median`` comes from).
It lands as **one value per cell**, broadcast to that cell's rows.

**Why this file exists.** ``dataset_all.parquet`` was assembled without it, so
``add_optortk_expression`` fell back to a surrogate: whole-cell C0 =
``0.5 * (mean_intensity_C0_nuc + mean_intensity_C0_ring)``. Measured against the
real thing on the same cells, that surrogate reaches only Spearman 0.60–0.71
within EGFR and puts **27–30% of cells in the wrong half** of a high/low split —
against a measurement whose own test–retest reliability is 0.92 over 5.5 h. The
surrogate is neither noise nor truth, and nothing in the pipeline said so.

**Naming.** Two upstream column names carry the same quantity, because faro
renamed the feature extractor: older Cedric-pipeline exports call it
``optocheck_mean_intensity``, newer faro exports (``RefFE``) call it
``ref_mean_intensity``. Both are the mean mCitrine intensity over the cell mask.
This module normalizes them to ``mcitrine``.

**Coverage.** 98.9% of the 72,441 cells in ``dataset_all.parquet`` resolve to a
finite positive per-cell value. The residual is cells whose (fov, particle) does
not join back upstream — a tracking/filter difference at ingest, not a missing
measurement.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from optoerk.core.utils import imaging_root

# Column that this module writes, and the two upstream spellings it accepts.
MCITRINE_COL = "mcitrine"
UPSTREAM_COLS = ("ref_mean_intensity", "optocheck_mean_intensity")

# original_experiment_name -> the acquisition folder it came from, relative to
# `imaging_root()`. Established by joining on (fov, particle) and taking the
# unambiguous winner: the mapped folder matches 97-100% of the bundle's cells for
# that experiment, while the next-best candidate sits at 88-89%. Recorded here
# because it is not otherwise recoverable — the bundle keeps only a short name.
SOURCE_FOLDERS: dict[str, str] = {
    "bo_v8": "PertzLab/Alex/31_bo_oscillation/2026-05-01_bo_erk_oscillation_v8_freq_range_wider",
    "bo_v10": "PertzLab/Alex/31_bo_oscillation/2026-05-07_bo_erk_oscillation_v10_led_power",
    "bo_v11_10s": "PertzLab/Alex/31_bo_oscillation/2026-05-08_bo_erk_oscillation_v11_light_budget_fixed10s_pi10",
    "bo_v11_20s": "PertzLab/Alex/31_bo_oscillation/2026-05-08_bo_erk_oscillation_v11_light_budget_fixed20s_pi10",
    "freepattern_v1": "PertzLab/Alex/14_free_pattern_TrkA_EGFR/2026-06-23_FreePatternStim_Jungfrau_v1",
    "freepattern_v2": "PertzLab/Alex/14_free_pattern_TrkA_EGFR/2026-06-26_FreePatternStim_Jungfrau_v2",
    "freepattern_Niesen_EGFR_v1": "PertzLab/Alex/14_free_pattern_TrkA_EGFR/2026-07-03_FreePatternStim_Niesen_EGFR_v1",
    "Sustained_1min": "PertzLab/optoRTK_CedricZ/experimental_data/2025-11-02_Sustained_1min",
    "3-2-1minIntervals": "PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals",
    "RampReverse": "PertzLab/optoRTK_CedricZ/experimental_data/2025-09-04_RampReverse",
    "DoseResponse": "PertzLab/optoRTK_CedricZ/experimental_data/2025-10-12_DoseResponse",
}


def _per_cell_mcitrine(exp_data: Path) -> pl.DataFrame:
    """``(fov, particle, mcitrine)`` from one acquisition's ``exp_data.parquet``.

    Filtered to finite positive values. That matters: the column is float, so a
    NaN is *not* a null and a plain ``drop_nulls`` would let it through and be
    counted as present.
    """
    import pyarrow.parquet as pq

    names = pq.read_schema(exp_data).names
    col = next((c for c in UPSTREAM_COLS if c in names), None)
    if col is None:
        raise ValueError(
            f"{exp_data} has neither of {UPSTREAM_COLS}. If this acquisition ran "
            f"with FE_ErkKtrReduced, the mCitrine columns were dropped at "
            f"extraction and the measurement is not recoverable from it."
        )
    return (
        pl.from_arrow(pq.read_table(exp_data, columns=["fov", "particle", col]))
        .with_columns([
            pl.col("fov").cast(pl.Int64),
            pl.col("particle").cast(pl.Int64),
            pl.col(col).cast(pl.Float64).alias(MCITRINE_COL),
        ])
        .filter(pl.col(MCITRINE_COL).is_finite() & (pl.col(MCITRINE_COL) > 0))
        .group_by(["fov", "particle"])
        .agg(pl.col(MCITRINE_COL).first())
    )


def join_mcitrine(
    bundle: pl.DataFrame, *, root: Path | None = None, strict: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add a ``mcitrine`` column to a training bundle; return ``(df, coverage)``.

    Joins on ``(original_experiment_name, fov, particle)``. Cells with no upstream
    match get null, which ``add_optortk_expression`` then ranks as the population
    median rather than dropping — a cell we must still be able to steer.

    ``strict`` fails on an experiment absent from :data:`SOURCE_FOLDERS`, so a
    bundle that grows a new experiment cannot silently ship without the feature.
    """
    root = root or imaging_root()
    have = set(bundle["original_experiment_name"].unique().to_list())
    missing = sorted(have - set(SOURCE_FOLDERS))
    if missing and strict:
        raise ValueError(
            f"no mCitrine source folder mapped for {missing}. Add them to "
            f"SOURCE_FOLDERS (verify by join fraction — the right folder matches "
            f"~100% of that experiment's cells, wrong ones sit well below)."
        )

    parts, rows = [], []
    for exp in sorted(have):
        cells = bundle.filter(pl.col("original_experiment_name") == exp)
        rel = SOURCE_FOLDERS.get(exp)
        if rel is None:
            parts.append(cells.with_columns(pl.lit(None, pl.Float64).alias(MCITRINE_COL)))
            rows.append({"experiment": exp, "cells": cells.height, "matched": 0,
                         "coverage": 0.0, "source": None})
            continue
        up = _per_cell_mcitrine(root / rel / "exp_data.parquet")
        joined = (
            cells.with_columns([pl.col("fov").cast(pl.Int64),
                                pl.col("particle").cast(pl.Int64)])
                 .join(up, on=["fov", "particle"], how="left")
        )
        parts.append(joined)
        n_cells = joined.select("uid").n_unique()
        n_hit = joined.filter(pl.col(MCITRINE_COL).is_not_null()).select("uid").n_unique()
        rows.append({"experiment": exp, "cells": n_cells, "matched": n_hit,
                     "coverage": round(n_hit / n_cells, 4) if n_cells else 0.0,
                     "source": rel.split("/")[-1]})
    return pl.concat(parts, how="vertical_relaxed"), pl.DataFrame(rows).sort("experiment")
