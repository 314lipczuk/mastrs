"""Load a serving run directory into dataframes.

One serving run is a directory holding per-FOV track parquets under ``tracks/``
and a ``*.jsonl`` server log. Neither half is usable alone: the parquets carry
the segmentation and the real acquisition clock, the log carries the exposure
the controller actually commanded and the reference it was tracking. This
module joins them and is the single place that knows how.

It exists because the join is not obvious. faro writes each track file as a
running snapshot rather than a per-phase slice, the log's optional fields are
absent for whole FOVs at a time, and the parquet ``stim_exposure`` column is
mangled. Every one of those has silently corrupted an analysis at least once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

__all__ = ["ServingRun", "load_serving_run"]


@dataclass
class ServingRun:
    """Everything one serving run directory contains, already joined.

    ``data_all`` is the analysis table: one row per (fov, timestep, particle),
    spanning every phase. Cut it to a single phase before plotting — phases do
    not share a cadence or a treatment.
    """

    data_all: pl.DataFrame
    serving: pl.DataFrame
    timing: pl.DataFrame
    gpu: pl.DataFrame | None
    startup: dict
    gpu_device: dict
    exp_dir: Path
    log_path: Path


def _phase_of(path: Path) -> int:
    m = re.search(r"_phase_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _load_tracks(files: list[Path]) -> pl.DataFrame:
    """Concatenate the per-(FOV, phase) parquets without double-counting.

    ``tracks/`` holds one file per (FOV, phase) — ``<fov>_phase_<n>_latest.parquet``
    — and faro writes each as a running snapshot of the tracker, spanning every
    frame seen so far rather than only that phase's. So ``0_phase_1_latest``
    already contains all of ``0_phase_0_latest``'s rows, under the same particle
    ids.

    Two consequences, both handled here:

    * the optocheck phase never stimulates, so faro writes no ``stim_exposure`` /
      ``stim_power`` / ``time_offset`` columns for it — a plain ``pl.concat`` dies
      on the width mismatch (29 vs 32).
    * concatenating every file double-counts every frame of the earlier phases.

    Read highest-phase-first, union the schemas (``diagonal`` fills the columns a
    phase lacks with null), and keep the first row per (fov, timestep, particle):
    the widest, latest version of a frame wins, and a frame that only the earlier
    file has still survives.
    """
    return pl.concat(
        [
            pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("source_file"))
            for f in sorted(files, key=_phase_of, reverse=True)
        ],
        how="diagonal",
    ).unique(subset=["fov", "timestep", "particle"], keep="first", maintain_order=True)


def _read_log(log_path: Path) -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    """One pass over the server log's four event kinds.

    ``predict`` — one per (fov, timestep), holding the scored cells and a
    ``timing`` block for that inference call.
    ``gpu`` — a periodic device sample, absent in pre-v4 runs.
    ``startup`` — engine, checkpoint and the resolved policy, logged once. This
    is the authority on what the run actually did; a policy file in the repo may
    have moved on, and the run directory's own copy may not be the one passed to
    ``--policy-file``.
    ``gpu_device`` — which physical GPU the samples describe, and whether that was
    verified against the model's device. Absent on runs that predate the record,
    which are exactly the runs where the two could silently disagree.
    """
    srv_rows: list[dict] = []
    timing_rows: list[dict] = []
    gpu_rows: list[dict] = []
    startup: dict = {}
    gpu_device: dict = {}

    with open(log_path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("event")
            if ev == "startup":
                startup = rec
                continue
            if ev == "gpu_device":
                gpu_device = rec
                continue
            if ev == "gpu":
                # `procs` is a long per-PID list nothing plots; keep the scalars.
                gpu_rows.append({k: v for k, v in rec.items() if k not in ("procs", "event")})
                continue
            if ev != "predict":
                continue
            tm = rec.get("timing") or {}
            timing_rows.append(
                {
                    "t": rec["t"],
                    "fov": rec["fov"],
                    "timestep": rec["timestep"],
                    "n_scored": rec.get("n_scored"),
                    "infer_s": tm.get("infer_s"),
                    "handler_s": tm.get("handler_s"),
                    "lock_wait_s": tm.get("lock_wait_s"),
                    "cuda_alloc_mb": tm.get("cuda_alloc_mb"),
                    "cuda_reserved_mb": tm.get("cuda_reserved_mb"),
                }
            )
            # Explode the cells to one row per (fov, timestep, particle) and keep
            # the true stim exposure the model commanded — the parquet's own
            # `stim_exposure` column was mangled by faro. Duplicate keys are
            # byte-identical repeats, so keeping the first is safe.
            for c in rec["cells"]:
                srv_rows.append(
                    {
                        "fov": rec["fov"],
                        "timestep": rec["timestep"],
                        "particle": c["particle"],
                        "exposure_ms": c["exposure_ms"],
                        "fluence_out": c["fluence_out"],
                        # The normalized signal the controller targets
                        # (= raw_cnr / server baseline); not in the parquet.
                        "cnr_norm": c.get("cnr_norm"),
                        "baseline": c.get("baseline"),
                        # The reference the controller was actually tracking for
                        # THIS cell at THIS frame, as the server recorded it
                        # (Objective.annotate). For a `hold` it is the constant
                        # target; for an `oscillation` it is the step train
                        # evaluated at the cell's own phase offset; for a
                        # `schedule` it is the piecewise-constant waveform. Which
                        # is why it is a per-row column and not a scalar.
                        "r_t": c.get("r_t"),
                        # Waveform segment (settle / low_hold / rise / high_hold /
                        # fall). `settle` marks the frames where the start-up
                        # transient and the tracking response overlap.
                        "segment": c.get("segment"),
                        "phase_offset_min": c.get("phase_offset_min"),
                        # What the controller BELIEVED would happen: the winning
                        # plan's cost, and the predicted CNR one step ahead under
                        # the dose it actually commanded. Without them a saturated
                        # cell and a mispredicted cell are indistinguishable.
                        "plan_cost": c.get("plan_cost"),
                        "pred_cnr_h1": c.get("pred_cnr_h1"),
                    }
                )
    return srv_rows, timing_rows, gpu_rows, startup, gpu_device


def _build_serving(srv_rows: list[dict]) -> pl.DataFrame:
    """SCHEMA IS DECLARED, NOT INFERRED.

    polars types a list of dicts from its first 100 rows, and every optional
    field here is absent for whole FOVs at a time rather than at random: a `hold`
    objective annotates no `segment` and no `phase_offset_min`, a run that
    predates an annotation has none of them anywhere. Whenever the first FOVs
    served are the ones WITHOUT a field, the column types as Null and the first
    real value — "settle" arriving from another FOV a few hundred rows later —
    kills the whole load with a builder-append error. That is not a corrupt log;
    it is a run whose FOVs carry different objectives, which is the normal case.

    Declaring the types costs nothing and removes the ordering dependence.
    `infer_schema_length=None` would also work but scans every row of a
    multi-million-row list to learn what is written here in seven lines.
    """
    return (
        pl.DataFrame(
            srv_rows,
            schema_overrides={
                "cnr_norm": pl.Float64,
                "baseline": pl.Float64,
                "r_t": pl.Float64,
                "segment": pl.Utf8,
                "phase_offset_min": pl.Float64,
                "plan_cost": pl.Float64,
                "pred_cnr_h1": pl.Float64,
            },
        )
        .unique(subset=["fov", "timestep", "particle"], keep="first")
        .with_columns(
            pl.col("fov").cast(pl.UInt16),
            pl.col("timestep").cast(pl.UInt32),
            pl.col("particle").cast(pl.UInt32),
            # On a run that predates the reference annotation these are all-null,
            # which polars infers as the Null dtype — cast so downstream
            # arithmetic and joins see a normal (empty) float/str column instead
            # of failing on a type they cannot subtract.
            pl.col("r_t").cast(pl.Float64),
            pl.col("phase_offset_min").cast(pl.Float64),
            pl.col("segment").cast(pl.Utf8),
            pl.col("plan_cost").cast(pl.Float64),
            pl.col("pred_cnr_h1").cast(pl.Float64),
        )
    )


def load_serving_run(exp_dir: str | Path) -> ServingRun:
    """Join a serving run's track parquets and server log into one table.

    Raises ``FileNotFoundError`` if the directory is not a serving run — that is,
    if it lacks either ``tracks/*.parquet`` or a top-level ``*.jsonl``.
    """
    exp_dir = Path(str(exp_dir).strip())
    # Only `tracks/` — the run dir also holds a top-level `exp_data.parquet` that
    # is the same rows concatenated, so a recursive glob would double-count them.
    files = sorted((exp_dir / "tracks").glob("*.parquet"))
    logs = sorted(exp_dir.glob("*.jsonl"))
    if not files or not logs:
        raise FileNotFoundError(
            f"{exp_dir} is not a serving run: needs tracks/*.parquet and a *.jsonl log "
            f"(found {len(files)} parquets, {len(logs)} logs)"
        )
    log_path = logs[0]

    tracks = _load_tracks(files)
    srv_rows, timing_rows, gpu_rows, startup, gpu_device = _read_log(log_path)
    serving = _build_serving(srv_rows)

    # Wall-clock `t` is epoch seconds; `predict` events carry both `t` and the
    # experiment `timestep`, so they define the map from epoch to experiment
    # hours that the `gpu` samples (which only have `t`) are placed on.
    timing = (
        pl.DataFrame(timing_rows).sort("t").with_columns((pl.col("timestep") / 60.0).alias("hours"))
    )
    gpu = (
        pl.DataFrame(gpu_rows, schema_overrides={"t": pl.Float64}).sort("t") if gpu_rows else None
    )

    ta = pl.col("time_acquired").str.strptime(pl.Datetime, "%Y-%m-%d-%H:%M:%S")
    data_all = tracks.join(serving, on=["fov", "timestep", "particle"], how="left").with_columns(
        # `hours` IS THE REAL CLOCK, read off `time_acquired` — the moment the
        # frame actually landed — relative to the first frame of the run.
        #
        # It used to be `time / 3600`, and `time` is the *planned* schedule the
        # acquisition was programmed with, i.e. `timestep / 60` under a nominal
        # 1-minute cadence. Whenever the loop cannot hold that cadence the two
        # diverge without warning, and every axis silently reads as the experiment
        # that was intended rather than the one that ran. Keep the plan under its
        # own name and check them against each other before reading any rate off
        # a plot.
        #
        # Per-row, not per-timestep: within one timestep the FOVs are imaged
        # sequentially and can be minutes apart, so each FOV keeps its own
        # acquisition time.
        ((ta - ta.min()).dt.total_seconds() / 3600.0).alias("hours"),
        (pl.col("time") / 3600.0).alias("hours_planned"),
        # True light-on flag from the server exposure, not the mangled parquet
        # `stim` column.
        (pl.col("exposure_ms") > 0).alias("light_on"),
        # Track id unique across FOVs, so pooling doesn't merge same-id particles.
        (pl.col("fov").cast(pl.Utf8) + "_" + pl.col("particle").cast(pl.Utf8)).alias("track_key"),
    )

    # Put the server's clock on the acquisition clock. `predict` events carry
    # epoch `t` and the experiment `timestep`; the tracks carry `time_acquired`
    # for that same timestep. Anchoring on the first shared timestep maps one to
    # the other exactly, without assuming anything about the machine's timezone
    # (the parquet stamps are naive local strings, the log is epoch seconds).
    # Without this the GPU and latency panels sit on a different, nominal axis
    # from the cell plots.
    anchor_ts = int(timing["timestep"].min())
    anchor_epoch = float(timing.filter(pl.col("timestep") == anchor_ts)["t"].min())
    anchor_hours = float(data_all.filter(pl.col("timestep") == anchor_ts)["hours"].min())
    timing = timing.with_columns(
        ((pl.col("t") - anchor_epoch) / 3600.0 + anchor_hours).alias("hours")
    )

    return ServingRun(
        data_all=data_all,
        serving=serving,
        timing=timing,
        gpu=gpu,
        startup=startup,
        gpu_device=gpu_device,
        exp_dir=exp_dir,
        log_path=log_path,
    )
