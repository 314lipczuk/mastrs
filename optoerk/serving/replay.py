"""Re-run a recorded serving run offline, through the real service, no faro.

A finished experiment leaves a ``predict`` record per (fov, timestep) holding
every cell's inputs and the exposure that was commanded. That is enough to drive
:class:`~optoerk.serving.service.InferenceService` again in-process and ask two
very different questions. **Keep them apart — conflating them over-claims.**

``replay_faithful``
    Same checkpoint, same policy. Must reproduce the recorded exposures exactly.
    This is a regression test, not a result: it is how you prove a refactor of the
    controller changed nothing, and how you debug "why did it dose that cell".

``replay_counterfactual``
    A *different* policy on the recorded CNR stream. This is **open-loop**: the
    recorded CNRs are the response the cells gave to the *original* doses, so
    feeding them to a policy that would have dosed differently is a fiction after
    the first frame where the two disagree. Useful for "how often would this
    policy have disagreed", useless for "would this policy have tracked better".

``simulate_closed_loop``
    Closes the loop with the model standing in for the cells: the commanded dose
    is rolled one step through the decoder and the prediction becomes the next
    frame's observation. This *can* compare tracking performance between
    policies — but it measures the policy against **the model's own beliefs**, so
    it rewards a controller that exploits model error. Report it as simulation.

The log is streamed, so a 200 MB run does not have to fit in memory.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from optoerk.serving.config import ServerConfig
from optoerk.serving.features import compute_crowding
from optoerk.serving.service import InferenceService


def iter_predict_records(log_path: str | Path) -> Iterator[dict]:
    """Stream the ``predict`` records of a serving log in file order."""
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line after a crash is expected
            if rec.get("event") == "predict":
                yield rec


def startup_record(log_path: str | Path) -> dict | None:
    """The run's ``startup`` record — checkpoint, engine and (for runs recorded
    after per-FOV policies landed) every resolved policy."""
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "startup":
                return rec
    return None


def load_positions(tracks_dir: str | Path) -> pl.DataFrame:
    """``(fov, timestep, particle) -> x, y`` from the per-FOV track parquets.

    The server log does not record positions (they are only used to derive the
    crowding channels), but objectives gate on them, so a replay that exercises a
    position-gated policy needs them joined back in. FOV is taken from the leading
    integer of the filename, matching how faro writes ``<fov>_phase_1_latest.parquet``.
    """
    frames = []
    for path in sorted(Path(tracks_dir).glob("*.parquet")):
        stem = path.stem.split("_")[0]
        if not stem.isdigit():
            continue
        df = pl.read_parquet(path, columns=["timestep", "particle", "x", "y"])
        frames.append(df.with_columns(pl.lit(int(stem)).alias("fov")))
    if not frames:
        raise FileNotFoundError(f"no <fov>_*.parquet track files under {tracks_dir}")
    return pl.concat(frames).select("fov", "timestep", "particle", "x", "y")


def record_to_payload(rec: dict, positions: dict | None = None) -> dict:
    """Rebuild the ``/predict`` payload that produced a recorded response.

    ``raw_cnr`` is replayed as ``cnr_median`` because that is the field
    :func:`~optoerk.serving.features.extract_raw_cnr` prefers, so the same scalar
    comes back out. Crowding is *not* injected — it is recomputed from x/y exactly
    as it was live, which :func:`replay_faithful` then cross-checks against the
    recorded values.
    """
    fov, ts = rec["fov"], rec["timestep"]
    cells = []
    for cell in rec.get("cells", []):
        particle = cell["particle"]
        xy = (positions or {}).get((fov, ts, particle))
        cells.append({
            "particle": particle,
            "cnr_median": cell["raw_cnr"],
            "x": float("nan") if xy is None else xy[0],
            "y": float("nan") if xy is None else xy[1],
        })
    return {"fov": fov, "timestep": ts, "cells": cells}


def _positions_index(tracks_dir) -> dict | None:
    if tracks_dir is None:
        return None
    df = load_positions(tracks_dir)
    return {
        (r["fov"], r["timestep"], r["particle"]): (r["x"], r["y"])
        for r in df.iter_rows(named=True)
    }


def replay(
    log_path: str | Path,
    cfg: ServerConfig,
    tracks_dir: str | Path | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Drive a fresh service with a recorded run's frames.

    Returns one row per (fov, timestep, particle): the recorded inputs, the
    recorded exposure, and the exposure this configuration commands instead.
    """
    positions = _positions_index(tracks_dir)
    service = InferenceService(cfg)
    rows = []
    try:
        for i, rec in enumerate(iter_predict_records(log_path)):
            if limit is not None and i >= limit:
                break
            payload = record_to_payload(rec, positions)
            # The crowding channels the replay will derive from the rebuilt
            # positions. Comparing these to the recorded ones is the check that
            # the positions were joined back correctly — without it, a silently
            # empty position join looks like a policy difference.
            density, n200 = compute_crowding(
                payload["cells"], radius=cfg.crowd_radius_px
            )
            out = service.predict(payload)
            got = out["exposures"]
            for j, cell in enumerate(rec.get("cells", [])):
                particle = cell["particle"]
                rows.append({
                    "fov": rec["fov"],
                    "timestep": rec["timestep"],
                    "particle": particle,
                    "raw_cnr": cell["raw_cnr"],
                    "cnr_norm_recorded": cell["cnr_norm"],
                    "fov_density_recorded": cell["fov_density"],
                    "fov_density_replayed": float(density[j]),
                    "n_cells_200px_recorded": cell["n_cells_200px"],
                    "n_cells_200px_replayed": float(n200[j]),
                    "exposure_recorded": cell["exposure_ms"],
                    "exposure_replayed": got.get(str(particle)),
                })
    finally:
        service.close()
    return pl.DataFrame(rows)


def replay_faithful(
    log_path: str | Path,
    cfg: ServerConfig,
    tracks_dir: str | Path | None = None,
    limit: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Replay with the original policy and report how exactly it reproduced.

    The returned summary is the acceptance gate for controller refactors:
    ``exposure_match_frac`` must be 1.0 for an unchanged policy.
    """
    df = replay(log_path, cfg, tracks_dir, limit)
    if df.is_empty():
        return df, {"n_rows": 0}
    df = df.with_columns(
        (pl.col("exposure_recorded") == pl.col("exposure_replayed")).alias("exposure_match")
    )
    summary = {
        "n_rows": df.height,
        "n_frames": df.select(pl.struct("fov", "timestep").n_unique()).item(),
        "exposure_match_frac": float(df["exposure_match"].mean()),
        "mean_abs_exposure_delta": float(
            (df["exposure_replayed"] - df["exposure_recorded"]).abs().mean()
        ),
        # If crowding disagrees, the replayed positions are wrong, and every
        # downstream comparison is suspect — surface it rather than hide it.
        "crowding_match_frac": float(
            df.select(
                (
                    (pl.col("fov_density_recorded") == pl.col("fov_density_replayed"))
                    & (pl.col("n_cells_200px_recorded") == pl.col("n_cells_200px_replayed"))
                ).mean()
            ).item()
        ),
    }
    return df, summary


def replay_counterfactual(
    log_path: str | Path,
    cfg: ServerConfig,
    tracks_dir: str | Path | None = None,
    limit: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Open-loop: what would a different policy have commanded on this CNR stream?

    Valid for measuring *disagreement* with the run that happened. NOT valid for
    claiming the new policy would have tracked better — the recorded CNRs are the
    cells' response to the original doses.
    """
    df = replay(log_path, cfg, tracks_dir, limit)
    if df.is_empty():
        return df, {"n_rows": 0}
    delta = df["exposure_replayed"] - df["exposure_recorded"]
    summary = {
        "n_rows": df.height,
        "disagreement_frac": float((delta != 0).mean()),
        "mean_exposure_recorded": float(df["exposure_recorded"].mean()),
        "mean_exposure_replayed": float(df["exposure_replayed"].mean()),
        "mean_signed_delta_ms": float(delta.mean()),
        "open_loop_caveat": (
            "recorded CNRs are the response to the ORIGINAL doses; this measures "
            "disagreement, not tracking performance"
        ),
    }
    return df, summary


def simulate_closed_loop(
    cfg: ServerConfig,
    n_cells: int = 32,
    n_frames: int = 60,
    fov: int = 0,
    cnr0: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Closed-loop rollout with the model standing in for the cells.

    Each frame: the service commands a dose, and the engine's own decoder is
    rolled one step with that dose to produce the next observed CNR. This is the
    only replay mode that can compare *tracking* between policies — but it scores
    a controller against the model's beliefs, so a controller that exploits model
    error will look good here and fail on the microscope. Simulation, not
    measurement.
    """
    import torch

    rng = np.random.default_rng(seed)
    service = InferenceService(cfg)
    engine = service.router.engine_for(fov)
    if not hasattr(engine, "rollout"):
        service.close()
        raise TypeError(
            "closed-loop simulation needs a RealModelEngine (a checkpoint); the "
            "stub has no forward model to close the loop with"
        )

    xs = rng.uniform(0, 1024, n_cells)
    ys = rng.uniform(0, 1024, n_cells)
    cnr = np.full(n_cells, float(cnr0))
    rows = []
    try:
        for t in range(n_frames):
            payload = {
                "fov": fov, "timestep": t,
                "cells": [
                    {"particle": i, "cnr_median": float(cnr[i]),
                     "x": float(xs[i]), "y": float(ys[i])}
                    for i in range(n_cells)
                ],
            }
            exposures = service.predict(payload)["exposures"]
            ms = np.array([exposures[str(i)] for i in range(n_cells)], float)

            for i in range(n_cells):
                rows.append({"timestep": t, "particle": i, "cnr": float(cnr[i]),
                             "exposure_ms": float(ms[i])})

            # Advance the "cells" one step with the model's own dynamics, using the
            # encoder state the service just wrote for each particle.
            with torch.no_grad():
                states = [service.store.get(fov, i) for i in range(n_cells)]
                h = torch.cat([s.h for s in states], dim=1)
                c = torch.cat([s.c for s in states], dim=1)
                fb = torch.tensor(
                    [[(cnr[i] - engine.mean_np[0]) / float(engine.std[0].item())]
                     for i in range(n_cells)],
                    dtype=torch.float32, device=engine.device,
                )
                fut = engine.std_fluence(
                    torch.tensor(ms, dtype=torch.float32, device=engine.device)
                ).view(n_cells, 1, 1)
                pred = engine.rollout(h, c, fb, fut)          # (N, 1) standardized
                cnr = engine.denorm_cnr(pred[:, 0]).cpu().numpy()
    finally:
        service.close()
    return pl.DataFrame(rows)
