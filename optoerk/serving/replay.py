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


def open_loop_ensemble(engine, ens: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-cell covariates for an ensemble of ``ens`` nominal cells.

    Only the channels the model actually conditions on per cell are varied; the
    rest are left to their population mean, which is what serving feeds when the
    payload omits them.

    Split out from :func:`simulate_open_loop` so the ensemble is an explicit,
    inspectable input rather than a hidden RNG draw. A schedule is only as good as
    the population it was optimised for, so that population belongs in the record.
    """
    rng = np.random.default_rng(seed)
    chans = list(engine.channels)
    out: dict[str, np.ndarray] = {}
    if "optortk_expr" in chans:
        # A session rank in (0, 1] — the range the live feature can produce.
        out["optortk_expr"] = rng.uniform(0.05, 0.95, ens).astype(np.float32)
    if "nuc_area" in chans:
        _m = float(engine.mean_np[chans.index("nuc_area")])
        out["nuc_area"] = (_m * rng.uniform(0.7, 1.3, ens)).astype(np.float32)
    return out


def simulate_open_loop(
    engine,
    seq_ms: np.ndarray,
    n_frames: int,
    ens: int,
    seed: int = 0,
    static: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Roll fixed dose schedules through the model, with no feedback anywhere.

    ``seq_ms`` is ``(C, P)``: ``C`` candidate schedules of ``P`` frames each, tiled
    to ``n_frames``. Returns predicted CNR of shape ``(C, ens, n_frames)``.

    This is what designs an open-loop control arm. Unlike
    :func:`simulate_closed_loop` no controller is consulted — the dose at frame
    ``t`` is ``seq_ms[:, t % P]`` whatever the cells are doing, which is precisely
    the property the arm exists to embody.

    **All ``C`` schedules are simulated in one pass**, as ``C * ens`` pseudo-cells,
    so an optimiser evaluating hundreds of candidates costs one sweep of encoder
    steps rather than one sweep per candidate.

    The ensemble varies the per-cell covariates the model conditions on, so a
    schedule is optimised for a POPULATION. Tuning one to the median cell would
    produce a weaker opponent and flatter any heterogeneity result the closed-loop
    arm is meant to show.

    Same standing warning as :func:`simulate_closed_loop`: the cells here are the
    model's own decoder, so this predicts what the model believes a schedule will
    do. It is the hypothesis the real run exists to falsify, never evidence.
    """
    import torch

    dev = engine.device
    seq_ms = np.asarray(seq_ms, dtype=np.float32)
    if seq_ms.ndim != 2:
        raise ValueError(f"seq_ms must be (C, P); got shape {seq_ms.shape}")
    if not hasattr(engine, "rollout"):
        raise TypeError(
            "open-loop simulation needs a RealModelEngine (a checkpoint); the stub "
            "has no forward model to roll"
        )
    C, P = seq_ms.shape
    N = C * ens
    chans = list(engine.channels)
    mean_np = engine.mean_np

    # Per-cell covariates, tiled so every candidate sees the SAME population — two
    # schedules must differ by the schedule alone, never by which cells they drew.
    per_cell = open_loop_ensemble(engine, ens, seed) if static is None else static
    tiled = {}
    for name, vals in per_cell.items():
        vals = np.asarray(vals, dtype=np.float32)
        if vals.shape != (ens,):
            raise ValueError(f"static[{name!r}] must have shape ({ens},), got {vals.shape}")
        tiled[name] = np.tile(vals, C)

    cnr = np.full(N, float(mean_np[chans.index("cnr")]), dtype=np.float32)
    last_flu = np.zeros(N, dtype=np.float32)
    h = torch.zeros(engine.num_layers, N, engine.hidden, device=dev)
    c = torch.zeros(engine.num_layers, N, engine.hidden, device=dev)
    doses = np.repeat(seq_ms, ens, axis=0)                     # (N, P)
    out = np.empty((N, n_frames), dtype=np.float32)

    with torch.no_grad():
        for t in range(n_frames):
            # Channel assembly by NAME, mirroring RealModelEngine.decide: a channel
            # the ensemble does not vary gets its population mean, which is exactly
            # what serving feeds when the payload omits it.
            raw = np.empty((N, len(chans)), dtype=np.float32)
            for j, name in enumerate(chans):
                if name == "cnr":
                    raw[:, j] = cnr
                elif name == "u_t":
                    raw[:, j] = last_flu
                elif name in tiled:
                    raw[:, j] = tiled[name]
                else:
                    raw[:, j] = mean_np[j]
            xs = (torch.tensor(raw, device=dev) - engine.mean) / engine.std
            _, (h, c) = engine.model.encoder.lstm(xs.unsqueeze(1), (h, c))

            ms = torch.tensor(doses[:, t % P], dtype=torch.float32, device=dev)
            fb = xs[:, chans.index("cnr")].reshape(N, 1)
            fut = engine.std_fluence(ms).view(N, 1, 1)
            pred = engine.rollout(h, c, fb, fut)                # (N, 1) standardized
            cnr = engine.denorm_cnr(pred[:, 0]).cpu().numpy()
            # RAW fluence, not standardized: this is the `u_t` channel value, and
            # the channel block above standardizes it along with everything else.
            # It is what `decide` writes to `state.last_fluence`
            # (calib.ms_to_fluence == ms * fluence_per_ms).
            last_flu = doses[:, t % P] * float(engine._flu_per_ms)
            out[:, t] = cnr
    return out.reshape(C, ens, n_frames)


def score_open_loop(engine, objective, cnr: np.ndarray, start_frame: int) -> np.ndarray:
    """Each candidate's cost under the objective's OWN cost function.

    ``cnr`` is ``(C, ens, T)`` from :func:`simulate_open_loop`. Returns ``(C,)``, the
    cost averaged over the ensemble.

    Deliberately calls :meth:`Objective.cost` rather than computing an L2 by hand:
    the open-loop schedule has to be optimised against exactly what the closed-loop
    arms minimise, kernel and regularizers included. A hand-rolled score would let
    the two arms drift apart silently, and the whole comparison rests on them
    optimising the same thing.
    """
    import torch

    from optoerk.serving.objectives import GoalContext, Prediction
    from optoerk.serving.runtime import CellFrame
    from optoerk.serving.state import CellState

    C, ens, T = cnr.shape
    N = C * ens
    cells = []
    for i in range(N):
        st = CellState()
        st.particle = i
        cells.append(CellFrame(state=st, cnr_norm=1.0, fov_density=float(ens),
                               n_cells_200px=5.0, x=0.0, y=0.0))
    ctx = GoalContext(fov=0, timestep=start_frame, cells=cells,
                      control_frame=start_frame)
    pred = Prediction(
        cnr=torch.tensor(cnr.reshape(N, 1, T), dtype=torch.float32, device=engine.device)
    )
    cost = objective.cost(pred, ctx)                            # (N, 1)
    return cost.view(C, ens).mean(dim=1).cpu().numpy()


def pulse_train_candidates(
    period: int,
    levels_ms,
    offset_step: int = 1,
    duration_step: int = 1,
    off_levels=(0.0,),
) -> tuple[np.ndarray, list[dict]]:
    """Every schedule of the shape "dose A for D frames from offset S, else B".

    Returns ``(schedules (C, period), params)`` — the candidate set for an
    open-loop arm, enumerated exhaustively rather than searched.

    **Why a shape and not free slots.** Optimising all ``period`` slots
    independently lets the search dither between rungs frame by frame, and a
    dithered schedule is where an open-loop arm goes wrong twice over: it exploits
    whatever the model is locally wrong about, so it beats its rivals in simulation
    and underperforms on the rig — which biases the closed-loop comparison in the
    direction the experimenter is hoping for. And it cannot be stated in a thesis
    except as a table of numbers.

    Constraining to a pulse train removes both problems. The space is small enough
    to enumerate, so the result is deterministic and reproducible with no optimiser
    to tune, and the answer is one sentence: this dose, for this long, starting
    here in each cycle.

    ``duration == period`` is a constant schedule, so a constant reference (which
    wants exactly that) is covered by the same enumeration with ``period = 1``.
    """
    levels = np.asarray(levels_ms, dtype=np.float32)
    P = int(period)
    if P < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    on_levels = [float(v) for v in levels if v > 0] or [float(levels[0])]

    rows, params = [], []
    for off in (float(v) for v in off_levels):
        for amp in on_levels:
            if amp == off:
                continue
            for start in range(0, P, max(1, int(offset_step))):
                for dur in range(max(1, int(duration_step)), P + 1, max(1, int(duration_step))):
                    seq = np.full(P, off, dtype=np.float32)
                    idx = (np.arange(dur) + start) % P
                    seq[idx] = amp
                    rows.append(seq)
                    params.append({"amp_ms": amp, "off_ms": off,
                                   "start": start, "duration": dur})
                    if dur == P:
                        break            # a full-period pulse is offset-invariant
    return np.stack(rows), params
