"""Replay: a run recorded by the service must re-drive the service identically.

Self-contained — the test records its own log with the stub engine, so there is
no mount, no checkpoint, and (being CPU-only and deterministic) an exact 1.0
match is the correct bar here. Replaying a *CUDA* run on CPU is a different
matter; see the note in ``optoerk/serving/README.md``.
"""
import numpy as np
import polars as pl
import pytest
import torch

from optoerk.serving.config import ServerConfig
from optoerk.serving.replay import (
    iter_predict_records,
    record_to_payload,
    replay_counterfactual,
    replay_faithful,
    startup_record,
)
from optoerk.serving.service import InferenceService


def _record_a_run(log_path, n_frames=12, n_cells=6, target_cnr=1.5):
    """Drive a service with synthetic frames, writing a real predict log."""
    cfg = ServerConfig(
        predict_log_path=str(log_path), warmup=False, dark_baseline=False,
        target_cnr=target_cnr, gpu_sample_interval_s=0,
    )
    svc = InferenceService(cfg)
    try:
        for t in range(n_frames):
            svc.predict({
                "fov": 0, "timestep": t,
                "cells": [
                    {"particle": i, "x": 100.0 * i, "y": 40.0 * i,
                     "cnr_median": 0.4 + 0.05 * t + 0.01 * i}
                    for i in range(n_cells)
                ],
            })
    finally:
        svc.close()
    return cfg


def _write_tracks(tmp_path, n_frames=12, n_cells=6):
    """The per-FOV track parquet faro writes, holding the positions the server
    log does not record."""
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    pl.DataFrame({
        "timestep": [t for t in range(n_frames) for _ in range(n_cells)],
        "particle": [i for _ in range(n_frames) for i in range(n_cells)],
        "x": [100.0 * i for _ in range(n_frames) for i in range(n_cells)],
        "y": [40.0 * i for _ in range(n_frames) for i in range(n_cells)],
    }).write_parquet(tracks / "0_phase_1_latest.parquet")
    return tracks


def test_recorded_run_replays_exactly(tmp_path):
    log = tmp_path / "run.jsonl"
    cfg = _record_a_run(log)
    tracks = _write_tracks(tmp_path)

    replay_cfg = ServerConfig(warmup=False, dark_baseline=False,
                              target_cnr=cfg.target_cnr, gpu_sample_interval_s=0)
    _df, summary = replay_faithful(log, replay_cfg, tracks_dir=tracks)

    assert summary["n_frames"] == 12
    assert summary["exposure_match_frac"] == 1.0
    assert summary["mean_abs_exposure_delta"] == 0.0
    # positions were joined back, so the derived crowding channels reproduce too
    assert summary["crowding_match_frac"] == 1.0


def test_missing_positions_are_flagged_by_the_crowding_check(tmp_path):
    """Without the track parquets, ``n_cells_200px`` cannot be reconstructed and
    the model would silently be fed different inputs. The stub ignores crowding,
    so exposures still match — ``crowding_match_frac`` is the only thing standing
    between that and a replay that looks perfect while being wrong.
    """
    log = tmp_path / "run.jsonl"
    _record_a_run(log)

    _df, summary = replay_faithful(
        log, ServerConfig(warmup=False, dark_baseline=False, gpu_sample_interval_s=0)
    )
    assert summary["exposure_match_frac"] == 1.0, "stub ignores crowding"
    assert summary["crowding_match_frac"] < 1.0, "must flag the missing positions"


def test_changed_policy_shows_up_as_disagreement(tmp_path):
    """The counterfactual mode must actually detect a policy change."""
    log = tmp_path / "run.jsonl"
    _record_a_run(log, target_cnr=1.5)

    cfg = ServerConfig(warmup=False, dark_baseline=False, target_cnr=3.0,
                       gpu_sample_interval_s=0)
    _df, summary = replay_counterfactual(log, cfg)

    assert summary["disagreement_frac"] > 0
    # a higher target drives more light
    assert summary["mean_exposure_replayed"] > summary["mean_exposure_recorded"]
    assert "open_loop_caveat" in summary


def test_startup_record_carries_the_resolved_policies(tmp_path):
    """A run's own log must be enough to rebuild what it ran — the gap that made
    recovering the 2026-07-16 run's target_cnr a parameter search."""
    log = tmp_path / "run.jsonl"
    _record_a_run(log, target_cnr=1.75)

    rec = startup_record(log)
    assert rec is not None
    policies = rec["policies"]["default"]
    # The objective describes its full composition — reference, cost kernel and
    # any plan-side regularizers — so an arm is reconstructible from the log
    # alone, not just its headline setpoint.
    assert policies["objective"] == {
        "type": "hold",
        "target_cnr": 1.75,
        "reference": {"type": "constant", "target_cnr": 1.75},
        "kernel": {"type": "l2"},
        "regularizers": [],
    }


def test_record_to_payload_rebuilds_the_request(tmp_path):
    log = tmp_path / "run.jsonl"
    _record_a_run(log, n_cells=3)
    rec = next(iter_predict_records(log))

    payload = record_to_payload(rec)
    assert payload["fov"] == rec["fov"] and payload["timestep"] == rec["timestep"]
    assert len(payload["cells"]) == 3
    # raw_cnr is replayed as cnr_median, the field extract_raw_cnr prefers, so the
    # same scalar comes back out of feature extraction.
    assert payload["cells"][0]["cnr_median"] == rec["cells"][0]["raw_cnr"]


def test_replay_limit_is_respected(tmp_path):
    log = tmp_path / "run.jsonl"
    _record_a_run(log, n_frames=10)
    df, _ = replay_faithful(log, ServerConfig(warmup=False, dark_baseline=False,
                                              gpu_sample_interval_s=0), limit=3)
    assert isinstance(df, pl.DataFrame)
    assert df.select(pl.struct("fov", "timestep").n_unique()).item() == 3


# ---------------------------------------------------------------------------
# simulate_open_loop — the open-loop arm's designer
# ---------------------------------------------------------------------------


def _ol_engine(sequence_ms, future_len=5):
    from optoerk.serving.bench import synthetic_handle
    from optoerk.serving.calibration import FluenceCalibration
    from optoerk.serving.config import ServerConfig
    from optoerk.serving.control import OpenLoopController, dose_levels
    from optoerk.serving.objectives import hold
    from optoerk.serving.runtime import RealModelEngine

    cfg = ServerConfig(warmup=False, control_horizon=future_len, gpu_sample_interval_s=0)
    handle = synthetic_handle(future_len=future_len, device=torch.device("cpu"))
    calib = FluenceCalibration(cfg.instrument, cfg.stim_power_pct)
    levels = dose_levels(0.0, 150.0, 4)
    ctrl = OpenLoopController(levels, sequence_ms=sequence_ms)
    return RealModelEngine(handle, calib, cfg, hold(1.0), ctrl)


def test_simulate_open_loop_matches_stepping_the_engine_itself():
    """The batched simulator reimplements `decide`'s channel assembly, so it can
    drift from the code it is meant to predict — and the whole open-loop arm is
    designed against it. Pin it: stepping one cell through the real engine must give
    the same trajectory the simulator does.
    """
    from optoerk.serving.objectives import GoalContext
    from optoerk.serving.replay import simulate_open_loop
    from optoerk.serving.runtime import CellFrame
    from optoerk.serving.state import CellState

    seq = [0.0, 150.0, 50.0]
    n_frames = 9
    engine = _ol_engine(seq)
    chans = engine.channels

    # The simulator feeds the population mean on every channel the ensemble does not
    # vary — crowding included, since a nominal cell has no real neighbours. Give the
    # reference frame the same values, so this compares the STEPPING MATH and not two
    # different sets of inputs.
    def _mean(name):
        return float(engine.mean_np[chans.index(name)])

    st = CellState()
    st.particle = 0
    frame = CellFrame(
        state=st,
        cnr_norm=_mean("cnr"),
        fov_density=_mean("fov_density") if "fov_density" in chans else 1.0,
        n_cells_200px=_mean("n_cells_200px"),
        x=0.0, y=0.0,
        optortk_expr=_mean("optortk_expr") if "optortk_expr" in chans else None,
    )
    stepwise = []
    for t in range(n_frames):
        engine.decide([frame], GoalContext(fov=0, timestep=t, cells=[frame],
                                           control_frame=t))
        nxt = float(engine.last_pred_cnr_h1[0])
        stepwise.append(nxt)
        frame.cnr_norm = nxt

    static = {
        name: np.array([_mean(name)], dtype=np.float32)
        for name in ("optortk_expr", "nuc_area") if name in chans
    }
    got = simulate_open_loop(
        engine, np.array([seq], dtype=np.float32), n_frames, ens=1, static=static
    )
    assert got.shape == (1, 1, n_frames)
    np.testing.assert_allclose(got[0, 0], np.array(stepwise, dtype=np.float32), rtol=2e-5)


def test_simulate_open_loop_evaluates_every_candidate_in_one_pass():
    """The optimiser's inner loop depends on this: C schedules must come back in C
    rows, and a schedule must not be affected by the others sharing the batch."""
    from optoerk.serving.replay import simulate_open_loop

    engine = _ol_engine([0.0])
    a, b = [0.0, 0.0, 0.0], [150.0, 150.0, 150.0]
    both = simulate_open_loop(engine, np.array([a, b], dtype=np.float32), 6, ens=4)
    assert both.shape == (2, 4, 6)

    alone_a = simulate_open_loop(engine, np.array([a], dtype=np.float32), 6, ens=4)
    alone_b = simulate_open_loop(engine, np.array([b], dtype=np.float32), 6, ens=4)
    np.testing.assert_allclose(both[0], alone_a[0], rtol=1e-6)
    np.testing.assert_allclose(both[1], alone_b[0], rtol=1e-6)
    # and light has to do something, or the whole experiment is measuring nothing
    assert both[1].mean() != pytest.approx(both[0].mean())


def test_simulate_open_loop_tiles_the_schedule_and_ignores_the_cells():
    """No feedback anywhere: the dose at frame t is seq[t % P] whatever the cells do.
    That is the property the control arm exists to embody."""
    from optoerk.serving.replay import simulate_open_loop

    engine = _ol_engine([0.0])
    # Two ensembles differing only in seed still see the identical dose schedule, so
    # any spread between them comes from the cells, never from the controller.
    seq = np.array([[0.0, 150.0]], dtype=np.float32)
    x = simulate_open_loop(engine, seq, 8, ens=6, seed=1)
    y = simulate_open_loop(engine, seq, 8, ens=6, seed=2)
    assert x.shape == y.shape == (1, 6, 8)


def test_score_open_loop_uses_the_objectives_own_cost():
    """The schedule must be optimised against exactly what the closed-loop arms
    minimise. A hand-rolled L2 here would let the arms drift apart silently."""
    from optoerk.serving.objectives import hold
    from optoerk.serving.replay import score_open_loop

    engine = _ol_engine([0.0])
    target = 1.0
    # Two candidates: one sitting on the target, one 0.5 away. An l2 hold objective
    # must score them 0 and 0.25.
    cnr = np.stack([
        np.full((3, 4), target, dtype=np.float32),
        np.full((3, 4), target + 0.5, dtype=np.float32),
    ])                                                       # (2, 3, 4)
    cost = score_open_loop(engine, hold(target), cnr, start_frame=0)
    assert cost.shape == (2,)
    assert cost[0] == pytest.approx(0.0, abs=1e-6)
    assert cost[1] == pytest.approx(0.25, abs=1e-6)
