"""Replay: a run recorded by the service must re-drive the service identically.

Self-contained — the test records its own log with the stub engine, so there is
no mount, no checkpoint, and (being CPU-only and deterministic) an exact 1.0
match is the correct bar here. Replaying a *CUDA* run on CPU is a different
matter; see the note in ``optoerk/serving/README.md``.
"""
import polars as pl

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
