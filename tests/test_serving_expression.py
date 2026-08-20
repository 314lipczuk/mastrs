"""Live optoRTK expression: does the online reconstruction match the offline feature?

The model was trained on ``preprocessing.add_optortk_expression``. If serving feeds
something that is only *approximately* that, the fifth channel is a different
covariate than the one the weights learned, and nothing anywhere would say so. The
first test here is therefore the load-bearing one: same cells, same C0, offline
pipeline vs online service, ranks must agree exactly.
"""
import json

import numpy as np
import pandas as pd
import pytest

from optoerk.data.preprocessing import add_optortk_expression
from optoerk.serving.config import ServerConfig
from optoerk.serving.expression import ExpressionCohort
from optoerk.serving.features import extract_optortk_value
from optoerk.serving.service import InferenceService
from optoerk.serving.state import CellState

BASELINE_FRAMES = 10


OPTOCHECK_FRAME = 0


def _c0_grid(n_fovs=3, n_cells=8, n_frames=14, seed=0):
    """One mCitrine value per cell, as the optocheck actually delivers it.

    Real acquisitions image the optoRTK channel in its own short reference
    acquisition once or twice per experiment, so the value exists on those frames
    and nowhere else — not on every frame.
    """
    rng = np.random.default_rng(seed)
    levels = {}
    for fov in range(n_fovs):
        for p in range(n_cells):
            levels[(fov, p)] = float(rng.uniform(400.0, 2000.0))
    return levels, levels


def _offline_ranks(levels, n_fovs, n_cells, n_frames):
    """What add_optortk_expression produces for the same cells. The cohort is the
    imaging SESSION — every FOV pooled — so uid is (fov, particle) but the cohort
    column is constant."""
    rows = [
        {"uid": f"{fov}_{p}", "frame": t,
         "original_experiment_name": "session", "mcitrine": v}
        for (fov, p), v in levels.items()
        for t in range(n_frames)
    ]
    df = add_optortk_expression(pd.DataFrame(rows))
    return df.groupby("uid")["optortk_expr"].first().to_dict(), df


def _payload(fov, t, n_cells, levels):
    """A frame. The optoRTK value rides along only on the optocheck frame."""
    cells = []
    for p in range(n_cells):
        cell = {"particle": p, "x": 100.0 * p, "y": 50.0 * fov, "cnr_median": 0.9}
        if t == OPTOCHECK_FRAME:
            cell["ref_mean_intensity"] = levels[(fov, p)]
        cells.append(cell)
    return {"fov": fov, "timestep": t, "cells": cells}


def _cfg(**kw):
    kw.setdefault("dark_baseline", False)
    kw.setdefault("warmup", False)
    kw.setdefault("gpu_sample_interval_s", 0)
    kw.setdefault("live_optortk_expr", True)
    kw.setdefault("optortk_baseline_frames", BASELINE_FRAMES)
    kw.setdefault("optortk_cohort_frames", BASELINE_FRAMES)
    return ServerConfig(**kw)


# ---------------------------------------------------------------------------
# the equivalence that matters
# ---------------------------------------------------------------------------


def test_online_ranks_reproduce_the_offline_feature_exactly(tmp_path):
    """Same cells, same C0 -> byte-identical percentile ranks.

    This is the whole point. The offline pipeline ranks per-cell C0 medians over
    `frame < baseline_frames` across the pooled session; the service has to arrive
    at the same numbers from a stream, having seen each frame once.
    """
    n_fovs, n_cells, n_frames = 3, 8, 14
    _levels, grid = _c0_grid(n_fovs, n_cells, n_frames)
    offline, _df = _offline_ranks(grid, n_fovs, n_cells, n_frames)

    log = tmp_path / "run.jsonl"
    svc = InferenceService(_cfg(predict_log_path=str(log)))
    try:
        for t in range(n_frames):
            for fov in range(n_fovs):
                svc.predict(_payload(fov, t, n_cells, grid))
    finally:
        svc.close()

    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    # the last frame's records carry every cell's settled rank
    last = [r for r in recs if r.get("event") == "predict"][-n_fovs:]
    online = {}
    for r in last:
        for c in r["cells"]:
            online[f"{r['fov']}_{c['particle']}"] = c["optortk_expr"]

    assert set(online) == set(offline)
    # Exact at the offline feature's own dtype: add_optortk_expression casts to
    # float32, so that is the precision the model was trained at. The online value
    # is float64 and agrees to every bit float32 can represent.
    for uid, want in offline.items():
        assert np.float32(online[uid]) == np.float32(want), uid

    # ...and they are real ranks, not all the same constant
    assert len(set(online.values())) == n_fovs * n_cells
    assert min(online.values()) == pytest.approx(1.0 / (n_fovs * n_cells))
    assert max(online.values()) == pytest.approx(1.0)


def test_cohort_pools_across_fovs_not_within_one():
    """Offline the cohort column is the imaging session, so a cell competes with
    every field's cells. Ranking per FOV would make each field its own population
    and silently change what the feature means."""
    n_fovs, n_cells, n_frames = 3, 8, 12
    _levels, grid = _c0_grid(n_fovs, n_cells, n_frames)
    svc = InferenceService(_cfg())
    try:
        for t in range(n_frames):
            for fov in range(n_fovs):
                svc.predict(_payload(fov, t, n_cells, grid))
        assert svc.cohort.n_cells == n_fovs * n_cells
    finally:
        svc.close()


# ---------------------------------------------------------------------------
# extracting the measurement
# ---------------------------------------------------------------------------


def test_accepts_either_upstream_spelling():
    """faro's RefFE writes `ref_mean_intensity`; the older Cedric-side export of
    the same quantity is `optocheck_mean_intensity`."""
    assert extract_optortk_value({"ref_mean_intensity": 812.0}) == pytest.approx(812.0)
    assert extract_optortk_value({"optocheck_mean_intensity": 640.0}) == pytest.approx(640.0)
    # preference order when both somehow appear
    assert extract_optortk_value(
        {"ref_mean_intensity": 1.0, "optocheck_mean_intensity": 2.0}
    ) == pytest.approx(1.0)


def test_never_reconstructs_the_measurement_from_the_C0_channels():
    """mCitrine is its own fluorescence channel. Whole-cell C0 (miRFP) was the old
    surrogate: Spearman 0.60-0.71 against the real thing, 27-30% of cells in the
    wrong half. It must never be silently substituted."""
    assert extract_optortk_value({}) is None
    assert extract_optortk_value(
        {"mean_intensity_C0_nuc": 400.0, "mean_intensity_C0_ring": 600.0}
    ) is None
    assert extract_optortk_value(
        {"mean_intensity_C1_nuc": 400.0, "mean_intensity_C1_ring": 600.0}
    ) is None
    # non-finite / non-positive are not measurements
    assert extract_optortk_value({"ref_mean_intensity": float("nan")}) is None
    assert extract_optortk_value({"ref_mean_intensity": 0.0}) is None


def test_a_frame_without_the_measurement_is_normal_not_an_error():
    """The optocheck is its own short acquisition, so MOST frames carry nothing.
    Treating that as missing data would abort every real run on frame 2."""
    _levels, grid = _c0_grid(1, 4, 6)
    svc = InferenceService(_cfg(optortk_cohort_frames=4))
    try:
        for t in range(6):
            out = svc.predict(_payload(0, t, 4, grid))   # value only at t=0
            assert len(out["exposures"]) == 4
        assert svc.cohort.sealed and svc.cohort.n_cells == 4
    finally:
        svc.close()


# ---------------------------------------------------------------------------
# the cohort's own contract
# ---------------------------------------------------------------------------


def _state(particle):
    s = CellState()
    s.particle = particle
    return s


def test_nobody_is_ranked_before_the_cohort_seals():
    """The streaming-obvious design ranks each cell when its own window fills,
    which scores the first finisher 1.0 against a cohort of one. Ranking must wait
    for the population it is a rank within."""
    co = ExpressionCohort(baseline_frames=4)
    states = {p: _state(p) for p in range(3)}
    for t in range(4):
        for p, st in states.items():
            assert co.observe(0, st, 100.0 * (p + 1), t) is None, f"ranked at t={t}"
    assert not co.sealed, "the seal is lazy — it happens on the first frame past it"

    # the first frame at/after cohort_frames seals, and everyone is ranked at once
    ranks = {p: co.observe(0, st, 100.0 * (p + 1), 4) for p, st in states.items()}
    assert co.sealed and co.n_cells == 3
    assert ranks == {0: pytest.approx(1 / 3), 1: pytest.approx(2 / 3),
                     2: pytest.approx(1.0)}


def test_ranks_freeze_and_late_cells_do_not_enter_the_cohort():
    """A percentile that drifts as cells appear is a different input distribution
    than the static feature the model was trained on."""
    co = ExpressionCohort(baseline_frames=3)
    early = {p: _state(p) for p in range(3)}
    for t in range(3):
        for p, st in early.items():
            co.observe(0, st, 100.0 * (p + 1), t)
    co.observe(0, early[0], 100.0, 3)
    first = {p: co.observe(0, st, 100.0 * (p + 1), 4) for p, st in early.items()}
    assert co.n_cells == 3

    # a cell born late gets ranked against the sealed cohort...
    late = _state(99)
    for t in range(5, 9):
        co.observe(0, late, 250.0, t)
    # ...and does not change it, nor anyone else's rank
    assert co.n_cells == 3
    after = {p: co.observe(0, st, 100.0 * (p + 1), 9) for p, st in early.items()}
    assert after == first


def test_rank_matches_pandas_pct_rank_including_ties():
    co = ExpressionCohort(baseline_frames=1)
    for p, v in enumerate([10.0, 20.0, 20.0, 40.0]):
        co.observe(0, _state(p), v, 0)
    co.seal()
    want = pd.Series([10.0, 20.0, 20.0, 40.0]).rank(pct=True).tolist()
    got = [co.rank_of(v) for v in (10.0, 20.0, 20.0, 40.0)]
    assert got == pytest.approx(want)


def test_replay_determinism():
    """Same payload sequence -> same ranks. No wall-clock, no dict order, no RNG,
    so a recorded run re-drives to the identical feature."""
    n_fovs, n_cells, n_frames = 2, 6, 12
    _levels, grid = _c0_grid(n_fovs, n_cells, n_frames)

    def run():
        svc = InferenceService(_cfg())
        try:
            out = {}
            for t in range(n_frames):
                for fov in range(n_fovs):
                    svc.predict(_payload(fov, t, n_cells, grid))
            for fov in range(n_fovs):
                for p in range(n_cells):
                    out[(fov, p)] = svc.store.get(fov, p).optortk_rank
            return out
        finally:
            svc.close()

    assert run() == run()


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


def test_a_cohort_that_seals_too_thin_keeps_serving_but_marks_every_frame():
    """The structural failure: the window closed and (almost) no cell supplied a
    value, so there is nothing to rank against and every cell falls back to the
    middle of the percentile scale.

    It must NOT stop the run — a false alarm would kill a good experiment, and a
    run on the neutral value is still a run. What it must do is make itself
    impossible to miss afterwards, because the silent version looks exactly like
    success. Previously this raised exactly once, disarmed itself, and let the
    rest of the run proceed unmarked.
    """
    svc = InferenceService(_cfg(optortk_cohort_frames=2))
    try:
        for t in range(5):
            out = svc.predict({
                "fov": 0, "timestep": t,
                "cells": [{"particle": p, "x": 1.0 * p, "y": 1.0, "cnr_median": 0.9}
                          for p in range(3)],
            })
            assert len(out["exposures"]) == 3, "the run must keep being steered"
        assert svc.cohort.sealed and svc.cohort.n_cells == 0
        assert svc.optortk_degraded is True, "the condition has to be sticky"
    finally:
        svc.close()


def test_degraded_cohort_is_stamped_on_every_predict_record(tmp_path):
    """One startup message is not a record. The log has to carry it frame by
    frame, or an analysis six weeks later cannot tell this run from a healthy one."""
    log = tmp_path / "run.jsonl"
    svc = InferenceService(_cfg(optortk_cohort_frames=2, predict_log_path=str(log)))
    try:
        for t in range(5):
            svc.predict({
                "fov": 0, "timestep": t,
                "cells": [{"particle": p, "x": 1.0 * p, "y": 1.0, "cnr_median": 0.9}
                          for p in range(3)],
            })
    finally:
        svc.close()

    recs = [json.loads(line) for line in log.read_text().splitlines()]
    preds = [r for r in recs if r.get("event") == "predict"]
    assert preds, "no predict records were written"
    assert preds[-1]["optortk_degraded"] is True
    # and the cohort's own size/spread is recorded once, where it sealed
    sealed = [r for r in recs if r.get("event") == "optortk_cohort"]
    assert len(sealed) == 1
    assert sealed[0]["spread"]["n"] == 0
    assert sealed[0]["degraded"] is True
    # every cell says where its number came from
    assert {c["optortk_source"] for c in preds[-1]["cells"]} == {"fallback"}


def test_one_cell_without_a_measurement_does_not_kill_the_run():
    """A cell the optocheck missed still has to be steered. It goes neutral; its
    neighbours are unaffected."""
    _levels, grid = _c0_grid(1, 4, 4)
    svc = InferenceService(_cfg(optortk_cohort_frames=2))
    try:
        for t in range(4):
            payload = _payload(0, t, 4, grid)
            if t == OPTOCHECK_FRAME:
                payload["cells"][2].pop("ref_mean_intensity")
            out = svc.predict(payload)
            assert len(out["exposures"]) == 4
        assert svc.cohort.n_cells == 3               # the missed cell is not in it
        assert svc.store.get(0, 2).optortk_rank is None
        assert svc.store.get(0, 1).optortk_rank is not None
    finally:
        svc.close()


def test_live_and_override_together_is_refused():
    with pytest.raises(ValueError, match="pick one"):
        InferenceService(_cfg(override_optortk_expr=True, optortk_expr_value=0.5))


def test_live_is_the_default_since_2026_08_14():
    """The default is now `live`, because that is what every real run since v13
    passed explicitly. A default nobody wants is not a safe default: it makes the
    flag mandatory ceremony, and OFF is the silently-degrading state — every cell
    gets the neutral median rank and the run looks fine."""
    cfg = ServerConfig(dark_baseline=False, warmup=False, gpu_sample_interval_s=0)
    assert cfg.live_optortk_expr is True
    svc = InferenceService(cfg)
    try:
        assert svc.optortk_expr_mode()["mode"] == "live"
    finally:
        svc.close()


def test_the_mode_is_never_decided_by_payload_content():
    """The guarantee the old default was standing in for, kept explicitly: which
    optoRTK regime a run is in comes from the config alone. A payload that happens
    to carry C0 must never promote a `neutral` run to `live` behind the operator."""
    cfg = ServerConfig(live_optortk_expr=False, dark_baseline=False,
                       warmup=False, gpu_sample_interval_s=0)
    svc = InferenceService(cfg)
    try:
        assert svc.cohort is None
        assert svc.optortk_expr_mode()["mode"] == "neutral"
        svc.predict(_payload(0, 0, 2, _c0_grid(1, 2, 2)[1]))
        assert svc.cohort is None, "a payload promoted the run to live"
        assert svc.optortk_expr_mode()["mode"] == "neutral"
    finally:
        svc.close()


def test_mode_is_recorded_in_the_startup_log(tmp_path):
    log = tmp_path / "run.jsonl"
    svc = InferenceService(_cfg(predict_log_path=str(log)))
    try:
        svc.predict(_payload(0, 0, 2, _c0_grid(1, 2, 2)[1]))
    finally:
        svc.close()
    startup = json.loads(log.read_text().splitlines()[0])
    assert startup["event"] == "startup"
    assert startup["optortk_expr"]["mode"] == "live"
    assert startup["optortk_expr"]["baseline_frames"] == BASELINE_FRAMES


# ---------------------------------------------------------------------------
# multi-FOV sealing
# ---------------------------------------------------------------------------


def test_the_seal_waits_for_every_reporting_fov():
    """Sealing on the first payload past the window ranks late fields against a
    partial population.

    faro interleaves FOVs within an acquisition cycle, so the skew is normally
    under one frame — but "normally" is not a guarantee and the failure is silent:
    the late field's cells simply get ranks drawn from a fraction of the session.
    """
    n_fovs, n_cells, n_frames = 3, 8, 12
    _levels, grid = _c0_grid(n_fovs, n_cells, n_frames)
    svc = InferenceService(_cfg(optortk_cohort_frames=6))
    try:
        # interleaved, as faro actually drives it: all FOVs at t, then t+1
        for t in range(n_frames):
            for fov in range(n_fovs):
                svc.predict(_payload(fov, t, n_cells, grid))
        assert svc.cohort.sealed
        assert svc.cohort.n_cells == n_fovs * n_cells, \
            "every FOV's cells must be in the cohort, not just the fastest field's"
    finally:
        svc.close()


def test_one_fov_running_ahead_does_not_close_the_cohort_early(tmp_path):
    """A field a few frames ahead of the others must not seal on its own clock.

    This needs the POLICY to declare the run's fields: a FOV that has not reported
    yet is otherwise indistinguishable from one that does not exist, and the cohort
    would rightly seal on the fields it has seen.
    """
    from optoerk.serving.policy import load_policy_file

    n_cells = 8
    _levels, grid = _c0_grid(2, n_cells, 20)
    pol = tmp_path / "p.toml"
    pol.write_text(
        '[default]\nobjective = { type = "hold", target_cnr = 1.0 }\n'
        "[fov.0]\narm = 1\n[fov.1]\narm = 1\n"
    )
    svc = InferenceService(_cfg(optortk_cohort_frames=5),
                           policy_file=load_policy_file(pol))
    try:
        for t in range(7):                     # fov 0 runs to t=6, past the window
            svc.predict(_payload(0, t, n_cells, grid))
        assert not svc.cohort.sealed, "fov 1 has not reached the window yet"
        for t in range(7):                     # fov 1 catches up
            svc.predict(_payload(1, t, n_cells, grid))
        assert svc.cohort.sealed
        assert svc.cohort.n_cells == 2 * n_cells
    finally:
        svc.close()


def test_a_stalled_fov_cannot_hold_the_cohort_open_forever():
    """The backstop. Without it, one withdrawn field would leave every cell on the
    neutral value for the whole run — worse than sealing without it."""
    n_cells = 8
    _levels, grid = _c0_grid(2, n_cells, 40)
    svc = InferenceService(_cfg(optortk_cohort_frames=5))
    try:
        svc.predict(_payload(1, 0, n_cells, grid))      # fov 1 reports once, then stalls
        for t in range(12):                             # fov 0 runs to 2x the window
            svc.predict(_payload(0, t, n_cells, grid))
        assert svc.cohort.sealed, "the backstop must fire at 2x cohort_frames"
    finally:
        svc.close()


# ---------------------------------------------------------------------------
# the window is measured in frames observed, not in faro's timestep numbers
# ---------------------------------------------------------------------------


def test_the_cohort_window_never_reads_faro_timesteps_as_elapsed_frames():
    """The v12 failure, in one assertion.

    An acquisition that runs an optocheck phase first hands the server its first
    controlled frame already numbered `timestep=40`. A window expressed in
    absolute timesteps is long past before a single measurement has been folded
    in, so the cohort sealed empty on frame one, `rank_of` returned 1.0 for
    everybody against a population of nothing, and the run continued 40 h on a
    constant. The backstop (`>= 2 * cohort_frames`) fired first, so it also
    bypassed the guard that waits for every declared field.

    The cohort must be indifferent to where faro starts numbering.
    """
    cohort = ExpressionCohort(
        baseline_frames=1, cohort_frames=2, expected_fovs=frozenset(range(3))
    )
    assert cohort.ready_to_seal(0, 40) is False, (
        "sealed on its own first frame — the window is being read off faro's "
        "timestep instead of frames this cohort has observed"
    )
    # ... and it still seals normally once the fields have actually been observed
    for offset in range(3):
        for fov in range(3):
            ready = cohort.ready_to_seal(fov, 40 + offset)
    assert ready is True


@pytest.mark.parametrize("t0", [0, 40, 1000])
def test_ranks_do_not_depend_on_where_faro_starts_numbering(t0):
    """End-to-end form of the same property.

    The identical experiment, offset only by how many timesteps earlier phases
    consumed, must produce identical ranks. Every test here used to start at
    timestep 0, which is exactly the case that works by accident.
    """
    n_fovs, n_cells, n_frames = 3, 8, 14
    levels, _ = _c0_grid(n_fovs, n_cells, n_frames)

    def _run(start):
        svc = InferenceService(_cfg())
        try:
            for i in range(n_frames):
                for fov in range(n_fovs):
                    cells = []
                    for p in range(n_cells):
                        cell = {"particle": p, "x": 100.0 * p, "y": 50.0 * fov,
                                "cnr_median": 0.9}
                        if i == 0:      # the optocheck value rides on frame one
                            cell["ref_mean_intensity"] = levels[(fov, p)]
                        cells.append(cell)
                    svc.predict({"fov": fov, "timestep": start + i, "cells": cells})
            ranks = {(fov, p): svc.store.get(fov, p).optortk_rank
                     for fov in range(n_fovs) for p in range(n_cells)}
            return ranks, svc.cohort.n_cells, svc.optortk_degraded
        finally:
            svc.close()

    ranks, n_cohort, degraded = _run(t0)
    base_ranks, base_n, _ = _run(0)

    assert n_cohort == base_n == n_fovs * n_cells, "every measured cell must be in it"
    assert degraded is False
    assert ranks == base_ranks
    assert len(set(ranks.values())) > 1, (
        "all cells scored the same — the cohort was empty when it sealed"
    )


# ---------------------------------------------------------------------------
# origin-shift equivalence: the whole service, twice, at two numberings
# ---------------------------------------------------------------------------


OSC_POLICY = """
[default]
control_horizon = 4
objective = { type = "oscillation", low = 0.87, high = 1.17, t_low_min = 2, \
t_rise_min = 2, t_high_min = 2, t_fall_min = 2, settle_periods = 1, n_phase_groups = 2 }
"""


def _drive(svc, start, n_frames, n_fovs, n_cells, levels):
    """One synthetic run, numbered from `start`."""
    out = []
    for i in range(n_frames):
        for fov in range(n_fovs):
            cells = []
            for p in range(n_cells):
                cell = {"particle": p, "x": 100.0 * p, "y": 50.0 * fov,
                        "cnr_median": 0.7 + 0.01 * ((i + p) % 11)}
                if i == 0:
                    cell["ref_mean_intensity"] = levels[(fov, p)]
                cells.append(cell)
            out.append(svc.predict({"fov": fov, "timestep": start + i, "cells": cells}))
    return out


@pytest.mark.parametrize("offset", [40, 1000])
def test_the_whole_run_is_invariant_to_where_faro_starts_numbering(offset, tmp_path):
    """The regression test for the v12 class of failure, at service level.

    The same experiment, differing only in how many frames earlier acquisition
    phases consumed, must produce the same doses, the same reference waveform and
    the same expression ranks. It exercises in one assertion every clock the
    server keeps: the objective's schedule, the expression cohort's window, the
    dark-baseline window and state eviction.

    Under the old code this failed twice over — the cohort sealed empty on frame
    one, and the waveform was already `offset` minutes into its settle before the
    first controlled frame.

    Any new clock that reads `timestep` as elapsed time will fail here too, which
    is the point: the assertion is about the property, not about the two bugs that
    motivated it.
    """
    # Comfortably above optortk_min_cohort_cells, so this exercises the healthy
    # path — a degraded cohort would make the rank comparison below vacuous.
    n_fovs, n_cells, n_frames = 2, 14, 16
    levels, _ = _c0_grid(n_fovs, n_cells, n_frames)
    pol = tmp_path / "policy.toml"
    pol.write_text(OSC_POLICY)

    def _run(start):
        log = tmp_path / f"run_{start}.jsonl"
        svc = InferenceService(_cfg(policy_file=str(pol), predict_log_path=str(log),
                                    optortk_cohort_frames=2))
        try:
            raw = _drive(svc, start, n_frames, n_fovs, n_cells, levels)
            # Keyed on frames-since-start: the response echoes faro's timestep
            # back, which is the one thing that is SUPPOSED to differ.
            exposures = [(r["fov"], r["timestep"] - start, r["exposures"]) for r in raw]
            ranks = {(f, p): svc.store.get(f, p).optortk_rank
                     for f in range(n_fovs) for p in range(n_cells)}
        finally:
            svc.close()
        recs = [json.loads(ln) for ln in log.read_text().splitlines()]
        preds = [r for r in recs if r.get("event") == "predict"]
        # keyed on frames-since-start, so the two runs are comparable at all
        waveform = [
            (r["fov"], r["timestep"] - start,
             [(c["particle"], c.get("r_t"), c.get("segment")) for c in r["cells"]])
            for r in preds
        ]
        return exposures, ranks, waveform, svc.optortk_degraded

    a_exp, a_ranks, a_wave, a_deg = _run(0)
    b_exp, b_ranks, b_wave, b_deg = _run(offset)

    assert a_deg is False and b_deg is False
    assert a_ranks == b_ranks, "expression ranks moved with faro's numbering"
    assert a_wave == b_wave, "the reference waveform moved with faro's numbering"
    assert a_exp == b_exp, "commanded doses moved with faro's numbering"

    # the run has to actually exercise a waveform, or the comparison is vacuous
    segments = {s for _f, _t, cs in a_wave for _p, _r, s in cs}
    assert len(segments) > 1, f"only saw segments {segments} — nothing was varying"
