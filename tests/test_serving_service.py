"""Policy routing, config parsing and the service contract.

Runs entirely on the stub engine plus a fake checkpoint, so no cluster mount and
no trained model are needed.
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from optoerk.serving.config import ServerConfig
from optoerk.serving.objectives import PolicyViolation
from optoerk.serving.policy import PolicyRouter, load_policy_file
from optoerk.serving.runtime import RealModelEngine, StubEngine, load_model
from optoerk.serving.service import InferenceService

POLICY_TOML = """
[default]
objective = { type = "hold", target_cnr = 1.5 }

[fov.1]
objective = { type = "hold", target_cnr = 2.5 }

[fov.2]
objective = { type = "gated", target_cnr = 1.8, after_t = 1000 }
control_horizon = 3
"""


def _cfg(**kw):
    # dark_baseline off: these tests are about routing, not the baseline window,
    # and the stub must be free to command a nonzero dose on frame 0.
    kw.setdefault("dark_baseline", False)
    kw.setdefault("warmup", False)
    return ServerConfig(**kw)


def _cells(n=2, cnr=0.5):
    return [
        {"particle": i, "x": 100.0 * i, "y": 50.0, "cnr_median": cnr}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# policy file parsing
# ---------------------------------------------------------------------------


def test_policy_file_toml_roundtrip_and_inheritance(tmp_path):
    p = tmp_path / "policies.toml"
    p.write_text(POLICY_TOML)
    pf = load_policy_file(p)

    assert pf.default.objective["target_cnr"] == 1.5
    assert set(pf.fov) == {1, 2}
    assert pf.fov[1].objective["target_cnr"] == 2.5
    # fov 2 sets its own horizon; fov 1 inherits the (unset) default
    assert pf.fov[2].control_horizon == 3
    assert pf.fov[1].control_horizon is None


def test_policy_file_json(tmp_path):
    p = tmp_path / "policies.json"
    p.write_text(json.dumps({
        "default": {"checkpoint": "somewhere", "objective": {"type": "hold", "target_cnr": 1.0}},
        "fov": {"3": {"objective": {"type": "hold", "target_cnr": 9.0}}},
    }))
    pf = load_policy_file(p)
    # fov 3 inherits the default's checkpoint but overrides the objective
    assert pf.fov[3].checkpoint == "somewhere"
    assert pf.fov[3].objective["target_cnr"] == 9.0


def test_policy_file_rejects_typos(tmp_path):
    """A silently-ignored typo means running the wrong experiment for 12 hours."""
    p = tmp_path / "policies.toml"
    p.write_text('[default]\nobjektive = { type = "hold" }\n')
    with pytest.raises(Exception):
        load_policy_file(p)

    p2 = tmp_path / "policies.yaml"
    p2.write_text("nope")
    with pytest.raises(ValueError, match="must be .toml or .json"):
        load_policy_file(p2)


def test_policy_file_carries_an_explicit_dose_ladder(tmp_path):
    """The ladder is an experiment parameter, not a code constant — and a rebinned
    ladder like [0, 20, 45, 85, 150] is not expressible as a linspace."""
    p = tmp_path / "policies.toml"
    p.write_text(
        '[default]\nlevels_ms = [0, 20, 45, 85, 150]\n'
        'objective = { type = "hold", target_cnr = 1.0 }\n'
        '[fov.1]\ncontroller = { type = "sequence_mpc" }\n'
    )
    pf = load_policy_file(p)
    assert pf.default.levels_ms == [0, 20, 45, 85, 150]
    assert pf.fov[1].levels_ms == [0, 20, 45, 85, 150], "inherited from [default]"

    router = PolicyRouter(_cfg(), pf)
    assert router.info_for(1)["levels_ms"] == [0.0, 20.0, 45.0, 85.0, 150.0]


def test_policy_file_defaults_to_the_config_ladder(tmp_path):
    p = tmp_path / "policies.toml"
    p.write_text('[default]\nobjective = { type = "hold", target_cnr = 1.0 }\n')
    router = PolicyRouter(_cfg(), load_policy_file(p))
    assert router.default_info["levels_ms"] == [0.0, 200.0, 400.0, 600.0, 800.0]


def test_unresolved_placeholders_refuse_to_serve(tmp_path):
    """A policy still carrying working assumptions must not run a real experiment."""
    p = tmp_path / "policies.toml"
    p.write_text(
        'placeholders_resolved = false\n'
        '[default]\nobjective = { type = "hold", target_cnr = 1.0 }\n'
    )
    pf = load_policy_file(p)
    assert pf.placeholders_resolved is False
    with pytest.raises(PolicyViolation, match="placeholders_resolved"):
        PolicyRouter(_cfg(), pf)


def test_placeholders_resolved_defaults_to_true(tmp_path):
    """Existing policy files predate the flag and must keep working."""
    p = tmp_path / "policies.toml"
    p.write_text('[default]\nobjective = { type = "hold", target_cnr = 1.0 }\n')
    assert load_policy_file(p).placeholders_resolved is True


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_router_routes_per_fov_and_falls_back_to_default(tmp_path):
    p = tmp_path / "policies.toml"
    p.write_text(POLICY_TOML)
    router = PolicyRouter(_cfg(), load_policy_file(p))

    assert router.engine_for(1) is not router.default_engine
    assert router.engine_for(2) is not router.engine_for(1)
    assert router.engine_for(99) is router.default_engine  # no override
    assert router.info_for(1)["objective"]["target_cnr"] == 2.5


def test_invalid_fov_policy_degrades_that_fov_only(tmp_path):
    """One bad objective must not take the server down for the other FOVs."""
    p = tmp_path / "policies.toml"
    p.write_text(
        '[default]\nobjective = { type = "hold", target_cnr = 1.5 }\n'
        '[fov.1]\nobjective = { type = "does_not_exist" }\n'
    )
    router = PolicyRouter(_cfg(), load_policy_file(p))
    assert isinstance(router.engine_for(1), StubEngine)
    assert router.info_for(1)["model_loaded"] is False
    assert router.default_info["objective"]["target_cnr"] == 1.5


def test_degraded_fov_still_records_what_it_was_asked_to_run(tmp_path):
    """A stub-degraded FOV used to log `{"policy": "stub"}` and nothing else, so
    the hold run's controller assignment had to be recovered from the .toml
    afterwards. The requested spec makes the startup record self-contained."""
    p = tmp_path / "policies.toml"
    p.write_text(
        '[default]\nobjective = { type = "hold", target_cnr = 1.5 }\n'
        '[fov.1]\nobjective = { type = "does_not_exist" }\n'
        'controller = { type = "sequence_mpc", n_samples = 256 }\n'
    )
    router = PolicyRouter(_cfg(), load_policy_file(p))
    info = router.info_for(1)
    assert info["model_loaded"] is False
    assert info["label"] == "fov 1"
    assert info["requested"]["objective"] == {"type": "does_not_exist"}
    assert info["requested"]["controller"]["n_samples"] == 256


def test_arm_overrides_compose_onto_the_shared_reference(tmp_path):
    """An arm may vary the kernel and the regularizers; it may NOT vary what is
    being tracked. Composing the arm-varying pieces onto an inherited objective
    makes that structural rather than a property of twelve blocks staying in sync.
    """
    p = tmp_path / "policies.toml"
    p.write_text(
        "[default]\ncontrol_horizon = 30\n"
        'objective = { type = "oscillation", low = 0.85, high = 1.15, '
        "t_low_min = 8, t_rise_min = 2, t_high_min = 15, t_fall_min = 15, "
        "n_phase_groups = 4 }\n"
        '[fov.1]\nkernel = "l2"\nlambda_move = 0.0\n'
        '[fov.2]\nkernel = "l2"\nlambda_move = 0.6\n'
        '[fov.3]\nkernel = { type = "band", half_width = 0.05 }\nlambda_move = 0.6\n'
    )
    router = PolicyRouter(_cfg(), load_policy_file(p))
    objs = {f: router.info_for(f)["objective"] for f in (1, 2, 3)}

    # the reference is byte-identical across every arm
    refs = [o["reference"] for o in objs.values()]
    assert refs[0] == refs[1] == refs[2]
    assert refs[0]["period_min"] == 40.0

    # and the arms differ in exactly one respect each, in the nested order
    assert objs[1]["kernel"] == {"type": "l2"}
    assert objs[1]["regularizers"] == []
    assert objs[2]["kernel"] == {"type": "l2"}
    assert objs[2]["regularizers"] == [
        {"type": "move_penalty", "lambda_move": 0.6}
    ]
    assert objs[3]["kernel"] == {"type": "band", "half_width": 0.05}
    assert objs[3]["regularizers"] == objs[2]["regularizers"]


def test_shipped_12fov_policy_encodes_the_nested_arms():
    """The real experiment definition, checked as a unit: interleaved arms, one
    change per adjacent arm, and one shared reference / ladder / horizon."""
    pf = load_policy_file(Path(__file__).parent.parent / "policies" / "policy_12fov_osc.toml")
    assert pf.placeholders_resolved is False, "must not ship ready-to-run"
    assert set(pf.fov) == set(range(12))

    arms = {
        f: (
            pf.fov[f].controller["type"],
            pf.fov[f].kernel if isinstance(pf.fov[f].kernel, str)
            else pf.fov[f].kernel["type"],
            pf.fov[f].lambda_move,
        )
        for f in range(12)
    }
    expected = {
        0: ("constant_dose", "l2", 0.0),
        1: ("sequence_mpc", "l2", 0.0),
        2: ("sequence_mpc", "l2", 0.6),
        3: ("sequence_mpc", "band", 0.6),
    }
    for f in range(12):
        assert arms[f] == expected[f % 4], f"fov {f} is not arm {f % 4 + 1}"

    # adjacent arms differ in exactly one respect
    for a, b in zip([expected[i] for i in range(3)], [expected[i] for i in range(1, 4)]):
        assert sum(x != y for x, y in zip(a, b)) == 1

    # reference, ladder and horizon are shared, not per-arm
    for f in range(12):
        assert pf.fov[f].objective == pf.default.objective
        assert pf.fov[f].levels_ms == [0, 20, 45, 85, 150]
        assert pf.fov[f].control_horizon == 30


def test_a_period_longer_than_the_horizon_still_serves(tmp_path):
    """The period/horizon bound is gone: the plan re-solves every frame, so a
    transition enters the window H frames ahead however long the period is."""
    p = tmp_path / "policies.toml"
    p.write_text(
        "[default]\ncontrol_horizon = 10\n"
        'objective = { type = "oscillation", low = 0.85, high = 1.15, '
        "t_low_min = 8, t_rise_min = 2, t_high_min = 15, t_fall_min = 15 }\n"
    )
    router = PolicyRouter(_cfg(), load_policy_file(p))
    assert router.default_info["objective"]["reference"]["period_min"] == 40.0


def test_per_fov_objective_reaches_the_engine(tmp_path):
    """fov 2's gate (after_t=1000) suppresses stimulation there while the default
    FOV still doses — end-to-end proof that routing selects the objective."""
    p = tmp_path / "policies.toml"
    p.write_text(POLICY_TOML)
    svc = InferenceService(_cfg(policy_file=str(p)))

    default_fov = svc.predict({"fov": 0, "timestep": 1, "cells": _cells()})
    gated_fov = svc.predict({"fov": 2, "timestep": 1, "cells": _cells()})

    assert any(v > 0 for v in default_fov["exposures"].values())
    assert set(gated_fov["exposures"].values()) == {0.0}
    svc.close()


# ---------------------------------------------------------------------------
# model cache
# ---------------------------------------------------------------------------


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(
            future_len=10, cnr_mode="norm", num_layers=1, hidden_dim=4,
            norm_channels=["cnr", "u_t", "fov_density", "n_cells_200px"],
            norm_mean=[1.5, 10.0, 50.0, 5.0], norm_std=[0.8, 20.0, 20.0, 3.0],
        )


def test_load_model_cache_shares_one_model_across_fovs(monkeypatch, tmp_path):
    """N FOVs on the same checkpoint must load and warm up ONE model."""
    calls = []

    def fake_load_experiment(path):
        calls.append(path)
        return SimpleNamespace(
            model_type="fake", reconstruct_model=lambda: _FakeModel()
        )

    import optoerk.core.experiment as exp

    monkeypatch.setattr(exp, "load_experiment", fake_load_experiment)

    # Real directories: the loader resolves a checkpoint before opening it, so a
    # name that exists nowhere never reaches load_experiment at all.
    ck_a, ck_b = tmp_path / "ckpt_a", tmp_path / "ckpt_b"
    ck_a.mkdir()
    ck_b.mkdir()

    cache = {}
    a = load_model(str(ck_a), "cpu", cache)
    b = load_model(str(ck_a), "cpu", cache)
    c = load_model(str(ck_b), "cpu", cache)

    assert a is b, "same checkpoint should hit the cache"
    assert c is not a
    assert calls == [str(ck_a), str(ck_b)], "cached load must not re-read the bundle"
    assert a.info["future_len"] == 10


def test_unresolvable_checkpoint_names_both_locations_tried(tmp_path):
    """A bare bundle name that resolves nowhere degrades the FOV to the stub, so
    the error has to say where it looked — otherwise the failure reads as a
    corrupt bundle rather than an unresolved name."""
    from optoerk.serving.runtime import _resolve_checkpoint

    with pytest.raises(FileNotFoundError, match="results_write_path"):
        _resolve_checkpoint("no_such_bundle_anywhere")


def test_bare_checkpoint_name_resolves_against_results_path(monkeypatch, tmp_path):
    """Policy files name bundles bare so one file works on cluster and laptop."""
    import optoerk.core.utils as utils
    import optoerk.serving.runtime as rt

    (tmp_path / "my_bundle").mkdir()
    monkeypatch.setattr(utils, "results_write_path", lambda: str(tmp_path))
    assert rt._resolve_checkpoint("my_bundle") == str(tmp_path / "my_bundle")


def test_router_reuses_one_handle_for_two_fovs(monkeypatch, tmp_path):
    import optoerk.core.experiment as exp

    monkeypatch.setattr(
        exp, "load_experiment",
        lambda path: SimpleNamespace(model_type="fake",
                                     reconstruct_model=lambda: _FakeModel()),
    )
    shared = tmp_path / "shared"
    shared.mkdir()
    p = tmp_path / "policies.toml"
    p.write_text(
        f'[default]\ncheckpoint = "{shared}"\n'
        'objective = { type = "hold", target_cnr = 1.5 }\n'
        '[fov.1]\nobjective = { type = "hold", target_cnr = 2.0 }\n'
        '[fov.2]\nobjective = { type = "hold", target_cnr = 3.0 }\n'
    )
    router = PolicyRouter(_cfg(), load_policy_file(p))
    assert router.describe()["n_models_loaded"] == 1
    assert isinstance(router.engine_for(1), RealModelEngine)
    # distinct engines, but the same underlying torch module
    assert router.engine_for(1) is not router.engine_for(2)
    assert router.engine_for(1).model is router.engine_for(2).model


# ---------------------------------------------------------------------------
# service contract (unchanged behaviour must stay unchanged)
# ---------------------------------------------------------------------------


def test_retry_is_idempotent_and_does_not_advance_state():
    svc = InferenceService(_cfg())
    payload = {"fov": 0, "timestep": 4, "cells": _cells()}
    first = svc.predict(payload)
    again = svc.predict(payload)
    assert first["exposures"] == again["exposures"]
    assert svc._n_predict == 2
    # one advance only
    assert svc.store.get(0, 0).n_frames == 1
    svc.close()


def test_positions_reach_the_cell_frame():
    """Objectives gate on x/y, so the payload's coordinates must survive the trip."""
    seen = {}
    svc = InferenceService(_cfg())
    engine = svc.router.default_engine
    original = engine.decide

    def spy(frames, ctx):
        seen["xs"] = [f.x for f in frames]
        seen["ts"] = ctx.timestep
        seen["fov"] = ctx.fov
        return original(frames, ctx)

    engine.decide = spy
    svc.predict({"fov": 7, "timestep": 3, "cells": _cells(3)})
    assert seen["xs"] == [0.0, 100.0, 200.0]
    assert (seen["ts"], seen["fov"]) == (3, 7)
    svc.close()


def test_missing_cnr_yields_zero_and_is_reported_skipped():
    svc = InferenceService(_cfg())
    out = svc.predict({"fov": 0, "timestep": 1,
                       "cells": [{"particle": 0, "x": 1.0, "y": 1.0}]})
    assert out["exposures"] == {"0": 0.0}
    svc.close()


def test_reset_clears_state_for_one_fov_only():
    svc = InferenceService(_cfg())
    svc.predict({"fov": 0, "timestep": 1, "cells": _cells()})
    svc.predict({"fov": 1, "timestep": 1, "cells": _cells()})
    svc.reset(fov=0)
    assert svc.store.get(0, 0) is None
    assert svc.store.get(1, 0) is not None
    svc.close()


# ---------------------------------------------------------------------------
# what the controller believed: plan_cost / pred_cnr_h1
# ---------------------------------------------------------------------------


def _real_engine(objective, controller):
    """A RealModelEngine on an untrained model of the production shape — enough to
    exercise the decide() path without a checkpoint or a mount."""
    from optoerk.serving.bench import synthetic_handle
    from optoerk.serving.calibration import FluenceCalibration

    cfg = ServerConfig(warmup=False, control_horizon=5, gpu_sample_interval_s=0)
    handle = synthetic_handle(future_len=5, device=torch.device("cpu"))
    calib = FluenceCalibration(cfg.instrument, cfg.stim_power_pct)
    return RealModelEngine(handle, calib, cfg, objective, controller)


def _frames(n):
    from optoerk.serving.runtime import CellFrame
    from optoerk.serving.state import CellState

    out = []
    for i in range(n):
        st = CellState()
        st.particle = i
        out.append(CellFrame(state=st, cnr_norm=1.0, fov_density=float(n),
                             n_cells_200px=5.0, x=float(i), y=0.0))
    return out


def test_decide_records_the_plan_cost_and_the_one_step_prediction():
    from optoerk.serving.control import ConstantDoseSearch, dose_levels
    from optoerk.serving.objectives import GoalContext, hold

    levels = dose_levels(0.0, 150.0, 5)
    engine = _real_engine(hold(1.2), ConstantDoseSearch(levels))
    frames = _frames(4)
    ms = engine.decide(frames, GoalContext(fov=0, timestep=3, cells=frames))

    assert len(engine.last_plan_cost) == 4
    assert len(engine.last_pred_cnr_h1) == 4
    assert all(c >= 0 for c in engine.last_plan_cost), "an l2 cost cannot be negative"
    # the prediction is in absolute CNR, not z-scores
    assert all(0.0 < p < 5.0 for p in engine.last_pred_cnr_h1)
    assert len(ms) == 4


def test_decide_clears_the_diagnostics_on_an_empty_frame():
    """Otherwise an empty FOV silently relogs the previous frame's beliefs."""
    from optoerk.serving.control import ConstantDoseSearch, dose_levels
    from optoerk.serving.objectives import GoalContext, hold

    engine = _real_engine(hold(1.2), ConstantDoseSearch(dose_levels(0.0, 150.0, 5)))
    frames = _frames(2)
    engine.decide(frames, GoalContext(fov=0, timestep=1, cells=frames))
    assert engine.last_plan_cost is not None
    engine.decide([], GoalContext(fov=0, timestep=2, cells=[]))
    assert engine.last_plan_cost is None
    assert engine.last_pred_cnr_h1 is None


def test_prediction_is_the_rollout_under_the_commanded_dose():
    """pred_cnr_h1 must follow the *gated* command, not the plan's raw pick.

    Asserted as the identity it is — the model's own one-step rollout, from the
    post-advance encoder state, under exactly the ms that decide() returned —
    rather than by hoping an untrained model prefers to dose.
    """
    from optoerk.serving.control import ConstantDoseSearch, dose_levels
    from optoerk.serving.objectives import GoalContext, gated
    from optoerk.serving.runtime import CNR

    levels = dose_levels(0.0, 150.0, 5)
    # after_t=1000 gates every cell out at timestep 3
    engine = _real_engine(gated(target_cnr=2.5, after_t=1000),
                          ConstantDoseSearch(levels))
    frames = _frames(3)
    ms = engine.decide(frames, GoalContext(fov=0, timestep=3, cells=frames))
    assert ms == [0.0, 0.0, 0.0], "the gate forces exactly zero"

    # decide() leaves the post-advance encoder state on each cell
    h = torch.cat([f.state.h for f in frames], dim=1)
    c = torch.cat([f.state.c for f in frames], dim=1)
    cnr_fb = torch.tensor(
        [[(f.cnr_norm - float(engine.mean[CNR])) / float(engine.std[CNR])]
         for f in frames], dtype=torch.float32,
    )

    def rollout1(doses):
        fut = engine.std_fluence(torch.tensor(doses, dtype=torch.float32).reshape(-1, 1, 1))
        return engine.denorm_cnr(engine.rollout(h, c, cnr_fb, fut))[:, 0].tolist()

    assert engine.last_pred_cnr_h1 == pytest.approx(rollout1(ms))
    # and the prediction really does depend on the dose, so the check above has teeth
    assert engine.last_pred_cnr_h1 != pytest.approx(rollout1([150.0] * 3))


def test_predict_log_carries_the_controller_beliefs(tmp_path):
    """End-to-end through the service. The stub has no forward model, so both
    fields are null rather than absent — the schema stays fixed either way."""
    log = tmp_path / "run.jsonl"
    cfg = _cfg()
    cfg.predict_log_path = str(log)
    svc = InferenceService(cfg)
    svc.predict({"fov": 0, "timestep": 1, "cells": _cells()})
    svc.close()

    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    cells = [c for r in recs if r["event"] == "predict" for c in r["cells"]]
    assert cells
    for c in cells:
        assert "plan_cost" in c and "pred_cnr_h1" in c
        assert c["plan_cost"] is None      # stub engine
        assert c["pred_cnr_h1"] is None


# ---------------------------------------------------------------------------
# arm labels derived from the policy file
# ---------------------------------------------------------------------------


def test_arm_map_groups_fovs_by_identical_policy():
    """The shipped 10-FOV layout is interleaved to balance stage position against
    arm, so `fov % 4 + 1` is wrong for it. Reading the grouping off the file
    cannot drift from the file."""
    from optoerk.serving.policy import arm_map

    pf = load_policy_file(
        Path(__file__).parent.parent / "policies" / "policy_10fov_osc.toml")
    arms = arm_map(pf)
    assert arms == {0: 3, 1: 4, 2: 1, 3: 2, 4: 3, 5: 4, 6: 2, 7: 1, 8: 3, 9: 4}
    assert arms != {f: f % 4 + 1 for f in range(10)}, "the formula must not hold"

    by_arm = {}
    for fov, arm in arms.items():
        by_arm.setdefault(arm, []).append(fov)
    assert sorted(by_arm) == [1, 2, 3, 4]
    assert [len(v) for _, v in sorted(by_arm.items())] == [2, 2, 3, 3]


def test_arm_map_holds_for_the_blocked_12fov_layout():
    """The one file the old formula was right about must still come out right."""
    from optoerk.serving.policy import arm_map

    pf = load_policy_file(
        Path(__file__).parent.parent / "policies" / "policy_12fov_osc.toml")
    assert arm_map(pf) == {f: f % 4 + 1 for f in range(12)}


def test_arm_map_separates_arms_that_differ_only_in_objective(tmp_path):
    """The pattern-zoo case: same controller everywhere, waveform is the arm."""
    from optoerk.serving.policy import arm_map

    p = tmp_path / "policies.toml"
    p.write_text(
        '[default]\ncontroller = { type = "constant_dose" }\nkernel = "l2"\n'
        '[fov.0]\nobjective = { type = "hold", target_cnr = 1.0 }\n'
        '[fov.1]\nobjective = { type = "hold", target_cnr = 2.0 }\n'
        '[fov.2]\nobjective = { type = "hold", target_cnr = 1.0 }\n'
    )
    assert arm_map(load_policy_file(p)) == {0: 1, 1: 2, 2: 1}


def test_arm_map_rejects_a_half_declared_file(tmp_path):
    """Mixing declared ids with derived ones is the exact failure the declaration
    exists to remove, so a partial labelling is an error rather than a guess."""
    from optoerk.serving.policy import arm_map

    p = tmp_path / "policies.toml"
    p.write_text(
        '[default]\nobjective = { type = "hold", target_cnr = 1.0 }\n'
        '[fov.0]\narm = 1\ncontroller = { type = "constant_dose" }\n'
        '[fov.1]\ncontroller = { type = "sequence_mpc" }\n'
    )
    with pytest.raises(PolicyViolation, match="Declare it for every FOV or for none"):
        arm_map(load_policy_file(p))


def test_shipped_pattern_zoo_policy_varies_waveform_and_nothing_else():
    """The pattern-zoo run inverts the previous design: the reference is the arm.

    Checked as a unit because the whole experiment rests on it — if the controller
    varies too, waveform and controller are confounded and no arm means anything.
    """
    from optoerk.serving.policy import arm_map

    pf = load_policy_file(
        Path(__file__).parent.parent / "policies" / "policy_10fov_patterns.toml")
    assert pf.placeholders_resolved is False, "must not ship ready-to-run"
    assert set(pf.fov) == set(range(10))

    assert arm_map(pf) == {0: 1, 1: 2, 2: 3, 3: 4, 4: 2, 5: 1, 6: 4, 7: 3, 8: 2, 9: 1}

    # ONE controller on every FOV, so waveform is the only varying factor
    for f in range(10):
        spec = pf.fov[f]
        assert spec.controller == {"type": "sequence_mpc"}, f"fov {f}"
        assert spec.kernel == "l2", f"fov {f}"
        assert spec.lambda_move == 0.0, f"fov {f}"
        assert spec.levels_ms == [0, 20, 45, 85, 150], f"fov {f}"
        assert spec.control_horizon == 30, f"fov {f}"

    # ...and every arm really does differ in reference
    refs = {}
    for f in range(10):
        refs.setdefault(json.dumps(pf.fov[f].objective, sort_keys=True), set()).add(f)
    assert len(refs) == 4, "four distinct waveforms"
    assert sorted(len(v) for v in refs.values()) == [2, 2, 3, 3]

    # arm 3 is arm 1's slowest block with the fall stretched: same endpoints, same
    # period, fall 18 -> 38. That one-factor contrast is the point of the arm.
    stair = pf.fov[0].objective["blocks"][0]
    brake = pf.fov[2].objective
    assert (brake["low"], brake["high"]) == (stair["low"], stair["high"])
    assert sum(brake[k] for k in
               ("t_low_min", "t_rise_min", "t_high_min", "t_fall_min")) == 50
    assert brake["t_fall_min"] > stair["t_fall_min"]


def test_pattern_zoo_arms_are_mean_matched_where_they_claim_to_be():
    """Arms 1, 3 and 4 share a reference mean by construction, so a drift
    difference between them is not an exposure difference. The staircase's blocks
    are mean-matched to each other too. Neither is automatic — both were solved
    for, and a retuned block table silently breaks them.
    """
    from optoerk.serving.objectives import build_objective

    pf = load_policy_file(
        Path(__file__).parent.parent / "policies" / "policy_10fov_patterns.toml")

    def trapezoid_mean(r):
        x = r["t_high_min"] + (r["t_rise_min"] + r["t_fall_min"]) / 2
        period = sum(r[k] for k in
                     ("t_low_min", "t_rise_min", "t_high_min", "t_fall_min"))
        return r["low"] + (r["high"] - r["low"]) * x / period

    blocks = pf.fov[0].objective["blocks"]
    block_means = [trapezoid_mean(b) for b in blocks]
    assert max(block_means) - min(block_means) < 0.012, "staircase varies f at fixed mean"

    # duration-weighted sweep mean == the hold arm's target
    durations = [
        b["n_cycles"] * sum(b[k] for k in
                            ("t_low_min", "t_rise_min", "t_high_min", "t_fall_min"))
        for b in blocks
    ]
    sweep_mean = sum(d * m for d, m in zip(durations, block_means)) / sum(durations)
    assert sweep_mean == pytest.approx(pf.fov[3].objective["target_cnr"], abs=0.002)
    # ...and arm 3 sits on it too
    assert trapezoid_mean(pf.fov[2].objective) == pytest.approx(sweep_mean, abs=0.01)

    # every FOV's objective actually builds
    for f in range(10):
        spec = pf.fov[f]
        build_objective({**spec.objective, "kernel": spec.kernel,
                         "lambda_move": spec.lambda_move})


def test_the_control_policy_differs_from_the_main_one_ONLY_by_the_server_flag():
    """The pattern-zoo pair separates "what the expression covariate contributes"
    from "what feedback contributes". That only works if the two files are
    otherwise identical — any drift between them lands squarely in the contrast
    and neither run can be attributed.

    The difference is a launch flag (`--live-optortk-expr`), deliberately NOT a
    field in the policy: the policy describes the experiment, the flag describes
    what the server is allowed to condition on.
    """
    from optoerk.serving.policy import arm_map

    root = Path(__file__).parent.parent / "policies"
    main = load_policy_file(root / "policy_10fov_patterns.toml")
    ctrl = load_policy_file(root / "policy_10fov_patterns_control.toml")

    assert arm_map(main) == arm_map(ctrl)
    assert main.placeholders_resolved == ctrl.placeholders_resolved is False
    assert main.default.model_dump() == ctrl.default.model_dump()
    assert set(main.fov) == set(ctrl.fov) == set(range(10))
    for f in sorted(main.fov):
        assert main.fov[f].model_dump() == ctrl.fov[f].model_dump(), f"fov {f} drifted"


def test_the_pattern_zoo_checkpoint_is_servable():
    """Every channel the checkpoint declares must be one serving can actually
    supply. A channel it cannot supply is silently fed the population mean — which
    is how `nuc_area` would have gone out as a constant on the second most
    important input."""
    import inspect
    import re

    from optoerk.serving.runtime import RealModelEngine

    src = inspect.getsource(RealModelEngine.decide)
    suppliable = set(re.findall(r'"(\w+)"\s*[]:]\s*=?\s*\(?\s*lambda', src)) | {
        "cnr", "optortk_expr", "nuc_area",
    }
    # the channels the pattern-zoo checkpoint was trained on
    needed = ["cnr", "u_t", "n_cells_200px", "optortk_expr", "nuc_area"]
    missing = [c for c in needed if c not in suppliable]
    assert not missing, f"serving cannot supply {missing}"


def test_an_unavailable_device_fails_loudly_instead_of_degrading_to_the_stub():
    """`--device cuda` on a CPU-only torch used to blow up inside load_model, get
    caught by the degrade-to-stub handler, and surface as
    `checkpoint load failed ('Torch not compiled with CUDA enabled')` — blaming the
    checkpoint for an install problem, while every FOV silently became a stub and
    the server came up "working" with no model at all.

    A device the process cannot use is a configuration error for the whole server.
    """
    from optoerk.serving.runtime import _resolve_device

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA; the failure path cannot be exercised")
    with pytest.raises(RuntimeError, match="cannot use CUDA"):
        _resolve_device("cuda")
    # the message has to say what to do, not just what happened
    try:
        _resolve_device("cuda")
    except RuntimeError as e:
        msg = str(e)
        assert "--device cpu" in msg
        assert "nvidia-smi" in msg

    # auto never raises — it is the "use what is here" request
    assert _resolve_device("auto").type in ("cpu", "cuda", "mps")
    assert _resolve_device("cpu").type == "cpu"


# ---------------------------------------------------------------------------
# inference batch bucketing (ServerConfig.batch_bucket)
# ---------------------------------------------------------------------------


def test_bucket_to_rounds_up_to_a_multiple():
    from optoerk.serving.runtime import _bucket_to

    assert _bucket_to(1, 32) == 32
    assert _bucket_to(32, 32) == 32, "an exact multiple must not grow a whole bucket"
    assert _bucket_to(33, 32) == 64
    assert _bucket_to(207, 32) == 224
    # 0 and 1 disable it; the count must pass through untouched
    assert _bucket_to(207, 1) == 207
    assert _bucket_to(207, 0) == 207


def test_padding_frame_cannot_write_back_to_the_cell_it_copied():
    """The padding cell shares a batch with real ones and is thrown away after.
    If it shared their state object, its discarded encoder state and dose would
    land on a real cell."""
    from optoerk.serving.runtime import _padding_frame

    proto = _frames(1)[0]
    proto.state.baseline_samples.append(1.0)
    proto.state.h = torch.ones(1, 1, 4)

    pad = _padding_frame(proto)
    assert pad.state is not proto.state
    assert pad.state.h is None and pad.state.c is None, "must start from zeros"
    # scalars are carried over, so the padding row is in-distribution
    assert pad.cnr_norm == proto.cnr_norm
    assert pad.x == proto.x

    pad.state.baseline_samples.append(99.0)
    pad.state.last_applied_ms = 123.0
    assert proto.state.baseline_samples == [1.0], "mutable state must not be shared"
    assert proto.state.last_applied_ms != 123.0


def _doses(engine, bucket, n_cells):
    from optoerk.serving.objectives import GoalContext

    engine.cfg = replace(engine.cfg, batch_bucket=bucket)
    frames = _frames(n_cells)
    ms = engine.decide(frames, GoalContext(fov=0, timestep=3, cells=frames))
    return ms, list(engine.last_plan_cost), list(engine.last_pred_cnr_h1)


@pytest.mark.parametrize("n_cells", [32, 64])
def test_bucketing_is_a_no_op_when_no_padding_is_needed(n_cells):
    """A cell count already on a bucket boundary must take the identical path."""
    from optoerk.serving.control import SequenceMPC, dose_levels
    from optoerk.serving.objectives import hold

    # ONE engine for both runs: `synthetic_handle` draws random weights, so a
    # second engine would differ for reasons unrelated to padding.
    engine = _real_engine(
        hold(1.2), SequenceMPC(dose_levels(0.0, 150.0, 5), n_samples=64, n_iters=2)
    )
    off = _doses(engine, 0, n_cells)
    on = _doses(engine, 32, n_cells)
    assert on == off


@pytest.mark.parametrize("n_cells", [1, 7, 33])
def test_bucketing_perturbs_the_cost_only_by_float_roundoff(n_cells):
    """Padding must not change what the controller *believes*, only the batch it
    computes it in.

    It is deliberately NOT asserted that the doses are identical. Batched matmul
    reassociates with the batch size, so the predicted cost moves by float32
    round-off, and argmin over a discrete dose ladder can land on the other side
    of a near-tie. The contract is that the cost itself stays within round-off —
    if padding ever changed it materially, a padding cell would be influencing a
    real one.
    """
    from optoerk.serving.control import SequenceMPC, dose_levels
    from optoerk.serving.objectives import hold

    engine = _real_engine(
        hold(1.2), SequenceMPC(dose_levels(0.0, 150.0, 5), n_samples=64, n_iters=2)
    )
    _, off_cost, off_pred = _doses(engine, 0, n_cells)
    _, on_cost, on_pred = _doses(engine, 32, n_cells)

    assert max(abs(a - b) for a, b in zip(on_cost, off_cost)) < 1e-4
    assert max(abs(a - b) for a, b in zip(on_pred, off_pred)) < 1e-4


def test_bucketing_returns_one_dose_per_real_cell_only():
    """The padded rows must never reach the caller — faro maps the returned list
    positionally onto the cells it sent."""
    from optoerk.serving.control import ConstantDoseSearch, dose_levels
    from optoerk.serving.objectives import GoalContext, hold

    engine = _real_engine(hold(1.2), ConstantDoseSearch(dose_levels(0.0, 150.0, 5)))
    engine.cfg = replace(engine.cfg, batch_bucket=32)
    frames = _frames(5)
    ms = engine.decide(frames, GoalContext(fov=0, timestep=1, cells=frames))

    assert len(ms) == 5
    assert len(engine.last_plan_cost) == 5
    assert len(engine.last_pred_cnr_h1) == 5
    # every real cell advanced; nothing else was created to advance
    assert all(f.state.h is not None for f in frames)
    assert len(frames) == 5, "the caller's list must not be left padded"


# ---------------------------------------------------------------------------
# NVML device resolution (gpu.resolve_nvml_handle)
# ---------------------------------------------------------------------------


class _FakeNvml:
    """Two physical GPUs, so an index handed straight across can pick the wrong one."""

    def __init__(self):
        self.uuids = {0: b"GPU-aaaaaaaa-0000", 1: b"GPU-bbbbbbbb-1111"}

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        if i not in self.uuids:
            raise RuntimeError(f"no such nvml device {i}")
        return f"handle-{i}"

    def nvmlDeviceGetHandleByUUID(self, uuid):
        for i, u in self.uuids.items():
            if u == uuid:
                return f"handle-{i}"
        raise RuntimeError("no such uuid")

    def nvmlDeviceGetUUID(self, handle):
        return self.uuids[int(str(handle).split("-")[1])]


def test_nvml_handle_resolves_by_uuid_over_the_index(monkeypatch):
    """The whole point: torch's cuda:0 can be the machine's physical GPU 1, and
    the UUID is the only identifier that means the same thing on both sides."""
    from optoerk.serving import gpu

    monkeypatch.setattr(gpu, "_torch_device_uuid", lambda i: "GPU-bbbbbbbb-1111")
    handle, meta = gpu.resolve_nvml_handle(_FakeNvml(), 0)

    assert handle == "handle-1", "index 0 must not win over a UUID that says otherwise"
    assert meta["resolved_by"] == "uuid"
    assert meta["verified"] is True
    assert meta["nvml_index"] == 1


def test_nvml_handle_falls_back_to_cuda_visible_devices(monkeypatch):
    """No UUID from torch (older driver), but the remapping is still recoverable."""
    from optoerk.serving import gpu

    monkeypatch.setattr(gpu, "_torch_device_uuid", lambda i: "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    handle, meta = gpu.resolve_nvml_handle(_FakeNvml(), 0)

    assert handle == "handle-1"
    assert meta["resolved_by"] == "visible-devices"
    assert meta["verified"] is True


def test_nvml_handle_marks_the_bare_index_as_unverified(monkeypatch):
    """Telemetry still flows when nothing can confirm the mapping — but it must
    say so, or a reader trusts numbers that may describe another card."""
    from optoerk.serving import gpu

    monkeypatch.setattr(gpu, "_torch_device_uuid", lambda i: "")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    handle, meta = gpu.resolve_nvml_handle(_FakeNvml(), 0)

    assert handle == "handle-0"
    assert meta["resolved_by"] == "index-unverified"
    assert meta["verified"] is False


def test_gpu_sampler_takes_a_torch_device_not_an_index():
    """`dev.index or 0` was the original bug: it silently coerced a device with
    no explicit index, and handed NVML a number from torch's namespace."""
    from optoerk.serving.gpu import GpuSampler

    assert GpuSampler(lambda r: None, torch.device("cuda"), 5.0)._device_index == 0
    assert GpuSampler(lambda r: None, torch.device("cuda:1"), 5.0)._device_index == 1
    # the index-0 case must survive the `or` that used to swallow it
    assert GpuSampler(lambda r: None, torch.device("cuda:0"), 5.0)._device_index == 0


# ---------------------------------------------------------------------------
# acquisition cadence guard
# ---------------------------------------------------------------------------


def _drive_at_cadence(svc, interval_s, n_frames=12, n_fovs=2, monkeypatch=None):
    """Drive the service with a controlled wall clock, so the test is not timing
    dependent. Fields are imaged sequentially inside a round, as faro does."""
    import optoerk.serving.service as svc_mod

    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(svc_mod.time, "time", lambda: clock["t"])
    for i in range(n_frames):
        for fov in range(n_fovs):
            clock["t"] += interval_s / n_fovs        # the per-field slot
            svc.predict({
                "fov": fov, "timestep": i,
                "cells": [{"particle": p, "x": 1.0 * p, "y": 1.0, "cnr_median": 0.9}
                          for p in range(3)],
            })


def test_cadence_guard_stays_quiet_when_the_rig_keeps_time(tmp_path, monkeypatch):
    log = tmp_path / "run.jsonl"
    svc = InferenceService(_cfg(predict_log_path=str(log)))
    try:
        _drive_at_cadence(svc, 60.0, monkeypatch=monkeypatch)
        assert svc.cadence_degraded is False
    finally:
        svc.close()
    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    cad = [r for r in recs if r.get("event") == "cadence"]
    assert len(cad) == 1 and cad[0]["degraded"] is False
    assert cad[0]["observed_s"] == pytest.approx(60.0, abs=1.0)


def test_cadence_guard_flags_a_run_that_is_silently_running_slow(tmp_path, monkeypatch):
    """The v12 failure: 12 h declared, 40 h delivered, discovered only afterwards
    from the parquet timestamps. Every reference period was multiplied by 3.4."""
    log = tmp_path / "run.jsonl"
    svc = InferenceService(_cfg(predict_log_path=str(log)))
    try:
        _drive_at_cadence(svc, 202.0, monkeypatch=monkeypatch)   # v12's real cadence
        assert svc.cadence_degraded is True
    finally:
        svc.close()

    recs = [json.loads(ln) for ln in log.read_text().splitlines()]
    cad = [r for r in recs if r.get("event") == "cadence"]
    assert len(cad) == 1, "reported once, not every frame"
    assert cad[0]["degraded"] is True
    assert cad[0]["ratio"] == pytest.approx(202.0 / 60.0, rel=0.05)
    # and it is sticky on every frame the analysis will read
    preds = [r for r in recs if r.get("event") == "predict"]
    assert preds[-1]["cadence_degraded"] is True


def test_cadence_is_measured_per_field_not_between_requests(monkeypatch):
    """Fields are imaged sequentially within a round, so consecutive requests are a
    slot apart, not a frame apart. Measuring across fields would read an 8-field
    round as 8x faster than it is and never fire."""
    svc = InferenceService(_cfg())
    try:
        # 8 fields inside a 60 s round: each field is 60 s apart from its own last
        # frame, but successive requests are only 7.5 s apart.
        _drive_at_cadence(svc, 60.0, n_frames=12, n_fovs=8, monkeypatch=monkeypatch)
        assert svc.cadence_degraded is False, (
            "read the gap between requests instead of between a field's own frames"
        )
    finally:
        svc.close()


# ---------------------------------------------------------------------------
# the cadence invariant is not configurable
# ---------------------------------------------------------------------------

def test_frame_interval_is_a_constant_not_a_setting():
    """One frame per minute, one inference per frame — nothing may configure it.

    It used to be a ServerConfig field behind --frame-interval-min, declared TWICE
    in the same dataclass. Raising it to match a slow rig silenced the only alarm
    that catches the slip AND rescaled every waveform's frame count, so two runs
    came out looking clean while measuring something else. This test is the guard
    against it coming back by either route: the config field or the objective spec.
    """
    from dataclasses import fields as dc_fields

    from optoerk.serving.config import FRAME_INTERVAL_MIN, FRAME_INTERVAL_S, ServerConfig

    assert FRAME_INTERVAL_MIN == 1.0
    assert FRAME_INTERVAL_S == 60.0
    names = {f.name for f in dc_fields(ServerConfig)}
    assert not {n for n in names if "frame_interval" in n or "cadence" in n}, (
        "the cadence invariant is back on ServerConfig"
    )
    with pytest.raises(TypeError):
        ServerConfig(frame_interval_min=2.0)


@pytest.mark.parametrize("spec", [
    {"type": "oscillation", "low": 1.0, "high": 1.2, "t_low_min": 12,
     "t_rise_min": 2, "t_high_min": 18, "t_fall_min": 18},
    {"type": "frequency_staircase", "blocks": [
        {"low": 1.0, "high": 1.2, "t_low_min": 12, "t_rise_min": 2,
         "t_high_min": 18, "t_fall_min": 18, "n_cycles": 1}]},
])
def test_no_objective_may_declare_its_own_frame_interval(spec):
    """The second route in, and the one a policy file can reach. This is exactly
    what policy_8fov_diverse.toml carried into v14: frame_interval_min = 2.0 on the
    staircase, which halved every period's frame count without touching the rig."""
    from optoerk.serving.config import FRAME_INTERVAL_MIN
    from optoerk.serving.objectives import build_objective

    assert build_objective(dict(spec)).reference.frame_interval_min == FRAME_INTERVAL_MIN
    with pytest.raises(TypeError, match="frame_interval_min"):
        build_objective({**spec, "frame_interval_min": 2.0})
