"""Policy routing, config parsing and the service contract.

Runs entirely on the stub engine plus a fake checkpoint, so no cluster mount and
no trained model are needed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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
        "tau_decay_min = 7.3, n_phase_groups = 4 }\n"
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


def test_oscillation_policy_refuses_a_period_the_horizon_cannot_see(tmp_path):
    """A misconfigured experiment stops the server; it does not quietly degrade
    that FOV to the stub and run for twelve hours measuring nothing."""
    p = tmp_path / "policies.toml"
    p.write_text(
        "[default]\ncontrol_horizon = 10\n"
        'objective = { type = "oscillation", low = 0.85, high = 1.15, '
        "t_low_min = 8, t_rise_min = 2, t_high_min = 15, t_fall_min = 15, "
        "tau_decay_min = 7.3 }\n"
    )
    with pytest.raises(PolicyViolation, match="exceeds 2 x the control horizon"):
        PolicyRouter(_cfg(), load_policy_file(p))


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
