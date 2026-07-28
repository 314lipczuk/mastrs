"""Policy routing, config parsing and the service contract.

Runs entirely on the stub engine plus a fake checkpoint, so no cluster mount and
no trained model are needed.
"""
import json
from types import SimpleNamespace

import pytest
import torch.nn as nn

from optoerk.serving.config import ServerConfig
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


def test_load_model_cache_shares_one_model_across_fovs(monkeypatch):
    """N FOVs on the same checkpoint must load and warm up ONE model."""
    calls = []

    def fake_load_experiment(path):
        calls.append(path)
        return SimpleNamespace(
            model_type="fake", reconstruct_model=lambda: _FakeModel()
        )

    import optoerk.core.experiment as exp

    monkeypatch.setattr(exp, "load_experiment", fake_load_experiment)

    cache = {}
    a = load_model("ckpt_a", "cpu", cache)
    b = load_model("ckpt_a", "cpu", cache)
    c = load_model("ckpt_b", "cpu", cache)

    assert a is b, "same checkpoint should hit the cache"
    assert c is not a
    assert calls == ["ckpt_a", "ckpt_b"], "cached load must not re-read the bundle"
    assert a.info["future_len"] == 10


def test_router_reuses_one_handle_for_two_fovs(monkeypatch, tmp_path):
    import optoerk.core.experiment as exp

    monkeypatch.setattr(
        exp, "load_experiment",
        lambda path: SimpleNamespace(model_type="fake",
                                     reconstruct_model=lambda: _FakeModel()),
    )
    p = tmp_path / "policies.toml"
    p.write_text(
        '[default]\ncheckpoint = "shared"\nobjective = { type = "hold", target_cnr = 1.5 }\n'
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
