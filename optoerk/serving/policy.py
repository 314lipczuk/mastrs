"""Per-FOV policies: which model, which goal, which controller — for which FOV.

A single microscope runs several FOVs, and they do not have to share an
experiment. A **policy** is the triple (checkpoint, objective, controller); a
**policy file** names a default policy plus per-FOV overrides:

```toml
[default]
checkpoint = "results/seq2scal_history_optortk_multilen_2026-07-14_09.48.21"
objective  = { type = "hold", target_cnr = 1.5 }
controller = { type = "sequence_mpc", n_samples = 128 }

[fov.1]
objective = { type = "hold", target_cnr = 2.0 }

[fov.2]
checkpoint = "results/some_other_model"
objective  = { type = "gated", target_cnr = 1.8, after_t = 10, x_gt = 512,
               max_neighbours_200px = 5 }
```

JSON with the same shape works too. A FOV entry inherits every field it does not
set from ``[default]``; ``objective`` and ``controller`` are replaced wholesale
rather than deep-merged, so a partially-overridden objective can never end up as a
surprising hybrid of two specs.

Models are cached by ``(checkpoint, device)``, so N FOVs sharing a checkpoint load
and warm up **one** model. A FOV whose policy fails to build degrades to the stub
engine for that FOV alone — a typo in one FOV's objective never takes down the
server for the others.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from optoerk.serving.config import ServerConfig
from optoerk.serving.control import build_controller, dose_levels
from optoerk.serving.objectives import PolicyViolation, build_objective


class PolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Which experimental arm this FOV belongs to. Pure metadata — nothing in the
    # control path reads it — but it is the experiment's own grouping, and until
    # it existed every consumer that wanted it reinvented `fov % 4 + 1`, which is
    # right for the blocked 12-FOV layout and wrong for every interleaved one.
    # Declared here it travels with the policy into the startup log.
    arm: int | None = None
    checkpoint: str | None = None
    objective: dict[str, Any] | None = None
    controller: dict[str, Any] | None = None
    # Per-FOV lookahead override. Still hard-capped by the checkpoint's future_len.
    control_horizon: int | None = None
    # --- arm-varying pieces of the objective ------------------------------
    # An objective decomposes into a reference (what to track), a cost kernel
    # (how to score the prediction against it) and plan-side regularizers. In a
    # controlled comparison the reference is a property of the EXPERIMENT and
    # must be byte-identical across arms, while the kernel and the regularizer
    # coefficients are exactly what the arms vary.
    #
    # ``objective`` is replaced wholesale on override (never deep-merged), so
    # restating it per FOV would put that invariant at the mercy of a typo in one
    # of twelve blocks. These three fields instead compose onto the inherited
    # objective, so an arm can only change the parts an arm is allowed to change.
    kernel: str | dict[str, Any] | None = None
    lambda_move: float | None = None
    lambda_dose: float | None = None
    # Explicit dose ladder in ms. When None the ladder is the evenly-spaced grid
    # implied by (min_exposure_ms, max_exposure_ms, n_candidates) on the config.
    # A linspace cannot express a rebinned ladder like [0, 20, 45, 85, 150], and
    # the ladder is an experiment parameter rather than a code constant, so it has
    # to be settable here. Normally set once in [default] and shared by every FOV —
    # arms that differ in their ladder are not comparable.
    levels_ms: list[float] | None = None


class PolicyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: PolicySpec = Field(default_factory=PolicySpec)
    fov: dict[int, PolicySpec] = Field(default_factory=dict)
    # Guard for experiments whose real parameters are not known yet (dose ladder,
    # setpoints, period, lambda_move, band half-width). A policy file ships with
    # working assumptions so it can be wired and tested, and refuses to serve
    # until whoever filled in the measured values flips this to true. Twelve hours
    # of microscope time run against a placeholder is not recoverable.
    placeholders_resolved: bool = True


def load_policy_file(path: str | Path) -> PolicyFile:
    """Parse a ``.toml`` or ``.json`` policy file, applying default-inheritance.

    Raises on an unknown key or a malformed spec — a policy file is the experiment
    definition, and a silently-ignored typo there means running the wrong
    experiment for twelve hours.
    """
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        raw = json.loads(raw_text)
    elif p.suffix == ".toml":
        raw = tomllib.loads(raw_text)
    else:
        raise ValueError(f"policy file must be .toml or .json, got {p.suffix!r}")

    default_raw = dict(raw.get("default", {}) or {})
    merged_fovs = {
        str(fov): {**default_raw, **(spec or {})}
        for fov, spec in (raw.get("fov", {}) or {}).items()
    }
    return PolicyFile(
        default=default_raw,
        fov=merged_fovs,
        placeholders_resolved=bool(raw.get("placeholders_resolved", True)),
    )


def _objective_spec(spec: PolicySpec) -> dict[str, Any]:
    """The inherited objective with this FOV's arm-varying pieces composed in.

    Only the kernel and the regularizer coefficients can be set this way — the
    reference comes from the inherited ``objective`` and is therefore identical
    across every FOV sharing a ``[default]``.
    """
    out = dict(spec.objective or {})
    if spec.kernel is not None:
        out["kernel"] = spec.kernel
    if spec.lambda_move is not None:
        out["lambda_move"] = spec.lambda_move
    if spec.lambda_dose is not None:
        out["lambda_dose"] = spec.lambda_dose
    return out


def arm_map(policy_file: PolicyFile) -> dict[int, int]:
    """``{fov: arm}`` — the declared ``arm`` of each FOV, or, failing that, FOVs
    grouped by identical resolved policy and numbered from 1 in FOV order.

    An arm used to be a comment-header convention rather than a field, so every
    consumer that wanted arm labels reinvented one, and they all reinvented the
    same wrong one: ``fov % 4 + 1``. That holds for ``policy_12fov_osc.toml`` and
    for neither 10-FOV file, whose layouts are interleaved so that arm is not
    confounded with stage position — mislabelling every per-arm row rather than
    failing. Declared arms end that.

    The derived fallback keeps files without declarations working, and gets the
    grouping right even when arms differ in ``objective`` rather than in
    ``controller`` (the pattern-zoo case). It cannot recover the file's own arm
    *numbering*, only its partition, which is why declaring is preferred.

    Half-declared is rejected: a file where some FOVs name an arm and others do
    not has no consistent numbering, and silently mixing declared ids with
    derived ones is the failure this function exists to remove.
    """
    fovs = sorted(policy_file.fov)
    declared = {f: policy_file.fov[f].arm for f in fovs}
    named = [f for f in fovs if declared[f] is not None]
    if named and len(named) != len(fovs):
        missing = [f for f in fovs if declared[f] is None]
        raise PolicyViolation(
            f"policy declares `arm` for FOVs {named} but not for {missing}. "
            f"Declare it for every FOV or for none — a partial labelling has no "
            f"consistent numbering."
        )
    if named:
        return {f: int(declared[f]) for f in fovs}

    seen: dict[str, int] = {}
    out: dict[int, int] = {}
    for fov in fovs:
        spec = policy_file.fov[fov]
        key = json.dumps(spec.model_dump(exclude_none=True), sort_keys=True)
        if key not in seen:
            seen[key] = len(seen) + 1
        out[fov] = seen[key]
    return out


class PolicyRouter:
    """Owns one engine per distinct policy and routes a FOV to its engine."""

    def __init__(self, cfg: ServerConfig, policy_file: PolicyFile | None = None):
        self.cfg = cfg
        self.policy_file = policy_file
        self._cache: dict = {}  # (checkpoint, device) -> ModelHandle
        if policy_file is not None and not policy_file.placeholders_resolved:
            raise PolicyViolation(
                "policy file has placeholders_resolved = false: it still carries "
                "working assumptions (dose ladder / setpoints / period / "
                "lambda_move / band half-width) rather than measured values. "
                "Fill them in from their stated sources and set "
                "placeholders_resolved = true to serve."
            )
        self.default_engine, self.default_info = self._build(
            policy_file.default if policy_file else PolicySpec(), label="default"
        )
        self.engines: dict[int, Any] = {}
        self.infos: dict[int, dict] = {}
        if policy_file:
            for fov, spec in policy_file.fov.items():
                self.engines[fov], self.infos[fov] = self._build(spec, label=f"fov {fov}")

    def _build(self, spec: PolicySpec, label: str):
        from optoerk.serving.runtime import load_engine

        cfg = self.cfg
        if spec.control_horizon is not None:
            cfg = replace(cfg, control_horizon=spec.control_horizon)
        levels = (
            np.asarray(spec.levels_ms, dtype=np.float64)
            if spec.levels_ms
            else dose_levels(cfg.min_exposure_ms, cfg.max_exposure_ms, cfg.n_candidates)
        )
        print(f"[serving] building policy for {label}")
        try:
            objective = (
                build_objective(_objective_spec(spec)) if spec.objective else None
            )
            controller = (
                build_controller(spec.controller, levels) if spec.controller else None
            )
        except PolicyViolation:
            raise  # a misconfigured experiment stops the server, it does not degrade
        except Exception as e:  # noqa: BLE001 - one bad FOV must not sink the server
            print(f"[serving] policy for {label} is invalid ({e!r}); using STUB")
            objective, controller = None, None
            cfg = replace(cfg, checkpoint_dir=None)
        checkpoint = spec.checkpoint if spec.checkpoint is not None else cfg.checkpoint_dir
        engine, info = load_engine(cfg, objective, controller, checkpoint, self._cache)
        # Record what this FOV was ASKED to run, not only what it managed to build.
        # A FOV that degraded to the stub previously logged `{"policy": "stub"}` and
        # nothing else, so the hold run's controller assignment had to be recovered
        # from the .toml afterwards. The requested spec makes the log self-contained.
        info = {
            **info,
            "label": label,
            "requested": spec.model_dump(exclude_none=True),
            "levels_ms": [float(x) for x in levels],
        }
        return engine, info

    def engine_for(self, fov: int):
        return self.engines.get(fov, self.default_engine)

    def info_for(self, fov: int) -> dict:
        return self.infos.get(fov, self.default_info)

    def describe(self) -> dict:
        """Serializable summary of every resolved policy — logged at startup and
        echoed by ``/info``, so a run's log records exactly what it ran."""
        return {
            "default": self.default_info,
            "fov": {str(k): v for k, v in self.infos.items()},
            "n_models_loaded": len(self._cache),
        }
