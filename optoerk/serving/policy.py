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

from pydantic import BaseModel, ConfigDict, Field

from optoerk.serving.config import ServerConfig
from optoerk.serving.control import build_controller, dose_levels
from optoerk.serving.objectives import build_objective


class PolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str | None = None
    objective: dict[str, Any] | None = None
    controller: dict[str, Any] | None = None
    # Per-FOV lookahead override. Still hard-capped by the checkpoint's future_len.
    control_horizon: int | None = None


class PolicyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: PolicySpec = Field(default_factory=PolicySpec)
    fov: dict[int, PolicySpec] = Field(default_factory=dict)


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
    return PolicyFile(default=default_raw, fov=merged_fovs)


class PolicyRouter:
    """Owns one engine per distinct policy and routes a FOV to its engine."""

    def __init__(self, cfg: ServerConfig, policy_file: PolicyFile | None = None):
        self.cfg = cfg
        self.policy_file = policy_file
        self._cache: dict = {}  # (checkpoint, device) -> ModelHandle
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
        levels = dose_levels(cfg.min_exposure_ms, cfg.max_exposure_ms, cfg.n_candidates)
        print(f"[serving] building policy for {label}")
        try:
            objective = build_objective(spec.objective) if spec.objective else None
            controller = (
                build_controller(spec.controller, levels) if spec.controller else None
            )
        except Exception as e:  # noqa: BLE001 - one bad FOV must not sink the server
            print(f"[serving] policy for {label} is invalid ({e!r}); using STUB")
            objective, controller = None, None
            cfg = replace(cfg, checkpoint_dir=None)
        checkpoint = spec.checkpoint if spec.checkpoint is not None else cfg.checkpoint_dir
        return load_engine(cfg, objective, controller, checkpoint, self._cache)

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
