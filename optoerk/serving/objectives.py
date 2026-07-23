"""What the controller is *trying to achieve* — the goal, as a cost function.

Two layers, deliberately:

  * :class:`Objective` — the general seam. ``cost(pred_cnr, ctx) -> (N, M)`` scores
    every candidate dose (or dose *sequence*) for every cell; the controller takes
    the argmin. Anything is expressible here: setpoint tracking, banded targets,
    per-step weighting, dose penalties, cell-conditional behaviour.

  * :class:`TargetTrajectory` — the convenience layer covering the common case:
    supply ``target_fn(cell, t)`` (a setpoint, possibly varying over time and over
    the forecast horizon) and optionally ``gate_fn(cell, t)`` (may this cell be
    stimulated at all?). Cost is then squared error to the target, and gated-out
    cells are forced to zero dose. All the built-ins below are built on it.

**Units.** ``pred_cnr`` is handed to :meth:`Objective.cost` in **absolute CNR
units** (already denormalized by the engine), in whatever convention the loaded
checkpoint uses — ``cnr_median_norm`` for a ``cnr_mode="norm"`` model (resting
baseline == 1.0) or absolute ``cnr_median`` for a ``"raw"`` model. Objectives are
therefore written in human-readable CNR and never touch the z-score stats. The
server prints the resolved objective and the checkpoint's ``cnr_mode`` side by
side at startup so a mismatch is visible.

Gating vs cost: a gate is *not* folded into the cost. Making a cell's cost huge
still leaves the controller picking its least-bad nonzero dose; :meth:`allow_stim`
instead forces exposure to exactly 0, which is what "do not stimulate this cell"
means physically.

Built-ins are registered by name (``hold``, ``schedule``, ``gated``) so a policy
file can name them; see :mod:`optoerk.serving.policy`. Register custom objectives
with :func:`register` to make them nameable too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass
class GoalContext:
    """Everything an objective may condition on, for one FOV at one frame."""
    fov: int
    timestep: int
    cells: list  # list[CellFrame] — avoids a circular import with runtime


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------


class Objective:
    """Score candidate dose plans. Subclass and override :meth:`cost`."""

    name = "objective"

    def cost(self, pred_cnr: torch.Tensor, ctx: GoalContext) -> torch.Tensor:
        """``pred_cnr`` (N, M, H) predicted **absolute CNR** per cell / candidate /
        horizon step -> (N, M) cost. Lower is better."""
        raise NotImplementedError

    def allow_stim(self, ctx: GoalContext) -> torch.Tensor | None:
        """Optional (N,) bool mask: cells that may be stimulated at all. ``None``
        (the default) means every cell is eligible. Cells masked out are commanded
        exactly 0 ms regardless of what :meth:`cost` says."""
        return None

    def describe(self) -> dict[str, Any]:
        """Serializable summary, logged at startup and echoed by ``/info``."""
        return {"type": self.name}


# ---------------------------------------------------------------------------
# the convenience layer
# ---------------------------------------------------------------------------


class TargetTrajectory(Objective):
    """Track a setpoint: ``cost = mean_h (pred[h] - target[h])**2``.

    ``target_fn(cell, t)`` returns either a scalar (held across the horizon) or a
    sequence of length H, i.e. the goal *trajectory* starting at frame ``t``.
    ``gate_fn(cell, t)`` returns whether the cell may be stimulated this frame.

    Both callables see a ``CellFrame``, so they can condition on ``x``, ``y``,
    ``cnr_norm``, ``fov_density``, ``n_cells_200px`` and ``state.n_frames`` as well
    as on ``t``. They are called once per cell per frame (a few hundred calls
    against a 60 s budget) — write them for clarity, not speed.
    """

    name = "target_trajectory"

    def __init__(
        self,
        target_fn: Callable[[Any, int], float | Sequence[float]],
        gate_fn: Callable[[Any, int], bool] | None = None,
        *,
        name: str | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.target_fn = target_fn
        self.gate_fn = gate_fn
        if name is not None:
            self.name = name
        self._params = params or {}

    def targets(self, ctx: GoalContext, horizon: int, device) -> torch.Tensor:
        """(N, H) setpoint per cell per horizon step, in absolute CNR units."""
        rows = []
        for cell in ctx.cells:
            v = self.target_fn(cell, ctx.timestep)
            if isinstance(v, (int, float)):
                rows.append([float(v)] * horizon)
            else:
                vals = [float(x) for x in v]
                if len(vals) != horizon:
                    raise ValueError(
                        f"{self.name}: target_fn returned {len(vals)} values, "
                        f"expected a scalar or {horizon} (the control horizon)"
                    )
                rows.append(vals)
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def cost(self, pred_cnr: torch.Tensor, ctx: GoalContext) -> torch.Tensor:
        horizon = pred_cnr.shape[-1]
        tgt = self.targets(ctx, horizon, pred_cnr.device)   # (N, H)
        return ((pred_cnr - tgt.unsqueeze(1)) ** 2).mean(dim=-1)

    def allow_stim(self, ctx: GoalContext) -> torch.Tensor | None:
        if self.gate_fn is None:
            return None
        return torch.tensor(
            [bool(self.gate_fn(cell, ctx.timestep)) for cell in ctx.cells],
            dtype=torch.bool,
        )

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, **self._params}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

OBJECTIVES: dict[str, Callable[..., Objective]] = {}

def register(name: str) -> Callable:
    """Decorator: make a builder available to policy files under ``name``."""

    def deco(fn: Callable[..., Objective]) -> Callable[..., Objective]:
        if name in OBJECTIVES:
            raise ValueError(f"objective {name!r} already registered")
        OBJECTIVES[name] = fn
        return fn

    return deco


def build_objective(spec: dict[str, Any]) -> Objective:
    """Build an objective from a policy-file spec: ``{"type": ..., **params}``."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind is None:
        raise ValueError(f"objective spec needs a 'type' key; got {spec!r}")
    if kind not in OBJECTIVES:
        raise KeyError(
            f"unknown objective type {kind!r}; registered: {sorted(OBJECTIVES)}"
        )
    try:
        return OBJECTIVES[kind](**spec)
    except TypeError as e:
        raise TypeError(f"bad params for objective {kind!r}: {e}") from e


# ---------------------------------------------------------------------------
# built-ins
# ---------------------------------------------------------------------------


@register("hold")
def hold(target_cnr: float) -> TargetTrajectory:
    """Drive every cell to a fixed CNR and hold it. The historical behaviour."""
    return TargetTrajectory(
        lambda cell, t: target_cnr,
        name="hold",
        params={"target_cnr": target_cnr},
    )


@register("schedule")
def schedule(points: Sequence[Sequence[float]]) -> TargetTrajectory:
    """Piecewise-constant setpoint over time: ``points = [[t0, cnr0], [t1, cnr1], ...]``.

    The setpoint at frame ``t`` is the last point with ``t_point <= t`` (before the
    first point, the first point's value is used). The target is evaluated *per
    horizon step*, so the controller sees an upcoming step change and can start
    driving toward it before it lands — the point of having a trajectory rather
    than a scalar.
    """
    pts = sorted((float(t), float(v)) for t, v in points)
    if not pts:
        raise ValueError("schedule needs at least one [timestep, target_cnr] point")

    def value_at(t: float) -> float:
        out = pts[0][1]
        for t_pt, v in pts:
            if t >= t_pt:
                out = v
            else:
                break
        return out

    # Horizon length is unknown here; TargetTrajectory passes it via the (cell, t)
    # call, so return a closure over t only and let the base class broadcast.
    class _Schedule(TargetTrajectory):
        def targets(self, ctx, horizon, device):
            row = [value_at(ctx.timestep + h) for h in range(horizon)]
            return torch.tensor(
                [row] * len(ctx.cells), dtype=torch.float32, device=device
            )

    return _Schedule(
        lambda cell, t: value_at(t),
        name="schedule",
        params={"points": [list(p) for p in pts]},
    )


@register("gated")
def gated(
    target_cnr: float,
    after_t: int | None = None,
    before_t: int | None = None,
    x_gt: float | None = None,
    x_lt: float | None = None,
    y_gt: float | None = None,
    y_lt: float | None = None,
    max_neighbours_200px: int | None = None,
    min_neighbours_200px: int | None = None,
    min_frames_seen: int | None = None,
) -> TargetTrajectory:
    """Hold ``target_cnr``, but only stimulate cells passing every supplied
    predicate; the rest are commanded 0 ms.

    Example — "activate from t=10 only for cells right of x=512 with fewer than 5
    neighbours within 200 px"::

        {type = "gated", target_cnr = 1.8, after_t = 10, x_gt = 512,
         max_neighbours_200px = 5}

    Unset predicates are simply not applied. Cells whose position is missing
    (``x``/``y`` NaN) fail any position predicate, so an untracked cell is never
    stimulated by accident.
    """
    params = {
        "target_cnr": target_cnr, "after_t": after_t, "before_t": before_t,
        "x_gt": x_gt, "x_lt": x_lt, "y_gt": y_gt, "y_lt": y_lt,
        "max_neighbours_200px": max_neighbours_200px,
        "min_neighbours_200px": min_neighbours_200px,
        "min_frames_seen": min_frames_seen,
    }

    def gate(cell, t: int) -> bool:
        if after_t is not None and t < after_t:
            return False
        if before_t is not None and t >= before_t:
            return False
        if x_gt is not None and not (cell.x > x_gt):
            return False
        if x_lt is not None and not (cell.x < x_lt):
            return False
        if y_gt is not None and not (cell.y > y_gt):
            return False
        if y_lt is not None and not (cell.y < y_lt):
            return False
        if max_neighbours_200px is not None and cell.n_cells_200px > max_neighbours_200px:
            return False
        if min_neighbours_200px is not None and cell.n_cells_200px < min_neighbours_200px:
            return False
        if min_frames_seen is not None and cell.state.n_frames < min_frames_seen:
            return False
        return True

    return TargetTrajectory(
        lambda cell, t: target_cnr,
        gate,
        name="gated",
        params={k: v for k, v in params.items() if v is not None},
    )
