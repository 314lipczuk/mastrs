"""What the controller is *trying to achieve* — the goal, as a cost function.

An objective is composed of three independent pieces, so that a new experiment
combines existing parts rather than adding a class:

  * :class:`Reference` — the target trajectory ``r_h`` over the horizon.
    :class:`ConstantReference`, :class:`ScheduleReference`, :class:`StepTrainReference`.
  * :class:`Kernel` — maps the predictive distribution at step ``h`` and the
    reference ``r_h`` to a scalar. :class:`L2Kernel`, :class:`BandKernel`.
  * :class:`Regularizer` — a function of the *dose plan alone*, independent of
    anything the model predicted. :class:`MovePenalty`, :class:`DosePenalty`.

The total cost of a candidate plan ``u[0..H-1]`` is::

    J(u) = (1/H) sum_h kernel(pi_h, mu_h, sigma_h, r_h)
         + lambda_move * (1/H) sum_h (u_h - u_{h-1})**2
         + lambda_dose * (1/H) sum_h u_h

where ``u`` is the **normalized** dose ``exposure_ms / max_exposure_ms`` (see
:class:`MovePenalty`) and ``u_{-1}`` is the dose actually applied on the previous
frame, carried in per-cell state.

Composition happens in :class:`Objective`; the controller only ever sees
``Objective.cost(pred, ctx) -> (N, M)`` and takes the argmin.

**Units.** :class:`Prediction` reaches :meth:`Objective.cost` in **absolute CNR
units** (already denormalized by the engine), in whatever convention the loaded
checkpoint uses — ``cnr_median_norm`` for a ``cnr_mode="norm"`` model (resting
baseline == 1.0) or absolute ``cnr_median`` for a ``"raw"`` model. This includes
the mixture parameters: ``mu`` and ``sigma`` are de-standardized by the engine
before they get here, because a band probability computed against a standardized
``mu`` with an absolute ``r`` is silently meaningless rather than obviously wrong.
Objectives are therefore written in human-readable CNR and never touch the z-score
stats. The server prints the resolved objective and the checkpoint's ``cnr_mode``
side by side at startup so a mismatch is visible.

Gating vs cost: a gate is *not* folded into the cost. Making a cell's cost huge
still leaves the controller picking its least-bad nonzero dose; :meth:`allow_stim`
instead forces exposure to exactly 0, which is what "do not stimulate this cell"
means physically.

Built-ins are registered by name (``hold``, ``schedule``, ``gated``,
``oscillation``) so a policy file can name them; see :mod:`optoerk.serving.policy`.
``hold`` is exactly ``constant`` + ``l2`` with both lambdas zero, so it remains
bit-for-bit the historical behaviour. Register custom objectives with
:func:`register` to make them nameable too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


class PolicyViolation(Exception):
    """A policy is internally inconsistent and the server must refuse to start.

    Distinct from a malformed spec (which degrades that one FOV to the stub):
    this means the experiment as configured would not measure what it claims to,
    e.g. an oscillation period the controller cannot see inside its horizon. It
    is deliberately *not* swallowed by the per-FOV degrade-to-stub handler.
    """


@dataclass
class GoalContext:
    """Everything an objective may condition on, for one FOV at one frame."""
    fov: int
    timestep: int
    cells: list  # list[CellFrame] — avoids a circular import with runtime


@dataclass
class Prediction:
    """What the plant predicts for a batch of candidate dose plans.

    Shapes are (N cells, M candidate plans, H horizon steps). ``cnr`` and the
    mixture parameters are in **absolute CNR units**; ``plan_norm`` / ``prev_norm``
    are **normalized dose** in [0, 1].

    ``pi`` / ``mu`` / ``sigma`` are (N, M, H, K) and are only populated when the
    objective asked for them (:attr:`Objective.needs_mixture`) — rolling the full
    mixture out costs allocations the L2 path has no use for.
    """

    cnr: torch.Tensor                       # (N, M, H) mixture mean, absolute CNR
    plan_norm: torch.Tensor | None = None   # (N, M, H) normalized dose in [0, 1]
    prev_norm: torch.Tensor | None = None   # (N,) normalized dose applied last frame
    pi: torch.Tensor | None = None          # (N, M, H, K) mixture weights
    mu: torch.Tensor | None = None          # (N, M, H, K) component means, abs CNR
    sigma: torch.Tensor | None = None       # (N, M, H, K) component sds, abs CNR

    def require_plan(self, who: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.plan_norm is None or self.prev_norm is None:
            raise ValueError(
                f"{who} needs the dose plan, but the controller did not build it. "
                f"The controller consults Objective.needs_plan to decide whether "
                f"to materialize it — if you added a regularizer, make sure it is "
                f"reachable from Objective.regularizers."
            )
        return self.plan_norm, self.prev_norm

    def require_mixture(self, who: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pi is None or self.mu is None or self.sigma is None:
            raise ValueError(
                f"{who} needs the predictive mixture, but the plant returned only "
                f"the mean. The controller consults Objective.needs_mixture to "
                f"decide whether to roll the mixture out — if you added a kernel, "
                f"give it needs_mixture = True."
            )
        return self.pi, self.mu, self.sigma


# ---------------------------------------------------------------------------
# references — the target trajectory r_h
# ---------------------------------------------------------------------------


class Reference:
    """The setpoint trajectory over the horizon, in absolute CNR units."""

    name = "reference"

    def values(self, ctx: GoalContext, horizon: int, device) -> torch.Tensor:
        """(N, H) setpoint per cell per horizon step."""
        raise NotImplementedError

    def annotate(self, ctx: GoalContext) -> list[dict[str, Any]]:
        """Per-cell record of what this reference was doing at ``ctx.timestep``.

        Emitted into the prediction log so the analysis never has to re-derive the
        waveform from parameters and get it subtly wrong. Default: just the value.
        """
        r = self.values(ctx, 1, torch.device("cpu"))
        return [{"r_t": float(r[i, 0])} for i in range(len(ctx.cells))]

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


class ConstantReference(Reference):
    """A fixed setpoint, held forever. ``hold``'s reference."""

    name = "constant"

    def __init__(self, target_cnr: float):
        self.target_cnr = float(target_cnr)

    def values(self, ctx, horizon, device):
        return torch.full(
            (len(ctx.cells), horizon), self.target_cnr,
            dtype=torch.float32, device=device,
        )

    def describe(self):
        return {"type": self.name, "target_cnr": self.target_cnr}


class ScheduleReference(Reference):
    """Piecewise-constant setpoint over time: ``points = [[t0, cnr0], ...]``.

    The setpoint at frame ``t`` is the last point with ``t_point <= t`` (before the
    first point, the first point's value is used). Evaluated *per horizon step*, so
    the controller sees an upcoming step change and can start driving toward it
    before it lands — the point of having a trajectory rather than a scalar.
    """

    name = "schedule"

    def __init__(self, points: Sequence[Sequence[float]]):
        self.points = sorted((float(t), float(v)) for t, v in points)
        if not self.points:
            raise ValueError("schedule needs at least one [timestep, target_cnr] point")

    def value_at(self, t: float) -> float:
        out = self.points[0][1]
        for t_pt, v in self.points:
            if t >= t_pt:
                out = v
            else:
                break
        return out

    def values(self, ctx, horizon, device):
        row = [self.value_at(ctx.timestep + h) for h in range(horizon)]
        return torch.tensor(
            [row] * len(ctx.cells), dtype=torch.float32, device=device
        )

    def describe(self):
        return {"type": self.name, "points": [list(p) for p in self.points]}


# segment labels, in the order they occur within one period.
SEGMENTS = ("low_hold", "rise", "high_hold", "fall")
SETTLE = "settle"


class StepTrainReference(Reference):
    """An oscillating **step train**: low hold -> linear rise -> high hold -> linear fall.

    A sine yields one blended error number. A step train decomposes into rise
    time, overshoot, hold error and fall time, which is what the diagnostics act
    on and what maps onto the cells' known asymmetry: rises are dose-limited,
    falls are decay-limited. Transitions are **linear** ramps, so "did it keep up"
    is a slope comparison rather than a shape fit.

    "Decay-limited" bounds one direction only. A fall *faster* than free decay is
    unreachable — the controller can only stop light and wait for CRY2 release plus
    ERK decay. A fall *slower* than free decay is fully controllable: the controller
    adds light to brake the descent. Nothing here enforces either, deliberately —
    see the note on removed guards below.

    **No feasibility guard.** This class used to carry a ``tau_decay_min`` parameter
    and refuse references whose fall or period were short relative to it. That
    number was hand-typed into the policy file alongside the durations it was
    checked against, so it could only ever catch self-contradiction, and lowering it
    legalized anything. Meanwhile ``high`` — the parameter most likely to put a
    reference out of reach — was never checked at all. Feasibility now lives in the
    pre-flight check, where it is argued against measured τ and reachable-ceiling
    distributions and recorded in the policy file, rather than asserted here.
    A reference that the cells cannot follow is therefore now expressible, which is
    what the border-probing experiments need.

    This is a *reference-tracking* target: the cost kernel is evaluated pointwise
    against ``r_h``. It is not waveform matching (match frequency and amplitude,
    phase-free) — that is what the later frequency-imposition experiment needs.
    :meth:`value_at` is a pure function of ``(t, params)`` with no controller state
    in it, so a waveform-matching objective can reuse it unchanged.

    All durations are in **minutes**; with the standard 1 frame = 1 min acquisition
    they are also frames, but the conversion is explicit everywhere (see
    ``frame_interval_min`` on :class:`~optoerk.serving.config.ServerConfig`).

    **Settle-in.** Cells start at resting CNR and the controller starts with no
    history, so the reference holds at ``low`` for ``settle_periods`` periods before
    the first rise. Those frames are labelled ``settle`` so the analysis can drop
    them; without it the first cycle mixes the initial transient with the tracking
    response and is uninterpretable.

    **Phase.** With ``n_phase_groups > 1`` a cell's reference is offset by
    ``(particle % n_phase_groups) * period / n_phase_groups``, fixed for its
    lifetime. This is a nuisance control, not a sweep: with aligned phase any
    global time-varying artifact (illumination drift, focus drift, evaporation,
    temperature cycling) is perfectly aliased with the reference and "the
    population follows the reference" becomes unfalsifiable. Offsetting by particle
    breaks the alias while staying perfectly reproducible under replay — an RNG
    here would not be. Analysis must align to each cell's own ``phase_offset_min``
    before pooling.
    """

    name = "step_train"

    def __init__(
        self,
        low: float,
        high: float,
        t_low_min: float,
        t_rise_min: float,
        t_high_min: float,
        t_fall_min: float,
        settle_periods: float = 2.0,
        n_phase_groups: int = 4,
        frame_interval_min: float = 1.0,
    ):
        self.low = float(low)
        self.high = float(high)
        self.t_low_min = float(t_low_min)
        self.t_rise_min = float(t_rise_min)
        self.t_high_min = float(t_high_min)
        self.t_fall_min = float(t_fall_min)
        self.settle_periods = float(settle_periods)
        self.n_phase_groups = int(n_phase_groups)
        self.frame_interval_min = float(frame_interval_min)

        for label, v in (
            ("t_low_min", self.t_low_min), ("t_rise_min", self.t_rise_min),
            ("t_high_min", self.t_high_min), ("t_fall_min", self.t_fall_min),
        ):
            if v < 0:
                raise ValueError(f"step_train: {label} must be >= 0, got {v}")
        if self.period_min <= 0:
            raise ValueError("step_train: all four segment durations are zero")
        if self.high <= self.low:
            raise ValueError(
                f"step_train: high ({self.high}) must exceed low ({self.low})"
            )
        if self.n_phase_groups < 1:
            raise ValueError(
                f"step_train: n_phase_groups must be >= 1, got {n_phase_groups}"
            )
        if self.frame_interval_min <= 0:
            raise ValueError("step_train: frame_interval_min must be > 0")

    @property
    def period_min(self) -> float:
        return self.t_low_min + self.t_rise_min + self.t_high_min + self.t_fall_min

    @property
    def settle_min(self) -> float:
        return self.settle_periods * self.period_min

    @property
    def amplitude(self) -> float:
        return self.high - self.low

    def phase_offset_min(self, cell) -> float:
        """This cell's fixed phase offset. Derived from its particle id, so a cell
        keeps the same offset for its whole lifetime and a replay reproduces it."""
        if self.n_phase_groups <= 1:
            return 0.0
        group = int(getattr(cell.state, "particle", 0)) % self.n_phase_groups
        return group * self.period_min / self.n_phase_groups

    def value_at(self, t_min: float, phase_offset_min: float = 0.0) -> float:
        """Reference at absolute time ``t_min``. Pure function of (t, params)."""
        return self._eval(t_min, phase_offset_min)[0]

    def segment_at(self, t_min: float, phase_offset_min: float = 0.0) -> str:
        """Which segment of the waveform ``t_min`` falls in."""
        return self._eval(t_min, phase_offset_min)[1]

    def _eval(self, t_min: float, phase_offset_min: float) -> tuple[float, str]:
        # Settle-in: hold at `low` and label it, so the analysis can exclude the
        # frames where the initial transient and the tracking response overlap.
        # The phase offset shifts where in the cycle the cell *starts*, so it is
        # applied inside the cycle rather than to the settle boundary — every cell
        # settles for the same wall-clock duration.
        if t_min < self.settle_min:
            return self.low, SETTLE
        u = (t_min - self.settle_min + phase_offset_min) % self.period_min

        a = self.t_low_min
        b = a + self.t_rise_min
        c = b + self.t_high_min
        if u < a:
            return self.low, "low_hold"
        if u < b:
            frac = (u - a) / self.t_rise_min if self.t_rise_min > 0 else 1.0
            return self.low + self.amplitude * frac, "rise"
        if u < c:
            return self.high, "high_hold"
        frac = (u - c) / self.t_fall_min if self.t_fall_min > 0 else 1.0
        return self.high - self.amplitude * frac, "fall"

    def values(self, ctx, horizon, device):
        dt = self.frame_interval_min
        # The row depends only on (timestep, phase offset), and there are at most
        # `n_phase_groups` distinct offsets — so compute one row per offset rather
        # than one per cell. Without this the reference is N x H Python calls on
        # the CEM's hot path (~19k per FOV-frame at 208 cells, H=30, 3 iters).
        # Cached per call, never across calls: ctx.timestep moves every frame.
        rows: dict[float, list[float]] = {}
        out = []
        for cell in ctx.cells:
            off = self.phase_offset_min(cell)
            row = rows.get(off)
            if row is None:
                row = [
                    self.value_at((ctx.timestep + h) * dt, off)
                    for h in range(horizon)
                ]
                rows[off] = row
            out.append(row)
        return torch.tensor(out, dtype=torch.float32, device=device)

    def annotate(self, ctx):
        dt = self.frame_interval_min
        out = []
        for cell in ctx.cells:
            off = self.phase_offset_min(cell)
            r, seg = self._eval(ctx.timestep * dt, off)
            out.append({"r_t": r, "segment": seg, "phase_offset_min": off})
        return out

    def describe(self):
        return {
            "type": self.name,
            "low": self.low, "high": self.high,
            "t_low_min": self.t_low_min, "t_rise_min": self.t_rise_min,
            "t_high_min": self.t_high_min, "t_fall_min": self.t_fall_min,
            "period_min": self.period_min,
            "settle_periods": self.settle_periods,
            "settle_min": self.settle_min,
            "n_phase_groups": self.n_phase_groups,
            "frame_interval_min": self.frame_interval_min,
        }


class FrequencyStaircaseReference(Reference):
    """A step train whose period steps down block by block, then repeats.

    This is the frequency-axis border probe. Each block is a
    :class:`StepTrainReference` run for a whole number of its own cycles; the
    blocks play in order and the whole sweep then loops for as long as the
    acquisition lasts.

    **Amplitude must shrink with period, and that is the design, not a
    compromise.** A fall from ``high`` to ``low`` toward a resting CNR takes
    ``tau * ln((high - rest) / (low - rest))`` of free decay, and the controller
    cannot beat it — there is no inhibitory actuator. At a fixed amplitude every
    period below ~50 min therefore demands an impossible fall, and the sweep would
    break at the first step down for a reason arithmetic already predicts. Instead
    each block demands the *largest* amplitude its own fall can deliver, so the
    fall is never the trivial binding constraint and what the sweep measures is
    the real thing: achieved amplitude divided by demanded amplitude, against
    frequency. That ratio is a Bode magnitude curve and its rolloff is the
    bandwidth. A ratio near 1 across the sweep says the first-order decay model
    was right; a ratio that falls says the cells are slower than it.

    **Blocks step in ``low`` as well as ``high``.** Raising the floor with
    frequency is what makes the fast falls reachable at all (decay from 1.1 to
    0.95 is far quicker than from 1.1 to 0.87), so ``low`` cannot be held fixed
    across blocks. The cost is a step in the reference at each block boundary,
    equal to the difference between adjacent blocks' lows. Because each block runs
    a whole number of its own cycles, every boundary sits at the bottom of a
    completed fall with a low hold immediately after — never mid-ramp. Going up
    the staircase those steps are small and upward (0.02-0.03 CNR, dose-limited
    and easy); the sweep wrap is one larger downward step back to the slowest
    block's low, which is decay-limited and must fit inside that block's low hold.

    **Phase** shifts the whole schedule rather than the position within a block,
    so each cell sees an unbroken waveform and its blocks simply land at different
    wall-clock times. The settle window stays common to every cell, as it is for
    :class:`StepTrainReference`. Analysis must group on the logged
    ``block_index`` / ``sweep_index``, never on wall-clock time.

    There is no ``n_sweeps``: the reference is periodic with period ``sweep_min``
    and the acquisition length decides how many sweeps happen. ``sweep_index`` is
    logged so the analysis can tell them apart — comparing a period's blocks
    between sweep 0 and sweep 1 is the drift-at-matched-frequency measurement.
    """

    name = "frequency_staircase"

    def __init__(
        self,
        blocks: Sequence[dict[str, Any]],
        settle_min: float = 0.0,
        n_phase_groups: int = 4,
        frame_interval_min: float = 1.0,
    ):
        if not blocks:
            raise ValueError("frequency_staircase: needs at least one block")
        self.settle_min = float(settle_min)
        self.n_phase_groups = int(n_phase_groups)
        self.frame_interval_min = float(frame_interval_min)
        if self.settle_min < 0:
            raise ValueError("frequency_staircase: settle_min must be >= 0")
        if self.n_phase_groups < 1:
            raise ValueError(
                f"frequency_staircase: n_phase_groups must be >= 1, got "
                f"{n_phase_groups}"
            )
        if self.frame_interval_min <= 0:
            raise ValueError("frequency_staircase: frame_interval_min must be > 0")

        self.n_cycles: list[int] = []
        self.refs: list[StepTrainReference] = []
        for i, block in enumerate(blocks):
            spec = dict(block)
            n = int(spec.pop("n_cycles", 1))
            if n < 1:
                raise ValueError(
                    f"frequency_staircase: block {i} has n_cycles={n}; a block must "
                    f"span a whole number of its own cycles so it starts and ends "
                    f"on a low hold"
                )
            self.n_cycles.append(n)
            # settle and phase live on the staircase, not inside a block.
            self.refs.append(
                StepTrainReference(
                    settle_periods=0.0, n_phase_groups=1,
                    frame_interval_min=self.frame_interval_min, **spec,
                )
            )

        self.block_min = [
            r.period_min * n for r, n in zip(self.refs, self.n_cycles)
        ]
        self.sweep_min = sum(self.block_min)
        self._starts: list[float] = []
        acc = 0.0
        for d in self.block_min:
            self._starts.append(acc)
            acc += d

    def phase_offset_min(self, cell) -> float:
        """Offset of this cell's whole schedule, from its particle id.

        Sized against the *first* block's period, so it is a phase of the slowest
        waveform in the sweep and stays small next to the block durations.
        """
        if self.n_phase_groups <= 1:
            return 0.0
        group = int(getattr(cell.state, "particle", 0)) % self.n_phase_groups
        return group * self.refs[0].period_min / self.n_phase_groups

    def _locate(
        self, t_min: float, phase_offset_min: float
    ) -> tuple[float, str, int, int]:
        """``(value, segment, block_index, sweep_index)`` at absolute ``t_min``.

        During settle the block and sweep indices are ``-1``: the schema stays
        fixed so the analysis never has to guess whether a key is missing or the
        frame was pre-sweep.
        """
        if t_min < self.settle_min:
            return self.refs[0].low, SETTLE, -1, -1
        u_total = t_min - self.settle_min + phase_offset_min
        sweep = int(u_total // self.sweep_min)
        u = u_total % self.sweep_min
        for i, (start, dur) in enumerate(zip(self._starts, self.block_min)):
            if u < start + dur:
                value, segment = self.refs[i]._eval(u - start, 0.0)
                return value, segment, i, sweep
        # Only reachable on a float boundary at the very end of the sweep.
        value, segment = self.refs[-1]._eval(u - self._starts[-1], 0.0)
        return value, segment, len(self.refs) - 1, sweep

    def value_at(self, t_min: float, phase_offset_min: float = 0.0) -> float:
        return self._locate(t_min, phase_offset_min)[0]

    def segment_at(self, t_min: float, phase_offset_min: float = 0.0) -> str:
        return self._locate(t_min, phase_offset_min)[1]

    def values(self, ctx, horizon, device):
        dt = self.frame_interval_min
        # One row per phase group, not per cell — same hot-path reason as
        # StepTrainReference.values.
        rows: dict[float, list[float]] = {}
        out = []
        for cell in ctx.cells:
            off = self.phase_offset_min(cell)
            row = rows.get(off)
            if row is None:
                row = [
                    self.value_at((ctx.timestep + h) * dt, off)
                    for h in range(horizon)
                ]
                rows[off] = row
            out.append(row)
        return torch.tensor(out, dtype=torch.float32, device=device)

    def annotate(self, ctx):
        dt = self.frame_interval_min
        out = []
        for cell in ctx.cells:
            off = self.phase_offset_min(cell)
            r, seg, block, sweep = self._locate(ctx.timestep * dt, off)
            out.append({
                "r_t": r, "segment": seg, "phase_offset_min": off,
                "block_index": block, "sweep_index": sweep,
                "block_period_min": (
                    self.refs[block].period_min if block >= 0 else None
                ),
            })
        return out

    def describe(self):
        return {
            "type": self.name,
            "settle_min": self.settle_min,
            "sweep_min": self.sweep_min,
            "n_phase_groups": self.n_phase_groups,
            "frame_interval_min": self.frame_interval_min,
            "blocks": [
                {
                    "period_min": r.period_min, "n_cycles": n,
                    "block_min": d,
                    "low": r.low, "high": r.high, "amplitude": r.amplitude,
                    "t_low_min": r.t_low_min, "t_rise_min": r.t_rise_min,
                    "t_high_min": r.t_high_min, "t_fall_min": r.t_fall_min,
                }
                for r, n, d in zip(self.refs, self.n_cycles, self.block_min)
            ],
        }


# ---------------------------------------------------------------------------
# kernels — predictive distribution + reference -> scalar
# ---------------------------------------------------------------------------


class Kernel:
    """Score one candidate plan's predicted trajectory against the reference."""

    name = "kernel"
    needs_mixture = False

    def cost(self, pred: Prediction, r: torch.Tensor) -> torch.Tensor:
        """``r`` is (N, H); returns (N, M). Lower is better."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


class L2Kernel(Kernel):
    """Mean squared error of the predictive **mean** against the reference."""

    name = "l2"

    def cost(self, pred, r):
        return ((pred.cnr - r.unsqueeze(1)) ** 2).mean(dim=-1)


class BandKernel(Kernel):
    """Probability of leaving a band of half-width ``delta`` around the reference::

        1 - sum_k pi_k [ Phi((r + d - mu_k)/sigma_k) - Phi((r - d - mu_k)/sigma_k) ]

    Scored under the full predictive mixture rather than its mean, so a plan that
    hits the reference on average but with wide spread is penalized where L2 would
    call it perfect. The band tracks a **time-varying** ``r_h``; it does not sit at
    a fixed level.

    ``mu`` and ``sigma`` arrive already de-standardized (absolute CNR), matching
    ``r`` and ``delta``. Computing this against standardized mixture parameters
    produces a plausible-looking but meaningless cost, so the de-standardization is
    done once in the engine rather than trusted to each kernel.

    Choosing ``delta``: it is not free. Too small and every plan scores ~1.0; too
    large and every plan scores ~0.0; either way the cost is flat and the band arm
    measures nothing. It must straddle the achievable tracking error — start near a
    third of the reference amplitude and check offline that scores actually spread
    across candidate plans.
    """

    name = "band"
    needs_mixture = True
    # Below this, a component is a delta function and the CDF difference is either
    # exactly 0 or exactly 1; the clamp keeps the division finite either way.
    SIGMA_FLOOR = 1e-6

    def __init__(self, half_width: float):
        self.half_width = float(half_width)
        if self.half_width <= 0:
            raise ValueError(
                f"band: half_width (delta) must be > 0, got {half_width}"
            )

    def cost(self, pred, r):
        pi, mu, sigma = pred.require_mixture("band kernel")
        r4 = r.unsqueeze(1).unsqueeze(-1)                    # (N, 1, H, 1)
        s = sigma.clamp(min=self.SIGMA_FLOOR)
        # ndtr is the standard-normal CDF; vectorized over
        # (cells x plans x horizon x components) — a Python loop over components
        # would not meet the CEM latency budget.
        hi = torch.special.ndtr((r4 + self.half_width - mu) / s)
        lo = torch.special.ndtr((r4 - self.half_width - mu) / s)
        p_in = (pi * (hi - lo)).sum(dim=-1)                  # (N, M, H)
        return (1.0 - p_in).mean(dim=-1)                     # (N, M)

    def describe(self):
        return {"type": self.name, "half_width": self.half_width}


KERNELS: dict[str, Callable[..., Kernel]] = {
    "l2": lambda **kw: L2Kernel(**kw),
    "band": lambda **kw: BandKernel(**kw),
}


def build_kernel(spec: str | dict[str, Any]) -> Kernel:
    """``"l2"`` or ``{type = "band", half_width = 0.05}``."""
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    kind = spec.pop("type", None)
    if kind not in KERNELS:
        raise KeyError(f"unknown kernel {kind!r}; known: {sorted(KERNELS)}")
    try:
        return KERNELS[kind](**spec)
    except TypeError as e:
        raise TypeError(f"bad params for kernel {kind!r}: {e}") from e


# ---------------------------------------------------------------------------
# regularizers — functions of the dose plan alone
# ---------------------------------------------------------------------------


class Regularizer:
    """A cost on the plan itself, independent of what the model predicted."""

    name = "regularizer"

    def cost(self, pred: Prediction) -> torch.Tensor:
        """(N, M). Lower is better."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


class MovePenalty(Regularizer):
    """``lambda_move * mean_h (u_h - u_{h-1})**2`` on the **normalized** dose.

    Suppresses slamming the dose between extremes on consecutive frames.

    **Why normalized.** ``u = exposure_ms / max_exposure_ms``, so ``u`` is in
    [0, 1] and ``du`` in [-1, 1]. This is not cosmetic: rebinning the ladder from
    0-800 ms to 0-150 ms would silently change an un-normalized penalty's meaning
    by ~28x. Normalized, ``lambda_move`` keeps its units (CNR^2 per unit^2
    normalized dose) and stays comparable across runs and ladders.

    **Why u_{-1} is the previously applied dose, not u[0].** Penalizing only the
    within-plan differences leaves the first move free, and the controller can
    still slam on every frame — it just never plans to do it twice in a row. The
    previous *applied* dose is carried in per-cell state; a cell's first frame and
    a freshly seeded daughter both take ``u_{-1} = 0`` (see
    :attr:`~optoerk.serving.state.CellState.last_applied_ms`).
    """

    name = "move_penalty"

    def __init__(self, lambda_move: float):
        self.lambda_move = float(lambda_move)

    def cost(self, pred):
        u, prev = pred.require_plan("move penalty")
        first = u[:, :, :1] - prev.view(-1, 1, 1)            # (N, M, 1)
        rest = u[:, :, 1:] - u[:, :, :-1]
        du = torch.cat([first, rest], dim=-1)
        return self.lambda_move * (du ** 2).mean(dim=-1)

    def describe(self):
        return {"type": self.name, "lambda_move": self.lambda_move}


class DosePenalty(Regularizer):
    """``lambda_dose * mean_h u_h`` on the normalized dose — a light budget."""

    name = "dose_penalty"

    def __init__(self, lambda_dose: float):
        self.lambda_dose = float(lambda_dose)

    def cost(self, pred):
        u, _prev = pred.require_plan("dose penalty")
        return self.lambda_dose * u.mean(dim=-1)

    def describe(self):
        return {"type": self.name, "lambda_dose": self.lambda_dose}


# ---------------------------------------------------------------------------
# the composite objective
# ---------------------------------------------------------------------------


class Objective:
    """Reference + kernel + regularizers, plus an optional stimulation gate.

    Regularizers with a zero coefficient are dropped at construction rather than
    added and multiplied by zero, so ``lambda_move = 0`` reproduces the pre-refactor
    cost expression exactly — arm 2 must be the existing controller, unchanged.
    """

    def __init__(
        self,
        reference: Reference,
        kernel: Kernel | None = None,
        regularizers: Sequence[Regularizer] = (),
        gate_fn: Callable[[Any, int], bool] | None = None,
        *,
        name: str = "objective",
        params: dict[str, Any] | None = None,
    ):
        self.reference = reference
        self.kernel = kernel if kernel is not None else L2Kernel()
        self.regularizers = list(regularizers)
        self.gate_fn = gate_fn
        self.name = name
        self._params = params or {}

    @property
    def needs_mixture(self) -> bool:
        """Whether the plant must roll out the full mixture, not just its mean."""
        return self.kernel.needs_mixture

    @property
    def needs_plan(self) -> bool:
        """Whether anything in this objective reads the dose plan.

        Only the regularizers do. With none of them (arms 1 and 2), the
        controller skips materializing the plan in ms and its normalized copy —
        two (N, M, H) tensors that would otherwise be built and never read.
        """
        return bool(self.regularizers)

    def targets(self, ctx: GoalContext, horizon: int, device) -> torch.Tensor:
        """(N, H) reference per cell per horizon step, in absolute CNR units."""
        return self.reference.values(ctx, horizon, device)

    def cost(self, pred: Prediction, ctx: GoalContext) -> torch.Tensor:
        r = self.targets(ctx, pred.cnr.shape[-1], pred.cnr.device)
        total = self.kernel.cost(pred, r)
        for reg in self.regularizers:
            total = total + reg.cost(pred)
        return total

    def allow_stim(self, ctx: GoalContext) -> torch.Tensor | None:
        """Optional (N,) bool mask: cells that may be stimulated at all. ``None``
        (the default) means every cell is eligible. Cells masked out are commanded
        exactly 0 ms regardless of what :meth:`cost` says."""
        if self.gate_fn is None:
            return None
        return torch.tensor(
            [bool(self.gate_fn(cell, ctx.timestep)) for cell in ctx.cells],
            dtype=torch.bool,
        )

    def annotate(self, ctx: GoalContext) -> list[dict[str, Any]]:
        """Per-cell reference annotations for the prediction log (``r_t``, and for
        an oscillating reference also ``segment`` and ``phase_offset_min``)."""
        return self.reference.annotate(ctx)

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "reference": self.reference.describe(),
            "kernel": self.kernel.describe(),
            "regularizers": [r.describe() for r in self.regularizers],
            **self._params,
        }


def _regularizers(lambda_move: float, lambda_dose: float) -> list[Regularizer]:
    """Build only the penalties that are actually switched on."""
    out: list[Regularizer] = []
    if lambda_move:
        out.append(MovePenalty(lambda_move))
    if lambda_dose:
        out.append(DosePenalty(lambda_dose))
    return out


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
def hold(
    target_cnr: float,
    kernel: str | dict[str, Any] = "l2",
    lambda_move: float = 0.0,
    lambda_dose: float = 0.0,
) -> Objective:
    """Drive every cell to a fixed CNR and hold it.

    With the defaults (``l2``, both lambdas zero) this is exactly the historical
    behaviour: ``constant`` reference, squared error to it, no plan-side penalties.
    """
    return Objective(
        ConstantReference(target_cnr),
        build_kernel(kernel),
        _regularizers(lambda_move, lambda_dose),
        name="hold",
        params={"target_cnr": target_cnr},
    )


@register("schedule")
def schedule(
    points: Sequence[Sequence[float]],
    kernel: str | dict[str, Any] = "l2",
    lambda_move: float = 0.0,
    lambda_dose: float = 0.0,
) -> Objective:
    """Piecewise-constant setpoint over time; see :class:`ScheduleReference`."""
    return Objective(
        ScheduleReference(points),
        build_kernel(kernel),
        _regularizers(lambda_move, lambda_dose),
        name="schedule",
    )


@register("oscillation")
def oscillation(
    low: float,
    high: float,
    t_low_min: float,
    t_rise_min: float,
    t_high_min: float,
    t_fall_min: float,
    settle_periods: float = 2.0,
    n_phase_groups: int = 4,
    frame_interval_min: float = 1.0,
    kernel: str | dict[str, Any] = "l2",
    lambda_move: float = 0.0,
    lambda_dose: float = 0.0,
) -> Objective:
    """Track an oscillating step train; see :class:`StepTrainReference`.

    When arms vary the *controller*, the reference is a property of the experiment
    rather than of an arm: every arm must configure it identically and vary only
    the kernel and the regularizer coefficients. When arms vary the *waveform* —
    as the pattern-zoo run does — the reference is what the arm is, and the
    controller fields are what must be held identical instead.
    """
    return Objective(
        StepTrainReference(
            low=low, high=high,
            t_low_min=t_low_min, t_rise_min=t_rise_min,
            t_high_min=t_high_min, t_fall_min=t_fall_min,
            settle_periods=settle_periods,
            n_phase_groups=n_phase_groups,
            frame_interval_min=frame_interval_min,
        ),
        build_kernel(kernel),
        _regularizers(lambda_move, lambda_dose),
        name="oscillation",
    )


@register("frequency_staircase")
def frequency_staircase(
    blocks: Sequence[dict[str, Any]],
    settle_min: float = 0.0,
    n_phase_groups: int = 4,
    frame_interval_min: float = 1.0,
    kernel: str | dict[str, Any] = "l2",
    lambda_move: float = 0.0,
    lambda_dose: float = 0.0,
) -> Objective:
    """Sweep the reference frequency block by block; see
    :class:`FrequencyStaircaseReference`.

    Each block sets its own ``low``/``high`` alongside its segment durations,
    because the reachable amplitude falls with the period. Sizing those is the
    pre-flight check's job, not this builder's.
    """
    return Objective(
        FrequencyStaircaseReference(
            blocks=blocks,
            settle_min=settle_min,
            n_phase_groups=n_phase_groups,
            frame_interval_min=frame_interval_min,
        ),
        build_kernel(kernel),
        _regularizers(lambda_move, lambda_dose),
        name="frequency_staircase",
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
) -> Objective:
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

    return Objective(
        ConstantReference(target_cnr),
        L2Kernel(),
        gate_fn=gate,
        name="gated",
        params={k: v for k, v in params.items() if v is not None},
    )
