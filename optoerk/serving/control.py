"""Controllers: turn "what CNR do we want" into "how many ms of light now".

The model predicts CNR *given* a future dose; faro needs a commanded dose, so the
controller inverts the model by searching over dose plans and scoring them with an
:class:`~optoerk.serving.objectives.Objective`.

Two controllers behind one ``solve()`` seam:

  * :class:`ConstantDoseSearch` — the historical behaviour. Scores a **constant**
    dose held across the whole horizon and commands the best one. Cheap, and a
    useful baseline to A/B against, but it is not MPC: it cannot express "a big
    pulse now, then nothing".

  * :class:`SequenceMPC` — real receding-horizon control. Optimizes a dose
    *sequence* ``u[0..H-1]`` with the cross-entropy method over the discrete DMD
    level set, then **applies only ``u[0]``** and re-plans next frame. Exhaustive
    search is ``L**H`` (5**10 ≈ 10M per cell) so the search is sampled; the
    constant-dose plans are always injected into the sample set, which makes
    SequenceMPC provably no worse than ConstantDoseSearch at equal horizon.

  * :class:`StaggeredCadenceMPC` — SequenceMPC constrained to stimulate each cell
    every ``k`` frames, with cells split into ``k`` phase-staggered groups so only
    ~1/k of them run the search on any frame (the rest coast). Enforces a fixed
    stimulation cadence and shaves the per-frame compute peak by ~k×.

Both are given a *plant* — anything exposing ``horizon``, ``device``,
``std_fluence(ms)``, ``rollout(h, c, cnr_fb, fut)`` and ``denorm_cnr(std)``. That
is :class:`~optoerk.serving.runtime.RealModelEngine`, kept behind an interface so
controllers stay testable against a toy plant.

The controller returns exposures in **ms**; the caller is responsible for the
``u_t`` bookkeeping (persisting the *applied* dose, i.e. ``u[0]``, as the fluence
fed at the next encoder step).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from optoerk.serving.objectives import GoalContext, Objective, Prediction


def dose_levels(min_ms: float, max_ms: float, n: int) -> np.ndarray:
    """The commandable exposure grid (ms). Evenly spaced, always including 0 =
    do-not-stimulate when ``min_ms`` is 0."""
    if n < 1:
        raise ValueError(f"need at least one dose level, got {n}")
    return np.linspace(float(min_ms), float(max_ms), int(n))


class Controller:
    name = "controller"

    def plan(self, plant, h, c, cnr_fb, objective: Objective, ctx: GoalContext):
        """``h``/``c``: encoder state (L, N, H_hidden) *after* this frame's advance.
        ``cnr_fb``: (N, 1) standardized CNR at the last real frame.

        Returns ``(ms, cost)``: the (N,) exposure to apply *now* and the (N,) cost
        of the plan it came from. The cost is what makes controllers comparable —
        tests assert that MPC never scores worse than the constant-dose search.
        """
        raise NotImplementedError

    def solve(self, plant, h, c, cnr_fb, objective: Objective, ctx: GoalContext) -> torch.Tensor:
        """The commanded (N,) exposure in ms, with the objective's gate applied."""
        ms, _cost = self.plan(plant, h, c, cnr_fb, objective, ctx)
        return self.apply_gate(ms, objective, ctx)

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}

    @property
    def max_ms(self) -> float:
        """The ladder's top rung, used to normalize the dose for the plan-side
        regularizers.

        Deliberately the **ladder** max rather than ``cfg.max_exposure_ms``: it is
        what keeps ``lambda_move`` meaning the same thing after the ladder is
        rebinned (0-800 ms -> 0-150 ms would otherwise rescale the penalty ~28x).
        An all-zero ladder normalizes by 1.0 rather than dividing by zero; every
        plan is then 0 and the penalties vanish, which is correct.
        """
        m = float(np.max(np.abs(self.levels_ms)))
        return m if m > 0 else 1.0

    # -- shared helpers ----------------------------------------------------

    def _prev_norm(self, ctx: GoalContext, device) -> torch.Tensor:
        """(N,) the normalized dose actually applied to each cell last frame.

        This is ``u_{-1}`` for :class:`~optoerk.serving.objectives.MovePenalty`.
        Read from per-cell state rather than taken as ``u[0]``, so the first move
        of a plan is not free. A cell's first frame and a freshly seeded daughter
        both report 0.0 — see ``CellState.last_applied_ms``.
        """
        return torch.tensor(
            [float(f.state.last_applied_ms) / self.max_ms for f in ctx.cells],
            dtype=torch.float32, device=device,
        )

    @torch.no_grad()
    def _score(self, plant, h, c, cnr_fb, fut_ms_std, plan_ms, objective, ctx) -> torch.Tensor:
        """Evaluate a batch of dose plans.

        ``fut_ms_std``: (N, S, H) standardized fluence per cell / plan / step.
        ``plan_ms``:    (N, S, H) the same plans in raw exposure ms, which the
        plan-side regularizers need (the standardized fluence is not a dose scale
        they can normalize). ``None`` when the objective has no regularizers.

        Returns (N, S) cost from the objective. Predictions are denormalized to
        absolute CNR first — objectives are written in CNR units, never z-scores.

        Two things are built only when something actually reads them: the
        predictive **mixture** (``needs_mixture``, i.e. a distributional kernel)
        and the **dose plan** (``needs_plan``, i.e. any regularizer). An
        objective with neither — ``constant`` + ``l2``, the pre-refactor
        controller — allocates nothing beyond what it always did.
        """
        N, S, H = fut_ms_std.shape
        h_b = h.repeat_interleave(S, dim=1)              # (L, N*S, H_hidden)
        c_b = c.repeat_interleave(S, dim=1)
        fb_b = cnr_fb.repeat_interleave(S, dim=0)        # (N*S, 1)
        fut_b = fut_ms_std.reshape(N * S, H, 1)

        pi = mu = sigma = None
        if objective.needs_mixture:
            pred_std, pi_s, mu_s, sigma_s = plant.rollout_mixture(h_b, c_b, fb_b, fut_b)
            K = pi_s.shape[-1]
            pi = pi_s.view(N, S, H, K)
            # De-standardize the components into absolute CNR, the units the
            # reference and the band half-width are written in.
            mu = plant.denorm_cnr(mu_s).view(N, S, H, K)
            sigma = plant.denorm_sigma(sigma_s).view(N, S, H, K)
        else:
            pred_std = plant.rollout(h_b, c_b, fb_b, fut_b)  # (N*S, H) standardized

        plan_norm = prev_norm = None
        if plan_ms is not None:
            plan_norm = plan_ms / self.max_ms
            prev_norm = self._prev_norm(ctx, fut_ms_std.device)

        pred = Prediction(
            cnr=plant.denorm_cnr(pred_std).view(N, S, H),
            plan_norm=plan_norm, prev_norm=prev_norm,
            pi=pi, mu=mu, sigma=sigma,
        )
        return objective.cost(pred, ctx)                 # (N, S)

    @staticmethod
    def apply_gate(ms: torch.Tensor, objective: Objective, ctx: GoalContext) -> torch.Tensor:
        """Force gated-out cells to exactly 0 ms.

        Public because the engine calls :meth:`plan` directly when it needs the
        plan's cost for the log, and must then gate the result itself — the same
        two steps :meth:`solve` bundles.
        """
        mask = objective.allow_stim(ctx)
        if mask is None:
            return ms
        return torch.where(mask.to(ms.device), ms, torch.zeros_like(ms))


class ConstantDoseSearch(Controller):
    """Score each constant dose over the horizon; command the best."""

    name = "constant_dose"

    def __init__(self, levels_ms: np.ndarray):
        self.levels_ms = np.asarray(levels_ms, dtype=np.float64)

    def plan(self, plant, h, c, cnr_fb, objective, ctx):
        N = h.shape[1]
        levels = torch.as_tensor(self.levels_ms, dtype=torch.float32, device=plant.device)
        M, H = levels.shape[0], plant.horizon
        std = plant.std_fluence(levels)                          # (M,)
        fut = std.view(1, M, 1).expand(N, M, H)                  # constant over H
        plan_ms = (
            levels.view(1, M, 1).expand(N, M, H) if objective.needs_plan else None
        )
        cost = self._score(plant, h, c, cnr_fb, fut, plan_ms, objective, ctx)
        best_cost, best = cost.min(dim=1)                        # (N,)
        return levels[best], best_cost

    def describe(self) -> dict[str, Any]:
        return {"type": self.name, "levels_ms": self.levels_ms.tolist()}


class SequenceMPC(Controller):
    """Receding-horizon MPC: optimize ``u[0..H-1]``, apply ``u[0]``, re-plan.

    Cross-entropy method over the discrete level set. Each iteration samples
    ``n_samples`` sequences per cell from a per-cell, per-step categorical
    distribution, keeps the ``n_elite`` cheapest, and refits the distribution to
    the elites (with ``smoothing`` mixed back toward uniform so it never collapses
    to a degenerate plan on iteration 1).

    The ``L`` constant-dose sequences are injected into every iteration's sample
    set, so the returned plan is always at least as good as the best constant dose
    — MPC can only improve on the old controller, never regress.

    **On ``n_samples``.** Each cell samples its own plans, so two cells with
    identical inputs can be commanded different doses purely from sampling noise.
    Measured on the multilen checkpoint at H=10 with 8 identical cells: at S=32
    the commanded dose splits 600/800 across them, at S=128 it still splits, and
    only by S=512 do all eight agree on 600. The default is therefore deliberately
    generous — inference costs ~0.1 s of a 60 s frame budget, so buying determinism
    with samples is nearly free. Lower it only if a benchmark says you must.
    """

    name = "sequence_mpc"

    def __init__(
        self,
        levels_ms: np.ndarray,
        n_samples: int = 512,
        n_iters: int = 3,
        elite_frac: float = 0.125,
        smoothing: float = 0.1,
        seed: int = 0,
    ):
        self.levels_ms = np.asarray(levels_ms, dtype=np.float64)
        self.n_samples = int(n_samples)
        self.n_iters = int(n_iters)
        self.elite_frac = float(elite_frac)
        self.smoothing = float(smoothing)
        self.seed = int(seed)

    def plan(self, plant, h, c, cnr_fb, objective, ctx):
        return self._cem(plant, h, c, cnr_fb, objective, ctx, stim_mask=None)

    def _cem(self, plant, h, c, cnr_fb, objective, ctx, stim_mask=None):
        """The CEM optimizer over a subset of cells with an optional horizon mask.

        ``stim_mask`` (H,) bool marks which horizon offsets are *stimulable*; the
        rest are forced to the 0 ms level for every plan, so the search optimizes
        only the doses it can actually apply and correctly predicts the un-lit
        "coast" steps in between. ``None`` = every step stimulable (plain MPC).
        Factored out so :class:`StaggeredCadenceMPC` can reuse it on a due subset.
        """
        device = plant.device
        N, H = h.shape[1], plant.horizon
        levels = torch.as_tensor(self.levels_ms, dtype=torch.float32, device=device)
        L = levels.shape[0]
        std_levels = plant.std_fluence(levels)                      # (L,)

        # Index of the 0 ms level, used to force coast steps dark. Only needed when
        # masking; StaggeredCadenceMPC guarantees a zero level exists in __init__.
        non_stim = None if stim_mask is None else ~stim_mask.to(device)
        zero_idx = None if stim_mask is None else int(torch.argmin(levels.abs()).item())

        # Deterministic per-call RNG: the same frame always yields the same plan,
        # which is what makes a replay reproduce a recorded run exactly.
        gen = torch.Generator(device="cpu").manual_seed(self.seed + ctx.timestep)

        # The L constant-dose plans, always evaluated.
        const_idx = (
            torch.arange(L, device=device).view(1, L, 1).expand(N, L, H)
        )

        probs = torch.full((N, H, L), 1.0 / L, device=device)
        n_elite = max(2, int(round(self.n_samples * self.elite_frac)))
        best_cost = torch.full((N,), float("inf"), device=device)
        best_first = torch.zeros(N, dtype=torch.long, device=device)

        for _ in range(self.n_iters):
            # multinomial wants 2D (rows, categories); sample per (cell, step).
            flat = probs.reshape(N * H, L).cpu()
            draw = torch.multinomial(flat, self.n_samples, replacement=True, generator=gen)
            idx = draw.to(device).view(N, H, self.n_samples).permute(0, 2, 1)  # (N,S,H)
            idx = torch.cat([const_idx, idx], dim=1)                          # (N,S+L,H)

            # Force the non-stimulable horizon steps to the 0 ms level for EVERY
            # plan (sampled and constant alike), so the coast between ticks is
            # modelled as dark and the "never worse than constant" guarantee holds
            # within the cadence-constrained problem.
            if non_stim is not None:
                idx[:, :, non_stim] = zero_idx

            fut = std_levels[idx]                                             # (N,S+L,H)
            # Second gather only when a regularizer will read it.
            plan_ms = levels[idx] if objective.needs_plan else None            # (N,S+L,H)
            cost = self._score(
                plant, h, c, cnr_fb, fut, plan_ms, objective, ctx
            )                                                                 # (N,S+L)

            # Track the global best plan's first action across all iterations.
            it_cost, it_arg = cost.min(dim=1)
            better = it_cost < best_cost
            best_cost = torch.where(better, it_cost, best_cost)
            first = idx[torch.arange(N, device=device), it_arg, 0]
            best_first = torch.where(better, first, best_first)

            # Refit the sampling distribution to the elites.
            elite = cost.topk(n_elite, dim=1, largest=False).indices          # (N,E)
            elite_idx = torch.gather(
                idx, 1, elite.unsqueeze(-1).expand(-1, -1, H)
            )                                                                 # (N,E,H)
            counts = torch.zeros(N, H, L, device=device)
            counts.scatter_add_(
                2,
                elite_idx.permute(0, 2, 1),                                   # (N,H,E)
                torch.ones(N, H, elite_idx.shape[1], device=device),
            )
            probs = counts / counts.sum(dim=2, keepdim=True).clamp(min=1.0)
            probs = (1 - self.smoothing) * probs + self.smoothing / L

        return levels[best_first], best_cost

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "levels_ms": self.levels_ms.tolist(),
            "n_samples": self.n_samples,
            "n_iters": self.n_iters,
            "elite_frac": self.elite_frac,
            "smoothing": self.smoothing,
            "seed": self.seed,
        }


class StaggeredCadenceMPC(SequenceMPC):
    """MPC that stimulates each cell every ``k`` frames, with the cells split into
    ``k`` phase-staggered groups so only ~1/k of them plan on any given frame.

    A cell's phase is ``particle % k`` (fixed for its lifetime), and it is *due*
    on frames where ``timestep % k == phase``. On a frame, only the due group runs
    the (expensive) CEM search — everyone else returns 0 ms this frame but was
    still integrated into the encoder upstream in ``decide()``. This does two
    things at once:

      * **Cadence.** Every cell is stimulated exactly every k frames. The horizon
        mask forces the un-lit coast steps dark, so the plan is optimized for the
        schedule the cell can actually follow (not a fictitious every-frame one).

      * **Peak-shaving.** Un-staggered, all cells plan on the same k-tick frames
        and idle otherwise — bursty, k× peak load. Staggered, the load is uniform
        at ~N/k planning cells per frame. Same total work, k× lower peak, which is
        the budget you can spend on a larger ``n_samples`` or a longer horizon
        (the latter still capped by the checkpoint's ``future_len``).

    Because every due cell shares the current frame as its own tick, they all
    share one horizon mask (offsets ``i % k == 0``), so the due subset is a single
    clean CEM call. Planning is per-cell independent (the model predicts each
    cell's CNR from its own state and its own future dose), so staggering the
    *timing* costs nothing in per-cell control quality.

    Requires a 0 ms level in the dose grid (``min_exposure_ms == 0``) — without a
    representable zero dose there is no way to express a coast frame. For any
    lookahead across a cell's *next* tick you also want ``control_horizon >= k``;
    with ``control_horizon < k`` only the current tick is in view and it degrades
    to single-pulse-with-decay planning (still correct, just myopic).
    """

    name = "staggered_mpc"

    def __init__(self, levels_ms, k: int, **kwargs):
        super().__init__(levels_ms, **kwargs)
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"staggered_mpc needs k >= 1, got {k}")
        if not bool(np.any(np.abs(self.levels_ms) < 1e-9)):
            raise ValueError(
                "staggered_mpc needs a 0 ms level (set min_exposure_ms=0) to "
                "represent the un-lit coast frames between stimulations"
            )

    def plan(self, plant, h, c, cnr_fb, objective, ctx):
        device = plant.device
        N, H = h.shape[1], plant.horizon
        ms = torch.zeros(N, device=device)
        cost = torch.full((N,), float("inf"), device=device)

        # Due group: cells whose fixed phase (particle % k) matches this frame.
        due = torch.tensor(
            [f.state.particle % self.k == ctx.timestep % self.k for f in ctx.cells],
            dtype=torch.bool, device=device,
        )
        if not bool(due.any()):
            return ms, cost  # idle frame for every cell here — no rollout at all

        due_idx = torch.nonzero(due, as_tuple=False).flatten()
        sub_cells = [ctx.cells[j] for j in due_idx.tolist()]
        sub_ctx = GoalContext(fov=ctx.fov, timestep=ctx.timestep, cells=sub_cells,
                              control_frame=ctx.control_frame)

        # Every due cell has its tick at offset 0, so they share one mask: the
        # stimulable future offsets are i % k == 0 (this tick, then every k-th).
        offsets = torch.arange(H, device=device)
        stim_mask = offsets % self.k == 0

        ms_d, cost_d = self._cem(
            plant, h[:, due_idx, :], c[:, due_idx, :], cnr_fb[due_idx],
            objective, sub_ctx, stim_mask=stim_mask,
        )
        ms[due_idx] = ms_d
        cost[due_idx] = cost_d
        return ms, cost

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "type": self.name, "k": self.k}


class OpenLoopController(Controller):
    """The control arm: a fixed dose sequence, identical for every cell, no feedback.

    This exists to be *compared against*. Every other controller here reads the
    cell's own encoder state and takes a per-cell ``argmin``, so an experiment using
    them cannot say whether the result came from closed-loop control or merely from
    delivering light. This one delivers the same predetermined dose to every cell on
    every frame, which makes the contrast a clean one-variable comparison: same
    model, same objective, same ladder, same reference — only feedback removed.

    ``sequence_ms`` is designed OFFLINE against the same model and objective (see
    ``experiments/open_loop_design.py``), so the arm is "the best open-loop schedule
    the model can produce", not a strawman. It is optimised over one period of the
    reference and tiled, since a periodic reference has a periodic optimal
    open-loop input.

    **Indexed on ``ctx.control_frame``, never ``ctx.timestep``.** The sequence has
    to line up with the reference waveform, and the waveform is clocked from the
    first controlled frame — faro's timestep starts wherever the acquisition's
    earlier phases left off. Reading the wrong one would slide the whole schedule
    against the reference it was designed for.

    The model still runs (``_score`` with a single candidate) even though nothing is
    chosen. That is deliberate: it keeps ``plan_cost`` on the same scale as the
    closed-loop arms, and it turns every frame of the open-loop arm into a free
    one-step model-error measurement under a known dose — which is exactly the
    quantity that says whether the model, rather than the controller, is the thing
    that is wrong.
    """

    name = "open_loop"

    def __init__(
        self,
        levels_ms: np.ndarray,
        sequence_ms: Sequence[float],
        repeat: bool = True,
    ):
        self.levels_ms = np.asarray(levels_ms, dtype=np.float64)
        self.sequence_ms = np.asarray(list(sequence_ms), dtype=np.float64)
        self.repeat = bool(repeat)
        if self.sequence_ms.size == 0:
            raise ValueError("open_loop needs a non-empty sequence_ms")
        if np.any(self.sequence_ms < 0):
            raise ValueError(f"open_loop sequence_ms has negative doses: {sequence_ms}")
        # Not a hard error — a designed sequence may legitimately interpolate
        # between rungs if the hardware accepts it — but a sequence off the ladder
        # makes the arms incomparable on dose, so it must not pass silently.
        off = [
            float(v) for v in self.sequence_ms
            if not np.any(np.isclose(v, self.levels_ms))
        ]
        if off:
            print(
                f"[serving] NOTE open_loop sequence uses doses outside the ladder: "
                f"{sorted(set(off))} not in {self.levels_ms.tolist()}. The closed-loop "
                f"arms can only choose ladder rungs, so the arms are not searching "
                f"the same dose set."
            )

    def dose_at(self, control_frame: int) -> float:
        """The dose this schedule commands at ``control_frame``, for every cell.

        Past the end of a non-repeating sequence the schedule holds its last value
        rather than going dark: an experiment that outruns its designed sequence
        should degrade to a constant, not silently stop stimulating.
        """
        i = int(control_frame)
        n = self.sequence_ms.size
        return float(self.sequence_ms[i % n] if self.repeat else self.sequence_ms[min(i, n - 1)])

    def plan(self, plant, h, c, cnr_fb, objective, ctx):
        N, H = h.shape[1], plant.horizon
        device = plant.device
        ms = float(self.dose_at(ctx.control_frame))

        # The plan this arm is committed to over the horizon: the next H entries of
        # the schedule. Scored, never searched — `cost` is what the model believed
        # would happen, logged so the arms are comparable.
        future = torch.tensor(
            [self.dose_at(ctx.control_frame + i) for i in range(H)],
            dtype=torch.float32, device=device,
        )                                                    # (H,)
        fut = plant.std_fluence(future).view(1, 1, H).expand(N, 1, H)
        plan_ms = future.view(1, 1, H).expand(N, 1, H) if objective.needs_plan else None
        cost = self._score(plant, h, c, cnr_fb, fut, plan_ms, objective, ctx)  # (N, 1)

        return torch.full((N,), ms, device=device), cost[:, 0]

    def describe(self) -> dict[str, Any]:
        # The full sequence goes in the startup record. A schedule that is only
        # named and not written down cannot be reconstructed from the log, and this
        # arm IS its schedule.
        return {
            "type": self.name,
            "levels_ms": self.levels_ms.tolist(),
            "sequence_ms": self.sequence_ms.tolist(),
            "repeat": self.repeat,
            "period_frames": int(self.sequence_ms.size),
            "mean_ms": float(self.sequence_ms.mean()),
        }


class PopulationMPC(SequenceMPC):
    """One dose for a whole group of cells, planned for a typical member of it.

    The comparison arm for per-cell control, and the reason it is worth having: an
    open-loop arm removes feedback AND the ability to treat cells differently at the
    same time, so when it loses you cannot say which of the two mattered. This
    removes exactly one thing. Same model, same objective, same ladder, same horizon,
    same optimizer — the group's dose still answers to what the cells are doing, frame
    by frame. What is gone is only the freedom to send two cells different doses.

    It also needs no recordings and no donor cells, so nothing in it depends on a
    second cell staying tracked.

    ``share``:
      ``"all"``   every cell in the field gets the same dose.
      ``"half"``  cells with ``(particle // group_stride) % 2 == 1`` share a dose;
                  the rest are planned individually. One field then carries both
                  arms, so anything field-level — focus, medium, illumination —
                  cancels inside it rather than landing in the contrast. Dividing
                  by ``group_stride`` before taking the parity keeps the split
                  independent of the phase group, which is ``particle % 4``; a plain
                  ``particle % 2`` would put phase groups 0 and 2 in one arm and 1
                  and 3 in the other.

    The broadcast group is (arm x PHASE GROUP), never the arm alone. With four phase
    groups, cells at one wall-clock frame are being asked for opposite things — half
    to sit at the low level, half at the high — and one dose for all of them is not
    population control, it is an incoherent arm that loses for a reason nobody is
    asking about. Measured on v19: the cross-cell median command taken WITHIN a phase
    group tracks the demand (17-20 ms through the low phase, 39-63 ms through the
    high, averaging 36.9 ms against a mean of 52.2). Taken across the whole field it
    reads 19.9 ms, dragged down by whichever half is meant to be resting.

    The representative is whichever cell sits at the median CNR of its broadcast group
    this frame, re-chosen every frame, and its own encoder state is what the plan is
    computed from. Averaging hidden states instead would build a state no cell was
    ever in and that the model has never been asked about. Re-choosing every frame is
    consistent with the receding horizon: the plan is recomputed from scratch anyway.

    Padding rows duplicate the last real cell, so the median is taken over distinct
    particle ids only — otherwise a bucket of 31 copies could decide who is typical.
    """

    name = "population_mpc"

    def __init__(
        self,
        levels_ms: np.ndarray,
        share: str = "half",
        group_stride: int = 4,
        **mpc_kwargs: Any,
    ):
        super().__init__(levels_ms=levels_ms, **mpc_kwargs)
        if share not in ("all", "half"):
            raise ValueError(f"share must be 'all' or 'half', got {share!r}")
        if int(group_stride) < 1:
            raise ValueError(f"group_stride must be >= 1, got {group_stride}")
        self.share = share
        self.group_stride = int(group_stride)

    def shares_dose(self, particle: int) -> bool:
        """Is this cell in the group that receives one shared dose?"""
        if self.share == "all":
            return True
        return ((int(particle) // self.group_stride) % 2) == 1

    def _shared_mask(self, ctx: GoalContext, device) -> torch.Tensor:
        return torch.tensor(
            [self.shares_dose(f.state.particle) for f in ctx.cells],
            dtype=torch.bool, device=device,
        )

    @staticmethod
    def _condition_of(objective, cell) -> float:
        """What this cell is currently being asked for, as a groupable key.

        Read from the objective's own reference rather than recomputed here, so the
        split can never drift from the waveform the cells are actually tracking. A
        reference with no per-cell phase puts every cell in one condition.
        """
        ref = getattr(objective, "reference", None)
        fn = getattr(ref, "phase_offset_min", None)
        return 0.0 if fn is None else float(fn(cell))

    def _representative(self, cnr_fb, idx) -> int:
        """Index of the cell at the median CNR of ``idx``."""
        if not idx:
            return -1
        order = torch.argsort(cnr_fb.reshape(-1)[idx])
        return int(idx[int(order[len(order) // 2])])

    def _broadcast_groups(self, shared, ctx, objective) -> list[list[int]]:
        """Row indices per broadcast group: shared cells, split by what they are
        being asked for, deduplicated by particle so padding copies cannot vote."""
        groups: dict[float, list[int]] = {}
        seen: set[int] = set()
        for i, f in enumerate(ctx.cells):
            if not bool(shared[i]):
                continue
            p = int(f.state.particle)
            if p in seen:
                continue
            seen.add(p)
            groups.setdefault(self._condition_of(objective, f), []).append(i)
        return list(groups.values())

    def plan(self, engine, h, c, cnr_fb, objective, ctx):
        ms, cost = super().plan(engine, h, c, cnr_fb, objective, ctx)
        shared = self._shared_mask(ctx, ms.device)
        if not bool(shared.any()):
            return ms, cost
        ms = ms.clone()
        cost = cost.clone()
        for idx in self._broadcast_groups(shared, ctx, objective):
            rep = self._representative(cnr_fb, idx)
            if rep < 0:
                continue
            same = torch.tensor(
                [bool(shared[j]) and self._condition_of(objective, ctx.cells[j])
                 == self._condition_of(objective, ctx.cells[rep])
                 for j in range(len(ctx.cells))],
                dtype=torch.bool, device=ms.device,
            )
            # The cost is broadcast with the dose on purpose: every cell in the group
            # is being commanded on the strength of one plan, and the log should say
            # so rather than report a per-cell cost nothing acted on.
            ms[same] = ms[rep]
            cost[same] = cost[rep]
        return ms, cost

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"share": self.share, "group_stride": self.group_stride})
        return d


CONTROLLERS: dict[str, type[Controller]] = {
    ConstantDoseSearch.name: ConstantDoseSearch,
    SequenceMPC.name: SequenceMPC,
    StaggeredCadenceMPC.name: StaggeredCadenceMPC,
    OpenLoopController.name: OpenLoopController,
    PopulationMPC.name: PopulationMPC,
}


def build_controller(spec: dict[str, Any], levels_ms: np.ndarray) -> Controller:
    """Build a controller from a policy-file spec: ``{"type": ..., **params}``."""
    spec = dict(spec)
    kind = spec.pop("type", ConstantDoseSearch.name)
    if kind not in CONTROLLERS:
        raise KeyError(
            f"unknown controller type {kind!r}; known: {sorted(CONTROLLERS)}"
        )
    return CONTROLLERS[kind](levels_ms=levels_ms, **spec)
