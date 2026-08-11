"""Objectives and controllers, tested against a toy plant.

No checkpoint and no cluster mount: the controllers only ever touch the model
through the plant interface (``horizon``, ``device``, ``std_fluence``,
``rollout``, ``rollout_mixture``, ``denorm_cnr``, ``denorm_sigma``), so a
few-line analytic plant exercises them completely — and lets the optimum be
computed by hand, which a real model never would.
"""
import math
import types

import numpy as np
import pytest
import torch

from optoerk.serving.control import (
    ConstantDoseSearch,
    SequenceMPC,
    StaggeredCadenceMPC,
    build_controller,
    dose_levels,
)
from optoerk.serving.objectives import (
    BandKernel,
    ConstantReference,
    DosePenalty,
    FrequencyStaircaseReference,
    GoalContext,
    L2Kernel,
    MovePenalty,
    Objective,
    Prediction,
    StepTrainReference,
    build_kernel,
    build_objective,
    gated,
    hold,
    oscillation,
    schedule,
)
from optoerk.serving.runtime import CellFrame
from optoerk.serving.state import CellState, StateStore

LEVELS = dose_levels(0.0, 800.0, 5)  # [0, 200, 400, 600, 800]


class ToyPlant:
    """Instantaneous, memoryless response: predicted CNR == the dose applied that
    step, scaled to [0, 1]. Trivial, but it makes the optimal plan exact:
    to track target trajectory ``g`` you must command ``g`` step by step.

    The mixture is a single component centred on the mean with a fixed width, so
    band-kernel plumbing is exercised without the band changing the optimum.
    """

    sigma = 0.05

    def __init__(self, horizon=2):
        self.horizon = horizon
        self.device = torch.device("cpu")

    def std_fluence(self, ms):
        return ms / 800.0

    def denorm_cnr(self, pred_std):
        return pred_std

    def denorm_sigma(self, sigma_std):
        return sigma_std

    def rollout(self, h, c, cnr_fb, fut):
        return fut[:, :, 0]  # (B, H)

    def rollout_mixture(self, h, c, cnr_fb, fut):
        mean = self.rollout(h, c, cnr_fb, fut)              # (B, H)
        mu = mean.unsqueeze(-1)                             # (B, H, 1)
        return mean, torch.ones_like(mu), mu, torch.full_like(mu, self.sigma)


class LeakyPlant(ToyPlant):
    """``cnr[h] = 0.6*cnr[h-1] + 0.9*u[h]`` — has memory, so plan *ordering*
    matters and a golden CEM result actually constrains the optimizer."""

    def rollout(self, h, c, cnr_fb, fut):
        out = []
        x = cnr_fb[:, 0]
        for i in range(fut.shape[1]):
            x = 0.6 * x + 0.9 * fut[:, i, 0]
            out.append(x)
        return torch.stack(out, dim=1)


def _cells(n, **kw):
    return [
        CellFrame(state=CellState(), cnr_norm=1.0, fov_density=float(n),
                  n_cells_200px=kw.pop("n_cells_200px", 1.0),
                  x=kw.pop("x", 0.0), y=kw.pop("y", 0.0))
        for _ in range(n)
    ]


def _state(n, plant):
    h = torch.zeros(1, n, 4)
    return h, h.clone(), torch.zeros(n, 1)


def _pred(cnr, plan_norm=None, prev_norm=None, **mix):
    """A Prediction for testing a kernel / regularizer in isolation."""
    cnr = torch.as_tensor(cnr, dtype=torch.float32)
    N, M, H = cnr.shape
    return Prediction(
        cnr=cnr,
        plan_norm=torch.zeros(N, M, H) if plan_norm is None else torch.as_tensor(
            plan_norm, dtype=torch.float32),
        prev_norm=torch.zeros(N) if prev_norm is None else torch.as_tensor(
            prev_norm, dtype=torch.float32),
        **mix,
    )


# ---------------------------------------------------------------------------
# objectives — reference / kernel / regularizer composition
# ---------------------------------------------------------------------------


def test_hold_cost_is_squared_error_to_target():
    obj = hold(0.5)
    ctx = GoalContext(fov=0, timestep=0, cells=_cells(3))
    cost = obj.cost(_pred(torch.zeros(3, 2, 4)), ctx)
    assert cost.shape == (3, 2)
    assert torch.allclose(cost, torch.full((3, 2), 0.25))


def test_hold_has_no_regularizers_by_default():
    """The regression-critical property: with lambda_move / lambda_dose at zero,
    nothing is added to the kernel at all — not even a multiply-by-zero term."""
    assert hold(1.2).regularizers == []
    assert hold(1.2, lambda_move=0.0, lambda_dose=0.0).regularizers == []
    assert len(hold(1.2, lambda_move=0.3).regularizers) == 1
    assert len(hold(1.2, lambda_move=0.3, lambda_dose=0.1).regularizers) == 2


def test_schedule_targets_step_over_the_horizon():
    """The setpoint is evaluated per horizon step, so the controller sees an
    upcoming step change before it lands."""
    obj = schedule([[0, 1.0], [5, 2.0]])
    ctx = GoalContext(fov=0, timestep=3, cells=_cells(1))
    tgt = obj.targets(ctx, horizon=4, device=torch.device("cpu"))
    # frames 3,4 -> 1.0 ; frames 5,6 -> 2.0
    assert torch.allclose(tgt, torch.tensor([[1.0, 1.0, 2.0, 2.0]]))


def test_gated_predicates_combine_and_default_open():
    obj = gated(1.5, after_t=10, x_gt=512, max_neighbours_200px=5)
    cells = [
        CellFrame(CellState(), 1.0, 8.0, 3.0, x=600.0, y=0.0),   # passes
        CellFrame(CellState(), 1.0, 8.0, 3.0, x=100.0, y=0.0),   # x too small
        CellFrame(CellState(), 1.0, 8.0, 9.0, x=600.0, y=0.0),   # too crowded
    ]
    assert obj.allow_stim(GoalContext(0, 20, cells)).tolist() == [True, False, False]
    # before after_t nobody is eligible
    assert obj.allow_stim(GoalContext(0, 5, cells)).tolist() == [False] * 3
    # no gate at all -> None, meaning "everyone"
    assert hold(1.5).allow_stim(GoalContext(0, 20, cells)) is None


def test_missing_position_fails_position_gates():
    """A cell with no reported x/y must never be stimulated by accident."""
    obj = gated(1.5, x_gt=100)
    cells = [CellFrame(CellState(), 1.0, 1.0, 1.0)]  # x, y default to NaN
    assert obj.allow_stim(GoalContext(0, 20, cells)).tolist() == [False]


def test_build_objective_from_spec_and_unknown_type():
    obj = build_objective({"type": "hold", "target_cnr": 1.9})
    d = obj.describe()
    assert d["type"] == "hold" and d["target_cnr"] == 1.9
    assert d["reference"] == {"type": "constant", "target_cnr": 1.9}
    assert d["kernel"] == {"type": "l2"} and d["regularizers"] == []
    with pytest.raises(KeyError, match="unknown objective type"):
        build_objective({"type": "nope"})
    with pytest.raises(ValueError, match="needs a 'type'"):
        build_objective({"target_cnr": 1.0})
    with pytest.raises(TypeError, match="bad params"):
        build_objective({"type": "hold", "nonsense": 1})


def test_build_kernel_from_string_or_table():
    assert isinstance(build_kernel("l2"), L2Kernel)
    k = build_kernel({"type": "band", "half_width": 0.05})
    assert isinstance(k, BandKernel) and k.half_width == 0.05
    with pytest.raises(KeyError, match="unknown kernel"):
        build_kernel("nope")
    with pytest.raises(ValueError, match="half_width"):
        build_kernel({"type": "band", "half_width": 0.0})


def test_band_kernel_requires_half_width_explicitly():
    """No default: a policy selecting the band kernel must state delta."""
    with pytest.raises(TypeError):
        BandKernel()


def test_needs_mixture_propagates_from_the_kernel():
    assert hold(1.0).needs_mixture is False
    assert hold(1.0, kernel={"type": "band", "half_width": 0.1}).needs_mixture is True


def test_needs_plan_is_true_only_when_a_regularizer_reads_it():
    """Arms 1-2 have no regularizers, so the controller must not materialize the
    dose plan or its normalized copy — two (N, M, H) tensors, never read."""
    assert hold(1.0).needs_plan is False
    assert hold(1.0, kernel={"type": "band", "half_width": 0.1}).needs_plan is False
    assert hold(1.0, lambda_move=0.3).needs_plan is True
    assert hold(1.0, lambda_dose=0.3).needs_plan is True
    assert hold(1.0, lambda_move=0.0).needs_plan is False


def test_regularizer_reports_a_missing_plan_clearly():
    for reg in (MovePenalty(1.0), DosePenalty(1.0)):
        with pytest.raises(ValueError, match="needs the dose plan"):
            reg.cost(Prediction(cnr=torch.zeros(1, 1, 2)))


def test_controller_skips_the_plan_tensors_when_nothing_reads_them():
    """The gate is on the controller side too, not just inside the objective —
    otherwise the tensors are built and then discarded."""
    plant = ToyPlant(horizon=3)
    ctx = GoalContext(0, 0, _cells(3))
    seen = {}

    class Spy(Objective):
        def cost(self, pred, ctx):
            seen["plan_norm"] = pred.plan_norm
            seen["prev_norm"] = pred.prev_norm
            return super().cost(pred, ctx)

    plain = Spy(ConstantReference(0.5), L2Kernel())
    ConstantDoseSearch(LEVELS).plan(plant, *_state(3, plant), plain, ctx)
    assert seen["plan_norm"] is None and seen["prev_norm"] is None

    damped = Spy(ConstantReference(0.5), L2Kernel(), [MovePenalty(0.1)])
    ConstantDoseSearch(LEVELS).plan(plant, *_state(3, plant), damped, ctx)
    assert seen["plan_norm"] is not None and seen["prev_norm"] is not None


def test_l2_kernel_never_asks_for_the_mixture():
    """Mean-only predictions must satisfy the L2 path — that is what keeps the
    cheap path cheap."""
    pred = _pred(torch.zeros(2, 3, 4))
    assert pred.pi is None
    L2Kernel().cost(pred, torch.zeros(2, 4))  # must not raise


def test_band_kernel_reports_a_missing_mixture_clearly():
    with pytest.raises(ValueError, match="needs the predictive mixture"):
        BandKernel(0.1).cost(_pred(torch.zeros(1, 1, 2)), torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# band kernel correctness
# ---------------------------------------------------------------------------


def _band_prob_numeric(pi, mu, sigma, r, delta, n=200_001):
    """P(|x - r| <= delta) by direct quadrature of the mixture density."""
    x = np.linspace(r - delta, r + delta, n)
    dens = np.zeros_like(x)
    for p, m, s in zip(pi, mu, sigma):
        dens += p * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return float(np.trapezoid(dens, x))


@pytest.mark.parametrize(
    "pi,mu,sigma,r,delta",
    [
        ([1.0], [0.9], [0.10], 0.9, 0.05),           # centred, single component
        ([1.0], [0.9], [0.10], 1.3, 0.05),           # far off the reference
        ([0.5, 0.5], [0.8, 1.2], [0.10, 0.20], 1.0, 0.15),
        ([0.3, 0.3, 0.4], [0.7, 1.0, 1.4], [0.05, 0.2, 0.1], 1.05, 0.08),
        ([0.999, 0.001], [1.0, 3.0], [0.12, 0.5], 1.0, 0.10),   # ~all mass in one
        ([0.5, 0.5], [0.9, 1.1], [0.10, 0.10], 1.0, 0.002),     # delta very small
        ([0.5, 0.5], [0.9, 1.1], [0.10, 0.10], 1.0, 5.0),       # delta very large
        ([0.6, 0.4], [0.85, 1.15], [0.30, 0.02], 1.15, 0.05),   # one narrow comp
    ],
)
def test_band_cost_matches_numerical_integration(pi, mu, sigma, r, delta):
    K = len(pi)
    pred = _pred(
        torch.zeros(1, 1, 1),
        pi=torch.tensor(pi).view(1, 1, 1, K),
        mu=torch.tensor(mu).view(1, 1, 1, K),
        sigma=torch.tensor(sigma).view(1, 1, 1, K),
    )
    got = float(BandKernel(delta).cost(pred, torch.tensor([[r]]))[0, 0])
    want = 1.0 - _band_prob_numeric(pi, mu, sigma, r, delta)
    assert got == pytest.approx(want, abs=1e-5)


def test_band_cost_with_sigma_at_the_floor_is_an_indicator():
    """A component with (effectively) zero width is inside the band or it is not;
    the clamp must keep that finite rather than dividing by zero."""
    inside = _pred(
        torch.zeros(1, 1, 1),
        pi=torch.tensor([1.0]).view(1, 1, 1, 1),
        mu=torch.tensor([1.0]).view(1, 1, 1, 1),
        sigma=torch.tensor([0.0]).view(1, 1, 1, 1),
    )
    assert float(BandKernel(0.1).cost(inside, torch.tensor([[1.0]]))) == pytest.approx(0.0)
    outside = _pred(
        torch.zeros(1, 1, 1),
        pi=torch.tensor([1.0]).view(1, 1, 1, 1),
        mu=torch.tensor([2.0]).view(1, 1, 1, 1),
        sigma=torch.tensor([0.0]).view(1, 1, 1, 1),
    )
    assert float(BandKernel(0.1).cost(outside, torch.tensor([[1.0]]))) == pytest.approx(1.0)


def test_band_cost_is_bounded_and_monotone_in_delta():
    pi = torch.tensor([0.4, 0.6]).view(1, 1, 1, 2)
    mu = torch.tensor([0.9, 1.2]).view(1, 1, 1, 2)
    sd = torch.tensor([0.15, 0.25]).view(1, 1, 1, 2)
    pred = _pred(torch.zeros(1, 1, 1), pi=pi, mu=mu, sigma=sd)
    r = torch.tensor([[1.0]])
    costs = [float(BandKernel(d).cost(pred, r)) for d in (0.01, 0.05, 0.2, 1.0, 10.0)]
    assert all(0.0 <= c <= 1.0 for c in costs)
    assert costs == sorted(costs, reverse=True), "wider band can only reduce the cost"


def test_band_tracks_a_time_varying_reference():
    """The band must follow r_h, not sit at a fixed level. Same mixture at both
    steps; the reference moves, so only the step it matches scores well."""
    mu = torch.tensor([1.0, 1.0]).view(1, 1, 2, 1)
    pred = _pred(
        torch.zeros(1, 1, 2),
        pi=torch.ones(1, 1, 2, 1),
        mu=mu,
        sigma=torch.full((1, 1, 2, 1), 0.02),
    )
    tracking = BandKernel(0.1).cost(pred, torch.tensor([[1.0, 1.0]]))
    moving_away = BandKernel(0.1).cost(pred, torch.tensor([[1.0, 2.0]]))
    assert float(tracking) == pytest.approx(0.0, abs=1e-4)
    assert float(moving_away) == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# move / dose penalties
# ---------------------------------------------------------------------------


def test_move_penalty_zero_for_a_constant_plan_from_the_same_previous_dose():
    plan = torch.full((1, 1, 5), 0.4)
    pred = _pred(torch.zeros(1, 1, 5), plan_norm=plan, prev_norm=torch.tensor([0.4]))
    assert float(MovePenalty(1.0).cost(pred)) == pytest.approx(0.0)


def test_move_penalty_is_maximal_for_a_full_swing_alternation():
    """Alternating 0 / max from a previous dose of 0 changes by 1.0 every step,
    so the mean squared change is exactly 1.0 — the ceiling."""
    plan = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    pred = _pred(torch.zeros(1, 1, 4), plan_norm=plan, prev_norm=torch.tensor([0.0]))
    assert float(MovePenalty(1.0).cost(pred)) == pytest.approx(1.0)


def test_move_penalty_is_invariant_to_rebinning_the_ladder():
    """Expressed in normalized dose, the penalty must not change when the ladder
    is rebinned. The 0-800 ms and 0-150 ms ladders below are the same rungs in
    normalized terms, so the same normalized plan must cost the same."""
    wide = ConstantDoseSearch(dose_levels(0.0, 800.0, 5))
    narrow = ConstantDoseSearch(np.array([0.0, 37.5, 75.0, 112.5, 150.0]))
    assert wide.max_ms == 800.0 and narrow.max_ms == 150.0

    plan_ms_wide = torch.tensor([[[0.0, 800.0, 400.0, 200.0]]])
    plan_ms_narrow = plan_ms_wide * (150.0 / 800.0)
    prev_ms_wide, prev_ms_narrow = 600.0, 600.0 * (150.0 / 800.0)

    pen = MovePenalty(0.7)
    a = pen.cost(_pred(torch.zeros(1, 1, 4),
                       plan_norm=plan_ms_wide / wide.max_ms,
                       prev_norm=torch.tensor([prev_ms_wide / wide.max_ms])))
    b = pen.cost(_pred(torch.zeros(1, 1, 4),
                       plan_norm=plan_ms_narrow / narrow.max_ms,
                       prev_norm=torch.tensor([prev_ms_narrow / narrow.max_ms])))
    assert torch.allclose(a, b, atol=1e-6)


def test_move_penalty_charges_the_first_step_against_the_previous_dose():
    """u_{-1} is the dose actually applied last frame, not u[0]. A cell that was
    at max and drops to zero is penalized; taking u[0] as the reference would
    make that first move free."""
    plan = torch.zeros(1, 1, 3)
    from_zero = _pred(torch.zeros(1, 1, 3), plan_norm=plan, prev_norm=torch.tensor([0.0]))
    from_max = _pred(torch.zeros(1, 1, 3), plan_norm=plan, prev_norm=torch.tensor([1.0]))
    assert float(MovePenalty(1.0).cost(from_zero)) == pytest.approx(0.0)
    assert float(MovePenalty(1.0).cost(from_max)) == pytest.approx(1.0 / 3.0)


def test_dose_penalty_is_mean_normalized_dose():
    plan = torch.tensor([[[0.0, 0.5, 1.0, 0.5]]])
    pred = _pred(torch.zeros(1, 1, 4), plan_norm=plan)
    assert float(DosePenalty(2.0).cost(pred)) == pytest.approx(2.0 * 0.5)


def test_move_penalty_suppresses_slamming_in_closed_loop():
    """End to end, over frames: against a reference that flips between 0 and max
    every frame, the unpenalized controller slams the full swing every time. The
    penalty is a cost on the *whole plan*, so it can legitimately raise u[0] on
    any single frame — the property it buys is lower total variation of the
    commanded dose over time, which is what this asserts.
    """
    plant = ToyPlant(horizon=4)
    ref = _AlternatingReference()

    def run(objective):
        cells = _cells(1)
        ctrl = SequenceMPC(LEVELS, n_samples=128, n_iters=3, seed=3)
        applied = []
        for t in range(10):
            ctx = GoalContext(0, t, cells)
            ms = float(ctrl.solve(plant, *_state(1, plant), objective, ctx)[0])
            applied.append(ms)
            cells[0].state.last_applied_ms = ms  # what runtime.decide persists
        return applied

    free = run(Objective(ref, L2Kernel()))
    damped = run(Objective(ref, L2Kernel(), [MovePenalty(5.0)]))

    def tv(xs):
        return sum(abs(b - a) for a, b in zip(xs, xs[1:]))

    assert tv(free) == pytest.approx(800.0 * 9), "unpenalized slams every frame"
    assert tv(damped) < tv(free) / 2


class _AlternatingReference(ConstantReference):
    """0.0 on even frames, 1.0 on odd — the worst case for a move penalty."""

    def __init__(self):
        super().__init__(0.0)

    def values(self, ctx, horizon, device):
        row = [float((ctx.timestep + h) % 2) for h in range(horizon)]
        return torch.tensor([row] * len(ctx.cells), dtype=torch.float32, device=device)


class _FixedReference(ConstantReference):
    """A literal per-step reference row, for tests that need an exact trajectory."""

    def __init__(self, row):
        self.row = [float(x) for x in row]
        super().__init__(self.row[0])

    def values(self, ctx, horizon, device):
        row = (self.row * horizon)[:horizon]
        return torch.tensor([row] * len(ctx.cells), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# u_{-1} plumbing through per-cell state
# ---------------------------------------------------------------------------


def test_prev_norm_reads_the_applied_dose_from_cell_state():
    ctrl = ConstantDoseSearch(LEVELS)
    cells = _cells(3)
    cells[0].state.last_applied_ms = 0.0
    cells[1].state.last_applied_ms = 400.0
    cells[2].state.last_applied_ms = 800.0
    prev = ctrl._prev_norm(GoalContext(0, 0, cells), torch.device("cpu"))
    assert prev.tolist() == [0.0, 0.5, 1.0]


def test_fresh_cell_takes_the_documented_zero_previous_dose():
    """A cell's first frame has applied no dose yet."""
    assert CellState().last_applied_ms == 0.0
    ctrl = ConstantDoseSearch(LEVELS)
    prev = ctrl._prev_norm(GoalContext(0, 0, _cells(2)), torch.device("cpu"))
    assert prev.tolist() == [0.0, 0.0]


def test_seeded_daughter_takes_the_documented_zero_previous_dose():
    """A daughter inherits the mother's encoder state and u_t channel — the light
    its lineage saw — but not a commanded dose of its own, so its first move is
    measured from zero."""
    store = StateStore()
    mother = store.get_or_create(fov=0, particle=1)
    mother.last_fluence = 12.5
    mother.last_applied_ms = 600.0
    daughter = store.get_or_create(fov=0, particle=2, parent=1)
    assert daughter.last_fluence == 12.5, "u_t channel IS inherited"
    assert daughter.last_applied_ms == 0.0, "u_{-1} for the move penalty is NOT"


# ---------------------------------------------------------------------------
# controllers
# ---------------------------------------------------------------------------


def test_constant_dose_search_picks_the_reachable_target():
    """Target 0.5 == 400 ms on the toy plant; the grid contains it exactly."""
    plant = ToyPlant(horizon=2)
    ctrl = ConstantDoseSearch(LEVELS)
    cells = _cells(4)
    ctx = GoalContext(0, 0, cells)
    ms = ctrl.solve(plant, *_state(4, plant), hold(0.5), ctx)
    assert ms.tolist() == [400.0] * 4


def test_sequence_mpc_beats_constant_on_a_time_varying_target():
    """A target that steps mid-horizon is unreachable by any constant dose but
    exactly reachable by a sequence — the whole point of true MPC."""
    plant = ToyPlant(horizon=2)
    obj = Objective(_FixedReference([0.0, 1.0]), L2Kernel())
    cells = _cells(4)
    ctx = GoalContext(0, 0, cells)
    h, c, fb = _state(4, plant)

    _, const_cost = ConstantDoseSearch(LEVELS).plan(plant, h, c, fb, obj, ctx)
    mpc_ms, mpc_cost = SequenceMPC(LEVELS, n_samples=64, n_iters=3).plan(
        plant, h, c, fb, obj, ctx
    )

    assert torch.all(mpc_cost < const_cost), "MPC must beat constant here"
    assert torch.allclose(mpc_cost, torch.zeros(4), atol=1e-6), "optimum is exact"
    assert mpc_ms.tolist() == [0.0] * 4, "applies u[0] only, which is 0 ms"


def test_sequence_mpc_never_worse_than_constant():
    """Constant plans are injected into every CEM iteration, so MPC's best plan
    is no worse than the best constant one — by construction, on any objective."""
    plant = ToyPlant(horizon=3)
    obj = hold(0.42)  # not on the dose grid, so nothing is exactly reachable
    cells = _cells(6)
    ctx = GoalContext(0, 0, cells)
    h, c, fb = _state(6, plant)

    _, const_cost = ConstantDoseSearch(LEVELS).plan(plant, h, c, fb, obj, ctx)
    _, mpc_cost = SequenceMPC(LEVELS, n_samples=16, n_iters=2).plan(
        plant, h, c, fb, obj, ctx
    )
    assert torch.all(mpc_cost <= const_cost + 1e-6)


def test_both_controllers_apply_the_gate():
    plant = ToyPlant(horizon=2)
    cells = [
        CellFrame(CellState(), 1.0, 2.0, 1.0, x=600.0, y=0.0),
        CellFrame(CellState(), 1.0, 2.0, 1.0, x=100.0, y=0.0),
    ]
    ctx = GoalContext(0, 20, cells)
    obj = gated(0.5, x_gt=512)
    for ctrl in (ConstantDoseSearch(LEVELS), SequenceMPC(LEVELS, n_samples=16, n_iters=1)):
        ms = ctrl.solve(plant, *_state(2, plant), obj, ctx)
        assert ms[1].item() == 0.0, f"{ctrl.name} ignored the gate"


def test_sequence_mpc_is_deterministic_for_a_given_frame():
    """Same frame, same plan — otherwise a replay could not reproduce a run."""
    plant = ToyPlant(horizon=3)
    obj = hold(0.42)
    ctx = GoalContext(0, 7, _cells(5))
    args = (plant, *_state(5, plant), obj, ctx)
    a = SequenceMPC(LEVELS, n_samples=32, n_iters=2, seed=1).solve(*args)
    b = SequenceMPC(LEVELS, n_samples=32, n_iters=2, seed=1).solve(*args)
    assert a.tolist() == b.tolist()


def test_sequence_mpc_is_deterministic_under_the_band_kernel_too():
    """Replay exactness must survive the distributional cost."""
    plant = ToyPlant(horizon=3)
    obj = hold(0.42, kernel={"type": "band", "half_width": 0.08}, lambda_move=0.4)
    ctx = GoalContext(0, 7, _cells(5))
    args = (plant, *_state(5, plant), obj, ctx)
    a = SequenceMPC(LEVELS, n_samples=32, n_iters=2, seed=1).solve(*args)
    b = SequenceMPC(LEVELS, n_samples=32, n_iters=2, seed=1).solve(*args)
    assert a.tolist() == b.tolist()


def test_band_kernel_reaches_the_optimum_on_the_toy_plant():
    """Sanity that the band path is wired end to end: the toy plant's mixture is
    centred on its mean, so the band-optimal dose is still the one that lands on
    the reference."""
    plant = ToyPlant(horizon=2)
    obj = hold(0.5, kernel={"type": "band", "half_width": 0.05})
    ms = ConstantDoseSearch(LEVELS).solve(plant, *_state(4, plant), obj,
                                          GoalContext(0, 0, _cells(4)))
    assert ms.tolist() == [400.0] * 4


def test_build_controller_defaults_to_constant_dose():
    assert build_controller({}, LEVELS).name == "constant_dose"
    ctrl = build_controller({"type": "sequence_mpc", "n_samples": 8}, LEVELS)
    assert isinstance(ctrl, SequenceMPC) and ctrl.n_samples == 8
    with pytest.raises(KeyError, match="unknown controller type"):
        build_controller({"type": "nope"}, LEVELS)


def test_dose_levels_grid():
    assert dose_levels(0, 800, 5).tolist() == [0, 200, 400, 600, 800]
    with pytest.raises(ValueError):
        dose_levels(0, 800, 0)


def test_max_ms_follows_the_ladder_not_the_config():
    assert ConstantDoseSearch(np.array([0.0, 20.0, 45.0, 85.0, 150.0])).max_ms == 150.0
    assert ConstantDoseSearch(np.array([0.0])).max_ms == 1.0, "no divide-by-zero"


# ---------------------------------------------------------------------------
# regression guard: the refactor must not move the existing controller
# ---------------------------------------------------------------------------

# Commanded doses and costs produced by the PRE-refactor objectives.py +
# control.py (commit b299eb9) on the LeakyPlant scenario below. Arm 2 of the
# 12-FOV run is the existing controller and must stay bit-for-bit identical, so
# these are pinned rather than recomputed. Regenerate ONLY against the old code.
GOLDEN = {
    "hold_H3": {
        "const_ms": [400.0] * 6,
        "const_cost": [0.027398081496357918] * 6,
        "mpc_ms": [400.0] * 6,
        "mpc_cost": [0.007253082003444433] * 6,
    },
    "hold_H8": {
        "const_ms": [200.0] * 6,
        "const_cost": [0.03705766052007675] * 6,
        "mpc_ms": [400.0, 400.0, 400.0, 600.0, 400.0, 400.0],
        "mpc_cost": [0.006545227486640215, 0.008303146809339523,
                     0.008606206625699997, 0.011577184312045574,
                     0.008303146809339523, 0.011737223714590073],
    },
    "schedule_H3": {
        "const_ms": [400.0] * 6,
        "const_cost": [0.050118084996938705] * 6,
        "mpc_ms": [200.0] * 6,
        "mpc_cost": [0.001051879022270441] * 6,
    },
    "schedule_H8": {
        "const_ms": [400.0] * 6,
        "const_cost": [0.01986347883939743] * 6,
        "mpc_ms": [200.0, 200.0, 400.0, 200.0, 200.0, 400.0],
        "mpc_cost": [0.012328704819083214, 0.014467976987361908,
                     0.012515337206423283, 0.00827869027853012,
                     0.006906174123287201, 0.012515337206423283],
    },
}


def _golden_cells(n):
    cells = _cells(n)
    for i, f in enumerate(cells):
        f.state.particle = i
        f.state.n_frames = 3
    return cells


def _golden_state(n):
    h = torch.zeros(1, n, 4)
    return h, h.clone(), torch.full((n, 1), 0.3)


@pytest.mark.parametrize("tag,horizon", [("hold", 3), ("hold", 8),
                                         ("schedule", 3), ("schedule", 8)])
def test_refactor_reproduces_the_pre_refactor_controller(tag, horizon):
    """THE regression guard. ``constant`` + ``l2`` + no regularizers must be the
    old controller, exactly — same commanded dose and same cost, on a plant with
    memory where the CEM's sampling actually matters."""
    obj = hold(0.7) if tag == "hold" else schedule([[0, 0.4], [6, 1.1]])
    plant = LeakyPlant(horizon)
    n = 6
    ctx = GoalContext(0, 5, _golden_cells(n))
    args = (plant, *_golden_state(n), obj, ctx)

    ms_c, cost_c = ConstantDoseSearch(LEVELS).plan(*args)
    ms_m, cost_m = SequenceMPC(LEVELS, n_samples=128, n_iters=3, seed=7).plan(*args)

    want = GOLDEN[f"{tag}_H{horizon}"]
    assert ms_c.tolist() == want["const_ms"]
    assert ms_m.tolist() == want["mpc_ms"]
    assert cost_c.tolist() == pytest.approx(want["const_cost"], abs=0, rel=1e-12)
    assert cost_m.tolist() == pytest.approx(want["mpc_cost"], abs=0, rel=1e-12)


def test_explicit_zero_lambdas_are_identical_to_omitting_them():
    """lambda_move = 0 must not perturb the cost by so much as a rounding step."""
    plant = LeakyPlant(6)
    ctx = GoalContext(0, 5, _golden_cells(4))
    args = (plant, *_golden_state(4), )
    a = SequenceMPC(LEVELS, n_samples=64, n_iters=2, seed=2).plan(
        *args, hold(0.7), ctx)
    b = SequenceMPC(LEVELS, n_samples=64, n_iters=2, seed=2).plan(
        *args, hold(0.7, lambda_move=0.0, lambda_dose=0.0), ctx)
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()


# ---------------------------------------------------------------------------
# oscillation reference (step train)
# ---------------------------------------------------------------------------


def _osc(**kw):
    params = dict(
        low=0.85, high=1.15,
        t_low_min=8.0, t_rise_min=2.0, t_high_min=15.0, t_fall_min=15.0,
        settle_periods=2.0, n_phase_groups=1,
    )
    params.update(kw)
    return StepTrainReference(**params)


def test_step_train_period_and_settle_are_the_sum_of_segments():
    ref = _osc()
    assert ref.period_min == 40.0
    assert ref.settle_min == 80.0
    assert ref.amplitude == pytest.approx(0.30)


def test_step_train_segments_partition_the_timeline():
    """Every minute of a period falls in exactly one labelled segment, with no
    gaps and no overlaps."""
    ref = _osc()
    seen = [ref.segment_at(ref.settle_min + t) for t in range(int(ref.period_min))]
    assert set(seen) == {"low_hold", "rise", "high_hold", "fall"}
    # segments occur as one contiguous run each, in waveform order
    order = [k for i, k in enumerate(seen) if i == 0 or k != seen[i - 1]]
    assert order == ["low_hold", "rise", "high_hold", "fall"]


def test_step_train_holds_low_through_the_settle_window():
    ref = _osc()
    for t in range(0, int(ref.settle_min)):
        assert ref.value_at(t) == pytest.approx(ref.low)
        assert ref.segment_at(t) == "settle"
    assert ref.segment_at(ref.settle_min) == "low_hold"


def test_step_train_levels_and_linear_ramps():
    ref = _osc()
    s = ref.settle_min
    assert ref.value_at(s + 0.0) == pytest.approx(0.85)
    assert ref.value_at(s + 7.9) == pytest.approx(0.85)
    # rise spans [8, 10): linear from low to high
    assert ref.value_at(s + 8.0) == pytest.approx(0.85)
    assert ref.value_at(s + 9.0) == pytest.approx(1.00)
    assert ref.value_at(s + 10.0) == pytest.approx(1.15)
    assert ref.value_at(s + 20.0) == pytest.approx(1.15)
    # fall spans [25, 40): linear back down
    assert ref.value_at(s + 25.0) == pytest.approx(1.15)
    assert ref.value_at(s + 32.5) == pytest.approx(1.00)
    assert ref.value_at(s + 39.9) == pytest.approx(0.852, abs=1e-3)


def test_step_train_is_periodic():
    ref = _osc()
    s = ref.settle_min
    for t in np.linspace(0, ref.period_min, 41):
        assert ref.value_at(s + t) == pytest.approx(
            ref.value_at(s + t + ref.period_min))


def test_step_train_accepts_a_fall_the_cells_cannot_follow():
    """Deliberately-unreachable references are expressible, by design.

    The border-probing arms demand exactly this: a reference the cells fail to
    track is the measurement, and where they depart from it is the answer. The
    old tau-based guards refused these, comparing one hand-typed number against
    another; feasibility is now argued in the pre-flight check instead.
    """
    ref = _osc(t_fall_min=1.0, t_high_min=29.0)
    assert ref.period_min == 40.0
    assert ref.value_at(ref.settle_min + 39.5) == pytest.approx(1.00)


def test_step_train_accepts_a_fall_slower_than_free_decay():
    """The other direction, which is genuinely controllable: the descent is
    stretched well past free decay, so the controller must add light to brake it."""
    ref = _osc(t_low_min=1.0, t_rise_min=3.0, t_high_min=8.0, t_fall_min=38.0)
    assert ref.period_min == 50.0
    s = ref.settle_min
    # fall spans [12, 50): 38 min to shed 0.30 CNR, vs ~17.5 min of free decay
    assert ref.segment_at(s + 12.0) == "fall"
    assert ref.value_at(s + 12.0) == pytest.approx(1.15)
    assert ref.value_at(s + 31.0) == pytest.approx(1.00)  # halfway down
    assert ref.segment_at(s + 49.9) == "fall"


def test_step_train_rejects_high_below_low():
    with pytest.raises(ValueError, match="must exceed low"):
        _osc(low=1.2, high=0.9)


def test_phase_groups_spread_cells_evenly_and_are_fixed_per_cell():
    ref = _osc(n_phase_groups=4)
    offsets = []
    for particle in range(8):
        st = CellState()
        st.particle = particle
        offsets.append(ref.phase_offset_min(types.SimpleNamespace(state=st)))
    assert offsets == [0.0, 10.0, 20.0, 30.0] * 2
    # aligned phase is the explicit single-group case
    st = CellState()
    st.particle = 3
    assert _osc(n_phase_groups=1).phase_offset_min(
        types.SimpleNamespace(state=st)) == 0.0


def test_phase_offset_shifts_a_cells_reference():
    ref = _osc(n_phase_groups=4)
    s = ref.settle_min
    # a cell offset by half a period sees the opposite part of the waveform
    assert ref.value_at(s + 0.0, 0.0) == pytest.approx(0.85)
    assert ref.value_at(s + 0.0, 20.0) == pytest.approx(1.15)


def test_step_train_rows_are_shared_across_cells_in_the_same_phase_group():
    """The row depends only on (timestep, offset), so at most n_phase_groups rows
    are computed however many cells there are — this is on the CEM's hot path."""
    ref = _osc(n_phase_groups=4)
    calls = {"n": 0}
    inner = ref.value_at

    def counting(t, off=0.0):
        calls["n"] += 1
        return inner(t, off)

    ref.value_at = counting
    cells = _cells(40)
    for i, f in enumerate(cells):
        f.state.particle = i
    vals = ref.values(GoalContext(0, 100, cells), horizon=30,
                      device=torch.device("cpu"))
    assert vals.shape == (40, 30)
    assert calls["n"] == 4 * 30, "one row per phase group, not per cell"
    # cells in the same group really do get the same row
    assert torch.equal(vals[0], vals[4])
    assert not torch.equal(vals[0], vals[2])


def test_oscillation_reference_is_per_cell_over_the_horizon():
    obj = oscillation(
        low=0.85, high=1.15, t_low_min=8.0, t_rise_min=2.0,
        t_high_min=15.0, t_fall_min=15.0,
        settle_periods=2.0, n_phase_groups=4,
    )
    cells = _cells(4)
    for i, f in enumerate(cells):
        f.state.particle = i
    ctx = GoalContext(0, 80, cells)  # first frame after settle
    tgt = obj.targets(ctx, horizon=4, device=torch.device("cpu"))
    assert tgt.shape == (4, 4)
    assert tgt[0].tolist() == pytest.approx([0.85] * 4)   # phase 0: low hold
    assert tgt[2, 0] == pytest.approx(1.15)               # phase 2: +20 min, high


def test_oscillation_annotations_carry_the_logging_contract():
    obj = oscillation(
        low=0.85, high=1.15, t_low_min=8.0, t_rise_min=2.0,
        t_high_min=15.0, t_fall_min=15.0,
        settle_periods=2.0, n_phase_groups=4,
    )
    cells = _cells(2)
    cells[0].state.particle = 0
    cells[1].state.particle = 2
    notes = obj.annotate(GoalContext(0, 80, cells))
    assert notes[0] == {"r_t": pytest.approx(0.85), "segment": "low_hold",
                        "phase_offset_min": 0.0}
    assert notes[1] == {"r_t": pytest.approx(1.15), "segment": "high_hold",
                        "phase_offset_min": 20.0}
    # during settle every cell reports the settle label
    early = obj.annotate(GoalContext(0, 3, cells))
    assert [n["segment"] for n in early] == ["settle", "settle"]


# ---------------------------------------------------------------------------
# frequency staircase
# ---------------------------------------------------------------------------

# The shipped sweep: at each period the amplitude is the largest that period's
# fall can deliver, so every block sits near its own decay limit and the fall is
# never the trivially binding constraint.
STAIRCASE_BLOCKS = [
    {"period": 50, "low": 0.87, "high": 1.17,
     "t_low_min": 12.0, "t_rise_min": 2.0, "t_high_min": 18.0, "t_fall_min": 18.0,
     "n_cycles": 2},
    {"period": 35, "low": 0.89, "high": 1.15,
     "t_low_min": 8.0, "t_rise_min": 2.0, "t_high_min": 13.0, "t_fall_min": 12.0,
     "n_cycles": 2},
    {"period": 25, "low": 0.92, "high": 1.12,
     "t_low_min": 6.0, "t_rise_min": 2.0, "t_high_min": 9.0, "t_fall_min": 8.0,
     "n_cycles": 3},
    {"period": 18, "low": 0.945, "high": 1.095,
     "t_low_min": 4.0, "t_rise_min": 2.0, "t_high_min": 6.0, "t_fall_min": 6.0,
     "n_cycles": 3},
]


def _stair(**kw):
    blocks = [{k: v for k, v in b.items() if k != "period"}
              for b in STAIRCASE_BLOCKS]
    params = dict(blocks=blocks, settle_min=100.0, n_phase_groups=1)
    params.update(kw)
    return FrequencyStaircaseReference(**params)


def test_staircase_block_and_sweep_durations():
    ref = _stair()
    assert [r.period_min for r in ref.refs] == [50.0, 35.0, 25.0, 18.0]
    assert ref.block_min == [100.0, 70.0, 75.0, 54.0]
    assert ref.sweep_min == 299.0


def test_staircase_amplitude_shrinks_but_the_mean_does_not():
    """The sweep varies frequency at (near) constant time-mean CNR, so a drift
    difference between blocks cannot be an exposure difference.

    Mean of a trapezoid over one period is ``low + A * X / period`` where
    ``X = t_high + (t_rise + t_fall) / 2`` — a linear ramp contributes its
    midpoint.
    """
    ref = _stair()
    amps = [r.amplitude for r in ref.refs]
    assert amps == pytest.approx([0.30, 0.26, 0.20, 0.15])
    assert amps == sorted(amps, reverse=True), "amplitude must fall with period"

    means = []
    for r in ref.refs:
        x = r.t_high_min + (r.t_rise_min + r.t_fall_min) / 2
        means.append(r.low + r.amplitude * x / r.period_min)
    assert means == pytest.approx([1.038, 1.039, 1.032, 1.028], abs=5e-4)
    assert max(means) - min(means) < 0.012


def test_staircase_holds_low_through_settle_then_starts_the_first_block():
    ref = _stair()
    for t in range(0, 100):
        assert ref.segment_at(t) == "settle"
        assert ref.value_at(t) == pytest.approx(0.87)
    assert ref._locate(100.0, 0.0) == (pytest.approx(0.87), "low_hold", 0, 0)


def test_staircase_blocks_end_on_a_completed_fall():
    """Each block spans a whole number of its own cycles, so every boundary sits at
    the bottom of a fall with a low hold immediately after — never mid-ramp, and
    never mid-high-hold."""
    ref = _stair()
    for start in ref._starts[1:] + [ref.sweep_min]:
        before = ref._locate(100.0 + start - 0.01, 0.0)
        after = ref._locate(100.0 + start + 0.01, 0.0)
        assert before[1] == "fall", f"boundary at {start} is mid-{before[1]}"
        assert after[1] == "low_hold"
        # the fall has run to completion, so `before` is already at that block's low
        assert before[0] == pytest.approx(ref.refs[before[2]].low, abs=1e-3)


def test_staircase_boundary_steps_are_small():
    """The price of stepping `low` with the period: the reference jumps by the
    difference between adjacent blocks' lows. Going up the staircase these are
    small upward steps; the sweep wrap is one larger downward step."""
    ref = _stair()
    lows = [r.low for r in ref.refs]
    steps = [b - a for a, b in zip(lows, lows[1:])]
    assert steps == pytest.approx([0.02, 0.03, 0.025])
    assert all(s > 0 for s in steps), "low rises as the period shortens"

    wrap = lows[0] - lows[-1]
    assert wrap == pytest.approx(-0.075)
    # the wrap is downward and must be absorbed inside the first block's low hold:
    # free decay 0.945 -> 0.87 toward rest 0.82 is tau * ln(0.125/0.05) = 6.7 min
    assert ref.refs[0].t_low_min >= 6.7


def test_staircase_repeats_and_counts_sweeps():
    ref = _stair()
    for t in np.linspace(0, ref.sweep_min, 61):
        assert ref.value_at(100.0 + t) == pytest.approx(
            ref.value_at(100.0 + t + ref.sweep_min))
    assert ref._locate(100.0 + 10.0, 0.0)[3] == 0
    assert ref._locate(100.0 + ref.sweep_min + 10.0, 0.0)[3] == 1
    assert ref._locate(100.0 + 2 * ref.sweep_min + 10.0, 0.0)[3] == 2


def test_staircase_phase_shifts_the_whole_schedule():
    """Offsetting moves where a cell is in the *sweep*, not just within a block,
    so the waveform each cell sees is unbroken and its blocks land at different
    wall-clock times."""
    ref = _stair(n_phase_groups=4)
    offsets = []
    for particle in range(8):
        st = CellState()
        st.particle = particle
        offsets.append(ref.phase_offset_min(types.SimpleNamespace(state=st)))
    assert offsets == [0.0, 12.5, 25.0, 37.5] * 2

    # a cell offset by 100 min is a whole block ahead: at sweep time 0 it is in
    # block 1 while the unoffset cell is still starting block 0
    assert ref._locate(100.0, 0.0)[2] == 0
    assert ref._locate(100.0, 100.0)[2] == 1
    # and the settle window is common — the offset does not shorten it
    assert ref.segment_at(99.0, 37.5) == "settle"


def test_staircase_annotations_carry_the_block_contract():
    obj = build_objective({
        "type": "frequency_staircase",
        "blocks": [{k: v for k, v in b.items() if k != "period"}
                   for b in STAIRCASE_BLOCKS],
        "settle_min": 100.0, "n_phase_groups": 1,
    })
    cells = _cells(1)
    notes = obj.annotate(GoalContext(0, 100, cells))
    assert notes[0] == {
        "r_t": pytest.approx(0.87), "segment": "low_hold", "phase_offset_min": 0.0,
        "block_index": 0, "sweep_index": 0, "block_period_min": 50.0,
    }
    # block starts within a sweep are [0, 100, 170, 245]; 130 is inside block 1
    late = obj.annotate(GoalContext(0, 100 + 130, cells))
    assert late[0]["block_index"] == 1
    assert late[0]["block_period_min"] == 35.0
    # settle keeps the schema, with sentinel indices
    early = obj.annotate(GoalContext(0, 3, cells))
    assert early[0]["segment"] == "settle"
    assert early[0]["block_index"] == -1
    assert early[0]["block_period_min"] is None


def test_staircase_rejects_a_fractional_block():
    with pytest.raises(ValueError, match="whole number of its own cycles"):
        _stair(blocks=[{"low": 0.87, "high": 1.17, "t_low_min": 12.0,
                        "t_rise_min": 2.0, "t_high_min": 18.0, "t_fall_min": 18.0,
                        "n_cycles": 0}])


def test_staircase_reference_is_per_cell_over_the_horizon():
    obj = build_objective({
        "type": "frequency_staircase",
        "blocks": [{k: v for k, v in b.items() if k != "period"}
                   for b in STAIRCASE_BLOCKS],
        "settle_min": 100.0, "n_phase_groups": 4,
    })
    cells = _cells(4)
    for i, f in enumerate(cells):
        f.state.particle = i
    tgt = obj.targets(GoalContext(0, 100, cells), horizon=30,
                      device=torch.device("cpu"))
    assert tgt.shape == (4, 30)
    assert tgt[0, 0] == pytest.approx(0.87)          # phase 0: block 0 low hold
    assert not torch.equal(tgt[0], tgt[2])           # phases really differ


def test_a_period_longer_than_the_horizon_is_allowed():
    """No reference binds against the control horizon any more.

    The old bound refused ``period > 2H`` on the grounds that the controller
    could never see a transition coming. It cannot: warning time before a
    transition is H, always — a transition at T enters the window at t = T - H
    whatever the period — because the plan re-solves and advances one frame at a
    time. The surviving constraint is the hard cap H <= future_len, in runtime.
    """
    obj = oscillation(
        low=0.85, high=1.15, t_low_min=8.0, t_rise_min=2.0,
        t_high_min=15.0, t_fall_min=15.0,
    )
    cells = _cells(1)
    tgt = obj.targets(GoalContext(0, 80, cells), horizon=4,
                      device=torch.device("cpu"))
    assert tgt.shape == (1, 4)
    assert not hasattr(obj, "validate_horizon")
    assert not hasattr(obj.reference, "validate_horizon")


# ---------------------------------------------------------------------------
# staggered cadence MPC
# ---------------------------------------------------------------------------


class CountingToyPlant(ToyPlant):
    """ToyPlant that records how much rollout work it was asked to do, so a test
    can assert idle cells never reach the (expensive) search."""

    def __init__(self, horizon=2):
        super().__init__(horizon)
        self.rollout_calls = 0
        self.rollout_batch = 0

    def rollout(self, h, c, cnr_fb, fut):
        self.rollout_calls += 1
        self.rollout_batch += fut.shape[0]
        return super().rollout(h, c, cnr_fb, fut)


def _staggered_cells(particles):
    """Cells with explicit particle ids — the staggered controller phases on
    ``particle % k``."""
    cells = []
    for p in particles:
        st = CellState()
        st.particle = p
        cells.append(CellFrame(state=st, cnr_norm=1.0, fov_density=1.0,
                               n_cells_200px=1.0))
    return cells


def _state_for(cells, plant):
    n = len(cells)
    h = torch.zeros(1, n, 4)
    return h, h.clone(), torch.zeros(n, 1)


def test_staggered_only_the_due_partition_is_stimulated():
    """k=3: on frame t, exactly the cells with particle % 3 == t % 3 may dose."""
    plant = ToyPlant(horizon=4)
    ctrl = StaggeredCadenceMPC(LEVELS, k=3, n_samples=32, n_iters=2)
    cells = _staggered_cells([0, 1, 2, 3, 4, 5])  # phases 0,1,2,0,1,2
    for t in range(3):
        ctx = GoalContext(0, t, cells)
        ms = ctrl.solve(plant, *_state_for(cells, plant), hold(0.5), ctx)
        due = [p for p in range(6) if p % 3 == t % 3]
        for i, m in enumerate(ms.tolist()):
            if i in due:
                assert m > 0, f"due cell {i} at t={t} should dose"
            else:
                assert m == 0, f"non-due cell {i} at t={t} must be 0"


def test_staggered_idle_frame_runs_no_rollout():
    """If no cell is due this frame, the search is skipped entirely — the whole
    point of staggering for compute."""
    plant = CountingToyPlant(horizon=4)
    ctrl = StaggeredCadenceMPC(LEVELS, k=3, n_samples=32, n_iters=2)
    cells = _staggered_cells([0, 3])  # both phase 0
    ms = ctrl.solve(plant, *_state_for(cells, plant), hold(0.5),
                    GoalContext(0, 1, cells))  # t%3==1, nobody due
    assert ms.tolist() == [0.0, 0.0]
    assert plant.rollout_calls == 0


def test_staggered_rollout_batch_covers_only_due_cells():
    """Compute scales with the due subset, not the whole population."""
    all_due = CountingToyPlant(horizon=4)
    part_due = CountingToyPlant(horizon=4)
    ctrl = StaggeredCadenceMPC(LEVELS, k=3, n_samples=16, n_iters=1)
    cells = _staggered_cells([0, 1, 2, 3, 4, 5])

    # t=0 -> phases {0,3} due (2 cells); build a comparison where all 6 are due
    ctrl.solve(part_due, *_state_for(cells, plant := ToyPlant(4)), hold(0.5),
               GoalContext(0, 0, cells))
    ctrl_k1 = StaggeredCadenceMPC(LEVELS, k=1, n_samples=16, n_iters=1)  # everyone due
    ctrl_k1.solve(all_due, *_state_for(cells, plant), hold(0.5),
                  GoalContext(0, 0, cells))

    # 2 of 6 due -> a third of the rollout batch
    assert part_due.rollout_batch * 3 == all_due.rollout_batch


def test_staggered_tick_spacing_is_exactly_k():
    """One cell, followed over time, may only dose on its own every-k-th frame."""
    plant = ToyPlant(horizon=4)
    ctrl = StaggeredCadenceMPC(LEVELS, k=4, n_samples=32, n_iters=2)
    cell = _staggered_cells([2])  # phase 2 -> due at t = 2, 6, 10, ...
    doses_when_nonzero = []
    for t in range(12):
        ctx = GoalContext(0, t, cell)
        ms = ctrl.solve(plant, *_state_for(cell, plant), hold(0.5), ctx)
        if ms.item() > 0:
            doses_when_nonzero.append(t)
    assert doses_when_nonzero == [2, 6, 10]


def test_staggered_requires_a_zero_dose_level():
    """No 0 ms level -> no way to represent a coast frame -> refuse to build."""
    with pytest.raises(ValueError, match="0 ms level"):
        StaggeredCadenceMPC(dose_levels(200, 800, 4), k=3)


def test_staggered_is_deterministic_per_frame():
    plant = ToyPlant(horizon=4)
    cells = _staggered_cells([0, 1, 2])
    args = (plant, *_state_for(cells, plant), hold(0.5), GoalContext(0, 0, cells))
    a = StaggeredCadenceMPC(LEVELS, k=3, n_samples=32, n_iters=2, seed=1).solve(*args)
    b = StaggeredCadenceMPC(LEVELS, k=3, n_samples=32, n_iters=2, seed=1).solve(*args)
    assert a.tolist() == b.tolist()


def test_build_staggered_from_spec():
    ctrl = build_controller({"type": "staggered_mpc", "k": 4, "n_samples": 64}, LEVELS)
    assert isinstance(ctrl, StaggeredCadenceMPC)
    assert ctrl.k == 4 and ctrl.n_samples == 64
    assert ctrl.describe()["type"] == "staggered_mpc"
    assert ctrl.describe()["k"] == 4


# ---------------------------------------------------------------------------
# open_loop — the control arm
# ---------------------------------------------------------------------------


def test_open_loop_commands_the_identical_dose_to_every_cell():
    """The whole point of the arm. Cells here have wildly different feedback, which
    every other controller turns into different doses; this one must not."""
    plant = LeakyPlant(horizon=4)
    ctrl = build_controller(
        {"type": "open_loop", "sequence_ms": [0.0, 200.0, 800.0, 400.0]}, LEVELS
    )
    cells = _cells(6)
    h, c, _ = _state_for(cells, plant)
    cnr_fb = torch.tensor([[0.1], [0.9], [0.5], [0.2], [1.4], [0.7]])

    ms = ctrl.solve(plant, h, c, cnr_fb, hold(0.5), GoalContext(0, 0, cells))
    assert ms.tolist() == [0.0] * 6, "frame 0 of the sequence, for everyone"

    ms = ctrl.solve(plant, h, c, cnr_fb, hold(0.5), GoalContext(0, 2, cells))
    assert ms.tolist() == [800.0] * 6
    assert len(set(ms.tolist())) == 1, "feedback must not reach the dose"


def test_open_loop_follows_the_control_clock_not_faros_timestep():
    """The sequence has to line up with the reference waveform, and the waveform is
    clocked from the first controlled frame. An acquisition that ran earlier phases
    hands the server timestep 40 on its first controlled frame — reading that as the
    schedule index would slide the arm against the reference it was designed for."""
    plant = ToyPlant(horizon=2)
    ctrl = build_controller(
        {"type": "open_loop", "sequence_ms": [0.0, 200.0, 400.0, 600.0]}, LEVELS
    )
    cells = _cells(3)
    args = (plant, *_state_for(cells, plant), hold(0.5))

    for offset in (0, 40, 1000):
        got = [
            ctrl.solve(*args, GoalContext(0, offset + i, cells, control_frame=i))[0].item()
            for i in range(6)
        ]
        assert got == [0.0, 200.0, 400.0, 600.0, 0.0, 200.0], f"offset {offset}"


def test_open_loop_scores_the_plan_it_is_committed_to():
    """It never searches, but it must still report what the model believed would
    happen — that is what keeps plan_cost comparable with the closed-loop arms, and
    it makes every frame a model-error measurement under a known dose."""
    plant = ToyPlant(horizon=2)
    cells = _cells(2)
    args = (plant, *_state_for(cells, plant), hold(0.5), GoalContext(0, 0, cells))

    # ToyPlant: predicted cnr == dose/800. A sequence sitting exactly on the target
    # costs nothing; one sitting away from it costs the squared miss.
    on_target = build_controller({"type": "open_loop", "sequence_ms": [400.0]}, LEVELS)
    _ms, cost = on_target.plan(*args)
    assert cost.shape == (2,)
    assert cost.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)

    off_target = build_controller({"type": "open_loop", "sequence_ms": [800.0]}, LEVELS)
    _ms, cost_off = off_target.plan(*args)
    assert cost_off.tolist() == pytest.approx([0.25, 0.25], abs=1e-6)


def test_open_loop_describes_its_whole_schedule():
    """The arm IS its schedule. A run whose log only names the controller cannot be
    reconstructed afterwards."""
    ctrl = build_controller(
        {"type": "open_loop", "sequence_ms": [0.0, 0.0, 600.0, 200.0]}, LEVELS
    )
    d = ctrl.describe()
    assert d["type"] == "open_loop"
    assert d["sequence_ms"] == [0.0, 0.0, 600.0, 200.0]
    assert d["period_frames"] == 4
    assert d["mean_ms"] == pytest.approx(200.0)
    assert d["repeat"] is True


def test_open_loop_rejects_an_empty_or_negative_schedule():
    with pytest.raises(ValueError, match="non-empty"):
        build_controller({"type": "open_loop", "sequence_ms": []}, LEVELS)
    with pytest.raises(ValueError, match="negative"):
        build_controller({"type": "open_loop", "sequence_ms": [200.0, -1.0]}, LEVELS)


def test_open_loop_without_repeat_holds_its_last_dose():
    """Outrunning a designed sequence must degrade to a constant, not go dark —
    silently stopping stimulation partway through a 12 h run would look like a
    control result."""
    ctrl = build_controller(
        {"type": "open_loop", "sequence_ms": [200.0, 600.0], "repeat": False}, LEVELS
    )
    assert [ctrl.dose_at(i) for i in range(5)] == [200.0, 600.0, 600.0, 600.0, 600.0]
