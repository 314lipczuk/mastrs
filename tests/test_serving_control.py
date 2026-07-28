"""Objectives and controllers, tested against a toy plant.

No checkpoint and no cluster mount: the controllers only ever touch the model
through the plant interface (``horizon``, ``device``, ``std_fluence``,
``rollout``, ``denorm_cnr``), so a five-line analytic plant exercises them
completely — and lets the optimum be computed by hand, which a real model never
would.
"""
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
    GoalContext,
    TargetTrajectory,
    build_objective,
    gated,
    hold,
    schedule,
)
from optoerk.serving.runtime import CellFrame
from optoerk.serving.state import CellState

LEVELS = dose_levels(0.0, 800.0, 5)  # [0, 200, 400, 600, 800]


class ToyPlant:
    """Instantaneous, memoryless response: predicted CNR == the dose applied that
    step, scaled to [0, 1]. Trivial, but it makes the optimal plan exact:
    to track target trajectory ``g`` you must command ``g`` step by step.
    """

    def __init__(self, horizon=2):
        self.horizon = horizon
        self.device = torch.device("cpu")

    def std_fluence(self, ms):
        return ms / 800.0

    def denorm_cnr(self, pred_std):
        return pred_std

    def rollout(self, h, c, cnr_fb, fut):
        return fut[:, :, 0]  # (B, H)


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


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------


def test_hold_cost_is_squared_error_to_target():
    obj = hold(0.5)
    cells = _cells(3)
    ctx = GoalContext(fov=0, timestep=0, cells=cells)
    pred = torch.zeros(3, 2, 4)  # (N, M, H) all-zero prediction
    cost = obj.cost(pred, ctx)
    assert cost.shape == (3, 2)
    assert torch.allclose(cost, torch.full((3, 2), 0.25))


def test_target_trajectory_rejects_wrong_length_target():
    obj = TargetTrajectory(lambda cell, t: [1.0, 2.0, 3.0])
    ctx = GoalContext(fov=0, timestep=0, cells=_cells(1))
    with pytest.raises(ValueError, match="expected a scalar or 2"):
        obj.cost(torch.zeros(1, 1, 2), ctx)


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
    assert obj.describe() == {"type": "hold", "target_cnr": 1.9}
    with pytest.raises(KeyError, match="unknown objective type"):
        build_objective({"type": "nope"})
    with pytest.raises(ValueError, match="needs a 'type'"):
        build_objective({"target_cnr": 1.0})
    with pytest.raises(TypeError, match="bad params"):
        build_objective({"type": "hold", "nonsense": 1})


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
    # step target: 0.0 at the first step, 1.0 at the second -> u = [0 ms, 800 ms]
    obj = TargetTrajectory(lambda cell, t: [0.0, 1.0])
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
