import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch

    from optoerk.serving.config import ServerConfig
    from optoerk.serving.control import dose_levels
    from optoerk.serving.objectives import GoalContext, Prediction
    from optoerk.serving.policy import PolicyRouter, arm_map, load_policy_file
    from optoerk.serving.replay import (
        open_loop_ensemble,
        score_open_loop,
        simulate_open_loop,
    )
    from optoerk.serving.runtime import CellFrame
    from optoerk.serving.state import CellState

    return (
        Path,
        PolicyRouter,
        ServerConfig,
        alt,
        arm_map,
        load_policy_file,
        mo,
        np,
        pl,
        score_open_loop,
        simulate_open_loop,
    )


@app.cell
def _(mo):
    mo.md("""
    # Open-loop schedule designer

    Computes the fixed dose schedule that an **open-loop control arm** will play:
    the same command to every cell, every frame, with no feedback.

    The arm only means something if the schedule is a *fair opponent*. Beating a
    badly chosen fixed pattern says nothing about feedback. So the schedule here
    is optimised against **the same model, the same objective and the same dose
    ladder** the closed-loop arms use — the best open-loop schedule this model
    can produce.

    Two things this cannot escape, both of which belong in the write-up:

    - the schedule **inherits the model's error**, because it is designed against
      the model. If the model is wrong the open-loop arm suffers more than the
      closed-loop one — which is the mechanism under test, not a flaw, but it
      means the claim is "feedback beats the best open-loop schedule *this model*
      can design".
    - the optimiser is free to spend more or less light than the closed-loop arms.
      **Dose is not matched by construction**, so the achieved mean dose is
      reported below and has to be carried into the analysis.
    """)
    return


@app.cell
def _(mo):
    policy_input = mo.ui.text(
        value="policies/policy_8fov_openloop.toml",
        label="Policy file", full_width=True,
    )
    policy_input
    return (policy_input,)


@app.cell
def _(Path, arm_map, load_policy_file, mo, policy_input):
    _path = Path(policy_input.value.strip())
    mo.stop(not _path.exists(), mo.md(f"**No such policy:** `{_path}`"))
    policy = load_policy_file(_path)
    arms = arm_map(policy)

    # One schedule per distinct (reference, ladder) — every open-loop FOV sharing a
    # reference must play the SAME schedule, or the repeats are not repeats.
    _rows = []
    for _fov, _spec in policy.fov.items():
        _ctrl = _spec.controller or {}
        _rows.append({
            "fov": int(_fov),
            "arm": arms[int(_fov)],
            "controller": _ctrl.get("type", "constant_dose"),
            "objective": (_spec.objective or {}).get("type"),
        })
    fov_table = sorted(_rows, key=lambda r: r["fov"])
    open_loop_fovs = [r["fov"] for r in fov_table if r["controller"] == "open_loop"]
    mo.vstack([
        mo.md(f"**{len(fov_table)} FOVs**, open-loop on {open_loop_fovs}"),
        mo.ui.table(fov_table, selection=None),
    ])
    return open_loop_fovs, policy


@app.cell
def _(mo, open_loop_fovs):
    # Design for ONE open-loop FOV at a time; its repeats share the answer.
    fov_sel = mo.ui.dropdown(
        options={str(f): f for f in open_loop_fovs},
        value=str(open_loop_fovs[0]) if open_loop_fovs else None,
        label="Design the schedule for FOV",
    )
    n_ensemble = mo.ui.slider(
        8, 256, value=64, step=8, label="Ensemble size (nominal cells)", full_width=True
    )
    n_candidates = mo.ui.slider(
        32, 512, value=256, step=32, label="CEM candidates per iteration", full_width=True
    )
    n_iters = mo.ui.slider(1, 8, value=4, step=1, label="CEM iterations", full_width=True)
    mo.vstack([fov_sel, n_ensemble, n_candidates, n_iters])
    return fov_sel, n_candidates, n_ensemble, n_iters


@app.cell
def _(PolicyRouter, ServerConfig, fov_sel, mo, policy):
    # Placeholders are expected here — this notebook exists to REMOVE them, so it
    # must be able to load a file that is still gated.
    _pf = policy.model_copy(update={"placeholders_resolved": True})
    cfg = ServerConfig(warmup=False, gpu_sample_interval_s=0, dark_baseline=False)
    router = PolicyRouter(cfg, _pf)
    engine = router.engine_for(fov_sel.value)
    objective = engine.objective
    mo.stop(
        not hasattr(engine, "rollout"),
        mo.md("**No checkpoint** — this needs a real model to design against."),
    )
    mo.md(
        f"Engine `{type(engine).__name__}` · channels `{engine.channels}` · "
        f"ladder `{engine.controller.levels_ms.tolist()}` · "
        f"objective `{objective.describe()['type']}`"
    )
    return engine, objective


@app.cell
def _(engine, mo, objective):
    # The schedule must repeat on the reference's own period, or it slides out of
    # phase over the run. A constant reference has no period, so a short block is
    # enough and 1 frame is the natural answer.
    _ref = objective.describe().get("reference") or {}
    _period = _ref.get("period_min")
    _dt = float(_ref.get("frame_interval_min", 1.0))
    period_frames = int(round(_period / _dt)) if _period else 1
    settle_frames = int(round(float(_ref.get("settle_min", 0.0)) / _dt))
    ladder = [float(v) for v in engine.controller.levels_ms]
    mo.md(
        f"Reference period **{period_frames} frames**, settle **{settle_frames} frames**. "
        f"The schedule is {period_frames} entries drawn from `{ladder}`, repeated."
    )
    return ladder, period_frames, settle_frames


@app.cell
def _(mo):
    mo.md("""
    ## The simulator

    The model stands in for the cells: each frame its own decoder is rolled one
    step under the commanded dose, and that prediction becomes the next observed
    CNR. This is the same construction `replay.simulate_closed_loop` uses, and it
    carries the same warning — it scores against the model's beliefs, so it is a
    **prediction of where the borders are, not a measurement of them**.

    Every candidate schedule is simulated at once: the batch is
    `(candidates × ensemble)` pseudo-cells, so a CEM iteration costs one pass of
    encoder steps rather than one per candidate.
    """)
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Design the schedule", kind="success")
    mo.md("Optimisation is the expensive step, so it is gated."), run_btn
    return (run_btn,)


@app.cell
def _(
    engine,
    ladder,
    mo,
    n_candidates,
    n_ensemble,
    n_iters,
    np,
    objective,
    period_frames,
    run_btn,
    score_open_loop,
    settle_frames,
    simulate_open_loop,
):
    mo.stop(not run_btn.value, mo.md("_Press the button to run the optimiser._"))

    # CEM over the discrete ladder, one categorical per slot of the period. Same
    # optimiser family as SequenceMPC, but solving ONCE for a schedule that repeats,
    # instead of re-solving every frame for one cell.
    _L, _P = len(ladder), period_frames
    _rng = np.random.default_rng(0)
    _probs = np.full((_P, _L), 1.0 / _L)
    _n_elite = max(2, int(round(n_candidates.value * 0.125)))
    # Simulate settle plus two full periods, and score only the last period: the
    # first cycle is the transient from a cold encoder, which no schedule can fix
    # and which would otherwise dominate the cost.
    _eval_from = settle_frames + _P
    _n_frames = _eval_from + _P
    _levels = np.asarray(ladder, dtype=np.float32)

    _best_seq, _best_cost, _history = None, np.inf, []
    for _it in range(n_iters.value):
        _idx = np.stack(
            [_rng.choice(_L, size=n_candidates.value, p=_probs[_p]) for _p in range(_P)],
            axis=1,
        )                                                   # (C, P)
        if _best_seq is not None:                           # elitism: keep the best
            _idx[0] = _best_seq
        _seq = _levels[_idx]                                # (C, P) in ms
        _cnr = simulate_open_loop(engine, _seq, _n_frames, n_ensemble.value, seed=_it)
        _cost = score_open_loop(engine, objective, _cnr[:, :, _eval_from:], _eval_from)

        _order = np.argsort(_cost)
        _elite = _idx[_order[:_n_elite]]
        if _cost[_order[0]] < _best_cost:
            _best_cost = float(_cost[_order[0]])
            _best_seq = _idx[_order[0]].copy()
        # refit, smoothed back toward uniform so it cannot collapse on iteration 1
        _counts = np.stack(
            [np.bincount(_elite[:, _p], minlength=_L) for _p in range(_P)]
        ).astype(float)
        _probs = 0.9 * (_counts / _counts.sum(axis=1, keepdims=True)) + 0.1 / _L
        _history.append({"iteration": _it, "best_cost": _best_cost,
                         "mean_cost": float(_cost.mean())})

    best_sequence = _levels[_best_seq].tolist()
    best_cost = _best_cost
    cem_history = _history
    mo.md(f"Best cost **{best_cost:.5f}** after {n_iters.value} iterations.")
    return best_cost, best_sequence, cem_history


@app.cell
def _(alt, best_sequence, cem_history, mo, pl):
    _h = pl.DataFrame(cem_history)
    _conv = (
        alt.Chart(_h).mark_line(point=True)
        .encode(x=alt.X("iteration:Q", title="CEM iteration"),
                y=alt.Y("best_cost:Q", title="best cost", scale=alt.Scale(zero=False)),
                tooltip=["iteration", "best_cost", "mean_cost"])
        .properties(width=320, height=200, title="Convergence")
    )
    _s = pl.DataFrame({"frame": range(len(best_sequence)), "ms": best_sequence})
    _sched = (
        alt.Chart(_s).mark_bar()
        .encode(x=alt.X("frame:Q", title="frame within the period"),
                y=alt.Y("ms:Q", title="commanded exposure (ms)"),
                tooltip=["frame", "ms"])
        .properties(width=420, height=200, title="Designed schedule")
    )
    mo.ui.altair_chart(_conv | _sched)
    return


@app.cell
def _(
    alt,
    best_sequence,
    engine,
    mo,
    n_ensemble,
    np,
    objective,
    pl,
    settle_frames,
    simulate_open_loop,
):
    # What the model predicts this schedule actually achieves, against the reference
    # it was designed for. The provenance block wants the mean dose from here.
    _P = len(best_sequence)
    _n = settle_frames + 3 * _P
    _traj = simulate_open_loop(
        engine, np.asarray([best_sequence], dtype=np.float32), _n, n_ensemble.value, seed=99
    )[0]                                                     # (ens, n_frames)

    _ref = objective.describe().get("reference") or {}
    _dt = float(_ref.get("frame_interval_min", 1.0))
    _r = [
        float(objective.reference.value_at(t * _dt, 0.0)[0])
        if hasattr(objective.reference, "value_at") else None
        for t in range(_n)
    ]
    _df = pl.DataFrame({
        "frame": list(range(_n)),
        "median": np.median(_traj, axis=0),
        "p25": np.percentile(_traj, 25, axis=0),
        "p75": np.percentile(_traj, 75, axis=0),
    })
    _band = alt.Chart(_df).mark_area(opacity=0.25).encode(
        x="frame:Q", y=alt.Y("p25:Q", title="predicted CNR", scale=alt.Scale(zero=False)),
        y2="p75:Q")
    _line = alt.Chart(_df).mark_line(color="#117733").encode(x="frame:Q", y="median:Q")
    mean_dose_ms = float(np.mean(best_sequence))
    mo.vstack([
        mo.md(
            f"**Mean dose {mean_dose_ms:.1f} ms/frame** over the {_P}-frame period. "
            "Carry this into the policy's provenance block and compare it against the "
            "closed-loop arms' achieved mean dose — the arms are not dose-matched by "
            "construction."
        ),
        mo.ui.altair_chart((_band + _line).properties(width=700, height=240,
                                                      title="Predicted open-loop trajectory (IQR)")),
    ])
    return (mean_dose_ms,)


@app.cell
def _(best_cost, best_sequence, fov_sel, mean_dose_ms, mo):
    _seq = ", ".join(f"{v:g}" for v in best_sequence)
    mo.md(
        f"""
        ## Paste into the policy

        Replaces the provisional `sequence_ms` on **every** open-loop FOV that shares
        this reference — the repeats must play the identical schedule.

        ```toml
        controller = {{ type = "open_loop", repeat = true, sequence_ms = [{_seq}] }}
        ```

        And into the PARAMETER PROVENANCE block:

        ```
        open loop    fov {fov_sel.value} sequence mean dose {mean_dose_ms:.1f} ms
                     designed cost (model's own belief) {best_cost:.5f}
        ```

        The cost is the model scoring itself, so it is not evidence the schedule
        works — it is the prediction the run exists to falsify.
        """
    )
    return


if __name__ == "__main__":
    app.run()
