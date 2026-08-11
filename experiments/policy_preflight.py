import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from optoerk.core.utils import materials_path, results_write_path
    from optoerk.serving.objectives import build_objective
    from optoerk.serving.policy import arm_map, load_policy_file

    # Working constants. `REST` is the checkpoint's own CNR z-score mean — the CNR
    # a cell relaxes to with no light, which sets how long any fall takes.
    REST = 0.82
    TAU_ASSUMED = 7.3
    return (
        Path,
        arm_map,
        build_objective,
        load_policy_file,
        materials_path,
        math,
        mo,
        np,
        pl,
        plt,
        results_write_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Policy pre-flight

    **What this replaces.** `StepTrainReference` used to carry a `tau_decay_min`
    parameter and refuse references whose fall or period were short relative to it.
    That number was typed into the policy file on the same line as the durations it
    was checked against, so it could only ever catch a self-contradiction — and
    lowering it legalized anything. Meanwhile `high`, the parameter most likely to
    put a reference out of reach, was never checked at all.

    Feasibility lives here instead, where it can be argued against measured
    distributions and **recorded** in the policy file rather than asserted in it.

    **What this is not.** Not a gate. Two arms of the pattern-zoo run use an
    unreachable reference deliberately — a cell failing to track the ceiling ramp is
    the measurement. The job here is to state, before the run, *where the model
    thinks the borders are*, so the run can falsify it.

    Four sections:

    | § | question | needs |
    |---|---|---|
    | 1 | at what τ does each arm's fall stop being reachable? | nothing — pure arithmetic |
    | 2 | what is τ actually, and how spread? | `tier0_tau_episodes.parquet` |
    | 3 | what fraction of cells can reach each arm's `high`? | a dataset parquet |
    | 4 | can the controller track it, under the model? | a checkpoint (cluster) |

    Section 1 runs anywhere and is the one that would have caught the headroom
    problem both previous policy files worried about in prose.
    """)
    return


@app.cell
def _(mo):
    policy_input = mo.ui.text(
        value="policies/policy_10fov_patterns.toml",
        label="Policy file",
        full_width=True,
    )
    policy_input
    return (policy_input,)


@app.cell
def _(Path, arm_map, build_objective, load_policy_file, mo, policy_input):
    POLICY_PATH = Path(policy_input.value.strip())
    mo.stop(
        not POLICY_PATH.exists(),
        mo.md(f"**No such policy file:** `{POLICY_PATH}`"),
    )

    POLICY = load_policy_file(POLICY_PATH)
    ARMS = arm_map(POLICY)

    # One representative FOV per arm — every FOV in an arm has the same reference by
    # construction, and section 4 would otherwise pay for the replicates.
    ARM_FOV = {}
    for _fov in sorted(POLICY.fov):
        ARM_FOV.setdefault(ARMS[_fov], _fov)

    OBJECTIVES = {}
    for _arm, _fov in sorted(ARM_FOV.items()):
        _spec = POLICY.fov[_fov]
        OBJECTIVES[_arm] = build_objective(
            {
                **_spec.objective,
                **({"kernel": _spec.kernel} if _spec.kernel is not None else {}),
                **({"lambda_move": _spec.lambda_move} if _spec.lambda_move is not None else {}),
            }
        )

    mo.md(
        f"**`{POLICY_PATH.name}`** — {len(POLICY.fov)} FOVs, {len(ARM_FOV)} arms, "
        f"`placeholders_resolved = {POLICY.placeholders_resolved}`\n\n"
        + "\n".join(
            f"- arm {a}: fov {[f for f in sorted(POLICY.fov) if ARMS[f] == a]} "
            f"→ `{OBJECTIVES[a].reference.name}`"
            for a in sorted(ARM_FOV)
        )
    )
    return ARM_FOV, OBJECTIVES, POLICY, POLICY_PATH


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 — The fall-feasibility frontier

    A fall from `high` to `low` toward a resting CNR is exponential relaxation, and
    the controller cannot beat it: there is no inhibitory actuator, only light and
    decay. So the shortest possible fall is

    $$t_{\text{free}} = \tau \ln\frac{high - rest}{low - rest}$$

    which depends on **where in the range the swing sits**, not only on its size. A
    small swing at an elevated mean falls far faster than the same swing near rest.
    The deleted guard used `t_fall ≥ 1.5·τ`, which is that formula frozen at one
    particular ratio — `(high−rest)/(low−rest) = e^1.5 ≈ 4.5` — and drifts in both
    directions away from it.

    The useful number is not pass/fail at an assumed τ. It is **the τ at which each
    segment stops being reachable**, i.e. the headroom the design has against a
    refit. Inverting the formula at `t_fall`:

    $$\tau_{\max} = \frac{t_{\text{fall}}}{\ln\frac{high-rest}{low-rest}}$$

    A segment whose `τ_max` sits below the measured τ demands a fall the cells
    cannot deliver. Whether that is a bug or the point depends on the arm.
    """)
    return


@app.cell
def _(mo):
    rest_input = mo.ui.number(value=0.82, step=0.01, label="resting CNR")
    tau_input = mo.ui.number(value=7.3, step=0.1, label="assumed τ (min)")
    mo.hstack([rest_input, tau_input])
    return rest_input, tau_input


@app.cell
def _(math, pl):
    def segments_of(ref) -> list[dict]:
        """Flatten any reference into (label, low, high, t_fall, period) rows.

        A step train is one row; a frequency staircase is one row per block; a
        constant reference has no fall and contributes none.
        """
        name = getattr(ref, "name", "?")
        if name == "step_train":
            return [{
                "segment": "cycle", "low": ref.low, "high": ref.high,
                "t_fall_min": ref.t_fall_min, "period_min": ref.period_min,
            }]
        if name == "frequency_staircase":
            return [
                {
                    "segment": f"block {i} (P={r.period_min:.0f})",
                    "low": r.low, "high": r.high,
                    "t_fall_min": r.t_fall_min, "period_min": r.period_min,
                }
                for i, r in enumerate(ref.refs)
            ]
        return []


    def feasibility(objectives, rest: float, tau: float) -> pl.DataFrame:
        rows = []
        for arm, obj in sorted(objectives.items()):
            for seg in segments_of(obj.reference):
                lo_gap, hi_gap = seg["low"] - rest, seg["high"] - rest
                if lo_gap <= 0:
                    # `low` at or below rest is unreachable from above at ANY tau:
                    # free decay approaches rest asymptotically and never crosses it.
                    rows.append({
                        "arm": arm, **seg, "t_free_min": float("inf"),
                        "tau_max_min": 0.0, "headroom": 0.0,
                        "note": "low <= rest: unreachable at any tau",
                    })
                    continue
                ratio = math.log(hi_gap / lo_gap)
                t_free = tau * ratio
                tau_max = seg["t_fall_min"] / ratio if ratio > 0 else float("inf")
                rows.append({
                    "arm": arm, **seg,
                    "t_free_min": t_free,
                    "tau_max_min": tau_max,
                    "headroom": tau_max / tau,
                    "note": "",
                })
        return pl.DataFrame(rows)

    return feasibility, segments_of


@app.cell
def _(OBJECTIVES, feasibility, mo, pl, rest_input, tau_input):
    FEAS = feasibility(OBJECTIVES, rest_input.value, tau_input.value)
    mo.stop(FEAS.is_empty(), mo.md("No falling segments in this policy — nothing to check."))

    _view = FEAS.select(
        "arm", "segment",
        pl.col("low").round(3), pl.col("high").round(3),
        pl.col("t_fall_min").round(1),
        pl.col("t_free_min").round(2).alias("t_free"),
        pl.col("tau_max_min").round(2).alias("tau_max"),
        pl.col("headroom").round(2),
        "note",
    )
    _view
    return (FEAS,)


@app.cell
def _(FEAS, mo, pl, tau_input):
    _tight = FEAS.filter(pl.col("headroom") < 1.0)
    _thin = FEAS.filter((pl.col("headroom") >= 1.0) & (pl.col("headroom") < 1.3))

    _lines = [
        f"τ assumed **{tau_input.value} min**. `tau_max` is the largest τ at which "
        f"that segment's fall is still free-decay reachable; `headroom` is "
        f"`tau_max / τ`.",
        "",
    ]
    if _tight.height:
        _lines.append(
            "**Demanding a fall faster than free decay** (headroom < 1) — deliberate "
            "on a border-probing arm, a bug anywhere else:"
        )
        _lines += [
            f"- arm {r['arm']} {r['segment']}: needs {r['t_free_min']:.1f} min, "
            f"has {r['t_fall_min']:.0f} — reachable only up to τ = {r['tau_max_min']:.1f}"
            for r in _tight.iter_rows(named=True)
        ]
        _lines.append("")
    if _thin.height:
        _lines.append(
            "**Thin headroom** (< 1.3x). The D6 refit is expected to RAISE τ; these "
            "segments are the ones that stop being reachable when it does:"
        )
        _lines += [
            f"- arm {r['arm']} {r['segment']}: fails at τ = {r['tau_max_min']:.1f} min"
            for r in _thin.iter_rows(named=True)
        ]
        _lines.append("")
    if not _tight.height and not _thin.height:
        _lines.append("Every falling segment has >1.3x headroom against the assumed τ.")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(FEAS, mo, np, plt, tau_input):
    # The whole design, swept over tau: read off where each segment crosses.
    _taus = np.linspace(3.0, 20.0, 200)
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _cmap = plt.cm.tab10
    for _i, _r in enumerate(FEAS.iter_rows(named=True)):
        if not np.isfinite(_r["tau_max_min"]) or _r["tau_max_min"] <= 0:
            continue
        _ratio = _r["t_free_min"] / tau_input.value  # ln((hi-rest)/(lo-rest))
        _ax.plot(_taus, _taus * _ratio, color=_cmap(_i % 10), lw=1.6,
                 label=f"arm {_r['arm']} {_r['segment']}")
        _ax.axhline(_r["t_fall_min"], color=_cmap(_i % 10), ls=":", lw=1.0)
        _ax.plot([_r["tau_max_min"]], [_r["t_fall_min"]], "o",
                 color=_cmap(_i % 10), ms=6)
    _ax.axvline(tau_input.value, color="k", ls="--", lw=1.0)
    _ax.text(tau_input.value, _ax.get_ylim()[1] * 0.97, " assumed τ",
             va="top", fontsize=9)
    _ax.set_xlabel("τ (min)")
    _ax.set_ylabel("free-decay time required (min)")
    _ax.set_title("Solid = free decay needed; dotted = fall allowed; marker = crossing")
    _ax.legend(fontsize=8, ncol=2)
    _fig.tight_layout()
    mo.output.append(_ax.get_figure())
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 — τ, measured

    From `tier0_tau_episodes.parquet`, written next to `tier0_summary.json` by
    `experiments/tier0_diagnostics.py`. The summary carries only the median and IQR,
    which cannot answer the question that matters: a fall feasible at the median τ
    can still be untrackable for the slow half of the cells, and **that fraction** is
    what a policy has to be argued against.

    Only `identifiable` fits count — a τ longer than its own observation window is an
    extrapolation, not a measurement (tier0 excludes them from its verdict too).
    """)
    return


@app.cell
def _(mo, results_write_path):
    tau_path_input = mo.ui.text(
        value=str(results_write_path()),
        label="Directory holding tier0_tau_episodes.parquet (searched recursively)",
        full_width=True,
    )
    tau_path_input
    return (tau_path_input,)


@app.cell
def _(FEAS, Path, mo, np, pl, tau_path_input):
    _root = Path(tau_path_input.value.strip())
    _hits = sorted(_root.rglob("tier0_tau_episodes.parquet")) if _root.exists() else []

    mo.stop(
        not _hits,
        mo.md(
            f"**No `tier0_tau_episodes.parquet` under `{_root}`.** Run "
            f"`experiments/tier0_diagnostics.py` first (on the cluster — it needs the "
            f"results mount). Section 1 stands without it; sections 2 and 3 do not."
        ),
    )

    TAU_SRC = _hits[-1]
    tau_df = pl.read_parquet(TAU_SRC)
    tau_ident = tau_df.filter(pl.col("identifiable"))
    TAUS = tau_ident["tau_min"].drop_nulls().to_numpy()

    _q = np.percentile(TAUS, [10, 25, 50, 75, 90]) if len(TAUS) else []
    _rows = []
    for _r in FEAS.iter_rows(named=True):
        if not np.isfinite(_r["tau_max_min"]):
            continue
        _frac = float((TAUS <= _r["tau_max_min"]).mean()) if len(TAUS) else float("nan")
        _rows.append({
            "arm": _r["arm"], "segment": _r["segment"],
            "tau_max_min": round(_r["tau_max_min"], 2),
            "frac_cells_fall_reachable": round(_frac, 3),
        })
    tau_feas = pl.DataFrame(_rows)

    mo.output.append(mo.md(
        f"`{TAU_SRC.parent.name}/{TAU_SRC.name}` — {tau_ident.height} identifiable "
        f"of {tau_df.height} fits, source `{tau_df['source'][0] if tau_df.height else '?'}`.\n\n"
        f"τ percentiles (min): "
        f"p10 **{_q[0]:.1f}** · p25 **{_q[1]:.1f}** · p50 **{_q[2]:.1f}** · "
        f"p75 **{_q[3]:.1f}** · p90 **{_q[4]:.1f}**\n\n"
        f"`frac_cells_fall_reachable` is the fraction of measured cells whose own τ "
        f"is at or below that segment's `tau_max` — i.e. for whom the fall is free-decay "
        f"reachable. **PASTE THE p50 AND THIS COLUMN INTO THE POLICY'S PROVENANCE BLOCK.**"
    ))
    mo.output.append(tau_feas)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 — Reachable ceiling

    The parameter no guard ever checked. For each arm's `high`, what fraction of
    cells can actually get there?

    Estimated as the per-cell maximum CNR observed under sustained top-of-ladder
    stimulation. It is a **lower bound** on the true ceiling — a cell that was never
    driven hard for long enough looks lower than it is — so read a low fraction as
    "not demonstrated", not as "impossible".

    For the ceiling-probe arm the reading is inverted: `high` there is *supposed* to
    be out of reach, and a materially non-zero fraction means the arm measures
    nothing and `high` must go up.
    """)
    return


@app.cell
def _(materials_path, mo):
    ceiling_input = mo.ui.text(
        value=str(materials_path("dataset_all.parquet")),
        label="Dataset parquet for the ceiling estimate",
        full_width=True,
    )
    hi_dose_input = mo.ui.number(value=100.0, step=5.0,
                                 label="'sustained' dose threshold (ms)")
    mo.hstack([ceiling_input, hi_dose_input])
    return ceiling_input, hi_dose_input


@app.cell
def _(OBJECTIVES, Path, ceiling_input, hi_dose_input, mo, np, pl, segments_of):
    _p = Path(ceiling_input.value.strip())
    mo.stop(not _p.exists(), mo.md(f"**Dataset not found:** `{_p}` — section 3 skipped."))

    _df = pl.read_parquet(_p)
    _cols = set(_df.columns)
    _cnr = next((c for c in ("cnr_median", "cnr", "raw_cnr") if c in _cols), None)
    _key = next((c for c in ("track_key", "particle", "cell_id") if c in _cols), None)
    _dose = next((c for c in ("exposure_ms", "u_t", "fluence_mJ_cm2") if c in _cols), None)
    mo.stop(
        _cnr is None or _key is None or _dose is None,
        mo.md(
            f"**Cannot find the needed columns** in `{_p.name}` "
            f"(cnr={_cnr}, key={_key}, dose={_dose}). Columns: `{sorted(_cols)}`"
        ),
    )

    # Per-cell ceiling: the highest CNR the cell reached while being driven hard.
    _driven = _df.filter(pl.col(_dose) >= hi_dose_input.value)
    CEILINGS = (
        _driven.group_by(_key).agg(pl.col(_cnr).max().alias("ceiling"))
        ["ceiling"].drop_nulls().to_numpy()
    )
    mo.stop(
        len(CEILINGS) < 20,
        mo.md(f"Only {len(CEILINGS)} cells driven at ≥ {hi_dose_input.value} ms — too "
              f"few for a ceiling distribution. Lower the threshold or use another dataset."),
    )

    _highs = {}
    for _arm, _obj in sorted(OBJECTIVES.items()):
        _segs = segments_of(_obj.reference)
        if _segs:
            for _s in _segs:
                _highs[(_arm, _s["segment"])] = _s["high"]
        else:
            _t = getattr(_obj.reference, "target_cnr", None)
            if _t is not None:
                _highs[(_arm, "target")] = _t

    ceiling_tbl = pl.DataFrame([
        {"arm": a, "segment": s, "high": round(h, 3),
         "frac_cells_reaching": round(float((CEILINGS >= h).mean()), 3)}
        for (a, s), h in sorted(_highs.items())
    ])

    mo.output.append(mo.md(
        f"{len(CEILINGS)} cells driven at ≥ {hi_dose_input.value} ms. Ceiling "
        f"percentiles: p10 **{np.percentile(CEILINGS, 10):.2f}** · "
        f"p50 **{np.percentile(CEILINGS, 50):.2f}** · "
        f"p90 **{np.percentile(CEILINGS, 90):.2f}** · "
        f"max **{CEILINGS.max():.2f}**\n\n"
        f"**PASTE `frac_cells_reaching` INTO THE POLICY'S PROVENANCE BLOCK.**"
    ))
    mo.output.append(ceiling_tbl)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 — Closed-loop dry run

    `optoerk.serving.replay.simulate_closed_loop` drives the real service with the
    model standing in for the cells, against each arm's actual reference.

    **This is not a feasibility verdict and must not be read as one.** Its own
    docstring says it: the controller is scored against the model's beliefs, using
    the same model the controller plans with. A reference the model calls reachable
    is one the controller will confidently chase — possibly off a cliff. What it
    gives is the model's **prediction** of where the borders are, recorded before the
    run so the run can falsify it. That is the whole point of the ceiling and
    bandwidth arms.

    Needs a checkpoint, so this section is cluster-only.
    """)
    return


@app.cell
def _(mo):
    dry_run_button = mo.ui.run_button(label="Run the dry run (slow, needs a checkpoint)")
    n_sim_frames = mo.ui.number(value=300, step=50, label="frames to simulate")
    mo.hstack([dry_run_button, n_sim_frames])
    return dry_run_button, n_sim_frames


@app.cell
def _(ARM_FOV, POLICY, POLICY_PATH, dry_run_button, mo, n_sim_frames, pl):
    mo.stop(not dry_run_button.value, mo.md("_Not run._"))

    from optoerk.serving.config import ServerConfig
    from optoerk.serving.replay import simulate_closed_loop

    _rows = []
    for _arm, _fov in sorted(ARM_FOV.items()):
        _pf = POLICY.model_copy(deep=True)
        _pf.placeholders_resolved = True  # a dry run is exactly what the gate is not for
        _cfg = ServerConfig(
            policy_file=str(POLICY_PATH), warmup=False, gpu_sample_interval_s=0,
            dark_baseline=False,
        )
        _sim = simulate_closed_loop(_cfg, n_cells=32,
                                    n_frames=int(n_sim_frames.value), fov=_fov)
        # `r_t` is annotated per frame by the objective, so tracking error needs no
        # re-derivation of the waveform here.
        _cols = set(_sim.columns)
        _err = (
            (_sim["cnr"] - _sim["r_t"]).abs()
            if {"cnr", "r_t"} <= _cols else None
        )
        _rows.append({
            "arm": _arm, "fov": _fov,
            "n_frames": int(n_sim_frames.value),
            "mean_abs_err": None if _err is None else round(float(_err.mean()), 4),
            "p90_abs_err": None if _err is None else round(float(_err.quantile(0.9)), 4),
            "frac_at_ladder_top": (
                round(float((_sim["exposure_ms"] >= _sim["exposure_ms"].max()).mean()), 3)
                if "exposure_ms" in _cols else None
            ),
        })

    dry_run = pl.DataFrame(_rows)
    mo.output.append(mo.md(
        "**PASTE THIS INTO THE POLICY'S PRE-FLIGHT BLOCK** — it is the prediction the "
        "run tests, not a verdict on the run."
    ))
    mo.output.append(dry_run)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Before flipping the gate

    `placeholders_resolved = true` is the operator's call, not this notebook's. What
    it should mean by the time it is flipped:

    1. §1's `tau_max` column has been read against §2's measured τ, and every segment
       with headroom below ~1.3x is either **intended** (a border arm) or fixed.
    2. §3's `frac_cells_reaching` is pasted in, and each arm's `high` is where the
       design wants it — comfortably reachable, or comfortably not.
    3. §4 has run and its numbers are recorded **as a prediction**.
    4. If τ moved, any amplitude that was solved from τ has been recomputed. In
       `policy_10fov_patterns.toml` that is arm 1's entire block table: every block's
       amplitude is the largest its own fall can deliver, so a larger τ shrinks all
       four.

    None of this makes an unreachable reference an error. It makes it a *choice on
    the record*, which is what the deleted guards were reaching for and could not
    express.
    """)
    return


if __name__ == "__main__":
    app.run()
