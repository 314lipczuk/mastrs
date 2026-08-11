import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import tempfile
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from optoerk.serving.config import ServerConfig
    from optoerk.serving.replay import (
        iter_predict_records,
        replay_counterfactual,
        replay_faithful,
        startup_record,
    )

    return (
        Path,
        ServerConfig,
        alt,
        iter_predict_records,
        json,
        mo,
        pl,
        replay_counterfactual,
        replay_faithful,
        startup_record,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md("""
    # Replay a serving run

    Re-drives `InferenceService` in-process with a finished run's frames — no faro,
    no HTTP. Three modes, and **they answer different questions**:

    | mode | question it answers | question it does NOT |
    |---|---|---|
    | **faithful** | did a refactor change any decision? | — |
    | **counterfactual** | how often would policy B have disagreed? | would B have tracked better |
    | **closed-loop sim** | how does B track *the model's* dynamics? | how B behaves on real cells |

    Counterfactual replay is **open-loop**: the recorded CNRs are the cells' response
    to the *original* doses, so past the first divergent frame the input stream is a
    fiction. It measures disagreement, not control quality.

    The last section correlates per-FOV request cadence against the acquisition
    clock — that, not inference cost, is where dropped frames come from.
    """)
    return


@app.cell
def _(mo):
    run_dir = mo.ui.text(
        value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-07-16_InferenceCNRhold_12h_v5",
        label="Run directory (a `*.jsonl` server log + a `tracks/` dir)",
        full_width=True,
    )
    checkpoint = mo.ui.text(
        value="/Volumes/imaging.data/mic01-imaging/314lipczuk/seq2scal_history_optortk_multilen_2026-07-14_09.48.21",
        label="Checkpoint bundle",
        full_width=True,
    )
    mo.vstack([run_dir, checkpoint])
    return checkpoint, run_dir


@app.cell
def _(mo, run_dir):
    n_frames = mo.ui.slider(20, 800, value=200, step=20,
                            label="Frames to replay (all FOVs pooled)")
    target_cnr = mo.ui.number(value=1.40, label="target_cnr")
    horizon = mo.ui.slider(1, 10, value=5, label="control_horizon")
    mo.vstack([mo.md(f"**Run:** `{run_dir.value}`"),
               mo.hstack([n_frames, target_cnr, horizon])])
    return horizon, n_frames, target_cnr


@app.cell
def _(Path, json, mo, run_dir, startup_record):
    _logs = sorted(Path(run_dir.value.strip()).glob("*.jsonl"))
    mo.stop(not _logs, mo.md("**No `*.jsonl` log found in that directory.**"))
    log_path = _logs[0]
    _start = startup_record(log_path)
    mo.md(f"""
    **Log:** `{log_path.name}` ({log_path.stat().st_size / 1e6:.0f} MB)

    **Startup record:**
    ```json
    {json.dumps(_start.get("info", {}) if _start else {}, indent=2)}
    ```

    Runs recorded before per-FOV policies landed do **not** carry the objective, so
    `target_cnr` and `control_horizon` have to be recovered by sweeping them until
    the replay reproduces the run (the v5 run turns out to be `target_cnr=1.40`,
    `control_horizon=5`). Newer runs log every resolved policy under `policies`, so
    this is a one-time archaeology cost.
    """)
    return (log_path,)


@app.cell
def _(ServerConfig, checkpoint, horizon, target_cnr):
    def make_cfg(**overrides):
        base = dict(
            checkpoint_dir=checkpoint.value.strip(),
            device="cpu",
            warmup=False,
            target_cnr=target_cnr.value,
            control_horizon=horizon.value,
            n_candidates=5,
            stim_power_pct=10.0,
            gpu_sample_interval_s=0,
        )
        base.update(overrides)
        return ServerConfig(**base)

    return (make_cfg,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Faithful replay — does this config reproduce the run?
    """)
    return


@app.cell
def _(Path, json, log_path, make_cfg, mo, n_frames, replay_faithful, run_dir):
    faithful_df, faithful_summary = replay_faithful(
        log_path, make_cfg(),
        tracks_dir=Path(run_dir.value.strip()) / "tracks",
        limit=n_frames.value,
    )
    mo.md(f"""
    ```json
    {json.dumps(faithful_summary, indent=2)}
    ```

    `crowding_match_frac` must be **1.0** — it proves the track positions were joined
    back correctly. If it drops, the replayed model inputs differ from the live ones
    and every number below is meaningless.

    `exposure_match_frac` will not be exactly 1.0 when replaying a **CUDA** run on
    **CPU**: float differences flip the occasional argmin near a decision boundary,
    and the divergence compounds slowly through the per-cell encoder state. Measured
    on the v5 run: 0.9979 over 76k decisions. A same-device replay is exact.
    """)
    return (faithful_df,)


@app.cell
def _(alt, faithful_df, mo, pl):
    _by_t = (
        faithful_df
        .with_columns((pl.col("exposure_replayed") == pl.col("exposure_recorded")).alias("ok"))
        .group_by("timestep")
        .agg(pl.col("ok").mean().alias("match_frac"), pl.len().alias("n"))
        .sort("timestep")
    )
    _chart = (
        alt.Chart(_by_t)
        .mark_line(point=False)
        .encode(
            x=alt.X("timestep:Q", title="timestep (min)"),
            y=alt.Y("match_frac:Q", title="fraction of cells matching",
                    scale=alt.Scale(zero=False)),
            tooltip=["timestep", "match_frac", "n"],
        )
        .properties(width=700, height=220, title="Faithful replay agreement over time")
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Counterfactual — how much would a different goal have changed?

    Open-loop. Read this as *disagreement*, never as *improvement*.
    """)
    return


@app.cell
def _(mo):
    cf_target = mo.ui.number(value=2.0, label="counterfactual target_cnr")
    cf_controller = mo.ui.dropdown(
        options=["constant_dose", "sequence_mpc"], value="sequence_mpc",
        label="counterfactual controller",
    )
    mo.hstack([cf_target, cf_controller])
    return cf_controller, cf_target


@app.cell
def _(
    Path,
    cf_controller,
    cf_target,
    json,
    log_path,
    make_cfg,
    mo,
    n_frames,
    replay_counterfactual,
    run_dir,
    tempfile,
):
    _policy = {
        "default": {
            "objective": {"type": "hold", "target_cnr": cf_target.value},
            "controller": {"type": cf_controller.value},
        }
    }
    _pf = Path(tempfile.mkdtemp()) / "policy.json"
    _pf.write_text(json.dumps(_policy))

    cf_df, cf_summary = replay_counterfactual(
        log_path, make_cfg(policy_file=str(_pf)),
        tracks_dir=Path(run_dir.value.strip()) / "tracks",
        limit=n_frames.value,
    )
    mo.md(f"""
    ```json
    {json.dumps(cf_summary, indent=2)}
    ```
    """)
    return (cf_df,)


@app.cell
def _(alt, cf_df, mo, pl):
    _cmp = (
        cf_df.group_by("timestep")
        .agg(
            pl.col("exposure_recorded").mean().alias("recorded"),
            pl.col("exposure_replayed").mean().alias("counterfactual"),
        )
        .sort("timestep")
        .unpivot(index="timestep", variable_name="policy", value_name="mean_exposure_ms")
    )
    _chart2 = (
        alt.Chart(_cmp)
        .mark_line()
        .encode(
            x=alt.X("timestep:Q", title="timestep (min)"),
            y=alt.Y("mean_exposure_ms:Q", title="mean commanded exposure (ms)"),
            color=alt.Color("policy:N"),
            tooltip=["timestep", "policy", "mean_exposure_ms"],
        )
        .properties(width=700, height=240, title="Commanded dose: recorded vs counterfactual")
    )
    mo.ui.altair_chart(_chart2)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Where does the latency actually go?

    The v5 run answered this: the server spends **~0.1 s of a 60 s frame budget**
    (p99 `infer_s` 0.12 s, `lock_wait_s` 0.00 s). Inference is not why frames drop,
    so JIT/quantisation would optimise a component that is idle 99.8% of the time.

    What *does* matter is the **request cadence**: gaps in `recv_epoch` are the
    upstream acquisition clock. A gap ≫ 60 s with a flat `handler_s` means the
    stall is upstream of the server — faro acquisition, segmentation/tracking, or
    the network — not in the model.
    """)
    return


@app.cell
def _(iter_predict_records, log_path, pl):
    _rows = []
    for _rec in iter_predict_records(log_path):
        _t = _rec.get("timing") or {}
        _rows.append({
            "fov": _rec["fov"], "timestep": _rec["timestep"],
            "recv_epoch": _t.get("recv_epoch"),
            "lock_wait_s": _t.get("lock_wait_s"), "infer_s": _t.get("infer_s"),
            "handler_s": _t.get("handler_s"), "n_scored": _rec.get("n_scored"),
        })
    timing = pl.DataFrame(_rows)
    has_timing = timing["recv_epoch"].null_count() < timing.height
    timing.head()
    return has_timing, timing


@app.cell
def _(has_timing, mo, pl, timing):
    mo.stop(
        not has_timing,
        mo.md("**This run predates the `timing` instrumentation — no latency block to plot.**"),
    )
    cadence = (
        timing.sort("fov", "recv_epoch")
        .with_columns(
            (pl.col("recv_epoch").diff().over("fov")).alias("gap_s")
        )
        .drop_nulls("gap_s")
    )
    mo.md(f"""
    | metric | p50 | p99 | max |
    |---|---|---|---|
    | `infer_s` | {timing["infer_s"].quantile(0.5):.3f} | {timing["infer_s"].quantile(0.99):.3f} | {timing["infer_s"].max():.3f} |
    | `lock_wait_s` | {timing["lock_wait_s"].quantile(0.5):.3f} | {timing["lock_wait_s"].quantile(0.99):.3f} | {timing["lock_wait_s"].max():.3f} |
    | `handler_s` | {timing["handler_s"].quantile(0.5):.3f} | {timing["handler_s"].quantile(0.99):.3f} | {timing["handler_s"].max():.3f} |
    | upstream `gap_s` | {cadence["gap_s"].quantile(0.5):.1f} | {cadence["gap_s"].quantile(0.99):.1f} | {cadence["gap_s"].max():.1f} |

    Server time is `handler_s`; everything else in a `gap_s` is upstream.
    """)
    return (cadence,)


@app.cell
def _(alt, cadence, mo):
    _chart3 = (
        alt.Chart(cadence)
        .mark_point(opacity=0.4, size=12)
        .encode(
            x=alt.X("timestep:Q", title="timestep"),
            y=alt.Y("gap_s:Q", title="gap since this FOV's previous request (s)"),
            color=alt.Color("fov:N"),
            tooltip=["fov", "timestep", "gap_s", "handler_s"],
        )
        .properties(width=700, height=260,
                    title="Upstream request cadence — flat 60 s means faro is keeping up")
    )
    mo.ui.altair_chart(_chart3)
    return


if __name__ == "__main__":
    app.run()
