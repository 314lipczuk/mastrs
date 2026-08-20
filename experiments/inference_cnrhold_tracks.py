import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import functools
    import json
    import re
    import tomllib
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import zarr
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    from scipy.stats import gaussian_kde

    # Every plot renders with matplotlib (a static raster), so the notebook output
    # size is independent of the number of tracks/rows. Pooling every FOV of an
    # arbitrarily large experiment can never overflow marimo's output cap the way
    # inline-Vega (altair, which embeds every row as JSON) did.
    #
    # The dose ladder is NOT a constant here — it is an experiment parameter set
    # per policy (`levels_ms`), and it has been rebinned across runs (0–800 ms in
    # the early holds, [0, 20, 45, 85, 150] in the oscillation runs). It is read
    # from the policy in the DOSE LADDER cell below; only the colormap is fixed.
    EXPO_CMAP = plt.cm.YlOrRd
    # One colour per phase group, for the oscillating references (cells are offset
    # by particle % n_phase_groups, so a pooled mean smears the waveform flat and
    # each group has to be drawn against its own reference).
    PHASE_CMAP = plt.cm.viridis
    return (
        EXPO_CMAP,
        FuncAnimation,
        Normalize,
        PHASE_CMAP,
        Path,
        alt,
        asyncio,
        functools,
        gaussian_kde,
        json,
        mo,
        np,
        pl,
        plt,
        re,
        tomllib,
        zarr,
    )


@app.cell
def _(mo):
    mo.md("""
    # Live experiment tracks — serving-run viewer

    Point this at any serving-run **directory**: per-FOV track parquets under
    `tracks/` plus a `.jsonl` server log (any name). Each row is one cell at one
    timestep; `particle` is the per-FOV track id, `timestep` is minutes (dt = 1 min).

    The log holds three event kinds: `startup` (engine + checkpoint), `predict`
    (one per FOV/timestep, with a `timing` block and the scored cells), and `gpu`
    (a periodic device sample). GPU load is plotted at the end.

    The parquet `stim_exposure` column was mangled by faro, so the true per-cell
    stim exposure is pulled from the server log and joined on
    `(fov, timestep, particle)`. Exposure takes a handful of discrete levels — the
    **dose ladder**, an experiment parameter (`levels_ms`) read from the policy, not
    a constant (it was rebinned from `0…800 ms` to `[0, 20, 45, 85, 150]`).

    **The setpoint is per-FOV and may be time-varying.** A `hold` objective has one
    `target_cnr`; an `oscillation` objective tracks a step train whose value depends
    on the frame *and* on the cell's phase offset. Every plot below draws the
    reference the server actually logged per cell (`r_t`), so nothing here assumes a
    single flat target.

    FOVs are **pooled by default**, or grouped by any set of policy facets you pick
    (controller / kernel / λ_move / reference / checkpoint …), or viewed one at a
    time; a `track_key` = `fov_particle` keeps track identities unique across FOVs.
    Plots are static matplotlib images, so pooling any-size experiment never
    overflows the output.
    """)
    return


@app.cell
def _(mo):
    # Experiment directory to load. Any serving run with per-FOV `*.parquet` track
    # files and a `*.jsonl` prediction log (found recursively) works.
    exp_dir_input = mo.ui.text( 
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-07-16_InferenceCNRhold_12h_v7",
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-07-30_InferenceCNRhold_12h_v10",
        value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-07-30_InferenceCNRhold_12h_v11",
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-08-07_InferenceCNRhold_12h_v12",
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-08-07_InferenceCNRhold_12h_v13_2",
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-08-07_InferenceCNRhold_12h_v15", 
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-08-07_InferenceCNRhold_12h_v16",
        #value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-08-19_InferenceCNRhold_12h_v17", 
        label="Experiment directory (per-FOV parquets + a .jsonl log)", 
        full_width=True,
    )
    exp_dir_input

    # \\izbkingston.unibe.ch\imaging.data\mic01-imaging\314lipczuk\2026-07-16_InferenceCNRhold_12h_v7
    return (exp_dir_input,)


@app.cell
def _(Path, exp_dir_input, pl):

    _exp_dir = Path(exp_dir_input.value.strip())
    # Only `tracks/jjj` — the run dir also holds a top-level `exp_data.parquet` that
    # is the same rows concatenated, so a recursive glob would double-count them.
    _files = sorted((_exp_dir / "tracks").glob("*.parquet"))
    #_files
    t1, t2 = [pl.read_parquet(_f) for _f in [_files[0], _files[1]]]
    t1
    return


@app.cell
def _():
    return


@app.cell
def _(Path, exp_dir_input, json, mo, pl, re):
    _exp_dir = Path(exp_dir_input.value.strip())
    # Only `tracks/` — the run dir also holds a top-level `exp_data.parquet` that
    # is the same rows concatenated, so a recursive glob would double-count them.
    _files = sorted((_exp_dir / "tracks").glob("*.parquet"))
    _logs = sorted(_exp_dir.glob("*.jsonl"))
    mo.stop(
        not _files or not _logs,
        mo.md(
            f"**Not a serving directory:** `{_exp_dir}` needs `tracks/*.parquet` "
            "and a `*.jsonl` server log."
        ),
    )
    _log = _logs[0]

    # `tracks/` holds one file per (FOV, phase) — `<fov>_phase_<n>_latest.parquet` —
    # and faro writes each as a running snapshot of the tracker, spanning every
    # frame seen so far rather than only that phase's. So `0_phase_1_latest` already
    # contains all of `0_phase_0_latest`'s rows, under the same particle ids.
    #
    # Two consequences, both of which have to be handled here:
    #   * the optocheck phase never stimulates, so faro writes no `stim_exposure` /
    #     `stim_power` / `time_offset` columns for it — a plain `pl.concat` dies on
    #     the width mismatch (29 vs 32).
    #   * concatenating every file double-counts every frame of the earlier phases.
    # Read highest-phase-first, union the schemas (`diagonal` fills the columns a
    # phase lacks with null), and keep the first row per (fov, timestep, particle):
    # the widest, latest version of a frame wins, and a frame that only the earlier
    # file has still survives.
    def _phase_of(_f):
        _m = re.search(r"_phase_(\d+)", _f.stem)
        return int(_m.group(1)) if _m else -1


    _tracks = pl.concat(
        [
            pl.read_parquet(_f).with_columns(
                pl.lit(_f.stem).alias("source_file"),
            )
            for _f in sorted(_files, key=_phase_of, reverse=True)
        ],
        how="diagonal",
    ).unique(subset=["fov", "timestep", "particle"], keep="first", maintain_order=True)

    # Server log, one pass over three event kinds:
    #   `predict` — one per (fov, timestep), holding the scored cells and (new
    #               format only) a `timing` block for that inference call
    #   `gpu`     — a periodic device sample, absent in pre-v4 runs
    #   `startup` — engine/checkpoint info, logged once
    _srv_rows = []
    _timing_rows = []
    _gpu_rows = []
    startup = {}
    # Which physical GPU the `gpu` samples describe, and whether that was actually
    # verified against the model's device. Absent on runs that predate the record —
    # and those are exactly the runs where the two could silently disagree.
    gpu_device = {}
    with open(_log) as _fh:
        for _line in _fh:
            _rec = json.loads(_line)
            _ev = _rec.get("event")
            if _ev == "startup":
                startup = _rec
                continue
            if _ev == "gpu_device":
                gpu_device = _rec
                continue
            if _ev == "gpu":
                # `procs` is a long per-PID list we never plot; drop it and keep
                # the scalar device metrics.
                _gpu_rows.append({_k: _v for _k, _v in _rec.items() if _k not in ("procs", "event")})
                continue
            if _ev != "predict":
                continue
            _tm = _rec.get("timing") or {}
            _timing_rows.append(
                {
                    "t": _rec["t"],
                    "fov": _rec["fov"],
                    "timestep": _rec["timestep"],
                    "n_scored": _rec.get("n_scored"),
                    "infer_s": _tm.get("infer_s"),
                    "handler_s": _tm.get("handler_s"),
                    "lock_wait_s": _tm.get("lock_wait_s"),
                    "cuda_alloc_mb": _tm.get("cuda_alloc_mb"),
                    "cuda_reserved_mb": _tm.get("cuda_reserved_mb"),
                }
            )
            # Explode the cells to one row per (fov, timestep, particle) and keep
            # the true stim exposure the model commanded. Duplicate keys are
            # byte-identical repeats, so keeping the first is safe.
            for _c in _rec["cells"]:
                _srv_rows.append(
                    {
                        "fov": _rec["fov"],
                        "timestep": _rec["timestep"],
                        "particle": _c["particle"],
                        "exposure_ms": _c["exposure_ms"],
                        "fluence_out": _c["fluence_out"],
                        # cnr_norm = the normalized signal the controller targets
                        # (= raw_cnr / server baseline); not in the parquet.
                        "cnr_norm": _c.get("cnr_norm"),
                        "baseline": _c.get("baseline"),
                        # The reference the controller was actually tracking for
                        # THIS cell at THIS frame, as the server recorded it
                        # (Objective.annotate). For a `hold` it is the constant
                        # target; for an `oscillation` it is the step train
                        # evaluated at the cell's own phase offset — which is why
                        # it is a per-row column and not a notebook-side scalar.
                        # Absent on runs that predate the annotation; the
                        # REFERENCE cell falls back to the policy's target then.
                        "r_t": _c.get("r_t"),
                        # Waveform segment (settle / low_hold / rise / high_hold /
                        # fall). `settle` marks the frames where the start-up
                        # transient and the tracking response overlap and which
                        # the analysis must drop.
                        "segment": _c.get("segment"),
                        "phase_offset_min": _c.get("phase_offset_min"),
                        # What the controller BELIEVED would happen: the winning plan's
                        # cost, and the predicted CNR one step ahead under the dose it
                        # actually commanded. Without them a saturated cell and a
                        # mispredicted cell are indistinguishable — both just show a
                        # dose and a CNR. `pred_cnr_h1` is also the only per-frame
                        # measurement of model error under a known input.
                        "plan_cost": _c.get("plan_cost"),
                        "pred_cnr_h1": _c.get("pred_cnr_h1"),
                    }
                )

    # Wall-clock `t` is epoch seconds; `predict` events carry both `t` and the
    # experiment `timestep`, so they define the map from epoch to experiment
    # hours that the `gpu` samples (which only have `t`) are placed on.
    timing = pl.DataFrame(_timing_rows).sort("t").with_columns(
        (pl.col("timestep") / 60.0).alias("hours")
    )
    gpu = pl.DataFrame(_gpu_rows, schema_overrides={"t": pl.Float64}).sort("t") if _gpu_rows else None

    serving = (
        # SCHEMA IS DECLARED, NOT INFERRED. polars types a list of dicts from its
        # first 100 rows, and every optional field here is absent for whole FOVs at a
        # time rather than at random: a `hold` objective annotates no `segment` and no
        # `phase_offset_min`, a run that predates an annotation has none of them
        # anywhere. Whenever the first FOVs served are the ones WITHOUT a field, the
        # column types as Null and the first real value — "settle" arriving from a
        # frequency_staircase FOV a few hundred rows later — kills the whole load with
        # a builder-append error. That is not a corrupt log; it is a run whose FOVs
        # carry different objectives, which is now the normal case.
        #
        # Declaring the types costs nothing and removes the ordering dependence
        # entirely. `infer_schema_length=None` would also work but scans every row of
        # a multi-million-row list to learn what is written here in seven lines.
        pl.DataFrame(
            _srv_rows,
            schema_overrides={
                "cnr_norm": pl.Float64,
                "baseline": pl.Float64,
                "r_t": pl.Float64,
                "segment": pl.Utf8,
                "phase_offset_min": pl.Float64,
                "plan_cost": pl.Float64,
                "pred_cnr_h1": pl.Float64,
            },
        )
        .unique(subset=["fov", "timestep", "particle"], keep="first")
        .with_columns(
            pl.col("fov").cast(pl.UInt16),
            pl.col("timestep").cast(pl.UInt32),
            pl.col("particle").cast(pl.UInt32),
            # On a run that predates the reference annotation these are all-null,
            # which polars infers as the Null dtype — cast so downstream
            # arithmetic and joins see a normal (empty) float/str column instead
            # of failing on a type it cannot subtract.
            pl.col("r_t").cast(pl.Float64),
            pl.col("phase_offset_min").cast(pl.Float64),
            pl.col("segment").cast(pl.Utf8),
            pl.col("plan_cost").cast(pl.Float64),
            pl.col("pred_cnr_h1").cast(pl.Float64),
        )
    )

    _ta = pl.col("time_acquired").str.strptime(pl.Datetime, "%Y-%m-%d-%H:%M:%S")

    data_all = (
        _tracks.join(serving, on=["fov", "timestep", "particle"], how="left")
        .with_columns(
            # `hours` IS THE REAL CLOCK, read off `time_acquired` — the moment the
            # frame actually landed — relative to the first frame of the run.
            #
            # It used to be `time / 3600`, and `time` is the *planned* schedule the
            # acquisition was programmed with, i.e. `timestep / 60` under a nominal
            # 1-minute cadence. Whenever the loop cannot hold that cadence the two
            # diverge without warning, and every axis silently reads as the
            # experiment that was intended rather than the one that ran. Keep the
            # plan under its own name and check them against each other (the
            # `jxvo` panel) before reading any rate off a plot.
            #
            # Per-row, not per-timestep: within one timestep the FOVs are imaged
            # sequentially and can be minutes apart, so each FOV keeps its own
            # acquisition time.
            ((_ta - _ta.min()).dt.total_seconds() / 3600.0).alias("hours"),
            (pl.col("time") / 3600.0).alias("hours_planned"),
            # True light-on flag from the server exposure, not the mangled parquet
            # `stim` column.
            (pl.col("exposure_ms") > 0).alias("light_on"),
            # Track id unique across FOVs, so pooling doesn't merge same-id particles.
            (pl.col("fov").cast(pl.Utf8) + "_" + pl.col("particle").cast(pl.Utf8)).alias("track_key"),
        )
    )

    # Put the server's clock on the acquisition clock. `predict` events carry epoch
    # `t` and the experiment `timestep`; the tracks carry `time_acquired` for that
    # same timestep. Anchoring on the first shared timestep maps one to the other
    # exactly, without assuming anything about the machine's timezone (the parquet
    # stamps are naive local strings, the log is epoch seconds). Without this the
    # GPU and latency panels sit on a different, nominal axis from the cell plots.
    _anchor_ts = int(timing["timestep"].min())
    _anchor_epoch = float(timing.filter(pl.col("timestep") == _anchor_ts)["t"].min())
    _anchor_hours = float(
        data_all.filter(pl.col("timestep") == _anchor_ts)["hours"].min()
    )
    timing = timing.with_columns(
        ((pl.col("t") - _anchor_epoch) / 3600.0 + _anchor_hours).alias("hours")
    )
    return data_all, gpu, gpu_device, startup, timing


@app.cell
def _(data_all, mo, pl):
    # An experiment runs as several phases back to back — typically an optocheck
    # under a fixed light pattern, then the controlled run — and they do not share a
    # cadence or a treatment, so pooling them draws two unrelated time bases on one
    # axis. Default to the last phase, which is the controlled experiment.
    _phases = (
        data_all.group_by("phase_id")
        .agg(pl.col("phase_name").first(), pl.col("treatment_name").first(), pl.len())
        .sort("phase_id")
    )
    phase_sel = mo.ui.dropdown(
        options={
            **{
                f"{_r['phase_id']} — {_r['phase_name']} ({_r['treatment_name']})": _r["phase_id"]
                for _r in _phases.to_dicts()
            },
            "all phases (mixed cadence)": -1,
        },
        value=(
            f"{_phases['phase_id'][-1]} — {_phases['phase_name'][-1]} "
            f"({_phases['treatment_name'][-1]})"
        ),
        label="Phase",
        full_width=True,
    )
    mo.vstack([phase_sel, _phases])
    return (phase_sel,)


@app.cell
def _(data_all, phase_sel, pl):
    # Everything below reads `data`, so the phase cut happens once, here.
    data = (
        data_all
        if phase_sel.value == -1
        else data_all.filter(pl.col("phase_id") == phase_sel.value)
    )
    return (data,)


@app.cell
def _(Path, exp_dir_input, mo):
    # Optional policy `.toml` (the file passed to `--policy-file`). The startup log
    # records the *resolved* per-FOV policy, but a run whose model loaded as a stub
    # logs no `controller` per FOV — so when the experiment varied the controller
    # across FOVs, the policy file is the only place that assignment survives.
    # Auto-detected next to the run, then the repo's example, else paste a path.
    _exp = Path(exp_dir_input.value.strip())
    _auto = sorted(_exp.glob("*.toml"))
    _default_path = str(_auto[0]) if _auto else (
        "policy_9fov_raw.toml" if Path("policy_9fov_raw.toml").exists() else ""
    )
    policy_input = mo.ui.text(
        value=_default_path,
        label="Policy file (.toml, optional — fills in per-FOV controller)",
        full_width=True,
    )
    policy_input
    return (policy_input,)


@app.cell
def _(Path, mo, policy_input, tomllib):
    # Resolve each `[fov.N]` against `[default]` (default keys, overridden by the
    # FOV's own), so every FOV carries its full policy (checkpoint, objective,
    # control_horizon, controller). `None` when no file is given.
    policy = None
    _p = policy_input.value.strip()
    if _p and Path(_p).exists():
        with open(_p, "rb") as _fh:
            _raw = tomllib.load(_fh)
        _default = _raw.get("default", {}) or {}
        policy = {
            "default": _default,
            "fov": {int(_k): {**_default, **_v} for _k, _v in (_raw.get("fov", {}) or {}).items()},
        }
    mo.md(
        f"Policy file: **{len(policy['fov'])} FOVs** from `{_p}`."
        if policy is not None
        else "_No policy file — per-FOV metadata comes from the server log only._"
    )
    return (policy,)


@app.cell
def _(Path, data_all, policy, startup):
    # Per-FOV policy metadata, normalized to ONE record shape from either source.
    # Prefers the policy `.toml` (authoritative for the controller, and the only
    # place the assignment survives when a FOV degraded to the stub); falls back to
    # the server startup record. Feeds the setpoint, the grouping facets, and the
    # per-FOV latency table.
    #
    # TWO SHAPES REACH THIS CELL AND THEY ARE NOT THE SAME:
    #
    #   policy .toml — `objective` carries the reference params INLINE, and the
    #     arm-varying pieces (`kernel`, `lambda_move`, `lambda_dose`, `levels_ms`)
    #     are TOP-LEVEL keys that compose onto it. That split is deliberate
    #     upstream: `objective` is replaced wholesale on override, so restating the
    #     waveform per FOV would let one typo desynchronize an arm's reference.
    #     See optoerk.serving.policy.PolicySpec.
    #   startup log — `objective` is the RESOLVED `Objective.describe()`:
    #     `{type, reference:{...}, kernel:{...}, regularizers:[{type, lambda_*}]}`,
    #     with the ladder under `levels_ms` (and again under the controller).
    #
    # Read the wrong one and the arm-varying fields silently read as absent — which
    # is exactly how four arms collapse into one group. Everything downstream reads
    # the normalized record; neither raw shape leaks past this cell.
    #
    #   The setpoint is in the checkpoint's cnr units:
    #     cnr_mode="norm" -> cnr_median_norm (resting baseline == 1.0), e.g. 1.4
    #     cnr_mode="raw"  -> absolute cnr_median, e.g. 1.2
    #   The server logs the controlled signal (whichever it is) as `cnr_norm`, so
    #   the reference always lives on the `cnr_norm` readout. For a raw model
    #   `cnr_norm == cnr_median`, so it is meaningful on that readout too
    #   (TARGET_COLS in the REFERENCE cell).
    #
    # A `hold` objective has a single `target_cnr`. An `oscillation` does NOT —
    # its setpoint moves with the frame and with each cell's phase offset, so no
    # scalar target exists to fall back on and none is invented here.

    # The policy file names the model `checkpoint`; the server log `checkpoint_dir`.
    def _ckpt(_info):
        return Path(str((_info or {}).get("checkpoint_dir") or (_info or {}).get("checkpoint") or "?")).name

    def _obj(_info):
        return (_info or {}).get("objective") or {}

    def _ref_spec(_info):
        # startup: objective.reference; toml: the objective spec itself.
        _o = _obj(_info)
        return _o.get("reference") or _o

    def _period(_r):
        _p = _r.get("period_min")
        if _p is not None:
            return float(_p)
        return sum(
            float(_r.get(_k) or 0.0)
            for _k in ("t_low_min", "t_rise_min", "t_high_min", "t_fall_min")
        )

    def _const_target(_info):
        # The one constant setpoint, when the objective has one (`hold`, `gated`).
        # None for an oscillation / schedule — deliberately, so a caller cannot
        # draw a flat line through a moving reference.
        _t = _ref_spec(_info).get("target_cnr")
        return float(_t) if _t is not None else None

    def _ref_label(_info):
        _r = _ref_spec(_info)
        _type = _obj(_info).get("type") or _r.get("type") or "?"
        _t = _const_target(_info)
        if _t is not None:
            return f"{_type} {_t:g}"
        if _r.get("low") is not None and _r.get("high") is not None:
            return f"osc {float(_r['low']):g}→{float(_r['high']):g} P={_period(_r):g}m"
        if _r.get("points"):
            return f"schedule({len(_r['points'])} pts)"
        return str(_type)

    def _mode_of(_info):
        # cnr units. Explicit `cnr_mode` when the record carries one — the startup
        # log always does, the policy .toml never does. That asymmetry is not an
        # oversight upstream: cnr_mode is a property of the CHECKPOINT, resolved by
        # the server when it loads the weights, not something a policy file sets.
        #
        # So when the .toml is the source, fall through to the startup record for the
        # SAME checkpoint before guessing from the name. The name heuristic is a last
        # resort and v13 is exactly where it fails: `enc_e_area_lean_2026-08-07_...`
        # IS a raw model (the server logged cnr_mode="raw", and cnr_norm == cnr_median
        # to the bit in this run) but its name never says so, so the heuristic returns
        # "norm". That drops `cnr_median` out of TARGET_COLS, which silently stops the
        # reference being drawn on the default readout and makes the phase-aligned
        # panel refuse to render — a mislabelled unit presenting as a missing plot.
        _m = (_info or {}).get("cnr_mode")
        if _m in ("norm", "raw"):
            return _m
        _srv = (startup or {}).get("policies", {}) or {}
        for _cand in [_srv.get("default") or {}, *((_srv.get("fov") or {}).values())]:
            if _ckpt(_cand) == _ckpt(_info) and _cand.get("cnr_mode") in ("norm", "raw"):
                return _cand["cnr_mode"]
        return "raw" if "raw" in _ckpt(_info).lower() else "norm"

    def _kernel_label(_info):
        # toml: top-level `kernel`. startup: the resolved objective's kernel.
        # Absent in both -> the objective builders' default, which is l2.
        _k = (_info or {}).get("kernel") or _obj(_info).get("kernel") or "l2"
        if isinstance(_k, str):
            return _k
        _k = dict(_k)
        _kind = _k.pop("type", "?")
        _extra = ",".join(
            f"{_a}={_b:g}" if isinstance(_b, (int, float)) else f"{_a}={_b}"
            for _a, _b in sorted(_k.items())
        )
        return f"{_kind}({_extra})" if _extra else str(_kind)

    def _lam(_info, _which):
        # toml: top-level `lambda_move` / `lambda_dose`. startup: the regularizer
        # is simply absent when its coefficient is zero (they are dropped at
        # construction so lambda=0 reproduces the pre-refactor cost exactly), so
        # "no regularizer" means 0.0 rather than unknown.
        _v = (_info or {}).get(f"lambda_{_which}")
        if _v is None:
            for _r in _obj(_info).get("regularizers") or []:
                if _r.get("type") == f"{_which}_penalty":
                    _v = _r.get(f"lambda_{_which}")
        return float(_v or 0.0)

    def _ladder(_info):
        _lv = (_info or {}).get("levels_ms") or ((_info or {}).get("controller") or {}).get("levels_ms")
        return tuple(float(_x) for _x in _lv) if _lv else None

    if policy is not None:
        _fov_info = policy["fov"]
        _default_info = policy["default"]
    else:
        _pol = (startup or {}).get("policies", {}) or {}
        _default_info = _pol.get("default") or (startup or {}).get("info") or {}
        _fov_info = {int(_k): _v for _k, _v in (_pol.get("fov", {}) or {}).items()}

    # Controller label. The startup log's `controller.describe()` carries the whole
    # CEM configuration (n_samples, n_iters, seed, …) and the ladder, so joining
    # every key gives an unreadable label that is identical across FOVs anyway.
    # Include only the keys that actually DIFFER between FOVs in this run — that
    # keeps `staggered_mpc(k=3)` distinguishable from `staggered_mpc(k=5)` without
    # inventing a hardcoded list of "interesting" parameters that a future
    # controller would fall outside of.
    _ctrls = [dict((_v or {}).get("controller") or {}) for _v in _fov_info.values()]
    _varying = {
        _k
        for _k in {_k for _c in _ctrls for _k in _c}
        if _k not in ("name", "type", "levels_ms")
        and len({repr(_c.get(_k)) for _c in _ctrls}) > 1
    }

    def _ctrl_label(_info):
        _c = (_info or {}).get("controller") or {}
        _name = _c.get("name") or _c.get("type")
        if not _name:
            return "?"
        _extra = ",".join(f"{_k}={_c[_k]}" for _k in sorted(_varying) if _k in _c)
        return f"{_name}({_extra})" if _extra else str(_name)

    def _meta(_info):
        _lm, _ld = _lam(_info, "move"), _lam(_info, "dose")
        _ctrl, _kern = _ctrl_label(_info), _kernel_label(_info)
        _lad = _ladder(_info)
        return {
            "checkpoint": _ckpt(_info),
            "mode": _mode_of(_info),
            "target": _const_target(_info),
            "reference": _ref_label(_info),
            "controller": _ctrl,
            "kernel": _kern,
            "lambda_move": _lm,
            "lambda_dose": _ld,
            "horizon": (_info or {}).get("control_horizon"),
            "ladder": _lad,
            # Cycle geometry, needed to fold the run onto one cycle. Both shapes
            # carry it: the startup log resolves `period_min` / `settle_min`
            # directly, the .toml gives the four segment durations and
            # `settle_periods` to multiply out.
            "period_min": _period(_ref_spec(_info)) or None,
            "settle_min": (
                float(_ref_spec(_info)["settle_min"])
                if _ref_spec(_info).get("settle_min") is not None
                else (_period(_ref_spec(_info)) or 0.0)
                * float(_ref_spec(_info).get("settle_periods") or 0.0)
                or None
            ),
            # The arm: the pieces a controlled comparison is allowed to vary.
            # `arm_spec` is the ground truth; `arm` gets a number attached below.
            "arm_spec": f"{_ctrl} · {_kern} · λm={_lm:g}" + (f" · λd={_ld:g}" if _ld else ""),
            # The policy's OWN arm number, when it declares one. Both shapes carry
            # it: the .toml has `arm = N` in the [fov.N] block, and the startup log
            # echoes it under `requested`. Declared beats derived — see the ARMS
            # cell for why deriving cannot always work.
            "arm_declared": (
                (_info or {}).get("arm")
                if (_info or {}).get("arm") is not None
                else ((_info or {}).get("requested") or {}).get("arm")
            ),
        }

    FOV_META = {int(_k): _meta(_v) for _k, _v in _fov_info.items()}
    DEFAULT_META = _meta(_default_info)

    # --- FOVs THE POLICY NEVER MENTIONED ----------------------------------
    # The policy file and the rig can disagree about how many fields exist, and in
    # v13 they did: `policy_8fov_openloop.toml` declares fov 0-7, the microscope
    # imaged 10, so fov 8 and 9 were served by `[default]`. They are real controlled
    # cells with a real reference — they are simply not part of the design.
    #
    # Left out of FOV_META they do not vanish, which is the trap: every FOV list
    # below is built from the DATA, so they stay inside "All (pooled)" while being
    # absent from every arm group. A pooled number then quietly averages designed
    # fields with accidental ones and nothing on the plot says so.
    #
    # They get their own arm instead. Their config is arm 1's byte for byte, but
    # they are NOT arm 1: the layout is mirrored so that each arm's mean field index
    # is 3.5 and arm is not confounded with plate position (see the policy's FOV
    # LAYOUT note), and folding two edge fields into one arm breaks both that
    # balance and the n=2-per-arm symmetry the analysis constraint rests on.
    _unpolicied = sorted(set(data_all["fov"].unique().to_list()) - set(FOV_META))
    if _unpolicied:
        _declared_max = max(
            (
                int(_m["arm_declared"])
                for _m in FOV_META.values()
                if _m["arm_declared"] is not None
            ),
            default=0,
        )
        for _f in _unpolicied:
            _dm = dict(DEFAULT_META)
            # Marked, or it collides with the arm whose config it copies and the two
            # merge back into one group — the exact silent pooling this exists to stop.
            _dm["arm_spec"] = f"{_dm['arm_spec']} · [default policy]"
            # Numbered, never None: ARM_NUM only honours the policy's OWN declared
            # numbers when EVERY arm has one, so a single un-numbered arm would throw
            # the run back on derived numbering and silently renumber arms 1-4.
            _dm["arm_declared"] = _declared_max + 1
            FOV_META[int(_f)] = _dm

    # --- ARMS -------------------------------------------------------------
    # The arms are read OUT OF THE POLICY, never from the FOV index. The policy
    # files say so explicitly: `arm = (fov % 4) + 1` held for the 12-FOV file and
    # does NOT hold for the 10-FOV ones, where fov 0 is arm 3 — anything deriving
    # an arm from the index produces silently wrong labels rather than noisy ones.
    #
    # WHAT IDENTIFIES AN ARM DEPENDS ON THE EXPERIMENT, which is why the number is
    # taken from the policy's declared `arm` whenever it declares one.
    #
    #   The oscillation runs varied the CONTROLLER and held one reference across
    #   every FOV, so the arm was (controller, kernel, regularizers) and the
    #   reference was deliberately excluded from the identity.
    #
    #   The pattern-zoo run inverts that: one controller everywhere, and the
    #   WAVEFORM is the arm. Under the old rule all ten of its FOVs collapse into
    #   a single arm — the same class of silent mislabelling as `fov % 4 + 1`.
    #
    # So `arm_spec` picks up the reference whenever references differ across FOVs,
    # and drops it when they do not, keeping the oscillation runs' labels as they
    # were. The derived NUMBER (nesting order: constant_dose -> planning -> move
    # suppression -> distributional cost) is the fallback for files predating the
    # declaration; it reproduces arm 1-4 of the 10-FOV files but cannot order a set
    # of waveforms, which is the other reason to declare.
    _refs_vary = len({_m["reference"] for _m in FOV_META.values()}) > 1
    if _refs_vary:
        for _m in list(FOV_META.values()) + [DEFAULT_META]:
            _m["arm_spec"] = f"{_m['reference']} · {_m['arm_spec']}"

    _arm_key = lambda _m: (  # noqa: E731
        _m["controller"] != "constant_dose",
        _m["lambda_move"], _m["lambda_dose"],
        _m["kernel"] != "l2", _m["kernel"], _m["controller"],
    )
    _specs = {}
    for _m in FOV_META.values():
        _specs.setdefault(_m["arm_spec"], _m)

    _declared = {_m["arm_spec"]: _m["arm_declared"] for _m in FOV_META.values()}
    if _specs and all(_v is not None for _v in _declared.values()):
        ARM_NUM = {_spec: int(_declared[_spec]) for _spec in _specs}
    else:
        ARM_NUM = {
            _spec: _i + 1
            for _i, _spec in enumerate(sorted(_specs, key=lambda _s: _arm_key(_specs[_s])))
        }
    for _m in list(FOV_META.values()) + [DEFAULT_META]:
        _n = ARM_NUM.get(_m["arm_spec"])
        _m["arm"] = f"arm {_n} · {_m['arm_spec']}" if _n else _m["arm_spec"]
        _m["label"] = " · ".join(
            [_m["mode"], _m["reference"], _m["arm"], _m["checkpoint"][:24]]
        )

    # fov -> arm number, the map every per-arm analysis needs and the one the
    # policy files warn must be passed explicitly rather than recomputed.
    ARM_OF = {_f: ARM_NUM.get(_m["arm_spec"]) for _f, _m in FOV_META.items()}

    # Facets the FOV selector can group on, in rough order of "how likely you are
    # to slice by it". `arm` is the composite; the rest let you cut at any single
    # level (e.g. group by `kernel` alone to pool the two λ_move settings).
    FACET_KEYS = [
        "arm", "controller", "kernel", "lambda_move", "lambda_dose",
        "reference", "checkpoint", "mode", "horizon", "ladder",
    ]

    def facet_value(_m, _key):
        """One facet of a FOV's policy, as the string the group label shows."""
        _v = _m.get(_key)
        if _key == "ladder":
            return "?" if not _v else "[" + ",".join(f"{_x:g}" for _x in _v) + "]"
        if _key in ("lambda_move", "lambda_dose"):
            return f"λ{_key[7]}={float(_v or 0.0):g}"
        if _key == "horizon":
            return f"H={_v}" if _v is not None else "H=?"
        return "?" if _v is None else str(_v)

    {"n_fovs": len(FOV_META), "ARM_OF": ARM_OF}
    return ARM_OF, DEFAULT_META, FACET_KEYS, FOV_META, facet_value


@app.cell
def _(ARM_OF, FOV_META, mo, pl):
    # The resolved policy, one row per FOV. This is the table to check BEFORE
    # reading any plot: it is what actually ran, and it is where a target that
    # differs per FOV, or a ladder that was rebinned between runs, is visible.
    mo.stop(not FOV_META, mo.md("_No per-FOV policy — neither a policy file nor a startup record._"))
    _rows = [
        {
            "fov": _f,
            "arm": ARM_OF.get(_f),
            "arm_spec": _m["arm_spec"],
            "controller": _m["controller"],
            "kernel": _m["kernel"],
            "λ_move": _m["lambda_move"],
            "reference": _m["reference"],
            "cnr_mode": _m["mode"],
            "horizon": _m["horizon"],
            "ladder_ms": "?" if not _m["ladder"] else ",".join(f"{_x:g}" for _x in _m["ladder"]),
            "checkpoint": _m["checkpoint"],
        }
        for _f, _m in sorted(FOV_META.items())
    ]
    _by_arm = {}
    for _r in _rows:
        _by_arm.setdefault(_r["arm"], []).append(_r["fov"])
    mo.vstack([
        mo.md(
            f"**Resolved policy — {len(_rows)} FOVs, {len(_by_arm)} arms.** "
            "The FOV→arm map is read from the policy itself, never from the FOV "
            "index: `arm = (fov % 4) + 1` does **not** hold for the 10-FOV files "
            "(fov 0 is arm 3 there).\n\n"
            + " · ".join(
                f"**arm {_a}**: FOVs {','.join(map(str, _fl))}"
                for _a, _fl in sorted(_by_arm.items(), key=lambda _kv: (_kv[0] is None, _kv[0]))
            )
        ),
        mo.ui.table(pl.DataFrame(_rows), selection=None),
    ])
    return


@app.cell
def _(FOV_META, Normalize, data, mo):
    # --- DOSE LADDER ------------------------------------------------------
    # The set of exposures the controller could command, `levels_ms`. NOT a
    # constant: it is an experiment parameter, deliberately non-uniform (a ladder
    # like [0, 20, 45, 85, 150] cannot be written as a linspace), and it was
    # rebinned by ~5x between the early hold runs (0-800 ms) and the oscillation
    # runs. Hardcoding it draws an exposure stackplot of five empty bands and a
    # colour scale on which every real dose is the same shade of pale.
    #
    # Policy first (it is the definition of the run), observed values only as a
    # fallback for a log with no policy at all. The observed set is a strictly
    # worse source: a level the controller never happened to pick is simply
    # missing from it, so the ladder silently shrinks.
    _pol_levels = {_m["ladder"] for _m in FOV_META.values() if _m["ladder"]}
    _seen = sorted(
        float(_x) for _x in data["exposure_ms"].drop_nulls().unique().to_list()
    )
    if len(_pol_levels) == 1:
        EXPO_LEVELS = list(next(iter(_pol_levels)))
        _src = "policy"
    elif len(_pol_levels) > 1:
        # Arms searching different ladders are not comparable and the move penalty
        # normalizes by the ladder max, so this is a real finding, not a display
        # detail — surface it rather than silently picking one.
        EXPO_LEVELS = sorted({_x for _l in _pol_levels for _x in _l})
        _src = f"UNION of {len(_pol_levels)} different per-FOV ladders"
    else:
        EXPO_LEVELS = _seen
        _src = "observed exposures (no ladder in the policy)"

    # Colour scale spans the ladder, so the top rung is the top colour whatever the
    # ladder is.
    EXPO_NORM = Normalize(0.0, max(EXPO_LEVELS) if EXPO_LEVELS else 1.0)
    # Doses the server actually commanded that are NOT on the ladder we resolved:
    # a mismatch between the policy file loaded here and the one the run used.
    _off = [_x for _x in _seen if not any(abs(_x - _l) < 1e-6 for _l in EXPO_LEVELS)]
    mo.md(
        f"**Dose ladder ({_src}):** "
        + " / ".join(f"`{_l:g}`" for _l in EXPO_LEVELS)
        + " ms."
        + (
            f"\n\n⚠️ **{len(_off)} commanded exposures are off this ladder** "
            f"({', '.join(f'{_x:g}' for _x in _off[:8])} ms) — the policy file "
            "shown above is probably not the one this run served."
            if _off
            else ""
        )
    )
    return EXPO_LEVELS, EXPO_NORM


@app.cell
def _(FACET_KEYS, mo):
    # Which policy facets partition the FOVs. Pick none to pool everything, one to
    # cut at a single level (e.g. `kernel`), or several to nest (`controller` +
    # `kernel` + `lambda_move` == `arm`). This is what makes an arbitrary
    # comparison available without editing the notebook: the nested-arm design
    # means the interesting contrasts are exactly "same everything except ONE
    # facet", and that is a grouping, not a hardcoded FOV list.
    group_by = mo.ui.multiselect(
        options=FACET_KEYS, value=["arm"], label="Group FOVs by policy facet(s)",
    )
    group_by
    return (group_by,)


@app.cell
def _(FOV_META, facet_value, group_by):
    # FOVs bucketed by the chosen facets. Key = the tuple of facet values, so two
    # FOVs share a group exactly when they agree on every selected facet.
    _keys = list(group_by.value)
    FOV_GROUPS = {}
    for _f, _m in sorted(FOV_META.items()):
        _vals = tuple(facet_value(_m, _k) for _k in _keys)
        _label = " · ".join(_vals) if _vals else "all"
        FOV_GROUPS.setdefault(_label, []).append(_f)
    FOV_GROUPS
    return (FOV_GROUPS,)


@app.cell
def _(FOV_GROUPS, data, mo):
    _fovs = sorted(data["fov"].unique().to_list())
    # `None` = pool every FOV. A tuple = pool one facet group. A bare int = one
    # FOV. Group options only appear when the chosen facets actually split the run
    # (more than one group), otherwise "All (pooled)" already is the single group.
    _opts = {"All (pooled)": None}
    if len(FOV_GROUPS) > 1:
        for _lbl, _fl in FOV_GROUPS.items():
            _opts[f"Group: {_lbl}  (FOVs {','.join(map(str, _fl))})"] = tuple(_fl)
    _opts.update({f"FOV {_f}": _f for _f in _fovs})

    fov_sel = mo.ui.dropdown(options=_opts, value="All (pooled)", label="FOV / partition")
    readout_sel = mo.ui.dropdown(
        options=["cnr_norm", "cnr", "cnr_median"],
        # `cnr_median` = the raw per-cell median CNR the model consumes; for a raw
        # model it equals the controlled signal `cnr_norm` (see the REFERENCE cell).
        value="cnr_median",
        label="Readout",
    )
    mo.hstack([fov_sel, readout_sel], justify="start")
    return fov_sel, readout_sel


@app.cell
def _(DEFAULT_META, FOV_META, data, fov_sel, re):
    # cnr units + the constant setpoint (if there IS one) for the CURRENT selection.
    # A single FOV or one facet group has one policy, so its mode is unambiguous;
    # when everything is pooled, use the shared value if uniform, else the server
    # default policy.
    _sel = fov_sel.value
    if _sel is None:
        _fovs = list(FOV_META.keys())
    elif isinstance(_sel, (list, tuple)):
        _fovs = list(_sel)
    else:
        _fovs = [_sel]
    SEL_FOVS = _fovs
    _metas = [FOV_META[_f] for _f in _fovs if _f in FOV_META]

    def _uniform(_vals):
        _s = set(_vals)
        return next(iter(_s)) if len(_s) == 1 else None

    # Last-resort setpoint for a run with neither a policy file nor a startup
    # record: the target embedded in the acquisition's treatment label. Only ever
    # reached for a CONSTANT-setpoint run — an oscillation's reference comes from
    # the log and is never guessed.
    _tn = str(data["treatment_name"].mode()[0]) if data.height else ""
    _lbl = re.search(r"=\s*([0-9.]+)", _tn)
    _ut = _uniform([_m["target"] for _m in _metas if _m["target"] is not None])
    _has_policy_target = any(_m["target"] is not None for _m in _metas) or (
        DEFAULT_META.get("target") is not None
    )
    POLICY_TARGET = (
        _ut
        if _ut is not None
        else DEFAULT_META.get("target")
        if DEFAULT_META.get("target") is not None
        else None
        if _has_policy_target  # mixed constant targets across the selection
        else (float(_lbl.group(1)) if _lbl else None)
    )
    _um = _uniform([_m["mode"] for _m in _metas])
    CNR_MODE = _um if _um is not None else (DEFAULT_META.get("mode") or "norm")
    # Readouts the reference is meaningful on. The controlled signal is always
    # `cnr_norm`; a raw model's `cnr_norm` equals the parquet `cnr_median`, so the
    # reference holds there as well.
    TARGET_COLS = {"cnr_norm"} if CNR_MODE == "norm" else {"cnr_norm", "cnr_median"}
    return CNR_MODE, POLICY_TARGET, SEL_FOVS, TARGET_COLS


@app.cell
def _(data, pl):
    # How long a track was actually followed, on the real clock — the span from its
    # first to its last frame. This used to be `n_frames / 60`, which is only a
    # duration if the rig really held one frame per minute; when it does not, every
    # track reads as shorter than it was and the cutoff below silently bites
    # harder. `n_frames` is kept alongside because the two together are the
    # dropout signal: many frames over a long span means gaps.
    track_len = (
        data.group_by(["fov", "particle", "track_key"])
        .agg(
            pl.len().alias("n_frames"),
            (pl.col("hours").max() - pl.col("hours").min()).alias("length_h"),
        )
        .sort(["fov", "particle"])
    )
    return (track_len,)


@app.cell
def _(mo, np, track_len):
    # Trajectory plots below only show tracks at least this long, in REAL hours —
    # so the range has to follow the run's real duration, not a nominal 12 h.
    _max_h = float(track_len["length_h"].max() or 1.0)
    min_len = mo.ui.slider(
        0.0,
        float(np.ceil(_max_h)),
        value=min(10.0, float(np.ceil(_max_h))),
        step=0.5,
        label="Min track length (h, real clock)",
        full_width=True,
    )
    min_len
    return (min_len,)


@app.cell
def _(min_len, pl, plt, track_len):
    # Per-FOV distribution of track lengths (step outlines) with the cutoff drawn in.
    _fovs = sorted(track_len["fov"].unique().to_list())
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    for _f in _fovs:
        _lh = track_len.filter(pl.col("fov") == _f)["length_h"].to_numpy()
        _ax.hist(_lh, bins=48, histtype="step", lw=1.4, alpha=0.85, label=f"FOV {_f}")
    _ax.axvline(min_len.value, color="black", ls="--", lw=1.3)
    _ax.set_xlabel("track length (h)")
    _ax.set_ylabel("# tracks")
    _ax.set_title(f"Track-length distribution per FOV (cutoff = {min_len.value:.1f} h)")
    _ax.legend(fontsize=8, title="FOV")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(data, fov_sel, min_len, pl, readout_sel, track_len):
    y_col = readout_sel.value

    # Tracks that clear the length cutoff — pooled over all FOVs, a same-config
    # group (tuple), or one FOV (int), per the selector.
    _keep_tl = track_len.filter(pl.col("length_h") >= min_len.value)
    _sel = fov_sel.value
    if _sel is not None:
        _fovs = list(_sel) if isinstance(_sel, (list, tuple)) else [_sel]
        _keep_tl = _keep_tl.filter(pl.col("fov").is_in(_fovs))
    _keep = _keep_tl["track_key"].to_list()

    # track_key already encodes the FOV, so this restricts to the selection.
    fov_df = data.filter(pl.col("track_key").is_in(_keep))

    # Slim, per-track-sorted frame so matplotlib draws clean per-track lines.
    # `r_t` rides along: the reference is per-cell and per-frame (an oscillating
    # one differs between two cells at the same instant, by their phase offset),
    # so it cannot be re-attached later from a scalar.
    plot_df = fov_df.select(
        ["hours", "timestep", "track_key", "particle", "light_on", "exposure_ms",
         "r_t", "segment", "phase_offset_min", y_col]
    ).sort(["track_key", "timestep"])
    # Every-2nd-minute view for the3 dense all-track backgrounds (halves the point
    # count); single-track detail below still uses full-resolution plot_df.
    plot_df_sparse = plot_df.filter(pl.col("timestep") % 2 == 0)

    # Per-timestep summary: mean readout + mean commanded exposure across cells.
    summary = (
        fov_df.group_by("timestep")
        .agg(
            pl.col("hours").first(),
            pl.col(y_col).mean().alias("mean"),
            pl.col(y_col).median().alias("median"),
            pl.col("exposure_ms").mean().alias("mean_exposure"),
            pl.len().alias("n_cells"),
        )
        .sort("timestep")
    )

    # Per-timestep breakdown of how many cells sat at each exposure level.
    expo_ts = (
        fov_df.group_by(["timestep", "exposure_ms"])
        .agg(pl.col("hours").first(), pl.len().alias("n"))
        .sort("timestep")
    )
    return expo_ts, fov_df, plot_df, plot_df_sparse, y_col


@app.cell
def _(POLICY_TARGET, fov_df, pl):
    # --- THE REFERENCE ----------------------------------------------------
    # What the controller was tracking, for the current selection. Three cases,
    # and the plots below branch on REF_KIND rather than assuming any of them:
    #
    #   "constant" — one setpoint for every cell at every frame (a `hold` run).
    #                Drawn as a horizontal line, as before.
    #   "varying"  — the setpoint moves (an `oscillation`), and differs BETWEEN
    #                cells at the same frame by their phase offset. Drawn as one
    #                curve per phase group. There is no scalar target here, and
    #                inventing one (the old fallback of 1.4) puts a flat line
    #                through a waveform and makes every tracking plot a lie.
    #   "none"     — the log predates the reference annotation and the policy has
    #                no constant target either. Nothing is drawn.
    #
    # Source of truth is the server's own per-cell `r_t`, logged by
    # Objective.annotate precisely so the analysis never re-derives the waveform
    # from its parameters and gets it subtly wrong. The policy's constant target is
    # only used when the log has no annotation at all.
    _r = fov_df["r_t"].drop_nulls() if fov_df.height else fov_df["r_t"]
    _uniq = _r.unique().to_list() if _r.len() else []

    if len(_uniq) == 1:
        REF_KIND, REF_VALUE = "constant", float(_uniq[0])
    elif len(_uniq) > 1:
        REF_KIND, REF_VALUE = "varying", None
    elif POLICY_TARGET is not None:
        REF_KIND, REF_VALUE = "constant", float(POLICY_TARGET)
    else:
        REF_KIND, REF_VALUE = "none", None

    # One reference curve per phase group, straight from the logged values —
    # (label, hours, r_t). Empty unless the reference actually varies.
    REF_CURVES = []
    if REF_KIND == "varying":
        _c = (
            fov_df.drop_nulls("r_t")
            .group_by(["phase_offset_min", "timestep"])
            .agg(pl.col("hours").first(), pl.col("r_t").median())
            .sort(["phase_offset_min", "timestep"])
        )
        for _off in sorted(_c["phase_offset_min"].unique().to_list(), key=lambda _v: (_v is None, _v)):
            _g = _c.filter(
                pl.col("phase_offset_min").is_null()
                if _off is None
                else pl.col("phase_offset_min") == _off
            )
            _lab = "reference" if _off is None else f"ref φ={_off:g}m"
            REF_CURVES.append((_lab, _g["hours"].to_numpy(), _g["r_t"].to_numpy()))

    # End of the settle-in hold, in hours. Those frames mix the start-up transient
    # with the tracking response and the analysis is meant to drop them, so every
    # time-axis plot shades them out rather than leaving them to be mistaken for
    # tracking.
    _st = fov_df.filter(pl.col("segment") == "settle") if "segment" in fov_df.columns else None
    SETTLE_END_H = float(_st["hours"].max()) if _st is not None and _st.height else None
    return REF_CURVES, REF_KIND, REF_VALUE, SETTLE_END_H


@app.cell
def _(FOV_META, REF_KIND, SEL_FOVS, fov_df, pl):
    # --- PHASE ALIGNMENT --------------------------------------------------
    # Every cell is deliberately offset from its neighbours by a quarter cycle, so
    # on a wall-clock axis the population is four superimposed copies of the same
    # waveform in antiphase. Averaging them cancels the oscillation; drawing them
    # separately is four curves plus four references. Neither is a plotting
    # problem — it is all an unaligned axis can show.
    #
    # The fix is to put every cell on one clock. A cell with offset φ reaches cycle
    # position (t - settle + φ); the φ=0 group reaches that same position at
    # t' = t + φ. So re-stamping every sample at `t + φ` puts the whole population
    # in phase with the time axis INTACT — the median then oscillates properly and
    # you can still see the response change across twelve hours.
    #
    # (Wrapping onto a single cycle as well — modulo the period — is the other
    # option, but it discards the run's time axis and averages any change over the
    # run into the band. Alignment alone keeps both, so that is what this does.)
    #
    # Settle frames are dropped: the reference is flat there and the start-up
    # transient is still running, so they describe neither tracking nor the cycle.
    #
    # When NO cell carries a phase spread — a `hold` run (one flat reference), a
    # `schedule` run (a step train every cell follows at the same frame), or a
    # `frequency_staircase` whose cells all sit at φ=0 — there is no antiphase
    # fold to do. The frame below then degenerates to the wall-clock frame:
    # offsets fill to 0, bins floor to whole minutes, the trim is a no-op. The
    # population plot renders that plain axis instead, so the panel's summary
    # view is available on every run, not just the oscillations.
    _offs = (
        fov_df["phase_offset_min"].drop_nulls()
        if "phase_offset_min" in fov_df.columns
        else fov_df["phase_offset_min"].clear()
    )
    # A phase shift worth folding exists only when the offsets actually SPREAD
    # the cells. An objective that logs one constant offset (e.g. a staircase
    # with every cell at φ=0) has no antiphase populations to fold, and its
    # wall-clock summary stays meaningful — treating a logged 0.0 as "phase
    # shift" would silently kill the plot for that arm and for any pooled view
    # that contains it.
    HAS_PHASE_SPREAD = _offs.len() > 0 and float(_offs.abs().max() or 0.0) > 0.0
    _pers = {FOV_META[_f].get("period_min") for _f in SEL_FOVS if _f in FOV_META}
    _pers = {_p for _p in _pers if _p}
    # Alignment is only meaningful when every field in scope shares one cycle —
    # two different periods on one axis is a meaningless overlay, not a cleaner
    # plot. The cells must also be spread in phase to be shifted: without spread
    # the "alignment" would claim a shift that did nothing.
    PERIOD_MIN = next(iter(_pers)) if len(_pers) == 1 else None
    CAN_ALIGN = (
        bool(PERIOD_MIN)
        and REF_KIND == "varying"
        and fov_df.height > 0
        and HAS_PHASE_SPREAD
    )

    # Computed unconditionally so `aligned` always has the same schema — every
    # plot below reads the same columns whether a shift happened or not. With no
    # phase spread the shift is zero and this IS the wall-clock frame.
    aligned = (
        fov_df.filter((pl.col("segment") != "settle") | pl.col("segment").is_null())
        .with_columns(
            (pl.col("timestep") + pl.col("phase_offset_min").fill_null(0.0)).alias("aligned_min")
        )
        # AGGREGATE ON `aligned_bin`, NEVER ON RAW `aligned_min`. The offsets are
        # period / n_phase_groups, a half-integer whenever the period does not
        # divide by the group count (50/4 = 12.5). Cells then sit on two
        # interleaved grids — whole minutes for the φ=0 and φ=25 groups, half
        # minutes for φ=12.5 and φ=37.5 — so consecutive raw positions hold
        # DISJOINT sets of cells and any statistic over them alternates between two
        # different populations. That renders as a sawtooth and reads as
        # measurement noise; it is entirely an artifact of the grid.
        #
        # Flooring to whole minutes puts every group in the same bin, which is the
        # acquisition's own resolution. It leaves one group up to half a frame
        # ahead of another inside a bin — bounded, quantified in the plot title,
        # and far milder than swinging between disjoint populations. On a period
        # that divides evenly (40/4 = 10) it is a no-op.
        .with_columns(pl.col("aligned_min").floor().alias("aligned_bin"))
        .drop_nulls("aligned_bin")
    )

    # Trim to the window every phase group actually covers. Group φ spans aligned
    # times [φ, T + φ], so only [max φ, T + min φ] holds all of them. Outside it
    # the median is taken over a shrinking subset of the groups — i.e. it decays
    # back toward the unaligned, partly-cancelling average exactly at the two ends.
    # Same class of error as the sawtooth: a statistic whose underlying population
    # silently changes along the axis. With no spread every offset is 0 and the
    # window is the full axis — a no-op.
    if aligned.height:
        _o = aligned["phase_offset_min"].fill_null(0.0)
        _tmax = float(aligned["timestep"].max())
        aligned = aligned.filter(
            (pl.col("aligned_min") >= float(_o.max()))
            & (pl.col("aligned_min") <= _tmax + float(_o.min()))
        )
    return CAN_ALIGN, HAS_PHASE_SPREAD, PERIOD_MIN, aligned


@app.cell
def _(CNR_MODE, PHASE_CMAP, REF_CURVES, REF_KIND, REF_VALUE, SETTLE_END_H, np):
    # Drawing the reference is the same decision in six plots, so it is made once
    # here. Every plot calls this instead of reaching for a scalar target: that is
    # what keeps a `hold` run and an `oscillation` run rendering correctly from the
    # same code, and what stops a flat line reappearing under a moving setpoint.
    def phase_colors(_n):
        return PHASE_CMAP(np.linspace(0.15, 0.9, max(_n, 1)))

    def draw_reference(_ax, colors=None, lw=1.4, label=True, zorder=4):
        """Overlay the reference on a (time, readout) axes. No-op if there is none.

        `colors`: None -> one colour per phase group; a colour string -> all
        curves in that colour (for plots whose own palette would clash).
        """
        if REF_KIND == "constant":
            _c = colors if isinstance(colors, str) else "black"
            _ax.axhline(
                REF_VALUE, color=_c, ls="--", lw=lw, zorder=zorder,
                label=f"reference {REF_VALUE:g} ({CNR_MODE})" if label else None,
            )
        elif REF_KIND == "varying":
            _cs = (
                [colors] * len(REF_CURVES)
                if isinstance(colors, str)
                else phase_colors(len(REF_CURVES))
            )
            for (_lab, _h, _r), _col in zip(REF_CURVES, _cs):
                _ax.plot(_h, _r, color=_col, ls="--", lw=lw, alpha=0.95, zorder=zorder,
                         label=_lab if label else None)

    def shade_settle(_ax, label=True):
        """Grey out the settle-in hold — those frames are not tracking."""
        if not SETTLE_END_H:
            return
        _ax.axvspan(
            0, SETTLE_END_H, color="#888888", alpha=0.13, lw=0, zorder=0,
            label="settle" if label else None,
        )

    return draw_reference, phase_colors, shade_settle


@app.cell
def _(
    ARM_OF,
    CNR_MODE,
    FOV_META,
    REF_CURVES,
    REF_KIND,
    REF_VALUE,
    SEL_FOVS,
    SETTLE_END_H,
    fov_df,
    fov_sel,
    min_len,
    mo,
):
    mo.stop(
        fov_df.is_empty(),
        mo.md(f"**No tracks ≥ {min_len.value:.1f} h.** Lower the cutoff."),
    )
    _n_tracks = fov_df["track_key"].n_unique()
    _n_rows = fov_df.shape[0]
    _hmax = fov_df["hours"].max()
    _sel = fov_sel.value
    if _sel is None:
        _scope = "All FOVs (pooled)"
    elif isinstance(_sel, (list, tuple)):
        _scope = f"FOVs {','.join(map(str, _sel))} (pooled group)"
    else:
        _scope = f"FOV {_sel}"
    # Arm(s) in scope — the experimental variable. Mixing arms in one pooled view
    # is legitimate (it is how you see the population as a whole) but it is also
    # how an arm contrast gets averaged away, so it is stated outright.
    _arms = sorted({ARM_OF.get(_f) for _f in SEL_FOVS if _f in FOV_META}, key=lambda _a: (_a is None, _a))
    _specs = sorted({FOV_META[_f]["arm_spec"] for _f in SEL_FOVS if _f in FOV_META})
    _arm_txt = (
        f"arm {_arms[0]} — {_specs[0]}"
        if len(_specs) == 1
        else f"**{len(_specs)} arms mixed** ({', '.join(f'arm {_a}' for _a in _arms)})"
        if _specs
        else "?"
    )
    if REF_KIND == "constant":
        _ref_txt = f"held to **{REF_VALUE:g}**"
    elif REF_KIND == "varying":
        _lo, _hi = min(_r.min() for _, _, _r in REF_CURVES), max(_r.max() for _, _, _r in REF_CURVES)
        _ref_txt = (
            f"tracking a **moving reference** over **{_lo:g}–{_hi:g}**, "
            f"{len(REF_CURVES)} phase group(s)"
            + (f", settle ends at {SETTLE_END_H:.1f} h" if SETTLE_END_H else "")
        )
    else:
        _ref_txt = "**no reference recorded** (no `r_t` in the log, no constant target in the policy)"
    mo.md(
        f"**{_scope}** — {_n_tracks} tracks ≥ {min_len.value:.1f} h, "
        f"{_n_rows} rows, {_hmax:.1f} h. "
        f"Control: **{CNR_MODE}**-cnr {_ref_txt}. Arm: {_arm_txt}."
    )
    return


@app.cell
def _(EXPO_CMAP, EXPO_LEVELS, expo_ts, np, pl, plt):
    # Population control signal: fraction of cells at each exposure level over time.
    _tot = expo_ts.group_by("timestep").agg(pl.col("n").sum().alias("tot"))
    _m = expo_ts.join(_tot, on="timestep").with_columns((pl.col("n") / pl.col("tot")).alias("frac"))
    _ts = sorted(_m["timestep"].unique().to_list())
    # Real acquisition time per frame, not `timestep / 60` — the frames are not a
    # minute apart whenever the loop runs behind.
    _tmap = dict(zip(expo_ts["timestep"].to_list(), expo_ts["hours"].to_list()))
    _hours = np.array([_tmap[_t] for _t in _ts], dtype=float)
    _series = []
    for _lvl in EXPO_LEVELS:
        _s = _m.filter(pl.col("exposure_ms") == _lvl)
        _map = dict(zip(_s["timestep"].to_list(), _s["frac"].to_list()))
        _series.append([_map.get(_t, 0.0) for _t in _ts])

    _fig, _ax = plt.subplots(figsize=(11, 2.8))
    _colors = EXPO_CMAP(np.linspace(0.15, 0.95, len(EXPO_LEVELS)))
    _ax.stackplot(_hours, *_series, labels=[f"{_l:g} ms" for _l in EXPO_LEVELS], colors=_colors)
    _ax.set_xlim(float(_hours.min()), float(_hours.max()))
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("time (h, real clock)")
    _ax.set_ylabel("fraction of cells")
    # The rungs are the policy's ladder, so this reads as "what the controller did
    # with the choices it had" rather than against a fixed 0-800 ms grid.
    _ax.set_title(
        "Commanded stim exposure across cells over time — ladder "
        + "/".join(f"{_l:g}" for _l in EXPO_LEVELS) + " ms"
    )
    _ax.legend(loc="upper right", fontsize=7, ncol=len(EXPO_LEVELS))
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Phase-aligned population

    Each cell is offset from its neighbours by a quarter cycle, so on a raw
    wall-clock axis the population is four copies of the same waveform in
    antiphase. Averaging them cancels the oscillation; drawing them apart is four
    curves against four references. Neither is readable, and neither is a plotting
    failure — it is all an unaligned axis can show.

    Aligning shifts every cell by **its own offset**, re-stamping its samples at
    `t + φ`. The whole population lands on one clock, in phase, with the time axis
    intact: the median oscillates properly *and* you can still see the response
    change across twelve hours. Settle frames are dropped.

    **This is the view to read for tracking quality.** With more than one arm in
    scope it draws one curve per arm against the shared reference — the comparison
    the run exists to make. The dashed reference doubles as the check that the
    shift is right; any residual disagreement between the phase groups is
    quantified in the title.

    When there is **no phase shift** — a `hold` run (flat reference), a `schedule`
    run (a step train every cell follows at the same frame), or any selection
    whose offsets carry no spread — there is nothing to fold, and this panel
    falls back to the plain wall-clock population summary: the same per-arm
    median + IQR, each arm drawn against its **own** reference (the arms may hold
    different setpoints, so there is no one shared curve to overlay).
    """)
    return


@app.cell
def _(
    ARM_OF,
    CAN_ALIGN,
    FOV_META,
    HAS_PHASE_SPREAD,
    PERIOD_MIN,
    REF_KIND,
    REF_VALUE,
    TARGET_COLS,
    aligned,
    mo,
    pl,
    plt,
    y_col,
):
    # The panel doubles as the plain population summary when there is no phase
    # shift to fold — a `hold` run (one flat reference), a `schedule` run (a step
    # train every cell follows at the same frame), or a `frequency_staircase`
    # whose cells all sit at φ=0 carry offsets with no spread, so `aligned`
    # degenerates to the wall-clock frame and this draws the per-arm median + IQR
    # on an honest time axis. Only a selection whose cells ARE spread in phase
    # but cannot be aligned (no one shared cycle) stops here: that axis would be
    # an antiphase overlay, not a cleaner plot.
    mo.stop(
        HAS_PHASE_SPREAD and not CAN_ALIGN,
        mo.md(
            "_The cells are offset in phase but the selection has no single shared "
            "cycle length — the populations are in antiphase and cannot be folded "
            "onto one clock._"
        ),
    )
    mo.stop(
        y_col not in TARGET_COLS,
        mo.md(
            f"_Not a controlled readout — the reference lives on "
            f"`{'/'.join(sorted(TARGET_COLS))}`, not `{y_col}`._"
        ),
    )

    # One curve per ARM, because that is the contrast; with a single arm in scope
    # this degrades to one curve and the band carries the spread.
    _d = aligned.with_columns(
        pl.col("fov").replace_strict(ARM_OF, default=None).alias("arm")
    )
    _arms = sorted([_a for _a in _d["arm"].unique().to_list() if _a is not None])
    _fig, _ax = plt.subplots(figsize=(11, 4.6))
    _cmap = plt.cm.tab10

    for _i, _a in enumerate(_arms):
        _g = (
            _d.filter(pl.col("arm") == _a)
            .group_by("aligned_bin")
            .agg(
                pl.col(y_col).median().alias("mid"),
                pl.col(y_col).quantile(0.25).alias("lo"),
                pl.col(y_col).quantile(0.75).alias("hi"),
            )
            .sort("aligned_bin")
        )
        _x = _g["aligned_bin"].to_numpy() / 60.0
        _col = _cmap(_i % 10)
        # The band is drawn only for a single arm — four overlapping IQR bands are
        # the same unreadable mess this section exists to remove.
        if len(_arms) == 1:
            _ax.fill_between(_x, _g["lo"].to_numpy(), _g["hi"].to_numpy(),
                             color=_col, alpha=0.22, lw=0, label="IQR")
        _spec = FOV_META[next(_f for _f in ARM_OF if ARM_OF[_f] == _a)]["arm_spec"]
        _ax.plot(_x, _g["mid"].to_numpy(), color=_col, lw=2.0,
                 label=f"arm {_a} — {_spec}")

    def _draw_ref(_g, _col, _lab, _lw):
        """One arm's reference as a dashed overlay. Falls back to the policy's
        flat target when the run predates the per-cell `r_t` annotation."""
        if _g.height:
            _ax.plot(_g["aligned_bin"].to_numpy() / 60.0, _g["r"].to_numpy(),
                     color=_col, ls="--", lw=_lw, alpha=0.9, zorder=4, label=_lab)
        elif REF_KIND == "constant":
            _ax.axhline(REF_VALUE, color=_col, ls="--", lw=_lw, alpha=0.9,
                        zorder=4, label=_lab)

    # Worst disagreement between the phase groups' own references inside one bin.
    # Meaningful only when a shift was actually applied — on an unaligned axis
    # every cell shares one reference, so the spread is zero by construction.
    #
    # Zero is NOT automatically a bug:
    #
    #   EXPECTED — when the period does not divide by the group count the offsets
    #     are half-integers (50/4 = 12.5) and whole-minute bins leave one group up
    #     to half a frame ahead of another. On the steepest ramp that is
    #     (max per-minute step)/2 of reference disagreement and nothing more.
    #   REAL — materially above that bound means the shift itself is wrong and
    #     every curve above is a blend of phases rather than a population in phase.
    _mtxt = ""
    if CAN_ALIGN:
        # One shared reference, put through the SAME alignment. Every arm tracks
        # this one curve — that is what makes them comparable at all. It doubles
        # as the check that the shift is right: `r_t` is per-cell, so a wrong
        # shift makes the phase groups' references disagree inside a bin and this
        # comes out smeared rather than as a clean waveform.
        _r = (
            _d.drop_nulls("r_t").group_by("aligned_bin")
            .agg(pl.col("r_t").median().alias("r"),
                 (pl.col("r_t").max() - pl.col("r_t").min()).alias("spread"))
            .sort("aligned_bin")
        )
        _draw_ref(_r, "black", "reference", 1.8)
        _mis = float(_r["spread"].max() or 0.0)
        _rv = _r["r"].to_numpy()
        _bound = 0.5 * float(abs(_rv[1:] - _rv[:-1]).max()) if len(_rv) > 1 else 0.0
        if _mis <= _bound * 1.5 + 1e-9:
            _mtxt = f"  ·  ±{_mis:.3f} half-frame quantisation" if _mis > 1e-6 else ""
        else:
            _mtxt = f"  ⚠ PHASE MISMATCH {_mis:.3f} (expected ≤ {_bound:.3f}) — curves are a blend"
    elif len(_arms) == 1:
        # No shift, one arm: the reference is a single curve (flat, schedule, or
        # staircase), drawn like the aligned case for readability.
        _ra = (
            _d.drop_nulls("r_t").group_by("aligned_bin")
            .agg(pl.col("r_t").median().alias("r"))
            .sort("aligned_bin")
        )
        _draw_ref(_ra, "black", "reference", 1.8)
    else:
        # No shift, several arms: their references DIFFER (a hold next to a
        # schedule next to a staircase), so a pooled `r_t` median would smear into
        # a line none of them followed. Draw each arm against its own.
        for _i, _a in enumerate(_arms):
            _col = _cmap(_i % 10)
            _ra = (
                _d.filter(pl.col("arm") == _a).drop_nulls("r_t")
                .group_by("aligned_bin")
                .agg(pl.col("r_t").median().alias("r"))
                .sort("aligned_bin")
            )
            _draw_ref(_ra, _col, f"arm {_a} ref", 1.2)

    if CAN_ALIGN:
        _title = (
            f"Phase-aligned {y_col} — {PERIOD_MIN:g}-min cycle, aligned by each "
            f"cell's own φ · {_d['track_key'].n_unique()} tracks, {_d.height} "
            f"cell-frames" + _mtxt
        )
        _xlab = "phase-aligned time (h)"
    else:
        _title = (
            f"Population {y_col} · {_d['track_key'].n_unique()} tracks, "
            f"{_d.height} cell-frames — no phase shift, wall-clock axis"
        )
        _xlab = "time (h)"
    _ax.set_xlabel(_xlab)
    _ax.set_ylabel(y_col)
    _ax.set_title(_title)
    _ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(REF_KIND, REF_VALUE, fov_df, mo, pl):
    # Order tracks by tracking error of the controlled signal (`cnr_norm`, whatever
    # metric the server drove) against the reference — RMSE, so it is comparable
    # across track lengths; best-tracking first. Total exposure is the light dose
    # the controller spent on that cell over its whole trajectory — two tracks can
    # track equally well for very different effort.
    #
    # The error is taken per-frame against that cell's OWN `r_t`, so it means the
    # same thing under a moving reference as under a hold. Scoring an oscillation
    # against a scalar would rank cells by how close they sat to the waveform's
    # mean, i.e. reward exactly the cells that failed to track it.
    _err = (
        (pl.col("cnr_norm") - pl.col("r_t"))
        if REF_KIND == "varying"
        else (pl.col("cnr_norm") - REF_VALUE)
        if REF_KIND == "constant"
        else None
    )
    # Settle frames are excluded: they are the start-up transient, identical by
    # construction across arms, and would dilute the ranking.
    # `!= "settle"` is null (i.e. drops the row) wherever `segment` is null, so a
    # run that logged the annotation only part-way through would silently lose its
    # unannotated frames. Keep them: unlabelled is not the same as settle.
    _scored = (
        fov_df.filter((pl.col("segment") != "settle") | pl.col("segment").is_null())
        if fov_df["segment"].drop_nulls().len()
        else fov_df
    )
    _aggs = [(pl.col("exposure_ms").sum() / 1000.0).alias("total_exposure_s")]
    if _err is not None:
        _aggs.insert(0, (_err ** 2).mean().sqrt().alias("rmse"))
    _l2 = _scored.group_by("track_key").agg(_aggs)
    _l2 = (
        _l2.drop_nulls("rmse").sort("rmse")
        if _err is not None
        else _l2.sort("total_exposure_s", descending=True)
    )
    mo.stop(_l2.is_empty(), mo.md("_No tracks pass the length cutoff._"))
    if _err is not None:
        _labels = [
            f"{k}  ·  RMSE-to-ref {r:.3f}  ·  {e:.1f} s light"
            for k, r, e in zip(_l2["track_key"], _l2["rmse"], _l2["total_exposure_s"])
        ]
        _lbl = "Highlight track (best-tracking first)"
    else:
        _labels = [
            f"{k}  ·  {e:.1f} s light"
            for k, e in zip(_l2["track_key"], _l2["total_exposure_s"])
        ]
        _lbl = "Highlight track (no reference logged — ordered by light dose)"
    particle_sel = mo.ui.dropdown(
        options=dict(zip(_labels, _l2["track_key"].to_list())),
        value=_labels[0],
        label=_lbl,
    )
    particle_sel
    return (particle_sel,)


@app.cell
def _(
    EXPO_CMAP,
    EXPO_NORM,
    REF_KIND,
    TARGET_COLS,
    draw_reference,
    particle_sel,
    pl,
    plot_df,
    plot_df_sparse,
    plt,
    shade_settle,
    y_col,
):
    # One track picked out against the faint background of all others; markers
    # colored by the exposure level the server commanded at each frame.
    _fig, _ax = plt.subplots(figsize=(11, 4.8))
    for _key, _g in plot_df_sparse.group_by("track_key"):
        _ax.plot(_g["hours"].to_numpy(), _g[y_col].to_numpy(), color="#bbbbbb", alpha=0.12, lw=0.5)

    _one = plot_df.filter(pl.col("track_key") == particle_sel.value).sort("timestep")
    _ax.plot(_one["hours"].to_numpy(), _one[y_col].to_numpy(), color="#333333", lw=1.4, zorder=2)
    _sc = _ax.scatter(
        _one["hours"].to_numpy(), _one[y_col].to_numpy(),
        c=_one["exposure_ms"].to_numpy(), cmap=EXPO_CMAP, norm=EXPO_NORM,
        s=26, zorder=3, edgecolor="none",
    )
    _fig.colorbar(_sc, ax=_ax, label="exposure (ms)")
    shade_settle(_ax)
    if y_col in TARGET_COLS:  # controller setpoint, in this run's cnr units
        if REF_KIND == "varying":
            # THIS cell's own reference, not the population's: under a phase-offset
            # waveform the neighbouring cell is chasing a different curve, and
            # drawing all four here would make a well-tracking cell look wrong.
            _r = _one.drop_nulls("r_t")
            _ax.plot(_r["hours"].to_numpy(), _r["r_t"].to_numpy(), color="black",
                     ls="--", lw=1.4, zorder=4, label="this cell's reference")
        else:
            draw_reference(_ax)
        _ax.legend(loc="upper right", fontsize=8)

    _n_on = int(_one.filter(pl.col("light_on")).height)
    _dose_s = float(_one["exposure_ms"].sum() or 0.0) / 1000.0
    _phase = _one["phase_offset_min"].drop_nulls()
    _ptxt = f", φ={_phase[0]:g} min" if _phase.len() else ""
    _ax.set_xlabel("time (h, real clock)")
    _ax.set_ylabel(y_col)
    _ax.set_title(
        f"Track {particle_sel.value} ({_one.height} pts, {_n_on} light-on, "
        f"{_dose_s:.1f} s total light{_ptxt}) — color = commanded exposure"
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(
    CNR_MODE,
    REF_KIND,
    REF_VALUE,
    TARGET_COLS,
    alt,
    mo,
    particle_sel,
    pl,
    plot_df,
    y_col,
):
    # Zoomable version of the single-cell trace above: the selected track's readout
    # over time, points coloured by the commanded exposure. One track (≤ ~720 rows)
    # is safe to inline as Vega. Scroll / drag to pan-zoom; double-click to reset.
    _one = plot_df.filter(pl.col("track_key") == particle_sel.value).sort("timestep")
    _base = alt.Chart(_one).encode(
        x=alt.X("hours:Q", title="time (h, real clock)"),
        y=alt.Y(f"{y_col}:Q", title=y_col, scale=alt.Scale(zero=False)),
    )
    _line = _base.mark_line(color="#999999", strokeWidth=1.0)
    _pts = _base.mark_point(filled=True, size=55).encode(
        color=alt.Color(
            "exposure_ms:Q", scale=alt.Scale(scheme="yelloworangered"), title="exposure (ms)"
        ),
        tooltip=[
            "timestep:Q", "hours:Q",
            alt.Tooltip(f"{y_col}:Q", format=".3f"), "exposure_ms:Q",
        ],
    )
    _layers = [_line, _pts]
    _tsuffix = ""
    if y_col in TARGET_COLS and REF_KIND == "constant":
        # Same field name as the y encoding so the layered y-scale is shared.
        _tgt = pl.DataFrame({y_col: [float(REF_VALUE)]})
        _layers.append(
            alt.Chart(_tgt).mark_rule(color="black", strokeDash=[6, 3]).encode(y=f"{y_col}:Q")
        )
        _tsuffix = f" · reference {REF_VALUE:g} ({CNR_MODE})"
    elif y_col in TARGET_COLS and REF_KIND == "varying":
        # A moving reference is a line, not a rule — and it is this cell's own
        # `r_t`, which already rides along in plot_df. Renaming it to the y field
        # shares the layered y-scale, exactly as the rule does above.
        _ref = _one.drop_nulls("r_t").select(
            pl.col("hours"), pl.col("r_t").alias(y_col)
        )
        _layers.append(
            alt.Chart(_ref)
            .mark_line(color="black", strokeDash=[6, 3], strokeWidth=1.4)
            .encode(x="hours:Q", y=f"{y_col}:Q")
        )
        _tsuffix = f" · tracking reference ({CNR_MODE})"
    _chart = (
        alt.layer(*_layers)
        .properties(
            height=340, width="container",
            title=f"{particle_sel.value}: {y_col} + exposure — drag/scroll to zoom{_tsuffix}",
        )
        .interactive()
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(fov_df, np, particle_sel, pl, plot_df, plt):
    # Cumulative light dose the controller spent on the selected track over its
    # whole trajectory (running sum of commanded exposure, in seconds), with the
    # population-median cumulative dose as a reference so you can see whether this
    # cell was driven harder or softer than typical. Per-frame exposure is drawn as
    # faint bars on the right axis, so bursts of stimulation are visible under the
    # smooth cumulative curve.
    _one = plot_df.filter(pl.col("track_key") == particle_sel.value).sort("timestep")

    # Population median cumulative: cumulative dose per track, then median per frame.
    _cum_all = (
        fov_df.select(["track_key", "timestep", "hours", "exposure_ms"])
        .sort(["track_key", "timestep"])
        .with_columns((pl.col("exposure_ms").cum_sum().over("track_key") / 1000.0).alias("cum_s"))
    )
    _ref = (
        _cum_all.group_by("timestep")
        .agg(pl.col("hours").first(), pl.col("cum_s").median().alias("med_cum_s"))
        .sort("timestep")
    )

    _h = _one["hours"].to_numpy()
    _cum = (_one["exposure_ms"].cum_sum().to_numpy()) / 1000.0
    # Bar width = the run's real median frame spacing, so the bars tile the axis.
    # A hardcoded 1/60 h assumes a 1-minute cadence and leaves gaps (or overlaps)
    # the moment the real one differs.
    _bar_w = (
        float(np.median(np.diff(_ref["hours"].to_numpy())))
        if _ref.height > 1
        else 1.0 / 60.0
    )

    _fig, _ax = plt.subplots(figsize=(11, 3.6))
    _axr = _ax.twinx()
    # Right axis first, so the cumulative curve draws on top of the exposure bars.
    _axr.bar(_h, _one["exposure_ms"].to_numpy(), width=_bar_w, color="#ee7733",
             alpha=0.30, align="center", label="per-frame exposure")
    _axr.set_ylabel("per-frame exposure (ms)", color="#ee7733")
    _axr.set_ylim(0, 850)

    _ax.plot(_ref["hours"].to_numpy(), _ref["med_cum_s"].to_numpy(),
             color="#999999", ls="--", lw=1.5, label="population median")
    _ax.plot(_h, _cum, color="#117733", lw=2.2, label="this track")
    _ax.fill_between(_h, 0.0, _cum, color="#117733", alpha=0.12)
    _ax.set_zorder(_axr.get_zorder() + 1)
    _ax.patch.set_visible(False)
    _ax.set_xlabel("time (h, real clock)")
    _ax.set_ylabel("cumulative light dose (s)", color="#117733")
    _ax.set_title(
        f"Cumulative stimulation — track {particle_sel.value} "
        f"({_cum[-1]:.1f} s total over {_one.height} frames)"
    )
    _ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Watch the cell

    **Click and drag** across the trace below to select a time window, then click
    **Render clip** to build a short movie of both channels (miRFP + mScarlet3) over
    that window, cropped around — and re-centred on — the tracked cell each frame.
    The player has **play/pause, frame stepping and a slider to skip to any point**.
    Hover still shows the per-frame tooltip; only the drag-select is new.

    Frames come from `acquisition.ome.zarr` `(t, fov, channel, y, x)`, sharded per
    plane so each crop reads only its window. Rendering is gated behind the button
    (and the window is capped / subsampled) so dragging never triggers a load — the
    slow part is one deliberate pass over the mount, not every mouse move.

    Note the parquet's `x` is the **row** index and `y` the **column** — verified
    against `labels/particles`, which stores the particle id per pixel. That same
    array gives the outline drawn on each frame, so you see the exact segmented cell.
    """)
    return


@app.cell
def _(alt, mo, particle_sel, pl, plot_df, y_col):
    # Small enough to inline as Vega (one track, <= ~720 rows).
    _trace = plot_df.filter(pl.col("track_key") == particle_sel.value).sort("timestep")
    # Interval brush on the time axis: click-drag selects the window to animate.
    _brush = alt.selection_interval(encodings=["x"], empty=False)
    _line = (
        alt.Chart(_trace)
        .mark_line(color="#4477aa", strokeWidth=1.4)
        .encode(
            x=alt.X("hours:Q", title="time (h)"),
            y=alt.Y(f"{y_col}:Q", title=y_col, scale=alt.Scale(zero=False)),
        )
    )
    _pts = (
        _line.mark_point(filled=True, size=45)
        .encode(
            color=alt.Color("exposure_ms:Q", scale=alt.Scale(scheme="yelloworangered"),
                            title="exposure (ms)"),
            tooltip=["timestep:Q", "hours:Q", f"{y_col}:Q", "exposure_ms:Q"],
        )
        .add_params(_brush)
    )
    range_chart = mo.ui.altair_chart(
        (_line + _pts).properties(height=220, width="container", title="drag to select a window")
    )
    range_chart
    return (range_chart,)


@app.cell
def _(mo, range_chart):
    # Live readout of the current brush (cheap — reruns on drag but loads nothing),
    # plus the button that actually triggers the render. A run_button's value is
    # True only on the click, so dragging the brush never kicks off a load.
    _sel = range_chart.value
    _n = 0 if _sel is None else len(_sel)
    if _n:
        _ts = list(_sel["timestep"]) if hasattr(_sel, "columns") else [_r["timestep"] for _r in _sel]
        _txt = f"Selected minutes **{int(min(_ts))}–{int(max(_ts))}** ({_n} frames)."
    else:
        _txt = "_Drag across the trace to select a window._"
    render_btn = mo.ui.run_button(label="Render clip", disabled=_n == 0)
    mo.vstack([mo.md(_txt), render_btn])
    return (render_btn,)


@app.cell
def _(mo):
    crop_half = mo.ui.slider(
        24, 200, value=64, step=8, label="Crop half-size (px)", full_width=True
    )
    # Display gain for the mScarlet3 (ERK biosensor) panel only. 1.0 = the same
    # auto 1–99.5 percentile stretch the other channels use; higher pulls the white
    # point down so dim cytoplasmic signal is visible. Purely a display transform —
    # it never touches the pixel data or the cnr readout.
    mscarlet_gain = mo.ui.slider(
        0.25, 6.0, value=1.0, step=0.25, label="mScarlet3 display gain", full_width=True
    )
    # Cap on frames pulled from the mount (window is subsampled to fit) and the
    # clip's playback rate.
    max_frames = mo.ui.slider(
        20, 150, value=60, step=10, label="Max frames", full_width=True
    )
    fps = mo.ui.slider(2, 15, value=6, step=1, label="Playback fps", full_width=True)
    mo.vstack([crop_half, mscarlet_gain, max_frames, fps])
    return crop_half, fps, max_frames, mscarlet_gain


@app.cell
def _(functools, zarr):
    # Cached handle to the OME-Zarr store: the pyramid level-0 image
    # `(t, fov, channel, y, x)`, the per-pixel particle-id label array, and the
    # channel names, parsed once for the click crop below.
    @functools.lru_cache(maxsize=8)
    def open_zarr(zarr_path):
        _g = zarr.open_group(zarr_path, mode="r")
        _labels = zarr.open_group(zarr_path + "/labels", mode="r")
        _names = [_c["label"] for _c in _g.attrs["ome"]["omero"]["channels"]]
        return _g["0"], _labels["particles/0"], _names

    return (open_zarr,)


@app.cell
def _(functools, np, open_zarr):
    # Cached so re-rendering (or stepping back to a frame) never re-reads the disk.
    # Keyed on the crop window, and only that window is fetched — the store is
    # sharded one shard per (t, fov, channel) plane, so this is a partial read.
    @functools.lru_cache(maxsize=256)
    def load_crop(zarr_path, t, fov, row, col, half):
        """(channels, particle-id mask, origin, channel names) around one cell.

        `row`/`col` are the parquet's `x`/`y` — see the note above. Runs off the
        main thread via asyncio.to_thread, so a slow network mount does not block
        the kernel.
        """
        _img, _part, _names = open_zarr(zarr_path)
        _r0, _r1 = max(0, row - half), min(_img.shape[3], row + half)
        _c0, _c1 = max(0, col - half), min(_img.shape[4], col + half)
        _chans = np.asarray(_img[t, fov, :, _r0:_r1, _c0:_c1])
        _mask = np.asarray(_part[t, fov, _r0:_r1, _c0:_c1])
        return _chans, _mask, (_r0, _c0), _names

    return (load_crop,)


@app.cell
async def _(
    FuncAnimation,
    Path,
    asyncio,
    crop_half,
    exp_dir_input,
    fov_df,
    fps,
    load_crop,
    max_frames,
    mo,
    mscarlet_gain,
    np,
    particle_sel,
    pl,
    plt,
    range_chart,
    render_btn,
):
    # Gated on the button: dragging the brush reruns this cell but stops here
    # (cheap) until you actually click Render.
    mo.stop(
        not render_btn.value,
        mo.md("_Select a window above and click **Render clip** to build the movie._"),
    )
    _zarr_path = Path(exp_dir_input.value.strip()) / "acquisition.ome.zarr"
    mo.stop(
        not _zarr_path.exists(),
        mo.md(f"_No `acquisition.ome.zarr` under `{_zarr_path.parent}` — nothing to show._"),
    )
    _sel = range_chart.value
    mo.stop(_sel is None or len(_sel) == 0, mo.md("_No window selected._"))
    # .value is a dataframe (pandas or polars depending on the input); read the
    # selected timesteps either way.
    _tss = (
        list(_sel["timestep"]) if hasattr(_sel, "columns") else [_r["timestep"] for _r in _sel]
    )
    _t0, _t1 = int(min(_tss)), int(max(_tss))

    _win = fov_df.filter(
        (pl.col("track_key") == particle_sel.value)
        & (pl.col("timestep") >= _t0)
        & (pl.col("timestep") <= _t1)
    ).sort("timestep")
    mo.stop(_win.is_empty(), mo.md("_No frames for this track in the window._"))
    # Subsample to the frame cap so the mount read stays bounded.
    _stride = max(1, (_win.height + max_frames.value - 1) // max_frames.value)
    _rows = _win.gather_every(_stride).to_dicts()

    # Load each frame's crop off the main thread (parquet x -> row, y -> column).
    def _load_all():
        _out = []
        for _r in _rows:
            _cy, _cx = int(round(_r["x"])), int(round(_r["y"]))
            _out.append(
                (_r, *load_crop(str(_zarr_path), int(_r["timestep"]), int(_r["fov"]), _cy, _cx, int(crop_half.value)))
            )
        return _out

    _loaded = await asyncio.to_thread(_load_all)

    # One display stretch per channel across the WHOLE window, so brightness is
    # stable frame-to-frame (per-frame auto-stretch would flicker). mScarlet gain
    # pulls its white point down, same as the still viewer.
    _names = _loaded[0][4]
    _nc = _loaded[0][1].shape[0]
    _vlim = []
    for _ci in range(_nc):
        _allpx = np.concatenate([_L[1][_ci].ravel().astype(float) for _L in _loaded])
        _lo, _hi = float(np.percentile(_allpx, 1)), float(np.percentile(_allpx, 99.5))
        if "scarlet" in str(_names[_ci]).lower():
            _hi = _lo + (_hi - _lo) / mscarlet_gain.value
        _vlim.append((_lo, max(_hi, _lo + 1)))

    # One shared figure animated by FuncAnimation, exported as a self-contained
    # HTML5 player: `to_jshtml()` embeds every frame as base64 and ships JS controls
    # — play/pause, step, a draggable frame slider (skip to any point), and a speed
    # menu — so no ffmpeg and no autoplay-only GIF. Redrawing reads from `_loaded`
    # (in memory), so scrubbing never touches the mount.
    plt.rcParams["animation.embed_limit"] = 64  # MB; default 20 is tight for long clips
    _fig, _axes = plt.subplots(1, _nc, figsize=(3.2 * _nc, 3.5), dpi=90)
    _axes = np.atleast_1d(_axes)

    def _draw(_i):
        _r, _chans, _mask, _origin, _cn = _loaded[_i]
        for _ci in range(_nc):
            _ax = _axes[_ci]
            _ax.clear()
            _lo, _hi = _vlim[_ci]
            _ax.imshow(_chans[_ci].astype(float), cmap="gray", vmin=_lo, vmax=_hi)
            _this = _mask == int(_r["particle"])
            if _this.any():
                _ax.contour(_this, levels=[0.5], colors="#66ccee", linewidths=1.0)
            _extra = f"  ×{mscarlet_gain.value:g}" if "scarlet" in str(_cn[_ci]).lower() else ""
            _ax.set_title(str(_cn[_ci]) + _extra, fontsize=9)
            _ax.axis("off")
        _on = "  •light" if float(_r.get("exposure_ms") or 0) > 0 else ""
        _fig.suptitle(
            f"{particle_sel.value}  ·  min {int(_r['timestep'])} "
            f"({_r['timestep'] / 60:.2f} h)  ·  exp {float(_r.get('exposure_ms') or 0):.0f} ms{_on}",
            fontsize=10,
        )
        return _axes

    _anim = FuncAnimation(_fig, _draw, frames=len(_loaded), interval=int(1000 / fps.value))
    _html = _anim.to_jshtml(fps=fps.value)
    plt.close(_fig)
    mo.vstack([
        mo.md(
            f"**{len(_loaded)} frames**, minutes {_t0}–{_t1}"
            + (f" (every {_stride}th)" if _stride > 1 else "")
            + f" · {fps.value} fps. Use the player's slider to skip to any frame."
        ),
        mo.iframe(_html, height="440px"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Population density at one timepoint

    The trajectory plots above show the mean, which hides shape: a population
    sitting tightly on target and one split between over- and under-driven cells
    can share a mean. This is the distribution across every cell at one frame.
    """)
    return


@app.cell
def _(fov_df, mo):
    # Frame to slice. `timestep` is a frame index, not a minute — under a cadence
    # that slipped, frame N is not minute N — so the readout carries the real hour
    # the frame was acquired.
    _ts = fov_df["timestep"]
    tp_sel = mo.ui.slider(
        int(_ts.min()), int(_ts.max()), value=int(_ts.max() // 2), step=1,
        label="Timepoint (frame index)", full_width=True,
    )
    tp_sel
    return (tp_sel,)


@app.cell
def _(
    CNR_MODE,
    REF_KIND,
    REF_VALUE,
    TARGET_COLS,
    fov_df,
    gaussian_kde,
    mo,
    np,
    phase_colors,
    pl,
    plt,
    tp_sel,
    y_col,
):
    _at = fov_df.filter(pl.col("timestep") == tp_sel.value)[y_col].drop_nulls().to_numpy()
    # The real hour this frame was acquired. `timestep / 60` would only be the
    # hour if the rig held one frame per minute, which it does not when the
    # loop runs behind.
    _tp_h = float(fov_df.filter(pl.col("timestep") == tp_sel.value)["hours"].min())
    _t0 = fov_df.filter(pl.col("timestep") == int(fov_df["timestep"].min()))[y_col]
    _t0 = _t0.drop_nulls().to_numpy()
    mo.stop(_at.size < 2, mo.md(f"_Fewer than 2 cells at frame {tp_sel.value}._"))

    _fig, _ax = plt.subplots(figsize=(11, 4.0))
    _lo, _hi = float(min(_at.min(), _t0.min())), float(max(_at.max(), _t0.max()))
    _grid = np.linspace(_lo, _hi, 400)

    # gaussian_kde needs spread: at the first frame every cell is normalized by its
    # own value, so cnr_norm is exactly 1.0 for all of them and the covariance is
    # singular. Skip the smooth curve in that case rather than crashing.
    def _kde(_v):
        return gaussian_kde(_v)(_grid) if _v.size > 1 and np.ptp(_v) > 0 else None

    # Histogram for the honest per-cell counts, KDE over it for the shape.
    _ax.hist(_at, bins=40, range=(_lo, _hi), density=True, color="#4477aa",
             alpha=0.35, label=f"minute {tp_sel.value} (n={_at.size})")
    _k = _kde(_at)
    if _k is not None:
        _ax.plot(_grid, _k, color="#4477aa", lw=2.0)
    # First frame as the reference shape — how far the population has moved.
    _k0 = _kde(_t0)
    if _k0 is not None:
        _ax.plot(_grid, _k0, color="#999999", lw=1.5, ls="--",
                 label=f"first frame (n={_t0.size})")
    elif _t0.size:
        _ax.axvline(float(_t0[0]), color="#999999", lw=1.5, ls="--",
                    label=f"first frame (all = {_t0[0]:.2f})")

    _med = float(np.median(_at))
    _ax.axvline(_med, color="#cc3311", lw=1.6, label=f"median {_med:.2f}")
    if y_col in TARGET_COLS:  # controller setpoint, in this run's cnr units
        if REF_KIND == "constant":
            _ax.axvline(REF_VALUE, color="black", ls="--", lw=1.3,
                        label=f"reference {REF_VALUE:g} ({CNR_MODE})")
        elif REF_KIND == "varying":
            # At ONE frame a phase-offset waveform gives the cells several
            # different setpoints at once, so this is several lines. It is also
            # why the spread here is not error: part of it is the design.
            _rs = (
                fov_df.filter(pl.col("timestep") == tp_sel.value)
                .drop_nulls("r_t")
                .group_by("phase_offset_min")
                .agg(pl.col("r_t").median(), pl.len().alias("n"))
                .sort("phase_offset_min")
            )
            for _row, _col in zip(_rs.iter_rows(named=True), phase_colors(_rs.height)):
                _off = _row["phase_offset_min"]
                _ax.axvline(
                    _row["r_t"], color=_col, ls="--", lw=1.4,
                    label=f"ref φ={_off:g}m = {_row['r_t']:.2f} (n={_row['n']})"
                    if _off is not None
                    else f"ref {_row['r_t']:.2f}",
                )
    _seg = fov_df.filter(pl.col("timestep") == tp_sel.value)["segment"].drop_nulls()
    _segtxt = f" · {_seg.mode()[0]}" if _seg.len() else ""
    _ax.set_xlabel(y_col)
    _ax.set_ylabel("density")
    _ax.set_title(
        f"{y_col} across all cells at frame {tp_sel.value} "
        f"({_tp_h:.2f} h real{_segtxt}) — IQR {np.percentile(_at, 25):.2f}–"
        f"{np.percentile(_at, 75):.2f}"
    )
    _ax.legend(fontsize=8)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(TARGET_COLS, draw_reference, fov_df, np, pl, plt, y_col):
    # The same density over every timepoint, so the slice above has context: each
    # column is one frame's distribution, normalized to its own cell count.
    _d = fov_df.select(["timestep", y_col]).drop_nulls()
    _v = _d[y_col].to_numpy()
    _lo, _hi = float(np.percentile(_v, 0.5)), float(np.percentile(_v, 99.5))
    _ts = _d["timestep"].to_numpy()
    _nb = 60
    _H, _xe, _ye = np.histogram2d(
        _ts, _v,
        bins=[np.arange(_ts.min(), _ts.max() + 2) - 0.5, np.linspace(_lo, _hi, _nb)],
    )
    _H = _H / np.clip(_H.sum(axis=1, keepdims=True), 1, None)  # per-frame density

    # Bin on frame index (that is the acquisition's own grid), then place the edges
    # at the hour each frame was really acquired — so a stretch in the cadence shows
    # as wider columns instead of being flattened into a uniform axis.
    _tmap = fov_df.group_by("timestep").agg(pl.col("hours").min()).sort("timestep")
    _h_of = lambda _x: np.interp(
        _x, _tmap["timestep"].to_numpy(), _tmap["hours"].to_numpy()
    )

    _fig, _ax = plt.subplots(figsize=(11, 3.8))
    # pcolormesh with shading="flat" wants the full edge arrays, one longer than
    # each axis of C — passing bin centres for x silently mismatches.
    _im = _ax.pcolormesh(_h_of(_xe), _ye, _H.T, cmap="magma", shading="flat")
    _fig.colorbar(_im, ax=_ax, label="fraction of cells")
    _ax.plot(
        _h_of(_xe[:-1] + 0.5),
        [np.median(_v[_ts == _t]) if (_ts == _t).any() else np.nan for _t in _xe[:-1] + 0.5],
        color="#66ccee", lw=1.2, label="median",
    )
    if y_col in TARGET_COLS:
        # White, not the phase palette — the magma background would swallow it.
        # Under a phase-offset reference this draws every group's curve, which is
        # the honest picture: the density is a superposition of all of them.
        draw_reference(_ax, colors="white", lw=1.2)
    _ax.set_xlabel("time (h, real clock)")
    _ax.set_ylabel(y_col)
    _ax.set_title(f"{y_col} density over time (0.5–99.5 percentile range)")
    _ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ARM_OF, FOV_META, data, mo, np, pl, plt):
    # --- DID THE SCHEDULE EXPLOIT MODEL ERROR? ----------------------------
    # The check that has to exist once a run carries an OPEN-LOOP arm.
    #
    # An open-loop schedule is designed offline against the model, so it is free to
    # find inputs the model is wrong about. When it does, the arm underperforms its own
    # prediction on real cells — and that failure biases the comparison in the direction
    # that flatters closed-loop control, because closed-loop corrects model error by
    # construction and open-loop cannot. A closed-loop win produced that way is an
    # artifact of the design, not a result, and nothing else in this notebook would
    # distinguish the two.
    #
    # `pred_cnr_h1` is the model's prediction of the NEXT frame under the dose actually
    # commanded; pairing it with that cell's next reading gives model error measured
    # under a known input. It is cleanest on an open-loop arm, where the input was not
    # chosen in reaction to the output.
    #
    # READ IT LIKE THIS:
    #   bias near zero on every arm         -> the model is honest wherever it was driven
    #   open-loop bias LARGER than closed   -> the schedule drove into a region the model
    #                                          gets wrong; discount the arm contrast by
    #                                          roughly that amount and say so
    #   skill below persistence             -> the model adds nothing over "no change",
    #                                          so an MPC built on it is steering on noise
    _err_src = data.filter(pl.col("pred_cnr_h1").is_not_null())
    mo.stop(
        _err_src.is_empty(),
        mo.md(
            "_This run predates `pred_cnr_h1` in the log — no model-error check is "
            "possible. Every run from 2026-08 onward carries it._"
        ),
    )

    # Pair each frame's prediction with the SAME cell's next frame, consecutive only.
    _nxt = (
        _err_src.select("track_key", "timestep", "cnr_norm")
        .with_columns((pl.col("timestep") - 1).alias("timestep"))
        .rename({"cnr_norm": "achieved_next"})
    )
    _pairs = (
        _err_src.select("fov", "track_key", "timestep", "hours", "cnr_norm",
                        "pred_cnr_h1", "exposure_ms", "segment")
        .join(_nxt, on=["track_key", "timestep"], how="inner")
        .filter((pl.col("segment") != "settle") | pl.col("segment").is_null())
        .with_columns(
            (pl.col("achieved_next") - pl.col("pred_cnr_h1")).alias("model_err"),
            # The baseline any forecast has to beat: "nothing changes". A model that
            # cannot beat it is not supplying the controller with information.
            (pl.col("achieved_next") - pl.col("cnr_norm")).alias("persist_err"),
            pl.col("fov").replace_strict(ARM_OF, default=None).alias("arm"),
        )
        .drop_nulls(["model_err", "arm"])
    )
    mo.stop(_pairs.is_empty(), mo.md("_No consecutive frame pairs to score._"))

    _arms = sorted(_pairs["arm"].unique().to_list())
    _fig, _axes = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True)
    _cmap = plt.cm.tab10

    _summary = []
    for _i, _a in enumerate(_arms):
        _g = _pairs.filter(pl.col("arm") == _a)
        _col = _cmap(_i % 10)
        _b = (
            _g.group_by("timestep")
            .agg(pl.col("hours").min(), pl.col("model_err").median().alias("mid"),
                 pl.col("model_err").quantile(0.25).alias("lo"),
                 pl.col("model_err").quantile(0.75).alias("hi"))
            .sort("timestep")
        )
        if len(_arms) <= 2:
            _axes[0].fill_between(_b["hours"], _b["lo"], _b["hi"], color=_col, alpha=0.18, lw=0)
        _spec = FOV_META[next(_f for _f in ARM_OF if ARM_OF[_f] == _a)]["arm_spec"]
        _axes[0].plot(_b["hours"], _b["mid"], color=_col, lw=1.6, label=f"arm {_a} — {_spec}")

        _m = _g["model_err"].to_numpy()
        _p = _g["persist_err"].to_numpy()
        _rm, _rp = float(np.sqrt((_m ** 2).mean())), float(np.sqrt((_p ** 2).mean()))
        _summary.append({
            "arm": _a, "n_pairs": _g.height,
            "bias": float(_m.mean()), "rmse": _rm,
            "persistence_rmse": _rp,
            "skill_vs_persistence_%": 100.0 * (1.0 - _rm / _rp) if _rp else None,
            "mean_dose_ms": float(_g["exposure_ms"].mean()),
        })

        # Cumulative bias: a schedule that is slightly wrong in one direction all run
        # shows up here long before it is visible frame by frame.
        _c = _g.group_by("timestep").agg(
            pl.col("hours").min(), pl.col("model_err").median().alias("mid")
        ).sort("timestep")
        _axes[1].plot(_c["hours"], np.cumsum(_c["mid"].to_numpy()), color=_col, lw=1.6)

    _axes[0].axhline(0.0, color="black", lw=1.0, ls="--")
    _axes[0].set_ylabel("achieved − predicted")
    _axes[0].legend(fontsize=8, ncol=2)
    _axes[0].set_title(
        "One-step model error under the dose actually commanded "
        "(positive = the cells outran the model)"
    )
    _axes[1].axhline(0.0, color="black", lw=1.0, ls="--")
    _axes[1].set_ylabel("cumulative median error")
    _axes[1].set_xlabel("time (h, real clock)")
    _axes[1].set_title("Cumulative drift of the same error")
    plt.tight_layout()
    model_error_by_arm = pl.DataFrame(_summary)
    mo.vstack([plt.gca(), model_error_by_arm])
    return


@app.cell
def _(mo):
    mo.md("""
    ## GPU load across the experiment

    The server samples the device on a timer (`gpu` events) and stamps each
    inference call with its own latency (`timing` on `predict` events). Both are
    plotted on experiment hours, so a spike here lines up with the trajectories
    above. Runs logged before the GPU instrumentation landed show nothing here.
    """)
    return


@app.cell
def _(gpu, np, pl, timing):
    # `gpu` samples carry only epoch `t`, so interpolate them onto experiment
    # hours using the (t, hours) pairs the predict events provide. Stays None
    # on pre-GPU logs; the plots below stop, the health check below does not.
    #
    # One point per timestep: a timestep's FOVs arrive at different `t`, so take the
    # earliest of each and its own hours (NOT `hours.first()`, which after the
    # real-clock change need not be the row that `t.min()` came from).
    _pt = (
        timing.group_by("timestep")
        .agg(pl.col("t").min(), pl.col("hours").min())
        .sort("t")
    )
    # Restrict to the predict window BEFORE interpolating. The sampler outlives the
    # experiment (it kept running ~12 h past the last predict in v5, half of all
    # samples), and np.interp clamps out-of-range input to the edge value — those
    # samples would otherwise pile onto the last bin and fake a huge terminal spike.
    _t0, _t1 = float(_pt["t"].min()), float(_pt["t"].max())
    gpu_h = None if gpu is None else gpu.filter(
        (pl.col("t") >= _t0) & (pl.col("t") <= _t1)
    ).with_columns(
        pl.Series(
            "hours",
            np.interp(
                gpu.filter((pl.col("t") >= _t0) & (pl.col("t") <= _t1))["t"].to_numpy(),
                _pt["t"].to_numpy(),
                _pt["hours"].to_numpy(),
            ),
        )
    )
    gpu_h
    return (gpu_h,)


@app.cell
def _(gpu, gpu_device, mo, timing):
    # --- IS THIS EVEN THE RIGHT GPU? --------------------------------------
    # NVML enumerates the machine's physical GPUs and ignores CUDA_VISIBLE_DEVICES;
    # torch enumerates only the visible ones. A server that hands its torch index
    # straight to NVML therefore samples a DIFFERENT CARD whenever anything remapped
    # the devices, and every panel below is then about somebody else's GPU while
    # looking perfectly healthy.
    #
    # Two checks, because the cheap one works on runs that predate the other:
    #
    #   arithmetic — `cuda_reserved_mb` is OUR process's allocator pool and
    #     `mem_total_mb` is the sampled card's capacity. Reserved cannot exceed the
    #     capacity of the card it lives on. If it does, the two counters are
    #     describing different devices and that is not a matter of opinion.
    #   declared  — newer servers log a `gpu_device` record saying how the NVML
    #     handle was resolved (`uuid` / `visible-devices` / `index-unverified`).
    _mem_total = float(gpu["mem_total_mb"].max()) if gpu is not None and "mem_total_mb" in gpu.columns else None
    _resv_max = float(timing["cuda_reserved_mb"].max()) if timing["cuda_reserved_mb"].null_count() < timing.height else None

    _msgs = []
    if _resv_max is not None and _mem_total is not None and _resv_max > _mem_total:
        _msgs.append(
            f"🔴 **The `gpu` samples are NOT the model's GPU.** This process reserved "
            f"**{_resv_max / 1000:.1f} GB** on its device, but the card being sampled "
            f"has only **{_mem_total / 1000:.1f} GB** total. Every panel in this "
            "section describes a different card — read none of it as the model's."
        )
    elif not gpu_device:
        _msgs.append(
            "🟡 This run predates the `gpu_device` record, so nothing confirms the "
            "sampled card is the model's. The arithmetic check above found no "
            "contradiction, which is weaker than a match."
        )
    elif not gpu_device.get("verified"):
        _msgs.append(
            f"🟡 NVML handle resolved by `{gpu_device.get('resolved_by')}` — "
            "unverified. Treat the device panels as indicative only."
        )
    else:
        _msgs.append(
            f"🟢 NVML handle verified by `{gpu_device.get('resolved_by')}` — "
            f"`{gpu_device.get('nvml_name', '?')}`, "
            f"{float(gpu_device.get('mem_total_mb') or 0) / 1000:.1f} GB."
        )

    # --- ALLOCATOR HEALTH -------------------------------------------------
    # `cuda_alloc_mb` is live tensors; `cuda_reserved_mb` is what torch holds from
    # the driver. Reserved climbing against a flat allocated is the caching
    # allocator never reusing its blocks — every frame asking for a shape it has
    # never seen, stranding the previous cache. It ends with the card full and every
    # allocation paying a synchronizing cudaFree/cudaMalloc, which is visible as
    # inference latency climbing with no change in the work being done.
    _t = timing.drop_nulls("cuda_reserved_mb").sort("t")
    if _t.height > 20:
        _r0 = float(_t["cuda_reserved_mb"].head(50).median())
        _r1 = float(_t["cuda_reserved_mb"].tail(50).median())
        _a1 = float(_t["cuda_alloc_mb"].tail(50).median())
        _i0 = float(_t["infer_s"].head(50).median())
        _i1 = float(_t["infer_s"].tail(50).median())
        _ratio = _r1 / max(_a1, 1e-9)
        _verdict = "🔴 leaking" if _r1 > 4 * _r0 else ("🟡 growing" if _r1 > 1.5 * _r0 else "🟢 stable")
        _msgs.append(
            f"{_verdict} **allocator:** reserved {_r0 / 1000:.1f} → **{_r1 / 1000:.1f} GB** "
            f"against {_a1:.0f} MB live ({_ratio:,.0f}x). "
            f"Inference {_i0:.2f} → **{_i1:.2f} s** over the same span."
        )

    mo.md("### GPU telemetry health\n\n" + "\n\n".join(_msgs))
    return


@app.cell
def _(gpu_h, mo, np, plt, timing):
    mo.stop(
        gpu_h is None,
        mo.md("_This run predates GPU logging — the log has no `gpu` events._"),
    )
    # Three stacked panels sharing the experiment clock: compute/memory load,
    # thermals and power draw, then per-call inference latency.
    #
    # Device load is almost all idle punctuated by brief spikes (median util ~0,
    # peaks to 100), and there are far more samples than pixels, so raw samples
    # draw as a solid block that hides both. Bin instead: a median line for the
    # typical level, a min-max band for the spikes each bin actually contains.
    _fig, _axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    _h = gpu_h["hours"].to_numpy()
    _nbin = 240
    _edges = np.linspace(_h.min(), _h.max(), _nbin + 1)
    _bin = np.clip(np.digitize(_h, _edges) - 1, 0, _nbin - 1)
    _bh = 0.5 * (_edges[:-1] + _edges[1:])

    def _agg(col):
        # Per-bin (median, min, max); bins with no samples come back as NaN.
        _v = gpu_h[col].to_numpy().astype(float)
        _out = np.full((3, _nbin), np.nan)
        for _b in np.unique(_bin):
            _s = _v[_bin == _b]
            _out[:, _b] = (np.median(_s), _s.min(), _s.max())
        return _out

    _ax0 = _axes[0]
    _u = _agg("gpu_util_pct")
    _ax0.fill_between(_bh, _u[1], _u[2], color="#4477aa", alpha=0.25, lw=0)
    _ax0.plot(_bh, _u[0], color="#4477aa", lw=1.0, label="GPU util % (median, min–max)")
    # NOTE: mem_util_pct is memory *bandwidth* activity (fraction of time the
    # memory interface was busy), NOT occupancy — it tracks gpu_util (r=0.98),
    # not mem_used_mb (r=-0.09). Occupancy is the green line on the right axis.
    _m = _agg("mem_util_pct")
    _ax0.plot(_bh, _m[0], color="#ee7733", lw=1.0, label="mem bandwidth % (median)")
    _ax0.set_ylabel("utilisation (%)")
    _ax0.set_ylim(0, 100)
    _ax0.legend(loc="upper right", fontsize=7)
    _ax0_mem = _ax0.twinx()
    _ax0_mem.plot(_bh, _agg("mem_used_mb")[0] / 1024.0, color="#117733", lw=1.0, alpha=0.8)
    _ax0_mem.set_ylabel("mem used (GB)", color="#117733")
    _ax0.set_title("GPU utilisation and memory (whole device, all processes)")

    _ax1 = _axes[1]
    _tc = _agg("temp_c")
    _ax1.plot(_bh, _tc[0], color="#cc3311", lw=1.0)
    _ax1.set_ylabel("temperature (°C)", color="#cc3311")
    _ax1_pw = _ax1.twinx()
    _pwr = _agg("power_w")
    _ax1_pw.fill_between(_bh, _pwr[1], _pwr[2], color="#882255", alpha=0.2, lw=0)
    _ax1_pw.plot(_bh, _pwr[0], color="#882255", lw=1.0, alpha=0.8)
    _ax1_pw.set_ylabel("power (W)", color="#882255")
    # `throttle` is an NVML bitmask, and bit 0 is GpuIdle — the card clocking down
    # because it has nothing to do. That is not throttling in any sense worth
    # flagging (it covers ~92% of samples on an idle-ish device), so mask it out
    # and shade only genuine slowdowns: power cap, hw slowdown, thermal.
    _REAL = ~0x1  # every reason except GpuIdle
    _thr_mask = (gpu_h["throttle"].to_numpy().astype(int) & _REAL) != 0
    _thr = _h[_thr_mask]
    if _thr.size:
        _ax1.plot(_thr, np.full(_thr.size, _tc[0][np.isfinite(_tc[0])].min()),
                  "|", color="black", alpha=0.5, ms=4)
    _ax1.set_title(
        f"Thermals and power ({_thr.size} genuinely throttled samples, "
        f"{int((gpu_h['throttle'].to_numpy() == 1).sum())} idle-clocked)"
    )

    _ax2 = _axes[2]
    _lat = timing.drop_nulls("infer_s").sort("hours")
    if _lat.height:
        _ax2.plot(
            _lat["hours"].to_numpy(), _lat["infer_s"].to_numpy() * 1000.0,
            color="#333333", lw=0.6, alpha=0.8, label="infer",
        )
        _ax2.plot(
            _lat["hours"].to_numpy(), _lat["handler_s"].to_numpy() * 1000.0,
            color="#ee7733", lw=0.6, alpha=0.6, label="handler",
        )
        _ax2.legend(loc="upper right", fontsize=8)
    _ax2.set_ylabel("latency (ms)")
    _ax2.set_xlabel("time (h, real clock)")
    _ax2.set_title("Per-call inference latency")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(gpu_h, mo, np, startup, timing):
    mo.stop(gpu_h is None)
    # Headline numbers: sustained load, and the tail latency that would actually
    # threaten the 1-minute acquisition cadence.
    _lat = timing.drop_nulls("infer_s")
    _ms = _lat["handler_s"].to_numpy() * 1000.0 if _lat.height else np.array([np.nan])
    mo.md(
        f"""
        | metric | median | p95 | max |
        |---|---|---|---|
        | GPU util (%) | {gpu_h["gpu_util_pct"].median():.0f} | {np.percentile(gpu_h["gpu_util_pct"].to_numpy(), 95):.0f} | {gpu_h["gpu_util_pct"].max():.0f} |
        | mem used (GB) | {gpu_h["mem_used_mb"].median() / 1024:.1f} | {np.percentile(gpu_h["mem_used_mb"].to_numpy(), 95) / 1024:.1f} | {gpu_h["mem_used_mb"].max() / 1024:.1f} |
        | temp (°C) | {gpu_h["temp_c"].median():.0f} | {np.percentile(gpu_h["temp_c"].to_numpy(), 95):.0f} | {gpu_h["temp_c"].max():.0f} |
        | power (W) | {gpu_h["power_w"].median():.0f} | {np.percentile(gpu_h["power_w"].to_numpy(), 95):.0f} | {gpu_h["power_w"].max():.0f} |
        | handler latency (ms) | {np.median(_ms):.1f} | {np.percentile(_ms, 95):.1f} | {np.max(_ms):.1f} |

        {gpu_h.height} GPU samples · {_lat.height} timed predict calls ·
        device `{startup.get("info", {}).get("device", "?")}` ·
        engine `{startup.get("engine", "?")}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-FOV inference latency

    Each FOV can be served by a different model / cnr-mode (see the per-FOV policy
    in the startup log), so their inference cost may differ. Below: one plot of the
    per-call inference-latency distribution for every FOV (log axis — latency is
    heavy-tailed), then a table pinning each FOV's latency to the model it ran.
    """)
    return


@app.cell
def _(mo, pl, plt, timing):
    _lat = timing.drop_nulls("infer_s")
    mo.stop(_lat.is_empty(), mo.md("_No per-call timing in this log._"))
    _fovs = sorted(_lat["fov"].unique().to_list())
    _cmap = plt.cm.tab10
    _data = [_lat.filter(pl.col("fov") == _f)["infer_s"].to_numpy() * 1000.0 for _f in _fovs]

    _fig, _ax = plt.subplots(figsize=(11, 4.2))
    _bp = _ax.boxplot(
        _data, positions=range(len(_fovs)), widths=0.6, showfliers=True,
        patch_artist=True, flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
        medianprops={"color": "black"},
    )
    for _i, _box in enumerate(_bp["boxes"]):
        _box.set_facecolor(_cmap(_i % 10))
        _box.set_alpha(0.65)
    # Latency is heavy-tailed, so keep a log y-axis — but the range spans well under
    # a decade, where the default log locator drops all but one labelled tick. Place
    # ticks at 1·2·3·5·×10ⁿ and label them as plain ms, and label the minor
    # (in-between) ticks too so the axis is actually readable.
    from matplotlib import ticker as _ticker

    _sf = _ticker.ScalarFormatter()
    _sf.set_scientific(False)
    _ax.set_yscale("log")
    _ax.yaxis.set_major_locator(_ticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    _ax.yaxis.set_major_formatter(_sf)
    _ax.yaxis.set_minor_locator(_ticker.LogLocator(base=10, subs=(1.5, 2.5, 4.0, 7.0)))
    _ax.yaxis.set_minor_formatter(_sf)
    _ax.tick_params(axis="y", which="minor", labelsize=7, labelcolor="#666666")
    _ax.set_xticks(range(len(_fovs)))
    _ax.set_xticklabels([f"FOV {_f}" for _f in _fovs])
    _ax.set_ylabel("inference latency (ms)")
    _ax.set_title(f"Per-call inference latency by FOV (n={_lat.height} calls)")
    _ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ARM_OF, FOV_META, mo, np, pl, timing):
    _lat = timing.drop_nulls("infer_s")
    mo.stop(_lat.is_empty())
    _rows = []
    for _f in sorted(_lat["fov"].unique().to_list()):
        _d = _lat.filter(pl.col("fov") == _f)
        _inf = _d["infer_s"].to_numpy() * 1000.0
        _hnd = _d["handler_s"].to_numpy() * 1000.0
        _meta = FOV_META.get(_f, {})
        _ns = _d.drop_nulls("n_scored")["n_scored"].to_numpy()
        _rows.append(
            {
                "fov": _f,
                # Latency is per-arm here because the arm IS the inference cost:
                # constant_dose scores one plan, sequence_mpc runs a CEM, and the
                # band kernel forces the full mixture rollout. Labelled from the
                # policy — an arm derived from the FOV index would mislabel every
                # row of this table against a 10-FOV file.
                "arm": ARM_OF.get(_f),
                "controller": _meta.get("controller", "?"),
                "kernel": _meta.get("kernel", "?"),
                "model": _meta.get("checkpoint", "?"),
                "mode": _meta.get("mode", "?"),
                "reference": _meta.get("reference", "?"),
                "n_calls": _d.height,
                "mean_cells": float(_ns.mean()) if _ns.size else None,
                "infer_med_ms": float(np.median(_inf)),
                "infer_p95_ms": float(np.percentile(_inf, 95)),
                "infer_max_ms": float(_inf.max()),
                "handler_med_ms": float(np.median(_hnd)),
                "handler_p95_ms": float(np.percentile(_hnd, 95)),
                "handler_max_ms": float(_hnd.max()),
            }
        )
    _tbl = pl.DataFrame(_rows).with_columns(
        pl.col(
            "mean_cells", "infer_med_ms", "infer_p95_ms", "infer_max_ms",
            "handler_med_ms", "handler_p95_ms", "handler_max_ms",
        ).round(1)
    )

    tbl = _tbl

    mo.ui.table(_tbl, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Server responsiveness

    Three independent signals, because they fail for different reasons:

    - **GPU sampler gaps** — the sampler fires on its own timer, so a gap means
      the *server process* stalled (GIL blocked, swap, a long CUDA sync). This is
      the only signal that separates "we were stuck" from "nobody called us".
      Unavailable on pre-GPU logs.
    - **Predict inter-arrival** — how often faro actually called. A gap here with
      an unbroken GPU sampler puts the stall upstream of us.
    - **Coverage** — missing `(fov, timestep)` pairs: predicts that never arrived
      or never completed.

    Both intervals are measured on the epoch clock (`t`), not on interpolated
    experiment hours, which are too coarse to difference.

    The log starts at `recv_epoch`, so time on the wire before a request landed is
    invisible here; if faro saw a timeout while all three signals look clean, the
    loss was in the network or in faro, and its own logs are needed to localize it.
    """)
    return


@app.cell
def _(gpu_h, np, plt, timing):
    # Inter-arrival deltas for both clocks. A healthy run is a flat line at the
    # median cadence, and every stall shows up as a spike above it.
    _fig, _axes = plt.subplots(2, 1, figsize=(11, 5.2))

    _ax0 = _axes[0]
    if gpu_h is None:
        _ax0.text(0.5, 0.5, "no gpu events in this log", ha="center", va="center",
                  transform=_ax0.transAxes, color="#888888")
    else:
        _gh = gpu_h.sort("t")
        _gd = np.diff(_gh["t"].to_numpy())          # elapsed seconds
        _gx = _gh["hours"].to_numpy()[1:]           # x position only
        _gmed = float(np.median(_gd))
        _ax0.plot(_gx, _gd, color="#4477aa", lw=0.7)
        _ax0.axhline(_gmed, color="black", ls="--", lw=1.0)
        _bad = _gd > 3 * _gmed  # 3x the timer period = a real process stall
        if _bad.any():
            _ax0.scatter(_gx[_bad], _gd[_bad], color="#cc3311", s=18, zorder=3)
        _ax0.set_yscale("log")
        _ax0.set_title(
            f"GPU sampler interval (median {_gmed:.1f}s, {int(_bad.sum())} gaps > 3x)"
        )
    _ax0.set_ylabel("seconds")

    _ax1 = _axes[1]
    _pt = timing.sort("t")
    _pd = np.diff(_pt["t"].to_numpy())
    _px = _pt["hours"].to_numpy()[1:]
    _pmed = float(np.median(_pd))
    _ax1.plot(_px, _pd, color="#ee7733", lw=0.7)
    _ax1.axhline(_pmed, color="black", ls="--", lw=1.0)
    _pbad = _pd > 3 * _pmed
    if _pbad.any():
        _ax1.scatter(_px[_pbad], _pd[_pbad], color="#cc3311", s=18, zorder=3)
    _ax1.set_yscale("log")
    _ax1.set_ylabel("seconds")
    _ax1.set_xlabel("time (h, real clock)")
    _ax1.set_title(
        f"Predict inter-arrival (median {_pmed:.1f}s, {int(_pbad.sum())} gaps > 3x)"
    )

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(gpu_h, mo, np, timing):
    # Coverage: every FOV should have been scored at every timestep in the range.
    _fovs = sorted(timing["fov"].unique().to_list())
    _ts = timing["timestep"].to_numpy()
    _lo, _hi = int(_ts.min()), int(_ts.max())
    _expected = {(_f, _t) for _f in _fovs for _t in range(_lo, _hi + 1)}
    _seen = set(zip(timing["fov"].to_list(), timing["timestep"].to_list()))
    _missing = sorted(_expected - _seen)

    _lines = []
    if gpu_h is None:
        _lines.append("- **GPU sampler:** not logged in this run.")
    else:
        _gd = np.diff(gpu_h.sort("t")["t"].to_numpy())
        _gmed = float(np.median(_gd))
        _ng = int((_gd > 3 * _gmed).sum())
        _lines.append(
            f"- **GPU sampler:** {'no process stalls — ' if _ng == 0 else ''}"
            f"{_ng} gaps > 3x the {_gmed:.1f}s period, worst {_gd.max():.1f}s."
        )
    _pd = np.diff(timing.sort("t")["t"].to_numpy())
    _lines.append(
        f"- **Predict arrivals:** median {np.median(_pd):.1f}s, worst {_pd.max():.1f}s, "
        f"{int((_pd > 3 * np.median(_pd)).sum())} gaps > 3x."
    )
    _lines.append(
        f"- **Coverage:** {len(_seen)}/{len(_expected)} (fov, timestep) pairs over "
        f"{len(_fovs)} FOVs × timesteps {_lo}–{_hi}"
        + (
            f" — **{len(_missing)} missing**, first: {_missing[:5]}."
            if _missing
            else " — complete."
        )
    )
    _lat = timing.drop_nulls("handler_s")
    if _lat.height:
        _ms = _lat["handler_s"].to_numpy() * 1000.0
        # The cadence is the real deadline: handler time as a share of the budget.
        _budget = float(np.median(_pd)) * 1000.0
        _lines.append(
            f"- **Handler latency:** p99 {np.percentile(_ms, 99):.0f} ms, max "
            f"{_ms.max():.0f} ms — {_ms.max() / _budget:.1%} of the {_budget / 1000:.0f}s "
            "inter-arrival budget at worst."
        )
        _lw = _lat.drop_nulls("lock_wait_s")["lock_wait_s"].to_numpy() * 1000.0
        if _lw.size:
            _lines.append(
                f"- **Lock contention:** max wait {_lw.max():.1f} ms"
                + (" — requests never queued." if _lw.max() < 1.0 else ".")
            )
    else:
        _lines.append("- **Handler latency:** not logged in this run.")

    mo.md("### Verdict\n\n" + "\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
