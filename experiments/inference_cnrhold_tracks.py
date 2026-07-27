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
    EXPO_LEVELS = [0.0, 200.0, 400.0, 600.0, 800.0]
    EXPO_NORM = Normalize(0.0, 800.0)
    EXPO_CMAP = plt.cm.YlOrRd
    return (
        EXPO_CMAP,
        EXPO_LEVELS,
        EXPO_NORM,
        FuncAnimation,
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
    `(fov, timestep, particle)`. Exposure takes 5 discrete levels: **0 / 200 / 400 /
    600 / 800 ms**.

    FOVs are **pooled by default** (or view one at a time); a `track_key` =
    `fov_particle` keeps track identities unique across FOVs. Plots are static
    matplotlib images, so pooling any-size experiment never overflows the output.
    """)
    return


@app.cell
def _(mo):
    # Experiment directory to load. Any serving run with per-FOV `*.parquet` track
    # files and a `*.jsonl` prediction log (found recursively) works.
    exp_dir_input = mo.ui.text(
        value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-07-16_InferenceCNRhold_12h_v7",
        label="Experiment directory (per-FOV parquets + a .jsonl log)",
        full_width=True,
    )
    exp_dir_input

    # \\izbkingston.unibe.ch\imaging.data\mic01-imaging\314lipczuk\2026-07-16_InferenceCNRhold_12h_v7
    return (exp_dir_input,)


@app.cell
def _(Path, exp_dir_input, json, mo, pl):
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

    _tracks = pl.concat(
        [
            pl.read_parquet(_f).with_columns(
                pl.lit(_f.stem).alias("source_file"),
            )
            for _f in _files
        ]
    )

    # Server log, one pass over three event kinds:
    #   `predict` — one per (fov, timestep), holding the scored cells and (new
    #               format only) a `timing` block for that inference call
    #   `gpu`     — a periodic device sample, absent in pre-v4 runs
    #   `startup` — engine/checkpoint info, logged once
    _srv_rows = []
    _timing_rows = []
    _gpu_rows = []
    startup = {}
    with open(_log) as _fh:
        for _line in _fh:
            _rec = json.loads(_line)
            _ev = _rec.get("event")
            if _ev == "startup":
                startup = _rec
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
        pl.DataFrame(_srv_rows)
        .unique(subset=["fov", "timestep", "particle"], keep="first")
        .with_columns(
            pl.col("fov").cast(pl.UInt16),
            pl.col("timestep").cast(pl.UInt32),
            pl.col("particle").cast(pl.UInt32),
        )
    )

    data = (
        _tracks.join(serving, on=["fov", "timestep", "particle"], how="left")
        .with_columns(
            (pl.col("time") / 3600.0).alias("hours"),
            # True light-on flag from the server exposure, not the mangled parquet
            # `stim` column.
            (pl.col("exposure_ms") > 0).alias("light_on"),
            # Track id unique across FOVs, so pooling doesn't merge same-id particles.
            (pl.col("fov").cast(pl.Utf8) + "_" + pl.col("particle").cast(pl.Utf8)).alias("track_key"),
        )
    )
    return data, gpu, startup, timing


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
def _(Path, json, policy, startup):
    # Per-FOV policy metadata. Prefers the policy `.toml` (authoritative for the
    # controller); falls back to the server startup record. Feeds three things: the
    # control metric/setpoint (next cell), the same-config FOV partitions in the
    # selector, and the per-FOV latency table.
    #
    #   objective.target_cnr is in the checkpoint's cnr units:
    #     cnr_mode="norm" -> cnr_median_norm (resting baseline == 1.0), e.g. 1.4
    #     cnr_mode="raw"  -> absolute cnr_median, e.g. 1.2
    #   The server logs the controlled signal (whichever it is) as `cnr_norm`, so
    #   the target line always lives on the `cnr_norm` readout at TARGET. For a raw
    #   model `cnr_norm == cnr_median`, so the target is meaningful on that readout
    #   too (TARGET_COLS in the next cell).
    # The policy file names the model `checkpoint`; the server log names it
    # `checkpoint_dir`. Read either.
    def _ckpt(_info):
        return Path(str((_info or {}).get("checkpoint_dir") or (_info or {}).get("checkpoint") or "?")).name

    def _obj_target(_info):
        _t = ((_info or {}).get("objective") or {}).get("target_cnr")
        return float(_t) if _t is not None else None

    def _mode_of(_info):
        # Explicit cnr_mode when the model loaded; else infer from the checkpoint
        # name (…_raw_cnr_… -> raw), which the stub startup / policy file still name.
        _m = (_info or {}).get("cnr_mode")
        if _m in ("norm", "raw"):
            return _m
        return "raw" if "raw" in _ckpt(_info).lower() else "norm"

    def _ctrl_label(_info):
        # Controller + its distinguishing params (e.g. staggered_mpc's cadence k).
        _c = (_info or {}).get("controller") or {}
        _name = _c.get("name") or _c.get("type")
        if not _name:
            return "?"
        _extra = ",".join(f"{_k}={_v}" for _k, _v in _c.items() if _k not in ("name", "type"))
        return f"{_name}({_extra})" if _extra else str(_name)

    def _sig(_info):
        # Identity of "same protocol/model/inference/controller": FOVs whose policy
        # matches on all of these pool together.
        _info = _info or {}
        return json.dumps(
            {
                "checkpoint": _ckpt(_info),
                "cnr_mode": _mode_of(_info),
                "model_type": _info.get("model_type"),
                "objective": _info.get("objective"),
                "controller": _info.get("controller"),
                "control_horizon": _info.get("control_horizon"),
            },
            sort_keys=True,
            default=str,
        )

    def _label(_info):
        _info = _info or {}
        _tgt = _obj_target(_info)
        _obj = (_info.get("objective") or {}).get("type", "?")
        _bits = [
            _mode_of(_info),
            f"{_obj}={_tgt:g}" if _tgt is not None else str(_obj),
            _ctrl_label(_info),
            _ckpt(_info)[:24],
        ]
        return " · ".join(_bits)

    if policy is not None:
        _fov_info = policy["fov"]
        _default_info = policy["default"]
    else:
        _pol = (startup or {}).get("policies", {}) or {}
        _default_info = _pol.get("default") or (startup or {}).get("info") or {}
        _fov_info = {int(_k): _v for _k, _v in (_pol.get("fov", {}) or {}).items()}

    FOV_META = {
        int(_k): {
            "checkpoint": _ckpt(_v),
            "mode": _mode_of(_v),
            "target": _obj_target(_v),
            "controller": _ctrl_label(_v),
            "sig": _sig(_v),
            "label": _label(_v),
        }
        for _k, _v in _fov_info.items()
    }
    DEFAULT_META = {"mode": _mode_of(_default_info), "target": _obj_target(_default_info)}

    # FOVs bucketed by identical policy signature — the "pool same-config" groups.
    FOV_GROUPS = {}
    for _f, _m in sorted(FOV_META.items()):
        FOV_GROUPS.setdefault(_m["sig"], {"label": _m["label"], "fovs": []})["fovs"].append(_f)

    {"n_groups": len(FOV_GROUPS), "groups": {g["label"]: g["fovs"] for g in FOV_GROUPS.values()}}
    return DEFAULT_META, FOV_GROUPS, FOV_META


@app.cell
def _(FOV_GROUPS, data, mo):
    _fovs = sorted(data["fov"].unique().to_list())
    # `None` = pool every FOV. A tuple = pool a same-config group. A bare int = one
    # FOV. Group options only appear when the run actually mixed configs (more than
    # one signature), otherwise "All (pooled)" already is the single group.
    _opts = {"All (pooled)": None}
    if len(FOV_GROUPS) > 1:
        for _g in FOV_GROUPS.values():
            _fl = _g["fovs"]
            _opts[f"Group: {_g['label']}  (FOVs {','.join(map(str, _fl))})"] = tuple(_fl)
    _opts.update({f"FOV {_f}": _f for _f in _fovs})

    fov_sel = mo.ui.dropdown(options=_opts, value="All (pooled)", label="FOV / partition")
    readout_sel = mo.ui.dropdown(
        options=["cnr_norm", "cnr", "cnr_median"],
        # `cnr_median` = the raw per-cell median CNR the model consumes; for a raw
        # model it equals the controlled signal `cnr_norm` (see the TARGET cell).
        value="cnr_median",
        label="Readout",
    )
    mo.hstack([fov_sel, readout_sel], justify="start")
    return fov_sel, readout_sel


@app.cell
def _(DEFAULT_META, FOV_META, data, fov_sel, re):
    # Control metric + setpoint for the CURRENT selection. A single FOV or a
    # same-config group has one policy, so its target/mode is unambiguous; when
    # everything is pooled, use the shared value if uniform, else the server
    # default policy, else the treatment label, else 1.4.
    _sel = fov_sel.value
    if _sel is None:
        _fovs = list(FOV_META.keys())
    elif isinstance(_sel, (list, tuple)):
        _fovs = list(_sel)
    else:
        _fovs = [_sel]
    _metas = [FOV_META[_f] for _f in _fovs if _f in FOV_META]

    def _uniform(_vals):
        _s = set(_vals)
        return next(iter(_s)) if len(_s) == 1 else None

    _tn = str(data["treatment_name"].mode()[0]) if data.height else ""
    _lbl = re.search(r"=\s*([0-9.]+)", _tn)
    _ut = _uniform([_m["target"] for _m in _metas if _m["target"] is not None])
    _um = _uniform([_m["mode"] for _m in _metas])
    TARGET = (
        _ut
        if _ut is not None
        else DEFAULT_META.get("target")
        if DEFAULT_META.get("target") is not None
        else (float(_lbl.group(1)) if _lbl else 1.4)
    )
    CNR_MODE = _um if _um is not None else (DEFAULT_META.get("mode") or "norm")
    # Readouts the target line is meaningful on. The controlled signal is always
    # `cnr_norm`; a raw model's `cnr_norm` equals the parquet `cnr_median`, so the
    # target holds there as well.
    TARGET_COLS = {"cnr_norm"} if CNR_MODE == "norm" else {"cnr_norm", "cnr_median"}
    return CNR_MODE, TARGET, TARGET_COLS


@app.cell
def _(data, pl):
    # Track length = number of frames a track is seen (dt = 1 min, so /60 = hours).
    # Many tracks are very short — tracking / segmentation / feature-extraction dropouts.
    track_len = (
        data.group_by(["fov", "particle", "track_key"])
        .agg(pl.len().alias("n_frames"))
        .with_columns((pl.col("n_frames") / 60.0).alias("length_h"))
        .sort(["fov", "particle"])
    )
    return (track_len,)


@app.cell
def _(mo):
    # Trajectory plots below only show tracks at least this long.
    min_len = mo.ui.slider(
        0.0, 12.0, value=10.0, step=0.5, label="Min track length (h)", full_width=True
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
    plot_df = fov_df.select(
        ["hours", "timestep", "track_key", "particle", "light_on", "exposure_ms", y_col]
    ).sort(["track_key", "timestep"])
    # Every-2nd-minute view for the dense all-track backgrounds (halves the point
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
    return expo_ts, fov_df, plot_df, plot_df_sparse, summary, y_col


@app.cell
def _(CNR_MODE, FOV_META, TARGET, fov_df, fov_sel, min_len, mo):
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
        _fovs = list(FOV_META.keys())
    elif isinstance(_sel, (list, tuple)):
        _scope = f"FOVs {','.join(map(str, _sel))} (pooled group)"
        _fovs = list(_sel)
    else:
        _scope = f"FOV {_sel}"
        _fovs = [_sel]
    # Controller(s) in scope — the experimental variable in a controller comparison.
    _ctrls = sorted({FOV_META[_f]["controller"] for _f in _fovs if _f in FOV_META})
    _ctrl_txt = _ctrls[0] if len(_ctrls) == 1 else ("mixed: " + ", ".join(_ctrls)) if _ctrls else "?"
    mo.md(
        f"**{_scope}** — {_n_tracks} tracks ≥ {min_len.value:.1f} h, "
        f"{_n_rows} rows, {_hmax:.1f} h. "
        f"Control: **{CNR_MODE}**-cnr held to **{TARGET:g}**, controller **{_ctrl_txt}**."
    )
    return


@app.cell
def _(
    CNR_MODE,
    TARGET,
    TARGET_COLS,
    fov_df,
    plot_df_sparse,
    plt,
    summary,
    y_col,
):
    # Spaghetti of every track (thin, faint) with the population mean overlaid bold.
    _fig, _ax = plt.subplots(figsize=(11, 4.8))
    for _key, _g in plot_df_sparse.group_by("track_key"):
        _ax.plot(_g["hours"].to_numpy(), _g[y_col].to_numpy(), color="#4477aa", alpha=0.15, lw=0.5)
    _ax.plot(
        summary["hours"].to_numpy(), summary["mean"].to_numpy(),
        color="#cc3311", lw=2.5, label="population mean",
    )
    if y_col in TARGET_COLS:  # controller setpoint, in this run's cnr units
        _ax.axhline(TARGET, color="black", ls="--", lw=1.3, label=f"target {TARGET:g} ({CNR_MODE})")
    _ax.set_xlabel("time (h)")
    _ax.set_ylabel(y_col)
    _ax.set_title(f"{y_col} tracks (bold = population mean, n={fov_df['track_key'].n_unique()})")
    _ax.legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(EXPO_CMAP, EXPO_LEVELS, expo_ts, np, pl, plt):
    # Population control signal: fraction of cells at each exposure level over time.
    _tot = expo_ts.group_by("timestep").agg(pl.col("n").sum().alias("tot"))
    _m = expo_ts.join(_tot, on="timestep").with_columns((pl.col("n") / pl.col("tot")).alias("frac"))
    _ts = sorted(_m["timestep"].unique().to_list())
    _hours = np.asarray(_ts) / 60.0
    _series = []
    for _lvl in EXPO_LEVELS:
        _s = _m.filter(pl.col("exposure_ms") == _lvl)
        _map = dict(zip(_s["timestep"].to_list(), _s["frac"].to_list()))
        _series.append([_map.get(_t, 0.0) for _t in _ts])

    _fig, _ax = plt.subplots(figsize=(11, 2.8))
    _colors = EXPO_CMAP(np.linspace(0.15, 0.95, len(EXPO_LEVELS)))
    _ax.stackplot(_hours, *_series, labels=[f"{int(_l)} ms" for _l in EXPO_LEVELS], colors=_colors)
    _ax.set_xlim(float(_hours.min()), float(_hours.max()))
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("time (h)")
    _ax.set_ylabel("fraction of cells")
    _ax.set_title("Commanded stim exposure across cells over time")
    _ax.legend(loc="upper right", fontsize=7, ncol=5)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(TARGET, fov_df, mo, pl):
    # Order tracks by L2 distance of the controlled signal (`cnr_norm`, whatever
    # metric the server drove) to the setpoint (RMSE, so it's comparable across
    # track lengths); nearest-to-target first. Total exposure is
    # the light dose the controller spent on that cell over its whole trajectory —
    # two tracks can sit equally close to target for very different effort.
    _l2 = (
        fov_df.group_by("track_key")
        .agg(
            ((pl.col("cnr_norm") - TARGET) ** 2).mean().sqrt().alias("rmse"),
            (pl.col("exposure_ms").sum() / 1000.0).alias("total_exposure_s"),
        )
        .drop_nulls("rmse")
        .sort("rmse")
    )
    mo.stop(_l2.is_empty(), mo.md("_No tracks pass the length cutoff._"))
    _labels = [
        f"{k}  ·  L2/√n {r:.3f}  ·  {e:.1f} s light"
        for k, r, e in zip(_l2["track_key"], _l2["rmse"], _l2["total_exposure_s"])
    ]
    particle_sel = mo.ui.dropdown(
        options=dict(zip(_labels, _l2["track_key"].to_list())),
        value=_labels[0],
        label="Highlight track (nearest target first)",
    )
    particle_sel
    return (particle_sel,)


@app.cell
def _(
    CNR_MODE,
    EXPO_CMAP,
    EXPO_NORM,
    TARGET,
    TARGET_COLS,
    particle_sel,
    pl,
    plot_df,
    plot_df_sparse,
    plt,
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
    if y_col in TARGET_COLS:  # controller setpoint, in this run's cnr units
        _ax.axhline(TARGET, color="black", ls="--", lw=1.3, label=f"target {TARGET:g} ({CNR_MODE})")
        _ax.legend(loc="upper right")

    _n_on = int(_one.filter(pl.col("light_on")).height)
    _dose_s = float(_one["exposure_ms"].sum() or 0.0) / 1000.0
    _ax.set_xlabel("time (h)")
    _ax.set_ylabel(y_col)
    _ax.set_title(
        f"Track {particle_sel.value} ({_one.height} pts, {_n_on} light-on, "
        f"{_dose_s:.1f} s total light) — color = commanded exposure"
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(
    CNR_MODE,
    TARGET,
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
        x=alt.X("hours:Q", title="time (h)"),
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
    if y_col in TARGET_COLS:
        # Same field name as the y encoding so the layered y-scale is shared.
        _tgt = pl.DataFrame({y_col: [float(TARGET)]})
        _layers.append(
            alt.Chart(_tgt).mark_rule(color="black", strokeDash=[6, 3]).encode(y=f"{y_col}:Q")
        )
        _tsuffix = f" · target {TARGET:g} ({CNR_MODE})"
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
def _(fov_df, particle_sel, pl, plot_df, plt):
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

    _fig, _ax = plt.subplots(figsize=(11, 3.6))
    _axr = _ax.twinx()
    # Right axis first, so the cumulative curve draws on top of the exposure bars.
    _axr.bar(_h, _one["exposure_ms"].to_numpy(), width=1.0 / 60.0, color="#ee7733",
             alpha=0.30, align="center", label="per-frame exposure")
    _axr.set_ylabel("per-frame exposure (ms)", color="#ee7733")
    _axr.set_ylim(0, 850)

    _ax.plot(_ref["hours"].to_numpy(), _ref["med_cum_s"].to_numpy(),
             color="#999999", ls="--", lw=1.5, label="population median")
    _ax.plot(_h, _cum, color="#117733", lw=2.2, label="this track")
    _ax.fill_between(_h, 0.0, _cum, color="#117733", alpha=0.12)
    _ax.set_zorder(_axr.get_zorder() + 1)
    _ax.patch.set_visible(False)
    _ax.set_xlabel("time (h)")
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
    # Timepoint to slice, in minutes (= timestep, dt = 1 min).
    _ts = fov_df["timestep"]
    tp_sel = mo.ui.slider(
        int(_ts.min()), int(_ts.max()), value=int(_ts.max() // 2), step=1,
        label="Timepoint (min)", full_width=True,
    )
    tp_sel
    return (tp_sel,)


@app.cell
def _(
    CNR_MODE,
    TARGET,
    TARGET_COLS,
    fov_df,
    gaussian_kde,
    mo,
    np,
    pl,
    plt,
    tp_sel,
    y_col,
):
    _at = fov_df.filter(pl.col("timestep") == tp_sel.value)[y_col].drop_nulls().to_numpy()
    _t0 = fov_df.filter(pl.col("timestep") == int(fov_df["timestep"].min()))[y_col]
    _t0 = _t0.drop_nulls().to_numpy()
    mo.stop(_at.size < 2, mo.md(f"_Fewer than 2 cells at minute {tp_sel.value}._"))

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
        _ax.axvline(TARGET, color="black", ls="--", lw=1.3, label=f"target {TARGET:g} ({CNR_MODE})")
    _ax.set_xlabel(y_col)
    _ax.set_ylabel("density")
    _ax.set_title(
        f"{y_col} across all cells at minute {tp_sel.value} "
        f"({tp_sel.value / 60:.2f} h) — IQR {np.percentile(_at, 25):.2f}–"
        f"{np.percentile(_at, 75):.2f}"
    )
    _ax.legend(fontsize=8)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(CNR_MODE, TARGET, TARGET_COLS, fov_df, np, plt, y_col):
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

    _fig, _ax = plt.subplots(figsize=(11, 3.8))
    # pcolormesh with shading="flat" wants the full edge arrays, one longer than
    # each axis of C — passing bin centres for x silently mismatches.
    _im = _ax.pcolormesh(_xe / 60.0, _ye, _H.T, cmap="magma", shading="flat")
    _fig.colorbar(_im, ax=_ax, label="fraction of cells")
    _ax.plot(
        (_xe[:-1] + 0.5) / 60.0,
        [np.median(_v[_ts == _t]) if (_ts == _t).any() else np.nan for _t in _xe[:-1] + 0.5],
        color="#66ccee", lw=1.2, label="median",
    )
    if y_col in TARGET_COLS:
        _ax.axhline(TARGET, color="white", ls="--", lw=1.2, label=f"target {TARGET:g} ({CNR_MODE})")
    _ax.set_xlabel("time (h)")
    _ax.set_ylabel(y_col)
    _ax.set_title(f"{y_col} density over time (0.5–99.5 percentile range)")
    _ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.gca()
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
    # hours using the (t, timestep) pairs the predict events provide. Stays None
    # on pre-GPU logs; the plots below stop, the health check below does not.
    #
    # One point per timestep: a timestep's FOVs arrive at different `t` but share
    # one `hours`, and feeding those duplicates to np.interp creates flat segments
    # (samples inside one would come out with zero elapsed time between them).
    _pt = (
        timing.group_by("timestep")
        .agg(pl.col("t").min(), pl.col("hours").first())
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
    _ax2.set_xlabel("time (h)")
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
def _(FOV_META, mo, np, pl, timing):
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
                "controller": _meta.get("controller", "?"),
                "model": _meta.get("checkpoint", "?"),
                "mode": _meta.get("mode", "?"),
                "target": _meta.get("target"),
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
    _ax1.set_xlabel("time (h)")
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
