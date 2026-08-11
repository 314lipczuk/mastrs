import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import tomllib
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    from matplotlib.colors import Normalize
    from scipy import signal
    from scipy.optimize import  curve_fit
    from scipy.stats import chi2, spearmanr

    from optoerk.core.experiment import ExperimentTracker, load_experiment
    from optoerk.core.utils import materials_path, results_write_path
    from optoerk.serving.calibration import FluenceCalibration

    # Every pooled plot is a static matplotlib raster, so the notebook output size
    # is independent of the number of tracks — same convention as
    # `inference_cnrhold_tracks.py`. Altair is reserved for single-cell views and
    # is deliberately not imported here: nothing below is per-cell interactive.
    DT_MIN = 1.0                      # one frame = one minute
    EXPO_LEVELS = [0.0, 200.0, 400.0, 600.0, 800.0]
    EXPO_NORM = Normalize(0.0, 800.0)
    EXPO_CMAP = plt.cm.YlOrRd
    ARM_COLORS = {
        "constant_dose": "#4477aa",
        "sequence_mpc": "#ee7733",
        "staggered_mpc(k=4)": "#117733",
    }
    return (
        ARM_COLORS,
        DT_MIN,
        ExperimentTracker,
        FluenceCalibration,
        Path,
        chi2,
        curve_fit,
        datetime,
        hashlib,
        json,
        load_experiment,
        materials_path,
        mo,
        np,
        pl,
        plt,
        results_write_path,
        signal,
        spearmanr,
        tomllib,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Tier 0 diagnostics

    Six offline diagnostics that decide the **arm design of the next wet experiment**.
    No microscope time; every input already exists on disk.

    | # | question | decides |
    |---|---|---|
    | **D1** | does the full mixture cost change the chosen action? | keep `n_gaussians=3` or drop to one |
    | **D2** | where does predicted response saturate? | dose-ladder re-gridding; Niesen contamination |
    | **D3** | what causes the wobble around target? | arm 4 = long horizon / move penalty / neither |
    | **D4** | where does the deployed dose sit in the training `u_t`? | is the frozen normalization compromised |
    | **D5** | is the 800 ms pile-up saturation or chatter? | raise power, or don't bother with more levels |
    | **D6** | how fast does CNR fall in the dark? | reference period for the next experiment |

    Each diagnostic ends in an explicit **verdict** with a stated decision rule.
    The final cell writes `tier0_summary.json` through the experiment class — that
    file, not the plots, is the deliverable. The plots are supporting evidence.

    **Fail-loud contract.** A missing checkpoint / policy / run directory makes the
    affected diagnostic emit `verdict: "unavailable"` with a reason string. It never
    produces a silently empty plot, and it never takes down the diagnostics that do
    not need that input (D3/D4/D5 are pure data and need no GPU at all).
    """)
    return


@app.cell
def _(mo):
    run_dir_input = mo.ui.text(
        value="/Volumes/imaging.data/mic01-imaging/314lipczuk/2026-07-16_InferenceCNRhold_12h_v7",
        label="Serving-run directory (`tracks/*.parquet` + a `*.jsonl` log)",
        full_width=True,
    )
    policy_file_input = mo.ui.text(
        value="policy_9fov_raw.toml",
        label="Policy `.toml` (authoritative for the per-FOV controller)",
        full_width=True,
    )
    # One checkpoint bundle per line. The first line that matches the run's served
    # checkpoint is used for D1/D6; D2 sweeps every line it can load.
    ckpt_input = mo.ui.text_area(
        value=(
            "/Volumes/imaging.data/mic01-imaging/314lipczuk/seq2scal_history_raw_cnr_2026-07-19_12.34.57\n"
            "/Volumes/imaging.data/mic01-imaging/314lipczuk/seq2scal_history_raw_cnr_multilen_2026-07-23_14.49.54"
        ),
        label="Checkpoint bundles (one path per line)",
        full_width=True,
        rows=4,
    )
    device_input = mo.ui.dropdown(
        options=["cpu", "cuda", "mps"], value="cpu", label="Torch device"
    )
    n_dec_d1 = mo.ui.slider(
        200, 4000, value=2000, step=100, label="D1 decisions to replay"
    )
    n_dec_d2 = mo.ui.slider(
        100, 2000, value=500, step=50, label="D2 cell/timepoint pairs"
    )
    mo.vstack(
        [
            run_dir_input,
            policy_file_input,
            ckpt_input,
            mo.hstack([device_input, n_dec_d1, n_dec_d2], justify="start"),
        ]
    )
    return (
        ckpt_input,
        device_input,
        n_dec_d1,
        n_dec_d2,
        policy_file_input,
        run_dir_input,
    )


@app.cell
def _(Path, mo, run_dir_input):
    # Resolve the run's three artifacts up front. A missing run directory is fatal
    # for the whole notebook (every diagnostic reads it), so this one stops hard.
    RUN_DIR = Path(run_dir_input.value.strip())
    TRACKS_DIR = RUN_DIR / "tracks"
    _logs = sorted(RUN_DIR.glob("*.jsonl"))
    _tps = sorted(TRACKS_DIR.glob("*.parquet")) if TRACKS_DIR.is_dir() else []
    mo.stop(
        not _logs or not _tps,
        mo.md(
            f"**Not a serving run directory:** `{RUN_DIR}`\n\n"
            f"Needs `tracks/*.parquet` (found {len(_tps)}) and a `*.jsonl` server "
            f"log (found {len(_logs)})."
        ),
    )
    LOG_PATH = _logs[0]
    TRACK_FILES = _tps
    mo.md(
        f"**Run:** `{RUN_DIR.name}` — {len(TRACK_FILES)} FOV parquets, log "
        f"`{LOG_PATH.name}` ({LOG_PATH.stat().st_size / 1e6:.0f} MB)."
    )
    return LOG_PATH, RUN_DIR, TRACK_FILES


@app.cell
def _(LOG_PATH, json, pl):
    # The server log is the authoritative record of what the model actually
    # consumed and commanded. Every `predict` cell record carries the full
    # 5-channel input the encoder saw that frame — raw_cnr, u_t_in (the fluence
    # persisted from the previous frame's applied dose), the two crowding channels
    # and the optoRTK value that was fed — plus the exposure that came out. The
    # parquet's `stim_exposure` is mangled by faro and is never used.
    _srv_rows = []
    _startup = {}
    with open(LOG_PATH, encoding="utf-8") as _fh:
        for _line in _fh:
            try:
                _rec = json.loads(_line)
            except json.JSONDecodeError:
                continue  # a torn last line after a crash is expected
            _ev = _rec.get("event")
            if _ev == "startup":
                # A run can log twice: once as a stub before the checkpoint loads,
                # then again as the real engine. Keep the last (real) one.
                if _rec.get("model_loaded") or not _startup:
                    _startup = _rec
                continue
            if _ev != "predict":
                continue
            for _c in _rec["cells"]:
                _srv_rows.append(
                    {
                        "fov": _rec["fov"],
                        "timestep": _rec["timestep"],
                        "particle": _c["particle"],
                        "raw_cnr": _c.get("raw_cnr"),
                        "cnr_norm": _c.get("cnr_norm"),
                        "fov_density": _c.get("fov_density"),
                        "n_cells_200px": _c.get("n_cells_200px"),
                        "u_t_in": _c.get("u_t_in"),
                        "optortk_expr": _c.get("optortk_expr"),
                        "n_frames_seen": _c.get("n_frames_seen"),
                        "exposure_ms": _c.get("exposure_ms"),
                        "fluence_out": _c.get("fluence_out"),
                        "dark": _c.get("dark"),
                    }
                )

    startup = _startup
    serving = (
        pl.DataFrame(_srv_rows)
        .unique(subset=["fov", "timestep", "particle"], keep="first")
        .with_columns(
            pl.col("fov").cast(pl.UInt16),
            pl.col("timestep").cast(pl.UInt32),
            pl.col("particle").cast(pl.UInt32),
            (
                pl.col("fov").cast(pl.Utf8) + "_" + pl.col("particle").cast(pl.Utf8)
            ).alias("track_key"),
        )
        .sort(["track_key", "timestep"])
    )
    serving.head()
    return serving, startup


@app.cell
def _(TRACK_FILES, pl, serving):
    # Per-FOV track parquets joined to the server log on (fov, timestep, particle).
    # `cnr_median` is the readout D3/D5/D6 analyse; `exposure_ms` / `fluence_out`
    # come from the log side, never from the parquet.
    _tracks = pl.concat([pl.read_parquet(_f) for _f in TRACK_FILES])
    data = (
        _tracks.join(
            serving.drop("cnr_norm"), on=["fov", "timestep", "particle"], how="left"
        )
        .with_columns(
            (pl.col("time") / 3600.0).alias("hours"),
            (pl.col("exposure_ms") > 0).alias("light_on"),
            (
                pl.col("fov").cast(pl.Utf8) + "_" + pl.col("particle").cast(pl.Utf8)
            ).alias("track_key"),
        )
        .sort(["track_key", "timestep"])
    )
    JOIN_FRAC = float(data["exposure_ms"].is_not_null().mean())
    data.select(
        pl.len().alias("rows"),
        pl.col("track_key").n_unique().alias("tracks"),
        pl.col("hours").max().alias("hours"),
    )
    return JOIN_FRAC, data


@app.cell
def _(JOIN_FRAC, Path, mo, policy_file_input, startup, tomllib):
    # Per-FOV controller. The policy `.toml` is authoritative: the run's own
    # startup record is written before the per-FOV assignment is meaningful in a
    # stub run, and the controller is the experimental variable here.
    _p = Path(policy_file_input.value.strip())
    POLICY_OK = _p.is_file()
    POLICY_REASON = "" if POLICY_OK else f"policy file not found: {_p}"

    _raw = {}
    if POLICY_OK:
        with open(_p, "rb") as _fh:
            _raw = tomllib.load(_fh)
    _default = _raw.get("default", {}) or {}
    _fovs = {int(_k): {**_default, **_v} for _k, _v in (_raw.get("fov", {}) or {}).items()}

    def _ctrl_label(_info):
        _c = (_info or {}).get("controller") or {}
        _name = _c.get("name") or _c.get("type")
        if not _name:
            return "?"
        _extra = ",".join(f"{_k}={_v}" for _k, _v in _c.items() if _k not in ("name", "type"))
        return f"{_name}({_extra})" if _extra else str(_name)

    FOV_ARM = {_f: _ctrl_label(_i) for _f, _i in _fovs.items()}
    ARMS = sorted(set(FOV_ARM.values()))

    # Setpoint + horizon: the policy file, falling back to the server's own
    # resolved startup info.
    _sinfo = (startup or {}).get("info", {}) or {}
    TARGET = float(
        (_default.get("objective") or {}).get("target_cnr")
        or (_sinfo.get("objective") or {}).get("target_cnr")
        or 1.2
    )
    CONTROL_HORIZON = int(
        _default.get("control_horizon") or _sinfo.get("control_horizon") or 8
    )
    SERVED_CKPT = str(
        _sinfo.get("checkpoint_dir") or _default.get("checkpoint") or ""
    )
    # The dose grid the run could actually command.
    RUN_LEVELS = [
        float(x)
        for x in ((_sinfo.get("controller") or {}).get("levels_ms") or [0, 200, 400, 600, 800])
    ]

    mo.md(
        f"""
    **Policy:** {"`" + str(_p) + "`" if POLICY_OK else "**MISSING** — " + POLICY_REASON}

    - target CNR **{TARGET:g}** (raw `cnr_median`), control horizon **H = {CONTROL_HORIZON}**
    - served checkpoint: `{Path(SERVED_CKPT).name or "?"}`
    - dose grid: {", ".join(f"{x:g}" for x in RUN_LEVELS)} ms
    - arms: {", ".join(f"**{a}** (FOV {','.join(str(f) for f, v in sorted(FOV_ARM.items()) if v == a)})" for a in ARMS) or "_none — no policy file_"}
    - log↔parquet join coverage: **{JOIN_FRAC:.1%}** of parquet rows have a server record
    """
    )
    return (
        CONTROL_HORIZON,
        FOV_ARM,
        POLICY_OK,
        POLICY_REASON,
        RUN_LEVELS,
        SERVED_CKPT,
        TARGET,
    )


@app.cell
def _(FOV_ARM, data, mo, pl):
    # Two track sets. The >=10 h set is the headline population, but it is
    # survivorship-selected: a cell only survives 10 h of tracking if segmentation
    # never lost it. D3 and D5 are therefore also computed on the >=4 h set and the
    # verdicts compared — if they disagree, the >=10 h result is a property of the
    # surviving subpopulation, not of the biology.
    track_len = (
        data.group_by("track_key")
        .agg(
            pl.len().alias("n_frames"),
            pl.col("fov").first(),
        )
        .with_columns((pl.col("n_frames") / 60.0).alias("length_h"))
    )
    KEYS_10H = track_len.filter(pl.col("length_h") >= 10.0)["track_key"].to_list()
    KEYS_4H = track_len.filter(pl.col("length_h") >= 4.0)["track_key"].to_list()

    # Arm label per row, so every downstream split is one column away.
    data_armed = data.with_columns(
        pl.col("fov")
        .cast(pl.Int64)
        .replace_strict(
            {int(k): v for k, v in FOV_ARM.items()}, default="unknown"
        )
        .alias("arm")
    )
    mo.md(
        f"**Track sets:** {len(KEYS_10H)} tracks ≥ 10 h, {len(KEYS_4H)} tracks ≥ 4 h "
        f"(of {track_len.height} total)."
    )
    return KEYS_10H, KEYS_4H, data_armed


@app.cell
def _(ExperimentTracker, Path, plt, results_write_path):
    # Artifact management goes through the repo's experiment class, as with every
    # other experiment: it owns the (timestamped) directory and the `figures/`
    # convention. There is no trained model here, so `save_final` does not apply —
    # figures and the summary JSON are written into the directory it registers.
    _tracker = ExperimentTracker(
        directory=str(Path(results_write_path()) / "tier0_diagnostics"),
        name="tier0_diagnostics",
        model_config={},
        training_config={"kind": "offline_diagnostics", "trains_nothing": True},
    )
    EXP_DIR = Path(_tracker.register_start())
    (EXP_DIR / "figures").mkdir(exist_ok=True)

    # Encoder states are expensive to rebuild and depend only on (checkpoint,
    # decision points), so they are cached OUTSIDE the per-run experiment dir and
    # survive across sessions.
    STATE_CACHE = Path(results_write_path()) / "tier0_state_cache"
    STATE_CACHE.mkdir(parents=True, exist_ok=True)

    def save_fig(name, fig):
        """Persist a figure into the experiment bundle and return it for display."""
        fig.savefig(EXP_DIR / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
        return fig

    def new_fig(*args, **kwargs):
        return plt.subplots(*args, **kwargs)

    EXP_DIR
    return EXP_DIR, STATE_CACHE, new_fig, save_fig


@app.cell
def _(mo):
    mo.md("""
    ## Shared model machinery

    D1, D2 and D6 need the encoder state the server carried for a cell at a given
    frame. That state is exactly reproducible: the log records the 5 input channels
    the encoder consumed at every frame, so replaying a cell's prefix through
    `model.encoder.lstm` reconstructs `(h, c)` bit-for-bit up to float ordering.

    Two things make this **serve-faithful** rather than merely plausible:

    1. `u_t` is taken from the log's `u_t_in` (the fluence persisted from the
       previously *applied* dose), not recomputed from the commanded exposure.
    2. `optortk_expr` is taken from the log too, which means the replay feeds the
       same population-mean constant the server fed online — **not** the true
       per-cell expression rank, which the server never had.

    Reconstructed states are cached to disk keyed by
    `(checkpoint, decision-point set)`; re-encoding thousands of full prefixes is
    the expensive part of this notebook.
    """)
    return


@app.cell
def _(Path, ckpt_input, device_input, load_experiment, mo, torch):
    # Checkpoints are loaded LAZILY — the data-only diagnostics (D3/D4/D5) run
    # with no model and no GPU. `get_model` memoizes within the session.
    DEVICE = torch.device(device_input.value)
    _paths = [p.strip() for p in ckpt_input.value.splitlines() if p.strip()]
    CKPT_SPECS = [
        {"path": p, "name": Path(p).name, "ok": Path(p).is_dir(), }
        for p in _paths
    ]
    for _s in CKPT_SPECS:
        _s["reason"] = "" if _s["ok"] else f"checkpoint directory not found: {_s['path']}"

    _MODELS = {}

    def get_model(path):
        """Load (and cache) a bundle's model in eval mode on DEVICE."""
        if path not in _MODELS:
            _bundle = load_experiment(path)
            _m = _bundle.reconstruct_model().to(DEVICE).eval()
            _MODELS[path] = _m
        return _MODELS[path]

    mo.md(
        "**Checkpoints**\n\n"
        + "\n".join(
            f"- {'✅' if s['ok'] else '❌'} `{s['name']}`" + ("" if s["ok"] else f" — {s['reason']}")
            for s in CKPT_SPECS
        )
        + f"\n\nDevice: **{DEVICE}**"
    )
    return CKPT_SPECS, DEVICE, get_model


@app.cell
def _(CKPT_SPECS, SERVED_CKPT, get_model, mo, np, pl):
    # Per-checkpoint metadata, read from the bundle's own config: the horizon it can
    # legally be rolled to (`future_len`), the frozen norm stats it was stamped with,
    # and whether the Niesen high-dose experiment was in its training bundle.
    CKPT_META = []
    for _s in CKPT_SPECS:
        if not _s["ok"]:
            CKPT_META.append({**_s})
            continue
        _cfg = get_model(_s["path"]).cfg
        CKPT_META.append(
            {
                **_s,
                "future_len": int(_cfg.future_len),
                "n_gaussians": int(_cfg.n_gaussians),
                "cnr_mode": _cfg.cnr_mode,
                "data_source": _cfg.data_source,
                "with_niesen": "niesen" in str(_cfg.data_source).lower(),
                "channels": list(_cfg.norm_channels),
                "norm_mean": list(_cfg.norm_mean),
                "norm_std": list(_cfg.norm_std),
            }
        )

    # The checkpoint the run was actually served with — D1 and D6 must use it, or
    # they describe a controller that never ran.
    _served = [m for m in CKPT_META if m["ok"] and m["name"] == SERVED_CKPT.rstrip("/").split("/")[-1]]
    SERVED_META = _served[0] if _served else next((m for m in CKPT_META if m["ok"]), None)

    mo.vstack(
        [
            pl.DataFrame(
                [
                    {
                        "checkpoint": m["name"],
                        "future_len": m.get("future_len"),
                        "K": m.get("n_gaussians"),
                        "cnr_mode": m.get("cnr_mode"),
                        "data_source": m.get("data_source"),
                        "niesen": m.get("with_niesen"),
                    }
                    for m in CKPT_META
                ]
            ),
            mo.md(
                f"Served checkpoint for D1/D6: **`{SERVED_META['name']}`**"
                if SERVED_META
                else "**No loadable checkpoint** — D1, D2 and D6 will report `unavailable`."
            ),
            mo.md(
                "⚠️ The served checkpoint's `norm_mean`/`norm_std` are the frozen "
                "stats D4 must use: "
                + (
                    f"`u_t` mean **{np.asarray(SERVED_META['norm_mean'])[1]:.4f}**, "
                    f"std **{np.asarray(SERVED_META['norm_std'])[1]:.4f}**."
                    if SERVED_META
                    else "unavailable."
                )
            ),
        ]
    )
    return CKPT_META, SERVED_META


@app.cell
def _(np, serving):
    # Per-track channel store, in the cell's OWN frame order (the encoder advances
    # once per frame the cell was seen, so gaps do not advance it — the same
    # semantics as the live `CellState`).
    _cols = ["raw_cnr", "u_t_in", "fov_density", "n_cells_200px", "optortk_expr"]
    _by = (
        serving.select(["track_key", "timestep", *_cols])
        .drop_nulls()
        .sort(["track_key", "timestep"])
        .partition_by("track_key", as_dict=True, include_key=False)
    )
    CHAN = {}
    for _k, _df in _by.items():
        _key = _k[0] if isinstance(_k, tuple) else _k
        CHAN[_key] = {
            "ts": _df["timestep"].to_numpy().astype(np.int64),
            "cnr": _df["raw_cnr"].to_numpy().astype(np.float32),
            "u_t": _df["u_t_in"].to_numpy().astype(np.float32),
            "fov_density": _df["fov_density"].to_numpy().astype(np.float32),
            "n_cells_200px": _df["n_cells_200px"].to_numpy().astype(np.float32),
            "optortk_expr": _df["optortk_expr"].to_numpy().astype(np.float32),
        }

    # Sanity check on the serve-faithfulness claim: the optoRTK channel really was
    # a single constant online.
    _expr = np.concatenate([v["optortk_expr"] for v in CHAN.values()])
    OPTORTK_FED = float(np.median(_expr))
    OPTORTK_CONSTANT = bool(np.ptp(_expr) < 1e-6)
    len(CHAN), OPTORTK_FED, OPTORTK_CONSTANT
    return CHAN, OPTORTK_CONSTANT, OPTORTK_FED


@app.cell
def _(CHAN, DEVICE, Path, STATE_CACHE, hashlib, np, torch):
    def _std_matrix(model, key):
        """Standardized (T, C) input matrix for one track, in the model's channel order."""
        mean = np.asarray(model.cfg.norm_mean, np.float32)
        std = np.asarray(model.cfg.norm_std, np.float32)
        src = CHAN[key]
        cols = []
        for i, name in enumerate(model.cfg.norm_channels):
            if name not in src:
                raise KeyError(
                    f"the server log carries no channel {name!r} required by this "
                    f"checkpoint (channels={list(model.cfg.norm_channels)})"
                )
            cols.append(src[name])
        X = np.stack(cols, axis=-1)
        return ((X - mean) / std).astype(np.float32)

    @torch.no_grad()
    def reconstruct_states(model, wanted, batch_tracks=96):
        """`(h, c, cnr_fb)` for each requested `(track_key, timestep)` decision point.

        The state returned is the one the controller planned from: the encoder has
        already ingested that frame, and `cnr_fb` is that frame's standardized CNR.
        Tracks are advanced in a batch, one timestep at a time, so the cost is one
        LSTM call per frame index rather than one per (cell, frame).
        """
        want = {}
        for k, t in wanted:
            want.setdefault(k, set()).add(int(t))
        keys = sorted(want)
        L, Hh = model.cfg.num_layers, model.cfg.hidden_dim
        C = len(model.cfg.norm_channels)
        cnr_idx = list(model.cfg.norm_channels).index("cnr")

        out_keys, out_h, out_c, out_fb = [], [], [], []
        for s in range(0, len(keys), batch_tracks):
            chunk = keys[s : s + batch_tracks]
            mats = [_std_matrix(model, k) for k in chunk]
            tss = [CHAN[k]["ts"] for k in chunk]
            B, Tmax = len(chunk), max(m.shape[0] for m in mats)
            X = np.zeros((B, Tmax, C), np.float32)
            M = np.zeros((B, Tmax), bool)
            for i, m in enumerate(mats):
                X[i, : m.shape[0]] = m
                M[i, : m.shape[0]] = True
            # Which (row, step) pairs must be snapshotted.
            snap = {}
            for i, k in enumerate(chunk):
                for step, t in enumerate(tss[i]):
                    if int(t) in want[k]:
                        snap.setdefault(step, []).append((i, k, int(t)))

            Xt = torch.from_numpy(X).to(DEVICE)
            Mt = torch.from_numpy(M).to(DEVICE)
            h = torch.zeros(L, B, Hh, device=DEVICE)
            c = torch.zeros(L, B, Hh, device=DEVICE)
            for step in range(Tmax):
                xs = Xt[:, step]                                     # (B, C)
                _, (hn, cn) = model.encoder.lstm(xs.unsqueeze(1), (h, c))
                live = Mt[:, step].view(1, -1, 1)
                # A cell that is absent this frame keeps its carried state, exactly
                # as the live per-cell store does across a tracking dropout.
                h = torch.where(live, hn, h)
                c = torch.where(live, cn, c)
                for i, k, t in snap.get(step, ()):
                    out_keys.append((k, t))
                    out_h.append(h[:, i].detach().clone())
                    out_c.append(c[:, i].detach().clone())
                    out_fb.append(xs[i, cnr_idx].detach().clone())

        if not out_keys:
            empty = torch.zeros(L, 0, Hh, device=DEVICE)
            return out_keys, empty, empty, torch.zeros(0, 1, device=DEVICE)
        return (
            out_keys,
            torch.stack(out_h, dim=1),                               # (L, D, Hh)
            torch.stack(out_c, dim=1),
            torch.stack(out_fb).view(-1, 1),                         # (D, 1)
        )

    def cached_states(model, ckpt_name, wanted, tag):
        """`reconstruct_states` with an on-disk cache keyed by (checkpoint, points)."""
        digest = hashlib.sha1(
            (ckpt_name + "|" + tag + "|" + "|".join(f"{k}@{t}" for k, t in wanted)).encode()
        ).hexdigest()[:16]
        path = Path(STATE_CACHE) / f"{ckpt_name}__{tag}__{digest}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            keys = [(str(k), int(t)) for k, t in zip(z["track_key"], z["timestep"])]
            return (
                keys,
                torch.from_numpy(z["h"]).to(DEVICE),
                torch.from_numpy(z["c"]).to(DEVICE),
                torch.from_numpy(z["fb"]).to(DEVICE),
            )
        keys, h, c, fb = reconstruct_states(model, wanted)
        np.savez_compressed(
            path,
            track_key=np.array([k for k, _ in keys]),
            timestep=np.array([t for _, t in keys], np.int64),
            h=h.cpu().numpy(),
            c=c.cpu().numpy(),
            fb=fb.cpu().numpy(),
        )
        return keys, h, c, fb

    @torch.no_grad()
    def rollout_mdn(model, h, c, cnr_fb, fut):
        """Decoder rollout returning the FULL mixture `(pi, mu, sigma)`, each (B, F, K).

        Identical to `RealModelEngine.rollout` (free-running feedback on the mixture
        mean, same FiLM and per-step sigma-bias handling) except that it keeps the
        components instead of collapsing them — which is precisely what D1 needs.
        Outputs are in STANDARDIZED cnr units.
        """
        dh, dc = h, c
        pis, mus, sgs = [], [], []
        for i in range(fut.shape[1]):
            flu_i = fut[:, i, :]
            dec_in = torch.cat([cnr_fb, flu_i], dim=-1).unsqueeze(1)
            out, (dh, dc) = model.decoder(dec_in, (dh, dc))
            h_step = out[:, -1, :]
            if model.film_layer is not None:
                gamma, beta = model.film_layer(flu_i)
                if model.cfg.film == "output":
                    h_step = gamma * h_step + beta
                else:
                    dh = gamma.unsqueeze(0) * dh + beta.unsqueeze(0)
                    h_step = dh[-1]
            sigma_bias = (
                model.sigma_step_bias_param[i]
                if model.sigma_step_bias_param is not None
                else 0.0
            )
            feats = model.trunk(torch.cat([h_step, flu_i], dim=-1))
            pi, mu, sigma = model.head(feats, sigma_bias=sigma_bias)
            pis.append(pi)
            mus.append(mu)
            sgs.append(sigma)
            cnr_fb = (pi * mu).sum(dim=-1, keepdim=True)
        return torch.stack(pis, 1), torch.stack(mus, 1), torch.stack(sgs, 1)

    return cached_states, rollout_mdn


@app.cell
def _(FluenceCalibration, np):
    # The served configuration: Niesen DMD at 10% power. Same curve the training
    # `u_t` was built with, so exposures and training fluences live on one axis.
    CALIB = FluenceCalibration("niesen", 10.0)
    FLU_PER_MS = float(CALIB.fluence_per_ms)

    def ms_to_flu(ms):
        return np.asarray(ms, np.float64) * FLU_PER_MS

    CALIB.describe()
    return (FLU_PER_MS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## D1 — Mixture-cost replay

    **Question.** Does scoring plans under the full mixture instead of the mixture
    *mean* change which action the controller picks?

    $$J_\text{mean} = \frac{1}{H}\sum_h \Big(\textstyle\sum_k \pi_k \mu_k - r\Big)^2
    \qquad
    J_\text{mix} = \frac{1}{H}\sum_h \sum_k \pi_k \big[(\mu_k - r)^2 + \sigma_k^2\big]$$

    Both are scored on **one shared candidate set** (the 5 constant-dose plans plus
    128 seeded random dose sequences), so the comparison is paired and independent
    of CEM sampling noise. Costs are evaluated in **absolute CNR** — `mu` and
    `sigma` are de-standardized first, as `objectives.py` requires; getting that
    wrong silently changes the answer because the two costs weight `sigma`
    differently.

    **Decision rule.** `< 5%` of decisions change action → the mixture is decoration
    for control; retrain single-Gaussian and report the clean negative.
    `> 15%` → the mixture-aware cost becomes the default for every MPC arm in the
    next run. In between → report the number, keep L2-on-mean, revisit after the
    band objective.
    """)
    return


@app.cell
def _(KEYS_10H, TARGET, data_armed, n_dec_d1, np, pl):
    # Decision points for D1, stratified over (experiment third) x (above/below
    # target) so the answer is not dominated by one regime. A decision needs a
    # prefix, so cells are only sampled once the encoder has seen >= 10 frames.
    _d = (
        data_armed.filter(
            pl.col("track_key").is_in(KEYS_10H)
            & pl.col("exposure_ms").is_not_null()
            & (pl.col("n_frames_seen") >= 10)
        )
        .select(["track_key", "timestep", "hours", "cnr_median", "exposure_ms", "arm", "fov"])
        .drop_nulls()
    )
    _tmax = float(_d["timestep"].max())
    d1_pool = _d.with_columns(
        pl.when(pl.col("timestep") < _tmax / 3)
        .then(pl.lit("early"))
        .when(pl.col("timestep") < 2 * _tmax / 3)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("third"),
        (pl.col("cnr_median") >= TARGET).alias("above_target"),
    )

    _rng = np.random.default_rng(0)
    _per = max(1, n_dec_d1.value // 6)
    _picks = []
    for _third in ("early", "mid", "late"):
        for _above in (False, True):
            _s = d1_pool.filter(
                (pl.col("third") == _third) & (pl.col("above_target") == _above)
            )
            if _s.is_empty():
                continue
            _take = min(_per, _s.height)
            _idx = _rng.choice(_s.height, size=_take, replace=False)
            _picks.append(_s[np.sort(_idx)])
    d1_points = pl.concat(_picks) if _picks else d1_pool.head(0)
    d1_points.group_by(["third", "above_target"]).agg(pl.len().alias("n")).sort(
        ["third", "above_target"]
    )
    return (d1_points,)


@app.cell
def _(
    CONTROL_HORIZON,
    FLU_PER_MS,
    RUN_LEVELS,
    SERVED_META,
    TARGET,
    cached_states,
    d1_points,
    get_model,
    np,
    pl,
    rollout_mdn,
    torch,
):
    # --- D1 core -----------------------------------------------------------
    D1 = {"verdict": "unavailable", "reason": "", "numbers": {}}
    d1_df = None

    if SERVED_META is None:
        D1["reason"] = "no loadable checkpoint; D1 needs the served model's mixture head"
    elif d1_points.is_empty():
        D1["reason"] = "no decision points survived the >=10 h / >=10 frames-seen filter"
    else:
        _model = get_model(SERVED_META["path"])
        _H = min(CONTROL_HORIZON, int(SERVED_META["future_len"]))
        _mean = np.asarray(_model.cfg.norm_mean, np.float32)
        _std = np.asarray(_model.cfg.norm_std, np.float32)
        _cnr_m, _cnr_s = float(_mean[0]), float(_std[0])
        _flu_m, _flu_s = float(_mean[1]), float(_std[1])
        _levels = np.asarray(RUN_LEVELS, np.float64)
        _L = len(_levels)

        # Shared candidate set: the L constant-dose plans + 128 seeded sequences.
        # Identical for both costs and for every decision, so the comparison is
        # exactly paired. Exact CEM replay is unnecessary — we need argmin over a
        # shared set, not the run's own search trajectory.
        _rng = np.random.default_rng(1234)
        _const = np.tile(np.arange(_L)[:, None], (1, _H))
        _samp = _rng.integers(0, _L, size=(128, _H))
        _plan_idx = np.concatenate([_const, _samp], axis=0)              # (S, H)
        _S = _plan_idx.shape[0]
        _lvl_std = ((_levels * FLU_PER_MS) - _flu_m) / _flu_s            # (L,)
        _fut = torch.tensor(
            _lvl_std[_plan_idx], dtype=torch.float32                     # (S, H)
        )

        _wanted = list(zip(d1_points["track_key"].to_list(), d1_points["timestep"].to_list()))
        _keys, _h, _c, _fb = cached_states(
            _model, SERVED_META["name"], _wanted, f"d1_H{_H}"
        )
        _D = len(_keys)

        _u_mean, _u_mix, _cost_mean, _cost_mix = [], [], [], []
        _ent, _spread = [], []
        _chunk = 64
        for _s0 in range(0, _D, _chunk):
            _s1 = min(_D, _s0 + _chunk)
            _n = _s1 - _s0
            _hb = _h[:, _s0:_s1].repeat_interleave(_S, dim=1)
            _cb = _c[:, _s0:_s1].repeat_interleave(_S, dim=1)
            _fbb = _fb[_s0:_s1].repeat_interleave(_S, dim=0)
            _futb = _fut.to(_h.device).unsqueeze(0).expand(_n, _S, _H).reshape(_n * _S, _H, 1)
            _pi, _mu, _sg = rollout_mdn(_model, _hb, _cb, _fbb, _futb)
            # De-standardize BEFORE costing: objectives.py works in absolute CNR.
            _mu_a = _mu * _cnr_s + _cnr_m
            _sg_a = _sg * _cnr_s
            _pt = (_pi * _mu_a).sum(-1)                                  # (B, H)
            _jm = ((_pt - TARGET) ** 2).mean(-1)                         # mean-only cost
            _jx = (
                (_pi * ((_mu_a - TARGET) ** 2 + _sg_a ** 2)).sum(-1).mean(-1)
            )                                                            # mixture cost
            _jm = _jm.view(_n, _S)
            _jx = _jx.view(_n, _S)
            _am, _ax = _jm.argmin(1), _jx.argmin(1)
            _u_mean.append(_levels[_plan_idx[_am.cpu().numpy(), 0]])
            _u_mix.append(_levels[_plan_idx[_ax.cpu().numpy(), 0]])
            _cost_mean.append(_jm.gather(1, _am[:, None]).squeeze(1).cpu().numpy())
            _cost_mix.append(_jx.gather(1, _ax[:, None]).squeeze(1).cpu().numpy())
            # Mixture shape at the FIRST horizon step of the mean-cost winner —
            # the step whose action is actually applied.
            _pi_w = _pi.view(_n, _S, _H, -1)[torch.arange(_n), _am, 0]   # (n, K)
            _mu_w = _mu_a.view(_n, _S, _H, -1)[torch.arange(_n), _am, 0]
            _ent.append(
                (-(_pi_w.clamp_min(1e-12) * _pi_w.clamp_min(1e-12).log()).sum(-1))
                .cpu()
                .numpy()
            )
            _spread.append(
                (_mu_w.max(-1).values - _mu_w.min(-1).values).cpu().numpy()
            )

        d1_df = pl.DataFrame(
            {
                "track_key": [k for k, _ in _keys],
                "timestep": [t for _, t in _keys],
                "u_mean": np.concatenate(_u_mean),
                "u_mix": np.concatenate(_u_mix),
                "cost_mean": np.concatenate(_cost_mean),
                "cost_mix": np.concatenate(_cost_mix),
                "pi_entropy": np.concatenate(_ent),
                "mu_spread": np.concatenate(_spread),
            }
        ).join(
            d1_points.select(["track_key", "timestep", "third", "above_target", "arm", "hours", "cnr_median"]),
            on=["track_key", "timestep"],
            how="left",
        )
        d1_df = d1_df.with_columns(
            (pl.col("u_mix") != pl.col("u_mean")).alias("changed"),
            (pl.col("u_mix") - pl.col("u_mean")).abs().alias("abs_du"),
        )

        _frac = float(d1_df["changed"].mean())
        D1["numbers"] = {
            "n_decisions": int(d1_df.height),
            "horizon": _H,
            "n_candidate_plans": int(_S),
            "frac_action_changed": _frac,
            "mean_abs_du_when_changed_ms": (
                float(d1_df.filter(pl.col("changed"))["abs_du"].mean())
                if d1_df["changed"].any()
                else 0.0
            ),
            "median_pi_entropy": float(d1_df["pi_entropy"].median()),
            "median_mu_spread_cnr": float(d1_df["mu_spread"].median()),
            "checkpoint": SERVED_META["name"],
        }
        if _frac < 0.05:
            D1["verdict"] = "mixture_is_decoration_retrain_single_gaussian"
        elif _frac > 0.15:
            D1["verdict"] = "adopt_mixture_aware_cost"
        else:
            D1["verdict"] = "inconclusive_keep_l2_on_mean"
    D1
    return D1, d1_df


@app.cell
def _(D1, TARGET, d1_df, mo, new_fig, np, pl, plt, save_fig):
    mo.stop(
        d1_df is None,
        mo.md(f"**D1 unavailable** — {D1['reason']}"),
    )
    _fig, _ax = new_fig(2, 2, figsize=(11, 7))

    # (a) commanded-action confusion between the two costs.
    _levels = sorted(set(d1_df["u_mean"].to_list()) | set(d1_df["u_mix"].to_list()))
    _M = np.zeros((len(_levels), len(_levels)))
    for _a, _b in zip(d1_df["u_mean"], d1_df["u_mix"]):
        _M[_levels.index(_a), _levels.index(_b)] += 1
    _M = _M / max(_M.sum(), 1)
    _im = _ax[0, 0].imshow(_M, cmap="magma", origin="lower")
    _ax[0, 0].set_xticks(range(len(_levels)), [f"{v:g}" for v in _levels])
    _ax[0, 0].set_yticks(range(len(_levels)), [f"{v:g}" for v in _levels])
    _ax[0, 0].set_xlabel("u[0] under $J_{mix}$ (ms)")
    _ax[0, 0].set_ylabel("u[0] under $J_{mean}$ (ms)")
    _ax[0, 0].set_title(
        f"action agreement — {1 - float(d1_df['changed'].mean()):.1%} on the diagonal"
    )
    _fig.colorbar(_im, ax=_ax[0, 0], label="fraction of decisions")

    # (b) where the disagreement sits in time.
    _bt = (
        d1_df.with_columns((pl.col("hours")).round(0).alias("hbin"))
        .group_by("hbin")
        .agg(pl.col("changed").mean().alias("frac"), pl.len().alias("n"))
        .sort("hbin")
        .filter(pl.col("n") >= 10)
    )
    _ax[0, 1].plot(_bt["hbin"], _bt["frac"], "o-", color="#cc3311")
    _ax[0, 1].axhline(0.05, color="#666666", ls=":", lw=1.2, label="5% rule")
    _ax[0, 1].axhline(0.15, color="#666666", ls="--", lw=1.2, label="15% rule")
    _ax[0, 1].set_xlabel("experiment time (h)")
    _ax[0, 1].set_ylabel("fraction of actions changed")
    _ax[0, 1].set_title("disagreement over time")
    _ax[0, 1].legend(fontsize=8)

    # (c) disagreement vs cell state (CNR relative to target).
    _err = (d1_df["cnr_median"] - TARGET).to_numpy()
    _ch = d1_df["changed"].to_numpy()
    _bins = np.linspace(np.percentile(_err, 1), np.percentile(_err, 99), 21)
    _cent = 0.5 * (_bins[1:] + _bins[:-1])
    _who = np.digitize(_err, _bins) - 1
    _frac = [
        _ch[_who == i].mean() if (_who == i).sum() >= 10 else np.nan
        for i in range(len(_cent))
    ]
    _ax[1, 0].plot(_cent, _frac, "o-", color="#4477aa")
    _ax[1, 0].axvline(0.0, color="black", ls="--", lw=1.0)
    _ax[1, 0].set_xlabel(f"cnr_median − target ({TARGET:g})")
    _ax[1, 0].set_ylabel("fraction of actions changed")
    _ax[1, 0].set_title("disagreement vs cell state")

    # (d) mixture shape at agreeing vs disagreeing decisions. If the mixture never
    # matters for control, these two should be indistinguishable.
    for _flag, _col, _lab in ((False, "#999999", "agree"), (True, "#cc3311", "disagree")):
        _s = d1_df.filter(pl.col("changed") == _flag)
        if _s.is_empty():
            continue
        _ax[1, 1].scatter(
            _s["pi_entropy"], _s["mu_spread"], s=6, alpha=0.25, color=_col,
            label=f"{_lab} (n={_s.height})",
        )
    _ax[1, 1].set_xlabel("π entropy (nats)")
    _ax[1, 1].set_ylabel("max |μ_i − μ_j| (absolute CNR)")
    _ax[1, 1].set_title("mixture shape at the applied step")
    _ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    save_fig("d1_mixture_cost_replay", _fig)
    plt.gca()
    return


@app.cell
def _(D1, mo):
    _n = D1["numbers"]
    mo.md(
        f"""
    ### D1 verdict — `{D1['verdict']}`

    {f"Over **{_n['n_decisions']}** paired decisions ({_n['n_candidate_plans']} shared candidate plans, H = {_n['horizon']}), the mixture-aware cost changed the applied action in **{_n['frac_action_changed']:.1%}** of them (mean |Δu[0]| when it did: **{_n['mean_abs_du_when_changed_ms']:.0f} ms**). Median π entropy **{_n['median_pi_entropy']:.3f}** nats, median component spread **{_n['median_mu_spread_cnr']:.3f}** CNR." if _n else "**Unavailable** — " + D1['reason']}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## D2 — Dose-response sweep

    **Question.** Where does predicted response saturate? This re-grids the dose
    ladder — and it is the decisive test of the Niesen contamination that D4
    quantifies from the data side.

    Constant exposure is swept **0 → 800 ms in 20 ms steps**, converted to fluence
    with the served calibration, held over the checkpoint's own horizon, and decoded.
    Every checkpoint supplied is swept; each is rolled only to its own `future_len`
    (never past it — beyond that the decoder is untrained and `sigma_step_bias`
    indexes out of range).

    **Decision rule.** Knee `< 400 ms` → 600/800 ms are dead levels; re-grid
    non-uniformly, as a config change applied to **every** arm rather than as an
    experimental condition. Near-linear to 800 ms → the ladder is fine and
    resolution is not the problem. Slope ratio (without-Niesen / with-Niesen)
    `> 1.3` → the contamination materially suppresses learned dose sensitivity and
    the accuracy-based decision to keep Niesen was made on the wrong metric.
    """)
    return


@app.cell
def _(KEYS_10H, TARGET, data_armed, n_dec_d2, np, pl):
    # D2 sample: stratified by current-CNR stratum (below / at / above target) and
    # by experiment third, so a flat curve cannot be an artefact of every probe
    # sitting in the same regime.
    _d = (
        data_armed.filter(
            pl.col("track_key").is_in(KEYS_10H)
            & pl.col("exposure_ms").is_not_null()
            & (pl.col("n_frames_seen") >= 10)
        )
        .select(["track_key", "timestep", "hours", "cnr_median", "arm"])
        .drop_nulls()
    )
    _tmax = float(_d["timestep"].max())
    _pool = _d.with_columns(
        pl.when(pl.col("cnr_median") < TARGET - 0.1)
        .then(pl.lit("below"))
        .when(pl.col("cnr_median") > TARGET + 0.1)
        .then(pl.lit("above"))
        .otherwise(pl.lit("at"))
        .alias("stratum"),
        pl.when(pl.col("timestep") < _tmax / 3)
        .then(pl.lit("early"))
        .when(pl.col("timestep") < 2 * _tmax / 3)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("third"),
    )
    _rng = np.random.default_rng(7)
    _per = max(1, n_dec_d2.value // 9)
    _picks = []
    for _st in ("below", "at", "above"):
        for _th in ("early", "mid", "late"):
            _s = _pool.filter((pl.col("stratum") == _st) & (pl.col("third") == _th))
            if _s.is_empty():
                continue
            _take = min(_per, _s.height)
            _picks.append(_s[np.sort(_rng.choice(_s.height, size=_take, replace=False))])
    d2_points = pl.concat(_picks) if _picks else _pool.head(0)

    # The exposure grid swept for every checkpoint.
    D2_EXPO = np.arange(0.0, 800.0 + 1e-9, 20.0)
    d2_points.group_by(["stratum", "third"]).agg(pl.len().alias("n")).sort(
        ["stratum", "third"]
    )
    return D2_EXPO, d2_points


@app.cell
def _(
    CKPT_META,
    CONTROL_HORIZON,
    D2_EXPO,
    FLU_PER_MS,
    cached_states,
    d2_points,
    get_model,
    np,
    pl,
    rollout_mdn,
    torch,
):
    # --- D2 core: sweep every loadable checkpoint --------------------------
    d2_curves = []          # long-format: checkpoint x stratum x exposure
    D2_PER_CKPT = {}
    D2_SKIPPED = [m["name"] + " (" + m["reason"] + ")" for m in CKPT_META if not m["ok"]]

    for _meta in [m for m in CKPT_META if m["ok"]]:
        if d2_points.is_empty():
            continue
        _model = get_model(_meta["path"])
        _H = min(CONTROL_HORIZON, int(_meta["future_len"]))
        _mean = np.asarray(_model.cfg.norm_mean, np.float32)
        _std = np.asarray(_model.cfg.norm_std, np.float32)
        _cnr_m, _cnr_s = float(_mean[0]), float(_std[0])
        _flu_std = ((D2_EXPO * FLU_PER_MS) - float(_mean[1])) / float(_std[1])
        _E = len(D2_EXPO)

        _wanted = list(zip(d2_points["track_key"].to_list(), d2_points["timestep"].to_list()))
        _keys, _h, _c, _fb = cached_states(_model, _meta["name"], _wanted, f"d2_H{_H}")
        _D = len(_keys)
        _futE = torch.tensor(_flu_std, dtype=torch.float32, device=_h.device)

        _term, _traj = [], []
        _chunk = 24
        for _s0 in range(0, _D, _chunk):
            _s1 = min(_D, _s0 + _chunk)
            _n = _s1 - _s0
            _hb = _h[:, _s0:_s1].repeat_interleave(_E, dim=1)
            _cb = _c[:, _s0:_s1].repeat_interleave(_E, dim=1)
            _fbb = _fb[_s0:_s1].repeat_interleave(_E, dim=0)
            _futb = _futE.view(1, _E, 1, 1).expand(_n, _E, _H, 1).reshape(_n * _E, _H, 1)
            _pi, _mu, _sg = rollout_mdn(_model, _hb, _cb, _fbb, _futb)
            _pred = ((_pi * _mu).sum(-1) * _cnr_s + _cnr_m).view(_n, _E, _H)
            _term.append(_pred[:, :, -1].cpu().numpy())
            _traj.append(_pred.cpu().numpy())
        _term = np.concatenate(_term, axis=0)                     # (D, E)
        _traj = np.concatenate(_traj, axis=0)                     # (D, E, H)

        _strat = (
            pl.DataFrame({"track_key": [k for k, _ in _keys], "timestep": [t for _, t in _keys]})
            .join(d2_points, on=["track_key", "timestep"], how="left")["stratum"]
            .to_numpy()
        )
        for _st in ["all", "below", "at", "above"]:
            _m = np.ones(len(_strat), bool) if _st == "all" else (_strat == _st)
            if _m.sum() < 5:
                continue
            _med = np.median(_term[_m], axis=0)
            _q1 = np.percentile(_term[_m], 25, axis=0)
            _q3 = np.percentile(_term[_m], 75, axis=0)
            d2_curves.append(
                pl.DataFrame(
                    {
                        "checkpoint": [_meta["name"]] * len(D2_EXPO),
                        "stratum": [_st] * len(D2_EXPO),
                        "exposure_ms": D2_EXPO,
                        "median": _med,
                        "q1": _q1,
                        "q3": _q3,
                        "n_cells": [int(_m.sum())] * len(D2_EXPO),
                    }
                )
            )

        # Curve descriptors, on the pooled ("all") median curve.
        _med_all = np.median(_term, axis=0)
        _d = np.gradient(_med_all, D2_EXPO)                       # CNR per ms
        _ref_lo = float(np.mean(_d[D2_EXPO <= 100.0]))
        _knee = None
        if _ref_lo > 0:
            _cand = np.where((D2_EXPO > 100.0) & (_d < 0.2 * _ref_lo))[0]
            _knee = float(D2_EXPO[_cand[0]]) if len(_cand) else None
        _i200 = int(np.argmin(np.abs(D2_EXPO - 200.0)))
        _slope_200 = float((_med_all[_i200] - _med_all[0]) / 200.0)
        _viol = int((np.diff(_med_all) < -1e-6).sum())
        D2_PER_CKPT[_meta["name"]] = {
            "horizon": _H,
            "future_len": int(_meta["future_len"]),
            "with_niesen": bool(_meta["with_niesen"]),
            "data_source": _meta["data_source"],
            "n_probes": int(_term.shape[0]),
            "knee_ms": _knee,
            "slope_0_200_cnr_per_ms": _slope_200,
            "slope_0_100_cnr_per_ms": _ref_lo,
            "monotonicity_violations": _viol,
            "cnr_at_0ms": float(_med_all[0]),
            "cnr_at_800ms": float(_med_all[-1]),
            "dynamic_range_cnr": float(_med_all[-1] - _med_all[0]),
        }

    d2_curve_df = pl.concat(d2_curves) if d2_curves else pl.DataFrame()
    D2_PER_CKPT
    return D2_PER_CKPT, D2_SKIPPED, d2_curve_df


@app.cell
def _(D2_PER_CKPT, d2_curve_df, mo, new_fig, pl, plt, save_fig):
    mo.stop(
        d2_curve_df.is_empty(),
        mo.md("**D2 unavailable** — no checkpoint produced a dose-response curve."),
    )
    _ckpts = d2_curve_df["checkpoint"].unique(maintain_order=True).to_list()
    _fig, _axes = new_fig(1, len(_ckpts), figsize=(5.6 * len(_ckpts), 4.4), squeeze=False)
    _cols = {"below": "#4477aa", "at": "#117733", "above": "#cc3311", "all": "#333333"}
    for _i, _ck in enumerate(_ckpts):
        _ax = _axes[0, _i]
        for _st in ("below", "at", "above"):
            _s = d2_curve_df.filter(
                (pl.col("checkpoint") == _ck) & (pl.col("stratum") == _st)
            )
            if _s.is_empty():
                continue
            _ax.plot(_s["exposure_ms"], _s["median"], lw=2.0, color=_cols[_st],
                     label=f"{_st} (n={_s['n_cells'][0]})")
            _ax.fill_between(_s["exposure_ms"], _s["q1"], _s["q3"], color=_cols[_st], alpha=0.15)
        _k = D2_PER_CKPT[_ck]["knee_ms"]
        if _k is not None:
            _ax.axvline(_k, color="black", ls="--", lw=1.3, label=f"knee {_k:.0f} ms")
        _ax.axvline(400.0, color="#888888", ls=":", lw=1.0)
        _ax.set_xlabel("held exposure (ms)")
        _ax.set_ylabel("predicted CNR at horizon end")
        _ax.set_title(
            f"{_ck[:34]}\nH={D2_PER_CKPT[_ck]['horizon']} · "
            f"{'with' if D2_PER_CKPT[_ck]['with_niesen'] else 'without'} Niesen",
            fontsize=9,
        )
        _ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig("d2_dose_response", _fig)
    plt.gca()
    return


@app.cell
def _(D2_PER_CKPT, D2_SKIPPED, SERVED_META, mo, pl):
    # --- D2 verdict --------------------------------------------------------
    D2 = {"verdict": "unavailable", "reason": "", "numbers": {}, "skipped": D2_SKIPPED}
    if not D2_PER_CKPT:
        D2["reason"] = "no loadable checkpoint produced a dose-response curve"
    else:
        _served = SERVED_META["name"] if SERVED_META else next(iter(D2_PER_CKPT))
        _s = D2_PER_CKPT.get(_served) or next(iter(D2_PER_CKPT.values()))
        _knee = _s["knee_ms"]
        if _knee is not None and _knee < 400.0:
            D2["verdict"] = "regrid_ladder_knee_below_400ms"
        elif _knee is None:
            D2["verdict"] = "ladder_ok_near_linear_to_800ms"
        else:
            D2["verdict"] = "knee_above_400ms_ladder_acceptable"

        # Matched with/without-Niesen pair, if one was supplied.
        _with = [v for v in D2_PER_CKPT.values() if v["with_niesen"]]
        _without = [v for v in D2_PER_CKPT.values() if not v["with_niesen"]]
        _ratio = None
        if _with and _without:
            _a = sum(v["slope_0_200_cnr_per_ms"] for v in _without) / len(_without)
            _b = sum(v["slope_0_200_cnr_per_ms"] for v in _with) / len(_with)
            _ratio = float(_a / _b) if _b != 0 else None
        D2["numbers"] = {
            "reference_checkpoint": _served,
            "per_checkpoint": D2_PER_CKPT,
            "niesen_slope_ratio_without_over_with": _ratio,
            "niesen_slope_ratio_verdict": (
                "unavailable_no_matched_pair"
                if _ratio is None
                else ("niesen_suppresses_dose_sensitivity" if _ratio > 1.3 else "niesen_effect_below_threshold")
            ),
        }

    mo.vstack(
        [
            pl.DataFrame(
                [{"checkpoint": k, **{kk: vv for kk, vv in v.items() if kk != "data_source"}}
                 for k, v in D2_PER_CKPT.items()]
            )
            if D2_PER_CKPT
            else mo.md("_no curves_"),
            mo.md(
                f"### D2 verdict — `{D2['verdict']}`\n\n"
                + (
                    f"Reference checkpoint `{D2['numbers']['reference_checkpoint']}`: "
                    f"knee **{D2['numbers']['per_checkpoint'][D2['numbers']['reference_checkpoint']]['knee_ms']}** ms, "
                    f"0–200 ms slope **{D2['numbers']['per_checkpoint'][D2['numbers']['reference_checkpoint']]['slope_0_200_cnr_per_ms']:.2e}** CNR/ms.\n\n"
                    f"Niesen slope ratio (without/with): **{D2['numbers']['niesen_slope_ratio_without_over_with']}** → "
                    f"`{D2['numbers']['niesen_slope_ratio_verdict']}`."
                    if D2["numbers"]
                    else "**Unavailable** — " + D2["reason"]
                )
                + (
                    "\n\n_Note: no checkpoint trained **without** the Niesen data was "
                    "supplied, so the slope-ratio test cannot run. Supply a matched "
                    "pair to close this out._"
                    if D2.get("numbers", {}).get("niesen_slope_ratio_without_over_with") is None
                    else ""
                )
            ),
        ]
    )
    return (D2,)


@app.cell
def _(mo):
    mo.md("""
    ## D3 — Tracking-error power spectrum

    **Question.** Is the wobble around target caused by the **horizon**, by
    **actuator lag plus a cost with no move penalty**, or by **the cells**?

    Three things this does that a naive PSD would not:

    1. **Detrend first.** The population drifts over 12 h; undetrended, that drift
       dominates the low-frequency end and buries every real peak. A per-track
       linear detrend is removed and its slope reported separately — that slope
       *is* the per-cell drift rate, and is useful in its own right.
    2. **Test against an AR(1) red-noise null**, fitted per track. Autocorrelated
       noise produces a sloped spectrum with no oscillation at all; without the null
       every track appears to have a "dominant period" at its lowest frequency.
    3. **Cross-check with the autocorrelation**, which is an independent estimate of
       the same period.

    **Known limitation — stated because it bounds the verdict.** This run is `raw`
    mode and stimulates from frame 0, so there is no unstimulated reference beyond
    the single `optocheck_t0` frame. The endogenous-pulsing branch therefore
    **cannot be confirmed from this run**, only made plausible by exclusion of the
    other two. That is the strongest argument for a dark subset in the next
    experiment.
    """)
    return


@app.cell
def _(DT_MIN, TARGET, chi2, np, pl, signal):
    def track_spectrum(cnr, fs=1.0 / DT_MIN, band=(3.0, 180.0), alpha=0.95):
        """Detrended Welch PSD of the tracking error, tested against an AR(1) null.

        Returns a dict with the removed linear trend, the dominant period (minutes)
        where the peak clears the red-noise null, and the ACF cross-check.
        `fs` is in samples per minute (1/60 Hz at dt = 1 min).
        """
        e = np.asarray(cnr, np.float64) - TARGET
        n = len(e)
        if n < 60:
            return None
        t = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(t, e, 1)
        r = e - (slope * t + intercept)
        if not np.isfinite(r).all() or r.std() < 1e-9:
            return None

        nperseg = int(min(256, max(32, n // 3)))
        f, p = signal.welch(r, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                            detrend=False, window="hann")
        keep = f > 0
        f, p = f[keep], p[keep]
        period = 1.0 / f                                   # minutes

        # AR(1) red-noise null (Gilman), scaled to the observed mean power, with a
        # chi-square confidence level whose dof come from the number of Welch
        # segments actually averaged.
        phi = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.0, 0.98))
        null = (1 - phi**2) / (1 - 2 * phi * np.cos(2 * np.pi * f / fs) + phi**2)
        null = null * (p.mean() / null.mean())
        n_seg = max(1, 2 * n // nperseg - 1)
        dof = 2 * n_seg
        inband = (period >= band[0]) & (period <= band[1])
        n_band = int(max(inband.sum(), 1))

        # MULTIPLE COMPARISONS. The peak is searched over every in-band frequency,
        # so a per-frequency 95% test has a per-TRACK false-positive rate of
        # 1 - 0.95**n_band — with ~120 in-band bins that is ~100%, and essentially
        # every track then reports a "significant" peak whether or not it has one.
        # The Sidak-corrected level is the honest one for a peak search; the
        # uncorrected one is kept alongside so the inflation is visible rather than
        # merely fixed.
        alpha_sidak = alpha ** (1.0 / n_band)
        thresh = null * chi2.ppf(alpha_sidak, dof) / dof
        thresh_uncorr = null * chi2.ppf(alpha, dof) / dof

        ratio = np.where(inband, p / np.maximum(thresh, 1e-30), 0.0)
        sig = inband & (p > thresh)
        dom = float(period[np.argmax(ratio)]) if sig.any() else None
        sig_uncorr = bool((inband & (p > thresh_uncorr)).any())

        # Independent cross-check: first interior local maximum of the ACF.
        rr = r - r.mean()
        acf = np.correlate(rr, rr, mode="full")[n - 1 :]
        acf = acf / max(acf[0], 1e-30)
        acf_period = None
        for lag in range(2, min(len(acf) - 1, int(band[1]))):
            if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1] and acf[lag] > 0.05:
                acf_period = float(lag * DT_MIN)
                break

        return {
            "drift_cnr_per_h": float(slope * 60.0 / DT_MIN),
            "phi": phi,
            "dominant_period_min": dom,
            "significant": bool(sig.any()),
            "significant_uncorrected": sig_uncorr,
            "n_band_freqs": n_band,
            "acf_period_min": acf_period,
            "resid_std": float(r.std()),
            "f": f,
            "psd": p,
            "thresh": thresh,
        }

    def spectra_for(df, keys, window=None):
        """Run `track_spectrum` over a track set, optionally restricted to a window
        (in hours). Returns a polars frame, one row per track."""
        d = df.filter(pl.col("track_key").is_in(keys))
        if window is not None:
            d = d.filter((pl.col("hours") >= window[0]) & (pl.col("hours") < window[1]))
        rows, curves = [], {}
        for (k,), g in d.sort("timestep").group_by("track_key", maintain_order=True):
            r = track_spectrum(g["cnr_median"].drop_nulls().to_numpy())
            if r is None:
                continue
            curves[k] = (r["f"], r["psd"], r["thresh"])
            rows.append(
                {
                    "track_key": k,
                    "arm": g["arm"][0],
                    "drift_cnr_per_h": r["drift_cnr_per_h"],
                    "phi": r["phi"],
                    "dominant_period_min": r["dominant_period_min"],
                    "significant": r["significant"],
                    "significant_uncorrected": r["significant_uncorrected"],
                    "acf_period_min": r["acf_period_min"],
                    "resid_std": r["resid_std"],
                }
            )
        return pl.DataFrame(rows), curves

    return (spectra_for,)


@app.cell
def _(KEYS_10H, KEYS_4H, data_armed, spectra_for):
    # Full run, plus the early / late windows that separate a limit cycle that is
    # present from the start from one that develops as the cells desensitize.
    d3_full, d3_curves = spectra_for(data_armed, KEYS_10H)
    d3_early, _ = spectra_for(data_armed, KEYS_10H, window=(0.0, 4.0))
    d3_late, _ = spectra_for(data_armed, KEYS_10H, window=(8.0, 12.0))
    d3_full_4h, _ = spectra_for(data_armed, KEYS_4H)
    d3_full.head()
    return d3_curves, d3_early, d3_full, d3_full_4h, d3_late


@app.cell
def _(
    ARM_COLORS,
    CONTROL_HORIZON,
    d3_curves,
    d3_early,
    d3_full,
    d3_late,
    mo,
    new_fig,
    np,
    pl,
    plt,
    save_fig,
):
    mo.stop(d3_full.is_empty(), mo.md("**D3 unavailable** — no track was long enough for a PSD."))
    _fig, _ax = new_fig(2, 2, figsize=(11, 7.5))

    # (a) dominant-period distribution per arm, with the three hypothesis bands.
    _arms = sorted(d3_full["arm"].unique().to_list())
    for _i, _a in enumerate(_arms):
        _v = (
            d3_full.filter((pl.col("arm") == _a) & pl.col("significant"))["dominant_period_min"]
            .drop_nulls()
            .to_numpy()
        )
        if _v.size:
            _ax[0, 0].hist(_v, bins=np.arange(2, 62, 2), histtype="step", lw=1.6,
                           color=ARM_COLORS.get(_a, f"C{_i}"), label=f"{_a} (n={_v.size})")
    # Shading = the bands the verdict rule actually applies (chatter is tested at
    # <= 12 min, slightly wider than the handoff's nominal 5–10, so the plot and the
    # rule cannot drift apart).
    _ax[0, 0].axvspan(0, 12, color="#ee7733", alpha=0.12)
    _ax[0, 0].axvspan(2 * CONTROL_HORIZON - 2, 2 * CONTROL_HORIZON + 4, color="#4477aa", alpha=0.12)
    _ax[0, 0].axvspan(15, 30, color="#117733", alpha=0.10)
    _ax[0, 0].set_xlabel("dominant period (min)")
    _ax[0, 0].set_ylabel("# tracks")
    _ax[0, 0].set_title(
        f"dominant period per arm — decision bands: chatter ≤12, "
        f"2H={2 * CONTROL_HORIZON} (±), endogenous 15–30"
    )
    _ax[0, 0].legend(fontsize=7)

    # (b) early vs late window.
    for _lab, _df, _col in (("0–4 h", d3_early, "#4477aa"), ("8–12 h", d3_late, "#cc3311")):
        _v = _df.filter(pl.col("significant"))["dominant_period_min"].drop_nulls().to_numpy()
        if _v.size:
            _ax[0, 1].hist(_v, bins=np.arange(2, 62, 2), histtype="step", lw=1.6,
                           color=_col, label=f"{_lab} (n={_v.size})")
    _ax[0, 1].set_xlabel("dominant period (min)")
    _ax[0, 1].set_ylabel("# tracks")
    _ax[0, 1].set_title("early vs late window")
    _ax[0, 1].legend(fontsize=8)

    # (c) median PSD per arm, log-log, with the median red-noise threshold.
    _grid = None
    for _i, _a in enumerate(_arms):
        _keys = d3_full.filter(pl.col("arm") == _a)["track_key"].to_list()
        _ps = [d3_curves[k] for k in _keys if k in d3_curves]
        if not _ps:
            continue
        _grid = _ps[0][0]
        _stack = np.vstack([np.interp(_grid, f, p) for f, p, _ in _ps])
        _th = np.vstack([np.interp(_grid, f, t) for f, _, t in _ps])
        _ax[1, 0].loglog(1.0 / _grid, np.median(_stack, 0), lw=1.8,
                         color=ARM_COLORS.get(_a, f"C{_i}"), label=_a)
        _ax[1, 0].loglog(1.0 / _grid, np.median(_th, 0), lw=0.9, ls=":",
                         color=ARM_COLORS.get(_a, f"C{_i}"))
    _ax[1, 0].set_xlabel("period (min)")
    _ax[1, 0].set_ylabel("PSD of detrended error")
    _ax[1, 0].set_title("median PSD per arm (dotted = AR(1) 95% null)")
    _ax[1, 0].legend(fontsize=8)

    # (d) the drift the detrend removed — a real per-cell quantity, not a nuisance.
    for _i, _a in enumerate(_arms):
        _v = d3_full.filter(pl.col("arm") == _a)["drift_cnr_per_h"].to_numpy()
        _ax[1, 1].hist(_v, bins=40, histtype="step", lw=1.6,
                       color=ARM_COLORS.get(_a, f"C{_i}"), label=_a)
    _ax[1, 1].axvline(0.0, color="black", ls="--", lw=1.0)
    _ax[1, 1].set_xlabel("removed linear drift (CNR / h)")
    _ax[1, 1].set_ylabel("# tracks")
    _ax[1, 1].set_title("per-cell drift rate (removed before the PSD)")
    _ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    save_fig("d3_error_spectrum", _fig)
    plt.gca()
    return


@app.cell
def _(CONTROL_HORIZON, d3_early, d3_full, d3_full_4h, d3_late, mo, np, pl):
    # --- D3 verdict --------------------------------------------------------
    def _classify(df, H):
        """Map a track set's dominant period onto the three mechanism hypotheses.

        The 2H band (horizon-limited limit cycle) and the 15–30 min endogenous-ERK
        band OVERLAP at H = 8 (2H = 16). A median landing in the overlap is reported
        as `undetermined`, not silently assigned — this run cannot separate them
        (see the no-dark-reference limitation above).
        """
        sig = df.filter(pl.col("significant"))["dominant_period_min"].drop_nulls().to_numpy()
        if sig.size < 5:
            return "no_significant_peak", None, sig.size, float(df.height)
        med = float(np.median(sig))
        return _band_of(med, H), med, sig.size, float(df.height)

    def _band_of(period, H):
        """Which mechanism a period implicates. `None` -> no call."""
        if period is None:
            return "no_significant_peak"
        lo_2h, hi_2h = 2 * H - 2, 2 * H + 4
        in_2h = lo_2h <= period <= hi_2h
        in_endo = 15.0 <= period <= 30.0
        if period <= 12.0:
            return "actuator_lag_chatter"
        if in_2h and in_endo:
            return "ambiguous_2H_overlaps_endogenous_band"
        if in_2h:
            return "horizon_limited_limit_cycle"
        if in_endo:
            return "endogenous_erk_pulsing"
        return "undetermined"

    _v10, _m10, _n10, _t10 = _classify(d3_full, CONTROL_HORIZON)
    _v4, _m4, _n4, _t4 = _classify(d3_full_4h, CONTROL_HORIZON)

    # Cross-check gate. The ACF is an INDEPENDENT estimate of the same period, and
    # arm 4 hangs entirely on this verdict, so a failed cross-check downgrades it
    # rather than being noted and ignored.
    #
    # The gate is on the MECHANISM, not on the number. A PSD peak at half the ACF
    # period is a harmonic — the exact period is then unresolved, but if both
    # candidate periods implicate the same mechanism the call still stands, and
    # discarding it would be over-conservative. Only a disagreement that changes
    # which mechanism is implicated invalidates the verdict.
    _acf_med = (
        float(d3_full["acf_period_min"].drop_nulls().median())
        if d3_full["acf_period_min"].drop_nulls().len()
        else None
    )
    _acf_band = _band_of(_acf_med, CONTROL_HORIZON) if _acf_med else None
    _period_agrees = (
        None if (_m10 is None or not _acf_med) else bool(1 / 1.5 <= (_m10 / _acf_med) <= 1.5)
    )
    _agree = None if _acf_band is None else bool(_acf_band == _v10)
    if _agree is False and _v10 != "no_significant_peak":
        _v10_final = f"{_v10}__unconfirmed_acf_implies_{_acf_band}"
    else:
        _v10_final = _v10

    D3 = {
        "verdict": _v10_final,
        "verdict_psd_only": _v10,
        "acf_implies": _acf_band,
        "acf_psd_same_mechanism": _agree,
        "acf_psd_same_period": _period_agrees,
        "verdict_4h": _v4,
        "verdict_changes_with_4h_set": _v10 != _v4,
        "reason": "",
        "numbers": {
            "control_horizon": CONTROL_HORIZON,
            "median_dominant_period_min_10h": _m10,
            "median_dominant_period_min_4h": _m4,
            "n_tracks_10h": int(_t10),
            "n_significant_10h": int(_n10),
            "frac_no_significant_peak_10h": float(1.0 - _n10 / max(_t10, 1)),
            # Same test WITHOUT the Sidak correction, to show how much of the
            # apparent significance was a multiple-comparisons artefact.
            "frac_significant_uncorrected_10h": float(
                d3_full["significant_uncorrected"].mean()
            ),
            "n_tracks_4h": int(_t4),
            "n_significant_4h": int(_n4),
            "median_drift_cnr_per_h": float(d3_full["drift_cnr_per_h"].median()),
            "median_acf_period_min": _acf_med,
            "acf_psd_ratio": (None if (_m10 is None or not _acf_med) else float(_m10 / _acf_med)),
            "median_phi_ar1": float(d3_full["phi"].median()),
            "per_arm_median_period_min": {
                r["arm"]: r["med"]
                for r in d3_full.filter(pl.col("significant"))
                .group_by("arm")
                .agg(pl.col("dominant_period_min").median().alias("med"))
                .iter_rows(named=True)
            },
            "median_period_early_min": (
                float(d3_early.filter(pl.col("significant"))["dominant_period_min"].drop_nulls().median())
                if d3_early.filter(pl.col("significant")).height
                else None
            ),
            "median_period_late_min": (
                float(d3_late.filter(pl.col("significant"))["dominant_period_min"].drop_nulls().median())
                if d3_late.filter(pl.col("significant")).height
                else None
            ),
        },
        "limitation": (
            "raw-mode run stimulating from frame 0: no unstimulated reference, so the "
            "endogenous-pulsing branch cannot be CONFIRMED here, only left standing by "
            "exclusion. A dark subset in the next experiment is what closes this."
        ),
    }
    _n = D3["numbers"]
    mo.md(
        f"""
    ### D3 verdict — `{D3['verdict']}` (≥4 h set: `{D3['verdict_4h']}`{", **verdict changes**" if D3['verdict_changes_with_4h_set'] else ", unchanged"})

    Median dominant period **{_n['median_dominant_period_min_10h']} min** over
    {_n['n_significant_10h']}/{_n['n_tracks_10h']} tracks whose peak clears the
    Šidák-corrected AR(1) null ({_n['frac_no_significant_peak_10h']:.0%} had none).
    Without the multiple-comparison correction
    **{_n['frac_significant_uncorrected_10h']:.0%}** of tracks would look
    "significant" — that inflation is why the correction is applied.
    Median AR(1) φ **{_n['median_phi_ar1']:.2f}**. Removed drift
    **{_n['median_drift_cnr_per_h']:+.3f} CNR/h**. Early window
    **{_n['median_period_early_min']}** min vs late **{_n['median_period_late_min']}** min.

    **ACF cross-check: {_n['median_acf_period_min']} min** vs PSD
    {_n['median_dominant_period_min_10h']} min (ratio
    {_n['acf_psd_ratio'] if _n['acf_psd_ratio'] is None else round(_n['acf_psd_ratio'], 2)}),
    which implicates `{D3['acf_implies']}`.
    {"Both estimates point at the same mechanism, so the call stands." if D3['acf_psd_same_mechanism'] else "**They implicate different mechanisms**, so the call is marked unconfirmed."}
    {"" if D3['acf_psd_same_period'] else "The *exact* period is nonetheless unresolved — the ratio is near ½, i.e. one estimate is a harmonic of the other — so use the mechanism, not the number."}

    ⚠️ {D3['limitation']}
    """
    )
    return (D3,)


@app.cell
def _(mo):
    mo.md("""
    ## D4 — `u_t` distribution by training source

    **Question.** Where does the deployed dose range sit inside the training
    distribution of standardized fluence, and how much does the Niesen high-dose
    experiment skew it?

    Standardization uses the **frozen stats stamped on the served checkpoint**, not
    freshly computed ones — those frozen stats are what the deployed model actually
    applies, so recomputing them would answer a question nobody asked.

    **Decision rule.** If the deployment range occupies `< 20%` of the standardized
    training spread, **or** excluding Niesen shifts the deployment range by more
    than 0.5 standardized units, the frozen `u_t` normalization is compromised for
    deployment. Combine with the D2 slope ratio for the final call.
    """)
    return


@app.cell
def _(FLU_PER_MS, SERVED_META, materials_path, np, pl):
    # Source grouping. `original_experiment_name` is the only session label in the
    # bundle; these five groups are the ones the training-mix decision was made over.
    D4_SOURCE_MAP = {
        "3-2-1minIntervals": "cedric",
        "DoseResponse": "cedric",
        "RampReverse": "cedric",
        "Sustained_1min": "cedric",
        "bo_v8": "bayesian_opt",
        "bo_v10": "bayesian_opt",
        "bo_v11_10s": "bayesian_opt",
        "bo_v11_20s": "bayesian_opt",
        "freepattern_v1": "freepattern_v1",
        "freepattern_v2": "freepattern_v2",
        "freepattern_Niesen_EGFR_v1": "niesen_egfr",
    }
    D4_PATH = materials_path("dataset_all.parquet")
    D4_OK = D4_PATH.exists() and SERVED_META is not None
    D4_REASON = (
        ""
        if D4_OK
        else (
            f"training bundle not found: {D4_PATH}"
            if not D4_PATH.exists()
            else "no loadable checkpoint, so the frozen norm stats are unknown"
        )
    )

    d4_raw = None
    if D4_OK:
        d4_raw = (
            pl.scan_parquet(D4_PATH)
            .select(["original_experiment_name", "u_t"])
            .collect()
            .with_columns(
                pl.col("original_experiment_name")
                .replace_strict(D4_SOURCE_MAP, default="other")
                .alias("source")
            )
        )

    # The deployed range, on the same fluence axis: 0–800 ms at Niesen 10%.
    D4_DEPLOY_MS = (0.0, 800.0)
    D4_DEPLOY_FLU = (D4_DEPLOY_MS[0] * FLU_PER_MS, D4_DEPLOY_MS[1] * FLU_PER_MS)
    D4_FROZEN = (
        (float(np.asarray(SERVED_META["norm_mean"])[1]), float(np.asarray(SERVED_META["norm_std"])[1]))
        if SERVED_META
        else (None, None)
    )
    D4_DEPLOY_FLU, D4_FROZEN
    return D4_DEPLOY_FLU, D4_FROZEN, D4_OK, D4_REASON, d4_raw


@app.cell
def _(D4_DEPLOY_FLU, D4_FROZEN, D4_OK, D4_REASON, d4_raw, np, pl):
    # --- D4 core -----------------------------------------------------------
    D4 = {"verdict": "unavailable", "reason": D4_REASON, "numbers": {}}
    d4_stats = None

    if D4_OK:
        _m, _s = D4_FROZEN
        _u = d4_raw["u_t"].to_numpy().astype(np.float64)
        _dep_lo, _dep_hi = (D4_DEPLOY_FLU[0] - _m) / _s, (D4_DEPLOY_FLU[1] - _m) / _s

        # Per-source summary in standardized units.
        d4_stats = (
            d4_raw.with_columns(((pl.col("u_t") - _m) / _s).alias("u_std"))
            .group_by("source")
            .agg(
                pl.len().alias("n"),
                pl.col("u_t").mean().alias("mean_flu"),
                pl.col("u_t").std().alias("std_flu"),
                pl.col("u_std").quantile(0.01).alias("p01"),
                pl.col("u_std").median().alias("p50"),
                pl.col("u_std").quantile(0.99).alias("p99"),
                pl.col("u_std").max().alias("max"),
                (pl.col("u_t") > D4_DEPLOY_FLU[1]).mean().alias("frac_above_deploy_max"),
            )
            .sort("source")
        )

        # Overlap between the deployed interval and the training spread (1–99 pct,
        # standardized). BOTH directions are reported, because they answer different
        # questions and only one of them is the dangerous one here: the deployed
        # range can be a strict SUPERSET of the training bulk, in which case
        # "deployment covers 100% of training" is true and reassuring-sounding while
        # most of the deployed range has no training support at all.
        _tr_lo, _tr_hi = float(np.percentile((_u - _m) / _s, 1)), float(np.percentile((_u - _m) / _s, 99))
        _overlap = max(0.0, min(_dep_hi, _tr_hi) - max(_dep_lo, _tr_lo))
        _coverage = _overlap / max(_tr_hi - _tr_lo, 1e-12)          # of training spread
        _supported = _overlap / max(_dep_hi - _dep_lo, 1e-12)       # of deployed range

        # Same deployment range, standardized with stats recomputed WITHOUT Niesen.
        _no_n = d4_raw.filter(pl.col("source") != "niesen_egfr")["u_t"].to_numpy().astype(np.float64)
        _m2, _s2 = float(_no_n.mean()), float(_no_n.std())
        _dep_lo2, _dep_hi2 = (D4_DEPLOY_FLU[0] - _m2) / _s2, (D4_DEPLOY_FLU[1] - _m2) / _s2
        _shift = float(max(abs(_dep_lo2 - _dep_lo), abs(_dep_hi2 - _dep_hi)))

        D4["numbers"] = {
            "frozen_u_t_mean": _m,
            "frozen_u_t_std": _s,
            "deployment_fluence_mJ_cm2": list(D4_DEPLOY_FLU),
            "deployment_standardized_with_niesen": [_dep_lo, _dep_hi],
            "deployment_standardized_without_niesen": [_dep_lo2, _dep_hi2],
            "u_t_mean_std_without_niesen": [_m2, _s2],
            "shift_standardized_units": _shift,
            "training_spread_1_99_standardized": [_tr_lo, _tr_hi],
            "deployment_coverage_of_training_spread": float(_coverage),
            "training_support_of_deployment_range": float(_supported),
            "frac_training_rows_above_deployment_max": float((_u > D4_DEPLOY_FLU[1]).mean()),
            "per_source": d4_stats.to_dicts(),
        }
        # The rule fires on EITHER direction being under 20%: a deployed range that
        # barely moves inside the training spread and one that runs far outside it
        # are both cases of the frozen scale not describing what is served.
        _bad = (min(_coverage, _supported) < 0.20) or (_shift > 0.5)
        D4["verdict"] = (
            "frozen_u_t_normalization_compromised" if _bad else "frozen_u_t_normalization_acceptable"
        )
    D4["verdict"], D4["numbers"].get("shift_standardized_units")
    return D4, d4_stats


@app.cell
def _(D4, d4_raw, d4_stats, mo, new_fig, np, pl, plt, save_fig):
    mo.stop(d4_raw is None, mo.md(f"**D4 unavailable** — {D4['reason']}"))
    _n = D4["numbers"]
    _m, _s = _n["frozen_u_t_mean"], _n["frozen_u_t_std"]
    _dlo, _dhi = _n["deployment_standardized_with_niesen"]

    _sources = d4_stats["source"].to_list()
    _fig, _axes = new_fig(2, 1, figsize=(11, 7), height_ratios=[2, 1])

    # (a) per-source histogram of standardized u_t, deployment range shaded.
    _bins = np.linspace(-0.5, max(_dhi * 1.15, 20.0), 220)
    for _i, _src in enumerate(_sources):
        _v = d4_raw.filter(pl.col("source") == _src)["u_t"].to_numpy()
        _v = (_v - _m) / _s
        _axes[0].hist(_v, bins=_bins, histtype="step", lw=1.5, density=True,
                      label=f"{_src} (n={len(_v):,})", color=f"C{_i}")
    _axes[0].axvspan(_dlo, _dhi, color="#cc3311", alpha=0.10)
    _axes[0].axvline(_dhi, color="#cc3311", ls="--", lw=1.4,
                     label=f"deployment max = {_dhi:.1f} σ (800 ms)")
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("standardized $u_t$ (frozen checkpoint stats)")
    _axes[0].set_ylabel("density (log)")
    _axes[0].set_title(
        "training $u_t$ per source vs the deployed range — "
        f"{_n['frac_training_rows_above_deployment_max']:.2%} of training rows sit above 800 ms-equivalent"
    )
    _axes[0].legend(fontsize=8)

    # (b) the same deployment range under both normalizations. If these are far
    # apart, the frozen stats are a Niesen artefact.
    _lo2, _hi2 = _n["deployment_standardized_without_niesen"]
    _axes[1].barh(["with Niesen (frozen)", "without Niesen"],
                  [_dhi - _dlo, _hi2 - _lo2], left=[_dlo, _lo2],
                  color=["#4477aa", "#ee7733"], height=0.45)
    _axes[1].axvline(0.0, color="black", lw=1.0)
    _axes[1].set_xlabel("standardized $u_t$")
    _axes[1].set_title(
        f"deployment range under each normalization — shift {_n['shift_standardized_units']:.1f} σ "
        f"(rule: > 0.5 σ compromises the frozen stats)"
    )
    for _y, (_a, _b) in enumerate([(_dlo, _dhi), (_lo2, _hi2)]):
        _axes[1].text(_b, _y, f"  {_a:.2f} → {_b:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    save_fig("d4_ut_distribution", _fig)
    plt.gca()
    return


@app.cell
def _(D4, d4_stats, mo):
    _n = D4["numbers"]
    mo.vstack(
        [
            d4_stats,
            mo.md(
                f"""
    ### D4 verdict — `{D4['verdict']}`

    Frozen `u_t` stats: mean **{_n['frozen_u_t_mean']:.3f}**, std **{_n['frozen_u_t_std']:.3f}** mJ/cm².
    The deployed 0–800 ms range is **{_n['deployment_fluence_mJ_cm2'][0]:.0f}–{_n['deployment_fluence_mJ_cm2'][1]:.0f} mJ/cm²**,
    i.e. **{_n['deployment_standardized_with_niesen'][0]:.2f} → {_n['deployment_standardized_with_niesen'][1]:.2f} σ**
    under the frozen stats. Recomputing the stats without Niesen
    (mean **{_n['u_t_mean_std_without_niesen'][0]:.3f}**, std **{_n['u_t_mean_std_without_niesen'][1]:.3f}**)
    moves the same physical range to
    **{_n['deployment_standardized_without_niesen'][0]:.2f} → {_n['deployment_standardized_without_niesen'][1]:.1f} σ** —
    a shift of **{_n['shift_standardized_units']:.1f} σ**.

    Overlap, both directions: the deployed range covers
    **{_n['deployment_coverage_of_training_spread']:.0%}** of the training 1–99
    percentile spread, but only **{_n['training_support_of_deployment_range']:.0%}**
    of the deployed range has training support inside that spread. Only
    **{_n['frac_training_rows_above_deployment_max']:.2%}** of training rows sit
    above the deployment maximum, and essentially all of them are Niesen — outside
    that one session, the top of the commanded ladder is extrapolation.
    """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## D5 — Rail-conditioned error sign

    **Question.** Is the pile-up at 800 ms **genuine saturation** (the controller
    wants more and cannot get it) or **model error / chatter**?

    The discriminator is the sign of the error while railed. A controller that is
    authority-limited sits at the rail with its cells *below* target. A controller
    that is chattering sits at the rail with cells scattered on both sides.

    Two supporting decompositions:

    * **Gain estimate** — rolling 30-min regression of ΔCNR on commanded fluence,
      per cell. `g(late)/g(early)` well below 1 is desensitization.
    * **Dose-variance decomposition** — commanded-dose variance split into
      within-cell and between-cell parts over sliding windows. Between-cell growth
      is heterogeneous desensitization; within-cell growth is chattering. These are
      competing explanations for the same exposure drift, and they are reported
      side by side rather than collapsed.
    """)
    return


@app.cell
def _(KEYS_10H, KEYS_4H, TARGET, data_armed, np, pl):
    def d5_rail_stats(keys):
        """Error-sign statistics conditioned on the commanded rail, per arm/window."""
        d = data_armed.filter(
            pl.col("track_key").is_in(keys) & pl.col("exposure_ms").is_not_null()
        ).with_columns(
            (pl.col("cnr_median") - TARGET).alias("err"),
            pl.when(pl.col("hours") < 4.0)
            .then(pl.lit("early"))
            .when(pl.col("hours") >= 8.0)
            .then(pl.lit("late"))
            .otherwise(pl.lit("mid"))
            .alias("window"),
        )
        return (
            d.filter(pl.col("exposure_ms").is_in([0.0, 800.0]))
            .group_by(["exposure_ms", "arm", "window"])
            .agg(
                pl.len().alias("n"),
                pl.col("err").median().alias("median_err"),
                (pl.col("err") < 0).mean().alias("frac_below_target"),
            )
            .sort(["exposure_ms", "arm", "window"])
        ), d

    d5_rails_10h, d5_data_10h = d5_rail_stats(KEYS_10H)
    d5_rails_4h, _d5_data_4h = d5_rail_stats(KEYS_4H)

    def d5_frac_below(rails, expo):
        s = rails.filter(pl.col("exposure_ms") == expo)
        if s.is_empty():
            return None
        w = s["n"].to_numpy()
        return float(np.average(s["frac_below_target"].to_numpy(), weights=w))

    D5_BELOW_800_10H = d5_frac_below(d5_rails_10h, 800.0)
    D5_BELOW_0_10H = d5_frac_below(d5_rails_10h, 0.0)
    D5_BELOW_800_4H = d5_frac_below(d5_rails_4h, 800.0)
    d5_rails_10h
    return (
        D5_BELOW_0_10H,
        D5_BELOW_800_10H,
        D5_BELOW_800_4H,
        d5_data_10h,
        d5_rails_10h,
    )


@app.cell
def _(d5_data_10h, np, pl):
    # Run lengths of consecutive 800 ms commands, and what CNR does in the 10 min
    # after a run of >= 5 railed frames ends.
    _runs, _after = [], []
    for (_k,), _g in d5_data_10h.sort("timestep").group_by("track_key", maintain_order=True):
        _e = _g["exposure_ms"].to_numpy()
        _c = _g["cnr_median"].to_numpy()
        _i = 0
        while _i < len(_e):
            if _e[_i] == 800.0:
                _j = _i
                while _j + 1 < len(_e) and _e[_j + 1] == 800.0:
                    _j += 1
                _runs.append({"track_key": _k, "arm": _g["arm"][0], "run_len": _j - _i + 1,
                              "start_h": float(_g["hours"][_i])})
                if (_j - _i + 1) >= 5 and _j + 10 < len(_e):
                    _after.append(_c[_j : _j + 11] - _c[_j])
                _i = _j + 1
            else:
                _i += 1
    d5_runs = pl.DataFrame(_runs) if _runs else pl.DataFrame(
        schema={"track_key": pl.Utf8, "arm": pl.Utf8, "run_len": pl.Int64, "start_h": pl.Float64}
    )
    d5_after = np.vstack(_after) if _after else np.zeros((0, 11))
    d5_runs.height, d5_after.shape
    return d5_after, d5_runs


@app.cell
def _(d5_data_10h, np, pl):
    # Per-cell gain: rolling 30-min regression of one-step dCNR on the fluence
    # commanded at that frame. A cell whose gain collapses between the first and
    # last four hours is desensitizing.
    _rows = []
    for (_k,), _g in d5_data_10h.sort("timestep").group_by("track_key", maintain_order=True):
        _c = _g["cnr_median"].to_numpy().astype(float)
        _f = _g["fluence_out"].to_numpy().astype(float)
        _h = _g["hours"].to_numpy().astype(float)
        _n = len(_c)
        if _n < 90:
            continue
        for _s in range(0, _n - 31, 10):
            _x = _f[_s : _s + 30]
            _y = np.diff(_c[_s : _s + 31])
            if np.ptp(_x) < 1e-9 or not np.isfinite(_y).all():
                continue
            _b = np.polyfit(_x, _y, 1)[0]
            _rows.append({"track_key": _k, "arm": _g["arm"][0],
                          "hours": float(_h[_s + 15]), "gain": float(_b)})
    d5_gain = pl.DataFrame(_rows) if _rows else pl.DataFrame(
        schema={"track_key": pl.Utf8, "arm": pl.Utf8, "hours": pl.Float64, "gain": pl.Float64}
    )

    _early = d5_gain.filter(pl.col("hours") < 4.0).group_by("track_key").agg(
        pl.col("gain").median().alias("g_early")
    )
    _late = d5_gain.filter(pl.col("hours") >= 8.0).group_by("track_key").agg(
        pl.col("gain").median().alias("g_late")
    )
    d5_gain_ratio = (
        _early.join(_late, on="track_key", how="inner")
        .filter(pl.col("g_early") > 1e-9)
        .with_columns((pl.col("g_late") / pl.col("g_early")).alias("ratio"))
    )
    D5_GAIN_RATIO = (
        float(d5_gain_ratio["ratio"].median()) if d5_gain_ratio.height else None
    )
    D5_GAIN_N = int(d5_gain_ratio.height)
    D5_GAIN_RATIO, D5_GAIN_N
    return D5_GAIN_N, D5_GAIN_RATIO, d5_gain_ratio


@app.cell
def _(d5_data_10h, np, pl):
    # Dose-variance decomposition over 60-min sliding windows: total commanded-dose
    # variance = mean within-cell variance + variance of the per-cell means.
    _rows = []
    _hmax = float(d5_data_10h["hours"].max())
    for _t0 in np.arange(0.0, max(_hmax - 1.0, 0.0), 0.5):
        _w = d5_data_10h.filter(
            (pl.col("hours") >= _t0) & (pl.col("hours") < _t0 + 1.0)
        )
        if _w.height < 50:
            continue
        _per = _w.group_by("track_key").agg(
            pl.col("exposure_ms").mean().alias("m"),
            pl.col("exposure_ms").var().alias("v"),
            pl.len().alias("n"),
        ).filter(pl.col("n") >= 20)
        if _per.height < 5:
            continue
        _rows.append(
            {
                "hours": float(_t0 + 0.5),
                "within": float(_per["v"].mean()),
                "between": float(_per["m"].var()),
                "n_cells": int(_per.height),
            }
        )
    d5_var = pl.DataFrame(_rows) if _rows else pl.DataFrame(
        schema={"hours": pl.Float64, "within": pl.Float64, "between": pl.Float64, "n_cells": pl.Int64}
    )
    d5_var.head()
    return (d5_var,)


@app.cell
def _(
    ARM_COLORS,
    TARGET,
    d5_after,
    d5_data_10h,
    d5_gain_ratio,
    d5_runs,
    d5_var,
    mo,
    new_fig,
    np,
    pl,
    plt,
    save_fig,
):
    mo.stop(d5_data_10h.is_empty(), mo.md("**D5 unavailable** — no railed frames to analyse."))
    _fig, _ax = new_fig(2, 3, figsize=(15, 7.5))

    # (a) error distribution at each rail.
    for _expo, _col in ((0.0, "#4477aa"), (800.0, "#cc3311")):
        _v = d5_data_10h.filter(pl.col("exposure_ms") == _expo)["err"].drop_nulls().to_numpy()
        if _v.size:
            _ax[0, 0].hist(_v, bins=60, histtype="step", lw=1.7, density=True, color=_col,
                           label=f"{_expo:g} ms (n={_v.size:,}, {(_v < 0).mean():.0%} below)")
    _ax[0, 0].axvline(0.0, color="black", ls="--", lw=1.2)
    _ax[0, 0].set_xlabel(f"cnr_median − target ({TARGET:g})")
    _ax[0, 0].set_ylabel("density")
    _ax[0, 0].set_title("error conditioned on the commanded rail")
    _ax[0, 0].legend(fontsize=8)

    # (b) fraction below target at 800 ms, per arm and window.
    _s = (
        d5_data_10h.filter(pl.col("exposure_ms") == 800.0)
        .group_by(["arm", "window"])
        .agg((pl.col("err") < 0).mean().alias("frac"), pl.len().alias("n"))
        .sort(["arm", "window"])
    )
    _arms = sorted(_s["arm"].unique().to_list())
    _wins = ["early", "mid", "late"]
    _x = np.arange(len(_wins))
    for _i, _a in enumerate(_arms):
        _vals = [
            (_s.filter((pl.col("arm") == _a) & (pl.col("window") == _w))["frac"].to_list() or [np.nan])[0]
            for _w in _wins
        ]
        _ax[0, 1].bar(_x + _i * 0.25, _vals, width=0.25, color=ARM_COLORS.get(_a, f"C{_i}"), label=_a)
    _ax[0, 1].axhline(0.70, color="black", ls="--", lw=1.2, label="70% rule")
    _ax[0, 1].set_xticks(_x + 0.25, _wins)
    _ax[0, 1].set_ylim(0, 1)
    _ax[0, 1].set_ylabel("fraction below target at 800 ms")
    _ax[0, 1].set_title("railed-and-below, per arm and window")
    _ax[0, 1].legend(fontsize=7)

    # (c) run-length distribution of consecutive 800 ms commands. Runs longer than
    # the axis are piled into a labelled overflow bin rather than dropped — the tail
    # is the whole point here (a cell pinned at the rail for hours is the
    # authority-limited case), so silently cutting it would invert the reading.
    _CAP = 40
    if d5_runs.height:
        for _i, _a in enumerate(sorted(d5_runs["arm"].unique().to_list())):
            _v = d5_runs.filter(pl.col("arm") == _a)["run_len"].to_numpy()
            _ax[0, 2].hist(np.minimum(_v, _CAP), bins=np.arange(0.5, _CAP + 1.5, 1),
                           histtype="step", lw=1.5, color=ARM_COLORS.get(_a, f"C{_i}"),
                           label=f"{_a} (n={_v.size}, {(_v > _CAP).sum()} > {_CAP})")
        _ax[0, 2].set_yscale("log")
        _ax[0, 2].axvline(_CAP, color="#888888", ls=":", lw=1.0)
    _ax[0, 2].set_xlabel(f"consecutive frames at 800 ms (≥{_CAP} pooled in the last bin)")
    _ax[0, 2].set_ylabel("# runs (log)")
    _ax[0, 2].set_title(
        f"rail run lengths — longest {int(d5_runs['run_len'].max()) if d5_runs.height else 0} frames"
    )
    _ax[0, 2].legend(fontsize=7)

    # (d) CNR in the 10 min after a run of >= 5 railed frames.
    if d5_after.shape[0]:
        _q = np.percentile(d5_after, [25, 50, 75], axis=0)
        _t = np.arange(d5_after.shape[1])
        _ax[1, 0].fill_between(_t, _q[0], _q[2], color="#cc3311", alpha=0.20)
        _ax[1, 0].plot(_t, _q[1], color="#cc3311", lw=2.0)
        _ax[1, 0].axhline(0.0, color="black", ls="--", lw=1.0)
        _ax[1, 0].set_title(f"ΔCNR after ≥5 railed frames (n={d5_after.shape[0]})")
    else:
        _ax[1, 0].set_title("no runs of ≥5 railed frames")
    _ax[1, 0].set_xlabel("minutes after the run ends")
    _ax[1, 0].set_ylabel("CNR − CNR at run end")

    # (e) gain ratio late/early. The tails are clipped for display; the clipped
    # counts are named in the legend so the pile-up at each edge is not misread as
    # a mode.
    if d5_gain_ratio.height:
        _v = d5_gain_ratio["ratio"].to_numpy()
        _v = _v[np.isfinite(_v)]
        _lo, _hi = -2.0, 4.0
        _ax[1, 1].hist(np.clip(_v, _lo, _hi), bins=50, color="#117733", alpha=0.7,
                       label=f"n={_v.size} ({(_v < _lo).sum()} below {_lo:g}, "
                             f"{(_v > _hi).sum()} above {_hi:g}, both clipped into the edge bins)")
        _ax[1, 1].axvline(1.0, color="black", ls="--", lw=1.2, label="no change")
        _ax[1, 1].axvline(float(np.median(_v)), color="#cc3311", lw=1.6,
                          label=f"median {np.median(_v):.2f}")
        _ax[1, 1].legend(fontsize=7)
    _ax[1, 1].set_xlabel("g(late) / g(early)  (display-clipped to [-2, 4])")
    _ax[1, 1].set_ylabel("# cells")
    _ax[1, 1].set_title("per-cell gain change (desensitization)")

    # (f) dose-variance decomposition.
    if d5_var.height:
        _ax[1, 2].plot(d5_var["hours"], d5_var["within"], lw=2.0, color="#ee7733", label="within-cell")
        _ax[1, 2].plot(d5_var["hours"], d5_var["between"], lw=2.0, color="#4477aa", label="between-cell")
        _ax[1, 2].legend(fontsize=8)
    _ax[1, 2].set_xlabel("time (h)")
    _ax[1, 2].set_ylabel("commanded-dose variance (ms²)")
    _ax[1, 2].set_title("dose-variance decomposition (1 h windows)")

    plt.tight_layout()
    save_fig("d5_rail_conditioned_error", _fig)
    plt.gca()
    return


@app.cell
def _(
    D5_BELOW_0_10H,
    D5_BELOW_800_10H,
    D5_BELOW_800_4H,
    D5_GAIN_N,
    D5_GAIN_RATIO,
    d5_after,
    d5_rails_10h,
    d5_runs,
    d5_var,
    mo,
    np,
    pl,
):
    # --- D5 verdict --------------------------------------------------------
    def _rail_verdict(frac):
        if frac is None:
            return "unavailable"
        return "genuine_upward_saturation" if frac > 0.70 else "model_error_or_chatter"

    _v10 = _rail_verdict(D5_BELOW_800_10H)
    _v4 = _rail_verdict(D5_BELOW_800_4H)
    _runs_free = d5_runs.filter(~pl.col("arm").str.starts_with("staggered_mpc"))
    _between_grows = None
    _within_grows = None
    if d5_var.height >= 4:
        _h = d5_var["hours"].to_numpy()
        _between_grows = bool(np.polyfit(_h, d5_var["between"].to_numpy(), 1)[0] > 0)
        _within_grows = bool(np.polyfit(_h, d5_var["within"].to_numpy(), 1)[0] > 0)

    D5 = {
        "verdict": _v10,
        "verdict_4h": _v4,
        "verdict_changes_with_4h_set": _v10 != _v4,
        "reason": "" if D5_BELOW_800_10H is not None else "no frames commanded 800 ms",
        "numbers": {
            "frac_below_target_at_800ms_10h": D5_BELOW_800_10H,
            "frac_below_target_at_800ms_4h": D5_BELOW_800_4H,
            "frac_below_target_at_0ms_10h": D5_BELOW_0_10H,
            # The rail statistic only discriminates against its own baseline: if the
            # population sits below target most of the time anyway, "below while
            # railed" is partly just that. This excess is the discriminating number.
            "excess_below_at_800ms_vs_0ms": (
                None
                if (D5_BELOW_800_10H is None or D5_BELOW_0_10H is None)
                else float(D5_BELOW_800_10H - D5_BELOW_0_10H)
            ),
            "n_frames_at_800ms": int(
                d5_rails_10h.filter(pl.col("exposure_ms") == 800.0)["n"].sum() or 0
            ),
            "n_frames_at_0ms": int(
                d5_rails_10h.filter(pl.col("exposure_ms") == 0.0)["n"].sum() or 0
            ),
            "median_rail_run_len": (
                float(d5_runs["run_len"].median()) if d5_runs.height else None
            ),
            # staggered_mpc physically cannot produce a run longer than 1 — it
            # stimulates each cell every k-th frame by construction — so its runs
            # say nothing about chatter and are excluded from the chatter reading.
            "median_rail_run_len_excl_staggered": (
                float(_runs_free["run_len"].median()) if _runs_free.height else None
            ),
            "frac_isolated_rail_frames_excl_staggered": (
                float((_runs_free["run_len"] == 1).mean()) if _runs_free.height else None
            ),
            "max_rail_run_len": (int(d5_runs["run_len"].max()) if d5_runs.height else None),
            "per_arm_median_rail_run_len": {
                r["arm"]: float(r["med"])
                for r in d5_runs.group_by("arm")
                .agg(pl.col("run_len").median().alias("med"))
                .iter_rows(named=True)
            }
            if d5_runs.height
            else {},
            "n_post_rail_episodes": int(d5_after.shape[0]),
            "median_dcnr_10min_after_rail": (
                float(np.median(d5_after[:, -1])) if d5_after.shape[0] else None
            ),
            "gain_ratio_late_over_early_median": D5_GAIN_RATIO,
            "n_cells_with_gain_ratio": D5_GAIN_N,
            "between_cell_dose_variance_growing": _between_grows,
            "within_cell_dose_variance_growing": _within_grows,
        },
        "desensitization": (
            "confirmed_competes_with_niesen_explanation"
            if (D5_GAIN_RATIO is not None and D5_GAIN_RATIO < 1.0 and _between_grows)
            else "not_confirmed"
        ),
    }
    _n = D5["numbers"]
    mo.md(
        f"""
    ### D5 verdict — `{D5['verdict']}` (≥4 h set: `{D5['verdict_4h']}`{", **verdict changes**" if D5['verdict_changes_with_4h_set'] else ", unchanged"})

    At 800 ms ({_n['n_frames_at_800ms']:,} frames), **{(_n['frac_below_target_at_800ms_10h'] or 0):.1%}**
    of cells sit below target (rule: > 70% ⇒ authority-limited). At 0 ms
    ({_n['n_frames_at_0ms']:,} frames): **{(_n['frac_below_target_at_0ms_10h'] or 0):.1%}** —
    so the *excess* attributable to being railed is
    **{(_n['excess_below_at_800ms_vs_0ms'] or 0):+.1%}**. Read the 800 ms number
    against that baseline, not against 50%: this population sits below target much
    of the time regardless of the command.
    Rail runs, excluding `staggered_mpc` (whose runs are capped at 1 by its own
    cadence and say nothing about chatter): median **{_n['median_rail_run_len_excl_staggered']}**
    frames, **{(_n['frac_isolated_rail_frames_excl_staggered'] or 0):.0%}** of them isolated
    single frames, longest **{_n['max_rail_run_len']}**. Isolated commands are chatter;
    the long tail is genuinely pinned cells — both are present, so the population
    verdict above hides a real split between cells.
    Gain ratio late/early median **{_n['gain_ratio_late_over_early_median']}**
    over {_n['n_cells_with_gain_ratio']} cells; between-cell dose variance
    growing: **{_n['between_cell_dose_variance_growing']}**, within-cell:
    **{_n['within_cell_dose_variance_growing']}** ⇒ desensitization
    `{D5['desensitization']}`.

    {"**Do not run a reduced ladder** — the controller is already saturated upward; consider raising `stim_power_pct` or `max_exposure_ms` instead." if D5['verdict'] == 'genuine_upward_saturation' else "More dose levels will not help — the rail is not authority-limited."}
    """
    )
    return (D5,)


@app.cell
def _(mo):
    mo.md("""
    ## D6 — τ_decay

    **Question.** How fast can CNR fall when the light stops? This sets the
    reference period for the next experiment and bounds *downward* control
    authority — the controller can only push a cell down as fast as the biology
    decays.

    Episodes are ≥ 10 min of sustained high dose followed by ≥ 15 consecutive dark
    frames; a single exponential `A·exp(−t/τ) + C` is fitted to the falling limb and
    poor fits are discarded at a stated R² threshold. The hold run rarely coasts
    that long — the controller keeps re-dosing — so if too few episodes exist the
    diagnostic **falls back to the freepattern training data**, which has real gaps,
    and says so explicitly in its verdict.

    **Decision rule.** Reference period for the next experiment = **4–6 × median τ**,
    reported in minutes. That number, not intuition, sets the step-train period.
    """)
    return


@app.cell
def _(curve_fit, np):
    D6_MIN_R2 = 0.5
    D6_TAU_BOUNDS = (0.5, 300.0)

    def fit_decay(y):
        """Fit `A·exp(-t/τ) + C` to a falling CNR limb. Returns None on a poor fit."""
        y = np.asarray(y, float)
        t = np.arange(len(y), dtype=float)
        if len(y) < 8 or not np.isfinite(y).all() or np.ptp(y) < 1e-6:
            return None
        p0 = (max(y[0] - y[-1], 1e-3), 20.0, y[-1])
        try:
            popt, _ = curve_fit(
                lambda tt, A, tau, C: A * np.exp(-tt / tau) + C,
                t, y, p0=p0,
                bounds=([0.0, D6_TAU_BOUNDS[0], -1.0], [10.0, D6_TAU_BOUNDS[1], 5.0]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            return None
        A, tau, C = (float(v) for v in popt)
        resid = y - (A * np.exp(-t / tau) + C)
        ss_res = float((resid**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        if r2 < D6_MIN_R2 or tau >= D6_TAU_BOUNDS[1] * 0.99:
            return None
        # IDENTIFIABILITY. A τ longer than the observation window is not measured,
        # it is extrapolated: over a 20-minute window an exponential with τ = 250
        # min is indistinguishable from a straight line, and curve_fit will happily
        # return one with a good R². Such fits are kept but flagged, and the verdict
        # is computed on the identifiable subset only.
        return {
            "A": A, "tau": tau, "C": C, "r2": r2, "y": y,
            "window_min": float(len(y)),
            "identifiable": bool(tau <= len(y)),
        }

    def find_episodes(expo, cnr, hi_thresh, min_hi=10, min_dark=15, max_dark=60):
        """Indices of (dark_start, dark_end) after >= `min_hi` sustained high-dose frames."""
        expo = np.asarray(expo, float)
        out = []
        i = 0
        n = len(expo)
        while i < n:
            if expo[i] >= hi_thresh:
                j = i
                while j + 1 < n and expo[j + 1] >= hi_thresh:
                    j += 1
                if (j - i + 1) >= min_hi and j + 1 < n and expo[j + 1] == 0.0:
                    k = j + 1
                    while k + 1 < n and expo[k + 1] == 0.0:
                        k += 1
                    if (k - j) >= min_dark:
                        out.append((j + 1, min(k, j + max_dark), float(expo[i : j + 1].mean())))
                i = j + 1
            else:
                i += 1
        return out

    return D6_MIN_R2, find_episodes, fit_decay


@app.cell
def _(
    KEYS_10H,
    OPTORTK_FED,
    data_armed,
    find_episodes,
    fit_decay,
    materials_path,
    pl,
):
    # --- D6: try the hold run first ---------------------------------------
    _rows, _examples = [], []
    for (_k,), _g in (
        data_armed.filter(pl.col("track_key").is_in(KEYS_10H) & pl.col("exposure_ms").is_not_null())
        .sort("timestep")
        .group_by("track_key", maintain_order=True)
    ):
        _e = _g["exposure_ms"].to_numpy()
        _c = _g["cnr_median"].to_numpy()
        _h = _g["hours"].to_numpy()
        for _s, _t, _dose in find_episodes(_e, _c, hi_thresh=600.0):
            _fit = fit_decay(_c[_s : _t + 1])
            if _fit is None:
                continue
            _rows.append(
                {
                    "source": "hold_run",
                    "track_key": _k,
                    "tau_min": _fit["tau"],
                    "asymptote": _fit["C"],
                    "amplitude": _fit["A"],
                    "r2": _fit["r2"],
                    "window_min": _fit["window_min"],
                    "identifiable": _fit["identifiable"],
                    "preceding_dose_ms": _dose,
                    "hours": float(_h[_s]),
                    "optortk_expr": OPTORTK_FED,
                }
            )
            if len(_examples) < 6:
                _examples.append((f"{_k} @ {_h[_s]:.1f} h", _fit))

    D6_FROM_RUN = len(_rows)
    d6_examples = _examples

    # --- fallback: the freepattern training data ---------------------------
    D6_FALLBACK_PATH = materials_path("dataset_all.parquet")
    D6_SOURCE = "hold_run"
    D6_FALLBACK_REASON = ""
    if D6_FROM_RUN < 20:
        if not D6_FALLBACK_PATH.exists():
            D6_FALLBACK_REASON = f"fallback bundle not found: {D6_FALLBACK_PATH}"
        else:
            D6_SOURCE = "freepattern_training_data"
            _fp = (
                pl.scan_parquet(D6_FALLBACK_PATH)
                .filter(
                    pl.col("original_experiment_name").is_in(
                        ["freepattern_v1", "freepattern_v2"]
                    )
                )
                .select(["uid", "frame", "cnr_median", "stim_exposure", "u_t"])
                .collect()
                .sort(["uid", "frame"])
            )
            _rows_fp, _ex_fp = [], []
            for (_u,), _g in _fp.group_by("uid", maintain_order=True):
                _e = _g["stim_exposure"].to_numpy().astype(float)
                _c = _g["cnr_median"].to_numpy().astype(float)
                _f = _g["frame"].to_numpy().astype(float)
                # The freepattern sessions ran on a different instrument, so a
                # 600 ms-equivalent fluence threshold is not transferable. The
                # relaxed criterion is "sustained light of ANY level", with the
                # preceding dose recorded as a covariate — stated, not hidden.
                for _s, _t, _dose in find_episodes(_e, _c, hi_thresh=1.0):
                    _fit = fit_decay(_c[_s : _t + 1])
                    if _fit is None:
                        continue
                    _rows_fp.append(
                        {
                            "source": "freepattern",
                            "track_key": str(_u),
                            "tau_min": _fit["tau"],
                            "asymptote": _fit["C"],
                            "amplitude": _fit["A"],
                            "r2": _fit["r2"],
                            "window_min": _fit["window_min"],
                            "identifiable": _fit["identifiable"],
                            "preceding_dose_ms": _dose,
                            "hours": float(_f[_s]) / 60.0,
                            "optortk_expr": float("nan"),
                        }
                    )
                    if len(_ex_fp) < 6:
                        _ex_fp.append((f"{_u} @ frame {int(_f[_s])}", _fit))
            _rows = _rows_fp
            d6_examples = _ex_fp
            D6_FALLBACK_REASON = (
                f"only {D6_FROM_RUN} usable episodes in the hold run (the controller "
                f"rarely coasts ≥15 dark frames), so τ is measured on the freepattern "
                f"training data with a relaxed 'sustained light' criterion"
            )

    d6_taus = pl.DataFrame(_rows) if _rows else pl.DataFrame(
        schema={
            "source": pl.Utf8, "track_key": pl.Utf8, "tau_min": pl.Float64,
            "asymptote": pl.Float64, "amplitude": pl.Float64, "r2": pl.Float64,
            "window_min": pl.Float64, "identifiable": pl.Boolean,
            "preceding_dose_ms": pl.Float64, "hours": pl.Float64, "optortk_expr": pl.Float64,
        }
    )
    D6_SOURCE, D6_FROM_RUN, d6_taus.height
    return D6_FALLBACK_REASON, D6_FROM_RUN, D6_SOURCE, d6_examples, d6_taus


@app.cell
def _(d6_examples, d6_taus, mo, new_fig, np, pl, plt, save_fig):
    mo.stop(
        d6_taus.is_empty(),
        mo.md("**D6 unavailable** — no decay episode passed the fit-quality threshold."),
    )
    _fig, _ax = new_fig(2, 3, figsize=(14, 7))

    # Identifiable fits (τ within their own observation window) carry the verdict;
    # the rest are drawn separately rather than dropped, so the size of the
    # extrapolated tail is visible instead of being quietly filtered away.
    _id = d6_taus.filter(pl.col("identifiable"))
    _un = d6_taus.filter(~pl.col("identifiable"))
    _tau = _id["tau_min"].to_numpy() if _id.height >= 10 else d6_taus["tau_min"].to_numpy()

    _bins = np.linspace(0, max(float(np.percentile(_tau, 99.5)), 5.0), 45)
    _ax[0, 0].hist(_tau, bins=_bins, color="#4477aa", alpha=0.8,
                   label=f"identifiable (n={len(_tau)})")
    _ax[0, 0].axvline(float(np.median(_tau)), color="#cc3311", lw=1.8,
                      label=f"median {np.median(_tau):.1f} min")
    _ax[0, 0].set_xlabel("τ (min)")
    _ax[0, 0].set_ylabel("# episodes")
    _ax[0, 0].set_title(
        f"τ_decay distribution — {_un.height} unidentifiable fits "
        f"(τ > observation window) excluded"
    )
    _ax[0, 0].legend(fontsize=8)

    for _a, _col, _xcol, _xlab, _title in (
        (_ax[0, 1], "#117733", "preceding_dose_ms", "mean preceding exposure (ms)", "τ vs preceding dose"),
        (_ax[0, 2], "#ee7733", "hours", "episode start (h)", "τ vs experiment time (a rising τ is desensitization)"),
    ):
        _a.scatter(_id[_xcol], _id["tau_min"], s=10, alpha=0.4, color=_col, label="identifiable")
        if _un.height:
            _a.scatter(_un[_xcol], _un["tau_min"], s=10, alpha=0.25, color="#bbbbbb",
                       label="unidentifiable")
        _nu = int(_id[_xcol].n_unique())
        _a.set_xlabel(_xlab)
        _a.set_ylabel("τ (min)")
        _a.set_title(_title + (f"\n(only {_nu} distinct x values — no correlation reported)" if _nu < 5 else ""),
                     fontsize=9)
        _a.legend(fontsize=7)

    # Example fits — the honest check that the exponential is the right model.
    for _i in range(3):
        _a = _ax[1, _i]
        if _i < len(d6_examples):
            _lab, _f = d6_examples[_i]
            _y = _f["y"]
            _t = np.arange(len(_y))
            _a.plot(_t, _y, "o", ms=3, color="#333333")
            _a.plot(_t, _f["A"] * np.exp(-_t / _f["tau"]) + _f["C"], lw=2.0, color="#cc3311")
            _a.set_title(f"{_lab}\nτ={_f['tau']:.1f} min, R²={_f['r2']:.2f}", fontsize=8)
        else:
            _a.set_title("—", fontsize=8)
        _a.set_xlabel("minutes into the dark window")
        _a.set_ylabel("cnr_median")

    plt.tight_layout()
    save_fig("d6_tau_decay", _fig)
    plt.gca()
    return


@app.cell
def _(
    D6_FALLBACK_REASON,
    D6_FROM_RUN,
    D6_MIN_R2,
    D6_SOURCE,
    d6_taus,
    mo,
    np,
    pl,
    spearmanr,
):
    # --- D6 verdict --------------------------------------------------------
    D6 = {"verdict": "unavailable", "reason": "", "numbers": {}}
    if d6_taus.is_empty():
        D6["reason"] = (
            D6_FALLBACK_REASON
            or "no decay episode passed the fit-quality threshold in either source"
        )
    else:
        # Verdict on the IDENTIFIABLE subset only (tau <= its own fit window);
        # the all-fits median is reported alongside so the difference is visible.
        _ident = d6_taus.filter(pl.col("identifiable"))
        _t_all = d6_taus["tau_min"].to_numpy()
        _t = _ident["tau_min"].to_numpy() if _ident.height >= 10 else _t_all
        _used_ident = bool(_ident.height >= 10)
        _med = float(np.median(_t))

        def _rho(col):
            """Spearman rho, or None when the covariate has no usable spread.

            A covariate taking two distinct values across the whole set (which is
            what `hours` degenerates to when every cell runs the same protocol)
            yields a number that looks like a correlation and is not one.
            """
            src = _ident if _used_ident else d6_taus
            v = src[col].to_numpy()
            m = np.isfinite(v) & np.isfinite(src["tau_min"].to_numpy())
            if m.sum() < 10 or len(np.unique(v[m])) < 5:
                return None
            return float(spearmanr(v[m], src["tau_min"].to_numpy()[m]).statistic)

        D6["verdict"] = f"reference_period_{4 * _med:.0f}_to_{6 * _med:.0f}_min"
        D6["numbers"] = {
            "source": D6_SOURCE,
            "n_episodes": int(len(_t)),
            "n_episodes_all_fits": int(len(_t_all)),
            "n_episodes_unidentifiable_excluded": int(len(_t_all) - _ident.height),
            "verdict_on_identifiable_subset": _used_ident,
            "median_tau_min": _med,
            "median_tau_min_all_fits": float(np.median(_t_all)),
            "median_fit_window_min": float(d6_taus["window_min"].median()),
            "iqr_tau_min": [float(np.percentile(_t, 25)), float(np.percentile(_t, 75))],
            "n_episodes_in_hold_run": int(D6_FROM_RUN),
            "reference_period_min_4x": float(4 * _med),
            "reference_period_min_6x": float(6 * _med),
            "min_r2": D6_MIN_R2,
            "spearman_tau_vs_preceding_dose": _rho("preceding_dose_ms"),
            "spearman_tau_vs_experiment_time": _rho("hours"),
            "spearman_tau_vs_optortk_expr": _rho("optortk_expr"),
            "fallback_reason": D6_FALLBACK_REASON,
        }
    _n = D6["numbers"]
    mo.md(
        f"""
    ### D6 verdict — `{D6['verdict']}`

    {f"Median τ **{_n['median_tau_min']:.1f} min** (IQR {_n['iqr_tau_min'][0]:.1f}–{_n['iqr_tau_min'][1]:.1f}) over **{_n['n_episodes']}** episodes from **{_n['source']}**. Reference period for the next experiment: **{_n['reference_period_min_4x']:.0f}–{_n['reference_period_min_6x']:.0f} min**. Spearman ρ(τ, preceding dose) = {_n['spearman_tau_vs_preceding_dose']}, ρ(τ, experiment time) = {_n['spearman_tau_vs_experiment_time']}, ρ(τ, optoRTK expr) = {_n['spearman_tau_vs_optortk_expr']} — a `None` means the covariate had too few distinct values to support a correlation at all (optoRTK was a fed constant; in the freepattern fallback every cell runs the same protocol clock)." if _n else "**Unavailable** — " + D6['reason']}

    {f"**Identifiability:** {_n['n_episodes_unidentifiable_excluded']} of {_n['n_episodes_all_fits']} fits returned a τ longer than their own {_n['median_fit_window_min']:.0f}-minute observation window. Over such a window an exponential of that τ is indistinguishable from a straight line, so those are extrapolations, not measurements, and they are excluded from the verdict. Including them would move the median to {_n['median_tau_min_all_fits']:.1f} min." if _n and _n['n_episodes_unidentifiable_excluded'] else ""}

    {("⚠️ **Source note:** " + _n["fallback_reason"]) if _n.get("fallback_reason") else ""}
    """
    )
    return (D6,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary artifact

    `tier0_summary.json` is written into the experiment directory. It is the actual
    output of this notebook: one verdict per diagnostic (or `"unavailable"` plus a
    reason), the headline numbers behind it, the inputs used, and a timestamp — plus
    an explicit `arm_4_recommendation` resolved from D3 and D5.
    """)
    return


@app.cell
def _(
    CKPT_META,
    CONTROL_HORIZON,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    EXP_DIR,
    KEYS_10H,
    KEYS_4H,
    LOG_PATH,
    OPTORTK_CONSTANT,
    OPTORTK_FED,
    POLICY_OK,
    POLICY_REASON,
    RUN_DIR,
    SERVED_META,
    TARGET,
    d6_taus,
    datetime,
    json,
    mo,
    n_dec_d1,
    n_dec_d2,
    policy_file_input,
):
    # --- arm 4, resolved from D3 and D5 -----------------------------------
    # The handoff's D3 bands overlap at H = 8: "2H (~16-20 min)" sits INSIDE
    # "15-30 min endogenous". A median landing there is genuinely not separable
    # from this run — reporting it as `long_horizon` would be an artefact of band
    # ordering, so it resolves to `undetermined` with the ambiguity named.
    _d3 = D3["verdict_psd_only"]
    if _d3 == "horizon_limited_limit_cycle":
        ARM4 = "long_horizon"
        ARM4_WHY = (
            f"D3 dominant period ≈ 2H ({2 * CONTROL_HORIZON} min) and outside the "
            f"15–30 min endogenous band; the F=20 checkpoint unblocks a longer horizon."
        )
    elif _d3 == "actuator_lag_chatter":
        ARM4 = "move_penalty"
        ARM4_WHY = "D3 dominant period 5–12 min: actuator-lag chattering, so sweep λ_Δ per cell."
    elif _d3 == "endogenous_erk_pulsing":
        ARM4 = "neither"
        ARM4_WHY = "D3 dominant period 15–30 min and outside the 2H band: endogenous pulsing; spend arm 4 elsewhere."
    elif _d3 == "ambiguous_2H_overlaps_endogenous_band":
        ARM4 = "undetermined"
        ARM4_WHY = (
            f"D3's median period lands where the 2H band (={2 * CONTROL_HORIZON} min) "
            f"overlaps the 15–30 min endogenous band. With no dark reference in this "
            f"run the two are not separable — resolve with the dark subset, not by "
            f"picking the more convenient band."
        )
    else:
        ARM4 = "undetermined"
        ARM4_WHY = f"D3 returned `{_d3}`; no mechanism is distinguished."
    if D3.get("acf_psd_same_mechanism") is False:
        ARM4 = "undetermined"
        ARM4_WHY = (
            f"D3's PSD points at `{_d3}`, but its independent ACF cross-check "
            f"implicates `{D3['acf_implies']}` instead (PSD "
            f"{D3['numbers']['median_dominant_period_min_10h']} min vs ACF "
            f"{D3['numbers']['median_acf_period_min']} min). The two estimates "
            f"disagree about the mechanism, not merely the number, so the finding "
            f"is not solid enough to spend an arm on. Resolve before committing."
        )
    elif D3.get("acf_psd_same_period") is False:
        ARM4_WHY += (
            f" The mechanism is confirmed by both estimates, but the exact period is "
            f"not: PSD {D3['numbers']['median_dominant_period_min_10h']} min vs ACF "
            f"{D3['numbers']['median_acf_period_min']} min is a harmonic relationship. "
            f"Size the arm from the mechanism, not from that number."
        )
    if D3.get("verdict_changes_with_4h_set"):
        ARM4_WHY += (
            f" NOTE: the ≥4 h track set gives `{D3['verdict_4h']}` instead, so the "
            f"≥10 h result is partly a survivorship effect."
        )

    SUMMARY = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "arm_4_recommendation": ARM4,
        "arm_4_rationale": ARM4_WHY,
        "inputs": {
            "run_dir": str(RUN_DIR),
            "log": str(LOG_PATH),
            "policy_file": policy_file_input.value.strip(),
            "policy_loaded": POLICY_OK,
            "policy_reason": POLICY_REASON,
            "checkpoints": [
                {k: v for k, v in m.items() if k not in ("norm_mean", "norm_std")}
                for m in CKPT_META
            ],
            "served_checkpoint": SERVED_META["name"] if SERVED_META else None,
            "target_cnr": TARGET,
            "control_horizon": CONTROL_HORIZON,
            "n_tracks_10h": len(KEYS_10H),
            "n_tracks_4h": len(KEYS_4H),
            "n_decisions_d1": int(n_dec_d1.value),
            "n_probes_d2": int(n_dec_d2.value),
            "optortk_fed_online": OPTORTK_FED,
            "optortk_was_constant_online": OPTORTK_CONSTANT,
        },
        "diagnostics": {"D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5, "D6": D6},
    }
    (EXP_DIR / "tier0_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))

    # The per-episode τ fits, not just their median. `experiments/policy_preflight.py`
    # needs the DISTRIBUTION: a reference whose fall is feasible at the median τ can
    # still be untrackable for the slow half of the cells, and that fraction is the
    # number a policy has to be argued against. The summary carries only p50 and the
    # IQR, which cannot answer it.
    d6_taus.write_parquet(EXP_DIR / "tier0_tau_episodes.parquet")

    _rows = "\n".join(
        f"| **{k}** | `{v['verdict']}` | {v.get('reason') or '—'} |"
        for k, v in SUMMARY["diagnostics"].items()
    )
    mo.md(
        f"""
    ### Verdicts

    | diagnostic | verdict | note |
    |---|---|---|
    {_rows}

    ### `arm_4_recommendation` = `{ARM4}`

    {ARM4_WHY}

    Written to `{EXP_DIR / "tier0_summary.json"}`; figures under `{EXP_DIR / "figures"}`.
    """
    )
    return


if __name__ == "__main__":
    app.run()
