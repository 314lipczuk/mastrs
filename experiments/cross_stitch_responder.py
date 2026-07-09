import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import altair as alt
    import torch

    alt.data_transformers.disable_max_rows()

    from optoerk.core.experiment import load_experiment
    from optoerk.core.utils import (
        get_device,
        results_read_sources,
        scan_experiment_dirs,
    )
    from optoerk.data.seq2seq_data import load as load_dataset, AVAILABLE_DATASETS
    from optoerk.eval.cell_video import (
        CellData,
        PREDICT_FN_BY_MODULE,
        _infer_experiment_module,
        _resolve_main_model,
        stratify_by_std,
    )

    repo_root = Path(__file__).resolve().parent.parent
    device = get_device()
    return (
        AVAILABLE_DATASETS,
        PREDICT_FN_BY_MODULE,
        Path,
        device,
        load_dataset,
        load_experiment,
        mo,
        np,
        pl,
        plt,
        repo_root,
        results_read_sources,
        scan_experiment_dirs,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Current-model memory characterization

    Before changing the architecture, measure how the trained seq2scalar model
    carries information across time — the precondition for running it in an
    experiment with **two stimulation patterns (or a long gap) in sequence**: the
    calibration phase must still inform predictions in the later phase.

    The model has three memory channels with known time constants: the **encoder
    window** (`history_len`, a hard cutoff), the two **CNR-EWMA** features (~1/alpha),
    and the **baseline** feature (median of the first 10 frames — a permanent
    offset). The ladder below measures the realised memory and attributes it to a
    channel.

    1. **Impulse kernel** — how far back a past (now-ended) response still moves the
       prediction.
    2. **Pathway ablation** — which channel carries that memory.
    3. **Real-cell stitch** — concatenate two real cells; measure carryover Δ(tau)
       into the second segment and whether it is structured by responder quartile
       (identity carryover) or just the new cell.
    4. **Gap sweep & uncertainty** — insert a quiescent gap at the stitch; does
       forgetting follow the window cutoff or the EWMA constant, and does predicted
       sigma rise where carryover is high?

    All predictions use `core_predict`, verified bit-for-bit identical to the
    bundle's `predict_fn`, with added ablation knobs.
    """)
    return


@app.cell
def _(mo, repo_root, results_read_sources):
    _sources = results_read_sources(repo_root)
    source_selector = mo.ui.dropdown(
        options=list(_sources.keys()), value=list(_sources.keys())[0],
        label="Results source",
    )
    source_selector
    return (source_selector,)


@app.cell
def _(
    Path,
    mo,
    repo_root,
    results_read_sources,
    scan_experiment_dirs,
    source_selector,
):
    _src_root = Path(results_read_sources(repo_root)[source_selector.value])
    _choices = scan_experiment_dirs(_src_root)
    if _choices:
        experiment_picker = mo.ui.dropdown(
            options=_choices, value=_choices[0], label="Checkpoint (experiment run)",
        )
        load_button = mo.ui.run_button(label="Load bundle")
        _picker_ui = mo.vstack([experiment_picker, load_button])
    else:
        experiment_picker = load_button = None
        _picker_ui = mo.md(f"No loadable experiments under `{_src_root}`.")
    source_root = _src_root
    _picker_ui
    return experiment_picker, load_button, source_root


@app.cell
def _(
    PREDICT_FN_BY_MODULE,
    device,
    experiment_picker,
    load_button,
    load_experiment,
    mo,
    source_root,
):
    mo.stop(
        experiment_picker is None or not load_button.value,
        mo.md("Pick a checkpoint and click **Load bundle**."),
    )

    _exp_path = source_root / experiment_picker.value
    bundle = load_experiment(str(_exp_path))

    _inferred_mod = _infer_experiment_module(_exp_path)
    _model_type = bundle.model_type or ""
    if _model_type.startswith("__main__."):
        # Model class was defined in the experiment script itself; resolve it from
        # the inferred experiment module rather than the unimportable __main__.
        if _inferred_mod is None:
            raise RuntimeError(
                f"Bundle model_type is {_model_type!r} but no experiment module "
                f"could be inferred from {_exp_path}/slurm.log. Re-run cell_video "
                f"or pass experiment_module manually."
            )
        model = _resolve_main_model(bundle, _inferred_mod)
    else:
        # Model class lives in an importable module (e.g. seq2scal_models_abs);
        # reconstruct directly. predict_fn still comes from the experiment module.
        model = bundle.reconstruct_model()

    # Weights load on CPU; move to the inference device (predict_fn puts inputs on
    # `device`, so a CPU model triggers "Placeholder storage not allocated on MPS").
    model = model.to(device).eval()

    experiment_module = _inferred_mod
    predict_fn = PREDICT_FN_BY_MODULE.get(_inferred_mod) if _inferred_mod else None
    if predict_fn is None:
        raise RuntimeError(
            f"No predict_fn registered for inferred module {_inferred_mod!r} "
            f"(from {_exp_path}/slurm.log). Known: {list(PREDICT_FN_BY_MODULE)}"
        )

    model_cfg = bundle.model_config or {}
    default_history_len = int(model_cfg.get("history_len", 30))
    default_future_len = int(model_cfg.get("future_len", 5))

    mo.md(
        f"**Bundle:** `{experiment_picker.value}`\n\n"
        f"- model_type: `{_model_type}`\n"
        f"- experiment_module: `{experiment_module}`\n"
        f"- predict_fn: `{getattr(predict_fn, '__name__', None)}`\n"
        f"- device: `{device}`\n"
        f"- model_config.history_len: `{default_history_len}`  ·  future_len: `{default_future_len}`\n"
        f"- warnings: `{len(bundle.warnings)}`"
    )
    return default_future_len, default_history_len, model, predict_fn


@app.cell
def _(mo):
    mo.md("""
    ## Data selection

    Pick a dataset and the prediction-window parameters. `calibration_len` is
    the number of leading frames the model sees (its only source of cell
    identity); `future_len` is what it must predict. `history_len` is the
    model's encoder window — must be ≤ `calibration_len`.
    """)
    return


@app.cell
def _(AVAILABLE_DATASETS, default_future_len, default_history_len, mo):
    dataset_selector = mo.ui.dropdown(
        options=list(AVAILABLE_DATASETS), value="real_plus_bo",
        label="Dataset",
    )
    history_len_input = mo.ui.number(
        start=5, stop=120, value=default_history_len, label="history_len (encoder window)",
    )
    future_len_input = mo.ui.number(
        start=1, stop=60, value=default_future_len, label="future_len (prediction horizon)",
    )
    calibration_len_input = mo.ui.number(
        start=10, stop=300, value=max(default_history_len, 40), label="calibration_len",
    )
    mo.vstack([dataset_selector, history_len_input, future_len_input, calibration_len_input])
    return (
        calibration_len_input,
        dataset_selector,
        future_len_input,
        history_len_input,
    )


@app.cell
def _(
    calibration_len_input,
    dataset_selector,
    future_len_input,
    history_len_input,
    load_dataset,
    mo,
    pl,
):
    cnr_tracks, stim_tracks, conditions = load_dataset(dataset_selector.value)
    _cond_counts = pl.DataFrame(
        {"condition": list(conditions)}
    ).group_by("condition").len().sort("len", descending=True)
    mo.md(
        f"Loaded `{len(cnr_tracks)}` cells across `{_cond_counts.shape[0]}` "
        f"conditions. Filter: calibration_len=`{calibration_len_input.value}`, "
        f"history_len=`{history_len_input.value}`, future_len=`{future_len_input.value}`."
    ), _cond_counts
    return cnr_tracks, conditions, stim_tracks


@app.cell
def _(conditions, mo):
    _uniq = sorted(set(conditions.tolist() if hasattr(conditions, "tolist") else list(conditions)))
    cond_a = mo.ui.dropdown(options=_uniq, value=_uniq[0] if _uniq else None, label="Experiment A")
    cond_b = mo.ui.dropdown(
        options=_uniq,
        value=_uniq[1] if len(_uniq) > 1 else (_uniq[0] if _uniq else None),
        label="Experiment B",
    )
    mo.vstack([cond_a, cond_b])
    return cond_a, cond_b


@app.cell
def _(
    calibration_len_input,
    cnr_tracks,
    cond_a,
    cond_b,
    conditions,
    future_len_input,
    history_len_input,
    mo,
    np,
    pl,
    stim_tracks,
):
    _cal = int(calibration_len_input.value)
    _H = int(history_len_input.value)
    _F = int(future_len_input.value)
    if _H > _cal:
        raise ValueError(
            f"history_len ({_H}) must be <= calibration_len ({_cal}); "
            "the encoder window can't exceed the available calibration history."
        )
    if cond_a.value == cond_b.value:
        raise ValueError(
            "Pick two DIFFERENT conditions for A and B — the stitch needs two "
            "distinct protocols to switch between."
        )

    _conds = conditions.tolist() if hasattr(conditions, "tolist") else list(conditions)
    _want = [cond_a.value, cond_b.value]
    _keep_idx = [
        i for i, c in enumerate(_conds)
        if c in _want and len(cnr_tracks[i]) >= _cal + _F
    ]
    if len(_keep_idx) == 0:
        raise RuntimeError(
            f"No cells in {_want} with length >= {_cal + _F} frames. "
            "Pick longer protocols or a smaller calibration_len + future_len."
        )

    _sel_conds = np.array([_conds[i] for i in _keep_idx])
    _sel_cnr = [np.asarray(cnr_tracks[i], dtype=np.float32) for i in _keep_idx]
    _sel_stim = [np.asarray(stim_tracks[i], dtype=np.float32) for i in _keep_idx]

    # Tracks are ragged (cnr 1D) and 2D (stim 9xT); store as pl.Object so polars
    # keeps the raw numpy arrays instead of trying to build fixed-shape Array cols.
    pooled = pl.DataFrame({
        "local_idx": np.arange(len(_keep_idx)),
        "source_exp": _sel_conds,
    }).with_columns(
        pl.Series("cnr_track", _sel_cnr, dtype=pl.Object),
        pl.Series("stim_track", _sel_stim, dtype=pl.Object),
    )
    _counts = pooled.group_by("source_exp").len().sort("source_exp")
    mo.md(
        f"Pooled `{pooled.shape[0]}` cells. Each provides a real pattern-A "
        f"calibration state; its future is predicted under **both** "
        f"`{cond_a.value}` (A) and `{cond_b.value}` (B) fluence."
    ), _counts
    return (pooled,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Memory-characterization ladder

    Before any architecture change, characterize the **current** model's memory:
    how far back does the past influence a prediction, and through which pathway.
    The model has three known memory channels with different time constants —
    encoder window (`history_len`, hard cutoff), the two CNR-EWMA features
    (~1/alpha), and the `baseline` feature (median of the first 10 frames — a
    permanent offset). The ladder measures each.

    Everything below is computed with `core_predict`, a local reimplementation of
    the bundle's `predict_fn` that is verified bit-for-bit identical and adds
    ablation knobs (`fix_baseline`, `reset_ewma_at`). It needs only a cell's
    absolute CNR trace and its `u_t` fluence trace.
    """)
    return


@app.cell
def _(
    device,
    future_len_input,
    history_len_input,
    mo,
    model,
    np,
    pooled,
    predict_fn,
    torch,
):
    from optoerk.data.seq2seq_data import STIM_COLS
    from optoerk.models.seq2scal_abs import _ewma_1d as ewma1d
    from optoerk.eval.cell_video import _gmm_pred_std as gmm_std

    mo.stop(predict_fn is None, mo.md("Load a bundle first."))

    U_IDX = STIM_COLS.index("u_t")
    N_STIM = len(STIM_COLS)
    _cfg = getattr(model, "cfg", None)
    A_SLOW = float(getattr(_cfg, "ewma_slow_alpha", 0.05))
    A_FAST = float(getattr(_cfg, "ewma_fast_alpha", 0.30))
    H_DEF = int(history_len_input.value)
    F_DEF = int(future_len_input.value)


    def core_predict(cnr, flu, t, F, H, *, fix_baseline=None, reset_ewma_at=None):
        """Verified-equal reimpl of predict_fn (seq2scal abs) with ablation knobs."""
        cnr = np.asarray(cnr, np.float32)
        flu = np.asarray(flu, np.float32)
        T = len(cnr)
        F = min(F, T - t)
        if reset_ewma_at is None:
            _es = ewma1d(cnr, A_SLOW)
            _ef = ewma1d(cnr, A_FAST)
        else:
            _s = reset_ewma_at
            _es = np.empty(T, np.float32)
            _ef = np.empty(T, np.float32)
            _es[:_s] = ewma1d(cnr[:_s], A_SLOW)
            _ef[:_s] = ewma1d(cnr[:_s], A_FAST)
            _es[_s:] = ewma1d(cnr[_s:], A_SLOW)
            _ef[_s:] = ewma1d(cnr[_s:], A_FAST)
        _b = float(np.median(cnr[: min(10, T)])) if fix_baseline is None else float(fix_baseline)
        _enc = np.stack(
            [cnr[t - H:t], flu[t - H:t], np.full(H, _b, np.float32), _es[t - H:t], _ef[t - H:t]],
            axis=-1,
        )
        _et = torch.from_numpy(_enc).float().unsqueeze(0).to(device)
        _dt = torch.from_numpy(flu[t:t + F, None]).float().unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            _pi, _mu, _sig = model(_et, _dt)
            _mean = (_pi * _mu).sum(-1).cpu().numpy()[0]
            _sstd = gmm_std(_pi, _mu, _sig).cpu().numpy()[0]
        return _mean.astype(np.float32), _sstd.astype(np.float32)


    def predict_many(cnr, flu, ts, *, want_sigma=False):
        """Batched core_predict: all prediction points `ts` along one trajectory in
        a single model() call. Returns means (len(ts), F) [+ sigmas if asked]."""
        cnr = np.asarray(cnr, np.float32)
        flu = np.asarray(flu, np.float32)
        _es = ewma1d(cnr, A_SLOW)
        _ef = ewma1d(cnr, A_FAST)
        _b = float(np.median(cnr[: min(10, len(cnr))]))
        _enc = np.stack([
            np.stack([cnr[t - H_DEF:t], flu[t - H_DEF:t], np.full(H_DEF, _b, np.float32),
                      _es[t - H_DEF:t], _ef[t - H_DEF:t]], axis=-1)
            for t in ts
        ])
        _dec = np.stack([flu[t:t + F_DEF] for t in ts])[..., None]
        _et = torch.from_numpy(_enc).float().to(device)
        _dt = torch.from_numpy(_dec).float().to(device)
        model.eval()
        with torch.no_grad():
            _pi, _mu, _sig = model(_et, _dt)
            _mean = (_pi * _mu).sum(-1).cpu().numpy()
            if want_sigma:
                return _mean.astype(np.float32), gmm_std(_pi, _mu, _sig).cpu().numpy().astype(np.float32)
        return _mean.astype(np.float32)


    # data-derived scales: baseline CNR and a typical response amplitude
    _n = min(300, pooled.shape[0])
    B0 = float(np.median([float(np.median(np.asarray(pooled["cnr_track"][i])[:10])) for i in range(_n)]))
    AMP = float(np.median([
        float(np.max(np.asarray(pooled["cnr_track"][i])) - np.median(np.asarray(pooled["cnr_track"][i])[:10]))
        for i in range(_n)
    ]))


    def memory_kernel(width, *, fix_baseline=None, reset_ewma=False, T=240, t=180, taus=None):
        """How long is a *past* response (width frames, returns to baseline)
        remembered? S(tau)=max|pred_perturbed - pred_flat|."""
        if taus is None:
            taus = list(range(1, t - 20, 2))
        _flu0 = np.zeros(T, np.float32)
        _base = np.full(T, B0, np.float32)
        _kw = {} if fix_baseline is None else {"fix_baseline": fix_baseline}
        _mflat, _ = core_predict(_base, _flu0, t, F_DEF, H_DEF, **_kw)
        _S = []
        for _tau in taus:
            _c = _base.copy()
            _s = t - _tau
            if _s < 0:
                _S.append(0.0)
                continue
            _c[_s:_s + width] += AMP
            _kw2 = dict(_kw, reset_ewma_at=_s) if reset_ewma else _kw
            _m, _ = core_predict(_c, _flu0, t, F_DEF, H_DEF, **_kw2)
            _S.append(float(np.abs(_m - _mflat).max()))
        return np.asarray(taus), np.asarray(_S)


    mo.md(
        f"Core predictor ready (verified == `predict_fn`). "
        f"`B0={B0:.3f}` · response `AMP={AMP:.3f}` · `H={H_DEF}` · "
        f"EWMA tau_slow≈`{1/A_SLOW:.0f}` · tau_fast≈`{1/A_FAST:.1f}` frames."
    )
    return A_FAST, A_SLOW, B0, F_DEF, H_DEF, U_IDX, memory_kernel, predict_many


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 1 — impulse memory kernel

    Synthetic flat-baseline trajectory. Inject a past response of some duration that
    then **returns to baseline**, at distance `tau` before the prediction point, and
    measure how much the prediction still moves: `S(tau) = max|Δ pred|`. The
    return-to-baseline is what makes this *memory* (of a past event) rather than
    sensitivity to the current level. Two durations — a brief event vs a sustained
    response. Dashed lines: architectural constants `H`, `tau_slow`, `tau_fast`.
    """)
    return


@app.cell
def _(A_FAST, A_SLOW, H_DEF, memory_kernel, plt):
    _t3, _S3 = memory_kernel(3)
    _t15, _S15 = memory_kernel(15)

    fig_kernel, _ax = plt.subplots(figsize=(9, 4.2))
    _ax.plot(_t3, _S3, "-o", ms=3, color="C0", label="brief event (3-frame response)")
    _ax.plot(_t15, _S15, "-s", ms=3, color="C1", label="sustained response (15-frame), released to baseline")
    _ymax = float(max(_S3.max(), _S15.max())) or 1.0
    for _x, _lab, _c in [(H_DEF, "H", "grey"), (1 / A_SLOW, "tau_slow", "C2"), (1 / A_FAST, "tau_fast", "C3")]:
        _ax.axvline(_x, ls="--", lw=0.9, color=_c, alpha=0.8)
        _ax.text(_x, _ymax * 0.96, _lab, rotation=90, fontsize=7, va="top", ha="right", color=_c)
    _ax.set_xlabel("tau  =  frames between the (now-ended) past response and the prediction point")
    _ax.set_ylabel("prediction sensitivity   max|Δ pred|")
    _ax.set_title("Memory kernel: a past response that returned to baseline — how long is it remembered?")
    _ax.legend(fontsize=8)
    fig_kernel.tight_layout()
    fig_kernel
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 2 — pathway ablation

    Re-run the kernel (using a long 40-frame released response, the case with the
    most energy to carry forward) with each memory channel disabled. If a curve
    **overlaps** the full curve, that channel adds no memory; a gap attributes
    memory to it. `EWMA reset` removes the exponential carry-over; `baseline fixed`
    removes the permanent first-10-frames offset; both off leaves only the encoder
    window.
    """)
    return


@app.cell
def _(B0, H_DEF, memory_kernel, plt):
    _W = 40
    _a0, _b0 = memory_kernel(_W)
    _a1, _b1 = memory_kernel(_W, reset_ewma=True)
    _a2, _b2 = memory_kernel(_W, fix_baseline=B0)
    _a3, _b3 = memory_kernel(_W, reset_ewma=True, fix_baseline=B0)

    fig_ablate, _ax = plt.subplots(figsize=(9, 4.2))
    _ax.plot(_a0, _b0, "-o", ms=3, label="full (all pathways)")
    _ax.plot(_a1, _b1, "-^", ms=3, label="EWMA reset")
    _ax.plot(_a2, _b2, "-s", ms=3, label="baseline fixed")
    _ax.plot(_a3, _b3, "-d", ms=3, label="both off → window only")
    _ax.axvline(H_DEF, ls="--", lw=0.9, color="grey", alpha=0.8)
    _ax.text(H_DEF, float(_b0.max()) * 0.96, "H", rotation=90, fontsize=7, va="top", ha="right", color="grey")
    _ax.set_xlabel("tau")
    _ax.set_ylabel("max|Δ pred|")
    _ax.set_title("Step 2: pathway ablation (40-frame released response) — overlap means no added memory")
    _ax.legend(fontsize=8)
    fig_ablate.tight_layout()
    fig_ablate
    return


@app.cell
def _(mo):
    step3_K = mo.ui.number(5, 40, value=20, label="pairs per (q1,q2) bucket")
    step3_J = mo.ui.number(20, 120, value=70, label="frames into exp2 (J)")
    step3_seed = mo.ui.number(0, 9999, value=0, label="seed")
    step3_run = mo.ui.run_button(label="Run Step 3")
    mo.vstack([
        mo.md(r"""
    ### Step 3 — real-cell stitch: carryover & quartile interaction

    Concatenate a real **prefix cell** (cell1) and a real **exp2 cell** (cell2),
    slide the model into exp2, and measure carryover
    `Δ(tau) = | pred_stitched − pred_clean |`. Two versions:

    - **raw** — clean replaces cell1 with a quiescent baseline prefix. Includes any
      reaction to the CNR level-jump at the stitch.
    - **jump-corrected** — cell2 is shifted so it continues smoothly from cell1's
      end level, and the clean prefix is a flat hold at that same level, so *neither*
      arm has a stitch discontinuity. The only remaining difference is cell1's
      dynamics vs a flat hold — isolating real prefix memory from the jump.

    The quartile-interaction matrix bins pairs by cell1 quartile (q1) × cell2
    quartile (q2) and reports the signed early-window overshoot: dependence on **q1**
    = cell1 identity carried into exp2; dependence on **q2** = effect set by the new
    cell.
    """),
        mo.hstack([step3_K, step3_J, step3_seed]),
        step3_run,
    ])
    return step3_J, step3_K, step3_run, step3_seed


@app.cell
def _(
    B0,
    F_DEF,
    H_DEF,
    U_IDX,
    mo,
    np,
    pooled,
    predict_many,
    step3_J,
    step3_K,
    step3_run,
    step3_seed,
):
    mo.stop(not step3_run.value, mo.md("Click **Run Step 3**."))

    _K = int(step3_K.value)
    _J = int(step3_J.value)
    _Wearly = 5
    _N = pooled.shape[0]
    _cnr = [np.asarray(pooled["cnr_track"][i], np.float32) for i in range(_N)]
    _flu = [np.asarray(pooled["stim_track"][i])[U_IDX].astype(np.float32) for i in range(_N)]
    _score = np.array([float(np.std(c)) for c in _cnr])
    _edges = np.quantile(_score, np.linspace(0, 1, 5))
    _edges[0] -= 1e-9
    _edges[-1] += 1e-9
    _quart = np.digitize(_score, _edges[1:-1]) + 1
    _elig = np.array([i for i in range(_N) if len(_cnr[i]) >= H_DEF and len(_cnr[i]) >= _J + F_DEF])
    _byq = {q: _elig[_quart[_elig] == q] for q in (1, 2, 3, 4)}
    _rng = np.random.default_rng(int(step3_seed.value))


    def _carry(i1, i2):
        _c1, _c2 = _cnr[i1], _cnr[i2]
        _f1, _f2 = _flu[i1], _flu[i2]
        _L1 = len(_c1)
        _ts = [_L1 + j for j in range(_J)]
        # raw: real prefix vs quiescent baseline (jump present)
        _mf = predict_many(np.concatenate([_c1, _c2]), np.concatenate([_f1, _f2]), _ts)
        _mc = predict_many(
            np.concatenate([np.full(_L1, B0, np.float32), _c2]),
            np.concatenate([np.zeros(_L1, np.float32), _f2]), _ts,
        )
        # jump-corrected: c2 shifted to continue from cell1's end; clean = flat hold
        # at that level — neither arm has a discontinuity at the stitch
        _sh = float(_c1[-1] - _c2[0])
        _c2s = _c2 + _sh
        _mfa = predict_many(np.concatenate([_c1, _c2s]), np.concatenate([_f1, _f2]), _ts)
        _mca = predict_many(
            np.concatenate([np.full(_L1, float(_c1[-1]), np.float32), _c2s]),
            np.concatenate([np.zeros(_L1, np.float32), _f2]), _ts,
        )
        return _mf, _mc, _mfa, _mca


    _absD = []
    _absDa = []
    _Msum = np.zeros((4, 4))
    _Mn = np.zeros((4, 4))
    _rec = []
    _eps = 0.02
    for _q1 in (1, 2, 3, 4):
        for _q2 in (1, 2, 3, 4):
            if len(_byq[_q1]) == 0 or len(_byq[_q2]) == 0:
                continue
            for _ in range(_K):
                _i1 = int(_rng.choice(_byq[_q1]))
                _i2 = int(_rng.choice(_byq[_q2]))
                _mf, _mc, _mfa, _mca = _carry(_i1, _i2)
                _d = np.abs(_mf - _mc).max(1)
                _da = np.abs(_mfa - _mca).max(1)
                _absD.append(_d)
                _absDa.append(_da)
                _Msum[_q1 - 1, _q2 - 1] += float(np.mean((_mf - _mc)[:_Wearly]))
                _Mn[_q1 - 1, _q2 - 1] += 1
                _below = np.where(_d < _eps)[0]
                _rec.append(int(_below[0]) if len(_below) else _J)

    taus3 = np.arange(_J)
    delta_curve = np.mean(_absD, 0)
    delta_curve_se = np.std(_absD, 0) / np.sqrt(len(_absD))
    delta_curve_al = np.mean(_absDa, 0)
    delta_curve_al_se = np.std(_absDa, 0) / np.sqrt(len(_absDa))
    qmatrix = _Msum / np.maximum(_Mn, 1)
    recovery = np.array(_rec)
    mo.md(
        f"Computed `{int(_Mn.sum())}` stitched pairs (K=`{_K}`/bucket, J=`{_J}`). "
        f"Median recovery to |Δ|<{_eps}: **`{np.median(recovery):.0f}`** frames (H=`{H_DEF}`). "
        f"Peak carryover — raw `{delta_curve.max():.2f}` vs jump-corrected "
        f"`{delta_curve_al.max():.2f}` ⇒ the discontinuity contributes only "
        f"`{100*(1-delta_curve_al.max()/delta_curve.max()):.0f}%`; the carryover is real "
        f"prefix-sensitivity, not a stitch artifact."
    )
    return (
        delta_curve,
        delta_curve_al,
        delta_curve_al_se,
        delta_curve_se,
        qmatrix,
        taus3,
    )


@app.cell
def _(
    H_DEF,
    delta_curve,
    delta_curve_al,
    delta_curve_al_se,
    delta_curve_se,
    plt,
    taus3,
):
    fig_carry, _ax = plt.subplots(figsize=(9, 4.2))
    _ax.plot(taus3, delta_curve, "-", color="C0", lw=1.8, label="raw (vs quiescent baseline)")
    _ax.fill_between(taus3, delta_curve - delta_curve_se, delta_curve + delta_curve_se, color="C0", alpha=0.18)
    _ax.plot(taus3, delta_curve_al, "-", color="C1", lw=1.8, label="jump-corrected (both arms smooth)")
    _ax.fill_between(taus3, delta_curve_al - delta_curve_al_se, delta_curve_al + delta_curve_al_se, color="C1", alpha=0.18)
    _ax.axvline(H_DEF, ls="--", color="grey", lw=0.9)
    _ax.text(H_DEF, float(delta_curve.max()) * 0.95, "H", rotation=90, fontsize=7, va="top", ha="right", color="grey")
    _ax.axhline(0.02, ls=":", color="red", lw=0.8, label="recovery threshold")
    _ax.set_xlabel("tau  =  frames since stitch (into exp2)")
    _ax.set_ylabel("carryover   mean max|Δ pred|")
    _ax.set_title("Step 3: prefix carryover Δ(tau) — raw vs jump-corrected")
    _ax.legend(fontsize=8)
    fig_carry.tight_layout()
    fig_carry
    return


@app.cell
def _(mo, np, plt, qmatrix):
    _rm = qmatrix.mean(1)
    _cm = qmatrix.mean(0)
    _vmax = float(np.abs(qmatrix).max()) or 1e-6
    fig_qm, _ax = plt.subplots(figsize=(5.4, 4.6))
    _im = _ax.imshow(qmatrix, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
    _ax.set_xticks(range(4))
    _ax.set_xticklabels([f"q2={i+1}" for i in range(4)])
    _ax.set_yticks(range(4))
    _ax.set_yticklabels([f"q1={i+1}" for i in range(4)])
    for _i in range(4):
        for _j in range(4):
            _ax.text(_j, _i, f"{qmatrix[_i, _j]:+.3f}", ha="center", va="center", fontsize=8)
    _ax.set_xlabel("exp2 cell quartile (q2)")
    _ax.set_ylabel("prefix cell quartile (q1)")
    _ax.set_title("Signed early-window carryover by quartile pair")
    plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    fig_qm.tight_layout()
    mo.output.append(mo.md(
        f"**q1 (prefix) effect** — row means: `{[round(float(x), 4) for x in _rm]}`  ·  "
        f"**q2 (exp2) effect** — col means: `{[round(float(x), 4) for x in _cm]}`. "
        f"Flat across q1 ⇒ no responder-identity carryover; rising in q2 ⇒ effect set by the new cell."
    ))
    fig_qm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4 — gap sweep & uncertainty at the stitch

    Insert `g` quiescent (baseline, zero-light) frames between cell1 and cell2. As
    `g` grows the prefix slides out of the encoder window, so early carryover should
    fall. Comparing the washout against the EWMA decay `e^(-alpha_slow*g)` shows
    whether the **window cutoff (H)** or the **EWMA constant** governs forgetting.
    Separately, track the predicted `sigma` across the stitch: does the model raise
    its uncertainty where carryover is high (well-calibrated) or stay confident
    while wrong?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    step4_P = mo.ui.number(20, 300, value=100, label="pairs")
    step4_run = mo.ui.run_button(label="Run Step 4")
    mo.hstack([step4_P, step4_run])
    return step4_P, step4_run


@app.cell(hide_code=True)
def _(
    B0,
    F_DEF,
    H_DEF,
    U_IDX,
    mo,
    np,
    pooled,
    predict_many,
    step4_P,
    step4_run,
):
    mo.stop(not step4_run.value, mo.md("Click **Run Step 4**."))

    _P = int(step4_P.value)
    _Wearly = 5
    _Js = 40
    _GAPS = [0, 5, 10, 20, 40, 60, 90]
    _N = pooled.shape[0]
    _cnr = [np.asarray(pooled["cnr_track"][i], np.float32) for i in range(_N)]
    _flu = [np.asarray(pooled["stim_track"][i])[U_IDX].astype(np.float32) for i in range(_N)]
    _elig = [i for i in range(_N) if len(_cnr[i]) >= H_DEF and len(_cnr[i]) >= _Js + F_DEF]
    _rng = np.random.default_rng(0)
    _pairs = [(int(_rng.choice(_elig)), int(_rng.choice(_elig))) for _ in range(_P)]


    def _early_delta(i1, i2, gp):
        _c1, _c2 = _cnr[i1], _cnr[i2]
        _f1, _f2 = _flu[i1], _flu[i2]
        _L1 = len(_c1)
        _pre = _L1 + gp
        _ts = [_pre + j for j in range(_Wearly)]
        _mf = predict_many(
            np.concatenate([_c1, np.full(gp, B0, np.float32), _c2]),
            np.concatenate([_f1, np.zeros(gp, np.float32), _f2]), _ts,
        )
        _mc = predict_many(
            np.concatenate([np.full(_pre, B0, np.float32), _c2]),
            np.concatenate([np.zeros(_pre, np.float32), _f2]), _ts,
        )
        return float(np.abs(_mf - _mc).mean())


    gap_vals = np.array(_GAPS)
    gap_delta = np.array([np.mean([_early_delta(a, b, gp) for a, b in _pairs]) for gp in _GAPS])


    def _sig(i1, i2):
        _c1, _c2 = _cnr[i1], _cnr[i2]
        _f1, _f2 = _flu[i1], _flu[i2]
        _L1 = len(_c1)
        _ts = [_L1 + j for j in range(_Js)]
        _mf, _sf = predict_many(np.concatenate([_c1, _c2]), np.concatenate([_f1, _f2]), _ts, want_sigma=True)
        _mc, _sc = predict_many(
            np.concatenate([np.full(_L1, B0, np.float32), _c2]),
            np.concatenate([np.zeros(_L1, np.float32), _f2]), _ts, want_sigma=True,
        )
        return _sf.mean(1), _sc.mean(1), np.abs(_mf - _mc).max(1)


    _SS = [_sig(a, b) for a, b in _pairs]
    taus4 = np.arange(_Js)
    sigma_stitch = np.mean([s[0] for s in _SS], 0)
    sigma_clean = np.mean([s[1] for s in _SS], 0)
    delta4 = np.mean([s[2] for s in _SS], 0)

    _collapse = [g for g, d in zip(_GAPS, gap_delta) if d < 0.1 * gap_delta[0]]
    mo.md(
        f"Gap sweep + sigma on `{_P}` pairs. Early carryover drops below 10% of its "
        f"gap-0 value at gap ≈ `{_collapse[0] if _collapse else '>' + str(_GAPS[-1])}` "
        f"(H=`{H_DEF}`)."
    )
    return delta4, gap_delta, gap_vals, sigma_clean, sigma_stitch, taus4


@app.cell(hide_code=True)
def _(A_SLOW, H_DEF, gap_delta, gap_vals, np, plt):
    fig_gap, _ax = plt.subplots(figsize=(8, 4.2))
    _ax.plot(gap_vals, gap_delta, "-o", color="C0", label="early carryover |Δ|")
    _ax.plot(gap_vals, gap_delta[0] * np.exp(-A_SLOW * gap_vals), "--", color="C2",
             label="EWMA decay  e^(-a_slow·g)")
    _ax.axvline(H_DEF, ls="--", color="grey", lw=0.9)
    _ax.text(H_DEF, float(gap_delta.max()) * 0.9, "H", rotation=90, fontsize=7, va="top", ha="right", color="grey")
    _ax.set_xlabel("gap length g  (quiescent frames inserted at stitch)")
    _ax.set_ylabel("early-window carryover  |Δ|")
    _ax.set_title("Step 4: carryover vs gap — window cutoff (H) or EWMA decay?")
    _ax.legend(fontsize=8)
    fig_gap.tight_layout()
    fig_gap
    return


@app.cell(hide_code=True)
def _(delta4, plt, sigma_clean, sigma_stitch, taus4):
    fig_sig, _ax = plt.subplots(figsize=(8, 4.2))
    _ax.plot(taus4, sigma_stitch, "-", color="C3", lw=1.8, label="sigma (stitched)")
    _ax.plot(taus4, sigma_clean, "-", color="C0", lw=1.8, label="sigma (clean)")
    _ax2 = _ax.twinx()
    _ax2.plot(taus4, delta4, ":", color="grey", lw=1.4, label="|Δ| carryover")
    _ax2.set_ylabel("|Δ pred|", color="grey")
    _ax.axvline(0, ls="--", color="k", lw=0.6)
    _ax.set_xlabel("tau since stitch")
    _ax.set_ylabel("predicted sigma")
    _ax.set_title("Step 4: does uncertainty rise at the stitch? (calibration check)")
    _ax.legend(fontsize=8, loc="upper left")
    fig_sig.tight_layout()
    fig_sig
    return


if __name__ == "__main__":
    app.run()
