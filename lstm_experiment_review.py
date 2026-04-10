import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns", app_title="LSTM Experiment explorer")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import base64
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import subprocess
    import altair as alt
    import polars as pl
    from experiment import ExperimentBundle, discover_experiments, load_experiment_group
    from interactive_viz import get_views
    from experiments.seq2seq_data import load_synthetic, load_real, STIM_COLS
    import torch
    import utils


@app.cell(hide_code=True)
def _():
    RESULTS_PATH = Path("/Volumes/imaging.data/ppilip/results/models")
    experiment_dirs = discover_experiments(RESULTS_PATH)

    mo.stop(not experiment_dirs, mo.md(f"No experiments found in `{RESULTS_PATH}`."))

    experiment_dropdown = mo.ui.dropdown(
        options=experiment_dirs,
        value=experiment_dirs[0],
        label="Experiment",
    )
    load_button = mo.ui.run_button(label="Load")
    mo.hstack([experiment_dropdown, load_button], justify="start", gap=1)
    return RESULTS_PATH, experiment_dropdown, load_button


@app.cell(hide_code=True)
def _(RESULTS_PATH, experiment_dropdown, load_button):
    mo.stop(not load_button.value, mo.md("Select an experiment and click **Load**."))

    exp_dir = RESULTS_PATH / experiment_dropdown.value
    is_grouped = (exp_dir / "experiment.json").exists()

    if is_grouped:
        loaded_bundles = load_experiment_group(str(exp_dir))
    else:
        loaded_bundles = None
    return exp_dir, is_grouped, loaded_bundles


@app.cell(hide_code=True)
def _(exp_dir):
    _started_file = exp_dir / "started.txt"
    _started_info = _started_file.read_text() if _started_file.exists() else ""

    mo.vstack([mo.md(f"""
    ## {exp_dir.name}

    {f"```{chr(10)}{_started_info}```" if _started_info else ""}
    """)])
    return


@app.cell(hide_code=True)
def _(exp_dir):
    _fig_dir = exp_dir / "figures"
    _pngs = sorted(_fig_dir.glob("*.png")) if _fig_dir.is_dir() else []

    mo.stop(not _pngs, mo.md("No global figures in experiment"))

    _cards = []
    for _png in _pngs:
        _b64 = base64.b64encode(_png.read_bytes()).decode()
        _label = _png.stem.replace("_", " ").title()
        _cards.append(
            mo.md(f"**{_label}**\n\n<img src=\"data:image/png;base64,{_b64}\" style=\"width:100%; border-radius:6px;\" />")
        )

    _cols = min(len(_cards), 2)
    _rows = [
        mo.vstack(_cards[i : i + _cols], heights="equal", gap=1)
        for i in range(0, len(_cards), _cols)
    ]

    mo.vstack([mo.md("## Figures"), *_rows])
    return


@app.cell(hide_code=True)
def _(is_grouped, loaded_bundles):
    if is_grouped:
        variants = list(loaded_bundles.keys())
        variant_dropdown = mo.ui.dropdown(
            options=variants,
            value=variants[0],
            label="Variant",
        )
    else:
        variant_dropdown = None

    mo.stop(variant_dropdown is None, mo.md("No variants found in this bundle"))
    variant_dropdown
    return (variant_dropdown,)


@app.cell(hide_code=True)
def _(exp_dir, is_grouped, loaded_bundles, variant_dropdown):
    if is_grouped:
        bundle = loaded_bundles[variant_dropdown.value]
    else:
        _has_final = (exp_dir / "bundle.pt").exists()
        _has_checkpoint = (exp_dir / "checkpoints" / "bundle.pt").exists()

        if _has_final:
            bundle = ExperimentBundle.load(str(exp_dir))
        elif _has_checkpoint:
            bundle = ExperimentBundle.load(str(exp_dir / "checkpoints"))
        else:
            bundle = ExperimentBundle.load(str(exp_dir))
    return (bundle,)


@app.cell(hide_code=True)
def _(bundle):
    scalar_metrics = {
        k: v for k, v in bundle.metrics.items()
        if isinstance(v, (int, float, str, np.floating, np.integer))
    }

    config_rows = "".join(
        f"| `{k}` | `{v}` |\n"
        for k, v in bundle.training_config.items()
    )

    metrics_rows = "".join(
        f"| `{k}` | `{v:.6f}` |\n" if isinstance(v, (float, np.floating))
        else f"| `{k}` | `{v}` |\n"
        for k, v in scalar_metrics.items()
    )

    _header = mo.md(f"""
    ### {bundle.name}

    **Timestamp:** {bundle.timestamp} | **Model:** `{bundle.model_type}`
    """)

    _model_cfg_md = mo.md(f"""
    #### Model config
    | Key | Value |
    |-----|-------|
    {"".join(f"| `{k}` | `{v}` |{chr(10)}" for k, v in bundle.model_config.items())}
    """)

    _training_cfg_md = mo.md(f"""
    #### Training config
    | Key | Value |
    |-----|-------|
    {config_rows}
    """)

    _metrics_md = mo.md(f"""
    #### Metrics
    | Metric | Value |
    |--------|-------|
    {metrics_rows}
    """)

    _parts = [
        _header,
        mo.hstack([_model_cfg_md, _training_cfg_md, _metrics_md], widths='equal', align="start"),
    ]

    if bundle.warnings:
        _parts.append(mo.callout(
            mo.md("\n".join(f"- {w}" for w in bundle.warnings)),
            kind="warn",
        ))

    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(bundle):
    _stats = bundle.training_results.get("stats", {})
    _history = bundle.training_results.get("history", {})
    _train_loss = _history.get("train_loss", [])
    _val_loss = _history.get("val_loss", [])

    mo.stop(not _stats and not _train_loss, mo.md(""))

    _parts = []

    if _stats:
        _elapsed = _stats.get("train_elapsed_s", 0)
        if _elapsed >= 3600:
            _time_str = f"{_elapsed / 3600:.1f}h"
        elif _elapsed >= 60:
            _time_str = f"{_elapsed / 60:.1f}m"
        else:
            _time_str = f"{_elapsed:.1f}s"

        _params = _stats.get("n_model_parameters")
        _params_str = f"{_params:,}" if _params else "—"

        _summary_rows = [
            ("Wall time", _time_str),
            ("Device", _stats.get("device", "—")),
            ("Parameters", _params_str),
            ("Epochs", str(_stats.get("epochs_completed", "—"))),
            ("Time / epoch", f"{_stats.get('time_per_epoch_s', '—')}s"),
            ("Train samples", f"{_stats.get('n_train_samples', 0):,}"),
            ("Val samples", f"{_stats.get('n_val_samples', 0):,}"),
            ("Throughput", f"{_stats.get('samples_per_second', '—')} samples/s"),
        ]

        _loss_rows = []
        if "final_train_loss" in _stats:
            _loss_rows.append(("Final train loss", f"{_stats['final_train_loss']:.6f}"))
        if "final_val_loss" in _stats:
            _loss_rows.append(("Final val loss", f"{_stats['final_val_loss']:.6f}"))
        if "best_val_loss" in _stats:
            _loss_rows.append(("Best val loss", f"{_stats['best_val_loss']:.6f} (epoch {_stats.get('best_epoch', '?')})"))
        if "convergence_epoch_95pct" in _stats:
            _loss_rows.append(("95% convergence", f"epoch {_stats['convergence_epoch_95pct']}"))

        _infra_md = mo.md(
            "#### Infrastructure\n"
            "| | |\n|---|---|\n"
            + "".join(f"| **{k}** | {v} |\n" for k, v in _summary_rows)
        )

        _loss_md = mo.md(
            "#### Loss summary\n"
            "| | |\n|---|---|\n"
            + "".join(f"| **{k}** | {v} |\n" for k, v in _loss_rows)
        ) if _loss_rows else mo.md("")

        _parts.append(mo.hstack([_infra_md, _loss_md], widths="equal", align="start"))

    if _train_loss:
        _epochs = list(range(len(_train_loss)))
        _df_parts = []
        _df_parts.append(pl.DataFrame({"epoch": _epochs, "loss": _train_loss, "series": ["train"] * len(_train_loss)}))
        if _val_loss:
            _df_parts.append(pl.DataFrame({"epoch": list(range(len(_val_loss))), "loss": _val_loss, "series": ["val"] * len(_val_loss)}))
        _df = pl.concat(_df_parts)

        _chart = alt.Chart(_df).mark_line(strokeWidth=1.5).encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="Loss", scale=alt.Scale(type="log")),
            color=alt.Color("series:N", title="", scale=alt.Scale(domain=["train", "val"], range=["#4c78a8", "#e45756"])),
            tooltip=["series", "epoch", alt.Tooltip("loss:Q", format=".6f")],
        ).properties(width=500, height=250, title="Training loss")

        if _stats.get("best_epoch") is not None and _val_loss:
            _best_epoch = _stats["best_epoch"]
            _best_val = _stats["best_val_loss"]
            _rule = alt.Chart(pl.DataFrame({"epoch": [_best_epoch], "loss": [_best_val], "label": [f"best: {_best_val:.6f}"]})).mark_point(
                shape="diamond", size=80, color="#e45756", filled=True,
            ).encode(x="epoch:Q", y="loss:Q", tooltip=["label"])
            _chart = _chart + _rule

        _parts.append(_chart)

    mo.vstack([mo.md("#### Training stats"), *_parts])
    return


@app.cell(hide_code=True)
def _(bundle):
    mo.stop(not bundle.figures, mo.md(f"No figures in the bundle {bundle.name}"))

    _cards = []
    for _name, _fig in bundle.figures.items():
        _label = _name.replace("_", " ").title()
        _buf = io.BytesIO()
        if isinstance(_fig, Figure):
            _fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
        else:
            plt.imsave(_buf, _fig, format="png")
        _buf.seek(0)
        _b64 = base64.b64encode(_buf.read()).decode()
        _cards.append(
            mo.md(f"**{_label}**\n\n<img src=\"data:image/png;base64,{_b64}\" style=\"width:100%; border-radius:6px;\" />")
        )

    _cols = min(len(_cards), 2)
    _rows = [
        mo.vstack(_cards[i : i + _cols], heights="equal", gap=1)
        for i in range(0, len(_cards), _cols)
    ]

    mo.vstack([mo.md(f"Variant: {bundle.name}"), *_rows])
    return


@app.cell(hide_code=True)
def _(bundle):
    views = get_views(bundle)
    mo.stop(not views, mo.md("_No interactive visualizations available for this model._"))
    mo.vstack([mo.md("## Interactive Visualizations"), mo.ui.tabs(views)])
    return


@app.cell(hide_code=True)
def _(bundle):
    H_eval = bundle.training_config["history_len"]
    F_eval = bundle.training_config["future_len"]
    return F_eval, H_eval


@app.cell(column=1)
def _(bundle):
    bundle.model_type = 'experiments.lstm_seq2seq.Seq2Seq'
    return


@app.cell
def _(bundle):
    m = bundle.reconstruct_model()
    dev = utils.get_device()
    m = m.to(dev)
    m.eval()
    return dev, m


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Does the 2-stepper trained on synthetic data predict real data well?
    TLDR: no
    """)
    return


@app.cell(hide_code=True)
def _(F_eval, H_eval, dev, m):
    mo.stop(True)

    total_eval = H_eval + F_eval

    cnr_real, stim_real, _cond_real = load_real(window_size=total_eval, stride=13)
    print('cnr_r shape:', cnr_real.shape)
    _samples = []

    for i in range(len(cnr_real)):
        t = 0
        while t + total_eval <= cnr_real.shape[1]:
            enc_cnr = cnr_real[i, t : t + H_eval]
            enc_stim = stim_real[i, :, t : t + H_eval]
            dec_stim = stim_real[i, :, t + H_eval : t + total_eval]
            full_window = cnr_real[i, t : t + total_eval]
            dec_target = np.diff(full_window)[H_eval - 1 : H_eval - 1 + F_eval]
            enc_in = np.concatenate([enc_cnr[:, np.newaxis], enc_stim.T], axis=-1)
            light_window = stim_real[i, 0, t : t + total_eval]
            _samples.append((enc_in, dec_stim.T, dec_target, full_window, light_window))
            t += 2

    _preds_all, _targets_all, _windows_all, _light_all = [], [], [], []
    with torch.no_grad():
        for _enc, _dec, _tgt, _win, _light in _samples:
            _enc_t = torch.tensor(_enc, dtype=torch.float32).unsqueeze(0).to(dev)
            _dec_t = torch.tensor(_dec, dtype=torch.float32).unsqueeze(0).to(dev)
            _pred = m(_enc_t, _dec_t).cpu().numpy().squeeze(0)
            _preds_all.append(_pred)
            _targets_all.append(_tgt)
            _windows_all.append(_win)
            _light_all.append(_light)

    preds_real = np.array(_preds_all)
    targets_real = np.array(_targets_all)
    windows_real = np.array(_windows_all)
    light_real = np.array(_light_all)

    mo.md(f"**Real data:** {len(_samples)} windows (H={H_eval}, F={F_eval}), "
           f"MSE = {np.mean((preds_real - targets_real)**2):.6f}")
    return light_real, preds_real, targets_real, windows_real


@app.cell(hide_code=True)
def _(F_eval, H_eval, preds_real, targets_real, windows_real):
    # Per-step metrics (delta space)
    _mse_per_step = np.mean((preds_real - targets_real) ** 2, axis=0)
    _mae_per_step = np.mean(np.abs(preds_real - targets_real), axis=0)

    # Cumulative absolute CNR reconstruction
    _last_cnr = windows_real[:, H_eval - 1]
    _pred_abs = _last_cnr[:, None] + np.cumsum(preds_real, axis=1)
    _true_abs = _last_cnr[:, None] + np.cumsum(targets_real, axis=1)
    _mse_abs_per_step = np.mean((_pred_abs - _true_abs) ** 2, axis=0)
    _mae_abs_per_step = np.mean(np.abs(_pred_abs - _true_abs), axis=0)

    # Aggregate
    _mse_delta = np.mean((preds_real - targets_real) ** 2)
    _mae_delta = np.mean(np.abs(preds_real - targets_real))
    _mse_abs = np.mean((_pred_abs - _true_abs) ** 2)
    _mae_abs = np.mean(np.abs(_pred_abs - _true_abs))

    # Naive baseline: predict zero delta (last value persists)
    _bl_mse_delta = np.mean(targets_real ** 2)
    _bl_mse_abs = np.mean((_last_cnr[:, None] - _true_abs) ** 2)

    # Per-window MSE for spread
    _per_window_mse = np.mean((preds_real - targets_real) ** 2, axis=1)

    _summary = mo.md(f"""
    ### Real-data evaluation ({len(preds_real)} windows)

    | Metric | Model | Naive baseline (Δ=0) |
    |--------|------:|---------------------:|
    | MSE (Δ) | {_mse_delta:.6f} | {_bl_mse_delta:.6f} |
    | MAE (Δ) | {_mae_delta:.6f} | — |
    | MSE (abs CNR) | {_mse_abs:.6f} | {_bl_mse_abs:.6f} |
    | MAE (abs CNR) | {_mae_abs:.6f} | — |
    | Median per-window MSE | {np.median(_per_window_mse):.6f} | — |
    | 90th pctl per-window MSE | {np.percentile(_per_window_mse, 90):.6f} | — |
    """)

    # Per-step error plot
    _steps = np.arange(1, F_eval + 1)
    _fig_metrics, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3))

    _ax1.plot(_steps, _mse_per_step, "o-", ms=4, label="MSE (Δ)")
    _ax1.plot(_steps, _mae_per_step, "s-", ms=4, label="MAE (Δ)")
    _ax1.set_xlabel("Future step")
    _ax1.set_ylabel("Error")
    _ax1.set_title("Per-step error (delta space)")
    _ax1.legend(fontsize=8)

    _ax2.plot(_steps, _mse_abs_per_step, "o-", ms=4, label="MSE (abs)")
    _ax2.plot(_steps, _mae_abs_per_step, "s-", ms=4, label="MAE (abs)")
    _ax2.set_xlabel("Future step")
    _ax2.set_ylabel("Error")
    _ax2.set_title("Cumulative error (absolute CNR)")
    _ax2.legend(fontsize=8)

    _fig_metrics.tight_layout()

    mo.vstack([_summary, _fig_metrics])
    return


@app.cell(hide_code=True)
def _(preds_real):
    idx_input = mo.ui.text(
        value="0, 1, 2, 3, 4, 5",
        label=f"Window indices (0–{len(preds_real)-1})",
        full_width=True,
    )
    idx_input
    return (idx_input,)


@app.cell(hide_code=True)
def _(
    F_eval,
    H_eval,
    idx_input,
    light_real,
    preds_real,
    targets_real,
    windows_real,
):
    _ids = [int(s.strip()) for s in idx_input.value.split(",")
            if s.strip().lstrip("-").isdigit()]
    _ids = [i for i in _ids if 0 <= i < len(preds_real)]
    mo.stop(not _ids, mo.md("Enter valid comma-separated indices."))

    _n_show = len(_ids)
    _fig, _axes = plt.subplots(_n_show, 1, figsize=(10, 3 * _n_show), sharex=True)
    if _n_show == 1:
        _axes = [_axes]

    _t_all = np.arange(H_eval + F_eval)
    _t_fut = np.arange(H_eval, H_eval + F_eval)

    for _j, (_idx, _ax) in enumerate(zip(_ids, _axes)):
        # Light stimulation as shaded background
        _ax_light = _ax.twinx()
        _ax_light.fill_between(_t_all, light_real[_idx], alpha=0.15, color="orange", label="light")
        _ax_light.set_ylim(0, light_real[_idx].max() * 3 if light_real[_idx].max() > 0 else 1)
        _ax_light.set_ylabel("light", color="orange", fontsize=8)
        _ax_light.tick_params(axis="y", labelcolor="orange", labelsize=7)

        # CNR traces
        _ax.plot(_t_all, windows_real[_idx], "k-", alpha=0.4, label="actual CNR")
        _last_cnr = windows_real[_idx, H_eval - 1]
        _pred_abs = _last_cnr + np.cumsum(preds_real[_idx])
        _true_abs = _last_cnr + np.cumsum(targets_real[_idx])
        _ax.plot(_t_fut, _true_abs, "b-", lw=1.5, label="true future")
        _ax.plot(_t_fut, _pred_abs, "r--", lw=1.5, label="predicted")
        _ax.axvline(H_eval, color="gray", ls=":", alpha=0.5)
        _ax.set_ylabel("CNR")
        _ax.set_title(f"window {_idx}", fontsize=9, loc="left")
        if _j == 0:
            _lines_cnr = _ax.get_legend_handles_labels()
            _lines_light = _ax_light.get_legend_handles_labels()
            _ax.legend(_lines_cnr[0] + _lines_light[0],
                       _lines_cnr[1] + _lines_light[1], fontsize=8)

    _axes[-1].set_xlabel("Time step")
    _fig.suptitle("Synthetic-trained model → real data", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Answer: pretty poor cross comprehension
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Stimulation ablation
    Does the model learn things about ERK conditioned on stimulation? Or is it just 'faking' understanding?

    Take trained model. At inference time, zero out the stimulation input channel — feed the model the cell's history but tell it "no future stimulation" regardless of what actually happened. Measure RMSE. Now feed it the true future stimulation. Compare.
    If RMSE barely changes, the model is essentially ignoring the stimulation input and predicting from autonomous dynamics alone. This is the failure mode to rule out. If RMSE drops substantially when you give it the true stimulation, the model is using the input — and you have something real.
    """)
    return


@app.cell
def _(F_eval, H_eval, dev, m):
    # Re-run inference with zeroed-out future stimulation
    from experiments.seq2seq_data import load_real as _load_real

    _total = H_eval + F_eval
    _cnr_abl, _stim_abl, _ = _load_real(window_size=_total, stride=50)

    from experiments.seq2seq_data import _stim_features

    _S = {name: i for i, name in enumerate(STIM_COLS)}

    _samples_abl = []
    for _i in range(len(_cnr_abl)):
        _t = 0
        while _t + _total <= _cnr_abl.shape[1]:
            _enc_cnr  = _cnr_abl[_i, _t : _t + H_eval]
            _enc_stim = _stim_abl[_i, :, _t : _t + H_eval]
            _dec_stim = _stim_abl[_i, :, _t + H_eval : _t + _total]
            _full_win = _cnr_abl[_i, _t : _t + _total]
            _dec_tgt  = np.diff(_full_win)[H_eval - 1 : H_eval - 1 + F_eval]
            _enc_in   = np.concatenate([_enc_cnr[:, np.newaxis], _enc_stim.T], axis=-1)
            _light_win = _stim_abl[_i, 0, _t : _t + _total]

            # Build zeroed-future stim features.
            # Recompute over [history light | zero future] so that window-local
            # channels (n_5, slope_5, burst_pos, m_t, u_t) are handled correctly.
            _hist_light = _stim_abl[_i, 0, _t : _t + H_eval]
            _full_light = np.concatenate([_hist_light, np.zeros(F_eval)])[np.newaxis, :]
            _full_feats = _stim_features(_full_light)          # (1, 9, H+F)
            _dec_stim_zeroed = _full_feats[0, :, H_eval:].T.copy()  # (F, 9)

            # The precomputed stim features were derived from the full trajectory,
            # so their boundary state (end of history) may differ from the window-
            # recomputed values.  Correct the stateful decay channels by
            # propagating forward from the precomputed encoder boundary value.

            # s_cum: no new stimulation -> stays flat at history-end value
            _dec_stim_zeroed[:, _S["s_cum"]] = _enc_stim[_S["s_cum"], -1]

            # ewma_fast (alpha=0.5): x=0 future -> ewma[t] = (1-alpha)^t * enc_boundary
            _dec_stim_zeroed[:, _S["ewma_fast"]] = (
                _enc_stim[_S["ewma_fast"], -1] * (0.5 ** np.arange(1, F_eval + 1))
            )

            # ewma_slow (alpha=0.1): same, decay factor (1-0.1)=0.9
            _dec_stim_zeroed[:, _S["ewma_slow"]] = (
                _enc_stim[_S["ewma_slow"], -1] * (0.9 ** np.arange(1, F_eval + 1))
            )

            # recency (tau=5): each step adds 1 to dt -> multiply by exp(-1/tau)
            _dec_stim_zeroed[:, _S["recency"]] = (
                _enc_stim[_S["recency"], -1] * (np.exp(-1.0 / 5.0) ** np.arange(1, F_eval + 1))
            )

            _samples_abl.append((_enc_in, _dec_stim.T, _dec_tgt, _full_win, _light_win, _dec_stim_zeroed))
            _t += 2

    #mo.stop(True)

    _preds_zeroed, _preds_true = [], []
    with torch.no_grad():
        for _enc, _dec, _tgt, _, _, _dec_z in _samples_abl:
            _enc_t = torch.tensor(_enc,   dtype=torch.float32).unsqueeze(0).to(dev)
            _dec_t = torch.tensor(_dec,   dtype=torch.float32).unsqueeze(0).to(dev)
            _dec_zero = torch.tensor(_dec_z, dtype=torch.float32).unsqueeze(0).to(dev)

            _pred_true = m(_enc_t, _dec_t).cpu().numpy().squeeze(0)
            _pred_zero = m(_enc_t, _dec_zero).cpu().numpy().squeeze(0)
            _preds_true.append(_pred_true)
            _preds_zeroed.append(_pred_zero)

    abl_preds_true = np.array(_preds_true)
    abl_preds_zeroed = np.array(_preds_zeroed)
    abl_targets = np.array([s[2] for s in _samples_abl])
    abl_windows = np.array([s[3] for s in _samples_abl])
    abl_light   = np.array([s[4] for s in _samples_abl])

    _rmse_true = np.sqrt(np.mean((abl_preds_true - abl_targets) ** 2))
    _rmse_zero = np.sqrt(np.mean((abl_preds_zeroed - abl_targets) ** 2))
    _rmse_diff = np.sqrt(np.mean((abl_preds_true - abl_preds_zeroed) ** 2))

    _rmse_true_step = np.sqrt(np.mean((abl_preds_true - abl_targets) ** 2, axis=0))
    _rmse_zero_step = np.sqrt(np.mean((abl_preds_zeroed - abl_targets) ** 2, axis=0))

    _fig_abl, _ax_abl = plt.subplots(figsize=(8, 3))
    _steps = np.arange(1, F_eval + 1)
    _ax_abl.plot(_steps, _rmse_true_step, "o-", ms=4, label="True stim")
    _ax_abl.plot(_steps, _rmse_zero_step, "s-", ms=4, label="Zeroed stim")
    _ax_abl.set_xlabel("Future step")
    _ax_abl.set_ylabel("RMSE")
    _ax_abl.set_title("Stimulation ablation: per-step RMSE")
    _ax_abl.legend(fontsize=8)
    _fig_abl.tight_layout()

    mo.vstack([
        mo.md(f"""
    ### Stimulation ablation results ({len(_samples_abl)} windows)

    | Condition | RMSE |
    |-----------|-----:|
    | True future stim | {_rmse_true:.6f} |
    | Zeroed future stim | {_rmse_zero:.6f} |
    | Pred difference (true vs zero) | {_rmse_diff:.6f} |

    **Interpretation:** {"Model uses stimulation input — predictions change substantially." if _rmse_diff > 0.1 * _rmse_true else "Model largely ignores stimulation input — predictions barely change when stim is zeroed."}
    """),
        _fig_abl,
    ])
    return (
        abl_light,
        abl_preds_true,
        abl_preds_zeroed,
        abl_targets,
        abl_windows,
    )


@app.cell
def _(abl_targets):
    abl_reshuffle = mo.ui.run_button(label="Reshuffle examples")
    mo.hstack([
        mo.md(f"### Example trajectories: true stim vs zeroed stim ({len(abl_targets)} windows)"),
        abl_reshuffle,
    ], justify="start", gap=1, align="center")
    return (abl_reshuffle,)


@app.cell
def _(
    F_eval,
    H_eval,
    abl_light,
    abl_preds_true,
    abl_preds_zeroed,
    abl_reshuffle,
    abl_targets,
    abl_windows,
):
    _ = abl_reshuffle.value

    _n = len(abl_targets)
    _example_ids = np.random.choice(_n, size=min(3, _n), replace=False)
    print('example ids', _example_ids)
    _t_all = np.arange(H_eval + F_eval)
    _t_fut = np.arange(H_eval, H_eval + F_eval)
    _fig_ex, _axes_ex = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for _j, (_idx, _ax) in enumerate(zip(_example_ids, _axes_ex)):
        _win = abl_windows[_idx]
        _last = _win[H_eval - 1]
        _true_abs = _last + np.cumsum(abl_targets[_idx])
        _pred_true_abs = _last + np.cumsum(abl_preds_true[_idx])
        _pred_zero_abs = _last + np.cumsum(abl_preds_zeroed[_idx])

        _ax2 = _ax.twinx()
        _ax2.fill_between(_t_all, abl_light[_idx], alpha=0.15, color="orange", label="light")
        _ax2.set_ylim(0, max(abl_light[_idx].max() * 3, 1))
        _ax2.set_ylabel("light", color="orange", fontsize=8)
        _ax2.tick_params(axis="y", labelcolor="orange", labelsize=7)

        _ax.plot(_t_all, _win, "k-", alpha=0.4, label="actual CNR")
        _ax.plot(_t_fut, _true_abs, "b-", lw=1.5, label="true future")
        _ax.plot(_t_fut, _pred_true_abs, "r--", lw=1.5, label="pred (true stim)")
        _ax.plot(_t_fut, _pred_zero_abs, "g--", lw=1.5, label="pred (zero stim)")
        _ax.axvline(H_eval, color="gray", ls=":", alpha=0.5)
        _ax.set_ylabel("CNR")
        _ax.set_title(f"window {_idx}", fontsize=9, loc="left")
        if _j == 0:
            _h1, _l1 = _ax.get_legend_handles_labels()
            _h2, _l2 = _ax2.get_legend_handles_labels()
            _ax.legend(_h1 + _h2, _l1 + _l2, fontsize=7, ncol=3, loc="upper right")

    _axes_ex[-1].set_xlabel("Time step")
    _fig_ex.suptitle("Ablation examples (click Reshuffle for new ones)", fontsize=11)
    _fig_ex.tight_layout()
    _fig_ex
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Fixed a bug where the generation of features would incorrectly initialize cumulative features from 0 instead of relying on features from a complete trajectory that we are windowing.

    Despite that, some trajectories still exhibit a behavior where zero stimulation results in an immediate peak that is higher than both true stimulation and actual trajectories in the future.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conterfactual sensitivity

    For a held-out cell, take its real history. Then generate predictions under several different hypothetical future stimulation sequences: always on, always off, a specific pulse train, random on/off. Plot all the predicted trajectories on the same axes. Do they diverge? By how much? For MPC to work, these trajectories need to be visibly different — ideally by amounts comparable to the dynamic range of the target ERK values you'd want to control toward.
    This is the most informative single test you can run right now. The spread of predictions across counterfactual inputs tells you the model's effective "control authority" — how much the predictions change based on control decisions. A model with tight counterfactual spread cannot drive useful MPC no matter how good its point-prediction RMSE is.
    """)
    return


@app.cell(hide_code=True)
def _(F_eval, H_eval):
    from experiments.seq2seq_data import load_real as _load_real_cf
    from experiments.seq2seq_data import _stim_features

    _total_cf = H_eval + F_eval
    cnr_cf, stim_cf, _ = _load_real_cf(window_size=_total_cf, stride=13)
    return cnr_cf, stim_cf


@app.cell(hide_code=True)
def _():
    cf_reshuffle = mo.ui.run_button(label="Reshuffle examples")
    mo.hstack([
        mo.md("### Counterfactual sensitivity"),
        cf_reshuffle,
    ], justify="start", gap=1, align="center")
    return (cf_reshuffle,)


@app.cell(hide_code=True)
def _(F_eval, H_eval, cf_reshuffle, cnr_cf, dev, m, stim_cf):
    _ = cf_reshuffle.value

    _S = {name: i for i, name in enumerate(STIM_COLS)}
    _total_cf = H_eval + F_eval

    _light_patterns = {}
    _light_patterns["Always OFF"] = np.zeros((1, F_eval))
    _light_patterns["Always ON"] = np.ones((1, F_eval))
    _lp = np.zeros((1, F_eval))
    for _k in range(F_eval):
        if (_k // 3) % 2 == 0:
            _lp[0, _k] = 1.0
    _light_patterns["Pulse (3on/3off)"] = _lp

    _colors = {"Always OFF": "#4c78a8", "Always ON": "#e45756",
               "Pulse (3on/3off)": "#72b7b2", "Actual": "#f58518"}

    _scenario_names = list(_light_patterns.keys()) + ["Actual"]
    _n_scenarios = len(_scenario_names)

    _n_cells = len(cnr_cf)
    _cell_ids = np.random.choice(_n_cells, size=min(3, _n_cells), replace=False)

    _t_all = np.arange(_total_cf)
    _t_hist = np.arange(H_eval)
    _t_fut = np.arange(H_eval, _total_cf)
    _n_show = len(_cell_ids)

    _rows_per_cell = 1 + _n_scenarios
    _cnr_h = 3.0
    _light_h = 0.4
    _ratios = []
    for _ in range(_n_show):
        _ratios.append(_cnr_h)
        _ratios.extend([_light_h] * _n_scenarios)
    _total_h = _n_show * (_cnr_h + _n_scenarios * _light_h) + 1.0

    _fig_cf, _axes_cf = plt.subplots(
        _n_show * _rows_per_cell, 1,
        figsize=(10, _total_h),
        gridspec_kw={"height_ratios": _ratios},
        sharex=True,
    )

    _all_spreads = []

    for _ci, _cell_idx in enumerate(_cell_ids):
        _base = _ci * _rows_per_cell
        _ax_cnr = _axes_cf[_base]

        _enc_cnr  = cnr_cf[_cell_idx, :H_eval]
        _enc_stim = stim_cf[_cell_idx, :, :H_eval]
        _enc_in   = np.concatenate([_enc_cnr[:, np.newaxis], _enc_stim.T], axis=-1)
        _enc_t    = torch.tensor(_enc_in, dtype=torch.float32).unsqueeze(0).to(dev)

        _hist_light  = stim_cf[_cell_idx, 0, :H_eval]
        _actual_dec  = stim_cf[_cell_idx, :, H_eval : _total_cf]
        _full_window = cnr_cf[_cell_idx, :_total_cf]
        _true_deltas = np.diff(_full_window)[H_eval - 1 : H_eval - 1 + F_eval]
        _last_cnr    = _full_window[H_eval - 1]

        _scenarios = {}
        for _name, _lp_arr in _light_patterns.items():
            _full_light = np.concatenate([_hist_light, _lp_arr.squeeze()])[np.newaxis, :]
            _full_feats = _stim_features(_full_light)
            _dec_feats  = _full_feats[0, :, H_eval:].T.copy()

            _dec_feats[:, _S["s_cum"]] = (
                _enc_stim[_S["s_cum"], -1] + np.cumsum(_lp_arr.squeeze())
            )
            _dec_feats[:, _S["ewma_fast"]] = (
                _enc_stim[_S["ewma_fast"], -1] * (0.5 ** np.arange(1, F_eval + 1))
                + 0.5 * np.array([
                    sum(0.5**j * _lp_arr[0, t - j] for j in range(t + 1))
                    for t in range(F_eval)
                ])
            )
            _dec_feats[:, _S["ewma_slow"]] = (
                _enc_stim[_S["ewma_slow"], -1] * (0.9 ** np.arange(1, F_eval + 1))
                + 0.1 * np.array([
                    sum(0.9**j * _lp_arr[0, t - j] for j in range(t + 1))
                    for t in range(F_eval)
                ])
            )
            _dec_feats[:, _S["recency"]] = _full_feats[0, _S["recency"], H_eval:]

            _scenarios[_name] = (_dec_feats, _lp_arr.squeeze())

        _actual_light = stim_cf[_cell_idx, 0, H_eval : _total_cf]
        _scenarios["Actual"] = (_actual_dec.T, _actual_light)

        _cf_preds = {}
        with torch.no_grad():
            for _label, (_dec_np, _) in _scenarios.items():
                _dec_t = torch.tensor(_dec_np, dtype=torch.float32).unsqueeze(0).to(dev)
                _pred = m(_enc_t, _dec_t).cpu().numpy().squeeze(0)
                _cf_preds[_label] = _last_cnr + np.cumsum(_pred)

        _true_abs = _last_cnr + np.cumsum(_true_deltas)

        _ax_cnr.plot(_t_hist, _full_window[:H_eval], "k-", alpha=0.5, label="History")
        _ax_cnr.plot(_t_fut, _true_abs, "k--", lw=1.5, alpha=0.5, label="True future")
        for _label, _pred_abs in _cf_preds.items():
            _ax_cnr.plot(_t_fut, _pred_abs, lw=2, color=_colors[_label], label=_label)
        _ax_cnr.axvline(H_eval, color="gray", ls=":", alpha=0.5)
        _ax_cnr.set_ylabel("CNR")
        _ax_cnr.set_title(f"Cell {_cell_idx}", fontsize=9, loc="left")
        if _ci == 0:
            _ax_cnr.legend(fontsize=7, ncol=3, loc="upper right")

        for _si, _sname in enumerate(_scenario_names):
            _ax_l = _axes_cf[_base + 1 + _si]
            _, _light_arr = _scenarios[_sname]
            _ax_l.fill_between(_t_fut, 0, _light_arr, color=_colors[_sname], alpha=0.5)
            _ax_l.fill_between(_t_hist, 0, _hist_light, color="gray", alpha=0.25)
            _ax_l.axvline(H_eval, color="gray", ls=":", alpha=0.5)
            _ax_l.set_ylim(-0.05, 1.15)
            _ax_l.set_yticks([])
            _ax_l.set_ylabel(_sname, fontsize=7, rotation=0, ha="right", va="center")

        _preds_stack = np.stack(list(_cf_preds.values()))
        _all_spreads.append(np.max(_preds_stack, axis=0) - np.min(_preds_stack, axis=0))

    _axes_cf[-1].set_xlabel("Time step")
    _fig_cf.suptitle(f"Counterfactual sensitivity — cells {list(_cell_ids)}", fontsize=12)
    _fig_cf.tight_layout()

    _spread = np.concatenate(_all_spreads)
    _cnr_range = np.ptp(cnr_cf[:, :_total_cf])

    mo.vstack([
        mo.md(f"""
    | Metric | Value |
    |--------|------:|
    | Mean counterfactual spread | {np.mean(_spread):.6f} |
    | Max counterfactual spread | {np.max(_spread):.6f} |
    | CNR dynamic range (dataset) | {_cnr_range:.4f} |
    | Spread / dynamic range | {np.mean(_spread) / max(_cnr_range, 1e-8):.2%} |

    **Interpretation:** {"Good control authority — predictions diverge meaningfully under different stimulation." if np.mean(_spread) > 0.05 * _cnr_range else "Low control authority — predictions are similar regardless of stimulation input."}
    """),
        _fig_cf,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Must be a bug here; The actual has no stimulation and always off has no stimulation, but the trajectories they produce differ wildly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Per-cond RMSE breakdown
    Split your validation set by stimulation regime (quiescent periods, active stimulation, different intensities, responders vs apparent non-responders). Is the model's improvement over persistence uniform across conditions, or concentrated in specific ones? If it's beating persistence mainly during quiescent periods (by capturing endogenous oscillation patterns) but not during active stimulation (where the controller actually needs to work), that's a targeted problem to fix. If it's the other way around, that's actually promising for MPC.
    """)
    return


@app.cell(hide_code=True)
def _(F_eval, H_eval, bundle, dev, m):
    from experiments.seq2seq_data import load_real as _load_real_pc

    _total_pc = H_eval + F_eval
    _cnr_pc, _stim_pc, _cond_pc = _load_real_pc(window_size=_total_pc, stride=13)

    _records = []
    for _i in range(len(_cnr_pc)):
        _t = 0
        while _t + _total_pc <= _cnr_pc.shape[1]:
            _enc_cnr = _cnr_pc[_i, _t : _t + H_eval]
            _enc_stim = _stim_pc[_i, :, _t : _t + H_eval]
            _dec_stim = _stim_pc[_i, :, _t + H_eval : _t + _total_pc]
            _full_win = _cnr_pc[_i, _t : _t + _total_pc]
            _dec_tgt = np.diff(_full_win)[H_eval - 1 : H_eval - 1 + F_eval]
            _enc_in = np.concatenate([_enc_cnr[:, np.newaxis], _enc_stim.T], axis=-1)

            _future_light = _stim_pc[_i, 0, _t + H_eval : _t + _total_pc]
            _frac_on = np.mean(_future_light > 0)
            if _frac_on < 0.1:
                _stim_regime = "Quiescent"
            elif _frac_on > 0.9:
                _stim_regime = "Sustained"
            else:
                _stim_regime = "Mixed"

            _records.append({
                "enc_in": _enc_in, "dec_stim": _dec_stim.T,
                "target": _dec_tgt, "last_cnr": _full_win[H_eval - 1],
                "stim_regime": _stim_regime,
                "condition": _cond_pc[_i],
            })
            _t += 2

    with torch.no_grad():
        for _r in _records:
            _enc_t = torch.tensor(_r["enc_in"], dtype=torch.float32).unsqueeze(0).to(dev)
            _dec_t = torch.tensor(_r["dec_stim"], dtype=torch.float32).unsqueeze(0).to(dev)
            _r["pred"] = m(_enc_t, _dec_t).cpu().numpy().squeeze(0)

    from collections import defaultdict

    def _group_metrics(key):
        _groups = defaultdict(list)
        for _r in _records:
            _groups[_r[key]].append(_r)
        _rows = []
        for _name, _recs in sorted(_groups.items()):
            _preds = np.array([_r["pred"] for _r in _recs])
            _tgts  = np.array([_r["target"] for _r in _recs])
            _rmse_model = np.sqrt(np.mean((_preds - _tgts) ** 2))
            _rmse_naive = np.sqrt(np.mean(_tgts ** 2))
            _rows.append({
                "Group": _name, "N": len(_recs),
                "RMSE (model)": _rmse_model,
                "RMSE (naive)": _rmse_naive,
                "Improvement": f"{(1 - _rmse_model / max(_rmse_naive, 1e-8)) * 100:.1f}%",
            })
        return _rows

    def _make_table(rows):
        lines = ["| Group | N | RMSE (model) | RMSE (naive) | Improvement |",
                 "|-------|--:|-------------:|-------------:|------------:|"]
        for _r in rows:
            lines.append(
                f"| {_r['Group']} | {_r['N']} "
                f"| {_r['RMSE (model)']:.6f} | {_r['RMSE (naive)']:.6f} "
                f"| {_r['Improvement']} |"
            )
        return "\n".join(lines)

    _regime_rows = _group_metrics("stim_regime")

    _regime_df = pl.DataFrame({
        "regime": [_r["Group"] for _r in _regime_rows],
        "model":  [_r["RMSE (model)"] for _r in _regime_rows],
        "naive":  [_r["RMSE (naive)"] for _r in _regime_rows],
    }).unpivot(index="regime", on=["model", "naive"], variable_name="method", value_name="RMSE")

    _bar = alt.Chart(_regime_df).mark_bar().encode(
        x=alt.X("regime:N", title="Stimulation regime"),
        y=alt.Y("RMSE:Q"),
        color=alt.Color("method:N", scale=alt.Scale(domain=["model", "naive"], range=["#4c78a8", "#e45756"])),
        xOffset="method:N",
        tooltip=["regime", "method", alt.Tooltip("RMSE:Q", format=".6f")],
    ).properties(width=400, height=250, title="RMSE by stimulation regime (all steps)")

    # Per-step RMSE: shows whether the model degrades quickly over the horizon
    _all_preds = np.array([_r["pred"] for _r in _records])   # (N, F_eval)
    _all_tgts  = np.array([_r["target"] for _r in _records]) # (N, F_eval)
    _steps = np.arange(1, F_eval + 1)
    _rmse_model_step = np.sqrt(np.mean((_all_preds - _all_tgts) ** 2, axis=0))
    _rmse_naive_step = np.sqrt(np.mean(_all_tgts ** 2, axis=0))

    _step_df = pl.concat([
        pl.DataFrame({"step": _steps, "RMSE": _rmse_model_step, "method": ["model"] * F_eval}),
        pl.DataFrame({"step": _steps, "RMSE": _rmse_naive_step, "method": ["naive (Δ=0)"] * F_eval}),
    ])
    _step_chart = alt.Chart(_step_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("step:O", title="Future step"),
        y=alt.Y("RMSE:Q", title="RMSE"),
        color=alt.Color("method:N", title="",
                        scale=alt.Scale(domain=["model", "naive (Δ=0)"],
                                        range=["#4c78a8", "#e45756"])),
        tooltip=["method", "step", alt.Tooltip("RMSE:Q", format=".6f")],
    ).properties(width=400, height=220, title="RMSE by future step (all regimes)")

    _parts = [
        mo.md(f"### Per-regime RMSE ({len(_records)} windows, averaged over all {F_eval} future steps)"),
        mo.md(_make_table(_regime_rows)),
        _bar,
        mo.md("### RMSE vs prediction horizon"),
        mo.md(
            f"Step 1 improvement: "
            f"{(1 - _rmse_model_step[0] / _rmse_naive_step[0]) * 100:.1f}%  |  "
            f"Step {F_eval} improvement: "
            f"{(1 - _rmse_model_step[-1] / _rmse_naive_step[-1]) * 100:.1f}%"
        ),
        _step_chart,
    ]

    if bundle.model_config.get("data_source") == "real":
        _cond_rows = _group_metrics("condition")
        _parts += [
            mo.md("#### By experimental condition"),
            mo.md(_make_table(_cond_rows)),
        ]

    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notes:

    Full real experiment shows big improvement, esp in the first frames (26.7% at the beginning, to 4.6% at steo 10);
    However, the synthetic data as it currently is gets way worse results.

    synthetic full
    ar still better than baseline; by far

    What can explain it?
    1. Synthetic generation is not that good - generated model kinda sucks / does not capture real dynamics, etc...
    2. Real data is being overfitted on - only a couple of patterns of stimulation, we learn some weird representation of those rather than the real deal.

    How do you discriminate between those?
    TODO: figure it out
    """)
    return


@app.cell(column=2)
def _(exp_dir):
    _log_file = exp_dir / "slurm.log"
    _log_content = _log_file.read_text() if _log_file.exists() else None

    mo.stop(
        _log_content is None,
        mo.md("_No SLURM log found for this experiment (local run or log not on Kingston)._"),
    )

    mo.vstack([
        mo.md("## Training log"),
        mo.plain_text(_log_content),
    ])
    return


@app.cell
def _(exp_dir):
    _html_file = exp_dir / "notebook.html"
    _html_exists = _html_file.exists()

    open_html_button = mo.ui.run_button(
        label="Open notebook HTML",
        disabled=not _html_exists,
    )
    _status = f"`{_html_file.name}`" if _html_exists else "_No HTML export found_"
    mo.hstack([open_html_button, mo.md(_status)], justify="start", gap=1, align="center")
    return (open_html_button,)


@app.cell
def _(exp_dir, open_html_button):
    mo.stop(not open_html_button.value)

    subprocess.Popen(["open", str(exp_dir / "notebook.html")])
    return


if __name__ == "__main__":
    app.run()
