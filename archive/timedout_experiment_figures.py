import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns", app_title="Timed-out experiment figures")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import io
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from experiment import ExperimentBundle, load_experiment_group


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Figures from a timed-out experiment

    The training loop never reached `save_final`, so `figures/` was never written.
    This notebook reconstructs what *can* be generated from the latest checkpoint
    + `training_results["history"]`: loss curves, training-stats summary, and
    weight histograms from the saved `model_state_dict`. Use the **Save** button
    to write the generated figures into the experiment's `figures/` directory.
    """)
    return


@app.cell
def _():
    exp_path_input = mo.ui.text(
        value="results/",
        label="Experiment directory",
        full_width=True,
    )
    load_button = mo.ui.run_button(label="Load")
    mo.hstack([exp_path_input, load_button], justify="start", gap=1, align="center")
    return exp_path_input, load_button


@app.cell
def _(exp_path_input, load_button):
    mo.stop(not load_button.value, mo.md("Enter a path and click **Load**."))

    exp_dir = Path(exp_path_input.value).expanduser().resolve()
    mo.stop(
        not exp_dir.is_dir(),
        mo.callout(mo.md(f"`{exp_dir}` is not a directory."), kind="danger"),
    )

    is_grouped = (exp_dir / "experiment.json").exists()
    return exp_dir, is_grouped


@app.cell(hide_code=True)
def _(exp_dir):
    _started = exp_dir / "started.txt"
    _info = _started.read_text() if _started.exists() else "(no started.txt)"
    mo.md(f"## `{exp_dir.name}`\n\n```\n{_info}\n```")
    return


@app.cell(hide_code=True)
def _(exp_dir, is_grouped):
    def _resolve_bundle_dir(d: Path) -> Path | None:
        if (d / "bundle.pt").exists():
            return d
        ckpt = d / "checkpoints"
        if (ckpt / "bundle.pt").exists():
            return ckpt
        return None

    if is_grouped:
        try:
            grouped = load_experiment_group(str(exp_dir))
        except Exception:
            grouped = {}
        for _variant_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            if _variant_dir.name in grouped:
                continue
            _bd = _resolve_bundle_dir(_variant_dir)
            if _bd is not None:
                try:
                    grouped[_variant_dir.name] = ExperimentBundle.load(str(_bd))
                except Exception as e:
                    print(f"  could not load variant '{_variant_dir.name}': {e}")
        bundles = grouped
        bundle_dirs = {
            name: (exp_dir / name / "checkpoints"
                   if (exp_dir / name / "checkpoints" / "bundle.pt").exists()
                   else exp_dir / name)
            for name in bundles
        }
    else:
        _bd = _resolve_bundle_dir(exp_dir)
        mo.stop(
            _bd is None,
            mo.callout(
                mo.md(f"No `bundle.pt` found under `{exp_dir}` or its `checkpoints/`."),
                kind="warn",
            ),
        )
        bundles = {exp_dir.name: ExperimentBundle.load(str(_bd))}
        bundle_dirs = {exp_dir.name: _bd}
    return bundle_dirs, bundles


@app.cell(hide_code=True)
def _(bundles):
    variant_dropdown = mo.ui.dropdown(
        options=list(bundles.keys()),
        value=next(iter(bundles.keys())),
        label="Variant / bundle",
    )
    variant_dropdown
    return (variant_dropdown,)


@app.cell
def _(bundle_dirs, bundles, variant_dropdown):
    bundle = bundles[variant_dropdown.value]
    bundle_source_dir = bundle_dirs[variant_dropdown.value]
    return bundle, bundle_source_dir


@app.cell(hide_code=True)
def _(bundle, bundle_source_dir):
    _scalars = {
        k: v for k, v in bundle.metrics.items()
        if isinstance(v, (int, float, str, np.floating, np.integer))
    }
    _model_cfg_md = mo.md(
        "#### Model config\n| Key | Value |\n|-----|-------|\n"
        + "".join(f"| `{k}` | `{v}` |\n" for k, v in bundle.model_config.items())
    )
    _training_cfg_md = mo.md(
        "#### Training config\n| Key | Value |\n|-----|-------|\n"
        + "".join(f"| `{k}` | `{v}` |\n" for k, v in bundle.training_config.items())
    )
    _metrics_md = mo.md(
        "#### Metrics\n| Metric | Value |\n|--------|-------|\n"
        + "".join(
            f"| `{k}` | `{v:.6f}` |\n" if isinstance(v, (float, np.floating))
            else f"| `{k}` | `{v}` |\n"
            for k, v in _scalars.items()
        )
    )
    _header = mo.md(
        f"### {bundle.name}\n\n**Timestamp:** {bundle.timestamp}  \n"
        f"**Model:** `{bundle.model_type or 'unknown'}`  \n"
        f"**Loaded from:** `{bundle_source_dir}`"
    )
    _parts = [
        _header,
        mo.hstack([_model_cfg_md, _training_cfg_md, _metrics_md],
                  widths="equal", align="start"),
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
    _history = bundle.training_results.get("history", {})
    _train_loss = list(_history.get("train_loss", []))
    _val_loss = list(_history.get("val_loss", []))

    mo.stop(not _train_loss, mo.callout(
        mo.md("No `history.train_loss` found in the checkpoint — cannot plot loss curves."),
        kind="warn",
    ))

    _epochs_t = np.arange(len(_train_loss))
    _fig_loss, (_ax_lin, _ax_log) = plt.subplots(1, 2, figsize=(12, 4))
    _ax_lin.plot(_epochs_t, _train_loss, label="train", color="#4c78a8", lw=1.8)
    if _val_loss:
        _ax_lin.plot(np.arange(len(_val_loss)), _val_loss,
                     label="val", color="#e45756", lw=1.8)
        _best_epoch = int(np.argmin(_val_loss))
        _ax_lin.axvline(_best_epoch, color="#e45756", ls=":", lw=1,
                        label=f"best epoch = {_best_epoch}")
    _ax_lin.set_xlabel("epoch")
    _ax_lin.set_ylabel("loss")
    _ax_lin.set_title("Training loss (linear)")
    _ax_lin.legend(fontsize=9)

    _ax_log.plot(_epochs_t, _train_loss, label="train", color="#4c78a8", lw=1.8)
    if _val_loss:
        _ax_log.plot(np.arange(len(_val_loss)), _val_loss,
                     label="val", color="#e45756", lw=1.8)
    _ax_log.set_yscale("log")
    _ax_log.set_xlabel("epoch")
    _ax_log.set_ylabel("loss (log)")
    _ax_log.set_title("Training loss (log)")
    _ax_log.legend(fontsize=9)
    _fig_loss.tight_layout()

    fig_loss = _fig_loss
    fig_loss
    return (fig_loss,)


@app.cell(hide_code=True)
def _(bundle):
    _history = bundle.training_results.get("history", {})
    _train_loss = list(_history.get("train_loss", []))
    _val_loss = list(_history.get("val_loss", []))

    mo.stop(not _val_loss,
            mo.md("_No val_loss history — skipping val-vs-train scatter._"))

    _n = min(len(_train_loss), len(_val_loss))
    _t = np.array(_train_loss[:_n])
    _v = np.array(_val_loss[:_n])
    _gap = _v - _t

    _fig_gap, (_ax_sc, _ax_gap) = plt.subplots(1, 2, figsize=(12, 4))
    _ax_sc.scatter(_t, _v, c=np.arange(_n), cmap="viridis", s=14)
    _lim = [min(_t.min(), _v.min()), max(_t.max(), _v.max())]
    _ax_sc.plot(_lim, _lim, color="gray", lw=0.8, ls="--")
    _ax_sc.set_xlabel("train loss")
    _ax_sc.set_ylabel("val loss")
    _ax_sc.set_title("Val vs train (color = epoch)")

    _ax_gap.plot(np.arange(_n), _gap, color="#7b4fa3", lw=1.5)
    _ax_gap.axhline(0, color="gray", lw=0.8)
    _ax_gap.set_xlabel("epoch")
    _ax_gap.set_ylabel("val − train")
    _ax_gap.set_title("Generalization gap")
    _fig_gap.tight_layout()

    fig_generalization = _fig_gap
    fig_generalization
    return (fig_generalization,)


@app.cell(hide_code=True)
def _(bundle):
    _sd = bundle.model_state_dict
    mo.stop(_sd is None,
            mo.md("_No `model_state_dict` — skipping weight histogram._"))

    _layer_stats = []
    _all_weights = []
    for _name, _tensor in _sd.items():
        if not isinstance(_tensor, torch.Tensor):
            continue
        if _tensor.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            continue
        _flat = _tensor.detach().float().cpu().numpy().ravel()
        if _flat.size == 0:
            continue
        _layer_stats.append((_name, _flat.size, float(_flat.mean()), float(_flat.std()),
                             float(np.abs(_flat).max())))
        _all_weights.append(_flat)

    mo.stop(not _all_weights,
            mo.md("_No floating-point tensors in state dict._"))

    _all = np.concatenate(_all_weights)

    _fig_w, (_ax_h, _ax_b) = plt.subplots(1, 2, figsize=(12, 4))
    _ax_h.hist(_all, bins=120, color="#4c78a8", edgecolor="none")
    _ax_h.set_yscale("log")
    _ax_h.set_xlabel("weight value")
    _ax_h.set_ylabel("count (log)")
    _ax_h.set_title(f"All weights ({_all.size:,} values)")

    _names = [s[0] for s in _layer_stats]
    _means = [s[2] for s in _layer_stats]
    _stds = [s[3] for s in _layer_stats]
    _y = np.arange(len(_names))
    _ax_b.errorbar(_means, _y, xerr=_stds, fmt="o", color="#4c78a8",
                   ecolor="#9ecae9", capsize=2, ms=4)
    _ax_b.set_yticks(_y)
    _ax_b.set_yticklabels(_names, fontsize=6)
    _ax_b.axvline(0, color="gray", lw=0.6)
    _ax_b.set_xlabel("mean ± std")
    _ax_b.set_title("Per-tensor weight stats")
    _ax_b.invert_yaxis()
    _fig_w.tight_layout()

    fig_weights = _fig_w
    fig_weights
    return (fig_weights,)


@app.cell(hide_code=True)
def _(bundle):
    _sd = bundle.model_state_dict
    mo.stop(_sd is None, mo.md(""))

    _rows = []
    _total = 0
    for _name, _tensor in _sd.items():
        if not isinstance(_tensor, torch.Tensor):
            continue
        _n = _tensor.numel()
        _total += _n
        _rows.append(
            f"| `{_name}` | `{tuple(_tensor.shape)}` | {_n:,} | `{_tensor.dtype}` |"
        )

    mo.md(
        f"#### State-dict inventory — **{_total:,}** parameters\n\n"
        "| Tensor | Shape | #params | dtype |\n|---|---|---:|---|\n"
        + "\n".join(_rows)
    )
    return


@app.cell
def _():
    n_windows_slider = mo.ui.slider(1, 12, value=6, step=1, label="Windows to plot")
    stride_slider = mo.ui.slider(5, 100, value=25, step=5, label="Window stride")
    run_pred_button = mo.ui.run_button(label="Run trajectory prediction")
    mo.vstack([
        mo.md("### Trajectory prediction\n\n"
              "Reconstructs the model from the checkpoint, loads the training dataset "
              "referenced in the bundle, and rolls out predictions on a handful of windows. "
              "Only works for the seq2seq models that expose "
              "`forward(encoder_input, future_stim, ...)` and whose `training_config` "
              "contains `history_len` and `future_len`."),
        mo.hstack([n_windows_slider, stride_slider, run_pred_button],
                  justify="start", gap=1, align="center"),
    ])
    return n_windows_slider, run_pred_button, stride_slider


@app.cell
def _(bundle):
    bundle.model_type = 'experiments.lstm_seq2scal_anneal.Seq2Seq'
    return


@app.cell
def _(bundle, n_windows_slider, run_pred_button, stride_slider):
    mo.stop(not run_pred_button.value, mo.md(""))

    _cfg = {**bundle.training_config, **bundle.model_config}
    H = _cfg.get("history_len")
    F = _cfg.get("future_len")
    mo.stop(
        H is None or F is None,
        mo.callout(
            mo.md("`history_len` / `future_len` missing from training/model config — "
                  "cannot run trajectory prediction."),
            kind="warn",
        ),
    )
    source = _cfg.get("data_source") or _cfg.get("source") or "synthetic"

    try:
        model = bundle.reconstruct_model()
    except Exception as e:
        mo.stop(True, mo.callout(
            mo.md(f"Could not reconstruct model: `{e}`"), kind="danger"))

    try:
        from experiments.seq2seq_data import load as load_ds
        cnr_all, stim_all, conditions_all = load_ds(source)
    except Exception as e:
        mo.stop(True, mo.callout(
            mo.md(f"Could not load dataset `{source}`: `{e}`"), kind="danger"))

    _device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = model.to(_device)
    model.eval()

    _total = H + F
    mo.stop(
        cnr_all.shape[1] < _total,
        mo.callout(mo.md(
            f"Trajectory length ({cnr_all.shape[1]}) shorter than H+F ({_total})."
        ), kind="warn"),
    )

    _rng = np.random.default_rng(0)
    _traj_idx = _rng.choice(cnr_all.shape[0], size=n_windows_slider.value, replace=False)

    _enc_batch, _dec_stim_batch, _hist_batch, _fut_actual_batch, _light_batch = [], [], [], [], []
    _stride = stride_slider.value
    for _i in _traj_idx:
        _t = min(_stride, cnr_all.shape[1] - _total)
        _t = max(0, _t)
        _enc_cnr = cnr_all[_i, _t : _t + H]
        _enc_stim = stim_all[_i, :, _t : _t + H]
        _dec_stim = stim_all[_i, :, _t + H : _t + _total]
        _full = cnr_all[_i, _t : _t + _total]
        _enc_in = np.concatenate([_enc_cnr[:, np.newaxis], _enc_stim.T], axis=-1)
        _enc_batch.append(_enc_in)
        _dec_stim_batch.append(_dec_stim.T)
        _hist_batch.append(_enc_cnr)
        _fut_actual_batch.append(_full[H:])
        _light_batch.append(stim_all[_i, 0, _t : _t + _total])

    _enc_t = torch.tensor(np.stack(_enc_batch), dtype=torch.float32, device=_device)
    _dec_t = torch.tensor(np.stack(_dec_stim_batch), dtype=torch.float32, device=_device)

    with torch.no_grad():
        _pred_delta = model(_enc_t, _dec_t).cpu().numpy()  # (B, F)

    _last_abs = np.array([h[-1] for h in _hist_batch])[:, None]
    _pred_abs = _last_abs + np.cumsum(_pred_delta, axis=1)

    _n = len(_enc_batch)
    _cols = min(3, _n)
    _rows = (_n + _cols - 1) // _cols
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(5.5 * _cols, 3.2 * _rows),
                               squeeze=False)
    _t_hist = np.arange(H)
    _t_fut = np.arange(H, H + F)
    for _k in range(_n):
        _ax = _axes[_k // _cols, _k % _cols]
        _ax.fill_between(np.arange(H + F), 0, _light_batch[_k],
                         alpha=0.15, color="gold", step="mid",
                         transform=_ax.get_xaxis_transform(), label="u_t")
        _ax.plot(_t_hist, _hist_batch[_k], color="navy", lw=1.8, label="history")
        _ax.plot(_t_fut, _fut_actual_batch[_k], color="navy", lw=1.8,
                 alpha=0.4, label="actual")
        _ax.plot(_t_fut, _pred_abs[_k], color="crimson", lw=1.8, ls="--",
                 label="pred")
        _ax.axvline(H - 0.5, color="gray", ls=":", lw=1)
        _ax.set_title(f"traj {_traj_idx[_k]} — {conditions_all[_traj_idx[_k]]}",
                      fontsize=9)
        if _k == 0:
            _ax.legend(fontsize=7, loc="best")
    for _k in range(_n, _rows * _cols):
        _axes[_k // _cols, _k % _cols].axis("off")
    _fig.suptitle(
        f"Trajectory predictions — source={source}, H={H}, F={F}, "
        f"mean |err|={np.mean(np.abs(_pred_abs - np.stack(_fut_actual_batch))):.4f}",
        fontsize=10,
    )
    _fig.tight_layout()

    fig_trajectories = _fig
    fig_trajectories
    return (fig_trajectories,)


@app.cell(hide_code=True)
def _(
    bundle_source_dir,
    fig_generalization,
    fig_loss,
    fig_trajectories,
    fig_weights,
):
    save_button = mo.ui.run_button(label="Save figures")

    if bundle_source_dir.name == "checkpoints":
        target_dir = bundle_source_dir.parent / "figures"
    else:
        target_dir = bundle_source_dir / "figures"

    figures_to_save = {
        k: v for k, v in {
            "loss_curves": fig_loss,
            "generalization_gap": fig_generalization,
            "weight_stats": fig_weights,
            "trajectory_predictions": fig_trajectories,
        }.items() if v is not None
    }

    mo.vstack([
        mo.md(f"### Save\n\nTarget directory: `{target_dir}`"),
        mo.md("Figures to write: " + ", ".join(f"`{k}.png`" for k in figures_to_save)),
        save_button,
    ])
    return figures_to_save, save_button, target_dir


@app.cell(hide_code=True)
def _(figures_to_save, save_button, target_dir):
    mo.stop(not save_button.value, mo.md(""))

    target_dir.mkdir(parents=True, exist_ok=True)
    _written = []
    for _name, _fig in figures_to_save.items():
        _path = target_dir / f"{_name}.png"
        _fig.savefig(_path, dpi=200, bbox_inches="tight")
        _written.append(_path)

    mo.callout(
        mo.md("Wrote:\n\n" + "\n".join(f"- `{p}`" for p in _written)),
        kind="success",
    )
    return


@app.cell(column=1, hide_code=True)
def _(exp_dir):
    _log = exp_dir / "slurm.log"
    mo.stop(not _log.exists(),
            mo.md("_No `slurm.log` present for this experiment._"))
    mo.vstack([mo.md("## SLURM log"), mo.plain_text(_log.read_text())])
    return


if __name__ == "__main__":
    app.run()
