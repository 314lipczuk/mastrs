import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns", app_title="Ensemble Ablation explorer")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import base64
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import altair as alt
    import polars as pl
    import torch
    from experiment import ExperimentBundle, discover_experiments, load_experiment_group
    from experiments.seq2seq_data import load as load_dataset, AVAILABLE_DATASETS, STIM_COLS
    from experiments.ablation import (
        window_samples,
        default_scenarios,
        ensemble_predict_scenarios,
    )
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

    mo.stop(
        not is_grouped,
        mo.md(f"`{exp_dir.name}` is not a grouped experiment — this reviewer only handles ensemble bundles."),
    )

    bundles = load_experiment_group(str(exp_dir))
    mo.stop(not bundles, mo.md("No member bundles loaded."))
    return bundles, exp_dir


@app.cell(hide_code=True)
def _(bundles, exp_dir):
    _first = next(iter(bundles.values()))
    _tc = _first.training_config

    mo.md(f"""
    ## {exp_dir.name}
    **Members:** {len(bundles)} (`{', '.join(bundles.keys())}`)
    **Model:** `{_first.model_type}` | **H:** {_tc.get("history_len")} | **F:** {_tc.get("future_len")}
    **Data source trained on:** `{_first.model_config.get("data_source", "?")}`
    """)
    return


@app.cell(hide_code=True)
def _(bundles):
    dev = utils.get_device()
    members = []
    for name, b in bundles.items():
        # Saved bundles reference __main__.Seq2Scalar (notebook-script execution);
        # redirect to the module-level class so reconstruct_model() can resolve it.
        if b.model_type.endswith("Seq2Scalar"):
            b.model_type = "experiments.seq2scal_model.Seq2Scalar"
        m = b.reconstruct_model().to(dev)
        m.eval()
        members.append((name, m))
    print(f"reconstructed {len(members)} members on {dev}")
    return dev, members


@app.cell(hide_code=True)
def _(bundles):
    _first = next(iter(bundles.values()))
    H_eval = int(_first.training_config["history_len"])
    F_eval = int(_first.training_config["future_len"])
    return F_eval, H_eval


@app.cell(hide_code=True)
def _(bundles):
    _first = next(iter(bundles.values()))
    _trained_on = _first.model_config.get("data_source", "synthetic")
    eval_source = mo.ui.dropdown(
        options=list(AVAILABLE_DATASETS),
        value=_trained_on,
        label="Eval data source",
    )
    stride_in = mo.ui.number(value=4, start=1, stop=50, step=1, label="Window stride")
    run_button = mo.ui.run_button(label="Run ensemble ablation")
    mo.hstack([eval_source, stride_in, run_button], justify="start", gap=1, align="center")
    return eval_source, run_button, stride_in


@app.cell
def _(F_eval, H_eval, dev, eval_source, members, run_button, stride_in):
    mo.stop(not run_button.value, mo.md("_Click **Run ensemble ablation** to execute inference._"))

    total = H_eval + F_eval
    cnr_real, stim_real, _cond = load_dataset(eval_source.value, window_size=total, stride=max(1, total // 4))
    samples = window_samples(cnr_real, stim_real, H_eval, F_eval, stride=int(stride_in.value))

    scenarios = default_scenarios(F_eval)
    models_only = [m for _, m in members]
    preds = ensemble_predict_scenarios(models_only, samples, scenarios, dev)

    targets = np.stack([s["dec_target"] for s in samples])       # (N, F)
    last_cnr = np.stack([s["full_window"][H_eval - 1] for s in samples])  # (N,)
    full_windows = np.stack([s["full_window"] for s in samples]) # (N, H+F)
    lights = np.stack([s["light"] for s in samples])             # (N, H+F)

    pred_abs = {
        name: last_cnr[None, :, None] + np.cumsum(arr, axis=2)    # (M, N, F)
        for name, arr in preds.items()
    }
    actual_abs = last_cnr[:, None] + np.cumsum(targets, axis=1)   # (N, F)

    mean_abs = {n: arr.mean(axis=0) for n, arr in pred_abs.items()}   # (N, F)
    std_abs = {n: arr.std(axis=0, ddof=1) for n, arr in pred_abs.items()}

    mo.md(f"**Ran {len(models_only)} members × {len(samples)} windows × {len(scenarios)} scenarios.**")
    return actual_abs, full_windows, lights, mean_abs, preds, std_abs


@app.cell(hide_code=True)
def _(actual_abs, mean_abs, preds, std_abs):
    _rows = []
    _true_mean = mean_abs["true"]
    _true_std = std_abs["true"]
    _ens_rmse_true = float(np.sqrt(((_true_mean - actual_abs) ** 2).mean()))

    for _name in preds:
        _mu = mean_abs[_name]
        _sg = std_abs[_name]
        _rmse_vs_true_mean = float(np.sqrt(((_mu - _true_mean) ** 2).mean()))
        _ens_spread = float(_sg.mean())
        _row = {
            "scenario": _name,
            "rmse_vs_actual": float(np.sqrt(((_mu - actual_abs) ** 2).mean())),
            "rmse_vs_true_scenario": _rmse_vs_true_mean,
            "mean_ensemble_sigma": _ens_spread,
        }
        _rows.append(_row)

    _md = ["| scenario | RMSE (vs actual) | RMSE (vs true-stim mean) | mean σ (members) |",
           "|----------|-----------------:|-------------------------:|-----------------:|"]
    for _r in _rows:
        _md.append(
            f"| `{_r['scenario']}` | {_r['rmse_vs_actual']:.5f} "
            f"| {_r['rmse_vs_true_scenario']:.5f} "
            f"| {_r['mean_ensemble_sigma']:.5f} |"
        )

    mo.md("### Per-scenario summary\n\n" + "\n".join(_md) + f"""

    - **RMSE (vs actual)** — only meaningful for `true`; counterfactuals have no ground truth, use for sanity only.
    - **RMSE (vs true-stim mean)** — how far a counterfactual pushes the ensemble mean away from the true-stim prediction. This is **control authority**.
    - **mean σ (members)** — ensemble disagreement under that scenario (model uncertainty).

    If control authority ≈ ensemble σ, stimulation choice is indistinguishable from model noise.
    """)
    return


@app.cell(hide_code=True)
def _(mean_abs, std_abs):
    # Control authority vs model uncertainty, per-sample scatter.
    _true_mean = mean_abs["true"]
    _true_std = std_abs["true"]

    _cfs = [k for k in mean_abs if k not in ("true", "zeroed")]

    _fig, _axes = plt.subplots(1, len(_cfs), figsize=(5 * len(_cfs), 4.5), squeeze=False)
    _axes = _axes.flatten()

    for _i, _name in enumerate(_cfs):
        _a = _axes[_i]
        _shift = np.abs(mean_abs[_name] - _true_mean).mean(axis=1)      # (N,)
        _sigma = _true_std.mean(axis=1)                                  # (N,)
        _m = max(_shift.max(), _sigma.max())
        _a.plot([0, _m], [0, _m], "k--", alpha=0.5, label="shift = σ")
        _a.scatter(_sigma, _shift, s=6, alpha=0.35, color="tab:blue")
        _a.set_xlabel("ensemble σ under true stim (model uncertainty)")
        _a.set_ylabel("|mean shift| vs true-stim (control authority)")
        _a.set_title(f"{_name} vs true")
        _a.grid(alpha=0.3)
        _a.legend(fontsize=8)

    _fig.suptitle("Above line = model distinguishes this control action from actual stim", fontsize=11)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(F_eval, mean_abs, std_abs):
    # Per-horizon: how does control authority and model uncertainty grow?
    _steps = np.arange(1, F_eval + 1)
    _true_mean = mean_abs["true"]
    _sigma_step = std_abs["true"].mean(axis=0)    # (F,)

    _rows = []
    for _name, _mu in mean_abs.items():
        if _name == "true":
            continue
        _shift_step = np.abs(_mu - _true_mean).mean(axis=0)
        for _k, (_s, _sh) in enumerate(zip(_steps, _shift_step)):
            _rows.append({"step": int(_s), "metric": f"shift ({_name})", "value": float(_sh)})
    for _k, _v in enumerate(_sigma_step):
        _rows.append({"step": int(_steps[_k]), "metric": "ensemble σ (true)", "value": float(_v)})

    _df = pl.DataFrame(_rows)
    _chart = alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("step:O", title="Future step"),
        y=alt.Y("value:Q", title="Absolute CNR units"),
        color=alt.Color("metric:N"),
        tooltip=["metric", "step", alt.Tooltip("value:Q", format=".5f")],
    ).properties(width=560, height=280, title="Per-horizon: scenario shift vs ensemble σ")
    _chart
    return


@app.cell(hide_code=True)
def _(actual_abs):
    n_show = mo.ui.slider(1, 8, value=4, label="Example cells")
    reshuffle = mo.ui.run_button(label="Reshuffle")
    mo.hstack([
        mo.md(f"### Example trajectories ({len(actual_abs)} windows)"),
        n_show, reshuffle,
    ], justify="start", gap=1, align="center")
    return n_show, reshuffle


@app.cell(hide_code=True)
def _(
    F_eval,
    H_eval,
    actual_abs,
    full_windows,
    lights,
    mean_abs,
    n_show,
    reshuffle,
    std_abs,
):
    _ = reshuffle.value
    _n = int(n_show.value)
    _ids = np.random.choice(len(actual_abs), size=min(_n, len(actual_abs)), replace=False)

    _colors = {
        "true": "#1f77b4",
        "zeroed": "#7f7f7f",
        "always_on": "#d62728",
        "always_off": "#2ca02c",
        "pulse_3on3off": "#9467bd",
    }

    _t_all = np.arange(H_eval + F_eval)
    _t_fut = np.arange(H_eval, H_eval + F_eval)

    _fig, _axes = plt.subplots(len(_ids), 1, figsize=(11, 3 * len(_ids)), sharex=True)
    if len(_ids) == 1:
        _axes = [_axes]

    for _j, (_idx, _ax) in enumerate(zip(_ids, _axes)):
        _ax2 = _ax.twinx()
        _ax2.fill_between(_t_all, lights[_idx], alpha=0.12, color="orange")
        _ax2.set_ylim(0, max(lights[_idx].max() * 3, 1))
        _ax2.set_yticks([])

        _ax.plot(_t_all, full_windows[_idx], "k-", alpha=0.5, label="actual CNR")
        _ax.plot(_t_fut, actual_abs[_idx], "k--", lw=1.3, alpha=0.7, label="true future")

        for _name, _mu in mean_abs.items():
            _sg = std_abs[_name]
            _c = _colors.get(_name, "tab:gray")
            _ax.plot(_t_fut, _mu[_idx], color=_c, lw=1.5, label=_name)
            _ax.fill_between(_t_fut, _mu[_idx] - _sg[_idx], _mu[_idx] + _sg[_idx],
                             color=_c, alpha=0.12)

        _ax.axvline(H_eval, color="gray", ls=":", alpha=0.5)
        _ax.set_ylabel("CNR")
        _ax.set_title(f"window {_idx}", fontsize=9, loc="left")
        if _j == 0:
            _ax.legend(fontsize=7, ncol=3, loc="upper right")

    _axes[-1].set_xlabel("Time step")
    _fig.suptitle("Ensemble mean ± σ per scenario", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading guide

    - **Per-scenario summary table** — `rmse_vs_true_scenario` is the key column. Small values = scenario doesn't change predictions → weak control authority.
    - **Scatter: shift vs ensemble σ** — points above the diagonal are windows where the model separates that counterfactual from actual stim beyond its own uncertainty.
    - **Per-horizon chart** — if ensemble σ grows faster than scenario shift, long-horizon MPC is hopeless.
    - **Example trajectories** — eyeball overlap: if `always_on`/`always_off` bands overlap `true`, stimulation is not driving the prediction.
    """)
    return


if __name__ == "__main__":
    app.run()
