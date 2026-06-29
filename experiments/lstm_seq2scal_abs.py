import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np

    from experiments.seq2scal_models_abs import (
        ModelConfig,
        TrainingConfig,
        Seq2ScalarSeq,
        prepare_data,
    )


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from hastyplot import qplot

    from experiment import load_experiment
    from utils import (
        get_device,
        get_username,
        running_on_cluster,
        results_write_path,
        results_read_sources,
        parse_bool,
        scan_experiment_dirs,
    )
    from experiments.scaffold import (
        form_from_configs,
        resolve_configs,
        run_experiment,
        save_bundle,
    )
    from experiments.seq2seq_data import AVAILABLE_DATASETS
    from experiments import eval_seq2scal_abs as ev

    device = get_device()
    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()
    repo_root = Path(__file__).resolve().parent.parent
    return (
        AVAILABLE_DATASETS,
        device,
        ev,
        form_from_configs,
        hostname,
        is_cluster,
        mo,
        parse_bool,
        pl,
        qplot,
        repo_root,
        resolve_configs,
        results_base,
        results_read_sources,
        run_experiment,
        save_bundle,
        scan_experiment_dirs,
    )


@app.cell
def _(mo):
    mo.md("""
    # Seq2Scalar absolute-output driver

    Same configurable encoder-decoder forecaster as `lstm_seq2scal_variant`,
    but the head predicts the **absolute CNR value** at each future step
    directly (per-step sigma is the band around that absolute value) instead
    of per-step delta-CNR. Uses `seq2scal_models_abs` + `eval_seq2scal_abs`.
    Architecture variant (Gaussian vs MDN head, continuous decoder, stim-gated
    init, FiLM, per-step sigma bias) is still selected by `ModelConfig` flags.
    """)
    return


@app.cell
def _(mo, parse_bool):
    MODE = mo.cli_args().get("mode", "train")
    IS_HEADLESS = "name" in mo.cli_args()
    EXPERIMENT_NAME = mo.cli_args().get("name", "lstm_seq2scal_abs")
    DRY_RUN = parse_bool(mo.cli_args().get("dry_run", True))

    if MODE not in ("train", "load"):
        raise ValueError(f"--mode must be 'train' or 'load', got {MODE!r}")

    mo.md(
        f"**Mode:** `{MODE}` · **Headless:** `{IS_HEADLESS}` · "
        f"**Experiment:** `{EXPERIMENT_NAME}` · **Dry run:** `{DRY_RUN}`"
    )
    return DRY_RUN, EXPERIMENT_NAME, IS_HEADLESS, MODE


@app.cell
def _(IS_HEADLESS, MODE, mo, repo_root, results_read_sources):
    if MODE == "load" and not IS_HEADLESS:
        _sources = results_read_sources(repo_root)
        source_selector = mo.ui.dropdown(
            options=list(_sources.keys()), value="Local", label="Results source",
        )
    else:
        source_selector = None
    source_selector if source_selector is not None else mo.md("")
    return (source_selector,)


@app.cell
def _(
    IS_HEADLESS,
    MODE,
    mo,
    repo_root,
    results_read_sources,
    scan_experiment_dirs,
    source_selector,
):
    if MODE == "load" and not IS_HEADLESS and source_selector is not None:
        _src_root = Path(results_read_sources(repo_root)[source_selector.value])
        _choices = scan_experiment_dirs(_src_root)
        if _choices:
            experiment_picker = mo.ui.dropdown(
                options=_choices, value=_choices[0], label="Experiment run",
            )
            load_button = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Load")
            source_root = _src_root
            _picker_ui = mo.vstack([experiment_picker, load_button])
        else:
            experiment_picker = load_button = source_root = None
            _picker_ui = mo.md(f"No experiments under `{_src_root}`.")
    else:
        experiment_picker = load_button = source_root = None
        _picker_ui = mo.md("")
    _picker_ui
    return experiment_picker, load_button, source_root


@app.cell
def _(experiment_picker, source_root):
    if experiment_picker is not None and source_root is not None:
        experiment_path = source_root / experiment_picker.value
    else:
        experiment_path = None
    return (experiment_path,)


@app.cell
def _(AVAILABLE_DATASETS, IS_HEADLESS, MODE, form_from_configs, mo):
    if MODE == "train" and not IS_HEADLESS:
        form = form_from_configs(
            mo,
            {"m": ModelConfig, "t": TrainingConfig},
            skip={"m": {"encoder_dim", "stim_dim", "variant"}},
            radio_choices={
                "m": {
                    "data_source": AVAILABLE_DATASETS,
                    "head_type": ("mdn", "gaussian"),
                    "decoder_type": ("encdec", "continuous"),
                    "film": ("none", "output", "hidden"),
                }
            },
        )
    else:
        form = None
    form if form is not None else mo.md("")
    return (form,)


@app.cell(hide_code=True)
def _(DRY_RUN, EXPERIMENT_NAME, IS_HEADLESS, MODE, form, mo, resolve_configs):
    _cfgs, data_source, _ctx_display = resolve_configs(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        form=form,
        config_classes={"m": ModelConfig, "t": TrainingConfig},
        always={"m": {"encoder_dim": 5, "stim_dim": 1}},
        default_source="real_plus_bo",
        experiment_name=EXPERIMENT_NAME,
        dry_run=DRY_RUN,
    )
    model_config, training_config = _cfgs["m"], _cfgs["t"]
    _ctx_display
    return data_source, model_config, training_config


@app.cell
def _(DRY_RUN, data_source, mo, model_config):
    _slow = model_config.ewma_slow_alpha if model_config is not None else 0.05
    _fast = model_config.ewma_fast_alpha if model_config is not None else 0.30
    _prepend = (
        model_config.history_len
        if (model_config is not None and model_config.prepend_baseline)
        else 0
    )
    _split = model_config.split_regime if model_config is not None else "random"
    prep = prepare_data(
        data_source,
        ewma_slow_alpha=_slow,
        ewma_fast_alpha=_fast,
        dry_run=DRY_RUN,
        baseline_prepend=_prepend,
        split_regime=_split,
    )
    mo.md(
        f"**Data:** `{data_source}` — {prep.n_cells} cells · "
        f"train={len(prep.train[0])} val={len(prep.val[0])} test={len(prep.test[0])} "
        f"· dry_run={DRY_RUN}"
    )
    return (prep,)


@app.cell(hide_code=True)
def _(IS_HEADLESS, MODE, mo):
    if MODE == "train" and not IS_HEADLESS:
        train_button = mo.ui.run_button(label="Start training")
    else:
        train_button = None
    train_button if train_button is not None else mo.md("")
    return (train_button,)


@app.cell
def _(
    EXPERIMENT_NAME,
    IS_HEADLESS,
    MODE,
    device,
    experiment_path,
    load_button,
    mo,
    model_config,
    prep,
    results_base,
    run_experiment,
    train_button,
    training_config,
):
    artifacts = run_experiment(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        model_cls=Seq2ScalarSeq,
        model_config_cls=ModelConfig,
        dataset={"train": prep.train, "val": prep.val},
        model_config=model_config,
        training_config=training_config,
        device=device,
        experiment_name=EXPERIMENT_NAME,
        results_base=results_base,
        experiment_path=experiment_path,
        load_button=load_button,
        train_button=train_button,
    )
    model = artifacts.model
    history = artifacts.history
    model_config_used = artifacts.model_config
    mo.md(
        f"**Run ready** · {type(model).__name__} · "
        f"variant=`{model_config_used.variant}` · "
        f"head=`{model_config_used.head_type}` decoder=`{model_config_used.decoder_type}` "
        f"stim_init={model_config_used.stim_init} film=`{model_config_used.film}` "
        f"sigma_step_bias={model_config_used.sigma_step_bias} · "
        f"{sum(p.numel() for p in model.parameters()):,} params"
    )
    return artifacts, history, model, model_config_used


@app.cell
def _(MODE, data_source, model_config_used, prep):
    if MODE == "load" and model_config_used.data_source != data_source:
        test_prep = prepare_data(
            model_config_used.data_source,
            ewma_slow_alpha=model_config_used.ewma_slow_alpha,
            ewma_fast_alpha=model_config_used.ewma_fast_alpha,
            baseline_prepend=(
                model_config_used.history_len
                if model_config_used.prepend_baseline
                else 0
            ),
            split_regime=model_config_used.split_regime,
        )
        test_arrays = test_prep.test
        test_ood_arrays = test_prep.test_ood
    else:
        test_arrays = prep.test
        test_ood_arrays = prep.test_ood
    return test_arrays, test_ood_arrays


@app.cell
def _(history, pl, qplot):
    loss_df = pl.DataFrame({
        "epoch": list(range(len(history["train_loss"]))) * 2,
        "nll": history["train_loss"] + history["val_loss"],
        "split": (
            ["train"] * len(history["train_loss"])
            + ["val"] * len(history["val_loss"])
        ),
    })
    fig_loss = qplot(
        loss_df, "epoch", "nll", color="split", mark="line",
        title="NLL curves", height=300,
    )
    fig_loss
    return (fig_loss,)


@app.cell
def _(device, ev, mo, model, model_config_used, test_arrays, test_ood_arrays):
    results_by_set = {
        "indist": ev.evaluate(
            model, test_arrays, model_config_used, device=device, test_stride=10
        )
    }
    for _cond, _arrs in test_ood_arrays.items():
        if len(_arrs[0]) == 0:
            continue
        results_by_set[f"ood/{_cond}"] = ev.evaluate(
            model, _arrs, model_config_used, device=device, test_stride=10
        )
    eval_result = results_by_set["indist"]
    step0_diag = ev.step0_diagnostics(
        model, test_arrays, model_config_used, device=device, test_stride=10
    )
    summary = ev.summary_table(eval_result)
    print(summary)
    if len(results_by_set) > 1:
        print()
        print("=== Per-test-set headlines ===")
        for _set, _res in results_by_set.items():
            _h = _res.headline
            print(
                f"  {_set:22s} test_nll={_h.get('test_nll', float('nan')):.4f}  "
                f"test_crps={_h.get('test_crps', float('nan')):.4f}  "
                f"q4_mae={_h.get('q4_mae', float('nan')):.4f}  "
                f"n_windows={_res.full.get('n_test_windows', '?')}"
            )
    print("\n=== STEP-0 DIAGNOSTICS (indist) ===")
    print(f"step0 mean residual : {step0_diag['step0_mean_residual']:+.5f}")
    print(f"step0 residual std  : {step0_diag['step0_residual_std']:.5f}")
    print(f"steps1+ residual std: {step0_diag['steps1plus_residual_std']:.5f}")
    for _r in step0_diag["step0_by_boundary_slope"]:
        print(
            f"  boundary-slope Q{_r['slope_quartile']} "
            f"(n={_r['n']}): step0 |resid|={_r['step0_abs_resid']:.5f}"
        )
    mo.md(f"## Evaluation\n\n```\n{summary}\n```")
    return eval_result, results_by_set, step0_diag, summary


@app.cell
def _(eval_result, pl, qplot):
    _ps = eval_result.per_step
    per_step_df = pl.DataFrame({
        "step": list(range(1, len(_ps["nll"]) + 1)) * 3,
        "value": _ps["nll"] + _ps["mae"] + _ps["coverage_1sigma"],
        "metric": (
            ["nll"] * len(_ps["nll"])
            + ["mae"] * len(_ps["mae"])
            + ["coverage_1sigma"] * len(_ps["coverage_1sigma"])
        ),
    })
    fig_per_step = qplot(
        per_step_df, "step", "value", color="metric", group="metric",
        facet_wrap="metric", columns=3, mark="line",
        title="Per-forecast-step metrics", height=240,
    )
    fig_per_step
    return (fig_per_step,)


@app.cell
def _(pl, qplot, step0_diag):
    _d = step0_diag["per_step"]
    diag_df = pl.DataFrame({
        "step": list(range(len(_d["mae"]))) * 2,
        "value": _d["mean_residual"] + _d["residual_std"],
        "metric": (
            ["mean_residual"] * len(_d["mean_residual"])
            + ["residual_std"] * len(_d["residual_std"])
        ),
    })
    fig_step0 = qplot(
        diag_df, "step", "value", color="metric", mark="line",
        title="Step-0 diagnostics: per-step residual bias and spread",
        height=280,
    )
    fig_step0
    return (fig_step0,)


@app.cell
def _(
    device,
    ev,
    eval_result,
    model,
    model_config_used,
    pl,
    qplot,
    test_arrays,
):
    abs_out = ev.absolute_outputs(
        model, test_arrays, model_config_used, device=device, test_stride=10
    )
    print("=== ABSOLUTE-CNR OUTPUT METRICS ===")
    for _k, _v in eval_result.full["absolute"].items():
        print(f"  {_k}: {_v:.5f}")
    _ps = eval_result.per_step
    abs_step_df = pl.DataFrame({
        "step": list(range(1, len(_ps["abs_mae"]) + 1)) * 2,
        "value": _ps["abs_mae"] + _ps["abs_rmse"],
        "metric": (
            ["abs_mae"] * len(_ps["abs_mae"])
            + ["abs_rmse"] * len(_ps["abs_rmse"])
        ),
    })
    fig_abs_step = qplot(
        abs_step_df, "step", "value", color="metric", mark="line",
        title="Absolute-CNR error vs forecast step", height=280,
    )
    fig_abs_step
    return abs_out, fig_abs_step


@app.cell
def _(abs_out, pl, qplot):
    _rs = abs_out["resp_std"]
    _idx = np.where(_rs >= np.quantile(_rs, 0.75))[0][:6]
    _F = abs_out["pred"].shape[1]
    _rows = []
    for _w, _i in enumerate(_idx):
        for _s in range(_F):
            _rows.append((f"win{_w}", _s + 1, float(abs_out["pred"][_i, _s]), "pred"))
            _rows.append((f"win{_w}", _s + 1, float(abs_out["target"][_i, _s]), "true"))
    abs_traj_df = pl.DataFrame(
        _rows, schema=["window", "step", "cnr", "kind"], orient="row"
    )
    fig_abs_traj = qplot(
        abs_traj_df, "step", "cnr", color="kind", group="window",
        facet_wrap="window", columns=3, mark="line",
        title="Absolute CNR: predicted vs true (Q4 response windows)", height=320,
    )
    fig_abs_traj
    return (fig_abs_traj,)


@app.cell
def _(
    artifacts,
    eval_result,
    fig_abs_step,
    fig_abs_traj,
    fig_loss,
    fig_per_step,
    fig_step0,
    hostname,
    is_cluster,
    mo,
    prep,
    results_by_set,
    save_bundle,
    step0_diag,
    summary,
):
    metrics = {
        **eval_result.headline,
        "summary_table": summary,
        "per_step": eval_result.per_step,
        "full": eval_result.full,
        "step0_diagnostics": step0_diag,
        "by_test_set": {
            k: {"headline": v.headline, "per_step": v.per_step, "full": v.full}
            for k, v in results_by_set.items()
        },
        "splits": prep.splits,
    }
    save_status = save_bundle(
        mo=mo,
        is_headless="name" in mo.cli_args(),
        artifacts=artifacts,
        figures={
            "loss": fig_loss,
            "per_step": fig_per_step,
            "step0": fig_step0,
            "abs_step": fig_abs_step,
            "abs_traj": fig_abs_traj,
        },
        metrics=metrics,
        n_train=len(prep.train[0]),
        n_val=len(prep.val[0]),
        hostname=hostname,
        is_cluster=is_cluster,
    )
    save_status
    return


if __name__ == "__main__":
    app.run()
