import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np

    from experiments.seq2scal_models_history import (
        HistoryConfig,
        HistoryTrainingConfig,
        Seq2ScalarHistory,
    )
    from experiments.history_data import load_history_tracks
    from experiments.history_dataset import CHANNELS, NormStats, make_split


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from hastyplot import qplot

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

    device = get_device()
    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()
    repo_root = Path(__file__).resolve().parent.parent
    return (
        device,
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
    # Full-history seq2scalar (long-gap model)

    Encodes the **entire variable-length past** (packed LSTM, no fixed history
    window) and decodes the future from commanded fluence; minimal raw inputs
    `[cnr, fluence, fov_density, n_cells_200px]`, absolute-CNR MDN head. Trained
    with self-concat-random-break augmentation so it learns the
    two-experiment / inter-experiment-break structure. See
    `memory_characterization_plan.md`. Evaluate with the memory ladder in
    `cross_stitch_responder.py`.
    """)
    return


@app.cell
def _():
    # Frozen train-population standardization (computed once, see history_dataset).
    norm_stats = NormStats.load()
    return (norm_stats,)


@app.cell
def _(mo, parse_bool):
    MODE = mo.cli_args().get("mode", "train")
    IS_HEADLESS = "name" in mo.cli_args()
    EXPERIMENT_NAME = mo.cli_args().get("name", "lstm_seq2scal_history")
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
def _(IS_HEADLESS, MODE, form_from_configs, mo):
    if MODE == "train" and not IS_HEADLESS:
        form = form_from_configs(
            mo,
            {"m": HistoryConfig, "t": HistoryTrainingConfig},
            skip={"m": {"input_dim", "variant", "norm_channels", "norm_mean", "norm_std"}},
            radio_choices={
                "m": {
                    "encoder_type": ("lstm",),
                    "head_type": ("mdn", "gaussian"),
                    "film": ("none", "output", "hidden"),
                }
            },
        )
    else:
        form = None
    form if form is not None else mo.md("")
    return (form,)


@app.cell(hide_code=True)
def _(
    DRY_RUN,
    EXPERIMENT_NAME,
    IS_HEADLESS,
    MODE,
    form,
    mo,
    norm_stats,
    resolve_configs,
):
    _cfgs, data_source, _ctx_display = resolve_configs(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        form=form,
        config_classes={"m": HistoryConfig, "t": HistoryTrainingConfig},
        always={
            "m": {
                "input_dim": len(CHANNELS),
                "norm_channels": norm_stats.channels,
                "norm_mean": norm_stats.mean,
                "norm_std": norm_stats.std,
            }
        },
        default_source="real_plus_bo",
        experiment_name=EXPERIMENT_NAME,
        dry_run=DRY_RUN,
    )
    model_config, training_config = _cfgs["m"], _cfgs["t"]
    _ctx_display
    return model_config, training_config


@app.cell
def _(DRY_RUN, MODE, mo, norm_stats):
    if MODE == "train":
        _cnr, _feats, _cond, _meta = load_history_tracks("dataset.parquet")
        _split = make_split(_cond, seed=0)
        _tr, _va = _split["train"], _split["val"]
        if DRY_RUN:
            _tr, _va = _tr[:800], _va[:200]
        dataset_dict = {
            "train": (_cnr[_tr], _feats[_tr]),
            "val": (_cnr[_va], _feats[_va]),
            "stats": norm_stats,
        }
        n_train, n_val = len(_tr), len(_va)
        _msg = mo.md(f"**Data:** `{n_train}` train · `{n_val}` val cells · dry_run={DRY_RUN}")
    else:
        dataset_dict, n_train, n_val = None, 0, 0
        _msg = mo.md("**Load mode** — data not loaded.")
    _msg
    return dataset_dict, n_train, n_val


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
    dataset_dict,
    device,
    experiment_path,
    load_button,
    mo,
    model_config,
    results_base,
    run_experiment,
    train_button,
    training_config,
):
    artifacts = run_experiment(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        model_cls=Seq2ScalarHistory,
        model_config_cls=HistoryConfig,
        dataset=dataset_dict,
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
        f"**Run ready** · {type(model).__name__} · encoder=`{model_config_used.encoder_type}` "
        f"head=`{model_config_used.head_type}` film=`{model_config_used.film}` · "
        f"{sum(p.numel() for p in model.parameters()):,} params"
    )
    return artifacts, history


@app.cell
def _(history, mo, pl, qplot):
    if history["train_loss"]:
        loss_df = pl.DataFrame({
            "epoch": list(range(len(history["train_loss"]))) * 2,
            "nll": history["train_loss"] + history["val_loss"],
            "split": ["train"] * len(history["train_loss"]) + ["val"] * len(history["val_loss"]),
        })
        fig_loss = qplot(
            loss_df, "epoch", "nll", color="split", mark="line",
            title="NLL curves", height=300,
        )
    else:
        fig_loss = mo.md("No history (load mode).")
    fig_loss
    return (fig_loss,)


@app.cell
def _(
    artifacts,
    fig_loss,
    hostname,
    is_cluster,
    mo,
    n_train,
    n_val,
    save_bundle,
):
    save_status = save_bundle(
        mo=mo,
        is_headless="name" in mo.cli_args(),
        artifacts=artifacts,
        figures={"loss": fig_loss} if not isinstance(fig_loss, type(mo.md(""))) else {},
        metrics={"final_val_loss": (artifacts.history["val_loss"][-1] if artifacts.history["val_loss"] else None)},
        n_train=n_train,
        n_val=n_val,
        hostname=hostname,
        is_cluster=is_cluster,
    )
    save_status
    return


if __name__ == "__main__":
    app.run()
