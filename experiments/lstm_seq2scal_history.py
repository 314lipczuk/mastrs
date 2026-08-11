import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    import numpy as np

    from optoerk.models.seq2scal_history import (
        HistoryConfig,
        HistoryTrainingConfig,
        Seq2ScalarHistory,
    )
    from optoerk.core.utils import materials_path
    from optoerk.data.history_data import load_history_tracks, resolve_feature_set
    from optoerk.data.history_dataset import (
        NormStats,
        compute_norm_stats,
        make_split,
    )


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from hastyplot import qplot

    from optoerk.core.utils import (
        get_device,
        get_username,
        running_on_cluster,
        results_write_path,
        results_read_sources,
        parse_bool,
        scan_experiment_dirs,
    )
    from optoerk.scaffold import (
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
def _(CNR_MODE, FEATURES, MODE, mo):
    # Training bundle to load (parquet in materials/). Default = the full `all`
    # bundle; jobs override via --dataset (e.g. dataset_niesen.parquet).
    # Default bundle carries the REAL mCitrine optoRTK measurement (built by
    # experiments/build_mcitrine_dataset.py). The older dataset_all.parquet has
    # no `mcitrine` column and will now fail loudly rather than fall back to the
    # C0 surrogate it used to use.
    DATASET = mo.cli_args().get("dataset", "dataset_all_mcitrine.parquet")
    if MODE == "train":
        hist_cnr, hist_feats, hist_cond, _hist_meta = load_history_tracks(
            materials_path(DATASET), cnr_mode=CNR_MODE, features=FEATURES
        )
        hist_split = make_split(hist_cond, seed=0)
    else:
        hist_cnr = hist_feats = hist_split = None
    return DATASET, hist_cnr, hist_feats, hist_split


@app.cell
def _(CNR_MODE, FEATURES, MODE, hist_cnr, hist_feats, hist_split):
    # Standardization computed from THIS bundle's train split (not a frozen file),
    # so stats always match the loaded data/features. Stamped onto the model
    # config for eval/serving. Load mode: values unused (config comes from bundle).
    if MODE == "train":
        norm_stats = compute_norm_stats(
            hist_cnr, hist_feats, hist_split["train"], features=FEATURES
        )
    else:
        norm_stats = NormStats.load(cnr_mode=CNR_MODE)
    return (norm_stats,)


@app.cell
def _(mo, parse_bool):
    MODE = mo.cli_args().get("mode", "train")
    IS_HEADLESS = "name" in mo.cli_args()
    EXPERIMENT_NAME = mo.cli_args().get("name", "lstm_seq2scal_history")
    DRY_RUN = parse_bool(mo.cli_args().get("dry_run", True))
    # cnr channel semantics: "norm" = baseline-normalized cnr_median_norm (default),
    # "raw" = absolute cnr_median. Stamped onto the model config so eval/serving
    # know whether to online-normalize and which frozen z-score stats to load.
    CNR_MODE = mo.cli_args().get("cnr_mode", "norm")
    # Which input channels the model gets, and what conditions FiLM. These are
    # the axes of the encoding comparison (see optoerk.data.history_data
    # FEATURE_SETS and HistoryConfig.film_cond); everything else is held fixed
    # across the runs so the comparison is one-factor.
    FEATURE_SET = mo.cli_args().get("feature_set", "base")
    FILM_COND = mo.cli_args().get("film_cond", "fluence")
    FEATURES, FUTURE_CHANNELS = resolve_feature_set(FEATURE_SET)

    if MODE not in ("train", "load"):
        raise ValueError(f"--mode must be 'train' or 'load', got {MODE!r}")
    if CNR_MODE not in ("norm", "raw"):
        raise ValueError(f"--cnr_mode must be 'norm' or 'raw', got {CNR_MODE!r}")

    mo.md(
        f"**Mode:** `{MODE}` · **CNR:** `{CNR_MODE}` · **Headless:** `{IS_HEADLESS}` · "
        f"**Experiment:** `{EXPERIMENT_NAME}` · **Dry run:** `{DRY_RUN}`\n\n"
        f"**Feature set:** `{FEATURE_SET}` → `{FEATURES}` · "
        f"**FiLM conditioned on:** `{FILM_COND}`"
    )
    return (
        CNR_MODE,
        DRY_RUN,
        EXPERIMENT_NAME,
        FEATURES,
        FEATURE_SET,
        FILM_COND,
        FUTURE_CHANNELS,
        IS_HEADLESS,
        MODE,
    )


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
            skip={"m": {"input_dim", "variant", "cnr_mode", "norm_channels", "norm_mean", "norm_std"}},
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
    CNR_MODE,
    DRY_RUN,
    EXPERIMENT_NAME,
    FILM_COND,
    FUTURE_CHANNELS,
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
                "input_dim": len(norm_stats.channels),
                "stim_dim": len(FUTURE_CHANNELS),
                "future_channels": FUTURE_CHANNELS,
                "film_cond": FILM_COND,
                "cnr_mode": CNR_MODE,
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
def _(
    DATASET,
    DRY_RUN,
    MODE,
    hist_cnr,
    hist_feats,
    hist_split,
    mo,
    norm_stats,
):
    if MODE == "train":
        _tr, _va = hist_split["train"], hist_split["val"]
        if DRY_RUN:
            _tr, _va = _tr[:800], _va[:200]
        dataset_dict = {
            "train": (hist_cnr[_tr], hist_feats[_tr]),
            "val": (hist_cnr[_va], hist_feats[_va]),
            "stats": norm_stats,
        }
        n_train, n_val = len(_tr), len(_va)
        _msg = mo.md(f"**Data:** `{DATASET}` · `{n_train}` train · `{n_val}` val cells · dry_run={DRY_RUN}")
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
    FEATURES,
    FEATURE_SET,
    FILM_COND,
    MODE,
    artifacts,
    hist_cnr,
    hist_feats,
    hist_split,
    mo,
    norm_stats,
):
    # --- ENCODING COMPARISON METRICS ---------------------------------------
    # Computed on the TEST split, which no run ever trains or early-stops on, and
    # with the same seed everywhere — so the four variants are scored on the same
    # samples in the same order and the comparison notebook can pair them.
    #
    # NLL alone cannot decide this. A covariate can shave the average loss while
    # being used for the wrong thing, so three further readouts are recorded:
    #   gain_spearman   does the model's believed dose effect rise with the cell's
    #                   expression? optoRTK expression IS a gain, so this is the
    #                   question the encoding variants exist to answer.
    #   perm_delta_nll  shuffle a channel across cells; a channel whose shuffle is
    #                   free is a channel the model ignores. This is what decides
    #                   whether `nuc_area` earns its place.
    #   decile_mae      error by expression decile. Helping on average while
    #                   leaving the high expressers just as wrong is not success.
    mo.stop(MODE != "train" or artifacts is None, mo.md("_No metrics — not a training run._"))

    from torch.utils.data import DataLoader

    from optoerk.data.history_dataset import HistoryDataset, collate_history
    from optoerk.eval.encoding_metrics import (
        dose_effect,
        evaluate,
        permutation_importance,
        stratified_error,
    )

    _model = artifacts.model.eval()
    _mcfg = _model.cfg
    _te = hist_split["test"]
    _ds = HistoryDataset(
        hist_cnr[_te], hist_feats[_te], np.arange(len(_te)), norm_stats,
        F=_mcfg.future_len, t_min=10, p_concat=0.0,
        future_channels=_mcfg.future_channels, seed=0,
    )
    _dl = DataLoader(_ds, batch_size=256, shuffle=False, collate_fn=collate_history)

    _ev = evaluate(_model, _dl)
    _de = dose_effect(_model, _dl)
    _st = stratified_error(_ev["abs_err_per_sample"], _de["expr_std"])
    _scored = [_c for _c in norm_stats.channels if _c != "cnr"]   # cnr is the target
    _perm = {
        _c: permutation_importance(_model, _dl, _c)["delta_nll"] for _c in _scored
    }
    # ...and the same shuffles with each channel moved ALONE. Not a valid input
    # state, so not an importance — a decomposition. Without it the interaction
    # variant is unreadable: linked, its `optortk_expr` scores ~9x every other
    # run, which looks like the encoding made the model use expression harder.
    # Decomposed, that number is almost entirely `u_t_x_expr` acting as a second
    # dose channel, and expression alone is used LESS than in the baseline.
    _perm_solo = {
        _c: permutation_importance(_model, _dl, _c, linked=False)["delta_nll"]
        for _c in _scored
    }

    encoding_metrics = {
        "feature_set": FEATURE_SET,
        "film_cond": FILM_COND,
        "features": FEATURES,
        "channels": list(norm_stats.channels),
        "n_test_samples": int(len(_ev["nll"])),
        "test_nll": float(_ev["nll"].mean()),
        "test_mae": _ev["mae"],
        "test_rmse": _ev["rmse"],
        "mae_per_step": [float(x) for x in _ev["mae_per_step"]],
        "rmse_per_step": [float(x) for x in _ev["rmse_per_step"]],
        "cov68": _ev["cov68"],
        "cov95": _ev["cov95"],
        "z_std": _ev["z_std"],
        "gain_spearman": _de["gain_spearman"],
        "mean_dose_effect": _de["mean_effect"],
        "perm_delta_nll": {k: float(v) for k, v in _perm.items()},
        "perm_delta_nll_solo": {k: float(v) for k, v in _perm_solo.items()},
        "decile_mae": [float(x) for x in _st["decile_mae"]],
        "decile_mae_spread": _st["spread"],
        # per-sample, for paired tests in the comparison notebook
        "_per_sample_nll": [float(x) for x in _ev["nll"]],
        "_per_sample_abs_err": [float(x) for x in _ev["abs_err_per_sample"]],
    }

    _rows = "\n".join(
        f"| `{_k}` | {_v:+.4f} | {encoding_metrics['perm_delta_nll_solo'][_k]:+.4f} |"
        for _k, _v in sorted(
            encoding_metrics["perm_delta_nll"].items(), key=lambda kv: -kv[1])
    )
    mo.md(
        f"""
    ### Encoding metrics — `{FEATURE_SET}` / FiLM on `{FILM_COND}`

    **{encoding_metrics['n_test_samples']}** test samples ·
    NLL **{encoding_metrics['test_nll']:.4f}** ·
    MAE **{encoding_metrics['test_mae']:.4f}** ·
    RMSE **{encoding_metrics['test_rmse']:.4f}**

    Calibration: 68% interval covers **{encoding_metrics['cov68']:.1%}**,
    95% covers **{encoding_metrics['cov95']:.1%}** (nominal 68 / 95).

    **Gain test:** Spearman(expression, believed dose effect) =
    **{encoding_metrics['gain_spearman']:+.3f}**. Positive means the model
    predicts a larger light effect for higher expressers, which is what optoRTK
    expression physically is.

    **Permutation importance** (NLL increase when the channel is shuffled across
    cells; ~0 means unused):

    | channel | Δ NLL (linked) | Δ NLL (alone) |
    |---|---|---|
    {_rows}

    *linked* moves a channel with anything derived from it, which is the causal
    importance; *alone* is the decomposition that says which of the linked group
    the effect actually came from.

    Error by expression decile spans **{encoding_metrics['decile_mae_spread']:.4f}** CNR.
    """
    )
    return (encoding_metrics,)


@app.cell
def _(
    artifacts,
    encoding_metrics,
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
        metrics={
            "final_val_loss": (
                artifacts.history["val_loss"][-1] if artifacts.history["val_loss"] else None
            ),
            # The whole point of the four-run comparison travels in the bundle,
            # so experiments/encoding_comparison.py needs nothing but the bundles.
            "encoding": encoding_metrics,
        },
        n_train=n_train,
        n_val=n_val,
        hostname=hostname,
        is_cluster=is_cluster,
    )
    save_status
    return


if __name__ == "__main__":
    app.run()
