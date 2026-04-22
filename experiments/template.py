"""Starter template for a new seq2* experiment notebook.

Copy this file, rename it, then edit the TODO-marked sections:
  - ``ModelConfig`` / ``TrainingConfig`` — per-experiment hyperparams
  - ``Model`` class — define your `nn.Module` and a static ``fit`` method
  - dataset cell — load/window data into a ``dict`` the ``fit`` method expects
  - viz cells — add plots that return figure handles named ``fig_*``
  - save cell — list the figure keys to persist

The scaffold takes care of: MODE/IS_HEADLESS parsing, config forms, load-mode
experiment picker, train-mode run button, tracker wiring, and save gating.
"""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    from pydantic import BaseModel, ConfigDict, Field

    from experiments.scaffold import TrainContext

    class ModelConfig(BaseModel):
        """TODO: per-experiment model hyperparameters."""

        model_config = ConfigDict(extra="forbid")
        encoder_dim: int = Field(..., ge=1)
        stim_dim: int = Field(..., ge=1)
        hidden_dim: int = Field(64, ge=1)
        dropout: float = Field(0.1, ge=0.0, le=0.9)
        history_len: int = Field(30, ge=1)
        future_len: int = Field(5, ge=1)
        data_source: str = "synthetic"
        variant: str = "template"

    class TrainingConfig(BaseModel):
        """TODO: per-experiment training hyperparameters."""

        model_config = ConfigDict(extra="forbid")
        lr: float = 1e-3
        weight_decay: float = 1e-4
        epochs: int = 100
        batch_size: int = 64
        patience: int = 50
        grad_clip: float = 1.0

    class Model(nn.Module):
        """TODO: your model.

        Contract the scaffold depends on:
          - constructor accepts a ``ModelConfig`` instance OR a ``dict`` (so
            ``bundle.reconstruct_model()`` round-trips).
          - ``Model.fit(dataset, ctx: TrainContext) -> (model, history_dict)``
            is the only hard entry-point. It must:
              * build model from ``ctx.model_config``
              * run the training loop, honoring ``ctx.training_config``
              * call ``ctx.progress_cb(epoch, total, metrics_dict)`` if set
              * call ``ctx.tracker.checkpoint(model, training_results=...)``
                per epoch when ``ctx.tracker`` is not None
              * return ``(model, history_dict)`` with at least
                ``history["train_loss"]`` and ``history["val_loss"]``.
        """

        Config = ModelConfig
        TrainingConfigCls = TrainingConfig

        def __init__(self, cfg=None, **kwargs):
            super().__init__()
            if cfg is None:
                cfg = kwargs
            if isinstance(cfg, dict):
                cfg = ModelConfig.model_validate(cfg)
            self.cfg = cfg
            # TODO: build modules from cfg
            self.placeholder = nn.Linear(cfg.encoder_dim, 1)

        def forward(self, x):
            # TODO
            return self.placeholder(x)

        @staticmethod
        def fit(dataset, ctx: TrainContext):
            # TODO: real training loop. Minimal stub so the notebook runs end-to-end.
            mcfg = ctx.model_config
            tcfg = ctx.training_config
            model = Model(mcfg).to(ctx.device)
            hist = {"train_loss": [], "val_loss": []}
            for ep in range(tcfg.epochs):
                t = float(1.0 / (ep + 1))
                v = float(1.0 / (ep + 1) + 0.01)
                hist["train_loss"].append(t)
                hist["val_loss"].append(v)
                if ctx.progress_cb is not None:
                    ctx.progress_cb(ep, tcfg.epochs, {"train": t, "val": v})
                if ctx.tracker is not None:
                    ctx.tracker.checkpoint(model, training_results={"history": hist})
            return model, hist


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from hastyplot import qplot

    from experiment import load_experiment
    from utils import (
        get_device,
        get_username,
        parse_bool,
        results_read_sources,
        results_write_path,
        running_on_cluster,
        scan_experiment_dirs,
    )
    from experiments.seq2seq_data import (
        AVAILABLE_DATASETS,
        STIM_COLS,
        load as load_dataset,
    )
    from experiments.scaffold import (
        form_from_configs,
        resolve_configs,
        run_experiment,
        save_bundle,
    )

    device = get_device()
    n_stim = len(STIM_COLS)
    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()
    repo_root = Path(__file__).resolve().parent.parent
    return (
        AVAILABLE_DATASETS,
        device,
        form_from_configs,
        hostname,
        is_cluster,
        load_dataset,
        load_experiment,
        mo,
        n_stim,
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
def _(mo, parse_bool):
    MODE = mo.cli_args().get("mode", "train")
    IS_HEADLESS = "name" in mo.cli_args()
    EXPERIMENT_NAME = mo.cli_args().get("name", "template_experiment")
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
            load_button = mo.ui.button(
                value=0, on_click=lambda n: n + 1, label="Load",
            )
            source_root = _src_root
            _picker_ui = mo.vstack([experiment_picker, load_button])
        else:
            experiment_picker = None
            load_button = None
            source_root = None
            _picker_ui = mo.md(f"No experiments under `{_src_root}`.")
    else:
        experiment_picker = None
        load_button = None
        source_root = None
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
            radio_choices={"m": {"data_source": AVAILABLE_DATASETS}},
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
    n_stim,
    resolve_configs,
):
    _cfgs, data_source, _ctx_display = resolve_configs(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        form=form,
        config_classes={"m": ModelConfig, "t": TrainingConfig},
        always={"m": {"encoder_dim": 1 + n_stim, "stim_dim": n_stim}},
        experiment_name=EXPERIMENT_NAME,
        dry_run=DRY_RUN,
    )
    model_config, training_config = _cfgs["m"], _cfgs["t"]

    _ctx_display
    return data_source, model_config, training_config


@app.cell
def _(DRY_RUN, data_source, load_dataset, mo):
    # Loaders return per-cell full trajectories (cnr/stim are object arrays
    # for real / variable-length, 2D arrays for synthetic). Downstream
    # Seq2SeqDataset is duck-typed — fancy indexing by cell id works either
    # way, so train/val/test split by cell id is all that's needed here.
    cnr_all, stim_all, conditions_all = load_dataset(data_source)

    _ids = np.arange(len(cnr_all))
    _tr, _te = train_test_split(_ids, test_size=0.2, random_state=42)
    _tr, _va = train_test_split(_tr, test_size=0.125, random_state=42)
    if DRY_RUN:
        _tr = _tr[: min(len(_tr), 800)]
        _va = _va[: min(len(_va), 200)]
        _te = _te[: min(len(_te), 200)]

    cnr_tr, stim_tr = cnr_all[_tr], stim_all[_tr]
    cnr_va, stim_va = cnr_all[_va], stim_all[_va]
    cnr_te, stim_te = cnr_all[_te], stim_all[_te]

    mo.md(
        f"**Data:** {len(cnr_all)} trajectories (`{data_source}`) · "
        f"train={len(_tr)} / val={len(_va)} / test={len(_te)} · dry_run={DRY_RUN}"
    )
    return cnr_tr, cnr_va, stim_tr, stim_va


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
    cnr_tr,
    cnr_va,
    device,
    experiment_path,
    load_button,
    mo,
    model_config,
    results_base,
    run_experiment,
    stim_tr,
    stim_va,
    train_button,
    training_config,
):
    artifacts = run_experiment(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        model_cls=Model,
        model_config_cls=ModelConfig,
        dataset={"train": (cnr_tr, stim_tr), "val": (cnr_va, stim_va)},
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
    tracker = artifacts.tracker
    model_config_used = artifacts.model_config

    mo.md(
        f"**Run ready** · {type(model).__name__} · "
        f"{sum(p.numel() for p in model.parameters()):,} params"
    )
    return artifacts, history, model_config_used, tracker


@app.cell(hide_code=True)
def _(MODE, experiment_path, load_experiment, mo):
    import json as _json

    if MODE == "load" and experiment_path is not None:
        _bundle_ld = load_experiment(str(experiment_path))
        _stats = _bundle_ld.training_results.get("stats", {})
        _elapsed = _bundle_ld.training_results.get("train_elapsed_s")

        _rows = [
            ("name", _bundle_ld.name),
            ("timestamp", _bundle_ld.timestamp),
            ("model_type", _bundle_ld.model_type),
            ("experiment_path", str(experiment_path)),
        ]
        if _elapsed is not None:
            _rows.append((
                "train_elapsed_s",
                f"{float(_elapsed):.1f} ({float(_elapsed) / 60:.1f} min)",
            ))
        for _k, _v in _stats.items():
            _rows.append((_k, _v))
        for _k, _v in (_bundle_ld.metrics or {}).items():
            _rows.append((f"metric.{_k}", _v))

        _summary_md = (
            "## Loaded run summary\n\n| field | value |\n|---|---|\n"
            + "\n".join(f"| `{_k}` | {_v} |" for _k, _v in _rows)
        )
        _cfg_md = (
            "## Saved configs\n\n"
            f"**model_config**\n```json\n{_json.dumps(_bundle_ld.model_config, indent=2, default=str)}\n```\n\n"
            f"**training_config**\n```json\n{_json.dumps(_bundle_ld.training_config, indent=2, default=str)}\n```"
        )
        run_summary = mo.vstack([mo.md(_summary_md), mo.md(_cfg_md)])
    else:
        run_summary = mo.md("")

    run_summary
    return


@app.cell
def _(history, pl, qplot):
    # TODO: replace with real viz. Stable name `fig_loss` is referenced in save cell.
    loss_df = pl.DataFrame({
        "epoch": list(range(len(history["train_loss"]))) * 2,
        "loss": history["train_loss"] + history["val_loss"],
        "split": ["train"] * len(history["train_loss"])
                 + ["val"] * len(history["val_loss"]),
    })
    fig_loss = qplot(loss_df, x="epoch", y="loss", color="split", mark="line")
    fig_loss
    return (fig_loss,)


@app.cell
def _(mo, model_config_used):
    # TODO: compute your evaluation metrics here.
    eval_metrics = {
        "variant": model_config_used.variant,
        "data_source": model_config_used.data_source,
    }
    mo.md(f"**Metrics:** `{eval_metrics}`")
    return (eval_metrics,)


@app.cell(hide_code=True)
def _(IS_HEADLESS, mo, tracker):
    if (not IS_HEADLESS) and tracker is not None:
        save_all_button = mo.ui.run_button(
            label="Save experiment (model + figures + stats)",
        )
    else:
        save_all_button = None

    save_all_button if save_all_button is not None else mo.md("")
    return (save_all_button,)


@app.cell
def _(
    IS_HEADLESS,
    artifacts,
    cnr_tr,
    cnr_va,
    eval_metrics,
    fig_loss,
    hostname,
    is_cluster,
    mo,
    save_all_button,
    save_bundle,
):
    save_bundle(
        mo=mo,
        is_headless=IS_HEADLESS,
        artifacts=artifacts,
        figures={"loss_curves": fig_loss},
        metrics=eval_metrics,
        n_train=len(cnr_tr),
        n_val=len(cnr_va),
        save_button=save_all_button,
        hostname=hostname,
        is_cluster=is_cluster,
    )
    return


if __name__ == "__main__":
    app.run()
