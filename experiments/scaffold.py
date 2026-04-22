"""Reusable scaffolding for seq2seq-style experiment notebooks.

Mode is a static CLI flag (``--mode train|load``) — each notebook cell runs
or no-ops based on it, so marimo's DAG is stable across a session. Model
class, dataset loading, and viz cells remain experiment-specific.

Usage pattern:

    # @app.setup (library scope):
    from experiments.scaffold import TrainContext
    class MyModel(nn.Module):
        @staticmethod
        def fit(dataset, ctx: TrainContext) -> tuple["MyModel", dict]: ...

    # Cell: model assembly (single source of `model`, branches on static MODE)
    if MODE == "train":
        if not IS_HEADLESS:
            mo.stop(not train_button.value, mo.md("Click Start training."))
        artifacts = train_model(
            mo=mo, model_cls=MyModel, dataset=my_dataset_dict,
            model_config=model_config, training_config=training_config,
            device=device, experiment_name=EXPERIMENT_NAME,
            results_base=results_base, is_headless=IS_HEADLESS,
        )
    else:  # MODE == "load"
        mo.stop(not load_button.value, mo.md("Pick experiment + click Load."))
        artifacts = load_model(
            experiment_path=results_path / experiment_picker.value,
            model_cls=MyModel, model_config_cls=ModelConfig, device=device,
        )
    model = artifacts.model
"""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from pydantic import BaseModel

from experiment import ExperimentTracker, compute_training_stats, load_experiment


@dataclass
class TrainContext:
    device: torch.device
    model_config: BaseModel
    training_config: BaseModel
    tracker: ExperimentTracker | None = None
    progress_cb: Callable[[int, int, dict], None] | None = None
    print_every: int = 20


def _is_optional(ann) -> tuple[bool, Any]:
    """Return (is_optional, inner_type) for `T | None` / `Optional[T]`."""
    origin = typing.get_origin(ann)
    if origin is typing.Union or str(origin) == "types.UnionType":
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, ann


def ui_from_pydantic(mo, cls: type[BaseModel], *, skip=(), overrides=None) -> dict:
    """Generate {field_name: mo.ui.Element} from a pydantic model's fields.

    - bool → checkbox, int → number(step=1), float → number(step=0.0001), str → text
    - Optional[T] → text (blank = None)
    - `overrides[name]` substitutes a caller-provided widget
    """
    overrides = overrides or {}
    fields: dict[str, Any] = {}
    for name, info in cls.model_fields.items():
        if name in skip:
            continue
        if name in overrides:
            fields[name] = overrides[name]
            continue
        ann = info.annotation
        is_opt, inner = _is_optional(ann)
        default = info.default if info.default is not None else None
        effective = inner if is_opt else ann
        if effective is bool:
            fields[name] = mo.ui.checkbox(label=name, value=bool(default) if default is not None else False)
        elif effective is int and not is_opt:
            fields[name] = mo.ui.number(label=name, value=int(default) if default is not None else 0, step=1)
        elif effective is float and not is_opt:
            fields[name] = mo.ui.number(label=name, value=float(default) if default is not None else 0.0, step=0.0001)
        else:
            fields[name] = mo.ui.text(
                label=name + (" (blank = None)" if is_opt else ""),
                value="" if default is None else str(default),
            )
    return fields


def form_from_configs(
    mo,
    config_classes: dict[str, type[BaseModel]],
    *,
    skip: dict[str, set[str]] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    radio_choices: dict[str, dict[str, typing.Iterable]] | None = None,
    label: str = "Experiment config (pydantic-validated)",
    submit_label: str = "Apply",
):
    """Build an `mo.ui.form` from a {prefix: pydantic_class} map.

    Fields are keyed `{prefix}.{field_name}` in form.value so multiple configs
    can share a single form.

    - `skip[prefix]`: set of fields to omit for that class
    - `overrides[prefix][field]`: substitute widget
    - `radio_choices[prefix][field]`: shortcut → builds a radio with those options
      (use instead of dropdown — dropdown is buggy inside forms)
    """
    skip = skip or {}
    overrides = overrides or {}
    radio_choices = radio_choices or {}
    merged_overrides: dict[str, dict[str, Any]] = {p: dict(overrides.get(p, {})) for p in config_classes}
    for prefix, choices in radio_choices.items():
        for field, opts in choices.items():
            if field in merged_overrides.get(prefix, {}):
                continue
            default = config_classes[prefix].model_fields[field].default
            merged_overrides.setdefault(prefix, {})[field] = mo.ui.radio(
                options=list(opts), value=default, label=field
            )

    flat: dict[str, Any] = {}
    for prefix, cls in config_classes.items():
        ui = ui_from_pydantic(mo, cls, skip=skip.get(prefix, ()), overrides=merged_overrides.get(prefix, {}))
        for k, v in ui.items():
            flat[f"{prefix}.{k}"] = v
    return mo.ui.form(mo.ui.dictionary(flat), label=label, submit_button_label=submit_label)


def _coerce_blanks(vals: dict) -> dict:
    """Treat blank strings and 'None' as None — covers Optional[int] text fields."""
    return {k: (None if isinstance(v, str) and v in ("", "None") else v) for k, v in vals.items()}


def configs_from_form(
    form_value: dict,
    config_classes: dict[str, type[BaseModel]],
    *,
    always: dict[str, dict] | None = None,
) -> dict[str, BaseModel]:
    """Split prefixed form values by prefix, apply `always` overrides, validate."""
    always = always or {}
    out: dict[str, BaseModel] = {}
    for prefix, cls in config_classes.items():
        vals = {k.removeprefix(f"{prefix}."): v for k, v in form_value.items() if k.startswith(f"{prefix}.")}
        vals = _coerce_blanks(vals)
        vals.update(always.get(prefix, {}))
        out[prefix] = cls.model_validate(vals)
    return out


def configs_from_cli(
    cli_args,
    config_classes: dict[str, type[BaseModel]],
    *,
    always: dict[str, dict] | None = None,
) -> dict[str, BaseModel]:
    """Parse `mo.cli_args()` into pydantic configs. pydantic v2 auto-coerces strings.

    Accepts both unprefixed (`--epochs 3`) and prefixed (`--t.epochs 3`) CLI args.
    """
    always = always or {}
    out: dict[str, BaseModel] = {}
    # cli_args supports dict-like access but not __contains__/__iter__ cleanly; normalize
    args_dict = dict(cli_args) if hasattr(cli_args, "keys") else {k: cli_args[k] for k in cli_args}
    for prefix, cls in config_classes.items():
        vals: dict[str, Any] = {}
        for k in cls.model_fields:
            pref_key = f"{prefix}.{k}"
            if pref_key in args_dict:
                vals[k] = args_dict[pref_key]
            elif k in args_dict:
                vals[k] = args_dict[k]
        vals = _coerce_blanks(vals)
        vals.update(always.get(prefix, {}))
        out[prefix] = cls.model_validate(vals)
    return out


@dataclass
class RunArtifacts:
    model: nn.Module
    history: dict
    train_elapsed: float
    tracker: ExperimentTracker | None
    model_config: BaseModel | None
    training_config: BaseModel | None


def train_model(
    *,
    mo,
    model_cls: type,
    dataset: Any,
    model_config: BaseModel,
    training_config: BaseModel,
    device: torch.device,
    experiment_name: str,
    results_base: str,
    is_headless: bool,
    progress_bar: bool = True,
) -> RunArtifacts:
    """Train a model under ``ExperimentTracker``. Works headless or interactive.

    ``model_cls.fit(dataset, TrainContext) -> (model, history_dict)`` is the
    only hard contract. The caller is responsible for upstream gating (the
    train button in interactive mode).
    """
    exp_dir = (
        mo.cli_args().get("results-dir", f"{results_base}/{experiment_name}")
        if is_headless
        else f"{results_base}/{experiment_name}"
    )
    tracker = ExperimentTracker(
        directory=exp_dir,
        name=experiment_name,
        model_config=model_config.model_dump(),
        training_config=training_config.model_dump(),
    )
    tracker.register_start()
    t0 = time.time()

    if progress_bar and not is_headless:
        with mo.status.progress_bar(total=training_config.epochs) as bar:
            def _cb(_ep, _total, m):
                bar.update(increment=1, subtitle=f"val={m['val']:.4f}")
            model, history = model_cls.fit(
                dataset,
                TrainContext(device=device, model_config=model_config,
                             training_config=training_config, tracker=tracker, progress_cb=_cb),
            )
    else:
        model, history = model_cls.fit(
            dataset,
            TrainContext(device=device, model_config=model_config,
                         training_config=training_config, tracker=tracker),
        )
    return RunArtifacts(model, history, time.time() - t0, tracker, model_config, training_config)


def load_model(
    *,
    experiment_path: str | Any,
    model_cls: type,
    model_config_cls: type[BaseModel],
    device: torch.device,
) -> RunArtifacts:
    """Reconstruct a saved model. Returns RunArtifacts with tracker=None.

    ``model_cls`` is unused at runtime — the bundle's own model class FQN is
    used — but passing it documents the contract and helps type-checkers.
    """
    del model_cls  # kept in signature for symmetry with train_model
    bundle = load_experiment(str(experiment_path))
    model = bundle.reconstruct_model().to(device)
    history = bundle.training_results.get(
        "history", {"train_loss": [], "val_loss": [], "tf_ratio": []}
    )
    train_elapsed = bundle.training_results.get("train_elapsed_s", 0.0)
    mc_used = model_config_cls.model_validate(bundle.model_config)
    return RunArtifacts(model, history, train_elapsed, None, mc_used, None)


def save_bundle(
    *,
    mo,
    is_headless: bool,
    artifacts: RunArtifacts,
    figures: dict,
    metrics: dict,
    n_train: int,
    n_val: int,
    save_button=None,
    hostname: str = "",
    is_cluster: bool = False,
):
    """Unified save: headless → auto-save; interactive → button-gated."""
    if artifacts.tracker is None:
        return mo.md("**Load mode** — nothing to save. (Viz rendered from loaded bundle.)")

    stats = compute_training_stats(
        train_elapsed_s=artifacts.train_elapsed,
        history=artifacts.history,
        n_train_samples=n_train,
        n_val_samples=n_val,
        model=artifacts.model,
    )
    payload = dict(
        model=artifacts.model,
        training_results={"history": artifacts.history, "train_elapsed_s": artifacts.train_elapsed, "stats": stats},
        metrics=metrics,
        figures=figures,
    )
    if is_headless:
        bundle = artifacts.tracker.save_final(**payload)
        env = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"
        return mo.md(f"**Saved** on {env}\n\n`{bundle.save_dir}`")

    mo.stop(
        save_button is None or not save_button.value,
        mo.md("Click **Save experiment** to persist."),
    )
    bundle = artifacts.tracker.save_final(**payload)
    return mo.md(f"**Saved** → `{bundle.save_dir}`")
