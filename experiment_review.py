import marimo

__generated_with = "0.21.1"
app = marimo.App(width="columns", app_title="Experiment explorer")

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


@app.cell
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


@app.cell(column=1)
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
