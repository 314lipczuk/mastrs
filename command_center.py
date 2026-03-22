import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import socket
    from pathlib import Path
    import os
    import getpass

    return Path, getpass, mo


@app.cell
def _(Path, getpass, mo):
    hostname = getpass.getuser()

    is_cluster = not hostname.startswith("polya")
    env_label = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"

    results_sources = {
        "Local": str(Path.cwd() / "results"),
        "Kingston": str('/Volumes/imaging.data/ppilip/results/models')
    }

    source_selector = mo.ui.dropdown(
        options=list(results_sources.keys()),
        value="Kingston",
        label="Results source",
    )

    mo.vstack([
        mo.md(f"# Command Center\n\nEnvironment: {env_label}"),
        source_selector,
    ])
    return results_sources, source_selector


@app.cell
def _(Path, mo, results_sources, source_selector):
    results_path = Path(results_sources[source_selector.value])

    experiment_dirs = []
    if results_path.is_dir():
        for subdir in sorted(results_path.iterdir()):
            if not subdir.is_dir():
                continue
            has_pt = any(subdir.glob("*.pt"))
            has_figures = (subdir / "figures").is_dir()
            if has_pt and has_figures:
                experiment_dirs.append(subdir.name)

    if not experiment_dirs:
        mo.stop(True, mo.md(
            f"No experiments found in `{results_path}`."
        ))

    experiment_dropdown = mo.ui.dropdown(
        options=experiment_dirs,
        value=experiment_dirs[0],
        label="Experiment",
    )
    load_button = mo.ui.run_button(label="Load experiment")
    mo.hstack([experiment_dropdown, load_button], justify="start", gap=1)
    return experiment_dropdown, load_button, results_path


@app.cell
def _(experiment_dropdown, load_button, mo, results_path):
    from experiment import ExperimentBundle

    mo.stop(not load_button.value, mo.md("Select an experiment and click **Load experiment**."))

    _exp_dir = results_path / experiment_dropdown.value
    bundle = ExperimentBundle.load(str(_exp_dir))
    return (bundle,)


@app.cell
def _(bundle, mo):
    import matplotlib.pyplot as plt
    import numpy as np

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

    info_md = mo.md(f"""
    ## {bundle.name}

    **Timestamp:** {bundle.timestamp} | **Model:** `{bundle.model_type}`

    ### Model config
    | Key | Value |
    |-----|-------|
    {"".join(f"| `{k}` | `{v}` |{chr(10)}" for k, v in bundle.model_config.items())}

    ### Training config
    | Key | Value |
    |-----|-------|
    {config_rows}

    ### Metrics
    | Metric | Value |
    |--------|-------|
    {metrics_rows}
    """)

    parts = [info_md]
    if bundle.warnings:
        parts.append(mo.md(
            "### Warnings\n" + "\n".join(f"- {w}" for w in bundle.warnings)
        ))
    mo.vstack(parts)
    return


@app.cell
def show_experiment_figures(experiment_dropdown, mo, results_path):
    import base64

    _fig_dir = results_path / experiment_dropdown.value / "figures"
    _pngs = sorted(_fig_dir.glob("*.png")) if _fig_dir.is_dir() else []

    if not _pngs:
        mo.stop(True, mo.md("No figures found."))

    _cards = []
    for _png in _pngs:
        _b64 = base64.b64encode(_png.read_bytes()).decode()
        _label = _png.stem.replace("_", " ").title()
        _cards.append(
            mo.md(f"""
    **{_label}**

    <img src="data:image/png;base64,{_b64}" style="width:100%; border-radius:6px;" />
    """)
        )

    _cols = min(len(_cards), 2)
    _rows = [
        mo.hstack(_cards[i : i + _cols], widths="equal", gap=1)
        for i in range(0, len(_cards), _cols)
    ]

    mo.vstack([mo.md("## Figures"), *_rows])
    return


@app.cell
def _(bundle, mo):
    from interactive_viz import get_views

    views = get_views(bundle)
    mo.stop(not views, mo.md("_No interactive visualizations available for this model._"))
    mo.vstack([mo.md("## Interactive Visualizations"), mo.ui.tabs(views)])
    return


if __name__ == "__main__":
    app.run()
