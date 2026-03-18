import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import socket
    from pathlib import Path

    return Path, mo, socket


@app.cell
def _(Path, mo, socket):
    hostname = socket.gethostname()
    is_cluster = not hostname.startswith("polya")
    env_label = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"

    results_sources = {
        "Local": str(Path.cwd() / "results"),
        "Cluster (mounted)": str(Path.home() / "mnt" / "cluster_results"),
    }

    source_selector = mo.ui.dropdown(
        options=list(results_sources.keys()),
        value="Local",
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
    import torch
    import matplotlib.image as mpimg
    from experiment import ExperimentBundle

    mo.stop(not load_button.value, mo.md("Select an experiment and click **Load experiment**."))

    _exp_dir = results_path / experiment_dropdown.value
    _pt_files = sorted(_exp_dir.glob("*.pt"))
    _data = torch.load(_pt_files[0], map_location="cpu", weights_only=False)

    _figures = {}
    _fig_dir = _exp_dir / "figures"
    if _fig_dir.is_dir():
        for _png in sorted(_fig_dir.glob("*.png")):
            _figures[_png.stem] = mpimg.imread(str(_png))

    bundle = ExperimentBundle(
        name=_data.get("name", _exp_dir.name),
        timestamp=_data.get("timestamp", "unknown"),
        model_type=_data.get("model_type", ""),
        model_config=_data.get("model_config", {}),
        model_state_dict=_data.get("model_state_dict"),
        training_config=_data.get("training_config", {}),
        training_results=_data.get("training_results", {}),
        metrics=_data.get("metrics", {}),
        figures=_figures,
        normalization=_data.get("normalization"),
    )
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
    return (plt,)


@app.cell
def _(bundle, mo, plt):
    fig_items = {}
    for fig_name, fig_data in bundle.figures.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig_data)
        ax.set_axis_off()
        ax.set_title(fig_name.replace("_", " ").title())
        fig.tight_layout()
        fig_items[fig_name] = fig

    mo.vstack([
        mo.md("## Figures"),
        mo.tabs(fig_items),
    ])
    return


if __name__ == "__main__":
    app.run()
