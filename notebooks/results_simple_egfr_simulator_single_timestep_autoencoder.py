import marimo

__generated_with = "0.21.1"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import socket
    from pathlib import Path
    import os

    from utils import get_username, running_on_cluster, results_read_sources

    return Path, get_username, mo, results_read_sources, running_on_cluster


@app.cell(hide_code=True)
def _(Path, get_username, mo, results_read_sources, running_on_cluster):
    hostname = get_username()

    is_cluster = running_on_cluster()
    env_label = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"

    results_sources = results_read_sources(Path.cwd())

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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(experiment_dropdown, load_button, mo, results_path):
    from experiment import ExperimentBundle

    mo.stop(not load_button.value, mo.md("Select an experiment and click **Load experiment**."))

    _exp_dir = results_path / experiment_dropdown.value
    bundle = ExperimentBundle.load(str(_exp_dir))
    return (bundle,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(bundle, mo):
    from interactive_viz import get_views

    views = get_views(bundle)
    mo.stop(not views, mo.md("_No interactive visualizations available for this model._"))
    mo.vstack([mo.md("## Interactive Visualizations"), mo.ui.tabs(views)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Observations
    ## L2: H_4-4
    $z_0$ looks like ERK itself (shape-wise, diff scaling).
    $z_1$ looks like $-MEK$; also scaled and shifted.

    ## L2 H_8-4
    $z_1$ looks like $-ERK$; also scaled and shifted.
    $z_0$ is not immediately recognizable; But in some trajectories is quite similar to RAS/RAF
    ## General

    The network is learning to figure out representations and paint in, but the represented states are not interesting - they are usually clones of some of the signals in the state variables.
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Mechanistic model

    Simplified EGFR → RAS → RAF → MEK → ERK cascade with negative feedback. Simplifications used in the `equation` field of `eq_desc`:

    - **5-node lumped cascade** — the full EGFR signaling network is collapsed into five active-fraction states: RAS, RAF, MEK, NFB, ERK
    - **Total protein conservation** — each node is normalized so Active + Inactive = 1; activation is proportional to the inactive pool `(1 - X_s)`, removing the need for separate inactive-state ODEs (but restricting any state from reaching values below or equal 0, at every timestep we take $max(1e-3 ,x)$)
    - **Single shared Km** — all Michaelis-Menten deactivation terms share one saturation constant, reducing the parameter count
    - **Mass-action activation, MM deactivation** — forward activation is linear (rate × upstream × inactive pool), while deactivation uses saturating Michaelis-Menten kinetics `X_s / (Km + X_s)`
    - **Explicit NFB intermediate** — ERK-driven negative feedback on RAF goes through a dedicated NFB state variable rather than a direct algebraic term, introducing a time delay in the feedback loop
    - **Light as direct RAS input** — optogenetic stimulation enters only through the RAS activation term (`light × k12 × (1 - RAS_s)`), treating upstream receptor dynamics as instantaneous

    There are many things wrong in this model, but it is simple enough to capture some general trends;
    However, one thing that might be too much to overlook even in the simple model is light being directly wired; Maybe it should get treated like an MM term? Explicit saturation based on some theoretical limits?
    """)
    return


@app.cell(hide_code=True)
def _():
    import model.mechanistic.egfr_simplified as es
    eq_desc = es.model_eqs(es.PARAM_NAMES, es.STATE_NAMES)
    eq_desc
    return


if __name__ == "__main__":
    app.run()
