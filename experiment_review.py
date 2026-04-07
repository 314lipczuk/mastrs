import marimo

__generated_with = "0.21.1"
app = marimo.App(width="columns", app_title="Experiment explorer")


@app.cell(column=0)
def _():
    import marimo as mo
    from pathlib import Path
    import base64

    return Path, base64, mo


@app.cell(hide_code=True)
def _(Path, mo):
    RESULTS_PATH = Path("/Volumes/imaging.data/ppilip/results/models")
    LOGS_PATH = Path("/Volumes/imaging.data/ppilip")

    experiment_dirs = []
    if RESULTS_PATH.is_dir():
        for subdir in sorted(RESULTS_PATH.iterdir(), reverse=True):
            if not subdir.is_dir():
                continue
            has_final = (subdir / "bundle.pt").exists()
            has_checkpoint = (subdir / "checkpoints" / "bundle.pt").exists()
            if has_final or has_checkpoint:
                experiment_dirs.append(subdir.name)

    mo.stop(not experiment_dirs, mo.md(f"No experiments found in `{RESULTS_PATH}`."))

    experiment_dropdown = mo.ui.dropdown(
        options=experiment_dirs,
        value=experiment_dirs[0],
        label="Experiment",
    )
    load_button = mo.ui.run_button(label="Load")
    mo.hstack([experiment_dropdown, load_button], justify="start", gap=1)
    return LOGS_PATH, RESULTS_PATH, experiment_dropdown, load_button


@app.cell(hide_code=True)
def _(RESULTS_PATH, experiment_dropdown, load_button, mo):
    from experiment import ExperimentBundle

    mo.stop(not load_button.value, mo.md("Select an experiment and click **Load**."))

    _exp_dir = RESULTS_PATH / experiment_dropdown.value
    _has_final = (_exp_dir / "bundle.pt").exists()
    _has_checkpoint = (_exp_dir / "checkpoints" / "bundle.pt").exists()

    is_incomplete = not _has_final and _has_checkpoint

    if _has_final:
        bundle = ExperimentBundle.load(str(_exp_dir))
    else:
        bundle = ExperimentBundle.load(str(_exp_dir / "checkpoints"))
    return bundle, is_incomplete


@app.cell(hide_code=True)
def _(bundle, mo):
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

    _started_file = bundle.save_dir and (__import__("pathlib").Path(bundle.save_dir) / "started.txt")
    _started_info = ""
    if _started_file and _started_file.exists():
        _started_info = _started_file.read_text()

    _header = mo.md(f"""
    ## {bundle.name}

    **Timestamp:** {bundle.timestamp} | **Model:** `{bundle.model_type}`

    {f"```{chr(10)}{_started_info}```" if _started_info else ""}
    """)

    _model_cfg_md = mo.md(f"""
    ### Model config
    | Key | Value |
    |-----|-------|
    {"".join(f"| `{k}` | `{v}` |{chr(10)}" for k, v in bundle.model_config.items())}
    """)

    _training_cfg_md = mo.md(f"""
    ### Training config
    | Key | Value |
    |-----|-------|
    {config_rows}
    """)

    _metrics_md = mo.md(f"""
    ### Metrics
    | Metric | Value |
    |--------|-------|
    {metrics_rows}
    """)

    parts = [_header, mo.hstack([_model_cfg_md, _training_cfg_md, _metrics_md], widths='equal', align="start")]
    mo.vstack(parts)
    return


@app.cell(hide_code=True)
def _(bundle, is_incomplete, mo):
    _rows = []
    if is_incomplete:
        _rows.append(mo.callout(
            mo.md("This experiment **did not finish**. Loaded from the latest checkpoint."),
            kind="warn",
        ))
    if bundle.warnings:
        _rows.append(mo.callout(
            mo.md("\n".join(f"- {w}" for w in bundle.warnings)),
            kind="warn",
        ))
    mo.vstack(_rows)
    return


@app.cell(hide_code=True)
def _(RESULTS_PATH, base64, experiment_dropdown, mo):
    _fig_dir = RESULTS_PATH / experiment_dropdown.value / "figures"

    _pngs = sorted(_fig_dir.glob("*.png")) if _fig_dir.is_dir() else []

    mo.stop(not _pngs, mo.md("No figures found."))

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
def _(bundle, mo):
    from interactive_viz import get_views

    views = get_views(bundle)
    mo.stop(not views, mo.md("_No interactive visualizations available for this model._"))
    mo.vstack([mo.md("## Interactive Visualizations"), mo.ui.tabs(views)])
    return


@app.cell(column=1, hide_code=True)
def _(LOGS_PATH, experiment_dropdown, mo):
    import re

    _exp_name = experiment_dropdown.value
    _match = re.search(r"_j(\d+)$", _exp_name)

    log_file = None
    _log_content = None
    if _match:
        _job_id = _match.group(1)
        _candidates = sorted(LOGS_PATH.glob(f"*_{_job_id}.log"))
        if _candidates:
            log_file = _candidates[0]
            _log_content = log_file.read_text()

    mo.stop(
        _log_content is None,
        mo.md("_No SLURM log found for this experiment (local run or log not on Kingston)._"),
    )

    mo.vstack([
        mo.md("## Training log"),
        mo.plain_text(_log_content),
    ])
    return (log_file,)


@app.cell(hide_code=True)
def _(log_file, mo):
    _html_file = log_file.with_suffix(".html") if log_file else None
    _html_exists = _html_file is not None and _html_file.exists()

    open_html_button = mo.ui.run_button(
        label="Open notebook HTML",
        disabled=not _html_exists,
    )
    _status = f"`{_html_file.name}`" if _html_exists else "_No HTML export found_"
    mo.hstack([open_html_button, mo.md(_status)], justify="start", gap=1, align="center")
    return (open_html_button,)


@app.cell(hide_code=True)
def _(log_file, mo, open_html_button):
    import subprocess

    mo.stop(not open_html_button.value)

    _html_file = log_file.with_suffix(".html")
    subprocess.Popen(["open", str(_html_file)])
    return


if __name__ == "__main__":
    app.run()
