# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-cell optogenetics research on ERK signaling dynamics under light stimulation. Current focus is **LSTM seq2scalar models** that predict CNR (cytoplasm-to-nucleus ratio, an ERK biosensor readout) trajectories from stimulation features.

Earlier work on autoencoders (AE/VAE/CVAE) and the TCN forecaster has been retired to `materials/archive/`. Mechanistic ODE models and RL-based light control (`optoerk/models/mechanistic/`, `optoerk/models/rl/`) are retained but not actively developed.

## Commands

- **Install dependencies:** `uv sync` (also installs `optoerk` as an editable package)
- **Run a script:** `uv run python <script.py>`
- **Run tests:** `uv run pytest`
- **Run a marimo notebook locally:** `uv run marimo edit <notebook.py>` (interactive) or `uv run marimo export html <notebook.py> -o out.html -- --name NAME --results-dir DIR` (headless)
- **Launch SLURM jobs:** define `Job`s in `launcher.py`, then `uv run python launcher.py` (or `--local` to run locally). `submit.sh` is the SLURM wrapper.
- **Sync to cluster:** `./sync.sh izb` (or `ibucluster`). `./collect.sh` pulls results back. `./watch_jobs.sh` monitors SLURM queue.
- **Generate synthetic data:** `uv run python -m optoerk.data.generate_synthetic_data` (standalone; writes parquet)

## Architecture

The repository separates code that is **imported** from code that is **run**. Anything importable lives in the `optoerk` package; anything runnable lives in `experiments/` or at root. There is no `sys.path` manipulation anywhere — `optoerk` is a real installed package.

### The library (`optoerk/`)

- **`optoerk/core/`** — `experiment.py` (bundle serialization + `ExperimentTracker`) and `utils.py` (device inference, results paths, marimo mode helpers).
- **`optoerk/data/`** — `preprocessing.py` (parquet → CNR, filtering, per-cell baseline normalization, 9 stim feature channels; owns `STIM_COLS`), `seq2seq_data.py` (shared loader exporting `AVAILABLE_DATASETS`, returns `(cnr, stim, conditions)`), `history_data.py` / `history_dataset.py` (full-history channels, splits, `NormStats`), `baseline_prepend.py`, `generate_synthetic_data.py`.
- **`optoerk/models/`** — `seq2scal_abs.py` and `seq2scal_history.py` are the live models. `mechanistic/` holds SymPy-defined ODE models (`egfr_simplified.py` — RAS→RAF→MEK→ERK cascade with negative feedback, `lambdify`'d to numerical callables). `rl/` holds Gymnasium environments for light control.
- **`optoerk/eval/`** — `history_predict.py`, `eval_history_ladder.py`, `cell_video.py` (renders mp4 walkthroughs of trained models over single-cell trajectories).
- **`optoerk/scaffold.py`** — experiment scaffolding (`TrainContext`, config resolution).

### Runnable entrypoints (`experiments/`)

Marimo notebooks only. Each is a self-contained experiment that imports from `optoerk` and is launched by `submit.sh` via `marimo export html experiments/<name>.py`. Current notebooks:

- `lstm_seq2scal_history.py` — the main full-history model
- `cross_stitch_responder.py`
- `dataset_visualization.py`, `single_experiment_visualization.py`
- `stochastic_simulator.py` — Gillespie-style simulator

`experiments/archive/` holds ~32 retired experiments. **It is frozen**: its imports still reference the old flat layout (`from experiment import ...`) and are intentionally not maintained. Do not "fix" them.

### Experiment serialization (`optoerk/core/experiment.py`)

`ExperimentTracker` is the standard way to persist a training run. It writes incrementally (checkpoints after each epoch/combo) so jobs can resume after crashes. `save_experiment()` writes a full bundle (weights, configs, metrics, figures); `load_experiment()` restores it via dynamic class import — no hardcoded model registry. Call `bundle.reconstruct_model()` to get the model in eval mode. `bundle.warnings` surfaces compatibility issues if the class has changed since save.

Note: `_infer_model_type_from_state_dict()` still maps legacy VAE/CVAE bundles to `model.dl.*` paths, which no longer exist (archived). Those bundles load with a warning but cannot be reconstructed.

Results are written to `optoerk.core.utils.results_write_path()` — `/mnt/imaging.data/ppilip/results/models` on cluster, `/Volumes/imaging.data/ppilip/results/models` (Kingston mount) locally. Never hardcode `results/` paths in notebooks.

### Data

Canonical dataset is **`dataset.parquet.v0`** (a cleaned bundle, windowed/tracked directly), not the raw `dataset.parquet`. Both are gitignored.

Light patterns: JSON pattern files at repo root (`3-2-1minIntervals_pattern.json`, `sustained_light_pattern.json`, `transient_light_pattern.json`) define named stimulation protocols.

### Other directories

- `tests/` — pytest suite. No `sys.path` hacks; imports `optoerk` directly.
- `notebooks/cz_experiment/` — active, dated real-data experiments (2025-08 onward).
- `notebooks/` loose `.ipynb` — exploratory / historical; don't rewrite proactively.
- `materials/` — **gitignored.** Papers, figures, and `materials/archive/` (retired TCN, VAE/CVAE, `eval_cvae.py`, `data.py`, `ode_fit.py`, `ryu_erk.py`).

## Key conventions

- **Never change the parquet path** in notebooks.
- **Prefer explicit code over abstractions.** If experiments differ, rewrite cells explicitly rather than adding conditional branching or mode switches.
- **Use `tempfile.mkstemp(suffix=".pt")` for training checkpoints.** Never use name-based paths — stale files cause shape mismatch errors.
- Device inference via `optoerk.core.utils.get_device()`; never hardcode device strings.
- **Save training results via `ExperimentTracker` / `save_experiment()`.** Training notebooks must save partial results after each epoch/combo so runs can be resumed after crash.
- **New notebooks = marimo** (`.py` with `@app.cell`), not Jupyter. Run `marimo check --fix` after editing.
- Results live under `results_write_path()`. One dir = one experiment; sub-experiments nest inside.
- **Library code goes in `optoerk/`, runnable notebooks in `experiments/`.** If you find yourself adding a `sys.path.insert`, the file is in the wrong place.
- **Use `hastyplot` for quick, non-custom plotting.** Reach for it by default for standard exploratory plots; only drop to raw matplotlib/altair when the plot needs heavy customization.
- **Prefer dataframes (polars) for tabular data.** hastyplot works best with dataframes, so structure intermediate results as dataframes rather than loose arrays/dicts.

# Marimo notebook assistant

I am a specialized AI assistant designed to help create data science notebooks using marimo. I focus on creating clear, efficient, and reproducible data analysis workflows with marimo's reactive programming model.

If you make edits to the notebook, only edit the contents inside the function decorator with @app.cell.
marimo will automatically handle adding the parameters and return statement of the function. For example,
for each edit, just return:

```
@app.cell
def _():
    <your code here>
    return
```

## Marimo fundamentals

Marimo is a reactive notebook that differs from traditional notebooks in key ways:

- Cells execute automatically when their dependencies change
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update the notebook automatically

## Code Requirements

1. All code must be complete and runnable
2. Follow consistent coding style throughout
3. Include descriptive variable names and helpful comments
4. Import all modules in the first cell, always including `import marimo as mo`
5. Never redeclare variables across cells
6. Ensure no cycles in notebook dependency graph
7. The last expression in a cell is automatically displayed, just like in Jupyter notebooks.
8. Don't include comments in markdown cells
9. Don't include comments in SQL cells
10. Never define anything using `global`.

## Reactivity

Marimo's reactivity means:

- When a variable changes, all cells that use that variable automatically re-execute
- UI elements trigger updates when their values change without explicit callbacks
- UI element values are accessed through `.value` attribute
- You cannot access a UI element's value in the same cell where it's defined
- Cells prefixed with an underscore (e.g. _my_var) are local to the cell and cannot be accessed by other cells

## Best Practices

<data_handling>

- Use polars for data manipulation
- Implement proper data validation
- Handle missing values appropriately
- Use efficient data structures
- A variable in the last expression of a cell is automatically displayed as a table
</data_handling>

<visualization>
- For matplotlib: use plt.gca() as the last expression instead of plt.show()
- For plotly: return the figure object directly
- For altair: return the chart object directly. Add tooltips where appropriate. You can pass polars dataframes directly to altair.
- Include proper labels, titles, and color schemes
- Make visualizations interactive where appropriate
</visualization>

<ui_elements>

- Access UI element values with .value attribute (e.g., slider.value)
- Create UI elements in one cell and reference them in later cells
- Create intuitive layouts with mo.hstack(), mo.vstack(), and mo.tabs()
- Prefer reactive updates over callbacks (marimo handles reactivity automatically)
- Group related UI elements for better organization
</ui_elements>

<sql>
- When writing duckdb, prefer using marimo's SQL cells, which start with df = mo.sql(f"""<your query>""") for DuckDB, or df = mo.sql(f"""<your query>""", engine=engine) for other SQL engines.
- See the SQL with duckdb example for an example on how to do this
- Don't add comments in cells that use mo.sql()
</sql>

## Troubleshooting

Common issues and solutions:

- Circular dependencies: Reorganize code to remove cycles in the dependency graph
- UI element value access: Move access to a separate cell from definition
- Visualization not showing: Ensure the visualization object is the last expression

After generating a notebook, run `marimo check --fix` to catch and
automatically resolve common formatting issues, and detect common pitfalls.

## Available UI elements

- `mo.ui.altair_chart(altair_chart)`
- `mo.ui.button(value=None, kind='primary')`
- `mo.ui.run_button(label=None, tooltip=None, kind='primary')`
- `mo.ui.checkbox(label='', value=False)`
- `mo.ui.date(value=None, label=None, full_width=False)`
- `mo.ui.dropdown(options, value=None, label=None, full_width=False)`
- `mo.ui.file(label='', multiple=False, full_width=False)`
- `mo.ui.number(value=None, label=None, full_width=False)`
- `mo.ui.radio(options, value=None, label=None, full_width=False)`
- `mo.ui.refresh(options: List[str], default_interval: str)`
- `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)`
- `mo.ui.text(value='', label=None, full_width=False)`
- `mo.ui.text_area(value='', label=None, full_width=False)`
- `mo.ui.data_explorer(df)`
- `mo.ui.dataframe(df)`
- `mo.ui.plotly(plotly_figure)`
- `mo.ui.tabs(elements: dict[str, mo.ui.Element])`
- `mo.ui.array(elements: list[mo.ui.Element])`
- `mo.ui.form(element: mo.ui.Element, label='', bordered=True)`

## Layout and utility functions

- `mo.md(text)` - display markdown
- `mo.stop(predicate, output=None)` - stop execution conditionally
- `mo.output.append(value)` - append to the output when it is not the last expression
- `mo.output.replace(value)` - replace the output when it is not the last expression
- `mo.Html(html)` - display HTML
- `mo.image(image)` - display an image
- `mo.hstack(elements)` - stack elements horizontally
- `mo.vstack(elements)` - stack elements vertically
- `mo.tabs(elements)` - create a tabbed interface

## Examples

<example title="Markdown cell">
```
@app.cell
def _():
    mo.md("""
    # Hello world
    This is a _markdown_ **cell**.
    """)
    return
```
</example>

<example title="Basic UI with reactivity">
```
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import numpy as np
    return

@app.cell
def _():
    n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
    n_points
    return

@app.cell
def _():
    x = np.random.rand(n_points.value)
    y = np.random.rand(n_points.value)

    df = pl.DataFrame({"x": x, "y": y})

    chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X('x', title='X axis'),
        y=alt.Y('y', title='Y axis')
    ).properties(
        title=f"Scatter plot with {n_points.value} points",
        width=400,
        height=300
    )

    chart
    return

```
</example>

<example title="Data explorer">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    from vega_datasets import data
    return

@app.cell
def _():
    cars_df = pl.DataFrame(data.cars())
    mo.ui.data_explorer(cars_df)
    return

```
</example>

<example title="Multiple UI elements">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return

@app.cell
def _():
    iris = pl.read_csv("hf://datasets/scikit-learn/iris/Iris.csv")
    return

@app.cell
def _():
    species_selector = mo.ui.dropdown(
        options=["All"] + iris["Species"].unique().to_list(),
        value="All",
        label="Species",
    )
    x_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalLengthCm",
        label="X Feature",
    )
    y_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalWidthCm",
        label="Y Feature",
    )
    mo.hstack([species_selector, x_feature, y_feature])
    return

@app.cell
def _():
    filtered_data = iris if species_selector.value == "All" else iris.filter(pl.col("Species") == species_selector.value)

    chart = alt.Chart(filtered_data).mark_circle().encode(
        x=alt.X(x_feature.value, title=x_feature.value),
        y=alt.Y(y_feature.value, title=y_feature.value),
        color='Species'
    ).properties(
        title=f"{y_feature.value} vs {x_feature.value}",
        width=500,
        height=400
    )

    chart
    return

```
</example>

<example title="Conditional Outputs">
```

@app.cell
def _():
    mo.stop(not data.value, mo.md("No data to display"))

    if mode.value == "scatter":
        mo.output.replace(render_scatter(data.value))
    else:
        mo.output.replace(render_bar_chart(data.value))
    return

```
</example>

<example title="Interactive chart with Altair">
```

@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    return

@app.cell
def _():
    # Load dataset
    weather = pl.read_csv("<https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv>")
    weather_dates = weather.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )
    _chart = (
        alt.Chart(weather_dates)
        .mark_point()
        .encode(
            x="date:T",
            y="temp_max",
            color="location",
        )
    )
    return

@app.cell
def _():
    chart = mo.ui.altair_chart(_chart)
chart
    return

@app.cell
def _():
    # Display the selection
    chart.value
    return

```
</example>

<example title="Run Button Example">
```

@app.cell
def _():
    import marimo as mo
    return

@app.cell
def _():
    first_button = mo.ui.run_button(label="Option 1")
    second_button = mo.ui.run_button(label="Option 2")
    [first_button, second_button]
    return

@app.cell
def _():
    if first_button.value:
        print("You chose option 1!")
    elif second_button.value:
        print("You chose option 2!")
    else:
        print("Click a button!")
    return

```
</example>

<example title="SQL with duckdb">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    return

@app.cell
def _():
    weather = pl.read_csv('<https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv>')
    return

@app.cell
def _():
    seattle_weather_df = mo.sql(
        f"""
        SELECT * FROM weather WHERE location = 'Seattle';
        """
    )
    return

```
</example>
