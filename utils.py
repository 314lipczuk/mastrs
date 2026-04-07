import getpass
import types
from pathlib import Path

import torch

# --- Path constants ---

CLUSTER_RESULTS_PATH = "/mnt/imaging.data/ppilip/results/models"
KINGSTON_RESULTS_PATH = "/Volumes/imaging.data/ppilip/results/models"


# --- Execution context ---

def running_on_cluster() -> bool:
    """True when not running as local user 'polya'."""
    return not getpass.getuser().startswith("polya")


def get_username() -> str:
    return getpass.getuser()


def results_write_path() -> str:
    """Results dir for saving experiments. Cluster -> NFS path, local -> Kingston mount."""
    if running_on_cluster():
        return CLUSTER_RESULTS_PATH
    return KINGSTON_RESULTS_PATH


def results_read_sources(project_root: str | Path | None = None) -> dict[str, str]:
    """Dict for UI dropdowns: local + Kingston mount paths."""
    local = str(Path(project_root) / "results") if project_root else str(Path.cwd() / "results")
    return {"Local": local, "Kingston": KINGSTON_RESULTS_PATH}


def parse_bool(value, default: bool = True) -> bool:
    """Parse bool from mo.cli_args() which may return str or bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


# --- Device ---

def get_device():
    dev = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('Device:', dev)
    return dev


# --- Marimo notebook widgets ---

def experiment_mode_widget(
    mo,
    experiment_name_default: str = "experiment",
    project_root: str | Path | None = None,
    experiment_labels: list[str] | None = None,
):
    """Build a train/load mode selector with source picker.

    Use in two cells — this function creates the top-level controls (cell 1),
    then ``experiment_mode_state`` reads their values and builds the experiment
    picker + state (cell 2).

        # Cell 1
        mode_widget, mode_ctx = experiment_mode_widget(mo, "my_exp", Path(__file__).resolve().parent.parent)
        mode_widget

        # Cell 2
        load_widget, mode = experiment_mode_state(mo, mode_ctx)
        load_widget

        # Cell 3+
        mode.is_train, mode.selected_experiments, ...

    Args:
        experiment_labels: labels for experiment dropdowns in load mode.
            Default ["Experiment"]. Pass e.g. ["AR experiment", "Baseline experiment"]
            for multiple slots.

    Default mode: "Train" in CLI (run mode), "Load" in editor.
    """
    if experiment_labels is None:
        experiment_labels = ["Experiment"]

    _is_cli = mo.app_meta().mode == "run"
    _default_mode = "Train new model" if _is_cli else "Load from disk"

    mode_selector = mo.ui.radio(
        options=["Train new model", "Load from disk"],
        value=_default_mode,
        label="Mode",
    )

    _sources = results_read_sources(project_root)
    _source_keys = list(_sources.keys())
    _default_source = "Kingston" if "Kingston" in _source_keys else _source_keys[0]
    source_selector = mo.ui.dropdown(
        options=_source_keys,
        value=_default_source,
        label="Results source",
    )

    widget = mo.hstack([mode_selector, source_selector], gap=2)

    _ctx = types.SimpleNamespace(
        mode_selector=mode_selector,
        source_selector=source_selector,
        experiment_name=experiment_name_default,
        experiment_labels=tuple(experiment_labels),
        sources=_sources,
    )

    return widget, _ctx


def experiment_mode_state(mo, ctx):
    """Read mode controls and build experiment picker + state.

    Call in the cell after ``experiment_mode_widget``. Returns
    ``(load_widget, state)`` — display ``load_widget``, read ``state`` downstream.

    State fields:
        is_train, is_load, experiment_name,
        load_button_clicked, selected_experiments, selected_experiment_paths,
        selected_experiment, selected_experiment_path (single-slot shortcut),
        results_source, results_path
    """
    _is_load = ctx.mode_selector.value == "Load from disk"
    _source_key = ctx.source_selector.value
    _sources = ctx.sources
    _results_path = Path(_sources[_source_key])

    # Scan for loadable experiments (sorted newest-first)
    # Detects: finished (bundle.pt), checkpointed, manifested, grouped, or started-only
    _experiment_dirs = []
    if _is_load and _results_path.is_dir():
        for subdir in _results_path.iterdir():
            if not subdir.is_dir():
                continue
            has_final = (subdir / "bundle.pt").exists()
            has_checkpoint = (subdir / "checkpoints" / "bundle.pt").exists()
            has_manifest = (subdir / "experiment.json").exists()
            has_started = (subdir / "started.txt").exists()
            has_sub_bundles = any(
                sub.glob("bundle.pt") for sub in subdir.iterdir() if sub.is_dir()
            )
            if has_final or has_checkpoint or has_manifest or has_sub_bundles or has_started:
                _experiment_dirs.append(subdir)
        _experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        _experiment_dirs = [d.name for d in _experiment_dirs]

    if _is_load and not _experiment_dirs:
        mo.stop(True, mo.md(f"No experiments found in `{_results_path}`."))

    # Build one dropdown per label, stagger defaults across available dirs
    _dropdowns = {}
    for i, label in enumerate(ctx.experiment_labels):
        default_idx = min(i, len(_experiment_dirs) - 1) if _experiment_dirs else None
        _dropdowns[label] = mo.ui.dropdown(
            options=_experiment_dirs or ["(none)"],
            value=_experiment_dirs[default_idx] if default_idx is not None else "(none)",
            label=label,
        )

    load_button = mo.ui.run_button(label="Load experiments")

    load_widget = (
        mo.hstack([*_dropdowns.values(), load_button], justify="start", gap=1)
        if _is_load
        else mo.md("")
    )

    class _ModeState:
        def __init__(self):
            self.is_train = not _is_load
            self.is_load = _is_load
            self.experiment_name = ctx.experiment_name
            self._load_button = load_button
            self.experiment_labels = ctx.experiment_labels
            self._dropdowns = _dropdowns
            self.results_source = _source_key
            self.results_path = _results_path

        @property
        def load_button_clicked(self):
            return self._load_button.value if self.is_load else False

        @property
        def selected_experiments(self):
            return {
                label: (dd.value if self.is_load and dd.value != "(none)" else None)
                for label, dd in self._dropdowns.items()
            }

        @property
        def selected_experiment_paths(self):
            return {
                label: (self.results_path / v if v else None)
                for label, v in self.selected_experiments.items()
            }

        @property
        def selected_experiment(self):
            return self.selected_experiments[self.experiment_labels[0]]

        @property
        def selected_experiment_path(self):
            return self.selected_experiment_paths[self.experiment_labels[0]]

    return load_widget, _ModeState()
