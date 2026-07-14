import getpass
import os
from pathlib import Path
# --- Path constants ---

CLUSTER_RESULTS_PATH = "/mnt/imaging.data/ppilip/results/models"
KINGSTON_RESULTS_PATH = "/Volumes/imaging.data/ppilip/results/models"

# Repo root, derived from this file: optoerk/core/utils.py -> optoerk/core -> optoerk -> root.
# Valid because the project is installed editable (`uv sync`), so sources stay in place.
REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Data files ---

def materials_dir() -> Path:
    """Directory holding datasets, light patterns, and reference material.

    Set ``OPTOERK_MATERIALS`` to point at data living outside the repo.
    """
    override = os.environ.get("OPTOERK_MATERIALS")
    return Path(override) if override else REPO_ROOT / "materials"


def materials_path(name: str | Path) -> Path:
    """Absolute path to a file under :func:`materials_dir`."""
    return materials_dir() / name


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
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('Device:', dev)
    return dev


# --- Experiment directory scanning ---

def scan_experiment_dirs(results_path: str | Path) -> list[str]:
    """Return **loadable** experiment paths under results_path, newest first.

    A path is loadable if it contains ``bundle.pt`` or
    ``checkpoints/bundle.pt``. For grouped parents (ensembles etc.) whose
    bundles live one level deeper, the nested children are surfaced as
    ``"parent/child"`` paths. Parents with no usable bundle are skipped.
    """
    results_path = Path(results_path)
    if not results_path.is_dir():
        return []

    def _is_loadable(p: Path) -> bool:
        return (p / "bundle.pt").exists() or (p / "checkpoints" / "bundle.pt").exists()

    entries: list[tuple[Path, float]] = []
    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue
        if _is_loadable(subdir):
            entries.append((subdir, subdir.stat().st_mtime))
            continue
        for nested in subdir.iterdir():
            if nested.is_dir() and _is_loadable(nested):
                entries.append((nested, nested.stat().st_mtime))

    entries.sort(key=lambda t: t[1], reverse=True)
    return [str(p.relative_to(results_path)) for p in (e[0] for e in entries)]
