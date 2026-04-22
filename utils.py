import getpass
from pathlib import Path
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
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('Device:', dev)
    return dev


# --- Experiment directory scanning ---

def scan_experiment_dirs(results_path: str | Path) -> list[str]:
    """Return experiment subdir names under results_path, newest first.

    An entry counts as an experiment if it contains any of: bundle.pt,
    checkpoints/bundle.pt, experiment.json, started.txt, or a nested dir
    with bundle.pt (grouped runs like ensembles).
    """
    results_path = Path(results_path)
    if not results_path.is_dir():
        return []
    dirs = []
    for subdir in results_path.iterdir():
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
            dirs.append(subdir)
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]
