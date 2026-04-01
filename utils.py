import getpass
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
