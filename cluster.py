#!/usr/bin/env python3
"""
cluster.py — Dispatch jupyter notebooks / python scripts to remote machines.

Usage:
    As a module:  from cluster import dispatch, Machine, SlurmCluster, ...
    As a script:  python cluster.py          # launches TUI
                  python cluster.py --help   # shows CLI options
"""

from __future__ import annotations

import abc
import argparse
import dataclasses
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ─── Utilities ───────────────────────────────────────────────────────────────

def _run(cmd: str | list[str], *, check=True, capture=True, **kw) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kw)


def _ssh(host: str, cmd: str, *, check=True, capture=True) -> subprocess.CompletedProcess:
    return _run(["ssh", host, cmd], check=check, capture=capture)


def _scp(src: str, dst: str, *, recursive=False) -> subprocess.CompletedProcess:
    flags = ["-r"] if recursive else []
    return _run(["scp", *flags, src, dst])


def _tar_create(archive: pathlib.Path, paths: list[pathlib.Path], base_dir: pathlib.Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=p.relative_to(base_dir))


# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_CONFIG = ".cluster.json"
MACHINE_CONFIG_DIR = pathlib.Path.home() / ".config" / "cluster" / "machines"


@dataclass
class ProjectConfig:
    """Per-project defaults, read from .cluster.json alongside the notebook."""
    extra_deps: list[str] = field(default_factory=list)  # extra dirs/files to bundle
    default_machine: str = ""
    default_gpu: bool = False
    default_ram_gb: int = 8
    default_time: str = "01:00:00"
    pre_run: list[str] = field(default_factory=list)
    post_run: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, directory: pathlib.Path) -> "ProjectConfig":
        cfg_path = directory / PROJECT_CONFIG
        if cfg_path.exists():
            with open(cfg_path) as f:
                return cls(**{k: v for k, v in json.load(f).items() if k in {f.name for f in dataclasses.fields(cls)}})
        return cls()


# ─── Job specification ───────────────────────────────────────────────────────

@dataclass
class Job:
    """Everything needed to describe a dispatch."""
    target_file: pathlib.Path            # notebook or script
    machine_name: str                    # ssh config host name
    deps: list[pathlib.Path] = field(default_factory=list)  # extra dirs/files
    gpu: bool = False
    cpus: int = 1
    ram_gb: int = 8
    time: str = "01:00:00"              # HH:MM:SS (slurm only)
    partition: str = ""
    venv_path: str = ".venv"            # path relative to project root


# ─── Abstract Machine ────────────────────────────────────────────────────────

class Machine(abc.ABC):
    """Base class for all remote execution targets."""

    name: str               # matches ssh config Host
    remote_base: str        # where projects land on the remote
    pre_hooks: list[str]    # shell commands run before the job
    post_hooks: list[str]   # shell commands run after the job

    def __init__(self, name: str, remote_base: str = "~/cluster_jobs",
                 pre_hooks: list[str] | None = None,
                 post_hooks: list[str] | None = None):
        self.name = name
        self.remote_base = remote_base
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []

    # ── archive & upload ──────────────────────────────────────────────────

    def bundle(self, job: Job) -> pathlib.Path:
        """Create a .tar.gz with the target file + dependencies."""
        project_root = job.target_file.parent.resolve()
        paths = [job.target_file.resolve()]
        for dep in job.deps:
            p = (project_root / dep).resolve() if not dep.is_absolute() else dep.resolve()
            if p.exists():
                paths.append(p)
            else:
                print(f"  warning: dependency {dep} not found, skipping")
        tmp = pathlib.Path(tempfile.mkdtemp()) / "bundle.tar.gz"
        _tar_create(tmp, paths, project_root)
        return tmp

    def upload(self, job: Job, archive: pathlib.Path) -> str:
        """SCP the bundle onto the machine, return the remote working dir."""
        run_id = job.target_file.stem
        remote_dir = f"{self.remote_base}/{run_id}"
        _ssh(self.name, f"mkdir -p {remote_dir}")
        _scp(str(archive), f"{self.name}:{remote_dir}/bundle.tar.gz")
        _ssh(self.name, f"cd {remote_dir} && tar xzf bundle.tar.gz && rm bundle.tar.gz")
        return remote_dir

    # ── hooks ─────────────────────────────────────────────────────────────

    def run_pre_hooks(self, job: Job, remote_dir: str) -> None:
        for hook in self.pre_hooks:
            _ssh(self.name, f"cd {remote_dir} && {hook}")

    def run_post_hooks(self, job: Job, remote_dir: str) -> None:
        for hook in self.post_hooks:
            _ssh(self.name, f"cd {remote_dir} && {hook}")

    # ── execution (subclasses implement) ──────────────────────────────────

    @abc.abstractmethod
    def execute(self, job: Job, remote_dir: str) -> None:
        """Run the job on the remote machine."""

    @abc.abstractmethod
    def kind(self) -> str:
        """Human label like 'simple' or 'slurm'."""

    # ── retrieve results ──────────────────────────────────────────────────

    def retrieve(self, job: Job, remote_dir: str) -> None:
        """Pull the executed notebook / outputs back."""
        fname = job.target_file.name
        local_out = job.target_file.parent / f"{job.target_file.stem}.executed{job.target_file.suffix}"
        _scp(f"{self.name}:{remote_dir}/{fname}", str(local_out))
        print(f"  result saved to {local_out}")

    # ── full dispatch pipeline ────────────────────────────────────────────

    def dispatch(self, job: Job) -> None:
        print(f"  bundling {job.target_file.name} ...")
        archive = self.bundle(job)
        print(f"  uploading to {self.name}:{self.remote_base} ...")
        remote_dir = self.upload(job, archive)
        self.run_pre_hooks(job, remote_dir)
        print(f"  executing on {self.name} ...")
        self.execute(job, remote_dir)
        self.run_post_hooks(job, remote_dir)
        print(f"  retrieving results ...")
        self.retrieve(job, remote_dir)
        print("  done.")


# ─── Simple (non-cluster) machine ────────────────────────────────────────────

class SimpleMachine(Machine):
    """A plain remote box — runs directly via ssh."""

    def kind(self) -> str:
        return "simple"

    def _build_run_cmd(self, job: Job, remote_dir: str) -> str:
        activate = ""
        if job.venv_path:
            activate = f"source {job.venv_path}/bin/activate && "

        fname = job.target_file.name
        if fname.endswith(".ipynb"):
            return (
                f"cd {remote_dir} && {activate}"
                f"jupyter nbconvert --to notebook --execute --inplace "
                f"--ExecutePreprocessor.timeout=-1 {shlex.quote(fname)}"
            )
        else:
            return f"cd {remote_dir} && {activate}python {shlex.quote(fname)}"

    def execute(self, job: Job, remote_dir: str) -> None:
        cmd = self._build_run_cmd(job, remote_dir)
        result = _ssh(self.name, cmd, check=False, capture=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"  remote execution failed (exit {result.returncode})")
            if result.stderr:
                print(result.stderr)
            raise RuntimeError("Remote execution failed")


# ─── Slurm cluster machine ───────────────────────────────────────────────────

class SlurmCluster(Machine):
    """A machine managed by Slurm — submits via sbatch."""

    default_partition: str
    gpu_partition: str
    modules: list[str]  # `module load ...` before running

    def __init__(self, name: str, remote_base: str = "~/cluster_jobs",
                 default_partition: str = "all", gpu_partition: str = "gpu",
                 modules: list[str] | None = None,
                 pre_hooks: list[str] | None = None,
                 post_hooks: list[str] | None = None):
        super().__init__(name, remote_base, pre_hooks, post_hooks)
        self.default_partition = default_partition
        self.gpu_partition = gpu_partition
        self.modules = modules or []

    def kind(self) -> str:
        return "slurm"

    def _sbatch_script(self, job: Job, remote_dir: str) -> str:
        partition = job.partition or (self.gpu_partition if job.gpu else self.default_partition)
        gres = f"#SBATCH --gres=gpu:1\n" if job.gpu else ""
        modules = "\n".join(f"module load {m}" for m in self.modules)
        activate = f"source {job.venv_path}/bin/activate" if job.venv_path else ""

        fname = job.target_file.name
        if fname.endswith(".ipynb"):
            run_cmd = (
                f"jupyter nbconvert --to notebook --execute --inplace "
                f"--ExecutePreprocessor.timeout=-1 {shlex.quote(fname)}"
            )
        else:
            run_cmd = f"python {shlex.quote(fname)}"

        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={job.target_file.stem}
            #SBATCH --partition={partition}
            #SBATCH --cpus-per-task={job.cpus}
            #SBATCH --mem={job.ram_gb}G
            #SBATCH --time={job.time}
            #SBATCH --output=slurm_%j.out
            #SBATCH --error=slurm_%j.err
            {gres}
            {modules}
            {activate}

            cd {remote_dir}
            {run_cmd}
        """)

    def execute(self, job: Job, remote_dir: str) -> None:
        script = self._sbatch_script(job, remote_dir)
        # Write the sbatch script remotely
        escaped = script.replace("'", "'\\''")
        _ssh(self.name, f"cat > {remote_dir}/run.sbatch << 'SBATCH_EOF'\n{script}SBATCH_EOF")

        # Submit and capture job id
        result = _ssh(self.name, f"cd {remote_dir} && sbatch run.sbatch")
        stdout = result.stdout.strip()
        print(f"  {stdout}")

        # Parse job id and wait
        job_id = stdout.split()[-1] if stdout else None
        if job_id and job_id.isdigit():
            print(f"  waiting for slurm job {job_id} to finish ...")
            _ssh(self.name, f"squeue -j {job_id} > /dev/null 2>&1; "
                            f"while squeue -j {job_id} 2>/dev/null | grep -q {job_id}; do sleep 15; done")
            # Print slurm output
            cat_res = _ssh(self.name, f"cat {remote_dir}/slurm_{job_id}.out 2>/dev/null", check=False)
            if cat_res.stdout.strip():
                print(cat_res.stdout)
            cat_err = _ssh(self.name, f"cat {remote_dir}/slurm_{job_id}.err 2>/dev/null", check=False)
            if cat_err.stdout.strip():
                print("  stderr:", cat_err.stdout)


# ─── Machine registry ────────────────────────────────────────────────────────

MACHINES: dict[str, Machine] = {
    # ── Slurm clusters ──
    "ubelix": SlurmCluster(
        name="ubelix",
        remote_base="~/cluster_jobs",
        default_partition="epyc2",
        gpu_partition="gpu",
        modules=["Anaconda3"],
    ),
    "ibucluster": SlurmCluster(
        name="ibucluster",
        remote_base="~/cluster_jobs",
        default_partition="all",
        gpu_partition="gpu",
        modules=[],
    ),
    "izb": SlurmCluster(
        name="izb",
        remote_base="~/cluster_jobs",
        default_partition="all",
        gpu_partition="gpu",
        modules=[],
    ),
    # ── Simple machines ──
    "aws": SimpleMachine(name="aws", remote_base="~/cluster_jobs"),
    "aws-2": SimpleMachine(name="aws-2", remote_base="~/cluster_jobs"),
    "aws-3": SimpleMachine(name="aws-3", remote_base="~/cluster_jobs"),
    "oracle-1": SimpleMachine(name="oracle-1", remote_base="~/cluster_jobs"),
}


def get_machine(name: str) -> Machine:
    if name in MACHINES:
        return MACHINES[name]
    # Check for user-defined machine configs
    cfg_path = MACHINE_CONFIG_DIR / f"{name}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        kind = cfg.pop("kind", "simple")
        if kind == "slurm":
            return SlurmCluster(**cfg)
        return SimpleMachine(**cfg)
    raise ValueError(f"Unknown machine: {name}. Available: {', '.join(MACHINES)}")


# ─── High-level dispatch API ─────────────────────────────────────────────────

def dispatch(target: str | pathlib.Path, machine: str, *,
             deps: list[str] | None = None, gpu: bool = False,
             cpus: int = 1, ram_gb: int = 8, time: str = "01:00:00",
             partition: str = "", venv: str = ".venv") -> None:
    """
    Main entry-point for programmatic use.

        from cluster import dispatch
        dispatch("my_notebook.ipynb", "ubelix", gpu=True, ram_gb=16)
    """
    target = pathlib.Path(target).resolve()
    if not target.exists():
        raise FileNotFoundError(target)

    project_cfg = ProjectConfig.load(target.parent)
    all_deps = [pathlib.Path(d) for d in (deps or [])] + [pathlib.Path(d) for d in project_cfg.extra_deps]

    job = Job(
        target_file=target,
        machine_name=machine,
        deps=all_deps,
        gpu=gpu,
        cpus=cpus,
        ram_gb=ram_gb,
        time=time,
        partition=partition,
        venv_path=venv,
    )

    m = get_machine(machine)
    m.dispatch(job)


# ─── TUI ─────────────────────────────────────────────────────────────────────

def _find_notebooks(root: pathlib.Path = pathlib.Path(".")) -> list[pathlib.Path]:
    nbs = sorted(root.rglob("*.ipynb"))
    return [nb for nb in nbs if ".ipynb_checkpoint" not in str(nb) and ".venv" not in str(nb)]


def _pick(prompt: str, options: list[str], *, allow_multi: bool = False) -> list[int] | int:
    """Minimal numbered-list selector."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        if allow_multi:
            try:
                indices = [int(x) - 1 for x in raw.replace(",", " ").split()]
                if all(0 <= i < len(options) for i in indices):
                    return indices
            except ValueError:
                pass
        else:
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return idx
            except ValueError:
                pass
        print("  invalid selection, try again")


def _input_default(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _confirm(summary: str) -> bool:
    print(f"\n{'=' * 60}")
    print("DISPATCH SUMMARY")
    print('=' * 60)
    print(summary)
    print('=' * 60)
    resp = input("Proceed? [Y/n] ").strip().lower()
    return resp in ("", "y", "yes")


def _auto_detect_deps(notebook: pathlib.Path) -> list[pathlib.Path]:
    """Heuristic: include ./data and .venv if they exist next to the notebook."""
    project_dir = notebook.parent
    deps = []
    for candidate in ["data", "dataset", "datasets", "src", "lib", "utils.py", "config.json"]:
        p = project_dir / candidate
        if p.exists():
            deps.append(p)
    return deps


def tui() -> None:
    """Interactive terminal UI for dispatching jobs."""
    print("╔══════════════════════════════════════════════╗")
    print("║        cluster.py — job dispatcher           ║")
    print("╚══════════════════════════════════════════════╝")

    # 1. Pick notebook
    notebooks = _find_notebooks()
    if not notebooks:
        print("No .ipynb files found in current directory tree.")
        sys.exit(1)

    nb_labels = [str(nb) for nb in notebooks]
    nb_idx = _pick("Select a notebook:", nb_labels)
    notebook = notebooks[nb_idx]
    project_cfg = ProjectConfig.load(notebook.parent)

    # 2. Pick machine
    machine_labels = [f"{name} ({m.kind()})" for name, m in MACHINES.items()]
    machine_names = list(MACHINES.keys())
    m_idx = _pick("Select target machine:", machine_labels)
    machine_name = machine_names[m_idx]
    machine = MACHINES[machine_name]

    # 3. Options
    is_slurm = isinstance(machine, SlurmCluster)
    gpu = False
    cpus = 1
    ram_gb = project_cfg.default_ram_gb
    time_limit = project_cfg.default_time
    partition = ""

    if is_slurm:
        node_type = _pick("Node type:", ["CPU", "GPU"])
        gpu = node_type == 1
        ram_gb = int(_input_default("RAM (GB)", str(ram_gb)))
        cpus = int(_input_default("CPUs", str(cpus)))
        time_limit = _input_default("Time limit (HH:MM:SS)", time_limit)
        partition = _input_default("Partition (empty = auto)", partition)

    venv = _input_default("Venv path (relative to project)", ".venv")

    # 4. Dependencies
    auto_deps = _auto_detect_deps(notebook)
    cfg_deps = [pathlib.Path(d) for d in project_cfg.extra_deps]
    all_deps = list({str(d): d for d in auto_deps + cfg_deps}.values())  # dedupe

    if all_deps:
        print(f"\nAuto-detected dependencies:")
        for d in all_deps:
            print(f"  • {d}")
        keep = input("Include these? [Y/n] ").strip().lower()
        if keep in ("n", "no"):
            all_deps = []

    extra = input("Additional deps (space-separated, or empty): ").strip()
    if extra:
        all_deps.extend(pathlib.Path(e) for e in extra.split())

    # Include venv in the bundle
    venv_path = notebook.parent / venv
    if venv_path.exists() and venv_path not in all_deps:
        include_venv = input(f"Include {venv} in bundle? [Y/n] ").strip().lower()
        if include_venv in ("", "y", "yes"):
            all_deps.append(venv_path)

    # 5. Summary & confirmation
    dep_list = "\n".join(f"    • {d}" for d in all_deps) if all_deps else "    (none)"
    summary = textwrap.dedent(f"""\
        Notebook:  {notebook}
        Machine:   {machine_name} ({machine.kind()})
        GPU:       {gpu}
        CPUs:      {cpus}
        RAM:       {ram_gb} GB
        Time:      {time_limit}
        Partition: {partition or '(auto)'}
        Venv:      {venv}
        Deps:
    {dep_list}
    """)

    if not _confirm(summary):
        print("Aborted.")
        sys.exit(0)

    # 6. Dispatch
    job = Job(
        target_file=notebook.resolve(),
        machine_name=machine_name,
        deps=[d.resolve() if d.is_absolute() else (notebook.parent / d).resolve() for d in all_deps],
        gpu=gpu,
        cpus=cpus,
        ram_gb=ram_gb,
        time=time_limit,
        partition=partition,
        venv_path=venv,
    )
    machine.dispatch(job)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch notebooks/scripts to remote machines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python cluster.py                              # interactive TUI
              python cluster.py run nb.ipynb ubelix --gpu    # direct dispatch
              python cluster.py list-machines                # show available machines
        """),
    )
    sub = parser.add_subparsers(dest="command")

    # --- tui (default) ---
    sub.add_parser("tui", help="Interactive terminal UI (default)")

    # --- run ---
    run_p = sub.add_parser("run", help="Dispatch a file directly")
    run_p.add_argument("file", help="Notebook or script to run")
    run_p.add_argument("machine", help="Target machine name")
    run_p.add_argument("--gpu", action="store_true")
    run_p.add_argument("--cpus", type=int, default=1)
    run_p.add_argument("--ram", type=int, default=8, help="RAM in GB")
    run_p.add_argument("--time", default="01:00:00", help="Time limit (HH:MM:SS)")
    run_p.add_argument("--partition", default="")
    run_p.add_argument("--venv", default=".venv")
    run_p.add_argument("--dep", action="append", default=[], help="Extra dependency (repeatable)")

    # --- list-machines ---
    sub.add_parser("list-machines", help="Show registered machines")

    args = parser.parse_args()

    if args.command == "run":
        dispatch(
            args.file, args.machine,
            deps=args.dep, gpu=args.gpu, cpus=args.cpus,
            ram_gb=args.ram, time=args.time, partition=args.partition,
            venv=args.venv,
        )
    elif args.command == "list-machines":
        for name, m in MACHINES.items():
            print(f"  {name:15s}  ({m.kind()})")
    else:
        tui()


if __name__ == "__main__":
    cli()
