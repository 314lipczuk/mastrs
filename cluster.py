#!/usr/bin/env python3
"""
cluster.py — Dispatch jupyter notebooks / python scripts to remote machines.

Usage:
    As a module:  from cluster import dispatch, wait_and_download, retrieve, ...
    As a script:  python cluster.py            # launches TUI
                  python cluster.py run nb.ipynb ubelix --gpu
                  python cluster.py retrieve   # pick & pull pending results
                  python cluster.py jobs       # list recorded jobs
                  python cluster.py list-machines
"""

from __future__ import annotations

import abc
import argparse
import dataclasses
import datetime
import json
import pathlib
import shlex
import sqlite3
import subprocess
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass, field


# ─── Utilities ───────────────────────────────────────────────────────────────

def _run(cmd: str | list[str], *, check=True, capture=True, **kw) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kw)


def _ssh(host: str, cmd: str, *, check=True, capture=True) -> subprocess.CompletedProcess:
    return _run(["ssh", host, cmd], check=check, capture=capture)


def _scp(src: str, dst: str, *, recursive=False, check=True) -> subprocess.CompletedProcess:
    flags = ["-r"] if recursive else []
    return _run(["scp", *flags, src, dst], check=check)


def _tar_create(archive: pathlib.Path, paths: list[pathlib.Path], base_dir: pathlib.Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=p.relative_to(base_dir))


# ─── Database ────────────────────────────────────────────────────────────────

DB_PATH = pathlib.Path.home() / ".config" / "cluster" / "jobs.db"


@dataclass
class JobRecord:
    id: int
    dispatched_at: str
    local_file: str
    machine: str
    remote_dir: str
    slurm_job_id: str | None
    retrieved: int          # 0 / 1
    retrieved_at: str | None


def _db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            dispatched_at TEXT NOT NULL,
            local_file   TEXT NOT NULL,
            machine      TEXT NOT NULL,
            remote_dir   TEXT NOT NULL,
            slurm_job_id TEXT,
            retrieved    INTEGER NOT NULL DEFAULT 0,
            retrieved_at TEXT
        )
    """)
    conn.commit()
    return conn


def db_save_job(local_file: str, machine: str, remote_dir: str,
                slurm_job_id: str | None) -> int:
    with _db_connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (dispatched_at, local_file, machine, remote_dir, slurm_job_id)"
            " VALUES (?, ?, ?, ?, ?)",
            (datetime.datetime.now().isoformat(timespec="seconds"),
             local_file, machine, remote_dir, slurm_job_id),
        )
        return cur.lastrowid


def db_get_jobs(*, retrieved: int | None = None) -> list[JobRecord]:
    with _db_connect() as conn:
        if retrieved is None:
            rows = conn.execute("SELECT * FROM jobs ORDER BY dispatched_at DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE retrieved=? ORDER BY dispatched_at DESC",
                (retrieved,),
            ).fetchall()
    return [JobRecord(**dict(r)) for r in rows]


def db_mark_retrieved(record_id: int) -> None:
    with _db_connect() as conn:
        conn.execute(
            "UPDATE jobs SET retrieved=1, retrieved_at=? WHERE id=?",
            (datetime.datetime.now().isoformat(timespec="seconds"), record_id),
        )


# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_CONFIG = ".cluster.json"
MACHINE_CONFIG_DIR = pathlib.Path.home() / ".config" / "cluster" / "machines"


@dataclass
class ProjectConfig:
    """Per-project defaults, read from .cluster.json alongside the notebook."""
    extra_deps: list[str] = field(default_factory=list)
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
                known = {f.name for f in dataclasses.fields(cls)}
                return cls(**{k: v for k, v in json.load(f).items() if k in known})
        return cls()


# ─── Job specification ───────────────────────────────────────────────────────

@dataclass
class Job:
    """Everything needed to describe a dispatch."""
    target_file: pathlib.Path       # notebook or script (absolute)
    machine_name: str               # ssh config Host
    deps: list[pathlib.Path] = field(default_factory=list)
    gpu: bool = False
    cpus: int = 1
    ram_gb: int = 8
    time: str = "01:00:00"          # HH:MM:SS (slurm only)
    partition: str = ""
    venv_path: str = ".venv"        # relative to project root on the remote


# ─── Abstract Machine ────────────────────────────────────────────────────────

class Machine(abc.ABC):
    """Base class for all remote execution targets."""

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
            p = dep.resolve() if dep.is_absolute() else (project_root / dep).resolve()
            if p.exists():
                paths.append(p)
            else:
                print(f"  warning: dependency {dep} not found, skipping")
        tmp = pathlib.Path(tempfile.mkdtemp()) / "bundle.tar.gz"
        _tar_create(tmp, paths, project_root)
        return tmp

    def upload(self, job: Job, archive: pathlib.Path) -> str:
        """SCP the bundle onto the machine, return the remote working dir."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{job.target_file.stem}_{ts}"
        remote_dir = f"{self.remote_base}/{run_id}"
        _ssh(self.name, f"mkdir -p {remote_dir}")
        _scp(str(archive), f"{self.name}:{remote_dir}/bundle.tar.gz")
        _ssh(self.name, f"cd {remote_dir} && tar xzf bundle.tar.gz && rm bundle.tar.gz")
        return remote_dir

    def run_pre_hooks(self, job: Job, remote_dir: str) -> None:
        for hook in self.pre_hooks:
            _ssh(self.name, f"cd {remote_dir} && {hook}")

    def run_post_hooks(self, job: Job, remote_dir: str) -> None:
        for hook in self.post_hooks:
            _ssh(self.name, f"cd {remote_dir} && {hook}")

    # ── execution — subclasses implement ──────────────────────────────────

    @abc.abstractmethod
    def execute(self, job: Job, remote_dir: str) -> str | None:
        """
        Fire-and-forget: start the job on the remote and return immediately.
        Returns a job identifier (slurm job id, or None for simple machines).
        """

    @abc.abstractmethod
    def kind(self) -> str:
        """Human label: 'simple' or 'slurm'."""

    # ── status & retrieval ────────────────────────────────────────────────

    @abc.abstractmethod
    def is_done(self, record: JobRecord) -> tuple[bool, str]:
        """
        Returns (finished, status_string).
        finished=True means results can be retrieved.
        """

    def retrieve_record(self, record: JobRecord) -> pathlib.Path:
        """Pull the executed notebook back from the remote."""
        local_file = pathlib.Path(record.local_file)
        fname = local_file.name
        local_out = local_file.parent / f"{local_file.stem}.executed{local_file.suffix}"
        _scp(f"{self.name}:{record.remote_dir}/{fname}", str(local_out))
        db_mark_retrieved(record.id)
        print(f"  saved → {local_out}")
        return local_out

    # ── full dispatch pipeline ────────────────────────────────────────────

    def dispatch(self, job: Job) -> JobRecord:
        """Bundle, upload, fire job, save record. Does NOT wait for completion."""
        print(f"  bundling {job.target_file.name} ...")
        archive = self.bundle(job)
        print(f"  uploading to {self.name}:{self.remote_base} ...")
        remote_dir = self.upload(job, archive)
        self.run_pre_hooks(job, remote_dir)
        print(f"  submitting job on {self.name} ...")
        slurm_id = self.execute(job, remote_dir)
        self.run_post_hooks(job, remote_dir)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        record_id = db_save_job(str(job.target_file), self.name, remote_dir, slurm_id)
        record = JobRecord(
            id=record_id,
            dispatched_at=now,
            local_file=str(job.target_file),
            machine=self.name,
            remote_dir=remote_dir,
            slurm_job_id=slurm_id,
            retrieved=0,
            retrieved_at=None,
        )
        label = f"slurm job {slurm_id}" if slurm_id else "background process"
        print(f"  dispatched as {label} (db id={record_id}). Use 'retrieve' to pull results.")
        return record


# ─── Simple (non-cluster) machine ────────────────────────────────────────────

class SimpleMachine(Machine):
    """A plain remote box — fires the job via nohup and disconnects."""

    def kind(self) -> str:
        return "simple"

    def _run_cmd(self, job: Job) -> str:
        activate = f"source {job.venv_path}/bin/activate && " if job.venv_path else ""
        fname = shlex.quote(job.target_file.name)
        if job.target_file.suffix == ".ipynb":
            return (
                f"{activate}jupyter nbconvert --to notebook --execute --inplace "
                f"--ExecutePreprocessor.timeout=-1 {fname}"
            )
        return f"{activate}python {fname}"

    def execute(self, job: Job, remote_dir: str) -> str | None:
        inner = self._run_cmd(job)
        script = textwrap.dedent(f"""\
            #!/bin/bash
            cd {remote_dir}
            {inner} > run.out 2> run.err && touch .done || touch .failed
        """)
        # Write via heredoc — avoids quoting/escaping pitfalls
        _ssh(self.name,
             f"cat > {remote_dir}/run.sh << 'RUN_EOF'\n{script}RUN_EOF\n"
             f"chmod +x {remote_dir}/run.sh")
        # Detach completely: redirect all fds so ssh returns immediately
        _ssh(self.name, f"nohup {remote_dir}/run.sh > /dev/null 2>&1 < /dev/null &")
        return None

    def is_done(self, record: JobRecord) -> tuple[bool, str]:
        r = _ssh(
            self.name,
            f"if [ -f {record.remote_dir}/.done ]; then echo done;"
            f" elif [ -f {record.remote_dir}/.failed ]; then echo failed;"
            f" else echo running; fi",
            check=False,
        )
        status = r.stdout.strip() if r.returncode == 0 else "unknown"
        return status in ("done", "failed"), status

    def retrieve_record(self, record: JobRecord) -> pathlib.Path:
        out = super().retrieve_record(record)
        # Also pull stdout/stderr logs
        local_file = pathlib.Path(record.local_file)
        for log in ("run.out", "run.err"):
            local_log = local_file.parent / f"{local_file.stem}.{log}"
            _scp(f"{self.name}:{record.remote_dir}/{log}", str(local_log), check=False)
        return out


# ─── Slurm cluster machine ───────────────────────────────────────────────────

class SlurmCluster(Machine):
    """A machine managed by Slurm — submits via sbatch and returns immediately."""

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
        gres_line = "#SBATCH --gres=gpu:1" if job.gpu else ""
        modules = "\n".join(f"module load {m}" for m in self.modules)
        activate = f"source {job.venv_path}/bin/activate" if job.venv_path else ""
        fname = shlex.quote(job.target_file.name)
        if job.target_file.suffix == ".ipynb":
            run_cmd = (
                f"jupyter nbconvert --to notebook --execute --inplace "
                f"--ExecutePreprocessor.timeout=-1 {fname}"
            )
        else:
            run_cmd = f"python {fname}"

        return textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={job.target_file.stem}
            #SBATCH --partition={partition}
            #SBATCH --cpus-per-task={job.cpus}
            #SBATCH --mem={job.ram_gb}G
            #SBATCH --time={job.time}
            #SBATCH --output=slurm_%j.out
            #SBATCH --error=slurm_%j.err
            {gres_line}
            {modules}
            {activate}

            cd {remote_dir}
            {run_cmd}
        """)

    def execute(self, job: Job, remote_dir: str) -> str | None:
        script = self._sbatch_script(job, remote_dir)
        # Write sbatch script via heredoc
        _ssh(self.name, f"cat > {remote_dir}/run.sbatch << 'SBATCH_EOF'\n{script}SBATCH_EOF")
        result = _ssh(self.name, f"cd {remote_dir} && sbatch run.sbatch")
        stdout = result.stdout.strip()
        print(f"  {stdout}")
        parts = stdout.split()
        job_id = parts[-1] if parts and parts[-1].isdigit() else None
        return job_id

    def is_done(self, record: JobRecord) -> tuple[bool, str]:
        if not record.slurm_job_id:
            return True, "unknown (no job id)"
        r = _ssh(
            self.name,
            f"squeue -j {record.slurm_job_id} --noheader --format='%T' 2>/dev/null",
            check=False,
        )
        state = r.stdout.strip()
        if state:
            return False, state          # still in queue: PENDING, RUNNING, …
        # Job is gone from squeue — check sacct for final state
        sacct = _ssh(
            self.name,
            f"sacct -j {record.slurm_job_id} --noheader --format=State --parsable2 2>/dev/null | head -1",
            check=False,
        )
        final = sacct.stdout.strip() or "COMPLETED"
        return True, final

    def retrieve_record(self, record: JobRecord) -> pathlib.Path:
        out = super().retrieve_record(record)
        # Also pull slurm stdout/stderr
        if record.slurm_job_id:
            local_file = pathlib.Path(record.local_file)
            for ext in (".out", ".err"):
                remote = f"{record.remote_dir}/slurm_{record.slurm_job_id}{ext}"
                local = local_file.parent / f"slurm_{record.slurm_job_id}{ext}"
                _scp(f"{self.name}:{remote}", str(local), check=False)
        return out


# ─── Machine registry ────────────────────────────────────────────────────────

MACHINES: dict[str, Machine] = {
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
    ),
    "izb": SlurmCluster(
        name="izb",
        remote_base="~/cluster_jobs",
        default_partition="all",
        gpu_partition="gpu",
    ),
    "aws":      SimpleMachine(name="aws",      remote_base="~/cluster_jobs"),
    "aws-2":    SimpleMachine(name="aws-2",    remote_base="~/cluster_jobs"),
    "aws-3":    SimpleMachine(name="aws-3",    remote_base="~/cluster_jobs"),
    "oracle-1": SimpleMachine(name="oracle-1", remote_base="~/cluster_jobs"),
}


def get_machine(name: str) -> Machine:
    if name in MACHINES:
        return MACHINES[name]
    cfg_path = MACHINE_CONFIG_DIR / f"{name}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        kind = cfg.pop("kind", "simple")
        if kind == "slurm":
            return SlurmCluster(**cfg)
        return SimpleMachine(**cfg)
    raise ValueError(f"Unknown machine: {name!r}. Available: {', '.join(MACHINES)}")


# ─── High-level API ──────────────────────────────────────────────────────────

def dispatch(target: str | pathlib.Path, machine: str, *,
             deps: list[str] | None = None, gpu: bool = False,
             cpus: int = 1, ram_gb: int = 8, time: str = "01:00:00",
             partition: str = "", venv: str = ".venv") -> JobRecord:
    """
    Programmatic dispatch. Returns the JobRecord saved to the DB.

        from cluster import dispatch
        record = dispatch("train.ipynb", "ubelix", gpu=True, ram_gb=16)
    """
    target = pathlib.Path(target).resolve()
    if not target.exists():
        raise FileNotFoundError(target)

    project_cfg = ProjectConfig.load(target.parent)
    all_deps = (
        [pathlib.Path(d) for d in (deps or [])]
        + [pathlib.Path(d) for d in project_cfg.extra_deps]
    )

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
    return get_machine(machine).dispatch(job)


def retrieve(record_id: int) -> pathlib.Path:
    """Pull results for a recorded job by its DB id."""
    records = {r.id: r for r in db_get_jobs()}
    if record_id not in records:
        raise ValueError(f"No job with id={record_id}")
    record = records[record_id]
    machine = get_machine(record.machine)
    done, status = machine.is_done(record)
    if not done:
        raise RuntimeError(f"Job is not finished yet (status: {status})")
    return machine.retrieve_record(record)


def wait_and_download(record_or_id: JobRecord | int, *,
                      poll_interval: float = 30,
                      timeout: float | None = None) -> pathlib.Path:
    """
    Block until a dispatched job finishes, then retrieve the result.

    Optional companion to the fire-and-forget dispatch():

        record = dispatch("train.ipynb", "ubelix", gpu=True)
        result = wait_and_download(record)           # blocks

    Or by DB id:

        result = wait_and_download(5, timeout=3600)  # 1h max

    Args:
        record_or_id: A JobRecord returned by dispatch(), or an integer DB id.
        poll_interval: Seconds between status checks (default 30).
        timeout: Maximum seconds to wait. None = wait forever.

    Returns:
        Path to the locally retrieved file.

    Raises:
        TimeoutError: If timeout is exceeded.
        RuntimeError: If the job finished with a failure status.
    """
    # Resolve to a JobRecord
    if isinstance(record_or_id, int):
        records = {r.id: r for r in db_get_jobs()}
        if record_or_id not in records:
            raise ValueError(f"No job with id={record_or_id}")
        record = records[record_or_id]
    else:
        record = record_or_id

    machine = get_machine(record.machine)
    start = time.monotonic()
    print(f"  waiting for job {record.id} on {record.machine} ...")

    while True:
        done, status = machine.is_done(record)
        if done:
            break
        elapsed = time.monotonic() - start
        if timeout is not None and elapsed >= timeout:
            raise TimeoutError(
                f"Job {record.id} still {status} after {elapsed:.0f}s (timeout={timeout}s)"
            )
        print(f"  [{elapsed:6.0f}s] status: {status}")
        time.sleep(poll_interval)

    elapsed = time.monotonic() - start
    print(f"  job finished ({status}) after {elapsed:.0f}s — retrieving ...")

    if status.upper() in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
        # Still retrieve so the user can inspect logs, but warn
        print(f"  warning: job ended with status {status}")

    return machine.retrieve_record(record)


# ─── TUI helpers ─────────────────────────────────────────────────────────────

def _pick(prompt: str, options: list[str]) -> int:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i:2}] {opt}")
    while True:
        raw = input("> ").strip()
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
    print("=" * 60)
    print(summary)
    print("=" * 60)
    return input("Proceed? [Y/n] ").strip().lower() in ("", "y", "yes")


def _find_notebooks(root: pathlib.Path = pathlib.Path(".")) -> list[pathlib.Path]:
    nbs = sorted(root.rglob("*.ipynb"))
    return [nb for nb in nbs if ".ipynb_checkpoint" not in str(nb) and ".venv" not in str(nb)]


def _auto_detect_deps(notebook: pathlib.Path) -> list[pathlib.Path]:
    project_dir = notebook.parent
    candidates = ["data", "dataset", "datasets", "src", "lib", "utils.py", "config.json"]
    return [project_dir / c for c in candidates if (project_dir / c).exists()]


# ─── TUI flows ───────────────────────────────────────────────────────────────

def tui_dispatch() -> None:
    # 1. Pick notebook
    notebooks = _find_notebooks()
    if not notebooks:
        print("No .ipynb files found in current directory tree.")
        return

    nb_idx = _pick("Select a notebook:", [str(nb) for nb in notebooks])
    notebook = notebooks[nb_idx]
    project_cfg = ProjectConfig.load(notebook.parent)

    # 2. Pick machine
    machine_names = list(MACHINES.keys())
    m_idx = _pick(
        "Select target machine:",
        [f"{n} ({MACHINES[n].kind()})" for n in machine_names],
    )
    machine_name = machine_names[m_idx]
    machine = MACHINES[machine_name]

    # 3. Slurm options
    is_slurm = isinstance(machine, SlurmCluster)
    gpu = False
    cpus = 1
    ram_gb = project_cfg.default_ram_gb
    time_limit = project_cfg.default_time
    partition = ""

    if is_slurm:
        gpu = _pick("Node type:", ["CPU", "GPU"]) == 1
        ram_gb = int(_input_default("RAM (GB)", str(ram_gb)))
        cpus = int(_input_default("CPUs", str(cpus)))
        time_limit = _input_default("Time limit (HH:MM:SS)", time_limit)
        partition = _input_default("Partition (empty = auto)", partition)

    venv = _input_default("Venv path (relative to project)", ".venv")

    # 4. Dependencies
    auto_deps = _auto_detect_deps(notebook)
    cfg_deps = [pathlib.Path(d) for d in project_cfg.extra_deps]
    all_deps: list[pathlib.Path] = list({str(d): d for d in auto_deps + cfg_deps}.values())

    if all_deps:
        print("\nAuto-detected dependencies:")
        for d in all_deps:
            print(f"  • {d}")
        if input("Include these? [Y/n] ").strip().lower() in ("n", "no"):
            all_deps = []

    extra = input("Additional deps (space-separated paths, or empty): ").strip()
    if extra:
        all_deps.extend(pathlib.Path(e) for e in extra.split())

    venv_path = notebook.parent / venv
    if venv_path.exists() and venv_path not in all_deps:
        if input(f"Include {venv} in bundle? [Y/n] ").strip().lower() in ("", "y", "yes"):
            all_deps.append(venv_path)

    # 5. Summary & confirm
    dep_list = "\n".join(f"    • {d}" for d in all_deps) or "    (none)"
    summary = (
        f"  Notebook:  {notebook}\n"
        f"  Machine:   {machine_name} ({machine.kind()})\n"
        f"  GPU:       {gpu}\n"
        f"  CPUs:      {cpus}\n"
        f"  RAM:       {ram_gb} GB\n"
        f"  Time:      {time_limit}\n"
        f"  Partition: {partition or '(auto)'}\n"
        f"  Venv:      {venv}\n"
        f"  Deps:\n{dep_list}"
    )
    if not _confirm(summary):
        print("Aborted.")
        return

    # 6. Dispatch
    job = Job(
        target_file=notebook.resolve(),
        machine_name=machine_name,
        deps=[
            (notebook.parent / d).resolve() if not d.is_absolute() else d.resolve()
            for d in all_deps
        ],
        gpu=gpu,
        cpus=cpus,
        ram_gb=ram_gb,
        time=time_limit,
        partition=partition,
        venv_path=venv,
    )
    machine.dispatch(job)


def tui_retrieve() -> None:
    pending = db_get_jobs(retrieved=0)
    if not pending:
        print("\nNo pending jobs to retrieve.")
        return

    # Show status for each
    print("\nChecking job statuses ...")
    rows: list[tuple[JobRecord, bool, str]] = []
    for rec in pending:
        try:
            m = get_machine(rec.machine)
            done, status = m.is_done(rec)
        except Exception as e:
            done, status = False, f"error: {e}"
        rows.append((rec, done, status))

    labels = []
    for rec, done, status in rows:
        fname = pathlib.Path(rec.local_file).name
        ready = "READY" if done else f"  {status}"
        labels.append(
            f"id={rec.id}  {rec.dispatched_at}  {rec.machine:12}  {fname}  → {ready}"
        )

    idx = _pick("Select a job to retrieve (must be READY):", labels)
    rec, done, status = rows[idx]

    if not done:
        print(f"  Job is not finished (status: {status}). Cannot retrieve yet.")
        return

    print(f"  retrieving from {rec.machine}:{rec.remote_dir} ...")
    get_machine(rec.machine).retrieve_record(rec)


def tui_list() -> None:
    records = db_get_jobs()
    if not records:
        print("\nNo jobs recorded yet.")
        return
    print(f"\n{'ID':>4}  {'Dispatched':19}  {'Machine':12}  {'Status':10}  {'File'}")
    print("─" * 80)
    for rec in records:
        status = "retrieved" if rec.retrieved else "pending"
        fname = pathlib.Path(rec.local_file).name
        slurm = f"  (slurm {rec.slurm_job_id})" if rec.slurm_job_id else ""
        print(f"{rec.id:4}  {rec.dispatched_at:19}  {rec.machine:12}  {status:10}  {fname}{slurm}")


def tui() -> None:
    print("╔══════════════════════════════════════════════╗")
    print("║        cluster.py — job dispatcher           ║")
    print("╚══════════════════════════════════════════════╝")

    actions = [
        ("Dispatch a new job",    tui_dispatch),
        ("Retrieve results",      tui_retrieve),
        ("List recorded jobs",    tui_list),
        ("Quit",                  None),
    ]

    while True:
        idx = _pick("What would you like to do?", [a for a, _ in actions])
        _, fn = actions[idx]
        if fn is None:
            print("Bye.")
            break
        fn()
        print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch notebooks/scripts to remote machines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python cluster.py                                       # interactive TUI
              python cluster.py run train.ipynb ubelix --gpu         # fire & forget
              python cluster.py run nb.ipynb aws --wait --poll 10   # block until done
              python cluster.py retrieve 3                           # pull job id=3
              python cluster.py jobs                                 # list all jobs
              python cluster.py list-machines
        """),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tui", help="Interactive TUI (default when no command given)")

    run_p = sub.add_parser("run", help="Dispatch a file directly")
    run_p.add_argument("file")
    run_p.add_argument("machine")
    run_p.add_argument("--gpu", action="store_true")
    run_p.add_argument("--cpus", type=int, default=1)
    run_p.add_argument("--ram", type=int, default=8, metavar="GB")
    run_p.add_argument("--time", default="01:00:00", metavar="HH:MM:SS")
    run_p.add_argument("--partition", default="")
    run_p.add_argument("--venv", default=".venv")
    run_p.add_argument("--dep", action="append", default=[], metavar="PATH",
                       help="Extra dependency (repeatable)")
    run_p.add_argument("--wait", action="store_true",
                       help="Block until job finishes, then retrieve result")
    run_p.add_argument("--poll", type=float, default=30, metavar="SEC",
                       help="Poll interval for --wait (default 30s)")
    run_p.add_argument("--timeout", type=float, default=None, metavar="SEC",
                       help="Max wait time for --wait (default: no limit)")

    ret_p = sub.add_parser("retrieve", help="Pull results for a job by DB id")
    ret_p.add_argument("id", type=int, nargs="?", help="Job DB id (omit for TUI picker)")

    sub.add_parser("jobs", help="List all recorded jobs")
    sub.add_parser("list-machines", help="Show registered machines")

    args = parser.parse_args()

    if args.command == "run":
        record = dispatch(
            args.file, args.machine,
            deps=args.dep, gpu=args.gpu, cpus=args.cpus,
            ram_gb=args.ram, time=args.time, partition=args.partition,
            venv=args.venv,
        )
        if args.wait:
            wait_and_download(record, poll_interval=args.poll, timeout=args.timeout)
    elif args.command == "retrieve":
        if args.id is not None:
            retrieve(args.id)
        else:
            tui_retrieve()
    elif args.command == "jobs":
        tui_list()
    elif args.command == "list-machines":
        for name, m in MACHINES.items():
            print(f"  {name:15s}  ({m.kind()})")
    else:
        tui()


if __name__ == "__main__":
    cli()
