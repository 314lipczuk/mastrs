# cluster.py

A self-contained tool for dispatching Jupyter notebooks and Python scripts to remote machines over SSH. Handles bundling code + dependencies, uploading via `scp`, executing remotely (plain SSH or Slurm), and retrieving results — all driven by your `~/.ssh/config`.

## Quick start

```bash
# Interactive TUI — pick a notebook, pick a machine, go
python cluster.py

# Direct dispatch (fire-and-forget)
python cluster.py run train.ipynb ubelix --gpu --ram 16

# Wait for it to finish, then auto-retrieve
python cluster.py run train.ipynb aws --wait

# Check what's been dispatched
python cluster.py jobs

# Pull results when ready
python cluster.py retrieve 3
```

From Python:

```python
from cluster import dispatch, wait_and_download, retrieve

record = dispatch("train.ipynb", "ubelix", gpu=True, ram_gb=16)

# Option A: come back later
retrieve(record.id)

# Option B: block until done
wait_and_download(record, poll_interval=30, timeout=7200)
```

## How it works

```
                         local                              remote
                    ┌─────────────┐     scp/ssh        ┌──────────────┐
                    │  bundle     │ ──────────────────► │  ~/cluster_  │
                    │  .tar.gz    │                     │  jobs/<id>/  │
                    │             │                     │              │
                    │  notebook   │                     │  notebook    │
                    │  + deps     │                     │  + deps      │
                    │  + .venv    │                     │  + .venv     │
                    └─────────────┘                     └──────┬───────┘
                                                               │
                                                    ┌──────────┴──────────┐
                                                    │                     │
                                              SimpleMachine          SlurmCluster
                                              nohup run.sh           sbatch run.sbatch
                                              (detach + sentinel)    (submit + squeue)
                                                    │                     │
                                                    ▼                     ▼
                    ┌─────────────┐     scp        .done / .failed    sacct COMPLETED
                    │  *.executed │ ◄────────────   result ready       result ready
                    │  .ipynb     │
                    └─────────────┘
                    + run.out / slurm_*.out
```

### Dispatch pipeline

1. **Bundle** — tar.gz the target file + all dependencies (data dirs, venv, utils, etc.)
2. **Upload** — `scp` the archive to `~/cluster_jobs/<stem>_<timestamp>/` on the remote, extract it
3. **Pre-hooks** — run any machine- or project-defined shell commands
4. **Execute** — fire-and-forget:
   - **SimpleMachine**: writes a `run.sh`, launches it via `nohup` with all fds redirected so SSH returns immediately. Sentinel files `.done` / `.failed` mark completion.
   - **SlurmCluster**: writes a `run.sbatch` with `#SBATCH` directives, submits via `sbatch`, returns the job ID.
5. **Post-hooks** — run any post-submission shell commands
6. **Record** — save job metadata (machine, remote path, slurm ID, timestamp) to a local SQLite DB

Retrieval is a separate step — either manually (`retrieve`), via TUI, or by passing `--wait` to block.

### Notebook execution

Notebooks are executed remotely via:

```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb
```

This runs all cells in order and writes outputs (plots, print statements, etc.) back into the `.ipynb` file in-place. When you retrieve, you get a fully rendered notebook with all outputs preserved.

## Architecture

### Class hierarchy

```
Machine (ABC)
├── SimpleMachine      — plain SSH box (aws, oracle, etc.)
└── SlurmCluster       — Slurm-managed cluster (ubelix, ibucluster, izb)
```

**`Machine`** defines the full lifecycle:

| Method | Purpose |
|---|---|
| `bundle(job)` | Create .tar.gz from target + deps |
| `upload(job, archive)` | SCP to remote, extract |
| `execute(job, remote_dir)` | *Abstract* — fire the job |
| `is_done(record)` | *Abstract* — check remote status |
| `retrieve_record(record)` | SCP result back, mark DB as retrieved |
| `dispatch(job)` | Orchestrates the full pipeline |
| `run_pre_hooks` / `run_post_hooks` | Run shell commands before/after execution |

Subclasses only need to implement `execute()`, `is_done()`, and `kind()`. They can optionally override `retrieve_record()` to pull extra files (slurm logs, stdout/stderr).

### Job tracking

All dispatched jobs are recorded in a SQLite database at:

```
~/.config/cluster/jobs.db
```

Schema:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `dispatched_at` | TEXT | ISO timestamp |
| `local_file` | TEXT | Absolute path to the original file |
| `machine` | TEXT | SSH host name |
| `remote_dir` | TEXT | Where files live on the remote |
| `slurm_job_id` | TEXT | Slurm job ID (NULL for simple machines) |
| `retrieved` | INTEGER | 0 = pending, 1 = retrieved |
| `retrieved_at` | TEXT | ISO timestamp of retrieval |

### Status checking

- **SimpleMachine**: checks for `.done` or `.failed` sentinel files in the remote working directory
- **SlurmCluster**: queries `squeue` first (for running/pending jobs), then falls back to `sacct` (for finished jobs) to get the final state (COMPLETED, FAILED, CANCELLED, etc.)

## CLI reference

### `python cluster.py` (or `tui`)

Interactive menu with four options:
1. **Dispatch a new job** — pick notebook, machine, configure resources, confirm, send
2. **Retrieve results** — shows all pending jobs with live status checks, pick one that's READY
3. **List recorded jobs** — table of all dispatched jobs and their retrieved/pending status
4. **Quit**

### `python cluster.py run <file> <machine> [options]`

| Flag | Default | Description |
|---|---|---|
| `--gpu` | off | Request a GPU node (slurm only) |
| `--cpus N` | 1 | CPUs per task |
| `--ram GB` | 8 | Memory in GB |
| `--time HH:MM:SS` | 01:00:00 | Wall-time limit (slurm only) |
| `--partition NAME` | auto | Slurm partition (auto-selects gpu/cpu partition) |
| `--venv PATH` | .venv | Path to virtualenv relative to project root |
| `--dep PATH` | — | Extra dependency to bundle (repeatable) |
| `--wait` | off | Block until job finishes, then retrieve |
| `--poll SEC` | 30 | Polling interval when using `--wait` |
| `--timeout SEC` | none | Max wait time when using `--wait` |

### `python cluster.py retrieve [ID]`

Pull results for a specific job by its DB id. Omit the id to get the interactive picker.

### `python cluster.py jobs`

Print a table of all recorded jobs.

### `python cluster.py list-machines`

Show all registered machines and their type (simple/slurm).

## Python API

### `dispatch(target, machine, **kwargs) -> JobRecord`

Fire-and-forget. Returns a `JobRecord` with the DB id, remote path, etc.

```python
record = dispatch("analysis.ipynb", "ubelix",
                  gpu=True, ram_gb=32, time="04:00:00",
                  deps=["data", "src"])
```

### `wait_and_download(record_or_id, poll_interval=30, timeout=None) -> Path`

Blocks until the job finishes, then retrieves the result. Accepts a `JobRecord` or a DB id (int).

```python
result_path = wait_and_download(record)
result_path = wait_and_download(42, timeout=3600)
```

Raises `TimeoutError` if the timeout is exceeded. On failure statuses (FAILED, CANCELLED, etc.) it still retrieves the file + logs so you can inspect what went wrong, but prints a warning.

### `retrieve(record_id) -> Path`

Immediately pull results for a finished job. Raises `RuntimeError` if the job is still running.

### `db_get_jobs(retrieved=None) -> list[JobRecord]`

Query the job database. Pass `retrieved=0` for pending, `retrieved=1` for retrieved, or `None` for all.

## Configuration

### Per-project: `.cluster.json`

Place a `.cluster.json` alongside your notebooks to set defaults:

```json
{
  "extra_deps": ["data", "src/models"],
  "default_machine": "ubelix",
  "default_gpu": false,
  "default_ram_gb": 16,
  "default_time": "02:00:00",
  "pre_run": ["pip install -r requirements.txt"],
  "post_run": []
}
```

The TUI reads these as defaults (you can still override interactively). The `extra_deps` are always included in the bundle.

### Per-machine: `~/.config/cluster/machines/<name>.json`

Define additional machines beyond the built-in ones:

```json
{
  "kind": "slurm",
  "name": "my-cluster",
  "remote_base": "~/jobs",
  "default_partition": "compute",
  "gpu_partition": "gpu-a100",
  "modules": ["CUDA/12.0", "Python/3.11"]
}
```

```json
{
  "kind": "simple",
  "name": "my-server",
  "remote_base": "/data/runs",
  "pre_hooks": ["export CUDA_VISIBLE_DEVICES=0"]
}
```

The `name` field must match a `Host` entry in your `~/.ssh/config`.

## Built-in machines

| Name | Type | Notes |
|---|---|---|
| `ubelix` | slurm | UBELIX cluster, epyc2/gpu partitions, loads Anaconda3 |
| `ibucluster` | slurm | IBU HPC cluster |
| `izb` | slurm | IZB Lisbon |
| `aws` | simple | EC2 instance |
| `aws-2` | simple | EC2 instance |
| `aws-3` | simple | EC2 instance |
| `oracle-1` | simple | Oracle Cloud VM |

All machines use host names from `~/.ssh/config` — no credentials are stored in the script.

## SSH requirements

- All remote access goes through `ssh` and `scp` using host names from `~/.ssh/config`
- The script never handles passwords or keys directly — set up key-based auth or ssh-agent beforehand
- For jump hosts (like `compute_ibu` via `ibucluster`), configure `ProxyJump` in your ssh config
- Remote machines need: `tar`, `bash`, and either `jupyter` (for notebooks) or `python` (for scripts)
- Slurm clusters additionally need: `sbatch`, `squeue`, `sacct`

## Dependency bundling

The TUI auto-detects these directories/files next to the selected notebook:

- `data/`, `dataset/`, `datasets/`
- `src/`, `lib/`
- `utils.py`, `config.json`

You're prompted to include or exclude these, and can add more. The `.venv` directory can also be bundled (useful when the remote doesn't have your packages installed).

Everything is tar'd relative to the notebook's parent directory, so the archive unpacks with the same structure on the remote.

## Test payload

`test_payload.ipynb` is included as a minimal test notebook. It:
- Plots a sine wave and a damped oscillation (requires `numpy` + `matplotlib`)
- Prints the hostname, timestamp, and Python version
- Saves `test_output.png`

Use it to verify your dispatch pipeline end-to-end:

```bash
python cluster.py run test_payload.ipynb aws --wait --venv ""
```

## File layout

```
cluster.py              — the entire dispatcher (single file, no external deps)
test_payload.ipynb      — dummy notebook for testing
.cluster.json           — (optional) per-project config
~/.config/cluster/
├── jobs.db             — SQLite job tracking database
└── machines/           — (optional) custom machine definitions
    └── <name>.json
```
