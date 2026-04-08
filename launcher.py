import subprocess
import os
from dataclasses import dataclass, field
from datetime import datetime

from utils import results_write_path


@dataclass
class Job:
    notebook: str
    name: str
    params: dict[str, str] = field(default_factory=dict)
    partition: str = "all"
    time: str = "24:00:00"
    mem: str = "16G"


def launch(job: Job, local: bool = False):
    """Run a marimo notebook either locally or via sbatch.

    The job name is used as both the SLURM job name and the experiment --name
    passed to the notebook via mo.cli_args().
    """
    assert os.path.exists(job.notebook), f"Notebook not found: {job.notebook}"

    ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_dir = f"{results_write_path()}/{job.name}_{ts}"
    os.makedirs(exp_dir, exist_ok=True)

    # Build the marimo cli args: --key value pairs
    cli_args = []
    for k, v in job.params.items():
        cli_args += [f"--{k}", str(v)]

    if local:
        cmd = [
            "uv", "run", "marimo", "run", job.notebook,
            "--", "--name", job.name, "--results-dir", exp_dir, *cli_args,
        ]
        print(f"[local] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        cmd = [
            "sbatch",
            f"--job-name={job.name}",
            f"--partition={job.partition}",
            f"--time={job.time}",
            f"--mem={job.mem}",
            f"--output={exp_dir}/slurm.log",
            f"--error={exp_dir}/slurm.log",
            "submit.sh",
            job.name,
            job.notebook,
            exp_dir,
            *cli_args,
        ]
        print(f"[sbatch] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ── Experiment definitions ────────────────────────────────────────────────────

JOBS = [
    #Job(
    #    notebook="experiments/gridsearch_lstm_seq2seq.py",
    #    name="lstm_gridsearch_synthetic",
    #    params=dict(
    #        source="synthetic",
    #        dry_run="false",
    #        history_len="10,20,30",
    #        future_len="2,5,10",
    #        lr="1e-3,5e-4",
    #        num_layers="1,2",
    #        hidden_dim="16,32,64",
    #        patience="30,50",
    #    ),
    #    time="24:00:00",
    #    mem="32G",
    #),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally instead of sbatch")
    parser.add_argument("--dry-run", action="store_true", help="Override dry_run=true for all jobs")
    args = parser.parse_args()

    for job in JOBS:
        if args.dry_run:
            job.params["dry_run"] = "true"
        launch(job, local=args.local)
