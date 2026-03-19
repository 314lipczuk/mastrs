import subprocess
import os
from dataclasses import dataclass, field


@dataclass
class Job:
    notebook: str
    name: str
    params: dict[str, str] = field(default_factory=dict)
    partition: str = "all"
    time: str = "04:00:00"
    mem: str = "4G"


def launch(job: Job, local: bool = False):
    """Run a marimo notebook either locally or via sbatch.

    The job name is used as both the SLURM job name and the experiment --name
    passed to the notebook via mo.cli_args().
    """
    assert os.path.exists(job.notebook), f"Notebook not found: {job.notebook}"

    # Build the marimo cli args: --key value pairs
    cli_args = []
    for k, v in job.params.items():
        cli_args += [f"--{k}", str(v)]

    if local:
        cmd = [
            "uv", "run", "marimo", "run", job.notebook,
            "--", "--name", job.name, *cli_args,
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
            f"--output=/mnt/imaging.data/ppilip/{job.name}_%j.log",
            f"--error=/mnt/imaging.data/ppilip/{job.name}_%j.log",
            "submit.sh",
            job.name,
            job.notebook,
            "--",
            *cli_args,
        ]
        print(f"[sbatch] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ── Experiment definitions ────────────────────────────────────────────────────

JOBS = [
    Job(
        notebook="experiments/VAE_single_timepoint_state_space.py",
        name="VAE_ss_H16-8_L2_b1",
        params=dict(hidden_dims="16,8", latent_dim="2", beta="1.0", dry_run="false"),
    ),
    Job(
        notebook="experiments/VAE_single_timepoint_state_space.py",
        name="VAE_ss_H16-8_L3_b1",
        params=dict(hidden_dims="16,8", latent_dim="3", beta="1.0", dry_run="false"),
    ),
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
