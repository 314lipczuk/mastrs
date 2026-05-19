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
    # "marimo": training notebook, creates a fresh exp_dir, --name + --results-dir injected.
    # "script": plain Python utility; runs as `uv run python notebook --key value` directly
    #          (no exp_dir). On cluster, dispatched via `sbatch --wrap`; slurm.log lands next
    #          to the operated-on result dir if `result-path` was in params, else under ./logs/.
    kind: str = "marimo"

def launch(job: Job, local: bool = False):
    """Run a job either locally or via sbatch.

    For ``kind="marimo"`` the job name is used as both the SLURM job name and
    the experiment ``--name`` passed to the notebook via ``mo.cli_args()``.
    For ``kind="script"`` the notebook path is just a Python entrypoint;
    params become ``--key value`` flags and no exp_dir is created.
    """
    assert os.path.exists(job.notebook), f"Notebook not found: {job.notebook}"

    cli_args = []
    for k, v in job.params.items():
        cli_args += [f"--{k}", str(v)]

    if job.kind == "script":
        if local:
            cmd = ["uv", "run", "python", job.notebook, *cli_args]
            print(f"[local-script] {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return

        # Cluster: wrap with sbatch --wrap; log next to the operated-on result
        # dir if one was passed via --result-path, else a per-run logs/ file.
        ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        result_path = job.params.get("result-path") or job.params.get("result_path")
        if result_path:
            log_path = f"{result_path}/{job.name}_{ts}.slurm.log"
        else:
            os.makedirs("logs", exist_ok=True)
            log_path = f"logs/{job.name}_{ts}.slurm.log"

        # Quote each argv element so it survives the shell layer inside --wrap.
        import shlex
        wrap_cmd = " ".join(
            shlex.quote(p) for p in ["uv", "run", "python", job.notebook, *cli_args]
        )
        # sbatch --wrap runs under /bin/sh which doesn't grok `set -o pipefail`;
        # invoke bash explicitly.
        inner = (
            'set -euo pipefail; '
            'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; '
            'source "$HOME/.bashrc" 2>/dev/null || source "$HOME/.profile" 2>/dev/null || true; '
            'echo "── GPU diag ──"; '
            'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"; '
            'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: N/A"; '
            'uv sync; '
            'uv run python -c "import torch; print(\'torch\', torch.__version__, \'cuda_available=\', torch.cuda.is_available(), \'cuda_version=\', torch.version.cuda)"; '
            'echo "──────────────"; '
            f'PYTHONUNBUFFERED=1 {wrap_cmd}'
        )
        wrap = f"bash -c {shlex.quote(inner)}"
        cmd = [
            "sbatch",
            f"--job-name={job.name}",
            f"--partition={job.partition}",
            f"--time={job.time}",
            f"--mem={job.mem}",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            f"--output={log_path}",
            f"--error={log_path}",
            f"--wrap={wrap}",
        ]
        print(f"[sbatch-script] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return

    ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_dir = f"{results_write_path()}/{job.name}_{ts}"
    os.makedirs(exp_dir, exist_ok=True)

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

# Shared hyperparams for the 6-way sweep (real vs real+BO × image/no-image × minfeats/fullfeats).
# history_len=25, future_len=10, batch_size=900.
_SWEEP = dict(
    dry_run="false",
    history_len="25",
    future_len="10",
    batch_size="900",
)

# Cluster path to the most recent extracted patches (used by image-enabled jobs).
_PATCHES_H5 = (
    "/mnt/imaging.data/ppilip/results/models/"
    "extract_patches_all_40_2026-05-06_11.50.00/patches.h5"
)

JOBS = [
    # bo + no image + minfeats: history_len=50, future_len=10, epochs=900,
    # TF annealing 0 → 120 (hold=0, anneal_frac=120/900).
    Job(
        notebook="experiments/lstm_seq2scal_mdn_minfeats_image_ewma.py",
        name="lstm_mdn_bo_noimage_minfeats_h50",
        params=dict(
            dry_run="false",
            history_len="50",
            future_len="10",
            batch_size="900",
            epochs="900",
            tf_anneal_frac=str(120.0 / 900.0),
            tf_hold_frac="0.0",
            source="real_plus_bo",
            use_images="false",
        ),
        mem="32G",
    ),

    # Job(
    #     notebook="cell_video.py",
    #     name="cell_video_plus_bo_noimages",
    #     kind="script",
    #     params=dict(
    #         **{
    #             "result-path": "/mnt/imaging.data/ppilip/results/models/lstm_mdn_bo_noimage_fullfeats_2026-05-19_14.37.27",
    #         },
    #         **{"fps": "4", "stride": "1", "display-history": "10"},
    #     ),
    # ),

    # # 1. real + image + minfeats (EWMA minfeats variant)
    # Job(
    #     notebook="experiments/lstm_seq2scal_mdn_minfeats_image_ewma.py",
    #     name="lstm_mdn_real_image_minfeats",
    #     params=dict(
    #         **_SWEEP,
    #         **{
    #             "source": "real",
    #             "use_images": "true",
    #             "patches-h5": _PATCHES_H5,
    #         },
    #     ),
    #     mem="32G",
    # ),

    # 1a. real + image + full features
    #Job(
    #    notebook="experiments/lstm_seq2scal_mdn_image.py",
    #    name="lstm_mdn_real_image_fullfeats",
    #    params=dict(
    #        **_SWEEP,
    #        **{
    #            "source": "real",
    #            "use_images": "true",
    #            "patches-h5": _PATCHES_H5,
    #            "tf_anneal_frac": str(120.0 / 900.0),
    #        },
    #    ),
    #    mem="32G",
    #),

    # 2. real + no image + minfeats
    # Job(
    #     notebook="experiments/lstm_seq2scal_mdn_minfeats_image_ewma.py",
    #     name="lstm_mdn_real_noimage_minfeats",
    #     params=dict(
    #         **_SWEEP,
    #         **{
    #             "source": "real",
    #             "use_images": "false",
    #         },
    #     ),
    #     mem="32G",
    # ),

    # 2a. real + no image + full features
    #Job(
    #    notebook="experiments/lstm_seq2scal_mdn.py",
    #    name="lstm_mdn_real_noimage_fullfeats",
    #    params=dict(
    #        **_SWEEP,
    #        source="real",
    #        tf_anneal_frac=str(120.0 / 900.0),
    #    ),
    #    mem="32G",
    #),

    # 3. real_plus_bo + no image + minfeats
    # Job(
        # notebook="experiments/lstm_seq2scal_mdn_minfeats_image_ewma.py",
        # name="lstm_mdn_bo_noimage_minfeats",
        # params=dict(
            # **_SWEEP,
            # **{
                # "source": "real_plus_bo",
                # "use_images": "false",
            # },
        # ),
        # mem="32G",
    # ),

    # 3a. real_plus_bo + no image + full features
    #Job(
    #    notebook="experiments/lstm_seq2scal_mdn.py",
    #    name="lstm_mdn_bo_noimage_fullfeats",
    #    params=dict(
    #        **_SWEEP,
    #        source="real_plus_bo",
    #        tf_anneal_frac=str(120.0 / 900.0),
    #    ),
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
