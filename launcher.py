import re
import subprocess
import os
from dataclasses import dataclass, field
from datetime import datetime

from optoerk.core.utils import results_write_path


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
    # Optional: SLURM job id(s) to gate this submission on (`--dependency=afterok:<ids>`).
    dependency: str | None = None
    # Optional: for marimo training jobs, names of sub-experiment directories
    # for which to launch a cell_video.py job with afterok dependency on this job.
    # Each entry maps subdir name → notebook module to pass as --experiment-module.
    video_subexps: dict[str, str] = field(default_factory=dict)
    # When True (script kind), launcher pre-creates the timestamped exp_dir under
    # results_write_path() and passes `--results-dir <path>` to the script, with
    # slurm.log routed into the same dir. Same contract as marimo jobs — keeps
    # every artifact (bundle, slurm log, videos) under one experiment directory.
    creates_exp_dir: bool = False


_SBATCH_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def _parse_jobid(stdout: str) -> str | None:
    m = _SBATCH_JOBID_RE.search(stdout or "")
    return m.group(1) if m else None


def launch(job: Job, local: bool = False) -> tuple[str | None, str | None]:
    """Run a job either locally or via sbatch.

    Returns ``(jobid, exp_dir)``. ``jobid`` is the SLURM job id (str) for cluster
    submissions, ``None`` locally. ``exp_dir`` is set only for ``kind="marimo"``.

    For ``kind="marimo"`` the job name is used as both the SLURM job name and
    the experiment ``--name`` passed to the notebook via ``mo.cli_args()``.
    For ``kind="script"`` the notebook path is just a Python entrypoint;
    params become ``--key value`` flags and no exp_dir is created.
    """
    assert os.path.exists(job.notebook), f"Notebook not found: {job.notebook}"

    cli_args = []
    for k, v in job.params.items():
        cli_args += [f"--{k}", str(v)]

    dep_flag = [f"--dependency=afterok:{job.dependency}"] if job.dependency else []

    if job.kind == "script":
        # Optionally pre-create the experiment dir on NFS and inject it as
        # --results-dir so slurm.log + bundle + figures all live together.
        script_exp_dir: str | None = None
        if job.creates_exp_dir:
            ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            script_exp_dir = f"{results_write_path()}/{job.name}_{ts}"
            os.makedirs(script_exp_dir, exist_ok=True)
            cli_args = ["--results-dir", script_exp_dir, *cli_args]

        if local:
            cmd = ["uv", "run", "python", job.notebook, *cli_args]
            print(f"[local-script] {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return None, script_exp_dir

        # Cluster: wrap with sbatch --wrap. Slurm log goes (in order of
        # preference): pre-created exp_dir from `creates_exp_dir`; the
        # `result-path` operated-on dir; else a per-run logs/ file.
        ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        result_path = job.params.get("result-path") or job.params.get("result_path")
        if script_exp_dir:
            log_path = f"{script_exp_dir}/slurm.log"
        elif result_path:
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
            *dep_flag,
            f"--wrap={wrap}",
        ]
        print(f"[sbatch-script] {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(proc.stdout, end="")
        return _parse_jobid(proc.stdout), script_exp_dir

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
        return None, exp_dir

    cmd = [
        "sbatch",
        f"--job-name={job.name}",
        f"--partition={job.partition}",
        f"--time={job.time}",
        f"--mem={job.mem}",
        f"--output={exp_dir}/slurm.log",
        f"--error={exp_dir}/slurm.log",
        *dep_flag,
        "submit.sh",
        job.name,
        job.notebook,
        exp_dir,
        *cli_args,
    ]
    print(f"[sbatch] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(proc.stdout, end="")
    return _parse_jobid(proc.stdout), exp_dir


def launch_with_videos(job: Job, local: bool = False) -> None:
    """Submit ``job`` and, for each entry in ``job.video_subexps``, submit a
    cell_video.py script job with ``--dependency=afterok:<train_jobid>``
    pointing at ``<exp_dir>/<subexp>/``.

    Locally (no sbatch), runs videos sequentially after training finishes.
    """
    jobid, exp_dir = launch(job, local=local)
    if not job.video_subexps:
        return
    assert exp_dir is not None, "video_subexps requires kind='marimo'"
    for subexp, exp_module in job.video_subexps.items():
        result_path = f"{exp_dir}/{subexp}" if subexp else exp_dir
        suffix = f"_{subexp}" if subexp else ""
        video_job = Job(
            notebook="cell_video.py",
            name=f"{job.name}__video{suffix}",
            kind="script",
            params={
                "result-path": result_path,
                "experiment-module": exp_module,
                "fps": "4",
                "stride": "1",
                "display-history": "10",
            },
            mem="16G",
            time="08:00:00",
            dependency=jobid,
        )
        launch(video_job, local=local)


# ── Experiment definitions ────────────────────────────────────────────────────

_BASE = dict(
    dry_run="false",
    history_len="25",
    future_len="10",
    batch_size="900",
    epochs="900",
)

# Pair each notebook with the (subdir → experiment_module) mapping cell_video
# needs to reconstruct the model class. lstm_seq2seq.py saves subexps "ar" and
# "baseline"; lstm_seq2scal.py saves "lstm" and "mlp".
_SEQ2SEQ_VIDEOS = {
    "ar": "experiments.lstm_seq2seq",
    "baseline": "experiments.lstm_seq2seq",
}
_SEQ2SCAL_VIDEOS = {
    "lstm": "experiments.lstm_seq2scal",
    "mlp": "experiments.lstm_seq2scal",
}


# ── Baseline-prepend experiment (long context H=60, real_plus_bo) ─────────────
# Tests: (1) does prepending block-bootstrapped baseline let models predict the
# stimulation onset that the history window otherwise eats; (2) MDN vs point on
# TCN; (3) absolute- vs delta-output parameterisation on the LSTM. Each model is
# paired ±prepend; eval reports interior- vs onset-window metrics for a clean read.

# Shared LSTM (seq2scal + MDN + FiLM-output) config; only output-param + prepend vary.
def _lstm_params(prepend: bool) -> dict:
    return dict(
        dry_run="false",
        history_len="60",
        future_len="10",
        batch_size="900",
        epochs="900",
        tf_anneal_frac=str(120.0 / 900.0),
        tf_hold_frac="0.0",
        source="real_plus_bo",
        head_type="mdn",
        film="output",
        use_stratified_sampler="true",
        n_strata="4",
        sampler_type="quartile_weighted",
        quartile_weights="[0.1,0.2,0.3,0.4]",
        prepend_baseline="true" if prepend else "false",
        # Option D: bo_osc cells only for train/val/test_indist; OOD conds
        # (Sustained, ramp1, 3-2-1minIntervals) reserved per-condition for test.
        split_regime="condition_held_out",
    )


# ── Full-history (long-gap) model ─────────────────────────────────────────────
# Encodes the entire past (packed LSTM, no fixed H), minimal raw features
# [cnr, fluence, fov_density, n_cells_200px], absolute-CNR MDN. Self-concat-
# random-break augmentation teaches the two-experiment / inter-experiment-break
# structure. Eval with the memory ladder (cross_stitch_responder.py).
_HISTORY = dict(
    dry_run="false",
    source="real_plus_bo",
    future_len="10",
    epochs="300",
    batch_size="256",
    head_type="mdn",
    film="output",
    p_concat="0.5",
    break_max="60",
    use_stratified_sampler="true",
    n_strata="4",
)

# Multi-length variant: sample horizon F ~ U[3, future_len] per batch, with a
# learnable per-step sigma bias for horizon calibration. Trains in parallel with
# the fixed-F baseline (separate sbatch job, no dependency) so the two are
# directly comparable.
_HISTORY_MULTILEN = dict(_HISTORY, future_len_min="3", sigma_step_bias="true")


JOBS = [
    # --- Full-history long-gap model (train + chained held-out video) ----------
    Job(notebook="experiments/lstm_seq2scal_history.py", name="seq2scal_history",
        mem="48G", time="24:00:00",
        params=_HISTORY,
        video_subexps={"": "experiments.lstm_seq2scal_history"}),
    Job(notebook="experiments/lstm_seq2scal_history.py", name="seq2scal_history_multilen",
        mem="48G", time="24:00:00",
        params=_HISTORY_MULTILEN,
        video_subexps={"": "experiments.lstm_seq2scal_history"}),

    # --- TCN: head (point|mdn) × prepend (off|on) -----------------------------
    #Job(notebook="TCN/train.py", name="tcn_point_off", kind="script", mem="32G", time="24:00:00",
    #    creates_exp_dir=True,
    #    params={"config": "TCN/config.yaml", "name": "tcn_point_off", "head": "point", "baseline-prepend": "false"}),
    #Job(notebook="TCN/train.py", name="tcn_point_on", kind="script", mem="32G", time="24:00:00",
    #    creates_exp_dir=True,
    #    params={"config": "TCN/config.yaml", "name": "tcn_point_on", "head": "point", "baseline-prepend": "true"}),
    #Job(notebook="TCN/train.py", name="tcn_mdn_off", kind="script", mem="32G", time="24:00:00",
    #    creates_exp_dir=True,
    #    params={"config": "TCN/config.yaml", "name": "tcn_mdn_off", "head": "mdn", "baseline-prepend": "false"}),
    #Job(notebook="TCN/train.py", name="tcn_mdn_on", kind="script", mem="32G", time="24:00:00",
    #    creates_exp_dir=True,
    #    params={"config": "TCN/config.yaml", "name": "tcn_mdn_on", "head": "mdn", "baseline-prepend": "true"}),

    # --- LSTM seq2scal + FiLM: output-param (delta|abs) × prepend (off|on) -----
    # video_subexps chains a held-out cell_video render after each run ("" = the
    # bundle lives at exp_dir itself, scaffold.save_bundle has no sub-experiments).
    #Job(notebook="experiments/lstm_seq2scal_variant.py", name="lstm_delta_off", mem="32G", time="48:00:00",
    #    params=_lstm_params(prepend=False),
    #    video_subexps={"": "experiments.lstm_seq2scal_variant"}),
    #Job(notebook="experiments/lstm_seq2scal_variant.py", name="lstm_delta_on", mem="32G", time="48:00:00",
    #    params=_lstm_params(prepend=True),
    #    video_subexps={"": "experiments.lstm_seq2scal_variant"}),
    #Job(notebook="experiments/lstm_seq2scal_abs.py", name="lstm_abs_off", mem="32G", time="48:00:00",
    #    params=_lstm_params(prepend=False),
    #    video_subexps={"": "experiments.lstm_seq2scal_abs"}),
    #Job(notebook="experiments/lstm_seq2scal_abs.py", name="lstm_abs_on", mem="32G", time="48:00:00",
    #    params=_lstm_params(prepend=True),
    #    video_subexps={"": "experiments.lstm_seq2scal_abs"}),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally instead of sbatch")
    parser.add_argument("--dry-run", action="store_true", help="Override dry_run=true for all jobs")
    parser.add_argument("--only", default=None,
                        help="Comma-separated job names to submit (subset of JOBS); default = all.")
    args = parser.parse_args()

    jobs = JOBS
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        jobs = [j for j in JOBS if j.name in wanted]
        missing = wanted - {j.name for j in jobs}
        if missing:
            parser.error(f"--only: unknown job name(s): {sorted(missing)}; "
                         f"have: {sorted(j.name for j in JOBS)}")
        print(f"[launcher] --only filter -> {len(jobs)} job(s): {[j.name for j in jobs]}")

    for job in jobs:
        if args.dry_run:
            job.params["dry_run"] = "true"
        launch_with_videos(job, local=args.local)
