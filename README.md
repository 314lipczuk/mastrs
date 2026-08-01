# optoerk

Single-cell optogenetics research on ERK signalling dynamics under light stimulation.

Current focus: LSTM seq2scalar models that predict CNR (cytoplasm-to-nucleus ratio,
an ERK biosensor readout) trajectories from stimulation features.

## Layout

The repository separates code that is **imported** from code that is **run**.

| Path | Role |
| --- | --- |
| `optoerk/` | The library. Everything importable. |
| `optoerk/core/` | Experiment tracking (`experiment.py`), device/path helpers (`utils.py`) |
| `optoerk/data/` | Loaders, preprocessing, synthetic data generation |
| `optoerk/models/` | Live seq2scal models; `mechanistic/` and `rl/` are dormant |
| `optoerk/eval/` | Prediction, evaluation, video rendering |
| `experiments/` | Runnable marimo entrypoints, launched via `submit.sh` |
| `experiments/archive/` | Frozen past experiments. Imports are stale by design. |
| `tests/` | pytest suite |
| `materials/` | Papers, figures, archived subsystems. Gitignored. |

## Commands

```sh
uv sync                                  # install (installs optoerk editable)
uv run pytest                            # run tests
uv run marimo edit experiments/<nb>.py   # interactive notebook
uv run python launcher.py                # dispatch SLURM jobs (--local to run here)
./sync.sh izb                            # push to cluster
./collect.sh                             # pull results back
```

## Conventions

- Results go to `optoerk.core.utils.results_write_path()`, never a hardcoded `results/`.
- Training runs persist via `ExperimentTracker`, checkpointing each epoch so jobs resume after a crash.
- New notebooks are marimo (`.py` with `@app.cell`), not Jupyter. Run `marimo check --fix` after editing.
- Data lives in `materials/`; address it via `optoerk.core.utils.materials_path(...)`, never a bare relative filename.

## Serving on the microscope

Run on the acquisition host, with the GPU:

```sh
uv run python -m optoerk.serving.app \
    --port 8080 --device cuda \
    --policy-file policies/policy_10fov_osc_p50.toml \
    --optortk-expr-value 0.5 \
    --stim-power 10 \
    --predict-log exp_v11.jsonl
```

- `--optortk-expr-value 0.5` short-circuits the optoRTK-expr channel to a fixed
  0.5 for every cell, so the run does not depend on a per-cell expression
  measurement. It implies `--override-optortk-expr`.
- `--policy-file` supersedes `--checkpoint` and `--target-cnr`: the checkpoint,
  objective and controller all come from the policy file, per FOV.
- `--predict-log` is the run's record. It is append-only, so give each run its own
  path — reusing one silently concatenates two experiments into one file.
