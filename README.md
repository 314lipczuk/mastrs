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
    --policy-file policies/policy_10fov_patterns.toml \
    --live-optortk-expr --optortk-cohort-frames 10 \
    --stim-power 10 \
    --predict-log exp_v12_patterns.jsonl
```

- `--policy-file` supersedes `--checkpoint` and `--target-cnr`: the checkpoint,
  objective and controller all come from the policy file, per FOV.
  `policy_10fov_patterns.toml` is the pattern-zoo run — one controller on every
  FOV, four different waveforms. **It ships with `placeholders_resolved = false`
  and the server will refuse to start** until `experiments/policy_preflight.py`
  has run and its numbers are pasted into the file's provenance block.
- `--live-optortk-expr` reconstructs the real per-cell optoRTK-expression rank
  online instead of feeding a constant: the **mCitrine** value from the payload's
  `ref_mean_intensity`, ranked against the session's cohort pooled across FOVs.
  `tests/test_serving_expression.py` pins it to reproduce
  `preprocessing.add_optortk_expression` exactly, at that feature's own float32
  precision.

  **It is NOT the C0 channels.** Whole-cell C0 (miRFP) was the surrogate the
  pipeline used before the real measurement was wired through; it reaches only
  Spearman 0.60–0.71 against mCitrine and misplaces 27–30% of cells across a
  high/low split.

  **This needed a faro change, made 2026-08-07.** Three things blocked the value
  from ever reaching `/predict`: it comes from `feature_extractor_ref`, not the
  stimulator's own extractor, so the column filter dropped it; the live pipeline
  computes the stim mask *before* feature extraction, so on the reference frame it
  is not on `tracks` yet; and `FeatureExtractorRef` writes per-row, so later frames
  are NaN. `InferenceServerStim._carry_forward` now carries each cell's last
  non-null value forward, which clears all three.

  **If nothing ever arrives** the cohort seals empty and the server aborts
  (`_check_optortk_coverage`) rather than feeding every cell the population mean,
  which would look exactly like a successful run. Cells the reference missed arrive
  as null, go neutral, and are counted in `n_optortk_values_seen`.

  **This changes the experimental condition**, so the pattern-zoo run ships as a
  PAIR: `policy_10fov_patterns.toml` with the flag, and
  `policy_10fov_patterns_control.toml` — byte-identical apart from its header —
  without it. Neither answers "do different cells need different steering" alone:
  with the flag, per-cell divergence is part covariate and part feedback; without
  it, there is no covariate and divergence is purely feedback-driven. The
  difference between the two runs is the measurement.

  **Expect about a third of the offline value.** Live coverage is ~37% of particle
  ids, not the 98.9% the model trained with — a reference acquisition only measures
  the ids alive when it runs, and track fragmentation issues far more ids than there
  are cells (median track lifetime 39 of 721 frames). Masking the trained model to
  that coverage gives test NLL 0.0272 against 0.0085 fully covered and 0.0390 with
  the constant. Unmeasured cells get the neutral value, which is what training gives
  them too (0.001 sigma apart).

  `--optortk-cohort-frames` must span the run's FIRST optocheck. The server is never
  called on frame 0 — faro starts `/predict` at timestep 1 — so the value only
  arrives because faro carries it forward from the frame that measured it.

  Without the flag the channel falls back to the training population mean — still
  one value for every cell, just not an operator-typed one. `--optortk-expr-value`
  remains for pinning a specific constant, and is refused alongside
  `--live-optortk-expr`. The regime is recorded per run under `optortk_expr` in the
  startup record, and per cell as `optortk_expr` / `optortk_live` in the predict
  log, so no run is ambiguous after the fact.
- `--predict-log` is the run's record. It is append-only, so give each run its own
  path — reusing one silently concatenates two experiments into one file.

## Quick reference — commands to run

### Benchmark on the microscope computer (do this first)

`soak.py` is the go/no-go: it drives the real HTTP server the way faro does —
real transport, real concurrency, real cadence — and reports `rho`, the fraction
of each acquisition cycle spent holding the inference lock. Every FOV serializes
on that one lock, so `rho >= 1` means the backlog grows without bound and faro
starts missing its 80 s `stim_mask` timeout hours into the run. No single-FOV
benchmark can see that.

```sh
uv run python -m optoerk.serving.soak \
    --policy-file policies/policy_10fov_patterns.toml --allow-placeholders \
    --live-optortk-expr --optortk-cohort-frames 10 \
    --device cuda --cycles 20 --cycle-seconds 60 \
    --from-log <previous_run>.jsonl --start-frame 600 \
    --predict-log soak_predict.jsonl
```

Read `rho`: **< 0.7** comfortable, **0.7–1.0** marginal (cell counts grow over 12 h,
so it drifts up — re-soak at the count you expect at hour 12), **>= 1.0** fail.

- `--n-fovs` is omitted on purpose: it defaults to the policy's field count, so a
  10-FOV policy is not benchmarked as 12 with two fields quietly on `[default]`.
- `--from-log --start-frame 600` replays the **crowded late** part of a previous
  run. Cell count is the biggest driver of inference time, and hour 12 is the
  question. Without `--from-log` it falls back to a flat synthetic count, which
  understates the tail. Drop both flags for a first smoke run.
- `--live-optortk-expr` benchmarks the configuration the real run uses. The
  payloads are augmented with a **fabricated** reference measurement so the
  expression cohort can seal — latency is representative, the ranks are not.
  Keep `--cycles` well above `--optortk-cohort-frames`, or most requests are
  measured before the cohort seals and never exercise the ranked path.
- `--allow-placeholders` is correct here and only here: latency depends on the
  ladder, horizon, kernel and sample count, none of which are the gated values.

Most faithful variant — point it at the already-running server instead of
starting one in-process, so it exercises the exact process faro will talk to.
`--predict-log` must then name the path **that server** was started with, or
there are no server-side timings and no `rho`:

```sh
uv run python -m optoerk.serving.soak --url http://localhost:8080 \
    --policy-file policies/policy_10fov_patterns.toml \
    --cycles 20 --cycle-seconds 60 --predict-log <the server's log>.jsonl
```

### The run itself

Both policies ship gated (`placeholders_resolved = false`) — the server refuses
to start until `experiments/policy_preflight.py` has run and its numbers are in
the provenance block.

```sh
# MAIN — live per-cell optoRTK expression ON
uv run python -m optoerk.serving.app --port 8080 --device cuda \
    --policy-file policies/policy_10fov_patterns.toml \
    --live-optortk-expr --optortk-cohort-frames 10 \
    --stim-power 10 --predict-log exp_patterns.jsonl

# CONTROL — identical policy, the one flag omitted
uv run python -m optoerk.serving.app --port 8080 --device cuda \
    --policy-file policies/policy_10fov_patterns_control.toml \
    --optortk-cohort-frames 10 \
    --stim-power 10 --predict-log exp_patterns_control.jsonl
```

`--optortk-cohort-frames` must span the run's **first** optocheck: nobody can be
ranked before the population they are ranked against exists. Acquire 700 min per
run, and give each its own `--predict-log` (the file is append-only).

