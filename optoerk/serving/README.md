# optoerk.serving — faro optogenetic inference server

Hosts a trained `Seq2ScalarHistory` model and serves **per-cell stimulation
exposures (milliseconds)** to the `faro` microscopy control system. Implements
the contract in `inference_server_spec.md`.

The server treats the model's encoder as an **online per-cell recurrent state**:
it advances each cell's LSTM by exactly one step per frame using the carried
`(h, c)`, which for a causal `nn.LSTM` is numerically identical to re-encoding
the full history every call (verified to ~1e-7). It never replays history.

## Endpoints

| Method | Path       | Body                                   | Returns |
|--------|------------|----------------------------------------|---------|
| POST   | `/predict` | `{fov, timestep, time, cells:[...]}`   | `{fov, timestep, exposures:{"<particle>": ms}}` |
| GET    | `/health`  | –                                      | `{status, model_loaded}` |
| POST   | `/reset`   | `{}` (all) or `{"fov": N}`             | `{status, reset}` |
| GET    | `/info`    | –                                      | model / calibration / units metadata |

Exposures are in **ms, clamped to `[min_exposure_ms, max_exposure_ms]`**
(default `[0, 800]`); `0` = do not stimulate.
Retries with the same `(fov, timestep)` return the cached response without
advancing any state.

## Running

Zero extra dependencies — pure stdlib HTTP (`http.server`) + torch. From the
`masters` repo root:

```bash
# Stub policy (no checkpoint) — runnable immediately, for faro integration:
uv run python -m optoerk.serving.app --host 0.0.0.0 --port 8080

# Real trained model:
uv run python -m optoerk.serving.app --host 0.0.0.0 --port 8080 \
    --checkpoint results/<bundle_dir> --device cuda --stim-power 10

# Smoke test (in-process; set the env var to test the real model):
uv run python -m optoerk.serving.smoke_test
OPTOERK_SERVE_CHECKPOINT_DIR=results/<bundle_dir> uv run python -m optoerk.serving.smoke_test
```

All `ServerConfig` fields are also settable via `OPTOERK_SERVE_<FIELD>` env vars
(e.g. `OPTOERK_SERVE_PORT`, `OPTOERK_SERVE_TARGET_CNR`, `OPTOERK_SERVE_CHECKPOINT_DIR`).

### Cluster GPU node + SSH tunnel

On the GPU node:

```bash
uv run python -m optoerk.serving.app --port 8080 --checkpoint results/<bundle_dir> --device cuda
```

From the faro host, tunnel to it (server bound on the node, reached via login host):

```bash
ssh -N -L 8080:<gpu_node_hostname>:8080 <user>@<cluster_login_host>
# faro then talks to http://localhost:8080
```

The model is loaded once at startup; `/health` reports `model_loaded` only after
loading succeeds. If the checkpoint fails to load, the server logs the error and
falls back to the stub policy (so it still answers faro).

## How the model plugs in

- **Loading:** `optoerk.core.experiment.load_experiment(dir).reconstruct_model()`
  (bundle format: `bundle.pt` with `model_type` + `model_config` + `model_state_dict`).
- **Norm stats:** taken from `model.cfg.norm_mean/std` if present, else from
  `optoerk/data/history_norm_stats.json`. Channel order `[cnr, u_t, fov_density, n_cells_200px]`.
- **Seam:** `runtime.load_engine(cfg)` returns a `RealModelEngine` or `StubEngine`,
  both exposing `decide(frames) -> [exposure_ms]`.

## Goals, controllers and per-FOV policies

The model predicts CNR *given* a dose; faro needs a commanded dose. Three pieces
split that job, each swappable:

| piece | module | question |
|---|---|---|
| **Objective** | `objectives.py` | what are we aiming for? |
| **Controller** | `control.py` | how do we search for the dose that gets there? |
| **Policy** | `policy.py` | which of each, for which FOV? |

### Objectives — the goal *is* a cost function

`Objective.cost(pred_cnr, ctx) -> (N, M)` scores every candidate plan for every
cell; the controller takes the argmin. `pred_cnr` arrives in **absolute CNR
units** (already denormalized), in the loaded checkpoint's `cnr_mode` convention,
so objectives are written in human-readable CNR and never touch z-score stats.

`TargetTrajectory(target_fn, gate_fn)` covers the common case: a setpoint that may
vary over time *and over the forecast horizon*, plus a predicate deciding whether
a cell may be stimulated at all. A gate is deliberately **not** folded into the
cost — an enormous cost still leaves the controller picking a least-bad nonzero
dose, whereas `allow_stim` forces exactly 0 ms.

Built-ins, nameable from a policy file: `hold` (fixed setpoint), `schedule`
(piecewise setpoint over time, evaluated per horizon step so the controller sees
a step change coming), `gated` (`hold` plus predicates on `x`, `y`,
`n_cells_200px`, `timestep`, `n_frames_seen`). Register your own with
`objectives.register`.

### Controllers

- `constant_dose` — scores a **constant** dose held over the horizon. Cheap; the
  historical behaviour; kept as the A/B baseline. Not MPC: it cannot express "a
  pulse now, then nothing".
- `sequence_mpc` — real receding-horizon MPC. Optimizes a dose *sequence*
  `u[0..H-1]` by the cross-entropy method over the discrete DMD level set, applies
  **only `u[0]`**, and re-plans next frame. The constant-dose plans are injected
  into every CEM iteration, so MPC is provably never worse than `constant_dose`.

Two things to know about `sequence_mpc`: each cell samples its own plans, so a low
`n_samples` gives *identical cells different doses* (measured: split at S=128,
stable by S=512 — hence the generous default); and the horizon is hard-capped at
the checkpoint's `future_len`, because rolling further is untrained and indexes
past `sigma_step_bias_param`.

### Per-FOV policies

```toml
[default]
checkpoint = "results/seq2scal_history_optortk_multilen_2026-07-14_09.48.21"
objective  = { type = "hold", target_cnr = 1.4 }
controller = { type = "sequence_mpc", n_samples = 512 }

[fov.1]
objective = { type = "hold", target_cnr = 2.0 }

[fov.2]
checkpoint = "results/some_other_model"
objective  = { type = "gated", target_cnr = 1.8, after_t = 10, x_gt = 512,
               max_neighbours_200px = 5 }
```

`--policy-file policies.toml` (JSON works too). A FOV inherits every field it does
not set; `objective`/`controller` are replaced wholesale, never deep-merged.
Models are cached by `(checkpoint, device)`, so N FOVs on one checkpoint load and
warm up **one** model. A FOV whose policy fails to build degrades to the stub for
that FOV alone. Every resolved policy is written into the log's `startup` record
and echoed by `/info`, so a run's log is a complete record of what it ran.

With no policy file, behaviour is exactly as before: one checkpoint, a `hold` at
`target_cnr`, searched by `constant_dose`.

## Replay and benchmarking

`python -m optoerk.serving.bench` sweeps `decide()` wall time over
horizon × candidates × cells × controller. It can build an **untrained** model at
any `future_len`, so "can we afford F=30?" is answerable before paying for the
retrain. Measured on CPU at 208 cells: `constant_dose` H=30 is 61 ms,
`sequence_mpc@128` H=30 is 4.2 s — against a **60 s** frame budget. Horizon is
limited by what the model was trained for, not by compute.

`optoerk.serving.replay` re-drives the service with a recorded run's frames, and
`experiments/replay_serving_run.py` wraps it in a notebook. Three modes that must
not be conflated: **faithful** (same policy; a regression gate), **counterfactual**
(different policy on the recorded CNR stream — open-loop, so it measures
disagreement, *not* whether the new policy would track better), and
`simulate_closed_loop` (the model stands in for the cells; compares tracking, but
against the model's own beliefs).

`crowding_match_frac` in the replay summary must be 1.0 — it proves the track
positions were joined back in. Without it, a missing `tracks/` dir silently feeds
the model different `n_cells_200px` while the summary still looks healthy.

A faithful replay is exact on the same device. Replaying a **CUDA** run on **CPU**
lands at ~0.998: float differences flip the occasional argmin near a decision
boundary and drift compounds through the per-cell encoder state.

## Performance — measured, before you optimize

From the 2026-07-16 v5 run (2880 predicts, 4 FOVs, CUDA, ~208 cells/predict):
p50/p99/max `infer_s` = 0.07/0.12/0.15 s, `lock_wait_s` = 0.00 throughout, and a
per-FOV request cadence of a flat 60.0 s.

**The server uses ~0.15 s of a 60 s budget and never contends on its lock.** If
frames are being dropped, it is not inference — a 400× speedup would save 0.15 s
per minute. Look upstream (faro acquisition, segmentation/tracking, `stim_mask`
handling, the network); the notebook's cadence panel is the tool for that. This is
also the headroom that makes `sequence_mpc` affordable.

## Fluence ↔ milliseconds

The model's `u_t` is fluence `mJ/cm2 = irradiance(mW/cm2) * exposure_ms * 1e-3`
(from `preprocessing.calc_power`), where irradiance comes from the instrument
power calibration at a given LED power `%`. `calibration.FluenceCalibration`
reuses the exact same `CALIBRATIONS` table. **faro does not send the LED power,
so a fixed `stim_power_pct` is assumed** (`--stim-power`, default 10 %). Set it
to the power faro actually drives the DMD at.

## Deltas to the faro contract (things to confirm)

1. **CNR — one scalar per cell, and which convention:** faro sends a single
   **`cnr_median`** per cell per frame — the ratio `median_intensity_C1_ring /
   median_intensity_C1_nuc` (channel C1 only; the per-region pixel reduction
   happens upstream in segmentation, so this is already a scalar, *not* pixels).
   `extract_raw_cnr` prefers the `cnr_median` field, else reconstructs it from the
   two C1 medians, and falls back to plain `cnr` (the *mean*-ratio) only as a last
   resort — the model was trained on the median ratio, so **send `cnr_median`.**

   What the server does with that scalar depends on the checkpoint's **`cnr_mode`**
   (printed in the startup banner). Note "raw" is overloaded: the scalar faro sends
   is "raw" in the sense of *not yet baseline-divided*, and separately a `cnr_mode
   ="raw"` **model** is one trained on that absolute scalar.

   - **`cnr_mode="norm"`** — the model's channel is `cnr_median_norm` = `cnr_median
     ÷ per-cell baseline` (baseline = median of the cell's first `baseline_frames`
     frames), so the server reconstructs the baseline **online**. This needs the
     cell tracked **from experiment start**; a mid-experiment birth never had its
     resting window observed and gets a provisional baseline (the `dark_baseline`
     field-mode machinery mitigates this by seeding births from the FOV field
     baseline, but it stays an approximation). This is the fragile part of the
     contract — a bias appears whenever faro's track ids aren't stable from t=0.

   - **`cnr_mode="raw"`** — the model consumes the absolute `cnr_median` directly:
     **no online normalization, no dark window, no baseline.** The entire
     tracking-from-start / provisional-baseline caveat above **does not apply.**
     The tradeoff moves into the model: it must generalize across absolute CNR
     levels that vary with cell line and sensor expression, rather than every cell
     starting at 1.0 (the OOD-scale concern in `NIESEN_TOCHECK.md`). `baseline_frames`,
     `dark_baseline` and `baseline_mode` are all ignored in this mode.

   Either way faro sends the same field (`cnr_median`); the checkpoint decides how
   it is interpreted, so serving a raw model with a norm checkpoint's target (or
   vice versa) is the mismatch the startup banner exists to make obvious.
2. **LED power:** not in the payload; assumed fixed (see above).
3. **`fov_density` / `n_cells_200px`:** derived server-side from all cells'
   `x, y` in the payload (replicating `preprocessing.add_crowding_features`), so
   faro need not send them — but it must send `x, y` for every cell.
4. **`parent_particle`:** optional; if present, a daughter's state is seeded
   from its mother.
