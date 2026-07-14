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

Exposures are in **ms, clamped to `[0, 3000]`**; `0` = do not stimulate.
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
    --checkpoint results/<bundle_dir> --device cuda --stim-power 100

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

## Control law (PLACEHOLDER — needs the real objective)

The model predicts CNR given future fluence; faro needs a commanded dose. No
controller existed in `masters`, so `RealModelEngine.decide` implements a simple
**model-predictive search**: for a grid of candidate exposures it rolls the
decoder forward `control_horizon` steps and picks the exposure whose predicted
CNR best tracks `target_cnr` (squared-error over the horizon). `target_cnr`,
`control_horizon`, and `n_candidates` are configurable. **Replace this objective
with the real experimental goal.**

## Fluence ↔ milliseconds

The model's `u_t` is fluence `mJ/cm2 = irradiance(mW/cm2) * exposure_ms * 1e-3`
(from `preprocessing.calc_power`), where irradiance comes from the instrument
power calibration at a given LED power `%`. `calibration.FluenceCalibration`
reuses the exact same `CALIBRATIONS` table. **faro does not send the LED power,
so a fixed `stim_power_pct` is assumed** (`--stim-power`, default 100 %). Set it
to the power faro actually drives the DMD at.

## Deltas to the faro contract (things to confirm)

1. **CNR normalization:** the model's cnr channel is `cnr_median_norm` =
   per-cell median CNR ÷ its baseline (median of first `baseline_frames`
   frames). faro sends raw `cnr`; the server prefers `cnr_median` if present and
   baseline-normalizes online. Cells must be tracked from experiment start for a
   correct baseline; mid-experiment births use a provisional baseline. Sending
   `cnr_median` is preferred over `cnr`.
2. **LED power:** not in the payload; assumed fixed (see above).
3. **`fov_density` / `n_cells_200px`:** derived server-side from all cells'
   `x, y` in the payload (replicating `preprocessing.add_crowding_features`), so
   faro need not send them — but it must send `x, y` for every cell.
4. **`parent_particle`:** optional; if present, a daughter's state is seeded
   from its mother.
