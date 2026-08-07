"""Configuration for the faro optogenetic inference server.

All fields are overridable via ``OPTOERK_SERVE_<UPPER_FIELD>`` environment
variables (see :meth:`ServerConfig.from_env`), so the same code runs unchanged
on a laptop (stub model) and on the cluster GPU node (real checkpoint).

Several fields encode *placeholders / assumptions* that need the user's real
decisions — they are called out in the README and in the field comments below:
  * ``target_cnr``            — the control objective (what CNR to drive cells to).
  * ``stim_power_pct``        — the LED power faro will actually use; sets the
                               fluence(mJ/cm2) <-> exposure(ms) conversion.
  * ``baseline_frames``       — window used to online-normalize CNR to the
                               ``cnr_median_norm`` the model was trained on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _env(name: str) -> str | None:
    return os.environ.get(f"OPTOERK_SERVE_{name.upper()}")


@dataclass
class ServerConfig:
    # --- transport --------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8080

    # --- model loading ----------------------------------------------------
    # Path to a saved experiment bundle directory (contains ``bundle.pt``),
    # loadable via ``optoerk.core.experiment.load_experiment``. When None (or
    # the load fails), the server falls back to a deterministic STUB policy so
    # faro can integrate end-to-end before a real checkpoint exists.
    checkpoint_dir: str | None = None
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    # --- per-FOV policies --------------------------------------------------
    # Path to a .toml / .json policy file giving a default policy (checkpoint +
    # objective + controller) plus per-FOV overrides. See optoerk.serving.policy.
    # When None, every FOV shares one policy built from the fields below:
    # ``checkpoint_dir``, a ``hold`` objective at ``target_cnr``, searched by
    # ``constant_dose``. That is exactly the pre-policy-file behaviour.
    policy_file: str | None = None

    # --- fluence <-> milliseconds conversion ------------------------------
    # The model's ``u_t`` channel is fluence in mJ/cm2 =
    #   irradiance(mW/cm2, from stim_power) * exposure_ms * 1e-3   (preprocessing.calc_power)
    # so exposure_ms and fluence are linearly related given the LED power.
    # PLACEHOLDER: faro's payload does not carry stim_power; we assume a fixed
    # power here. Change to the value faro actually drives the DMD at.
    instrument: str = "niesen"     # selects the calibration curve
    stim_power_pct: float = 10.0    # LED power (%) faro stimulates at

    # --- control law ------------------------------------------------------
    # The model predicts CNR given future fluence; the controller inverts that by
    # searching dose plans and scoring them with an Objective. These three fields
    # configure the *default* policy used when no ``policy_file`` is given:
    # ``target_cnr`` is the setpoint of a ``hold`` objective, searched over an
    # ``n_candidates``-wide exposure grid, ``control_horizon`` frames ahead.
    # A policy file overrides all of this per FOV — see optoerk.serving.policy.
    #
    # ``target_cnr`` is in the loaded checkpoint's cnr units: cnr_median_norm for a
    # cnr_mode="norm" model (resting baseline == 1.0; training population mean
    # ~1.64), absolute cnr_median for a "raw" model. The startup banner prints both
    # so a mismatch is visible.
    #
    # ``control_horizon`` is hard-capped at the checkpoint's ``future_len``:
    # rolling further is untrained and indexes past ``sigma_step_bias_param``.
    target_cnr: float = 1.5
    control_horizon: int = 5         # frames to look ahead when scoring a dose
    n_candidates: int = 5           # exposure grid resolution (0..max_exposure_ms)

    # Acquisition cadence. The training data is built on a 1-frame = 1-minute grid
    # (``preprocessing.add_stim_features``), so this is 1.0 for every dataset so
    # far. It is explicit because time-parameterized references (the oscillation
    # step train) are configured in *minutes* while the horizon is in *frames*,
    # and the bound relating them is load-bearing: get this wrong and the startup
    # period check passes on a period the controller cannot actually see.
    frame_interval_min: float = 1.0

    # --- exposure output bounds (faro contract) ---------------------------
    min_exposure_ms: float = 0.0
    max_exposure_ms: float = 800.0
    dmd_quantum_ms: float = 25.0     # faro's DMD switch granularity (for logging)

    # --- online CNR baseline normalization --------------------------------
    # NOTE: this whole section applies only when the loaded checkpoint's
    # ``cnr_mode == "norm"``. A ``cnr_mode == "raw"`` model is fed raw
    # ``cnr_median`` directly and the server does NO online normalization and NO
    # dark window (baseline_frames / dark_baseline / baseline_mode are ignored).
    #
    # For a norm-mode model, the cnr input channel is ``cnr_median_norm`` =
    # cnr_median divided by the cell's baseline (median of its first
    # ``baseline_frames`` frames). faro sends raw cnr per frame, so the server
    # accumulates the baseline online. Before ``baseline_frames`` frames are seen
    # the baseline is a provisional running median.
    baseline_frames: int = 10

    # --- dark baseline (do NOT poison the baseline with immediate stimulation) --
    # The baseline must be a *resting* CNR. If stimulation starts on frame 0 the
    # baseline window is captured under light and comes out inflated, which biases
    # cnr_norm downward and makes the controller over-drive. When ``dark_baseline``
    # is on the server withholds stimulation until a resting baseline is measured.
    #   * "per_cell": every cell is held dark for its own first ``baseline_frames``
    #     frames, whenever it first appears. Most accurate per cell; pauses control
    #     on each newborn and re-baselines cleanly after track fragmentation.
    #   * "field": the FOV is held dark for its first ``baseline_frames`` frames to
    #     measure per-cell resting baselines AND a FOV-median field baseline. Cells
    #     born after the window are seeded with the field baseline and stimulated
    #     immediately — keeps control continuous and fixes fragmentation re-births.
    dark_baseline: bool = True
    baseline_mode: str = "field"  # "per_cell" | "field"

    # --- optoRTK expression feature override ------------------------------
    # optocheck / faro does not send a reliable per-cell optoRTK-expression rank.
    # When ``override_optortk_expr`` is on the server IGNORES any payload value and
    # feeds a fixed raw ``optortk_expr_value`` on the optoRTK channel for every cell
    # and every prediction. When the value is None it defaults to the training
    # population mean for that channel (history_norm_stats.json → ~0.5, the neutral
    # median rank), i.e. the value implicitly used today. Only the real model uses
    # this channel; the stub ignores it.
    override_optortk_expr: bool = False
    optortk_expr_value: float | None = None

    # --- LIVE per-cell optoRTK expression ---------------------------------
    # When on, the server reconstructs the real feature instead of feeding any
    # constant: the per-cell mCitrine value from the payload's optocheck /
    # reference measurement, ranked against the session's cohort pooled across
    # FOVs (see `optoerk.serving.expression`).
    #
    # OFF BY DEFAULT, and switching it on is a change of EXPERIMENTAL CONDITION,
    # not a bug fix: it gives the controller a per-cell gain covariate it has never
    # had, so a run with it on is not comparable to one with it off.
    #
    # It requires the payload to carry the optocheck/reference measurement
    # (`ref_mean_intensity`, or the older `optocheck_mean_intensity`). faro's
    # `RefFE` writes it into the tracks, but `InferenceServerStim._current_cells`
    # keeps only identity columns plus the columns ITS OWN feature extractor
    # produced — and the ref extractor is a different one — so it has to be let
    # through explicitly on the faro side. If nothing ever arrives the server
    # aborts once the cohort seals empty (`_check_optortk_coverage`) rather than
    # quietly feeding the constant, because a silent fallback looks exactly like a
    # successful run and produces a whole experiment of median-expresser
    # predictions.
    live_optortk_expr: bool = False
    # How many optocheck samples one cell contributes before its rank freezes. One
    # optocheck per run is the normal case, so 1 is the sensible default.
    optortk_baseline_frames: int = 1
    # When the session cohort closes. Must be long enough to span the FIRST
    # optocheck of the run — nobody can be ranked before the population exists.
    optortk_cohort_frames: int = 10

    # --- prediction logging -----------------------------------------------
    # When set, the server appends a structured JSONL record per prediction
    # (startup + per-cell decisions) to this path, matching the schema the
    # analysis notebooks parse. Off by default. Highly recommended for real
    # runs — it is the only record of raw_cnr / baseline / exposure per cell.
    predict_log_path: str | None = None

    # --- latency instrumentation ------------------------------------------
    # Every predict-log record carries a ``timing`` block decomposing server
    # latency: ``recv_epoch`` (wall-clock the request entered the service),
    # ``lock_wait_s`` (time blocked on the single service lock — i.e. another
    # FOV's inference serializing ahead of this one), ``infer_s`` (engine
    # ``decide`` wall time), and ``handler_s`` (total recv→done). Together they
    # separate an upstream stall (recv cadence drift) from a serialization
    # backlog (lock_wait) from the model itself being slow (infer). This is the
    # decomposition needed to diagnose faro ``stim_mask`` timeouts.
    #
    # Independently of the JSONL log, when a prediction's total ``handler_s``
    # exceeds this many seconds a one-line warning is printed to the server's
    # stderr, so slow frames are visible even with logging off. 0 disables the
    # warning. Keep it well under faro's stim-mask timeout (80 s) so a jam is
    # surfaced before faro gives up on the frame.
    slow_predict_warn_s: float = 30.0

    # --- startup warmup ---------------------------------------------------
    # Run a few throwaway inferences right after the model loads, priming the
    # CUDA context, cuDNN autotune and allocator pools so the FIRST real frame
    # hits a warm GPU instead of paying cold-start latency (a prime suspect for
    # the observed startup jam). No effect off CUDA.
    warmup: bool = True

    # --- GPU telemetry sampler --------------------------------------------
    # When the predict log is enabled and the engine is on CUDA, a background
    # thread samples NVML GPU telemetry every this many seconds and appends
    # ``{"event": "gpu", ...}`` records (util, memory, temperature, power,
    # clock-throttle reasons, and the processes on the device). It runs off the
    # prediction path, so it keeps recording through a stall. Requires
    # ``nvidia-ml-py``. 0 disables it.
    gpu_sample_interval_s: float = 5.0

    # --- crowding features -------------------------------------------------
    crowd_radius_px: float = 200.0   # n_cells_200px neighbourhood radius

    # --- state store bookkeeping ------------------------------------------
    evict_after_frames: int = 60     # drop a cell unseen for this many timesteps
    use_parent_seed: bool = True     # seed a daughter's state from parent_particle

    # --- stub policy (used only when no real model is loaded) -------------
    stub_gain_ms_per_cnr: float = 800.0  # exposure = gain * max(0, target - cnr)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Build a config from defaults overridden by OPTOERK_SERVE_* env vars.

        The conversion is driven by each field's *default value* type, so e.g.
        ``OPTOERK_SERVE_PORT=9000`` becomes an int and
        ``OPTOERK_SERVE_CHECKPOINT_DIR=/path`` a str. ``checkpoint_dir`` also
        treats the empty string / "none" as ``None`` (stub model).
        """
        defaults = cls()
        kwargs: dict = {}
        for f in fields(cls):
            raw = _env(f.name)
            if raw is None:
                continue
            default = getattr(defaults, f.name)
            if f.name in ("checkpoint_dir", "predict_log_path", "policy_file"):
                kwargs[f.name] = None if raw.strip().lower() in ("", "none") else raw
            elif f.name == "optortk_expr_value":
                # float | None: empty / "none" clears the override to the default.
                kwargs[f.name] = None if raw.strip().lower() in ("", "none") else float(raw)
            elif isinstance(default, bool):
                kwargs[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                kwargs[f.name] = int(raw)
            elif isinstance(default, float):
                kwargs[f.name] = float(raw)
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)
