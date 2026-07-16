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

    # --- fluence <-> milliseconds conversion ------------------------------
    # The model's ``u_t`` channel is fluence in mJ/cm2 =
    #   irradiance(mW/cm2, from stim_power) * exposure_ms * 1e-3   (preprocessing.calc_power)
    # so exposure_ms and fluence are linearly related given the LED power.
    # PLACEHOLDER: faro's payload does not carry stim_power; we assume a fixed
    # power here. Change to the value faro actually drives the DMD at.
    instrument: str = "niesen"     # selects the calibration curve
    stim_power_pct: float = 10.0    # LED power (%) faro stimulates at

    # --- control law (PLACEHOLDER objective) ------------------------------
    # The model predicts CNR given future fluence; the controller inverts that
    # by searching candidate exposures and picking the one whose predicted CNR
    # best tracks ``target_cnr`` over ``control_horizon`` future frames.
    # ``target_cnr`` is in cnr_median_norm units (baseline == 1.0; population
    # mean in the training data ~1.64). This is a stand-in for the real
    # experimental objective — set it to what the experiment actually wants.
    target_cnr: float = 1.5
    control_horizon: int = 5         # frames to look ahead when scoring a dose
    n_candidates: int = 5           # exposure grid resolution (0..max_exposure_ms)

    # --- exposure output bounds (faro contract) ---------------------------
    min_exposure_ms: float = 0.0
    max_exposure_ms: float = 800.0
    dmd_quantum_ms: float = 25.0     # faro's DMD switch granularity (for logging)

    # --- online CNR baseline normalization --------------------------------
    # The model's cnr input channel is ``cnr_median_norm`` = cnr_median divided
    # by the cell's baseline (median of its first ``baseline_frames`` frames).
    # faro sends raw cnr per frame, so the server accumulates the baseline
    # online. Before ``baseline_frames`` frames are seen the baseline is a
    # provisional running median.
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
            if f.name == "checkpoint_dir":
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
