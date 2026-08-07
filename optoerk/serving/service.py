"""Transport-agnostic inference service: the pure ``/predict`` + ``/reset`` logic.

Owns the model engine and the per-(fov, particle) state store, and enforces the
faro contract: namespacing by FOV, first-seen cells, idempotent retries per
(fov, timestep), online CNR baseline normalization, crowding features derived
from the payload, and memory eviction. A single lock serializes the whole
critical section (torch + state are not thread-safe); the HTTP layer stays
concurrent.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any

from optoerk.serving.config import ServerConfig
from optoerk.serving.expression import ExpressionCohort
from optoerk.serving.features import (
    OPTORTK_KEYS,
    compute_crowding,
    extract_nuc_area,
    extract_optortk_value,
    extract_raw_cnr,
)
from optoerk.serving.gpu import GpuSampler, cuda_mem_mb
from optoerk.serving.objectives import GoalContext
from optoerk.serving.policy import PolicyRouter, load_policy_file
from optoerk.serving.predict_log import PredictLogger
from optoerk.serving.runtime import CellFrame
from optoerk.serving.state import StateStore


class InferenceService:
    def __init__(self, cfg: ServerConfig | None = None, policy_file=None):
        """``policy_file``: an already-parsed :class:`~optoerk.serving.policy.PolicyFile`
        to use instead of loading ``cfg.policy_file`` from disk. Only the soak
        benchmark passes this — it needs to run a policy whose placeholders are
        still unresolved, which the router rightly refuses to load for a real run.
        """
        self.cfg = cfg or ServerConfig()
        self.lock = threading.Lock()
        # One engine per distinct policy; FOVs without an override share the
        # default. With no policy file this is a single engine, as before.
        if policy_file is None and self.cfg.policy_file:
            policy_file = load_policy_file(self.cfg.policy_file)
        self.router = PolicyRouter(self.cfg, policy_file)
        self.engine = self.router.default_engine
        self.info = self.router.default_info
        self.store = StateStore(evict_after_frames=self.cfg.evict_after_frames)
        self.model_loaded = bool(self.info.get("model_loaded", False))
        # cnr convention the loaded checkpoint expects: "norm" -> reconstruct
        # cnr_median_norm online (baseline division); "raw" -> feed raw cnr_median
        # with NO online normalization. Governs the normalization branch below.
        self.cnr_mode = self.info.get("cnr_mode", "norm")
        self._n_predict = 0

        # Live optoRTK expression. ONE cohort for the whole service, not one per
        # FOV: offline the cohort is the imaging session, so ranking per field
        # would make each field its own population and change what the feature
        # means. Everything else here is namespaced by FOV; this deliberately is
        # not.
        self._n_optortk_seen = 0
        self._optortk_checked = False
        self.cohort: ExpressionCohort | None = (
            ExpressionCohort(
                baseline_frames=self.cfg.optortk_baseline_frames,
                cohort_frames=self.cfg.optortk_cohort_frames,
                # The policy declares the run's fields, so the cohort can wait for
                # ALL of them rather than sealing on whichever reports first.
                expected_fovs=(frozenset(policy_file.fov)
                               if policy_file and policy_file.fov else None),
            )
            if self.cfg.live_optortk_expr
            else None
        )
        if self.cohort is not None and self.cfg.override_optortk_expr:
            raise ValueError(
                "live_optortk_expr and override_optortk_expr are both set. The "
                "override feeds a fixed value to every cell, which is exactly what "
                "the live feature exists to stop doing — pick one."
            )
        print(f"[serving] optoRTK expression: {self.optortk_expr_mode()}")

        # Optional structured prediction log (see optoerk.serving.predict_log).
        self._logger: PredictLogger | None = None
        self._gpu_sampler: GpuSampler | None = None
        if self.cfg.predict_log_path:
            self._logger = PredictLogger(self.cfg.predict_log_path)
            self._logger.write(
                {
                    "t": time.time(),
                    "n_predict": 0,
                    "event": "startup",
                    "engine": type(self.engine).__name__,
                    "model_loaded": self.model_loaded,
                    "info": self.info,
                    # Every resolved policy, verbatim: the run's log is then a
                    # complete record of what it actually ran, and the replay
                    # harness can rebuild it without the original policy file.
                    "policies": self.router.describe(),
                    # Which of the three optoRTK-expression regimes this run used.
                    # It changes what the controller can condition on, so a run is
                    # only comparable to another with the same value here.
                    "optortk_expr": self.optortk_expr_mode(),
                }
            )
            # Background GPU telemetry (CUDA only) — runs off the prediction path
            # so it keeps sampling through a stall. See optoerk.serving.gpu.
            dev = getattr(self.engine, "device", None)
            if self.cfg.gpu_sample_interval_s > 0 and getattr(dev, "type", None) == "cuda":
                self._gpu_sampler = GpuSampler(
                    self._logger.write, dev.index or 0, self.cfg.gpu_sample_interval_s
                )
                self._gpu_sampler.start()

    # -- health / info -----------------------------------------------------
    def health(self) -> dict:
        # Ready to serve as soon as an engine (real or stub) is loaded.
        return {"status": "ok", "model_loaded": self.model_loaded}

    def _check_optortk_coverage(self, fov: int, timestep: int, n_seen: int) -> None:
        """Fail when the cohort seals with nobody in it.

        Most frames legitimately carry no optoRTK value: the optocheck is its own
        short acquisition, run once or twice per experiment, so per-frame absence
        is the normal case and never an error.

        The real failure is structural — the cohort window closed and not one cell
        ever supplied a measurement. That means the acquisition has no optocheck /
        reference channel wired up, or faro dropped the column: `RefFE` writes
        `ref_mean_intensity`, but `InferenceServerStim._current_cells` keeps only
        identity columns plus the columns ITS OWN feature extractor produced, and
        the ref extractor is a different one. Aborting beats feeding every cell the
        population mean, which would look exactly like a successful run.
        """
        self._n_optortk_seen += n_seen
        if not self.cohort.sealed or self._optortk_checked:
            return
        self._optortk_checked = True
        if self.cohort.n_cells > 0:
            return
        raise ValueError(
            f"live_optortk_expr is on but the expression cohort sealed EMPTY at "
            f"fov {fov} timestep {timestep}: no cell supplied any of {OPTORTK_KEYS} "
            f"in the first {self.cohort.cohort_frames} frames. Either the "
            f"acquisition has no optocheck/reference channel, or faro is not "
            f"forwarding it — `RefFE` writes `ref_mean_intensity` into the tracks, "
            f"but InferenceServerStim keeps only its own feature extractor's "
            f"columns, so it has to be let through explicitly. Raise "
            f"optortk_cohort_frames if the first optocheck lands later than that."
        )

    def optortk_expr_mode(self) -> dict:
        """Which optoRTK-expression regime this run is in, for the log and /info.

        Three regimes, and they are not interchangeable — the feature is a per-cell
        gain covariate, so it changes what the controller can condition on:

          ``live``     per-cell session-relative rank, reconstructed online
          ``constant`` one operator-chosen value for every cell
          ``neutral``  the channel's training population mean for every cell,
                       i.e. every cell treated as a median expresser

        ``neutral`` is the historical default and is still a constant — dropping
        ``--optortk-expr-value`` moves between the two constant regimes, it does
        not turn the feature on.
        """
        if self.cohort is not None:
            return {
                "mode": "live",
                "source": "payload ref_mean_intensity (mCitrine optocheck), session-ranked",
                # How many (cell, frame) pairs actually carried a measurement.
                # Small by design — the optocheck runs once or twice per run.
                "n_optortk_values_seen": self._n_optortk_seen,
                **self.cohort.describe(),
            }
        fed = getattr(self.engine, "optortk_fed", None)
        return {
            "mode": "constant" if self.cfg.override_optortk_expr else "neutral",
            "value_fed": None if fed is None else float(fed),
        }

    def info_dict(self) -> dict:
        with self.lock:
            return {
                **self.info,
                "expected_cell_columns": [
                    "particle", "x", "y", "cnr", "cnr_median",
                ],
                "input_channels": ["cnr", "u_t", "fov_density", "n_cells_200px"],
                "cnr_mode": self.cnr_mode,
                "output_units": "exposure_milliseconds",
                "exposure_range_ms": [self.cfg.min_exposure_ms, self.cfg.max_exposure_ms],
                "policies": self.router.describe(),
                "optortk_expr": self.optortk_expr_mode(),
                "calibration": {
                    "instrument": self.cfg.instrument,
                    "stim_power_pct": self.cfg.stim_power_pct,
                },
                "n_tracked_cells": self.store.n_cells(),
                "n_predict_calls": self._n_predict,
            }

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Stop the GPU sampler and close the predict log. Idempotent."""
        if self._gpu_sampler is not None:
            self._gpu_sampler.stop()
            self._gpu_sampler.join(timeout=2.0)
            self._gpu_sampler = None
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    # -- reset -------------------------------------------------------------
    def reset(self, fov: int | None = None) -> dict:
        with self.lock:
            self.store.reset(fov)
        return {"status": "ok", "reset": "all" if fov is None else fov}

    # -- predict -----------------------------------------------------------
    def predict(self, payload: dict[str, Any]) -> dict:
        # Latency clock starts the moment the request enters the service, BEFORE
        # contending for the lock — lock_wait then captures serialization backlog.
        t_recv = time.perf_counter()
        recv_epoch = time.time()
        fov = int(payload["fov"])
        timestep = int(payload["timestep"])
        cells = payload.get("cells", []) or []

        with self.lock:
            lock_wait_s = time.perf_counter() - t_recv
            self._n_predict += 1
            # Idempotency: a retried (fov, timestep) returns the cached response
            # WITHOUT advancing any recurrent state.
            cached = self.store.cached_response(fov, timestep)
            if cached is not None:
                return {"fov": fov, "timestep": timestep, "exposures": dict(cached)}

            exposures = self._predict_locked(
                fov, timestep, cells,
                t_recv=t_recv, recv_epoch=recv_epoch, lock_wait_s=lock_wait_s,
            )
            self.store.store_response(fov, timestep, exposures)
            self.store.evict(fov, timestep)
        return {"fov": fov, "timestep": timestep, "exposures": exposures}

    def _predict_locked(
        self,
        fov: int,
        timestep: int,
        cells: list[dict],
        *,
        t_recv: float,
        recv_epoch: float,
        lock_wait_s: float,
    ) -> dict[str, float]:
        cfg = self.cfg
        # Crowding features from all cells' positions in this frame.
        fov_density, n200 = compute_crowding(cells, radius=cfg.crowd_radius_px)

        # cnr convention is a property of THIS FOV's checkpoint, not of the server:
        # per-FOV policies may mix a raw-CNR model with a norm-CNR one.
        cnr_mode = self.router.info_for(fov).get("cnr_mode", self.cnr_mode)

        # Field-mode dark window: the FOV's first ``baseline_frames`` frames are
        # held dark to measure resting baselines (per-cell + a FOV field baseline).
        # Norm-mode only — raw-CNR models do no online baseline normalization.
        window_start = self.store.fov_window_start(fov, timestep)
        in_field_window = (
            cnr_mode == "norm"
            and cfg.dark_baseline
            and cfg.baseline_mode == "field"
            and timestep < window_start + cfg.baseline_frames
        )

        n_with_optortk = 0
        frames: list[CellFrame] = []
        particles: list[int] = []
        dark_flags: list[bool] = []
        exposures: dict[str, float] = {}
        # Per-cell log records (partial; exposure filled in after decide) + the
        # particles skipped this frame. Only assembled when logging is enabled.
        log_on = self._logger is not None
        log_cells: list[dict] = []
        skipped: list[int] = []

        for i, cell in enumerate(cells):
            if "particle" not in cell:
                continue
            particle = int(cell["particle"])
            raw_cnr = extract_raw_cnr(cell)
            if raw_cnr is None:
                exposures[str(particle)] = 0.0  # can't score without CNR
                skipped.append(particle)
                continue

            parent = cell.get("parent_particle") if cfg.use_parent_seed else None
            st = self.store.get_or_create(
                fov, particle, parent=int(parent) if parent is not None else None
            )

            # Out-of-order / duplicate-for-this-cell guard: don't re-advance.
            if timestep <= st.last_timestep:
                exposures[str(particle)] = 0.0
                st.last_seen_timestep = max(st.last_seen_timestep, timestep)
                skipped.append(particle)
                continue

            # Resolve cnr_norm and whether this cell is held dark this frame so its
            # baseline is measured at rest rather than under stimulation.
            dark = False
            if cnr_mode == "raw":
                # Model trained on absolute cnr_median: feed the raw scalar with no
                # online baseline normalization and no dark window (both are
                # norm-mode machinery). Cells are stimulated from their first frame.
                cnr_norm = raw_cnr
            elif not cfg.dark_baseline:
                cnr_norm = st.update_baseline(raw_cnr, cfg.baseline_frames)
            elif cfg.baseline_mode == "per_cell":
                # Hold each cell dark until it has measured its own resting baseline.
                dark = not st.baseline_ready(cfg.baseline_frames)
                cnr_norm = st.update_baseline(raw_cnr, cfg.baseline_frames)
            else:  # "field"
                if in_field_window:
                    dark = True
                    cnr_norm = st.update_baseline(raw_cnr, cfg.baseline_frames)
                    self.store.add_field_sample(fov, raw_cnr)
                else:
                    # Window over: a birth without its own dark baseline inherits the
                    # FOV field baseline instead of normalizing by a stimulated frame.
                    if not st.baseline_ready(cfg.baseline_frames):
                        fb = self.store.field_baseline(fov)
                        if fb is not None:
                            st.seed_baseline(fb, cfg.baseline_frames)
                    cnr_norm = st.normalize(raw_cnr)

            # Live optoRTK expression rank, if enabled. `observe` folds this
            # frame's measurement into the cell's window and returns its rank
            # while the window fills, frozen once it is full.
            expr = None
            if self.cohort is not None:
                # None on the many frames that are not optochecks — normal, not a
                # gap. `observe` also returns None while the cohort is still open.
                value = extract_optortk_value(cell)
                if value is not None:
                    n_with_optortk += 1
                expr = self.cohort.observe(fov, st, value, timestep)

            frames.append(
                CellFrame(
                    state=st,
                    cnr_norm=cnr_norm,
                    fov_density=float(fov_density[i]),
                    n_cells_200px=float(n200[i]),
                    optortk_expr=expr,
                    nuc_area=extract_nuc_area(cell),
                    # Not model inputs — objectives gate on position (e.g. "only
                    # the right half of the field"). Missing -> NaN, which fails
                    # every position predicate rather than silently passing.
                    x=float(cell.get("x", float("nan"))),
                    y=float(cell.get("y", float("nan"))),
                )
            )
            particles.append(particle)
            dark_flags.append(dark)

            if log_on:
                # Snapshot the inputs BEFORE decide(): u_t_in is the fluence carried
                # into this encoder step, which decide() overwrites with the new dose.
                log_cells.append(
                    {
                        "particle": particle,
                        "raw_cnr": float(raw_cnr),
                        "cnr_norm": float(cnr_norm),
                        "baseline": None if st.baseline is None else float(st.baseline),
                        "fov_density": float(fov_density[i]),
                        "n_cells_200px": float(n200[i]),
                        "u_t_in": float(st.last_fluence),
                        "n_frames_seen": int(st.n_frames),
                        "first_seen": st.n_frames == 0,
                        "dark": bool(dark),
                    }
                )

        if self.cohort is not None:
            self._check_optortk_coverage(fov, timestep, n_with_optortk)

        engine = self.router.engine_for(fov)
        goal_ctx = GoalContext(fov=fov, timestep=timestep, cells=frames)
        t_infer = time.perf_counter()
        ms_list = engine.decide(frames, goal_ctx)
        infer_s = time.perf_counter() - t_infer

        for particle, f, ms, dark in zip(particles, frames, ms_list, dark_flags):
            if dark:
                # No stimulation while the resting baseline is being measured. Zero
                # last_fluence too, so the next encoder step sees the true (zero)
                # applied dose as its u_t input — and last_applied_ms with it, so
                # the move penalty measures the next move from the dose that was
                # really applied rather than the one the controller proposed.
                ms = 0.0
                f.state.last_fluence = 0.0
                f.state.last_applied_ms = 0.0
            f.state.last_timestep = timestep
            f.state.last_seen_timestep = timestep
            f.state.n_frames += 1
            exposures[str(particle)] = float(ms)

        # Latency decomposition: recv -> (lock_wait) -> ... -> (infer) -> done.
        handler_s = time.perf_counter() - t_recv
        warn_s = self.cfg.slow_predict_warn_s
        if warn_s and handler_s > warn_s:
            print(
                f"[serving] SLOW predict fov={fov} timestep={timestep}: "
                f"handler={handler_s:.1f}s (lock_wait={lock_wait_s:.1f}s "
                f"infer={infer_s:.1f}s n_scored={len(frames)})",
                file=sys.stderr,
                flush=True,
            )

        if log_on:
            optortk = getattr(engine, "optortk_fed", None)
            # What the controller was actually tracking this frame, per cell: r_t,
            # and for an oscillating reference the segment label and the cell's
            # phase offset. Emitted explicitly so the analysis never re-derives the
            # waveform from parameters and gets it subtly wrong.
            objective = getattr(engine, "objective", None)
            notes = (
                objective.annotate(goal_ctx)
                if objective is not None and hasattr(objective, "annotate")
                else [{}] * len(frames)
            )
            # What the controller believed about the plan it picked. `None` on the
            # stub, which has neither a cost nor a forward model.
            costs = getattr(engine, "last_plan_cost", None) or [None] * len(frames)
            preds = getattr(engine, "last_pred_cnr_h1", None) or [None] * len(frames)
            for rec, f, ms, note, cost, pred in zip(
                log_cells, frames, ms_list, notes, costs, preds
            ):
                rec["exposure_ms"] = float(0.0 if rec["dark"] else ms)
                rec["fluence_out"] = float(f.state.last_fluence)
                # Per-cell when the live feature is on, the engine-wide constant
                # otherwise. Same key either way, so the analysis reads one column
                # and `optortk_live` next to it says which it is.
                rec["optortk_expr"] = (
                    float(f.optortk_expr) if f.optortk_expr is not None
                    else (None if optortk is None else float(optortk))
                )
                rec["optortk_live"] = f.optortk_expr is not None
                rec["plan_cost"] = None if cost is None else float(cost)
                # decide() predicted under the dose it commanded; a dark cell is
                # forced to 0 ms afterwards, so that prediction is for a dose that
                # was never applied. Null rather than log a number that is wrong.
                rec["pred_cnr_h1"] = (
                    None if pred is None or rec["dark"] else float(pred)
                )
                rec.update(note)
            self._logger.write(
                {
                    "t": time.time(),
                    "n_predict": self._n_predict,
                    "event": "predict",
                    "fov": fov,
                    "timestep": timestep,
                    "n_cells_in": len(cells),
                    "n_scored": len(frames),
                    "engine": type(engine).__name__,
                    "timing": {
                        "recv_epoch": recv_epoch,
                        "lock_wait_s": round(lock_wait_s, 4),
                        "infer_s": round(infer_s, 4),
                        "handler_s": round(handler_s, 4),
                        **cuda_mem_mb(getattr(engine, "device", None)),
                    },
                    "cells": log_cells,
                    "skipped": skipped,
                }
            )
        return exposures
