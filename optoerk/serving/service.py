"""Transport-agnostic inference service: the pure ``/predict`` + ``/reset`` logic.

Owns the model engine and the per-(fov, particle) state store, and enforces the
faro contract: namespacing by FOV, first-seen cells, idempotent retries per
(fov, timestep), online CNR baseline normalization, crowding features derived
from the payload, and memory eviction. A single lock serializes the whole
critical section (torch + state are not thread-safe); the HTTP layer stays
concurrent.
"""
from __future__ import annotations

import threading
from typing import Any

from optoerk.serving.config import ServerConfig
from optoerk.serving.features import compute_crowding, extract_raw_cnr
from optoerk.serving.runtime import CellFrame, load_engine
from optoerk.serving.state import StateStore

from pprint import pp as pprint

class InferenceService:
    def __init__(self, cfg: ServerConfig | None = None):
        self.cfg = cfg or ServerConfig()
        self.lock = threading.Lock()
        self.engine, self.info = load_engine(self.cfg)
        self.store = StateStore(evict_after_frames=self.cfg.evict_after_frames)
        self.model_loaded = bool(self.info.get("model_loaded", False))
        self._n_predict = 0

    # -- health / info -----------------------------------------------------
    def health(self) -> dict:
        # Ready to serve as soon as an engine (real or stub) is loaded.
        return {"status": "ok", "model_loaded": self.model_loaded}

    def info_dict(self) -> dict:
        with self.lock:
            return {
                **self.info,
                "expected_cell_columns": [
                    "particle", "x", "y", "cnr", "cnr_median",
                ],
                "input_channels": ["cnr", "u_t", "fov_density", "n_cells_200px"],
                "output_units": "exposure_milliseconds",
                "exposure_range_ms": [self.cfg.min_exposure_ms, self.cfg.max_exposure_ms],
                "target_cnr": self.cfg.target_cnr,
                "control_horizon": self.cfg.control_horizon,
                "calibration": {
                    "instrument": self.cfg.instrument,
                    "stim_power_pct": self.cfg.stim_power_pct,
                },
                "n_tracked_cells": self.store.n_cells(),
                "n_predict_calls": self._n_predict,
            }

    # -- reset -------------------------------------------------------------
    def reset(self, fov: int | None = None) -> dict:
        with self.lock:
            self.store.reset(fov)
        return {"status": "ok", "reset": "all" if fov is None else fov}

    # -- predict -----------------------------------------------------------
    def predict(self, payload: dict[str, Any]) -> dict:
        fov = int(payload["fov"])
        timestep = int(payload["timestep"])
        cells = payload.get("cells", []) or []
        print(f'fov {fov}, ts:{timestep}')
        pprint(type(cells))
        pprint(cells[0:2])

        with self.lock:
            self._n_predict += 1
            # Idempotency: a retried (fov, timestep) returns the cached response
            # WITHOUT advancing any recurrent state.
            cached = self.store.cached_response(fov, timestep)
            if cached is not None:
                return {"fov": fov, "timestep": timestep, "exposures": dict(cached)}

            exposures = self._predict_locked(fov, timestep, cells)
            self.store.store_response(fov, timestep, exposures)
            self.store.evict(fov, timestep)
        print('exposures')
        pprint(exposures)
        print(f'---'*20)
        return {"fov": fov, "timestep": timestep, "exposures": exposures}

    def _predict_locked(self, fov: int, timestep: int, cells: list[dict]) -> dict[str, float]:
        # Crowding features from all cells' positions in this frame.
        fov_density, n200 = compute_crowding(cells, radius=self.cfg.crowd_radius_px)

        frames: list[CellFrame] = []
        particles: list[int] = []
        exposures: dict[str, float] = {}

        for i, cell in enumerate(cells):
            if "particle" not in cell:
                continue
            particle = int(cell["particle"])
            raw_cnr = extract_raw_cnr(cell)
            if raw_cnr is None:
                exposures[str(particle)] = 0.0  # can't score without CNR
                continue

            parent = cell.get("parent_particle") if self.cfg.use_parent_seed else None
            st = self.store.get_or_create(
                fov, particle, parent=int(parent) if parent is not None else None
            )

            # Out-of-order / duplicate-for-this-cell guard: don't re-advance.
            if timestep <= st.last_timestep:
                exposures[str(particle)] = 0.0
                st.last_seen_timestep = max(st.last_seen_timestep, timestep)
                continue

            cnr_norm = st.update_baseline(raw_cnr, self.cfg.baseline_frames)
            frames.append(
                CellFrame(
                    state=st,
                    cnr_norm=cnr_norm,
                    fov_density=float(fov_density[i]),
                    n_cells_200px=float(n200[i]),
                )
            )
            particles.append(particle)
        print('pre-engine')
        ms_list = self.engine.decide(frames)
        print('post-engine')

        for particle, f, ms in zip(particles, frames, ms_list):
            f.state.last_timestep = timestep
            f.state.last_seen_timestep = timestep
            f.state.n_frames += 1
            exposures[str(particle)] = float(ms)
        return exposures
