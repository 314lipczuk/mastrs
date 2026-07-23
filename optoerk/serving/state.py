"""Per-(fov, particle) streaming state for the online recurrent encoder.

The central design requirement: the model's encoder LSTM is run as an
**online per-cell recurrent state**, advanced by exactly one step per frame,
instead of re-encoding the whole past every call. For a causal ``nn.LSTM`` the
one-step advance with the carried ``(h, c)`` is numerically identical to
encoding the full history (verified to ~1e-8).

Each :class:`CellState` holds:
  * ``h`` / ``c``          — the encoder LSTM hidden/cell state (or None until
                            the first frame; a fresh cell starts from zeros).
  * ``last_fluence``       — the last commanded fluence (mJ/cm2). This is the
                            ``u_t`` input fed at the *next* encoder step, since
                            the dose applied after frame k is what drives k+1.
  * baseline bookkeeping   — to online-normalize CNR into ``cnr_median_norm``.
  * ``last_timestep``      — for idempotency (retries must not double-advance).

All state is namespaced by ``fov``; particle ids are only unique within a FOV.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class CellState:
    # encoder LSTM state; None means "not yet initialized" (start from zeros).
    h: torch.Tensor | None = None
    c: torch.Tensor | None = None
    # last commanded fluence (mJ/cm2) -> u_t input for the next encoder step.
    last_fluence: float = 0.0
    # online CNR baseline (median of first `baseline_frames` cnr_median values).
    baseline_samples: list[float] = field(default_factory=list)
    baseline: float | None = None
    last_cnr_norm: float = 1.0
    # bookkeeping
    last_timestep: int = -1
    last_seen_timestep: int = -1
    n_frames: int = 0
    # this cell's particle id (within its FOV). Set at creation. Used by
    # StaggeredCadenceMPC to assign the cell a fixed cadence phase (particle % k),
    # so a cell keeps its stimulation slot for its whole lifetime.
    particle: int = -1

    def baseline_ready(self, baseline_frames: int) -> bool:
        """True once a full baseline window has been accumulated (or seeded)."""
        return len(self.baseline_samples) >= baseline_frames

    def normalize(self, raw_cnr: float) -> float:
        """cnr_median_norm = raw / baseline, WITHOUT folding raw into the baseline.

        Used once the baseline is fixed (a completed window or a seeded field
        baseline), so stimulated frames never contaminate the reference.
        """
        base = self.baseline if self.baseline else float(raw_cnr)
        if base == 0:
            base = 1.0
        norm = float(raw_cnr) / base
        self.last_cnr_norm = norm
        return norm

    def seed_baseline(self, value: float, baseline_frames: int) -> None:
        """Fix the baseline to an externally-measured resting value (the FOV field
        baseline) so a mid-experiment birth is never normalized by its own
        stimulated frames. Fills the sample window so it counts as complete.
        """
        self.baseline = float(value)
        self.baseline_samples = [float(value)] * baseline_frames

    def update_baseline(self, raw_cnr: float, baseline_frames: int) -> float:
        """Fold a new raw CNR into the baseline estimate; return cnr_median_norm.

        The baseline is the median of the cell's first ``baseline_frames`` raw
        CNR values (matching ``preprocessing.clean``). Before that many frames
        exist it is a provisional running median so early frames still get a
        sane normalization (~1.0 at rest). NOTE: for the baseline to be a
        *resting* reference, the caller must ensure these frames are unstimulated
        (see ``dark_baseline`` in the config / service).
        """
        if len(self.baseline_samples) < baseline_frames:
            self.baseline_samples.append(float(raw_cnr))
            self.baseline = float(np.median(self.baseline_samples))
        return self.normalize(raw_cnr)


class StateStore:
    """Thread-unsafe store of per-(fov, particle) cell state + idempotency cache.

    Guard access with the service-level lock. Memory is bounded by evicting
    cells unseen for ``evict_after_frames`` timesteps.
    """

    def __init__(self, evict_after_frames: int = 60):
        self.evict_after_frames = evict_after_frames
        self._cells: dict[tuple[int, int], CellState] = {}
        # per-fov idempotency cache: fov -> (timestep, response_exposures).
        self._last_response: dict[int, tuple[int, dict[str, float]]] = {}
        # per-fov dark-baseline bookkeeping (field mode): the first timestep seen
        # (window start), the resting CNRs collected during the window, and the
        # resulting FOV field baseline (cached once the window closes).
        self._fov_start: dict[int, int] = {}
        self._field_samples: dict[int, list[float]] = {}
        self._field_baseline: dict[int, float] = {}

    # -- cells -------------------------------------------------------------
    def get(self, fov: int, particle: int) -> CellState | None:
        return self._cells.get((fov, particle))

    def get_or_create(
        self, fov: int, particle: int, parent: int | None = None
    ) -> CellState:
        key = (fov, particle)
        st = self._cells.get(key)
        if st is None:
            st = CellState()
            # Optional lineage: seed a daughter from its mother's state.
            if parent is not None:
                pst = self._cells.get((fov, parent))
                if pst is not None:
                    st.h = None if pst.h is None else pst.h.clone()
                    st.c = None if pst.c is None else pst.c.clone()
                    st.last_fluence = pst.last_fluence
                    st.baseline_samples = list(pst.baseline_samples)
                    st.baseline = pst.baseline
                    st.last_cnr_norm = pst.last_cnr_norm
            # The daughter carries its OWN particle id (its cadence phase), not the
            # mother's — a division must not put both cells in the same slot.
            st.particle = particle
            self._cells[key] = st
        return st

    # -- field / dark baseline (per-fov) -----------------------------------
    def fov_window_start(self, fov: int, timestep: int) -> int:
        """First timestep seen for this FOV — the start of its dark-baseline window."""
        return self._fov_start.setdefault(fov, timestep)

    def add_field_sample(self, fov: int, raw_cnr: float) -> None:
        """Collect a resting-CNR sample during the FOV's dark window."""
        self._field_samples.setdefault(fov, []).append(float(raw_cnr))

    def field_baseline(self, fov: int) -> float | None:
        """FOV-median resting baseline from the dark window (cached), or None if
        no dark samples were collected."""
        if fov in self._field_baseline:
            return self._field_baseline[fov]
        samples = self._field_samples.get(fov)
        if not samples:
            return None
        base = float(np.median(samples))
        self._field_baseline[fov] = base
        return base

    # -- idempotency -------------------------------------------------------
    def cached_response(self, fov: int, timestep: int) -> dict[str, float] | None:
        entry = self._last_response.get(fov)
        if entry is not None and entry[0] == timestep:
            return entry[1]
        return None

    def store_response(self, fov: int, timestep: int, exposures: dict[str, float]) -> None:
        self._last_response[fov] = (timestep, exposures)

    # -- maintenance -------------------------------------------------------
    def evict(self, fov: int, current_timestep: int) -> int:
        """Drop cells in ``fov`` unseen for > evict_after_frames. Returns count."""
        cutoff = current_timestep - self.evict_after_frames
        dead = [
            key
            for key, st in self._cells.items()
            if key[0] == fov and st.last_seen_timestep < cutoff
        ]
        for key in dead:
            del self._cells[key]
        return len(dead)

    def reset(self, fov: int | None = None) -> None:
        if fov is None:
            self._cells.clear()
            self._last_response.clear()
            self._fov_start.clear()
            self._field_samples.clear()
            self._field_baseline.clear()
        else:
            for key in [k for k in self._cells if k[0] == fov]:
                del self._cells[key]
            self._last_response.pop(fov, None)
            self._fov_start.pop(fov, None)
            self._field_samples.pop(fov, None)
            self._field_baseline.pop(fov, None)

    def n_cells(self) -> int:
        return len(self._cells)
