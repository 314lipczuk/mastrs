"""Fluence(mJ/cm2) <-> exposure(ms) conversion, reusing the training-time curve.

The model consumes / commands ``u_t`` = fluence in mJ/cm2. faro drives the DMD
in exposure milliseconds. The relationship is fixed by the microscope's power
calibration used to *build* the training data
(``optoerk.data.preprocessing.calc_power``)::

    fluence_mJ_cm2 = irradiance(mW/cm2) * exposure_ms * 1e-3

where ``irradiance`` is looked up from the instrument calibration at a given LED
power (``stim_power`` %). We reuse the exact same ``CALIBRATIONS`` table so the
server's conversion is consistent with how ``u_t`` was defined during training.

Because faro's payload does not carry ``stim_power``, we assume a *fixed* power
(``ServerConfig.stim_power_pct``). If faro later drives multiple powers, pass the
per-cell power through and rebuild the calibration per cell.
"""
from __future__ import annotations

import numpy as np

from optoerk.data.preprocessing import CALIBRATIONS, DEFAULT_INSTRUMENT


class FluenceCalibration:
    def __init__(self, instrument: str = DEFAULT_INSTRUMENT, stim_power_pct: float = 100.0):
        if instrument not in CALIBRATIONS:
            raise KeyError(
                f"unknown instrument {instrument!r}; known: {list(CALIBRATIONS)}"
            )
        calib = CALIBRATIONS[instrument]
        self.instrument = instrument
        self.stim_power_pct = float(stim_power_pct)
        # mW/cm2 at the configured LED power (piecewise-linear, same as training).
        self.irradiance_mW_cm2 = float(
            np.interp(self.stim_power_pct, calib["pct"], calib["mW_cm2"])
        )
        # fluence per ms of exposure at this power.
        self._fluence_per_ms = self.irradiance_mW_cm2 * 1e-3

    @property
    def fluence_per_ms(self) -> float:
        """mJ/cm2 per ms of exposure at the configured power. Exposed so the
        controllers can do the ms->fluence conversion in torch, on-device, without
        a numpy round-trip per candidate."""
        return self._fluence_per_ms

    def ms_to_fluence(self, exposure_ms: float | np.ndarray):
        """exposure (ms) -> fluence (mJ/cm2)."""
        return np.asarray(exposure_ms, dtype=np.float64) * self._fluence_per_ms

    def fluence_to_ms(self, fluence_mJ_cm2: float | np.ndarray):
        """fluence (mJ/cm2) -> exposure (ms). Zero irradiance -> 0 ms."""
        if self._fluence_per_ms <= 0:
            return np.zeros_like(np.asarray(fluence_mJ_cm2, dtype=np.float64))
        return np.asarray(fluence_mJ_cm2, dtype=np.float64) / self._fluence_per_ms

    def describe(self) -> dict:
        return {
            "instrument": self.instrument,
            "stim_power_pct": self.stim_power_pct,
            "irradiance_mW_cm2": self.irradiance_mW_cm2,
            "fluence_mJ_cm2_per_ms": self._fluence_per_ms,
        }
