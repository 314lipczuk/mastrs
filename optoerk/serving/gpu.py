"""GPU telemetry for the inference server: cheap in-process memory counters and
a background NVML sampler.

Two pieces, both **best-effort** — a telemetry failure must never break serving:

  * :func:`cuda_mem_mb` — our process's ``torch.cuda`` allocator footprint
    (allocated + reserved MB). Essentially free; attached to every ``timing``
    block so allocator growth / fragmentation (a cause of ``cudaMalloc`` stalls)
    is visible per prediction.

  * :class:`GpuSampler` — a daemon thread that samples NVML device telemetry
    (util, memory, temperature, power, clock-throttle reasons, and the list of
    processes on the device) at a fixed interval and emits one
    ``{"event": "gpu", ...}`` record per sample. It runs OFF the prediction/lock
    path, so it keeps recording through a stall — exactly when the per-frame
    ``predict`` records go dark. Requires ``nvidia-ml-py`` (import name
    ``pynvml``); if that is missing or a read fails it logs once and stops.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable


def cuda_mem_mb(device) -> dict[str, float]:
    """Our process's CUDA allocator footprint for ``device`` (MB), or ``{}`` when
    not on CUDA / unavailable. Best-effort, never raises."""
    try:
        if device is None or getattr(device, "type", None) != "cuda":
            return {}
        import torch

        return {
            "cuda_alloc_mb": round(torch.cuda.memory_allocated(device) / 1e6, 1),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1e6, 1),
        }
    except Exception:  # noqa: BLE001 - telemetry must never break serving
        return {}


class GpuSampler(threading.Thread):
    """Daemon thread emitting periodic ``gpu`` telemetry records via ``write``."""

    def __init__(
        self,
        write: Callable[[dict[str, Any]], None],
        device_index: int,
        interval_s: float,
    ):
        super().__init__(daemon=True, name="gpu-sampler")
        self._write = write
        self._device_index = device_index
        self._interval_s = interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            import pynvml
        except Exception as e:  # noqa: BLE001
            self._write({"t": time.time(), "event": "gpu",
                         "error": f"pynvml unavailable: {e!r}"})
            return
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception as e:  # noqa: BLE001
            self._write({"t": time.time(), "event": "gpu",
                         "error": f"nvml init failed: {e!r}"})
            return
        try:
            while True:
                self._write(self._sample(pynvml, handle))
                if self._stop.wait(self._interval_s):
                    break
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def _sample(self, pynvml, handle) -> dict[str, Any]:
        rec: dict[str, Any] = {"t": time.time(), "event": "gpu"}
        # Each read is guarded independently: a driver that doesn't support one
        # metric should not cost us the others.
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(handle)
            rec["gpu_util_pct"] = int(u.gpu)
            rec["mem_util_pct"] = int(u.memory)
        except Exception:  # noqa: BLE001
            pass
        try:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            rec["mem_used_mb"] = round(m.used / 1e6, 1)
            rec["mem_total_mb"] = round(m.total / 1e6, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            rec["temp_c"] = int(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            rec["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1)
        except Exception:  # noqa: BLE001
            pass
        try:
            # Nonzero => the GPU is clocking down (thermal / power / etc.); the
            # raw bitmask is enough to flag it and decode offline.
            rec["throttle"] = int(
                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            rec["n_procs"] = len(procs)
            rec["procs"] = [
                {"pid": p.pid, "mem_mb": _proc_mem_mb(p)} for p in procs
            ]
        except Exception:  # noqa: BLE001
            pass
        return rec

    def stop(self) -> None:
        self._stop.set()


def _proc_mem_mb(p) -> float | None:
    """Per-process GPU memory (MB), or None when NVML reports it unavailable
    (the field is a large sentinel rather than a real byte count)."""
    m = getattr(p, "usedGpuMemory", None)
    if isinstance(m, (int, float)) and 0 <= m < 2 ** 60:
        return round(m / 1e6, 1)
    return None
