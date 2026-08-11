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

import os
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


def _torch_device_uuid(index: int) -> str:
    """The CUDA device's hardware UUID as torch sees it, or ``""``.

    This is the only identifier that means the same thing on both sides of the
    NVML/CUDA divide — see :func:`resolve_nvml_handle`.
    """
    try:
        import torch

        return str(getattr(torch.cuda.get_device_properties(index), "uuid", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _norm_uuid(u) -> str:
    if isinstance(u, bytes):
        u = u.decode("ascii", "replace")
    return str(u).strip().lower().removeprefix("gpu-")


def resolve_nvml_handle(pynvml, index: int) -> tuple[Any, dict[str, Any]]:
    """NVML handle for the physical GPU that torch device ``cuda:index`` runs on.

    **NVML does not honour ``CUDA_VISIBLE_DEVICES``.** Its indices enumerate the
    machine's physical GPUs; torch's enumerate only the visible subset, in the
    order that variable lists them. So on any shared or scheduled box the two
    disagree, and ``nvmlDeviceGetHandleByIndex(torch_index)`` silently samples a
    DIFFERENT CARD than the one running the model — telemetry that looks healthy
    because it is describing somebody else's GPU. (That is not hypothetical: it
    is how a run reported a 21 GB device at 2% utilisation while torch reserved
    81 GB on the real one.)

    Resolution, most trustworthy first, with how it was resolved returned so the
    log records whether the numbers can be believed:

      1. **uuid** — match torch's device UUID against every NVML device. Correct
         under any remapping, and the only strategy that actually verifies.
      2. **visible-devices** — index into ``CUDA_VISIBLE_DEVICES`` ourselves.
         Right whenever that variable is what did the remapping.
      3. **index-unverified** — the old behaviour. Kept so telemetry still flows
         on a driver too old for the above, but labelled so nothing downstream
         treats it as confirmed.
    """
    meta: dict[str, Any] = {"torch_index": index}
    uuid = _torch_device_uuid(index)
    if uuid:
        meta["torch_uuid"] = uuid
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                if _norm_uuid(pynvml.nvmlDeviceGetUUID(h)) == _norm_uuid(uuid):
                    meta.update(nvml_index=i, resolved_by="uuid", verified=True)
                    return h, meta
        except Exception as e:  # noqa: BLE001
            meta["uuid_lookup_error"] = repr(e)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        meta["cuda_visible_devices"] = visible
        entries = [e.strip() for e in visible.split(",") if e.strip()]
        if index < len(entries):
            entry = entries[index]
            try:
                if entry.startswith(("GPU-", "MIG-")):
                    h = pynvml.nvmlDeviceGetHandleByUUID(entry.encode())
                    meta.update(resolved_by="visible-devices", verified=True)
                    return h, meta
                h = pynvml.nvmlDeviceGetHandleByIndex(int(entry))
                meta.update(nvml_index=int(entry), resolved_by="visible-devices",
                            verified=True)
                return h, meta
            except Exception as e:  # noqa: BLE001
                meta["visible_devices_error"] = repr(e)

    meta.update(nvml_index=index, resolved_by="index-unverified", verified=False)
    return pynvml.nvmlDeviceGetHandleByIndex(index), meta


class GpuSampler(threading.Thread):
    """Daemon thread emitting periodic ``gpu`` telemetry records via ``write``.

    ``device`` is the **torch device the model is on**, not an NVML index — the
    two are not interchangeable (see :func:`resolve_nvml_handle`). One
    ``gpu_device`` record is emitted before sampling starts, naming the card that
    was resolved and how, so a reader can tell whether the ``gpu`` stream
    describes the model's GPU or an unverified guess at it.
    """

    def __init__(
        self,
        write: Callable[[dict[str, Any]], None],
        device,
        interval_s: float,
    ):
        super().__init__(daemon=True, name="gpu-sampler")
        self._write = write
        self._device_index = 0 if getattr(device, "index", None) is None else device.index
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
            handle, meta = resolve_nvml_handle(pynvml, self._device_index)
        except Exception as e:  # noqa: BLE001
            self._write({"t": time.time(), "event": "gpu",
                         "error": f"nvml init failed: {e!r}"})
            return
        # Identify the card once, up front: name and total memory make the
        # per-sample `mem_used_mb` interpretable, and `verified` says whether
        # this is the model's GPU or only assumed to be.
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            meta["nvml_name"] = name.decode() if isinstance(name, bytes) else str(name)
            meta["nvml_uuid"] = _norm_uuid(pynvml.nvmlDeviceGetUUID(handle))
            meta["mem_total_mb"] = round(
                pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1e6, 1
            )
        except Exception:  # noqa: BLE001
            pass
        self._write({"t": time.time(), "event": "gpu_device", **meta})
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
