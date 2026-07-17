"""Optional structured JSONL log of the server's per-prediction decisions.

One line per record. Two event types, matching the schema the analysis notebooks
already parse (``experiments/inference_cnrhold_tracks.py``):

  * ``startup`` — ``{t, n_predict, event, engine, model_loaded, info}``
  * ``predict`` — ``{t, n_predict, event, fov, timestep, n_cells_in, n_scored,
    engine, timing:{...}, cells:[...], skipped:[...]}`` where each cell carries
    ``{particle, raw_cnr, cnr_norm, baseline, fov_density, n_cells_200px,
    u_t_in, n_frames_seen, first_seen, exposure_ms, fluence_out, dark,
    optortk_expr}``.

``t`` is the *completion* timestamp; ``timing`` decomposes how the latency to
that point was spent: ``recv_epoch`` (wall-clock the request entered the
service), ``lock_wait_s`` (blocked on the service lock behind another FOV's
inference), ``infer_s`` (engine ``decide`` time), ``handler_s`` (total
recv→done), plus ``cuda_alloc_mb`` / ``cuda_reserved_mb`` (our process's CUDA
allocator footprint) when on GPU. Per-FOV gaps in ``recv_epoch`` are the true
upstream acquisition cadence, distinct from gaps in ``t``; this is what tells an
upstream stall apart from a serialization backlog apart from a slow model when
diagnosing faro ``stim_mask`` timeouts.

A third event type is written on GPU runs when a ``gpu_sample_interval_s`` is
set (see :mod:`optoerk.serving.gpu`):

  * ``gpu`` — ``{t, event, gpu_util_pct, mem_util_pct, mem_used_mb,
    mem_total_mb, temp_c, power_w, throttle, n_procs, procs:[{pid, mem_mb}]}``,
    sampled by a background thread so it keeps recording through a stall (when
    ``predict`` records stop). Fields are best-effort and may be absent.

Enabled by setting ``predict_log_path`` on the config. The file is opened in
append mode and line-buffered, so records survive a crash without an explicit
close. Writing is best-effort: a logging failure must never break serving.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class PredictLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered text append: each record is flushed as it is written.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        # write() is called from both the request threads and the GPU sampler
        # thread; serialize so records never interleave mid-line.
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, default=str) + "\n"
            with self._lock:
                self._fh.write(line)
        except Exception as e:  # noqa: BLE001 - logging must never break serving
            print(f"[serving] predict-log write failed: {e!r}")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
