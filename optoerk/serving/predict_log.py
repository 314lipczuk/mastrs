"""Optional structured JSONL log of the server's per-prediction decisions.

One line per record. Two event types, matching the schema the analysis notebooks
already parse (``experiments/inference_cnrhold_tracks.py``):

  * ``startup`` — ``{t, n_predict, event, engine, model_loaded, info, policies,
    optortk_expr}``
  * ``predict`` — ``{t, n_predict, event, fov, timestep, n_cells_in, n_scored,
    engine, optortk_degraded, timing:{...}, cells:[...], skipped:[...]}`` where
    each cell carries ``{particle, raw_cnr, cnr_norm, baseline, fov_density,
    n_cells_200px, u_t_in, n_frames_seen, first_seen, exposure_ms, fluence_out,
    dark, optortk_expr, optortk_source, nuc_area, plan_cost, pred_cnr_h1}``, plus
    ``shared_dose`` when the controller splits the field into arms, plus
    whatever the objective's reference annotates (``r_t`` always; ``segment`` /
    ``phase_offset_min`` for an oscillation; also ``block_index`` /
    ``sweep_index`` / ``block_period_min`` for a frequency staircase).
  * ``optortk_cohort`` — ``{t, event, fov, timestep, spread:{n, min, p25, median,
    p75, max}, degraded}``, written once when the expression cohort closes. It
    records the distribution every rank in the run is measured against, so a
    degenerate cohort is visible directly instead of only through the ranks it
    produced.
  * ``cadence`` — ``{t, event, fov, timestep, declared_s, observed_s, ratio,
    n_samples, degraded}``, written once after enough per-FOV frame intervals have
    been seen. ``observed_s`` is the median gap between two successive frames of the
    SAME field (fields are imaged sequentially inside a round, so the gap between
    requests is a slot, not a frame). A ratio above tolerance means every reference
    period is stretched by that factor in real time and the checkpoint is being run
    at an interval it was not trained on; ``cadence_degraded`` is then sticky on
    every subsequent ``predict`` record.

``plan_cost`` and ``pred_cnr_h1`` record what the controller *believed* about the
plan it chose: the winning plan's cost, and the predictive mean one step ahead
under the dose it commanded. Without them a saturated cell and a mispredicted
cell are indistinguishable in the log — both just show a dose and a CNR — and
telling them apart needs a full replay. ``pred_cnr_h1`` also makes every frame a
one-step model-error measurement: the drift of achieved-delta over predicted-delta
is a per-cell sensitivity readout. Both are ``None`` on the stub engine, and
``pred_cnr_h1`` is ``None`` for a dark-window cell, whose commanded dose is
overridden to zero after the prediction was made.

``optortk_expr`` is the value fed on the optoRTK-expression channel, and
``optortk_source`` says where that number came from: ``"measured"`` for a real
per-cell session rank (``--live-optortk-expr``, and this cell was in the
optocheck), ``"fallback"`` for the middle of the percentile scale — the channel's
training population mean, which standardizes to exactly 0. The regime is also
recorded once, whole, in the ``startup`` record under ``optortk_expr``. It changes
what the controller can condition on, so two runs are only comparable when it
matches.

Expect ``"fallback"`` to be common and to grow through a run: faro attaches the
value only to the tracks it followed through the optocheck, there is no lineage
column, so cells appearing later never acquire one. Reading the per-cell source is
the only way to tell a run where the feature was live from one where it decayed to
a constant. ``nuc_area`` is recorded for the same reason — ``None`` means the model
was fed that channel's population mean rather than a real area.

``optortk_degraded`` on the ``predict`` record is sticky: once the cohort closes
with too few cells to rank against (``optortk_min_cohort_cells``) every subsequent
frame carries it, so the condition lives in the log rather than in one startup
message.

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
