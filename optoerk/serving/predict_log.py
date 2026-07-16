"""Optional structured JSONL log of the server's per-prediction decisions.

One line per record. Two event types, matching the schema the analysis notebooks
already parse (``experiments/inference_cnrhold_tracks.py``):

  * ``startup`` — ``{t, n_predict, event, engine, model_loaded, info}``
  * ``predict`` — ``{t, n_predict, event, fov, timestep, n_cells_in, n_scored,
    engine, cells:[...], skipped:[...]}`` where each cell carries
    ``{particle, raw_cnr, cnr_norm, baseline, fov_density, n_cells_200px,
    u_t_in, n_frames_seen, first_seen, exposure_ms, fluence_out, dark,
    optortk_expr}``.

Enabled by setting ``predict_log_path`` on the config. The file is opened in
append mode and line-buffered, so records survive a crash without an explicit
close. Writing is best-effort: a logging failure must never break serving.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PredictLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered text append: each record is flushed as it is written.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
        except Exception as e:  # noqa: BLE001 - logging must never break serving
            print(f"[serving] predict-log write failed: {e!r}")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
