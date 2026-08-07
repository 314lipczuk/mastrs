"""HTTP transport for the faro inference server (stdlib only, no extra deps).

Implements the contract in ``inference_server_spec.md``:
  * ``POST /predict`` — per-cell features in, per-cell exposure (ms) out.
  * ``GET  /health``  — readiness.
  * ``POST /reset``   — clear per-cell state (all, or one ``{"fov": N}``).
  * ``GET  /info``    — model / calibration / units metadata.

Uses ``ThreadingHTTPServer`` so faro's up-to-4 concurrent FOV calls are served
in parallel; :class:`InferenceService` serializes the torch + state critical
section behind a lock. Latency budget is generous (60 s acquisition interval).

Run::

    python -m optoerk.serving.app --host 0.0.0.0 --port 8080 \
        --checkpoint results/<bundle_dir>          # omit -> stub policy

FastAPI/uvicorn are intentionally NOT required; this keeps the server runnable
on a bare cluster node. The core logic lives in :mod:`optoerk.serving.service`
and is transport-agnostic, so a FastAPI wrapper can be dropped in unchanged.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from optoerk.serving.config import ServerConfig
from optoerk.serving.service import InferenceService

import traceback


def make_handler(service: InferenceService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quieter default logging
            pass

        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw)

        def do_GET(self):
            try:
                if self.path == "/health":
                    self._send(200, service.health())
                elif self.path == "/info":
                    self._send(200, service.info_dict())
                else:
                    self._send(404, {"error": f"unknown path {self.path}"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": repr(e)})

        def do_POST(self):
            try:
                body = self._read_json()
                if self.path == "/predict":
                    self._send(200, service.predict(body))
                elif self.path == "/reset":
                    fov = body.get("fov")
                    self._send(200, service.reset(None if fov is None else int(fov)))
                else:
                    self._send(404, {"error": f"unknown path {self.path}"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": repr(e)})

    return Handler


def build_service(args) -> InferenceService:
    cfg = ServerConfig.from_env()
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.checkpoint is not None:
        cfg.checkpoint_dir = args.checkpoint or None
    if args.device:
        cfg.device = args.device
    if args.stim_power is not None:
        cfg.stim_power_pct = args.stim_power
    if args.target_cnr is not None:
        cfg.target_cnr = args.target_cnr
    if args.dark_baseline is not None:
        cfg.dark_baseline = args.dark_baseline
    if args.baseline_mode:
        cfg.baseline_mode = args.baseline_mode
    if args.override_optortk_expr is not None:
        cfg.override_optortk_expr = args.override_optortk_expr
    if args.optortk_expr_value is not None:
        cfg.optortk_expr_value = args.optortk_expr_value
        cfg.override_optortk_expr = True  # a value implies the override is on
    if args.live_optortk_expr is not None:
        cfg.live_optortk_expr = args.live_optortk_expr
    if args.optortk_baseline_frames is not None:
        cfg.optortk_baseline_frames = args.optortk_baseline_frames
    if args.optortk_cohort_frames is not None:
        cfg.optortk_cohort_frames = args.optortk_cohort_frames
    if args.predict_log is not None:
        cfg.predict_log_path = args.predict_log or None
    if args.policy_file is not None:
        cfg.policy_file = args.policy_file or None
    return InferenceService(cfg)


def main():
    p = argparse.ArgumentParser(description="faro optogenetic inference server")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--checkpoint", default=None,
                   help="experiment bundle dir (omit -> stub policy)")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--stim-power", dest="stim_power", type=float, default=None,
                   help="LED power %% for fluence<->ms conversion")
    p.add_argument("--target-cnr", dest="target_cnr", type=float, default=None)
    p.add_argument("--dark-baseline", action=argparse.BooleanOptionalAction, default=None,
                   help="measure the baseline in the dark before stimulating (default: on)")
    p.add_argument("--baseline-mode", default=None, choices=["per_cell", "field"],
                   help="dark-baseline strategy (default: field)")
    p.add_argument("--override-optortk-expr", action=argparse.BooleanOptionalAction,
                   default=None, help="feed a fixed optoRTK-expr value for every cell")
    p.add_argument("--optortk-expr-value", dest="optortk_expr_value", type=float, default=None,
                   help="raw optoRTK-expr value to feed (implies --override-optortk-expr)")
    p.add_argument("--live-optortk-expr", dest="live_optortk_expr",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="reconstruct the real per-cell optoRTK-expr rank from the "
                        "payload's mCitrine optocheck measurement instead of a "
                        "constant. Needs ref_mean_intensity in /predict; aborts "
                        "if the cohort seals with nobody in it. CHANGES THE "
                        "EXPERIMENTAL CONDITION — a run with this on is not "
                        "comparable to one without.")
    p.add_argument("--optortk-baseline-frames", dest="optortk_baseline_frames",
                   type=int, default=None,
                   help="optocheck samples to median before freezing a rank "
                        "(default 1 — one optocheck per run is the normal case)")
    p.add_argument("--optortk-cohort-frames", dest="optortk_cohort_frames",
                   type=int, default=None,
                   help="when the session expression cohort closes; must span the "
                        "run's FIRST optocheck (default 10)")
    p.add_argument("--predict-log", dest="predict_log", default=None,
                   help="path to append a per-prediction JSONL log (off by default)")
    p.add_argument("--policy-file", dest="policy_file", default=None,
                   help="per-FOV policy file (.toml/.json): checkpoint + objective "
                        "+ controller per FOV. Overrides --checkpoint/--target-cnr.")
    args = p.parse_args()

    service = build_service(args)
    cfg = service.cfg
    handler = make_handler(service)
    server = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    mode = "REAL model" if service.model_loaded else "STUB policy"
    print(f"[serving] {mode} | listening on http://{cfg.host}:{cfg.port}")
    print(f"[serving] info: {json.dumps(service.info, default=str)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serving] shutting down")
        server.shutdown()
    finally:
        service.close()


if __name__ == "__main__":
    main()
