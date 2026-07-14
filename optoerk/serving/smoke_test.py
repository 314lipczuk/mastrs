"""End-to-end smoke test for the inference server.

Runs two checks against a live server (started in-process on an ephemeral port,
or against ``--url`` if given):

  1. The spec's example ``/predict`` payload -> a valid response with an
     exposure per cell in [0, max_ms].
  2. A sequence of frames for the SAME cell, to show the per-cell recurrent
     encoder state advancing across calls (idempotent retries do not
     double-advance).

Usage::

    python -m optoerk.serving.smoke_test                 # in-process, stub model
    OPTOERK_SERVE_CHECKPOINT_DIR=results/<dir> python -m optoerk.serving.smoke_test
    python -m optoerk.serving.smoke_test --url http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from optoerk.serving.app import make_handler
from optoerk.serving.config import ServerConfig
from optoerk.serving.service import InferenceService


def _post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def _start_inprocess() -> tuple[str, ThreadingHTTPServer, InferenceService]:
    service = InferenceService(ServerConfig.from_env())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return base, server, service


EXAMPLE = {
    "fov": 3, "timestep": 17, "time": 1020.0,
    "cells": [
        {"particle": 42, "x": 512.3, "y": 128.9, "cnr": 1.83, "cnr_median": 1.80,
         "area_nuc": 210.0, "mean_intensity_C1_nuc": 812.0, "mean_intensity_C1_ring": 1490.0},
        {"particle": 43, "x": 640.1, "y": 300.2, "cnr": 0.94, "cnr_median": 0.95,
         "area_nuc": 188.0, "mean_intensity_C1_nuc": 733.0, "mean_intensity_C1_ring": 690.0},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="target a running server instead of in-process")
    args = ap.parse_args()

    server = None
    if args.url:
        base = args.url.rstrip("/")
    else:
        base, server, _svc = _start_inprocess()
        time.sleep(0.2)

    ok = True
    try:
        print("== /health =="); print(_get(base, "/health"))
        print("== /info ==");   print(json.dumps(_get(base, "/info"), indent=2))

        # 1) spec example ----------------------------------------------------
        print("\n== /predict (spec example) ==")
        resp = _post(base, "/predict", EXAMPLE)
        print(resp)
        assert resp["fov"] == 3 and resp["timestep"] == 17
        exps = resp["exposures"]
        assert set(exps) == {"42", "43"}, exps
        for v in exps.values():
            assert 0.0 <= float(v) <= 3000.0, v

        # 2) idempotent retry (same fov,timestep) must return identical result
        print("\n== /predict retry (idempotency) ==")
        resp2 = _post(base, "/predict", EXAMPLE)
        assert resp2["exposures"] == exps, (resp2["exposures"], exps)
        print("retry matched ->", resp2["exposures"])

        # 3) streaming: same cell across advancing timesteps -----------------
        print("\n== streaming one cell across frames (encoder state advances) ==")
        _post(base, "/reset", {})
        cnr_track = [1.0, 1.05, 1.2, 1.5, 1.9, 2.3, 2.1, 1.8]
        for k, cnr in enumerate(cnr_track):
            payload = {"fov": 7, "timestep": k, "time": 60.0 * k,
                       "cells": [{"particle": 1, "x": 100.0, "y": 100.0,
                                  "cnr": cnr, "cnr_median": cnr}]}
            r = _post(base, "/predict", payload)
            print(f"  t={k}  cnr={cnr:>4}  ->  exposure_ms={r['exposures']['1']:.1f}")

        print("\nSMOKE TEST PASSED")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"\nSMOKE TEST FAILED: {e!r}")
        raise
    finally:
        if server is not None:
            server.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
