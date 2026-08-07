"""Dress rehearsal: will the server keep up during the real 12-FOV run?

``bench.py`` times ``decide()`` in isolation. That answers "how fast is the
model", which is *not* the question that decides whether an experiment survives
12 hours. This module answers the actual question by driving the real HTTP
server the way faro drives it:

  * real transport — ``POST /predict`` over the real ``ThreadingHTTPServer``,
    including JSON encode/decode and thread handoff;
  * real concurrency — faro issues up to ``--concurrency`` overlapping FOV calls,
    which all serialize behind :class:`InferenceService`'s single lock;
  * real cadence — 12 FOVs per acquisition cycle, arriving spread across the
    cycle as each field is imaged, not all at once;
  * real payloads — replayed from a previous run's predict log, so cell counts,
    CNR distributions, births and track fragmentation are the ones that actually
    occur (``--from-log``), rather than a flat synthetic count.

**The constraint that actually binds.** Every FOV's inference runs inside one
process-wide lock, so the arms do not get a per-FOV budget in parallel — they
share one. The system is stable only if the *total* lock occupancy per cycle
stays below the cycle duration::

    rho = (sum over FOVs of infer_time) / cycle_seconds

``rho < 1`` and the queue drains every cycle. ``rho > 1`` and the backlog grows
without bound: the first frames look fine, ``lock_wait_s`` creeps up, and faro
starts missing its ``stim_mask`` timeout hours in. That is the failure mode the
hold run's timing decomposition was pointing at, and no single-FOV benchmark can
see it. This soak reports ``rho`` directly and fails the run if it is near 1.

Run on the microscope computer (or the GPU node that will serve it)::

    uv run python -m optoerk.serving.soak \\
        --policy-file policies/policy_12fov_osc.toml --allow-placeholders \\
        --device cuda --from-log <previous_run>.jsonl \\
        --cycles 20 --cycle-seconds 60 --concurrency 4

Point it at an already-running server instead with ``--url http://host:8080``,
which is the most faithful test of all — it exercises the exact process faro
will talk to.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import ThreadingHTTPServer
from pathlib import Path

import polars as pl

# faro gives up on a frame's stim_mask after this long. Exceeding it means the
# cells for that FOV are not stimulated at all for that frame.
FARO_STIM_MASK_TIMEOUT_S = 80.0


# ---------------------------------------------------------------------------
# payload sources
# ---------------------------------------------------------------------------


def payloads_from_log(
    log_path: str | Path, n_fovs: int, start_frame: int = 0
) -> dict[int, list[dict]]:
    """Replay a previous run's per-frame cell payloads, remapped onto ``n_fovs``.

    Returns ``{fov: [payload_per_frame, ...]}``. Source FOVs are cycled if the
    previous run had fewer, and particle ids are namespaced per target FOV so the
    server's per-(fov, particle) state behaves as it would with distinct fields.

    Using a real log matters: cell count is the single biggest driver of inference
    time, and it is neither constant nor uniform across FOVs in a real run.

    ``start_frame`` skips into the log. Cycle *N* of the soak replays frame
    ``start_frame + N``, so without it a 20-cycle soak only ever sees the first
    20 frames — the sparsest part of the run, before tracks have accumulated.
    The question that matters is whether the server keeps up at hour 12, so start
    where the cell count is at its worst. ``frames_summary`` prints the counts so
    you can see which part of the run you actually selected.
    """
    from optoerk.serving.replay import iter_predict_records, record_to_payload

    by_src: dict[int, list[dict]] = defaultdict(list)
    for rec in iter_predict_records(log_path):
        by_src[int(rec["fov"])].append(record_to_payload(rec))
    if not by_src:
        raise ValueError(f"no predict records in {log_path}")

    src_fovs = sorted(by_src)
    out: dict[int, list[dict]] = {}
    for fov in range(n_fovs):
        src = src_fovs[fov % len(src_fovs)]
        src_frames = by_src[src]
        if start_frame >= len(src_frames):
            raise ValueError(
                f"--start-frame {start_frame} is past the end of source fov "
                f"{src} ({len(src_frames)} frames)"
            )
        frames = []
        for t, payload in enumerate(src_frames[start_frame:]):
            cells = [
                {**c, "particle": int(c["particle"]) + fov * 1_000_000}
                for c in payload.get("cells", [])
            ]
            frames.append({"fov": fov, "timestep": t, "cells": cells})
        out[fov] = frames
    return out


def frames_summary(frames: dict[int, list[dict]], cycles: int) -> str:
    """Cell counts over the window that will actually be replayed."""
    counts = [
        len(frames[f][c % len(frames[f])].get("cells", []))
        for f in frames for c in range(cycles)
    ]
    counts.sort()
    return (f"cells/FOV over the replayed window: min={counts[0]} "
            f"p50={counts[len(counts) // 2]} max={counts[-1]}")


def augment_cells(frames: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Add the per-cell fields a modern checkpoint needs, where they are missing.

    Two channels arrived after the existing predict logs were written, so a
    replayed payload does not carry them and a synthetic one never did:

      * ``ref_mean_intensity`` — the mCitrine optoRTK measurement. WITHOUT IT a
        server running ``--live-optortk-expr`` aborts: the expression cohort seals
        with nobody in it, which is the correct response to a run whose reference
        channel never arrives, and a fatal one for a benchmark.
      * ``area_nuc`` — the payload spelling of the ``nuc_area`` channel.

    Values are deterministic in ``(fov, particle)`` so a soak is reproducible, and
    static per cell because that is what the real quantities are. They are
    fabricated: fine for a LATENCY benchmark, where compute depends on the number
    of cells and not on what the numbers say, and meaningless for anything else.
    """
    for fov, seq in frames.items():
        for payload in seq:
            for c in payload.get("cells", []):
                pid = int(c.get("particle", 0))
                if not any(k in c for k in ("ref_mean_intensity",
                                            "optocheck_mean_intensity")):
                    c["ref_mean_intensity"] = 400.0 + (pid * 7919 % 2100)
                if "area_nuc" not in c and "nuc_area" not in c:
                    c["area_nuc"] = 120.0 + (pid * 104729 % 180)
    return frames


def payloads_synthetic(n_fovs: int, n_frames: int, n_cells: int) -> dict[int, list[dict]]:
    """Flat synthetic fallback when no previous log is available.

    Deliberately second choice: a constant cell count understates the tail, and
    the tail is what breaks a run.
    """
    out = {}
    for fov in range(n_fovs):
        frames = []
        for t in range(n_frames):
            cells = [
                {
                    "particle": fov * 1_000_000 + i,
                    "x": float(50 * (i % 32)), "y": float(50 * (i // 32)),
                    # a plausible spread around the raw-CNR resting level
                    "cnr_median": 0.82 + 0.30 * ((i % 7) / 7.0),
                }
                for i in range(n_cells)
            ]
            frames.append({"fov": fov, "timestep": t, "cells": cells})
        out[fov] = frames
    return out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------


def _post(url: str, body: dict, timeout: float) -> tuple[float, int, str | None]:
    """POST and return (seconds, http_status, error). Never raises."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return time.perf_counter() - t0, resp.status, None
    except urllib.error.HTTPError as e:
        return time.perf_counter() - t0, e.code, e.reason
    except Exception as e:  # noqa: BLE001 - a timeout IS the result we're measuring
        return time.perf_counter() - t0, 0, repr(e)


def drive(
    url: str,
    frames: dict[int, list[dict]],
    cycles: int,
    cycle_seconds: float,
    concurrency: int,
    arms: dict[int, int] | None = None,
) -> pl.DataFrame:
    """Issue one predict per (cycle, FOV) on the real acquisition schedule.

    FOVs are spread across the cycle (``cycle_seconds / n_fovs`` apart), which is
    how they actually arrive — one field is imaged, then the next. A request is
    dispatched at its scheduled wall-clock time regardless of whether earlier ones
    have returned, so a backlog builds up exactly as it would in the real run.
    """
    n_fovs = len(frames)
    stride = cycle_seconds / n_fovs
    sem = threading.Semaphore(concurrency)
    rows: list[dict] = []
    lock = threading.Lock()
    t_start = time.perf_counter()

    def one(fov: int, cycle: int, due: float):
        with sem:
            payload = frames[fov][cycle % len(frames[fov])]
            payload = {**payload, "timestep": cycle}
            secs, status, err = _post(url, payload, FARO_STIM_MASK_TIMEOUT_S + 20)
            with lock:
                rows.append({
                    "cycle": cycle, "fov": fov, "arm": (arms or {}).get(fov, 1),
                    "n_cells": len(payload.get("cells", [])),
                    "seconds": secs, "status": status, "error": err,
                    # how late the request was dispatched relative to its slot:
                    # the client-side view of a backlog
                    "dispatch_lag_s": max(0.0, time.perf_counter() - t_start - due),
                })

    threads = []
    for cycle in range(cycles):
        for fov in range(n_fovs):
            due = cycle * cycle_seconds + fov * stride
            wait = due - (time.perf_counter() - t_start)
            if wait > 0:
                time.sleep(wait)
            th = threading.Thread(target=one, args=(fov, cycle, due), daemon=True)
            th.start()
            threads.append(th)
        done = sum(1 for t in threads if not t.is_alive())
        print(f"[soak] cycle {cycle + 1}/{cycles} dispatched "
              f"({done}/{len(threads)} complete)")
    for th in threads:
        th.join()
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# server-side timing, from the predict log
# ---------------------------------------------------------------------------


def server_timings(log_path: str | Path, since: float,
                   arms: dict[int, int] | None = None) -> pl.DataFrame:
    """The server's own latency decomposition per prediction, since ``since``.

    Client-side round-trip says *whether* it was slow; this says *why* —
    ``infer_s`` is the model, ``lock_wait_s`` is time queued behind another FOV.
    A run that fails with high lock_wait and low infer is a serialization problem
    and is fixed by batching or fewer samples, not a faster model.

    ``since`` (epoch seconds) is **load-bearing**, not a convenience. The predict
    log is opened in append mode, and in ``--url`` mode it is the live server's
    own log — which already contains every prediction it has served. Reading the
    whole file would compute rho over a mix of this soak and whatever ran before
    it, and rho is the headline verdict. Only records written during the soak
    window count.
    """
    rows = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "predict":
                continue
            if float(rec.get("t", 0.0)) < since:
                continue
            t = rec.get("timing", {}) or {}
            rows.append({
                "fov": rec.get("fov"), "timestep": rec.get("timestep"),
                "arm": (arms or {}).get(rec.get("fov"), 1),
                "n_scored": rec.get("n_scored"),
                "lock_wait_s": t.get("lock_wait_s"),
                "infer_s": t.get("infer_s"),
                "handler_s": t.get("handler_s"),
            })
    return pl.DataFrame(rows)


def verdict(client: pl.DataFrame, server: pl.DataFrame,
            cycle_seconds: float, n_fovs: int) -> None:
    """Print the go / no-go summary."""
    print("\n" + "=" * 72)
    print("PER-ARM CLIENT-SIDE ROUND TRIP (what faro actually experiences)")
    print("=" * 72)
    ok = client.filter(pl.col("status") == 200)
    summary = (
        ok.group_by("arm")
        .agg(
            pl.len().alias("n"),
            pl.col("n_cells").median().alias("cells_med"),
            (pl.col("seconds") * 1000).median().round(0).alias("ms_p50"),
            (pl.col("seconds") * 1000).quantile(0.95).round(0).alias("ms_p95"),
            (pl.col("seconds") * 1000).max().round(0).alias("ms_max"),
        )
        .sort("arm")
    )
    with pl.Config(tbl_rows=-1):
        print(summary)

    failed = client.filter(pl.col("status") != 200)
    if failed.height:
        print(f"\n!! {failed.height} request(s) did not return 200:")
        print(failed.select("cycle", "fov", "status", "error").head(10))

    timeouts = ok.filter(pl.col("seconds") > FARO_STIM_MASK_TIMEOUT_S)
    print(f"\nrequests exceeding faro's {FARO_STIM_MASK_TIMEOUT_S:.0f} s "
          f"stim_mask timeout: {timeouts.height} / {ok.height}")

    # --- the stability condition -------------------------------------------
    print("\n" + "=" * 72)
    print("LOCK UTILISATION  rho = total inference time per cycle / cycle time")
    print("=" * 72)
    if server.height and server["infer_s"].null_count() < server.height:
        per_cycle = float(server["infer_s"].sum()) / max(
            1, server["timestep"].n_unique())
        rho = per_cycle / cycle_seconds
        print(f"  mean inference per FOV : {server['infer_s'].mean() * 1000:.0f} ms")
        print(f"  total per cycle        : {per_cycle:.2f} s "
              f"({n_fovs} FOVs, all serialized on one lock)")
        print(f"  cycle duration         : {cycle_seconds:.1f} s")
        print(f"  rho                    : {rho:.2f}")
        if rho >= 1.0:
            print("\n  VERDICT: FAIL — the server cannot keep up. The backlog grows\n"
                  "  every cycle; early frames will look fine and faro will start\n"
                  "  missing stim_mask hours in. Reduce n_samples (equally across\n"
                  "  ALL arms, or the nesting breaks), shorten the horizon, or\n"
                  "  batch FOVs.")
        elif rho > 0.7:
            print("\n  VERDICT: MARGINAL — under 1 but with little headroom. Cell\n"
                  "  counts grow over a 12 h run, so this will drift upward.\n"
                  "  Re-run the soak at the cell count you expect at hour 12.")
        else:
            print("\n  VERDICT: OK — comfortable headroom.")

        print("\n  lock_wait vs infer (is it the model, or the queue?):")
        print(server.group_by("arm").agg(
            (pl.col("infer_s") * 1000).median().round(0).alias("infer_ms_p50"),
            (pl.col("lock_wait_s") * 1000).median().round(0).alias("lockwait_ms_p50"),
            (pl.col("lock_wait_s") * 1000).max().round(0).alias("lockwait_ms_max"),
        ).sort("arm"))
    else:
        print("  (no server-side timings — enable --predict-log to get them)")

    lag = client["dispatch_lag_s"]
    if lag.max() and lag.max() > 1.0:
        print(f"\n  NOTE: client dispatch lagged its own schedule by up to "
              f"{lag.max():.1f} s — the driver itself was saturated at "
              f"concurrency limit, so these numbers are optimistic.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def assert_real_models(info: dict, allow_stub: bool) -> None:
    """Refuse to benchmark the stub.

    A checkpoint that fails to load degrades that FOV to a deterministic
    proportional policy with no neural net — microseconds instead of hundreds of
    milliseconds. The soak would then print a comfortable ``rho`` and a clean
    verdict for a server that does none of the work the real run does. Silent
    false confidence is worse than no benchmark, so this is fatal by default.
    """
    policies = info.get("policies", {})
    stubs = [
        name for name, pol in
        [("default", policies.get("default", {}))]
        + list((f"fov {k}", v) for k, v in (policies.get("fov", {}) or {}).items())
        if not pol.get("model_loaded", False)
    ]
    if not stubs:
        return
    msg = (f"{len(stubs)} policy/policies fell back to the STUB engine "
           f"(no model loaded): {', '.join(stubs)}")
    if allow_stub:
        print(f"[soak] WARNING: {msg}. Timings are MEANINGLESS for the real run.")
        return
    raise SystemExit(
        f"[soak] REFUSING TO RUN: {msg}.\n"
        f"        The stub has no neural net, so it would benchmark ~0 ms per FOV\n"
        f"        and report a comfortable verdict for a server doing none of the\n"
        f"        real work. Fix the checkpoint path (the server printed why), or\n"
        f"        pass --allow-stub if you are deliberately testing the transport."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default=None,
                   help="drive an already-running server (most faithful). "
                        "Omit to start one in-process from --policy-file.")
    p.add_argument("--policy-file", dest="policy_file", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--allow-placeholders", action="store_true",
                   help="run against a policy whose placeholders_resolved is "
                        "false — correct for a benchmark, NEVER for a real run")
    p.add_argument("--allow-stub", action="store_true",
                   help="proceed even if a checkpoint failed to load. Only for "
                        "testing the transport; the numbers mean nothing.")
    p.add_argument("--from-log", dest="from_log", default=None,
                   help="replay cell payloads from a previous run's predict log")
    p.add_argument("--start-frame", dest="start_frame", type=int, default=0,
                   help="skip this many frames into --from-log. Use it to soak "
                        "against the CROWDED late part of a run rather than the "
                        "sparse first frames — hour 12 is the question, not hour 1.")
    p.add_argument("--n-fovs", dest="n_fovs", type=int, default=None,
                   help="fields to drive (default: the policy's FOV count, else 12)")
    p.add_argument("--live-optortk-expr", dest="live_optortk_expr",
                   action="store_true",
                   help="benchmark with live per-cell optoRTK expression on, as "
                        "the real run uses it. Payloads are augmented with a "
                        "fabricated reference measurement so the cohort can seal; "
                        "latency is representative, the ranks are not.")
    p.add_argument("--optortk-cohort-frames", dest="optortk_cohort_frames",
                   type=int, default=None)
    p.add_argument("--n-cells", dest="n_cells", type=int, default=208,
                   help="synthetic cells per FOV when --from-log is not given")
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--cycle-seconds", dest="cycle_seconds", type=float, default=60.0)
    p.add_argument("--concurrency", type=int, default=4,
                   help="max overlapping predict calls, as faro issues them")
    p.add_argument("--predict-log", dest="predict_log", default="soak_predict.jsonl")
    p.add_argument("--out", default=None, help="write client latencies parquet here")
    args = p.parse_args()

    # Default the field count to the policy's, so a 10-FOV policy is not
    # benchmarked as 12 with two FOVs silently falling back to [default].
    if args.n_fovs is None:
        if args.policy_file:
            from optoerk.serving.policy import load_policy_file as _lpf

            args.n_fovs = len(_lpf(args.policy_file).fov) or 12
            print(f"[soak] --n-fovs not given; using {args.n_fovs} from the policy")
        else:
            args.n_fovs = 12

    frames = augment_cells(
        payloads_from_log(args.from_log, args.n_fovs, args.start_frame)
        if args.from_log
        else payloads_synthetic(args.n_fovs, args.cycles, args.n_cells)
    )
    print(f"[soak] {args.n_fovs} FOVs, {args.cycles} cycles; "
          f"{frames_summary(frames, args.cycles)}")

    # Arm labels come from the policy file, never from a formula over the FOV
    # index — the 10-FOV layouts interleave arms to balance stage position, so
    # `fov % 4 + 1` mislabels every row against them. Without a policy file there
    # is only one arm, and saying so beats inventing four.
    arms: dict[int, int] = {}
    if args.policy_file:
        from optoerk.serving.policy import arm_map, load_policy_file

        arms = arm_map(load_policy_file(args.policy_file))
        by_arm: dict[int, list[int]] = defaultdict(list)
        for fov, arm in sorted(arms.items()):
            by_arm[arm].append(fov)
        print("[soak] arms from policy: " + "; ".join(
            f"arm {a} -> fov {fovs}" for a, fovs in sorted(by_arm.items())))
    elif args.url is None:
        print("[soak] no --policy-file: every FOV is reported as arm 1")

    server = None
    url = args.url
    if url is None:
        from optoerk.serving.app import make_handler
        from optoerk.serving.config import ServerConfig
        from optoerk.serving.policy import load_policy_file
        from optoerk.serving.service import InferenceService

        cfg = ServerConfig(
            device=args.device, checkpoint_dir=args.checkpoint,
            policy_file=args.policy_file, predict_log_path=args.predict_log,
            port=0, live_optortk_expr=args.live_optortk_expr,
        )
        if args.optortk_cohort_frames is not None:
            cfg.optortk_cohort_frames = args.optortk_cohort_frames
        if args.live_optortk_expr:
            print("[soak] live optoRTK expression ON — the cohort ranks a "
                  "FABRICATED reference measurement. Latency is representative; "
                  "the per-cell ranks are not.")
        pf = None
        if args.allow_placeholders and args.policy_file:
            # Benchmarking before the measured values exist is exactly the case
            # the gate is not meant to block: latency depends on the ladder size,
            # horizon, kernel and sample count, none of which are placeholders.
            # Flip it here, loudly, rather than weakening the gate itself.
            pf = load_policy_file(args.policy_file)
            pf.placeholders_resolved = True
            print("[soak] WARNING: running with UNRESOLVED placeholders. Latency "
                  "is representative; the science is NOT. Never do this for a run.")
        service = InferenceService(cfg, policy_file=pf)

        assert_real_models(service.info_dict(), args.allow_stub)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        port = httpd.server_address[1]
        server = threading.Thread(target=httpd.serve_forever, daemon=True)
        server.start()
        url = f"http://127.0.0.1:{port}/predict"
        print(f"[soak] in-process server on {url}")
    else:
        base = url.rstrip("/")
        with urllib.request.urlopen(f"{base}/info", timeout=30) as resp:
            assert_real_models(json.loads(resp.read()), args.allow_stub)
        url = f"{base}/predict"
        print(f"[soak] driving external server at {url}")
        # In --url mode we do NOT create the predict log; the server owns it.
        # --predict-log must therefore name the path that server was started
        # with, or there are no server-side timings and hence no rho.
        if not Path(args.predict_log).exists():
            print(f"[soak] NOTE: {args.predict_log} does not exist. In --url mode "
                  f"this must be the path the RUNNING server was started with "
                  f"(--predict-log). Without it you get client round-trip times "
                  f"but no rho and no lock_wait/infer split.")

    # Everything the server writes from here on belongs to this soak. Anything
    # already in the log (a previous soak, or the live run in --url mode) does
    # not, and must not contribute to rho.
    t_window = time.time()
    t0 = time.perf_counter()
    client = drive(url, frames, args.cycles, args.cycle_seconds, args.concurrency,
                   arms=arms)
    print(f"[soak] {client.height} requests in {time.perf_counter() - t0:.1f} s")

    srv = (
        server_timings(args.predict_log, since=t_window, arms=arms)
        if args.predict_log and Path(args.predict_log).exists()
        else pl.DataFrame()
    )
    if srv.height and srv.height < client.filter(pl.col("status") == 200).height:
        print(f"[soak] NOTE: {srv.height} server-side records for "
              f"{client.height} requests — the log may belong to a different "
              f"server process than the one being driven.")
    verdict(client, srv, args.cycle_seconds, args.n_fovs)

    if args.out:
        client.write_parquet(args.out)
        print(f"\n[soak] wrote {args.out}")


if __name__ == "__main__":
    main()
