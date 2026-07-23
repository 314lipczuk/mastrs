"""How fast is ``decide()`` per standard FOV, across controllers and horizon?

Times one ``decide()`` — one FOV's worth of cells for one frame — for every
controller (``constant_dose``, ``sequence_mpc`` at a few sample counts, and
``staggered_mpc`` at a few k). The headline output is a **per-standard-FOV** table
(ms per 60 s frame) so the three experiment conditions can be compared directly;
the full horizon × cell-count sweep is printed underneath.

Two things it gets right that a naive timer would not:

  * **Staggering is priced per frame, honestly.** ``staggered_mpc`` only searches
    ~1/k of the cells on any given frame, and *which* cells differ frame to frame,
    so its per-frame cost is the mean over one full k-cycle (every cell searched
    exactly once across the cycle). The ``n_searched`` column shows how many cells
    actually hit the search each frame.
  * **F is answerable before a retrain.** Wall time depends on tensor shapes, not
    weights, so an untrained model at the target ``future_len`` gives the right
    timing — price H=15 (or 30) before committing GPU time to training it. Pass
    ``--checkpoint`` to time a real bundle instead (capped at its ``future_len``).

Context: the live server has a **60 s** budget per FOV frame, and the 2026-07-16
v5 run measured p99 ``infer_s`` = 0.12 s at ~208 cells with ``constant_dose`` at
H=5. Anything here under a second is comfortably free.

Run (on the cluster GPU — see CLAUDE.md; a laptop CPU answers a different
question)::

    uv run python -m optoerk.serving.bench --device cuda
    uv run python -m optoerk.serving.bench --device cuda --horizons 5,8,15 --stagger-k 4
    uv run python -m optoerk.serving.bench --device cuda --plot bench.png
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import torch

from optoerk.serving.calibration import FluenceCalibration
from optoerk.serving.config import ServerConfig
from optoerk.serving.control import (
    ConstantDoseSearch,
    SequenceMPC,
    StaggeredCadenceMPC,
    dose_levels,
)
from optoerk.serving.objectives import GoalContext, hold
from optoerk.serving.runtime import (
    CellFrame,
    ModelHandle,
    RealModelEngine,
    _resolve_device,
    load_model,
)
from optoerk.serving.state import CellState


def synthetic_handle(future_len: int, device: torch.device,
                     hidden: int = 64, layers: int = 2) -> ModelHandle:
    """An untrained model of the production shape at an arbitrary ``future_len``.

    This is what lets us price F=30 before committing GPU time to training one.
    """
    from optoerk.models.seq2scal_history import HistoryConfig, Seq2ScalarHistory

    channels = ["cnr", "u_t", "fov_density", "n_cells_200px", "optortk_expr"]
    cfg = HistoryConfig(
        input_dim=len(channels), hidden_dim=hidden, num_layers=layers,
        future_len=future_len, norm_channels=channels,
        norm_mean=[1.57, 44.0, 200.0, 6.0, 0.5],
        norm_std=[0.79, 90.0, 80.0, 4.0, 0.29],
    )
    model = Seq2ScalarHistory(cfg).to(device).eval()
    return ModelHandle(
        model=model,
        mean=np.asarray(cfg.norm_mean, np.float32),
        std=np.asarray(cfg.norm_std, np.float32),
        device=device,
        info={"model_type": "synthetic", "checkpoint_dir": None, "device": str(device),
              "future_len": future_len, "cnr_mode": "norm", "norm_channels": channels},
    )


def _fov_frames(n_cells: int) -> list[CellFrame]:
    """One FOV's worth of cells. ``state.particle`` is set to 0..n-1 so
    StaggeredCadenceMPC spreads them across its k phase groups — without ids they
    would all share one phase and the due subset would be all-or-nothing."""
    frames = []
    for i in range(n_cells):
        st = CellState()
        st.particle = i
        frames.append(CellFrame(state=st, cnr_norm=1.0, fov_density=float(n_cells),
                                n_cells_200px=5.0, x=float(i), y=0.0))
    return frames


def time_decide(engine, n_cells: int, timestep: int = 50, repeats: int = 3) -> float:
    """Median wall time of one ``decide()`` over ``n_cells`` at a given frame, warm.

    ``timestep`` matters only for the staggered controller (it selects which phase
    group is due); the others are timestep-independent.
    """
    frames = _fov_frames(n_cells)
    ctx = GoalContext(fov=0, timestep=timestep, cells=frames)
    engine.decide(frames, ctx)  # warm
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.decide(frames, ctx)
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_per_frame(engine, n_cells: int, repeats: int = 3) -> tuple[float, float, int]:
    """Steady-state per-frame cost, honest about staggering.

    For a staggered controller only ~1/k of the cells run the search on any frame,
    and which cells differ frame to frame — so the fair per-frame number is the
    mean over one full k-cycle (every cell searched exactly once across the cycle).
    Returns ``(mean_seconds, peak_seconds, n_searched_per_frame)``.
    """
    k = getattr(engine.controller, "k", 1)
    if k <= 1:
        secs = time_decide(engine, n_cells, timestep=50, repeats=repeats)
        return secs, secs, n_cells
    per_t = [time_decide(engine, n_cells, timestep=t, repeats=repeats) for t in range(k)]
    n_searched = round(n_cells / k)
    return float(np.mean(per_t)), float(np.max(per_t)), n_searched


def _build_controllers(levels, mpc_samples, stagger_k):
    """Every controller variant to benchmark, labelled for the results table."""
    variants = [("constant_dose", ConstantDoseSearch(levels))]
    for s in mpc_samples:
        variants.append((f"sequence_mpc@{s}", SequenceMPC(levels, n_samples=s)))
    for s in mpc_samples:
        for k in stagger_k:
            variants.append(
                (f"staggered_mpc@{s}_k{k}", StaggeredCadenceMPC(levels, k=k, n_samples=s))
            )
    return variants


def sweep(
    device: str = "auto",
    checkpoint: str | None = None,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30, 50),
    cell_counts: tuple[int, ...] = (64, 208, 512),
    n_candidates: int = 5,
    mpc_samples: tuple[int, ...] = (128, 512),
    stagger_k: tuple[int, ...] = (4,),
) -> pl.DataFrame:
    dev = _resolve_device(device)
    cfg = ServerConfig(device=device, warmup=False)
    calib = FluenceCalibration(cfg.instrument, cfg.stim_power_pct)
    levels = dose_levels(cfg.min_exposure_ms, cfg.max_exposure_ms, n_candidates)
    objective = hold(cfg.target_cnr)

    rows = []
    for H in horizons:
        if checkpoint:
            handle = load_model(checkpoint, device)
            if H > handle.info["future_len"]:
                print(f"[bench] skipping H={H}: checkpoint future_len="
                      f"{handle.info['future_len']}")
                continue
        else:
            handle = synthetic_handle(H, dev)

        for label, ctrl in _build_controllers(levels, mpc_samples, stagger_k):
            engine = RealModelEngine(
                handle, calib, ServerConfig(device=device, warmup=False,
                                            control_horizon=H,
                                            n_candidates=n_candidates),
                objective, ctrl,
            )
            for n in cell_counts:
                secs, peak, n_searched = time_per_frame(engine, n)
                rows.append({
                    "controller": label, "horizon": H, "n_cells": n,
                    "n_searched": n_searched,          # cells run through the search/frame
                    "seconds": round(secs, 4),         # steady-state per-frame (staggered: k-cycle mean)
                    "peak_seconds": round(peak, 4),    # worst frame in the cycle
                    "frame_budget_pct": round(100 * secs / 60.0, 2),
                    "device": str(dev),
                })
                tag = f" (search {n_searched}/{n})" if n_searched != n else ""
                print(f"[bench] {label:20s} H={H:3d} n={n:4d}{tag} -> {secs*1000:8.1f} ms "
                      f"({100*secs/60.0:.2f}% of a 60 s frame)")
    return pl.DataFrame(rows)


def per_fov_summary(df: pl.DataFrame, standard_cells: int) -> pl.DataFrame:
    """Headline table: one row per controller × horizon at a standard FOV size,
    the ms-per-FOV-frame the user asked for. Falls back to the nearest available
    cell count if the exact one wasn't swept."""
    if df.is_empty():
        return df
    sizes = df["n_cells"].unique().to_list()
    n = standard_cells if standard_cells in sizes else min(sizes, key=lambda s: abs(s - standard_cells))
    return (
        df.filter(pl.col("n_cells") == n)
        .select("controller", "horizon", "n_searched",
                (pl.col("seconds") * 1000).round(1).alias("ms_per_frame"),
                "frame_budget_pct")
        .sort("horizon", "controller")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--checkpoint", default=None,
                   help="time a real bundle (capped at its future_len) instead of "
                        "a synthetic model")
    p.add_argument("--horizons", default="1,5,10,20,30,50")
    p.add_argument("--cells", default="64,208,512")
    p.add_argument("--mpc-samples", dest="mpc_samples", default="128,512")
    p.add_argument("--stagger-k", dest="stagger_k", default="4",
                   help="k values for StaggeredCadenceMPC (comma-separated)")
    p.add_argument("--standard-cells", dest="standard_cells", type=int, default=208,
                   help="FOV size for the per-FOV summary table (v5 run was ~208)")
    p.add_argument("--out", default=None, help="write the results parquet here")
    p.add_argument("--plot", default=None, help="write a PNG of seconds vs horizon")
    args = p.parse_args()

    df = sweep(
        device=args.device,
        checkpoint=args.checkpoint,
        horizons=tuple(int(x) for x in args.horizons.split(",")),
        cell_counts=tuple(int(x) for x in args.cells.split(",")),
        mpc_samples=tuple(int(x) for x in args.mpc_samples.split(",")),
        stagger_k=tuple(int(x) for x in args.stagger_k.split(",")),
    )

    print(f"\n=== predictions per standard FOV ({args.standard_cells} cells), "
          f"ms per 60 s frame ===")
    with pl.Config(tbl_rows=-1):
        print(per_fov_summary(df, args.standard_cells))
    print("\n=== full sweep ===")
    if args.out:
        df.write_parquet(args.out)
        print(f"[bench] wrote {args.out}")
    if args.plot:
        import hastyplot as hp

        hp.line(df, x="horizon", y="seconds", color="controller", facet="n_cells",
                title="decide() per-frame wall time vs control horizon").save(args.plot)
        print(f"[bench] wrote {args.plot}")

    worst = df.filter(pl.col("frame_budget_pct") > 50)
    if worst.height:
        print("\n[bench] configurations using >50% of the frame budget:")
        print(worst)
    else:
        print(f"\n[bench] every configuration fits the 60 s frame budget "
              f"(worst: {df['frame_budget_pct'].max()}%)")


if __name__ == "__main__":
    main()
