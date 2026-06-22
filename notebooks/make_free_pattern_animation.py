"""Animate how 4 numbers generate every stimulation shape.

The free-pattern parameterisation (see ``free_pattern_lowdim_representations.ipynb``
and ``bo_erk_oscillation_testv15_free_pattern_level3.ipynb``) describes an
*arbitrary* temporal optogenetic stimulation pattern with just **four dials**:

    c1  -- tilt / ramp       (Legendre P1)
    c2  -- curvature         (Legendre P2)   hump (<0) <-> U (>0)
    c3  -- cubic / asymmetry (Legendre P3)
    pulse_interval -- timing (how many pulses, how far apart)

The three coefficients define a smooth shape on normalised time [0, 1]; a
softmax then turns that shape into per-pulse exposure times that are always
non-negative and always sum to the fixed light budget (4000 ms). The pulse
interval resamples that shape onto real frames.

This script renders a guided tour through the four dials as a GIF (and an MP4 if
an ffmpeg writer is available).

Run with the project venv:
    ../../.venv/Scripts/python.exe make_free_pattern_animation.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

# Point matplotlib at a usable ffmpeg: prefer one on PATH, else the binary
# bundled with imageio-ffmpeg (pip-installable, no system install needed).
if not matplotlib.animation.FFMpegWriter.isAvailable():
    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        print(e)
        pass

# ---------------------------------------------------------------------------
# faro BO pitch-deck dark theme (shared across the oscillation/exploitation
# animations). Semantic conventions kept consistent:
#   AMBER  -> light / stimulus (the shape & schedule we DESIGN; never a readout)
#   CORAL  -> the dial currently being turned ("the active knob / the win")
#   TAN    -> held dials, tick labels, secondary annotation
#   INK    -> headline text
# ---------------------------------------------------------------------------
BG, PANEL, INK, TAN = "#000000", "#0a0b10", "#FFFFFF", "#E0C1B3"
CORAL, AMBER, GREEN, BLUE, PERI = "#F15F47", "#FBA91E", "#6EA56C", "#5BC0EB", "#717EC3"
EDGE, GRID, GREY = "#2a2d36", "#23252e", "#7f8792"
matplotlib.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": PANEL,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": TAN, "ytick.color": TAN, "axes.edgecolor": EDGE,
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "svg.fonttype": "path",
})

# ---------------------------------------------------------------------------
# Production parameterisation -- copied verbatim from
# bo_erk_oscillation_testv15_free_pattern_level3.ipynb so the animation shows
# exactly what the microscope would deliver.
# ---------------------------------------------------------------------------
BUDGET_MS = 4000.0          # fixed total LED-on time per condition (5 s -> 4 s in v15)
FIRST_FRAME_STIM = 10       # stim window start (frame == minute)
LAST_FRAME_STIM = 70        # stim window end


def pulse_positions(pulse_interval, first=FIRST_FRAME_STIM, last=LAST_FRAME_STIM):
    """Frame indices where pulses fire: range(first, last, pulse_interval)."""
    return np.arange(first, last, int(pulse_interval))


def shape_normalized_time(n):
    """n pulse slots -> positions in [0, 1] (independent of n)."""
    if n <= 1:
        return np.array([0.5])
    return np.linspace(0.0, 1.0, n)


def shape_basis_poly(t, K=3):
    """Legendre P1..PK on [0, 1] (constant P0 dropped): tilt, curvature, cubic."""
    x = 2 * t - 1
    P = [np.ones_like(x), x]
    for k in range(2, K + 1):
        P.append(((2 * k - 1) * x * P[k - 1] - (k - 1) * P[k - 2]) / k)
    return np.column_stack([P[k] for k in range(1, K + 1)])


def shape_exposures(coeffs, n_pulses, budget=BUDGET_MS):
    """3-coef polynomial -> per-pulse exposures, softmax-normalised to budget."""
    if n_pulses <= 0:
        return np.array([])
    t = shape_normalized_time(n_pulses)
    logits = shape_basis_poly(t, 3) @ np.asarray(coeffs, dtype=float)
    w = np.exp(logits - logits.max())
    return budget * w / w.sum()


# ---------------------------------------------------------------------------
# Tour construction: a sequence of "acts", each a smooth sweep of one or more
# dials between waypoints, with eased (smoothstep) interpolation.
# ---------------------------------------------------------------------------
C_MIN, C_MAX = -4.0, 4.0
PI_MIN, PI_MAX = 1, 20


def smoothstep(n):
    """n eased fractions in [0, 1] (ease-in-out, no jerk at the joints)."""
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    return u * u * (3 - 2 * u)


def sweep(waypoints, frames_per_leg):
    """Interpolate a (c1, c2, c3, pi) path through `waypoints` with easing.

    Returns an (N, 4) array of states.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    legs = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        s = smoothstep(frames_per_leg)[:, None]
        legs.append(a[None, :] + (b[None, :] - a[None, :]) * s)
    legs.append(waypoints[-1][None, :])  # land exactly on the final waypoint
    return np.vstack(legs)


def hold(state, n):
    return np.tile(np.asarray(state, dtype=float), (n, 1))


# Each act: (title, subtitle, highlighted_dial_index_or_None, state_array)
FPL = 18  # frames per leg

ACTS = [
    (
        "Four dials → every stimulation shape",
        "fixed 4000 ms budget, always split across the pulses",
        None,
        hold([0, 0, 0, 5], 22),
    ),
    (
        "Dial 1  —  c1: tilt / ramp",
        "weak→strong (c1>0)   vs   strong→weak (c1<0)",
        0,
        sweep([[0, 0, 0, 5], [3, 0, 0, 5], [-3, 0, 0, 5], [0, 0, 0, 5]], FPL),
    ),
    (
        "Dial 2  —  c2: curvature",
        "centre hump / sustain (c2<0)   vs   ends-loaded U (c2>0)",
        1,
        sweep([[0, 0, 0, 5], [0, -3, 0, 5], [0, 3, 0, 5], [0, 0, 0, 5]], FPL),
    ),
    (
        "Dial 3  —  c3: cubic / asymmetry",
        "fast-up/slow-down   vs   slow-up/fast-down",
        2,
        sweep([[0, 0, 0, 5], [0, -1.5, 2.5, 5], [0, -1.5, -2.5, 5], [0, 0, 0, 5]], FPL),
    ),
    (
        "Dial 4  —  pulse_interval: timing",
        "summation→sustained (pi<5 min)   vs   discrete peaks (pi>5 min)",
        3,
        sweep([[0, -2, 0, 2], [0, -2, 0, 12], [0, -2, 0, 2]], FPL * 2),
    ),
    (
        "All four together → the full shape space",
        "ramp, hump, plateau, U, asymmetry, bursts — from 4 numbers",
        None,
        sweep(
            [
                [0, 0, 0, 5],
                [1.5, -2.0, 1.0, 4],
                [-2.0, 1.5, -1.5, 8],
                [3.0, 2.0, 0.0, 3],
                [0.0, -3.0, 2.0, 6],
                [-1.5, -1.5, -2.0, 10],
                [2.5, -2.5, 1.5, 4],
                [0, 0, 0, 5],
            ],
            FPL,
        ),
    ),
]

# Flatten into per-frame metadata + states.
STATES = []
META = []  # (title, subtitle, highlight) per frame
for title, subtitle, hl, arr in ACTS:
    for st in arr:
        STATES.append(st)
        META.append((title, subtitle, hl))
STATES = np.asarray(STATES)
N_FRAMES = len(STATES)

DIAL_LABELS = ["c1  tilt", "c2  curve", "c3  cubic", "pi  interval"]
DIAL_RANGES = [(C_MIN, C_MAX), (C_MIN, C_MAX), (C_MIN, C_MAX), (PI_MIN, PI_MAX)]

# Fine grid for the continuous shape (peak-normalised, relative).
TF = np.linspace(0, 1, 250)
PHI_FINE = shape_basis_poly(TF, 3)


def setup_figure():
    fig = plt.figure(figsize=(11.5, 5.4), dpi=300)
    fig.patch.set_facecolor(BG)
    gs = GridSpec(
        2, 2, figure=fig, width_ratios=[1.0, 1.05], height_ratios=[1.0, 0.5],
        hspace=0.55, wspace=0.28,
        left=0.07, right=0.93, top=0.91, bottom=0.13,
    )
    ax_shape = fig.add_subplot(gs[0, 0])
    ax_dials = fig.add_subplot(gs[1, 0])
    ax_sched = fig.add_subplot(gs[:, 1])
    return fig, ax_shape, ax_dials, ax_sched


def render(fig, ax_shape, ax_dials, ax_sched, frame):
    c1, c2, c3, pi = STATES[frame]
    pi = int(round(pi))
    coeffs = np.array([c1, c2, c3])
    _title, _subtitle, hl = META[frame]  # title/subtitle added later in PowerPoint

    # --- continuous shape (relative, peak-normalised) -- it's light we DESIGN -> AMBER ---
    shape = np.exp(PHI_FINE @ coeffs)
    shape /= shape.max()
    ax_shape.clear()
    ax_shape.set_facecolor(PANEL)
    ax_shape.plot(TF, shape, color=AMBER, lw=2.4)
    ax_shape.fill_between(TF, shape, color=AMBER, alpha=0.16)
    ax_shape.set_ylim(0, 1.06)
    ax_shape.set_xlim(0, 1)
    ax_shape.set_xlabel("normalized time", fontsize=9)
    ax_shape.set_ylabel("relative exposure", fontsize=9)
    ax_shape.set_title(
        f"continuous shape   c = [{c1:+.2f}, {c2:+.2f}, {c3:+.2f}]",
        fontsize=10, color=INK,
    )
    ax_shape.grid(color=GRID, alpha=0.9, lw=0.8)
    ax_shape.tick_params(labelsize=8)

    # --- dial gauges (each dial mapped onto a common 0..1 track) ---
    # active dial -> CORAL (the knob being turned), held dials -> TAN.
    ax_dials.clear()
    ax_dials.set_facecolor(PANEL)
    vals = [c1, c2, c3, pi]
    ypos = np.arange(4)[::-1]
    T0, T1 = 0.0, 1.0  # common track span in axis coords
    for y, label, val, (lo, hi) in zip(ypos, DIAL_LABELS, vals, DIAL_RANGES):
        is_hl = (hl is not None) and (label.startswith(DIAL_LABELS[hl][:2]))
        bar_c = CORAL if is_hl else TAN
        txt_c = CORAL if is_hl else TAN
        frac = (val - lo) / (hi - lo)          # 0..1 position of the value
        x = T0 + frac * (T1 - T0)
        ax_dials.barh(y, T1 - T0, left=T0, height=0.42, color=EDGE, zorder=1)
        # zero / mid reference tick
        mid = 0 if lo < 0 < hi else (lo + hi) / 2
        xmid = T0 + (mid - lo) / (hi - lo) * (T1 - T0)
        ax_dials.plot([xmid, xmid], [y - 0.24, y + 0.24], color=GREY, lw=1, zorder=2)
        ax_dials.plot([x], [y], "o", ms=11, color=bar_c, zorder=3,
                      markeredgecolor=BG, markeredgewidth=1.2)
        vtxt = f"{int(round(val))} min" if label.startswith("pi") else f"{val:+.2f}"
        ax_dials.text(T0 - 0.04, y, label, ha="right", va="center",
                      fontsize=8.5, color=txt_c,
                      fontweight="bold" if is_hl else "normal")
        ax_dials.text(T1 + 0.04, y, vtxt, ha="left", va="center",
                      fontsize=8.5, color=txt_c, fontfamily="monospace")
    ax_dials.set_xlim(-0.42, 1.42)
    ax_dials.set_ylim(-0.6, 3.6)
    ax_dials.axis("off")
    ax_dials.set_title("the four numbers", fontsize=9.5, color=TAN, pad=2)

    # --- realized schedule (what the microscope delivers) -- light pulses -> AMBER ---
    frames = pulse_positions(pi)
    expo = shape_exposures(coeffs, len(frames))
    ax_sched.clear()
    ax_sched.set_facecolor(PANEL)
    ax_sched.axvspan(FIRST_FRAME_STIM, LAST_FRAME_STIM, color=AMBER, alpha=0.06)
    if len(frames):
        ax_sched.vlines(frames, 0, expo, color=AMBER, lw=3)
        ax_sched.scatter(frames, expo, color=AMBER, s=26, zorder=5,
                         edgecolors=BG, linewidths=0.6)
    ax_sched.set_xlim(FIRST_FRAME_STIM - 2, LAST_FRAME_STIM + 1)
    ax_sched.set_ylim(0, 1400)
    ax_sched.set_xlabel("frame  (minutes)", fontsize=9)
    ax_sched.set_ylabel("exposure per pulse (ms)", fontsize=9)
    peak = expo.max() if len(expo) else 0.0
    ax_sched.set_title(
        f"delivered schedule   pi = {pi} min   →   "
        f"{len(frames)} pulses,  Σ = {expo.sum():.0f} ms,  peak = {peak:.0f} ms",
        fontsize=10, color=INK,
    )
    ax_sched.grid(color=GRID, alpha=0.9, lw=0.8)
    ax_sched.tick_params(labelsize=8)

    # No figure headline/subtitle by design -- titles are added later in
    # PowerPoint. (`title`/`subtitle` from META still drive which dial is
    # highlighted via `hl`.)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fig, ax_shape, ax_dials, ax_sched = setup_figure()

    def update(i):
        render(fig, ax_shape, ax_dials, ax_sched, i)
        return []

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=91, blit=False)

    gif_path = os.path.join(here, "free_pattern_shapes.gif")
    print(f"Rendering {N_FRAMES} frames -> {gif_path}")
    anim.save(gif_path, writer=PillowWriter(fps=11))
    print(f"  wrote GIF: {gif_path}  ({os.path.getsize(gif_path) / 1e6:.2f} MB)")

    # MP4 if an ffmpeg-backed writer is available; skip cleanly otherwise.
    try:
        from matplotlib.animation import FFMpegWriter

        if FFMpegWriter.isAvailable():
            mp4_path = os.path.join(here, "free_pattern_shapes.mp4")
            # H.264 + yuv420p so it plays in Windows Media Player / PowerPoint.
            writer = FFMpegWriter(
                fps=11, codec="libx264", bitrate=2400,
                extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(mp4_path, writer=writer)
            print(f"  wrote MP4: {mp4_path}")
        else:
            print("  (ffmpeg not found -> skipped MP4; GIF only)")
    except Exception as exc:  # noqa: BLE001
        print(f"  (MP4 skipped: {exc})")

    plt.close(fig)


if __name__ == "__main__":
    main()
