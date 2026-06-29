"""Generate mp4 videos walking a trained model through full single-cell trajectories.

Default behaviour: **one video per per-cell condition**, each video a 2×2 grid
of panels — one panel per response-magnitude quartile (Q1..Q4 of ``std(cnr)``
*within that condition*). One cell per panel by default; ``--n-per-bucket k``
renders ``k`` videos per condition (set0..set{k-1}), each cycling through a
different cell pick. Each frame shows the true trajectory plus the model's
``future_len``-step prediction with bands from the GMM's per-step ``pred_std``
(no MC dropout — matches notebook eval semantics).

Both prediction and per-frame plotting are pluggable. ``predict_fn`` is
required (model contracts differ); reference impls for the MDN minfeats family
live in this module (``predict_mdn_minfeats``, ``predict_mdn_minfeats_ewma``).
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from tqdm.auto import tqdm

from experiment import ExperimentBundle, load_experiment
from experiments import seq2seq_data
from experiments.seq2seq_data import STIM_COLS
from utils import get_device


# --------------------------------------------------------------------- types

@dataclass
class CellData:
    idx: int
    cnr: np.ndarray            # (T,) absolute CNR
    stim: np.ndarray           # (n_stim, T)
    condition: str
    stratum: str               # "q1" | "q2" | "q3" | "q4"


PredictFn = Callable[
    ["torch.nn.Module", CellData, int, int, int, torch.device],
    tuple[np.ndarray, np.ndarray],   # (mean_abs (F,), sigma_abs (F,))
]

FrameFn = Callable[
    ["plt.Axes", CellData, int, int, np.ndarray, np.ndarray, "int | None", "tuple[float, float]"],
    None,
]


# ------------------------------------------------- marimo class resolution

def _infer_experiment_module(result_path: Path) -> str | None:
    """Parse ``<result_path>/slurm.log`` for the ``Notebook :`` header.

    Header line looks like ``Notebook   : experiments/foo.py``; converted to the
    importable module path ``experiments.foo``. Returns None if absent.
    """
    log = result_path / "slurm.log"
    if not log.exists():
        return None
    with log.open() as f:
        for line in f:
            if line.startswith("Notebook"):
                _, _, rhs = line.partition(":")
                nb = rhs.strip()
                if not nb:
                    return None
                return nb.removesuffix(".py").replace("/", ".")
            if line.startswith("=="):
                continue
            if "Experiment" in line or "Args" in line:
                continue
            # Header ends within the first ~10 lines; bail if we're past it.
    return None


def _resolve_main_model(
    bundle: ExperimentBundle,
    experiment_module: str | None,
) -> "torch.nn.Module":
    """Reconstruct the model, optionally pulling its class from a named module.

    Bundles trained inside a marimo notebook record ``model_type='__main__.<Cls>'``.
    Caller must specify ``experiment_module`` (e.g.
    ``"experiments.lstm_seq2scal_mdn_minfeats_image_ewma"``) so the right class
    is registered under ``__main__`` before :meth:`reconstruct_model` runs.
    """
    mt = bundle.model_type or ""
    if not mt.startswith("__main__."):
        return bundle.reconstruct_model()

    if experiment_module is None:
        raise RuntimeError(
            f"Bundle model_type is {mt!r} (class lived in __main__ at train time). "
            "Pass experiment_module='experiments.<notebook>' so cell_video can "
            "locate the class."
        )

    class_name = mt.split(".", 1)[1]
    mod = importlib.import_module(experiment_module)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"Module {experiment_module!r} has no attribute {class_name!r}. "
            f"Check that the notebook defines it in an `app.setup` block."
        )
    main_mod = sys.modules.setdefault("__main__", type(sys)("__main__"))
    setattr(main_mod, class_name, cls)
    return bundle.reconstruct_model()


# ----------------------------------------------------------- stratification

def stratify_by_std(
    cnr_tracks: Sequence[np.ndarray],
    indices: Sequence[int] | None = None,
    n_strata: int = 4,
) -> list[np.ndarray]:
    """Quantile bins of per-cell ``std(cnr)``, returned low→high.

    If ``indices`` is given, stratification is done *within* that subset and the
    returned arrays contain values from ``indices`` — useful for stratifying
    inside a per-condition subgroup. With fewer cells than bins, some bins may
    come back empty (caller should skip them).
    """
    if indices is None:
        pool = np.arange(len(cnr_tracks))
    else:
        pool = np.asarray(list(indices), dtype=int)
    if len(pool) == 0:
        return [np.array([], dtype=int) for _ in range(n_strata)]
    scores = np.array([np.std(np.asarray(cnr_tracks[i])) for i in pool])
    edges = np.quantile(scores, np.linspace(0.0, 1.0, n_strata + 1))
    # nudge endpoints so digitize includes the extremes.
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(scores, edges[1:-1])
    return [pool[bin_idx == k] for k in range(n_strata)]


def pick_per_quartile(
    quartiles: list[np.ndarray],
    n_per_bucket: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Sample ``min(n_per_bucket, |q|)`` cell indices from each quartile.

    Returns a list of 4 lists (one per quartile, in Q1..Q4 order). Empty
    quartiles return ``[]`` and produce blank panels downstream.
    """
    out: list[list[int]] = []
    for qpool in quartiles:
        k = min(n_per_bucket, len(qpool))
        if k == 0:
            out.append([])
        else:
            out.append([int(x) for x in rng.choice(qpool, size=k, replace=False)])
    return out


def _safe_filename(s: str) -> str:
    """Reduce a condition label to something safe for a filename."""
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "unlabeled"


# ---------------------------------------------------- reference predict_fn

def _gmm_pred_std(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-step marginal std of a Gaussian mixture: sqrt((π·(σ² + (μ - μ̄)²)).sum(-1))."""
    mean = (pi * mu).sum(dim=-1, keepdim=True)
    var = (pi * (sigma**2 + (mu - mean) ** 2)).sum(dim=-1)
    return torch.sqrt(var.clamp(min=1e-12))


def predict_mdn_minfeats_ewma(
    model: "torch.nn.Module",
    cell: CellData,
    t: int,
    future_len: int,
    history_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Single deterministic forward (no dropout) for ``Seq2ScalarMDN`` (minfeats + EWMA).

    Encoder window features (5): ``[cnr, fluence, baseline, ewma_slow, ewma_fast]``
    with ``baseline = median(cnr[:9])``; EWMA channels read from ``stim``.
    Predicts at most ``future_len`` steps ahead (clipped at trajectory end).

    Mean trajectory: ``last_val + cumsum(E[delta_k])``.
    Sigma: cumulative ``sqrt(cumsum(pred_std_k²))`` — uncertainty about the absolute
    CNR at step k under the (approx.) independence assumption used by the eval
    notebooks for ``pred_std``.
    """
    H = history_len
    T = cell.cnr.shape[0]
    if t < H or t >= T:
        raise ValueError(f"t={t} out of bounds for history_len={H}, T={T}")
    F = min(future_len, T - t)

    u_idx = STIM_COLS.index("u_t")
    s_idx = STIM_COLS.index("ewma_slow")
    f_idx = STIM_COLS.index("ewma_fast")

    cnr = cell.cnr.astype(np.float32)
    flu = cell.stim[u_idx].astype(np.float32)
    ewma_s = cell.stim[s_idx].astype(np.float32)
    ewma_f = cell.stim[f_idx].astype(np.float32)
    baseline = float(np.median(cnr[:9]))

    enc_in = np.stack(
        [
            cnr[t - H:t],
            flu[t - H:t],
            np.full(H, baseline, dtype=np.float32),
            ewma_s[t - H:t],
            ewma_f[t - H:t],
        ],
        axis=-1,
    )                                                       # (H, 5)
    dec_stim = flu[t:t + F, None]                           # (F, 1)

    enc_t = torch.from_numpy(enc_in).float().unsqueeze(0).to(device)
    dec_t = torch.from_numpy(dec_stim).float().unsqueeze(0).to(device)
    last_val = float(cnr[t - 1])

    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(enc_t, dec_t)                 # each (1, F, K)
        mean_delta = (pi * mu).sum(dim=-1).cpu().numpy()[0]
        sigma_step = _gmm_pred_std(pi, mu, sigma).cpu().numpy()[0]

    abs_mean = last_val + np.cumsum(mean_delta)
    abs_sigma = np.sqrt(np.cumsum(sigma_step ** 2))
    return abs_mean.astype(np.float32), abs_sigma.astype(np.float32)


def predict_mdn_minfeats(
    model: "torch.nn.Module",
    cell: CellData,
    t: int,
    future_len: int,
    history_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic forward for the no-EWMA minfeats variant (encoder: 3 features)."""
    H = history_len
    T = cell.cnr.shape[0]
    if t < H or t >= T:
        raise ValueError(f"t={t} out of bounds for history_len={H}, T={T}")
    F = min(future_len, T - t)

    u_idx = STIM_COLS.index("u_t")
    cnr = cell.cnr.astype(np.float32)
    flu = cell.stim[u_idx].astype(np.float32)
    baseline = float(np.median(cnr[:9]))

    enc_in = np.stack(
        [
            cnr[t - H:t],
            flu[t - H:t],
            np.full(H, baseline, dtype=np.float32),
        ],
        axis=-1,
    )                                                       # (H, 3)
    dec_stim = flu[t:t + F, None]                           # (F, 1)

    enc_t = torch.from_numpy(enc_in).float().unsqueeze(0).to(device)
    dec_t = torch.from_numpy(dec_stim).float().unsqueeze(0).to(device)
    last_val = float(cnr[t - 1])

    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(enc_t, dec_t)
        mean_delta = (pi * mu).sum(dim=-1).cpu().numpy()[0]
        sigma_step = _gmm_pred_std(pi, mu, sigma).cpu().numpy()[0]

    abs_mean = last_val + np.cumsum(mean_delta)
    abs_sigma = np.sqrt(np.cumsum(sigma_step ** 2))
    return abs_mean.astype(np.float32), abs_sigma.astype(np.float32)


def predict_seq2seq_deltas(
    model: "torch.nn.Module",
    cell: CellData,
    t: int,
    future_len: int,
    history_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic delta predictor for the vanilla seq2seq / seq2scal notebooks.

    Encoder input: ``[cnr, *STIM_COLS]`` over the H-step history; decoder consumes
    the full stim-feature vector over the F-step future. Model returns ``(B, F)``
    per-step CNR deltas. No aleatoric output, so ``sigma`` comes back as zeros —
    the video bands collapse to the mean line, matching what those models give us.
    """
    H = history_len
    T = cell.cnr.shape[0]
    if t < H or t >= T:
        raise ValueError(f"t={t} out of bounds for history_len={H}, T={T}")
    F = min(future_len, T - t)

    cnr = cell.cnr.astype(np.float32)
    stim = cell.stim.astype(np.float32)  # (n_stim, T)

    enc_in = np.concatenate(
        [cnr[t - H:t, None], stim[:, t - H:t].T], axis=-1,
    )                                                       # (H, 1 + n_stim)
    dec_stim = stim[:, t:t + F].T                           # (F, n_stim)

    enc_t = torch.from_numpy(enc_in).float().unsqueeze(0).to(device)
    dec_t = torch.from_numpy(dec_stim).float().unsqueeze(0).to(device)
    last_val = float(cnr[t - 1])

    model.eval()
    with torch.no_grad():
        delta = model(enc_t, dec_t).cpu().numpy()[0]        # (F,)

    abs_mean = last_val + np.cumsum(delta)
    abs_sigma = np.zeros_like(abs_mean, dtype=np.float32)
    return abs_mean.astype(np.float32), abs_sigma


def _predict_seq2scal_minfeats(
    model, cell, t, future_len, history_len, device, *, absolute
):
    """Correct predictor for the seq2scal_models(_abs) + FiLM family.

    These models train on the 5 minfeats ``[cnr, fluence, baseline,
    ewma_slow(cnr), ewma_fast(cnr)]`` where the EWMA channels are EWMAs of the
    *CNR* (not the light) — so we recompute them here from ``cell.cnr`` using the
    model's own ``cfg`` alphas, rather than reading the light-EWMA stim channels.
    ``absolute``: the abs model emits the future CNR directly (no delta cumsum).
    """
    from experiments.seq2scal_models import _ewma_1d

    H = history_len
    cnr = cell.cnr.astype(np.float32)
    T = cnr.shape[0]
    if t < H or t >= T:
        raise ValueError(f"t={t} out of bounds for history_len={H}, T={T}")
    F = min(future_len, T - t)

    flu = cell.stim[STIM_COLS.index("u_t")].astype(np.float32)
    cfg = getattr(model, "cfg", None)
    a_slow = float(getattr(cfg, "ewma_slow_alpha", 0.05))
    a_fast = float(getattr(cfg, "ewma_fast_alpha", 0.30))
    ewma_s = _ewma_1d(cnr, a_slow)
    ewma_f = _ewma_1d(cnr, a_fast)
    baseline = float(np.median(cnr[: min(10, T)]))

    enc_in = np.stack(
        [
            cnr[t - H:t],
            flu[t - H:t],
            np.full(H, baseline, dtype=np.float32),
            ewma_s[t - H:t],
            ewma_f[t - H:t],
        ],
        axis=-1,
    )
    dec_stim = flu[t:t + F, None]
    enc_t = torch.from_numpy(enc_in).float().unsqueeze(0).to(device)
    dec_t = torch.from_numpy(dec_stim).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(enc_t, dec_t)
        mean = (pi * mu).sum(dim=-1).cpu().numpy()[0]              # (F,)
        step_std = _gmm_pred_std(pi, mu, sigma).cpu().numpy()[0]   # (F,)

    if absolute:
        abs_mean = mean
        abs_sigma = step_std
    else:
        last_val = float(cnr[t - 1])
        abs_mean = last_val + np.cumsum(mean)
        abs_sigma = np.sqrt(np.cumsum(step_std ** 2))
    return abs_mean.astype(np.float32), abs_sigma.astype(np.float32)


def predict_seq2scal_minfeats_delta(model, cell, t, future_len, history_len, device):
    return _predict_seq2scal_minfeats(model, cell, t, future_len, history_len, device, absolute=False)


def predict_seq2scal_minfeats_abs(model, cell, t, future_len, history_len, device):
    return _predict_seq2scal_minfeats(model, cell, t, future_len, history_len, device, absolute=True)


PREDICT_FN_BY_MODULE: dict[str, PredictFn] = {
    "experiments.lstm_seq2scal_mdn_minfeats": predict_mdn_minfeats,
    "experiments.lstm_seq2scal_mdn_minfeats_image_ewma": predict_mdn_minfeats_ewma,
    # seq2scal_variant / _abs train on EWMA-of-CNR minfeats (recomputed in the
    # predictor from the model's cfg alphas). variant = delta output, abs = absolute.
    "experiments.lstm_seq2scal_variant": predict_seq2scal_minfeats_delta,
    "experiments.lstm_seq2scal_abs": predict_seq2scal_minfeats_abs,
    # Deterministic notebooks: forward returns (B, F) future deltas directly.
    "experiments.lstm_seq2seq": predict_seq2seq_deltas,
    "experiments.lstm_seq2scal": predict_seq2seq_deltas,
}


# ------------------------------------------------------------ default frame

def default_frame_fn(
    ax: "plt.Axes",
    cell: CellData,
    t: int,
    history_len: int,
    mean: np.ndarray,
    sigma: np.ndarray,
    display_history: int | None,
    ylim: tuple[float, float],
) -> None:
    """True traj + GMM-mean prediction + 1σ/2σ bands. ``display_history`` clips left x-axis."""
    T = cell.cnr.shape[0]
    time = np.arange(T)
    F = mean.shape[0]
    fut = time[t:t + F]

    ax.clear()
    ax.plot(time, cell.cnr, color="black", lw=1.2, label="true CNR")
    ax.axvline(t, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axvspan(t - history_len, t, color="C0", alpha=0.08, label="context")

    ax.fill_between(fut, mean - 2 * sigma, mean + 2 * sigma, color="C3", alpha=0.15, label="±2σ")
    ax.fill_between(fut, mean - sigma, mean + sigma, color="C3", alpha=0.30, label="±1σ")
    ax.plot(fut, mean, color="C3", lw=1.5, label="pred")

    u_idx = STIM_COLS.index("u_t") if "u_t" in STIM_COLS else 0
    u_t = cell.stim[u_idx]
    if u_t.max() > u_t.min():
        ax2 = ax.twinx()
        ax2.fill_between(time, 0, u_t, color="gold", alpha=0.25, step="pre")
        ax2.set_ylim(0, max(u_t.max() * 4, 1e-3))
        ax2.set_yticks([])

    if display_history is not None:
        x_lo = max(0, t - display_history)
        ax.set_xlim(x_lo, T - 1)
    else:
        ax.set_xlim(0, T - 1)
    ax.set_ylim(*ylim)
    ax.set_xlabel("frame")
    ax.set_ylabel("CNR")
    ax.set_title(
        f"[{cell.stratum.upper()}] cell #{cell.idx}  t={t}/{T - 1}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7)


# ------------------------------------------------------------- main entry

def _render_one_video(
    panel_cells: list[CellData | None],
    *,
    predict_fn: PredictFn,
    frame_fn: FrameFn,
    model,
    history_len: int,
    future_len: int,
    stride: int,
    display_history: int | None,
    ylim: tuple[float, float],
    fps: int,
    dpi: int,
    device,
    title: str,
    out_path: Path,
) -> Path:
    """Render a 2×2 grid mp4 for up to 4 cells (None entries leave blank panels).

    Panels are laid out:
        TL=Q1   TR=Q2
        BL=Q3   BR=Q4
    Frame schedule advances each panel independently — a panel that runs out of
    trajectory just stays frozen on its last frame.
    """
    # Each panel's frame schedule (list of ts). Panels with None or too-short
    # trajectories contribute an empty schedule.
    schedules: list[list[int]] = []
    for cell in panel_cells:
        if cell is None or cell.cnr.shape[0] <= history_len + 1:
            schedules.append([])
            continue
        T = cell.cnr.shape[0]
        schedules.append(list(range(history_len, T - 1, max(stride, 1))))

    n_frames = max((len(s) for s in schedules), default=0)
    if n_frames == 0:
        print(f"[cell_video] skipping {out_path.name}: no panel had a usable schedule")
        return out_path

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=dpi)
    fig.suptitle(title, fontsize=12)
    axes_flat = list(axes.flat)

    def render(frame_i: int):
        for panel_i, cell in enumerate(panel_cells):
            ax = axes_flat[panel_i]
            if cell is None or not schedules[panel_i]:
                ax.clear()
                ax.set_axis_off()
                ax.set_title(f"[Q{panel_i + 1}] (no cell)", fontsize=10)
                continue
            sched = schedules[panel_i]
            t = sched[min(frame_i, len(sched) - 1)]
            mean, sigma = predict_fn(model, cell, t, future_len, history_len, device)
            frame_fn(ax, cell, t, history_len, mean, sigma, display_history, ylim)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    anim = FuncAnimation(fig, render, frames=n_frames, interval=1000 // max(fps, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: FFMpegWriter | PillowWriter
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    else:
        gif_path = out_path.with_suffix(".gif")
        print(f"[cell_video] ffmpeg not on PATH; writing {gif_path} instead")
        out_path = gif_path
        writer = PillowWriter(fps=fps)

    pbar = tqdm(total=n_frames, desc=out_path.name, unit="frame", leave=False)
    try:
        anim.save(
            str(out_path),
            writer=writer,
            dpi=dpi,
            progress_callback=lambda i, n: pbar.update(1),
        )
    finally:
        pbar.close()
        plt.close(fig)
    return out_path


def make_cell_video(
    result_path: str | Path,
    predict_fn: PredictFn,
    *,
    experiment_module: str | None = None,
    dataset_name: str | None = None,
    n_per_bucket: int = 1,
    conditions_filter: Sequence[str] | None = None,
    future_len: int | None = None,
    fps: int = 8,
    stride: int = 1,
    history_len: int | None = None,
    display_history: int | None = None,
    ylim: tuple[float, float] | None = None,
    ylim_pad: float = 0.15,
    frame_fn: FrameFn | None = None,
    out_dir: str | Path | None = None,
    seed: int = 0,
    dpi: int = 100,
    holdout_only: bool = True,
    split_seed: int = 42,
    baseline_prepend: int | None = None,
) -> list[Path]:
    """Render one mp4 per per-cell condition, each a 2×2 grid of quartile panels.

    By default only **held-out test cells** are rendered (``holdout_only=True``),
    reproducing the project-standard split
    ``train_test_split(arange(n), test_size=0.2, random_state=split_seed)`` that
    every seq2seq/seq2scal notebook uses — so videos never show cells the model
    trained on. Pass ``holdout_only=False`` to render all cells.

    Parameters
    ----------
    result_path
        Directory of a saved experiment (loadable by :func:`load_experiment`).
    predict_fn
        Required. ``(model, cell, t, future_len, history_len, device) → (mean (F,), sigma (F,))``
        absolute CNR prediction with cumulative-variance uncertainty.
    experiment_module
        Module containing the model class for ``__main__``-saved bundles.
        Inferred from ``<result_path>/slurm.log`` if omitted.
    dataset_name
        Override the dataset; defaults to ``training_config["data_source"]``.
    n_per_bucket
        Cells per (condition, quartile). ``1`` (default) = one mp4 per
        condition. ``k > 1`` = ``k`` mp4s per condition (set0..set{k-1}),
        each picking a different cell per panel.
    conditions_filter
        Whitelist of condition labels to render. Default: render all.
    out_dir
        Directory for the family of mp4s. Default: ``<result_path>/videos``.

    Other args
    ----------
    future_len, stride, history_len, display_history, ylim, ylim_pad, fps, dpi —
        see module docstring; defaults pulled from the bundle's configs.

    Returns
    -------
    List of written paths.
    """
    result_path = Path(result_path)
    bundle_dir = result_path
    if not (result_path / "bundle.pt").exists() and (result_path / "checkpoints" / "bundle.pt").exists():
        bundle_dir = result_path / "checkpoints"
        print(f"[cell_video] no top-level bundle.pt; loading from {bundle_dir}")
    bundle: ExperimentBundle = load_experiment(str(bundle_dir))

    if dataset_name is None:
        dataset_name = (
            bundle.training_config.get("data_source")
            or bundle.model_config.get("data_source")
            or "real"
        )
    if history_len is None:
        history_len = int(
            bundle.model_config.get("history_len")
            or bundle.training_config["history_len"]
        )
    if future_len is None:
        future_len = int(
            bundle.model_config.get("future_len")
            or bundle.training_config.get("future_len", 5)
        )

    if experiment_module is None:
        experiment_module = _infer_experiment_module(result_path)
        if experiment_module is not None:
            print(f"[cell_video] inferred experiment_module={experiment_module!r} from slurm.log")

    device = get_device()
    model = _resolve_main_model(bundle, experiment_module).to(device)

    # Match the training-time baseline prepend so the video shows the same
    # onset-inclusive trajectories the model was trained/evaluated on.
    if baseline_prepend is None:
        baseline_prepend = history_len if bundle.model_config.get("prepend_baseline") else 0
    if baseline_prepend:
        print(f"[cell_video] baseline_prepend={baseline_prepend} (prepended baseline frames)")
    cnr_arr, stim_arr, conditions = seq2seq_data.load(
        dataset_name, baseline_prepend=baseline_prepend
    )
    cnr_list = [np.asarray(c, dtype=np.float32) for c in cnr_arr]
    if isinstance(stim_arr, np.ndarray) and stim_arr.dtype != object:
        stim_list = [stim_arr[i].astype(np.float32) for i in range(len(stim_arr))]
    else:
        stim_list = [np.asarray(s, dtype=np.float32) for s in stim_arr]

    keep = [i for i, c in enumerate(cnr_list) if len(c) > history_len + 1]
    if holdout_only:
        # Prefer the bundle's persisted split (correct for any regime, incl.
        # condition_held_out); fall back to reproducing the random 70/10/20.
        _persisted = bundle.metrics.get("splits") if isinstance(bundle.metrics, dict) else None
        if _persisted and "test_indist" in _persisted:
            _test_set = {int(i) for i in _persisted.get("test_indist", [])}
            for _cids in (_persisted.get("test_ood") or {}).values():
                _test_set |= {int(i) for i in _cids}
            _source = f"bundle.splits ({_persisted.get('regime', '?')})"
        else:
            from sklearn.model_selection import train_test_split

            _, _test_ids = train_test_split(
                np.arange(len(cnr_list)), test_size=0.2, random_state=split_seed
            )
            _test_set = {int(i) for i in _test_ids}
            _source = f"reproduced random split (seed={split_seed})"
        before = len(keep)
        keep = [i for i in keep if i in _test_set]
        print(
            f"[cell_video] holdout_only: {len(keep)}/{before} cells are held out "
            f"via {_source}; rendering only those"
        )
    cnr_list = [cnr_list[i] for i in keep]
    stim_list = [stim_list[i] for i in keep]
    conditions = np.asarray(conditions)[keep]

    unique_conditions = sorted({str(c) for c in conditions})
    if conditions_filter is not None:
        keepset = set(conditions_filter)
        unique_conditions = [c for c in unique_conditions if c in keepset]
        missing = sorted(keepset - set(unique_conditions))
        if missing:
            print(f"[cell_video] warning: requested conditions not in dataset: {missing}")

    if not unique_conditions:
        raise RuntimeError(
            f"No conditions to render from dataset {dataset_name!r} "
            f"(filter={conditions_filter})."
        )

    _frame_fn = frame_fn or default_frame_fn
    if ylim is None:
        all_cnr = np.concatenate(cnr_list)
        lo, hi = float(all_cnr.min()), float(all_cnr.max())
        pad = (hi - lo) * ylim_pad
        # Cap the upper bound at 4: rare cells with extreme CNR spikes blow
        # the y-axis up to ~12 and squash all the interesting dynamics flat.
        ylim = (lo - pad, min(hi + pad, 4.0))

    if out_dir is None:
        out_dir = result_path / "videos"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    written: list[Path] = []

    print(
        f"[cell_video] dataset={dataset_name}  n_cells={len(cnr_list)}  "
        f"H={history_len} F={future_len}  conditions={len(unique_conditions)}  "
        f"n_per_bucket={n_per_bucket}  out_dir={out_dir}"
    )

    for cond in unique_conditions:
        cond_idx = np.where(conditions == cond)[0]
        if len(cond_idx) == 0:
            continue
        quartiles = stratify_by_std(cnr_list, indices=cond_idx, n_strata=4)
        picks = pick_per_quartile(quartiles, n_per_bucket, rng)
        bucket_counts = [len(q) for q in quartiles]
        print(
            f"[cell_video] condition={cond!r}  n_cells={len(cond_idx)}  "
            f"quartile sizes Q1..Q4 = {bucket_counts}"
        )

        for set_i in range(n_per_bucket):
            panel_cells: list[CellData | None] = []
            for q_i in range(4):
                if set_i < len(picks[q_i]):
                    i = picks[q_i][set_i]
                    panel_cells.append(
                        CellData(
                            idx=i,
                            cnr=cnr_list[i],
                            stim=stim_list[i],
                            condition=str(conditions[i]),
                            stratum=f"q{q_i + 1}",
                        )
                    )
                else:
                    panel_cells.append(None)

            tag = f"_set{set_i}" if n_per_bucket > 1 else ""
            out_path = out_dir / f"video_{_safe_filename(cond)}{tag}.mp4"
            title = f"{cond}{'  · set ' + str(set_i) if n_per_bucket > 1 else ''}"
            written_path = _render_one_video(
                panel_cells,
                predict_fn=predict_fn,
                frame_fn=_frame_fn,
                model=model,
                history_len=history_len,
                future_len=future_len,
                stride=stride,
                display_history=display_history,
                ylim=ylim,
                fps=fps,
                dpi=dpi,
                device=device,
                title=title,
                out_path=out_path,
            )
            written.append(written_path)

    print(f"[cell_video] wrote {len(written)} videos under {out_dir}")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", nargs="?", default=None)
    parser.add_argument(
        "--result-path",
        dest="result_path_flag",
        default=None,
        help="Alternative to positional; convenient for --key value launchers.",
    )
    parser.add_argument(
        "--experiment-module",
        default=None,
        help="Module that defines the bundle's model class. Inferred from slurm.log if omitted.",
    )
    parser.add_argument("--future-len", type=int, default=None,
                        help="Steps ahead per frame; defaults to model's training future_len.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--display-history", type=int, default=None)
    parser.add_argument("--n-per-bucket", type=int, default=1,
                        help="Cells per (condition, quartile). >1 renders that many mp4s per condition.")
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated whitelist of condition labels.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <result_path>/videos).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-cells", action="store_true",
                        help="Render all cells, not just the held-out test split (default: test only).")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="random_state for the train/test split (must match training; default 42).")
    args = parser.parse_args()
    result_path = args.result_path or args.result_path_flag
    if not result_path:
        parser.error("result_path required (positional or --result-path).")

    exp_mod = args.experiment_module or _infer_experiment_module(Path(result_path))
    if exp_mod is None:
        parser.error("Could not infer --experiment-module from slurm.log; pass it explicitly.")
    if exp_mod not in PREDICT_FN_BY_MODULE:
        parser.error(
            f"No predict_fn registered for {exp_mod!r}. "
            f"Known: {sorted(PREDICT_FN_BY_MODULE)}. "
            "Add an entry to PREDICT_FN_BY_MODULE or call make_cell_video() from Python."
        )

    conds = None
    if args.conditions:
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    paths = make_cell_video(
        result_path,
        PREDICT_FN_BY_MODULE[exp_mod],
        experiment_module=exp_mod,
        n_per_bucket=args.n_per_bucket,
        conditions_filter=conds,
        future_len=args.future_len,
        fps=args.fps,
        stride=args.stride,
        display_history=args.display_history,
        out_dir=args.out_dir,
        seed=args.seed,
        holdout_only=not args.all_cells,
        split_seed=args.split_seed,
    )
    print(f"wrote {len(paths)} files")
    for p in paths:
        print(f"  {p}")
