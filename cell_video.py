"""Generate an mp4 walking a trained model through full single-cell trajectories.

Cells are stratified by trajectory std (quartiles) and sampled
``(n_low, n_mid, n_high)`` per stratum. For each picked cell, the context
window slides forward in time; each frame shows the true trajectory plus the
model's ``future_len``-step prediction with bands from the GMM's per-step
``pred_std`` (no MC dropout — matches notebook eval semantics).

Both prediction and per-frame plotting are pluggable. ``predict_fn`` is
required (model contracts differ); reference impls for the MDN minfeats family
live in this module (``predict_mdn_minfeats``, ``predict_mdn_minfeats_ewma``).
"""

from __future__ import annotations

import importlib
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
    stratum: str               # "low" | "mid" | "high"


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
    n_strata: int = 4,
) -> list[np.ndarray]:
    """Quartile bins of per-cell ``std(cnr)``. Returns list of index arrays low→high."""
    scores = np.array([np.std(np.asarray(c)) for c in cnr_tracks])
    edges = np.quantile(scores, np.linspace(0.0, 1.0, n_strata + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_idx = np.digitize(scores, edges[1:-1])
    return [np.where(bin_idx == k)[0] for k in range(n_strata)]


def select_cells(
    strata: list[np.ndarray],
    n_low: int,
    n_mid: int,
    n_high: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Pick ``(idx, stratum_label)`` from low / mid (Q2∪Q3) / high pools."""
    low_pool = strata[0]
    mid_pool = np.concatenate(strata[1:-1]) if len(strata) > 2 else strata[1]
    high_pool = strata[-1]

    def _sample(pool, k):
        k = min(k, len(pool))
        return rng.choice(pool, size=k, replace=False)

    picks: list[tuple[int, str]] = []
    picks += [(int(i), "low") for i in _sample(low_pool, n_low)]
    picks += [(int(i), "mid") for i in _sample(mid_pool, n_mid)]
    picks += [(int(i), "high") for i in _sample(high_pool, n_high)]
    return picks


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


PREDICT_FN_BY_MODULE: dict[str, PredictFn] = {
    "experiments.lstm_seq2scal_mdn_minfeats": predict_mdn_minfeats,
    "experiments.lstm_seq2scal_mdn_minfeats_image_ewma": predict_mdn_minfeats_ewma,
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
        f"cell #{cell.idx}  [{cell.stratum}]  cond={cell.condition}  t={t}/{T - 1}"
    )
    ax.legend(loc="upper right", fontsize=8)


# ------------------------------------------------------------- main entry

def make_cell_video(
    result_path: str | Path,
    predict_fn: PredictFn,
    *,
    experiment_module: str | None = None,
    dataset_name: str | None = None,
    n_low: int = 0,
    n_mid: int = 2,
    n_high: int = 2,
    future_len: int | None = None,
    fps: int = 8,
    stride: int = 1,
    history_len: int | None = None,
    display_history: int | None = None,
    ylim: tuple[float, float] | None = None,
    ylim_pad: float = 0.15,
    frame_fn: FrameFn | None = None,
    out_path: str | Path | None = None,
    seed: int = 0,
    dpi: int = 100,
) -> Path:
    """Render an mp4 of a model rolling through full single-cell trajectories.

    Parameters
    ----------
    result_path
        Directory of a saved experiment (loadable by :func:`load_experiment`).
    predict_fn
        Required. ``(model, cell, t, future_len, history_len, device) → (mean (F,), sigma (F,))``
        absolute CNR prediction with cumulative-variance uncertainty.
        Use :func:`predict_mdn_minfeats_ewma` for the MDN minfeats+EWMA models.
    experiment_module
        Module containing the model class for ``__main__``-saved bundles, e.g.
        ``"experiments.lstm_seq2scal_mdn_minfeats_image_ewma"``. If omitted,
        inferred from ``<result_path>/slurm.log`` (``Notebook : experiments/<x>.py``).
    dataset_name
        Override the dataset; defaults to ``training_config["data_source"]`` then
        ``model_config["data_source"]`` if present.
    n_low, n_mid, n_high
        Cells per stratum (Q1, Q2∪Q3, Q4 of std(cnr)).
    future_len
        Steps ahead to predict per frame. Defaults to ``model_config["future_len"]``
        (matches training horizon).
    stride
        Step (in frames) between successive context-window positions.
    history_len
        Encoder window size. Defaults to ``model_config["history_len"]`` then
        ``training_config["history_len"]``.
    display_history
        If set, frame plots clip the left x-axis to ``[t - display_history, T-1]``
        instead of showing the full trajectory from frame 0.
    ylim
        Fixed y-axis range for every frame. If None, computed from the true CNR
        across all selected cells with ``ylim_pad`` fractional padding.
    out_path
        mp4 destination. Default: ``<result_path>/video.mp4``.
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

    cnr_arr, stim_arr, conditions = seq2seq_data.load(dataset_name)
    cnr_list = [np.asarray(c, dtype=np.float32) for c in cnr_arr]
    if isinstance(stim_arr, np.ndarray) and stim_arr.dtype != object:
        stim_list = [stim_arr[i].astype(np.float32) for i in range(len(stim_arr))]
    else:
        stim_list = [np.asarray(s, dtype=np.float32) for s in stim_arr]

    keep = [i for i, c in enumerate(cnr_list) if len(c) > history_len + 1]
    cnr_list = [cnr_list[i] for i in keep]
    stim_list = [stim_list[i] for i in keep]
    conditions = np.asarray(conditions)[keep]

    rng = np.random.default_rng(seed)
    strata = stratify_by_std(cnr_list, n_strata=4)
    picks = select_cells(strata, n_low, n_mid, n_high, rng)
    if not picks:
        raise RuntimeError("No cells selected — check n_low/n_mid/n_high and dataset size.")

    cells: list[CellData] = [
        CellData(idx=i, cnr=cnr_list[i], stim=stim_list[i],
                 condition=str(conditions[i]), stratum=label)
        for i, label in picks
    ]

    _frame_fn = frame_fn or default_frame_fn

    if ylim is None:
        all_cnr = np.concatenate([c.cnr for c in cells])
        lo, hi = float(all_cnr.min()), float(all_cnr.max())
        pad = (hi - lo) * ylim_pad
        ylim = (lo - pad, hi + pad)

    schedule: list[tuple[CellData, int]] = []
    for cell in cells:
        T = cell.cnr.shape[0]
        ts = range(history_len, T - 1, max(stride, 1))
        schedule.extend((cell, int(t)) for t in ts)

    if not schedule:
        raise RuntimeError("Empty frame schedule — history_len too large for selected cells.")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)

    def render(frame_i: int):
        cell, t = schedule[frame_i]
        mean, sigma = predict_fn(model, cell, t, future_len, history_len, device)
        _frame_fn(ax, cell, t, history_len, mean, sigma, display_history, ylim)

    anim = FuncAnimation(fig, render, frames=len(schedule), interval=1000 // max(fps, 1))

    out_path = Path(out_path) if out_path is not None else result_path / "video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: FFMpegWriter | PillowWriter
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    else:
        gif_path = out_path.with_suffix(".gif")
        print(f"[cell_video] ffmpeg not on PATH; writing {gif_path} instead")
        out_path = gif_path
        writer = PillowWriter(fps=fps)

    pbar = tqdm(total=len(schedule), desc="cell_video", unit="frame")
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
    parser.add_argument("--future-len", type=int, default=None, help="Steps ahead per frame; defaults to model's training future_len.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--display-history", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
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

    path = make_cell_video(
        result_path,
        PREDICT_FN_BY_MODULE[exp_mod],
        experiment_module=exp_mod,
        future_len=args.future_len,
        fps=args.fps,
        stride=args.stride,
        display_history=args.display_history,
        out_path=args.out,
        seed=args.seed,
    )
    print(f"wrote {path}")
