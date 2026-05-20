"""Q4-stratified counterfactual ratio diagnostic.

Recomputes `mean_abs_point_diff_on_off / target_std` per response-magnitude
quartile (Q1..Q4), plus an `all_except_q1` aggregate that drops the flat
windows. The all-windows ratio in the existing run summary blends responsive
windows (Q4) with flat ones (Q1) where both numerator and denominator are
tiny — the Q4 number is what actually matters for MPC viability.

Usage
-----
    uv run python q4_counterfactual.py --run_dir /path/to/run_dir \
        [--n_windows 2000] [--device cuda]

Outputs go to ``<run_dir>/diagnostics/q4_counterfactual/``:
  * ``metrics.json``         — per-bin numbers + quartile edges + metadata
  * ``q4_counterfactual.png`` — ratio bar chart + per-window |on-off| violin
  * ``q4_counterfactual.log`` — plain-text table (also printed to stdout)

Conventions matched against ``experiments/lstm_seq2scal_mdn_minfeats_image_ewma.py``:
  * "point prediction" is the **mixture mean** (Σ_k π_k μ_k), matching the
    existing ``ratio_point_diff_over_std`` metric — NOT a top-component pick.
    This is what lets the all-windows ratio reproduce the run summary.
  * response magnitude per window is ``std(concat(history_cnr, future_cnr))``
    on absolute CNR (with future reconstructed as last_history + cumsum(Δ)),
    matching the existing STRATIFIED BY RESPONSE MAGNITUDE block.
  * ``target_std`` is pooled std of ΔCNR over all (window, step) pairs in
    the bin, matching ``_y_cf.std()`` in the source notebook.
  * stim is clamped to per-channel training-set max (on) or zero (off),
    same as the notebook's counterfactual cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import load_experiment  # noqa: E402
from experiments.seq2seq_data import REAL_DATASET_PATHS  # noqa: E402

# The model classes (Seq2ScalarMDN / Seq2ScalarMDNImage / ImageEncoder /
# Seq2SeqDataset / compute_ewma) live in the notebook's `app.setup` block,
# which executes on import. Saved bundles record the class under `__main__`
# (notebook-script execution), so we also need to mirror them into __main__
# for `bundle.reconstruct_model()` (or rewrite `bundle.model_type`).
import experiments.lstm_seq2scal_mdn_minfeats_image_ewma as _nb  # noqa: E402

_NB_MODULE_PATH = "experiments.lstm_seq2scal_mdn_minfeats_image_ewma"


# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------

def _load_model(run_dir: Path, device: torch.device):
    bundle = load_experiment(str(run_dir))
    # Saved as __main__.Seq2ScalarMDNImage; redirect to the importable module.
    if bundle.model_type.startswith("__main__."):
        cls_name = bundle.model_type.split(".", 1)[1]
        if not hasattr(_nb, cls_name):
            raise RuntimeError(
                f"Model class {cls_name!r} not found in {_NB_MODULE_PATH}; "
                f"the run was trained from a different notebook."
            )
        bundle.model_type = f"{_NB_MODULE_PATH}.{cls_name}"
    model = bundle.reconstruct_model().to(device).eval()
    return bundle, model


def _build_test_dataset(bundle):
    """Reconstruct the test dataset exactly as the training notebook does.

    Splits: ``train_test_split(test_size=0.2, random_state=42)`` then
    ``train_test_split(test_size=0.125, random_state=42)`` on the remainder —
    the same seeds used in
    ``lstm_seq2scal_mdn_minfeats_image_ewma.py`` so test indices match.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from notebooks.experiment.preprocessing import make_tracks

    cfg = bundle.model_config
    tcfg = bundle.training_config or {}
    data_source = cfg["data_source"]
    if data_source not in REAL_DATASET_PATHS:
        raise ValueError(
            f"Only real-data sources supported by this diagnostic; "
            f"run uses data_source={data_source!r}"
        )
    parquet_path = REAL_DATASET_PATHS[data_source]
    df = pd.read_parquet(parquet_path)

    cnr_all, stim_all, meta = make_tracks(
        df, value_col="cnr_median_norm", stim_cols=["fluence_mJ_cm2"],
    )
    fluence_all = np.empty(len(stim_all), dtype=object)
    for i, s in enumerate(stim_all):
        fluence_all[i] = s[0]

    uids_all = meta["uid"].to_numpy()
    base_lookup = df.groupby("uid")["median_cnr_0_9"].first()
    baseline_all = np.array(
        [float(base_lookup[u]) for u in uids_all], dtype=np.float32,
    )

    a_s = float(cfg["ewma_slow_alpha"])
    a_f = float(cfg["ewma_fast_alpha"])
    ewma_slow_all = np.empty(len(cnr_all), dtype=object)
    ewma_fast_all = np.empty(len(cnr_all), dtype=object)
    for i in range(len(cnr_all)):
        c = np.asarray(cnr_all[i], dtype=np.float32)
        ewma_slow_all[i] = _nb.compute_ewma(c, a_s)
        ewma_fast_all[i] = _nb.compute_ewma(c, a_f)

    n_traj = len(cnr_all)
    traj_ids = np.arange(n_traj)
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    H = int(cfg["history_len"])
    F = int(cfg["future_len"])
    test_stride = int(tcfg.get("test_stride", 10))

    test_ds = _nb.Seq2SeqDataset(
        cnr_all[te_ids], fluence_all[te_ids], baseline_all[te_ids],
        ewma_slow_all[te_ids], ewma_fast_all[te_ids],
        H, F, stride=test_stride,
    )

    # Per-channel training-set max for the all-on counterfactual stim level
    # (matches `_stim_max_arr` in the notebook — uses **training/test stim**
    # ranges to define "fully stimulated"; we use test here since training
    # arrays aren't easily recoverable from the bundle. fluence is bounded
    # per channel by the protocol anyway, so train vs test max is typically
    # identical for these datasets).
    stim_max = np.zeros(1, dtype=np.float32)
    for i in te_ids:
        stim_max[0] = max(stim_max[0], float(np.asarray(fluence_all[i]).max()))
    return test_ds, H, F, stim_max


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Per-window outputs needed for stratified ratios."""
    pt_on: np.ndarray   # (F,) mixture mean under all-on stim
    pt_off: np.ndarray  # (F,) mixture mean under all-off stim
    target: np.ndarray  # (F,) ground-truth ΔCNR
    hist_cnr: np.ndarray  # (H,) ground-truth history CNR (for resp_std)


def _forward_counterfactual(
    model, test_ds, indices, stim_max, F, device, batch_size=256
) -> list[WindowResult]:
    """Run on-stim and off-stim forward passes on the given window indices."""
    subset = Subset(test_ds, list(indices))
    stim_on_const = torch.tensor(stim_max, dtype=torch.float32).view(1, 1, -1)
    has_images = False  # use_images=False for the target run

    pt_on_l, pt_off_l, tgt_l, hist_l = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=batch_size):
            if len(batch) == 4:
                eb, sb, tb, imb = batch
                has_images = True
            else:
                eb, sb, tb = batch
                imb = None
            B = eb.shape[0]
            s_on = stim_on_const.repeat(B, F, 1).to(device)
            s_off = torch.zeros_like(sb).to(device)
            eb_d = eb.to(device)
            imb_d = imb.to(device) if (has_images and imb is not None) else None

            kwargs_on = {"images": imb_d} if has_images else {}
            kwargs_off = {"images": imb_d} if has_images else {}
            pi_on, mu_on, _ = model(eb_d, s_on, **kwargs_on)
            pi_off, mu_off, _ = model(eb_d, s_off, **kwargs_off)

            # mixture mean = Σ π_k μ_k  →  shape (B, F)
            pt_on = (pi_on * mu_on).sum(-1).cpu().numpy()
            pt_off = (pi_off * mu_off).sum(-1).cpu().numpy()
            pt_on_l.append(pt_on)
            pt_off_l.append(pt_off)
            tgt_l.append(tb.numpy())
            # encoder feature 0 is CNR (see Seq2SeqDataset.__init__)
            hist_l.append(eb[..., 0].numpy())

    pt_on = np.concatenate(pt_on_l)
    pt_off = np.concatenate(pt_off_l)
    tgt = np.concatenate(tgt_l)
    hist = np.concatenate(hist_l)
    return [
        WindowResult(pt_on[i], pt_off[i], tgt[i], hist[i])
        for i in range(len(pt_on))
    ]


# ---------------------------------------------------------------------------
# Magnitude binning + per-bin ratios
# ---------------------------------------------------------------------------

def _response_magnitude(hist_cnr: np.ndarray, dec_target: np.ndarray) -> float:
    """std over the full (history+future) absolute CNR trajectory.

    Matches the source notebook's STRATIFIED BY RESPONSE MAGNITUDE block:
    future is reconstructed as ``last_history + cumsum(dec_target)``.
    """
    last = float(hist_cnr[-1])
    future = last + np.cumsum(dec_target)
    return float(np.concatenate([hist_cnr, future]).std())


def _select_uniform_subsample(resp_mag: np.ndarray, n: int, rng) -> np.ndarray:
    """Stratified uniform sampling across the magnitude distribution.

    Cuts ``resp_mag`` into 4 equal-count quartiles and samples ``n/4`` from
    each. Falls back to "take all" when a quartile has fewer than n/4 windows.
    """
    edges = np.quantile(resp_mag, [0.25, 0.5, 0.75])
    bin_idx = np.digitize(resp_mag, edges)  # 0..3
    per_bin = n // 4
    picked = []
    for b in range(4):
        idx_b = np.where(bin_idx == b)[0]
        if len(idx_b) <= per_bin:
            picked.append(idx_b)
        else:
            picked.append(rng.choice(idx_b, size=per_bin, replace=False))
    return np.sort(np.concatenate(picked))


def _bin_ratios(
    results: list[WindowResult], resp_mag: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Compute {bin: {mean_abs_point_diff, target_std, ratio, n_windows}}.

    Returns the metrics dict and the quartile-edge array used.
    """
    edges = np.quantile(resp_mag, [0.25, 0.5, 0.75])
    bin_idx = np.digitize(resp_mag, edges)  # 0=Q1, 1=Q2, 2=Q3, 3=Q4

    diff = np.stack([np.abs(r.pt_on - r.pt_off) for r in results])  # (N, F)
    target = np.stack([r.target for r in results])  # (N, F)

    def _stat(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return dict(
                mean_abs_point_diff=float("nan"),
                target_std=float("nan"),
                ratio=float("nan"),
                n_windows=0,
            )
        d = diff[mask]
        t = target[mask]
        pp = float(d.mean())
        ts = float(t.std())
        return dict(
            mean_abs_point_diff=pp,
            target_std=ts,
            ratio=pp / max(ts, 1e-12),
            n_windows=int(mask.sum()),
        )

    out = {}
    for b, key in enumerate(["q1", "q2", "q3", "q4"]):
        out[key] = _stat(bin_idx == b)
    out["all_except_q1"] = _stat(bin_idx >= 1)
    out["all_windows"] = _stat(np.ones_like(bin_idx, dtype=bool))
    return out, edges


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_LABELS = {
    "q1": "Q1 (flat)",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4 (responsive)",
    "all_except_q1": "All except Q1",
    "all_windows": "All windows",
}
_ORDER = ["q1", "q2", "q3", "q4", "all_except_q1", "all_windows"]


def _format_table(metrics: dict) -> str:
    lines = [
        "COUNTERFACTUAL STIMULATION — Q4 STRATIFIED",
        f" {'bin':<20s} {'point_diff':>12s} {'target_std':>12s} "
        f"{'ratio':>10s} {'n':>8s}",
    ]
    for key in _ORDER:
        m = metrics[key]
        lines.append(
            f" {_LABELS[key]:<20s} "
            f"{m['mean_abs_point_diff']:>12.5f} "
            f"{m['target_std']:>12.5f} "
            f"{m['ratio']:>10.3f} "
            f"{m['n_windows']:>8d}"
        )
    return "\n".join(lines)


def _make_figure(
    results: list[WindowResult], resp_mag: np.ndarray, metrics: dict, out_png: Path
) -> None:
    import matplotlib.pyplot as plt

    edges = np.quantile(resp_mag, [0.25, 0.5, 0.75])
    bin_idx = np.digitize(resp_mag, edges)
    per_win_diff = np.array(
        [np.abs(r.pt_on - r.pt_off).mean() for r in results]
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    bars_order = ["q1", "q2", "q3", "q4", "all_except_q1", "all_windows"]
    bar_labels = ["Q1\n(flat)", "Q2", "Q3", "Q4\n(responsive)", "All\nexcept Q1", "All"]
    bar_vals = [metrics[k]["ratio"] for k in bars_order]
    colors = ["#bdbdbd", "#9ecae1", "#4292c6", "#08519c", "#41ab5d", "#737373"]
    ax_l.bar(bar_labels, bar_vals, color=colors, edgecolor="black")
    ax_l.axhline(1.0, color="red", lw=1.2, ls="--",
                 label="MPC plausible if ≳ 1")
    ax_l.set_ylabel("ratio = mean |on − off| / target std")
    ax_l.set_title("Counterfactual sensitivity by response quartile")
    ax_l.legend(loc="best", fontsize=9)
    for x, v in zip(bar_labels, bar_vals):
        ax_l.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Right: per-window mean |on-off| stratified by quartile (box plot —
    # robust to long tails; violin can be misleading with skewed data).
    data = [per_win_diff[bin_idx == b] for b in range(4)]
    ax_r.boxplot(data, labels=["Q1", "Q2", "Q3", "Q4"], showfliers=False)
    ax_r.set_ylabel("per-window mean |μ_on − μ_off|  (ΔCNR)")
    ax_r.set_title("Per-window stim sensitivity by quartile")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(run_dir: Path, n_windows: int | None, device_str: str | None) -> dict:
    device = torch.device(
        device_str if device_str is not None
        else ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    )
    print(f"[q4_counterfactual] run_dir={run_dir}  device={device}")

    bundle, model = _load_model(run_dir, device)
    test_ds, H, F, stim_max = _build_test_dataset(bundle)
    print(f"[q4_counterfactual] test windows={len(test_ds)}  H={H}  F={F}  "
          f"stim_max={stim_max.tolist()}")

    # Compute response magnitude for ALL windows first so subsampling can
    # stratify across the full distribution.
    resp_mag_all = np.empty(len(test_ds), dtype=np.float32)
    for i in range(len(test_ds)):
        enc_in, _dec_stim, dec_target = test_ds[i][:3]
        resp_mag_all[i] = _response_magnitude(
            enc_in[:, 0].numpy(), dec_target.numpy()
        )

    rng = np.random.default_rng(0)
    if n_windows is not None and n_windows < len(test_ds):
        sel_idx = _select_uniform_subsample(resp_mag_all, n_windows, rng)
        print(f"[q4_counterfactual] subsampled {len(sel_idx)} / "
              f"{len(test_ds)} windows uniformly across magnitude")
    else:
        sel_idx = np.arange(len(test_ds))
    resp_mag = resp_mag_all[sel_idx]

    results = _forward_counterfactual(
        model, test_ds, sel_idx, stim_max, F, device,
    )
    metrics, edges = _bin_ratios(results, resp_mag)

    # Sanity check vs saved summary
    saved_ratio = None
    if bundle.metrics:
        for k in ("ratio_point_diff_over_std",
                  "counterfactual.ratio_point_diff_over_std"):
            if k in bundle.metrics:
                saved_ratio = float(bundle.metrics[k])
                break
    if saved_ratio is None:
        # Try the counterfactual_summary stored as nested dict
        for v in (bundle.metrics or {}).values():
            if isinstance(v, dict) and "ratio_point_diff_over_std" in v:
                saved_ratio = float(v["ratio_point_diff_over_std"])
                break

    all_ratio = metrics["all_windows"]["ratio"]
    if saved_ratio is not None:
        rel = abs(all_ratio - saved_ratio) / max(abs(saved_ratio), 1e-12)
        if rel > 0.01:
            print(
                f"[q4_counterfactual] WARNING: all-windows ratio {all_ratio:.4f} "
                f"differs from saved {saved_ratio:.4f} by {rel*100:.2f}% "
                f"(>1%). Subsampling / data-split mismatch?"
            )
        else:
            print(f"[q4_counterfactual] all-windows ratio matches saved "
                  f"({all_ratio:.4f} vs {saved_ratio:.4f}, {rel*100:.2f}%).")
    else:
        print("[q4_counterfactual] no saved ratio found in bundle.metrics — "
              "cannot run the sanity check.")

    # Quartile size sanity check
    n_total = len(resp_mag)
    expected = n_total / 4
    for k in ("q1", "q2", "q3", "q4"):
        if abs(metrics[k]["n_windows"] - expected) > 1:
            print(
                f"[q4_counterfactual] note: {k} has {metrics[k]['n_windows']} "
                f"windows (expected ≈ {expected:.0f})"
            )

    out_dir = run_dir / "diagnostics" / "q4_counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metrics,
        "forecast_len": int(F),
        "history_len": int(H),
        "n_windows_total": int(len(test_ds)),
        "n_windows_evaluated": int(n_total),
        "magnitude_quartile_edges": [float(x) for x in edges],
        "saved_all_windows_ratio": saved_ratio,
        "stim_max": [float(x) for x in stim_max],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    table = _format_table(metrics)
    print(table)
    (out_dir / "q4_counterfactual.log").write_text(table + "\n")

    _make_figure(results, resp_mag, metrics, out_dir / "q4_counterfactual.png")
    print(f"[q4_counterfactual] wrote → {out_dir}")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--n_windows", type=int, default=None,
                    help="Cap evaluation to this many windows "
                         "(stratified uniform subsample). Default: full test set.")
    ap.add_argument("--device", default=None,
                    help="torch device (cuda/mps/cpu). Default: auto.")
    args = ap.parse_args(argv)
    run(args.run_dir.resolve(), args.n_windows, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
