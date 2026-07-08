"""Q4-stratified counterfactual ratio — variant family (Seq2ScalarSeq).

Mirror of ``q4_counterfactual.py`` for runs trained by
``experiments/lstm_seq2scal_variant.py`` (model class
``experiments.seq2scal_models.Seq2ScalarSeq``). Same metric definitions —
``mean_abs_point_diff_on_off / target_std`` per response-magnitude quartile,
plus ``all_except_q1`` and ``all_windows`` aggregates — but data is built via
``seq2scal_models.prepare_data`` / ``Seq2SeqDataset`` so the encoder layout
and test split match the variant training notebook (5 minfeats channels,
70/10/20 split with seed=42).

Usage::

    uv run python q4_counterfactual_variant.py --run_dir /path/to/run_dir \
        [--n_windows 2000] [--device cuda]

Outputs land in ``<run_dir>/diagnostics/q4_counterfactual/`` (same layout as
the original script).
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
from experiments.seq2scal_models import (  # noqa: E402
    Seq2SeqDataset,
    prepare_data,
)


# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------

def _load_model(run_dir: Path, device: torch.device):
    bundle = load_experiment(str(run_dir))
    # Variant bundles record the full importable path already; the __main__
    # rewrite only fires for legacy notebook-script saves.
    if bundle.model_type.startswith("__main__."):
        bundle.model_type = (
            "experiments.seq2scal_models." + bundle.model_type.split(".", 1)[1]
        )
    model = bundle.reconstruct_model().to(device).eval()
    return bundle, model


def _build_test_dataset(bundle):
    """Reconstruct the test dataset via ``seq2scal_models.prepare_data``.

    Uses the same data_source, ewma alphas, history_len, future_len, and
    test_stride that were saved on the bundle so windowing is byte-identical
    to the variant notebook's eval path.
    """
    cfg = bundle.model_config
    tcfg = bundle.training_config or {}
    data_source = cfg["data_source"]

    prep = prepare_data(
        data_source,
        ewma_slow_alpha=float(cfg["ewma_slow_alpha"]),
        ewma_fast_alpha=float(cfg["ewma_fast_alpha"]),
        dry_run=False,
        seed=int(tcfg.get("seed", 42)),
    )
    H = int(cfg["history_len"])
    F = int(cfg["future_len"])
    test_stride = int(tcfg.get("test_stride", 10))
    test_ds = Seq2SeqDataset(prep.test, H, F, stride=test_stride)

    # Per-channel "fully stimulated" level: max u_t over the train split, the
    # closest analogue to ``_stim_max_arr`` in the legacy notebook. The
    # variant's stim_dim is 1 (fluence only).
    train_fluence = prep.train[1]  # tuple = (cnr, fluence, baseline, ewma_s, ewma_f)
    stim_max = np.zeros(1, dtype=np.float32)
    for arr in train_fluence:
        if len(arr):
            stim_max[0] = max(stim_max[0], float(np.asarray(arr).max()))
    return test_ds, H, F, stim_max


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    pt_on: np.ndarray   # (F,) mixture mean under all-on stim
    pt_off: np.ndarray  # (F,) mixture mean under all-off stim
    target: np.ndarray  # (F,) ground-truth ΔCNR
    hist_cnr: np.ndarray  # (H,) ground-truth history CNR


def _forward_counterfactual(
    model, test_ds, indices, stim_max, F, device, batch_size=256
) -> list[WindowResult]:
    subset = Subset(test_ds, list(indices))
    stim_on_const = torch.tensor(stim_max, dtype=torch.float32).view(1, 1, -1)

    pt_on_l, pt_off_l, tgt_l, hist_l = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=batch_size):
            eb, sb, tb = batch
            B = eb.shape[0]
            s_on = stim_on_const.repeat(B, F, 1).to(device)
            s_off = torch.zeros_like(sb).to(device)
            eb_d = eb.to(device)

            pi_on, mu_on, _ = model(eb_d, s_on)
            pi_off, mu_off, _ = model(eb_d, s_off)

            # mixture mean = Σ π_k μ_k  →  shape (B, F); for gaussian-head
            # variants this still works (pi==1, K==1).
            pt_on = (pi_on * mu_on).sum(-1).cpu().numpy()
            pt_off = (pi_off * mu_off).sum(-1).cpu().numpy()
            pt_on_l.append(pt_on)
            pt_off_l.append(pt_off)
            tgt_l.append(tb.numpy())
            # encoder feature 0 is CNR (see Seq2SeqDataset)
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
# Magnitude binning + per-bin ratios  (identical to q4_counterfactual.py)
# ---------------------------------------------------------------------------

def _response_magnitude(hist_cnr: np.ndarray, dec_target: np.ndarray) -> float:
    last = float(hist_cnr[-1])
    future = last + np.cumsum(dec_target)
    return float(np.concatenate([hist_cnr, future]).std())


def _select_uniform_subsample(resp_mag: np.ndarray, n: int, rng) -> np.ndarray:
    edges = np.quantile(resp_mag, [0.25, 0.5, 0.75])
    bin_idx = np.digitize(resp_mag, edges)
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
    edges = np.quantile(resp_mag, [0.25, 0.5, 0.75])
    bin_idx = np.digitize(resp_mag, edges)  # 0=Q1, 1=Q2, 2=Q3, 3=Q4

    diff = np.stack([np.abs(r.pt_on - r.pt_off) for r in results])
    target = np.stack([r.target for r in results])

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
        "COUNTERFACTUAL STIMULATION — Q4 STRATIFIED (variant)",
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
    ax_l.axhline(1.0, color="red", lw=1.2, ls="--", label="MPC plausible if ≳ 1")
    ax_l.set_ylabel("ratio = mean |on − off| / target std")
    ax_l.set_title("Counterfactual sensitivity by response quartile")
    ax_l.legend(loc="best", fontsize=9)
    for x, v in zip(bar_labels, bar_vals):
        ax_l.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

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
    print(f"[q4_counterfactual_variant] run_dir={run_dir}  device={device}")

    bundle, model = _load_model(run_dir, device)
    test_ds, H, F, stim_max = _build_test_dataset(bundle)
    print(f"[q4_counterfactual_variant] test windows={len(test_ds)}  H={H}  "
          f"F={F}  stim_max={stim_max.tolist()}")

    resp_mag_all = np.empty(len(test_ds), dtype=np.float32)
    for i in range(len(test_ds)):
        enc_in, _dec_stim, dec_target = test_ds[i]
        resp_mag_all[i] = _response_magnitude(
            enc_in[:, 0].numpy(), dec_target.numpy()
        )

    rng = np.random.default_rng(0)
    if n_windows is not None and n_windows < len(test_ds):
        sel_idx = _select_uniform_subsample(resp_mag_all, n_windows, rng)
        print(f"[q4_counterfactual_variant] subsampled {len(sel_idx)} / "
              f"{len(test_ds)} windows uniformly across magnitude")
    else:
        sel_idx = np.arange(len(test_ds))
    resp_mag = resp_mag_all[sel_idx]

    results = _forward_counterfactual(
        model, test_ds, sel_idx, stim_max, F, device,
    )
    metrics, edges = _bin_ratios(results, resp_mag)

    # Sanity check vs saved scalar. The variant notebook stores it under
    # ``counterfactual_point_shift_ratio`` (see summary.txt of any variant run).
    saved_ratio = None
    if bundle.metrics:
        for k in (
            "counterfactual_point_shift_ratio",
            "ratio_point_diff_over_std",
            "counterfactual.ratio_point_diff_over_std",
        ):
            if k in bundle.metrics:
                saved_ratio = float(bundle.metrics[k])
                break
        if saved_ratio is None:
            for v in bundle.metrics.values():
                if isinstance(v, dict) and "ratio_point_diff_over_std" in v:
                    saved_ratio = float(v["ratio_point_diff_over_std"])
                    break

    all_ratio = metrics["all_windows"]["ratio"]
    if saved_ratio is not None:
        rel = abs(all_ratio - saved_ratio) / max(abs(saved_ratio), 1e-12)
        if rel > 0.05:
            print(
                f"[q4_counterfactual_variant] NOTE: all-windows ratio "
                f"{all_ratio:.4f} differs from saved {saved_ratio:.4f} by "
                f"{rel*100:.2f}%. The variant notebook's saved counterfactual "
                f"may use a different stim_max baseline (e.g. test-set max or "
                f"a fixed protocol level); the per-quartile ratios are still "
                f"the right thing to read."
            )
        else:
            print(f"[q4_counterfactual_variant] all-windows ratio matches "
                  f"saved ({all_ratio:.4f} vs {saved_ratio:.4f}, "
                  f"{rel*100:.2f}%).")
    else:
        print("[q4_counterfactual_variant] no saved ratio found in "
              "bundle.metrics — cannot run the sanity check.")

    n_total = len(resp_mag)
    expected = n_total / 4
    for k in ("q1", "q2", "q3", "q4"):
        if abs(metrics[k]["n_windows"] - expected) > 1:
            print(
                f"[q4_counterfactual_variant] note: {k} has "
                f"{metrics[k]['n_windows']} windows (expected ≈ {expected:.0f})"
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
        "source_script": "q4_counterfactual_variant.py",
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    table = _format_table(metrics)
    print(table)
    (out_dir / "q4_counterfactual.log").write_text(table + "\n")

    _make_figure(results, resp_mag, metrics, out_dir / "q4_counterfactual.png")
    print(f"[q4_counterfactual_variant] wrote → {out_dir}")
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
