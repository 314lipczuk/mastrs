"""Memory-ladder acceptance test for the full-history model.

Applies the transferable ladder diagnostics (Step 1 memory kernel, Step 3
real-cell carryover + quartile interaction, Step 4 gap sweep) to a trained
``Seq2ScalarHistory`` bundle via ``history_predict.predict_many``. Step 2
(pathway ablation) is N/A — the new model has no EWMA/baseline channels.

Acceptance vs the old model (memory ~recency/≤H≈60, no q1-dependence, hard gap
cutoff at H):
  * Step 1 — does memory extend well past 60 frames?
  * Step 3 — does q1 (prefix responder class) bias exp2 predictions (identity
             carryover)?
  * Step 4 — does carryover survive breaks > 60 frames?

Usage: ``uv run python experiments/eval_history_ladder.py <bundle_dir>``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from optoerk.core.experiment import load_experiment
from optoerk.core.utils import materials_path
from optoerk.data.history_data import load_history_tracks
from optoerk.eval.history_predict import predict_many


def memory_kernel(model, B0, AMP, width=15, T=260, t=200, taus=None):
    """S(tau) = max|pred_perturbed - pred_flat| for a past released block of
    `width` frames at distance tau before the prediction point (full history)."""
    if taus is None:
        taus = list(range(1, t - 10, 3))
    flu0 = np.zeros(T, np.float32)
    base = np.full(T, B0, np.float32)
    m_flat, _ = predict_many(model, base, flu0, [t])
    S = []
    for tau in taus:
        c = base.copy()
        s = t - tau
        c[s:s + width] += AMP
        m, _ = predict_many(model, c, flu0, [t])
        S.append(float(np.abs(m[0] - m_flat[0]).max()))
    return np.asarray(taus), np.asarray(S)


def _horizon(taus, S, frac):
    s0 = S[0] if S[0] > 0 else 1.0
    below = [int(taus[k]) for k in range(len(S)) if S[k] < frac * s0]
    return below[0] if below else int(taus[-1]) + 1


def carryover(model, cells, F, J=120, n_pairs=150, K=20, seed=0):
    """Stitched real-cell carryover Δ(tau) (full c1 prefix vs quiescent baseline)
    + signed early-window quartile-interaction matrix M[q1,q2]."""
    rng = np.random.default_rng(seed)
    cnr = [c[0] for c in cells]
    flu = [c[1] for c in cells]
    fov = [c[2] for c in cells]
    n200 = [c[3] for c in cells]
    B0 = float(np.median([np.median(c[:10]) for c in cnr]))
    score = np.array([float(np.std(c)) for c in cnr])
    e = np.quantile(score, np.linspace(0, 1, 5)); e[0] -= 1e-9; e[-1] += 1e-9
    quart = np.digitize(score, e[1:-1]) + 1
    elig = np.array([i for i in range(len(cnr)) if len(cnr[i]) >= J + F])
    byq = {q: elig[quart[elig] == q] for q in (1, 2, 3, 4)}

    def stitch(i1, i2):
        c1, c2 = cnr[i1], cnr[i2]; L1 = len(c1)
        ts = [L1 + j for j in range(J)]
        st_cnr = np.concatenate([c1, c2]); st_flu = np.concatenate([flu[i1], flu[i2]])
        st_fov = np.concatenate([fov[i1], fov[i2]]); st_n = np.concatenate([n200[i1], n200[i2]])
        cl_cnr = np.concatenate([np.full(L1, B0, np.float32), c2])
        cl_flu = np.concatenate([np.zeros(L1, np.float32), flu[i2]])
        cl_fov = np.concatenate([np.full(L1, np.median(fov[i1]), np.float32), fov[i2]])
        cl_n = np.concatenate([np.full(L1, np.median(n200[i1]), np.float32), n200[i2]])
        mf, _ = predict_many(model, st_cnr, st_flu, ts, fov=st_fov, n200=st_n)
        mc, _ = predict_many(model, cl_cnr, cl_flu, ts, fov=cl_fov, n200=cl_n)
        return mf, mc

    absD, Msum, Mn = [], np.zeros((4, 4)), np.zeros((4, 4))
    W = 5
    for q1 in (1, 2, 3, 4):
        for q2 in (1, 2, 3, 4):
            if len(byq[q1]) == 0 or len(byq[q2]) == 0:
                continue
            for _ in range(K):
                mf, mc = stitch(int(rng.choice(byq[q1])), int(rng.choice(byq[q2])))
                absD.append(np.abs(mf - mc).max(1))
                Msum[q1 - 1, q2 - 1] += float(np.mean((mf - mc)[:W]))
                Mn[q1 - 1, q2 - 1] += 1
    return np.arange(J), np.mean(absD, 0), Msum / np.maximum(Mn, 1)


def gap_sweep(model, cells, F, gaps=(0, 10, 30, 60, 90, 120), n_pairs=100, W=5, seed=1):
    """Early-window carryover vs inserted quiescent break length g."""
    rng = np.random.default_rng(seed)
    cnr = [c[0] for c in cells]; flu = [c[1] for c in cells]
    fov = [c[2] for c in cells]; n200 = [c[3] for c in cells]
    B0 = float(np.median([np.median(c[:10]) for c in cnr]))
    Js = 40
    elig = [i for i in range(len(cnr)) if len(cnr[i]) >= Js + F]
    pairs = [(int(rng.choice(elig)), int(rng.choice(elig))) for _ in range(n_pairs)]

    def early(i1, i2, g):
        c1, c2 = cnr[i1], cnr[i2]; L1 = len(c1); pre = L1 + g
        ts = [pre + j for j in range(W)]
        zb = np.full(g, B0, np.float32); zf = np.zeros(g, np.float32)
        zfov = np.full(g, np.median(fov[i1]), np.float32); zn = np.full(g, np.median(n200[i1]), np.float32)
        st = predict_many(model, np.concatenate([c1, zb, c2]), np.concatenate([flu[i1], zf, flu[i2]]),
                          ts, fov=np.concatenate([fov[i1], zfov, fov[i2]]), n200=np.concatenate([n200[i1], zn, n200[i2]]))[0]
        cl = predict_many(model, np.concatenate([np.full(pre, B0, np.float32), c2]), np.concatenate([np.zeros(pre, np.float32), flu[i2]]),
                          ts, fov=np.concatenate([np.full(pre, np.median(fov[i1]), np.float32), fov[i2]]),
                          n200=np.concatenate([np.full(pre, np.median(n200[i1]), np.float32), n200[i2]]))[0]
        return float(np.abs(st - cl).mean())

    return np.array(gaps), np.array([np.mean([early(a, b, g) for a, b in pairs]) for g in gaps])


def main(bundle_dir):
    bundle = load_experiment(bundle_dir)
    model = bundle.reconstruct_model()
    F = model.cfg.future_len
    print(f"=== {Path(bundle_dir).name} (F={F}, sigma_bias={model.cfg.sigma_step_bias}) ===")

    cnr_o, feats_o, cond, _ = load_history_tracks(materials_path("dataset_all.parquet"))
    cells = [(np.asarray(cnr_o[i], np.float32), np.asarray(feats_o[i])[0],
              np.asarray(feats_o[i])[1], np.asarray(feats_o[i])[2]) for i in range(len(cnr_o))]
    B0 = float(np.median([np.median(c[0][:10]) for c in cells[:300]]))
    AMP = float(np.median([np.max(c[0]) - np.median(c[0][:10]) for c in cells[:300]]))

    print("\n[Step 1] memory kernel (released 15-frame block):")
    taus, S = memory_kernel(model, B0, AMP)
    print(f"  S0={S[0]:.3f}  half@{_horizon(taus,S,.5)}  10%@{_horizon(taus,S,.1)}  5%@{_horizon(taus,S,.05)}  (old ≈ 10 frames; H_old=60)")

    print("\n[Step 3] real-cell carryover + quartile interaction:")
    tco, dco, qm = carryover(model, cells, F)
    rec = int(tco[np.argmax(dco < 0.02)]) if (dco < 0.02).any() else int(tco[-1]) + 1
    print(f"  Δ peak {dco.max():.3f}@{int(tco[np.argmax(dco)])}  recovery(|Δ|<0.02)@{rec}  (old ~38, gone by H=60)")
    print(f"  quartile q1(prefix) row-means: {[round(float(x),4) for x in qm.mean(1)]}  (old ~flat = no identity carryover)")
    print(f"  quartile q2(exp2)  col-means: {[round(float(x),4) for x in qm.mean(0)]}")

    print("\n[Step 4] gap sweep (early carryover vs break length g):")
    gaps, gd = gap_sweep(model, cells, F)
    print("  " + "  ".join(f"g={g}:{v:.4f}" for g, v in zip(gaps, gd)) + "  (old: hard cutoff at g≈60)")


if __name__ == "__main__":
    main(sys.argv[1])
