import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import json
    from datetime import datetime

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.stats import norm as scipy_norm

    from experiment import ExperimentBundle, load_experiment, save_experiment
    from experiments.seq2scal_pipeline import build_loaders
    from experiments.seq2seq_data import STIM_COLS
    from utils import get_device, parse_bool

    device = get_device()
    return (
        Path,
        STIM_COLS,
        build_loaders,
        datetime,
        device,
        json,
        load_experiment,
        mo,
        np,
        parse_bool,
        plt,
        save_experiment,
        scipy_norm,
        torch,
    )


@app.cell
def _(mo, parse_bool):
    args = mo.cli_args()
    PARENT_DIR = args.get("results-dir")
    assert PARENT_DIR, "Must pass --results-dir pointing at the ensemble parent directory"
    EXPERIMENT_NAME = args.get("name", "ensemble")
    DRY_RUN = parse_bool(args.get("dry_run", False))

    mo.md(f"# Ensemble aggregate — `{EXPERIMENT_NAME}` at `{PARENT_DIR}`")
    return DRY_RUN, EXPERIMENT_NAME, PARENT_DIR


@app.cell
def _(PARENT_DIR, Path, load_experiment, mo):
    _parent = Path(PARENT_DIR)
    seed_dirs = sorted(d for d in _parent.iterdir() if d.is_dir() and d.name.startswith("seed_") and (d / "bundle.pt").exists())
    mo.stop(not seed_dirs, mo.md(f"No seed_*/bundle.pt found under `{_parent}`."))

    bundles = {}
    for d in seed_dirs:
        bundles[d.name] = load_experiment(str(d))

    mo.md(f"**Loaded {len(bundles)} seed bundles:** `{', '.join(bundles.keys())}`")
    return (bundles,)


@app.cell
def _(bundles):
    _first = next(iter(bundles.values()))
    config = dict(_first.training_config)
    shared_model_config = dict(_first.model_config)
    config.pop("seed", None)
    shared_model_config.pop("seed", None)
    F_ = int(config["future_len"])
    H = int(config["history_len"])

    _loaders = {} if False else None
    return F_, H, config, shared_model_config


@app.cell
def _(DRY_RUN, build_loaders, config, mo):
    loaders = build_loaders(config, dry_run=DRY_RUN)
    test_loader = loaders["test_loader"]
    test_conditions = loaders["test_conditions"]
    mo.md(f"**Test windows:** {len(test_loader.dataset)} | "
          f"conditions: {sorted(set(test_conditions.tolist()))}")
    return test_conditions, test_loader


@app.cell
def _(bundles, device):
    # Legacy bundles saved the model class as `__main__.Seq2Scalar`; redirect to
    # the module-level class so reconstruct_model() can resolve it.
    members = []
    for name, b in bundles.items():
        if b.model_type.endswith("Seq2Scalar") and "__main__" in b.model_type:
            b.model_type = "experiments.seq2scal_model.Seq2Scalar"
        _m = b.reconstruct_model().to(device)
        _m.eval()
        members.append((name, _m, b))
    print(f"reconstructed {len(members)} members on {device}")
    return (members,)


@app.cell
def _(device, members, np, test_loader, torch):
    per_member_preds = []
    all_targets = None
    all_last = None
    all_enc_last_stim = None
    all_fut_stim_sum = None

    for _name, _model, _bundle in members:
        _model.eval()
        preds_chunks = []
        tgt_chunks = []
        last_chunks = []
        enc_last_chunks = []
        fut_sum_chunks = []
        with torch.no_grad():
            for enc_in, dec_stim, dec_target in test_loader:
                enc_d = enc_in.to(device)
                stim_d = dec_stim.to(device)
                preds = _model(enc_d, stim_d).cpu().numpy()
                preds_chunks.append(preds)
                tgt_chunks.append(dec_target.numpy())
                last_chunks.append(enc_in[:, -1, 0].numpy())
                enc_last_chunks.append(enc_in[:, -1, 1:].numpy())
                fut_sum_chunks.append(dec_stim.sum(dim=1).numpy())
        per_member_preds.append(np.concatenate(preds_chunks, axis=0))
        if all_targets is None:
            all_targets = np.concatenate(tgt_chunks, axis=0)
            all_last = np.concatenate(last_chunks, axis=0)
            all_enc_last_stim = np.concatenate(enc_last_chunks, axis=0)
            all_fut_stim_sum = np.concatenate(fut_sum_chunks, axis=0)

    ens_preds = np.stack(per_member_preds, axis=0)
    mean_pred = ens_preds.mean(axis=0)
    std_pred = ens_preds.std(axis=0, ddof=1)

    actual_abs = all_last[:, None] + np.cumsum(all_targets, axis=1)
    mean_abs = all_last[:, None] + np.cumsum(mean_pred, axis=1)
    per_member_abs = all_last[None, :, None] + np.cumsum(ens_preds, axis=2)
    std_abs = per_member_abs.std(axis=0, ddof=1)

    print(f"ensemble tensor shape (M, N, F): {ens_preds.shape}")
    return (
        actual_abs,
        all_enc_last_stim,
        all_fut_stim_sum,
        all_last,
        mean_abs,
        per_member_abs,
        std_abs,
    )


@app.cell
def _(
    actual_abs,
    mean_abs,
    members,
    mo,
    np,
    per_member_abs,
    scipy_norm,
    std_abs,
):
    ensemble_abs_mse = float(((mean_abs - actual_abs) ** 2).mean())
    per_member_abs_mse = np.array([((per_member_abs[m] - actual_abs) ** 2).mean() for m in range(len(members))])

    sigma_floor = 1e-3
    sig = np.maximum(std_abs, sigma_floor)
    nll = 0.5 * np.log(2 * np.pi * sig ** 2) + ((actual_abs - mean_abs) ** 2) / (2 * sig ** 2)
    mean_nll = float(nll.mean())

    levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    coverage = {}
    for _p in levels:
        _z = scipy_norm.ppf(0.5 + _p / 2)
        _lo = mean_abs - _z * sig
        _hi = mean_abs + _z * sig
        coverage[_p] = float(((actual_abs >= _lo) & (actual_abs <= _hi)).mean())

    err = np.abs(actual_abs - mean_abs).flatten()
    spread = sig.flatten()
    spread_skill_corr = float(np.corrcoef(err, spread)[0, 1])

    sigma_per_step = sig.mean(axis=0)
    err_per_step = np.abs(actual_abs - mean_abs).mean(axis=0)

    mo.md(f"""
    ### Ensemble uncertainty metrics

    | metric | value |
    |--------|-------|
    | ensemble mean MSE (abs) | {ensemble_abs_mse:.5f} |
    | mean per-member MSE (abs) | {per_member_abs_mse.mean():.5f} ± {per_member_abs_mse.std():.5f} |
    | best per-member MSE | {per_member_abs_mse.min():.5f} |
    | worst per-member MSE | {per_member_abs_mse.max():.5f} |
    | mean Gaussian NLL | {mean_nll:.5f} |
    | spread-skill corr | {spread_skill_corr:.3f} |

    **Calibration (observed / nominal):**
    {"  ".join(f"{int(_p*100)}%→{coverage[_p]*100:.1f}%" for _p in levels)}
    """)
    return (
        coverage,
        ensemble_abs_mse,
        err_per_step,
        levels,
        mean_nll,
        per_member_abs_mse,
        sig,
        sigma_per_step,
        spread_skill_corr,
    )


@app.cell
def _(
    F_,
    H,
    actual_abs,
    all_last,
    mean_abs,
    members,
    np,
    per_member_abs,
    plt,
    sig,
):
    n_ex = 8
    idx = np.linspace(0, actual_abs.shape[0] - 1, n_ex, dtype=int)
    fig_traj, _axes = plt.subplots(2, 4, figsize=(18, 8))
    _axes = _axes.flatten()
    t_fut = np.arange(H, H + F_)
    for ax_i, i in enumerate(idx):
        _a = _axes[ax_i]
        mu = mean_abs[i]
        s = sig[i]
        _a.axhline(all_last[i], color="gray", linestyle=":", alpha=0.4)
        _a.fill_between(t_fut, mu - 2 * s, mu + 2 * s, color="tab:blue", alpha=0.15, label="±2σ")
        _a.fill_between(t_fut, mu - s, mu + s, color="tab:blue", alpha=0.30, label="±1σ")
        for m in range(len(members)):
            _a.plot(t_fut, per_member_abs[m, i], color="tab:blue", alpha=0.25, lw=0.8)
        _a.plot(t_fut, mu, color="tab:blue", lw=2, label="mean")
        _a.plot(t_fut, actual_abs[i], color="navy", lw=1.5, label="actual")
        _a.set_title(f"#{i}", fontsize=9)
        if ax_i == 0:
            _a.legend(fontsize=7)
    fig_traj.suptitle("Ensemble prediction intervals (absolute CNR, test set)", fontsize=12)
    fig_traj.tight_layout()
    fig_traj
    return (fig_traj,)


@app.cell
def _(coverage, levels, np, plt):
    fig_cal, _a = plt.subplots(1, 1, figsize=(5, 5))
    nominal = np.array(levels)
    observed = np.array([coverage[_p] for _p in levels])
    _a.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    _a.plot(nominal, observed, "o-", color="tab:blue", lw=2, ms=8, label="ensemble")
    for _p, _o in zip(nominal, observed):
        _a.annotate(f"{int(_p*100)}%", (_p, _o), xytext=(5, -10), textcoords="offset points", fontsize=8)
    _a.set_xlabel("nominal coverage")
    _a.set_ylabel("observed coverage")
    _a.set_title("Calibration diagram")
    _a.legend()
    _a.grid(alpha=0.3)
    fig_cal
    return (fig_cal,)


@app.cell
def _(STIM_COLS, all_enc_last_stim, all_fut_stim_sum, np, test_conditions):
    _idx = {c: i for i, c in enumerate(STIM_COLS)}

    fut_fluence = all_fut_stim_sum[:, _idx["u_t"]]
    fut_n_pulses = all_fut_stim_sum[:, _idx["m_t"]]
    hist_recency = all_enc_last_stim[:, _idx["recency"]]
    hist_n_5 = all_enc_last_stim[:, _idx["n_5"]]
    hist_s_cum = all_enc_last_stim[:, _idx["s_cum"]]
    hist_burst = all_enc_last_stim[:, _idx["burst_pos"]] > 0

    def _quartile_labels(x, name):
        q = np.quantile(x, [0.25, 0.5, 0.75])
        labels = np.where(
            x <= q[0], f"{name}:Q1",
            np.where(x <= q[1], f"{name}:Q2",
            np.where(x <= q[2], f"{name}:Q3", f"{name}:Q4")),
        )
        return labels

    strata = {
        "condition / protocol": np.asarray(test_conditions),
        "future fluence (Σ u_t)": _quartile_labels(fut_fluence, "fluence"),
        "future pulse count": np.where(fut_n_pulses == 0, "n=0",
                              np.where(fut_n_pulses <= 2, "n=1-2", "n≥3")),
        "recency at boundary": np.where(hist_recency < 0.05, "cold",
                               np.where(hist_recency < 0.5, "warm", "hot")),
        "local rate n_5 (boundary)": _quartile_labels(hist_n_5, "n5"),
        "cumulative fluence so far": _quartile_labels(hist_s_cum, "s_cum"),
        "inside burst at boundary": np.where(hist_burst, "in-burst", "off"),
    }
    return (strata,)


@app.cell
def _(actual_abs, levels, mean_abs, np, plt, scipy_norm, sig, strata):
    def _coverage_curve(mu, truth, sigma, levels):
        out = []
        for p in levels:
            z = scipy_norm.ppf(0.5 + p / 2)
            inside = ((truth >= mu - z * sigma) & (truth <= mu + z * sigma)).mean()
            out.append(float(inside))
        return np.array(out)

    n_features = len(strata)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig_strat_cal, _axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    _axes = _axes.flatten()
    _nominal = np.array(levels)

    for _i, (_feat_name, _labels) in enumerate(strata.items()):
        _a = _axes[_i]
        _groups = sorted(set(_labels.tolist()))
        _a.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1)
        for _g in _groups:
            _mask = _labels == _g
            if _mask.sum() < 20:
                continue
            _mu = mean_abs[_mask].flatten()
            _truth = actual_abs[_mask].flatten()
            _sigma = sig[_mask].flatten()
            obs = _coverage_curve(_mu, _truth, _sigma, levels)
            _a.plot(_nominal, obs, "o-", lw=1.5, ms=5, label=f"{_g} (n={int(_mask.sum())})")
        _a.set_title(_feat_name, fontsize=10)
        _a.set_xlabel("nominal"); _a.set_ylabel("observed")
        _a.grid(alpha=0.3)
        _a.legend(fontsize=7)
    for _i in range(n_features, len(_axes)):
        _axes[_i].set_visible(False)

    fig_strat_cal.suptitle(
        "Stratified calibration — curves ABOVE ideal = overconfident on that stratum", fontsize=11,
    )
    fig_strat_cal.tight_layout()
    fig_strat_cal
    return (fig_strat_cal,)


@app.cell
def _(actual_abs, mean_abs, np, plt, scipy_norm, sig, strata):
    def _per_stratum_stats(labels):
        z95 = scipy_norm.ppf(0.975)
        rows = []
        for g in sorted(set(labels.tolist())):
            m = labels == g
            if m.sum() < 20:
                continue
            mu = mean_abs[m]; tr = actual_abs[m]; sg = sig[m]
            nll = float((0.5 * np.log(2 * np.pi * sg ** 2) + ((tr - mu) ** 2) / (2 * sg ** 2)).mean())
            cov95 = float(((tr >= mu - z95 * sg) & (tr <= mu + z95 * sg)).mean())
            rows.append(dict(
                group=g, n=int(m.sum()),
                mean_sigma=float(sg.mean()),
                mean_abs_err=float(np.abs(tr - mu).mean()),
                nll=nll, cov95=cov95, miscal95=cov95 - 0.95,
            ))
        return rows

    fig_strat_bar, _axes = plt.subplots(len(strata), 1, figsize=(10, 2.2 * len(strata)))
    if len(strata) == 1:
        _axes = [_axes]
    for _i, (_feat_name, _labels) in enumerate(strata.items()):
        rows = _per_stratum_stats(_labels)
        names = [r["group"] for r in rows]
        miscal = np.array([r["miscal95"] for r in rows])
        sigmas = np.array([r["mean_sigma"] for r in rows])
        errs = np.array([r["mean_abs_err"] for r in rows])
        counts = [r["n"] for r in rows]
        x = np.arange(len(names))

        _a = _axes[_i]
        _a.bar(x - 0.2, miscal, width=0.4,
               color=np.where(miscal < 0, "tab:red", "tab:green"), alpha=0.7)
        _a.set_xticks(x); _a.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)], fontsize=8)
        _a.axhline(0, color="k", lw=0.5)
        _a.set_ylabel("miscal @ 95%", fontsize=8)
        _a.set_title(_feat_name, fontsize=9)

        _b = _a.twinx()
        _b.plot(x, sigmas, "o-", color="tab:blue", ms=5, label="mean σ")
        _b.plot(x, errs, "s--", color="tab:orange", ms=5, label="mean |err|")
        _b.set_ylabel("σ / |err|", fontsize=8)
        _b.legend(fontsize=7, loc="upper right")

    fig_strat_bar.suptitle("Per-stratum miscalibration (red = overconfident)", fontsize=11)
    fig_strat_bar.tight_layout()
    fig_strat_bar
    return (fig_strat_bar,)


@app.cell
def _(actual_abs, mean_abs, np, plt, sig):
    _err = np.abs(actual_abs - mean_abs).flatten()
    _spread = sig.flatten()
    fig_ss, _ax = plt.subplots(1, 2, figsize=(12, 4))

    order = np.argsort(_spread)
    nb = 20
    bin_size = max(1, len(_spread) // nb)
    s_sorted = _spread[order]
    e_sorted = _err[order]
    bs = [s_sorted[i:i+bin_size] for i in range(0, len(s_sorted), bin_size)]
    be = [e_sorted[i:i+bin_size] for i in range(0, len(e_sorted), bin_size)]
    s_means = np.array([x.mean() for x in bs])
    e_means = np.array([x.mean() for x in be])

    _ax[0].scatter(_spread[::50], _err[::50], s=3, alpha=0.2, color="tab:blue")
    _ax[0].plot(s_means, e_means, "o-", color="tab:red", lw=2, label="binned mean")
    _max = max(_spread.max(), _err.max())
    _ax[0].plot([0, _max], [0, _max], "k--", alpha=0.5, label="y=x")
    _ax[0].set_xlabel("ensemble σ (spread)")
    _ax[0].set_ylabel("|actual − mean| (skill)")
    _ax[0].set_title("Spread vs skill")
    _ax[0].legend()

    _ax[1].hist(sig.flatten(), bins=60, color="tab:blue", alpha=0.8)
    _ax[1].set_xlabel("σ")
    _ax[1].set_ylabel("count")
    _ax[1].set_title("Histogram of predictive σ")
    fig_ss.tight_layout()
    fig_ss
    return (fig_ss,)


@app.cell
def _(F_, err_per_step, np, plt, sigma_per_step):
    fig_horizon, _a = plt.subplots(1, 1, figsize=(6, 4))
    steps = np.arange(1, F_ + 1)
    _a.plot(steps, sigma_per_step, "o-", color="tab:blue", lw=2, label="mean σ")
    _a.plot(steps, err_per_step, "s--", color="tab:red", lw=2, label="mean |error|")
    _a.set_xlabel("forecast horizon (steps ahead)")
    _a.set_ylabel("absolute CNR units")
    _a.set_title("Uncertainty and error vs horizon")
    _a.legend()
    _a.grid(alpha=0.3)
    fig_horizon
    return (fig_horizon,)


@app.cell
def _(bundles, plt):
    fig_curves, _axs = plt.subplots(1, 2, figsize=(12, 4))
    for _name, _b in bundles.items():
        _hist = _b.training_results.get("history", {})
        if _hist.get("train_loss"):
            _axs[0].plot(_hist["train_loss"], label=_name, alpha=0.8)
            _axs[1].plot(_hist["val_loss"], label=_name, alpha=0.8)
    for _a, _title in zip(_axs, ["train", "val"]):
        _a.set_yscale("log")
        _a.set_xlabel("epoch")
        _a.set_ylabel("MSE")
        _a.set_title(f"{_title} loss per seed")
        _a.legend(fontsize=8)
    fig_curves.tight_layout()
    fig_curves
    return (fig_curves,)


@app.cell
def _(
    EXPERIMENT_NAME,
    PARENT_DIR,
    Path,
    bundles,
    config,
    coverage,
    datetime,
    ensemble_abs_mse,
    fig_cal,
    fig_curves,
    fig_horizon,
    fig_ss,
    fig_strat_bar,
    fig_strat_cal,
    fig_traj,
    json,
    levels,
    mean_nll,
    members,
    mo,
    per_member_abs_mse,
    save_experiment,
    shared_model_config,
    spread_skill_corr,
):
    metrics = {
        "n_members": len(members),
        "ensemble_mse_abs": ensemble_abs_mse,
        "mean_member_mse_abs": float(per_member_abs_mse.mean()),
        "std_member_mse_abs": float(per_member_abs_mse.std()),
        "best_member_mse_abs": float(per_member_abs_mse.min()),
        "worst_member_mse_abs": float(per_member_abs_mse.max()),
        "mean_nll_gaussian": mean_nll,
        "spread_skill_corr": spread_skill_corr,
        **{f"coverage_{int(p*100)}": coverage[p] for p in levels},
    }
    figures = {
        "trajectories_with_uncertainty": fig_traj,
        "calibration": fig_cal,
        "calibration_stratified": fig_strat_cal,
        "calibration_miscal_by_stratum": fig_strat_bar,
        "spread_skill_and_sigma_hist": fig_ss,
        "uncertainty_vs_horizon": fig_horizon,
        "training_curves_per_seed": fig_curves,
    }

    _representative = members[0][1]
    save_experiment(
        directory=PARENT_DIR,
        model=_representative,
        model_config=shared_model_config,
        training_config={**config, "seeds": [b.training_config.get("seed") for b in bundles.values()]},
        training_results={"per_seed_best_val": [b.metrics.get("best_val_loss") for b in bundles.values()]},
        metrics=metrics,
        figures=figures,
        name=EXPERIMENT_NAME,
    )

    manifest = {
        "name": EXPERIMENT_NAME,
        "variants": list(bundles.keys()),
        "shared_config": config,
        "created": datetime.now().isoformat(),
        "subexperiments": {
            name: {
                "name": b.name,
                "model_config": b.model_config,
                "training_config": b.training_config,
                "status": "completed",
            }
            for name, b in bundles.items()
        },
    }
    (Path(PARENT_DIR) / "experiment.json").write_text(json.dumps(manifest, indent=2, default=str))

    mo.md(f"**Aggregate bundle saved → `{PARENT_DIR}`**. Manifest lists {len(bundles)} variants.")
    return


if __name__ == "__main__":
    app.run()
