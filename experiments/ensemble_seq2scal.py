import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import time
    import tempfile
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset, Subset

    from experiment import ExperimentTracker, load_experiment, compute_training_stats
    from utils import get_device, running_on_cluster, results_write_path, parse_bool
    from experiments.seq2seq_data import load, AVAILABLE_DATASETS, STIM_COLS
    from scipy.stats import norm as scipy_norm

    device = get_device()
    n_stim = len(STIM_COLS)
    results_base = results_write_path()

    def _init_forget_bias(lstm):
        for name, param in lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    class LSTMEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout,
            )
            _init_forget_bias(self.lstm)

        def forward(self, x):
            _, (h_n, c_n) = self.lstm(x)
            return h_n, c_n

    class Seq2Scalar(nn.Module):
        """LSTM encoder + MLP head, autoregressive with teacher forcing."""

        def __init__(
            self,
            encoder_dim,
            stim_dim,
            hidden_dim,
            num_layers,
            mlp_hidden=None,
            n_mlp_layers=2,
            dropout=0.1,
            **kwargs,
        ):
            super().__init__()
            if mlp_hidden is None:
                mlp_hidden = hidden_dim
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers, dropout)
            layers = [nn.Linear(hidden_dim + stim_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(n_mlp_layers - 1):
                layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
            layers += [nn.Linear(mlp_hidden, 1)]
            self.head = nn.Sequential(*layers)

        def _predict_step(self, h_top, stim_i):
            return self.head(torch.cat([h_top, stim_i], dim=-1)).squeeze(-1)

        def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
            F = future_stim.shape[1]
            current_window = encoder_input
            predictions = []
            for i in range(F):
                h, _ = self.encoder(current_window)
                pred = self._predict_step(h[-1], future_stim[:, i, :])
                predictions.append(pred)
                if i < F - 1:
                    last_abs = current_window[:, -1, 0:1]
                    use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                    delta = targets[:, i:i+1] if use_teacher else pred.unsqueeze(-1)
                    next_cnr_abs = last_abs + delta
                    next_input = torch.cat([next_cnr_abs, future_stim[:, i, :]], dim=-1).unsqueeze(1)
                    current_window = torch.cat([current_window[:, 1:, :], next_input], dim=1)
            return torch.stack(predictions, dim=1)

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)

    return (
        AVAILABLE_DATASETS,
        DataLoader,
        Dataset,
        ExperimentTracker,
        Path,
        STIM_COLS,
        Seq2Scalar,
        Subset,
        device,
        load,
        load_experiment,
        mo,
        n_stim,
        nn,
        np,
        optim,
        os,
        parse_bool,
        plt,
        results_base,
        scipy_norm,
        tempfile,
        time,
        torch,
        train_test_split,
    )


@app.cell
def _(mo, parse_bool):
    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "ensemble_seq2scal")
    DRY_RUN = parse_bool(args.get("dry_run", True))
    N_MODELS = int(args.get("n_models", "5"))
    SEEDS = [int(s) for s in args.get("seeds", ",".join(str(i) for i in range(N_MODELS))).split(",")]

    mo.md(f"# Seq2Scalar ensemble — `{EXPERIMENT_NAME}` (seeds={SEEDS}, dry_run={DRY_RUN})")
    return DRY_RUN, EXPERIMENT_NAME, SEEDS, args


@app.cell
def _(AVAILABLE_DATASETS, DRY_RUN, args, mo):
    _source = args.get("source", "synthetic_v2")
    assert _source in AVAILABLE_DATASETS, f"Unknown source {_source!r}. Available: {list(AVAILABLE_DATASETS)}"
    config = dict(
        data_source=_source,
        hidden_dim=int(args.get("hidden_dim", "16" if DRY_RUN else "64")),
        num_layers=int(args.get("num_layers", "2")),
        history_len=int(args.get("history_len", "30")),
        future_len=int(args.get("future_len", "5")),
        lr=float(args.get("lr", "1e-3")),
        epochs=int(args.get("epochs", "20" if DRY_RUN else "400")),
        batch_size=int(args.get("batch_size", "64")),
        patience=int(args.get("patience", "10" if DRY_RUN else "100")),
        tf_ratio_start=float(args.get("tf_ratio_start", "1.0")),
        tf_ratio_end=float(args.get("tf_ratio_end", "0.0")),
        dropout=float(args.get("dropout", "0.1")),
        mlp_hidden=int(args["mlp_hidden"]) if args.get("mlp_hidden") else None,
        n_mlp_layers=int(args.get("n_mlp_layers", "5")),
    )
    mo.md(f"**config:** `{config}`")
    return (config,)


@app.cell
def _(
    DRY_RUN,
    DataLoader,
    Dataset,
    STIM_COLS,
    Subset,
    config,
    load,
    mo,
    np,
    torch,
    train_test_split,
):
    H = config["history_len"]
    F_ = config["future_len"]
    total_window = H + F_

    DATA_SOURCE = config["data_source"]
    cnr_all, stim_all, conditions_all = load(DATA_SOURCE)

    n_traj = len(cnr_all)
    traj_len = cnr_all.shape[1]

    class Seq2SeqDataset(Dataset):
        def __init__(self, cnr, stim, history_len, future_len, stride=5):
            self.samples = []
            total = history_len + future_len
            for i in range(len(cnr)):
                t = 0
                while t + total <= cnr.shape[1]:
                    enc_cnr = cnr[i, t : t + history_len]
                    enc_stim = stim[i, :, t : t + history_len]
                    dec_stim = stim[i, :, t + history_len : t + total]
                    full_window = cnr[i, t : t + total]
                    dec_target = np.diff(full_window)[history_len - 1 : history_len - 1 + future_len]
                    enc_in = np.concatenate([enc_cnr[:, np.newaxis], enc_stim.T], axis=-1)
                    self.samples.append((enc_in, dec_stim.T, dec_target))
                    t += stride

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            enc_in, dec_stim, dec_target = self.samples[idx]
            return (
                torch.tensor(enc_in, dtype=torch.float32),
                torch.tensor(dec_stim, dtype=torch.float32),
                torch.tensor(dec_target, dtype=torch.float32),
            )

    traj_ids = np.arange(n_traj)
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    stride = 15
    train_ds = Seq2SeqDataset(cnr_all[tr_ids], stim_all[tr_ids], H, F_, stride=stride)
    val_ds = Seq2SeqDataset(cnr_all[va_ids], stim_all[va_ids], H, F_, stride=stride)
    test_ds = Seq2SeqDataset(cnr_all[te_ids], stim_all[te_ids], H, F_, stride=stride)

    if DRY_RUN:
        n_dry = 2000
        train_ds = Subset(train_ds, range(min(n_dry, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_dry, len(val_ds)) // 4))
        test_ds = Subset(test_ds, range(min(n_dry, len(test_ds))))

    BS = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    mo.md(f"""
    **synthetic v2:** {n_traj} trajectories × {traj_len} timepoints — stim feats: {', '.join(STIM_COLS)}
    Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} windows
    """)
    return F_, H, test_loader, train_loader, val_loader


@app.cell
def _(
    EXPERIMENT_NAME,
    ExperimentTracker,
    Path,
    SEEDS,
    Seq2Scalar,
    args,
    config,
    device,
    load_experiment,
    mo,
    n_stim,
    nn,
    np,
    optim,
    os,
    results_base,
    tempfile,
    time,
    torch,
    train_loader,
    val_loader,
):
    shared_model_config = dict(
        encoder_dim=1 + n_stim,
        stim_dim=n_stim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        mlp_hidden=config["mlp_hidden"] if config["mlp_hidden"] is not None else config["hidden_dim"],
        n_mlp_layers=config["n_mlp_layers"],
        dropout=config["dropout"],
        history_len=config["history_len"],
        future_len=config["future_len"],
        data_source=config["data_source"],
    )

    _exp_dir = args.get("results-dir", f"{results_base}/{EXPERIMENT_NAME}")
    parent_tracker = ExperimentTracker(
        directory=_exp_dir,
        name=EXPERIMENT_NAME,
        model_config=shared_model_config,
        training_config={**config, "seeds": SEEDS},
    )
    parent_tracker.register_start()

    def _run_epoch(model, loader, optimizer, cfg, epoch, is_train):
        if is_train:
            model.train()
            tf_start, tf_end, epochs = cfg["tf_ratio_start"], cfg["tf_ratio_end"], cfg["epochs"]
            tf_ratio = tf_start - (tf_start - tf_end) * epoch / max(epochs - 1, 1)
        else:
            model.eval()
            tf_ratio = 0.0
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in = enc_in.to(device)
                dec_stim = dec_stim.to(device)
                dec_target = dec_target.to(device)
                targets = dec_target if is_train else None
                preds = model(enc_in, dec_stim, targets=targets, tf_ratio=tf_ratio)
                loss = model.loss(preds, dec_target)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses)), tf_ratio

    def _checkpoint_best(tracker, model, best_ckpt_path, hist):
        _cur = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        tracker.checkpoint(model, training_results={"history": hist})
        model.load_state_dict(_cur)

    def train_one_seed(seed, tracker, train_loader, val_loader):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Seq2Scalar(**{k: v for k, v in shared_model_config.items() if k not in ("history_len", "future_len", "data_source")}).to(device)
        opt = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        hist = {"train_loss": [], "val_loss": []}
        fd, ckpt = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        best = float("inf")
        wait = 0
        for epoch in range(config["epochs"]):
            t_loss, tf = _run_epoch(model, train_loader, opt, config, epoch, True)
            v_loss, _ = _run_epoch(model, val_loader, opt, config, epoch, False)
            hist["train_loss"].append(t_loss)
            hist["val_loss"].append(v_loss)
            sched.step(v_loss)
            if v_loss < best:
                best = v_loss
                wait = 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= config["patience"]:
                    print(f"[seed {seed}] early stop @ epoch {epoch}")
                    break
            if epoch % 10 == 0:
                print(f"[seed {seed}] epoch {epoch:3d} tf={tf:.2f} train={t_loss:.5f} val={v_loss:.5f}")
            _checkpoint_best(tracker, model, ckpt, hist)

        model.load_state_dict(torch.load(ckpt, weights_only=True))
        os.remove(ckpt)
        return model, hist, best

    trained = []  # list of dicts: {seed, model, history, best_val, loaded_from_disk}
    train_elapsed_total = 0.0
    for seed in SEEDS:
        sub_dir = Path(parent_tracker.directory) / f"seed_{seed}"
        finished_bundle = sub_dir / "bundle.pt"
        if finished_bundle.exists():
            print(f"[seed {seed}] found finished bundle — loading, skipping training")
            bundle = load_experiment(str(sub_dir))
            model = bundle.reconstruct_model().to(device)
            hist = bundle.training_results.get("history", {"train_loss": [], "val_loss": []})
            best_val = min(hist["val_loss"]) if hist["val_loss"] else float("nan")
            trained.append(dict(seed=seed, model=model, history=hist, best_val=best_val, loaded=True))
            continue

        sub_tracker = parent_tracker.make_subexperiment(
            f"seed_{seed}",
            model_config={**shared_model_config, "seed": seed},
            training_config={**config, "seed": seed},
            checkpoint_interval_s=300,
        )
        sub_tracker.register_start()
        _t0 = time.time()
        model, hist, best_val = train_one_seed(seed, sub_tracker, train_loader, val_loader)
        elapsed = time.time() - _t0
        train_elapsed_total += elapsed
        sub_tracker.save_final(
            model,
            training_results={"history": hist, "train_elapsed_s": elapsed},
            metrics={"best_val_loss": best_val, "final_train_loss": hist["train_loss"][-1], "final_val_loss": hist["val_loss"][-1]},
            figures={},
        )
        trained.append(dict(seed=seed, model=model, history=hist, best_val=best_val, loaded=False))

    mo.md(f"**Ensemble trained.** Total new-training time: {train_elapsed_total:.0f}s | members: {len(trained)} | loaded-from-disk: {sum(1 for t in trained if t['loaded'])}")
    return parent_tracker, trained


@app.cell
def _(device, np, test_loader, torch, trained):
    # Collect per-member predictions on test set
    per_member_preds = []  # (M, N, F) deltas
    all_targets = None
    all_last = None
    all_fut_stim = None
    for _member in trained:
        _model = _member["model"]
        _model.eval()
        preds_chunks = []
        tgt_chunks = []
        last_chunks = []
        stim_chunks = []
        with torch.no_grad():
            for enc_in, dec_stim, dec_target in test_loader:
                enc_d = enc_in.to(device)
                stim_d = dec_stim.to(device)
                preds = _model(enc_d, stim_d).cpu().numpy()
                preds_chunks.append(preds)
                tgt_chunks.append(dec_target.numpy())
                last_chunks.append(enc_in[:, -1, 0].numpy())
                stim_chunks.append(dec_stim[:, :, 0].mean(dim=1).numpy())
        per_member_preds.append(np.concatenate(preds_chunks, axis=0))
        if all_targets is None:
            all_targets = np.concatenate(tgt_chunks, axis=0)
            all_last = np.concatenate(last_chunks, axis=0)
            all_fut_stim = np.concatenate(stim_chunks, axis=0)

    ens_preds = np.stack(per_member_preds, axis=0)  # (M, N, F) deltas
    mean_pred = ens_preds.mean(axis=0)               # (N, F)
    std_pred = ens_preds.std(axis=0, ddof=1)         # (N, F) — epistemic

    # Reconstruct absolute CNR (cumulative sum over delta predictions)
    actual_abs = all_last[:, None] + np.cumsum(all_targets, axis=1)
    mean_abs = all_last[:, None] + np.cumsum(mean_pred, axis=1)
    per_member_abs = all_last[None, :, None] + np.cumsum(ens_preds, axis=2)  # (M, N, F)
    std_abs = per_member_abs.std(axis=0, ddof=1)

    print(f"ensemble tensor shape (M,N,F): {ens_preds.shape}")
    return actual_abs, all_last, mean_abs, per_member_abs, std_abs


@app.cell
def _(
    actual_abs,
    mean_abs,
    mo,
    np,
    per_member_abs,
    scipy_norm,
    std_abs,
    trained,
):
    ensemble_abs_mse = float(((mean_abs - actual_abs) ** 2).mean())
    per_member_abs_mse = np.array([((per_member_abs[m] - actual_abs) ** 2).mean() for m in range(len(trained))])

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

    # Spread-skill correlation (per-sample flattened)
    err = np.abs(actual_abs - mean_abs).flatten()
    spread = sig.flatten()
    spread_skill_corr = float(np.corrcoef(err, spread)[0, 1])

    # Per-horizon uncertainty
    sigma_per_step = sig.mean(axis=0)    # (F,)
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
    np,
    per_member_abs,
    plt,
    sig,
    trained,
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
        for m in range(len(trained)):
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
def _(plt, trained):
    fig_curves, _axs = plt.subplots(1, 2, figsize=(12, 4))
    for t in trained:
        _hist = t["history"]
        if _hist["train_loss"]:
            _axs[0].plot(_hist["train_loss"], label=f"seed {t['seed']}", alpha=0.8)
            _axs[1].plot(_hist["val_loss"], label=f"seed {t['seed']}", alpha=0.8)
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
    coverage,
    ensemble_abs_mse,
    fig_cal,
    fig_curves,
    fig_horizon,
    fig_ss,
    fig_traj,
    levels,
    mean_nll,
    parent_tracker,
    per_member_abs_mse,
    spread_skill_corr,
    trained,
):
    metrics = {
        "n_members": len(trained),
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
        "spread_skill_and_sigma_hist": fig_ss,
        "uncertainty_vs_horizon": fig_horizon,
        "training_curves_per_seed": fig_curves,
    }
    parent_tracker.save_final(
        model=trained[0]["model"],  # representative; per-seed weights live in subexperiments
        training_results={"per_seed_best_val": [t["best_val"] for t in trained]},
        metrics=metrics,
        figures=figures,
    )
    print(f"[ensemble] final bundle saved → {parent_tracker.directory}")
    metrics
    return


if __name__ == "__main__":
    app.run()
