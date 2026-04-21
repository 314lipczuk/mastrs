import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import math
    import torch
    import torch.nn as nn

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

    class MDNHead(nn.Module):
        """Mixture Density Network head: predicts (pi, mu, sigma) for a scalar target.

        out_feat hardcoded to 1 (delta CNR). sigma parametrized as exp(log_sigma) for positivity.
        """
        def __init__(self, in_feat, n_gaussians):
            super().__init__()
            self.n_gaussians = n_gaussians
            self.pi_linear = nn.Linear(in_feat, n_gaussians)
            self.mu = nn.Linear(in_feat, n_gaussians)
            self.log_sigma = nn.Linear(in_feat, n_gaussians)

        def forward(self, x):
            pi = torch.softmax(self.pi_linear(x), dim=-1)
            mu = self.mu(x)
            sigma = torch.exp(self.log_sigma(x)).clamp(min=1e-3)
            return pi, mu, sigma

    def mdn_nll(pi, mu, sigma, target):
        """Negative log-likelihood of scalar target under mixture. All shapes broadcast-compatible.

        pi, mu, sigma: (..., K). target: (...,).
        Returns scalar mean NLL.
        """
        y = target.unsqueeze(-1)
        log_gauss = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((y - mu) / sigma) ** 2
        )
        log_mix = torch.log(pi + 1e-12) + log_gauss
        return -torch.logsumexp(log_mix, dim=-1).mean()

    class Seq2ScalarMDN(nn.Module):
        """Autoregressive sliding-window encoder + MLP trunk + MDN head.

        At each future step i:
          - Encode current window -> top hidden state h
          - MLP trunk on [h, stim_i] -> features
          - MDN head -> (pi, mu, sigma) over delta CNR
          - Point estimate = pi-weighted mean of mu; roll window forward with it (or teacher delta).
        """
        def __init__(
            self,
            encoder_dim,
            stim_dim,
            hidden_dim,
            num_layers,
            n_gaussians=3,
            mlp_hidden=None,
            n_mlp_layers=2,
            dropout=0.1,
            **kwargs,
        ):
            super().__init__()
            if mlp_hidden is None:
                mlp_hidden = hidden_dim
            self.n_gaussians = n_gaussians
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers, dropout)

            layers = [nn.Linear(hidden_dim + stim_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(n_mlp_layers - 1):
                layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout)]
            self.trunk = nn.Sequential(*layers)
            self.head = MDNHead(mlp_hidden, n_gaussians)

        def _step(self, h_top, stim_i):
            feats = self.trunk(torch.cat([h_top, stim_i], dim=-1))
            return self.head(feats)  # (B,K),(B,K),(B,K)

        def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
            B, H, _ = encoder_input.shape
            F = future_stim.shape[1]

            current_window = encoder_input
            pis, mus, sigmas = [], [], []

            for i in range(F):
                h, _ = self.encoder(current_window)
                pi, mu, sigma = self._step(h[-1], future_stim[:, i, :])
                pis.append(pi); mus.append(mu); sigmas.append(sigma)

                if i < F - 1:
                    last_abs = current_window[:, -1, 0:1]  # (B, 1)
                    use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                    if use_teacher:
                        delta = targets[:, i:i+1]
                    else:
                        point = (pi * mu).sum(dim=-1, keepdim=True)  # (B,1)
                        delta = point
                    next_cnr_abs = last_abs + delta
                    next_input = torch.cat([next_cnr_abs, future_stim[:, i, :]], dim=-1).unsqueeze(1)
                    current_window = torch.cat([current_window[:, 1:, :], next_input], dim=1)

            return (
                torch.stack(pis, dim=1),     # (B, F, K)
                torch.stack(mus, dim=1),     # (B, F, K)
                torch.stack(sigmas, dim=1),  # (B, F, K)
            )

        def point_pred(self, pi, mu):
            return (pi * mu).sum(dim=-1)  # (B, F)

        def pred_std(self, pi, mu, sigma):
            mean = (pi * mu).sum(dim=-1, keepdim=True)
            var = (pi * (sigma ** 2 + (mu - mean) ** 2)).sum(dim=-1)
            return torch.sqrt(var.clamp(min=1e-12))

        def loss(self, preds, target):
            pi, mu, sigma = preds
            return mdn_nll(pi, mu, sigma, target)


@app.cell
def _():
    import marimo as mo
    import torch.optim as optim
    import numpy as np
    import polars as pl
    import os
    import time
    import tempfile
    from hastyplot import qplot
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset, Subset

    from experiment import ExperimentTracker, compute_training_stats
    from utils import get_device, get_username, running_on_cluster, results_write_path, parse_bool
    from experiments.seq2seq_data import load, AVAILABLE_DATASETS, STIM_COLS

    device = get_device()
    n_stim = len(STIM_COLS)

    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()
    return (
        AVAILABLE_DATASETS,
        DataLoader,
        Dataset,
        ExperimentTracker,
        STIM_COLS,
        Subset,
        compute_training_stats,
        device,
        hostname,
        is_cluster,
        load,
        mo,
        n_stim,
        np,
        optim,
        os,
        parse_bool,
        pl,
        qplot,
        results_base,
        tempfile,
        time,
        train_test_split,
    )


@app.cell
def _(AVAILABLE_DATASETS, mo, parse_bool):
    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "lstm_seq2scal_mdn")
    DRY_RUN = parse_bool(args.get("dry_run", True))
    _cli_source = args.get("source", None)

    source_selector = mo.ui.dropdown(
        options=list(AVAILABLE_DATASETS), value=_cli_source or "synthetic", label="Data source"
    )
    mo.hstack([source_selector], gap=2)
    return DRY_RUN, EXPERIMENT_NAME, args, source_selector


@app.cell
def _(DRY_RUN, EXPERIMENT_NAME, args, mo, source_selector):
    DATA_SOURCE = source_selector.value

    config = dict(
        hidden_dim=int(args.get("hidden_dim", "16" if DRY_RUN else "64")),
        num_layers=int(args.get("num_layers", "2")),
        history_len=int(args.get("history_len", "30")),
        future_len=int(args.get("future_len", "5")),
        lr=float(args.get("lr", "1e-3")),
        epochs=int(args.get("epochs", "50" if DRY_RUN else "400")),
        batch_size=int(args.get("batch_size", "64")),
        patience=int(args.get("patience", "20" if DRY_RUN else "100")),
        tf_ratio_start=float(args.get("tf_ratio_start", "1.0")),
        tf_ratio_end=float(args.get("tf_ratio_end", "0.0")),
        tf_anneal_frac=float(args.get("tf_anneal_frac", "0.5")),
        tf_hold_frac=float(args.get("tf_hold_frac", "0.3")),
        dropout=float(args.get("dropout", "0.1")),
        mlp_hidden=int(args["mlp_hidden"]) if args.get("mlp_hidden") else None,
        n_mlp_layers=int(args.get("n_mlp_layers", "5")),
        n_gaussians=int(args.get("n_gaussians", "3")),
    )
    print(config)

    mo.md(f"""
    # LSTM encoder + MLP trunk + MDN head: `{EXPERIMENT_NAME}`

    Same AR rollout as `lstm_seq2scal_anneal`, but the scalar regressor on the delta is replaced
    by an MDN head with {config['n_gaussians']} Gaussian components. Training objective is
    mixture NLL per step (summed, mean-reduced).

    Point prediction for rollout feedback = π-weighted mean of μ. Uncertainty surfaces as
    mixture std dev at eval time.

    | param | value |
    |-------|-------|
    | source | {DATA_SOURCE} |
    | hidden_dim | {config['hidden_dim']} |
    | num_layers | {config['num_layers']} |
    | history_len / future_len | {config['history_len']} / {config['future_len']} |
    | lr | {config['lr']} |
    | epochs | {config['epochs']} |
    | batch_size | {config['batch_size']} |
    | patience | {config['patience']} |
    | tf schedule (linear) | {config['tf_ratio_start']} → {config['tf_ratio_end']} (hold={config['tf_hold_frac']}, anneal={config['tf_anneal_frac']}) |
    | dropout | {config['dropout']} |
    | mlp_hidden | {config['mlp_hidden'] if config['mlp_hidden'] is not None else 'auto'} |
    | n_mlp_layers | {config['n_mlp_layers']} |
    | n_gaussians | {config['n_gaussians']} |
    | dry_run | {DRY_RUN} |
    """)
    return DATA_SOURCE, config


@app.cell
def _(mo):
    _headless = "name" in mo.cli_args()
    load_data_button = mo.ui.run_button(label="Load data & prepare datasets")
    train_button = mo.ui.run_button(label="Start training")
    mo.hstack([load_data_button, train_button], gap=1) if not _headless else None
    return load_data_button, train_button


@app.cell
def _(
    DATA_SOURCE,
    DRY_RUN,
    DataLoader,
    Dataset,
    STIM_COLS,
    Subset,
    config,
    load,
    load_data_button,
    mo,
    n_stim,
    np,
    train_test_split,
):
    _headless = "name" in mo.cli_args()
    mo.stop(not _headless and not load_data_button.value, mo.md("Click **Load data & prepare datasets** to continue."))
    H = config["history_len"]
    F_ = config["future_len"]
    total_window = H + F_

    if DATA_SOURCE == "real":
        cnr_all, stim_all, conditions_all = load(
            "real", window_size=total_window, stride=max(1, total_window // 4),
        )
    else:
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
                    enc_in = np.concatenate(
                        [enc_cnr[:, np.newaxis], enc_stim.T], axis=-1,
                    )
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
        n_dry = 5000
        train_ds = Subset(train_ds, range(min(n_dry, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_dry, len(val_ds)) // 4))
        test_ds = Subset(test_ds, range(min(n_dry, len(test_ds))))

    BS = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

    mo.md(f"""
    **Data:** {n_traj} trajectories × {traj_len} timepoints ({DATA_SOURCE})

    Encoder input: CNR + {n_stim} stim features ({', '.join(STIM_COLS)}) over {H} history steps
    Decoder input: {n_stim} stim features over {F_} future steps → MDN over delta CNR

    Train: {len(train_ds)} windows | Val: {len(val_ds)} | Test: {len(test_ds)}
    """)
    return F_, H, test_ds, train_loader, val_loader


@app.cell
def _(config, device, mo, n_stim):
    encoder_dim = 1 + n_stim
    stim_dim = n_stim

    model = Seq2ScalarMDN(
        encoder_dim=encoder_dim,
        stim_dim=stim_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        n_gaussians=config["n_gaussians"],
        mlp_hidden=config["mlp_hidden"],
        n_mlp_layers=config["n_mlp_layers"],
        dropout=config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    _mlp_h = config["mlp_hidden"] if config["mlp_hidden"] is not None else config["hidden_dim"]
    mo.md(f"""
    | model | params | K | mlp_hidden | n_mlp_layers |
    |-------|-------:|--:|-----------:|-------------:|
    | `Seq2ScalarMDN` | {n_params:,} | {config['n_gaussians']} | {_mlp_h} | {config['n_mlp_layers']} |

    encoder_in={encoder_dim} | stim_dim={stim_dim} | hidden={config['hidden_dim']} | layers={config['num_layers']} | `{device}`
    """)
    return (model,)


@app.cell
def _(
    DATA_SOURCE,
    EXPERIMENT_NAME,
    ExperimentTracker,
    config,
    device,
    mo,
    model,
    n_stim,
    np,
    optim,
    os,
    results_base,
    tempfile,
    time,
    train_button,
    train_loader,
    val_loader,
):
    _model_config = dict(
        encoder_dim=1 + n_stim,
        stim_dim=n_stim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        history_len=config["history_len"],
        future_len=config["future_len"],
        data_source=DATA_SOURCE,
        n_gaussians=config["n_gaussians"],
        mlp_hidden=config["mlp_hidden"] if config["mlp_hidden"] is not None else config["hidden_dim"],
        n_mlp_layers=config["n_mlp_layers"],
        variant="seq2scalar_mdn_ar_tf",
    )

    def _tf_schedule_linear(cfg):
        start, end = cfg["tf_ratio_start"], cfg["tf_ratio_end"]
        frac, hold = cfg["tf_anneal_frac"], cfg["tf_hold_frac"]

        def schedule(epoch, total):
            hold_epochs = int(total * hold)
            anneal_epochs = max(int(total * frac) - 1, 1)
            if epoch < hold_epochs:
                p = 0.0
            else:
                p = min((epoch - hold_epochs) / anneal_epochs, 1.0)
            return start + (end - start) * p
        return schedule

    def _run_epoch(model, loader, device, optimizer, cfg, epoch, is_train, tf_fn=None):
        if is_train:
            model.train()
            tf_ratio = tf_fn(epoch, cfg["epochs"]) if tf_fn is not None else 0.0
        else:
            model.eval()
            tf_ratio = 0.0
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in, dec_stim, dec_target = enc_in.to(device), dec_stim.to(device), dec_target.to(device)
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

    def train_model(mdl, train_loader, val_loader, cfg, device, tracker):
        opt = optim.Adam(mdl.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        tf_fn = _tf_schedule_linear(cfg)
        hist = {"train_loss": [], "val_loss": [], "tf_ratio": []}

        ckpt_fd, ckpt = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd)

        best, wait = float("inf"), 0
        for epoch in range(cfg["epochs"]):
            t, tf_r = _run_epoch(mdl, train_loader, device, opt, cfg, epoch, True, tf_fn=tf_fn)
            v, _ = _run_epoch(mdl, val_loader, device, opt, cfg, epoch, False)
            hist["train_loss"].append(t)
            hist["val_loss"].append(v)
            hist["tf_ratio"].append(tf_r)
            sched.step(v)
            if v < best:
                best, wait = v, 0
                torch.save(mdl.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | tf={tf_r:.2f} T:{t:.5f} V:{v:.5f}")

            _cur = {k: w.clone() for k, w in mdl.state_dict().items()}
            mdl.load_state_dict(torch.load(ckpt, weights_only=True))
            tracker.checkpoint(mdl, training_results={"history": hist})
            mdl.load_state_dict(_cur)

        mdl.load_state_dict(torch.load(ckpt, weights_only=True))
        os.remove(ckpt)
        return hist

    _headless = "name" in mo.cli_args()
    mo.stop(not _headless and not train_button.value, mo.md("Click **Start training** when ready."))

    _exp_dir = mo.cli_args().get("results-dir", f"{results_base}/{EXPERIMENT_NAME}")
    tracker = ExperimentTracker(
        directory=_exp_dir,
        name=EXPERIMENT_NAME,
        model_config=_model_config,
        training_config=config,
    )
    tracker.register_start()

    _t0 = time.time()
    history = train_model(model, train_loader, val_loader, config, device, tracker)
    train_elapsed = time.time() - _t0

    mo.md(f"""
    **Training complete** in {train_elapsed:.0f}s — {len(history['train_loss'])} epochs

    Final train NLL: {history['train_loss'][-1]:.4f}  |  Final val NLL: {history['val_loss'][-1]:.4f}
    """)
    return history, tracker, train_elapsed


@app.cell
def _(history, pl, qplot):
    loss_df = pl.DataFrame({
        "epoch": list(range(len(history["train_loss"]))) * 2,
        "nll": history["train_loss"] + history["val_loss"],
        "split": ["train"] * len(history["train_loss"]) + ["val"] * len(history["val_loss"]),
    })
    fig_loss = qplot(loss_df, "epoch", "nll", color="split", mark="line",
                     title="MDN NLL curves", height=300)
    fig_loss
    return (fig_loss,)


@app.cell
def _(history, pl, qplot):
    tf_df = pl.DataFrame({
        "epoch": list(range(len(history["tf_ratio"]))),
        "tf_ratio": history["tf_ratio"],
    })
    fig_tf = qplot(tf_df, "epoch", "tf_ratio", mark="line",
                   title="Teacher-forcing schedule", height=200)
    fig_tf
    return (fig_tf,)


@app.cell
def _(DataLoader, device, model, np, test_ds):
    """Collect full-test-set MDN outputs + derived point pred / std."""
    _last, _act, _pi_all, _mu_all, _sig_all, _stim_all = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for _enc, _stim, _tgt in DataLoader(test_ds, batch_size=512):
            _enc_d, _stim_d = _enc.to(device), _stim.to(device)
            pi_t, mu_t, sig_t = model(_enc_d, _stim_d)
            _last.append(_enc_d[:, -1, 0].cpu().numpy())
            _act.append(_tgt.numpy())
            _pi_all.append(pi_t.cpu().numpy())
            _mu_all.append(mu_t.cpu().numpy())
            _sig_all.append(sig_t.cpu().numpy())
            _stim_all.append(_stim_d[:, :, 0].mean(dim=1).cpu().numpy())

    test_last = np.concatenate(_last)
    test_act = np.concatenate(_act)                 # (N, F) deltas
    test_pi = np.concatenate(_pi_all)               # (N, F, K)
    test_mu = np.concatenate(_mu_all)
    test_sigma = np.concatenate(_sig_all)
    test_stim = np.concatenate(_stim_all)

    test_point = (test_pi * test_mu).sum(axis=-1)   # (N, F)
    _mean_exp = test_point[..., None]
    test_std = np.sqrt((test_pi * (test_sigma ** 2 + (test_mu - _mean_exp) ** 2)).sum(axis=-1))

    test_act_abs = test_last[:, None] + np.cumsum(test_act, axis=1)
    test_point_abs = test_last[:, None] + np.cumsum(test_point, axis=1)
    return test_act, test_point, test_std


@app.cell
def _(F_, np, pl, qplot, test_act, test_point, test_std):
    """Per-step residual + uncertainty stats."""
    per_step = pl.DataFrame({
        "step": np.repeat(np.arange(1, F_ + 1), test_act.shape[0]),
        "residual": (test_act - test_point).flatten(order="F"),
        "pred_std": test_std.flatten(order="F"),
    })

    fig_residuals = qplot(
        per_step, "residual",
        facet_wrap="step", columns=F_,
        title="Residual distribution per forecast step",
        height=200, bins=50,
    )
    fig_residuals
    return (fig_residuals,)


@app.cell
def _(F_, np, pl, qplot, test_std):
    std_df = pl.DataFrame({
        "step": np.repeat(np.arange(1, F_ + 1), test_std.shape[0]),
        "pred_std": test_std.flatten(order="F"),
    })
    fig_std = qplot(std_df, "step", "pred_std", mark="boxplot",
                    title="Predicted std by forecast step", height=300)
    fig_std
    return (fig_std,)


@app.cell
def _(F_, H, device, mo, model, np, pl, qplot, test_ds):
    """Sample trajectories with uncertainty band."""
    _n = 8
    _idx_arr = np.linspace(0, len(test_ds) - 1, _n, dtype=int)
    _rows = []
    model.eval()
    with torch.no_grad():
        for _i in _idx_arr:
            _enc_in, _dec_stim, _dec_target = test_ds[int(_i)]
            _pi_t, _mu_t, _sig_t = model(
                _enc_in.unsqueeze(0).to(device),
                _dec_stim.unsqueeze(0).to(device),
            )
            _pi_np = _pi_t.cpu().numpy()[0]
            _mu_np = _mu_t.cpu().numpy()[0]
            _sig_np = _sig_t.cpu().numpy()[0]
            _point = (_pi_np * _mu_np).sum(axis=-1)
            _std = np.sqrt((_pi_np * (_sig_np ** 2 + (_mu_np - _point[:, None]) ** 2)).sum(axis=-1))

            _hist_cnr = _enc_in[:, 0].numpy()
            _last_val = float(_hist_cnr[-1])
            _actual_abs = _last_val + np.cumsum(_dec_target.numpy())
            _pred_abs = _last_val + np.cumsum(_point)
            _std_abs = np.sqrt(np.cumsum(_std ** 2))

            for _t, _v in enumerate(_hist_cnr):
                _rows.append(dict(window=int(_i), t=int(_t), cnr=float(_v), kind="history"))
            for _t in range(F_):
                _rows.append(dict(window=int(_i), t=H + _t, cnr=float(_actual_abs[_t]), kind="actual"))
                _rows.append(dict(window=int(_i), t=H + _t, cnr=float(_pred_abs[_t]), kind="pred_mean"))
                _rows.append(dict(window=int(_i), t=H + _t, cnr=float(_pred_abs[_t] + _std_abs[_t]), kind="pred_upper"))
                _rows.append(dict(window=int(_i), t=H + _t, cnr=float(_pred_abs[_t] - _std_abs[_t]), kind="pred_lower"))

    traj_df = pl.DataFrame(_rows)
    fig_traj = qplot(
        traj_df, "t", "cnr", color="kind", group="kind",
        facet_wrap="window", columns=4, mark="line",
        title="Sample trajectories: actual vs MDN point ± cum-std", height=250,
    )
    mo.vstack([mo.md("## Sample trajectories"), fig_traj])
    return (fig_traj,)


@app.cell
def _(F_, np, pl, qplot, test_act, test_point, test_std):
    """Calibration: fraction of residuals within ±k·σ."""
    _abs_resid = np.abs(test_act - test_point)
    _rows_cal = []
    for _k in [1.0, 2.0, 3.0]:
        for _s in range(F_):
            _cov = float(np.mean(_abs_resid[:, _s] <= _k * test_std[:, _s]))
            _rows_cal.append(dict(step=_s + 1, k_sigma=f"{_k:.0f}σ", coverage=_cov))
    calib_df = pl.DataFrame(_rows_cal)

    fig_calib = qplot(
        calib_df, "step", "coverage", color="k_sigma", mark="line",
        title="Empirical coverage (|residual| ≤ k·σ) per step", height=300,
    )
    fig_calib
    return (fig_calib,)


@app.cell
def _(mo, np, test_act, test_point, test_std):
    _mse = float(np.mean((test_act - test_point) ** 2))
    _mae = float(np.mean(np.abs(test_act - test_point)))
    _nll_proxy = float(np.mean(0.5 * ((test_act - test_point) / test_std) ** 2 + np.log(test_std)))
    _mean_std = float(test_std.mean())

    eval_metrics = dict(
        test_mse_point=_mse,
        test_mae_point=_mae,
        test_nll_gaussian_proxy=_nll_proxy,
        test_mean_std=_mean_std,
    )
    mo.md(f"""
    ## Evaluation summary

    | metric | value |
    |--------|------:|
    | Point MSE (π·μ vs actual δ) | {_mse:.6f} |
    | Point MAE | {_mae:.6f} |
    | Mean predicted σ | {_mean_std:.4f} |
    | Gaussian-proxy NLL | {_nll_proxy:.4f} |
    """)
    return (eval_metrics,)


@app.cell
def _(mo, test_ds):
    traj_selector = mo.ui.slider(0, len(test_ds) - 1, value=0, label="Test window index")
    traj_selector
    return (traj_selector,)


@app.cell
def _(F_, H, device, mo, model, np, pl, test_ds, traj_selector):
    import altair as _alt

    _N_MC = 200
    _idx = traj_selector.value
    _enc_in, _dec_stim, _dec_target = test_ds[_idx]

    _enc_batch = _enc_in.unsqueeze(0).repeat(_N_MC, 1, 1).to(device)
    _stim_batch = _dec_stim.unsqueeze(0).repeat(_N_MC, 1, 1).to(device)
    _current = _enc_batch.clone()

    _sample_paths = []
    model.eval()
    with torch.no_grad():
        for _i in range(F_):
            _h, _ = model.encoder(_current)
            _pi, _mu, _sig = model._step(_h[-1], _stim_batch[:, _i, :])
            _comp = torch.multinomial(_pi, 1).squeeze(-1)
            _rng = torch.arange(_N_MC, device=device)
            _mu_sel = _mu[_rng, _comp]
            _sig_sel = _sig[_rng, _comp]
            _delta = torch.normal(_mu_sel, _sig_sel).unsqueeze(-1)
            _last_abs = _current[:, -1, 0:1]
            _next_abs = _last_abs + _delta
            _next_in = torch.cat(
                [_next_abs, _stim_batch[:, _i, :]], dim=-1
            ).unsqueeze(1)
            _current = torch.cat([_current[:, 1:, :], _next_in], dim=1)
            _sample_paths.append(_next_abs.squeeze(-1).cpu().numpy())

    _samples = np.stack(_sample_paths, axis=1)

    _hist_cnr = _enc_in[:, 0].numpy()
    _last_val = float(_hist_cnr[-1])
    _actual_abs = _last_val + np.cumsum(_dec_target.numpy())
    _t_hist = np.arange(H)
    _t_fut = np.arange(H, H + F_)

    _q05 = np.quantile(_samples, 0.05, axis=0)
    _q25 = np.quantile(_samples, 0.25, axis=0)
    _q50 = np.quantile(_samples, 0.50, axis=0)
    _q75 = np.quantile(_samples, 0.75, axis=0)
    _q95 = np.quantile(_samples, 0.95, axis=0)
    _mean = _samples.mean(axis=0)

    stats_df = pl.DataFrame(
        {
            "t": _t_fut,
            "mean": _mean,
            "q05": _q05,
            "q25": _q25,
            "q50": _q50,
            "q75": _q75,
            "q95": _q95,
        }
    )
    history_df = pl.DataFrame({"t": _t_hist, "cnr": _hist_cnr})
    actual_df = pl.DataFrame({"t": _t_fut, "cnr": _actual_abs})
    _bridge = pl.DataFrame(
        {"t": [H - 1, H], "cnr": [_hist_cnr[-1], _actual_abs[0]]}
    )
    actual_line_df = pl.concat([_bridge, actual_df])

    _band_outer = (
        _alt.Chart(stats_df)
        .mark_area(opacity=0.15, color="#c0392b")
        .encode(
            x=_alt.X("t:Q", title="timestep"),
            y=_alt.Y("q05:Q", title="CNR"),
            y2="q95:Q",
        )
    )
    _band_inner = (
        _alt.Chart(stats_df)
        .mark_area(opacity=0.30, color="#c0392b")
        .encode(
            x="t:Q",
            y="q25:Q",
            y2="q75:Q",
        )
    )
    _median = (
        _alt.Chart(stats_df)
        .mark_line(color="#c0392b", strokeWidth=2)
        .encode(
            x="t:Q",
            y="q50:Q",
            tooltip=["t", "q50", "q05", "q95"],
        )
    )
    _mean_line = (
        _alt.Chart(stats_df)
        .mark_line(color="#c0392b", strokeDash=[4, 3], opacity=0.6)
        .encode(
            x="t:Q",
            y="mean:Q",
        )
    )
    _hist_line = (
        _alt.Chart(history_df)
        .mark_line(color="#2c3e50", strokeWidth=2)
        .encode(x="t:Q", y="cnr:Q")
    )
    _act_line = (
        _alt.Chart(actual_line_df)
        .mark_line(color="#2c3e50", strokeWidth=2, opacity=0.5)
        .encode(x="t:Q", y="cnr:Q")
    )
    _boundary = (
        _alt.Chart(pl.DataFrame({"t": [H]}))
        .mark_rule(color="gray", strokeDash=[2, 3])
        .encode(x="t:Q")
    )

    _chart = (
        _band_outer
        + _band_inner
        + _mean_line
        + _median
        + _hist_line
        + _act_line
        + _boundary
    ).properties(
        width=750,
        height=400,
        title=f"Window {_idx}: history (solid) | actual future (faded) | MDN median + 50/90% MC bands",
    )

    mo.vstack(
        [
            mo.md(
                f"**Window {_idx}** — {_N_MC} Monte Carlo rollouts from MDN (sample component + Gaussian per step)"
            ),
            _chart,
        ]
    )
    return


@app.cell
def _(traj_selector):
    traj_selector
    return


@app.cell(hide_code=True)
def _(F_, H, device, mo, model, np, pl, test_ds, traj_selector):
    import altair as _alt

    _idx_k = traj_selector.value
    _enc_in_k, _dec_stim_k, _dec_target_k = test_ds[_idx_k]

    _K = model.n_gaussians
    _enc_b = _enc_in_k.unsqueeze(0).repeat(_K, 1, 1).to(device)
    _stim_b = _dec_stim_k.unsqueeze(0).repeat(_K, 1, 1).to(device)
    _cur = _enc_b.clone()
    _rng = torch.arange(_K, device=device)

    _abs_paths, _pi_paths, _mu_paths, _sig_paths = [], [], [], []
    model.eval()
    with torch.no_grad():
        for _i in range(F_):
            _h, _ = model.encoder(_cur)
            _pi_b, _mu_b, _sig_b = model._step(_h[-1], _stim_b[:, _i, :])
            _mu_sel = _mu_b[_rng, _rng]
            _pi_sel = _pi_b[_rng, _rng]
            _sig_sel = _sig_b[_rng, _rng]
            _delta = _mu_sel.unsqueeze(-1)
            _last_abs = _cur[:, -1, 0:1]
            _next_abs = _last_abs + _delta
            _next_in = torch.cat([_next_abs, _stim_b[:, _i, :]], dim=-1).unsqueeze(
                1
            )
            _cur = torch.cat([_cur[:, 1:, :], _next_in], dim=1)
            _abs_paths.append(_next_abs.squeeze(-1).cpu().numpy())
            _pi_paths.append(_pi_sel.cpu().numpy())
            _mu_paths.append(_mu_sel.cpu().numpy())
            _sig_paths.append(_sig_sel.cpu().numpy())

    _abs_k = np.stack(_abs_paths, axis=0)
    _pi_k = np.stack(_pi_paths, axis=0)
    _mu_k = np.stack(_mu_paths, axis=0)
    _sig_k = np.stack(_sig_paths, axis=0)

    with torch.no_grad():
        _pi_f, _mu_f, _sig_f = model(
            _enc_in_k.unsqueeze(0).to(device),
            _dec_stim_k.unsqueeze(0).to(device),
        )
    _pi_f = _pi_f.cpu().numpy()[0]
    _mu_f = _mu_f.cpu().numpy()[0]
    _sig_f = _sig_f.cpu().numpy()[0]
    _point_delta = (_pi_f * _mu_f).sum(axis=-1)
    _pred_std = np.sqrt(
        (_pi_f * (_sig_f**2 + (_mu_f - _point_delta[:, None]) ** 2)).sum(axis=-1)
    )

    _hist_cnr_k = _enc_in_k[:, 0].numpy()
    _last_val_k = float(_hist_cnr_k[-1])
    _actual_abs_k = _last_val_k + np.cumsum(_dec_target_k.numpy())
    _pred_abs_k = _last_val_k + np.cumsum(_point_delta)
    _pred_std_abs = np.sqrt(np.cumsum(_pred_std**2))
    _t_hist_k = np.arange(H)
    _t_fut_k = np.arange(H, H + F_)

    # --- light stimulation (channel 0): history from enc_in[:, 1], future from dec_stim[:, 0] ---
    _stim_hist = _enc_in_k[:, 1].numpy()
    _stim_fut = _dec_stim_k[:, 0].numpy()
    _all_t = np.concatenate([_t_hist_k, _t_fut_k])
    _all_stim = np.concatenate([_stim_hist, _stim_fut])
    _stim_max = max(float(_all_stim.max()), 1e-8)

    # --- color scale: unified legend for everything, including light stim ---
    _TABLEAU = [
        "#4C78A8",
        "#F58518",
        "#E45756",
        "#72B7B2",
        "#54A24B",
        "#EEC94C",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]
    _comp_names = [f"k{_i}" for _i in range(_K)]
    _series_domain = ["real", "model (π-weighted)", "light stim", *_comp_names]
    _series_range = ["black", "#7f3fbf", "#e6a200", *_TABLEAU[:_K]]
    _color_scale = _alt.Scale(domain=_series_domain, range=_series_range)
    _color_enc = _alt.Color(
        "series:N", scale=_color_scale, legend=_alt.Legend(title="series")
    )

    # --- scale stim into a band at the bottom of the CNR range ---
    _y_data_min = float(
        min(
            _real_cnr_min := min(
                _hist_cnr_k.min(),
                _actual_abs_k.min(),
                _pred_abs_k.min(),
                _abs_k.min(),
            ),
            (_pred_abs_k - _pred_std_abs).min(),
        )
    )
    _y_data_max = float(
        max(
            _hist_cnr_k.max(),
            _actual_abs_k.max(),
            _pred_abs_k.max(),
            _abs_k.max(),
            (_pred_abs_k + _pred_std_abs).max(),
        )
    )
    _y_range_full = max(_y_data_max - _y_data_min, 1e-6)
    _stim_band_bottom = _y_data_min - 0.22 * _y_range_full
    _stim_band_height = 0.15 * _y_range_full
    _stim_top_vals = (
        _stim_band_bottom + (_all_stim / _stim_max) * _stim_band_height
    )

    _stim_df = pl.DataFrame(
        {
            "t": _all_t,
            "stim_top": _stim_top_vals,
            "stim_bottom": [_stim_band_bottom] * len(_all_t),
            "stim_raw": _all_stim,
            "series": ["light stim"] * len(_all_t),
        }
    )

    # --- trajectory dataframes (same as before) ---
    _rows_k = []
    for _k in range(_K):
        _rows_k.append(
            dict(
                t=H - 1,
                series=f"k{_k}",
                cnr=_last_val_k,
                pi=float(_pi_k[0, _k]),
                mu=0.0,
                sigma=0.0,
            )
        )
        for _i in range(F_):
            _rows_k.append(
                dict(
                    t=int(_t_fut_k[_i]),
                    series=f"k{_k}",
                    cnr=float(_abs_k[_i, _k]),
                    pi=float(_pi_k[_i, _k]),
                    mu=float(_mu_k[_i, _k]),
                    sigma=float(_sig_k[_i, _k]),
                )
            )
    comp_k_df = pl.DataFrame(_rows_k)

    _real_cnr = np.concatenate([_hist_cnr_k, _actual_abs_k])
    _real_t = np.concatenate([_t_hist_k, _t_fut_k])
    _real_df = pl.DataFrame(
        {"t": _real_t, "cnr": _real_cnr, "series": ["real"] * len(_real_t)}
    )

    _pred_t = np.array([H - 1, *_t_fut_k.tolist()])
    _pred_y = np.array([_hist_cnr_k[-1], *_pred_abs_k.tolist()])
    _pred_lo = np.array([_hist_cnr_k[-1], *(_pred_abs_k - _pred_std_abs).tolist()])
    _pred_hi = np.array([_hist_cnr_k[-1], *(_pred_abs_k + _pred_std_abs).tolist()])
    _pred_df = pl.DataFrame(
        {
            "t": _pred_t,
            "cnr": _pred_y,
            "lo": _pred_lo,
            "hi": _pred_hi,
            "series": ["model (π-weighted)"] * len(_pred_t),
        }
    )

    # --- layers ---
    _stim_layer = (
        _alt.Chart(_stim_df)
        .mark_area(interpolate="step-after", opacity=0.35)
        .encode(
            x=_alt.X("t:Q", title="timestep"),
            y=_alt.Y("stim_bottom:Q", title="CNR"),
            y2="stim_top:Q",
            color=_color_enc,
            tooltip=["series", "t", "stim_raw"],
        )
    )

    _pred_band = (
        _alt.Chart(_pred_df)
        .mark_area(opacity=0.18)
        .encode(
            x="t:Q",
            y=_alt.Y("lo:Q"),
            y2="hi:Q",
            color=_color_enc,
        )
    )
    _pred_line = (
        _alt.Chart(_pred_df)
        .mark_line(strokeWidth=3.5)
        .encode(
            x="t:Q",
            y="cnr:Q",
            color=_color_enc,
            tooltip=["series", "t", "cnr"],
        )
    )
    _pred_pts = (
        _alt.Chart(_pred_df.filter(pl.col("t") >= H))
        .mark_point(
            size=90,
            filled=True,
            stroke="white",
            strokeWidth=1.5,
        )
        .encode(x="t:Q", y="cnr:Q", color=_color_enc)
    )

    _real_line = (
        _alt.Chart(_real_df)
        .mark_line(strokeWidth=3)
        .encode(
            x="t:Q",
            y="cnr:Q",
            color=_color_enc,
        )
    )
    _real_pts = (
        _alt.Chart(_real_df.filter(pl.col("t") >= H))
        .mark_point(
            size=70,
            filled=True,
            stroke="white",
            strokeWidth=1.2,
        )
        .encode(
            x="t:Q", y="cnr:Q", color=_color_enc, tooltip=["series", "t", "cnr"]
        )
    )

    _comp_lines = (
        _alt.Chart(comp_k_df)
        .mark_line(strokeWidth=1.5, strokeDash=[3, 2])
        .encode(
            x="t:Q",
            y="cnr:Q",
            color=_color_enc,
            detail="series:N",
            opacity=_alt.value(0.45),
        )
    )
    _comp_pts = (
        _alt.Chart(comp_k_df.filter(pl.col("t") >= H))
        .mark_circle(stroke="black", strokeWidth=0.3)
        .encode(
            x="t:Q",
            y="cnr:Q",
            color=_color_enc,
            size=_alt.Size(
                "pi:Q",
                scale=_alt.Scale(range=[10, 500]),
                legend=_alt.Legend(title="π_k (size)"),
            ),
            opacity=_alt.Opacity(
                "pi:Q",
                scale=_alt.Scale(domain=[0, 1], range=[0.15, 1.0]),
                legend=None,
            ),
            tooltip=["series", "t", "pi", "mu", "sigma"],
        )
    )

    _boundary_k = (
        _alt.Chart(pl.DataFrame({"t": [H]}))
        .mark_rule(color="gray", strokeDash=[2, 3])
        .encode(x="t:Q")
    )

    chart_components = (
        (
            _stim_layer
            + _pred_band
            + _comp_lines
            + _comp_pts
            + _pred_line
            + _pred_pts
            + _real_line
            + _real_pts
            + _boundary_k
        )
        .properties(
            width=820,
            height=460,
            title=f"Window {_idx_k}: real | model π-weighted ±σ | K={_K} components | light stim (bottom strip)",
        )
        .resolve_scale(size="independent", opacity="independent")
        .interactive()
    )

    mo.vstack(
        [
            mo.md(
                f"**Window {_idx_k}** — legend covers every series. "
                f"**real** (black): ground truth. "
                f"**model (π-weighted)** (purple): whole-model rollout with ±1 mixture-σ band. "
                f"**k0…kN**: forced per-component rollouts; dot size/opacity = π_k at step. "
                f"**light stim** (amber strip at bottom): stim channel 0 — shape only, scaled to a band below the CNR data."
            ),
            chart_components,
        ]
    )
    return


@app.cell
def _(
    compute_training_stats,
    eval_metrics,
    fig_calib,
    fig_loss,
    fig_residuals,
    fig_std,
    fig_tf,
    fig_traj,
    history,
    hostname,
    is_cluster,
    mo,
    model,
    tracker,
    train_elapsed,
    train_loader,
    val_loader,
):
    _stats = compute_training_stats(
        train_elapsed_s=train_elapsed,
        history=history,
        n_train_samples=len(train_loader.dataset),
        n_val_samples=len(val_loader.dataset),
        model=model,
    )

    _figures = {
        "loss_curves": fig_loss,
        "tf_schedule": fig_tf,
        "residuals": fig_residuals,
        "pred_std_by_step": fig_std,
        "sample_trajectories": fig_traj,
        "coverage": fig_calib,
    }

    _bundle = tracker.save_final(
        model=model,
        training_results={"history": history, "train_elapsed_s": train_elapsed, "stats": _stats},
        metrics=eval_metrics,
        figures=_figures,
    )

    _env = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"
    mo.md(f"**Saved** on {_env}\n\n`{_bundle.save_dir}`")
    return


if __name__ == "__main__":
    app.run()
