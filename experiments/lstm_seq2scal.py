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

    from experiment import ExperimentTracker, compute_training_stats
    from utils import get_device, get_username, running_on_cluster, results_write_path, parse_bool
    from experiments.seq2seq_data import load_synthetic, load_real, STIM_COLS
    from notebooks.experiment.preprocessing import DEFAULT_STIM_COLS

    device = get_device()
    n_stim = len(STIM_COLS)

    hostname = get_username()
    is_cluster = running_on_cluster()
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

    class LSTMDecoder(nn.Module):
        def __init__(self, stim_dim, hidden_dim, num_layers, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                stim_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout,
            )
            _init_forget_bias(self.lstm)
            self.fc_out = nn.Linear(hidden_dim, 1)

        def forward(self, future_stim, h_0, c_0):
            out, _ = self.lstm(future_stim, (h_0, c_0))
            return self.fc_out(out).squeeze(-1)

    class Seq2Seq(nn.Module):
        """Autoregressive rollout with teacher forcing."""
        def __init__(self, encoder_dim, stim_dim, hidden_dim, num_layers, dropout=0.1, **kwargs):
            super().__init__()
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers, dropout)
            self.decoder = LSTMDecoder(stim_dim, hidden_dim, num_layers, dropout)

        def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
            B, H, _ = encoder_input.shape
            F = future_stim.shape[1]
            current_window = encoder_input
            predictions = []
            for i in range(F):
                h, c = self.encoder(current_window)
                pred = self.decoder(future_stim[:, i:i+1, :], h, c)  # (B, 1)
                predictions.append(pred.squeeze(1))
                if i < F - 1:
                    # Window stores absolute CNR; reconstruct from delta before appending
                    last_abs = current_window[:, -1, 0:1]  # (B, 1)
                    use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                    next_cnr_abs = last_abs + (targets[:, i:i+1] if use_teacher else pred)
                    next_input = torch.cat([next_cnr_abs, future_stim[:, i, :]], dim=-1).unsqueeze(1)
                    current_window = torch.cat([current_window[:, 1:, :], next_input], dim=1)
            return torch.stack(predictions, dim=1)  # (B, F)

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)

    class Seq2SeqBaseline(nn.Module):
        """Single-pass: encode history once, decode all future steps in parallel."""
        def __init__(self, encoder_dim, stim_dim, hidden_dim, num_layers, dropout=0.1):
            super().__init__()
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers, dropout)
            self.decoder = LSTMDecoder(stim_dim, hidden_dim, num_layers, dropout)

        def forward(self, encoder_input, future_stim, **kwargs):
            h, c = self.encoder(encoder_input)
            return self.decoder(future_stim, h, c)

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)


    class Seq2Scalar(nn.Module):
        """Autoregressive one-step predictor: sliding-window LSTM encoder + MLP readout.

        At each future step i:
          - Encode the current window -> (h, c)
          - Predict delta from top hidden state + stim_i via MLP
          - Reconstruct absolute CNR, slide window, repeat
        """
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
            # h_top: (B, hidden_dim), stim_i: (B, stim_dim) -> (B,)
            return self.head(torch.cat([h_top, stim_i], dim=-1)).squeeze(-1)

        def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
            B, H, _ = encoder_input.shape
            F = future_stim.shape[1]

            current_window = encoder_input
            predictions = []

            for i in range(F):
                h, _ = self.encoder(current_window)
                pred = self._predict_step(h[-1], future_stim[:, i, :])  # (B,)
                predictions.append(pred)

                if i < F - 1:
                    last_abs = current_window[:, -1, 0:1]  # (B, 1)
                    use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                    delta = targets[:, i:i+1] if use_teacher else pred.unsqueeze(-1)
                    next_cnr_abs = last_abs + delta
                    next_input = torch.cat([next_cnr_abs, future_stim[:, i, :]], dim=-1).unsqueeze(1)
                    current_window = torch.cat([current_window[:, 1:, :], next_input], dim=1)

            return torch.stack(predictions, dim=1)  # (B, F)

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)

    return (
        DataLoader,
        Dataset,
        ExperimentTracker,
        Path,
        STIM_COLS,
        Seq2Scalar,
        Seq2Seq,
        Subset,
        compute_training_stats,
        device,
        hostname,
        is_cluster,
        load_real,
        load_synthetic,
        mo,
        n_stim,
        nn,
        np,
        optim,
        os,
        parse_bool,
        plt,
        results_base,
        tempfile,
        time,
        torch,
        train_test_split,
    )


@app.cell
def _(mo, parse_bool):
    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "lstm_seq2seq")
    DRY_RUN = parse_bool(args.get("dry_run", True))
    _cli_source = args.get("source", None)

    source_selector = mo.ui.dropdown(
        options=["synthetic", "real"], value=_cli_source or "synthetic", label="Data source"
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
        dropout=float(args.get("dropout", "0.1")),
        mlp_hidden=int(args["mlp_hidden"]) if args.get("mlp_hidden") else None,
        n_mlp_layers=int(args.get("n_mlp_layers", "5")),
    )
    print(config)

    mo.md(f"""
    # LSTM encoder + LSTM vs MLP decoder: `{EXPERIMENT_NAME}`

    Compare two autoregressive CNR predictors sharing the same LSTM encoder:
    - **`Seq2Seq`** — LSTM decoder, sliding-window AR rollout with teacher forcing.
    - **`Seq2Scalar`** — MLP head on top hidden state, same sliding-window AR rollout.

    Encoder compresses CNR + stim features history → hidden state.
    Decoder (LSTM or MLP) takes hidden state + future stim features → next-step delta CNR.
    Both use annealed teacher forcing.

    | param | value |
    |-------|-------|
    | source | {DATA_SOURCE} |
    | hidden_dim | {config['hidden_dim']} |
    | num_layers | {config['num_layers']} |
    | history_len | {config['history_len']} |
    | future_len | {config['future_len']} |
    | lr | {config['lr']} |
    | epochs | {config['epochs']} |
    | batch_size | {config['batch_size']} |
    | patience | {config['patience']} |
    | tf_ratio_start | {config['tf_ratio_start']} |
    | tf_ratio_end | {config['tf_ratio_end']} |
    | dropout | {config['dropout']} |
    | mlp_hidden | {config['mlp_hidden'] if config['mlp_hidden'] is not None else f"auto (= hidden_dim = {config['hidden_dim']})"} |
    | n_mlp_layers | {config['n_mlp_layers']} |
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
    load_data_button,
    load_real,
    load_synthetic,
    mo,
    n_stim,
    np,
    torch,
    train_test_split,
):
    _headless = "name" in mo.cli_args()
    mo.stop(not _headless and not load_data_button.value, mo.md("Click **Load data & prepare datasets** to continue."))
    H = config["history_len"]
    F_ = config["future_len"]
    total_window = H + F_

    if DATA_SOURCE == "synthetic":
        cnr_all, stim_all, conditions_all = load_synthetic()
    else:
        cnr_all, stim_all, conditions_all = load_real(
            window_size=total_window, stride=max(1, total_window // 4),
        )

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
                    # Delta targets: diff of full window, starting at the history/future boundary
                    full_window = cnr[i, t : t + total]
                    dec_target = np.diff(full_window)[history_len - 1 : history_len - 1 + future_len]
                    enc_in = np.concatenate(
                        [enc_cnr[:, np.newaxis], enc_stim.T], axis=-1,
                    )
                    self.samples.append((
                        enc_in,          # (H, 1 + n_stim)
                        dec_stim.T,      # (F, n_stim)
                        dec_target,      # (F,) — CNR deltas
                    ))
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
        val_ds = Subset(val_ds, range(min(n_dry, len(val_ds))//4))
        test_ds = Subset(test_ds, range(min(n_dry, len(test_ds))))

    BS = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

    mo.md(f"""
    **Data:** {n_traj} trajectories × {traj_len} timepoints ({DATA_SOURCE})

    Encoder input: CNR + {n_stim} stim features ({', '.join(STIM_COLS)}) over {H} history steps
    Decoder input: {n_stim} stim features over {F_} future steps → predicts CNR

    Train: {len(train_ds)} windows | Val: {len(val_ds)} | Test: {len(test_ds)}
    """)
    return F_, H, test_ds, train_loader, val_loader


@app.cell
def _(Seq2Scalar, Seq2Seq, config, device, mo, n_stim):
    encoder_dim = 1 + n_stim
    stim_dim = n_stim

    model = Seq2Seq(
        encoder_dim=encoder_dim,
        stim_dim=stim_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config['dropout']
    ).to(device)

    model_mlp = Seq2Scalar(
        encoder_dim=encoder_dim,
        stim_dim=stim_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        mlp_hidden=config["mlp_hidden"],
        n_mlp_layers=config["n_mlp_layers"],
        dropout=config['dropout']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_m = sum(p.numel() for p in model_mlp.parameters())
    _mlp_h = config["mlp_hidden"] if config["mlp_hidden"] is not None else config["hidden_dim"]
    mo.md(f"""
    | model | type | params |
    |-------|------|--------|
    | LSTM decoder (AR + TF) | `Seq2Seq` | {n_params:,} |
    | MLP decoder (AR + TF)  | `Seq2Scalar` (mlp_hidden={_mlp_h}, n_mlp_layers={config['n_mlp_layers']}) | {n_params_m:,} |

    encoder_in={encoder_dim} | decoder_in={stim_dim} | hidden={config['hidden_dim']} | layers={config['num_layers']} | `{device}`
    """)
    return model, model_mlp


@app.cell
def _(
    DATA_SOURCE,
    EXPERIMENT_NAME,
    ExperimentTracker,
    config,
    device,
    mo,
    model,
    model_mlp,
    n_stim,
    nn,
    np,
    optim,
    os,
    results_base,
    tempfile,
    time,
    torch,
    train_button,
    train_loader,
    val_loader,
):
    _model_config_shared = dict(
        encoder_dim=1 + n_stim,
        stim_dim=n_stim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        history_len=config["history_len"],
        future_len=config["future_len"],
        data_source=DATA_SOURCE,
    )

    def _run_epoch_ar(model, loader, device, optimizer, cfg, epoch, is_train):
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
        return np.mean(losses), tf_ratio

    def _checkpoint_with_best_weights(tracker, mdl, best_ckpt_path, hist):
        """Swap in best weights, checkpoint, swap back."""
        _cur = {k: v.clone() for k, v in mdl.state_dict().items()}
        mdl.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        tracker.checkpoint(mdl, training_results={"history": hist})
        mdl.load_state_dict(_cur)

    def train_both(model_lstm, model_mlp, train_loader, val_loader, cfg, device,
                   tracker_lstm, tracker_mlp):
        opt_lstm = optim.Adam(model_lstm.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        opt_mlp = optim.Adam(model_mlp.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        sched_lstm = optim.lr_scheduler.ReduceLROnPlateau(opt_lstm, patience=10, factor=0.5)
        sched_mlp = optim.lr_scheduler.ReduceLROnPlateau(opt_mlp, patience=10, factor=0.5)

        epochs, patience = cfg["epochs"], cfg["patience"]
        hist_lstm = {"train_loss": [], "val_loss": []}
        hist_mlp = {"train_loss": [], "val_loss": []}

        ckpt_fd_lstm, ckpt_lstm = tempfile.mkstemp(suffix=".pt")
        ckpt_fd_mlp, ckpt_mlp = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd_lstm)
        os.close(ckpt_fd_mlp)

        best_lstm, best_mlp, wait_lstm, wait_mlp = float("inf"), float("inf"), 0, 0
        done_lstm, done_mlp = False, False

        for epoch in range(epochs):
            if not done_lstm:
                t_lstm, tf = _run_epoch_ar(model_lstm, train_loader, device, opt_lstm, cfg, epoch, True)
                v_lstm, _ = _run_epoch_ar(model_lstm, val_loader, device, opt_lstm, cfg, epoch, False)
                hist_lstm["train_loss"].append(t_lstm)
                hist_lstm["val_loss"].append(v_lstm)
                sched_lstm.step(v_lstm)
                if v_lstm < best_lstm:
                    best_lstm, wait_lstm = v_lstm, 0
                    torch.save(model_lstm.state_dict(), ckpt_lstm)
                else:
                    wait_lstm += 1
                    if wait_lstm >= patience:
                        print(f"[LSTM] Early stopping at epoch {epoch}")
                        done_lstm = True

            if not done_mlp:
                t_mlp, _ = _run_epoch_ar(model_mlp, train_loader, device, opt_mlp, cfg, epoch, True)
                v_mlp, _ = _run_epoch_ar(model_mlp, val_loader, device, opt_mlp, cfg, epoch, False)
                hist_mlp["train_loss"].append(t_mlp)
                hist_mlp["val_loss"].append(v_mlp)
                sched_mlp.step(v_mlp)
                if v_mlp < best_mlp:
                    best_mlp, wait_mlp = v_mlp, 0
                    torch.save(model_mlp.state_dict(), ckpt_mlp)
                else:
                    wait_mlp += 1
                    if wait_mlp >= patience:
                        print(f"[MLP]  Early stopping at epoch {epoch}")
                        done_mlp = True

            if done_lstm and done_mlp:
                break

            if epoch % 20 == 0:
                lstm_str = f"LSTM tf={tf:.2f} T:{t_lstm:.5f} V:{v_lstm:.5f}" if not done_lstm else "LSTM done"
                mlp_str  = f"MLP  tf={tf:.2f} T:{t_mlp:.5f} V:{v_mlp:.5f}"   if not done_mlp  else "MLP done"
                print(f"Epoch {epoch:3d} | {lstm_str} | {mlp_str}")

            _checkpoint_with_best_weights(tracker_lstm, model_lstm, ckpt_lstm, hist_lstm)
            _checkpoint_with_best_weights(tracker_mlp, model_mlp, ckpt_mlp, hist_mlp)

        model_lstm.load_state_dict(torch.load(ckpt_lstm, weights_only=True))
        model_mlp.load_state_dict(torch.load(ckpt_mlp, weights_only=True))
        os.remove(ckpt_lstm)
        os.remove(ckpt_mlp)
        return hist_lstm, hist_mlp

    _headless = "name" in mo.cli_args()
    mo.stop(not _headless and not train_button.value, mo.md("Click **Start training** when ready."))

    _exp_dir = mo.cli_args().get("results-dir", f"{results_base}/{EXPERIMENT_NAME}")
    tracker = ExperimentTracker(
        directory=_exp_dir,
        name=EXPERIMENT_NAME,
        model_config=_model_config_shared,
        training_config=config,
    )
    tracker.register_start()
    tracker_lstm = tracker.make_subexperiment(
        "lstm", model_config=dict(**_model_config_shared, variant="seq2seq_lstm_ar_tf"),
    )
    tracker_mlp = tracker.make_subexperiment(
        "mlp",
        model_config=dict(
            **_model_config_shared,
            variant="seq2scalar_mlp_ar_tf",
            mlp_hidden=config["mlp_hidden"] if config["mlp_hidden"] is not None else config["hidden_dim"],
            n_mlp_layers=config["n_mlp_layers"],
        ),
    )
    tracker_lstm.register_start()
    tracker_mlp.register_start()

    _t0 = time.time()
    history, history_mlp = train_both(
        model, model_mlp, train_loader, val_loader, config, device,
        tracker_lstm, tracker_mlp,
    )
    train_elapsed = time.time() - _t0

    _models_md = mo.md(f"""
    **Training complete** in {train_elapsed:.0f}s

    | model | epochs |
    |-------|--------|
    | LSTM decoder (Seq2Seq) | {len(history['train_loss'])} |
    | MLP decoder (Seq2Scalar) | {len(history_mlp['train_loss'])} |
    """)
    _config_md = mo.md(f"""
    **Config**

    | param | value |
    |-------|-------|
    | hidden_dim | {config['hidden_dim']} |
    | num_layers | {config['num_layers']} |
    | mlp_hidden | {config['mlp_hidden'] if config['mlp_hidden'] is not None else f"= hidden_dim ({config['hidden_dim']})"} |
    | n_mlp_layers | {config['n_mlp_layers']} |
    | history / future | {config['history_len']} / {config['future_len']} |
    | lr | {config['lr']} |
    | batch_size | {config['batch_size']} |
    | patience | {config['patience']} |
    """)
    _metrics_md = mo.md(f"""
    **Metrics**

    | model | final train | final val |
    |-------|-------------|-----------|
    | LSTM | {history['train_loss'][-1]:.4f} | {history['val_loss'][-1]:.4f} |
    | MLP  | {history_mlp['train_loss'][-1]:.4f} | {history_mlp['val_loss'][-1]:.4f} |
    """)
    mo.output.replace(mo.hstack([_models_md, _config_md, _metrics_md], gap=2))
    return history, history_mlp, tracker_lstm, tracker_mlp, train_elapsed


@app.cell
def _(history, history_mlp, plt):
    skip = 3
    fig_loss, _ax = plt.subplots(1, 3, figsize=(18, 4))

    _ax[0].plot(history["train_loss"], label="train", color="tab:red")
    _ax[0].plot(history["val_loss"], label="val", color="tab:red", linestyle="--")
    _ax[0].set_title("LSTM decoder (Seq2Seq AR)")
    _ax[0].set_xlabel("epoch")
    _ax[0].set_ylabel("MSE")
    _ax[0].set_yscale("log")
    _ax[0].legend()

    _ax[1].plot(history_mlp["train_loss"], label="train", color="tab:blue")
    _ax[1].plot(history_mlp["val_loss"], label="val", color="tab:blue", linestyle="--")
    _ax[1].set_title("MLP decoder (Seq2Scalar AR)")
    _ax[1].set_xlabel("epoch")
    _ax[1].set_yscale("log")
    _ax[1].legend()

    _ax[2].plot(history["val_loss"][skip:], label="LSTM val", color="tab:red")
    _ax[2].plot(history_mlp["val_loss"][skip:], label="MLP val", color="tab:blue")
    _ax[2].set_title(f"Val loss comparison (epoch {skip}+)")
    _ax[2].set_xlabel("epoch")
    _ax[2].set_yscale("log")
    _ax[2].legend()

    fig_loss.tight_layout()
    fig_loss
    return (fig_loss,)


@app.cell
def _(F_, H, device, model, model_mlp, np, plt, test_ds, torch):
    _n_examples = 8
    _indices = np.linspace(0, len(test_ds) - 1, _n_examples, dtype=int)

    fig_recon, _axes = plt.subplots(2, 4, figsize=(18, 8))
    _axes = _axes.flatten()

    model.eval()
    model_mlp.eval()
    with torch.no_grad():
        for _ax_i, _idx in enumerate(_indices):
            _enc_in, _dec_stim, _dec_target = test_ds[_idx]
            _enc_in_d = _enc_in.unsqueeze(0).to(device)
            _dec_stim_d = _dec_stim.unsqueeze(0).to(device)

            _pred_lstm_d = model(_enc_in_d, _dec_stim_d).cpu().numpy()[0]
            _pred_mlp_d  = model_mlp(_enc_in_d, _dec_stim_d).cpu().numpy()[0]
            _hist_cnr = _enc_in[:, 0].numpy()
            _hist_u_t = _enc_in[:, 1].numpy()
            _fut_u_t = _dec_stim[:, 0].numpy()
            _last_val = _hist_cnr[-1]
            # reconstruct absolute CNR from deltas
            _actual   = _last_val + np.cumsum(_dec_target.numpy())
            _pred_lstm = _last_val + np.cumsum(_pred_lstm_d)
            _pred_mlp  = _last_val + np.cumsum(_pred_mlp_d)

            _ax = _axes[_ax_i]
            _t_hist = np.arange(H)
            _t_fut = np.arange(H, H + F_)

            _u_max = max(_hist_u_t.max(), _fut_u_t.max(), 1e-8)
            _cnr_max = max(_hist_cnr.max(), _actual.max(), _pred_lstm.max(), _pred_mlp.max(), 1e-8)
            _ls_h = _hist_u_t / _u_max * _cnr_max * 0.5
            _ls_f = _fut_u_t / _u_max * _cnr_max * 0.5

            _ax.fill_between(_t_hist, 0, _ls_h, alpha=0.15, color="gold", step="mid")
            _ax.fill_between(_t_fut, 0, _ls_f, alpha=0.15, color="gold", step="mid")
            _ax.plot(_t_hist, _hist_cnr, color="navy", lw=1.5, label="history")
            _ax.plot(_t_fut, _actual, color="navy", lw=1.5, alpha=0.5, label="actual")
            _ax.plot(_t_fut, _pred_lstm, color="tab:red", lw=1.5, linestyle="--", label="LSTM")
            _ax.plot(_t_fut, _pred_mlp,  color="tab:blue", lw=1.5, linestyle=":", label="MLP")
            _ax.axvline(H, color="gray", linestyle=":", alpha=0.5)
            _mse_lstm = np.mean((_actual - _pred_lstm) ** 2)
            _mse_mlp  = np.mean((_actual - _pred_mlp) ** 2)
            _ax.set_title(f"#{_idx} LSTM:{_mse_lstm:.4f} MLP:{_mse_mlp:.4f}", fontsize=8)
            if _ax_i == 0:
                _ax.legend(fontsize=7)

    fig_recon.suptitle("LSTM decoder (red dashed) vs MLP decoder (blue dotted)", fontsize=12)
    fig_recon.tight_layout()
    fig_recon
    return (fig_recon,)


@app.cell
def _(DataLoader, device, model, model_mlp, np, test_ds, torch):
    # --- collect full test-set predictions (shared across eval cells) ---
    _last_cnr, _actual_all, _pred_lstm_all, _pred_mlp_all = [], [], [], []
    _pred_lstm_nz, _pred_mlp_nz, _fut_stim_all = [], [], []

    model.eval()
    model_mlp.eval()
    with torch.no_grad():
        for _enc, _stim, _tgt in DataLoader(test_ds, batch_size=512):
            _enc_d, _stim_d = _enc.to(device), _stim.to(device)
            _zero_d = torch.zeros_like(_stim_d)

            _last_cnr.append(_enc_d[:, -1, 0].cpu().numpy())
            _actual_all.append(_tgt.numpy())
            _pred_lstm_all.append(model(_enc_d, _stim_d).cpu().numpy())
            _pred_mlp_all.append(model_mlp(_enc_d, _stim_d).cpu().numpy())
            _pred_lstm_nz.append(model(_enc_d, _zero_d).cpu().numpy())
            _pred_mlp_nz.append(model_mlp(_enc_d, _zero_d).cpu().numpy())
            _fut_stim_all.append(_stim_d[:, :, 0].mean(dim=1).cpu().numpy())

    test_last = np.concatenate(_last_cnr)
    test_act   = np.concatenate(_actual_all)      # (N, F) — deltas
    test_lstm  = np.concatenate(_pred_lstm_all)   # (N, F) — predicted deltas
    test_mlp   = np.concatenate(_pred_mlp_all)
    test_lstm0 = np.concatenate(_pred_lstm_nz)
    test_mlp0  = np.concatenate(_pred_mlp_nz)
    test_stim  = np.concatenate(_fut_stim_all)

    # reconstruct absolute CNR for plots that need it
    test_act_abs  = test_last[:, None] + np.cumsum(test_act,  axis=1)
    test_lstm_abs = test_last[:, None] + np.cumsum(test_lstm, axis=1)
    test_mlp_abs  = test_last[:, None] + np.cumsum(test_mlp,  axis=1)

    test_stim_on = test_stim > test_stim.mean()
    return (
        test_act,
        test_act_abs,
        test_last,
        test_lstm,
        test_lstm0,
        test_lstm_abs,
        test_mlp,
        test_mlp0,
        test_mlp_abs,
        test_stim,
        test_stim_on,
    )


@app.cell(hide_code=True)
def _(
    F_,
    np,
    plt,
    test_act,
    test_act_abs,
    test_last,
    test_lstm,
    test_lstm0,
    test_lstm_abs,
    test_mlp,
    test_mlp0,
    test_mlp_abs,
    test_stim_on,
):
    fig_diag, _ax = plt.subplots(2, 3, figsize=(18, 10))

    _ax[0, 0].hist(test_act[:, 0], bins=60, alpha=0.6, color="navy", label=f"step1 μ={test_act[:,0].mean():.3f}")
    _ax[0, 0].hist(test_act[:, -1], bins=60, alpha=0.6, color="steelblue", label=f"step{test_act.shape[1]} μ={test_act[:,-1].mean():.3f}")
    _ax[0, 0].axvline(0, color="black", lw=1, linestyle="--")
    _ax[0, 0].set_xlabel("actual delta CNR")
    _ax[0, 0].set_title("Actual delta distribution (target)")
    _ax[0, 0].legend(fontsize=8)

    for _pred, _color, _lbl in [(test_lstm[:, 0], "tab:red", "LSTM"), (test_mlp[:, 0], "tab:blue", "MLP")]:
        _ax[0, 1].scatter(test_act[:, 0], _pred, s=3, alpha=0.15, color=_color, label=_lbl)
    _lim2 = [test_act[:, 0].min(), test_act[:, 0].max()]
    _ax[0, 1].plot(_lim2, _lim2, "k--", lw=1)
    _ax[0, 1].axvline(0, color="gray", lw=0.5)
    _ax[0, 1].axhline(0, color="gray", lw=0.5)
    _ax[0, 1].set_xlabel("actual delta (step 1)")
    _ax[0, 1].set_ylabel("predicted delta")
    _ax[0, 1].set_title("Delta calibration (step 1)")
    _ax[0, 1].legend(fontsize=8, markerscale=4)

    for _i in range(min(F_, 3)):
        _ax[0, 2].hist(test_act[:, _i] - test_lstm[:, _i], bins=60, alpha=0.4, color="tab:red",  label=f"LSTM step{_i+1}")
        _ax[0, 2].hist(test_act[:, _i] - test_mlp[:, _i],  bins=60, alpha=0.4, color="tab:blue", label=f"MLP step{_i+1}")
    _ax[0, 2].axvline(0, color="black", lw=1)
    _ax[0, 2].set_xlabel("actual delta − predicted delta")
    _ax[0, 2].set_title("Residual distribution (deltas)")
    _ax[0, 2].legend(fontsize=7)

    _sens_lstm = np.abs(test_lstm - test_lstm0).mean(axis=1)
    _sens_mlp  = np.abs(test_mlp  - test_mlp0).mean(axis=1)
    _ax[1, 0].hist(_sens_lstm, bins=60, alpha=0.6, color="tab:red",  label=f"LSTM (mean={_sens_lstm.mean():.4f})")
    _ax[1, 0].hist(_sens_mlp,  bins=60, alpha=0.6, color="tab:blue", label=f"MLP (mean={_sens_mlp.mean():.4f})")
    _ax[1, 0].set_xlabel("|pred(stim) − pred(zero stim)|")
    _ax[1, 0].set_title("Stimulus sensitivity (ablation)")
    _ax[1, 0].legend(fontsize=8)

    _steps = np.arange(1, F_ + 1)
    for _mask, _ls, _lbl in [(test_stim_on, "-", "stim ON"), (~test_stim_on, "--", "stim OFF")]:
        _ax[1, 1].plot(_steps, test_act_abs[_mask].mean(axis=0),  color="navy",     ls=_ls, label=f"actual {_lbl}")
        _ax[1, 1].plot(_steps, test_lstm_abs[_mask].mean(axis=0), color="tab:red",  ls=_ls, label=f"LSTM {_lbl}")
        _ax[1, 1].plot(_steps, test_mlp_abs[_mask].mean(axis=0),  color="tab:blue", ls=_ls, label=f"MLP {_lbl}")
    _ax[1, 1].set_xlabel("future step")
    _ax[1, 1].set_ylabel("mean CNR (absolute)")
    _ax[1, 1].set_title("Mean prediction: stim ON vs OFF")
    _ax[1, 1].legend(fontsize=7)

    _ratio_lstm = test_lstm_abs[:, 0] / np.where(np.abs(test_last) > 1e-8, test_last, 1e-8)
    _ratio_mlp  = test_mlp_abs[:, 0]  / np.where(np.abs(test_last) > 1e-8, test_last, 1e-8)
    _ratio_act  = test_act_abs[:, 0]  / np.where(np.abs(test_last) > 1e-8, test_last, 1e-8)
    _bins = np.linspace(0, 2, 60)
    _ax[1, 2].hist(_ratio_act,  bins=_bins, alpha=0.4, color="navy",     label=f"actual (μ={_ratio_act.mean():.2f})")
    _ax[1, 2].hist(_ratio_lstm, bins=_bins, alpha=0.5, color="tab:red",  label=f"LSTM pred (μ={_ratio_lstm.mean():.2f})")
    _ax[1, 2].hist(_ratio_mlp,  bins=_bins, alpha=0.5, color="tab:blue", label=f"MLP pred (μ={_ratio_mlp.mean():.2f})")
    _ax[1, 2].axvline(1.0, color="black", lw=1, linestyle="--", label="1×")
    _ax[1, 2].axvline(0.5, color="red",   lw=1, linestyle="--", label="0.5×")
    _ax[1, 2].set_xlabel("reconstructed abs pred / last_history_cnr")
    _ax[1, 2].set_title("Prediction ratio to last seen CNR")
    _ax[1, 2].legend(fontsize=7)

    fig_diag.suptitle("Diagnostic plots", fontsize=13)
    fig_diag.tight_layout()
    fig_diag
    return (fig_diag,)


@app.cell
def _(DataLoader, np, plt, test_ds):
    _last_cnr2, _hist_mean2 = [], []
    for _enc, _stim, _tgt in DataLoader(test_ds, batch_size=512):
        _last_cnr2.append(_enc[:, -1, 0].numpy())
        _hist_mean2.append(_enc[:, :, 0].mean(dim=1).numpy())

    _last = np.concatenate(_last_cnr2)
    _mean = np.concatenate(_hist_mean2)
    _ratio = _last / np.where(np.abs(_mean) > 1e-8, _mean, 1e-8)

    fig_enc_diag, _ax = plt.subplots(1, 2, figsize=(12, 5))

    _ax[0].scatter(_mean, _last, s=3, alpha=0.15, color="navy")
    _lim = [min(_mean.min(), _last.min()), max(_mean.max(), _last.max())]
    _ax[0].plot(_lim, _lim, "k--", lw=1, label="last = mean (ratio 1×)")
    _ax[0].plot(_lim, [v * 2 for v in _lim], "r--", lw=1, label="last = 2× mean")
    _ax[0].set_xlabel("history window mean CNR")
    _ax[0].set_ylabel("last history CNR")
    _ax[0].set_title("Last CNR vs history mean\n(encoder transmits mean ≈ 0.5× last?)")
    _ax[0].legend(fontsize=9)

    _ax[1].hist(_ratio, bins=80, color="navy", alpha=0.7)
    _ax[1].axvline(1.0, color="black", lw=1, linestyle="--", label="ratio = 1")
    _ax[1].axvline(2.0, color="red", lw=1, linestyle="--", label="ratio = 2")
    _ax[1].axvline(_ratio.mean(), color="orange", lw=1.5, label=f"mean = {_ratio.mean():.2f}")
    _ax[1].set_xlabel("last_cnr / history_mean")
    _ax[1].set_title("Distribution of last/mean ratio")
    _ax[1].legend(fontsize=9)

    fig_enc_diag.suptitle("Encoder input diagnostic: does history mean ≈ 0.5× last value?", fontsize=12)
    fig_enc_diag.tight_layout()
    fig_enc_diag
    return


@app.cell
def _(DataLoader, np, plt, test_ds):
    # conditional ratio: does actual_step1 / last_cnr drop toward 0.5 for high-CNR windows?
    _last3, _step1_actual3 = [], []
    for _enc, _stim, _tgt in DataLoader(test_ds, batch_size=512):
        _last3.append(_enc[:, -1, 0].numpy())
        _step1_actual3.append(_tgt[:, 0].numpy())

    _last3 = np.concatenate(_last3)
    _delta1 = np.concatenate(_step1_actual3)          # step-1 delta (new target format)
    _step1_abs = _last3 + _delta1                      # reconstruct absolute step-1 CNR
    _cond_ratio = _step1_abs / np.where(np.abs(_last3) > 1e-8, _last3, 1e-8)

    _q25, _q50, _q75 = np.quantile(_last3, [0.25, 0.50, 0.75])
    _masks = {
        "low (< Q25)":    _last3 < _q25,
        "mid (Q25–Q75)": (_last3 >= _q25) & (_last3 < _q75),
        "high (> Q75)":   _last3 >= _q75,
    }

    fig_cond, _ax = plt.subplots(1, 2, figsize=(13, 5))

    _colors = ["steelblue", "goldenrod", "tomato"]
    for (_lbl, _m), _c in zip(_masks.items(), _colors):
        _ax[0].scatter(_last3[_m], _step1_abs[_m], s=3, alpha=0.3, color=_c, label=_lbl)
    _lim = [_last3.min(), _last3.max()]
    _ax[0].plot(_lim, _lim, "k--", lw=1, label="y = x")
    _ax[0].plot(_lim, [v * 0.5 for v in _lim], "r--", lw=1, label="y = 0.5x")
    _ax[0].set_xlabel("last history CNR")
    _ax[0].set_ylabel("actual step-1 CNR (reconstructed)")
    _ax[0].set_title("Step-1 target vs last CNR\n(colored by last-CNR quartile)")
    _ax[0].legend(fontsize=8, markerscale=3)

    for (_lbl, _m), _c in zip(_masks.items(), _colors):
        _ax[1].hist(_cond_ratio[_m], bins=60, alpha=0.5, color=_c,
                    label=f"{_lbl}  μ={_cond_ratio[_m].mean():.2f}")
    _ax[1].axvline(1.0, color="black", lw=1, linestyle="--")
    _ax[1].axvline(0.5, color="red", lw=1, linestyle="--")
    _ax[1].set_xlabel("actual_step1_abs / last_cnr")
    _ax[1].set_title("Conditional ratio by last-CNR quartile")
    _ax[1].legend(fontsize=8)

    fig_cond.tight_layout()
    fig_cond
    return


@app.cell
def _(
    np,
    test_act,
    test_act_abs,
    test_lstm,
    test_lstm_abs,
    test_mlp,
    test_mlp_abs,
):
    mse_per_step_lstm = np.mean((test_lstm - test_act) ** 2, axis=0)
    mse_per_step_mlp  = np.mean((test_mlp  - test_act) ** 2, axis=0)
    mse_per_step_zero = np.mean(test_act ** 2, axis=0)
    _act_mean = test_act.mean(axis=0)
    mse_per_step_mean = np.mean((test_act - _act_mean) ** 2, axis=0)

    _ss_tot = np.sum((test_act - _act_mean) ** 2, axis=0)
    _ss_tot = np.where(_ss_tot > 0, _ss_tot, 1e-8)
    r2_per_step_lstm = 1 - np.sum((test_lstm - test_act) ** 2, axis=0) / _ss_tot
    r2_per_step_mlp  = 1 - np.sum((test_mlp  - test_act) ** 2, axis=0) / _ss_tot

    mae_cum_lstm = np.mean(np.abs(test_lstm_abs - test_act_abs), axis=0)
    mae_cum_mlp  = np.mean(np.abs(test_mlp_abs  - test_act_abs), axis=0)
    mae_cum_lstm_std = np.std(np.abs(test_lstm_abs - test_act_abs), axis=0)
    mae_cum_mlp_std  = np.std(np.abs(test_mlp_abs  - test_act_abs), axis=0)

    mse_window_lstm = np.mean((test_lstm - test_act) ** 2, axis=1)
    mse_window_mlp  = np.mean((test_mlp  - test_act) ** 2, axis=1)

    overall_mse_lstm = float(np.mean(mse_window_lstm))
    overall_mse_mlp  = float(np.mean(mse_window_mlp))
    overall_mse_zero = float(np.mean(test_act ** 2))
    overall_mse_mean = float(mse_per_step_mean.mean())
    _ss_tot_all = np.sum((test_act - test_act.mean()) ** 2)
    overall_r2_lstm = float(1 - np.sum((test_lstm - test_act) ** 2) / max(_ss_tot_all, 1e-8))
    overall_r2_mlp  = float(1 - np.sum((test_mlp  - test_act) ** 2) / max(_ss_tot_all, 1e-8))
    lstm_win_rate = float(np.mean(mse_window_lstm < mse_window_mlp))

    eval_metrics = {
        "test_mse_lstm": overall_mse_lstm,
        "test_mse_mlp": overall_mse_mlp,
        "test_mse_persist_last": overall_mse_zero,
        "test_mse_predict_mean": overall_mse_mean,
        "test_r2_lstm": overall_r2_lstm,
        "test_r2_mlp": overall_r2_mlp,
        "lstm_win_rate": lstm_win_rate,
        "mse_per_step_lstm": mse_per_step_lstm.tolist(),
        "mse_per_step_mlp": mse_per_step_mlp.tolist(),
        "r2_per_step_lstm": r2_per_step_lstm.tolist(),
        "r2_per_step_mlp": r2_per_step_mlp.tolist(),
    }
    return (
        eval_metrics,
        lstm_win_rate,
        mae_cum_lstm,
        mae_cum_lstm_std,
        mae_cum_mlp,
        mae_cum_mlp_std,
        mse_per_step_lstm,
        mse_per_step_mean,
        mse_per_step_mlp,
        mse_per_step_zero,
        mse_window_lstm,
        mse_window_mlp,
        overall_mse_lstm,
        overall_mse_mean,
        overall_mse_mlp,
        overall_mse_zero,
        overall_r2_lstm,
        overall_r2_mlp,
        r2_per_step_lstm,
        r2_per_step_mlp,
    )


@app.cell
def _(
    F_,
    mo,
    mse_per_step_lstm,
    mse_per_step_mean,
    mse_per_step_mlp,
    mse_per_step_zero,
    np,
    plt,
    r2_per_step_lstm,
    r2_per_step_mlp,
):
    _steps = np.arange(1, F_ + 1)
    _w = 0.2

    fig_step_metrics, (_ax_mse, _ax_r2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax_mse.bar(_steps - 1.5 * _w, mse_per_step_zero, _w, label="Persist last", color="#aaa")
    _ax_mse.bar(_steps - 0.5 * _w, mse_per_step_mean, _w, label="Predict mean", color="#ccc")
    _ax_mse.bar(_steps + 0.5 * _w, mse_per_step_mlp,  _w, label="MLP",  color="#dd8452")
    _ax_mse.bar(_steps + 1.5 * _w, mse_per_step_lstm, _w, label="LSTM", color="#4c72b0")
    _ax_mse.set_xlabel("Forecast step")
    _ax_mse.set_ylabel("MSE")
    _ax_mse.set_title("Per-step MSE vs naive baselines")
    _ax_mse.set_xticks(_steps)
    _ax_mse.legend(fontsize=8)

    _ax_r2.plot(_steps, r2_per_step_lstm, "o-", color="#4c72b0", label="LSTM")
    _ax_r2.plot(_steps, r2_per_step_mlp,  "s-", color="#dd8452", label="MLP")
    _ax_r2.axhline(0, color="black", lw=1, linestyle="--", label="R²=0 (naive)")
    _ax_r2.set_xlabel("Forecast step")
    _ax_r2.set_ylabel("R²")
    _ax_r2.set_title("Per-step R² (variance explained)")
    _ax_r2.set_xticks(_steps)
    _ax_r2.legend(fontsize=8)

    fig_step_metrics.tight_layout()

    mo.md("## Per-step forecast quality")
    return (fig_step_metrics,)


@app.cell
def _(
    mo,
    overall_mse_lstm,
    overall_mse_mean,
    overall_mse_mlp,
    overall_mse_zero,
    plt,
):
    _names = ["Persist last\n(zero delta)", "Predict\nmean delta", "MLP", "LSTM"]
    _vals = [overall_mse_zero, overall_mse_mean, overall_mse_mlp, overall_mse_lstm]
    _colors = ["#aaa", "#ccc", "#dd8452", "#4c72b0"]

    fig_baselines, _ax = plt.subplots(figsize=(8, 5))
    _bars = _ax.bar(_names, _vals, color=_colors, edgecolor="black", linewidth=0.5)
    for _bar, _v in zip(_bars, _vals):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _v, f"{_v:.6f}",
                 ha="center", va="bottom", fontsize=9)
    _ax.set_ylabel("Overall MSE")
    _ax.set_title("Models vs naive baselines")
    fig_baselines.tight_layout()

    _pct_lstm = (1 - overall_mse_lstm / overall_mse_zero) * 100
    _pct_mlp  = (1 - overall_mse_mlp  / overall_mse_zero) * 100

    mo.vstack([
        fig_baselines,
        mo.md(f"""
    **LSTM decoder** achieves **{_pct_lstm:.1f}%** lower MSE than persist-last baseline.
    **MLP decoder**  achieves **{_pct_mlp:.1f}%** lower MSE than persist-last baseline.
    """),
    ])
    return (fig_baselines,)


@app.cell
def _(
    F_,
    mae_cum_lstm,
    mae_cum_lstm_std,
    mae_cum_mlp,
    mae_cum_mlp_std,
    mo,
    np,
    plt,
):
    _steps = np.arange(1, F_ + 1)

    fig_cumulative, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_steps, mae_cum_lstm, "o-", color="#4c72b0", label="LSTM")
    _ax.fill_between(_steps, mae_cum_lstm - mae_cum_lstm_std, mae_cum_lstm + mae_cum_lstm_std,
                     alpha=0.15, color="#4c72b0")
    _ax.plot(_steps, mae_cum_mlp, "s-", color="#dd8452", label="MLP")
    _ax.fill_between(_steps, mae_cum_mlp - mae_cum_mlp_std, mae_cum_mlp + mae_cum_mlp_std,
                     alpha=0.15, color="#dd8452")
    _ax.set_xlabel("Forecast step")
    _ax.set_ylabel("MAE (absolute CNR)")
    _ax.set_title("Cumulative trajectory error (absolute CNR)")
    _ax.legend()
    fig_cumulative.tight_layout()

    mo.md("## Cumulative error in reconstructed CNR")
    _ax
    return (fig_cumulative,)


@app.cell
def _(lstm_win_rate, mo, mse_window_lstm, mse_window_mlp, plt, test_stim):
    fig_head2head, _ax = plt.subplots(figsize=(7, 7))
    _sc = _ax.scatter(
        mse_window_mlp, mse_window_lstm,
        s=8, alpha=0.3, c=test_stim, cmap="viridis", edgecolors="none",
    )
    _lim = [0, max(mse_window_lstm.max(), mse_window_mlp.max()) * 1.05]
    _ax.plot(_lim, _lim, "k--", lw=1, label="y = x")
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel("MLP per-window MSE")
    _ax.set_ylabel("LSTM per-window MSE")
    _ax.set_title("LSTM vs MLP head-to-head")
    _ax.text(0.05, 0.95, f"LSTM wins {lstm_win_rate * 100:.1f}% of windows",
             transform=_ax.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    _ax.legend(loc="lower right")
    fig_head2head.colorbar(_sc, ax=_ax, label="Mean stimulus intensity")
    fig_head2head.tight_layout()

    mo.md("## LSTM vs MLP head-to-head")
    _ax
    return (fig_head2head,)


@app.cell
def _(
    lstm_win_rate,
    mo,
    overall_mse_lstm,
    overall_mse_mean,
    overall_mse_mlp,
    overall_mse_zero,
    overall_r2_lstm,
    overall_r2_mlp,
):
    mo.md(f"""
    ## Evaluation summary

    | Metric | LSTM | MLP | Persist-last | Predict-mean |
    |--------|-----:|----:|-------------:|-------------:|
    | **Overall MSE** | {overall_mse_lstm:.6f} | {overall_mse_mlp:.6f} | {overall_mse_zero:.6f} | {overall_mse_mean:.6f} |
    | **Overall R²** | {overall_r2_lstm:.4f} | {overall_r2_mlp:.4f} | — | — |
    | **LSTM win rate** | {lstm_win_rate * 100:.1f}% | — | — | — |
    """)
    return


@app.cell
def _(mo, test_ds):
    traj_selector = mo.ui.slider(
        0, len(test_ds) - 1, value=0, label="Test window index"
    )
    traj_selector
    return (traj_selector,)


@app.cell
def _(
    F_,
    H,
    device,
    mo,
    model,
    model_mlp,
    np,
    plt,
    test_ds,
    torch,
    traj_selector,
):
    _idx = traj_selector.value
    _enc_in, _dec_stim, _dec_target = test_ds[_idx]

    model.eval()
    model_mlp.eval()
    with torch.no_grad():
        _enc_in_d = _enc_in.unsqueeze(0).to(device)
        _dec_stim_d = _dec_stim.unsqueeze(0).to(device)
        _pred_lstm_d = model(_enc_in_d, _dec_stim_d).cpu().numpy()[0]
        _pred_mlp_d  = model_mlp(_enc_in_d, _dec_stim_d).cpu().numpy()[0]

    _hist_cnr = _enc_in[:, 0].numpy()
    _hist_u_t = _enc_in[:, 1].numpy()
    _fut_u_t = _dec_stim[:, 0].numpy()
    _last_val = _hist_cnr[-1]
    # reconstruct absolute CNR from deltas
    _actual    = _last_val + np.cumsum(_dec_target.numpy())
    _pred_lstm = _last_val + np.cumsum(_pred_lstm_d)
    _pred_mlp  = _last_val + np.cumsum(_pred_mlp_d)
    _t_hist = np.arange(H)
    _t_fut = np.arange(H, H + F_)
    _umax = max(_hist_u_t.max(), _fut_u_t.max(), 1e-8)
    _cmax = max(_hist_cnr.max(), _actual.max(), _pred_lstm.max(), _pred_mlp.max(), 1e-8)
    _ls_h = _hist_u_t / _umax * _cmax * 0.5
    _ls_f = _fut_u_t / _umax * _cmax * 0.5

    _fig, _axes2 = plt.subplots(2, 2, figsize=(16, 8), height_ratios=[3, 1], sharex=True)

    for _col, (_pred, _color, _label) in enumerate([
        (_pred_lstm, "tab:red",  "LSTM"),
        (_pred_mlp,  "tab:blue", "MLP"),
    ]):
        _ax_main = _axes2[0, _col]
        _ax_err = _axes2[1, _col]

        _ax_main.fill_between(_t_hist, 0, _ls_h, alpha=0.15, color="gold", step="mid", label="u_t")
        _ax_main.fill_between(_t_fut, 0, _ls_f, alpha=0.15, color="gold", step="mid")
        _ax_main.plot(_t_hist, _hist_cnr, color="navy", lw=2, label="history CNR")
        _ax_main.plot(_t_fut, _actual, color="navy", lw=2, alpha=0.5, label="actual")
        _ax_main.plot(_t_fut, _pred, color=_color, lw=2, linestyle="--", label=f"{_label} pred")
        _ax_main.axvline(H, color="gray", linestyle=":", lw=1.5)
        _ax_main.legend(fontsize=9)
        _ax_main.set_ylabel("CNR")
        _ax_main.set_title(f"Window {_idx} — {_label}")

        _err = _actual - _pred
        _ax_err.bar(_t_fut, _err, color=_color, alpha=0.6, width=0.8)
        _ax_err.axhline(0, color="black", lw=0.5)
        _ax_err.set_ylabel("error")
        _ax_err.set_xlabel("timestep")

    _mse_lstm = np.mean((_actual - _pred_lstm) ** 2)
    _mse_mlp  = np.mean((_actual - _pred_mlp) ** 2)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"**Window {_idx}** | LSTM MSE: {_mse_lstm:.6f} | MLP MSE: {_mse_mlp:.6f}")
    ])
    return


@app.cell
def _(
    Path,
    compute_training_stats,
    eval_metrics,
    fig_baselines,
    fig_cumulative,
    fig_diag,
    fig_head2head,
    fig_loss,
    fig_recon,
    fig_step_metrics,
    history,
    history_mlp,
    hostname,
    is_cluster,
    mo,
    model,
    model_mlp,
    tracker_lstm,
    tracker_mlp,
    train_elapsed,
    train_loader,
    val_loader,
):

    _stats_lstm = compute_training_stats(
        train_elapsed_s=train_elapsed,
        history=history,
        n_train_samples=len(train_loader.dataset),
        n_val_samples=len(val_loader.dataset),
        model=model,
    )
    _stats_mlp = compute_training_stats(
        train_elapsed_s=train_elapsed,
        history=history_mlp,
        n_train_samples=len(train_loader.dataset),
        n_val_samples=len(val_loader.dataset),
        model=model_mlp,
    )

    _figures = {
        "loss_curves": fig_loss,
        "reconstructions": fig_recon,
        "diagnostics": fig_diag,
        "step_metrics": fig_step_metrics,
        "baselines": fig_baselines,
        "cumulative_error": fig_cumulative,
        "head_to_head": fig_head2head,
    }

    _bundle_lstm = tracker_lstm.save_final(
        model=model,
        training_results={"history": history, "train_elapsed_s": train_elapsed, "stats": _stats_lstm},
        metrics=eval_metrics,
        figures=_figures,
    )

    _bundle_mlp = tracker_mlp.save_final(
        model=model_mlp,
        training_results={"history": history_mlp, "train_elapsed_s": train_elapsed, "stats": _stats_mlp},
        metrics=eval_metrics,
        figures=_figures,
    )

    _env_label = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"
    _parent_dir = str(Path(_bundle_lstm.save_dir).parent)
    mo.md(f"**Saved** on {_env_label}\n\n`{_parent_dir}`")
    return


if __name__ == "__main__":
    app.run()
