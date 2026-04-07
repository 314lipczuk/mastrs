import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
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

    from experiment import ExperimentTracker, compute_training_stats, load_experiment_group
    from utils import get_device, get_username, running_on_cluster, results_write_path, results_read_sources, parse_bool, experiment_mode_widget, experiment_mode_state
    from experiments.seq2seq_data import load_synthetic, load_real, STIM_COLS
    from notebooks.experiment.preprocessing import DEFAULT_STIM_COLS

    device = get_device()
    n_stim = len(STIM_COLS)

    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()
    return (
        DEFAULT_STIM_COLS,
        DataLoader,
        Dataset,
        ExperimentTracker,
        Path,
        STIM_COLS,
        Subset,
        compute_training_stats,
        device,
        experiment_mode_state,
        experiment_mode_widget,
        hostname,
        is_cluster,
        load_experiment_group,
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
def _(Path, experiment_mode_widget, mo, parse_bool):
    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "lstm_seq2seq")
    DRY_RUN = parse_bool(args.get("dry_run", True))
    _cli_source = args.get("source", None)

    _mode_widget, mode_ctx = experiment_mode_widget(
        mo,
        experiment_name_default=EXPERIMENT_NAME,
        project_root=Path(__file__).resolve().parent.parent,
        experiment_labels=["Experiment"],
    )

    source_selector = mo.ui.dropdown(
        options=["synthetic", "real"], value=_cli_source or "synthetic", label="Data source"
    )

    mo.vstack([
        mo.hstack([source_selector], gap=2),
        _mode_widget,
    ])
    return DRY_RUN, EXPERIMENT_NAME, args, mode_ctx, source_selector


@app.cell
def _(experiment_mode_state, mo, mode_ctx):
    _load_widget, mode = experiment_mode_state(mo, mode_ctx)
    _load_widget
    return (mode,)


@app.cell(hide_code=True)
def _(DRY_RUN, EXPERIMENT_NAME, args, mo, mode, source_selector):
    DATA_SOURCE = source_selector.value

    # When loading from disk, inherit config from the loaded experiment bundle
    _loaded_cfg = {}
    if mode.is_load and mode.load_button_clicked and mode.selected_experiment_path:
        from experiment import ExperimentBundle as _EB, load_experiment_group as _leg
        _exp_path = mode.selected_experiment_path
        try:
            if (_exp_path / "experiment.json").exists():
                _bundles = _leg(str(_exp_path))
                _b = next(iter(_bundles.values()))
            else:
                _b = _EB.load(str(_exp_path))
            _loaded_cfg = {**(_b.training_config or {}), **(_b.model_config or {})}
            if _b.model_config.get("data_source"):
                DATA_SOURCE = _b.model_config["data_source"]
        except Exception:
            pass

    def _get(key, default):
        if _loaded_cfg and key in _loaded_cfg:
            return _loaded_cfg[key]
        return args.get(key, default)

    config = dict(
        hidden_dim=int(_get("hidden_dim", "16" if DRY_RUN else "32")),
        num_layers=int(_get("num_layers", "2")),
        history_len=int(_get("history_len", "15")),
        future_len=int(_get("future_len", "10")),
        lr=float(_get("lr", "1e-3")),
        epochs=int(_get("epochs", "50" if DRY_RUN else "200")),
        batch_size=int(_get("batch_size", "64")),
        patience=int(_get("patience", "20" if DRY_RUN else "50")),
        tf_ratio_start=float(_get("tf_ratio_start", "1.0")),
        tf_ratio_end=float(_get("tf_ratio_end", "0.0")),
    )

    _source_label = f"**{DATA_SOURCE}** (from loaded experiment)" if _loaded_cfg else DATA_SOURCE

    mo.md(f"""
    # LSTM Encoder-Decoder: `{EXPERIMENT_NAME}`

    Sequence-to-sequence model for CNR prediction.
    Encoder compresses CNR + stim features history → hidden state.
    Decoder takes hidden state + future stim features → predicted CNR.
    Autoregressive rollout with teacher forcing (annealed).

    | param | value |
    |-------|-------|
    | source | {_source_label} |
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
    | dry_run | {DRY_RUN} |
    """)
    return DATA_SOURCE, config


@app.cell
def _(mo, mode):
    load_data_button = mo.ui.run_button(label="Load data & prepare datasets")
    train_button = mo.ui.run_button(label="Start training")

    _buttons = [load_data_button]
    if mode.is_train:
        _buttons.append(train_button)
    mo.output.replace(mo.hstack(_buttons, gap=1))
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
    mode,
    n_stim,
    np,
    torch,
    train_test_split,
):
    mo.stop(not load_data_button.value, mo.md("Click **Load data & prepare datasets** to continue."))
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

    if DRY_RUN and mode.is_train:
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
    return F_, H, cnr_all, stim_all, test_ds, train_loader, val_loader


@app.cell
def _(DATA_SOURCE, DEFAULT_STIM_COLS, cnr_all, load_real, np, plt, stim_all):
    # Second xcorr plot: all available stim features, including ones not in the model.
    # For synthetic: derive ewma_fast + n_5 (pulse count window) from the raw light signal.
    # For real: reload trajectories with all 9 DEFAULT_STIM_COLS.
    if DATA_SOURCE == "synthetic":
        _u_t = stim_all[:, 0, :]  # (N, T)
        # derive ewma_fast (alpha=0.5) inline
        _ef = np.empty_like(_u_t)
        _ef[:, 0] = _u_t[:, 0]
        for _t in range(1, _u_t.shape[1]):
            _ef[:, _t] = 0.5 * _u_t[:, _t] + 0.5 * _ef[:, _t - 1]
        # n_5: number of on-frames in last 5 steps
        _n5 = np.stack([
            np.concatenate([np.zeros((_u_t.shape[0], min(_k, 5))),
                            np.array([(_u_t[:, max(0, _i-5):_i] > 0).sum(axis=1)
                                      for _i in range(_k, _u_t.shape[1])]).T], axis=1)
            for _k in [5]
        ], axis=0)[0].astype(np.float32)
        _all_stim = np.stack([_u_t, stim_all[:, 1, :], _ef, stim_all[:, 2, :],
                               stim_all[:, 3, :], _n5], axis=1)
        _all_cols = ["u_t", "m_t", "ewma_fast", "ewma_slow", "s_cum", "n_5"]
        _cnr_all = cnr_all
    else:
        _cnr_all, _all_stim, _ = load_real()
        _all_cols = DEFAULT_STIM_COLS

    _n_traj2, _traj_len2 = _cnr_all.shape
    _max_lag2 = min(50, _traj_len2 // 2)
    _lags2 = np.arange(-_max_lag2, _max_lag2 + 1)
    _xcorr2 = np.zeros((len(_all_cols), len(_lags2)))

    for _ti in range(_n_traj2):
        _cnr = _cnr_all[_ti] - _cnr_all[_ti].mean()
        _cnr_norm = np.linalg.norm(_cnr) + 1e-8
        for _si in range(len(_all_cols)):
            _s = _all_stim[_ti, _si] - _all_stim[_ti, _si].mean()
            _s_norm = np.linalg.norm(_s) + 1e-8
            _full = np.correlate(_cnr, _s, mode="full") / (_cnr_norm * _s_norm)
            _center = _traj_len2 - 1
            _xcorr2[_si] += _full[_center - _max_lag2 : _center + _max_lag2 + 1]
    _xcorr2 /= _n_traj2

    fig_xcorr_all, _ax = plt.subplots(figsize=(12, 4))
    for _si, _col in enumerate(_all_cols):
        _ls = "-" if _col in DEFAULT_STIM_COLS else "--"
        _ax.plot(_lags2, _xcorr2[_si], label=_col, alpha=0.8, linestyle=_ls)
    _ax.axvline(0, color="black", lw=1, linestyle="--")
    _ax.axhline(0, color="gray", lw=0.5)

    for _si, _col in enumerate(_all_cols):
        _pos = _xcorr2[_si, _lags2 > 0]
        _peak_lag = _lags2[_lags2 > 0][np.argmax(_pos)]
        _ax.axvline(_peak_lag, color=f"C{_si}", lw=1, linestyle=":", alpha=0.5)

    _ax.set_xlabel("lag (timesteps)  [positive = stim leads CNR]")
    _ax.set_ylabel("normalized cross-correlation")
    _ax.set_title(f"All stim features → CNR cross-correlation ({DATA_SOURCE})\n(solid = currently in model)")
    _ax.legend(fontsize=8)
    fig_xcorr_all.tight_layout()
    fig_xcorr_all
    return


@app.cell
def _(config, device, mo, n_stim, nn, torch):
    def _init_forget_bias(lstm):
        for name, param in lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    class LSTMEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=0.1 if num_layers > 1 else 0.0,
            )
            _init_forget_bias(self.lstm)

        def forward(self, x):
            _, (h_n, c_n) = self.lstm(x)
            return h_n, c_n

    class LSTMDecoder(nn.Module):
        def __init__(self, stim_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                stim_dim, hidden_dim, num_layers,
                batch_first=True, dropout=0.1 if num_layers > 1 else 0.0,
            )
            _init_forget_bias(self.lstm)
            self.fc_out = nn.Linear(hidden_dim, 1)

        def forward(self, future_stim, h_0, c_0):
            out, _ = self.lstm(future_stim, (h_0, c_0))
            return self.fc_out(out).squeeze(-1)

    class Seq2Seq(nn.Module):
        """Autoregressive rollout with teacher forcing."""
        def __init__(self, encoder_dim, stim_dim, hidden_dim, num_layers):
            super().__init__()
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers)
            self.decoder = LSTMDecoder(stim_dim, hidden_dim, num_layers)

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
        def __init__(self, encoder_dim, stim_dim, hidden_dim, num_layers):
            super().__init__()
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers)
            self.decoder = LSTMDecoder(stim_dim, hidden_dim, num_layers)

        def forward(self, encoder_input, future_stim, **kwargs):
            h, c = self.encoder(encoder_input)
            return self.decoder(future_stim, h, c)

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)

    encoder_dim = 1 + n_stim
    stim_dim = n_stim

    model = Seq2Seq(
        encoder_dim=encoder_dim,
        stim_dim=stim_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    model_baseline = Seq2SeqBaseline(
        encoder_dim=encoder_dim,
        stim_dim=stim_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_b = sum(p.numel() for p in model_baseline.parameters())
    mo.md(f"""
    | model | type | params |
    |-------|------|--------|
    | AR + teacher forcing | `Seq2Seq` | {n_params:,} |
    | Single-pass baseline | `Seq2SeqBaseline` | {n_params_b:,} |

    encoder_in={encoder_dim} | decoder_in={stim_dim} | hidden={config['hidden_dim']} | layers={config['num_layers']} | `{device}`
    """)
    return model, model_baseline


@app.cell
def _(
    DATA_SOURCE,
    EXPERIMENT_NAME,
    ExperimentTracker,
    config,
    device,
    load_experiment_group,
    mo,
    mode,
    model,
    model_baseline,
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

    def _run_epoch_baseline(model, loader, device, optimizer, is_train):
        model.train() if is_train else model.eval()
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in, dec_stim, dec_target = enc_in.to(device), dec_stim.to(device), dec_target.to(device)
                preds = model(enc_in, dec_stim)
                loss = model.loss(preds, dec_target)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                losses.append(loss.item())
        return np.mean(losses)

    def _checkpoint_with_best_weights(tracker, mdl, best_ckpt_path, hist):
        """Swap in best weights, checkpoint, swap back."""
        _cur = {k: v.clone() for k, v in mdl.state_dict().items()}
        mdl.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        tracker.checkpoint(mdl, training_results={"history": hist})
        mdl.load_state_dict(_cur)

    def train_both(model_ar, model_bl, train_loader, val_loader, cfg, device,
                   tracker_ar, tracker_bl):
        opt_ar = optim.Adam(model_ar.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        opt_bl = optim.Adam(model_bl.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        sched_ar = optim.lr_scheduler.ReduceLROnPlateau(opt_ar, patience=10, factor=0.5)
        sched_bl = optim.lr_scheduler.ReduceLROnPlateau(opt_bl, patience=10, factor=0.5)

        epochs, patience = cfg["epochs"], cfg["patience"]
        hist_ar = {"train_loss": [], "val_loss": []}
        hist_bl = {"train_loss": [], "val_loss": []}

        ckpt_fd_ar, ckpt_ar = tempfile.mkstemp(suffix=".pt")
        ckpt_fd_bl, ckpt_bl = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd_ar)
        os.close(ckpt_fd_bl)

        best_ar, best_bl, wait_ar, wait_bl = float("inf"), float("inf"), 0, 0
        done_ar, done_bl = False, False

        for epoch in range(epochs):
            if not done_ar:
                t_ar, tf = _run_epoch_ar(model_ar, train_loader, device, opt_ar, cfg, epoch, True)
                v_ar, _ = _run_epoch_ar(model_ar, val_loader, device, opt_ar, cfg, epoch, False)
                hist_ar["train_loss"].append(t_ar)
                hist_ar["val_loss"].append(v_ar)
                sched_ar.step(v_ar)
                if v_ar < best_ar:
                    best_ar, wait_ar = v_ar, 0
                    torch.save(model_ar.state_dict(), ckpt_ar)
                else:
                    wait_ar += 1
                    if wait_ar >= patience:
                        print(f"[AR]       Early stopping at epoch {epoch}")
                        done_ar = True

            if not done_bl:
                t_bl = _run_epoch_baseline(model_bl, train_loader, device, opt_bl, True)
                v_bl = _run_epoch_baseline(model_bl, val_loader, device, opt_bl, False)
                hist_bl["train_loss"].append(t_bl)
                hist_bl["val_loss"].append(v_bl)
                sched_bl.step(v_bl)
                if v_bl < best_bl:
                    best_bl, wait_bl = v_bl, 0
                    torch.save(model_bl.state_dict(), ckpt_bl)
                else:
                    wait_bl += 1
                    if wait_bl >= patience:
                        print(f"[Baseline] Early stopping at epoch {epoch}")
                        done_bl = True

            if done_ar and done_bl:
                break

            if epoch % 20 == 0:
                ar_str = f"AR tf={tf:.2f} T:{t_ar:.5f} V:{v_ar:.5f}" if not done_ar else "AR done"
                bl_str = f"BL T:{t_bl:.5f} V:{v_bl:.5f}" if not done_bl else "BL done"
                print(f"Epoch {epoch:3d} | {ar_str} | {bl_str}")

            _checkpoint_with_best_weights(tracker_ar, model_ar, ckpt_ar, hist_ar)
            _checkpoint_with_best_weights(tracker_bl, model_bl, ckpt_bl, hist_bl)

        model_ar.load_state_dict(torch.load(ckpt_ar, weights_only=True))
        model_bl.load_state_dict(torch.load(ckpt_bl, weights_only=True))
        os.remove(ckpt_ar)
        os.remove(ckpt_bl)
        return hist_ar, hist_bl

    if mode.is_train:
        mo.stop(not train_button.value, mo.md("Click **Start training** when ready."))

        tracker = ExperimentTracker(
            directory=f"{results_base}/{EXPERIMENT_NAME}",
            name=EXPERIMENT_NAME,
            model_config=_model_config_shared,
            training_config=config,
        )
        tracker.register_start()
        tracker_ar = tracker.make_subexperiment(
            "ar", model_config=dict(**_model_config_shared, variant="autoregressive_tf"),
        )
        tracker_bl = tracker.make_subexperiment(
            "baseline", model_config=dict(**_model_config_shared, variant="single_pass"),
        )
        tracker_ar.register_start()
        tracker_bl.register_start()

        _t0 = time.time()
        history, history_baseline = train_both(
            model, model_baseline, train_loader, val_loader, config, device,
            tracker_ar, tracker_bl,
        )
        train_elapsed = time.time() - _t0

        _models_md = mo.md(f"""
    **Training complete** in {train_elapsed:.0f}s

    | model | epochs |
    |-------|--------|
    | AR + teacher forcing | {len(history['train_loss'])} |
    | Single-pass baseline | {len(history_baseline['train_loss'])} |
    """)
        _config_md = mo.md(f"""
    **Config**

    | param | value |
    |-------|-------|
    | hidden_dim | {config['hidden_dim']} |
    | num_layers | {config['num_layers']} |
    | history / future | {config['history_len']} / {config['future_len']} |
    | lr | {config['lr']} |
    | batch_size | {config['batch_size']} |
    | patience | {config['patience']} |
    """)
        _metrics_md = mo.md(f"""
    **Metrics**

    | model | final train | final val |
    |-------|-------------|-----------|
    | AR | {history['train_loss'][-1]:.4f} | {history['val_loss'][-1]:.4f} |
    | Baseline | {history_baseline['train_loss'][-1]:.4f} | {history_baseline['val_loss'][-1]:.4f} |
    """)
        mo.output.replace(mo.hstack([_models_md, _config_md, _metrics_md], gap=2))
    else:
        from experiment import ExperimentBundle

        mo.stop(not mode.load_button_clicked, mo.md("Select experiments and click **Load experiments**."))

        _exp_name = mode.selected_experiment
        mo.stop(not _exp_name, mo.md("Select an experiment."))
        _exp_path = mode.selected_experiment_path

        if (_exp_path / "experiment.json").exists():
            _bundles = load_experiment_group(str(_exp_path))
            mo.stop("ar" not in _bundles or "baseline" not in _bundles,
                     mo.md(f"Expected 'ar' and 'baseline' variants, got: {list(_bundles.keys())}"))
            _bundle_ar = _bundles["ar"]
            _bundle_bl = _bundles["baseline"]
        else:
            _bundle_ar = ExperimentBundle.load(str(_exp_path))
            _bundle_bl = _bundle_ar  # single flat experiment fallback

        _cfg_ar = _bundle_ar.model_config
        _cfg_bl = _bundle_bl.model_config
        _expected_encoder_dim = 1 + n_stim
        _dim_mismatch = (
            _cfg_ar.get("encoder_dim") != _expected_encoder_dim
            or _cfg_bl.get("encoder_dim") != _expected_encoder_dim
        )
        mo.stop(_dim_mismatch, mo.callout(
            mo.md(
                f"**Dimension mismatch:** loaded model expects encoder_dim="
                f"{_cfg_ar.get('encoder_dim')}/{_cfg_bl.get('encoder_dim')} "
                f"but current data source has {_expected_encoder_dim}. "
                f"Change the **Data source** dropdown to match the loaded experiment "
                f"(`{_cfg_ar.get('data_source', '?')}`)."
            ),
            kind="warn",
        ))

        model.load_state_dict(_bundle_ar.model_state_dict)
        model.to(device)
        model.eval()

        model_baseline.load_state_dict(_bundle_bl.model_state_dict)
        model_baseline.to(device)
        model_baseline.eval()

        history = _bundle_ar.training_results.get("history", {"train_loss": [], "val_loss": []})
        history_baseline = _bundle_bl.training_results.get("history", {"train_loss": [], "val_loss": []})
        train_elapsed = _bundle_ar.training_results.get("train_elapsed_s", 0.0)

        tracker_ar = None
        tracker_bl = None

        _models_md = mo.md(f"""
    **Loaded from disk**

    | model | source | epochs |
    |-------|--------|--------|
    | AR | `{_exp_name}` | {len(history.get('train_loss', []))} |
    | Baseline | `{_exp_name}` | {len(history_baseline.get('train_loss', []))} |
    """)
        _cfg_display = _bundle_ar.training_config or {}
        _config_md = mo.md(f"""
    **Config**

    | param | value |
    |-------|-------|
    | hidden_dim | {_cfg_ar.get('hidden_dim', '?')} |
    | num_layers | {_cfg_ar.get('num_layers', '?')} |
    | history / future | {_cfg_ar.get('history_len', '?')} / {_cfg_ar.get('future_len', '?')} |
    | lr | {_cfg_display.get('lr', '?')} |
    | batch_size | {_cfg_display.get('batch_size', '?')} |
    | data_source | {_cfg_ar.get('data_source', '?')} |
    """)
        _ar_final_t = history['train_loss'][-1] if history.get('train_loss') else '?'
        _ar_final_v = history['val_loss'][-1] if history.get('val_loss') else '?'
        _bl_final_t = history_baseline['train_loss'][-1] if history_baseline.get('train_loss') else '?'
        _bl_final_v = history_baseline['val_loss'][-1] if history_baseline.get('val_loss') else '?'
        _metrics_md = mo.md(f"""
    **Metrics**

    | model | final train | final val |
    |-------|-------------|-----------|
    | AR | {_ar_final_t:.4f if isinstance(_ar_final_t, float) else _ar_final_t} | {_ar_final_v:.4f if isinstance(_ar_final_v, float) else _ar_final_v} |
    | Baseline | {_bl_final_t:.4f if isinstance(_bl_final_t, float) else _bl_final_t} | {_bl_final_v:.4f if isinstance(_bl_final_v, float) else _bl_final_v} |
    """)
        _warnings = []
        if _bundle_ar.warnings:
            _warnings.append(mo.callout(mo.md("**AR warnings:** " + ", ".join(_bundle_ar.warnings)), kind="warn"))
        if _bundle_bl.warnings:
            _warnings.append(mo.callout(mo.md("**BL warnings:** " + ", ".join(_bundle_bl.warnings)), kind="warn"))

        _output = mo.vstack([
            mo.hstack([_models_md, _config_md, _metrics_md], gap=2),
            *_warnings,
        ])
        mo.output.replace(_output)
    return history, history_baseline, tracker_ar, tracker_bl, train_elapsed


@app.cell
def _(history, history_baseline, plt):
    skip = 3
    fig_loss, _ax = plt.subplots(1, 3, figsize=(18, 4))

    _ax[0].plot(history["train_loss"], label="train", color="tab:red")
    _ax[0].plot(history["val_loss"], label="val", color="tab:red", linestyle="--")
    _ax[0].set_title("AR (autoregressive)")
    _ax[0].set_xlabel("epoch")
    _ax[0].set_ylabel("MSE")
    _ax[0].set_yscale("log")
    _ax[0].legend()

    _ax[1].plot(history_baseline["train_loss"], label="train", color="tab:blue")
    _ax[1].plot(history_baseline["val_loss"], label="val", color="tab:blue", linestyle="--")
    _ax[1].set_title("Baseline (single-pass)")
    _ax[1].set_xlabel("epoch")
    _ax[1].set_yscale("log")
    _ax[1].legend()

    _ax[2].plot(history["val_loss"][skip:], label="AR val", color="tab:red")
    _ax[2].plot(history_baseline["val_loss"][skip:], label="Baseline val", color="tab:blue")
    _ax[2].set_title(f"Val loss comparison (epoch {skip}+)")
    _ax[2].set_xlabel("epoch")
    _ax[2].set_yscale("log")
    _ax[2].legend()

    fig_loss.tight_layout()
    fig_loss
    return (fig_loss,)


@app.cell
def _(F_, H, device, model, model_baseline, np, plt, test_ds, torch):
    _n_examples = 8
    _indices = np.linspace(0, len(test_ds) - 1, _n_examples, dtype=int)

    fig_recon, _axes = plt.subplots(2, 4, figsize=(18, 8))
    _axes = _axes.flatten()

    model.eval()
    model_baseline.eval()
    with torch.no_grad():
        for _ax_i, _idx in enumerate(_indices):
            _enc_in, _dec_stim, _dec_target = test_ds[_idx]
            _enc_in_d = _enc_in.unsqueeze(0).to(device)
            _dec_stim_d = _dec_stim.unsqueeze(0).to(device)

            _pred_ar_d = model(_enc_in_d, _dec_stim_d).cpu().numpy()[0]
            _pred_bl_d = model_baseline(_enc_in_d, _dec_stim_d).cpu().numpy()[0]
            _hist_cnr = _enc_in[:, 0].numpy()
            _hist_u_t = _enc_in[:, 1].numpy()
            _fut_u_t = _dec_stim[:, 0].numpy()
            _last_val = _hist_cnr[-1]
            # reconstruct absolute CNR from deltas
            _actual = _last_val + np.cumsum(_dec_target.numpy())
            _pred_ar = _last_val + np.cumsum(_pred_ar_d)
            _pred_bl = _last_val + np.cumsum(_pred_bl_d)

            _ax = _axes[_ax_i]
            _t_hist = np.arange(H)
            _t_fut = np.arange(H, H + F_)

            _u_max = max(_hist_u_t.max(), _fut_u_t.max(), 1e-8)
            _cnr_max = max(_hist_cnr.max(), _actual.max(), _pred_ar.max(), _pred_bl.max(), 1e-8)
            _ls_h = _hist_u_t / _u_max * _cnr_max * 0.5
            _ls_f = _fut_u_t / _u_max * _cnr_max * 0.5

            _ax.fill_between(_t_hist, 0, _ls_h, alpha=0.15, color="gold", step="mid")
            _ax.fill_between(_t_fut, 0, _ls_f, alpha=0.15, color="gold", step="mid")
            _ax.plot(_t_hist, _hist_cnr, color="navy", lw=1.5, label="history")
            _ax.plot(_t_fut, _actual, color="navy", lw=1.5, alpha=0.5, label="actual")
            _ax.plot(_t_fut, _pred_ar, color="tab:red", lw=1.5, linestyle="--", label="AR")
            _ax.plot(_t_fut, _pred_bl, color="tab:blue", lw=1.5, linestyle=":", label="baseline")
            _ax.axvline(H, color="gray", linestyle=":", alpha=0.5)
            _mse_ar = np.mean((_actual - _pred_ar) ** 2)
            _mse_bl = np.mean((_actual - _pred_bl) ** 2)
            _ax.set_title(f"#{_idx} AR:{_mse_ar:.4f} BL:{_mse_bl:.4f}", fontsize=8)
            if _ax_i == 0:
                _ax.legend(fontsize=7)

    fig_recon.suptitle("AR (red dashed) vs Baseline (blue dotted)", fontsize=12)
    fig_recon.tight_layout()
    fig_recon
    return (fig_recon,)


@app.cell
def _(DataLoader, device, model, model_baseline, np, test_ds, torch):
    # --- collect full test-set predictions (shared across eval cells) ---
    _last_cnr, _actual_all, _pred_ar_all, _pred_bl_all = [], [], [], []
    _pred_ar_nz, _pred_bl_nz, _fut_stim_all = [], [], []

    model.eval()
    model_baseline.eval()
    with torch.no_grad():
        for _enc, _stim, _tgt in DataLoader(test_ds, batch_size=512):
            _enc_d, _stim_d = _enc.to(device), _stim.to(device)
            _zero_d = torch.zeros_like(_stim_d)

            _last_cnr.append(_enc_d[:, -1, 0].cpu().numpy())
            _actual_all.append(_tgt.numpy())
            _pred_ar_all.append(model(_enc_d, _stim_d).cpu().numpy())
            _pred_bl_all.append(model_baseline(_enc_d, _stim_d).cpu().numpy())
            _pred_ar_nz.append(model(_enc_d, _zero_d).cpu().numpy())
            _pred_bl_nz.append(model_baseline(_enc_d, _zero_d).cpu().numpy())
            _fut_stim_all.append(_stim_d[:, :, 0].mean(dim=1).cpu().numpy())

    test_last = np.concatenate(_last_cnr)
    test_act  = np.concatenate(_actual_all)   # (N, F) — deltas
    test_ar   = np.concatenate(_pred_ar_all)  # (N, F) — predicted deltas
    test_bl   = np.concatenate(_pred_bl_all)
    test_ar0  = np.concatenate(_pred_ar_nz)
    test_bl0  = np.concatenate(_pred_bl_nz)
    test_stim = np.concatenate(_fut_stim_all)

    # reconstruct absolute CNR for plots that need it
    test_act_abs = test_last[:, None] + np.cumsum(test_act, axis=1)
    test_ar_abs  = test_last[:, None] + np.cumsum(test_ar,  axis=1)
    test_bl_abs  = test_last[:, None] + np.cumsum(test_bl,  axis=1)

    test_stim_on = test_stim > test_stim.mean()
    return (
        test_act,
        test_act_abs,
        test_ar,
        test_ar0,
        test_ar_abs,
        test_bl,
        test_bl0,
        test_bl_abs,
        test_last,
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
    test_ar,
    test_ar0,
    test_ar_abs,
    test_bl,
    test_bl0,
    test_bl_abs,
    test_last,
    test_stim_on,
):
    fig_diag, _ax = plt.subplots(2, 3, figsize=(18, 10))

    _ax[0, 0].hist(test_act[:, 0], bins=60, alpha=0.6, color="navy", label=f"step1 μ={test_act[:,0].mean():.3f}")
    _ax[0, 0].hist(test_act[:, -1], bins=60, alpha=0.6, color="steelblue", label=f"step{test_act.shape[1]} μ={test_act[:,-1].mean():.3f}")
    _ax[0, 0].axvline(0, color="black", lw=1, linestyle="--")
    _ax[0, 0].set_xlabel("actual delta CNR")
    _ax[0, 0].set_title("Actual delta distribution (target)")
    _ax[0, 0].legend(fontsize=8)

    for _pred, _color, _lbl in [(test_ar[:, 0], "tab:red", "AR"), (test_bl[:, 0], "tab:blue", "BL")]:
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
        _ax[0, 2].hist(test_act[:, _i] - test_ar[:, _i], bins=60, alpha=0.4, color="tab:red", label=f"AR step{_i+1}")
        _ax[0, 2].hist(test_act[:, _i] - test_bl[:, _i], bins=60, alpha=0.4, color="tab:blue", label=f"BL step{_i+1}")
    _ax[0, 2].axvline(0, color="black", lw=1)
    _ax[0, 2].set_xlabel("actual delta − predicted delta")
    _ax[0, 2].set_title("Residual distribution (deltas)")
    _ax[0, 2].legend(fontsize=7)

    _sens_ar = np.abs(test_ar - test_ar0).mean(axis=1)
    _sens_bl = np.abs(test_bl - test_bl0).mean(axis=1)
    _ax[1, 0].hist(_sens_ar, bins=60, alpha=0.6, color="tab:red", label=f"AR (mean={_sens_ar.mean():.4f})")
    _ax[1, 0].hist(_sens_bl, bins=60, alpha=0.6, color="tab:blue", label=f"BL (mean={_sens_bl.mean():.4f})")
    _ax[1, 0].set_xlabel("|pred(stim) − pred(zero stim)|")
    _ax[1, 0].set_title("Stimulus sensitivity (ablation)")
    _ax[1, 0].legend(fontsize=8)

    _steps = np.arange(1, F_ + 1)
    for _mask, _ls, _lbl in [(test_stim_on, "-", "stim ON"), (~test_stim_on, "--", "stim OFF")]:
        _ax[1, 1].plot(_steps, test_act_abs[_mask].mean(axis=0), color="navy", ls=_ls, label=f"actual {_lbl}")
        _ax[1, 1].plot(_steps, test_ar_abs[_mask].mean(axis=0),  color="tab:red",  ls=_ls, label=f"AR {_lbl}")
        _ax[1, 1].plot(_steps, test_bl_abs[_mask].mean(axis=0),  color="tab:blue", ls=_ls, label=f"BL {_lbl}")
    _ax[1, 1].set_xlabel("future step")
    _ax[1, 1].set_ylabel("mean CNR (absolute)")
    _ax[1, 1].set_title("Mean prediction: stim ON vs OFF")
    _ax[1, 1].legend(fontsize=7)

    _ratio_ar  = test_ar_abs[:, 0]  / np.where(np.abs(test_last) > 1e-8, test_last, 1e-8)
    _ratio_bl  = test_bl_abs[:, 0]  / np.where(np.abs(test_last) > 1e-8, test_last, 1e-8)
    _ratio_act = test_act_abs[:, 0] / np.where(np.abs(test_last) > 1e-8, test_last, 1e-8)
    _bins = np.linspace(0, 2, 60)
    _ax[1, 2].hist(_ratio_act, bins=_bins, alpha=0.4, color="navy",     label=f"actual (μ={_ratio_act.mean():.2f})")
    _ax[1, 2].hist(_ratio_ar,  bins=_bins, alpha=0.5, color="tab:red",  label=f"AR pred (μ={_ratio_ar.mean():.2f})")
    _ax[1, 2].hist(_ratio_bl,  bins=_bins, alpha=0.5, color="tab:blue", label=f"BL pred (μ={_ratio_bl.mean():.2f})")
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
def _(np, test_act, test_act_abs, test_ar, test_ar_abs, test_bl, test_bl_abs):
    mse_per_step_ar = np.mean((test_ar - test_act) ** 2, axis=0)
    mse_per_step_bl = np.mean((test_bl - test_act) ** 2, axis=0)
    mse_per_step_zero = np.mean(test_act ** 2, axis=0)
    _act_mean = test_act.mean(axis=0)
    mse_per_step_mean = np.mean((test_act - _act_mean) ** 2, axis=0)

    _ss_tot = np.sum((test_act - _act_mean) ** 2, axis=0)
    _ss_tot = np.where(_ss_tot > 0, _ss_tot, 1e-8)
    r2_per_step_ar = 1 - np.sum((test_ar - test_act) ** 2, axis=0) / _ss_tot
    r2_per_step_bl = 1 - np.sum((test_bl - test_act) ** 2, axis=0) / _ss_tot

    mae_cum_ar = np.mean(np.abs(test_ar_abs - test_act_abs), axis=0)
    mae_cum_bl = np.mean(np.abs(test_bl_abs - test_act_abs), axis=0)
    mae_cum_ar_std = np.std(np.abs(test_ar_abs - test_act_abs), axis=0)
    mae_cum_bl_std = np.std(np.abs(test_bl_abs - test_act_abs), axis=0)

    mse_window_ar = np.mean((test_ar - test_act) ** 2, axis=1)
    mse_window_bl = np.mean((test_bl - test_act) ** 2, axis=1)

    overall_mse_ar = float(np.mean(mse_window_ar))
    overall_mse_bl = float(np.mean(mse_window_bl))
    overall_mse_zero = float(np.mean(test_act ** 2))
    overall_mse_mean = float(mse_per_step_mean.mean())
    _ss_tot_all = np.sum((test_act - test_act.mean()) ** 2)
    overall_r2_ar = float(1 - np.sum((test_ar - test_act) ** 2) / max(_ss_tot_all, 1e-8))
    overall_r2_bl = float(1 - np.sum((test_bl - test_act) ** 2) / max(_ss_tot_all, 1e-8))
    ar_win_rate = float(np.mean(mse_window_ar < mse_window_bl))

    eval_metrics = {
        "test_mse_ar": overall_mse_ar,
        "test_mse_bl": overall_mse_bl,
        "test_mse_persist_last": overall_mse_zero,
        "test_mse_predict_mean": overall_mse_mean,
        "test_r2_ar": overall_r2_ar,
        "test_r2_bl": overall_r2_bl,
        "ar_win_rate": ar_win_rate,
        "mse_per_step_ar": mse_per_step_ar.tolist(),
        "mse_per_step_bl": mse_per_step_bl.tolist(),
        "r2_per_step_ar": r2_per_step_ar.tolist(),
        "r2_per_step_bl": r2_per_step_bl.tolist(),
    }
    return (
        ar_win_rate,
        eval_metrics,
        mae_cum_ar,
        mae_cum_ar_std,
        mae_cum_bl,
        mae_cum_bl_std,
        mse_per_step_ar,
        mse_per_step_bl,
        mse_per_step_mean,
        mse_per_step_zero,
        mse_window_ar,
        mse_window_bl,
        overall_mse_ar,
        overall_mse_bl,
        overall_mse_mean,
        overall_mse_zero,
        overall_r2_ar,
        overall_r2_bl,
        r2_per_step_ar,
        r2_per_step_bl,
    )


@app.cell
def _(
    F_,
    mo,
    mse_per_step_ar,
    mse_per_step_bl,
    mse_per_step_mean,
    mse_per_step_zero,
    np,
    plt,
    r2_per_step_ar,
    r2_per_step_bl,
):
    _steps = np.arange(1, F_ + 1)
    _w = 0.2

    fig_step_metrics, (_ax_mse, _ax_r2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax_mse.bar(_steps - 1.5 * _w, mse_per_step_zero, _w, label="Persist last", color="#aaa")
    _ax_mse.bar(_steps - 0.5 * _w, mse_per_step_mean, _w, label="Predict mean", color="#ccc")
    _ax_mse.bar(_steps + 0.5 * _w, mse_per_step_bl,   _w, label="Baseline", color="#dd8452")
    _ax_mse.bar(_steps + 1.5 * _w, mse_per_step_ar,   _w, label="AR", color="#4c72b0")
    _ax_mse.set_xlabel("Forecast step")
    _ax_mse.set_ylabel("MSE")
    _ax_mse.set_title("Per-step MSE vs naive baselines")
    _ax_mse.set_xticks(_steps)
    _ax_mse.legend(fontsize=8)

    _ax_r2.plot(_steps, r2_per_step_ar, "o-", color="#4c72b0", label="AR")
    _ax_r2.plot(_steps, r2_per_step_bl, "s-", color="#dd8452", label="Baseline")
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
    overall_mse_ar,
    overall_mse_bl,
    overall_mse_mean,
    overall_mse_zero,
    plt,
):
    _names = ["Persist last\n(zero delta)", "Predict\nmean delta", "Baseline", "AR"]
    _vals = [overall_mse_zero, overall_mse_mean, overall_mse_bl, overall_mse_ar]
    _colors = ["#aaa", "#ccc", "#dd8452", "#4c72b0"]

    fig_baselines, _ax = plt.subplots(figsize=(8, 5))
    _bars = _ax.bar(_names, _vals, color=_colors, edgecolor="black", linewidth=0.5)
    for _bar, _v in zip(_bars, _vals):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _v, f"{_v:.6f}",
                 ha="center", va="bottom", fontsize=9)
    _ax.set_ylabel("Overall MSE")
    _ax.set_title("Model vs naive baselines")
    fig_baselines.tight_layout()

    _pct_ar = (1 - overall_mse_ar / overall_mse_zero) * 100
    _pct_bl = (1 - overall_mse_bl / overall_mse_zero) * 100

    mo.vstack([
        fig_baselines,
        mo.md(f"""
    **AR** achieves **{_pct_ar:.1f}%** lower MSE than persist-last baseline.
    **Baseline** achieves **{_pct_bl:.1f}%** lower MSE than persist-last baseline.
    """),
    ])
    return (fig_baselines,)


@app.cell
def _(F_, mae_cum_ar, mae_cum_ar_std, mae_cum_bl, mae_cum_bl_std, mo, np, plt):
    _steps = np.arange(1, F_ + 1)

    fig_cumulative, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_steps, mae_cum_ar, "o-", color="#4c72b0", label="AR")
    _ax.fill_between(_steps, mae_cum_ar - mae_cum_ar_std, mae_cum_ar + mae_cum_ar_std,
                     alpha=0.15, color="#4c72b0")
    _ax.plot(_steps, mae_cum_bl, "s-", color="#dd8452", label="Baseline")
    _ax.fill_between(_steps, mae_cum_bl - mae_cum_bl_std, mae_cum_bl + mae_cum_bl_std,
                     alpha=0.15, color="#dd8452")
    _ax.set_xlabel("Forecast step")
    _ax.set_ylabel("MAE (absolute CNR)")
    _ax.set_title("Cumulative trajectory error (absolute CNR)")
    _ax.legend()
    fig_cumulative.tight_layout()

    mo.md("## Cumulative error in reconstructed CNR")
    return (fig_cumulative,)


@app.cell
def _(ar_win_rate, mo, mse_window_ar, mse_window_bl, plt, test_stim):
    fig_head2head, _ax = plt.subplots(figsize=(7, 7))
    _sc = _ax.scatter(
        mse_window_bl, mse_window_ar,
        s=8, alpha=0.3, c=test_stim, cmap="viridis", edgecolors="none",
    )
    _lim = [0, max(mse_window_ar.max(), mse_window_bl.max()) * 1.05]
    _ax.plot(_lim, _lim, "k--", lw=1, label="y = x")
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel("Baseline per-window MSE")
    _ax.set_ylabel("AR per-window MSE")
    _ax.set_title("AR vs Baseline head-to-head")
    _ax.text(0.05, 0.95, f"AR wins {ar_win_rate * 100:.1f}% of windows",
             transform=_ax.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    _ax.legend(loc="lower right")
    fig_head2head.colorbar(_sc, ax=_ax, label="Mean stimulus intensity")
    fig_head2head.tight_layout()

    mo.md("## AR vs Baseline head-to-head")
    return (fig_head2head,)


@app.cell
def _(
    ar_win_rate,
    mo,
    overall_mse_ar,
    overall_mse_bl,
    overall_mse_mean,
    overall_mse_zero,
    overall_r2_ar,
    overall_r2_bl,
):
    mo.md(f"""
    ## Evaluation summary

    | Metric | AR | Baseline | Persist-last | Predict-mean |
    |--------|---:|--------:|-----------:|-----------:|
    | **Overall MSE** | {overall_mse_ar:.6f} | {overall_mse_bl:.6f} | {overall_mse_zero:.6f} | {overall_mse_mean:.6f} |
    | **Overall R²** | {overall_r2_ar:.4f} | {overall_r2_bl:.4f} | — | — |
    | **AR win rate** | {ar_win_rate * 100:.1f}% | — | — | — |
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
    model_baseline,
    np,
    plt,
    test_ds,
    torch,
    traj_selector,
):
    _idx = traj_selector.value
    _enc_in, _dec_stim, _dec_target = test_ds[_idx]

    model.eval()
    model_baseline.eval()
    with torch.no_grad():
        _enc_in_d = _enc_in.unsqueeze(0).to(device)
        _dec_stim_d = _dec_stim.unsqueeze(0).to(device)
        _pred_ar_d = model(_enc_in_d, _dec_stim_d).cpu().numpy()[0]
        _pred_bl_d = model_baseline(_enc_in_d, _dec_stim_d).cpu().numpy()[0]

    _hist_cnr = _enc_in[:, 0].numpy()
    _hist_u_t = _enc_in[:, 1].numpy()
    _fut_u_t = _dec_stim[:, 0].numpy()
    _last_val = _hist_cnr[-1]
    # reconstruct absolute CNR from deltas
    _actual = _last_val + np.cumsum(_dec_target.numpy())
    _pred_ar = _last_val + np.cumsum(_pred_ar_d)
    _pred_bl = _last_val + np.cumsum(_pred_bl_d)
    _t_hist = np.arange(H)
    _t_fut = np.arange(H, H + F_)
    _umax = max(_hist_u_t.max(), _fut_u_t.max(), 1e-8)
    _cmax = max(_hist_cnr.max(), _actual.max(), _pred_ar.max(), _pred_bl.max(), 1e-8)
    _ls_h = _hist_u_t / _umax * _cmax * 0.5
    _ls_f = _fut_u_t / _umax * _cmax * 0.5

    _fig, _axes2 = plt.subplots(2, 2, figsize=(16, 8), height_ratios=[3, 1], sharex=True)

    for _col, (_pred, _color, _label) in enumerate([
        (_pred_ar, "tab:red", "AR"),
        (_pred_bl, "tab:blue", "Baseline"),
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

    _mse_ar = np.mean((_actual - _pred_ar) ** 2)
    _mse_bl = np.mean((_actual - _pred_bl) ** 2)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"**Window {_idx}** | AR MSE: {_mse_ar:.6f} | Baseline MSE: {_mse_bl:.6f}")
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
    history_baseline,
    hostname,
    is_cluster,
    mo,
    mode,
    model,
    model_baseline,
    tracker_ar,
    tracker_bl,
    train_elapsed,
    train_loader,
    val_loader,
):
    mo.stop(not mode.is_train, mo.md("_Loaded from disk — skipping save._"))

    _stats_ar = compute_training_stats(
        train_elapsed_s=train_elapsed,
        history=history,
        n_train_samples=len(train_loader.dataset),
        n_val_samples=len(val_loader.dataset),
        model=model,
    )
    _stats_bl = compute_training_stats(
        train_elapsed_s=train_elapsed,
        history=history_baseline,
        n_train_samples=len(train_loader.dataset),
        n_val_samples=len(val_loader.dataset),
        model=model_baseline,
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

    _bundle_ar = tracker_ar.save_final(
        model=model,
        training_results={"history": history, "train_elapsed_s": train_elapsed, "stats": _stats_ar},
        metrics=eval_metrics,
        figures=_figures,
    )

    _bundle_bl = tracker_bl.save_final(
        model=model_baseline,
        training_results={"history": history_baseline, "train_elapsed_s": train_elapsed, "stats": _stats_bl},
        metrics=eval_metrics,
        figures=_figures,
    )

    _env_label = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"
    _parent_dir = str(Path(_bundle_ar.save_dir).parent)
    mo.md(f"**Saved** on {_env_label}\n\n`{_parent_dir}`")
    return


if __name__ == "__main__":
    app.run()
