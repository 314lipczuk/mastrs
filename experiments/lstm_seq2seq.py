import marimo

__generated_with = "0.21.1"
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
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import time
    import tempfile
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset, Subset

    from experiment import save_experiment
    from utils import get_device

    device = get_device()
    return (
        DataLoader,
        Dataset,
        Subset,
        datetime,
        device,
        mo,
        nn,
        np,
        optim,
        os,
        pd,
        plt,
        save_experiment,
        tempfile,
        time,
        torch,
        train_test_split,
    )


@app.cell
def _(mo):
    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "lstm_seq2seq")
    DRY_RUN = args.get("dry_run", "true").lower() == "true"
    DRY_RUN = False

    config = dict(
        hidden_dim=int(args.get("hidden_dim", "32" if DRY_RUN else "64")),
        num_layers=int(args.get("num_layers", "1" if DRY_RUN else "2")),
        history_len=int(args.get("history_len", "40")),
        future_len=int(args.get("future_len", "30")),
        lr=float(args.get("lr", "1e-3")),
        epochs=int(args.get("epochs", "20" if DRY_RUN else "200")),
        batch_size=int(args.get("batch_size", "256")),
        patience=int(args.get("patience", "10" if DRY_RUN else "50")),
        teacher_forcing=float(args.get("teacher_forcing", "0.5")),
    )

    mo.md(f"""
    # LSTM Encoder-Decoder: `{EXPERIMENT_NAME}`

    Sequence-to-sequence model for CNR prediction.
    Encoder compresses CNR + light history → latent vector.
    Decoder takes latent + future light → predicted CNR step-by-step.

    | param | value |
    |-------|-------|
    | hidden_dim | {config['hidden_dim']} |
    | num_layers | {config['num_layers']} |
    | history_len | {config['history_len']} |
    | future_len | {config['future_len']} |
    | lr | {config['lr']} |
    | epochs | {config['epochs']} |
    | batch_size | {config['batch_size']} |
    | patience | {config['patience']} |
    | teacher_forcing | {config['teacher_forcing']} |
    | dry_run | {DRY_RUN} |
    """)
    return DRY_RUN, EXPERIMENT_NAME, config


@app.cell
def _(
    DRY_RUN,
    DataLoader,
    Dataset,
    Subset,
    config,
    mo,
    np,
    os,
    pd,
    torch,
    train_test_split,
):
    PARQUET_PATH = "stochastic_sim_output.parquet"
    mo.stop(
        not os.path.exists(PARQUET_PATH),
        mo.md(f"**`{PARQUET_PATH}` not found.** Run the stochastic simulator first."),
    )

    df = pd.read_parquet(PARQUET_PATH)
    traj_len = len(df["cnr"].iloc[0])

    cnr_all = np.stack(df["cnr"].values).astype(np.float32)     # (N, 100)
    light_all = np.stack(df["light"].values).astype(np.float32)  # (N, 100)

    H = config["history_len"]
    F_ = config["future_len"]

    class Seq2SeqDataset(Dataset):
        def __init__(self, cnr, light, history_len, future_len, stride=5):
            self.samples = []
            total = history_len + future_len
            for i in range(len(cnr)):
                t = 0
                while t + total <= cnr.shape[1]:
                    enc_cnr = cnr[i, t : t + history_len]
                    enc_light = light[i, t : t + history_len]
                    dec_light = light[i, t + history_len : t + total]
                    dec_target = cnr[i, t + history_len : t + total]
                    self.samples.append((
                        np.stack([enc_cnr, enc_light], axis=-1),  # (H, 2)
                        dec_light[:, np.newaxis],                  # (F, 1)
                        dec_target,                                # (F,)
                    ))
                    t += stride

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            enc_in, dec_light, dec_target = self.samples[idx]
            return (
                torch.tensor(enc_in),
                torch.tensor(dec_light),
                torch.tensor(dec_target),
            )

    traj_ids = np.arange(len(df))
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    stride = 5 if not DRY_RUN else 15
    train_ds = Seq2SeqDataset(cnr_all[tr_ids], light_all[tr_ids], H, F_, stride=stride)
    val_ds = Seq2SeqDataset(cnr_all[va_ids], light_all[va_ids], H, F_, stride=stride)
    test_ds = Seq2SeqDataset(cnr_all[te_ids], light_all[te_ids], H, F_, stride=stride)

    if DRY_RUN:
        n_dry = 2000
        train_ds = Subset(train_ds, range(min(n_dry, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_dry, len(val_ds))))
        test_ds = Subset(test_ds, range(min(n_dry, len(test_ds))))

    BS = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

    metadata = pd.DataFrame({
        "condition": df["generator"].values[te_ids],
    })

    mo.md(f"""
    **Data:** {len(df)} trajectories × {traj_len} timepoints

    Encoder input: CNR + light history ({H} steps) → Decoder predicts CNR ({F_} steps) given future light

    Train: {len(train_ds)} windows | Val: {len(val_ds)} | Test: {len(test_ds)}
    """)
    return F_, H, test_ds, train_loader, val_loader


@app.cell
def _(config, device, mo, nn, torch):
    class LSTMEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=0.1 if num_layers > 1 else 0.0,
            )

        def forward(self, x):
            # x: (batch, history_len, input_dim)
            _, (h_n, c_n) = self.lstm(x)
            # h_n: (num_layers, batch, hidden_dim)
            return h_n, c_n

    class LSTMDecoder(nn.Module):
        def __init__(self, hidden_dim, num_layers):
            super().__init__()
            # Input: previous CNR prediction (1) + current light (1) = 2
            self.lstm = nn.LSTM(
                2, hidden_dim, num_layers,
                batch_first=True, dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.fc_out = nn.Linear(hidden_dim, 1)

        def forward(self, future_light, h_0, c_0, target=None, teacher_forcing_ratio=0.0):
            # future_light: (batch, future_len, 1)
            # target: (batch, future_len) — actual CNR for teacher forcing
            batch_size, future_len, _ = future_light.shape
            outputs = []
            # First input: zero (no previous prediction yet)
            prev_erk = torch.zeros(batch_size, 1, 1, device=future_light.device)
            h, c = h_0, c_0

            for t in range(future_len):
                light_t = future_light[:, t : t + 1, :]  # (batch, 1, 1)
                dec_input = torch.cat([prev_erk, light_t], dim=-1)  # (batch, 1, 2)
                out, (h, c) = self.lstm(dec_input, (h, c))
                pred = self.fc_out(out)  # (batch, 1, 1)
                outputs.append(pred.squeeze(-1))  # (batch, 1)

                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    prev_erk = target[:, t].unsqueeze(1).unsqueeze(2)
                else:
                    prev_erk = pred.detach()

            return torch.cat(outputs, dim=1)  # (batch, future_len)

    class Seq2Seq(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
            self.decoder = LSTMDecoder(hidden_dim, num_layers)

        def forward(self, encoder_input, future_light, target=None, teacher_forcing_ratio=0.0):
            h, c = self.encoder(encoder_input)
            predictions = self.decoder(future_light, h, c, target, teacher_forcing_ratio)
            return predictions

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)

    model = Seq2Seq(
        input_dim=2,  # CNR + light
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    mo.md(f"**Model:** LSTM Seq2Seq | hidden_dim={config['hidden_dim']} | num_layers={config['num_layers']} | {n_params:,} params | `{device}`")
    return (model,)


@app.cell
def _(
    config,
    datetime,
    device,
    mo,
    model,
    nn,
    np,
    optim,
    os,
    tempfile,
    time,
    torch,
    train_loader,
    val_loader,
):
    def train_model(model, train_loader, val_loader, cfg, device):
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        epochs = cfg["epochs"]
        patience = cfg["patience"]
        tf_start = cfg["teacher_forcing"]

        best_val = float("inf")
        wait = 0
        history = {"train_loss": [], "val_loss": []}

        ckpt_fd, ckpt_path = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd)

        for epoch in range(epochs):
            tf_ratio = tf_start * max(0, 1 - epoch / (epochs * 0.7))

            model.train()
            t_losses = []
            for enc_in, dec_light, dec_target in train_loader:
                enc_in = enc_in.to(device)
                dec_light = dec_light.to(device)
                dec_target = dec_target.to(device)

                preds = model(enc_in, dec_light, target=dec_target, teacher_forcing_ratio=tf_ratio)
                loss = model.loss(preds, dec_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                t_losses.append(loss.item())

            model.eval()
            v_losses = []
            with torch.no_grad():
                for enc_in, dec_light, dec_target in val_loader:
                    enc_in = enc_in.to(device)
                    dec_light = dec_light.to(device)
                    dec_target = dec_target.to(device)

                    preds = model(enc_in, dec_light)
                    loss = model.loss(preds, dec_target)
                    v_losses.append(loss.item())

            t_mean = np.mean(t_losses)
            v_mean = np.mean(v_losses)
            history["train_loss"].append(t_mean)
            history["val_loss"].append(v_mean)

            scheduler.step(v_mean)

            if v_mean < best_val:
                best_val = v_mean
                wait = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train: {t_mean:.6f} | Val: {v_mean:.6f} | TF: {tf_ratio:.3f}")

        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        os.remove(ckpt_path)
        return history

    train_start = datetime.now()
    t0 = time.time()
    history = train_model(model, train_loader, val_loader, config, device)
    train_elapsed = time.time() - t0

    mo.md(f"**Training complete** in {train_elapsed:.0f}s — {len(history['train_loss'])} epochs")
    return history, train_elapsed


@app.cell
def _(history, plt):
    skip = 3

    fig_loss, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(history["train_loss"], label="train")
    _ax[0].plot(history["val_loss"], label="val")
    _ax[0].set_title("Loss (all epochs)")
    _ax[0].set_xlabel("epoch")
    _ax[0].set_ylabel("MSE")
    _ax[0].legend()

    _ax[1].plot(history["train_loss"][skip:], label="train")
    _ax[1].plot(history["val_loss"][skip:], label="val")
    _ax[1].set_title(f"Loss (epoch {skip}+)")
    _ax[1].set_xlabel("epoch")
    _ax[1].legend()

    fig_loss.tight_layout()
    fig_loss
    return (fig_loss,)


@app.cell
def _(F_, H, device, model, np, plt, test_ds, torch):
    _n_examples = 8
    _indices = np.linspace(0, len(test_ds) - 1, _n_examples, dtype=int)

    fig_recon, _axes = plt.subplots(2, 4, figsize=(18, 8))
    _axes = _axes.flatten()

    model.eval()
    with torch.no_grad():
        for _ax_i, _idx in enumerate(_indices):
            _enc_in, _dec_light, _dec_target = test_ds[_idx]
            _enc_in_d = _enc_in.unsqueeze(0).to(device)
            _dec_light_d = _dec_light.unsqueeze(0).to(device)

            _pred = model(_enc_in_d, _dec_light_d).cpu().numpy()[0]
            _actual = _dec_target.numpy()
            _hist_cnr = _enc_in[:, 0].numpy()
            _hist_light = _enc_in[:, 1].numpy()
            _fut_light = _dec_light[:, 0].numpy()

            _ax = _axes[_ax_i]
            _t_hist = np.arange(H)
            _t_fut = np.arange(H, H + F_)

            _light_max = max(_hist_light.max(), _fut_light.max(), 1e-8)
            _cnr_max = max(_hist_cnr.max(), _actual.max(), _pred.max(), 1e-8)
            _light_scaled_h = _hist_light / _light_max * _cnr_max * 0.5
            _light_scaled_f = _fut_light / _light_max * _cnr_max * 0.5

            _ax.fill_between(_t_hist, 0, _light_scaled_h, alpha=0.15, color="gold", step="mid")
            _ax.fill_between(_t_fut, 0, _light_scaled_f, alpha=0.15, color="gold", step="mid")
            _ax.plot(_t_hist, _hist_cnr, color="navy", lw=1.5, label="history CNR")
            _ax.plot(_t_fut, _actual, color="navy", lw=1.5, alpha=0.5, label="actual future")
            _ax.plot(_t_fut, _pred, color="tab:red", lw=1.5, linestyle="--", label="predicted")
            _ax.axvline(H, color="gray", linestyle=":", alpha=0.5)
            _ax.set_title(f"sample {_idx}", fontsize=9)
            if _ax_i == 0:
                _ax.legend(fontsize=7)

    fig_recon.suptitle("Encoder history → Decoder predictions", fontsize=12)
    fig_recon.tight_layout()
    fig_recon
    return (fig_recon,)


@app.cell
def _(mo, test_ds):
    traj_selector = mo.ui.slider(
        0, len(test_ds) - 1, value=0, label="Test window index"
    )
    traj_selector
    return (traj_selector,)


@app.cell
def _(F_, H, device, mo, model, np, plt, test_ds, torch, traj_selector):
    _idx = traj_selector.value
    _enc_in, _dec_light, _dec_target = test_ds[_idx]

    model.eval()
    with torch.no_grad():
        _enc_in_d = _enc_in.unsqueeze(0).to(device)
        _dec_light_d = _dec_light.unsqueeze(0).to(device)
        _pred = model(_enc_in_d, _dec_light_d).cpu().numpy()[0]

    _actual = _dec_target.numpy()
    _hist_cnr = _enc_in[:, 0].numpy()
    _hist_light = _enc_in[:, 1].numpy()
    _fut_light = _dec_light[:, 0].numpy()

    _t_hist = np.arange(H)
    _t_fut = np.arange(H, H + F_)

    _fig, (_ax_main, _ax_err) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1], sharex=True)

    _lmax = max(_hist_light.max(), _fut_light.max(), 1e-8)
    _cmax = max(_hist_cnr.max(), _actual.max(), _pred.max(), 1e-8)
    _ls_h = _hist_light / _lmax * _cmax * 0.5
    _ls_f = _fut_light / _lmax * _cmax * 0.5

    _ax_main.fill_between(_t_hist, 0, _ls_h, alpha=0.15, color="gold", step="mid", label="light")
    _ax_main.fill_between(_t_fut, 0, _ls_f, alpha=0.15, color="gold", step="mid")
    _ax_main.plot(_t_hist, _hist_cnr, color="navy", lw=2, label="encoder input (CNR)")
    _ax_main.plot(_t_fut, _actual, color="navy", lw=2, alpha=0.5, label="actual future CNR")
    _ax_main.plot(_t_fut, _pred, color="tab:red", lw=2, linestyle="--", label="predicted CNR")
    _ax_main.axvline(H, color="gray", linestyle=":", lw=1.5, label="encode|decode boundary")
    _ax_main.legend(fontsize=9)
    _ax_main.set_ylabel("CNR")
    _ax_main.set_title(f"Window {_idx}")

    _err = _actual - _pred
    _ax_err.bar(_t_fut, _err, color="tab:red", alpha=0.6, width=0.8)
    _ax_err.axhline(0, color="black", lw=0.5)
    _ax_err.set_ylabel("error (actual - pred)")
    _ax_err.set_xlabel("timestep")

    _mse = np.mean(_err ** 2)
    mo.vstack([
        _fig,
        mo.md(f"**Window {_idx}** | MSE: {_mse:.6f} | Max error: {np.max(np.abs(_err)):.4f}")
    ])
    return


@app.cell
def _(
    EXPERIMENT_NAME,
    config,
    fig_loss,
    fig_recon,
    history,
    mo,
    model,
    save_experiment,
    train_elapsed,
):
    output_dir = f"results/{EXPERIMENT_NAME}"

    save_experiment(
        directory=output_dir,
        model=model,
        model_config=dict(
            input_dim=2,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            history_len=config["history_len"],
            future_len=config["future_len"],
        ),
        training_config=config,
        training_results={
            "history": history,
            "train_elapsed_s": train_elapsed,
        },
        metrics={},
        figures={"loss_curves": fig_loss, "reconstructions": fig_recon},
        name=EXPERIMENT_NAME,
    )

    mo.md(f"**Saved** to `{output_dir}/`")
    return


if __name__ == "__main__":
    app.run()
