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

    from model.dl.cvae import ConditionalBetaVAE
    from eval_cvae import evaluate
    from experiment import save_experiment
    from utils import get_device

    device = get_device()
    return (
        ConditionalBetaVAE,
        DataLoader,
        Dataset,
        Subset,
        datetime,
        device,
        evaluate,
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

    EXPERIMENT_NAME = args.get("name", "default_name")
    DRY_RUN = args.get("dry_run", "true").lower() == "true"

    config = dict(
        hidden_dim=int(args.get("hidden_dim", "32")),
        latent_dim=int(args.get("latent_dim", "2")),
        beta=float(args.get("beta", "0.1")),
        alpha=float(args.get("alpha", "0.3")),
        lr=float(args.get("lr", "1e-3")),
        epochs=int(args.get("epochs", "200")),
        batch_size=int(args.get("batch_size", "256")),
        patience=int(args.get("patience", "100")),
    )

    mo.md(f"""
    # Experiment: `{EXPERIMENT_NAME}`

    Conditional beta-VAE on synthetic EGFR trajectories.
    Encodes ERK + light stimulus, reconstructs ERK conditioned on light.

    | param | value |
    |-------|-------|
    | hidden_dim | {config['hidden_dim']} |
    | latent_dim | {config['latent_dim']} |
    | beta | {config['beta']} |
    | alpha | {config['alpha']} |
    | lr | {config['lr']} |
    | epochs | {config['epochs']} |
    | batch_size | {config['batch_size']} |
    | patience | {config['patience']} |
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
    pd,
    torch,
    train_test_split,
):
    STATE_COLS = ["RAS_s", "RAF_s", "MEK_s", "NFB_s", "ERK_s"]

    df = pd.read_parquet("synthetic_EGFR_data.parquet")

    traj_len = len(df["ERK_s"].iloc[0])

    erk_all = np.stack(df["ERK_s"].values).astype(np.float32)
    light_all = np.stack(df["light"].values).astype(np.float32)

    class EGFRDataset(Dataset):
        def __init__(self, erk, light):
            erk_ch = torch.tensor(erk[:, np.newaxis, :])
            light_ch = torch.tensor(light[:, np.newaxis, :])
            self.encoder_input = torch.cat([erk_ch, light_ch], dim=1)
            self.stim_cond = light_ch.clone()
            self.target = erk_ch.clone()

        def __len__(self):
            return len(self.encoder_input)

        def __getitem__(self, idx):
            return self.encoder_input[idx], self.stim_cond[idx], self.target[idx]

    traj_ids = np.arange(len(df))
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    train_ds = EGFRDataset(erk_all[tr_ids], light_all[tr_ids])
    val_ds = EGFRDataset(erk_all[va_ids], light_all[va_ids])
    test_ds = EGFRDataset(erk_all[te_ids], light_all[te_ids])

    if DRY_RUN:
        n_dry = 5000
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
    **Data:** {len(df)} trajectories x {traj_len} timepoints

    Encoder input: ERK + light (2 channels) | Stim cond: light (1 channel) | Target: ERK (1 channel)

    Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}
    """)
    return (
        STATE_COLS,
        df,
        metadata,
        te_ids,
        test_ds,
        train_loader,
        traj_len,
        val_loader,
    )


@app.cell
def _(ConditionalBetaVAE, config, device, mo, traj_len):
    model = ConditionalBetaVAE(
        in_channels=2,
        stim_channels=1,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        seq_length=traj_len,
        beta=config["beta"],
        alpha=config["alpha"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    mo.md(f"**Model:** CVAE | hidden_dim={config['hidden_dim']} | latent_dim={config['latent_dim']} | beta={config['beta']} | alpha={config['alpha']} | {n_params:,} params | `{device}`")
    return (model,)


@app.cell
def _(
    DRY_RUN,
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
    def train_model(model, train_loader, val_loader, cfg, device, dry_run):
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        epochs = cfg["epochs"]
        patience = cfg["patience"]

        best_val = float("inf")
        wait = 0
        history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kl": [],
                    "val_recon": [], "val_kl": []}

        ckpt_fd, ckpt_path = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd)

        for epoch in range(epochs):
            model.train()
            t_total, t_recon, t_kl = [], [], []
            for encoder_input, stim_cond, target in train_loader:
                encoder_input = encoder_input.to(device)
                stim_cond = stim_cond.to(device)
                target = target.to(device)
                recon, mu, logvar = model(encoder_input, stim_cond)
                total, recon_l, kl_l = model.loss(recon, target, mu, logvar)
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                t_total.append(total.item())
                t_recon.append(recon_l.item())
                t_kl.append(kl_l.item())

            model.eval()
            v_total, v_recon, v_kl = [], [], []
            with torch.no_grad():
                for encoder_input, stim_cond, target in val_loader:
                    encoder_input = encoder_input.to(device)
                    stim_cond = stim_cond.to(device)
                    target = target.to(device)
                    recon, mu, logvar = model(encoder_input, stim_cond)
                    total, recon_l, kl_l = model.loss(recon, target, mu, logvar)
                    v_total.append(total.item())
                    v_recon.append(recon_l.item())
                    v_kl.append(kl_l.item())

            t_mean = np.mean(t_total)
            v_mean = np.mean(v_total)
            history["train_loss"].append(t_mean)
            history["val_loss"].append(v_mean)
            history["train_recon"].append(np.mean(t_recon))
            history["train_kl"].append(np.mean(t_kl))
            history["val_recon"].append(np.mean(v_recon))
            history["val_kl"].append(np.mean(v_kl))

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
                print(f"Epoch {epoch:3d} | Train: {t_mean:.6f} (recon={np.mean(t_recon):.6f} kl={np.mean(t_kl):.6f}) | Val: {v_mean:.6f}")

        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        os.remove(ckpt_path)
        return history

    train_start = datetime.now()
    t0 = time.time()
    history = train_model(model, train_loader, val_loader, config, device, DRY_RUN)
    train_elapsed = time.time() - t0

    mo.md(f"**Training complete** in {train_elapsed:.0f}s — {len(history['train_loss'])} epochs")
    return history, train_elapsed


@app.cell
def _(history, plt):
    skip = 5

    fig_loss, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_recon"][skip:], label="train recon")
    axes[1].plot(history["val_recon"][skip:], label="val recon")
    axes[1].set_title(f"Reconstruction loss (epoch {skip}+)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    axes[2].plot(history["train_kl"], label="train KL")
    axes[2].plot(history["val_kl"], label="val KL")
    axes[2].set_title("KL divergence")
    axes[2].set_xlabel("epoch")
    axes[2].legend()

    fig_loss.tight_layout()
    fig_loss
    return (fig_loss,)


@app.cell
def _(EXPERIMENT_NAME, evaluate, metadata, model, test_ds):
    result = evaluate(
        model=model,
        dataset=test_ds,
        metadata=metadata,
        condition_col="condition",
        name=EXPERIMENT_NAME,
    )
    result.summary()
    return (result,)


@app.cell
def _(result):
    result.figures["reconstructions"]
    return


@app.cell
def _(result):
    result.figures["mse_distribution"]
    return


@app.cell
def _(result):
    result.figures["power_spectra"]
    return


@app.cell
def _(result):
    result.figures["kl_per_dim"]
    return


@app.cell
def _(result):
    result.figures["encoder_uncertainty"]
    return


@app.cell
def _(result):
    result.figures["stimulus_invariance"]
    return


@app.cell
def _(result):
    result.figures["within_condition"]
    return


@app.cell
def _(result):
    result.figures["latent_traversals"]
    return


@app.cell
def _(mo, te_ids):
    traj_selector = mo.ui.slider(
        0, len(te_ids) - 1, value=0, label="Test trajectory index"
    )
    traj_selector
    return (traj_selector,)


@app.cell
def _(mo, traj_len):
    time_slider = mo.ui.slider(
        0, traj_len - 1, value=0, label="Timestep"
    )
    time_slider
    return (time_slider,)


@app.cell
def _(
    STATE_COLS,
    device,
    df,
    mo,
    model,
    np,
    plt,
    te_ids,
    time_slider,
    torch,
    traj_len,
    traj_selector,
):
    _traj_idx = traj_selector.value
    _t = time_slider.value
    _row = df.iloc[te_ids[_traj_idx]]

    _erk = np.array(_row["ERK_s"], dtype=np.float32)
    _light = np.array(_row["light"], dtype=np.float32)

    _erk_t = torch.tensor(_erk[np.newaxis, np.newaxis, :]).to(device)
    _light_t = torch.tensor(_light[np.newaxis, np.newaxis, :]).to(device)
    _enc_input = torch.cat([_erk_t, _light_t], dim=1)

    model.eval()
    with torch.no_grad():
        _mu, _logvar = model.encoder(_enc_input)
        _recon = model.decoder(_mu, _light_t).cpu().numpy()[0, 0]
    _latent = _mu.cpu().numpy()[0]

    _all_states = np.stack([np.array(_row[c], dtype=np.float32) for c in STATE_COLS], axis=1)
    _state_names = ["RAS", "RAF", "MEK", "NFB", "ERK"]
    _times = np.arange(traj_len)

    _light_scaled = _light / max(_light.max(), 1e-8) * max(_erk.max(), _all_states.max())

    _fig, (_ax_orig, _ax_latent) = plt.subplots(1, 2, figsize=(14, 5))

    _ax_orig.fill_between(
        _times, 0, _light_scaled,
        color="gold", alpha=0.25, label="light", step="mid"
    )
    for _i, _name in enumerate(_state_names):
        _alpha = 0.7 if _name == "ERK" else 0.25
        _lw = 1.5 if _name == "ERK" else 0.8
        _ax_orig.plot(_times, _all_states[:, _i], label=_name, alpha=_alpha, linewidth=_lw)
    _ax_orig.plot(_times, _recon, color="tab:red", linestyle="--", linewidth=1.5, label="ERK recon")
    _ax_orig.axvline(_t, color="red", linewidth=2, alpha=0.6)
    _ax_orig.scatter([_t], [_erk[_t]], c="red", zorder=5, s=80)
    _ax_orig.scatter([_t], [_recon[_t]], c="tab:red", zorder=5, s=80, marker="x")
    _ax_orig.set_xlabel("Timestep")
    _ax_orig.set_ylabel("Value")
    _ax_orig.set_title(f"Trajectory {te_ids[_traj_idx]} ({_row['generator']})")
    _ax_orig.legend(loc="upper right", fontsize=8)

    _n_latent = _latent.shape[0]
    if _n_latent >= 2:
        _ax_latent.set_xlabel("z₁")
        _ax_latent.set_ylabel("z₂")
    else:
        _ax_latent.set_xlabel("z₁")
        _ax_latent.set_ylabel("(1D latent)")

    _ax_latent.set_title(f"Latent embedding (t={_t})")

    if _n_latent >= 2:
        _ax_latent.scatter(
            _latent[0], _latent[1],
            c="red", s=120, zorder=5, edgecolors="black", linewidths=1.5
        )
    _ax_latent.legend(fontsize=8)

    _fig.tight_layout()

    _erk_val = f"{_erk[_t]:.3f}"
    _recon_val = f"{_recon[_t]:.3f}"
    _latent_str = ", ".join(f"{v:.3f}" for v in _latent)

    mo.vstack([
        _fig,
        mo.md(f"**Trajectory {te_ids[_traj_idx]}** ({_row['generator']}) | t=**{_t}/{traj_len-1}** | ERK: {_erk_val} | Recon: {_recon_val} | Latent: ({_latent_str})")
    ])
    return


@app.cell
def _(
    EXPERIMENT_NAME,
    config,
    fig_loss,
    history,
    mo,
    model,
    result,
    save_experiment,
    train_elapsed,
    traj_len,
):
    output_dir = f"results/{EXPERIMENT_NAME}"

    save_experiment(
        directory=output_dir,
        model=model,
        model_config=dict(
            in_channels=2,
            stim_channels=1,
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            seq_length=traj_len,
            beta=config["beta"],
            alpha=config["alpha"],
        ),
        training_config=config,
        training_results={
            "history": history,
            "train_elapsed_s": train_elapsed,
        },
        metrics=result.metrics,
        figures={"loss_curves": fig_loss, **result.figures},
        name=EXPERIMENT_NAME,
    )

    mo.md(f"**Saved** to `{output_dir}/`")
    return


if __name__ == "__main__":
    app.run()
