import itertools
from pathlib import Path

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


# ---------------------------------------------------------------------------
# Module-level helpers (importable, run outside marimo cells)
# ---------------------------------------------------------------------------

def _build_param_grid(dry_run):
    if dry_run:
        return {
            "hidden_dim": [16, 32],
            "num_layers": [1, 2],
            "history_len": [15],
            "future_len": [5],
            "lr": [1e-3],
            "batch_size": [64],
            "patience": [5],
            "epochs": [10],
            "tf_ratio_start": [1.0],
            "tf_ratio_end": [0.0],
        }
    return {
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "history_len": [10, 15, 20],
        "future_len": [5, 10, 15],
        "lr": [1e-3, 5e-4, 1e-4],
        "batch_size": [32, 64],
        "patience": [30, 50],
        "epochs": [300],
        "tf_ratio_start": [1.0],
        "tf_ratio_end": [0.0],
    }


def _run_single(source, config, results_base, experiment_name, device):
    """Train AR + baseline for one config. Returns (hist_ar, hist_bl, best_val_ar, best_val_bl, elapsed)."""
    import os
    import tempfile
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset, Subset

    from experiment import ExperimentTracker, compute_training_stats
    from experiments.seq2seq_data import load_synthetic, load_real, STIM_COLS

    n_stim = len(STIM_COLS)
    H = config["history_len"]
    F_ = config["future_len"]
    total_window = H + F_

    # --- data ---
    if source == "synthetic":
        cnr_all, stim_all, conditions_all = load_synthetic()
    else:
        cnr_all, stim_all, conditions_all = load_real(
            window_size=total_window,
            stride=max(1, total_window // 4),
        )

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
                    dec_target = np.diff(full_window)[
                        history_len - 1 : history_len - 1 + future_len
                    ]
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

    traj_ids = np.arange(len(cnr_all))
    tr_ids, te_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    tr_ids, va_ids = train_test_split(tr_ids, test_size=0.125, random_state=42)

    stride = 15
    train_ds = Seq2SeqDataset(cnr_all[tr_ids], stim_all[tr_ids], H, F_, stride=stride)
    val_ds = Seq2SeqDataset(cnr_all[va_ids], stim_all[va_ids], H, F_, stride=stride)

    BS = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

    # --- model ---
    def _init_forget_bias(lstm):
        for name, param in lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    class LSTMEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                                dropout=0.1 if num_layers > 1 else 0.0)
            _init_forget_bias(self.lstm)

        def forward(self, x):
            _, (h_n, c_n) = self.lstm(x)
            return h_n, c_n

    class LSTMDecoder(nn.Module):
        def __init__(self, stim_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(stim_dim, hidden_dim, num_layers, batch_first=True,
                                dropout=0.1 if num_layers > 1 else 0.0)
            _init_forget_bias(self.lstm)
            self.fc_out = nn.Linear(hidden_dim, 1)

        def forward(self, future_stim, h_0, c_0):
            out, _ = self.lstm(future_stim, (h_0, c_0))
            return self.fc_out(out).squeeze(-1)

    class Seq2Seq(nn.Module):
        def __init__(self, encoder_dim, stim_dim, hidden_dim, num_layers):
            super().__init__()
            self.encoder = LSTMEncoder(encoder_dim, hidden_dim, num_layers)
            self.decoder = LSTMDecoder(stim_dim, hidden_dim, num_layers)

        def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
            B, H_len, _ = encoder_input.shape
            F_len = future_stim.shape[1]
            current_window = encoder_input
            predictions = []
            for i in range(F_len):
                h, c = self.encoder(current_window)
                pred = self.decoder(future_stim[:, i : i + 1, :], h, c)
                predictions.append(pred.squeeze(1))
                if i < F_len - 1:
                    last_abs = current_window[:, -1, 0:1]
                    use_teacher = targets is not None and torch.rand(1).item() < tf_ratio
                    next_cnr_abs = last_abs + (targets[:, i : i + 1] if use_teacher else pred)
                    next_input = torch.cat([next_cnr_abs, future_stim[:, i, :]], dim=-1).unsqueeze(1)
                    current_window = torch.cat([current_window[:, 1:, :], next_input], dim=1)
            return torch.stack(predictions, dim=1)

        def loss(self, predictions, target):
            return nn.functional.mse_loss(predictions, target)

    class Seq2SeqBaseline(nn.Module):
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
    model_ar = Seq2Seq(encoder_dim, n_stim, config["hidden_dim"], config["num_layers"]).to(device)
    model_bl = Seq2SeqBaseline(encoder_dim, n_stim, config["hidden_dim"], config["num_layers"]).to(device)

    model_config = dict(
        encoder_dim=encoder_dim, stim_dim=n_stim,
        hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
        history_len=H, future_len=F_, data_source=source,
    )

    # --- tracker ---
    tracker = ExperimentTracker(
        directory=f"{results_base}/{experiment_name}",
        name=experiment_name,
        model_config=model_config,
        training_config=config,
    )
    tracker.register_start()
    tracker_ar = tracker.make_subexperiment(
        "ar", model_config=dict(**model_config, variant="autoregressive_tf"),
    )
    tracker_bl = tracker.make_subexperiment(
        "baseline", model_config=dict(**model_config, variant="single_pass"),
    )
    tracker_ar.register_start()
    tracker_bl.register_start()

    # --- train ---
    def _run_epoch_ar(mdl, loader, optimizer, cfg, epoch, is_train):
        if is_train:
            mdl.train()
            tf_start, tf_end, epochs = cfg["tf_ratio_start"], cfg["tf_ratio_end"], cfg["epochs"]
            tf_ratio = tf_start - (tf_start - tf_end) * epoch / max(epochs - 1, 1)
        else:
            mdl.eval()
            tf_ratio = 0.0
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in, dec_stim, dec_target = enc_in.to(device), dec_stim.to(device), dec_target.to(device)
                targets = dec_target if is_train else None
                preds = mdl(enc_in, dec_stim, targets=targets, tf_ratio=tf_ratio)
                loss = mdl.loss(preds, dec_target)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
                    optimizer.step()
                losses.append(loss.item())
        return np.mean(losses)

    def _run_epoch_bl(mdl, loader, optimizer, is_train):
        mdl.train() if is_train else mdl.eval()
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in, dec_stim, dec_target = enc_in.to(device), dec_stim.to(device), dec_target.to(device)
                preds = mdl(enc_in, dec_stim)
                loss = mdl.loss(preds, dec_target)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
                    optimizer.step()
                losses.append(loss.item())
        return np.mean(losses)

    opt_ar = optim.Adam(model_ar.parameters(), lr=config["lr"], weight_decay=1e-5)
    opt_bl = optim.Adam(model_bl.parameters(), lr=config["lr"], weight_decay=1e-5)
    sched_ar = optim.lr_scheduler.ReduceLROnPlateau(opt_ar, patience=10, factor=0.5)
    sched_bl = optim.lr_scheduler.ReduceLROnPlateau(opt_bl, patience=10, factor=0.5)

    epochs, patience = config["epochs"], config["patience"]
    hist_ar = {"train_loss": [], "val_loss": []}
    hist_bl = {"train_loss": [], "val_loss": []}

    ckpt_fd_ar, ckpt_ar = tempfile.mkstemp(suffix=".pt")
    ckpt_fd_bl, ckpt_bl = tempfile.mkstemp(suffix=".pt")
    os.close(ckpt_fd_ar)
    os.close(ckpt_fd_bl)

    best_ar, best_bl, wait_ar, wait_bl = float("inf"), float("inf"), 0, 0
    done_ar, done_bl = False, False

    t0 = time.time()
    for epoch in range(epochs):
        if not done_ar:
            t_ar = _run_epoch_ar(model_ar, train_loader, opt_ar, config, epoch, True)
            v_ar = _run_epoch_ar(model_ar, val_loader, opt_ar, config, epoch, False)
            hist_ar["train_loss"].append(t_ar)
            hist_ar["val_loss"].append(v_ar)
            sched_ar.step(v_ar)
            if v_ar < best_ar:
                best_ar, wait_ar = v_ar, 0
                torch.save(model_ar.state_dict(), ckpt_ar)
            else:
                wait_ar += 1
                if wait_ar >= patience:
                    done_ar = True

        if not done_bl:
            t_bl = _run_epoch_bl(model_bl, train_loader, opt_bl, True)
            v_bl = _run_epoch_bl(model_bl, val_loader, opt_bl, False)
            hist_bl["train_loss"].append(t_bl)
            hist_bl["val_loss"].append(v_bl)
            sched_bl.step(v_bl)
            if v_bl < best_bl:
                best_bl, wait_bl = v_bl, 0
                torch.save(model_bl.state_dict(), ckpt_bl)
            else:
                wait_bl += 1
                if wait_bl >= patience:
                    done_bl = True

        if done_ar and done_bl:
            break

    elapsed = time.time() - t0

    # restore best weights
    model_ar.load_state_dict(torch.load(ckpt_ar, weights_only=True))
    model_bl.load_state_dict(torch.load(ckpt_bl, weights_only=True))
    os.remove(ckpt_ar)
    os.remove(ckpt_bl)

    # --- save ---
    stats_ar = compute_training_stats(elapsed, hist_ar, len(train_loader.dataset),
                                       len(val_loader.dataset), model_ar)
    stats_bl = compute_training_stats(elapsed, hist_bl, len(train_loader.dataset),
                                       len(val_loader.dataset), model_bl)
    tracker_ar.save_final(
        model=model_ar,
        training_results={"history": hist_ar, "train_elapsed_s": elapsed, "stats": stats_ar},
        metrics={}, figures={},
    )
    tracker_bl.save_final(
        model=model_bl,
        training_results={"history": hist_bl, "train_elapsed_s": elapsed, "stats": stats_bl},
        metrics={}, figures={},
    )

    print(f"  [{experiment_name}] done in {elapsed:.0f}s  AR={best_ar:.5f}  BL={best_bl:.5f}  → {tracker.directory}")
    return hist_ar, hist_bl, best_ar, best_bl, elapsed


def _run_gridsearch(source, dry_run, results_base, experiment_name, device):
    """Run a grid search over hyperparameters. Returns (results_df, directory)."""
    import pandas as pd

    grid = _build_param_grid(dry_run)
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Grid search: {len(combos)} configurations ({source}, dry_run={dry_run})")

    rows = []
    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        tag = f"gs_{source}_h{config['hidden_dim']}_l{config['num_layers']}_H{config['history_len']}_F{config['future_len']}_lr{config['lr']}_bs{config['batch_size']}"
        print(f"\n[{i+1}/{len(combos)}] {tag}")
        try:
            _, _, best_ar, best_bl, elapsed = _run_single(
                source, config, results_base, tag, device,
            )
            rows.append({**config, "best_val_ar": best_ar, "best_val_bl": best_bl,
                         "elapsed_s": elapsed, "source": source, "status": "ok"})
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({**config, "best_val_ar": None, "best_val_bl": None,
                         "elapsed_s": None, "source": source, "status": str(e)})

    df = pd.DataFrame(rows)
    out_dir = Path(results_base) / f"{experiment_name}_gridsearch_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nGrid search complete. {len(df)} runs. Results: {csv_path}")
    return df, str(out_dir)


# ---------------------------------------------------------------------------
# Notebook cells
# ---------------------------------------------------------------------------

@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from utils import get_device, parse_bool, results_write_path
    from experiments.lstm_gridsearch_seq2seq import (
        _build_param_grid as build_param_grid,
        _run_gridsearch as run_gridsearch,
    )

    device = get_device()
    results_base = results_write_path()
    return Path, build_param_grid, device, mo, np, parse_bool, plt, results_base, run_gridsearch


@app.cell
def _(mo, parse_bool):
    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "lstm_gridsearch")
    DRY_RUN = parse_bool(args.get("dry_run", True))
    _cli_source = args.get("source", None)
    source_selector = mo.ui.dropdown(
        options=["synthetic", "real"],
        value=_cli_source or "synthetic",
        label="Data source",
    )
    source_selector
    return DRY_RUN, EXPERIMENT_NAME, args, source_selector


@app.cell
def _(DRY_RUN, EXPERIMENT_NAME, build_param_grid, mo, source_selector):
    DATA_SOURCE = source_selector.value

    _grid = build_param_grid(DRY_RUN)
    _n_combos = 1
    for v in _grid.values():
        _n_combos *= len(v)

    mo.md(f"""
    # LSTM Seq2Seq Grid Search: `{EXPERIMENT_NAME}`

    | | |
    |---|---|
    | source | {DATA_SOURCE} |
    | dry_run | {DRY_RUN} |
    | configurations | {_n_combos} |

    **Grid:**

    | param | values |
    |-------|--------|
    """ + "\n".join(f"| {k} | {v} |" for k, v in _grid.items()))
    return (DATA_SOURCE,)


@app.cell
def _(mo):
    _is_cli = mo.app_meta().mode == "run"
    run_button = mo.ui.run_button(label="Launch grid search")
    run_button if not _is_cli else mo.md("")
    return (run_button,)


@app.cell
def _(DATA_SOURCE, DRY_RUN, EXPERIMENT_NAME, device, mo, results_base, run_gridsearch, run_button):
    _is_cli = mo.app_meta().mode == "run"
    mo.stop(not _is_cli and not run_button.value, mo.md("Click **Launch grid search** to start."))

    gs_results_df, gs_directory = run_gridsearch(
        DATA_SOURCE, DRY_RUN, results_base, EXPERIMENT_NAME, device
    )
    mo.md(f"Grid search complete. Results in `{gs_directory}`")
    return gs_directory, gs_results_df


@app.cell
def _(gs_results_df, mo, np, plt):
    mo.stop(gs_results_df is None)

    df = gs_results_df.dropna(subset=["best_val_ar", "best_val_bl"]).copy()
    df["best_val_min"] = df[["best_val_ar", "best_val_bl"]].min(axis=1)
    df_sorted = df.sort_values("best_val_min")

    n_show = min(15, len(df_sorted))
    top = df_sorted.head(n_show)
    labels = [
        f"h{int(r.hidden_dim)}_l{int(r.num_layers)}_H{int(r.history_len)}_F{int(r.future_len)}_lr{r.lr}_bs{int(r.batch_size)}"
        for r in top.itertuples()
    ]

    fig_top, _ax = plt.subplots(figsize=(12, max(4, n_show * 0.35)))
    _y = np.arange(n_show)
    _ax.barh(_y - 0.15, top["best_val_ar"].values, height=0.3, label="AR", color="#4c72b0")
    _ax.barh(_y + 0.15, top["best_val_bl"].values, height=0.3, label="Baseline", color="#dd8452")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(labels, fontsize=8)
    _ax.set_xlabel("Best validation loss")
    _ax.set_title(f"Top {n_show} configurations (lower is better)")
    _ax.legend()
    _ax.invert_yaxis()
    fig_top.tight_layout()

    mo.md("## Grid Search Results — Top Configurations")
    return (df,)


@app.cell
def _(df, mo, plt):
    mo.stop(df is None or len(df) == 0)

    hp_cols = ["hidden_dim", "num_layers", "history_len", "future_len", "lr", "batch_size"]
    hp_cols = [c for c in hp_cols if df[c].nunique() > 1]

    _n_hp = len(hp_cols)
    if _n_hp == 0:
        mo.stop(True, mo.md("_All hyperparameters are constant — nothing to plot._"))

    fig_hp, _axes = plt.subplots(1, _n_hp, figsize=(4 * _n_hp, 4), squeeze=False)
    _axes = _axes[0]

    for _i, _col in enumerate(hp_cols):
        _grouped = df.groupby(_col)["best_val_min"].agg(["mean", "std", "min"]).reset_index()
        _x = _grouped[_col].values.astype(float)
        _axes[_i].errorbar(_x, _grouped["mean"], yerr=_grouped["std"], fmt="o-", capsize=4, label="mean +/- std")
        _axes[_i].scatter(_x, _grouped["min"], marker="^", color="green", zorder=5, label="best")
        _axes[_i].set_xlabel(_col)
        _axes[_i].set_ylabel("Val loss" if _i == 0 else "")
        _axes[_i].legend(fontsize=7)
        _axes[_i].set_title(_col)

    fig_hp.suptitle("Hyperparameter effect on validation loss", y=1.02)
    fig_hp.tight_layout()

    mo.md("## Hyperparameter Effects")
    return


@app.cell
def _(df, mo, plt):
    mo.stop(df is None or len(df) == 0)

    if df["hidden_dim"].nunique() > 1 and df["num_layers"].nunique() > 1:
        pivot = df.pivot_table(
            values="best_val_min",
            index="num_layers",
            columns="hidden_dim",
            aggfunc="mean",
        )
        fig_heat, _ax = plt.subplots(figsize=(6, 4))
        _im = _ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
        _ax.set_xticks(range(len(pivot.columns)))
        _ax.set_xticklabels(pivot.columns)
        _ax.set_yticks(range(len(pivot.index)))
        _ax.set_yticklabels(pivot.index)
        _ax.set_xlabel("hidden_dim")
        _ax.set_ylabel("num_layers")
        _ax.set_title("Mean val loss: hidden_dim × num_layers")
        for _i in range(len(pivot.index)):
            for _j in range(len(pivot.columns)):
                _ax.text(_j, _i, f"{pivot.values[_i, _j]:.5f}", ha="center", va="center", fontsize=9, color="white")
        fig_heat.colorbar(_im, ax=_ax)
        fig_heat.tight_layout()

    mo.md("## Architecture Heatmap")
    return


@app.cell
def _(df, mo, plt):
    mo.stop(df is None or len(df) == 0)

    _complexity = df["hidden_dim"] * df["num_layers"]
    fig_scatter, _ax = plt.subplots(figsize=(8, 5))
    _sc = _ax.scatter(
        df["elapsed_s"] / 60,
        df["best_val_min"],
        s=_complexity * 2,
        c=df["lr"],
        cmap="coolwarm",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    _ax.set_xlabel("Training time (minutes)")
    _ax.set_ylabel("Best validation loss")
    _ax.set_title("Training time vs performance (size = hidden_dim × num_layers, color = lr)")
    fig_scatter.colorbar(_sc, ax=_ax, label="Learning rate")
    fig_scatter.tight_layout()

    mo.md("## Cost vs Performance")
    return


if __name__ == "__main__":
    app.run()
