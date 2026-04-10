import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
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

    from utils import get_device, get_username, running_on_cluster, results_write_path, parse_bool
    from experiments.seq2seq_data import load_synthetic, load_real, STIM_COLS
    from experiments.dropout_uncertainty_lstm_seq2seq import Seq2Seq, Seq2SeqBaseline
    import altair as alt

    device = get_device()
    n_stim = len(STIM_COLS)

    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()


@app.cell
def _():
    import itertools
    import polars as pl

    args = mo.cli_args()

    EXPERIMENT_NAME = args.get("name", "lstm_gridsearch_seq2seq")
    DRY_RUN = parse_bool(args.get("dry_run", True))
    _cli_source = args.get("source", None)

    source_selector = mo.ui.dropdown(
        options=["synthetic", "real"], value=_cli_source or "synthetic", label="Data source"
    )

    mo.hstack([source_selector], gap=2)
    return DRY_RUN, EXPERIMENT_NAME, args, itertools, pl, source_selector


@app.cell
def _(DRY_RUN, EXPERIMENT_NAME, args, itertools, source_selector):
    DATA_SOURCE = source_selector.value

    def _parse_grid(raw, cast):
        return [cast(x) for x in str(raw).split(",")]

    grid = dict(
        history_len=_parse_grid(args.get("history_len", "10,20,30" if not DRY_RUN else "10"), int),
        future_len=_parse_grid(args.get("future_len", "2,5,10" if not DRY_RUN else "2,5"), int),
        lr=_parse_grid(args.get("lr", "1e-3,5e-4" if not DRY_RUN else "1e-3"), float),
        num_layers=_parse_grid(args.get("num_layers", "1,2" if not DRY_RUN else "1"), int),
        hidden_dim=_parse_grid(args.get("hidden_dim", "16,32,64" if not DRY_RUN else "16,32"), int),
        patience=_parse_grid(args.get("patience", "30,50" if not DRY_RUN else "15"), int),
        dropout=_parse_grid(args.get("dropout", "0.1,0.25,0.5" if not DRY_RUN else "0.1,0.5"), float),
    )

    grid_keys = list(grid.keys())
    grid_combos = list(itertools.product(*grid.values()))

    _grid_rows = "".join(f"| `{k}` | {v} |\n" for k, v in grid.items())

    mo.md(f"""
    # LSTM Seq2Seq Gridsearch: `{EXPERIMENT_NAME}`

    **Source:** {DATA_SOURCE} | **Dry run:** {DRY_RUN} | **Combinations:** {len(grid_combos)}

    | Parameter | Values |
    |-----------|--------|
    {_grid_rows}
    """)
    return DATA_SOURCE, grid_combos, grid_keys


@app.cell
def _():
    _headless = "name" in mo.cli_args()
    run_button = mo.ui.run_button(label="Run gridsearch")
    run_button if not _headless else None
    return (run_button,)


@app.cell
def _(
    DATA_SOURCE,
    DRY_RUN,
    EXPERIMENT_NAME,
    args,
    grid_combos,
    grid_keys,
    run_button,
):
    _headless = "name" in mo.cli_args()
    mo.stop(not _headless and not run_button.value, mo.md("Click **Run gridsearch** when ready."))

    _epochs = int(args.get("epochs", "50" if DRY_RUN else "200"))
    _batch_size = int(args.get("batch_size", "64"))

    _exp_dir = mo.cli_args().get("results-dir", f"{results_base}/{EXPERIMENT_NAME}")
    import json
    os.makedirs(_exp_dir, exist_ok=True)
    _out_path = Path(_exp_dir) / "gridsearch_results.json"

    # Resume: reload already-completed rows so a crashed run can continue
    _done_keys = set()
    if _out_path.exists():
        with open(_out_path) as _f:
            _existing = json.load(_f)
        gs_results = list(_existing)
        for _r in _existing:
            _done_keys.add(tuple(_r[k] for k in grid_keys))
    else:
        gs_results = []

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

    def _run_epoch_ar(mdl, loader, dev, optimizer, cfg, epoch, is_train):
        if is_train:
            mdl.train()
            tf_start, tf_end = cfg["tf_ratio_start"], cfg["tf_ratio_end"]
            tf_ratio = tf_start - (tf_start - tf_end) * epoch / max(cfg["epochs"] - 1, 1)
        else:
            mdl.eval()
            tf_ratio = 0.0
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in, dec_stim, dec_target = enc_in.to(dev), dec_stim.to(dev), dec_target.to(dev)
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

    def _run_epoch_bl(mdl, loader, dev, optimizer, is_train):
        mdl.train() if is_train else mdl.eval()
        losses = []
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for enc_in, dec_stim, dec_target in loader:
                enc_in, dec_stim, dec_target = enc_in.to(dev), dec_stim.to(dev), dec_target.to(dev)
                preds = mdl(enc_in, dec_stim)
                loss = mdl.loss(preds, dec_target)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
                    optimizer.step()
                losses.append(loss.item())
        return np.mean(losses)

    def _train_single(model_ar, model_bl, train_ldr, val_ldr, cfg, dev):
        opt_ar = optim.Adam(model_ar.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        opt_bl = optim.Adam(model_bl.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        sched_ar = optim.lr_scheduler.ReduceLROnPlateau(opt_ar, patience=10, factor=0.5)
        sched_bl = optim.lr_scheduler.ReduceLROnPlateau(opt_bl, patience=10, factor=0.5)

        epochs, patience = cfg["epochs"], cfg["patience"]
        best_ar, best_bl, wait_ar, wait_bl = float("inf"), float("inf"), 0, 0
        done_ar, done_bl = False, False
        hist_ar = {"train_loss": [], "val_loss": []}
        hist_bl = {"train_loss": [], "val_loss": []}

        ckpt_fd_ar, ckpt_ar = tempfile.mkstemp(suffix=".pt")
        ckpt_fd_bl, ckpt_bl = tempfile.mkstemp(suffix=".pt")
        os.close(ckpt_fd_ar); os.close(ckpt_fd_bl)

        for epoch in range(epochs):
            if not done_ar:
                t_ar = _run_epoch_ar(model_ar, train_ldr, dev, opt_ar, cfg, epoch, True)
                v_ar = _run_epoch_ar(model_ar, val_ldr, dev, opt_ar, cfg, epoch, False)
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
                t_bl = _run_epoch_bl(model_bl, train_ldr, dev, opt_bl, True)
                v_bl = _run_epoch_bl(model_bl, val_ldr, dev, opt_bl, False)
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

        model_ar.load_state_dict(torch.load(ckpt_ar, weights_only=True))
        model_bl.load_state_dict(torch.load(ckpt_bl, weights_only=True))
        os.remove(ckpt_ar); os.remove(ckpt_bl)
        return best_ar, best_bl, hist_ar, hist_bl

    def _eval_test(mdl, test_ldr, dev):
        mdl.eval()
        _preds, _targets = [], []
        with torch.no_grad():
            for enc_in, dec_stim, dec_target in test_ldr:
                enc_in, dec_stim = enc_in.to(dev), dec_stim.to(dev)
                _preds.append(mdl(enc_in, dec_stim).cpu().numpy())
                _targets.append(dec_target.numpy())
        preds = np.concatenate(_preds)
        targets = np.concatenate(_targets)
        mse = float(np.mean((preds - targets) ** 2))
        mse_baseline = float(np.mean(targets ** 2))  # persist-last
        r2 = 1 - np.sum((preds - targets) ** 2) / max(np.sum((targets - targets.mean()) ** 2), 1e-8)
        return mse, mse_baseline, float(r2)

    # --- Load data once per unique (history_len, future_len) ---
    _unique_windows = set((c[0], c[1]) for c in grid_combos)
    _data_cache = {}
    for _h, _f in _unique_windows:
        _total = _h + _f
        if DATA_SOURCE == "synthetic":
            _cnr, _stim, _cond = load_synthetic()
        else:
            _cnr, _stim, _cond = load_real(
                window_size=_total, stride=max(1, _total // 4),
            )
        _n_traj = len(_cnr)
        _traj_ids = np.arange(_n_traj)
        _tr, _te = train_test_split(_traj_ids, test_size=0.2, random_state=42)
        _tr, _va = train_test_split(_tr, test_size=0.125, random_state=42)
        _stride = 15
        _train_ds = Seq2SeqDataset(_cnr[_tr], _stim[_tr], _h, _f, stride=_stride)
        _val_ds = Seq2SeqDataset(_cnr[_va], _stim[_va], _h, _f, stride=_stride)
        _test_ds = Seq2SeqDataset(_cnr[_te], _stim[_te], _h, _f, stride=_stride)
        if DRY_RUN:
            _n_dry = 5000
            _train_ds = Subset(_train_ds, range(min(_n_dry, len(_train_ds))))
            _val_ds = Subset(_val_ds, range(min(_n_dry, len(_val_ds)) // 4))
            _test_ds = Subset(_test_ds, range(min(_n_dry, len(_test_ds))))
        _data_cache[(_h, _f)] = (_train_ds, _val_ds, _test_ds)

    # --- Run gridsearch ---
    _encoder_dim = 1 + n_stim
    _stim_dim = n_stim

    _total_combos = len(grid_combos)
    for _ci, _combo in enumerate(grid_combos):
        _params = dict(zip(grid_keys, _combo))
        if tuple(_combo) in _done_keys:
            print(f"[{_ci+1}/{_total_combos}] Skipping (already done): {_params}")
            continue
        _h, _f = _params["history_len"], _params["future_len"]
        _train_ds, _val_ds, _test_ds = _data_cache[(_h, _f)]

        _cfg = dict(
            **_params,
            epochs=_epochs,
            batch_size=_batch_size,
            tf_ratio_start=1.0,
            tf_ratio_end=0.0,
        )

        _train_ldr = DataLoader(_train_ds, batch_size=_batch_size, shuffle=True)
        _val_ldr = DataLoader(_val_ds, batch_size=_batch_size, shuffle=False)
        _test_ldr = DataLoader(_test_ds, batch_size=_batch_size, shuffle=False)

        _mdl_ar = Seq2Seq(
            encoder_dim=_encoder_dim, stim_dim=_stim_dim,
            hidden_dim=_params["hidden_dim"], num_layers=_params["num_layers"],
            dropout=_params['dropout']
        ).to(device)
        _mdl_bl = Seq2SeqBaseline(
            encoder_dim=_encoder_dim, stim_dim=_stim_dim,
            hidden_dim=_params["hidden_dim"], num_layers=_params["num_layers"],
            dropout=_params['dropout']
        ).to(device)

        _t0 = time.time()
        _best_val_ar, _best_val_bl, _hist_ar, _hist_bl = _train_single(
            _mdl_ar, _mdl_bl, _train_ldr, _val_ldr, _cfg, device,
        )
        _elapsed = time.time() - _t0

        _test_mse_ar, _test_mse_persist, _test_r2_ar = _eval_test(_mdl_ar, _test_ldr, device)
        _test_mse_bl, _, _test_r2_bl = _eval_test(_mdl_bl, _test_ldr, device)

        _n_params_ar = sum(p.numel() for p in _mdl_ar.parameters())

        _row = {
            **_params,
            "best_val_ar": _best_val_ar,
            "best_val_bl": _best_val_bl,
            "test_mse_ar": _test_mse_ar,
            "test_mse_bl": _test_mse_bl,
            "test_mse_persist": _test_mse_persist,
            "test_r2_ar": _test_r2_ar,
            "test_r2_bl": _test_r2_bl,
            "epochs_ar": len(_hist_ar["train_loss"]),
            "epochs_bl": len(_hist_bl["train_loss"]),
            "n_params": _n_params_ar,
            "train_time_s": _elapsed,
        }
        gs_results.append(_row)
        with open(_out_path, "w") as _f:
            json.dump(gs_results, _f, indent=2)

        print(f"[{_ci+1}/{_total_combos}] H={_h} F={_f} hid={_params['hidden_dim']} "
              f"layers={_params['num_layers']} lr={_params['lr']} pat={_params['patience']} "
              f"| val_ar={_best_val_ar:.6f} test_ar={_test_mse_ar:.6f} "
              f"r2_ar={_test_r2_ar:.4f} ({_elapsed:.0f}s)")

    mo.md(f"**Gridsearch complete:** {len(gs_results)} / {_total_combos} combinations")
    return (gs_results,)


@app.cell
def _(gs_results, pl):
    gs_df = pl.DataFrame(gs_results)
    gs_df_sorted = gs_df.sort("test_mse_ar")
    gs_df_sorted
    return gs_df, gs_df_sorted


@app.cell
def _(gs_df_sorted):
    mo.vstack([
        mo.md("## Best configurations (by AR test MSE)"),
        mo.md("**Top-5:**"),
        gs_df_sorted.head(5),
        mo.md("**Worst-5:**"),
        gs_df_sorted.tail(5),
    ])
    return


@app.cell
def _(gs_df, pl):
    _param_names = ["history_len", "future_len", "lr", "num_layers", "hidden_dim", "patience"]
    _charts = []
    for _p in _param_names:
        _unique_vals = gs_df[_p].unique().sort().to_list()
        if len(_unique_vals) < 2:
            continue
        _agg = (
            gs_df
            .group_by(_p)
            .agg([
                pl.col("test_mse_ar").mean().alias("mean_mse_ar"),
                pl.col("test_mse_ar").min().alias("min_mse_ar"),
                pl.col("test_r2_ar").mean().alias("mean_r2_ar"),
                pl.col("test_mse_bl").mean().alias("mean_mse_bl"),
            ])
            .sort(_p)
        )
        _c = alt.Chart(_agg).mark_bar().encode(
            x=alt.X(f"{_p}:N", title=_p),
            y=alt.Y("mean_mse_ar:Q", title="Mean test MSE (AR)"),
            tooltip=[_p, "mean_mse_ar", "min_mse_ar", "mean_r2_ar"],
        ).properties(width=180, height=200, title=_p)
        _charts.append(_c)

    mo.vstack([
        mo.md("## Parameter sensitivity"),
        mo.hstack([alt.hconcat(*_charts)] if _charts else [mo.md("Not enough variation")]),
    ])
    return


@app.cell
def _(gs_df):
    _best = gs_df.sort("test_mse_ar").head(1).row(0, named=True)
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    # AR vs Baseline test MSE
    _axes[0].scatter(gs_df["test_mse_bl"].to_list(), gs_df["test_mse_ar"].to_list(),
                     s=30, alpha=0.6, c=gs_df["hidden_dim"].to_list(), cmap="viridis")
    _lim = [0, max(gs_df["test_mse_ar"].max(), gs_df["test_mse_bl"].max()) * 1.05]
    _axes[0].plot(_lim, _lim, "k--", lw=1)
    _axes[0].set_xlabel("Baseline test MSE")
    _axes[0].set_ylabel("AR test MSE")
    _axes[0].set_title("AR vs Baseline (color=hidden_dim)")

    # Test MSE vs params
    _axes[1].scatter(gs_df["n_params"].to_list(), gs_df["test_mse_ar"].to_list(),
                     s=30, alpha=0.6, c=gs_df["num_layers"].to_list(), cmap="Set1")
    _axes[1].set_xlabel("# Parameters")
    _axes[1].set_ylabel("AR test MSE")
    _axes[1].set_title("Capacity vs performance (color=layers)")

    # R² distribution
    _axes[2].hist(gs_df["test_r2_ar"].to_list(), bins=20, alpha=0.6, color="tab:blue", label="AR")
    _axes[2].hist(gs_df["test_r2_bl"].to_list(), bins=20, alpha=0.6, color="tab:orange", label="Baseline")
    _axes[2].set_xlabel("Test R²")
    _axes[2].set_title("R² distribution")
    _axes[2].legend()

    _fig.tight_layout()

    mo.vstack([
        mo.md("## Gridsearch diagnostics"),
        _fig,
        mo.md(f"""
    **Best config:** H={_best['history_len']} F={_best['future_len']} hidden={_best['hidden_dim']} layers={_best['num_layers']} lr={_best['lr']} patience={_best['patience']}
    → test MSE={_best['test_mse_ar']:.6f}, R²={_best['test_r2_ar']:.4f}
    """),
    ])
    return


@app.cell
def _(EXPERIMENT_NAME, gs_df):
    _exp_dir = mo.cli_args().get("results-dir", f"{results_base}/{EXPERIMENT_NAME}")
    gs_df.write_parquet(str(Path(_exp_dir) / "gridsearch_results.parquet"))

    _env_label = f"**Cluster** (`{hostname}`)" if is_cluster else f"**Local** (`{hostname}`)"
    mo.md(f"**Saved** on {_env_label}\n\n`{_exp_dir}`")
    return


if __name__ == "__main__":
    app.run()
