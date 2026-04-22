import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import marimo as mo
    import numpy as np
    import os
    import tempfile
    import time

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from experiment import ExperimentTracker
    from experiments.seq2scal_model import Seq2Scalar
    from experiments.seq2scal_pipeline import build_loaders
    from experiments.seq2seq_data import AVAILABLE_DATASETS, STIM_COLS
    from utils import get_device, parse_bool

    device = get_device()
    n_stim = len(STIM_COLS)
    return (
        AVAILABLE_DATASETS,
        ExperimentTracker,
        Path,
        Seq2Scalar,
        build_loaders,
        device,
        mo,
        n_stim,
        nn,
        np,
        optim,
        os,
        parse_bool,
        tempfile,
        time,
        torch,
    )


@app.cell
def _(mo, parse_bool):
    args = mo.cli_args()
    SEED = int(args.get("seed", "0"))
    RESULTS_DIR = args.get("results-dir")
    assert RESULTS_DIR, "Must pass --results-dir pointing at the seed's subdirectory"
    EXPERIMENT_NAME = args.get("name", f"seed_{SEED}")
    DRY_RUN = parse_bool(args.get("dry_run", True))

    mo.md(f"# Seq2Scalar single seed — `{EXPERIMENT_NAME}` (seed={SEED}, dry_run={DRY_RUN})")
    return DRY_RUN, EXPERIMENT_NAME, RESULTS_DIR, SEED, args


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
def _(DRY_RUN, build_loaders, config, mo):
    loaders = build_loaders(config, dry_run=DRY_RUN)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    mo.md(f"""
    **{config['data_source']}:** {loaders['n_traj']} tracks × {loaders['traj_len']} timepoints
    Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} windows
    """)
    return train_loader, val_loader


@app.cell
def _(
    EXPERIMENT_NAME,
    ExperimentTracker,
    Path,
    RESULTS_DIR,
    SEED,
    Seq2Scalar,
    config,
    device,
    mo,
    n_stim,
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
    _bundle_path = Path(RESULTS_DIR) / "bundle.pt"
    _skip_training = _bundle_path.exists()

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
        seed=SEED,
    )

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
        cur = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        tracker.checkpoint(model, training_results={"history": hist})
        model.load_state_dict(cur)

    if _skip_training:
        mo.output.replace(mo.md(f"**Skipped** — bundle already exists at `{_bundle_path}`."))
    else:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = Seq2Scalar(**{
            k: v for k, v in shared_model_config.items()
            if k not in ("history_len", "future_len", "data_source", "seed")
        }).to(device)
        opt = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        tracker = ExperimentTracker(
            directory=RESULTS_DIR,
            name=EXPERIMENT_NAME,
            model_config=shared_model_config,
            training_config={**config, "seed": SEED},
            checkpoint_interval_s=300,
        )
        tracker.register_start()

        hist = {"train_loss": [], "val_loss": []}
        fd, ckpt = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        best = float("inf")
        best_epoch = 0
        wait = 0
        t0 = time.time()

        for epoch in range(config["epochs"]):
            t_loss, tf = _run_epoch(model, train_loader, opt, config, epoch, True)
            v_loss, _ = _run_epoch(model, val_loader, opt, config, epoch, False)
            hist["train_loss"].append(t_loss)
            hist["val_loss"].append(v_loss)
            sched.step(v_loss)

            if v_loss < best:
                best = v_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= config["patience"]:
                    print(f"[seed {SEED}] early stop @ epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"[seed {SEED}] epoch {epoch:3d} tf={tf:.2f} train={t_loss:.5f} val={v_loss:.5f}")

            _checkpoint_best(tracker, model, ckpt, hist)

        elapsed = time.time() - t0
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        os.remove(ckpt)

        tracker.save_final(
            model,
            training_results={"history": hist, "train_elapsed_s": elapsed},
            metrics={
                "best_val_loss": best,
                "best_epoch": best_epoch,
                "final_train_loss": hist["train_loss"][-1],
                "final_val_loss": hist["val_loss"][-1],
            },
            figures={},
        )

        mo.output.replace(mo.md(
            f"**Done.** seed={SEED}, best_val={best:.5f} @ epoch {best_epoch}, elapsed {elapsed:.0f}s → `{RESULTS_DIR}`"
        ))
    return


if __name__ == "__main__":
    app.run()
