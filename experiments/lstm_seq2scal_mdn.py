import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import math
    import os
    import time
    import tempfile
    from dataclasses import dataclass
    from typing import Callable

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset
    from sklearn.model_selection import train_test_split

    from pydantic import BaseModel, Field, ConfigDict, model_validator

    from experiment import ExperimentTracker
    from experiments.scaffold import TrainContext


    def _init_forget_bias(lstm):
        for name, param in lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)


    class LSTMEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout,
            )
            _init_forget_bias(self.lstm)

        def forward(self, x):
            _, (h_n, c_n) = self.lstm(x)
            return h_n, c_n


    class MDNHead(nn.Module):
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
        y = target.unsqueeze(-1)
        log_gauss = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((y - mu) / sigma) ** 2
        )
        log_mix = torch.log(pi + 1e-12) + log_gauss
        return -torch.logsumexp(log_mix, dim=-1).mean()


    class ModelConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        encoder_dim: int = Field(..., ge=1)
        stim_dim: int = Field(..., ge=1)
        hidden_dim: int = Field(64, ge=1)
        num_layers: int = Field(2, ge=1)
        n_gaussians: int = Field(3, ge=1)
        n_mlp_layers: int = Field(5, ge=1)
        mlp_hidden: int | None = None
        dropout: float = Field(0.1, ge=0.0, le=0.9)
        history_len: int = Field(30, ge=1)
        future_len: int = Field(5, ge=1)
        data_source: str = "synthetic"
        variant: str = "seq2scalar_mdn_ar_tf"

        @model_validator(mode="after")
        def _fill_mlp_hidden(self):
            if self.mlp_hidden is None:
                object.__setattr__(self, "mlp_hidden", self.hidden_dim)
            return self


    class TrainingConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        lr: float = 1e-3
        weight_decay: float = 1e-4
        epochs: int = 400
        batch_size: int = 64
        patience: int = 100
        tf_ratio_start: float = 1.0
        tf_ratio_end: float = 0.0
        tf_anneal_frac: float = 0.5
        tf_hold_frac: float = 0.3
        grad_clip: float = 1.0
        train_stride: int = Field(5, ge=1)
        test_stride: int = Field(10, ge=1)


    class Seq2SeqDataset(Dataset):
        """Duck-typed on cnr/stim: accepts either 2D numpy (uniform length)
        or object arrays / lists of per-cell arrays (variable length)."""

        def __init__(self, cnr, stim, history_len, future_len, stride=5):
            self.samples = []
            total = history_len + future_len
            for i in range(len(cnr)):
                cnr_i = cnr[i]
                stim_i = stim[i]
                T = len(cnr_i)
                t = 0
                while t + total <= T:
                    enc_cnr = cnr_i[t : t + history_len]
                    enc_stim = stim_i[:, t : t + history_len]
                    dec_stim = stim_i[:, t + history_len : t + total]
                    full_window = cnr_i[t : t + total]
                    dec_target = np.diff(full_window)[
                        history_len - 1 : history_len - 1 + future_len
                    ]
                    enc_in = np.concatenate(
                        [enc_cnr[:, np.newaxis], enc_stim.T], axis=-1
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


    def _tf_schedule_linear(tcfg, total_epochs):
        start, end = tcfg.tf_ratio_start, tcfg.tf_ratio_end
        frac, hold = tcfg.tf_anneal_frac, tcfg.tf_hold_frac

        def schedule(epoch):
            hold_epochs = int(total_epochs * hold)
            anneal_epochs = max(int(total_epochs * frac) - 1, 1)
            if epoch < hold_epochs:
                p = 0.0
            else:
                p = min((epoch - hold_epochs) / anneal_epochs, 1.0)
            return start + (end - start) * p

        return schedule


    def _run_epoch(
        model, loader, device, optimizer, grad_clip, tf_ratio, is_train
    ):
        if is_train:
            model.train()
        else:
            model.eval()
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
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )
                    optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))


    class Seq2ScalarMDN(nn.Module):
        """Autoregressive sliding-window encoder + MLP trunk + MDN head.

        Config attached as class attribute so external callers can do
        `Seq2ScalarMDN.Config(...)`. `fit` is the self-contained trainer
        (named `fit` to avoid shadowing nn.Module.train, which is reserved
        for setting train/eval mode).
        """

        Config = ModelConfig
        TrainingConfigCls = TrainingConfig

        def __init__(self, cfg=None, **kwargs):
            super().__init__()
            if cfg is None:
                cfg = kwargs
            if isinstance(cfg, dict):
                cfg = ModelConfig.model_validate(cfg)
            self.cfg = cfg
            self.n_gaussians = cfg.n_gaussians
            self.encoder = LSTMEncoder(
                cfg.encoder_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout
            )
            layers = [
                nn.Linear(cfg.hidden_dim + cfg.stim_dim, cfg.mlp_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            for _ in range(cfg.n_mlp_layers - 1):
                layers += [
                    nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            self.trunk = nn.Sequential(*layers)
            self.head = MDNHead(cfg.mlp_hidden, cfg.n_gaussians)

        def _step(self, h_top, stim_i):
            feats = self.trunk(torch.cat([h_top, stim_i], dim=-1))
            return self.head(feats)

        def forward(self, encoder_input, future_stim, targets=None, tf_ratio=0.0):
            F = future_stim.shape[1]
            current_window = encoder_input
            pis, mus, sigmas = [], [], []
            for i in range(F):
                h, _ = self.encoder(current_window)
                pi, mu, sigma = self._step(h[-1], future_stim[:, i, :])
                pis.append(pi)
                mus.append(mu)
                sigmas.append(sigma)
                if i < F - 1:
                    last_abs = current_window[:, -1, 0:1]
                    use_teacher = (
                        targets is not None and torch.rand(1).item() < tf_ratio
                    )
                    if use_teacher:
                        delta = targets[:, i : i + 1]
                    else:
                        delta = (pi * mu).sum(dim=-1, keepdim=True)
                    next_cnr_abs = last_abs + delta
                    next_input = torch.cat(
                        [next_cnr_abs, future_stim[:, i, :]], dim=-1
                    ).unsqueeze(1)
                    current_window = torch.cat(
                        [current_window[:, 1:, :], next_input], dim=1
                    )
            return (
                torch.stack(pis, dim=1),
                torch.stack(mus, dim=1),
                torch.stack(sigmas, dim=1),
            )

        def point_pred(self, pi, mu):
            return (pi * mu).sum(dim=-1)

        def pred_std(self, pi, mu, sigma):
            mean = (pi * mu).sum(dim=-1, keepdim=True)
            var = (pi * (sigma**2 + (mu - mean) ** 2)).sum(dim=-1)
            return torch.sqrt(var.clamp(min=1e-12))

        def loss(self, preds, target):
            pi, mu, sigma = preds
            return mdn_nll(pi, mu, sigma, target)

        @staticmethod
        def fit(dataset, ctx):
            """Self-contained training.

            dataset: {"train": (cnr, stim), "val": (cnr, stim)} with numpy arrays
                     shape (n_traj, T) and (n_traj, n_stim, T).
            ctx: TrainContext with device, model_config, training_config,
                 optional tracker and progress_cb.
            Returns (trained_model, history_dict).
            """
            mcfg = ctx.model_config
            tcfg = ctx.training_config

            cnr_tr, stim_tr = dataset["train"]
            cnr_va, stim_va = dataset["val"]

            train_ds = Seq2SeqDataset(
                cnr_tr,
                stim_tr,
                mcfg.history_len,
                mcfg.future_len,
                stride=tcfg.train_stride,
            )
            val_ds = Seq2SeqDataset(
                cnr_va,
                stim_va,
                mcfg.history_len,
                mcfg.future_len,
                stride=tcfg.train_stride,
            )
            train_loader = DataLoader(
                train_ds, batch_size=tcfg.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=tcfg.batch_size, shuffle=False
            )

            model = Seq2ScalarMDN(mcfg).to(ctx.device)
            opt = torch.optim.Adam(
                model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
            )
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=10, factor=0.5
            )
            tf_fn = _tf_schedule_linear(tcfg, tcfg.epochs)

            hist = {"train_loss": [], "val_loss": [], "tf_ratio": []}
            ckpt_fd, ckpt = tempfile.mkstemp(suffix=".pt")
            os.close(ckpt_fd)

            best, wait = float("inf"), 0
            for ep in range(tcfg.epochs):
                tf_r = tf_fn(ep)
                t = _run_epoch(
                    model,
                    train_loader,
                    ctx.device,
                    opt,
                    tcfg.grad_clip,
                    tf_r,
                    True,
                )
                v = _run_epoch(
                    model, val_loader, ctx.device, opt, tcfg.grad_clip, 0.0, False
                )
                hist["train_loss"].append(t)
                hist["val_loss"].append(v)
                hist["tf_ratio"].append(tf_r)
                sched.step(v)
                if v < best:
                    best, wait = v, 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    wait += 1
                    if wait >= tcfg.patience:
                        print(f"Early stopping at epoch {ep}")
                        break
                if ep % ctx.print_every == 0:
                    print(f"Epoch {ep:3d} | tf={tf_r:.2f} T:{t:.5f} V:{v:.5f}")
                if ctx.progress_cb is not None:
                    ctx.progress_cb(
                        ep, tcfg.epochs, {"train": t, "val": v, "tf": tf_r}
                    )
                if ctx.tracker is not None:
                    _cur = {k: w.clone() for k, w in model.state_dict().items()}
                    model.load_state_dict(torch.load(ckpt, weights_only=True))
                    ctx.tracker.checkpoint(
                        model, training_results={"history": hist}
                    )
                    model.load_state_dict(_cur)

            model.load_state_dict(torch.load(ckpt, weights_only=True))
            os.remove(ckpt)
            return model, hist


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from hastyplot import qplot

    from experiment import load_experiment
    from utils import (
        get_device,
        get_username,
        running_on_cluster,
        results_write_path,
        results_read_sources,
        parse_bool,
        scan_experiment_dirs,
    )
    from experiments.seq2seq_data import (
        load as load_dataset,
        AVAILABLE_DATASETS,
        STIM_COLS,
    )
    # re-bind so downstream cells can see it
    stim_col_names = list(STIM_COLS)
    from experiments.scaffold import (
        form_from_configs,
        resolve_configs,
        run_experiment,
        save_bundle,
    )

    device = get_device()
    n_stim = len(STIM_COLS)
    hostname = get_username()
    is_cluster = running_on_cluster()
    results_base = results_write_path()
    repo_root = Path(__file__).resolve().parent.parent
    return (
        AVAILABLE_DATASETS,
        device,
        form_from_configs,
        hostname,
        is_cluster,
        load_dataset,
        load_experiment,
        mo,
        n_stim,
        parse_bool,
        pl,
        qplot,
        repo_root,
        resolve_configs,
        results_base,
        results_read_sources,
        run_experiment,
        save_bundle,
        scan_experiment_dirs,
        stim_col_names,
    )


@app.cell
def _(mo, parse_bool):
    MODE = mo.cli_args().get("mode", "train")
    IS_HEADLESS = "name" in mo.cli_args()
    EXPERIMENT_NAME = mo.cli_args().get("name", "lstm_seq2scal_mdn")
    DRY_RUN = parse_bool(mo.cli_args().get("dry_run", True))

    if MODE not in ("train", "load"):
        raise ValueError(f"--mode must be 'train' or 'load', got {MODE!r}")

    mo.md(
        f"**Mode:** `{MODE}` · **Headless:** `{IS_HEADLESS}` · **Experiment:** `{EXPERIMENT_NAME}`  · **Dry run :** `{DRY_RUN}` "
    )
    return DRY_RUN, EXPERIMENT_NAME, IS_HEADLESS, MODE


@app.cell
def _(IS_HEADLESS, MODE, mo, repo_root, results_read_sources):
    if MODE == "load" and not IS_HEADLESS:
        _sources = results_read_sources(repo_root)
        source_selector = mo.ui.dropdown(
            options=list(_sources.keys()), value="Local", label="Results source",
        )
    else:
        source_selector = None

    source_selector if source_selector is not None else mo.md("")
    return (source_selector,)


@app.cell
def _(
    IS_HEADLESS,
    MODE,
    mo,
    repo_root,
    results_read_sources,
    scan_experiment_dirs,
    source_selector,
):
    if MODE == "load" and not IS_HEADLESS and source_selector is not None:
        _src_name = source_selector.value
        _src_root = Path(results_read_sources(repo_root)[_src_name])
        _choices = scan_experiment_dirs(_src_root)
        if _choices:
            experiment_picker = mo.ui.dropdown(
                options=_choices, value=_choices[0], label="Experiment run",
            )
            load_button = mo.ui.button(
                value=0, on_click=lambda n: n + 1, label="Load",
            )
            source_root = _src_root
            _picker_ui = mo.vstack([experiment_picker, load_button])
        else:
            experiment_picker = None
            load_button = None
            source_root = None
            _picker_ui = mo.md(f"No experiments under `{_src_root}`.")
    else:
        experiment_picker = None
        load_button = None
        source_root = None
        _picker_ui = mo.md("")

    _picker_ui
    return experiment_picker, load_button, source_root


@app.cell
def _(experiment_picker, source_root):
    if experiment_picker is not None and source_root is not None:
        experiment_path = source_root / experiment_picker.value
    else:
        experiment_path = None
    return (experiment_path,)


@app.cell
def _(AVAILABLE_DATASETS, IS_HEADLESS, MODE, form_from_configs, mo):
    if MODE == "train" and not IS_HEADLESS:
        form = form_from_configs(
            mo,
            {"m": ModelConfig, "t": TrainingConfig},
            skip={"m": {"encoder_dim", "stim_dim", "variant"}},
            radio_choices={"m": {"data_source": AVAILABLE_DATASETS}},
        )
    else:
        form = None

    form if form is not None else mo.md("")
    return (form,)


@app.cell(hide_code=True)
def _(
    DRY_RUN,
    EXPERIMENT_NAME,
    IS_HEADLESS,
    MODE,
    form,
    mo,
    n_stim,
    resolve_configs,
):
    _cfgs, data_source, _ctx_display = resolve_configs(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        form=form,
        config_classes={"m": ModelConfig, "t": TrainingConfig},
        always={"m": {"encoder_dim": 1 + n_stim, "stim_dim": n_stim}},
        experiment_name=EXPERIMENT_NAME,
        dry_run=DRY_RUN,
    )
    model_config, training_config = _cfgs["m"], _cfgs["t"]

    _ctx_display
    return data_source, model_config, training_config


@app.cell
def _(DRY_RUN, data_source, load_dataset, mo):
    if data_source == "real":
        cnr_all, stim_all, conditions_all = load_dataset("real")
    else:
        cnr_all, stim_all, conditions_all = load_dataset(data_source)

    n_traj = len(cnr_all)
    _lens = np.array([len(cnr_all[i]) for i in range(n_traj)])
    _traj_len_min, _traj_len_max = int(_lens.min()), int(_lens.max())

    _traj_ids = np.arange(n_traj)
    _tr_ids, _te_ids = train_test_split(_traj_ids, test_size=0.2, random_state=42)
    _tr_ids, _va_ids = train_test_split(_tr_ids, test_size=0.125, random_state=42)

    if DRY_RUN:
        _tr_ids = _tr_ids[: min(len(_tr_ids), 800)]
        _va_ids = _va_ids[: min(len(_va_ids), 200)]
        _te_ids = _te_ids[: min(len(_te_ids), 200)]

    cnr_tr, stim_tr = cnr_all[_tr_ids], stim_all[_tr_ids]
    cnr_va, stim_va = cnr_all[_va_ids], stim_all[_va_ids]
    cnr_te, stim_te = cnr_all[_te_ids], stim_all[_te_ids]

    _len_str = (
        f"{_traj_len_min}"
        if _traj_len_min == _traj_len_max
        else f"{_traj_len_min}-{_traj_len_max}"
    )
    mo.md(f"""
    **Data:** {n_traj} trajectories x {_len_str} timepoints (`{data_source}`)

    Splits: train={len(_tr_ids)} | val={len(_va_ids)} | test={len(_te_ids)} * dry_run={DRY_RUN}
    """)
    return cnr_te, cnr_tr, cnr_va, stim_te, stim_tr, stim_va


@app.cell(hide_code=True)
def _(IS_HEADLESS, MODE, mo):
    if MODE == "train" and not IS_HEADLESS:
        train_button = mo.ui.run_button(label="Start training")
    else:
        train_button = None

    train_button if train_button is not None else mo.md("")
    return (train_button,)


@app.cell
def _(
    EXPERIMENT_NAME,
    IS_HEADLESS,
    MODE,
    cnr_tr,
    cnr_va,
    device,
    experiment_path,
    load_button,
    mo,
    model_config,
    results_base,
    run_experiment,
    stim_tr,
    stim_va,
    train_button,
    training_config,
):
    artifacts = run_experiment(
        mo=mo,
        mode=MODE,
        is_headless=IS_HEADLESS,
        model_cls=Seq2ScalarMDN,
        model_config_cls=ModelConfig,
        dataset={"train": (cnr_tr, stim_tr), "val": (cnr_va, stim_va)},
        model_config=model_config,
        training_config=training_config,
        device=device,
        experiment_name=EXPERIMENT_NAME,
        results_base=results_base,
        experiment_path=experiment_path,
        load_button=load_button,
        train_button=train_button,
    )

    model = artifacts.model
    history = artifacts.history
    tracker = artifacts.tracker
    model_config_used = artifacts.model_config

    mo.md(
        f"**Run ready** · {type(model).__name__} · {sum(p.numel() for p in model.parameters()):,} params"
    )
    return artifacts, history, model, model_config_used, tracker


@app.cell
def _(model):
    model
    return


@app.cell(hide_code=True)
def _(MODE, experiment_path, load_experiment, mo):
    import json as _json

    if MODE == "load" and experiment_path is not None:
        _exp_path_ld = experiment_path
        _bundle_ld = load_experiment(str(_exp_path_ld))

        _stats = _bundle_ld.training_results.get("stats", {})
        _elapsed = _bundle_ld.training_results.get("train_elapsed_s")

        _rows = [
            ("name", _bundle_ld.name),
            ("timestamp", _bundle_ld.timestamp),
            ("model_type", _bundle_ld.model_type),
            ("experiment_path", str(_exp_path_ld)),
        ]
        if _elapsed is not None:
            _rows.append(
                (
                    "train_elapsed_s",
                    f"{float(_elapsed):.1f} ({float(_elapsed) / 60:.1f} min)",
                )
            )
        for _k, _v in _stats.items():
            _rows.append((_k, _v))
        for _k, _v in (_bundle_ld.metrics or {}).items():
            _rows.append((f"metric.{_k}", _v))

        _summary_md = (
            "## Loaded run summary\n\n| field | value |\n|---|---|\n"
            + "\n".join(f"| `{_k}` | {_v} |" for _k, _v in _rows)
        )

        _cfg_md = (
            "## Saved configs\n\n"
            f"**model_config**\n```json\n{_json.dumps(_bundle_ld.model_config, indent=2, default=str)}\n```\n\n"
            f"**training_config**\n```json\n{_json.dumps(_bundle_ld.training_config, indent=2, default=str)}\n```"
        )

        _slurm_path = _exp_path_ld / "slurm.log"
        if _slurm_path.exists():
            _slurm_txt = _slurm_path.read_text(errors="replace")
            _slurm_md = f"```\n{_slurm_txt}\n```"
            _slurm_section = mo.accordion(
                {
                    f"slurm.log ({_slurm_path.stat().st_size:,} bytes)": mo.md(
                        _slurm_md
                    ),
                }
            )
        else:
            _slurm_section = mo.md(f"_slurm.log not found at `{_slurm_path}`_")

        run_summary = mo.vstack(
            [
                mo.md(_summary_md),
                mo.md(_cfg_md),
                mo.md("## slurm.log"),
                _slurm_section,
            ]
        )
    else:
        run_summary = mo.md("")

    run_summary
    return


@app.cell
def _(
    MODE,
    cnr_te,
    data_source,
    load_dataset,
    mo,
    model_config_used,
    stim_te,
    training_config,
):
    H = model_config_used.history_len
    F_ = model_config_used.future_len

    if MODE == "load" and model_config_used.data_source != data_source:
        _ds_for_test = model_config_used.data_source
        _cnr_ld, _stim_ld, _ = load_dataset(_ds_for_test)
        _ids = np.arange(len(_cnr_ld))
        _tr, _te = train_test_split(_ids, test_size=0.2, random_state=42)
        cnr_te_used, stim_te_used = _cnr_ld[_te], _stim_ld[_te]
    else:
        cnr_te_used, stim_te_used = cnr_te, stim_te

    _test_stride = (
        training_config.test_stride
        if training_config is not None and hasattr(training_config, "test_stride")
        else 10
    )
    test_ds = Seq2SeqDataset(cnr_te_used, stim_te_used, H, F_, stride=_test_stride)

    mo.md(
        f"Test windows: {len(test_ds)} (H={H}, F={F_}, source=`{model_config_used.data_source}`)"
    )
    return F_, H, cnr_te_used, stim_te_used, test_ds


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
def _(device, model, test_ds):
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
    return test_act, test_mu, test_pi, test_point, test_sigma, test_std


@app.cell
def _(F_, pl, test_act, test_point):
    """Per-step residual histograms — precomputed bins to avoid altair max_rows."""

    import altair as _alt

    _bins = 50
    _resid_flat = (test_act - test_point).flatten(order="F")
    _step_flat = np.repeat(np.arange(1, F_ + 1), test_act.shape[0])

    _lo, _hi = float(_resid_flat.min()), float(_resid_flat.max())
    _edges = np.linspace(_lo, _hi, _bins + 1)

    _rows_h = []
    for _s in range(1, F_ + 1):
        _counts, _ = np.histogram(_resid_flat[_step_flat == _s], bins=_edges)
        for _i, _n in enumerate(_counts):
            _rows_h.append(
                {
                    "step": _s,
                    "bin_start": float(_edges[_i]),
                    "bin_end": float(_edges[_i + 1]),
                    "count": int(_n),
                }
            )
    resid_hist_df = pl.DataFrame(_rows_h)

    fig_residuals = (
        _alt.Chart(resid_hist_df)
        .mark_bar()
        .encode(
            x=_alt.X("bin_start:Q", bin="binned", title="residual"),
            x2="bin_end:Q",
            y=_alt.Y("count:Q", title="count"),
        )
        .properties(width=160, height=180)
        .facet(column=_alt.Column("step:N", title=None))
        .properties(title="Residual distribution per forecast step")
    )
    fig_residuals
    return (fig_residuals,)


@app.cell
def _(F_, pl, test_std):
    """Predicted std per forecast step — precomputed quantiles to avoid altair max_rows."""

    import altair as _alt

    _std_stats_df = (
        pl.DataFrame(
            {
                "step": np.repeat(np.arange(1, F_ + 1), test_std.shape[0]),
                "pred_std": test_std.flatten(order="F"),
            }
        )
        .group_by("step")
        .agg(
            [
                pl.col("pred_std").quantile(0.05).alias("q05"),
                pl.col("pred_std").quantile(0.25).alias("q25"),
                pl.col("pred_std").quantile(0.50).alias("q50"),
                pl.col("pred_std").quantile(0.75).alias("q75"),
                pl.col("pred_std").quantile(0.95).alias("q95"),
            ]
        )
        .sort("step")
    )

    _whisk = (
        _alt.Chart(_std_stats_df)
        .mark_rule()
        .encode(
            x=_alt.X("step:O", title="forecast step"),
            y=_alt.Y("q05:Q", title="pred_std"),
            y2="q95:Q",
        )
    )
    _box = (
        _alt.Chart(_std_stats_df)
        .mark_bar(size=22, color="#4C78A8")
        .encode(x="step:O", y="q25:Q", y2="q75:Q")
    )
    _median = (
        _alt.Chart(_std_stats_df)
        .mark_tick(color="white", size=22, thickness=2)
        .encode(x="step:O", y="q50:Q")
    )
    fig_std = (_whisk + _box + _median).properties(
        title="Predicted std by forecast step (5/25/50/75/95 quantiles)",
        height=300,
        width=380,
    )
    fig_std
    return (fig_std,)


@app.cell
def _(F_, H, device, mo, model, pl, qplot, test_ds):
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
def _(F_, pl, qplot, test_act, test_point, test_std):
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
def _(mo):
    mo.md("""
    # Extended evaluation

    Distributional, calibration, feature-importance, and stratified
    diagnostics. Each section below has a quick summary of what it
    measures, why it matters, and how to interpret the plot.
    """)
    return


@app.cell
def _(F_, mo, pl, test_act, test_mu, test_pi, test_sigma):
    from scipy.special import logsumexp as _logsumexp
    from scipy.stats import norm as _norm

    _y = test_act[..., None]
    _log_pi = np.log(test_pi + 1e-12)
    _log_gauss = (
        -0.5 * np.log(2 * np.pi)
        - np.log(test_sigma)
        - 0.5 * ((_y - test_mu) / test_sigma) ** 2
    )
    _nll_per = -_logsumexp(_log_pi + _log_gauss, axis=-1)

    _z = (_y - test_mu) / test_sigma
    _pit_per = (test_pi * _norm.cdf(_z)).sum(axis=-1)


    def _A(u, s):
        z_ = u / (s + 1e-12)
        return u * (2 * _norm.cdf(z_) - 1) + 2 * s * _norm.pdf(z_)


    _term1 = (test_pi * _A(_y - test_mu, test_sigma)).sum(axis=-1)
    _mu_j = test_mu[..., :, None]
    _mu_k = test_mu[..., None, :]
    _sig_j = test_sigma[..., :, None]
    _sig_k = test_sigma[..., None, :]
    _pi_jk = test_pi[..., :, None] * test_pi[..., None, :]
    _term2 = 0.5 * (
        _pi_jk * _A(_mu_j - _mu_k, np.sqrt(_sig_j**2 + _sig_k**2))
    ).sum(axis=(-1, -2))
    _crps_per = _term1 - _term2

    _mean_exp = (test_pi * test_mu).sum(axis=-1, keepdims=True)
    _sigma_eff = np.sqrt(
        (test_pi * (test_sigma**2 + (test_mu - _mean_exp) ** 2)).sum(axis=-1)
    )
    _resid = test_act - _mean_exp.squeeze(-1)

    _N = test_pi.shape[0]
    mixture_metrics_df = pl.DataFrame(
        {
            "window": np.repeat(np.arange(_N), F_),
            "step": np.tile(np.arange(1, F_ + 1), _N),
            "nll": _nll_per.flatten(),
            "crps": _crps_per.flatten(),
            "pit": _pit_per.flatten(),
            "sigma_eff": _sigma_eff.flatten(),
            "resid": _resid.flatten(),
        }
    )

    _per_step = (
        mixture_metrics_df.group_by("step")
        .agg(
            [
                pl.col("nll").mean().alias("nll"),
                pl.col("crps").mean().alias("crps"),
                pl.col("sigma_eff").mean().alias("sigma"),
            ]
        )
        .sort("step")
    )

    mo.vstack(
        [
            mo.md(
                """
            ## Mixture metrics (exact NLL + CRPS + PIT)

            Three per-(window, step) scores, one dataframe — downstream
            cells (PIT histogram, stratification, sharpness) all index into this.

            **Mixture NLL** — `-log Σ_k π_k · N(y; μ_k, σ_k)`. The model's own
            training objective, evaluated on held-out data. The existing
            `test_nll_gaussian_proxy` in the summary collapsed the mixture
            into a single π-weighted (mean, variance) Gaussian before scoring
            — it discards the actual multi-modal shape. This column is the
            exact mixture log-likelihood. Lower is better. Can be **negative**
            because Gaussian densities can exceed 1 (narrow σ), making log-
            density positive. Not comparable across datasets with different
            target scales.

            **CRPS** (Continuous Ranked Probability Score) — a proper scoring
            rule that generalizes MAE to distributions: `∫ (F(z) − 1{z ≥ y})² dz`.
            Closed form for Gaussian mixtures (Grimit et al. 2006) using
            `A(u, σ) = u·(2Φ(u/σ) − 1) + 2σ·φ(u/σ)`. Reported in **ΔCNR units**,
            so a mean CRPS of 0.05 means "the predicted distribution is on
            average 0.05 ΔCNR away from the observed target". Penalizes both
            bad location and bad spread. More stable than NLL for tail
            outliers (CRPS grows linearly; NLL grows quadratically).

            **PIT** — `F_mix(y) = Σ_k π_k · Φ((y − μ_k) / σ_k)`, the predicted
            mixture CDF evaluated at the observed target. By the probability
            integral transform theorem, if the model's predicted distribution
            matches the truth, these values are **uniform on [0, 1]**. Stored
            here; histogrammed in the next section.

            **Why all three**: NLL is good for ranking models on the same
            dataset but uninterpretable in absolute units. CRPS is
            interpretable but less tail-sensitive. PIT is calibration-only.
            Together they catch failure modes any single metric misses.

            **Per-step table below**: later forecast steps are expected to be
            harder under autoregressive rollout (errors compound). A steep
            gap between step 1 and step F indicates the model is drifting
            through its own predicted history.
            """
            ),
            _per_step,
        ]
    )
    return (mixture_metrics_df,)


@app.cell
def _(F_, mixture_metrics_df, mo, pl):
    import altair as _alt

    _bins = 20
    _edges = np.linspace(0.0, 1.0, _bins + 1)
    _rows = []
    for _s in range(1, F_ + 1):
        _vals = mixture_metrics_df.filter(pl.col("step") == _s)["pit"].to_numpy()
        _counts, _ = np.histogram(_vals, bins=_edges)
        _n = max(len(_vals), 1)
        for _i, _c in enumerate(_counts):
            _rows.append(
                {
                    "step": _s,
                    "bin_start": float(_edges[_i]),
                    "bin_end": float(_edges[_i + 1]),
                    "frac": float(_c) / _n,
                }
            )
    pit_df = pl.DataFrame(_rows)

    pit_df = pit_df.with_columns(pl.lit(1.0 / _bins).alias("ref"))
    _hist = (
        _alt.Chart()
        .mark_bar()
        .encode(
            x=_alt.X("bin_start:Q", bin="binned", title="PIT"),
            x2="bin_end:Q",
            y=_alt.Y("frac:Q", title="fraction"),
        )
    )
    _ref = (
        _alt.Chart()
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(
            y="ref:Q",
        )
    )
    fig_pit = (
        _alt.layer(_hist, _ref, data=pit_df)
        .properties(width=160, height=180)
        .facet(column=_alt.Column("step:N", title=None))
        .properties(title="PIT histograms per forecast step")
    )

    mo.vstack(
        [
            mo.md(
                """
            ## PIT (Probability Integral Transform) histograms

            **The question this answers**: "Is my predicted *distribution*
            right?" — not just "is the mean right", but the whole shape.

            **Ground-up intuition**. Suppose I predict `N(0, 1)` for every
            target, and the truth is also `N(0, 1)`. Take a test sample `y`,
            apply my predicted CDF to it: `u = F_pred(y) = Φ(y)`. Since `y`
            was drawn from the same distribution I'm using to transform it,
            `u` is **uniform on [0, 1]**. This is the probability integral
            transform theorem: applying the true CDF to a sample from the
            true distribution gives uniform output.

            Now swap: truth is `N(0, 2)` (wider), my prediction is `N(0, 1)`
            (too narrow). When `y` is drawn from the wider truth, it often
            lands far from 0. `Φ_pred(y) = Φ(y / 1)` is near 0 for very
            negative `y`, near 1 for very positive `y`. Result: **PIT values
            pile up at 0 and 1** — a U-shape. That U-shape *is* the signature
            of σ being too small (underdispersed).

            Flip it: truth is `N(0, 0.5)`, prediction is `N(0, 1)`. Targets
            stay near 0, predicted CDF at small `y` stays near 0.5. **PIT
            values clump at 0.5** — inverted-U, the signature of σ too large
            (overdispersed).

            **For a Gaussian mixture** the same argument holds with
            `F_mix(y) = Σ_k π_k · Φ((y − μ_k) / σ_k)` — probability-weighted
            sum of component CDFs. If the model fits perfectly, PIT values
            are uniform.

            **How to read the histogram** (20 bins, red line = uniform
            reference at `1/20`):

            - **Flat** → calibrated. Predicted quantiles match empirical
              frequencies.
            - **U-shape** (mass at 0 and 1) → **underdispersed**. Targets
              land in the predicted tails more often than the model
              expects. σ is systematically too small.
            - **Inverted-U / hump at 0.5** → **overdispersed**. Targets rarely
              reach predicted tails. σ is systematically too large (wasting
              uncertainty).
            - **Monotonic slope** (e.g. high at 0, low at 1) → **biased point
              prediction**. The predicted distribution is centered above the
              truth (observed `y` tends to fall in the lower quantiles of
              the predicted distribution). Slope the other way → centered
              below truth.
            - **Stair-steps / gaps** → numerical artifact, usually from
              extremely narrow σ collapsing many targets into the same bin.

            **vs k·σ coverage**: the existing coverage plot uses symmetric
            `|residual| ≤ k·σ` bands — it can miss **asymmetric**
            miscalibration. A model biased high with right-sized variance
            can pass symmetric coverage while producing a visibly skewed
            PIT histogram. PIT is quantile-free and location-aware; they're
            complementary, not redundant.

            **Per-step facets**: step 1 uses true-history encoder input,
            later steps roll autoregressively through predicted values.
            Calibration typically degrades with step — a flat step-1
            histogram turning U-shaped by step F means the rollout
            compounds error faster than the model's σ accounts for.

            **Scalar summary**: the `test_pit_ks` metric in the summary
            table is the Kolmogorov-Smirnov distance from the uniform
            distribution (max deviation of the empirical CDF from the
            diagonal). 0 = perfectly uniform.
            """
            ),
            fig_pit,
        ]
    )
    return (fig_pit,)


@app.cell
def _(mo, pl, test_pi):
    import altair as _alt

    _THRESH = 0.05
    _active = (test_pi > _THRESH).sum(axis=-1)
    _entropy = -(test_pi * np.log(test_pi + 1e-12)).sum(axis=-1)
    _K = test_pi.shape[-1]

    _active_flat = _active.flatten()
    _active_counts = np.bincount(_active_flat, minlength=_K + 1)
    active_df = pl.DataFrame(
        {
            "active_count": np.arange(len(_active_counts)),
            "count": _active_counts.astype(np.int64),
        }
    )

    _ent_flat = _entropy.flatten()
    _ent_bins = 30
    _ent_edges = np.linspace(0.0, float(np.log(_K)) + 1e-9, _ent_bins + 1)
    _ent_counts, _ = np.histogram(_ent_flat, bins=_ent_edges)
    entropy_df = pl.DataFrame(
        {
            "bin_start": _ent_edges[:-1],
            "bin_end": _ent_edges[1:],
            "count": _ent_counts.astype(np.int64),
        }
    )

    _fig_a = (
        _alt.Chart(active_df)
        .mark_bar()
        .encode(
            x=_alt.X("active_count:O", title=f"active components (π > {_THRESH})"),
            y=_alt.Y("count:Q"),
        )
        .properties(
            title=f"Active-component count (K={_K})", width=260, height=200
        )
    )
    _fig_e = (
        _alt.Chart(entropy_df)
        .mark_bar()
        .encode(
            x=_alt.X("bin_start:Q", bin="binned", title="π entropy"),
            x2="bin_end:Q",
            y=_alt.Y("count:Q"),
        )
        .properties(
            title=f"π entropy (max={np.log(_K):.3f})",
            width=260,
            height=200,
        )
    )
    fig_modes = _alt.hconcat(_fig_a, _fig_e)

    _mean_active = float(_active.mean())
    _frac_collapsed = float((_active == 1).mean())

    mo.vstack(
        [
            mo.md(
                f"""
            ## Mode usage — is the MDN actually using K > 1?

            **Background**. An MDN with K=**{_K}** components can in principle
            learn per-input distributions with up to {_K} peaks. But MDN
            training is notoriously prone to **mode collapse**: the model
            learns to put essentially all π mass on one component, and the
            other K−1 components become dead weight (their μ, σ, and
            gradients decay into noise). A mode-collapsed MDN has the same
            expressive power as a single Gaussian head but costs K× more
            parameters to train and store. You'd never know from the loss
            curve — NLL will still go down, just not as far as it could.

            **Two diagnostics on the mixture weights π** (shape `(N, F, K)`):

            **1. Active-component count** (left plot). For each prediction,
            count how many of the K weights exceed a threshold (here 0.05 —
            i.e. components carrying at least 5% mass). If this count is
            always 1, the MDN is effectively single-Gaussian per prediction.
            If it spreads across 1…K, different inputs trigger different
            numbers of components — good use of capacity.

            **2. Entropy of π** (right plot). `H(π) = -Σ_k π_k · log π_k`.
            Zero when one π is 1 and the rest are 0 (concentrated — one
            component carries everything). Maximum `log K ≈ {np.log(_K):.3f}`
            when π is uniform (every component equally weighted).
            Intermediate values mean partial sharing. Entropy is a
            smoother measure than "count active" — it registers the
            *shape* of the weight distribution, not just a threshold.

            **Ideal outcome**: a **broad** entropy distribution. Easy /
            unambiguous windows produce low-entropy predictions (one
            component is clearly best); ambiguous windows produce
            higher-entropy predictions (real uncertainty between several
            modes). The MDN is *adaptively* using its capacity.

            **How to read**:
            - **Active histogram spike at 1** → **mode collapse**. K > 1 is
              wasted; train K=1 next and compare — if NLL barely moves,
              save the compute.
            - **Active histogram uniform at K** → mixture is fully used,
              but you may want to check that the *dominant* component
              isn't still carrying all the weight (active > 1 could mean
              π = [0.9, 0.05, 0.05], which isn't really multimodal).
            - **Entropy mass near 0** → single-component predictions
              dominate.
            - **Entropy mass near log K** (uniform π) → predictions can't
              commit to any component. Unusual; may indicate underfitting
              or numerical issues in the π softmax.
            - **Bimodal entropy** → the MDN splits predictions into "I'm
              confident, use one mode" and "I'm uncertain, mix components".
              Often the best outcome.

            **Current numbers**: mean active = **{_mean_active:.2f}** ·
            fraction single-mode predictions = **{_frac_collapsed:.1%}**.
            """
            ),
            fig_modes,
        ]
    )
    return (fig_modes,)


@app.cell
def _(device, mo, model, n_stim, pl, qplot, stim_col_names, test_ds):
    from scipy.special import logsumexp as _logsumexp
    from scipy.stats import norm as _norm


    def _forward_all(enc_ch, stim_ch):
        _pi_l, _mu_l, _sig_l, _tgt_l = [], [], [], []
        model.eval()
        with torch.no_grad():
            for _eb, _sb, _tb in DataLoader(test_ds, batch_size=512):
                if enc_ch is not None or stim_ch is not None:
                    _eb = _eb.clone()
                    _sb = _sb.clone()
                    if enc_ch is not None:
                        _eb[..., enc_ch] = 0.0
                    if stim_ch is not None:
                        _sb[..., stim_ch] = 0.0
                _pi, _mu, _sig = model(_eb.to(device), _sb.to(device))
                _pi_l.append(_pi.cpu().numpy())
                _mu_l.append(_mu.cpu().numpy())
                _sig_l.append(_sig.cpu().numpy())
                _tgt_l.append(_tb.numpy())
        return (
            np.concatenate(_pi_l),
            np.concatenate(_mu_l),
            np.concatenate(_sig_l),
            np.concatenate(_tgt_l),
        )


    def _agg_metrics(pi, mu, sig, y):
        y_ = y[..., None]
        log_g = (
            -0.5 * np.log(2 * np.pi) - np.log(sig) - 0.5 * ((y_ - mu) / sig) ** 2
        )
        nll = float(-_logsumexp(np.log(pi + 1e-12) + log_g, axis=-1).mean())

        def _A(u, s):
            z = u / (s + 1e-12)
            return u * (2 * _norm.cdf(z) - 1) + 2 * s * _norm.pdf(z)

        t1 = (pi * _A(y_ - mu, sig)).sum(axis=-1)
        mu_j, mu_k = mu[..., :, None], mu[..., None, :]
        sg_j, sg_k = sig[..., :, None], sig[..., None, :]
        pi_jk = pi[..., :, None] * pi[..., None, :]
        t2 = 0.5 * (pi_jk * _A(mu_j - mu_k, np.sqrt(sg_j**2 + sg_k**2))).sum(
            axis=(-1, -2)
        )
        crps = float((t1 - t2).mean())
        mean_exp = (pi * mu).sum(axis=-1, keepdims=True)
        sigma_eff = float(
            np.sqrt((pi * (sig**2 + (mu - mean_exp) ** 2)).sum(axis=-1)).mean()
        )
        return nll, crps, sigma_eff


    _pi0, _mu0, _sig0, _y0 = _forward_all(None, None)
    _base_nll, _base_crps, _base_sigma = _agg_metrics(_pi0, _mu0, _sig0, _y0)

    _rows = [
        {
            "channel": "baseline",
            "idx": -1,
            "nll": _base_nll,
            "crps": _base_crps,
            "sigma": _base_sigma,
            "d_nll": 0.0,
            "d_crps": 0.0,
            "sigma_ratio": 1.0,
        }
    ]
    for _k in range(n_stim):
        _pi, _mu, _sig, _y = _forward_all(enc_ch=1 + _k, stim_ch=_k)
        _nll, _crps, _sig_eff = _agg_metrics(_pi, _mu, _sig, _y)
        _name = stim_col_names[_k] if _k < len(stim_col_names) else f"stim_{_k}"
        _rows.append(
            {
                "channel": _name,
                "idx": _k,
                "nll": _nll,
                "crps": _crps,
                "sigma": _sig_eff,
                "d_nll": _nll - _base_nll,
                "d_crps": _crps - _base_crps,
                "sigma_ratio": _sig_eff / max(_base_sigma, 1e-12),
            }
        )
    ablation_df = pl.DataFrame(_rows)

    fig_ablation = qplot(
        ablation_df.filter(pl.col("idx") >= 0).sort("d_nll", descending=True),
        "channel",
        "d_nll",
        mark="bar",
        title="ΔNLL vs baseline when channel is zeroed (bigger = channel more load-bearing)",
        height=260,
    )

    mo.vstack(
        [
            mo.md(
                f"""
            ## Input feature ablation

            **The question**: which of the {n_stim} stim channels is the
            model actually using, vs quietly ignoring? And if we lose a
            channel at deployment, does the model honestly widen its
            uncertainty, or does it silently become overconfident in a
            wrong answer?

            **Why ablation (not gradient saliency)**. Gradient-based
            importance ("how sensitive is the output to a tiny nudge of
            input x?") is a **local** measure — valid around the current
            input. Ablation is **global and counterfactual**: "if this
            channel simply weren't available, how bad would the model
            actually be?" That's the question you care about when deciding
            whether to drop a sensor, and it's a better match for
            distribution-level metrics like NLL and CRPS.

            **Protocol**. For each stim channel `k ∈ [0, {n_stim})`:
            clone the test-set batches, **zero channel `k` everywhere it
            appears** (encoder history column `1 + k` AND decoder future
            column `k`), leave all other channels and the CNR history
            intact, re-run the full forward pass, and compute aggregate
            mixture NLL / CRPS / mean σ. Compare to a baseline run with
            nothing ablated.

            **Why zero, not mean or permutation**. Each choice has
            tradeoffs:
            - **Zero-imputation** (used here): mimics "sensor off /
              missing". Clean, reproducible, fast. Downside: zero may be
              out-of-distribution for channels that never take zero in
              training — can over-estimate reliance.
            - Mean-imputation: replace with the channel's training mean.
              In-distribution but carries no information. Better if your
              channels aren't zero-centered.
            - Permutation: shuffle the channel across windows. Preserves
              marginal distribution but destroys per-window correlation
              with target. Slower, noisier.

            Since the stim channels here are one-hot / pulse encodings
            roughly centered near zero, zero-imputation is "between
            pulses" and isn't wildly OOD.

            **How to read each column**:
            - **ΔNLL (nll_ablated − nll_baseline)**: the key number. Units
              are log-density, so not directly interpretable in ΔCNR, but
              **bigger = more load-bearing**.
              - **ΔNLL > 0 and large**: removing this channel genuinely
                hurts. Keep it.
              - **ΔNLL ≈ 0**: the model ignores this channel. Either it's
                uninformative, or it's redundant with another kept
                channel, or the model failed to learn to use it.
                Candidate for dropping from the input.
              - **ΔNLL < 0**: removing the channel **helps**. The model
                was being misled by it (overfit noise, spurious
                correlation in training). Investigate — may signal
                data-preprocessing issues.
            - **ΔCRPS**: same signal as ΔNLL, in ΔCNR units. Good
              sanity check — if ΔNLL and ΔCRPS disagree on rank ordering,
              something is off.
            - **σ_ratio (σ_ablated / σ_baseline)**: **honest uncertainty**
              meter. Removing information *should* widen predictions.
              - **σ_ratio > 1**: model correctly says "I'm less sure now".
                Healthy calibration under shift.
              - **σ_ratio ≈ 1** while ΔNLL > 0: the model got wronger but
                **didn't realize it was wronger**. This is overconfidence
                under distribution shift — a deployment risk.
              - **σ_ratio < 1**: model gets *more* confident when an input
                vanishes. Pathological; usually means the channel was
                contributing variance the model was fighting, not signal.

            **Baseline**: NLL = {_base_nll:.4f} · CRPS = {_base_crps:.4f} ·
            mean σ = {_base_sigma:.4f}.
            """
            ),
            ablation_df,
            fig_ablation,
        ]
    )
    return ablation_df, fig_ablation


@app.cell
def _(device, mo, model, n_stim, pl, test_ds):
    from scipy.special import logsumexp as _logsumexp
    from scipy.stats import norm as _norm


    def _fwd(mod_enc, mod_stim):
        _pi_l, _mu_l, _sig_l, _y_l = [], [], [], []
        model.eval()
        with torch.no_grad():
            for _eb, _sb, _tb in DataLoader(test_ds, batch_size=512):
                _eb = _eb.clone()
                _sb = _sb.clone()
                if mod_enc is not None:
                    mod_enc(_eb)
                if mod_stim is not None:
                    mod_stim(_sb)
                _pi, _mu, _sig = model(_eb.to(device), _sb.to(device))
                _pi_l.append(_pi.cpu().numpy())
                _mu_l.append(_mu.cpu().numpy())
                _sig_l.append(_sig.cpu().numpy())
                _y_l.append(_tb.numpy())
        return tuple(np.concatenate(x) for x in (_pi_l, _mu_l, _sig_l, _y_l))


    def _mix_metrics(pi, mu, sig, y):
        y_ = y[..., None]
        log_g = (
            -0.5 * np.log(2 * np.pi) - np.log(sig) - 0.5 * ((y_ - mu) / sig) ** 2
        )
        nll = float(-_logsumexp(np.log(pi + 1e-12) + log_g, axis=-1).mean())
        mean_exp = (pi * mu).sum(-1, keepdims=True)
        sigma_eff = float(
            np.sqrt((pi * (sig**2 + (mu - mean_exp) ** 2)).sum(-1)).mean()
        )

        def _A(u, s):
            z = u / (s + 1e-12)
            return u * (2 * _norm.cdf(z) - 1) + 2 * s * _norm.pdf(z)

        t1 = (pi * _A(y_ - mu, sig)).sum(-1)
        mu_j, mu_k = mu[..., :, None], mu[..., None, :]
        sg_j, sg_k = sig[..., :, None], sig[..., None, :]
        pi_jk = pi[..., :, None] * pi[..., None, :]
        t2 = 0.5 * (pi_jk * _A(mu_j - mu_k, np.sqrt(sg_j**2 + sg_k**2))).sum(
            axis=(-1, -2)
        )
        crps = float((t1 - t2).mean())
        return nll, crps, sigma_eff


    _pi0, _mu0, _sig0, _y0 = _fwd(None, None)
    _b_nll, _b_crps, _b_sig = _mix_metrics(_pi0, _mu0, _sig0, _y0)


    def _zero_enc_cnr(b):
        b[..., 0] = 0.0


    def _zero_enc_stim(b):
        b[..., 1:] = 0.0


    def _zero_dec_stim(b):
        b[..., :] = 0.0


    _rows = []
    _pi, _mu, _sig, _y = _fwd(_zero_enc_cnr, None)
    _nll, _crps, _se = _mix_metrics(_pi, _mu, _sig, _y)
    _rows.append(
        {
            "variant": "cnr_only_zero",
            "nll": _nll,
            "crps": _crps,
            "sigma": _se,
            "d_nll": _nll - _b_nll,
            "d_crps": _crps - _b_crps,
            "sigma_ratio": _se / max(_b_sig, 1e-12),
        }
    )

    _pi, _mu, _sig, _y = _fwd(_zero_enc_stim, _zero_dec_stim)
    _nll, _crps, _se = _mix_metrics(_pi, _mu, _sig, _y)
    _rows.append(
        {
            "variant": "all_stim_zero",
            "nll": _nll,
            "crps": _crps,
            "sigma": _se,
            "d_nll": _nll - _b_nll,
            "d_crps": _crps - _b_crps,
            "sigma_ratio": _se / max(_b_sig, 1e-12),
        }
    )

    full_ablation_df = pl.DataFrame(_rows)

    mo.vstack(
        [
            mo.md(f"""
        ## Block ablations — CNR history vs all stim at once

        Per-channel ablation above is noisy because stim channels are
        correlated; zeroing one leaves redundant signal in others. These two
        **block** ablations answer the blunt questions:

        - **cnr_only_zero**: zero the CNR column in the encoder input
          (stim channels untouched). ΔNLL near zero → model ignores
          history. Large ΔNLL → model is primarily a CNR-autoregressor.
        - **all_stim_zero**: zero every stim channel everywhere (encoder
          history columns 1..{n_stim} AND decoder future stim). ΔNLL near
          zero → stim features carry **no** information the model uses;
          MPC infeasible with this model. Large ΔNLL → stim matters.

        Baseline NLL = **{_b_nll:.4f}** · CRPS = **{_b_crps:.4f}** · σ_eff = {_b_sig:.4f}.
        """),
            full_ablation_df,
        ]
    )
    return


@app.cell
def _(device, mo, model, n_stim, pl, qplot, stim_te_used, test_ds):
    from scipy.special import logsumexp as _logsumexp

    _N_WIN = min(500, len(test_ds))
    _idx = np.linspace(0, len(test_ds) - 1, _N_WIN, dtype=int).tolist()
    _subset = Subset(test_ds, _idx)

    _stim_max_arr = np.zeros(n_stim, dtype=np.float32)
    for _s in stim_te_used:
        _stim_max_arr = np.maximum(_stim_max_arr, np.asarray(_s).max(axis=1))


    def _run_cf(cond):
        _pi_l, _mu_l, _sig_l, _y_l = [], [], [], []
        model.eval()
        with torch.no_grad():
            for _eb, _sb, _tb in DataLoader(_subset, batch_size=256):
                _sb = _sb.clone()
                if cond == "on":
                    _sb[:] = torch.tensor(_stim_max_arr).view(1, 1, n_stim)
                elif cond == "off":
                    _sb.zero_()
                _pi, _mu, _sig = model(_eb.to(device), _sb.to(device))
                _pi_l.append(_pi.cpu().numpy())
                _mu_l.append(_mu.cpu().numpy())
                _sig_l.append(_sig.cpu().numpy())
                _y_l.append(_tb.numpy())
        return tuple(np.concatenate(x) for x in (_pi_l, _mu_l, _sig_l, _y_l))


    _pi_a, _mu_a, _sig_a, _y_cf = _run_cf("actual")
    _pi_on, _mu_on, _sig_on, _ = _run_cf("on")
    _pi_off, _mu_off, _sig_off, _ = _run_cf("off")

    _pt_on = (_pi_on * _mu_on).sum(-1)
    _pt_off = (_pi_off * _mu_off).sum(-1)

    _pp_diff = np.abs(_pt_on - _pt_off)
    _pp_mean = float(_pp_diff.mean())
    _pp_per_step = _pp_diff.mean(axis=0)

    _top_on = _pi_on.argmax(-1)
    _top_off = _pi_off.argmax(-1)
    _ix_n = np.arange(_pi_on.shape[0])[:, None]
    _ix_f = np.arange(_pi_on.shape[1])[None, :]
    _top_mu_on = _mu_on[_ix_n, _ix_f, _top_on]
    _top_mu_off = _mu_off[_ix_n, _ix_f, _top_off]
    _top_mu_diff = float(np.abs(_top_mu_on - _top_mu_off).mean())

    _m = 0.5 * (_pi_on + _pi_off)
    _js = 0.5 * (_pi_on * np.log((_pi_on + 1e-12) / (_m + 1e-12))).sum(
        -1
    ) + 0.5 * (_pi_off * np.log((_pi_off + 1e-12) / (_m + 1e-12))).sum(-1)
    _js_mean = float(_js.mean())
    _top_flip = float((_top_on != _top_off).mean())

    _y_std = float(_y_cf.std())


    def _nll_only(pi, mu, sig, y):
        y_ = y[..., None]
        log_g = (
            -0.5 * np.log(2 * np.pi) - np.log(sig) - 0.5 * ((y_ - mu) / sig) ** 2
        )
        return float(-_logsumexp(np.log(pi + 1e-12) + log_g, axis=-1).mean())


    _nll_a = _nll_only(_pi_a, _mu_a, _sig_a, _y_cf)
    _nll_on = _nll_only(_pi_on, _mu_on, _sig_on, _y_cf)
    _nll_off = _nll_only(_pi_off, _mu_off, _sig_off, _y_cf)

    counterfactual_step_df = pl.DataFrame(
        {
            "step": np.arange(1, _pp_per_step.shape[0] + 1),
            "mean_abs_point_diff": _pp_per_step.astype(float),
        }
    )
    fig_counterfactual = qplot(
        counterfactual_step_df,
        "step",
        "mean_abs_point_diff",
        mark="bar",
        title="Mean |point_on − point_off| per forecast step",
        height=260,
    )

    counterfactual_summary = dict(
        n_windows=int(_N_WIN),
        mean_abs_point_diff_on_off=_pp_mean,
        mean_abs_top_mu_diff_on_off=_top_mu_diff,
        target_std=_y_std,
        ratio_point_diff_over_std=_pp_mean / max(_y_std, 1e-12),
        mean_js_pi_on_off=_js_mean,
        frac_top_component_flips=_top_flip,
        nll_actual=_nll_a,
        nll_all_on=_nll_on,
        nll_all_off=_nll_off,
    )

    mo.vstack(
        [
            mo.md(f"""
        ## Counterfactual stimulation ({_N_WIN} test windows)

        Swap the *future* stim (decoder input) between three settings and
        compare predictions against a single fixed history. Encoder input
        (true history) is identical across the three conditions.

        - **actual**: real future stim from the test set.
        - **all_on**: every future stim channel set to its per-channel
          training-set max (so the model sees a "fully stimulated" future).
        - **all_off**: future stim zeroed everywhere.

        **The MPC viability question**. If swapping all-on ↔ all-off barely
        moves the point prediction, the model isn't using stim to forecast.
        A controller can't optimise a signal the model ignores — no matter
        how calibrated the marginal uncertainty looks.

        | metric | value |
        |---|---:|
        | Mean \|point_on − point_off\| (ΔCNR) | **{_pp_mean:.5f}** |
        | Target std (ΔCNR) | {_y_std:.5f} |
        | **Ratio — point diff / target std** | **{_pp_mean / max(_y_std, 1e-12):.3f}** |
        | Mean \|top-component μ(on) − μ(off)\| | {_top_mu_diff:.5f} |
        | Mean JS(π_on ‖ π_off) (nats) | {_js_mean:.4f} |
        | Fraction of steps where top-π flips | {_top_flip:.3f} |
        | NLL (actual stim) | {_nll_a:.4f} |
        | NLL (all-on) | {_nll_on:.4f} |
        | NLL (all-off) | {_nll_off:.4f} |

        **Reading the ratio**: ≪ 1 → the model's mean prediction is
        insensitive to stim, MPC is not viable with this model. ~1 or
        larger → stim materially shifts forecasts, MPC is at least
        plausible.

        **Mixing-weight shift**: if JS(π_on, π_off) ≈ 0 and the top-π
        component rarely flips, the model has **not** learned
        stimulation-dependent mode assignment — the
        responder/non-responder structure isn't encoded in π. A
        stim-driven π shift (JS > 0, flip frac > 0) is direct evidence
        the MDN uses the mixture to route inputs into stim-conditioned
        regimes.
        """),
            fig_counterfactual,
        ]
    )
    return


@app.cell
def _(F_, H, device, mo, model, n_stim, pl, stim_te_used, test_ds):
    import altair as _alt

    _N_GRID = 12
    _idx_grid = np.linspace(0, len(test_ds) - 1, _N_GRID, dtype=int).tolist()

    _stim_max_g = np.zeros(n_stim, dtype=np.float32)
    for _s in stim_te_used:
        _stim_max_g = np.maximum(_stim_max_g, np.asarray(_s).max(axis=1))
    _stim_max_gt = torch.tensor(_stim_max_g).view(1, 1, n_stim)

    _rows_g = []
    model.eval()
    with torch.no_grad():
        for _wi in _idx_grid:
            _enc_in, _dec_stim, _dec_target = test_ds[int(_wi)]
            _enc_b = _enc_in.unsqueeze(0).to(device)
            _s_act = _dec_stim.unsqueeze(0).to(device)
            _s_on = _stim_max_gt.repeat(1, F_, 1).to(device)
            _s_off = torch.zeros_like(_s_act)

            _hist = _enc_in[:, 0].numpy()
            _last_v = float(_hist[-1])
            _act_abs = _last_v + np.cumsum(_dec_target.numpy())
            for _t, _v in enumerate(_hist):
                _rows_g.append(
                    dict(window=int(_wi), t=int(_t), cnr=float(_v), cond="history")
                )
            for _t in range(F_):
                _rows_g.append(
                    dict(
                        window=int(_wi),
                        t=int(H + _t),
                        cnr=float(_act_abs[_t]),
                        cond="truth",
                    )
                )

            for _name, _s in [
                ("actual", _s_act),
                ("all_on", _s_on),
                ("all_off", _s_off),
            ]:
                _pi, _mu, _sig = model(_enc_b, _s)
                _pi_np = _pi.cpu().numpy()[0]
                _mu_np = _mu.cpu().numpy()[0]
                _pt = (_pi_np * _mu_np).sum(-1)
                _abs_pt = _last_v + np.cumsum(_pt)
                _rows_g.append(
                    dict(window=int(_wi), t=int(H - 1), cnr=_last_v, cond=_name)
                )
                for _t in range(F_):
                    _rows_g.append(
                        dict(
                            window=int(_wi),
                            t=int(H + _t),
                            cnr=float(_abs_pt[_t]),
                            cond=_name,
                        )
                    )

    cf_grid_df = pl.DataFrame(_rows_g)

    _colors_g = {
        "history": "#2c3e50",
        "truth": "#000000",
        "actual": "#4C78A8",
        "all_on": "#E45756",
        "all_off": "#54A24B",
    }
    _dom_g = list(_colors_g.keys())
    _rng_g = [_colors_g[k] for k in _dom_g]
    _enc_cg = _alt.Color(
        "cond:N",
        scale=_alt.Scale(domain=_dom_g, range=_rng_g),
        legend=_alt.Legend(title="series"),
    )

    _line_main = (
        _alt.Chart()
        .mark_line(strokeWidth=1.8)
        .encode(
            x=_alt.X("t:Q", title="t"),
            y=_alt.Y("cnr:Q", title="CNR", scale=_alt.Scale(zero=False)),
            color=_enc_cg,
            detail="cond:N",
            tooltip=["window", "cond", "t", "cnr"],
        )
        .transform_filter(_alt.datum.cond != "truth")
    )

    _line_truth = (
        _alt.Chart()
        .mark_line(strokeWidth=2, strokeDash=[4, 3])
        .encode(
            x="t:Q",
            y="cnr:Q",
            color=_enc_cg,
            detail="cond:N",
        )
        .transform_filter(_alt.datum.cond == "truth")
    )

    _boundary_g = (
        _alt.Chart(pl.DataFrame({"t": [H]}))
        .mark_rule(
            color="gray",
            strokeDash=[2, 3],
        )
        .encode(x="t:Q")
    )

    fig_cf_grid = (
        _alt.layer(_line_main, _line_truth, _boundary_g, data=cf_grid_df)
        .properties(
            width=220,
            height=150,
        )
        .facet(facet=_alt.Facet("window:N", title=None), columns=4)
        .properties(
            title=f"Counterfactual future stim — {_N_GRID} sample windows (point predictions)",
        )
        .resolve_scale(y="independent")
    )

    mo.vstack(
        [
            mo.md(f"""
        ## Counterfactual grid — quick orientation

        {_N_GRID} test windows sampled uniformly. Each panel shows the same
        history (dark blue) and truth-future (dashed black), with three
        point-prediction rollouts overlaid: **actual** stim, **all-on**
        stim, **all-off** stim.

        Panels where the three coloured lines overlap → model ignores stim
        for this input. Panels where all-on and all-off fan out →
        stim-responsive windows.
        """),
            fig_cf_grid,
        ]
    )
    return


@app.cell
def _(mixture_metrics_df, mo, pl, qplot, test_ds):
    _resp_mag = []
    for _i in range(len(test_ds)):
        _enc_in, _dec_stim, _dec_target = test_ds[int(_i)]
        _hist_cnr = _enc_in[:, 0].numpy()
        _last = float(_hist_cnr[-1])
        _future = _last + np.cumsum(_dec_target.numpy())
        _full = np.concatenate([_hist_cnr, _future])
        _resp_mag.append(float(_full.std()))
    resp_mag = np.array(_resp_mag)

    _q = np.quantile(resp_mag, [0.25, 0.5, 0.75])
    _labels = ["Q1 (flat)", "Q2", "Q3", "Q4 (responsive)"]
    _bin_idx = np.digitize(resp_mag, _q)
    _bin_name = np.array([_labels[_b] for _b in _bin_idx])

    _window_bin = pl.DataFrame(
        {
            "window": np.arange(len(resp_mag)),
            "bin": _bin_name,
            "resp_std": resp_mag,
        }
    )

    strat_df = (
        mixture_metrics_df.join(_window_bin, on="window")
        .group_by(["bin", "step"])
        .agg(
            [
                pl.col("nll").mean().alias("nll"),
                pl.col("crps").mean().alias("crps"),
                pl.col("sigma_eff").mean().alias("sigma"),
                pl.col("resid").abs().mean().alias("mae"),
                pl.len().alias("n"),
            ]
        )
        .sort(["bin", "step"])
    )

    fig_stratified = qplot(
        strat_df,
        "step",
        "nll",
        color="bin",
        mark="line",
        title="Mixture NLL per step, stratified by response magnitude quartile",
        height=280,
    )

    mo.vstack(
        [
            mo.md(
                f"""
            ## Stratified metrics by response magnitude

            **The problem with a single averaged NLL**. In this dataset, the
            majority of cells are **non-responsive** — their CNR trajectory
            is nearly flat throughout the window. A trivial model that
            predicts "nothing will change" gets a near-perfect NLL on those
            windows. Since those windows dominate the test set by count,
            they dominate the *mean* NLL. The model could be catastrophically
            bad at predicting responsive cells (the cells we actually care
            about for biology) and the average would barely budge. Averaging
            over a lopsided distribution hides exactly the failures you're
            trying to detect.

            **The fix: stratify before averaging**. Compute a per-window
            scalar difficulty measure, split windows into bins of
            comparable difficulty, and report metrics within each bin.

            **The difficulty measure used here**: for each test window,
            `resp_std = std(CNR)` computed over the full history + target
            trajectory (length H + F). This is 0 for a perfectly flat
            trajectory and grows with how much the cell's CNR moves during
            the window. Proxy for "did this cell respond". Other proxies
            are reasonable (`max(CNR) − min(CNR)`, total absolute
            increment, etc.) but std is simple and robust to single-point
            outliers.

            **The split**: quartiles of `resp_std` across the test set.
            - **Q1 (flat)**: resp_std ≤ {_q[0]:.3f}. Mostly non-responsive.
            - **Q2**: resp_std ≤ {_q[1]:.3f}.
            - **Q3**: resp_std ≤ {_q[2]:.3f}.
            - **Q4 (responsive)**: resp_std > {_q[2]:.3f}. The cells that
              actually show dynamics.

            Each bin has roughly 25% of the test set.

            **How to read the table and the per-step line plot**:
            - **Flat NLL across Q1 → Q4** → **balanced fit**. Model
              handles flat and responsive cells equally well. This is the
              target.
            - **Monotonic rise Q1 → Q4** (most common failure): the model
              is overfitting to the easy flat majority. Responsive cells
              are substantially worse-predicted. This is the canonical
              signal to try **sample reweighting**: build a
              `WeightedRandomSampler` keyed on `resp_std` bin and
              upsample Q3/Q4 during training. Alternatives: focal-style
              loss (downweight easy/low-loss samples), or curriculum
              (start with responsive subset, expand).
            - **U-shape** (Q1 and Q4 worst, Q2/Q3 fine): rare; usually
              indicates the difficulty measure or binning is capturing
              the wrong thing — investigate specific bad windows.
            - **Q4 much better than Q1**: surprising; usually means resp_std
              is confounded with something the model *is* good at
              predicting (e.g., specific stim protocols). Worth debugging.

            **Don't just watch NLL** — the `mae` column is in ΔCNR units
            and is directly interpretable. If `mae` is flat across bins
            while NLL rises, the model's point prediction is fine but its
            σ is badly miscalibrated on responsive cells (a different
            failure mode than "gets the prediction wrong").
            """
            ),
            strat_df,
            fig_stratified,
        ]
    )
    return (fig_stratified,)


@app.cell
def _(mixture_metrics_df, mo, pl):
    import altair as _alt

    _df = mixture_metrics_df.with_columns(pl.col("resid").abs().alias("abs_resid"))
    _sig = _df["sigma_eff"].to_numpy()
    _edges = np.quantile(_sig, np.linspace(0.0, 1.0, 11))
    _edges[-1] = _edges[-1] + 1e-9
    _bin_idx = np.digitize(_sig, _edges[1:-1])
    _df = _df.with_columns(pl.Series("sigma_bin", _bin_idx))

    sharp_df = (
        _df.group_by("sigma_bin")
        .agg(
            [
                pl.col("sigma_eff").mean().alias("sigma_mean"),
                pl.col("abs_resid").mean().alias("abs_resid_mean"),
                pl.len().alias("n"),
            ]
        )
        .sort("sigma_bin")
        .with_columns(
            (pl.col("sigma_mean") * float(np.sqrt(2.0 / np.pi))).alias(
                "ideal_abs_resid"
            ),
        )
    )

    _pts = (
        _alt.Chart(sharp_df)
        .mark_point(size=90, filled=True, color="#4C78A8")
        .encode(
            x=_alt.X("sigma_mean:Q", title="predicted σ (bin mean)"),
            y=_alt.Y("abs_resid_mean:Q", title="E|residual|"),
            tooltip=["sigma_mean", "abs_resid_mean", "n"],
        )
    )
    _ideal = (
        _alt.Chart(sharp_df)
        .mark_line(color="red", strokeDash=[4, 4])
        .encode(x="sigma_mean:Q", y="ideal_abs_resid:Q")
    )
    fig_sharpness = (_pts + _ideal).properties(
        title="Sharpness vs accuracy (red = ideal σ·√(2/π))",
        width=420,
        height=300,
    )

    mo.vstack(
        [
            mo.md(
                """
            ## Sharpness vs accuracy (conditional calibration)

            **The gap in what PIT and coverage tell us**. PIT histograms
            and k·σ coverage both answer **marginal** calibration
            questions: averaged over all test predictions, do the predicted
            quantiles match empirical frequencies? A model can pass both
            while being completely useless at **ranking predictions by
            difficulty**.

            **Concrete failure case**. Suppose a model predicts σ = 0.1 on
            every test window, regardless of input. Half the windows have
            actual |residual| ≈ 0.05 (easy, low-variance cells). Half have
            actual |residual| ≈ 0.15 (hard, responsive cells). The overall
            mean |residual| is 0.10 — roughly consistent with marginal
            calibration checks. But **σ tells you nothing about whether
            any specific prediction is trustworthy**. If you downstream-
            gate on σ (e.g., "only use predictions where σ < 0.08"), you
            get no useful filter — every prediction has the same σ.

            **The fix: conditional calibration test**. Ask: "*Among
            predictions with similar predicted σ*, does the actual error
            match what σ implies?"

            **Construction**:
            1. Group the `N × F` test predictions into **10 deciles by
               predicted σ** (computed as the mixture's effective std
               `√(Σ π_k · (σ_k² + (μ_k − mean)²))`).
            2. Within each decile, compute the mean predicted σ
               (x-axis) and the mean |residual| (y-axis).
            3. Plot the 10 resulting points.
            4. Overlay the Gaussian ideal.

            **The Gaussian ideal line**: if a prediction is Gaussian with
            standard deviation σ, the expected absolute residual is
            `E|Z| · σ = √(2/π) · σ ≈ 0.7979 · σ` (because a standard
            normal's absolute value has mean `√(2/π)`). For a
            *calibrated* Gaussian, the scatter of (mean σ, mean
            |residual|) points should fall on the line `y = 0.7979 · x`.

            **How to read the four possible outcomes**:

            - **Points on the red line** → **conditionally calibrated**.
              High-σ predictions really do produce large errors; low-σ
              predictions really are accurate. σ is a trustworthy
              per-prediction uncertainty — safe to use for downstream
              filtering.

            - **Points above the line** → **underconfident**. Actual
              |residual| exceeds `0.798 · σ`. The model's uncertainty is
              too small for the errors it's producing. Often a sign of
              residual bias the σ isn't tracking.

            - **Points below the line** → **overconfident**. Actual
              |residual| is smaller than `0.798 · σ`. The model is
              *more* accurate than it claims. Not a performance problem
              per se, but if you propagate σ downstream you'll over-hedge.

            - **Flat curve** (y doesn't rise with x) → **σ uninformative**.
              Predictions with high σ are no worse on average than
              predictions with low σ. σ is effectively a constant with
              noise. The model produces distributions, but the
              uncertainty estimate contains no signal about error. MC
              dropout uncertainty and badly-trained MDNs both show this.

            - **Reversed slope** (high σ → low error) → **pathological**.
              Very rare. Usually a bug in data split, target
              normalization, or training loop.

            **Gaussian-mixture caveat**. The ideal line assumes unimodal
            Gaussian predictions. Genuinely multimodal MDN predictions
            can have `E|resid|` that deviates from `0.798·σ` even when
            perfectly calibrated (the extra spread from inter-mode
            distance pushes `E|resid|` above `0.798·σ`). In practice the
            line is a useful reference; don't over-interpret small
            deviations as miscalibration if mode usage is high.

            **Why this is complementary to PIT**: PIT can be flat while σ
            is uninformative (as long as marginal tail frequencies work
            out). Sharpness-vs-accuracy can show σ tracking error per-
            prediction even while PIT is skewed from point-prediction
            bias. Different axes of calibration; both worth checking.
            """
            ),
            fig_sharpness,
        ]
    )
    return (fig_sharpness,)


@app.cell
def _(ablation_df, mixture_metrics_df, mo, pl, test_act, test_point, test_std):
    _mse = float(np.mean((test_act - test_point) ** 2))
    _mae = float(np.mean(np.abs(test_act - test_point)))
    _nll_proxy = float(
        np.mean(0.5 * ((test_act - test_point) / test_std) ** 2 + np.log(test_std))
    )
    _mean_std = float(test_std.mean())

    _nll_exact = float(mixture_metrics_df["nll"].mean())
    _crps_mean = float(mixture_metrics_df["crps"].mean())
    _pit_vals = mixture_metrics_df["pit"].to_numpy()
    # Simple PIT uniformity score: KS stat vs uniform
    _sorted = np.sort(_pit_vals)
    _n = len(_sorted)
    _ks = float(
        np.max(
            np.abs(
                _sorted - np.linspace(1.0 / _n, 1.0, _n)
            )
        )
    )

    _base = ablation_df.filter(pl.col("idx") == -1).row(0, named=True)
    _top_ablation = (
        ablation_df.filter(pl.col("idx") >= 0)
        .sort("d_nll", descending=True)
        .head(3)
    )
    _top_str = " · ".join(
        f"{_r['channel']} (ΔNLL={_r['d_nll']:.3f})"
        for _r in _top_ablation.iter_rows(named=True)
    )

    eval_metrics = dict(
        test_mse_point=_mse,
        test_mae_point=_mae,
        test_nll_gaussian_proxy=_nll_proxy,
        test_nll_mixture_exact=_nll_exact,
        test_crps_mean=_crps_mean,
        test_mean_std=_mean_std,
        test_pit_ks=_ks,
        ablation_baseline_nll=float(_base["nll"]),
    )
    mo.md(
        f"""
    ## Evaluation summary

    | metric | value |
    |--------|------:|
    | Point MSE (π·μ vs actual δ) | {_mse:.6f} |
    | Point MAE | {_mae:.6f} |
    | Mean predicted σ | {_mean_std:.4f} |
    | Gaussian-proxy NLL | {_nll_proxy:.4f} |
    | **Mixture NLL (exact)** | **{_nll_exact:.4f}** |
    | **Mean CRPS** | **{_crps_mean:.4f}** |
    | PIT KS stat (0 = uniform) | {_ks:.4f} |

    **Top load-bearing channels (largest ΔNLL when ablated):** {_top_str}.
    """
    )
    return (eval_metrics,)


@app.cell
def _(mo):
    get_win_start, set_win_start = mo.state(0)
    return get_win_start, set_win_start


@app.cell
def _(cnr_te_used, mo, model_config_used):
    _n_tracks = len(cnr_te_used)
    track_selector = mo.ui.dropdown(
        options=[str(i) for i in range(_n_tracks)],
        value="0",
        searchable=True,
        label=f"Test track ({_n_tracks} total, source=`{model_config_used.data_source}`)",
    )
    track_selector
    return (track_selector,)


@app.cell
def _(get_win_start, mo, set_win_start):
    def _step(delta):
        return lambda _: set_win_start(get_win_start() + delta)


    btn_m10 = mo.ui.button(label="⟨⟨ −10", on_click=_step(-10))
    btn_m5 = mo.ui.button(label="⟨ −5", on_click=_step(-5))
    btn_m1 = mo.ui.button(label="⟨ −1", on_click=_step(-1))
    btn_p1 = mo.ui.button(label="+1 ⟩", on_click=_step(+1))
    btn_p5 = mo.ui.button(label="+5 ⟩", on_click=_step(+5))
    btn_p10 = mo.ui.button(label="+10 ⟩⟩", on_click=_step(+10))
    btn_reset = mo.ui.button(
        label="⟲ 0", on_click=lambda _: set_win_start(0), kind="warn"
    )

    mo.hstack(
        [btn_m10, btn_m5, btn_m1, btn_p1, btn_p5, btn_p10, btn_reset],
        justify="start",
    )
    return btn_m1, btn_m10, btn_m5, btn_p1, btn_p10, btn_p5, btn_reset


@app.cell
def _(
    cnr_te_used,
    get_win_start,
    mo,
    model_config_used,
    stim_te_used,
    track_selector,
):
    _track_idx = int(track_selector.value)
    _cnr_tr = cnr_te_used[_track_idx]
    _stim_tr = stim_te_used[_track_idx]
    _traj_len = int(_cnr_tr.shape[0])
    _H = int(model_config_used.history_len)
    _F = int(model_config_used.future_len)
    _max_start = max(0, _traj_len - (_H + _F))

    _raw = get_win_start()
    _start = max(0, min(_raw, _max_start))

    _enc_cnr = _cnr_tr[_start : _start + _H]
    _enc_stim = _stim_tr[:, _start : _start + _H]
    _dec_stim = _stim_tr[:, _start + _H : _start + _H + _F]
    _full = _cnr_tr[_start : _start + _H + _F]
    _dec_target = np.diff(_full)[_H - 1 : _H - 1 + _F]
    _enc_in = np.concatenate([_enc_cnr[:, np.newaxis], _enc_stim.T], axis=-1)

    win_enc_in = torch.tensor(_enc_in, dtype=torch.float32)
    win_dec_stim = torch.tensor(_dec_stim.T, dtype=torch.float32)
    win_dec_target = torch.tensor(_dec_target, dtype=torch.float32)
    win_track = _track_idx
    win_start = _start
    win_label = f"Track {_track_idx} · start {_start}"

    _src = model_config_used.data_source
    _clamp_note = f" _(clamped from {_raw})_" if _raw != _start else ""
    _range_note = (
        " _(window fills the whole trajectory — frame-by-frame is a no-op for this dataset)_"
        if _max_start == 0
        else ""
    )
    mo.md(
        f"**{win_label}** · source=`{_src}` · start {_start} of {_max_start} · "
        f"traj_len={_traj_len} · H={_H}, F={_F}{_clamp_note}{_range_note}"
    )
    return win_dec_stim, win_dec_target, win_enc_in, win_label


@app.cell
def _(
    F_,
    H,
    device,
    mo,
    model,
    pl,
    win_dec_stim,
    win_dec_target,
    win_enc_in,
    win_label,
):
    import altair as _alt

    _N_MC = 200
    _enc_in, _dec_stim, _dec_target = win_enc_in, win_dec_stim, win_dec_target

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
        title=f"{win_label}: history (solid) | actual future (faded) | MDN median + 50/90% MC bands",
    )

    mo.vstack(
        [
            mo.md(
                f"**{win_label}** — {_N_MC} Monte Carlo rollouts from MDN (sample component + Gaussian per step)"
            ),
            _chart,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    F_,
    H,
    device,
    mo,
    model,
    pl,
    win_dec_stim,
    win_dec_target,
    win_enc_in,
    win_label,
):
    import altair as _alt

    _enc_in_k, _dec_stim_k, _dec_target_k = (
        win_enc_in,
        win_dec_stim,
        win_dec_target,
    )

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
            title=f"{win_label}: real | model π-weighted ±σ | K={_K} components | light stim (bottom strip)",
        )
        .resolve_scale(size="independent", opacity="independent")
        .interactive()
    )

    mo.vstack(
        [
            mo.md(
                f"""
                - **{win_label}** — legend covers every series.
                - **real** (black): ground truth. 
                - **model (π-weighted)** (purple): whole-model rollout with ±1 mixture-σ band. 
                - **k0…kN**: forced per-component rollouts; dot size/opacity = π_k at step. 
                - **light stim** (amber strip at bottom): stim channel 0 — shape only, scaled to a band below the CNR data."""
            ),
            chart_components,
        ]
    )
    return


@app.cell
def _(
    F_,
    H,
    device,
    mo,
    model,
    n_stim,
    pl,
    stim_te_used,
    win_dec_stim,
    win_dec_target,
    win_enc_in,
    win_label,
):
    import altair as _alt

    _stim_max_w = np.zeros(n_stim, dtype=np.float32)
    for _s in stim_te_used:
        _stim_max_w = np.maximum(_stim_max_w, np.asarray(_s).max(axis=1))

    _enc_w = win_enc_in.unsqueeze(0).to(device)
    _stim_act = win_dec_stim.unsqueeze(0).to(device)
    _stim_on = (
        torch.tensor(_stim_max_w).view(1, 1, n_stim).repeat(1, F_, 1).to(device)
    )
    _stim_off = torch.zeros_like(_stim_act)

    model.eval()
    with torch.no_grad():
        _cond_out = {}
        for _name, _s in [
            ("actual", _stim_act),
            ("all_on", _stim_on),
            ("all_off", _stim_off),
        ]:
            _pi, _mu, _sig = model(_enc_w, _s)
            _pi_np = _pi.cpu().numpy()[0]
            _mu_np = _mu.cpu().numpy()[0]
            _sig_np = _sig.cpu().numpy()[0]
            _pt = (_pi_np * _mu_np).sum(-1)
            _std = np.sqrt(
                (_pi_np * (_sig_np**2 + (_mu_np - _pt[:, None]) ** 2)).sum(-1)
            )
            _cond_out[_name] = (_pt, _std, _pi_np)

    _hist_cnr = win_enc_in[:, 0].numpy()
    _last = float(_hist_cnr[-1])
    _act_abs = _last + np.cumsum(win_dec_target.numpy())

    _rows = []
    for _t, _v in enumerate(_hist_cnr):
        _rows.append(
            dict(
                t=int(_t),
                cnr=float(_v),
                lo=float(_v),
                hi=float(_v),
                cond="history",
            )
        )
    for _t in range(F_):
        _rows.append(
            dict(
                t=int(H + _t),
                cnr=float(_act_abs[_t]),
                lo=float(_act_abs[_t]),
                hi=float(_act_abs[_t]),
                cond="truth",
            )
        )
    for _name, (_pt, _std, _pi_np) in _cond_out.items():
        _abs = _last + np.cumsum(_pt)
        _std_abs = np.sqrt(np.cumsum(_std**2))
        _rows.append(dict(t=int(H - 1), cnr=_last, lo=_last, hi=_last, cond=_name))
        for _t in range(F_):
            _rows.append(
                dict(
                    t=int(H + _t),
                    cnr=float(_abs[_t]),
                    lo=float(_abs[_t] - _std_abs[_t]),
                    hi=float(_abs[_t] + _std_abs[_t]),
                    cond=_name,
                )
            )
    cf_win_df = pl.DataFrame(_rows)

    _colors = {
        "history": "#2c3e50",
        "truth": "#000000",
        "actual": "#4C78A8",
        "all_on": "#E45756",
        "all_off": "#54A24B",
    }
    _domain = list(_colors.keys())
    _range = [_colors[k] for k in _domain]
    _enc_c = _alt.Color(
        "cond:N",
        scale=_alt.Scale(domain=_domain, range=_range),
        legend=_alt.Legend(title="series"),
    )

    _pred = cf_win_df.filter(pl.col("cond").is_in(["actual", "all_on", "all_off"]))
    _band = (
        _alt.Chart(_pred)
        .mark_area(opacity=0.15)
        .encode(
            x=_alt.X("t:Q", title="timestep"),
            y=_alt.Y("lo:Q", title="CNR"),
            y2="hi:Q",
            color=_enc_c,
        )
    )
    _line = (
        _alt.Chart(_pred)
        .mark_line(strokeWidth=2.5)
        .encode(
            x="t:Q",
            y="cnr:Q",
            color=_enc_c,
            tooltip=["cond", "t", "cnr"],
        )
    )
    _hst = (
        _alt.Chart(cf_win_df.filter(pl.col("cond") == "history"))
        .mark_line(
            strokeWidth=2.5,
        )
        .encode(x="t:Q", y="cnr:Q", color=_enc_c)
    )
    _tr = (
        _alt.Chart(cf_win_df.filter(pl.col("cond") == "truth"))
        .mark_line(
            strokeWidth=2.5,
            strokeDash=[4, 3],
        )
        .encode(x="t:Q", y="cnr:Q", color=_enc_c)
    )
    _b = (
        _alt.Chart(pl.DataFrame({"t": [H]}))
        .mark_rule(
            color="gray",
            strokeDash=[2, 3],
        )
        .encode(x="t:Q")
    )

    fig_cf_window = (
        (_band + _line + _hst + _tr + _b)
        .properties(
            width=820,
            height=380,
            title=f"{win_label}: counterfactual future stim (actual / all-on / all-off)",
        )
        .interactive()
    )

    _pi_on_w = _cond_out["all_on"][2]
    _pi_off_w = _cond_out["all_off"][2]
    _pt_on_w, _pt_off_w = _cond_out["all_on"][0], _cond_out["all_off"][0]
    _m_w = 0.5 * (_pi_on_w + _pi_off_w)
    _js_w = 0.5 * (_pi_on_w * np.log((_pi_on_w + 1e-12) / (_m_w + 1e-12))).sum(
        -1
    ) + 0.5 * (_pi_off_w * np.log((_pi_off_w + 1e-12) / (_m_w + 1e-12))).sum(-1)
    _pp_abs_w = float(np.abs(_pt_on_w - _pt_off_w).sum())

    mo.vstack(
        [
            mo.md(f"""
        **{win_label}** — counterfactual future stim.

        - Σ \|point(on) − point(off)\| over F={F_} steps = **{_pp_abs_w:.5f}**
        - Mean JS(π_on ‖ π_off) = **{_js_w.mean():.4f}**

        If the three bands overlap almost completely the model has not
        learned to respond to stim for this window. If they separate, this
        window is informative for control.
        """),
            fig_cf_window,
        ]
    )
    return


@app.cell
def _(
    btn_m1,
    btn_m10,
    btn_m5,
    btn_p1,
    btn_p10,
    btn_p5,
    btn_reset,
    mo,
    track_selector,
):
    track_selector, mo.hstack(
        [btn_m10, btn_m5, btn_m1, btn_p1, btn_p5, btn_p10, btn_reset],
        justify="start",
    )
    return


@app.cell(hide_code=True)
def _(IS_HEADLESS, mo, tracker):
    if (not IS_HEADLESS) and tracker is not None:
        save_all_button = mo.ui.run_button(
            label="Save experiment (model + figures + stats)"
        )
    else:
        save_all_button = None

    save_all_button if save_all_button is not None else mo.md("")
    return (save_all_button,)


@app.cell
def _(
    IS_HEADLESS,
    artifacts,
    cnr_tr,
    cnr_va,
    eval_metrics,
    fig_ablation,
    fig_calib,
    fig_loss,
    fig_modes,
    fig_pit,
    fig_residuals,
    fig_sharpness,
    fig_std,
    fig_stratified,
    fig_tf,
    fig_traj,
    hostname,
    is_cluster,
    mo,
    save_all_button,
    save_bundle,
):
    save_bundle(
        mo=mo,
        is_headless=IS_HEADLESS,
        artifacts=artifacts,
        figures={
            "loss_curves": fig_loss,
            "tf_schedule": fig_tf,
            "residuals": fig_residuals,
            "pred_std_by_step": fig_std,
            "sample_trajectories": fig_traj,
            "coverage": fig_calib,
            "pit": fig_pit,
            "mode_usage": fig_modes,
            "feature_ablation": fig_ablation,
            "stratified_by_response": fig_stratified,
            "sharpness_vs_accuracy": fig_sharpness,
        },
        metrics=eval_metrics,
        n_train=len(cnr_tr),
        n_val=len(cnr_va),
        save_button=save_all_button,
        hostname=hostname,
        is_cluster=is_cluster,
    )
    return


if __name__ == "__main__":
    app.run()
