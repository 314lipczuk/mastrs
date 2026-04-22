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
                cnr_tr, stim_tr, mcfg.history_len, mcfg.future_len, stride=15
            )
            val_ds = Seq2SeqDataset(
                cnr_va, stim_va, mcfg.history_len, mcfg.future_len, stride=15
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
    from experiments.scaffold import (
        form_from_configs,
        configs_from_cli,
        configs_from_form,
        train_model,
        load_model,
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
        configs_from_cli,
        configs_from_form,
        device,
        form_from_configs,
        hostname,
        is_cluster,
        load_dataset,
        load_experiment,
        load_model,
        mo,
        n_stim,
        parse_bool,
        pl,
        qplot,
        repo_root,
        results_base,
        results_read_sources,
        save_bundle,
        scan_experiment_dirs,
        train_model,
    )


@app.cell(hide_code=True)
def _(mo, parse_bool):
    MODE = mo.cli_args().get("mode", "train")
    IS_HEADLESS = "name" in mo.cli_args()
    EXPERIMENT_NAME = mo.cli_args().get("name", "lstm_seq2scal_mdn")
    DRY_RUN = parse_bool(mo.cli_args().get("dry_run", True))

    if MODE not in ("train", "load"):
        raise ValueError(f"--mode must be 'train' or 'load', got {MODE!r}")

    mo.md(f"**Mode:** `{MODE}` · **Headless:** `{IS_HEADLESS}` · **Experiment:** `{EXPERIMENT_NAME}`  · **Dry run :** `{DRY_RUN}` ")
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
        _src_root = Path(results_read_sources(repo_root)[_src_name]) #/ EXPERIMENT_NAME
        _choices = scan_experiment_dirs(_src_root)
        if _choices:
            print(_choices)
            experiment_picker = mo.ui.dropdown(
                options=_choices, value=_choices[0], label="Experiment run",
            )
            load_button = mo.ui.button(
                value=0, on_click=lambda n: n + 1, label="Load",
            )
            _picker_ui = mo.vstack([experiment_picker, load_button])
        else:
            experiment_picker = None
            load_button = None
            _picker_ui = mo.md(f"No experiments under `{_src_root}`.")
    else:
        experiment_picker = None
        load_button = None
        _picker_ui = mo.md("")

    _picker_ui
    return experiment_picker, load_button


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
    configs_from_cli,
    configs_from_form,
    form,
    mo,
    n_stim,
):
    _always = {"m": {"encoder_dim": 1 + n_stim, "stim_dim": n_stim}}
    _config_classes = {"m": ModelConfig, "t": TrainingConfig}

    if MODE == "load":
        data_source = "synthetic"
        model_config = None
        training_config = None
        ctx_display = mo.md(
            "**Load mode** — pick an experiment above and click **Load**. Config comes from the bundle."
        )
    elif IS_HEADLESS:
        data_source = mo.cli_args().get("source", "synthetic")
        _always["m"]["data_source"] = data_source
        _cfgs = configs_from_cli(mo.cli_args(), _config_classes, always=_always)
        model_config, training_config = _cfgs["m"], _cfgs["t"]
        ctx_display = mo.md(
            f"**Headless train** — `{EXPERIMENT_NAME}` · source `{data_source}` · dry_run={DRY_RUN}"
        )
    else:
        mo.stop(
            form.value is None,
            mo.md("Fill in the form above and click **Apply**."),
        )
        _cfgs = configs_from_form(form.value, _config_classes, always=_always)
        model_config, training_config = _cfgs["m"], _cfgs["t"]
        data_source = model_config.data_source
        ctx_display = mo.md(
            f"**Interactive train** — `{EXPERIMENT_NAME}` · source `{data_source}` · pydantic ✓"
        )

    ctx_display
    return data_source, model_config, training_config


@app.cell
def _(DRY_RUN, data_source, load_dataset, mo):
    if data_source == "real":
        _total_window_guess = 30 + 5
        cnr_all, stim_all, conditions_all = load_dataset(
            "real",
            window_size=_total_window_guess,
            stride=max(1, _total_window_guess // 4),
        )
    else:
        cnr_all, stim_all, conditions_all = load_dataset(data_source)

    n_traj = len(cnr_all)
    traj_len = cnr_all.shape[1]

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

    mo.md(f"""
    **Data:** {n_traj} trajectories × {traj_len} timepoints (`{data_source}`)

    Splits: train={len(_tr_ids)} | val={len(_va_ids)} | test={len(_te_ids)} · dry_run={DRY_RUN}
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
    experiment_picker,
    load_button,
    load_model,
    mo,
    model_config,
    repo_root,
    results_base,
    results_read_sources,
    source_selector,
    stim_tr,
    stim_va,
    train_button,
    train_model,
    training_config,
):
    if MODE == "load":
        mo.stop(
            experiment_picker is None
            or load_button is None
            or not bool(load_button.value),
            mo.md("Pick experiment and click **Load**."),
        )
        _src_name = source_selector.value if source_selector is not None else "Local"
        _src_root = Path(results_read_sources(repo_root)[_src_name]) #/ EXPERIMENT_NAME
        artifacts = load_model(
            experiment_path=_src_root / experiment_picker.value,
            model_cls=Seq2ScalarMDN,
            model_config_cls=ModelConfig,
            device=device,
        )
    elif IS_HEADLESS:
        artifacts = train_model(
            mo=mo,
            model_cls=Seq2ScalarMDN,
            dataset={"train": (cnr_tr, stim_tr), "val": (cnr_va, stim_va)},
            model_config=model_config,
            training_config=training_config,
            device=device,
            experiment_name=EXPERIMENT_NAME,
            results_base=results_base,
            is_headless=True,
        )
    else:
        mo.stop(not train_button.value, mo.md("Click **Start training**."))
        artifacts = train_model(
            mo=mo,
            model_cls=Seq2ScalarMDN,
            dataset={"train": (cnr_tr, stim_tr), "val": (cnr_va, stim_va)},
            model_config=model_config,
            training_config=training_config,
            device=device,
            experiment_name=EXPERIMENT_NAME,
            results_base=results_base,
            is_headless=False,
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
def _(
    MODE,
    experiment_picker,
    load_experiment,
    mo,
    repo_root,
    results_read_sources,
    source_selector,
):
    import json as _json

    if (
        MODE == "load"
        and experiment_picker is not None
        and source_selector is not None
    ):
        _src_root_ld = Path(results_read_sources(repo_root)[source_selector.value])
        _exp_path_ld = _src_root_ld / experiment_picker.value
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
def _(MODE, cnr_te, data_source, load_dataset, mo, model_config_used, stim_te):
    H = model_config_used.history_len
    F_ = model_config_used.future_len

    if MODE == "load" and model_config_used.data_source != data_source:
        _ds_for_test = model_config_used.data_source
        if _ds_for_test == "real":
            _tw = H + F_
            _cnr_ld, _stim_ld, _ = load_dataset(
                "real", window_size=_tw, stride=max(1, _tw // 4)
            )
        else:
            _cnr_ld, _stim_ld, _ = load_dataset(_ds_for_test)
        _ids = np.arange(len(_cnr_ld))
        _tr, _te = train_test_split(_ids, test_size=0.2, random_state=42)
        cnr_te_used, stim_te_used = _cnr_ld[_te], _stim_ld[_te]
    else:
        cnr_te_used, stim_te_used = cnr_te, stim_te

    test_ds = Seq2SeqDataset(cnr_te_used, stim_te_used, H, F_, stride=15)

    mo.md(
        f"Test windows: {len(test_ds)} (H={H}, F={F_}, source=`{model_config_used.data_source}`)"
    )
    return F_, H, test_ds


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
    return test_act, test_point, test_std


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
def _(mo, test_act, test_point, test_std):
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
def _(F_, H, device, mo, model, pl, test_ds, traj_selector):
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
def _(F_, H, device, mo, model, pl, test_ds, traj_selector):
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
    fig_calib,
    fig_loss,
    fig_residuals,
    fig_std,
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
