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
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset

    from experiment import ExperimentBundle
    from utils import get_device, results_read_sources
    from experiments.seq2seq_data import load_synthetic, load_real, STIM_COLS

    device = get_device()
    return


@app.cell
def _():
    mo.md("""
    # LSTM Seq2Seq: Synthetic vs Real comparison

    Load two trained experiment bundles (one synthetic, one real) and evaluate
    both models on both test sets to measure in-domain vs cross-domain performance.
    """)
    return


@app.cell
def _():
    _results_sources = results_read_sources(Path(__file__).resolve().parent.parent)
    results_source_selector = mo.ui.dropdown(
        options=list(_results_sources.keys()),
        value=list(_results_sources.keys())[0],
        label="Results source",
    )
    results_source_selector
    return


@app.cell
def _():
    _results_path = Path(results_read_sources(Path(__file__).resolve().parent.parent)[results_source_selector.value])

    _exp_dirs = []
    if _results_path.is_dir():
        for _subdir in sorted(_results_path.iterdir()):
            if _subdir.is_dir() and any(_subdir.glob("*.pt")):
                _exp_dirs.append(_subdir.name)

    synthetic_exp_selector = mo.ui.dropdown(
        options=_exp_dirs,
        label="Synthetic-trained experiment",
    )
    real_exp_selector = mo.ui.dropdown(
        options=_exp_dirs,
        label="Real-trained experiment",
    )
    load_btn = mo.ui.run_button(label="Load & compare")
    mo.vstack([
        mo.hstack([synthetic_exp_selector, real_exp_selector], gap=2),
        load_btn,
    ])
    return


@app.cell
def _():
    mo.stop(not load_btn.value, mo.md("Select two experiments and click **Load & compare**."))

    _results_path = Path(results_read_sources(Path(__file__).resolve().parent.parent)[results_source_selector.value])

    bundle_syn = ExperimentBundle.load(str(_results_path / synthetic_exp_selector.value))
    bundle_real = ExperimentBundle.load(str(_results_path / real_exp_selector.value))

    mo.md(f"""
    **Loaded:**
    - Synthetic: `{bundle_syn.name}` ({bundle_syn.timestamp})
    - Real: `{bundle_real.name}` ({bundle_real.timestamp})
    """)
    return


@app.cell
def _():
    model_syn = bundle_syn.reconstruct_model().to(device)
    model_real = bundle_real.reconstruct_model().to(device)
    model_syn.eval()
    model_real.eval()

    H_syn = bundle_syn.model_config.get("history_len", 15)
    F_syn = bundle_syn.model_config.get("future_len", 10)
    H_real = bundle_real.model_config.get("history_len", 15)
    F_real = bundle_real.model_config.get("future_len", 10)

    mo.md(f"""
    **Model configs:**
    - Synthetic model: H={H_syn}, F={F_syn}, hidden={bundle_syn.model_config.get('hidden_dim')}, layers={bundle_syn.model_config.get('num_layers')}
    - Real model: H={H_real}, F={F_real}, hidden={bundle_real.model_config.get('hidden_dim')}, layers={bundle_real.model_config.get('num_layers')}
    """)
    return


@app.cell
def _():
    class Seq2SeqDataset(torch.utils.data.Dataset):
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

    H_shared = min(H_syn, H_real)
    F_shared = min(F_syn, F_real)
    _total = H_shared + F_shared

    cnr_syn, stim_syn, cond_syn = load_synthetic()
    cnr_real, stim_real, cond_real = load_real(
        window_size=_total,
        stride=max(1, _total // 4),
    )

    _ids_syn = np.arange(len(cnr_syn))
    _, _te_syn = train_test_split(_ids_syn, test_size=0.2, random_state=42)
    test_ds_syn = Seq2SeqDataset(cnr_syn[_te_syn], stim_syn[_te_syn], H_shared, F_shared, stride=15)

    _ids_real = np.arange(len(cnr_real))
    _, _te_real = train_test_split(_ids_real, test_size=0.2, random_state=42)
    test_ds_real = Seq2SeqDataset(cnr_real[_te_real], stim_real[_te_real], H_shared, F_shared, stride=15)

    mo.md(f"""
    **Test sets** (H={H_shared}, F={F_shared}):
    - Synthetic: {len(test_ds_syn)} windows
    - Real: {len(test_ds_real)} windows
    """)
    return


@app.cell
def _():
    def _evaluate(mdl, ds):
        _last_list, _act_list, _pred_list = [], [], []
        mdl.eval()
        with torch.no_grad():
            for _enc, _stim, _tgt in DataLoader(ds, batch_size=512):
                _enc_d, _stim_d = _enc.to(device), _stim.to(device)
                _last_list.append(_enc_d[:, -1, 0].cpu().numpy())
                _act_list.append(_tgt.numpy())
                _pred_list.append(mdl(_enc_d, _stim_d).cpu().numpy())
        _last = np.concatenate(_last_list)
        _act = np.concatenate(_act_list)
        _pred = np.concatenate(_pred_list)
        _act_abs = _last[:, None] + np.cumsum(_act, axis=1)
        _pred_abs = _last[:, None] + np.cumsum(_pred, axis=1)
        _mse_per_step = np.mean((_pred - _act) ** 2, axis=0)
        _mse_window = np.mean((_pred - _act) ** 2, axis=1)
        _ss_tot = np.sum((_act - _act.mean()) ** 2)
        _r2 = float(1 - np.sum((_pred - _act) ** 2) / max(_ss_tot, 1e-8))
        _mae_cum = np.mean(np.abs(_pred_abs - _act_abs), axis=0)
        return {
            "mse_overall": float(np.mean(_mse_window)),
            "mse_per_step": _mse_per_step,
            "mse_window": _mse_window,
            "r2": _r2,
            "mae_cum": _mae_cum,
            "last": _last,
            "act": _act,
            "pred": _pred,
        }

    results_2x2 = {
        ("synthetic_model", "synthetic_data"): _evaluate(model_syn, test_ds_syn),
        ("synthetic_model", "real_data"):      _evaluate(model_syn, test_ds_real),
        ("real_model",      "synthetic_data"): _evaluate(model_real, test_ds_syn),
        ("real_model",      "real_data"):      _evaluate(model_real, test_ds_real),
    }
    return


@app.cell
def _():
    _rows = []
    for (_mdl, _data), _r in results_2x2.items():
        _rows.append(f"| {_mdl} | {_data} | {_r['mse_overall']:.6f} | {_r['r2']:.4f} |")

    mo.md(f"""
## 2x2 Evaluation Summary

| Model | Test data | MSE | R² |
|-------|-----------|----:|---:|
{chr(10).join(_rows)}
""")
    return


@app.cell
def _():
    _hist_syn = bundle_syn.training_results.get("history", {})
    _hist_real = bundle_real.training_results.get("history", {})

    fig_train_curves, _ax = plt.subplots(figsize=(10, 5))
    if "val_loss" in _hist_syn:
        _ax.plot(_hist_syn["val_loss"], color="#4c72b0", label="Synthetic model (val)")
    if "val_loss" in _hist_real:
        _ax.plot(_hist_real["val_loss"], color="#dd8452", label="Real model (val)")
    if "train_loss" in _hist_syn:
        _ax.plot(_hist_syn["train_loss"], color="#4c72b0", alpha=0.3, linestyle="--", label="Synthetic model (train)")
    if "train_loss" in _hist_real:
        _ax.plot(_hist_real["train_loss"], color="#dd8452", alpha=0.3, linestyle="--", label="Real model (train)")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.set_title("Training curves overlay")
    _ax.set_yscale("log")
    _ax.legend()
    fig_train_curves.tight_layout()

    mo.md("## Training curves")
    return


@app.cell
def _():
    _steps = np.arange(1, F_shared + 1)

    fig_step_2x2, _axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    _titles = [
        ("synthetic_model", "synthetic_data", "Syn model + Syn data (in-domain)"),
        ("synthetic_model", "real_data",      "Syn model + Real data (transfer)"),
        ("real_model",      "synthetic_data", "Real model + Syn data (transfer)"),
        ("real_model",      "real_data",      "Real model + Real data (in-domain)"),
    ]
    for _idx, (_m, _d, _title) in enumerate(_titles):
        _ax = _axes[_idx // 2, _idx % 2]
        _r = results_2x2[(_m, _d)]
        _ax.bar(_steps, _r["mse_per_step"], color="#4c72b0" if "synthetic_model" in _m else "#dd8452")
        _ax.set_title(f"{_title}\nMSE={_r['mse_overall']:.6f}  R²={_r['r2']:.4f}", fontsize=10)
        _ax.set_xlabel("Forecast step")
        _ax.set_ylabel("MSE")
    fig_step_2x2.suptitle("Per-step MSE: model × data", fontsize=13)
    fig_step_2x2.tight_layout()

    mo.md("## Per-step MSE by model and data source")
    return


@app.cell
def _():
    _in_syn  = results_2x2[("synthetic_model", "synthetic_data")]["mse_overall"]
    _cross_syn_to_real = results_2x2[("synthetic_model", "real_data")]["mse_overall"]
    _in_real = results_2x2[("real_model", "real_data")]["mse_overall"]
    _cross_real_to_syn = results_2x2[("real_model", "synthetic_data")]["mse_overall"]

    fig_domain_gap, _ax = plt.subplots(figsize=(8, 5))
    _x = np.arange(2)
    _w = 0.35
    _ax.bar(_x - _w / 2, [_in_syn, _in_real], _w, label="In-domain", color="#4c72b0")
    _ax.bar(_x + _w / 2, [_cross_syn_to_real, _cross_real_to_syn], _w, label="Cross-domain", color="#dd8452")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(["Synthetic model", "Real model"])
    _ax.set_ylabel("MSE")
    _ax.set_title("Domain gap: in-domain vs cross-domain performance")
    _ax.legend()

    for _i, (_in, _cross) in enumerate([(_in_syn, _cross_syn_to_real), (_in_real, _cross_real_to_syn)]):
        _gap_pct = (_cross / _in - 1) * 100 if _in > 0 else 0
        _ax.text(_i + _w / 2, _cross, f"+{_gap_pct:.0f}%", ha="center", va="bottom", fontsize=9)

    fig_domain_gap.tight_layout()

    mo.md("## Domain gap analysis")
    return


@app.cell
def _():
    fig_cum_compare, _axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _steps = np.arange(1, F_shared + 1)
    _colors = {"synthetic_model": "#4c72b0", "real_model": "#dd8452"}

    for _di, (_data_label, _data_name) in enumerate([("Synthetic test data", "synthetic_data"), ("Real test data", "real_data")]):
        _ax = _axes[_di]
        for _m_label, _m_name in [("Syn model", "synthetic_model"), ("Real model", "real_model")]:
            _r = results_2x2[(_m_name, _data_name)]
            _ax.plot(_steps, _r["mae_cum"], "o-", color=_colors[_m_name], label=_m_label)
        _ax.set_xlabel("Forecast step")
        _ax.set_ylabel("MAE (absolute CNR)")
        _ax.set_title(_data_label)
        _ax.legend()

    fig_cum_compare.suptitle("Cumulative trajectory error by domain", fontsize=13)
    fig_cum_compare.tight_layout()

    mo.md("## Cumulative error comparison")
    return


@app.cell
def _():
    _stim_bins = 5

    fig_stim_compare, _axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _colors = {"synthetic_model": "#4c72b0", "real_model": "#dd8452"}

    for _di, (_data_label, _data_name, _ds) in enumerate([
        ("Synthetic test", "synthetic_data", test_ds_syn),
        ("Real test", "real_data", test_ds_real),
    ]):
        _ax = _axes[_di]
        for _m_label, _m_name in [("Syn model", "synthetic_model"), ("Real model", "real_model")]:
            _r = results_2x2[(_m_name, _data_name)]
            _mse_w = _r["mse_window"]
            _stim_mean = np.abs(_r["act"]).mean(axis=1)
            _edges = np.quantile(_stim_mean, np.linspace(0, 1, _stim_bins + 1))
            _edges[-1] += 1e-8
            _bin_centers, _bin_mses = [], []
            for _bi in range(_stim_bins):
                _mask = (_stim_mean >= _edges[_bi]) & (_stim_mean < _edges[_bi + 1])
                if _mask.sum() > 0:
                    _bin_centers.append((_edges[_bi] + _edges[_bi + 1]) / 2)
                    _bin_mses.append(_mse_w[_mask].mean())
            _ax.plot(_bin_centers, _bin_mses, "o-", color=_colors[_m_name], label=_m_label)
        _ax.set_xlabel("Mean |delta| (activity level)")
        _ax.set_ylabel("MSE")
        _ax.set_title(_data_label)
        _ax.legend()

    fig_stim_compare.suptitle("MSE by activity level", fontsize=13)
    fig_stim_compare.tight_layout()

    mo.md("## Performance by activity level")
    return


if __name__ == "__main__":
    app.run()
