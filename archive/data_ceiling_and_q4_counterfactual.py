import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from experiment import load_experiment
    from utils import get_device, KINGSTON_RESULTS_PATH
    from experiments.seq2seq_data import (
        load as load_dataset,
        STIM_COLS,
    )
    from experiments.lstm_seq2scal_mdn_strat_cosine import (
        Seq2ScalarMDN,
        Seq2SeqDataset,
    )

    # bundles save model_type as "__main__.Seq2ScalarMDN" (saved from a
    # marimo notebook where Seq2ScalarMDN was top-level). expose it on
    # __main__ here so reconstruct_model() resolves.
    sys.modules["__main__"].Seq2ScalarMDN = Seq2ScalarMDN


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from sklearn.model_selection import train_test_split
    from scipy.special import logsumexp as scipy_logsumexp
    from scipy.stats import pearsonr, spearmanr

    alt.data_transformers.disable_max_rows()

    device = get_device()
    n_stim = len(STIM_COLS)
    stim_col_names = list(STIM_COLS)
    return (
        alt,
        device,
        mo,
        n_stim,
        pearsonr,
        pl,
        scipy_logsumexp,
        spearmanr,
        stim_col_names,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Data ceiling and Q4-stratified counterfactual diagnostics

    Two diagnostics across three trained `Seq2ScalarMDN` checkpoints to
    decide whether the disappointing real-data counterfactual ratio
    (~0.305 / ~0.199) is a **data** problem or a **model** problem.

    - **Diagnostic 1.** Stim → ΔCNR correlation per response-magnitude
      quartile, model-free. Tests whether the realised stim-response
      coupling in real data is strong enough that any model could ever
      hit ratio ~1 on real Q4.
    - **Diagnostic 2.** Counterfactual `all_on` vs `all_off` ratio,
      stratified by response-magnitude quartile, **stratified-sampled**
      so Q4 gets a full-sized sample per model.
    """)
    return


@app.cell
def _(mo):
    CKPT_ROOT = Path(KINGSTON_RESULTS_PATH)

    CKPTS = {
        "real": {
            "path": CKPT_ROOT
            / "lstm_seq2scal_mdn_strat_cosine_k3_real_2026-04-29_17.48.58",
            "data_source": "real",
        },
        "synthetic_v2": {
            "path": CKPT_ROOT
            / "lstm_seq2scal_mdn_strat_cosine_k3_synthetic_v2_2026-04-24_17.05.42",
            "data_source": "synthetic_v2",
        },
        "pretrained": {
            "path": CKPT_ROOT
            / "lstm_seq2scal_mdn_strat_cosine_pretrained_k3_2026-04-29_17.48.58",
            "data_source": "real",
        },
    }

    _missing = [n for n, c in CKPTS.items() if not c["path"].exists()]
    if _missing:
        raise FileNotFoundError(
            f"Missing checkpoints under {CKPT_ROOT}: {_missing}"
        )

    mo.md(
        "**Checkpoints**\n\n"
        + "\n".join(
            f"- `{name}` → `{cfg['path'].name}` (data_source = `{cfg['data_source']}`)"
            for name, cfg in CKPTS.items()
        )
    )
    return (CKPTS,)


@app.cell
def _(mo):
    stim_summary_selector = mo.ui.dropdown(
        options=["integ_u", "delta_s_cum", "n_pulses"],
        value="integ_u",
        label="Stim summary for D1 scatter / Q4 markdown",
    )
    n_cf_per_bin = mo.ui.slider(
        50, 800, value=500, step=50, label="D2 windows per (model, bin)"
    )
    grid_seed = mo.ui.number(value=0, label="Q4 grid seed")
    mo.hstack([stim_summary_selector, n_cf_per_bin, grid_seed])
    return grid_seed, n_cf_per_bin, stim_summary_selector


@app.cell
def _(CKPTS, device, mo):
    BUNDLES = {}
    MODELS = {}
    for _name, _cfg in CKPTS.items():
        _b = load_experiment(str(_cfg["path"]))
        _m = _b.reconstruct_model().to(device).eval()
        BUNDLES[_name] = _b
        MODELS[_name] = _m

    _rows_md = []
    for _name, _b in BUNDLES.items():
        _mc = _b.model_config
        _rows_md.append(
            f"| `{_name}` | {_mc.get('history_len')} | {_mc.get('future_len')} | "
            f"`{_mc.get('data_source')}` | {sum(p.numel() for p in MODELS[_name].parameters()):,} |"
        )
    mo.md(
        "**Loaded models**\n\n"
        "| name | H | F | data_source | params |\n"
        "|---|---:|---:|---|---:|\n"
        + "\n".join(_rows_md)
    )
    return BUNDLES, MODELS


@app.cell
def _(BUNDLES, train_test_split):
    """Load test cnr/stim per unique data source. Real and pretrained
    share `data_source='real'` so the trajectories are loaded once."""

    _data_sources = {b.model_config["data_source"] for b in BUNDLES.values()}
    DATA = {}
    for _src in _data_sources:
        _cnr, _stim, _ = load_dataset(_src)
        _ids = np.arange(len(_cnr))
        _, _te = train_test_split(_ids, test_size=0.2, random_state=42)
        DATA[_src] = {
            "cnr_te": _cnr[_te],
            "stim_te": _stim[_te],
        }
    return (DATA,)


@app.cell
def _(BUNDLES, DATA, mo):
    """Build per-model test_ds. Stride matches each bundle's training_config
    (test_stride=10). H, F come from each model's model_config."""

    TEST_DS = {}
    for _name, _b in BUNDLES.items():
        _mc = _b.model_config
        _tcfg = _b.training_config
        _src = _mc["data_source"]
        _stride = _tcfg.get("test_stride", 10)
        _ds = Seq2SeqDataset(
            DATA[_src]["cnr_te"],
            DATA[_src]["stim_te"],
            _mc["history_len"],
            _mc["future_len"],
            stride=_stride,
        )
        TEST_DS[_name] = _ds

    mo.md(
        "**Test windows per model**\n\n"
        + "\n".join(
            f"- `{n}` → {len(d)} windows (H={BUNDLES[n].model_config['history_len']}, "
            f"F={BUNDLES[n].model_config['future_len']})"
            for n, d in TEST_DS.items()
        )
    )
    return (TEST_DS,)


@app.cell
def _(BUNDLES, TEST_DS, mo, pl):
    """Per-window response magnitude (std of full window CNR) and quartile
    bins. Same logic as the existing stratification cell. Computed
    per-model so the bins reflect that model's test set distribution."""

    _bin_labels = ["Q1 (flat)", "Q2", "Q3", "Q4 (responsive)"]

    BINS = {}
    _rows_b = []
    for _name, _ds in TEST_DS.items():
        _mc = BUNDLES[_name].model_config
        _resp_mag = np.empty(len(_ds), dtype=np.float32)
        for _i in range(len(_ds)):
            _enc_in, _dec_stim, _dec_target = _ds[_i]
            _hist_cnr = _enc_in[:, 0].numpy()
            _last = float(_hist_cnr[-1])
            _future = _last + np.cumsum(_dec_target.numpy())
            _full = np.concatenate([_hist_cnr, _future])
            _resp_mag[_i] = float(_full.std())
        _q = np.quantile(_resp_mag, [0.25, 0.5, 0.75])
        _bin_idx = np.digitize(_resp_mag, _q)
        _bin_name = np.array([_bin_labels[b] for b in _bin_idx])
        BINS[_name] = {
            "resp_std": _resp_mag,
            "edges": _q,
            "bin_idx": _bin_idx,
            "bin_name": _bin_name,
        }
        for _lbl_i, _lbl in enumerate(_bin_labels):
            _rows_b.append(
                {
                    "model": _name,
                    "bin": _lbl,
                    "n": int((_bin_idx == _lbl_i).sum()),
                    "edge_low": float(
                        _q[_lbl_i - 1] if _lbl_i > 0 else _resp_mag.min()
                    ),
                    "edge_high": float(
                        _q[_lbl_i] if _lbl_i < 3 else _resp_mag.max()
                    ),
                }
            )

    BIN_LABELS = _bin_labels
    bin_summary_df = pl.DataFrame(_rows_b)

    mo.vstack(
        [
            mo.md(
                "## Response-magnitude quartile bins per model\n\n"
                "`resp_std` = std of the full (H+F) CNR window. Edges are the "
                "25/50/75 quantiles of `resp_std` over each model's test set."
            ),
            bin_summary_df,
        ]
    )
    return BINS, BIN_LABELS


@app.cell
def _(mo):
    mo.md("""
    ## Diagnostic 1 — data ceiling (stim → ΔCNR correlation)

    Per test window, three scalar summaries of integrated stim over the
    forecast window `[H, H+F)`, plus realised ΔCNR over the same window:

    - `integ_u` = `sum(u_t[H:H+F])`
    - `delta_s_cum` = `s_cum[H+F-1] - s_cum[H-1]`
    - `n_pulses` = count of `u_t > 0.5` rising edges in `[H, H+F)`
    - `delta_cnr` = `cnr[H+F-1] - cnr[H-1]`

    Forecast slice has `F=10` for both real and synthetic. Stim
    channels in `STIM_COLS` order; `u_t` is index 0, `s_cum` is index 8.
    """)
    return


@app.cell
def _(BINS, BUNDLES, TEST_DS, pl, stim_col_names):
    """Build the per-window dataframe with raw stim summaries and Δcnr.

    For each window, recover the absolute slice indices into the source
    trajectory using the dataset's stored sample tensors. Stim channels
    are looked up by name to be order-agnostic.
    """

    _u_idx = stim_col_names.index("u_t")
    _scum_idx = stim_col_names.index("s_cum")

    _frames = []
    for _name, _ds in TEST_DS.items():
        _mc = BUNDLES[_name].model_config
        _src = _mc["data_source"]
        _bin_name = BINS[_name]["bin_name"]
        _resp = BINS[_name]["resp_std"]

        _integ_u = np.empty(len(_ds), dtype=np.float64)
        _delta_scum = np.empty(len(_ds), dtype=np.float64)
        _n_pulses = np.empty(len(_ds), dtype=np.int32)
        _delta_cnr = np.empty(len(_ds), dtype=np.float64)

        for _i in range(len(_ds)):
            _enc_in, _dec_stim, _dec_target = _ds.samples[_i]
            # _enc_in: (H, 1+9), col 0 = cnr, cols 1.. = STIM_COLS
            # _dec_stim: (F, 9) in STIM_COLS order
            _u_future = _dec_stim[:, _u_idx]
            _scum_future = _dec_stim[:, _scum_idx]
            _scum_last_hist = float(_enc_in[-1, 1 + _scum_idx])

            _integ_u[_i] = float(_u_future.sum())
            _delta_scum[_i] = float(_scum_future[-1] - _scum_last_hist)
            # pulse rising edges: u_t > 0.5 transitions from <=0.5
            _u_bin = (_u_future > 0.5).astype(np.int8)
            _prev_bin = 1 if float(_enc_in[-1, 1 + _u_idx]) > 0.5 else 0
            _full_bin = np.concatenate([[_prev_bin], _u_bin])
            _n_pulses[_i] = int(((_full_bin[1:] - _full_bin[:-1]) == 1).sum())

            _hist_cnr = _enc_in[:, 0]
            _future_abs = float(_hist_cnr[-1]) + np.cumsum(_dec_target)
            _delta_cnr[_i] = float(_future_abs[-1] - _hist_cnr[-1])

        _frames.append(
            pl.DataFrame(
                {
                    "model": [_name] * len(_ds),
                    "data_source": [_src] * len(_ds),
                    "window": np.arange(len(_ds), dtype=np.int64),
                    "bin": _bin_name,
                    "resp_std": _resp.astype(np.float64),
                    "integ_u": _integ_u,
                    "delta_s_cum": _delta_scum,
                    "n_pulses": _n_pulses,
                    "delta_cnr": _delta_cnr,
                }
            )
        )

    diag1_df = pl.concat(_frames)
    diag1_df.head(8)
    return (diag1_df,)


@app.cell
def _(BIN_LABELS, diag1_df, mo, pearsonr, pl, spearmanr):
    """Pearson + Spearman correlations of delta_cnr against each stim
    summary, per (data_source, bin). delta_cnr ↔ stim is data-only
    (model-free), so we deduplicate by data_source — pretrained shares
    real's test data, so its correlations are identical to real's."""

    _stim_keys = ["integ_u", "delta_s_cum", "n_pulses"]

    _rows_c = []
    for _src in sorted(diag1_df["data_source"].unique().to_list()):
        _src_df = diag1_df.filter(pl.col("data_source") == _src).unique(
            subset=["window", "data_source"]
        )
        for _bin in BIN_LABELS:
            _bdf = _src_df.filter(pl.col("bin") == _bin)
            _y = _bdf["delta_cnr"].to_numpy()
            for _k in _stim_keys:
                _x = _bdf[_k].to_numpy().astype(np.float64)
                if len(_x) >= 3 and np.std(_x) > 0 and np.std(_y) > 0:
                    _r_p = float(pearsonr(_x, _y).statistic)
                    _r_s = float(spearmanr(_x, _y).statistic)
                else:
                    _r_p = float("nan")
                    _r_s = float("nan")
                _rows_c.append(
                    {
                        "data_source": _src,
                        "bin": _bin,
                        "stim_summary": _k,
                        "n": int(len(_x)),
                        "pearson": round(_r_p, 3),
                        "spearman": round(_r_s, 3),
                    }
                )

    diag1_corr_df = pl.DataFrame(_rows_c).sort(["data_source", "bin", "stim_summary"])

    mo.vstack(
        [
            mo.md(
                "### D1 correlation table\n\n"
                "Rows: (data_source × bin × stim_summary). Pretrained shares "
                "real's test data, so its row is omitted to avoid duplication. "
                "Total rows = "
                f"{diag1_corr_df.height}."
            ),
            diag1_corr_df,
        ]
    )
    return (diag1_corr_df,)


@app.cell
def _(BIN_LABELS, alt, diag1_df, mo, pl, stim_summary_selector):
    """Faceted scatter: delta_cnr vs the selected stim summary, faceted
    by (data_source × bin). Subsample to ≤2000 points per panel so altair
    doesn't choke. Fitted line + R² shown via altair regression layer."""

    _stim = stim_summary_selector.value

    _max_per_panel = 2000
    _rng = np.random.default_rng(0)
    _parts = []
    for _src in sorted(diag1_df["data_source"].unique().to_list()):
        for _bin in BIN_LABELS:
            _sub = diag1_df.filter(
                (pl.col("data_source") == _src) & (pl.col("bin") == _bin)
            ).unique(subset=["window", "data_source"])
            if _sub.height > _max_per_panel:
                _idx = _rng.choice(_sub.height, _max_per_panel, replace=False)
                _sub = _sub[_idx]
            _parts.append(_sub)
    _plot_df = pl.concat(_parts).select(
        ["data_source", "bin", _stim, "delta_cnr"]
    )

    _bin_order = BIN_LABELS

    _points = (
        alt.Chart()
        .mark_circle(size=10, opacity=0.45)
        .encode(
            x=alt.X(f"{_stim}:Q", title=_stim),
            y=alt.Y("delta_cnr:Q", title="ΔCNR (forecast window)"),
        )
    )
    _line = (
        alt.Chart()
        .mark_line(color="#E45756", strokeWidth=1.5)
        .transform_regression(_stim, "delta_cnr")
        .encode(x=f"{_stim}:Q", y="delta_cnr:Q")
    )
    _r2 = (
        alt.Chart()
        .mark_text(align="left", baseline="top", dx=4, dy=4, color="#E45756")
        .transform_regression(_stim, "delta_cnr", params=True)
        .transform_calculate(label='"R² = " + format(datum.rSquared, ".3f")')
        .encode(
            x=alt.value(4),
            y=alt.value(4),
            text=alt.Text("label:N"),
        )
    )

    fig_d1_scatter = (
        alt.layer(_points, _line, _r2)
        .properties(width=200, height=160)
        .facet(
            column=alt.Column("data_source:N", title=None),
            row=alt.Row(
                "bin:N",
                title=None,
                sort=_bin_order,
                header=alt.Header(labelOrient="left"),
            ),
            data=_plot_df,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(title=f"ΔCNR vs {_stim} — per (data_source × bin)")
    )

    mo.vstack(
        [
            mo.md(
                f"### D1 scatter grid — `{_stim}`\n\n"
                "Each panel shows `delta_cnr` vs the selected stim summary "
                "for one (data_source × bin) cell, with an OLS fit (red) "
                "and R²."
            ),
            fig_d1_scatter,
        ]
    )
    return


@app.cell
def _(diag1_corr_df, mo, pl, stim_summary_selector):
    """Q4 Pearson(delta_cnr, integ_u) — real vs synthetic side-by-side."""

    _stim = stim_summary_selector.value
    _q4 = diag1_corr_df.filter(
        (pl.col("bin") == "Q4 (responsive)") & (pl.col("stim_summary") == _stim)
    )

    _real_r = _q4.filter(pl.col("data_source") == "real")["pearson"][0]
    _real_n = _q4.filter(pl.col("data_source") == "real")["n"][0]
    _synth_rows = _q4.filter(pl.col("data_source") == "synthetic_v2")
    if _synth_rows.height > 0:
        _synth_r = _synth_rows["pearson"][0]
        _synth_n = _synth_rows["n"][0]
    else:
        _synth_r = float("nan")
        _synth_n = 0

    _ratio_real_synth = (
        abs(_real_r) / abs(_synth_r) if _synth_r and not np.isnan(_synth_r) else float("nan")
    )

    diag1_q4_summary = {
        "stim_summary": _stim,
        "real_pearson_q4": float(_real_r),
        "real_n_q4": int(_real_n),
        "synthetic_pearson_q4": float(_synth_r),
        "synthetic_n_q4": int(_synth_n),
        "abs_ratio_real_over_synth": float(_ratio_real_synth),
    }

    mo.md(
        f"""
    ### D1 — Q4 correlation summary (`{_stim}`)

    | | real | synthetic_v2 |
    |---|---:|---:|
    | Q4 Pearson(ΔCNR, {_stim}) | **{_real_r:+.3f}** | **{_synth_r:+.3f}** |
    | n (Q4 windows) | {_real_n} | {_synth_n} |

    |real|/|synth| = **{_ratio_real_synth:.3f}**.
    """
    )
    return (diag1_q4_summary,)


@app.cell
def _(mo):
    mo.md("""
    **How to read.** If real Q4 |r| is substantially below synthetic Q4
    |r|, the realised stim → ΔCNR coupling in real data is weaker than
    in synthetic data — a *data ceiling*. If they are comparable, the
    signal is in the data and the model is leaving it on the table.

    Diagnostic 1 is model-free: it is purely a property of the data
    windows.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Diagnostic 2 — Q4-stratified counterfactual ratio

    Replicates the existing 500-window counterfactual with two changes:

    1. **Stratified sampling**: per (model, bin), sample up to N
       windows uniformly from that bin (not from all windows). Q4
       gets a full sample per model.
    2. **Per-model, per-bin reporting**: the existing analysis
       reports a single ratio per model; here we get four.

    Counterfactual `all_on` is constructed verbatim from the existing
    cell — per-channel max of the loaded test stim trajectories.
    """)
    return


@app.cell
def _(
    BINS,
    BIN_LABELS,
    BUNDLES,
    DATA,
    MODELS,
    TEST_DS,
    device,
    n_cf_per_bin,
    n_stim,
    pl,
    scipy_logsumexp,
):
    """For each (model, bin):
       - sample up to N_CF windows from that bin
       - run inference on actual / all_on / all_off
       - aggregate metrics
       Reuses _stim_max_arr construction verbatim from the existing
       counterfactual cell (per-channel max over the loaded test stim
       trajectories of the model's data source).
    """

    _N_CF = int(n_cf_per_bin.value)
    _rng = np.random.default_rng(0)

    def _stim_max_for_source(_src):
        _arr = np.zeros(n_stim, dtype=np.float32)
        for _s in DATA[_src]["stim_te"]:
            _arr = np.maximum(_arr, np.asarray(_s).max(axis=1))
        return _arr

    _stim_max_cache = {
        _src: _stim_max_for_source(_src) for _src in DATA.keys()
    }

    def _run_cf_subset(_model, _ds, _idx, _stim_max_arr, _cond):
        _pi_l, _mu_l, _sig_l, _y_l = [], [], [], []
        _subset = Subset(_ds, _idx)
        _model.eval()
        with torch.no_grad():
            for _eb, _sb, _tb in DataLoader(_subset, batch_size=256):
                _sb = _sb.clone()
                if _cond == "on":
                    _sb[:] = torch.tensor(_stim_max_arr).view(1, 1, n_stim)
                elif _cond == "off":
                    _sb.zero_()
                _pi, _mu, _sig = _model(_eb.to(device), _sb.to(device))
                _pi_l.append(_pi.cpu().numpy())
                _mu_l.append(_mu.cpu().numpy())
                _sig_l.append(_sig.cpu().numpy())
                _y_l.append(_tb.numpy())
        return tuple(np.concatenate(x) for x in (_pi_l, _mu_l, _sig_l, _y_l))

    def _nll_only(pi, mu, sig, y):
        y_ = y[..., None]
        log_g = (
            -0.5 * np.log(2 * np.pi)
            - np.log(sig)
            - 0.5 * ((y_ - mu) / sig) ** 2
        )
        return float(
            -scipy_logsumexp(np.log(pi + 1e-12) + log_g, axis=-1).mean()
        )

    _rows = []
    _sample_idx_by_cell = {}
    for _name in MODELS.keys():
        _m = MODELS[_name]
        _ds = TEST_DS[_name]
        _src = BUNDLES[_name].model_config["data_source"]
        _stim_max_arr = _stim_max_cache[_src]
        _bin_idx = BINS[_name]["bin_idx"]

        for _b_i, _b_lbl in enumerate(BIN_LABELS):
            _all_idx = np.where(_bin_idx == _b_i)[0]
            if len(_all_idx) == 0:
                continue
            _take = min(_N_CF, len(_all_idx))
            _sel = _rng.choice(_all_idx, _take, replace=False).tolist()
            _sample_idx_by_cell[(_name, _b_lbl)] = _sel

            _pi_a, _mu_a, _sig_a, _y_cf = _run_cf_subset(
                _m, _ds, _sel, _stim_max_arr, "actual"
            )
            _pi_on, _mu_on, _sig_on, _ = _run_cf_subset(
                _m, _ds, _sel, _stim_max_arr, "on"
            )
            _pi_off, _mu_off, _sig_off, _ = _run_cf_subset(
                _m, _ds, _sel, _stim_max_arr, "off"
            )

            _pt_on = (_pi_on * _mu_on).sum(-1)
            _pt_off = (_pi_off * _mu_off).sum(-1)
            _pp_diff = np.abs(_pt_on - _pt_off)
            _pp_mean = float(_pp_diff.mean())

            _top_on = _pi_on.argmax(-1)
            _top_off = _pi_off.argmax(-1)
            _ix_n = np.arange(_pi_on.shape[0])[:, None]
            _ix_f = np.arange(_pi_on.shape[1])[None, :]
            _top_mu_on = _mu_on[_ix_n, _ix_f, _top_on]
            _top_mu_off = _mu_off[_ix_n, _ix_f, _top_off]
            _top_mu_diff = float(np.abs(_top_mu_on - _top_mu_off).mean())

            _m_mix = 0.5 * (_pi_on + _pi_off)
            _js = 0.5 * (
                _pi_on * np.log((_pi_on + 1e-12) / (_m_mix + 1e-12))
            ).sum(-1) + 0.5 * (
                _pi_off * np.log((_pi_off + 1e-12) / (_m_mix + 1e-12))
            ).sum(-1)
            _js_mean = float(_js.mean())
            _top_flip = float((_top_on != _top_off).mean())

            _y_std = float(_y_cf.std())

            _nll_a = _nll_only(_pi_a, _mu_a, _sig_a, _y_cf)
            _nll_on = _nll_only(_pi_on, _mu_on, _sig_on, _y_cf)
            _nll_off = _nll_only(_pi_off, _mu_off, _sig_off, _y_cf)

            _rows.append(
                {
                    "model": _name,
                    "bin": _b_lbl,
                    "n_windows": int(_take),
                    "mean_abs_point_diff_on_off": _pp_mean,
                    "target_std": _y_std,
                    "ratio_point_diff_over_std": _pp_mean / max(_y_std, 1e-12),
                    "mean_abs_top_mu_diff_on_off": _top_mu_diff,
                    "mean_js_pi_on_off": _js_mean,
                    "frac_top_component_flips": _top_flip,
                    "nll_actual": _nll_a,
                    "nll_all_on": _nll_on,
                    "nll_all_off": _nll_off,
                }
            )

    diag2_df = (
        pl.DataFrame(_rows)
        .sort(["model", "bin"])
        .with_columns(
            [
                pl.col("mean_abs_point_diff_on_off").round(5),
                pl.col("target_std").round(5),
                pl.col("ratio_point_diff_over_std").round(4),
                pl.col("mean_abs_top_mu_diff_on_off").round(5),
                pl.col("mean_js_pi_on_off").round(4),
                pl.col("frac_top_component_flips").round(3),
                pl.col("nll_actual").round(4),
                pl.col("nll_all_on").round(4),
                pl.col("nll_all_off").round(4),
            ]
        )
    )
    DIAG2_SAMPLE_IDX = _sample_idx_by_cell
    diag2_df
    return (diag2_df,)


@app.cell
def _(BIN_LABELS, alt, diag2_df, mo):
    """Bar plot: ratio_point_diff_over_std per (bin, model) with reference
    line at y=1 marking the "MPC plausible" threshold."""

    _bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "ratio_point_diff_over_std:Q",
                title="ratio (point diff / target std)",
            ),
            color=alt.Color("model:N", legend=None),
            tooltip=[
                "model",
                "bin",
                "n_windows",
                "ratio_point_diff_over_std",
                "mean_abs_point_diff_on_off",
                "target_std",
            ],
        )
    )
    _ref_line = (
        alt.Chart()
        .mark_rule(strokeDash=[4, 3], color="#888", strokeWidth=1)
        .encode(y=alt.datum(1.0))
    )

    fig_d2_ratio = (
        alt.layer(_bars, _ref_line)
        .properties(width=160, height=240)
        .facet(
            column=alt.Column("bin:N", title=None, sort=BIN_LABELS),
            data=diag2_df,
        )
        .properties(
            title="Counterfactual ratio per (model, bin) — y=1 is MPC plausible"
        )
    )

    mo.vstack(
        [mo.md("### D2 ratio bar plot"), fig_d2_ratio]
    )
    return


@app.cell
def _(BIN_LABELS, alt, diag2_df, mo):
    """JS divergence bar plot: did the mixture weights become
    stim-conditioned for responsive cells?"""

    _bars_js = (
        alt.Chart(diag2_df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean_js_pi_on_off:Q", title="mean JS(π_on ‖ π_off)"),
            color=alt.Color("model:N", legend=None),
            tooltip=[
                "model",
                "bin",
                "mean_js_pi_on_off",
                "frac_top_component_flips",
            ],
        )
        .properties(width=160, height=220)
        .facet(column=alt.Column("bin:N", title=None, sort=BIN_LABELS))
        .properties(
            title="Mixture-weight shift on↔off — bigger = stim-conditioned mode assignment"
        )
    )

    mo.vstack([mo.md("### D2 JS-divergence bar plot"), _bars_js])
    return


@app.cell
def _(
    BINS,
    BUNDLES,
    DATA,
    MODELS,
    TEST_DS,
    alt,
    device,
    grid_seed,
    mo,
    n_stim,
    pl,
):
    """Q4 counterfactual grid: for each model, pick 4 Q4 windows from the
    D2-sampled set (so they were just used in the metrics) and show
    history + truth-future + actual/all_on/all_off rollouts.

    Layout: rows = window slot (0..3), columns = model.
    """

    _Q4 = "Q4 (responsive)"
    _N_PANELS = 4

    _rng = np.random.default_rng(int(grid_seed.value))
    _grid_cells = {}
    for _name in MODELS.keys():
        _bin_idx = BINS[_name]["bin_idx"]
        _q4_all = np.where(_bin_idx == 3)[0]
        if len(_q4_all) == 0:
            continue
        _take = min(_N_PANELS, len(_q4_all))
        _sel = _rng.choice(_q4_all, _take, replace=False)
        _grid_cells[_name] = list(map(int, _sel))

    def _stim_max_for_source(_src):
        _arr = np.zeros(n_stim, dtype=np.float32)
        for _s in DATA[_src]["stim_te"]:
            _arr = np.maximum(_arr, np.asarray(_s).max(axis=1))
        return _arr

    _rows_g = []
    for _name, _wins in _grid_cells.items():
        _model = MODELS[_name]
        _ds = TEST_DS[_name]
        _mc = BUNDLES[_name].model_config
        _src = _mc["data_source"]
        _H = _mc["history_len"]
        _F = _mc["future_len"]
        _stim_max_t = torch.tensor(
            _stim_max_for_source(_src)
        ).view(1, 1, n_stim)

        _model.eval()
        with torch.no_grad():
            for _slot, _wi in enumerate(_wins):
                _enc_in, _dec_stim, _dec_target = _ds[int(_wi)]
                _enc_b = _enc_in.unsqueeze(0).to(device)
                _s_act = _dec_stim.unsqueeze(0).to(device)
                _s_on = _stim_max_t.repeat(1, _F, 1).to(device)
                _s_off = torch.zeros_like(_s_act)

                _hist = _enc_in[:, 0].numpy()
                _last_v = float(_hist[-1])
                _act_abs = _last_v + np.cumsum(_dec_target.numpy())
                for _t, _v in enumerate(_hist):
                    _rows_g.append(
                        dict(
                            model=_name,
                            slot=_slot,
                            window=int(_wi),
                            t=int(_t),
                            cnr=float(_v),
                            cond="history",
                        )
                    )
                for _t in range(_F):
                    _rows_g.append(
                        dict(
                            model=_name,
                            slot=_slot,
                            window=int(_wi),
                            t=int(_H + _t),
                            cnr=float(_act_abs[_t]),
                            cond="truth",
                        )
                    )

                for _cname, _s in [
                    ("actual", _s_act),
                    ("all_on", _s_on),
                    ("all_off", _s_off),
                ]:
                    _pi, _mu, _sig = _model(_enc_b, _s)
                    _pi_np = _pi.cpu().numpy()[0]
                    _mu_np = _mu.cpu().numpy()[0]
                    _pt = (_pi_np * _mu_np).sum(-1)
                    _abs_pt = _last_v + np.cumsum(_pt)
                    _rows_g.append(
                        dict(
                            model=_name,
                            slot=_slot,
                            window=int(_wi),
                            t=int(_H - 1),
                            cnr=_last_v,
                            cond=_cname,
                        )
                    )
                    for _t in range(_F):
                        _rows_g.append(
                            dict(
                                model=_name,
                                slot=_slot,
                                window=int(_wi),
                                t=int(_H + _t),
                                cnr=float(_abs_pt[_t]),
                                cond=_cname,
                            )
                        )

    cf_grid_q4_df = pl.DataFrame(_rows_g)

    _colors = {
        "history": "#2c3e50",
        "truth": "#000000",
        "actual": "#4C78A8",
        "all_on": "#E45756",
        "all_off": "#54A24B",
    }
    _dom = list(_colors.keys())
    _rng_c = [_colors[k] for k in _dom]
    _enc_c = alt.Color(
        "cond:N",
        scale=alt.Scale(domain=_dom, range=_rng_c),
        legend=alt.Legend(title="series"),
    )

    _line_main = (
        alt.Chart()
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("t:Q", title="t"),
            y=alt.Y("cnr:Q", title="CNR", scale=alt.Scale(zero=False)),
            color=_enc_c,
            detail="cond:N",
            tooltip=["model", "window", "cond", "t", "cnr"],
        )
        .transform_filter(alt.datum.cond != "truth")
    )
    _line_truth = (
        alt.Chart()
        .mark_line(strokeWidth=2, strokeDash=[4, 3])
        .encode(
            x="t:Q", y="cnr:Q", color=_enc_c, detail="cond:N",
        )
        .transform_filter(alt.datum.cond == "truth")
    )

    fig_cf_q4_grid = (
        alt.layer(_line_main, _line_truth)
        .properties(width=220, height=140)
        .facet(
            row=alt.Row(
                "slot:N",
                title=None,
                header=alt.Header(labelOrient="left", labelExpr=""),
            ),
            column=alt.Column("model:N", title=None),
            data=cf_grid_q4_df,
        )
        .resolve_scale(y="independent")
        .properties(title="Q4 counterfactual grid — 3 models × 4 Q4 windows")
    )

    mo.vstack(
        [
            mo.md(
                "### Q4 counterfactual grid\n\n"
                "Same windows are not shared across models (each model's Q4 "
                "set comes from its own test windows; pretrained and real "
                "share data so their pools overlap)."
            ),
            fig_cf_q4_grid,
        ]
    )
    return (cf_grid_q4_df,)


@app.cell
def _(cf_grid_q4_df, diag1_corr_df, diag1_df, diag2_df, mo):
    """Save result frames next to the notebook for follow-up analysis."""

    _out_dir = Path(__file__).resolve().parent / "data_ceiling_q4_outputs"
    _out_dir.mkdir(exist_ok=True)

    diag1_df.write_parquet(_out_dir / "diag1_per_window.parquet")
    diag1_corr_df.write_parquet(_out_dir / "diag1_correlation_table.parquet")
    diag2_df.write_parquet(_out_dir / "diag2_counterfactual_metrics.parquet")
    cf_grid_q4_df.write_parquet(_out_dir / "diag2_q4_grid_trajectories.parquet")

    mo.md(
        f"**Saved parquets to `{_out_dir}`**\n\n"
        f"- `diag1_per_window.parquet` ({diag1_df.height} rows)\n"
        f"- `diag1_correlation_table.parquet` ({diag1_corr_df.height} rows)\n"
        f"- `diag2_counterfactual_metrics.parquet` ({diag2_df.height} rows)\n"
        f"- `diag2_q4_grid_trajectories.parquet` ({cf_grid_q4_df.height} rows)"
    )
    return


@app.cell
def _(diag1_q4_summary, diag2_df, mo, pl):
    """Final summary — numbers only, no interpretation."""

    _q4_d2 = diag2_df.filter(pl.col("bin") == "Q4 (responsive)").sort("model")
    _q4_ratio = {
        r["model"]: float(r["ratio_point_diff_over_std"])
        for r in _q4_d2.iter_rows(named=True)
    }
    _q4_js = {
        r["model"]: float(r["mean_js_pi_on_off"])
        for r in _q4_d2.iter_rows(named=True)
    }

    _rl = diag1_q4_summary["real_pearson_q4"]
    _sy = diag1_q4_summary["synthetic_pearson_q4"]
    _stim = diag1_q4_summary["stim_summary"]

    _ratio_strs = ", ".join(f"{n}={_q4_ratio.get(n, float('nan')):.3f}" for n in ["real", "pretrained", "synthetic_v2"] if n in _q4_ratio)
    _js_strs = ", ".join(f"{n}={_q4_js.get(n, float('nan')):.3f}" for n in ["real", "pretrained", "synthetic_v2"] if n in _q4_js)

    mo.md(
        f"""
    ## Summary (numbers only)

    - **D1 Q4 Pearson(ΔCNR, {_stim})**: real = {_rl:+.3f}, synthetic_v2 = {_sy:+.3f}; |real|/|synth| = {abs(_rl) / max(abs(_sy), 1e-9):.3f}.
    - **D2 Q4 ratio_point_diff_over_std** (per model): {_ratio_strs}.
    - **D2 Q4 mean_js_pi_on_off** (per model): {_js_strs}.
    - D1 correlation table written with {len(['integ_u', 'delta_s_cum', 'n_pulses'])} stim summaries × 4 bins × {len(set(_q4_ratio.keys()) & {'real', 'pretrained', 'synthetic_v2'}) and 2} unique data sources.
    """
    )
    return


if __name__ == "__main__":
    app.run()
