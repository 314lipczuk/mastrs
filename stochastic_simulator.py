import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from notebooks.experiment.preprocessing import augment

    return augment, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Looking at the original data
    To quantify how much stochasticity to introduce.
    """)
    return


@app.cell(hide_code=True)
def _(augment, pd):
    df = augment(pd.read_parquet('dataset.parquet'))
    df.info()
    return (df,)


@app.cell(hide_code=True)
def _(df, plt):
    df['median_cnr_0_9'].hist(bins=50)
    plt.xlabel('median CNR (frames 0-9)')
    plt.ylabel('count')
    plt.title('Baseline ERK activity distribution')
    plt.gca()
    return


@app.cell
def _(df):
    df['uid'].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interesting shape; I wonder if i filter harder for noisy cells in the baseline (w.r.t to area), would the right part disappear and the entire thing turn to gaussian
    """)
    return


@app.cell
def _(df, plt):
    _high_baseline = df[df['median_cnr_0_9'] > 0.7]
    _high_uids = _high_baseline['uid'].unique()
    _rng = __import__('numpy').random.default_rng(42)
    _sample_uids = _rng.choice(_high_uids, size=min(10, len(_high_uids)), replace=False)

    _cmap = plt.cm.tab20
    _colors = {uid: _cmap(i / len(_sample_uids)) for i, uid in enumerate(_sample_uids)}

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    _titles = ['CNR (median, normalized)', 'Area', 'Nucleus intensity', 'Cytoplasm (ring) intensity']
    _cols = ['cnr_median_norm', 'area', 'median_intensity_C1_nuc', 'median_intensity_C1_ring']

    for _uid in _sample_uids:
        _cell = df[df['uid'] == _uid].sort_values('frame')
        _c = _colors[_uid]
        for _ax, _col in zip(_axes.flat, _cols):
            _ax.plot(_cell['frame'], _cell[_col], color=_c, alpha=0.7, lw=1)

    for _ax, _title in zip(_axes.flat, _titles):
        _ax.set_title(_title)
        _ax.axvline(10, color='gray', ls=':', lw=1, alpha=0.5)

    _axes[1, 0].set_xlabel('frame')
    _axes[1, 1].set_xlabel('frame')
    _fig.suptitle(f'High-baseline cells (median CNR₀₋₉ > 0.7, n={len(_sample_uids)})', y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Noise at three scales

    1. **Dataset-level** — pooled std of `cnr_median_norm` during baseline (frames 0-9) across all cells. Single scalar representing overall measurement noise.
    2. **Trajectory-level** — per-cell std of `cnr_median_norm` during baseline. Shows how noise varies across cells.
    3. **Rolling-window** — per-cell, per-timepoint std in a sliding window. Captures how local variability evolves over time (e.g. noisier during stimulation vs quiet baseline).
    """)
    return


@app.cell
def _(df, pd):
    baseline = df[df['frame'] < 10]

    dataset_noise = baseline['cnr_median_norm'].std()

    trajectory_noise = baseline.groupby('uid')['cnr_median_norm'].std().rename('trajectory_std')

    window_size = 10
    df_sorted = df.sort_values(['uid', 'frame'])
    rolling_std = (
        df_sorted.groupby('uid')['cnr_median_norm']
        .transform(lambda s: s.rolling(window_size, min_periods=3, center=True).std())
    )
    df_sorted = df_sorted.assign(rolling_noise=rolling_std)

    rolling_by_frame = df_sorted.groupby('frame')['rolling_noise'].agg(['median', 'mean', 'std']).rename(
        columns={'median': 'rolling_median', 'mean': 'rolling_mean', 'std': 'rolling_std'}
    )

    noise_summary = pd.DataFrame({
        'level': ['dataset', 'trajectory (median)', 'trajectory (mean)', 'trajectory (IQR)'],
        'value': [
            f'{dataset_noise:.4f}',
            f'{trajectory_noise.median():.4f}',
            f'{trajectory_noise.mean():.4f}',
            f'{trajectory_noise.quantile(0.25):.4f} – {trajectory_noise.quantile(0.75):.4f}',
        ]
    })
    return dataset_noise, noise_summary, rolling_by_frame, trajectory_noise


@app.cell
def _(noise_summary):
    noise_summary
    return


@app.cell
def _(dataset_noise, plt, trajectory_noise):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(trajectory_noise.values, bins=80, edgecolor='white', alpha=0.8)
    _ax.axvline(dataset_noise, color='red', ls='--', lw=2, label=f'dataset pooled σ = {dataset_noise:.4f}')
    _ax.axvline(trajectory_noise.median(), color='orange', ls='--', lw=2, label=f'trajectory median σ = {trajectory_noise.median():.4f}')
    _ax.set_xlabel('baseline std (cnr_median_norm)')
    _ax.set_ylabel('number of cells')
    _ax.set_title('Trajectory-level noise distribution (frames 0-9)')
    _ax.set_xlim(0, trajectory_noise.quantile(0.99))
    _ax.legend()
    _fig
    return


@app.cell
def _(dataset_noise, np, plt, rolling_by_frame):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _frames = rolling_by_frame.index.values
    _median_vals = rolling_by_frame['rolling_median'].values
    _std_vals = rolling_by_frame['rolling_std'].values

    _ax.plot(_frames, _median_vals, color='steelblue', lw=2, label='median rolling σ across cells')
    _ax.fill_between(
        _frames,
        np.maximum(_median_vals - _std_vals, 0),
        _median_vals + _std_vals,
        alpha=0.2, color='steelblue', label='±1 std'
    )
    _ax.axhline(dataset_noise, color='red', ls='--', lw=1.5, label=f'dataset pooled σ = {dataset_noise:.4f}')
    _ax.axvline(10, color='gray', ls=':', lw=1, label='stim onset (frame 10)')
    _ax.set_xlabel('frame')
    _ax.set_ylabel('rolling std (window=10)')
    _ax.set_title('Rolling noise over time (median ± std across cells)')
    _ax.legend(loc='upper right')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sensitivity and responder analysis
    """)
    return


@app.cell
def _(df, plt):
    _cell_summary = df.drop_duplicates('uid')

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    _axes[0].hist(_cell_summary['amplitude'].values, bins=60, edgecolor='white', alpha=0.8)
    _axes[0].set_xlabel('amplitude (peak CNR − 1)')
    _axes[0].set_ylabel('cells')
    _axes[0].set_title('Response amplitude distribution')

    _stimulated = _cell_summary[_cell_summary['total_fluence'] > 0]
    _axes[1].scatter(_stimulated['total_fluence'], _stimulated['amplitude'], alpha=0.3, s=10)
    _axes[1].set_xlabel('total fluence (mJ/cm²)')
    _axes[1].set_ylabel('amplitude')
    _axes[1].set_title('Amplitude vs total dose')

    _resp_by_exp = _cell_summary.groupby('ramp_pattern_name')['responder'].mean()
    _resp_by_exp.plot(kind='bar', ax=_axes[2], edgecolor='white', alpha=0.8)
    _axes[2].set_ylabel('fraction responders')
    _axes[2].set_title('Responder fraction by experiment')
    _axes[2].set_ylim(0, 1)
    _axes[2].tick_params(axis='x', rotation=30)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df, plt):
    _cs = df.drop_duplicates('uid')

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].hist(_cs['baseline_nuc'].values, bins=60, edgecolor='white', alpha=0.7, label='nucleus')
    _axes[0].hist(_cs['baseline_ring'].values, bins=60, edgecolor='white', alpha=0.7, label='ring (cytoplasm)')
    _axes[0].set_xlabel('median intensity (frames 0-9)')
    _axes[0].set_ylabel('cells')
    _axes[0].set_title('Baseline component distributions')
    _axes[0].legend()

    _axes[1].scatter(_cs['baseline_nuc'], _cs['baseline_ring'], alpha=0.3, s=10)
    _axes[1].set_xlabel('baseline nucleus intensity')
    _axes[1].set_ylabel('baseline ring intensity')
    _axes[1].set_title('Ring vs nucleus baseline')
    _axes[1].plot([0, _cs['baseline_nuc'].max()], [0, _cs['baseline_nuc'].max()], 'r--', lw=1, alpha=0.5)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stratified responder analysis

    Responder fraction broken down **by experiment × stim_exposure** (the per-pulse
    exposure in ms that each cell receives) and **by experiment × cumulative fluence bin**
    (how much total light the cell has received up to its peak frame).
    """)
    return


@app.cell
def _(df, np, pd):
    _cells = df.drop_duplicates('uid')

    _stim_frames = df[df['m_t'] == 1]
    _exposure_per_cell = (
        _stim_frames.groupby('uid')['stim_exposure']
        .apply(lambda s: tuple(sorted(s.unique())))
        .rename('exposure_set')
    )
    _cells = _cells.merge(_exposure_per_cell, on='uid', how='left')
    _cells['exposure_set'] = _cells['exposure_set'].fillna('no stim')

    _cells_with_single_exposure = _cells[_cells['exposure_set'].apply(
        lambda x: isinstance(x, tuple) and len(x) == 1
    )].copy()
    _cells_with_single_exposure['stim_exposure_ms'] = _cells_with_single_exposure['exposure_set'].apply(lambda x: x[0])

    resp_by_exp_exposure = (
        _cells_with_single_exposure
        .groupby(['ramp_pattern_name', 'stim_exposure_ms'])
        .agg(
            n_cells=('responder', 'count'),
            n_responders=('responder', 'sum'),
            frac_responders=('responder', 'mean'),
            mean_amplitude=('amplitude', 'mean'),
        )
        .reset_index()
    )

    _peak_frame = df.loc[df.groupby('uid')['cnr_median_norm'].idxmax()][['uid', 's_cum']].rename(columns={'s_cum': 'fluence_at_peak'})
    _cells_fluence = _cells.merge(_peak_frame, on='uid', how='left')

    _bins = np.array([0, 1, 5, 10, 25, 50, 100, 200, 500])
    _bins = _bins[_bins <= _cells_fluence['fluence_at_peak'].max() * 1.01]
    _bins = np.append(_bins, _cells_fluence['fluence_at_peak'].max() + 1)
    _cells_fluence['fluence_bin'] = pd.cut(_cells_fluence['fluence_at_peak'], bins=_bins, right=False)

    resp_by_exp_fluence = (
        _cells_fluence
        .groupby(['ramp_pattern_name', 'fluence_bin'], observed=True)
        .agg(
            n_cells=('responder', 'count'),
            n_responders=('responder', 'sum'),
            frac_responders=('responder', 'mean'),
            mean_amplitude=('amplitude', 'mean'),
        )
        .reset_index()
    )
    return resp_by_exp_exposure, resp_by_exp_fluence


@app.cell
def _(resp_by_exp_exposure):
    resp_by_exp_exposure
    return


@app.cell
def _(plt, resp_by_exp_exposure):
    _experiments = resp_by_exp_exposure['ramp_pattern_name'].unique()
    _fig, _axes = plt.subplots(1, len(_experiments), figsize=(5 * len(_experiments), 4), sharey=True)
    if len(_experiments) == 1:
        _axes = [_axes]

    for _ax, _exp in zip(_axes, _experiments):
        _sub = resp_by_exp_exposure[resp_by_exp_exposure['ramp_pattern_name'] == _exp].sort_values('stim_exposure_ms')
        _bars = _ax.bar(
            range(len(_sub)), _sub['frac_responders'],
            edgecolor='white', alpha=0.8,
        )
        for _i, (_bar, _n) in enumerate(zip(_bars, _sub['n_cells'])):
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.02,
                     f'n={_n}', ha='center', va='bottom', fontsize=8)
        _ax.set_xticks(range(len(_sub)))
        _ax.set_xticklabels([f'{v:.0f}' for v in _sub['stim_exposure_ms']], rotation=45)
        _ax.set_xlabel('stim exposure (ms)')
        _ax.set_title(_exp)
        _ax.set_ylim(0, 1.15)

    _axes[0].set_ylabel('fraction responders')
    _fig.suptitle('Responder fraction by experiment × stim exposure', y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(resp_by_exp_fluence):
    resp_by_exp_fluence
    return


@app.cell
def _(plt, resp_by_exp_fluence):
    _experiments = resp_by_exp_fluence['ramp_pattern_name'].unique()
    _fig, _axes = plt.subplots(1, len(_experiments), figsize=(5 * len(_experiments), 4), sharey=True)
    if len(_experiments) == 1:
        _axes = [_axes]

    for _ax, _exp in zip(_axes, _experiments):
        _sub = resp_by_exp_fluence[resp_by_exp_fluence['ramp_pattern_name'] == _exp]
        if _sub.empty:
            continue
        _bars = _ax.bar(
            range(len(_sub)), _sub['frac_responders'],
            edgecolor='white', alpha=0.8,
        )
        for _i, (_bar, _n) in enumerate(zip(_bars, _sub['n_cells'])):
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.02,
                     f'n={_n}', ha='center', va='bottom', fontsize=8)
        _ax.set_xticks(range(len(_sub)))
        _ax.set_xticklabels([str(_sub['fluence_bin'].iloc[i]) for i in range(len(_sub))], rotation=45, fontsize=7)
        _ax.set_xlabel('cumulative fluence at peak (mJ/cm²)')
        _ax.set_title(_exp)
        _ax.set_ylim(0, 1.15)

    _axes[0].set_ylabel('fraction responders')
    _fig.suptitle('Responder fraction by experiment × cumulative fluence bin', y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df):
    df['ramp_pattern_name'].unique()
    return


@app.cell
def _(df, np, plt):
    _sustained = df
    _rng = np.random.default_rng(42)
    _uids = _sustained['uid'].unique()
    _sample_uids = _rng.choice(_uids, size=min(10, len(_uids)), replace=False)

    _fig, _ax = plt.subplots(figsize=(12, 5))
    _cmap = plt.cm.tab10
    for _i, _uid in enumerate(_sample_uids):
        _cell = _sustained[_sustained['uid'] == _uid].sort_values('frame')
        _ax.plot(_cell['frame'], _cell['cnr_median_norm'], color=_cmap(_i / 10), alpha=0.8, lw=1.2, label=str(_uid)[:8])

    _ax.axvline(10, color='gray', ls=':', lw=1, alpha=0.5, label='stim onset')
    _ax.set_xlabel('frame')
    _ax.set_ylabel('CNR (median, normalized)')
    _ax.set_title(f'Sample trajectories (n={len(_sample_uids)})')
    _ax.legend(fontsize=7, ncol=2, loc='upper right')
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature correlations

    Spearman correlations between the five features available in real data,
    computed on baseline frames (0–9) to avoid stimulation-driven covariance.
    Density = number of cell neighbours within an 80-pixel radius.
    """)
    return


@app.cell
def _(df, np, pd):
    from scipy.spatial import cKDTree as _cKDTree

    _RADIUS = 1000  # pixels

    _baseline = df[df['frame'] < 10].copy()

    def _count_nb(grp):
        xy = grp[['x', 'y']].values
        if len(xy) < 2:
            return pd.Series(np.zeros(len(grp), dtype=int), index=grp.index)
        tree = _cKDTree(xy)
        counts = tree.query_ball_point(xy, r=_RADIUS, return_length=True) - 1
        return pd.Series(counts, index=grp.index)

    _nb = _baseline.groupby(['fov', 'frame'], group_keys=False).apply(_count_nb)
    _baseline = _baseline.assign(neighbour_count=_nb)

    _feat_cols = {
        'cnr_median_norm':         'CNR',
        'median_intensity_C1_nuc': 'Nuc intensity',
        'median_intensity_C1_ring':'Cyt intensity',
        'area':                    'Area',
        'neighbour_count':         'Density',
    }
    corr_features = (
        _baseline.groupby('uid')[list(_feat_cols.keys())].median()
        .rename(columns=_feat_cols)
        .dropna()
    )
    corr_matrix = corr_features.corr(method='spearman')
    return corr_features, corr_matrix


@app.cell
def _(corr_features, corr_matrix, plt):
    import matplotlib.gridspec as _gs

    _cols = list(corr_matrix.columns)
    _n = len(_cols)
    _data = corr_features.values

    _fig = plt.figure(figsize=(20, 8))
    _outer = _gs.GridSpec(1, 2, figure=_fig, width_ratios=[1, 1.8], wspace=0.25)

    # --- annotated heatmap ---
    _ax_h = _fig.add_subplot(_outer[0])
    _im = _ax_h.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap='RdBu_r')
    _ax_h.set_xticks(range(_n))
    _ax_h.set_yticks(range(_n))
    _ax_h.set_xticklabels(_cols, rotation=40, ha='right', fontsize=9)
    _ax_h.set_yticklabels(_cols, fontsize=9)
    for _i in range(_n):
        for _j in range(_n):
            _v = corr_matrix.values[_i, _j]
            _ax_h.text(_j, _i, f'{_v:.2f}', ha='center', va='center',
                       fontsize=8, color='white' if abs(_v) > 0.5 else 'black')
    _fig.colorbar(_im, ax=_ax_h, fraction=0.046, pad=0.04, label='Spearman r')
    _ax_h.set_title('Spearman r  (baseline frames 0–9, per-cell median)', fontsize=10)

    # --- scatter matrix (lower=scatter, diag=hist, upper=r text) ---
    _inner = _gs.GridSpecFromSubplotSpec(_n, _n, subplot_spec=_outer[1], hspace=0.08, wspace=0.08)
    for _i in range(_n):
        for _j in range(_n):
            _ax_s = _fig.add_subplot(_inner[_i, _j])
            if _i == _j:
                _col = _data[:, _i]
                try:
                    _ax_s.hist(_col, bins=25, color='steelblue', alpha=0.7, lw=0)
                except ValueError:
                    _ax_s.axvline(_col.mean(), color='steelblue', lw=2)
            elif _i > _j:
                _ax_s.scatter(_data[:, _j], _data[:, _i],
                               s=2, alpha=0.2, color='steelblue', rasterized=True)
            else:
                _v = corr_matrix.values[_i, _j]
                _rgba = plt.cm.RdBu_r((_v + 1) / 2)
                _ax_s.set_facecolor((*_rgba[:3], 0.15))
                _ax_s.text(0.5, 0.5, f'r={_v:.2f}', ha='center', va='center',
                            transform=_ax_s.transAxes, fontsize=8,
                            fontweight='bold' if abs(_v) > 0.3 else 'normal',
                            color='crimson' if abs(_v) > 0.3 else 'gray')
            _ax_s.tick_params(labelsize=5, length=2)
            if _j > 0:
                _ax_s.set_yticks([])
            if _i < _n - 1:
                _ax_s.set_xticks([])
            if _j == 0:
                _ax_s.set_ylabel(_cols[_i], fontsize=7)
            if _i == _n - 1:
                _ax_s.set_xlabel(_cols[_j], fontsize=7)

    _fig.suptitle('Feature correlation analysis', fontsize=12, y=1.01)
    _fig
    return


@app.cell
def _(np):
    def sample_image_quality(n_cells, rng=None):
        # TODO: replace with image-based quantification — e.g. Laplacian variance
        # of the per-cell bounding-box crop, Cellpose flow confidence, or
        # per-cell SNR estimated from background pixels in the raw TIFF.
        if rng is None:
            rng = np.random.default_rng()
        # Heuristic: Beta(5, 2) — most cells score ~0.7–0.9, a tail of poor-quality ones.
        return rng.beta(a=5, b=2, size=n_cells)

    return (sample_image_quality,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stochastic EGFR cascade simulator

    Same RAS → RAF → MEK → ERK cascade with negative feedback as in `model/mechanistic/egfr_simplified.py`,
    but with a per-cell **total kinase concentration** `K` sampled from a lognormal distribution.

    When `K=1` this reduces to the original deterministic model where `Active + Inactive = 1`.
    With `K ≠ 1`, the inactive pool becomes `(K - Active)`, scaling each node's activation capacity.
    """)
    return


@app.cell
def _(np):
    from scipy.integrate import solve_ivp as _solve_ivp

    def egfr_system(t, y, params, K, light_fn):
        RAS, RAF, MEK, NFB, ERK = y
        K_RAS, K_RAF, K_MEK, K_NFB, K_ERK = K
        Km, k12, k21, k34, knfb, k43, k56, k65, k78, k87, f12, f21 = params
        light = light_fn(t)

        dRAS = light * k12 * (K_RAS - RAS) - k21 * (RAS / (Km + RAS))
        dRAF = k34 * RAS * (K_RAF - RAF) - (knfb * NFB + k43) * (RAF / (Km + RAF))
        dMEK = k56 * RAF * (K_MEK - MEK) - k65 * (MEK / (Km + MEK))
        dNFB = f12 * ERK * (K_NFB - NFB) - f21 * (NFB / (Km + NFB))
        dERK = k78 * MEK * (K_ERK - ERK) - k87 * (ERK / (Km + ERK))
        return [dRAS, dRAF, dMEK, dNFB, dERK]

    DEFAULT_PARAMS = np.array([
        1.0,   # Km
        1.0,   # k12
        1.0,   # k21
        1.0,   # k34
        1.0,   # knfb
        1.0,   # k43
        1.0,   # k56
        1.0,   # k65
        1.0,   # k78
        1.0,   # k87
        1.0,   # f12
        1.0,   # f21
    ])

    def simulate_population(n_cells, params, K_mean, K_std, light_fn, t_max=100.0, dt=1, seed=42):
        rng = np.random.default_rng(seed)

        sigma_sq = np.log(1 + (K_std / K_mean) ** 2)
        mu = np.log(K_mean) - sigma_sq / 2
        K_values = rng.lognormal(mean=mu, sigma=np.sqrt(sigma_sq), size=(n_cells, 5))

        times = np.arange(0, t_max, dt)
        y0 = [0.05, 0.05, 0.05, 0.05, 0.05]
        trajectories = np.zeros((n_cells, 5, len(times)))

        for i in range(n_cells):
            K_i = K_values[i]
            sol = _solve_ivp(
                lambda t, y, _K=K_i: egfr_system(t, y, params, _K, light_fn),
                [0, t_max], y0,
                t_eval=times, method='LSODA', rtol=1e-8,
            )
            if sol.success:
                trajectories[i] = sol.y
            else:
                trajectories[i] = np.nan

        return times, trajectories, K_values

    return DEFAULT_PARAMS, simulate_population


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Real vs Simulated Comparison

    Compares real ERK measurements (`cnr_median_norm` from `dataset.parquet`) against
    the simulator output `sim_erk_noisy` resimulated on-the-fly from the sliders above.

    Sim is baseline-normalised (divided by per-cell mean over the first 10 timepoints)
    so both sit on the "CNR ratio" scale centred on 1 during baseline.
    """)
    return


@app.cell(hide_code=True)
def _(df, np, pd, sim_cnr):
    # Task 1 — distribution compare (real vs sim CNR)
    from scipy import stats as _stats

    _real = df[["uid", "frame", "cnr_median_norm"]].dropna()
    real_peak = _real.groupby("uid")["cnr_median_norm"].max().to_numpy()
    real_pool = _real["cnr_median_norm"].to_numpy()

    sim_peak = sim_cnr.max(axis=1)
    sim_pool = sim_cnr.flatten()

    def _moments(x, name):
        x = x[np.isfinite(x)]
        return {"name": name, "n": len(x),
                "mean": float(x.mean()), "std": float(x.std()),
                "skew": float(_stats.skew(x)), "kurt": float(_stats.kurtosis(x))}

    _ks_peak = _stats.ks_2samp(real_peak, sim_peak)
    _ks_pool = _stats.ks_2samp(real_pool[::20], sim_pool[::20])

    moments_table = pd.DataFrame([
        _moments(real_peak, "real peak"),
        _moments(sim_peak,  "sim  peak"),
        _moments(real_pool, "real pool"),
        _moments(sim_pool,  "sim  pool"),
    ])
    moments_table["KS D"] = [_ks_peak.statistic, _ks_peak.statistic, _ks_pool.statistic, _ks_pool.statistic]
    moments_table["KS p"] = [_ks_peak.pvalue, _ks_peak.pvalue, _ks_pool.pvalue, _ks_pool.pvalue]
    moments_table

    return real_peak, real_pool, sim_peak, sim_pool


@app.cell(hide_code=True)
def _(np, plt, real_peak, real_pool, sim_peak, sim_pool):
    # Task 1 — overlay hist
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

    _bins_peak = np.linspace(
        min(real_peak.min(), sim_peak.min()),
        max(real_peak.max(), sim_peak.max()), 50,
    )
    _axes[0].hist(real_peak, bins=_bins_peak, alpha=0.55, color="crimson",
                  edgecolor="white", lw=0.4, label=f"real (n={len(real_peak)})", density=True)
    _axes[0].hist(sim_peak, bins=_bins_peak, alpha=0.55, color="steelblue",
                  edgecolor="white", lw=0.4, label=f"sim  (n={len(sim_peak)})", density=True)
    _axes[0].set_xlabel("peak CNR (per cell)")
    _axes[0].set_ylabel("density")
    _axes[0].set_title("Peak CNR distribution")
    _axes[0].legend(fontsize=8)

    _lo, _hi = np.nanpercentile(real_pool, [0.5, 99.5])
    _bins_pool = np.linspace(_lo, _hi, 60)
    _axes[1].hist(real_pool, bins=_bins_pool, alpha=0.55, color="crimson",
                  edgecolor="white", lw=0.4, label="real", density=True)
    _axes[1].hist(sim_pool, bins=_bins_pool, alpha=0.55, color="steelblue",
                  edgecolor="white", lw=0.4, label="sim", density=True)
    _axes[1].set_xlabel("CNR (pooled over cells × time)")
    _axes[1].set_ylabel("density")
    _axes[1].set_title("Pooled CNR distribution")
    _axes[1].legend(fontsize=8)

    _fig.suptitle("Real vs Sim CNR distributions", y=1.02)
    _fig.tight_layout()
    _fig

    return


@app.cell(hide_code=True)
def _(
    dataset_noise,
    noise_baseline_scale,
    noise_jitter_scale,
    noise_jitter_timescale,
    np,
    sim_K_values,
    sim_trajectories,
    trajectory_noise,
):
    # Synthesise per-cell CNR from sim trajectories (same formula as CLI block)
    # cnr = cyt / nuc; nuc = b*(1 - ERK/K·s);  cyt = b*(1 + ERK/K·s)
    from scipy.ndimage import gaussian_filter1d as _gf1d_cmp
    _rng_cmp = np.random.default_rng(0)

    _erk_clean = sim_trajectories[:, 4, :]           # (n_cells, n_t)
    _K_erk     = sim_K_values[:, 4][:, None]         # (n_cells, 1)
    _erk_norm  = _erk_clean / np.maximum(_K_erk, 1e-6)

    _n_cells_s, _n_t_s = _erk_clean.shape
    _b_nuc  = _rng_cmp.lognormal(np.log(0.5), 0.2, size=(_n_cells_s, 1))
    _scale  = _rng_cmp.lognormal(0.0, 0.6, size=(_n_cells_s, 1))

    _jitter_std   = dataset_noise * noise_jitter_scale.value
    _baseline_std = trajectory_noise.median() * noise_baseline_scale.value
    _raw_n  = _rng_cmp.normal(0, 1.0, size=(_n_cells_s, _n_t_s))
    _smooth = _gf1d_cmp(_raw_n, sigma=noise_jitter_timescale.value, axis=1)
    _smooth = _smooth / (_smooth.std(axis=1, keepdims=True) + 1e-9) * _jitter_std
    _bshift = _rng_cmp.normal(0, _baseline_std, size=(_n_cells_s, 1))

    _prod = np.clip(_erk_norm * _scale, 0.0, 0.9)
    _nuc = _b_nuc * (1.0 - _prod) + _smooth * 0.5
    _cyt = _b_nuc * (1.0 + _prod) + _smooth
    _nuc_floored = np.maximum(_nuc, 0.3 * _b_nuc)
    sim_cnr = _cyt / _nuc_floored + _bshift  # already ~1 at baseline by construction
    sim_cnr.shape

    return (sim_cnr,)


@app.cell(hide_code=True)
def _(
    DEFAULT_PARAMS,
    dataset_noise,
    df,
    k12_std_ui,
    noise_baseline_scale,
    noise_jitter_scale,
    noise_jitter_timescale,
    np,
    pd,
    sim_K_mean,
    sim_K_std,
    simulate_population_het,
    trajectory_noise,
):
    # Task 2 — trajectory envelope: real vs sim per real light pattern
    # Reconstruct real stim profile → feed to simulate_population → synthesise CNR → compare envelopes.
    from scipy.ndimage import gaussian_filter1d as _gf1d_env

    _patterns = ['Single', 'Sustained', 'ramp1', '3-2-1minIntervals']
    _N_CELLS_SIM = 200

    envelope = {}

    for _pname in _patterns:
        _sub = df[df['ramp_pattern_name'] == _pname]
        _nf = int(_sub['frame'].max()) + 1

        _pivot = (_sub.pivot_table(index='uid', columns='frame',
                                    values='cnr_median_norm', aggfunc='mean')
                       .reindex(columns=range(_nf)))
        _real_mean = _pivot.mean(axis=0).to_numpy()
        _real_std  = _pivot.std(axis=0).to_numpy()

        _light_frame = _sub.groupby('frame')['stim_exposure'].mean().reindex(range(_nf)).fillna(0).to_numpy()
        _max_l = _light_frame.max() if _light_frame.max() > 0 else 1.0
        _light_norm = _light_frame / _max_l
        def _lfn(t, _p=_light_norm, _n=_nf):
            _i = int(t)
            if _i < 0 or _i >= _n:
                return 0.0
            return float(_p[_i])

        _times_s, _trajs, _Kvals, _k12vals = simulate_population_het(
            n_cells=_N_CELLS_SIM, params=DEFAULT_PARAMS,
            K_mean=sim_K_mean.value, K_std=sim_K_std.value, k12_std=k12_std_ui.value,
            light_fn=_lfn, t_max=float(_nf), dt=1.0, seed=17,
        )

        _rng_e = np.random.default_rng(17)
        _erk   = _trajs[:, 4, :]
        _K_erk = _Kvals[:, 4][:, None]
        _enorm = np.clip(_erk / np.maximum(_K_erk, 1e-6), 0.0, 1.0)
        _b_nuc = _rng_e.lognormal(np.log(0.5), 0.2, size=(_N_CELLS_SIM, 1))
        _scale = _rng_e.lognormal(0.0, 0.6, size=(_N_CELLS_SIM, 1))  # wider biosensor gain

        _jstd = dataset_noise * noise_jitter_scale.value
        _bstd = trajectory_noise.median() * noise_baseline_scale.value
        _rawn = _rng_e.normal(0, 1.0, size=_erk.shape)
        _smth = _gf1d_env(_rawn, sigma=noise_jitter_timescale.value, axis=1)
        # Global std normalisation — per-cell division blows up for short sequences
        _smth = _smth / (_smth.std() + 1e-9) * _jstd
        _bsh  = _rng_e.normal(0, _bstd, size=(_N_CELLS_SIM, 1))

        _prod = np.clip(_enorm * _scale, 0.0, 0.9)   # keep nuc positive
        _nuc  = _b_nuc * (1.0 - _prod) + _smth * 0.5
        _cyt  = _b_nuc * (1.0 + _prod) + _smth
        _nuc  = np.maximum(_nuc, 0.3 * _b_nuc)        # physical floor: nuc stays > 30% of baseline
        _cnr_raw = _cyt / _nuc + _bsh                 # cnr is already ~1 at baseline by construction
        _cnr = _cnr_raw

        envelope[_pname] = dict(
            frames=np.arange(_nf),
            real_mean=_real_mean, real_std=_real_std,
            sim_mean=_cnr.mean(axis=0), sim_std=_cnr.std(axis=0),
            light=_light_norm,
            n_real=_pivot.shape[0], n_sim=_N_CELLS_SIM,
        )

    envelope_summary = pd.DataFrame([
        {'pattern': p, 'n_frames': len(v['frames']), 'n_real': v['n_real'], 'n_sim': v['n_sim'],
         'sim_peak': float(v['sim_mean'].max()), 'real_peak': float(v['real_mean'].max())}
        for p, v in envelope.items()
    ])
    envelope_summary

    return (envelope,)


@app.cell(hide_code=True)
def _(envelope, plt):
    # Plot 4 subplots: real envelope vs sim envelope per pattern
    _fig, _axes = plt.subplots(2, 2, figsize=(15, 8))
    _axes = _axes.flatten()

    for _ax, _pname in zip(_axes, ['Single', 'Sustained', 'ramp1', '3-2-1minIntervals']):
        _e = envelope[_pname]
        _f = _e['frames']

        _ax_l = _ax.twinx()
        _ax_l.fill_between(_f, _e['light'], alpha=0.12, color='gold')
        _ax_l.set_ylim(-0.05, 4.0)
        _ax_l.set_yticks([])

        _n_real = _e['n_real']
        _n_sim  = _e['n_sim']
        _ax.plot(_f, _e['real_mean'], color='crimson', lw=1.8, label='real (n=' + str(_n_real) + ')')
        _ax.fill_between(_f, _e['real_mean'] - _e['real_std'], _e['real_mean'] + _e['real_std'],
                         color='crimson', alpha=0.18)
        _ax.plot(_f, _e['sim_mean'],  color='steelblue', lw=1.8, label='sim  (n=' + str(_n_sim) + ')')
        _ax.fill_between(_f, _e['sim_mean'] - _e['sim_std'], _e['sim_mean'] + _e['sim_std'],
                         color='steelblue', alpha=0.18)

        _ax.axvline(10, color='gray', ls=':', lw=1, alpha=0.6)
        _ax.set_title(_pname)
        _ax.set_xlabel('frame')
        _ax.set_ylabel('CNR (baseline-norm.)')
        _ax.legend(fontsize=8, loc='upper right')

    _fig.suptitle('Real vs Sim trajectory envelopes per light pattern  (mean ± 1σ)', y=1.01)
    _fig.tight_layout()
    _fig

    return


@app.cell(hide_code=True)
def _(np):
    # Per-cell heterogeneous simulator: varies k12 (RAS input sensitivity)
    # across cells on top of lognormal K. Reduces to homogeneous when k12_std=0.
    from scipy.integrate import solve_ivp as _solve_ivp_het

    def simulate_population_het(n_cells, params, K_mean, K_std, k12_std,
                                 light_fn, t_max=100.0, dt=1.0, seed=42):
        rng = np.random.default_rng(seed)

        _sig_K = np.log(1 + (K_std / K_mean) ** 2)
        _mu_K  = np.log(K_mean) - _sig_K / 2
        K_values = rng.lognormal(mean=_mu_K, sigma=np.sqrt(_sig_K), size=(n_cells, 5))

        k12_base = float(params[1])
        if k12_std > 0:
            _sig12 = np.log(1 + k12_std ** 2)
            _mu12  = np.log(k12_base) - _sig12 / 2
            k12_values = rng.lognormal(mean=_mu12, sigma=np.sqrt(_sig12), size=n_cells)
        else:
            k12_values = np.full(n_cells, k12_base)

        times = np.arange(0, t_max, dt)
        y0 = [0.05] * 5
        trajectories = np.zeros((n_cells, 5, len(times)))

        for i in range(n_cells):
            p_i = params.copy()
            p_i[1] = k12_values[i]
            K_i = K_values[i]

            def _sys(t, y, _p=p_i, _K=K_i, _lf=light_fn):
                RAS, RAF, MEK, NFB, ERK = y
                K_RAS, K_RAF, K_MEK, K_NFB, K_ERK = _K
                Km, k12, k21, k34, knfb, k43, k56, k65, k78, k87, f12, f21 = _p
                light = _lf(t)
                return [
                    light * k12 * (K_RAS - RAS) - k21 * (RAS / (Km + RAS)),
                    k34 * RAS * (K_RAF - RAF) - (knfb * NFB + k43) * (RAF / (Km + RAF)),
                    k56 * RAF * (K_MEK - MEK) - k65 * (MEK / (Km + MEK)),
                    f12 * ERK * (K_NFB - NFB) - f21 * (NFB / (Km + NFB)),
                    k78 * MEK * (K_ERK - ERK) - k87 * (ERK / (Km + ERK)),
                ]

            sol = _solve_ivp_het(_sys, [0, t_max], y0, t_eval=times,
                                 method='LSODA', rtol=1e-8)
            trajectories[i] = sol.y if sol.success else np.nan

        return times, trajectories, K_values, k12_values


    return (simulate_population_het,)


@app.cell(hide_code=True)
def _(mo):
    k12_std_ui = mo.ui.slider(0.0, 2.0, value=1.0, step=0.05,
                              label='k12 heterogeneity σ (lognormal CV)')
    k12_std_ui

    return (k12_std_ui,)


@app.cell(column=1)
def _(
    np,
    plt,
    sim_erk_mean,
    sim_erk_noisy,
    sim_erk_std,
    sim_light_pattern,
    sim_times,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 5))

    # ERK trajectories: noisy individual traces, clean population mean
    _ax = _axes[0]
    for _i in range(sim_erk_noisy.shape[0]):
        _ax.plot(sim_times, sim_erk_noisy[_i], color='steelblue', alpha=0.15, lw=0.5)
    _ax.plot(sim_times, sim_erk_mean, color='navy', lw=2, label='mean (clean)')
    _ax.fill_between(
        sim_times,
        np.maximum(sim_erk_mean - sim_erk_std, 0),
        sim_erk_mean + sim_erk_std,
        alpha=0.3, color='navy', label='±1 std',
    )
    _ax.set_xlabel('time (s)')
    _ax.set_ylabel('ERK readout')
    _ax.set_title(f'Simulated ERK trajectories (n={sim_erk_noisy.shape[0]})')
    _ax.legend(fontsize=8)

    # # K distribution (one per state variable)
    # _ax = _axes[1]
    # _state_names = ['RAS', 'RAF', 'MEK', 'NFB', 'ERK']
    # _kcolors = ['#e41a1c', '#ff7f00', '#4daf4a', '#984ea3', '#377eb8']
    # for _j, (_name, _col) in enumerate(zip(_state_names, _kcolors)):
    #     _ax.hist(sim_K_values[:, _j], bins=30, edgecolor='white', alpha=0.5, color=_col, label=_name)
    # _ax.set_xlabel('K (total kinase concentration)')
    # _ax.set_ylabel('cells')
    # _ax.set_title('Sampled K distributions per state variable')
    # _ax.legend(fontsize=8)

    # Light stimulus
    _ax = _axes[1]
    _light_signal = np.array([
        (1.0 if sim_light_pattern.value == "constant" else
         (1.0 if 10.0 <= t <= 40.0 else 0.0) if sim_light_pattern.value == "pulse" else
         min(t / 30.0, 1.0) if t <= 60.0 else 0.0)
        for t in sim_times
    ])
    _ax.plot(sim_times, _light_signal, color='orange', lw=1.5)
    _ax.set_xlabel('time (s)')
    _ax.set_ylabel('light intensity')
    _ax.set_title('Light stimulus')
    _ax.set_ylim(-0.05, 1.2)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    sim_n_cells = mo.ui.slider(10, 200, value=50, step=10, label="Number of cells")
    sim_K_mean = mo.ui.slider(0.5, 2.0, value=1.0, step=0.05, label="K mean")
    sim_K_std = mo.ui.slider(0.01, 0.5, value=0.15, step=0.01, label="K std")
    sim_light_pattern = mo.ui.dropdown(
        options=["constant", "pulse", "ramp"],
        value="pulse",
        label="Light pattern",
    )
    mo.hstack([sim_n_cells, sim_K_mean, sim_K_std, sim_light_pattern])
    return sim_K_mean, sim_K_std, sim_light_pattern, sim_n_cells


@app.cell(hide_code=True)
def _(dataset_noise, mo, trajectory_noise):
    noise_jitter_scale = mo.ui.slider(0.0, 3.0, value=1.0, step=0.1,
        label=f"Jitter scale  (×dataset σ={dataset_noise:.4f})")
    noise_baseline_scale = mo.ui.slider(0.0, 3.0, value=1.0, step=0.1,
        label=f"Baseline shift scale  (×traj median σ={trajectory_noise.median():.4f})")
    noise_jitter_timescale = mo.ui.slider(1, 50, value=10, step=1,
        label="Jitter timescale (frames)")
    mo.hstack([noise_jitter_scale, noise_baseline_scale, noise_jitter_timescale])
    return noise_baseline_scale, noise_jitter_scale, noise_jitter_timescale


@app.cell(hide_code=True)
def _(
    DEFAULT_PARAMS,
    k12_std_ui,
    np,
    sim_K_mean,
    sim_K_std,
    sim_light_pattern,
    sim_n_cells,
    simulate_population_het,
):
    def _make_light_fn(pattern):
        if pattern == "constant":
            return lambda t: 1.0
        elif pattern == "pulse":
            return lambda t: 1.0 if 10.0 <= t <= 40.0 else 0.0
        elif pattern == "ramp":
            return lambda t: min(t / 30.0, 1.0) if t <= 60.0 else 0.0
        return lambda t: 0.0

    _light_fn = _make_light_fn(sim_light_pattern.value)

    sim_times, sim_trajectories, sim_K_values, sim_k12_values = simulate_population_het(
        n_cells=sim_n_cells.value,
        params=DEFAULT_PARAMS,
        K_mean=sim_K_mean.value,
        K_std=sim_K_std.value,
        k12_std=k12_std_ui.value,
        light_fn=_light_fn,
        t_max=100.0,
        dt=0.1,
        seed=42,
    )

    sim_erk = sim_trajectories[:, 4, :]
    sim_erk_mean = np.nanmean(sim_erk, axis=0)
    sim_erk_std = np.nanstd(sim_erk, axis=0)
    return (
        sim_K_values,
        sim_erk,
        sim_erk_mean,
        sim_erk_std,
        sim_times,
        sim_trajectories,
    )


@app.cell(hide_code=True)
def _(
    dataset_noise,
    noise_baseline_scale,
    noise_jitter_scale,
    noise_jitter_timescale,
    np,
    sim_erk,
    trajectory_noise,
):
    from scipy.ndimage import gaussian_filter1d as _gf1d
    _rng = np.random.default_rng(0)
    _jitter_std = dataset_noise * noise_jitter_scale.value
    _baseline_std = trajectory_noise.median() * noise_baseline_scale.value
    _baseline_shifts = _rng.normal(0, _baseline_std, size=(sim_erk.shape[0], 1))
    _raw = _rng.normal(0, 1.0, size=sim_erk.shape)
    _smooth = _gf1d(_raw, sigma=noise_jitter_timescale.value, axis=1)
    _smooth = _smooth / _smooth.std(axis=1, keepdims=True) * _jitter_std
    sim_erk_noisy = sim_erk + _baseline_shifts + _smooth
    return (sim_erk_noisy,)


@app.cell
def _(np, plt, sim_erk, sim_times):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Peak ERK vs time of peak
    _peak_vals = np.nanmax(sim_erk, axis=1)
    _peak_times = sim_times[np.nanargmax(sim_erk, axis=1)]
    _axes[0].scatter(_peak_times, _peak_vals, alpha=0.6, s=20, color='steelblue')
    _axes[0].set_xlabel('time of peak (s)')
    _axes[0].set_ylabel('peak ERK')
    _axes[0].set_title('Peak ERK activity vs timing')

    # ERK at steady state (last 10% of sim)
    _ss_start = int(0.9 * sim_erk.shape[1])
    _ss_erk = np.nanmean(sim_erk[:, _ss_start:], axis=1)
    _axes[1].hist(_ss_erk, bins=30, edgecolor='white', alpha=0.8, color='steelblue')
    _axes[1].set_xlabel('ERK (late-phase mean)')
    _axes[1].set_ylabel('cells')
    _axes[1].set_title('Late-phase ERK distribution')

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Randomized trajectory demo

    Generate a fresh single-cell trajectory with a randomly sampled light pattern.
    Only active in interactive mode — clicking the button picks a new random seed each time.
    """)
    return


@app.cell
def _(mo):
    demo_run_btn = mo.ui.run_button(label="New random trajectory")
    demo_generator = mo.ui.dropdown(
        options=["random", "stochastic", "sequential", "functional", "smoothed"],
        value="random",
        label="Light pattern generator",
    )
    mo.hstack([demo_run_btn, demo_generator])
    return demo_generator, demo_run_btn


@app.cell
def _(
    DEFAULT_PARAMS,
    dataset_noise,
    demo_generator,
    demo_run_btn,
    mo,
    noise_baseline_scale,
    noise_jitter_scale,
    noise_jitter_timescale,
    np,
    plt,
    simulate_population,
    trajectory_noise,
):
    from model.mechanistic.egfr_simplified import (
        generate_stochastic_pulses as _gen_stochastic,
        generate_sequential_pulses as _gen_sequential,
        generate_functional_pulses as _gen_functional,
        generate_smoothed_pulses as _gen_smoothed,
    )
    from scipy.ndimage import gaussian_filter1d as _gf1d_demo

    mo.stop(not demo_run_btn.value)

    _rng = np.random.default_rng()  # new seed each click
    _gen_map = {
        "stochastic": _gen_stochastic,
        "sequential": _gen_sequential,
        "functional": _gen_functional,
        "smoothed":   _gen_smoothed,
    }
    _gen_fn = _gen_map.get(demo_generator.value) or _rng.choice(list(_gen_map.values()))
    _pat = _gen_fn(t_max=100.0, rng=_rng)
    _pulses = _pat["pulses"]
    _light_fn = lambda t, _p=_pulses: sum(p["amplitude"] for p in _p if p["t_on"] <= t <= p["t_off"])

    _times, _trajs, _ = simulate_population(
        n_cells=1, params=DEFAULT_PARAMS,
        K_mean=1.0, K_std=0.15,
        light_fn=_light_fn, t_max=100.0, dt=1.0,
        seed=int(_rng.integers(0, 2**31)),
    )
    _erk = _trajs[0, 4, :]
    _jitter_std = dataset_noise * noise_jitter_scale.value
    _baseline_shift = _rng.normal(0, trajectory_noise.median() * noise_baseline_scale.value)
    _raw = _rng.normal(0, 1.0, len(_erk))
    _smooth_noise = _gf1d_demo(_raw, sigma=noise_jitter_timescale.value)
    _smooth_noise = _smooth_noise / (_smooth_noise.std() + 1e-9) * _jitter_std
    _erk_noisy = _erk + _baseline_shift + _smooth_noise

    _light_vals = np.array([_light_fn(t) for t in _times])
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _ax_l = _ax.twinx()
    _ax_l.fill_between(_times, _light_vals, alpha=0.12, color='gold')
    _ax_l.set_ylim(-0.05, max(_light_vals, default=1) * 4)
    _ax_l.set_ylabel('light intensity', color='orange', fontsize=8)
    _ax_l.tick_params(axis='y', labelcolor='orange', labelsize=7)
    _ax.plot(_times, _erk, color='navy', lw=1.5, alpha=0.7, label='ERK (clean)')
    _ax.plot(_times, _erk_noisy, color='steelblue', lw=1.0, alpha=0.85, label='ERK (noisy)')
    _ax.set_xlabel('time (s)')
    _ax.set_ylabel('ERK active fraction')
    _ax.set_title(f'Random trajectory  [{demo_generator.value} generator]')
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    dataset_noise,
    k12_std_ui,
    mo,
    noise_baseline_scale,
    noise_jitter_scale,
    noise_jitter_timescale,
    np,
    pd,
    sample_image_quality,
    trajectory_noise,
):
    # In notebook mode, just show a note and stop. In CLI mode (app.run() from __main__),
    # mo.running_in_notebook() is False so the stop is skipped and generation runs.
    mo.stop(mo.running_in_notebook(), mo.md(
        "*Run `python stochastic_simulator.py` to generate a 30 K-trajectory parquet (v2).*"
    ))

    import json as _json
    from tqdm import tqdm as _tqdm
    from scipy.integrate import solve_ivp as _solve_ivp_cli
    from scipy.ndimage import gaussian_filter1d as _gf1d_cli
    from model.mechanistic.egfr_simplified import (
        generate_stochastic_pulses as _gs,
        generate_sequential_pulses as _gseq,
        generate_functional_pulses as _gfunc,
        generate_smoothed_pulses as _gsm,
    )

    _N_TOTAL = 30_000
    _N_TP    = 100
    _T_MAX   = 100.0
    _OUTPUT  = "stochastic_sim_v2_output.parquet"

    def _egfr_cli(t, y, params, K, light_fn):
        RAS, RAF, MEK, NFB, ERK = y
        K_RAS, K_RAF, K_MEK, K_NFB, K_ERK = K
        Km, k12, k21, k34, knfb, k43, k56, k65, k78, k87, f12, f21 = params
        light = light_fn(t)
        dRAS = light * k12 * (K_RAS - RAS) - k21 * (RAS / (Km + RAS))
        dRAF = k34 * RAS * (K_RAF - RAF) - (knfb * NFB + k43) * (RAF / (Km + RAF))
        dMEK = k56 * RAF * (K_MEK - MEK) - k65 * (MEK / (Km + MEK))
        dNFB = f12 * ERK * (K_NFB - NFB) - f21 * (NFB / (Km + NFB))
        dERK = k78 * MEK * (K_ERK - ERK) - k87 * (ERK / (Km + ERK))
        return [dRAS, dRAF, dMEK, dNFB, dERK]

    _PARAMS = np.ones(12)
    _K_MEAN, _K_STD = 1.0, 0.15
    _sigma_sq = np.log(1 + (_K_STD / _K_MEAN) ** 2)
    _mu_K = np.log(_K_MEAN) - _sigma_sq / 2

    # k12 heterogeneity — RAS input sensitivity varies per cell
    _K12_STD = float(k12_std_ui.value)
    if _K12_STD > 0:
        _sig12 = np.log(1 + _K12_STD ** 2)
        _mu12  = np.log(_PARAMS[1]) - _sig12 / 2

    _rng = np.random.default_rng(42)
    _patterns = []
    for _gname, _gfn in [
        ("stochastic", _gs), ("sequential", _gseq),
        ("functional", _gfunc), ("smoothed", _gsm),
    ]:
        for _ in range(_N_TOTAL // 4):
            _p = _gfn(t_max=_T_MAX, rng=_rng)
            _p["generator"] = _gname
            _patterns.append(_p)
    _rng.shuffle(_patterns)

    _times_cli = np.linspace(0, _T_MAX, _N_TP)
    _rows, _n_fail = [], 0

    for _i, _pat in enumerate(_tqdm(_patterns, desc="Simulating v2")):
        _K_i   = _rng.lognormal(mean=_mu_K, sigma=np.sqrt(_sigma_sq), size=5)
        _k12_i = (_rng.lognormal(mean=_mu12, sigma=np.sqrt(_sig12))
                  if _K12_STD > 0 else _PARAMS[1])
        _params_i = _PARAMS.copy()
        _params_i[1] = _k12_i

        _plist = _pat["pulses"]
        _lfn = lambda t, _p=_plist: sum(p["amplitude"] for p in _p if p["t_on"] <= t <= p["t_off"])

        _sol = _solve_ivp_cli(
            lambda t, y, _K=_K_i, _P=_params_i: _egfr_cli(t, y, _P, _K, _lfn),
            [_times_cli[0], _times_cli[-1]], [0.05] * 5,
            t_eval=_times_cli, method='LSODA', rtol=1e-8,
        )
        if not _sol.success:
            _n_fail += 1
            continue

        _erk = _sol.y[4]
        _erk_norm = np.clip(_erk / max(_K_i[4], 1e-6), 0.0, 1.0)

        # CNR synthesis — wider biosensor gain + product clip to keep nuc positive
        _b_nuc = _rng.lognormal(np.log(0.5), 0.2)
        _scale = _rng.lognormal(0.0, 0.6)
        _prod  = np.clip(_erk_norm * _scale, 0.0, 0.9)

        _jitter_std = dataset_noise * noise_jitter_scale.value
        _baseline_shift = _rng.normal(0, trajectory_noise.median() * noise_baseline_scale.value)
        _raw_j = _rng.normal(0, 1.0, _N_TP)
        _smooth_j = _gf1d_cli(_raw_j, sigma=noise_jitter_timescale.value)
        _smooth_j = _smooth_j / (_smooth_j.std() + 1e-9) * _jitter_std

        _nuc = _b_nuc * (1.0 - _prod) + _smooth_j * 0.5
        _cyt = _b_nuc * (1.0 + _prod) + _smooth_j
        _nuc = np.maximum(_nuc, 0.3 * _b_nuc)
        _cnr = _cyt / _nuc + _baseline_shift

        _rows.append({
            "trajectory_id":  _i,
            "generator":      _pat["generator"],
            "pulses_json":    _json.dumps(_pat["pulses"]),
            "K_values":       _K_i.tolist(),
            "k12":            float(_k12_i),
            "times":          _times_cli.tolist(),
            "light":          [_lfn(t) for t in _times_cli],
            "RAS_s": _sol.y[0].tolist(), "RAF_s": _sol.y[1].tolist(),
            "MEK_s": _sol.y[2].tolist(), "NFB_s": _sol.y[3].tolist(),
            "ERK_s": _erk.tolist(),
            "cnr":            _cnr.tolist(),
            "nuc_intensity":  _nuc.tolist(),
            "cyt_intensity":  _cyt.tolist(),
            "area":           float(_rng.lognormal(np.log(500), 0.3)),
            "density":        int(_rng.poisson(5)),
            "image_quality":  float(sample_image_quality(1, _rng)[0]),
        })

    _df_out = pd.DataFrame(_rows)
    _df_out.to_parquet(_OUTPUT, index=False)
    print(f"Saved {len(_df_out)} trajectories to {_OUTPUT}  (k12_std={_K12_STD})")
    if _n_fail:
        print(f"WARNING: {_n_fail} simulations failed and were skipped")

    return


@app.cell(column=2)
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return nn, torch


@app.cell
def _(nn, torch):
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5,3,10)
    h0 = torch.randn(2,3,20)
    c0 = torch.randn(2,3,20)
    rnn
    return c0, h0, input, rnn


@app.cell
def _(c0, h0, input, rnn):
    output, (hn,cn) = rnn(input, (h0, c0))
    output, (hn, cn)
    return


@app.cell
def _():
    EPOCHS = 5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated data inspection

    Load the parquet produced by `python stochastic_simulator.py` and visually inspect sample trajectories and feature distributions.
    """)
    return


@app.cell
def _(mo):
    inspect_path = mo.ui.text(value="stochastic_sim_output.parquet", label="Parquet path")
    inspect_load_btn = mo.ui.run_button(label="Load")
    inspect_n = mo.ui.slider(1, 20, value=6, step=1, label="Samples to show")
    inspect_generator = mo.ui.dropdown(
        options=["all", "stochastic", "sequential", "functional", "smoothed"],
        value="all", label="Generator filter",
    )
    mo.vstack([
        mo.hstack([inspect_path, inspect_load_btn]),
        mo.hstack([inspect_n, inspect_generator]),
    ])
    return inspect_generator, inspect_load_btn, inspect_n, inspect_path


@app.cell
def _(
    inspect_generator,
    inspect_load_btn,
    inspect_n,
    inspect_path,
    mo,
    np,
    pd,
    plt,
):
    import os as _os
    mo.stop(not inspect_load_btn.value, mo.md("Click **Load** to inspect generated trajectories."))
    mo.stop(not _os.path.exists(inspect_path.value), mo.md(f"`{inspect_path.value}` not found — run `python stochastic_simulator.py` first."))

    _df_insp = pd.read_parquet(inspect_path.value)
    _sub = _df_insp if inspect_generator.value == "all" else _df_insp[_df_insp["generator"] == inspect_generator.value]
    _sample = _sub.sample(n=min(inspect_n.value, len(_sub)), random_state=0)

    _n_rows = len(_sample)
    _fig_t, _axes_t = plt.subplots(_n_rows, 1, figsize=(12, 3.5 * _n_rows), squeeze=False)
    for _ax_row, (_, _row) in zip(_axes_t, _sample.iterrows()):
        _ax = _ax_row[0]
        _t = np.array(_row["times"])
        _light = np.array(_row["light"])
        _erk = np.array(_row["ERK_s"])
        _cnr = np.array(_row["cnr"])
        _ax_l = _ax.twinx()
        _ax_l.fill_between(_t, _light, alpha=0.12, color='gold')
        _max_l = _light.max() if _light.max() > 0 else 1
        _ax_l.set_ylim(-0.05, _max_l * 4)
        _ax_l.set_ylabel("light", color='orange', fontsize=7)
        _ax_l.tick_params(axis='y', labelcolor='orange', labelsize=6)
        _ax.plot(_t, _erk, color='navy', lw=1.5, alpha=0.85, label='ERK_s')
        _ax.plot(_t, _cnr, color='steelblue', lw=1.0, alpha=0.7, label='CNR')
        _ax.set_ylabel("value")
        _ax.set_title(f"id={_row['trajectory_id']}  [{_row['generator']}]", fontsize=9, fontweight='bold')
        _ax.legend(fontsize=7, loc='upper right')
    _axes_t[-1][0].set_xlabel("time (s)")
    _fig_t.tight_layout()
    _fig_t
    return


@app.cell
def _(inspect_load_btn, inspect_path, mo, np, pd, plt):
    import os as _os2
    mo.stop(not inspect_load_btn.value)
    mo.stop(not _os2.path.exists(inspect_path.value))

    _df_dist = pd.read_parquet(inspect_path.value)
    _fig_d, _axd = plt.subplots(1, 4, figsize=(16, 4))

    _axd[0].hist(_df_dist["area"].values, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    _axd[0].set_title("Area")
    _axd[0].set_xlabel("area (px²)")

    _density_vals = _df_dist["density"].values.astype(int)
    _axd[1].hist(_density_vals, bins=range(0, max(_density_vals) + 2), color='coral', alpha=0.8, edgecolor='white')
    _axd[1].set_title("Density")
    _axd[1].set_xlabel("neighbours in radius")

    _axd[2].hist(_df_dist["image_quality"].values, bins=30, color='seagreen', alpha=0.8, edgecolor='white')
    _axd[2].set_title("Image quality")
    _axd[2].set_xlabel("quality score")

    _peak_cnr = np.array([np.max(row) for row in _df_dist["cnr"].values])
    _axd[3].hist(_peak_cnr, bins=40, color='mediumpurple', alpha=0.8, edgecolor='white')
    _axd[3].set_title("Peak CNR per trajectory")
    _axd[3].set_xlabel("peak CNR")

    _fig_d.suptitle(f"Feature distributions  (n={len(_df_dist):,} trajectories)", fontsize=11)
    _fig_d.tight_layout()
    _fig_d
    return


if __name__ == "__main__":
    app.run()
