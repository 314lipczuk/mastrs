import marimo

__generated_with = "0.21.1"
app = marimo.App(width="columns")


@app.cell(column=0)
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


@app.cell
def _(augment, pd):
    df = augment(pd.read_parquet('dataset.parquet'))
    df.info()
    return (df,)


@app.cell
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
        Km, k12, k21, k34, knfb, k43, k56, k65, k78, k87, f12, f21 = params
        light = light_fn(t)

        dRAS = light * k12 * (K - RAS) - k21 * (RAS / (Km + RAS))
        dRAF = k34 * RAS * (K - RAF) - (knfb * NFB + k43) * (RAF / (Km + RAF))
        dMEK = k56 * RAF * (K - MEK) - k65 * (MEK / (Km + MEK))
        dNFB = f12 * ERK * (K - NFB) - f21 * (NFB / (Km + NFB))
        dERK = k78 * MEK * (K - ERK) - k87 * (ERK / (Km + ERK))
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

    def simulate_population(n_cells, params, K_mean, K_std, light_fn, t_max=100.0, dt=0.1, seed=42):
        rng = np.random.default_rng(seed)

        sigma_sq = np.log(1 + (K_std / K_mean) ** 2)
        mu = np.log(K_mean) - sigma_sq / 2
        K_values = rng.lognormal(mean=mu, sigma=np.sqrt(sigma_sq), size=n_cells)

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


@app.cell(column=1)
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


@app.cell
def _(
    DEFAULT_PARAMS,
    np,
    sim_K_mean,
    sim_K_std,
    sim_light_pattern,
    sim_n_cells,
    simulate_population,
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

    sim_times, sim_trajectories, sim_K_values = simulate_population(
        n_cells=sim_n_cells.value,
        params=DEFAULT_PARAMS,
        K_mean=sim_K_mean.value,
        K_std=sim_K_std.value,
        light_fn=_light_fn,
        t_max=100.0,
        dt=0.1,
        seed=42,
    )

    sim_erk = sim_trajectories[:, 4, :]
    sim_erk_mean = np.nanmean(sim_erk, axis=0)
    sim_erk_std = np.nanstd(sim_erk, axis=0)
    return sim_K_values, sim_erk, sim_erk_mean, sim_erk_std, sim_times


@app.cell
def _(
    np,
    plt,
    sim_K_values,
    sim_erk,
    sim_erk_mean,
    sim_erk_std,
    sim_light_pattern,
    sim_times,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))

    # ERK trajectories
    _ax = _axes[0]
    for _i in range(sim_erk.shape[0]):
        _ax.plot(sim_times, sim_erk[_i], color='steelblue', alpha=0.15, lw=0.5)
    _ax.plot(sim_times, sim_erk_mean, color='navy', lw=2, label='mean')
    _ax.fill_between(
        sim_times,
        np.maximum(sim_erk_mean - sim_erk_std, 0),
        sim_erk_mean + sim_erk_std,
        alpha=0.3, color='navy', label='±1 std',
    )
    _ax.set_xlabel('time (s)')
    _ax.set_ylabel('ERK active fraction')
    _ax.set_title(f'Simulated ERK trajectories (n={sim_erk.shape[0]})')
    _ax.legend(fontsize=8)

    # K distribution
    _ax = _axes[1]
    _ax.hist(sim_K_values, bins=30, edgecolor='white', alpha=0.8, color='coral')
    _ax.axvline(np.mean(sim_K_values), color='red', ls='--', lw=2, label=f'mean={np.mean(sim_K_values):.3f}')
    _ax.set_xlabel('K (total kinase concentration)')
    _ax.set_ylabel('cells')
    _ax.set_title('Sampled K distribution')
    _ax.legend(fontsize=8)

    # Light stimulus
    _ax = _axes[2]
    _light_signal = np.array([
        (1.0 if sim_light_pattern.value == "constant" else
         (1.0 if 10.0 <= t <= 40.0 else 0.0) if sim_light_pattern.value == "pulse" else
         min(t / 30.0, 1.0) if t <= 60.0 else 0.0)
        for t in sim_times
    ])
    _ax.fill_between(sim_times, _light_signal, alpha=0.3, color='gold')
    _ax.plot(sim_times, _light_signal, color='orange', lw=1.5)
    _ax.set_xlabel('time (s)')
    _ax.set_ylabel('light intensity')
    _ax.set_title('Light stimulus')
    _ax.set_ylim(-0.05, 1.2)

    _fig.tight_layout()
    _fig
    return


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


if __name__ == "__main__":
    app.run()
