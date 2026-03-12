import os
from pathlib import Path
import pandas as pd
import numpy as np

def load_and_clean(
    parquet_path: str,
    cell_selection_csv: str | None = None,
    tracking_threshold: float = 0.9,
    norm_until_timepoint: int = 10,
    baseline_cnr_max: float | None = 0.8,
    cell_line: str | None = 'EGFR'
) -> pd.DataFrame:
    """Read a parquet experiment file and return a cleaned dataframe.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file with raw tracking data.
    cell_selection_csv : str or None
        Path to a ``cell_selection.csv`` for manual filtering.  Rows whose
        ``deleted`` column is ``True`` are removed.  Pass ``None`` to skip.
    tracking_threshold : float
        Fraction of the maximum frame count a cell must reach to be kept
        (default 0.9 = 90 %).
    norm_until_timepoint : int
        Frames ``< norm_until_timepoint`` are used to compute the per-cell
        baseline for normalisation (default 10).
    baseline_cnr_max : float or None
        If set, cells whose ``median_cnr_0_9`` exceeds this value are dropped.
    """
    df = pd.read_parquet(parquet_path)

    # --- derived columns ---------------------------------------------------
    df["cnr"] = df["mean_intensity_C1_ring"] / df["mean_intensity_C1_nuc"]
    df["cnr_median"] = df["median_intensity_C1_ring"] / df["median_intensity_C1_nuc"]
    df["uid"] = df['ramp_pattern_name'] + df["fov"].astype("string") + "_" + df["particle"].astype("string")
    df["frame"] = df["timestep"]

    # --- drop short tracks --------------------------------------------------
    frame_counts = df["uid"].value_counts()
    threshold = tracking_threshold * frame_counts.max()
    valid_uids = frame_counts[frame_counts >= threshold].index
    df = df[df["uid"].isin(valid_uids)]

    # --- normalise CNR (median) ---------------------------------------------
    baseline_median = (
        df.loc[df["frame"] < norm_until_timepoint]
        .groupby("uid")["cnr_median"]
        .median()
    )
    df["cnr_median_norm"] = df["uid"].map(baseline_median)
    df["cnr_median_norm"] = df["cnr_median"] / df["cnr_median_norm"]
    df.dropna(subset=["cnr_median_norm"], inplace=True)

    # --- normalise CNR (mean) -----------------------------------------------
    baseline_mean = (
        df.loc[df["frame"] < norm_until_timepoint]
        .groupby("uid")["cnr"]
        .median()
    )
    df["cnr_mean_norm"] = df["uid"].map(baseline_mean)
    df["cnr_mean_norm"] = df["cnr"] / df["cnr_mean_norm"]
    df.dropna(subset=["cnr_mean_norm"], inplace=True)

    # --- manual cell selection ----------------------------------------------
    if cell_selection_csv is not None and os.path.isfile(cell_selection_csv):
        print('cell line selection for ', parquet_path , ' running...')
        sel = pd.read_csv(cell_selection_csv)
        uids_to_delete = sel.loc[sel["deleted"] == True, "uid"]
        df = df[~df["uid"].isin(uids_to_delete)]

    # --- baseline ERK activity per cell -------------------------------------
    df_baseline = df.loc[
        (df["frame"] >= 0) & (df["frame"] <= (norm_until_timepoint - 1))
    ].copy()
    median_cnr = (
        df_baseline.groupby("uid")["cnr"]
        .median()
        .reset_index()
        .rename(columns={"cnr": "median_cnr_0_9"})
    )
    df = df.merge(median_cnr, on="uid", how="left")

    # --- optional baseline threshold ----------------------------------------
    if baseline_cnr_max is not None:
        df = df[df["median_cnr_0_9"] <= baseline_cnr_max]


    # --- Cell line filtering ----------------------------------------
    df = df[df['cell_line'] == cell_line]

    
    # --- Power calculation -----------------------------------------

    df['stim_exposure'] = df['stim_exposure'].fillna(0) # if you're not giving me exposure, it's probably zero 

    df = calc_power(df)

    df = add_stim_features(df)

    df = df.reset_index(drop=True)
    return df




cal_power_pct = [0,1,2,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
cal_uW        = [8,11,17,26.2,29.5,49.9,71.3,90.5,111,132,153,172,192,212,231,249,268,287,305,321,339,356,374,389]
cal_mW_cm2    = [1.26,1.73,2.67,4.12,4.64,7.84,11.21,14.23,17.45,20.75,24.05,27.04,30.18,33.32,36.31,39.14,42.13,45.11,47.94,50.46,53.29,55.96,58.79,61.15]

def calc_power(df):
    P_uW = np.interp(df["stim_power"], cal_power_pct, cal_uW)           # microwatts
    irradiance = np.interp(df["stim_power"], cal_power_pct, cal_mW_cm2) # mW/cm2

    # Energy per pulse
    # More precisely: energy (uJ) = P (uW) * t (ms) * 1e-3
    df["energy_uJ"] = P_uW * df["stim_exposure"] * 1e-3

    # Fluence (energy dose per unit area) per pulse
    df["fluence_mJ_cm2"] = irradiance * df["stim_exposure"] * 1e-3  # mW/cm2 * ms * 1e-3 = mJ/cm2

    df['energy_per_cell'] = df['fluence_mJ_cm2'] * df['area'] # not in any units, since area is just pixels.
    # warning - never use this - the area is the things we stain for (ERK in nucl - so size actually changes from our pertrbation)

    return df

def add_stim_features(df, window_min=5, ewma_alpha_fast=0.5, ewma_alpha_slow=0.1, tau=5 ):
    """Add 9 stimulation input channels per cell on a 1-minute grid.

    Assumes each frame = 1 minute. Operates per cell (grouped by ``uid``).

    Columns added
    -------------
    u_t             : Raw pulse amplitude (fluence_mJ_cm2).
    m_t             : Activation indicator (1 if stimulated, 0 otherwise).
    dt_since_pulse  : Minutes since last pulse (NaN before first pulse).
    ewma_fast       : EWMA of u_t with ``ewma_alpha_fast`` (half-life ~1 min).
    ewma_slow       : EWMA of u_t with ``ewma_alpha_slow`` (half-life ~7 min).
    n_5             : Number of pulses in the last ``window_min`` frames.
    slope_5         : OLS slope of u_t over the last ``window_min`` frames.
    burst_pos       : Position (1-indexed) within current consecutive burst; 0 when unstimulated.
    s_cum           : Cumulative sum of u_t up to and including this frame.
    """
    df = df.sort_values(["uid", "frame"]).copy()

    df["u_t"] = df["fluence_mJ_cm2"]
    df["m_t"] = df["stim"].astype(int)

    def _cell_features(g):
        u = g["u_t"].values
        m = g["m_t"].values
        n = len(u)

        # --- dt_since_pulse ---
        dt = np.full(n, np.nan)
        last_pulse = -1
        for i in range(n):
            if m[i]:
                last_pulse = i
            if last_pulse >= 0:
                dt[i] = i - last_pulse

        # --- burst_pos ---
        bp = np.zeros(n, dtype=int)
        for i in range(n):
            if m[i]:
                bp[i] = (bp[i - 1] + 1) if i > 0 and m[i - 1] else 1

        # --- n_5 and slope_5 (rolling window) ---
        n5 = np.zeros(n, dtype=int)
        sl5 = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - window_min + 1)
            window_m = m[start:i + 1]
            window_u = u[start:i + 1]
            n5[i] = window_m.sum()
            if len(window_u) >= 2:
                x_w = np.arange(len(window_u), dtype=float)
                x_w_mean = x_w.mean()
                ss = ((x_w - x_w_mean) ** 2).sum()
                if ss > 0:
                    sl5[i] = ((x_w - x_w_mean) * (window_u - window_u.mean())).sum() / ss
        
        g = g.copy()

        g["recency"] = np.exp(-dt / tau)                                                     
        g["recency"] = g["recency"].fillna(0.0) 
        g["burst_pos"] = bp
        g["n_5"] = n5
        g["slope_5"] = sl5
        return g

    saved_uid = df[["uid"]].copy()
    df = df.groupby("uid", group_keys=False).apply(_cell_features)
    if "uid" not in df.columns:
        df["uid"] = saved_uid["uid"]

    # --- vectorised columns (EWMA, cumsum) ---
    df["ewma_fast"] = (
        df.groupby("uid")["u_t"]
        .transform(lambda s: s.ewm(alpha=ewma_alpha_fast, adjust=False).mean())
    )
    df["ewma_slow"] = (
        df.groupby("uid")["u_t"]
        .transform(lambda s: s.ewm(alpha=ewma_alpha_slow, adjust=False).mean())
    )
    df["s_cum"] = df.groupby("uid")["u_t"].cumsum()

    return df

DEFAULT_STIM_COLS = ["u_t", "m_t", "recency", "ewma_fast", "ewma_slow",
                     "n_5", "slope_5", "burst_pos", "s_cum"]

def make_windows(df, window_size=None, stride=None, value_col="cnr_median_norm",
                 stim_cols=None):
    """Slice per-cell trajectories into fixed-length windows.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataset with ``uid``, ``frame``, *value_col*, and *stim_cols*.
    window_size : int or None
        Length of each window in frames.  ``None`` → shortest experiment length.
    stride : int or None
        Step between consecutive windows.  ``None`` → ``window_size`` (no overlap).
    value_col : str
        Column name for the target signal (e.g. ERK readout).
    stim_cols : list of str or None
        Stimulus feature columns to extract.  ``None`` → ``DEFAULT_STIM_COLS``.

    Returns
    -------
    erk : np.ndarray, shape (n_windows, window_size)
    stim : np.ndarray, shape (n_windows, n_stim_cols, window_size)
    meta : pd.DataFrame  — one row per window with uid, window_start, and first-row metadata.
    """
    if stim_cols is None:
        stim_cols = DEFAULT_STIM_COLS
    if window_size is None:
        window_size = int(df.groupby("ramp_pattern_name")["frame"].nunique().min())
    if stride is None:
        stride = window_size

    erk_windows, stim_windows, meta_rows = [], [], []

    for uid, g in df.groupby("uid"):
        g = g.sort_values("frame")
        vals = g[value_col].values
        stim_mat = g[stim_cols].values  # (n_frames, n_stim_cols)
        frames = g["frame"].values
        n = len(vals)

        first_row = g.iloc[0]

        for start in range(0, n - window_size + 1, stride):
            chunk_v = vals[start : start + window_size]
            chunk_s = stim_mat[start : start + window_size]  # (window_size, n_stim_cols)
            if np.isnan(chunk_v).any() or np.isnan(chunk_s).any():
                continue
            erk_windows.append(chunk_v)
            stim_windows.append(chunk_s.T)  # (n_stim_cols, window_size)
            meta_rows.append({
                "uid": uid,
                "window_start": int(frames[start]),
                "cell_line": first_row.get("cell_line", None),
                "ramp_pattern_name": first_row.get("ramp_pattern_name", None),
                "fov": first_row.get("fov", None),
            })

    erk = np.array(erk_windows, dtype=np.float32)
    stim = np.array(stim_windows, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return erk, stim, meta


def plot_nan_audit(df):
    """Plot NaN counts per column broken down by experiment, and per-cell NaN fraction.

    Parameters
    ----------
    df : pd.DataFrame
        The combined dataset (must contain a ``ramp_pattern_name`` and ``uid`` column).
    """
    import matplotlib.pyplot as plt

    # Per-column NaN count per experiment
    nan_by_exp = pd.DataFrame({
        pat: df[df['ramp_pattern_name'] == pat].isna().sum()
        for pat in df['ramp_pattern_name'].unique()
    })
    nan_by_exp = nan_by_exp[nan_by_exp.sum(axis=1) > 0]

    if len(nan_by_exp) == 0:
        print("No NaN values found in any column — data is clean.")
    else:
        fig, ax = plt.subplots(figsize=(10, max(3, len(nan_by_exp) * 0.4)))
        nan_by_exp.plot(kind='barh', ax=ax)
        ax.set_title('NaN counts per column per experiment')
        ax.set_xlabel('NaN count')
        plt.tight_layout()
        plt.show()
        print("\nColumns with NaNs:")
        print(nan_by_exp)

    # Per-cell NaN fraction
    cell_nan_frac = df.groupby('uid')[df.columns.difference(['uid'])].apply(
        lambda g: g.isna().any(axis=1).mean()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cell_nan_frac, bins=50)
    ax.set_xlabel('Fraction of frames with any NaN')
    ax.set_ylabel('Number of cells')
    ax.set_title('Per-cell NaN contamination')
    plt.tight_layout()
    plt.show()

    threshold = 0.1
    dirty = cell_nan_frac[cell_nan_frac > threshold]
    print(f"Cells with >{threshold*100:.0f}% NaN frames: {len(dirty)} / {len(cell_nan_frac)}")
    return cell_nan_frac


def check_df(df):
    # TODO: make this better.
    for c in 'u_t m_t recency burst_pos	n_5	slope_5	ewma_fast ewma_slow	s_cum '.split():
      assert df[c].notna().all(), f'{c} failed notNA check'

if __name__ == '__main__':
    exp_paths = [
    '/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-03_3-2-1minIntervals/', # 660 cells, # 180 min
    '/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-10-12_DoseResponse', # 800 cells, 40min
    '/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-11-02_Sustained_1min', # 1350 cells, 120 min
    '/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data/2025-09-04_RampReverse',
    ]
    # load these and process according to the above-defined pipeline, then save as dataset.parquet
    dfs = []
    for ep in exp_paths:    
        exp = Path(ep)
        p = exp / 'exp_data.parquet' 
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

        cell_sel_path = exp / 'cell_selection.csv'
        if cell_sel_path.exists():
            df_i = load_and_clean(p, cell_selection_csv=cell_sel_path)
        else:
            df_i = load_and_clean(p)
        dfs.append(df_i.copy())

    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    check_df(df)
    df.to_parquet('dataset.parquet')
    print('saved dataset...')

def filter_dead_cells(df,
                      intensity_col="mean_intensity_C0_nuc",
                      drop_ratio=0.4,
                      min_area=50,
                      remove_from="entire"):
    """Remove cells that die during the experiment.

    A cell is flagged as dead at the first frame where its nuclear
    intensity drops below ``drop_ratio`` of its per-cell median, or
    its nuclear area falls below ``min_area`` pixels.

    Parameters
    ----------
    intensity_col : str
        Column used to detect intensity collapse.
    drop_ratio : float
        Fraction of per-cell baseline median intensity below which a
        cell is considered dead (default 0.4 = 40 %). Baseline is the
        first 10 frames of each cell's track.
    min_area : int
        Minimum nuclear area in pixels.  Below this the cell is
        considered dead/fragmented.
    remove_from : {"death", "entire"}
        ``"death"`` removes only frames from the death timepoint onward.
        ``"entire"`` removes the whole cell track.
    """
    df = df.copy()
    area_col = "area_nuc" if "area_nuc" in df.columns else "area"

    # per-cell baseline median intensity as reference (first 10 frames)
    baseline = df.loc[df["frame"] < 10]
    med_int = baseline.groupby("uid")[intensity_col].median()
    df["_int_threshold"] = df["uid"].map(med_int) * drop_ratio

    dead_mask = (df[intensity_col] < df["_int_threshold"]) | (df[area_col] < min_area)

    if remove_from == "entire":
        dead_uids = df.loc[dead_mask, "uid"].unique()
        df = df[~df["uid"].isin(dead_uids)]
    else:
        # find first death frame per cell, drop everything from there on
        dead_frames = df.loc[dead_mask].groupby("uid")["frame"].min()
        for uid, death_frame in dead_frames.items():
            df = df[~((df["uid"] == uid) & (df["frame"] >= death_frame))]

    df.drop(columns=["_int_threshold"], inplace=True)
    return df


