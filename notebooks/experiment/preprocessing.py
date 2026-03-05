import os
from pathlib import Path
import pandas as pd
import numpy as np

def load_and_clean(
    parquet_path: str,
    cell_selection_csv: str | None = None,
    tracking_threshold: float = 0.9,
    norm_until_timepoint: int = 10,
    baseline_cnr_max: float | None = None,
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
    df["cnr_norm"] = df["uid"].map(baseline_mean)
    df["cnr_norm"] = df["cnr"] / df["cnr_norm"]
    df.dropna(subset=["cnr_norm"], inplace=True)

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

    df = df.groupby("uid", group_keys=False).apply(_cell_features)

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




