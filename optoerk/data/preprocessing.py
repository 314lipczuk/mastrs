import os
from pathlib import Path
import pandas as pd
import numpy as np

from optoerk.core.utils import materials_path

# ===========================================================================
# Canonical schema — the single source of truth
# ===========================================================================
# Every raw experiment parquet carries its own column names; an ADAPTER renames
# and *projects* each family into CANONICAL_RAW_COLS (nothing legacy survives).
# clean() + derive_features() then operate only on canonical columns. A "bundle"
# is a row-wise concat of cleaned per-experiment canonical frames.

# The 9 stimulation feature channels (order is load-bearing — models index it).
STIM_COLS = [
    "u_t", "m_t", "recency", "ewma_fast", "ewma_slow",
    "n_5", "slope_5", "burst_pos", "s_cum",
]

# Exactly what an adapter must emit (everything else is dropped by projection).
CANONICAL_RAW_COLS = [
    "original_experiment_name",   # registry key, e.g. "bo_v8", "freepattern_v1"
    "stim_condition",             # canonical per-trajectory label (str)
    "fov", "particle",            # particle retained: builds uid + regroups
    "frame",                      # int per-cell frame index (1 frame = 1 min)
    "time_min",                   # real acquisition time (minutes)
    "x", "y",
    "nuc_area",
    "mean_intensity_C0_nuc", "mean_intensity_C1_nuc",
    "mean_intensity_C0_ring", "mean_intensity_C1_ring",
    "median_intensity_C0_nuc", "median_intensity_C1_nuc",
    "median_intensity_C0_ring", "median_intensity_C1_ring",
    "stim", "stim_power", "stim_exposure",
    "cell_line",                  # the EGFR filter in clean() keys on it
]

# Added by _derive_identity + clean + derive_features (stored in the bundle).
CANONICAL_DERIVED_COLS = [
    "uid", "cnr", "cnr_median", "cnr_median_norm", "cnr_mean_norm",
    "median_cnr_0_9", "energy_uJ", "fluence_mJ_cm2", "energy_per_cell",
    *STIM_COLS,
]

# Raw names that must never survive a hard-cut adapter (validator guards these).
_LEGACY_COLS = [
    "ramp_pattern_name", "treatment_name", "area", "area_nuc",
    "timestep", "time", "fov_name", "condition_idx",
    "channels", "ref_channels", "img_shape",
]

# Nested/object columns dropped up front (unrenderable, not needed downstream).
_OBJECT_COLS = [
    "channels", "ref_channels", "img_shape", "optocheck_channels",
    "stim_timestep", "stim_exposure_list", "stim_channel_name",
    "stim_channel_group", "stim_channel_device_name",
    "stim_channel_power_property_name",
]


# ---------------------------------------------------------------------------
# Adapters — raw family -> canonical raw columns
# ---------------------------------------------------------------------------

def _canonicalize_common(df, experiment_name, cell_line_default="EGFR"):
    """Mechanical raw->canonical renames shared by every adapter (not the label).

    Maps the time axes (``timestep``->``frame``, ``time``->``time_min``), the
    nuclear area (``area``/``area_nuc``->``nuc_area``), defaults ``cell_line``
    when the raw file lacks it, and fills missing stim scalars with 0.
    """
    df = df.copy()
    df["original_experiment_name"] = experiment_name

    df["frame"] = df["timestep"]
    if "time" in df.columns:
        df["time_min"] = df["time"]
    elif "time_min" not in df.columns:
        df["time_min"] = df["timestep"].astype(float)  # 1 frame = 1 min fallback

    if "area_nuc" in df.columns:
        df["nuc_area"] = df["area_nuc"]
    elif "area" in df.columns:
        df["nuc_area"] = df["area"]

    if "cell_line" not in df.columns:
        df["cell_line"] = cell_line_default

    if "stim_power" in df.columns:
        df["stim_power"] = df["stim_power"].fillna(0)
    df["stim_exposure"] = df["stim_exposure"].fillna(0)
    return df


def _project_canonical(df):
    """Enforce the hard cut: keep exactly CANONICAL_RAW_COLS, drop everything else."""
    missing = [c for c in CANONICAL_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"adapter output missing canonical columns: {missing}")
    return df[CANONICAL_RAW_COLS].copy()


def adapt_standard(df, experiment_name, cell_line="EGFR", **kw):
    """Standard optoRTK experiments — ``ramp_pattern_name`` is the label."""
    df = df.copy()
    df["stim_condition"] = df["ramp_pattern_name"].astype(str)
    df = _canonicalize_common(df, experiment_name, cell_line)
    return _project_canonical(df)


def adapt_bo(df, experiment_name, bo_tag="v8", cell_line="EGFR", **kw):
    """BO oscillation experiments — synthesize the label from ``condition_idx``."""
    df = df.drop(columns=[c for c in _OBJECT_COLS if c in df.columns])
    if "condition_idx" in df.columns:
        df["stim_condition"] = f"bo_osc_{bo_tag}_c" + df["condition_idx"].astype(str)
    else:
        df["stim_condition"] = f"bo_osc_{bo_tag}"
    df = _canonicalize_common(df, experiment_name, cell_line)
    return _project_canonical(df)


def adapt_freepattern(df, experiment_name, cell_line="EGFR", **kw):
    """FreePatternStim experiments — ``treatment_name`` is the label.

    The raw integer ``uid`` is a *treatment* id (one FOV per treatment), NOT a
    cell id; it is dropped so _derive_identity rebuilds a real per-cell uid.
    """
    df = df.drop(columns=[c for c in _OBJECT_COLS if c in df.columns])
    df["stim_condition"] = df["treatment_name"].astype(str)
    df = _canonicalize_common(df, experiment_name, cell_line)
    return _project_canonical(df)


ADAPTERS = {
    "standard": adapt_standard,
    "bo": adapt_bo,
    "freepattern": adapt_freepattern,
}


def adapt(df, experiment, experiment_name, **kw):
    """Dispatch to the adapter for ``experiment`` (unknown -> standard)."""
    fn = ADAPTERS.get(experiment or "standard", adapt_standard)
    return fn(df, experiment_name, **kw)


# ---------------------------------------------------------------------------
# Identity derivation / cleaning / feature derivation (all on canonical cols)
# ---------------------------------------------------------------------------

def _derive_identity(df):
    """cnr / cnr_median / uid from canonical raw columns.

    uid = ``{original_experiment_name}__{stim_condition}__{fov}__{particle}`` —
    globally unique across a concatenated bundle.
    """
    df = df.copy()
    df["cnr"] = df["mean_intensity_C1_ring"] / df["mean_intensity_C1_nuc"]
    df["cnr_median"] = df["median_intensity_C1_ring"] / df["median_intensity_C1_nuc"]
    df["uid"] = (
        df["original_experiment_name"].astype(str) + "__"
        + df["stim_condition"].astype(str) + "__"
        + df["fov"].astype(str) + "__"
        + df["particle"].astype(str)
    )
    return df


def clean(
    df,
    *,
    cell_selection_csv=None,
    tracking_threshold=0.9,
    norm_until_timepoint=10,
    baseline_cnr_max=0.8,
    cell_line="EGFR",
):
    """Drop short tracks, baseline-normalize CNR, apply selection/filters.

    Operates on a canonical frame that already has ``uid``/``frame``/``cnr``.
    """
    # --- drop short tracks --------------------------------------------------
    frame_counts = df["uid"].value_counts()
    threshold = tracking_threshold * frame_counts.max()
    valid_uids = frame_counts[frame_counts >= threshold].index
    df = df[df["uid"].isin(valid_uids)].copy()

    # --- normalise CNR (median) ---------------------------------------------
    baseline_median = (
        df.loc[df["frame"] < norm_until_timepoint].groupby("uid")["cnr_median"].median()
    )
    df["cnr_median_norm"] = df["uid"].map(baseline_median)
    df["cnr_median_norm"] = df["cnr_median"] / df["cnr_median_norm"]
    df = df.dropna(subset=["cnr_median_norm"])

    # --- normalise CNR (mean) -----------------------------------------------
    baseline_mean = (
        df.loc[df["frame"] < norm_until_timepoint].groupby("uid")["cnr"].median()
    )
    df["cnr_mean_norm"] = df["uid"].map(baseline_mean)
    df["cnr_mean_norm"] = df["cnr"] / df["cnr_mean_norm"]
    df = df.dropna(subset=["cnr_mean_norm"])

    # --- manual cell selection ----------------------------------------------
    # NOTE: cell_selection.csv `uid` must use the canonical uid format; legacy
    # selection files (ramp_pattern_name+fov+particle) need regeneration.
    if cell_selection_csv is not None and os.path.isfile(cell_selection_csv):
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

    # --- cell line filtering ------------------------------------------------
    df = df[df["cell_line"] == cell_line]
    return df


def derive_features(df, calib=None):
    """calc_power(calib) + the 9 stim features; returns a fresh-indexed frame."""
    df = calc_power(df, calib=calib)
    df = add_stim_features(df)
    return df.reset_index(drop=True)


def load_and_clean(
    parquet_path: "str | pd.DataFrame",
    *,
    experiment: str = "standard",
    experiment_name: str | None = None,
    instrument: str | None = None,
    cell_selection_csv: str | None = None,
    tracking_threshold: float = 0.9,
    norm_until_timepoint: int = 10,
    baseline_cnr_max: float | None = 0.8,
    cell_line: str | None = "EGFR",
    **adapter_kw,
) -> pd.DataFrame:
    """Read one raw experiment parquet and return a cleaned canonical frame.

    Orchestrates: read -> adapt(experiment) -> _derive_identity -> clean ->
    derive_features. ``experiment`` selects the adapter (``standard``/``bo``/
    ``freepattern``); ``experiment_name`` is the canonical
    ``original_experiment_name`` value (defaults to ``experiment``);
    ``instrument`` selects the power-calibration curve from :data:`CALIBRATIONS`.
    """
    if isinstance(parquet_path, pd.DataFrame):
        df = parquet_path.copy()
    else:
        df = pd.read_parquet(parquet_path)

    if experiment_name is None:
        experiment_name = experiment

    df = adapt(df, experiment, experiment_name, cell_line=cell_line, **adapter_kw)
    df = _derive_identity(df)
    df = clean(
        df,
        cell_selection_csv=cell_selection_csv,
        tracking_threshold=tracking_threshold,
        norm_until_timepoint=norm_until_timepoint,
        baseline_cnr_max=baseline_cnr_max,
        cell_line=cell_line,
    )
    df = derive_features(df, calib=CALIBRATIONS[instrument or DEFAULT_INSTRUMENT])
    return df


# Power calibration curves, one per microscope. `stim_power` (0..100 %) maps to
# optical power (uW) and irradiance (mW/cm2) by piecewise-linear interpolation.
# Keyed by instrument name; add a new entry when a second scope comes online.
# No time dimension — curves are assumed stable per scope.
# Both curves convert uW -> mW/cm2 with the SAME illuminated-area constant:
# irradiance = uW * 1e-3 / A, A = 0.006362 cm2 (a 900 um-diameter field). This is
# the lab convention from `Lichtintensität Mönch.xlsx`; the "jungfrau" numbers are
# that sheet's **ND5** column (uW + mW/cm2) verbatim. Niesen was measured with no
# ND filter (dichroics only), so its uW is ~70x higher at the same %; applying the
# same area keeps its dose on the identical mJ/cm2 axis as every other experiment.
CALIBRATIONS = {
    "jungfrau": dict(
        pct=[0,1,2,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
        uW=[8,11,17,26.2,29.5,49.9,71.3,90.5,111,132,153,172,192,212,231,249,268,287,305,321,339,356,374,389],
        mW_cm2=[1.26,1.73,2.67,4.12,4.64,7.84,11.21,14.23,17.45,20.75,24.05,27.04,30.18,33.32,36.31,39.14,42.13,45.11,47.94,50.46,53.29,55.96,58.79,61.15],
    ),
    # Niesen DMD (470/40), "1st run on Niesen", no ND filter. mW_cm2 = uW * 0.15719
    # (same 0.006362 cm2 area as jungfrau/Mönch). Experiment ran at a fixed 10%.
    "niesen": dict(
        pct=[0, 2, 3, 5, 10, 30, 99],
        uW=[0, 321, 936, 1630, 3499, 9888, 23800],
        mW_cm2=[0.0, 50.46, 147.13, 256.22, 550.01, 1554.29, 3741.12],
    ),
}
DEFAULT_INSTRUMENT = "jungfrau"


def calc_power(df, calib=None):
    """Add energy/fluence columns from ``stim_power`` via an instrument curve.

    ``calib`` is one entry of :data:`CALIBRATIONS` (keys ``pct``/``uW``/
    ``mW_cm2``); defaults to the :data:`DEFAULT_INSTRUMENT` curve.
    """
    if calib is None:
        calib = CALIBRATIONS[DEFAULT_INSTRUMENT]
    P_uW = np.interp(df["stim_power"], calib["pct"], calib["uW"])           # microwatts
    irradiance = np.interp(df["stim_power"], calib["pct"], calib["mW_cm2"]) # mW/cm2

    # Energy per pulse
    # More precisely: energy (uJ) = P (uW) * t (ms) * 1e-3
    df["energy_uJ"] = P_uW * df["stim_exposure"] * 1e-3

    # Fluence (energy dose per unit area) per pulse
    df["fluence_mJ_cm2"] = irradiance * df["stim_exposure"] * 1e-3  # mW/cm2 * ms * 1e-3 = mJ/cm2

    df['energy_per_cell'] = df['fluence_mJ_cm2'] * df['nuc_area'] # not in any units, since area is just pixels.
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

def augment(df, baseline_frames=10, responder_sigma=2.0):
    """Add per-cell baseline decomposition, sensitivity, and responder columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataset produced by ``load_and_clean`` (must have
        ``uid``, ``frame``, ``cnr_median_norm``, ``median_intensity_C1_ring``,
        ``median_intensity_C1_nuc``, ``fluence_mJ_cm2``, ``energy_uJ``,
        ``ewma_slow``).
    baseline_frames : int
        Number of initial frames (0 .. baseline_frames-1) used to compute
        per-cell baselines (default 10).
    responder_sigma : float
        A cell is classified as a responder when its amplitude exceeds
        ``responder_sigma`` × baseline noise std (default 2.0).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the following columns added:

        Per-cell scalars (broadcast to every row):
            baseline_ring, baseline_nuc, peak_cnr, amplitude,
            total_fluence, total_energy, sensitivity_fluence,
            sensitivity_energy, responder

        Per-timepoint:
            delta_cnr, sensitivity_ewma
    """
    df = df.copy()
    bl = df.loc[df["frame"] < baseline_frames]

    # --- 1. Baseline decomposition (per-cell) ------------------------------
    baseline_ring = bl.groupby("uid")["median_intensity_C1_ring"].median()
    baseline_nuc = bl.groupby("uid")["median_intensity_C1_nuc"].median()
    df["baseline_ring"] = df["uid"].map(baseline_ring)
    df["baseline_nuc"] = df["uid"].map(baseline_nuc)

    # --- 2. Peak / amplitude (per-cell) ------------------------------------
    peak_cnr = df.groupby("uid")["cnr_median_norm"].max()
    df["peak_cnr"] = df["uid"].map(peak_cnr)
    df["amplitude"] = df["peak_cnr"] - 1.0

    # --- 3. Total dose (per-cell) ------------------------------------------
    total_fluence = df.groupby("uid")["fluence_mJ_cm2"].sum()
    total_energy = df.groupby("uid")["energy_uJ"].sum()
    df["total_fluence"] = df["uid"].map(total_fluence)
    df["total_energy"] = df["uid"].map(total_energy)

    # --- 4. Dose-response sensitivity (per-cell) --------------------------
    df["sensitivity_fluence"] = np.where(
        df["total_fluence"] > 0,
        df["amplitude"] / df["total_fluence"],
        np.nan,
    )
    df["sensitivity_energy"] = np.where(
        df["total_energy"] > 0,
        df["amplitude"] / df["total_energy"],
        np.nan,
    )

    # --- 5. Per-timepoint sensitivity --------------------------------------
    df["delta_cnr"] = df["cnr_median_norm"] - 1.0
    df["sensitivity_ewma"] = np.where(
        df["ewma_slow"] > 0,
        df["delta_cnr"] / df["ewma_slow"],
        np.nan,
    )

    # --- 6. Responder classification (per-cell) ----------------------------
    baseline_noise_std = bl["cnr_median_norm"].std()
    threshold = responder_sigma * baseline_noise_std
    df["responder"] = df["amplitude"] > threshold

    return df


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
        Stimulus feature columns to extract.  ``None`` → ``STIM_COLS``.

    Returns
    -------
    erk : np.ndarray, shape (n_windows, window_size)
    stim : np.ndarray, shape (n_windows, n_stim_cols, window_size)
    meta : pd.DataFrame  — one row per window with uid, window_start, and first-row metadata.
    """
    if stim_cols is None:
        stim_cols = STIM_COLS
    if window_size is None:
        window_size = int(df.groupby("stim_condition")["frame"].nunique().min())
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
                "stim_condition": first_row.get("stim_condition", None),
                "original_experiment_name": first_row.get("original_experiment_name", None),
                "fov": first_row.get("fov", None),
            })

    erk = np.array(erk_windows, dtype=np.float32)
    stim = np.array(stim_windows, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return erk, stim, meta


def make_tracks(df, value_col="cnr_median_norm", stim_cols=None,
                drop_nan_cells=True):
    """Return per-cell full trajectories (no windowing, variable length).

    Each cell contributes one entry in the returned object arrays. Downstream
    code should treat ``cnr`` / ``stim`` as sequences of per-cell arrays (fancy
    indexing by track id works; ``.shape[1]`` does not).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataset with ``uid``, ``frame``, *value_col*, and *stim_cols*.
    value_col, stim_cols
        See ``make_windows``. ``stim_cols`` defaults to ``STIM_COLS``.
    drop_nan_cells : bool
        Skip cells whose trajectory contains any NaN in value or stim columns.

    Returns
    -------
    cnr : np.ndarray(dtype=object), shape (n_cells,)
        Per-cell CNR trajectories, each a 1D float32 array of shape ``(T_cell,)``.
    stim : np.ndarray(dtype=object), shape (n_cells,)
        Per-cell stim arrays, each 2D float32 of shape ``(n_stim, T_cell)``.
    meta : pd.DataFrame
        One row per kept cell with uid + first-frame metadata + ``T``.
    """
    if stim_cols is None:
        stim_cols = STIM_COLS

    cnr_list, stim_list, meta_rows = [], [], []
    for uid, g in df.groupby("uid"):
        g = g.sort_values("frame")
        vals = g[value_col].to_numpy(dtype=np.float32)
        stim_mat = g[stim_cols].to_numpy(dtype=np.float32)  # (T, n_stim)
        if drop_nan_cells and (np.isnan(vals).any() or np.isnan(stim_mat).any()):
            continue
        first = g.iloc[0]
        cnr_list.append(vals)
        stim_list.append(stim_mat.T)  # (n_stim, T)
        meta_rows.append({
            "uid": uid,
            "cell_line": first.get("cell_line", None),
            "stim_condition": first.get("stim_condition", None),
            "original_experiment_name": first.get("original_experiment_name", None),
            "fov": first.get("fov", None),
            "T": int(len(vals)),
        })

    cnr = np.empty(len(cnr_list), dtype=object)
    stim = np.empty(len(stim_list), dtype=object)
    for i, v in enumerate(cnr_list):
        cnr[i] = v
    for i, s in enumerate(stim_list):
        stim[i] = s
    meta = pd.DataFrame(meta_rows)
    return cnr, stim, meta


def add_crowding_features(
    df,
    radius: float = 200.0,
    group_cols=("original_experiment_name", "fov", "frame"),
):
    """Add per-row spatial-crowding features from ``x``/``y`` positions.

    Adds two columns:
      ``fov_density``   — number of detected cells in the same field at the same
                          frame.
      ``n_cells_200px`` — number of *other* cells within ``radius`` pixels (same
                          field + frame).

    Call on the **raw** dataframe, BEFORE dropping short tracks, so neighbour
    counts include every detected cell. ``group_cols`` is the physical
    (experiment, fov, frame) key matching ``uid`` construction in
    ``load_and_clean``.
    """
    from scipy.spatial import cKDTree

    xy = df[["x", "y"]].to_numpy(dtype=float)
    n = len(df)
    fov_density = np.ones(n, dtype=np.float32)
    n_in_radius = np.zeros(n, dtype=np.float32)
    for _, idx in df.groupby(list(group_cols)).indices.items():
        m = len(idx)
        fov_density[idx] = m
        if m > 1:
            cnts = cKDTree(xy[idx]).query_ball_point(xy[idx], r=radius, return_length=True)
            n_in_radius[idx] = cnts - 1  # exclude self
    out = df.copy()
    out["fov_density"] = fov_density
    out["n_cells_200px"] = n_in_radius
    return out


def add_optortk_expression(
    df,
    *,
    baseline_frames: int = 10,
    cohort_col: str = "original_experiment_name",
):
    """Add per-cell optoRTK expression as a session-relative percentile rank.

    Expression proxy = whole-cell C0 intensity (mean of nuclear + ring C0),
    summarized per cell by its **median over the baseline frames**
    (``frame < baseline_frames``, i.e. pre-stimulation), then **rank-normalized
    to (0, 1] within each cohort** (default: ``original_experiment_name`` = one
    imaging session). Broadcast to every frame of the cell — expression is static
    (within-cell temporal CV ~1%). Column added: ``optortk_expr``.

    Ranking within the session is invariant to the per-session imaging gain (raw
    C0 differs ~2x across sessions), so the feature is comparable across
    experiments and reproducible **live** from the current session's co-imaged
    cells — no frozen dataset statistics required (the population analog of the
    per-cell baseline normalization already used for CNR).

    Cells with no baseline frames get NaN (dropped downstream by
    ``make_tracks(drop_nan_cells=True)``).
    """
    df = df.copy()
    bl = df.loc[
        df["frame"] < baseline_frames,
        ["uid", cohort_col, "mean_intensity_C0_nuc", "mean_intensity_C0_ring"],
    ].copy()
    bl["_c0"] = 0.5 * (bl["mean_intensity_C0_nuc"] + bl["mean_intensity_C0_ring"])
    per_cell = bl.groupby("uid").agg(
        _expr=("_c0", "median"), _cohort=(cohort_col, "first")
    )
    per_cell["optortk_expr"] = (
        per_cell.groupby("_cohort")["_expr"].rank(pct=True).astype(np.float32)
    )
    df["optortk_expr"] = df["uid"].map(per_cell["optortk_expr"]).astype(np.float32)
    return df


def make_windows_sample(df, window_size=None, value_col="cnr_median_norm",
                        stim_cols=None, rng=None):
    """Sample one random window per cell track.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataset with ``uid``, ``frame``, *value_col*, and *stim_cols*.
    window_size : int or None
        Length of each window in frames.  ``None`` → shortest experiment length.
    value_col : str
        Column name for the target signal.
    stim_cols : list of str or None
        Stimulus feature columns.  ``None`` → ``STIM_COLS``.
    rng : np.random.Generator or int or None
        Random generator or seed for reproducibility.

    Returns
    -------
    erk : np.ndarray, shape (n_cells, window_size)
    stim : np.ndarray, shape (n_cells, n_stim_cols, window_size)
    meta : pd.DataFrame  — one row per cell with uid, window_start, and first-row metadata.
    """
    if stim_cols is None:
        stim_cols = STIM_COLS
    if window_size is None:
        window_size = int(df.groupby("stim_condition")["frame"].nunique().min())
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    erk_windows, stim_windows, meta_rows = [], [], []

    for uid, g in df.groupby("uid"):
        g = g.sort_values("frame")
        vals = g[value_col].values
        stim_mat = g[stim_cols].values
        frames = g["frame"].values
        n = len(vals)

        if n < window_size:
            continue

        start = rng.integers(0, n - window_size + 1)
        chunk_v = vals[start : start + window_size]
        chunk_s = stim_mat[start : start + window_size]
        if np.isnan(chunk_v).any() or np.isnan(chunk_s).any():
            continue

        erk_windows.append(chunk_v)
        stim_windows.append(chunk_s.T)
        meta_rows.append({
            "uid": uid,
            "window_start": int(frames[start]),
            "cell_line": g.iloc[0].get("cell_line", None),
            "stim_condition": g.iloc[0].get("stim_condition", None),
            "original_experiment_name": g.iloc[0].get("original_experiment_name", None),
            "fov": g.iloc[0].get("fov", None),
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
        The combined dataset (must contain ``original_experiment_name`` and
        ``uid`` columns).
    """
    import matplotlib.pyplot as plt

    # Per-column NaN count per experiment
    nan_by_exp = pd.DataFrame({
        pat: df[df['original_experiment_name'] == pat].isna().sum()
        for pat in df['original_experiment_name'].unique()
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


def validate_canonical(df, *, derived=True):
    """Assert a frame conforms to the canonical schema (the hard-cut contract).

    Checks the canonical raw columns are present, no legacy column leaked, and
    (when ``derived``) the derived columns exist and the 9 stim features are
    non-NaN.
    """
    for c in CANONICAL_RAW_COLS:
        assert c in df.columns, f"missing canonical column {c!r}"
    for legacy in _LEGACY_COLS:
        assert legacy not in df.columns, f"legacy column {legacy!r} leaked"
    if derived:
        for c in STIM_COLS:
            assert df[c].notna().all(), f"{c} has NaN"
        for c in CANONICAL_DERIVED_COLS:
            assert c in df.columns, f"missing derived column {c!r}"


# ===========================================================================
# Experiment registry + bundles
# ===========================================================================
# One declarative entry per raw experiment (name -> dir + adapter + kwargs).
# A BUNDLE is a named list of experiments concatenated into one canonical frame.

_OPTORTK = "/Volumes/imaging.data/PertzLab/optoRTK_CedricZ/experimental_data"
_BO = "/Volumes/imaging.data/PertzLab/Alex/Oscillation_BO"
_FPS = "/Volumes/imaging.data/PertzLab/Alex/FreePatternStimulation"
_BSC ="/Volumes/imaging.data/PertzLab/optoRTK_BA/experimental_data"

# `instrument` selects the power-calibration curve from CALIBRATIONS. All
# experiments so far ran on "jungfrau"; a second scope adds its own curve there
# and sets instrument= on its entries here (nothing else changes).
EXPERIMENTS = {
    "3-2-1minIntervals": dict(dir=f"{_OPTORTK}/2025-11-03_3-2-1minIntervals", adapter="standard", instrument="jungfrau", kwargs={}),
    "DoseResponse":      dict(dir=f"{_OPTORTK}/2025-10-12_DoseResponse",       adapter="standard", instrument="jungfrau", kwargs={}),
    "Sustained_1min":    dict(dir=f"{_OPTORTK}/2025-11-02_Sustained_1min",     adapter="standard", instrument="jungfrau", kwargs={}),
    "RampReverse":       dict(dir=f"{_OPTORTK}/2025-09-04_RampReverse",        adapter="standard", instrument="jungfrau", kwargs={}),
    "bo_v8":     dict(dir=f"{_BO}/2026-05-01_bo_erk_oscillation_v8_freq_range_wider",             adapter="bo", instrument="jungfrau", kwargs={"bo_tag": "v8"}),
    "bo_v10":    dict(dir=f"{_BO}/2026-05-07_bo_erk_oscillation_v10_led_power",                   adapter="bo", instrument="jungfrau", kwargs={"bo_tag": "v10"}),
    "bo_v11_10s": dict(dir=f"{_BO}/2026-05-08_bo_erk_oscillation_v11_light_budget_fixed10s_pi10", adapter="bo", instrument="jungfrau", kwargs={"bo_tag": "v11_10s"}),
    "bo_v11_20s": dict(dir=f"{_BO}/2026-05-08_bo_erk_oscillation_v11_light_budget_fixed20s_pi10", adapter="bo", instrument="jungfrau", kwargs={"bo_tag": "v11_20s"}),
    "freepattern_v1":       dict(dir=f"{_FPS}/2026-06-23_FreePatternStim_Jungfrau_v1",        adapter="freepattern", instrument="jungfrau", kwargs={}),
    "freepattern_v2":       dict(dir=f"{_FPS}/2026-06-26_FreePatternStim_Jungfrau_v2",        adapter="freepattern", instrument="jungfrau", kwargs={}),
    "freepattern_TrKA1_v1": dict(dir=f"{_FPS}/2026-06-30_FreePatternStim_Jungfrau_TrKA1_v1",  adapter="freepattern", instrument="jungfrau", kwargs={}),
    "freepattern_TrKA1_v2": dict(dir=f"{_FPS}/2026-07-03_FreePatternStim_Jungfrau_TrKA1_v2",  adapter="freepattern", instrument="jungfrau", kwargs={}),
    "freepattern_Niesen_EGFR_v1": dict(dir=f"{_FPS}/2026-07-03_FreePatternStim_Niesen_EGFR_v1", adapter="freepattern", instrument="niesen", kwargs={}),
    "long": dict(dir=f"{_BSC}/LongTermExperiments/2026-03-21_RampLongTerm_Drug",  adapter="freepattern", instrument="jungfrau", kwargs={}),
}

# Instrument recoverable per row from original_experiment_name (no stored column).
EXPERIMENT_INSTRUMENT = {name: spec["instrument"] for name, spec in EXPERIMENTS.items()}

# Named bundles -> output parquet paths. `real` is the pre-BO snapshot;
# `real_plus_bo` is the post-BO union kept at the canonical `dataset.parquet`.
BUNDLES = {
    #"real": ["3-2-1minIntervals", "DoseResponse", "Sustained_1min", "RampReverse",
    #                 "bo_v8", "bo_v10", "bo_v11_10s", "bo_v11_20s"], # legacy reasons, same as cedric_and_bo
    "cedric":         ["3-2-1minIntervals", "DoseResponse", "Sustained_1min", "RampReverse"],

    "bo": ["bo_v8", "bo_v10", "bo_v11_10s", "bo_v11_20s"],
    "cedric_and_bo": ["3-2-1minIntervals", "DoseResponse", "Sustained_1min", "RampReverse",
                     "bo_v8", "bo_v10", "bo_v11_10s", "bo_v11_20s"],
    "freepattern":  ["freepattern_v1", "freepattern_v2",],
    "all": ["3-2-1minIntervals", "DoseResponse", "Sustained_1min", "RampReverse",
                     "bo_v8", "bo_v10", "bo_v11_10s", "bo_v11_20s", "freepattern_v1", "freepattern_v2",
                     "freepattern_Niesen_EGFR_v1"],
    "niesen": ["freepattern_Niesen_EGFR_v1"],
}

OUT_PATHS = {
    "real": materials_path("dataset.parquet.v0"),
    "real_plus_bo": materials_path("dataset.parquet"),
    "freepattern": materials_path("dataset_freepattern.parquet"),
    # `all` = every training experiment incl. the high-dose Niesen run (see
    # NIESEN_TOCHECK.md). This is the bundle the full-history model now trains on.
    "all": materials_path("dataset_all.parquet"),
    # `niesen` = the high-dose Niesen run alone (isolated training probe).
    "niesen": materials_path("dataset_niesen.parquet"),
}


def build_experiment(name, **clean_kwargs):
    """Load + clean one registered experiment into a canonical frame."""
    spec = EXPERIMENTS[name]
    exp_dir = Path(spec["dir"])
    p = exp_dir / "exp_data.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    cell_sel = exp_dir / "cell_selection.csv"
    return load_and_clean(
        p,
        experiment=spec["adapter"],
        experiment_name=name,
        instrument=spec["instrument"],
        cell_selection_csv=str(cell_sel) if cell_sel.exists() else None,
        **spec["kwargs"],
        **clean_kwargs,
    )


def build_bundle(names, **clean_kwargs):
    """Concatenate cleaned canonical frames for a list of experiment names."""
    dfs = [build_experiment(n, **clean_kwargs) for n in names]
    df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    validate_canonical(df, derived=True)
    return df


if __name__ == '__main__':
    import sys as _sys
    # Usage: python preprocessing.py [bundle_name ...] [--out PATH]
    # Default rebuilds `real` and `real_plus_bo` to their canonical OUT_PATHS.
    args = _sys.argv[1:]
    out_override = None
    if "--out" in args:
        i = args.index("--out")
        out_override = args[i + 1]
        args = args[:i] + args[i + 2:]
    targets = args or BUNDLES.keys()
    for bundle_name in targets:
        print(f"building bundle {bundle_name!r} ...", flush=True)
        df = build_bundle(BUNDLES[bundle_name])
        out = out_override or OUT_PATHS.get(bundle_name) or materials_path(f"{bundle_name}.parquet")
        df.to_parquet(out)
        print(f"  -> {out}  ({len(df):,} rows, {df['uid'].nunique():,} cells)")

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

    # per-cell baseline median intensity as reference (first 10 frames)
    baseline = df.loc[df["frame"] < 10]
    med_int = baseline.groupby("uid")[intensity_col].median()
    df["_int_threshold"] = df["uid"].map(med_int) * drop_ratio

    dead_mask = (df[intensity_col] < df["_int_threshold"]) | (df["nuc_area"] < min_area)

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


