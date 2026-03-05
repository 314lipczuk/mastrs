# `exp_data.parquet` Column Reference

This document describes the DataFrame produced by the rtm-pymmcore pipeline.
Each row represents **one tracked cell at one timepoint**.

---

## Acquisition Plan Columns

These columns originate from `generate_df_acquire()` in `utils.py`. They describe *what* the microscope should do. One row per (FOV, timestep) pair is created, then broadcast onto every detected cell.

| Column | Type | Description |
|---|---|---|
| `fov` | uint16 | Field-of-view index (0-based). |
| `fov_x` | float64 | Stage X coordinate of this FOV (from MDA position list). |
| `fov_y` | float64 | Stage Y coordinate of this FOV. |
| `fov_name` | object | Human-readable FOV name. Defaults to `str(fov_index)` if not set in MDA. |
| `timestep` | uint32 | Frame index within the experiment (0-based). |
| `time` | float32 | Planned wall-clock time offset in seconds from experiment start. Accounts for FOV interleaving: `start_time + fov_group * cycle_time + timestep * time_between_timesteps`. |
| `fname` | object | Filename stem for the raw TIFF, e.g. `"003_01_00045"` = FOV 3, phase 1, timestep 45. |
| `cell_line` | object | Biological condition label assigned to this FOV (e.g. `"WT"`, `"KO"`). Set via the `condition` argument. |
| `channels` | object | Tuple of dicts, one per imaging channel. Each dict has keys: `name`, `exposure` (ms), `group`, `power`, `device_name`, `property_name`. See [Channels object](#channels-object). |
| `optocheck` | bool | `True` at timesteps where an optogenetic expression check image is acquired. |
| `optocheck_channels` | object | Tuple of channel dicts for the optocheck acquisition. Same structure as `channels`. |
| `phase` | object | Phase name (multi-phase experiments only). |
| `phase_id` | int | Phase index (multi-phase experiments only). |

### Channels object

The `channels` column stores a **tuple of dicts** derived from `Channel` dataclass instances:

```python
(
  {"name": "miRFP", "exposure": 150, "group": None, "power": 80, ...},
  {"name": "mScarlet3", "exposure": 150, "group": None, "power": 20, ...},
)
```

Channel index matters: C0 is the first dict, C1 the second, etc. This determines which channel is used for segmentation (typically C0) and which is the biosensor readout (typically C1).

---

## Stimulation Columns

Added by `apply_stim_treatments_to_df_acquire()`, which merges `StimTreatment` dataclass instances onto `df_acquire` by FOV.

| Column | Type | Description |
|---|---|---|
| `stim_power` | int64 | Hardware power setting for the stimulation light source (dimensionless, typically 0-100). Not physical mW -- requires calibration to convert. |
| `stim_channel_name` | object | Micro-Manager channel config name for stimulation (e.g. `"Cyan"`). |
| `stim_channel_group` | object | Micro-Manager channel group. |
| `stim_channel_device_name` | object | Hardware device name (e.g. `"Spectra"`). |
| `stim_channel_power_property_name` | object | Device property controlling intensity (e.g. `"Cyan_Level"`). |
| `stim_timestep` | object | Tuple of all timestep indices where stimulation occurs for this FOV. Same value in every row of a given FOV. |
| `stim_exposure_list` | object | Tuple of exposure durations (ms), 1:1 with `stim_timestep`. Same value in every row of a given FOV. |
| `stim_exposure` | float32 | **Per-row** stimulation exposure in ms. Looked up from `stim_exposure_list` at the matching timestep; `0.0` if no stimulation at this frame. |
| `stim` | bool | `True` if `timestep in stim_timestep AND stim_exposure > 0`. |
| `ramp_pattern_name` | object | Human label for stimulation ramp pattern (used in ramp/dose-response experiments). |

### Computing light dose

`stim_power` is a hardware setting (0-100), not physical power. To convert to absolute units, use the calibration table from `microscope calibration curves.xlsx` with the **ND5** filter.

#### ND5 calibration table

| LED power (%) | Power (uW) | Irradiance (mW/cm2) |
|---:|---:|---:|
| 0 | 8 | 1.26 |
| 1 | 11 | 1.73 |
| 2 | 17 | 2.67 |
| 4 | 26.2 | 4.12 |
| 5 | 29.5 | 4.64 |
| 10 | 49.9 | 7.84 |
| 15 | 71.3 | 11.21 |
| 20 | 90.5 | 14.23 |
| 25 | 111 | 17.45 |
| 30 | 132 | 20.75 |
| 35 | 153 | 24.05 |
| 40 | 172 | 27.04 |
| 45 | 192 | 30.18 |
| 50 | 212 | 33.32 |
| 55 | 231 | 36.31 |
| 60 | 249 | 39.14 |
| 65 | 268 | 42.13 |
| 70 | 287 | 45.11 |
| 75 | 305 | 47.94 |
| 80 | 321 | 50.46 |
| 85 | 339 | 53.29 |
| 90 | 356 | 55.96 |
| 95 | 374 | 58.79 |
| 100 | 389 | 61.15 |

#### Converting to physical dose

```python
import numpy as np

# ND5 calibration data
cal_power_pct = [0,1,2,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
cal_uW        = [8,11,17,26.2,29.5,49.9,71.3,90.5,111,132,153,172,192,212,231,249,268,287,305,321,339,356,374,389]
cal_mW_cm2    = [1.26,1.73,2.67,4.12,4.64,7.84,11.21,14.23,17.45,20.75,24.05,27.04,30.18,33.32,36.31,39.14,42.13,45.11,47.94,50.46,53.29,55.96,58.79,61.15]

# Interpolate to get physical power for any stim_power setting
P_uW = np.interp(df["stim_power"], cal_power_pct, cal_uW)           # microwatts
irradiance = np.interp(df["stim_power"], cal_power_pct, cal_mW_cm2) # mW/cm2

# Energy per pulse
df["energy_uJ"] = P_uW * df["stim_exposure"]  # uW * ms = nJ * 1e3 -> multiply by 1e-3 for uJ
# More precisely: energy (uJ) = P (uW) * t (ms) * 1e-3
df["energy_uJ"] = P_uW * df["stim_exposure"] * 1e-3

# Fluence (energy dose per unit area) per pulse
df["fluence_mJ_cm2"] = irradiance * df["stim_exposure"] * 1e-3  # mW/cm2 * ms * 1e-3 = mJ/cm2
```

---

## Image Measurement Columns

Produced by `FE_ErkKtr.extract_features()` in `feature_extraction/erk_ktr.py`. For each frame, the segmented nuclear mask and a derived cytosolic ring mask are measured with `skimage.measure.regionprops_table`.

### Nuclear measurements (suffix `_nuc`)

Measured inside the segmentation label mask (nuclei).

| Column | Type | Description |
|---|---|---|
| `mean_intensity_C0_nuc` | float64 | Mean pixel intensity of channel 0 inside each nucleus. |
| `mean_intensity_C1_nuc` | float64 | Mean pixel intensity of channel 1 inside each nucleus. |
| `median_intensity_C0_nuc` | float64 | Median pixel intensity of channel 0 inside each nucleus. |
| `median_intensity_C1_nuc` | float64 | Median pixel intensity of channel 1 inside each nucleus. |
| `label` | uint32 | Segmentation label ID (unique integer per cell in the mask for that frame). |
| `x` | float64 | Centroid row coordinate (pixels). From `regionprops` `centroid-0`. |
| `y` | float64 | Centroid column coordinate (pixels). From `regionprops` `centroid-1`. |
| `area` | float64 | Nuclear area in pixels. Named `area_nuc` when using `FE_ErkKtr`; `area` when using `SimpleFE`. |

### Ring measurements (suffix `_ring`)

Measured on an annular cytosolic ring around each nucleus: `expand_labels(distance) - expand_labels(margin)`. Default margin=2px, distance=4px.

| Column | Type | Description |
|---|---|---|
| `mean_intensity_C0_ring` | float64 | Mean pixel intensity of channel 0 in the cytosolic ring. |
| `mean_intensity_C1_ring` | float64 | Mean pixel intensity of channel 1 in the cytosolic ring. |
| `median_intensity_C0_ring` | float64 | Median pixel intensity of channel 0 in the cytosolic ring. |
| `median_intensity_C1_ring` | float64 | Median pixel intensity of channel 1 in the cytosolic ring. |

### Derived ratios

| Column | Type | Description |
|---|---|---|
| `cnr` | float64 | Cytoplasm-to-nucleus ratio: `mean_intensity_C1_ring / mean_intensity_C1_nuc`. The primary ERK-KTR biosensor readout. High = ERK active, low = ERK inactive. |
| `cnr_median` | float64 | Same ratio using median intensities. More robust to outlier pixels. |

---

## Tracking Columns

Produced by `TrackerTrackpy` in `tracking/trackpy.py` using the `trackpy` linking library.

| Column | Type | Description |
|---|---|---|
| `particle` | uint32 | Unique cell identity linked across frames by nearest-neighbor tracking within `search_range` pixels (default 50px). Persistent: same cell keeps the same `particle` ID across timepoints. |
| `fov_timestep` | uint32 | FOV-local frame counter. May differ from `timestep` in multi-phase experiments. |

---

## Optocheck Column

Produced by `OptoCheckFE` in `feature_extraction/optocheck_fe.py`.

| Column | Type | Description |
|---|---|---|
| `optocheck_mean_intensity` | float64 | Mean intensity of the optogenetic construct expression channel, measured per tracked cell (matched by `particle` ID). Used to verify that cells express the optogenetic tool. `NaN` for timepoints without an optocheck acquisition. |

---

## Post-hoc Analysis Columns

These are **not stored** in `exp_data.parquet`. They are computed at load time by viewer scripts or analysis notebooks.

| Column | Type | Description |
|---|---|---|
| `uid` | string | Unique cell identifier across the whole experiment: `str(fov) + "_" + str(particle)`. E.g. `"3_17"`. |
| `frame` | uint32 | Legacy alias for `timestep`. Present in older data formats. |
| `cnr_norm` | float64 | `cnr` normalized to each cell's pre-stimulation baseline: `cnr / mean(cnr over first N frames)`. Default N=10. |
| `cnr_median_norm` | float64 | Same normalization applied to `cnr_median`. |

---

## Preprocessing (`load_and_clean`)

Implemented in `notebooks/experiment/preprocessing.py`. This function loads a parquet file, computes derived columns, filters cells, normalizes biosensor readouts, and converts stimulation power to physical units. It is the standard entry point for analysis notebooks.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `parquet_path` | *(required)* | Path to the raw `exp_data.parquet`. |
| `cell_selection_csv` | `None` | Path to a `cell_selection.csv` for manual cell exclusion. Rows with `deleted == True` are removed. |
| `tracking_threshold` | `0.9` | Minimum fraction of the maximum frame count a cell must reach to be kept (e.g. 0.9 = 90%). |
| `norm_until_timepoint` | `10` | Frames `< norm_until_timepoint` define the per-cell baseline window for CNR normalization. |
| `baseline_cnr_max` | `None` | If set, cells whose baseline median CNR (`median_cnr_0_9`) exceeds this value are dropped. |
| `cell_line` | `"EGFR"` | Keep only rows matching this `cell_line` value. |

### Pipeline steps

1. **Derived columns** -- Computes `cnr`, `cnr_median`, `uid`, and `frame` (see columns table below).
2. **Short-track removal** -- Counts frames per `uid`. Cells with fewer frames than `tracking_threshold * max(frame_counts)` are dropped.
3. **CNR normalization (median)** -- For each cell, computes the median of `cnr_median` over the baseline window (`frame < norm_until_timepoint`). Divides every row's `cnr_median` by that baseline to produce `cnr_median_norm`. Rows with `NaN` are dropped.
4. **CNR normalization (mean)** -- Same procedure using `cnr` to produce `cnr_norm`. Rows with `NaN` are dropped.
5. **Manual cell selection** -- If `cell_selection_csv` is provided and exists, removes cells flagged as `deleted`.
6. **Baseline ERK summary** -- Computes per-cell median CNR over frames 0 to `norm_until_timepoint - 1` and merges it as `median_cnr_0_9`.
7. **Baseline threshold** -- If `baseline_cnr_max` is set, drops cells whose `median_cnr_0_9` exceeds the threshold.
8. **Cell-line filtering** -- Keeps only rows where `cell_line` matches the parameter value.
9. **Power calibration** (`calc_power`) -- Converts `stim_power` (hardware %) to physical units using ND5 calibration via linear interpolation. Adds `energy_uJ`, `fluence_mJ_cm2`, and `energy_per_cell`.

### Columns added by preprocessing

| Column | Type | Description |
|---|---|---|
| `cnr` | float64 | Cytoplasm-to-nucleus ratio: `mean_intensity_C1_ring / mean_intensity_C1_nuc`. |
| `cnr_median` | float64 | Same ratio using median intensities. |
| `uid` | string | Unique cell identifier: `ramp_pattern_name + str(fov) + "_" + str(particle)`. Incorporates the ramp pattern so cells from different stimulation patterns are distinguishable. |
| `frame` | uint32 | Alias for `timestep`. |
| `cnr_median_norm` | float64 | `cnr_median` divided by the per-cell baseline median of `cnr_median` (over frames `< norm_until_timepoint`). |
| `cnr_norm` | float64 | `cnr` divided by the per-cell baseline median of `cnr` (over frames `< norm_until_timepoint`). |
| `median_cnr_0_9` | float64 | Per-cell median of `cnr` over the baseline window (frames 0 to `norm_until_timepoint - 1`). Merged onto every row for that cell. |
| `energy_uJ` | float64 | Energy per stimulation pulse in microjoules: `P_uW * stim_exposure * 1e-3`. Interpolated from the ND5 calibration table. `0` when `stim_exposure` is `0`. |
| `fluence_mJ_cm2` | float64 | Fluence (energy dose per unit area) per pulse: `irradiance * stim_exposure * 1e-3` (mJ/cm2). |
| `energy_per_cell` | float64 | `fluence_mJ_cm2 * area`. Not in physical units since `area` is in pixels. **Warning:** nuclear area changes with ERK perturbation, so this column should not be used for quantitative comparisons. |

---

## Stimulation Input Features (`add_stim_features`)

Implemented in `notebooks/experiment/preprocessing.py`. This function augments the DataFrame with 9 per-cell stimulation input channels intended for modeling single-cell ERK responses. Assumes a uniform 1-minute grid (1 frame = 1 minute). Must be called **after** `load_and_clean` (which provides `fluence_mJ_cm2` and `stim`).

```python
from notebooks.experiment.preprocessing import add_stim_features
df = add_stim_features(df)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `df` | *(required)* | DataFrame produced by `load_and_clean`. |
| `window_min` | `5` | Rolling window size in frames (minutes) for `n_5` and `slope_5`. |
| `ewma_alpha_fast` | `0.5` | Smoothing factor for fast EWMA (half-life ~1 min). |
| `ewma_alpha_slow` | `0.1` | Smoothing factor for slow EWMA (half-life ~7 min). |

### Columns added

| Column | Type | Description |
|---|---|---|
| `u_t` | float64 | Raw pulse amplitude (`fluence_mJ_cm2`). Primary input signal. |
| `m_t` | int | Activation indicator: 1 if stimulated, 0 otherwise. Disambiguates rest from zero-amplitude frames. |
| `dt_since_pulse` | float64 | Minutes (frames) since the most recent pulse for this cell. `NaN` before the cell's first pulse. Captures gap dynamics. |
| `ewma_fast` | float64 | Exponentially weighted moving average of `u_t` with `alpha=0.5`. Short-term effective stimulation level (equivalent to a discretised first-order ODE with fast decay). |
| `ewma_slow` | float64 | EWMA of `u_t` with `alpha=0.1`. Medium-term accumulation (slow decay). |
| `n_5` | int | Number of pulses (`m_t == 1`) in the last `window_min` frames. Measures burst density. |
| `slope_5` | float64 | OLS slope of `u_t` over the last `window_min` frames. Detects ramp-up / ramp-down patterns. |
| `burst_pos` | int | 1-indexed position within the current consecutive burst of stimulated frames. 0 when the cell is not stimulated. Captures adaptation/facilitation within a burst. |
| `s_cum` | float64 | Cumulative sum of `u_t` up to and including the current frame. Total light exposure history. |

### Design rationale

- **`u_t`, `m_t`, `dt_since_pulse`** are direct encodings of the stimulation protocol — essentially free and universally useful.
- **`ewma_fast`, `ewma_slow`** bridge deep learning and classical compartmental modeling: each EWMA is a discretised first-order ODE with a different time constant.
- **`n_5`, `slope_5`, `burst_pos`** capture higher-level temporal patterns (burst density, ramps, within-burst position).
- **`s_cum`** provides the model with long-term exposure history.

---

## Stratifying by Experimental Groups

The DataFrame is designed to be sliced and grouped by several axes:

### By biological condition
```python
df.groupby("cell_line")
```
Each FOV is assigned a `cell_line` label (e.g. `"WT"`, `"KO"`, `"drug_X"`). This is the primary experimental grouping.

### By stimulation treatment
```python
# Stimulated vs. unstimulated frames
df.groupby("stim")

# By stimulation dose (power x exposure)
df["dose"] = df["stim_power"] * df["stim_exposure"]
df.groupby("dose")

# By stimulation pattern (ramp experiments)
df.groupby("ramp_pattern_name")
```

### By field of view
```python
df.groupby("fov")       # by FOV index
df.groupby("fov_name")  # by FOV name
```

### By single cell
```python
# uid = fov + particle, uniquely identifies one tracked cell
df.groupby("uid")
```

### By optogenetic tool expression
```python
# Filter for cells expressing the optogenetic construct
expressing = df[df["optocheck_mean_intensity"] > threshold]
```

### By experiment phase (multi-phase)
```python
df.groupby("phase")     # by phase name
df.groupby("phase_id")  # by phase index
```

### Combined example
```python
# Mean ERK activity per condition over time, normalized to baseline
groups = df.groupby(["cell_line", "timestep"])["cnr_norm"].mean()
```

