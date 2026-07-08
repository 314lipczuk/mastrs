"""_derive_identity / clean / derive_features / validate_canonical."""
import numpy as np
import pytest

import notebooks.experiment.preprocessing as pp
from notebooks.experiment.preprocessing import (
    STIM_COLS,
    _derive_identity,
    adapt_standard,
    clean,
    load_and_clean,
    validate_canonical,
)


def test_full_pipeline_is_canonical_and_valid(make_raw):
    raw = make_raw("standard")
    df = load_and_clean(raw, experiment="standard", experiment_name="exp1",
                        baseline_cnr_max=None)
    validate_canonical(df, derived=True)  # raises if anything is off

    # derived stim channels present + non-NaN
    for c in STIM_COLS:
        assert df[c].notna().all()
    # u_t is the fluence channel; m_t is the stim indicator
    assert np.allclose(df["u_t"].values, df["fluence_mJ_cm2"].values)
    assert (df["m_t"] == df["stim"].astype(int)).all()
    # uid format: experiment__condition__fov__particle
    assert df["uid"].str.match(r"exp1__Sustained__\d+__\d+").all()


def test_derive_identity_cnr_formula(make_raw):
    raw = make_raw("standard")
    adapted = adapt_standard(raw, "exp1")
    df = _derive_identity(adapted)
    assert np.allclose(
        df["cnr"].values,
        (raw["mean_intensity_C1_ring"] / raw["mean_intensity_C1_nuc"]).values,
    )


def test_clean_drops_short_tracks(make_raw):
    # one full-length cell + one short cell (5 of 15 frames) -> short dropped
    long_raw = make_raw("standard", cells=[("Sustained", 0, 0)], n_frames=15)
    short_raw = make_raw("standard", cells=[("Sustained", 0, 1)], n_frames=5)
    raw = __import__("pandas").concat([long_raw, short_raw], ignore_index=True)
    df = load_and_clean(raw, experiment="standard", experiment_name="e",
                        baseline_cnr_max=None)
    assert df["uid"].nunique() == 1
    assert df["uid"].str.endswith("__0__0").all()


def test_clean_baseline_normalizes_to_one(make_raw):
    raw = make_raw("standard")
    df = load_and_clean(raw, experiment="standard", experiment_name="e",
                        baseline_cnr_max=None)
    baseline = df[df["frame"] < 10].groupby("uid")["cnr_median_norm"].median()
    assert np.allclose(baseline.values, 1.0, atol=1e-6)


def test_n5_pulse_count(make_raw):
    # pulses at frames 3,4,5 -> n_5 (trailing window of 5) == 3 at frame 5
    raw = make_raw("standard", cells=[("Sustained", 0, 0)], n_frames=15)
    df = load_and_clean(raw, experiment="standard", experiment_name="e",
                        baseline_cnr_max=None).sort_values("frame")
    assert df.loc[df["frame"] == 5, "n_5"].iloc[0] == 3


def test_calibration_selected_by_instrument(make_raw):
    # a 2x-scaled duplicate curve must scale fluence proportionally where stimmed
    raw = make_raw("standard", cells=[("Sustained", 0, 0)], n_frames=15)
    base = load_and_clean(raw, experiment="standard", experiment_name="e",
                          baseline_cnr_max=None, instrument="jungfrau")
    j = pp.CALIBRATIONS["jungfrau"]
    pp.CALIBRATIONS["_test2x"] = dict(
        pct=list(j["pct"]),
        uW=[2 * v for v in j["uW"]],
        mW_cm2=[2 * v for v in j["mW_cm2"]],
    )
    try:
        alt = load_and_clean(raw, experiment="standard", experiment_name="e",
                             baseline_cnr_max=None, instrument="_test2x")
    finally:
        del pp.CALIBRATIONS["_test2x"]

    nz = base["fluence_mJ_cm2"].to_numpy() > 0
    assert nz.any()  # pulses at frames 3,4,5 give nonzero fluence
    assert np.allclose(
        alt["fluence_mJ_cm2"].to_numpy()[nz],
        2 * base["fluence_mJ_cm2"].to_numpy()[nz],
    )


def test_unknown_instrument_raises(make_raw):
    raw = make_raw("standard")
    with pytest.raises(KeyError):
        load_and_clean(raw, experiment="standard", experiment_name="e",
                       baseline_cnr_max=None, instrument="nonexistent_scope")


def test_validate_canonical_rejects_legacy_and_nan(make_raw):
    raw = make_raw("standard")
    df = load_and_clean(raw, experiment="standard", experiment_name="e",
                        baseline_cnr_max=None)
    leaked = df.copy()
    leaked["ramp_pattern_name"] = leaked["stim_condition"]
    with pytest.raises(AssertionError, match="legacy column"):
        validate_canonical(leaked)

    nan_stim = df.copy()
    nan_stim.loc[nan_stim.index[0], "u_t"] = np.nan
    with pytest.raises(AssertionError, match="has NaN"):
        validate_canonical(nan_stim)
