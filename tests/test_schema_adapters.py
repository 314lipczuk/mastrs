"""Adapters project each raw family onto the canonical schema (hard cut)."""
import pytest

from notebooks.experiment.preprocessing import (
    CANONICAL_RAW_COLS,
    _LEGACY_COLS,
    adapt,
    adapt_bo,
    adapt_freepattern,
    adapt_standard,
    load_and_clean,
)


def test_standard_adapter_projects_to_canonical(make_raw):
    raw = make_raw("standard")
    out = adapt_standard(raw, "3-2-1minIntervals")

    assert list(out.columns) == CANONICAL_RAW_COLS
    for legacy in _LEGACY_COLS:
        assert legacy not in out.columns
    assert (out["stim_condition"] == raw["ramp_pattern_name"].values).all()
    assert (out["original_experiment_name"] == "3-2-1minIntervals").all()
    assert (out["frame"].values == raw["timestep"].values).all()
    assert (out["time_min"].values == raw["time"].values).all()
    assert (out["nuc_area"].values == raw["area"].values).all()


def test_bo_adapter_synthesizes_label_and_maps_area(make_raw):
    raw = make_raw("bo")
    out = adapt_bo(raw, "bo_v8", bo_tag="v8")

    assert list(out.columns) == CANONICAL_RAW_COLS
    assert "channels" not in out.columns
    expected = "bo_osc_v8_c" + raw["condition_idx"].astype(str)
    assert (out["stim_condition"].values == expected.values).all()
    assert (out["nuc_area"].values == raw["area_nuc"].values).all()
    assert (out["cell_line"] == "EGFR").all()


def test_freepattern_adapter_drops_int_uid(make_raw):
    raw = make_raw("freepattern")
    out = adapt_freepattern(raw, "freepattern_v1")

    assert list(out.columns) == CANONICAL_RAW_COLS
    # raw integer uid (a treatment id) must not survive
    assert "uid" not in out.columns
    assert (out["stim_condition"] == raw["treatment_name"].values).all()
    assert (out["nuc_area"].values == raw["area_nuc"].values).all()


def test_freepattern_uid_is_per_cell_not_per_treatment(make_raw):
    # 2 treatments x 2 particles = 4 distinct cells (fov is 1:1 with treatment)
    raw = make_raw("freepattern")
    df = load_and_clean(raw, experiment="freepattern", experiment_name="fp1",
                        baseline_cnr_max=None)
    n_cells = raw.groupby(["fov", "particle"]).ngroups
    assert df["uid"].nunique() == n_cells == 4
    # raw int uid had only 4 rows-worth of ids too here, but semantics differ:
    assert df["uid"].str.startswith("fp1__").all()


def test_adapt_dispatch_and_unknown_fallback(make_raw):
    raw_std = make_raw("standard")
    raw_bo = make_raw("bo")
    # explicit dispatch
    assert (adapt(raw_bo, "bo", "bo_v8", bo_tag="v8")["stim_condition"]
            .str.startswith("bo_osc_v8_c")).all()
    # unknown experiment name falls back to the standard adapter
    out = adapt(raw_std, "totally_unknown", "x")
    assert list(out.columns) == CANONICAL_RAW_COLS
    assert (out["stim_condition"] == raw_std["ramp_pattern_name"].values).all()


def test_projection_raises_on_missing_canonical_column(make_raw):
    raw = make_raw("standard").drop(columns=["x"])
    with pytest.raises(ValueError, match="missing canonical columns"):
        adapt_standard(raw, "exp")
