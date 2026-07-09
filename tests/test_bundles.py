"""build_bundle concatenates cleaned canonical frames; uid is globally unique."""
import optoerk.data.preprocessing as pp
import optoerk.data.seq2seq_data as s2s


def test_stim_cols_single_source():
    assert pp.STIM_COLS is s2s.STIM_COLS


def _register(monkeypatch, tmp_path, make_raw, specs):
    """Write raw exp_data.parquet files and monkeypatch EXPERIMENTS."""
    reg = {}
    for name, cells in specs.items():
        d = tmp_path / name
        d.mkdir()
        make_raw("standard", cells=cells).to_parquet(d / "exp_data.parquet")
        reg[name] = dict(dir=str(d), adapter="standard", instrument="jungfrau", kwargs={})
    monkeypatch.setattr(pp, "EXPERIMENTS", reg)
    return reg


def test_build_bundle_concats_and_validates(make_raw, tmp_path, monkeypatch):
    _register(monkeypatch, tmp_path, make_raw, {
        "exp1": [("Sustained", 0, 0), ("Sustained", 0, 1)],
        "exp2": [("ramp1", 0, 0), ("ramp1", 0, 1)],
    })
    df = pp.build_bundle(["exp1", "exp2"], baseline_cnr_max=None)

    pp.validate_canonical(df, derived=True)
    assert set(df["original_experiment_name"].unique()) == {"exp1", "exp2"}
    assert df["uid"].nunique() == 4
    assert df.groupby("original_experiment_name")["uid"].nunique().to_dict() == {
        "exp1": 2, "exp2": 2,
    }


def test_uid_unique_across_experiments_with_same_label(make_raw, tmp_path, monkeypatch):
    # identical (stim_condition, fov, particle) in two experiments must NOT collide
    _register(monkeypatch, tmp_path, make_raw, {
        "a": [("Sustained", 0, 0)],
        "b": [("Sustained", 0, 0)],
    })
    df = pp.build_bundle(["a", "b"], baseline_cnr_max=None)
    assert df["uid"].nunique() == 2  # experiment prefix disambiguates
