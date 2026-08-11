"""The policy files themselves, as artifacts.

A policy is twelve hours of microscope time expressed as a config. The errors that
matter are not syntax — pydantic's ``extra="forbid"`` catches those — but design
errors that parse perfectly: two arms whose references drifted apart by a typo, a
gate flipped before the measurements behind it were taken, an open-loop schedule
whose period does not match the waveform it is meant to track. None of those raise,
and all of them are only discoverable after the run.
"""
import json
from pathlib import Path

import pytest

from optoerk.serving.policy import arm_map, load_policy_file

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"
POLICIES = sorted(POLICY_DIR.glob("*.toml"))
OPEN_LOOP_POLICY = POLICY_DIR / "policy_8fov_openloop.toml"


def _skip_if_self_declared_erroneous(path: Path) -> None:
    """One committed policy is knowingly broken — a shell command was pasted into
    the TOML body — and is kept only so the runs it mislabels can still be analysed,
    per its own header. Keyed on that declaration rather than on the filename, so the
    exemption vanishes the moment the header does (or the file is deleted), instead
    of quietly covering for the next one.
    """
    header = path.read_text()[:2000]
    if "erroneously committed" in header:
        pytest.skip(f"{path.name} declares itself erroneously committed, pending deletion")


def _period_frames(objective: dict) -> float:
    """Frames per cycle of a step-train reference."""
    dt = float(objective.get("frame_interval_min", 1.0))
    total_min = sum(
        float(objective.get(k, 0.0))
        for k in ("t_low_min", "t_rise_min", "t_high_min", "t_fall_min")
    )
    return total_min / dt


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.name)
def test_every_policy_loads_and_labels_its_arms(path):
    _skip_if_self_declared_erroneous(path)
    pf = load_policy_file(path)
    arms = arm_map(pf)
    assert arms, f"{path.name} resolved to no FOVs"
    assert set(arms) == {int(f) for f in pf.fov}


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.name)
def test_a_resolved_policy_has_no_unfilled_blanks(path):
    """The house convention writes `____` where a measurement has to be pasted in,
    and `placeholders_resolved` is the gate that stops the file serving until it is.
    Flipping the gate with blanks still in the provenance block means the run went
    ahead on numbers nobody checked — which is exactly what the gate exists to stop,
    and the one failure mode it cannot catch by itself."""
    _skip_if_self_declared_erroneous(path)
    pf = load_policy_file(path)
    if not pf.placeholders_resolved:
        return
    text = path.read_text()
    assert "____" not in text, (
        f"{path.name} declares placeholders_resolved = true but still contains "
        f"`____` blanks — the gate was flipped before the measurements landed"
    )


# ---------------------------------------------------------------------------
# closed loop vs open loop (policy_8fov_openloop.toml)
# ---------------------------------------------------------------------------


def test_open_loop_run_is_four_arms_of_two_fields():
    pf = load_policy_file(OPEN_LOOP_POLICY)
    arms = arm_map(pf)
    counts = {a: sum(1 for v in arms.values() if v == a) for a in set(arms.values())}
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}
    # mirrored layout: every arm has the same mean field index, so arm is not
    # confounded with position on the plate
    means = {
        a: sum(f for f, v in arms.items() if v == a) / 2 for a in sorted(set(arms.values()))
    }
    assert set(means.values()) == {3.5}, means


def test_the_two_arms_of_a_pattern_track_a_byte_identical_reference():
    """`objective` is replaced wholesale per [fov.N], so the four copies of the
    oscillation are four chances to typo one of them. A reference that differs
    between the closed-loop and open-loop arms of a pattern makes the comparison
    meaningless while looking completely normal in every plot."""
    pf = load_policy_file(OPEN_LOOP_POLICY)
    arms = arm_map(pf)
    by_objective = {}
    for fov, spec in pf.fov.items():
        key = json.dumps(spec.objective, sort_keys=True)
        by_objective.setdefault(key, set()).add(arms[int(fov)])

    assert len(by_objective) == 2, "expected exactly two patterns"
    # each distinct reference must span exactly one closed-loop and one open-loop arm
    assert sorted(sorted(v) for v in by_objective.values()) == [[1, 2], [3, 4]]


def test_no_per_cell_phase_offsets_anywhere_in_the_open_loop_run():
    """An open-loop arm sends ONE command to every cell, so it cannot give two cells
    different references. Leaving n_phase_groups > 1 would guarantee it fails for a
    structural reason that has nothing to do with feedback."""
    pf = load_policy_file(OPEN_LOOP_POLICY)
    for fov, spec in pf.fov.items():
        obj = spec.objective or {}
        if "n_phase_groups" in obj:
            assert obj["n_phase_groups"] == 1, f"fov {fov} offsets cells by phase"


def test_each_open_loop_schedule_repeats_on_the_waveforms_own_period():
    """A schedule whose period differs from the reference's slides out of phase over
    twelve cycles and measures nothing. For a constant reference any period is fine."""
    pf = load_policy_file(OPEN_LOOP_POLICY)
    seen = 0
    for fov, spec in pf.fov.items():
        ctrl = spec.controller or {}
        if ctrl.get("type") != "open_loop":
            continue
        seen += 1
        seq = ctrl["sequence_ms"]
        obj = spec.objective or {}
        if obj.get("type") == "hold":
            continue
        assert len(seq) == _period_frames(obj), (
            f"fov {fov}: schedule is {len(seq)} frames but the reference cycles "
            f"every {_period_frames(obj)}"
        )
    assert seen == 4, "expected four open-loop fields"


def test_both_control_modes_choose_from_the_same_dose_ladder():
    """The closed-loop arms can only pick ladder rungs. An open-loop schedule off the
    ladder would mean the arms are not searching the same dose set, and any dose
    difference between them stops being attributable to control."""
    pf = load_policy_file(OPEN_LOOP_POLICY)
    ladder = set(pf.default.levels_ms)
    assert ladder, "the run must declare an explicit ladder"
    for fov, spec in pf.fov.items():
        ctrl = spec.controller or {}
        if ctrl.get("type") != "open_loop":
            continue
        off = sorted({float(v) for v in ctrl["sequence_ms"]} - {float(x) for x in ladder})
        assert not off, f"fov {fov} commands {off}, which is not on the ladder {sorted(ladder)}"
