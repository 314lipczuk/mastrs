"""Live optoRTK expression: per-cell mCitrine -> session-relative percentile.

The model's fifth channel, ``optortk_expr``, is not an intensity. It is a cell's
**rank** among the cells imaged alongside it. ``preprocessing.add_optortk_expression``
builds it offline as:

    proxy    = mCitrine intensity from the optocheck / reference acquisition
    per cell = that value (one per cell; median if several optochecks land)
    feature  = percentile rank of it within the imaging session

Ranking is what makes the feature portable: raw mCitrine differs ~3x between sessions
with the imaging gain, but a cell's position within its own session does not. That
is also what makes it reconstructible live, from the cells currently on the scope,
with no frozen dataset statistics — the population analogue of the per-cell CNR
baseline normalization.

Three things differ online, and getting them wrong is silent.

**The cohort must be complete before anyone is ranked.** The obvious streaming
design — rank each cell the moment its own window fills — is wrong: the first cell
to finish would be ranked against a cohort of one and score 1.0. This module
instead runs an explicit **cohort window**: every cell's measurement is accumulated over
frames ``timestep < cohort_frames`` and nobody is ranked at all until the window
closes, at which point the cohort is *sealed* and every cell in it is ranked
against the whole thing at once. That is the offline computation, performed at the
earliest moment the data for it exists. Before the seal there is no rank and
callers feed the neutral population mean, as they always did.

**Ranks never move afterwards.** The model was trained on a static per-cell
feature, so a percentile that drifts as new cells appear is a different input
distribution, not a more accurate one. Sealed means sealed: cells born later are
ranked against the sealed distribution and do not enter it. Offline those cells
have no baseline frames at all and are dropped; online they must still be steered,
so they get the best available answer instead.

**The value arrives on a handful of frames, not every frame.** The optocheck is
its own short acquisition run once or twice per experiment, so most frames carry
nothing for most cells. The cohort window must therefore be long enough to span
the first optocheck; a frame with no value is not an error, an empty cohort at
seal time is.

Drift does not enter: the ranks are frozen at the seal. Between two optochecks
5.5 h apart the per-cell rank correlates at 0.92 and only 9% of cells cross the
median, even though absolute intensity falls 6.4% — everything drifts together and
a within-session ordering absorbs it.

**TRACK FRAGMENTATION IS THE LIMIT ON THIS FEATURE, and it is the only thing that
is.** A reference acquisition can only measure the particle ids alive when it runs.
Over a 12 h run trackpy issues far more ids than there are cells — measured on the
2026-07-15 run: ~50 cells present per frame but 220 distinct ids, median track
lifetime 39 of 721 frames, and only 26 tracks spanning the run. So **28-53% of ids
ever receive a value** (median ~37%); the rest are continuations of cells that were
measured, under a new id, and they get the neutral population mean.

That is the correct answer for them and it matches training exactly — offline,
``add_optortk_expression`` gives an unmeasured cell ``fillna(0.5)`` and the runtime
feeds 0.499677, a difference of 0.001 sigma. What it is *not* is the distribution
the model was trained on: the training bundles were filtered to fully-tracked cells
(median track length equals the experiment length in all eleven experiments, and
they run 40-210 frames, not 721), so ~99% of training cells carried a real rank
against ~37% live.

Nothing here can fix that. There is no lineage column in the tracks, so the value
cannot be propagated across a break or from mother to daughter; and inheriting it
online would diverge from training, where the continuation is a separate uid that
also gets 0.5. The levers are upstream — track re-identification, or more frequent
reference acquisitions — and neither belongs in this module.

Worth being clear about what is NOT affected: with ``cnr_mode="raw"`` there is no
online CNR baseline to reconstruct, so fragmentation costs the CNR channel nothing.
Expression is the only per-cell static quantity that has to survive a track break.

Deterministic under replay throughout — a rank depends only on the multiset of
values in the sealed cohort and on the cell's own samples. No wall-clock, no dict
iteration order, no RNG.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass
class ExpressionCohort:
    """The session's mCitrine cohort, sealed at ``cohort_frames``, and its ranks.

    ``cohort_frames`` is when the cohort closes. It must be long enough to span
    the experiment's first optocheck, since before that no cell has a value at all.
    ``baseline_frames`` bounds how many optocheck samples one cell contributes
    before its rank is frozen; with a single optocheck per run it is reached on the
    first value.
    """

    baseline_frames: int = 1
    cohort_frames: int | None = None
    # FOVs the run is expected to cover, from the policy file. When known, the
    # cohort waits for ALL of them; without it a field that has not reported yet
    # is indistinguishable from a field that does not exist.
    expected_fovs: frozenset[int] | None = None

    sealed: bool = False
    # (fov, particle) -> measurements, while the cohort window is open.
    _building: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # Sealed cohort of per-cell values, sorted so ranking is a bisect.
    _sorted_c0: list[float] = field(default_factory=list)
    # (fov, particle) -> frozen rank in (0, 1]. Never revised once written.
    _frozen: dict[tuple[int, int], float] = field(default_factory=dict)
    # Post-seal arrivals still filling their own window.
    _late: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # Highest timestep seen per FOV, so the seal waits for EVERY reporting field
    # rather than firing on whichever one crosses the threshold first.
    _fov_clock: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.baseline_frames < 1:
            raise ValueError("baseline_frames must be >= 1")
        if self.cohort_frames is None:
            self.cohort_frames = self.baseline_frames
        if self.cohort_frames < 1:
            raise ValueError("cohort_frames must be >= 1")

    @property
    def n_cells(self) -> int:
        """Cells in the sealed cohort (0 until it seals)."""
        return len(self._sorted_c0)

    def rank_of(self, value: float) -> float:
        """Percentile rank of ``value`` in the sealed cohort, in (0, 1].

        Matches pandas ``rank(pct=True)``: the fraction of the cohort at or below
        this value, with ties taking the average rank. The largest member scores
        exactly 1.0 and nothing scores 0.
        """
        n = len(self._sorted_c0)
        if n == 0:
            return 1.0
        lo = bisect.bisect_left(self._sorted_c0, value)
        hi = bisect.bisect_right(self._sorted_c0, value)
        return ((lo + hi + 1) / 2.0) / n

    def seal(self) -> None:
        """Close the cohort window and rank everyone in it, all at once."""
        if self.sealed:
            return
        self._sorted_c0 = sorted(_median(v) for v in self._building.values() if v)
        self.sealed = True
        for key, samples in self._building.items():
            if samples:
                self._frozen[key] = self.rank_of(_median(samples))
        self._building.clear()

    def ready_to_seal(self, fov: int, timestep: int) -> bool:
        """Has every reporting FOV reached the cohort window's end?

        Sealing on the first payload past ``cohort_frames`` is wrong when the
        fields are skewed: whichever FOV runs ahead closes the cohort, and every
        field behind it is then ranked against a partial population. faro
        interleaves FOVs tightly within a cycle so the skew is normally under one
        frame, but "normally" is not a guarantee and the failure is silent — the
        late field's cells simply get ranks drawn from a fraction of the session.

        The backstop bounds the wait: if any FOV reaches twice the window, seal
        regardless, so one stalled or withdrawn field cannot hold the cohort open
        forever and leave every cell on the neutral value for the whole run.
        """
        self._fov_clock[int(fov)] = max(self._fov_clock.get(int(fov), -1), int(timestep))
        end = int(self.cohort_frames)
        if max(self._fov_clock.values()) >= 2 * end:
            return True
        if self.expected_fovs is not None:
            if not self.expected_fovs <= set(self._fov_clock):
                return False        # a declared field has not reported at all yet
        return min(self._fov_clock.values()) >= end

    def observe(self, fov: int, state, value: float | None, timestep: int) -> float | None:
        """Fold in this frame's measurement (if any); return the rank, or None.

        ``value`` is None on the many frames that are not optochecks — that is the
        normal case and must not be treated as missing data. None is also returned
        while the cohort is still open, since nobody can be ranked before the
        population they are ranked against exists; the caller feeds the neutral
        population mean for those frames.
        """
        key = (int(fov), int(state.particle))

        if not self.sealed:
            if self.ready_to_seal(fov, timestep):
                self.seal()
            else:
                if value is not None:
                    self._building.setdefault(key, []).append(float(value))
                    state.c0_samples = self._building[key]
                return None

        frozen = self._frozen.get(key)
        if frozen is not None:
            state.optortk_rank = frozen
            return frozen

        # A cell first seen after the seal, or one whose optocheck landed late:
        # rank it against the sealed cohort. It does not enter that cohort — the
        # model was trained on a static feature, and a percentile that moves as
        # cells appear is a different input distribution, not a better one.
        if value is None:
            return state.optortk_rank      # neutral until its first measurement
        samples = self._late.setdefault(key, [])
        samples.append(float(value))
        state.c0_samples = samples
        rank = self.rank_of(_median(samples))
        if len(samples) >= self.baseline_frames:
            self._frozen[key] = rank
            self._late.pop(key, None)
        state.optortk_rank = rank
        return rank

    def describe(self) -> dict:
        return {
            "baseline_frames": self.baseline_frames,
            "cohort_frames": self.cohort_frames,
            "sealed": self.sealed,
            "n_cells_in_cohort": self.n_cells,
            "n_frozen": len(self._frozen),
            "expected_fovs": (None if self.expected_fovs is None
                              else sorted(self.expected_fovs)),
            "fov_clock": dict(sorted(self._fov_clock.items())),
        }


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])
