# Rig checklist — run this before the experiment

For whoever (or whatever) is driving the microscope machine. Work top to bottom.
**Every gate is a stop, not a warning.** The last run passed none of these and
produced 40 hours of data that answered no question.

## Environment

You are on the Windows rig, in `C:\Users\Niesen\Documents\Przemek\mastrs`, with the
`(optoerk)` conda environment already active.

**Use plain `python`. Do NOT use `uv run`.** The dev machine drives everything
through uv; this machine does not. `uv run` here will try to build a second,
separate environment, and at best you wait, at worst you serve a twelve-hour
experiment from a different set of packages than the one you tested.

```powershell
python -c "import optoerk, torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `True`. If CUDA is False, stop — everything below is meaningless on CPU.

---

## Gate 1 — the test suite

```powershell
python -m pytest tests/ -q
```

**Pass = every test green.** No skips other than the two policy files that declare
themselves erroneous.

If something fails here, do not "just try the run". These tests encode failures that
have already cost a full experiment each — a cohort that seals empty, a schedule
indexed on the wrong clock, an open-loop arm that quietly commands different doses
to different cells.

## Gate 2 — the policy is internally consistent

```powershell
python -m pytest tests/test_policies.py -q
```

This checks the things that parse perfectly and are still wrong: the two arms of a
pattern tracking references that drifted apart by a typo, per-cell phase offsets
left on (which would make the open-loop arm fail structurally), an open-loop
schedule whose period does not match its waveform, arms searching different ladders.

Then look at the file yourself:

```powershell
python -c "from optoerk.serving.policy import load_policy_file, arm_map; p=load_policy_file('policies/policy_8fov_openloop.toml'); print(arm_map(p)); print(p.placeholders_resolved)"
```

Expect `{0:1, 1:2, 2:3, 3:4, 4:4, 5:3, 6:2, 7:1}` and `False` — it should still be
gated at this point.

## Gate 3 — soak (server side)

```powershell
python -m optoerk.serving.soak --policy-file policies/policy_8fov_openloop.toml ^
    --allow-placeholders --device cuda --n-fovs 8 --cycles 20 --cycle-seconds 60
```

**Pass = `rho < 0.7`.** 0.7–1.0 is marginal and will drift upward as cells divide;
`>= 1.0` is a hard fail — the backlog grows without bound.

Re-soak at the cell count you expect at hour 12, not hour 0:

```powershell
python -m optoerk.serving.soak --from-log <previous_run>.jsonl --start-frame 600 ^
    --device cuda --policy-file policies/policy_8fov_openloop.toml --allow-placeholders
```

## Gate 4 — the cadence dry run (THE IMPORTANT ONE)

**Soak cannot see this and it is what broke the last run.** `rho` measures the
inference server only. The last experiment spent 17 s per field on acquisition,
segmentation and tracking — work the server never touches — and ran 3.4× slower than
declared for 40 hours without anyone noticing.

Run the real acquisition, all 8 fields, for ~20 frames. Then measure the actual
round time:

```powershell
python -c "import polars as pl, glob, numpy as np; d=pl.concat([pl.read_parquet(f, columns=['fov','timestep','time_acquired']) for f in glob.glob('<dryrun>/tracks/*.parquet')]); s=d.filter(pl.col('fov')==d['fov'].min()).group_by('timestep').agg(pl.col('time_acquired').first()).sort('timestep'); t=s['time_acquired'].str.strptime(pl.Datetime,'%Y-%m-%d-%H:%M:%S').to_numpy(); print('median round:', np.median(np.diff(t).astype('timedelta64[s]').astype(float)), 's')"
```

**Pass = median round time <= 45 s** (25% headroom on the 60 s frame).

If it misses, the fallback was decided in advance so it is not decided at 2 a.m.:

1. **Default: do not run.** The 5.6 → 17.3 s per field regression is unexplained.
   Running before it is understood repeats the last failure exactly.
2. If it must go ahead: cut to **4 fields — one pattern, 2 arms, 2 repeats.** Never
   cut the repeats to 1; that leaves nothing to compare.

## Gate 5 — preflight (is the target reachable?)

```powershell
python -m marimo edit experiments/policy_preflight.py
```

Point it at `policies/policy_8fov_openloop.toml`. Paste the τ, ceiling and dry-run
tables into the policy's `PARAMETER PROVENANCE` block, over the `____` blanks.

The hold target must be reachable by most cells. If it is not, both arms saturate
and the experiment measures nothing — which is different from the pattern-zoo runs,
where an unreachable target was the point.

## Gate 6 — the open-loop schedules (ALREADY DESIGNED — just check)

**These are done.** Designed 2026-08-11 against this checkpoint and these
objectives, and already in the policy:

| arm | fields | schedule | mean dose |
|---|---|---|---|
| 2 (hold) | 1, 6 | constant 85 ms | 85.0 ms |
| 4 (oscillation) | 3, 4 | 45 ms for 48 of 50 frames, dark for 2 | 43.2 ms |

Do **not** redesign them unless Gate 5 moved the target — they are optimised against
`target_cnr = 1.034` and the 0.87→1.17 waveform, and are stale if either changes.
If Gate 5 does move a reference:

```powershell
python -m marimo edit experiments/open_loop_design.py
```

Run once per open-loop FOV (1 and 3; their repeats 6 and 4 share the answer), paste
`sequence_ms` into **both** fields of that arm, and update the provenance block.

Either way, check:

- the oscillation schedule has **exactly 50 entries** (the reference period)
- every value is on the ladder `[0, 20, 45, 85, 150]`
- the mean dose is within roughly 2× of the closed-loop arms' dry-run mean. A wild
  mismatch means the arms are not comparable on dose — record it, do not ignore it

## Gate 7 — flip the gate

Only when gates 1–6 have all passed and every `____` in the policy has been
replaced:

```toml
placeholders_resolved = true
```

Then confirm nothing was missed — this fails if any blank survives:

```powershell
python -m pytest tests/test_policies.py -q
```

---

## Running it

```powershell
python -m optoerk.serving.app --port 8080 --device cuda ^
    --policy-file policies/policy_8fov_openloop.toml ^
    --live-optortk-expr --optortk-cohort-frames 10 ^
    --no-dark-baseline --frame-interval-min 1.0 ^
    --stim-power 10 --predict-log run.jsonl
```

Then start faro against it.

## The first five minutes — watch stderr

The server now says three things out loud. Read them; they are cheap to act on now
and impossible to fix afterwards.

| what you see | means | do |
|---|---|---|
| `optoRTK cohort sealed: N cells, median M` | expression feature is live | good, continue |
| `*** optoRTK EXPRESSION DEGRADED ***` | too few cells carried a measurement | stop; the optocheck did not reach the server |
| `cadence OK: Xs per frame vs 60s` | the rig is keeping time | good, continue |
| `*** CADENCE SLIP ***` | rig slower than declared | **abort.** Every reference period is stretched by that factor and the model is running at an interval it was not trained on |

The cadence check reports once, after ~10 frames per field. If you see the slip
message, the run is not the experiment the policy describes. Aborting at minute 10
costs ten minutes; not aborting costs the whole session.

## Abort criteria during the run

- `cadence_degraded` appears on the predict records
- the expression cohort sealed degraded
- the server throws anything on a `/predict` — one 500 is one lost frame, but a
  repeated one means the run is not doing what it says

## Afterwards

Copy `run.jsonl` and the `tracks/` directory back, then open
`experiments/inference_cnrhold_tracks.py` on the dev machine and point it at the run
directory. Check, in this order:

1. **GPU telemetry health** panel — is the allocator leaking, is the sampled card
   even the model's card
2. **model error by arm** panel — if the open-loop arms show a much larger bias than
   the closed-loop ones, the open-loop schedule was exploiting model error and the
   arm contrast has to be discounted by roughly that much
3. only then the tracking comparison
