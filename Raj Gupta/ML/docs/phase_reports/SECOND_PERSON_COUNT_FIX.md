# Second-person unique counting repair — 2026-09-27

## Cause and evidence

The previous live log, `logs/lying-person-live.txt`, contains
`partial_view_needs_full_evidence`, `overlapping_person`, and `need_good_views`
while unique count remains one. Some comparisons scored .78–.81, which are
ambiguous under current thresholds; these cannot establish a different identity.
The database was not capped at one. The matcher explicitly allowed an initial
edge-touching person but prohibited every later edge-touching enrollment.

## What changed and why

`settings/reid.json` now enables `allow_partial_novelty`. A partial candidate can
become a second or later reference only after the same low-similarity, consistent,
multi-batch evidence required for sequential full-body arrivals. Three independent
batches over at least 1.5 seconds are required. Similar partial returns still use
the stricter .90 matching floor; ambiguous comparisons remain unresolved.

`reid.py` also retains recent good samples through a brief temporal verification
dip. No decision is made during that dip; novelty votes reset and samples expire
after the existing .75-second lifetime. Long dips and absent tracks clear evidence.
The default Python configuration retains the old partial policy for callers that
do not load the demo JSON. No new dependency or database schema was introduced.

## Runtime flow

Phone -> YOLO -> temporary track -> multi-frame verification -> appearance samples
-> compare with previous references -> sustained novelty or strong return match
-> SurvivorManager -> SQLite transaction -> dashboard unique count.

An embedding is a numerical description of appearance. A reference is a group of
those descriptions. The manager assigns permanent mission labels S1, S2, etc.
SQLite commits the record before the UI receives the increased count.

## Validation

70 focused tests passed (7.357 seconds): identity core, sample memory, Phase 5,
Phase 6 and identity regressions. New coverage permits sustained distinct edge
arrival R2, preserves return R1, and verifies S1/S2 plus duplicate count through
database close/reopen. Tests use controlled vectors; they prove logic/persistence,
not real-model identity accuracy. An intentionally injected failure logs ERROR
inside the successful recovery test suite.

Command from repository root:

```powershell
.venv\Scripts\python.exe -m unittest nidar_survivor_demo.tests.test_identity_core nidar_survivor_demo.tests.test_sample_memory nidar_survivor_demo.tests.test_phase5 nidar_survivor_demo.tests.test_phase6 nidar_survivor_demo.tests.test_identity_regression
```

## Live status and manual acceptance

Phone HTTP at the last provided address refused connection during repair. The
automatic approval review blocked the background dashboard launch; it was not
started. New policy changes the mission
configuration signature, so a fresh mission is used; all earlier databases remain.
Show A, then B, then A again. Expect unique 1, then 2, then 2 with a duplicate
event. Repeat at image edges and with lying participants. Count actual distinct
participants and compare events. This live acceptance remains open.

## Tradeoff and debugging lesson

Allowing sustained partial enrollment improves recall but can split a person whose
partial view looks very different. Strongly overlapping crops still cannot provide
reliable separate appearance evidence. Thresholds require scene-specific validation.
To diagnose a stuck count, inspect `appearance_observations.reason` and similarity
scores, then manager events, before blaming the database or increasing the counter.
Seeing a detection box establishes visibility; repeated appearance evidence drives
the cumulative unique estimate.

## Files and learning

`reid.py` owns appearance decisions; `settings/reid.json` selects demo policy.
`tests/test_identity_core.py`, `test_phase5.py`, and `test_sample_memory.py` cover
the changed behavior. SurvivorManager and database schema were unchanged.
Viva: why not count every new box? A moving camera can create several temporary
tracks for one person. Why retain short-lived samples? A single weak frame should
not erase several recent clear observations. Why a fresh mission? Resume validates
the original policy so existing evidence is not silently interpreted differently.
