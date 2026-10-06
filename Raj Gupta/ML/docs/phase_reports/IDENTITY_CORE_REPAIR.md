# Identity lifecycle core repair — 2026-09-27

## Objective and evidence

The user confirmed that the pending person was the SAME physical person as S1.
Therefore unique=1 was numerically correct, but lack of S1 recovery/duplicate event
was a live identity failure. The original gallery stored only the initial three
views, and the crop gate rejected edge views before even considering an existing
identity. These are recognition limitations, not evidence of a broken SQLite count.
New-person failure and same-person recovery must be evaluated separately.

This repair remains within existing identity functionality; it does not advance
another major phase or certify field reliability. Historical missions are retained.

## Core change 1: separate enrollment from recognition

Enrollment means creating a new reference and eventually a new S record.
Recognition means recovering an already enrolled reference.

- A complete, clear, consistent batch is still required for new enrollment.
- An edge-touching crop may now be used for recognition ONLY when a gallery exists.
- Three recent consistent views, confidence/size/blur/overlap/visible-fraction
  checks remain required. A partial batch uses a stricter 0.90 match floor rather
  than 0.815, with the same ambiguity margin and occupancy/conflict checks.
- No partial batch can create a new reference or extend an appearance gallery.
- If the partial match is uncertain, its reason explicitly says so. Novelty votes
  are reset rather than counting a different-looking body fragment as a new person.

The 0.90 floor is a conservative PROJECT-DECISION, not a locally calibrated
all-condition accuracy guarantee. Truly clipped boxes failing visible-fraction,
overlapping people, low confidence and blur remain rejected.

## Core change 2: bounded appearance history with fixed anchors

The first three reference vectors remain immutable anchors (a stable reference
point). A successful full-view return can add a sufficiently different example
only when its batch ALSO matches those original anchors. Partial matches never
teach the gallery. This prevents a chain of progressively weaker matches from
silently replacing the original identity.

Maximum 12 vectors per identity, including the initial three. Older supplemental
views roll out when capacity is reached. Matching still needs consistent evidence
across query views; there is no blanket threshold reduction. This can represent
more accepted appearance variation but does not recover an entirely dissimilar
back/lying view automatically. Already-bound tracks still avoid per-frame Re-ID.

## Data flow and counter semantics

Phone -> YOLO -> temporary track -> temporal confirmation -> quality-gated Re-ID
-> reference association -> SurvivorManager transaction -> dashboard.

1. A confirmed track increments CURRENT occupancy, not automatically UNIQUE.
2. A new accepted reference survives manager confirmation: create S1, unique=1.
3. A leaves and a new track strongly matches R1: reuse S1, unique stays 1,
   duplicate events increases once for the accepted return episode.
4. B produces sustained distinct full-view evidence: create S2, unique=2.
5. SQLite saves records, gallery and event totals together before publishing them.

No UI-only increments, every-frame duplicates, destructive resets or rewriting of
old identity evidence. Changing gallery configuration changes the compatibility
signature; a fresh mission is used. Old databases remain readable as archives.

## Files / dependencies / important blocks

- `reid.py`: `quality_crop(..., allow_edge=True)` bypasses only the edge-margin
  gate for recognition; all other quality gates remain. Pending batches track
  partial/full flags alongside vectors/timestamps. Partial batches cannot enter
  the novelty/enrollment branch. Matched full batches can add anchored templates.
  Restore validates variable gallery length within configured bounds.
- `settings/reid.json`: `partial_match_threshold=0.9`,
  `max_reference_samples=12`; other thresholds unchanged.
- `dashboard.py`: explicit partial-view uncertainty label.
- `tests/test_identity_core.py`: recognition vs enrollment, stricter partial floor,
  no partial gallery learning, no initial partial enrollment, preserved quality
  gates, bounded anchors, SQLite return/new-person transitions and resume.

No new package, model, training or schema dependency. Existing Python/NumPy,
OpenCV, PyTorch/OSNet and SQLite are reused. Existing dependency pins remain.
Configuration, rather than hard-coded camera paths, controls runtime behavior.

## Validation results

- **165 automated tests passed in 45.159 seconds** (`logs/identity-core-tests.txt`).
- Real OSNet synthetic-view replay: 10 labelled COCO pairs, 2,760 frames,
  20 return attempts, **15 correct, 5 unresolved**, zero false-merge pairs and zero
  false splits in this dataset. Not 100% recall and not live accuracy.
- 10 database resumes: 15 persisted accepted identities, 15 duplicate events,
  zero return-count inflation. Evidence: `logs/identity-core-replay.json`.
- Synthetic identity vectors + actual SQLite: A -> unique1; edge-view A return
  -> unique1/duplicate1; continued visibility keeps duplicate1; full-view B ->
  unique2; resumed store preserves unique2/duplicate1.
- The phone refused connection at final retest. Updated local dashboard was
  restarted and is waiting to reconnect. Live acceptance is OPEN.

Tests prove the code paths/invariants and measured proxy behavior, not recognition
of the user's actual difficult view. No claim that every remaining miss is fixed.

## Windows launch and manual acceptance

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Rotation 90 -Dashboard -DashboardPort 8766
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
```

Rotation90 is for the user's confirmed head-left camera mounting; use0 for an
already-upright camera. A lying person does not alone imply camera roll.

Show A clearly for first enrollment; turn away; show A at a partial edge view;
verify S1 recovery without unique inflation. Then show genuinely different B in a
usable full view and verify S2. Test B return, two people together, similar clothes,
front/back/sitting/lying views separately. Save actual labels/timings and report
both misses and false associations. Do not reset away failed evidence.

## Debugging and viva

**Why can current=1 but unique=0?** Temporal evidence confirms a person, while
identity evidence has not yet justified a unique record.

**Why is duplicate=0 while the same person stays visible?** Duplicate events count
accepted returns, not frames. If a new track is pending, return matching has failed.

**Why allow partial recognition but not partial enrollment?** Existing templates
provide comparison evidence; a fragment alone cannot establish a genuinely new
identity safely.

**Can more templates cause drift?** Yes; fixed-anchor admission, full-view-only
learning, ambiguity checks and bounded storage constrain it but do not certify
real-world accuracy. Labelled field evaluation is still required.

**What remains unresolved?** Five proxy returns, the user's live changed-view
case, severe blur/occlusion and robust arbitrary-pose identity. Further improvement
needs labelled local footage/calibration and potentially a better domain model,
not simply counting all confirmed tracks as distinct people.
