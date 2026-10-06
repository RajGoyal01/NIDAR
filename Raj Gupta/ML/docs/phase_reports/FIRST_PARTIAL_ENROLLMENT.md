# First partial-person enrollment — 2026-09-27

## Root cause and scope

The supplied screenshot had current=1, unique=0, pending=1. Runtime events show
`frame_edge_crop`, with occasional low-confidence/collecting states. D-046 allowed
partial recognition only after a reference existed, so it could not help this
empty-gallery enrollment case. Earlier partial-recognition repair did not address
the user's first-count failure; this report records the distinction explicitly.

## Policy change (PROJECT-DECISION)

`allow_initial_partial=true` allows an edge-touching first person to enroll when
the gallery is empty. Temporal confirmation, sufficient crop size, visible-fraction,
confidence, blur and overlap gates, three recent mutually consistent embeddings,
and the manager's three confirmation hits remain required. No single box alone
increments unique count. The first reference still represents an appearance-based
estimate, not a verified human identity or medical assessment.

After any reference exists, a partial batch remains recognition-only. It cannot
create another identity from a changed body fragment. A same-frame proposal guard
also prevents multiple partial bootstrap enrollments against one initially empty
gallery. A genuinely different person with only partial evidence can therefore
still remain pending; this repair is NOT general partial-person counting.

Disable the new setting to restore conservative full-view-only bootstrap. Initial
partial anchors can be less representative than full-body anchors; cross-pose
recognition still needs independent evaluation. No match/blur/confidence threshold
was lowered. Do not claim zero delay or arbitrary-condition detection.

## Files and runtime flow

- `reid.py`: boolean configuration validation, empty-gallery crop-policy exception,
  partial-vs-novel branch separation, assignment-time same-frame guard.
- `settings/reid.json`: explicit `allow_initial_partial: true`.
- `tests/test_identity_core.py`: multi-frame initial edge enrollment, opt-out,
  actual SQLite first partial -> return -> new full person -> resume transitions.
- `tests/test_sample_memory.py`: conservative clipping-reset test explicitly uses
  the opt-out policy; overlap protection remains unchanged.

Phone -> YOLO box -> temporary track -> temporal evidence -> acceptable partial
crop -> three consistent appearance samples -> R1 -> manager confirmation ->
SQLite S1 commit -> dashboard unique=1. Strong accepted re-entry reuses S1 and
increments duplicate events once; it does not create S2.

No new dependency, model download or schema migration. Old databases are retained.
Config signature changes, so the deployed test uses a fresh mission and does not
rewrite old signatures or erase historical records.

## Test evidence

- **166 tests passed in 42.208 seconds**, `logs/initial-partial-tests.txt`.
- Actual OSNet, real COCO crop (image18150/annotation521385), synthetic edge
  placement and oracle boxes: first accepted count at frame9; unique1 and
  duplicate1 after return. Simulated frame spacing was0.1s, giving0.8s from first
  observation to first count. This is NOT measured live latency or a YOLO benchmark.
- SQLite regression starts with an edge crop, confirms S1, recovers its partial
  return, enrolls distinct full-view B as S2, and preserves unique2/duplicate1
  after reopening the database.
- Corrected dashboard restarted; live acceptance remains unverified while phone
  connection is unavailable. Runtime log: `logs/initial-partial-phone.txt`.

## Windows commands / manual test

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Rotation 90 -Dashboard -DashboardPort 8766
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
```

Use rotation0 for upright camera input. In a fresh mission show one clear person
touching an image edge. Expect first enrollment after enough consistent views,
not indefinite `frame_edge_crop`. Show a same-person return, then a genuinely
different person with usable full-view evidence. Record counts and IDs separately.
Blur/confidence/overlap rejection may still delay enrollment; pending reasons must
remain visible. Preserve failures rather than resetting to conceal them.

## Learning / viva

**Why did previous changes fail here?** Recognition needs an existing reference;
an empty-gallery rule must independently permit initial enrollment.

**Why only bootstrap one partial identity?** Without usable comparison evidence,
different partial appearances may belong to one person. Automatically counting
every fragment would reintroduce duplicate inflation.

**Is every remaining miss fixed?** No. This fixes a specific enrollment gate.
Poor imagery, different poses and similar appearances still require calibrated
local evaluation; broader identity reliability remains OPEN.
