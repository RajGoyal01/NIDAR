# Multi-person persistent identity scaling repair

Date: 2026-10-07  
Status: **implemented and automated acceptance passed**

## Objective

Persistent labels must continue as `S1`, `S2`, `S3`, `S4`, `S5`, `S6`, and so
on. Detecting many people in a frame and assigning persistent identities are
different stages, but neither stage should have an accidental two-person limit.

## Root cause

YOLO was not limited to two detections (`max_det=100`). The survivor database also
already generated IDs with `S{number of records + 1}` and the appearance gallery
capacity was 64.

The actual scaling defect was in selective Re-ID. `max_batch=2` intentionally
limits OSNet to two crops per GPU call, protecting latency and VRAM. Tracks were
processed in the same order each frame, so the first tracks could repeatedly take
those two slots while later tracks waited indefinitely. A performance batch limit
had accidentally behaved like an identity-count limit.

## Repair

`nidar_survivor_demo/reid.py` now:

1. evaluates all unresolved confirmed tracks for crop quality;
2. orders usable tracks by the time each was last sampled;
3. sends the two least-recently-sampled tracks to OSNet;
4. gives remaining tracks priority on following frames.

This is **least-recently-sampled fair scheduling**. The safe GPU batch remains two,
but identity processing continues across frames until every visible qualified track
has received evidence.

## Runtime flow

```text
YOLO boxes (up to 100)
  -> BoT-SORT temporary tracks
  -> temporal confirmation
  -> all unresolved confirmed tracks enter fair queue
  -> OSNet processes 2 crops per frame
  -> R1, R2, R3 ... appearance references
  -> repeated manager evidence
  -> S1, S2, S3, S4, S5, S6 ... persistent mission records
  -> atomic SQLite save and dashboard/overlay update
```

`T` IDs are temporary tracker IDs, `R` IDs are internal appearance references, and
`S` IDs are persistent records within one mission. A detector box alone does not
immediately create an S record.

## Files involved

- `nidar_survivor_demo/reid.py`: fair least-recently-sampled crop scheduling.
- `nidar_survivor_demo/survivors.py`: already allocates sequential S IDs with no
  two-person branch or constant.
- `nidar_survivor_demo/settings/reid.json`: `max_batch=2` remains a GPU-work limit;
  `max_gallery=64` is the reviewed mission identity capacity.
- `nidar_survivor_demo/tests/test_multi_identity_scaling.py`: eight-person full
  Re-ID, manager, duplicate-return and SQLite regression.

## Automated evidence

The new integration test uses eight different appearance vectors and eight
simultaneously confirmed temporary tracks while keeping `max_batch=2`.

- eight distinct records created with exact IDs `S1` through `S8`;
- eight SQLite survivor rows saved;
- all people leave and return with different temporary tracker IDs;
- all returns map to the original `S1` through `S8` records;
- unique count remains 8 and duplicate-prevention count becomes 8;
- SQLite integrity check returns `ok`.

Focused identity suite: 54 tests passed.  
Complete project suite after the code repair: **193 tests + 36 subtests passed**.  
Dependency check: no broken requirements.

## Why batch size was not simply changed to eight

Increasing batch size consumes more GPU memory and creates a latency spike exactly
when a crowded rescue scene appears. Fair scheduling keeps live workload bounded
and processes the full crowd over successive fresh frames. This is safer for the
RTX 3050 prototype and closer to an edge-computer design.

## Manual test

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Manage
```

Keep each consenting participant visible, separated and sufficiently clear until
temporal and appearance evidence completes. Expected labels progress through
`S1`, `S2`, `S3`, `S4`, `S5`, `S6`, etc. On re-entry, the same people should reuse
their earlier S IDs and the unique count must not increase.

For USB:

```powershell
.\Start-USBCameraDemo.ps1 -CameraIndex 0 -Mode Full
```

## Important interpretation

- **Current people**: temporally tracked people visible now.
- **Unique people**: accepted appearance-based records during this mission.
- **Duplicate prevented**: a returning track matched an existing S record.

The demo now scales beyond six identities in automated tests. Appearance-only
matching can still be uncertain for blur, heavy overlap or very similar clothing;
such evidence remains pending instead of inventing an ID. Final-drone spatial
duplicate suppression still requires depth and SLAM/map position.

## Viva questions

**Was YOLO limited to two people?** No. YOLO permits up to 100 detections. The
starvation was in the downstream Re-ID crop scheduler.

**Why keep max_batch at two?** It bounds per-frame OSNet GPU work. Fair rotation
turns it into a throughput limit, not a total identity limit.

**What is the current identity capacity?** The reviewed gallery configuration
allows 64 mission references. It is not limited to S1/S2.

**Does this prove perfect field identity?** No. It proves allocation, persistence
and recovery beyond six without a coded two-person ceiling. Field accuracy still
needs labelled multi-person camera tests.
