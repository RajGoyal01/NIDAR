# Counting input and orientation repair — 2026-09-27

## Scope and actual diagnosis

User reports that one person is counted but a second is not. In
`logs/final-phone-retest.txt`, sampled pending observations were dominated by
`frame_edge_crop` and `low_confidence` (including 67 sampled edge rejections for
E1:T9 across the run). These are observations, not 67 people. The user confirmed
that the camera image is sideways with the head on the left. Logs do not establish
which physical person belongs to every track; no false-merge rate is claimed.

This is a repair within existing perception functionality, not acceptance of a
new major phase. The quality gate can reject evidence before SQLite ever receives
a new identity. Clearing the database would not fix that problem.

## What changed and why

1. Added explicit clockwise camera correction: 0/90/180/270 degrees. Head-left
   input uses 90. Rotation occurs before detection, tracking, Re-ID and display,
   so every component uses the same pixel coordinates. It does not crop or stretch
   the image. It corrects camera roll, not all human poses or viewpoints.
2. Separated sampling and comparison clocks. Previously the 0.5-second comparison
   cooldown also blocked collecting frames, despite the 0.15-second sample setting.
   Now fresh samples can arrive during that cooldown. Comparison remains paced;
   the quality, consistency, three-batch and minimum novelty-duration safeguards
   remain unchanged. Existing resolved tracks still avoid every-frame embeddings.
3. Pending video labels explain the reason: move back, improve light/straighten
   camera, hold steady, or uncertain identity. Previously one generic label hid
   very different problems. This is operator guidance, not proof of the cause of
   a low-confidence detection.

## Runtime flow and files

Phone JPEG -> `camera/phone_stream.py` decode/orient/latest-frame slot -> YOLO ->
tracker -> temporal verification -> `reid.py` quality/spaced samples/comparison ->
SQLite manager -> `main.py` -> `dashboard.py` annotated frame -> browser.

- `config.py`: validates `--rotation`; default 0 preserves existing behavior.
- `Start-PhoneDemo.ps1`: exposes `-Rotation` and passes it to Python.
- `camera/phone_stream.py`: shared orientation helper and capture integration.
- `reid.py`: independent evidence collection and comparison timers.
- `dashboard.py`, `main.py`: actual pending-reason labels.
- `tests/test_counting_input.py`: rotation pixels, capture integration, CLI
  validation, sampling during cooldown and explanatory labels.

No new dependencies, model downloads, training, schema migration, or threshold
relaxation. Existing OpenCV handles quarter-turn rotation; NumPy checks pixel
preservation. Existing model/database files are preserved. A fresh mission is used
for the corrected live run; prior evidence is not edited.

## Tests and measured results

- Full suite: **151 tests passed in 42.511 seconds**.
  Evidence: `logs/counting-input-tests.txt`.
- Actual GPU local-video integration: **164 consumed frames**, S1 and S2 created,
  clean shutdown. Evidence: `logs/counting-input-gpu.txt`. This is a two-person
  static-image diagnostic clip, not proof of live cross-angle identity accuracy.
- Rotation tests check all four quarter turns pixel-for-pixel and verify rotated
  dimensions before publishing frames. They do not certify detection at arbitrary
  view angles.
- A new test initially used the wrong fake-camera dimensions; corrected the test
  expectation to the actual 24x32 input / 32x24 output. Final suite is green.
- Corrected phone launcher attempt failed because the phone refused HTTP. The
  corrected dashboard was then started in reconnect mode with 90-degree rotation;
  live acceptance remains OPEN until actual phone frames and participant results
  are available. No live performance gain is claimed from this unavailable input.

## Exact Windows commands

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Rotation 90 -Dashboard -DashboardPort 8766
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
```

Use rotation 0 when the incoming image is already upright. Stop the old dashboard
before restarting on the same port. Keep the application loopback-only.

## Manual acceptance and debugging

1. Ensure the phone server is running and both devices share the 5 GHz network.
2. Confirm that the browser shows an upright person. Wrong direction: choose the
   correct quarter turn, not a CSS-only visual rotation.
3. Show A clearly inside the image boundary; wait for S1. Note time to acceptance.
4. A leaves, genuinely different B enters: expect S2 after sufficient evidence.
5. A returns: expect S1 without a third unique record. Repeat with front, back,
   side views and similar clothes, recording actual outcomes separately.
6. Test near-edge and blurred views: pending reason should explain the rejection,
   not silently force a unique identity. A low-quality frame cannot prove novelty.
7. Preserve mission reports on failures; record the displayed reason and temporary
   track ID. A pending count means unresolved identity, not an absent detection.

## Limitations / viva questions

**Does rotation give perfect all-angle detection?** No. Camera orientation and
person pose are different. Occlusion, lying people, poor light and similar clothes
need labelled project-specific evaluation and possibly fine-tuning.

**Why not count every bounding box as unique?** The same person can get another
temporary track after a pan; counting boxes would inflate totals.

**Why not lower the matching/novelty thresholds immediately?** Wrong changes can
merge different people or split one person. Calibrate with labelled positive and
negative re-entry examples first.

**What is the database's role?** Store accepted decisions atomically. It cannot
repair rejected or ambiguous visual evidence by itself.

**Is the user's second-person failure fully resolved?** Not yet established.
Sampling/orientation support and diagnostic feedback are verified in engineering
tests; corrected live A/B/re-entry acceptance is still required.
