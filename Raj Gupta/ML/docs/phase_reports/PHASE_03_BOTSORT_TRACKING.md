# Phase 3 — BoT-SORT tracking: completed learning handoff

Date: 2026-09-27. Status: **functional acceptance COMPLETE for the controlled laptop demo**.
Phase 4 has not started. This is not a field-rescue or autonomous-flight certification.

## 1. What we built

Phase 2 found people separately in each image. Phase 3 adds a short-term memory:
BoT-SORT associates those boxes over consecutive processed frames and displays a
temporary label such as `E1:T5`. A person walking across the view should usually
keep the same label while continuously visible.

- Existing camera-only and detection-only modes remain available.
- Person-only YOLO11n, CUDA FP16 and fixed 640 inference remain unchanged.
- BoT-SORT uses motion prediction, overlap and detection confidence.
- Sparse optical flow compensates for camera movement.
- Low-confidence detections can recover existing tracks without starting new ones.
- Empty detections still advance the tracker, so missing people disappear from view.
- Short-term lost tracks can be recovered; expired tracks receive new IDs.
- Camera outage, resolution change or long frame gap resets association safely.
- Frame, bounding boxes and IDs remain paired; no stale-video backlog.
- Optional per-frame metadata log stores IDs/coordinates, not camera images.

**Not added:** Re-ID, survivor confirmation, permanent identities, unique-survivor
counting, database, dashboard, training or flight-control commands.

## 2. Full data flow

```text
Moto camera -> IP Webcam -> 5 GHz network
 -> latest JPEG -> latest decoded frame
 -> YOLO person candidates
 -> BoT-SORT motion/overlap matching
 -> temporary tracks on the SAME frame
 -> freshness check -> labelled preview + JSON metrics/events
```

The camera keeps receiving while AI works. If capture is faster, intermediate
frames are skipped, not queued. The tracker advances once per processed frame,
not once per captured frame. It must never process the same frame twice.

## 3. Concepts in simple language

**Detection versus tracking:** detection says “a person is here”; tracking says
“this box probably continues the box seen a moment ago.” It is a best estimate.

**Kalman filter:** predicts where a moving box is likely to be next, then corrects
that prediction using the next observed box. Think of following a ball briefly
behind a player's arm—not remembering that ball forever.

**Association:** pairing old tracks with new boxes. The `lap` solver chooses a
low-cost one-to-one pairing so two existing tracks do not claim the same box.

**IoU:** intersection-over-union, the overlapping area of two boxes divided by
their combined area. Motion prediction and camera compensation make comparison
more useful when either the person or camera moves.

**Global motion compensation (GMC):** follows image features to estimate camera
movement. The tracker adjusts predicted boxes for that shared motion. Low texture,
blur or abrupt movement can still make it unreliable.

**Occlusion:** another object hides a person. The tracker retains a lost track for
a limited number of updates; a short compatible reappearance can reuse the ID.
It does not draw an invisible lost person as though detection were current.

**Temporary ID:** `E1:T5` means tracker epoch 1, local track 5. After a reset,
`E2:T5` is a different temporary track. Neither label is persistent `S1` identity.
IDs may skip numbers because unconfirmed internal tracks also allocate numbers.
IDs and episode counts are NOT counts of unique people.

**Tracker activation is not survivor verification:** BoT-SORT has its own track
activation rules. These do not implement the later multi-frame survivor policy.

## 4. Files and responsibilities

| File | Responsibility |
|---|---|
| `nidar_survivor_demo/tracking.py` | Validated tracker configuration, BoT-SORT adapter, resets, temporary records, labels |
| `nidar_survivor_demo/settings/botsort.json` | Editable tracking parameters outside implementation code |
| `nidar_survivor_demo/config.py` | `--track`, configuration and metadata-log options; preserves Phase 2 defaults |
| `nidar_survivor_demo/main.py` | Runs detection/tracking in order, handles camera health, logging and cleanup |
| `nidar_survivor_demo/tests/test_phase3.py` | 14 tracker/configuration/application tests using real BoT-SORT |
| `nidar_survivor_demo/benchmark_tracking.py` | Ground-truth-labelled synthetic association regression |
| `Start-PhoneDemo.ps1` | `-Track` launch switch with verified phone profile |
| `requirements.txt`, `requirements-lock.txt` | Added pinned `lap==0.5.12` |

The project stays on Ultralytics 8.4.163. Its installed BoT-SORT API accepts
`BOTSORT(args)` and `update(Boxes, frame)`. Keeping this adapter separate avoids
rerunning YOLO or changing Phase 2 detector behavior. Internal APIs are version
sensitive: rerun tests before any library upgrade.

`TrackerConfig.load()` rejects unknown JSON keys, wrong types and invalid ranges.
It intentionally offers no Re-ID switch in Phase 3. `PersonTracker` explicitly
checks the optional `lap` dependency before importing tracking, avoiding a surprise
automatic installation during a demo. One tracker instance per application process
is the supported design because the library uses a shared local-ID allocator.

`update()` converts plain detection records into Ultralytics `Boxes`. It runs even
with zero detections. Track data is copied back into project-owned dataclasses.
Our own bookkeeping is restricted to current/lost tracks, not an unlimited history.

`clear()` drops motion/history and advances the epoch only once for an outage.
`annotate_tracks()` copies the frame before drawing, so capture data is not modified.

## 5. Selected settings and why

| Setting | Value | Meaning |
|---|---|---|
| Detector confidence in tracking mode | 0.10 | Preserve weaker candidates for recovery |
| High association threshold | 0.35 | First-stage matching |
| Low association threshold | 0.10 | Lower-score recovery cutoff |
| New-track threshold | 0.40 | Stronger evidence to start a new track |
| Track buffer | 30 processed frames | Short lost-track retention; NOT 30 seconds |
| Match threshold | 0.80 | Maximum accepted association cost; larger is more lenient |
| Confidence fusion | true | Include confidence when computing match costs |
| GMC | sparseOptFlow, downscale 2 | Half-width/height motion image; full input retained for YOLO/display |
| Maximum frame gap | 1.0 second | Reset rather than trust old motion across a long interruption |
| Re-ID | disabled | Required phase separation; no extra encoder/weights |

At 15 processed FPS, 30 lost updates are roughly two seconds; this is not a precise
wall-clock identity lifetime. A long gap in receiving/processing frames has a
separate reset rule. Phase 2 `--detect` still defaults to confidence 0.35.

We tried GMC downscale 4 for speed. It failed the synthetic pan regression
(27 ID switches, 64.69% matched coverage), so it was rejected. Default 2 is the
same motion setting used in the user-accepted live run. Do not silently re-enable
the faster trial; changing it requires a fresh regression comparison.

## 6. Run and test commands

From `C:\path\to\NIDAR\Raj Gupta\ML`, with the phone serving on a trusted 5 GHz network:

```powershell
# Phase 3; replace PHONE_IP with the current address
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Track

# Known-good Phase 2 detection-only fallback
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Detect

# Camera-only fallback
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080

# Optional per-frame tracking metadata; use a NEW output filename each run
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --transport mjpeg --track --track-log logs/my_tracking_test.jsonl

# Repeat automated verification
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m nidar_survivor_demo.benchmark_tracking
.\.venv\Scripts\python.exe -m pip check
```

`--track` implies detection. `--tracker-config` accepts a validated JSON file.
Do not set detector confidence higher than the tracker's low threshold: the code
rejects this because it would remove candidates required by low-score recovery.
Q/Esc closes the preview; `-Seconds 60` limits launcher runtime. Model warmup occurs
before the live timer. No repeated physical test is required for this handoff.

Metadata output uses exclusive creation: an existing filename is not overwritten.
The optional file can grow during long runs; omit it for the ordinary demo and
use a bounded `--seconds` duration for diagnostics. It contains no images, names,
biometric embeddings or credentials, but trajectories should still be handled
privately. No server, authentication system or database is needed in this phase.

## 7. Verification evidence

Automated suite: **50 tests passed in 28.288 seconds** at final closeout; dependency
consistency passed with no broken requirements. Final-run evidence is in
`logs/phase3_tests_final.txt`. Includes all Phase 1/2 regression tests plus tracker
continuity, low-score recovery, missed detections, expiry, reset idempotence,
connection/shape/time resets, duplicate-input rejection, annotation immutability,
Re-ID disabled, invalid configuration and application/metadata-log integration.

Synthetic tracker regression: 80 frames per scenario, four labelled trajectories,
seed 42, known detector boxes. Steady motion, crossing paths, four-frame occlusion
and camera pan all passed with 100% matched visible observations and zero switches
using the selected downscale-2 setting. Crossing paths have vertical separation;
this does not prove recovery from fully coincident, indistinguishable people.
These tests isolate association, not YOLO detection quality or real-world accuracy.
Final metrics: `logs/phase3_tracking_regression.json`.

| Synthetic scenario | Matched / expected | ID switches | Median / p95 tracker ms |
|---|---|---|---|
| Steady motion | 320 / 320 | 0 | 9.42 / 11.32 |
| Crossing paths | 320 / 320 | 0 | 10.81 / 11.91 |
| Short occlusion | 316 / 316 | 0 | 10.89 / 12.19 |
| Camera pan | 320 / 320 | 0 | 15.38 / 17.87 |

Live acceptance: 720p phone stream, 367 processed frames over ~25.4 seconds,
24 sampled states all ONLINE, one connection, no tracker resets, clean shutdown.
Mean sampled displayed FPS: 14.67; range 11.78-16.77. Highest rolling tracker p95:
34.09 ms. This is slower than detection-only; a sustained 20-30 FPS claim would
be false. Performance optimisation remains a later phase, with this accepted
functional baseline preserved.

User was asked about walking/crossing, partial occlusion, camera pan/near-far and
ID switches and replied: **"Yes, All working fine"**, then requested closeout
without repeated physical testing. This records qualitative user acceptance.
The user did NOT give a numerical live switch count, so live `id_switches` remains
`null` (unknown), not zero. New/lost track events are not automatically ID switches:
ground-truth identity is needed to tell the difference.

## 8. Acceptance and limits

- [x] Actual BoT-SORT integrated after person detection; Re-ID off.
- [x] Temporary IDs displayed on matching fresh frames.
- [x] Low-score recovery, empty-frame ageing and short-occlusion behavior tested.
- [x] Camera-loss/gap/shape reset semantics tested; IDs separated by epoch.
- [x] Synthetic ID-switch/coverage and timing metrics recorded.
- [x] Real-phone tracking tested and user accepted walking/crossing/occlusion/pan behavior qualitatively.
- [x] Dependency pin, fallback modes, tests, report and project context updated.

No Phase 3 functional acceptance item remains open. Known limits are not hidden:
IDs can change after long absence or difficult overlap; GMC can fail with blur or
low texture; no real-world zero-switch guarantee; no numerical glass-to-glass
latency; no lifetime identity; no controlled real-video accuracy benchmark yet.
The existing COCO detection baseline also has misses (see Phase 2 report).

## 9. Debugging guide

- **No IDs but occasional boxes/candidates:** candidates may be below new-track
  confidence or not yet activated. Check `detection_candidates` versus visible IDs.
- **IDs skip numbers:** tentative internal tracks allocate numbers. Not an error
  and not evidence that the same number of people passed through the room.
- **ID changes after disconnect:** expected epoch reset; old motion cannot safely
  establish identity after a camera outage.
- **GMC warnings:** texture/flow estimate unavailable; the installed library can
  fall back to identity motion. Use better lighting/moderate pan; don't call this
  appearance Re-ID. Check actual track stability before changing parameters.
- **Low FPS:** inspect separate inference and tracking p95, not capture FPS alone.
- **Missing lap:** install the pinned requirements into project `.venv`, not global Python.
- **Log file exists:** choose a new filename; do not delete prior evidence to rerun.
- **Configuration error:** validate JSON keys, thresholds and the confidence ordering.

## 10. Viva questions and answers

**Is T5 the fifth unique survivor?** No. It is a temporary track number.

**Why BoT-SORT?** It offers motion-based association and camera-motion compensation
for the moving-phone prototype, and was the agreed project tracker.

**Why no Re-ID yet?** We must stabilize detection and short-term tracking first.
Appearance matching adds cost and a different class of identity errors.

**Why keep weaker boxes?** A partly hidden person may have a low detection score;
motion and overlap can still connect that box to an existing track.

**Is a tracker-confirmed person a verified survivor?** No. Survivor verification
and persistent identity are separate later layers.

**Can you report zero ID switches?** Only in the labelled synthetic regression.
The physical test was accepted qualitatively without a numerical switch annotation.

**What did we learn?** Detection, association, temporary identity, optical flow,
bounded memory, reset safety, configuration validation and evidence-based tuning.

**What comes next?** Phase 4 temporal survivor verification, only after explicit
authorization. Nothing from Phase 4 was implemented during Phase 3.

## 11. Sources

Ultralytics tracking documentation: https://docs.ultralytics.com/modes/track/
Installed source inspected: `ultralytics/trackers/bot_sort.py`, `byte_tracker.py`,
`utils/matching.py`, `utils/gmc.py`, plus `cfg/trackers/botsort.yaml` in version
8.4.163. Tracker implementation remains part of the existing Ultralytics licensing
scope documented in Phase 2. No new model weights were downloaded.
