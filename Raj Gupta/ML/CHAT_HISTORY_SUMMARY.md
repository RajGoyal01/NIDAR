# NIDAR AirMouse — Conversation History Summary

## Latest development handoff — 2026-09-27

2026-10-05 D-052: user requested USB phone-camera input while preserving every
existing demo. Added a separate Windows webcam discovery/validation utility,
dashboard-free USB launcher, and VS Code tasks/debug profile. Full USB mode feeds
the same perception/identity/database pipeline and starts a fresh mission; Wi-Fi
and offline launchers are unchanged. A USB cable must expose a Windows webcam
(native phone Webcam mode or a separately installed virtual-webcam solution).
179 tests pass. A Windows 1280x720 camera ran camera-only at about 29.6-29.8 FPS
and the Full CUDA pipeline with clean shutdown; it was not verified as the Moto,
so physical phone USB acceptance remains open. Read
docs/phase_reports/USB_PHONE_CAMERA_FALLBACK.md.

2026-10-01 D-051: user correctly reported that the offline presentation kept
showing one two-person photograph. Root cause was the selected static fixture,
not a manager capacity limit. Added `Start-OfflineDemo.ps1` and a real-YOLO
multi-image slideshow with measured counts 1,2,4,6,4,6. It is intentionally a
current-frame P-box detection demo; unrelated COCO photos are not presented as
persistent S identities. Unit and CUDA scene validation pass. Read
docs/phase_reports/OFFLINE_MULTI_PERSON_SHOWCASE.md.

2026-09-30 D-050: user clarified that the goal is person/possible-survivor
detection from diverse poses/views, not death prediction. Added reproducible
VisDrone conversion and leakage-aware fallen + COCO + VisDrone datasets. Trained
v5, balanced v6 and curriculum v7 candidates. V7 held-out recall: fallen .949,
lying .932, sitting 1.0, standing .929, small .227 and aerial .116; p95 38.97 ms.
General COCO recall/precision regressed to .469/.685 and 43 negative COCO images
had false positives. Promotion gate passed 7/10, so candidate was correctly rejected
and `models/yolo11n.pt` remains default. 176 tests + 36 subtests pass; pip check is
clean. VisDrone terms, final OAK-D data, locked field test and edge export remain OPEN.

D-049: unique count stuck at1 traced to first-only partial enrollment and repeated
sample resets. Demo permits sustained dissimilar partial newcomers; recent good
samples survive brief verification dips. 70 tests pass including S1/S2 persistence
and return. Phone refused connection; live accuracy remains open. See
docs/phase_reports/SECOND_PERSON_COUNT_FIX.md.

Latest: user says lying body must count as people2..6. Screenshot crop YOLO scores
up to .299 but tracker .40/verifier .35/Re-ID .50 blocked; D-048 aligned gates .20,
horizontal crop normalization, current card now tracks candidates rather than
waiting for unique identity. Phone unreachable; no live validation. See
docs/phase_reports/LYING_PERSON_COUNTING_FIX.md.

Latest D-047: user screenshot again current1/unique0; D-046 partial recovery
could not help empty gallery. Allow first edge-view bootstrap after temporal,
quality and multi-view checks (allow_initial_partial=true); subsequent partial
novelty still blocked.166 tests, actual OSNet/oracle-edge firstcount+return passed.
New dashboard log initial-partial-phone.txt; fresh mission, old data retained.
Live acceptance still OPEN. Read FIRST_PARTIAL_ENROLLMENT.md.

Latest D-046: user demanded core repair. Partial edge views now can recover existing
identities at stricter .90 but never enroll/update galleries. Full matches can add
bounded templates (12) against immutable initial anchors. 165 tests pass,10-pair
actual-model replay:15/20 correct returns,5 unresolved, no false merges/splits in
proxy;10 resumes preserve counts. New config requires fresh mission. Updated
dashboard running in reconnect mode, phone refused HTTP. Live user complaint not
fully validated/fixed; see docs/phase_reports/IDENTITY_CORE_REPAIR.md.

Latest user remains frustrated by zero counts and asks all-condition counting.
D-045 adds bounded recent-good-view memory through brief blur/confidence dips;
bad frames cannot create records. Candidate overlay distinguishes confirmation
from identity. Real-model proxy retains two identities/two returns; phone was
RECONNECTING and live failure remains OPEN. See BLUR_EVIDENCE_REPAIR.md. Do not
describe this scoped repair as perfect detection or full live acceptance.

Latest: second-person counting complaint diagnosed with edge/low-confidence
rejections; user confirms head-left sideways input. Added --rotation/-Rotation
(90 clockwise for that mounting), split sampling and comparison clocks, and
pending-reason labels. 151 tests and real-GPU static S1/S2 smoke passed. Phone
refused connection; corrected dashboard started in reconnect mode. Live identity
acceptance remains OPEN. Read docs/phase_reports/COUNTING_INPUT_REPAIR.md.

User approved explicit Reset Demo plus testing. Refresh must never reset counts.
Confirmed new mission preserves old database, clears perception identity state,
and is shared across tabs. Protected POST command runs on perception thread.
Previous reset summaries are exportable (last 20 this session). Read RESET_DEMO.md.

Latest: user confirmed different sequential arrivals were blocked. D-041's
co-visible policy removed; sustained novelty batches now permit A then B without
requiring A in view. Read SEQUENTIAL_IDENTITY_DATABASE_FIX.md. Database audit found
consistent storage, old records preserved. Actual-model replay (276 frames) passed
two enrollments/two returns and resume; live test still needed. Earlier co-visible
policy notes below are historical and must not override this change.

Latest user reported live duplicate failures/false unique counts. Fixed two unsafe
enrollment paths, added conservative managed novelty policy. 137 tests pass;
real-model replay returns A correctly but B remains unresolved under the policy.
Full live identity accuracy still OPEN; don't claim all errors solved. Phone server
refused corrected test. New test must use a new mission; old records preserved.
Read `docs/phase_reports/IDENTITY_FALSE_SPLIT_FIX.md` before further identity work.

Phase 8 update: user authorized performance fixes. Dashboard ceiling changed from
10 to configurable 30 FPS, browser pacing/unique decode fixed, redundant drawings
removed and CPU workers bounded. 133 tests passed (final 46.031s), pip check clean,
real GPU local-video run consumed 1288 frames and exited cleanly. Browser showed
S1/S2 and search worked. Phone stream unavailable: wireless smoothness acceptance
is not yet verified. Do not claim zero delay, live 30 FPS or full Phase 8 acceptance.
Full teaching/handoff: `docs/phase_reports/PHASE_08_PERFORMANCE.md`.

Phases 0-4 accepted through controlled live tests. User subsequently accepted
Phase 5's offline completion because the phone was unavailable, then authorized
Phase 6. Phase 6 implementation, camera-free acceptance and structured learning
handoff are complete: persistent SQLite S identities, duplicate suppression,
explicit mission resume and JSON reports. 112 tests passed; ten offline missions
resumed without accepted-return count inflation. Five replay returns remained
unresolved. Live identity accuracy is not certified.
User then authorized Phase 7: local read-only browser dashboard is implemented,
offline/browser acceptance and learning handoff complete. 130 tests pass, plus
actual-model replay and full GPU local-video mission/resume checks. Start with
Start-Dashboard.ps1 -Replay or -MissionDb for archive review. See
docs/phase_reports/PHASE_07_DASHBOARD.md. Phase 8 is not started.
See docs/phase_reports/PHASE_06_SURVIVOR_MANAGER.md for the complete handoff.
Older planning/history below is preserved and does not override this latest status.

## 1. Scope and limitation

This file summarises the technically relevant content available from the referenced conversation titled **“Analyse NIDAR Challenge”**. The available archive contained the recent detailed planning turns, not a complete verbatim export of every earlier message. Therefore this is a structured engineering handoff, not a claim of preserving unavailable wording.

## 2. Project direction established in the conversation

The team selected NIDAR AirMouse: an indoor, GPS-denied autonomous drone challenge focused on exploration, 2D mapping, survivor detection/localisation, and operator reporting.

The discussion converged on a three-part onboard role split:

- **OAK-D Lite** for RGB/depth sensing and possible edge inference;
- **Raspberry Pi 5** for ROS 2, SLAM, mission logic, sensor fusion, persistent survivor management, and communication;
- **Pixhawk-class flight controller** for stabilisation, motor control, flight modes, and safety.

The final system is intended to transform a camera detection into a physical map/grid location and suppress duplicate survivor counts across revisits.

## 3. Why the laptop demo was chosen

Before flight hardware integration, the conversation proposed a focused ML demonstration for a professor/review. An Android phone is moved through a room like a drone camera. A Windows laptop with RTX 3050 4 GB and 24 GB RAM runs the AI pipeline.

The demo is deliberately described as a **survivor-perception and duplicate-suppression proof of concept**, not a complete autonomous-drone demonstration.

## 4. Demo architecture agreed

```text
Android phone
  -> local wireless video stream
  -> low-latency OpenCV latest-frame capture
  -> lightweight YOLO person detection
  -> BoT-SORT temporary tracking
  -> temporal survivor verification
  -> selective appearance Re-ID
  -> persistent survivor manager
  -> duplicate suppression
  -> professional dashboard
```

Important distinctions:

- YOLO detects a person in a frame.
- BoT-SORT creates a temporary track while visible.
- Re-ID helps recognise a person after disappearance/re-entry.
- The Survivor Manager owns stable `S1`, `S2`, etc.
- Final drone identity will use depth and SLAM/map position in addition to appearance.

## 5. Performance decisions

The conversation did not promise zero lag. It selected these optimisation directions:

- phone video around 720p and 30 FPS;
- 5 GHz Wi-Fi/hotspot where available;
- Nano-class YOLO model;
- 512 or 640 inference size after testing;
- CUDA and FP16 on the RTX 3050 where supported;
- person class only;
- latest-frame-only processing;
- capture thread separate from inference/display;
- Re-ID only for new/reappearing tracks;
- no unnecessary 4K input;
- recording disabled initially;
- measure actual FPS and latency.

A visually smooth 20–30 FPS-class experience was considered realistic, but dependent on the phone stream, model, tracker, and configuration.

## 6. Development phases established

The conversation divided the demo into:

1. environment setup;
2. phone camera stream;
3. YOLO person detection;
4. BoT-SORT tracking;
5. temporal survivor verification;
6. persistent Re-ID;
7. duplicate suppression and Survivor Manager;
8. dashboard;
9. performance optimisation;
10. stress testing;
11. final professor demo;
12. explanation/migration to the real drone.

The strongest sequencing rule was to make phone streaming, detection, and tracking stable before integrating Re-ID.

## 7. Final professor-demo story

The planned highlight sequence is:

1. Person A is detected as `S1`; unique count is 1.
2. Person B enters as `S2`; unique count becomes 2.
3. The camera turns away and Person A disappears.
4. The camera returns to Person A under a new temporary track.
5. Re-ID maps that track back to `S1`; “duplicate prevented” increases while unique remains 2.

The dashboard should show module health, current people, unique survivors, duplicates prevented, FPS, annotated boxes, and an event log.

## 8. Dataset and model direction

The conversation chose not to begin with custom training. The agreed order is:

1. make the full pipeline work using a pretrained person detector;
2. measure actual misses and false detections;
3. collect a domain-specific dataset with drone-like views, survivor poses, dummies if relevant, low light, blur, occlusion, debris, and hard negatives;
4. fine-tune only if the baseline does not meet requirements;
5. export and revalidate on the selected edge hardware.

## 9. Context-package request

The user asked for the conversation knowledge to be moved into `C:\path\to\NIDAR\Raj Gupta\ML` without depending on ChatGPT Work. The resulting approach was to create structured Codex-readable files instead of a noisy raw transcript. `AGENTS.md` now requires future Codex sessions to read the entire package before modifying code.

## 10. Items not recoverable from the available archive

- Any exact earlier wording not included in the referenced archive.
- A complete official current-version rulebook PDF.
- Previously discussed exact vendor quotations, if any.
- Final model, threshold, SLAM stack, airframe, battery, or radio selections.

These gaps are intentionally marked as open rather than filled with invented details.

## 11. Domain dataset acquisition (2026-09-27)

Before Phase 9, the user authorized and supplied the download for Roboflow Fallen
Person v2. The 129,018,934-byte archive (SHA-256
`2055abd525e8903047682cb8c55dd69bfb87376b87aadcefa4a2c86a2d29e517`) was
downloaded; all 2,876 images and labels validate. Original classes are fallen,
lying, sitting and standing (3,297 boxes). A derived one-class `person_candidate`
view is training-ready. COCO YOLO11n baseline recall on the supplied 300-image test
split is .6144. Split leakage risk means this is not field accuracy. Fine-tuning is
not yet performed; Phase 9 has not started. See the dataset acquisition report.

## 12. Full VisDrone integration (2026-10-06)

The user asked whether VisDrone had been fully trained, then requested complete
integration. Inspection showed that the local conversion was complete but earlier
candidate training used limited VisDrone subsets. The project now has an exact-count
full mixture and a resumable prepare -> train -> held-out evaluate -> gate workflow.
All 6,471 train, 548 validation and 1,610 test-dev VisDrone images are included.
Vehicles are not mapped to people. The default demo checkpoint is not silently
replaced; final promotion still needs measured gates, deployment licence review,
project-owned OAK-D evidence and edge export validation.

## 13. Hybrid detector promotion (2026-10-07)

A validation-calibrated COCO-primary plus v7 posture/aerial-specialist fusion
passed the unchanged held-out engineering gate 10/10 and is integrated into phone,
USB and offline launchers. Full suite: 192 tests + 36 subtests. Offline exact counts
and motion variants: 1/2/4/6/4/6. This improves visible-person/body detection but
does not prove medical status, universal field accuracy or final-drone duplicate
suppression. See `docs/phase_reports/HYBRID_PERSON_DETECTOR_PROMOTION.md`.

## 14. Multi-person identity scaling repair (2026-10-07)

Fixed Re-ID starvation caused by stable track order with `max_batch=2`.
Least-recently-sampled fair scheduling keeps the GPU batch at two while progressing
all crowded-scene tracks. Integrated evidence creates/persists S1-S8 and correctly
recovers eight return tracks with unique=8 and duplicates=8. Configured gallery
capacity is 64, not two. See
`docs/phase_reports/MULTI_PERSON_IDENTITY_SCALING.md`.
