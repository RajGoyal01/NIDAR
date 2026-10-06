# NIDAR AirMouse — Phase-by-Phase ML Demo Plan

## Guiding rule

Move to the next phase only after the previous phase passes its tests. Keep a known-good tag or commit at every milestone.

## Phase 0 — Environment setup

**Goal:** create a reproducible Windows development environment.

Build and verify:

- Python 3.10 or 3.11 virtual environment;
- CUDA-compatible PyTorch installation;
- Ultralytics, graphical OpenCV and NumPy; tracker/Re-ID-specific dependencies
  deferred until their relevant phases (2026-09-27 user clarification);
- deterministic configuration and dependency lock;
- RTX 3050 detection.

Pass criteria:

```text
torch.cuda.is_available() == True
GPU name is the expected NVIDIA device
OpenCV can open a local camera/test video
All required imports succeed
```

2026-09-27 status: PASS using Python 3.11.15, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, NumPy 2.2.6, opencv-python 4.13.0.92 and Ultralytics
8.4.163. Verified actual CUDA matrix calculation, synthetic-video decode, GUI
window API and pip consistency. Run `nidar_survivor_demo/check_environment.py`
with the project `.venv` interpreter to repeat. User subsequently authorised Phase 1.

## Phase 1 — Moving phone stream

**2026-09-27 final status: functional acceptance COMPLETE.** Actual 720p capture
ran 120 seconds with no disconnect; user confirmed movement responsiveness and
automatic phone-server restart recovery. 25 automated tests passed. Working
launcher applies/verifies the phone profile. See `docs/phase_reports/PHASE_01_HANDOFF.md`.
Earlier status notes below are historical. Numerical total latency and sustained
minimum FPS are not certified. User authorized Phase 2 on 2026-09-27.

**Goal:** use the Android phone as a moving drone-camera simulator.

```text
Android phone -> local 5 GHz Wi-Fi/hotspot OR USB webcam mode -> laptop -> OpenCV
```

Required behaviour:

- webcam index and configurable stream URL support;
- 720p around 30 FPS input target;
- capture in a separate thread;
- keep only the latest valid frame;
- connection/FPS display;
- reconnect after temporary failure;
- clean shutdown and resource release.

Tests: rotate, pan, walk, move near/far, briefly disconnect network, reconnect. Pass when the feed stays responsive and does not accumulate seconds of delay.

2026-09-27: Implemented and 17 automated tests passed. Initial actual-phone
connection passed, but FPS dips remain under investigation. Preview rendering
and scheduling optimised; phone server unavailable for the post-fix comparison.
Acceptance remains pending. See `docs/phase_reports/PHASE_01_PHONE_STREAM.md`.

Follow-up: phone retested at 720p; latest-snapshot mode visibly reduced delay
(user-confirmed), but ~8.4 FPS remains below smoothness target. Keep Phase 1 active.

Further experiment: single-decoder mode worsened delay and was rolled back.
Direct MJPEG mode displayed ~20.5 FPS over 40 seconds including a recovered
interruption; visible-delay validation pending. Snapshot fallback retained.

Latest user feedback rejected direct-mode latency. Wireless-only constraint
rules out USB troubleshooting. Trial 960x540/quality35/motion-off profile tested;
5 GHz comparison pending user phone-side band change. Do not advance to Phase 2.

5 GHz comparison completed: user confirms smoothness and reduced visible delay
at 960x540/quality35/motion-off, direct MJPEG. ~26 FPS average including startup
over ~22.3 seconds, but rolling dips occurred. Retain profile; 720p and longer
acceptance testing remain pending before Phase 2.

2026-10-05 additive fallback: `Start-USBCameraDemo.ps1` lets a phone that Windows
recognises as a webcam provide the same live input without IP Webcam. Camera
discovery, one-frame validation, guaranteed release and VS Code tasks are added.
Wi-Fi remains available and unchanged. Automated acceptance passed; physical
phone USB acceptance is OPEN until a compatible phone is attached and tested.
See `docs/phase_reports/USB_PHONE_CAMERA_FALLBACK.md`.

## Phase 2 — YOLO person detection

**2026-09-27 final status: functional acceptance COMPLETE.** CUDA FP16 model and 512/640 COCO sample
benchmark verified. 36 automated tests pass. User confirmed one-person boxes and
accepted the fixed moving-camera preview, then explicitly confirmed 2-4-person
detection in the final live test. No Phase 2 acceptance item remains open.
Phase 3 was subsequently authorized and completed; see its section below.
See `docs/phase_reports/PHASE_02_PERSON_DETECTION.md`.

**Goal:** detect people in the moving stream.

- lightweight pretrained YOLO Nano-class model;
- person class only;
- configurable confidence and IoU thresholds;
- CUDA and FP16 where supported;
- 512 or 640 inference size after benchmarking;
- bounding box, confidence, inference latency, and FPS overlay.

Pass when one to four people are detected smoothly under moderate phone movement with acceptable false detections.

## Phase 3 — BoT-SORT tracking

**2026-09-27 final status: functional acceptance COMPLETE.** 50 automated tests,
synthetic motion/crossing/occlusion/pan regression, and user-accepted live tracking.
Re-ID off; temporary IDs only. Numerical live ID-switch count is unknown, not zero;
synthetic labelled tests have zero switches. Mean sampled live display 14.67 FPS.
See `docs/phase_reports/PHASE_03_BOTSORT_TRACKING.md`. Phase 4 subsequently authorized below.

**Goal:** keep a stable temporary ID while a person remains visible.

```text
YOLO boxes -> BoT-SORT -> track_id
```

Test people crossing, partial occlusion, camera panning, and near/far movement. Record ID switches. Pass when continuously visible people usually retain their IDs during the controlled test route.

## Phase 4 — Temporal survivor verification

**2026-09-27 functional acceptance COMPLETE.** User accepted the live demo:
"Haan, Phase 4 demo working". Implemented `--verify` / launcher `-Verify`.
3 good observations in 5 processed fresh frames, confidence >= .35 and clipped
visible area >= .0001 of the frame; current observation must also qualify.
Missing/weak observations reduce evidence; five missing updates expire history.
Outages, tracker epochs and >1s gaps reset evidence. No persistent unique counter
exists yet: the Phase 4 acceptance gate is zero confirmation for isolated flashes,
timely confirmation of sustained tracks, and verified-visible count only.
This clarifies the original unique-count wording without implementing Phase 6.
See `docs/phase_reports/PHASE_04_TEMPORAL_VERIFICATION.md`.

**Goal:** prevent one-frame noise from becoming a survivor.

Initial configurable rule:

```text
candidate seen in at least 3 of last 5 frames
AND confidence/quality above threshold
-> confirmed track
```

Pass when brief false boxes and partial flashes do not increase the unique count, while real people are confirmed without excessive delay.

## Phase 5 — Persistent Re-ID

2026-09-27: implementation and offline engineering validation COMPLETE under the
user's revised request to finish without a camera. Live/field reliability OPEN. OSNet x0.25
MSMT17 encoder, normalized three-view matching, quality gates, bounded session
gallery and uncertainty/conflict handling. 87 tests passed. One live match recorded,
but three references appeared with only one participant available; numeric logs
alone cannot establish ground-truth splits. Similar-clothing two-person live testing
is unavailable, not passed. Offline real-model replay: 15/20 returns resolved,
4 ambiguous + 1 blurred-crop unresolved, zero false-merge pairs/false splits in
1,000 synthetic-view frames. All 15 enrolled references recovered. The COCO proxy
calibration/evaluation and limitations are documented
separately in `docs/phase_reports/PHASE_05_APPEARANCE_REID.md`.
R references are session-local appearance hypotheses. Persistent S records and
unique-survivor/duplicate counters remain Phase 6.

**Goal:** recover a prior identity after the person disappears and returns.

- pretrained Re-ID encoder;
- several good embeddings per survivor;
- normalised embeddings and cosine similarity;
- quality filtering;
- Re-ID only for new/reappearing tracks;
- uncertain state for ambiguous matches.

Controlled test:

```text
Track 4 -> Survivor S1
camera turns away
Track 19 appears later
Re-ID match -> Survivor S1, not new S2
```

Pass when re-entry works reliably in a documented test set. Also measure false merges between different people wearing similar clothes.

## Phase 6 — Survivor Manager and duplicate suppression

2026-09-27: COMPLETE for the authorized camera-free implementation/acceptance
scope, following user acceptance of Phase 5. SQLite stores mission-scoped S IDs,
appearance gallery, bounded observations and transactional decision audit.
112 tests pass. Ten real-model/oracle-track synthetic-view replay missions:
1,000 frames, ten restarts, 15 accepted returns, zero return-count inflations,
five unresolved returns safely pending. Full GPU static-image integration passes.
This does not certify live re-entry accuracy. See the Phase 6 learning report.

**Goal:** combine tracking and Re-ID into persistent identity management.

Maintain:

- `current_persons`;
- `unique_survivors`;
- `duplicates_prevented`;
- first/last seen time;
- current and historic track IDs;
- embedding gallery and observation history.

Pass when repeat appearances do not inflate the unique count and every decision is traceable in the event log.

## Phase 7 — Professional dashboard

2026-09-27: implementation, offline/browser engineering acceptance and handoff
COMPLETE. Local read-only browser interface provides fresh annotated frames,
estimated counts, health/metrics, identity search, event filters and JSON export.
130 tests pass; real OSNet replay, full GPU local-video processing/resume and
desktop/mobile browser checks pass. Live phone and independent user comprehension
testing are not claimed. Original preview publishing capped at 10 FPS; Phase 8
has now removed that limit (see Phase 8 report; live wireless acceptance unverified).
See docs/phase_reports/PHASE_07_DASHBOARD.md for scope, evidence and commands.

**Goal:** make the system understandable without reading terminal output.

Recommended layout:

```text
+----------------------------------------------------------+
| NIDAR AIRMOUSE — SURVIVOR PERCEPTION PROTOTYPE           |
+----------------------------------+-----------------------+
| LIVE CAMERA                      | AI STATUS             |
| [S1 96%]                         | Camera     ONLINE     |
|                                  | YOLO       ACTIVE     |
|               [S2 94%]           | Tracker    ACTIVE     |
|                                  | Re-ID      ACTIVE     |
|                                  | Current          2    |
|                                  | Unique           2    |
|                                  | Duplicates       1    |
|                                  | FPS            27.4   |
+----------------------------------+-----------------------+
| EVENT LOG: S1 confirmed; S1 re-identified; duplicate...  |
+----------------------------------------------------------+
```

Pass when a new observer can correctly explain the system state and identity events from the screen alone.

## Phase 8 — Performance optimisation

2026-09-27: Authorized and implemented. 133 regression tests passed; actual GPU
local-video and browser checks passed. Phone server unavailable during final work;
wireless movement acceptance remains unverified. See `docs/phase_reports/PHASE_08_PERFORMANCE.md`.

Optimise only after measurement:

- CUDA inference;
- FP16 where safe;
- Nano-class model;
- 512/640 input comparison;
- latest-frame capture;
- threaded I/O;
- selective Re-ID;
- bounded logs and no unnecessary frame saving;
- recording OFF during the primary live demo unless performance permits.

Target a visually smooth 20–30 FPS-class experience. Report measured latency and FPS; do not promise zero lag.

## Phase 9 — Stress testing

Run and record:

1. one person;
2. two people;
3. three to four people;
4. people crossing;
5. slow and fast camera pan;
6. near/far movement;
7. temporary occlusion;
8. person disappears and returns;
9. genuinely new person enters;
10. similar clothing;
11. low light;
12. lying/crouching person or dummy if available;
13. stream interruption;
14. GPU unavailable fallback behaviour.

For every test, record configuration, expected result, actual result, metrics, PASS/FAIL, and video/log reference.

## Phase 10 — Professor/demo mode

2026-10-01 presentation helper: `Start-OfflineDemo.ps1` provides a phone-free,
dashboard-free real-YOLO slideshow with six different local scenes and measured
current-frame counts `1,2,4,6,4,6`. This specifically proves that detection is
not capped at two people. It does not assign S identities across unrelated COCO
photos and therefore does not replace the controlled re-entry/duplicate story
below. This helper is complete; full final-drone or field acceptance is not.

Freeze dependencies and configuration. Keep a backup recorded input and a known-good model.

Presentation sequence:

1. Point at Person A: `S1`, unique = 1.
2. Person B enters: `S2`, unique = 2.
3. Turn away so Person A disappears.
4. Return to Person A: `S1 RE-IDENTIFIED`; duplicate prevented; unique remains 2.
5. Move the phone dynamically to show live detection and tracking.

Explain honestly:

> The current demo validates moving-camera survivor perception and appearance-based duplicate suppression. In the final drone, depth and SLAM-derived physical position will strengthen identity and place each confirmed survivor on the 2D map.

## Phase 11 — Transition to the real drone

Replace components incrementally:

| Prototype | Final system |
|---|---|
| Android RGB stream | OAK-D RGB + stereo depth + IMU |
| Laptop CUDA | OAK accelerator and/or Raspberry Pi runtime |
| Screen-space position | camera-to-map 3D/2D position |
| Appearance-only history | appearance + depth + SLAM spatial fusion |
| Handheld camera motion | autonomous Pixhawk-controlled flight |
| Demo dashboard | GCS live map, feed, telemetry, survivor markers |

Use recorded sensor bags/videos before free flight, then tethered/guarded tests, then controlled autonomous flight. Never test new perception and new flight-control behaviour simultaneously without a rollback path.

## Dataset precondition added before Phase 9 (2026-09-27)

Roboflow Fallen Person v2 has been downloaded, validated and converted into a
one-class `person_candidate` training view. The unchanged YOLO11n baseline recall is
.6144 on its supplied test split. Fine-tuning and promotion remain separate
controlled work; supplied split metrics cannot serve as field certification because
related source groups overlap.

## Complete VisDrone follow-up before Phase 9 (2026-10-06)

All 8,629 converted VisDrone DET images are wired into the reproducible unified
training/evaluation path; earlier limited experiments are not presented as full
training. Dataset preparation and schema acceptance are complete. Candidate training
and gate results must be recorded before stress-testing a custom model. Phase 9 is
not automatically accepted by this data integration, and the known-good COCO
checkpoint remains the presentation fallback.

## Hybrid detector follow-up (2026-10-07)

The laptop demo now uses a gate-passing hybrid detector through the phone, USB and
offline launchers. The general primary and posture/aerial specialist produce the
same `person_candidate` interface. Offline scene counts 1/2/4/6/4/6 and all
simulated motion variants pass. `-BaselineDetector` selects the COCO-only
comparison. This improves Phase 9 readiness but does not complete Phase 9: live
blur, occlusion, lighting, multi-person re-entry and OAK-D stress tests still need
recorded expected/actual results.

## Multi-person S-ID follow-up (2026-10-07)

The persistent identity pipeline is not limited to S1/S2. Fair Re-ID scheduling
keeps `max_batch=2` as a GPU workload bound while rotating across every qualified
unresolved track. Integrated automated evidence now covers S1-S8 creation,
persistence and return matching. The configured mission capacity is 64 appearance
references. Live Phase 9 crowd accuracy remains a separate recorded stress test.
