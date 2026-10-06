# NIDAR AirMouse — Project Context

## 1. Mission in one sentence

Build an autonomous indoor drone system that can operate without GPS, explore an unknown maze-like environment, detect and localise survivors, build a usable 2D map, mark survivor locations, and return/exit safely while giving a single operator a clear mission view.

## 2. Why this project exists

The AirMouse scenario represents an earthquake- or collapse-damaged building. Human rescuers may not be able to enter safely because of debris, unstable structures, confined passages, and low visibility. The drone acts as a fast first scout: it explores, maps, detects possible survivors, and reports where they are.

## 3. Two systems must be kept distinct

### A. Current ML demo

The immediate deliverable is a controlled perception proof of concept:

```text
Android phone camera
        -> local Wi-Fi / hotspot stream OR Windows USB-webcam device
        -> Windows laptop
        -> lightweight YOLO person detector
        -> BoT-SORT temporary tracks
        -> temporal verification
        -> selective appearance Re-ID
        -> persistent survivor IDs
        -> duplicate suppression
        -> professional live dashboard
```

It validates moving-camera detection, tracking, re-entry recognition, counting, and user presentation. It does not validate autonomous flight, SLAM, obstacle avoidance, grid localisation, or complete competition compliance.

### B. Final NIDAR drone

The planned final architecture is:

```text
OAK-D Lite RGB + stereo depth + IMU
        -> perception / depth observations
Raspberry Pi 5 companion computer
        -> ROS 2 integration
        -> visual-inertial odometry / SLAM
        -> map, planning, survivor fusion and mission logic
Pixhawk-class flight controller
        -> attitude stabilisation, motor control and flight safety
        -> receives bounded high-level setpoints
Ground Control Station
        -> live feed, 2D map, survivor markers, telemetry and mission report
```

Appearance, depth, and map position are fused so the same person is not counted repeatedly when revisited.

## 4. Core technical problem

Person detection alone is not enough. A detector answers “is there a person in this frame?” It does not reliably answer “is this the same survivor observed earlier in another room view?”

The identity layers are therefore separated:

- **Detection** finds a person in one image.
- **Tracking** keeps a temporary ID while the person remains visible.
- **Temporal verification** rejects brief false detections.
- **Re-ID** compares appearance when a track disappears and later returns.
- **Spatial fusion** uses depth and SLAM coordinates in the final drone.
- **Survivor Manager** owns the persistent identity (`S1`, `S2`, ...), evidence, timestamps, map position, and duplicate decision.

## 5. Current prototype goals

- Responsive 720p-class phone stream at about 30 FPS input.
- Visually smooth end-to-end behaviour, expected in the 20–30 FPS class after tuning; this is a target, not a guarantee.
- Person-only detection using a lightweight pretrained YOLO model.
- Stable BoT-SORT track IDs under moderate camera movement.
- A detection must persist across multiple frames before confirmation.
- Reappearing people should recover the same persistent survivor ID in controlled tests.
- Dashboard must show current people, unique survivors, duplicates prevented, FPS, module status, and event history.
- Demo must remain useful if Re-ID is disabled: phone stream, detection, and tracking should still work independently.

## 6. Final-system goals

- Autonomous GPS-denied indoor flight.
- Real-time or continuously updated 2D map.
- Obstacle-aware exploration of corridors, junctions, turns, and rooms.
- Survivor detection and map/grid localisation.
- Persistent duplicate suppression across revisits.
- Live camera and mission status at the Ground Control Station.
- Safe mission abort and fault handling.
- A hardware plan whose team budget target is below INR 2,00,000, excluding already-owned development equipment unless the official rules say otherwise.

## 7. Data and fine-tuning strategy

Start with a pretrained person detector. This reduces initial risk and lets the team validate the complete pipeline before spending time collecting and labelling data.

Fine-tuning comes later, only after baseline measurements show a real problem. The custom dataset should represent the deployment domain:

- overhead and oblique drone-like camera angles;
- people standing, sitting, lying, crouching, partially hidden, or covered;
- real humans and competition dummies if both can appear;
- low light, shadows, blur, debris, similar clothing, and partial occlusion;
- empty-room and human-shaped hard negatives;
- multiple distances and camera heights.

Split data by scene or recording session, not by adjacent frames, to avoid train/test leakage. Track dataset licences and provenance. Never scrape or train on data whose licence or consent is unclear.

## 8. Definition of project success

The project is successful only when the perception demo is reproducible and the final architecture has a validated integration path. A polished UI without reliable identity logic is not success; a high-accuracy model without safe, low-latency integration is also not success.

## 9. Current status

**2026-10-05 additive USB camera fallback complete (D-052):** the existing
IP Webcam and offline launchers remain unchanged. `Start-USBCameraDemo.ps1`
discovers/validates Windows camera indices and launches the same native preview
and perception pipeline from a phone exposed to Windows as a USB webcam. Full,
detection, tracking and camera-only modes are available; Full is the default and
uses a fresh local mission database. 179 tests pass. A Windows camera ran
1280x720 camera-only at about 29.6-29.8 FPS and the Full CUDA pipeline; both shut
down cleanly. The exercised device was not verified as the user's Moto, so a real
phone USB run remains the manual hardware acceptance check. See
`docs/phase_reports/USB_PHONE_CAMERA_FALLBACK.md`.

**2026-10-01 phone-free multi-person showcase complete (D-051):** the previous
`coco-static.avi` demonstration looped one two-person photograph, which made the
presentation look capped at S1/S2 even though the manager has no two-record cap.
`Start-OfflineDemo.ps1` now opens a dashboard-free continuous real-YOLO replay
in the original camera-test layout over six different local COCO images. Actual RTX inference validated counts
`1, 2, 4, 6, 4, 6` at the demo's .25 presentation threshold. P labels are
current-frame boxes, not persistent S identities; unrelated stock photos are not
used to manufacture survivor identity claims. See
`docs/phase_reports/OFFLINE_MULTI_PERSON_SHOWCASE.md`.

**2026-09-30 detector-domain training complete, promotion rejected (D-050):**
death/liveness classification is explicitly out of scope. Reproducible one-class
`person_candidate` datasets combine group-separated fallen poses, COCO people and
negatives, and official-split VisDrone aerial people. Three mixed candidates were
tested. Best pose candidate v7 found .949 of held-out fallen boxes and .932 of
lying boxes at 38.97 ms p95, but general COCO recall/precision regressed to
.469/.685 and false positives appeared in 43 COCO negative images. The automated
engineering gate passed 7/10 and rejected promotion. The live default therefore
remains `models/yolo11n.pt`; v7 is a research checkpoint, not final-drone evidence.
VisDrone terms require review, and project-owned OAK-D indoor footage, new locked
field data, edge export and depth/SLAM validation remain OPEN.

Latest D-049: second-person edge enrollment now allowed after sustained novelty;
brief verification dips retain recent good samples. 70 focused tests passed,
including S2 database persistence and S1 return. Live phone unavailable; see
docs/phase_reports/SECOND_PERSON_COUNT_FIX.md. This supersedes D-047's first-only
partial enrollment restriction for the configured demo.

Latest D-048 lying-body repair: screenshot inference confidence max .299 was below
tracker .40, verifier .35, Re-ID .50. Settings aligned to .20, horizontal crops
normalized, current occupancy now counts tracked candidates separately from unique
accepted identities. Phone unreachable at closeout; live lying-pose accuracy and
false positives remain unverified. See LYING_PERSON_COUNTING_FIX.md.

Latest D-047 fixes empty-gallery edge enrollment: allow the first consistent
partial reference after existing evidence gates; later partial views remain
recognition-only.166 tests passed, actual-model/oracle-edge first-count/return
test passed. Live acceptance OPEN. See docs/phase_reports/FIRST_PARTIAL_ENROLLMENT.md.

Latest D-046 identity core separates partial-view recognition from enrollment and
adds bounded, fixed-anchor-validated full-view templates. 165 tests passed;20-return
real-model proxy:15 correct/5 unresolved, no count inflation after10 resumes.
Live changed-view failure remains OPEN, phone unavailable at retest. See
docs/phase_reports/IDENTITY_CORE_REPAIR.md. Do not claim all-condition counting.

Short-blur repair D-045 retains recent clear Re-ID samples for <=0.75s through
brief blur/confidence dips, with individual expiry and unchanged novelty safeguards.
Candidate labels distinguish verification from identity. Live counting complaint
remains unaccepted; phone disconnected during repair. See BLUR_EVIDENCE_REPAIR.md
under docs/phase_reports; do not claim all-pose/all-blur reliability.

Latest counting-input repair: optional pre-perception rotation for confirmed
sideways phone input, independent sampling/comparison timers, and explanatory
pending labels. 151 tests passed; actual GPU static two-person smoke passed.
Corrected phone A/B/re-entry acceptance remains OPEN. See COUNTING_INPUT_REPAIR.md
under docs/phase_reports. No claim of perfect detection from every angle.

Explicit reset feature: refresh retains mission/counts. New Mission / Reset Demo
requires confirmation, preserves the old SQLite database and replaces tracker,
verification and appearance history. A guarded POST is now the only dashboard
write control. See `docs/phase_reports/RESET_DEMO.md`; older read-only-only notes
describe the original dashboard, not this authorized extension.

Latest correction supersedes D-041's co-visible rule: sequential different people
can enroll after sustained low-similarity, consistent fresh evidence (three batches,
>=1.5s). Co-visible requirement removed; same-frame conflicts/edge gates retained.
SQLite live stores audited consistent, not corrupted; historical records preserved.
Actual-model 276-frame one-pair replay: two sequential enrollments, two correct
returns, two persisted identities/two duplicate events after resume. Live validation
still OPEN. See `docs/phase_reports/SEQUENTIAL_IDENTITY_DATABASE_FIX.md`.

Latest repair, 2026-09-27: user reported live false unique counts and missed
duplicate associations. Edge-crop rejection and managed co-visible novelty guard
implemented; same-frame similar enrollment conflict prevented. 137 tests pass.
One-pair actual-model replay: A recovered, B safely unresolved, no false merge/split;
this undercounts sequential new arrivals and is NOT complete identity accuracy.
Old missions preserved; changed config requires a new mission. Live retest awaits
phone availability. See `docs/phase_reports/IDENTITY_FALSE_SPLIT_FIX.md`.

**Authoritative latest status, 2026-09-27: Phases 0-4 functional acceptance COMPLETE.**
Phase 4 adds configurable 3-of-5 temporal verification with confidence/visible-area
checks, current-evidence requirement, bounded history, expiry and outage/epoch reset.
User confirmed "Haan, Phase 4 demo working" after the live label/count/pan check.
Live run: 779 frames, 55 sampled states ONLINE, one connection, zero resets;
sampled display mean 13.70 FPS and maximum rolling verification p95 0.072 ms.
Up to three simultaneously verified temporary tracks; NOT three unique survivors.
No injury/liveness assessment or persistent count.
See `docs/phase_reports/PHASE_04_TEMPORAL_VERIFICATION.md` for tests and teaching.

**Phase 5 (2026-09-27): implementation + offline engineering validation COMPLETE;
live/field reliability OPEN.** User's phone became unavailable and they explicitly
requested completion without the camera. No further physical test required this turn.
Selective OSNet x0.25 MSMT17 appearance matching, quality filtering, three-view
comparison and uncertain states are implemented with session-only R references.
87 automated tests passed; 20 held-out COCO hard-colour negative pairs had zero
threshold crossings (proxy, not a local re-entry accuracy certification).
One-participant live test produced three references and one R3 re-entry match;
the three references cannot be ground-truth-labelled retrospectively from numeric
logs, and must not be claimed as successful identity continuity.
Known-input offline replay: 1,000 frames, 20 returns, 15 correct recoveries, zero
false-merge pairs/splits, 4 ambiguous and 1 blurred-crop unresolved return. All 15
enrolled references recovered in this synthetic-view replay; this is not live accuracy.
Final 87 tests passed; real YOLO/BoT-SORT/verifier/OSNet GPU smoke also passed.
User subsequently accepted Phase 5 and explicitly authorized Phase 6.
See `docs/phase_reports/PHASE_05_APPEARANCE_REID.md`.

**Phase 6 (2026-09-27): implementation, camera-free acceptance and learning handoff COMPLETE.**
Persistent mission-scoped S records, SQLite atomic audit/count/gallery storage,
conservative identity confirmation, duplicate-event suppression, explicit resume,
and read-only JSON reporting implemented. 112 tests pass. Ten offline missions,
1,000 frames and ten resumes: 15 accepted return associations, zero return-count
inflations; five uncertain returns remain unresolved. Real full GPU pipeline
smoke passed (24 static-image frames, two current/two unique/zero duplicates).
Embeddings now persist locally only in managed missions; no raw camera images.
Counts are appearance-based estimates, not medically verified survivors or
guaranteed real-world identities. Live/field reliability remains OPEN.
See `docs/phase_reports/PHASE_06_SURVIVOR_MANAGER.md`.

**Phase 7 (2026-09-27): implementation, camera-free/browser acceptance and handoff COMPLETE.**
Local FastAPI/Uvicorn read-only dashboard, latest annotated frame, estimated counts,
module health/metrics, searchable identities, filtered events and JSON export.
Explicit LIVE/LOCAL VIDEO/OFFLINE REPLAY/ARCHIVE modes; stale/disconnected current
occupancy is unknown, not zero. 130 tests passed; actual-model replay and full GPU
local-video/new-and-resumed mission checks passed. Live phone/field accuracy remains
OPEN. Original Phase 7 preview publishing was capped at 10 FPS.
Phase 8 is now authorized and performance fixes are implemented: configurable
30 FPS preview ceiling, duplicate-encode suppression, paced single-request browser
delivery, one annotation pass and bounded CPU workers. Offline regression/browser
checks passed; final wireless acceptance awaits a reachable phone stream. No live
20–30 FPS or zero-latency claim. See `docs/phase_reports/PHASE_08_PERFORMANCE.md`.
See `docs/phase_reports/PHASE_07_DASHBOARD.md` for launch commands and teaching.

Preserved Phase 3 baseline:
Phase 3 adds BoT-SORT temporary epoch-qualified IDs, sparse optical-flow GMC,
low-score recovery and safe resets. Re-ID remains off. 50 automated tests passed;
synthetic motion regressions had zero ID switches. User accepted live tracking
qualitatively; numerical live switch count was not supplied. Live sampled mean
display rate was 14.67 FPS, not a guaranteed 20-30 FPS.
See `docs/phase_reports/PHASE_03_BOTSORT_TRACKING.md`.
User explicitly confirmed successful detection of 2-4 people in the final live test.
25 automated tests, actual 120-second 720p wireless capture and working launcher;
user confirmed final movement responsiveness and automatic phone-server recovery.
Use 5 GHz, 720p, JPEG35, motion-off, direct MJPEG; 540p remains a fallback.
No guaranteed minimum FPS or numerical glass-to-glass latency. Phase 2 uses local
YOLO11n weights, CUDA FP16, fixed 640 input, person-only boxes, configurable
confidence/IoU. Balanced 200-image COCO smoke benchmark and live GPU inference ran.
Intermittent bad-JPEG reconnect bug fixed; 36 tests pass. User confirmed one-person
boxes and accepted the fixed moving-camera preview ("Perfect, ok"), then confirmed
2-4-person detection. This is controlled-demo acceptance, not field-rescue certification.
See `docs/phase_reports/PHASE_02_PERSON_DETECTION.md` for the preserved detection baseline.
See `docs/phase_reports/PHASE_01_HANDOFF.md`. Bullets below retain earlier history.

- Context package created.
- No application source code was present in the target folder when this package was created.
- 2026-09-27: Phase 0 completed in repository-root `.venv` (Python 3.11.15).
  CUDA PyTorch, RTX 3050 matrix operation, imports, dependency consistency,
  synthetic local video and OpenCV GUI calls passed. See
  `docs/phase_reports/PHASE_00_ENVIRONMENT_SETUP.md` for evidence and teaching notes.
- 2026-09-27: Phase 1 implemented; 17 automated tests pass. Real phone capture
  and preview worked, but frame-rate dips were observed. Rendering optimisation
  reduced isolated mean render time from 9.9 to 3.0 ms. Actual-phone retest is
  pending because the phone server became unavailable. Phase 1 acceptance is
  not yet complete; Phase 2 has not started. See the Phase 1 learning report.
- 2026-09-27: user-requested COCO preparation completed separately: 5,000 COCO
  validation images verified, original annotations preserved, and 4,773-image
  person-only evaluation subset prepared. See `docs/COCO_DATASET_SETUP.md`.
- Official current-version rulebook PDF still needs to be stored and clause-checked locally.
- Follow-up actual-phone test: changed 1080p input to 720p. Continuous MJPEG
  still visibly delayed; optional latest-snapshot mode reduced delay according
  to the user, at ~8.4 FPS. Phase 1 smoothness and measured end-to-end latency
  remain pending. See Phase 1 report follow-up section.
- Further lag work: single-decoder experiment rejected by user; default rolled
  back. Direct MJPEG mode added and live-tested (~20.5 displayed FPS over 40 s,
  including one recovered interruption). Its visible-delay acceptance is pending.
- Latest feedback supersedes pending status: direct MJPEG still has seconds of
  delay after the 720p retest. Latency acceptance FAILED. USB-tethered diagnostic
  proposed; physical phone/cable action pending. Battery was 2% at last check.
- Latest user constraint supersedes USB proposal: wireless only. Trial profile
  960x540, JPEG quality 35, phone motion detection off; direct MJPEG retained
  for comparison. User confirms hotspot supports 5 GHz; band switch and retest
  pending. Charging now reports AC. Delay acceptance remains unresolved.
- Latest wireless comparison: verified 5 GHz and reapplied 960x540/quality35/
  motion-off after app restart. User confirmed smoother video AND less visible
  delay. Test consumed 580 frames over ~22.3 s (~26 FPS); rolling dips to 14 FPS
  still occurred. Current working profile, not full Phase 1 acceptance; 720p,
  sustained performance and measured total latency remain unverified.

**Domain dataset preparation (2026-09-27): COMPLETE; training not started.**
Downloaded and fully validated a 2,876-image Roboflow Fallen Person v2 YOLO11
export. Preserved its four original pose labels and generated a one-class
`person_candidate` view for future fine-tuning. An unchanged COCO YOLO11n baseline
reached .6144 recall on the supplied 300-image test split, confirming a pose-domain
gap; supplied splits share coarse source groups, so this is diagnostic rather than
field accuracy. See `docs/phase_reports/FALLEN_PERSON_DATASET_ACQUISITION.md`.

**Complete VisDrone integration (2026-10-06): dataset path COMPLETE; candidate not
promoted.** The full 8,629-image converted VisDrone DET source is included in
`datasets/nidar_person_v4_full_visdrone` with Fallen and COCO data. Exact manifest,
pair and label-schema checks pass. A resumable train/evaluate/gate workflow and
Windows/VS Code launchers are present. This is a research candidate path: the live
runtime still defaults to COCO YOLO11n while training/evaluation, licence review,
project-owned OAK-D evidence and edge export remain separate gates. See
`docs/phase_reports/FULL_VISDRONE_INTEGRATION.md`.

**Hybrid detector promotion (2026-10-07): laptop demo integration COMPLETE.**
Validation-calibrated COCO-primary plus posture/aerial-specialist fusion passed the
unchanged public-data engineering gate 10/10. Held-out lying recall is .9318,
fallen .9222, sitting .9375, standing 1.0, small-person .2371 and aerial .1217;
COCO precision/recall are .7771/.7214. Phone, USB and offline detection launchers
now select hybrid mode by default; `-BaselineDetector` retains an explicit rollback.
The six offline scenes and motion variants preserve counts 1/2/4/6/4/6. These are
person/body candidates, not medical survivor labels. Project-owned OAK-D field
acceptance and depth/SLAM duplicate suppression remain OPEN. See
`docs/phase_reports/HYBRID_PERSON_DETECTOR_PROMOTION.md`.

**Multi-person identity scaling repair (2026-10-07): COMPLETE.** The selective
Re-ID `max_batch=2` GPU bound no longer starves tracks after the first two. Fair
least-recently-sampled scheduling processes all qualified people over successive
frames. Integrated automated evidence creates/persists S1-S8 and recovers all eight
returns without unique-count inflation. The configured mission gallery holds 64
references; there is no S1/S2 limit. See
`docs/phase_reports/MULTI_PERSON_IDENTITY_SCALING.md`.
