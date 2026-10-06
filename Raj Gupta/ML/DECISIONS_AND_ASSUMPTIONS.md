# NIDAR AirMouse — Decisions, Assumptions, and Open Questions

## 1. Decision log

2026-10-05 — D-052 — PROJECT-DECISION: add USB phone-camera input as an
alternative, without changing or removing the accepted Wi-Fi IP Webcam and
offline demonstrations. Windows must expose the phone as a normal webcam device;
a cable alone is not assumed to provide video. Added bounded camera-index
discovery/validation, DirectShow default with MSMF fallback, guaranteed device
release, a separate dashboard-free `Start-USBCameraDemo.ps1`, and VS Code tasks.
The default USB Full mode reuses the unchanged YOLO -> BoT-SORT -> temporal
verification -> Re-ID -> persistent manager pipeline and creates a fresh mission
database. 179 tests pass. Windows camera index 0 ran camera-only at about 29.6-
29.8 FPS/1280x720 (166 consumed frames), then ran the Full CUDA pipeline (243
consumed frames), both with clean shutdown. This device was not ground-truth-
verified as the user's Moto; Moto USB compatibility remains HARDWARE-VERIFY.
See `docs/phase_reports/USB_PHONE_CAMERA_FALLBACK.md`.

2026-10-01 — D-051 — PROJECT-DECISION: replace the misleading single-photo
offline presentation with a separate dashboard-free multi-scene detector
showcase. Six local COCO images are processed continuously by the real default
YOLO11n at presentation confidence .25 and cycle through measured counts
1,2,4,6,4,6 in the original camera-test layout. Use P labels for
current-frame boxes; do not assign persistent S IDs across unrelated stock
photos because there is no valid temporal/re-entry identity relationship.
Identity/duplicate demonstration remains separate. Three focused tests, all 174
project regression tests, dependency consistency and the full six-scene CUDA
validation pass. See OFFLINE_MULTI_PERSON_SHOWCASE.md.

2026-09-30 — D-050 — PROJECT-DECISION: final detector semantics are one class,
`person_candidate`, covering visible people/bodies in standing, sitting, crouching,
lying, partial, occluded and aerial views. It must never infer injury, liveness or
death. Added VisDrone2019-DET aerial data (pedestrian + people collapsed), rebuilt
fallen data with coarse source-group separation, and mixed it with COCO positives
and negatives. V5 aerial-heavy, v6 balanced and v7 short curriculum candidates
were trained and evaluated. V7 achieved fallen/lying recall .949/.932, small-target
recall .227 and 38.97 ms p95 inference, but general COCO recall/precision were only
.469/.685 and 43 COCO negative images produced false positives. The explicit gate
passed 7/10 and rejected promotion. `models/yolo11n.pt` therefore remains the safe
demo default; v7 is retained as research evidence, not a final model. VisDrone
licence clearance is REVIEW-REQUIRED. Project-owned indoor OAK-D footage, a new
locked field test, edge export and depth/SLAM validation remain mandatory before
final-drone acceptance. See NIDAR_DETECTOR_DATASET_AND_TRAINING.md.

2026-09-27 — D-049 — Enable sustained partial novelty in the demo settings:
remove first-person-only edge enrollment restriction, retain .55 novelty and
multi-batch/time checks; preserve recent good samples across brief verification
dips with existing expiry. 70 focused tests passed including edge S2 persistence
and S1 return. Live phone refused connection; real-world identity accuracy OPEN.
See docs/phase_reports/SECOND_PERSON_COUNT_FIX.md. Old mission data preserved.

2026-09-27 — D-048 — Lying-person gate root cause measured on provided screenshot:
YOLO score up to .299; BoT-SORT new-track .40, verifier .35, Re-ID crop .50
blocked path. Align those gates at .20 (tracker low .10), preserve temporal 3/5
and identity checks; rotate horizontal crops for OSNet; current card reflects
tracked candidates. No live retest: camera unreachable. False-positive/delay impact
remains open. See docs/phase_reports/LYING_PERSON_COUNTING_FIX.md.

2026-09-27 — D-047 — First-count gate: allow multi-frame consistent edge-view
bootstrap only for an empty gallery, controlled by allow_initial_partial=true.
All other quality/temporal/manager checks stay; later partial views cannot enroll.
166 tests pass; actual OSNet/oracle-edge test counts1 then duplicate1, first count
frame9 (0.8s simulated, not live latency). Prior D-046 did not fix empty-gallery
failure. Read docs/phase_reports/FIRST_PARTIAL_ENROLLMENT.md; live acceptance OPEN.

2026-09-27 — D-046 — Identity core: separate edge-view recognition from new
enrollment; partial batches may recover existing IDs only at >=.90 with ambiguity
and quality gates, never create/update gallery entries. Full accepted returns may
add diverse templates only against immutable original anchors; cap12 vectors.
165 tests pass; real-model 20-return proxy:15 correct/5 unresolved, zero false
merge pairs/splits;10 resumes preserve counts. Live phone refused connection;
user's live identity failure not certified solved. See IDENTITY_CORE_REPAIR.md.

2026-09-27 — D-045 — Short blur/confidence dips preserve individually timestamped
good Re-ID samples for <=0.75s; bad frames never authorize counting. Novelty resets,
clipping/overlap/track-loss rejection and thresholds remain. Candidate labels now
distinguish temporal verification from identity acceptance. Real-model proxy replay
retains two identities/two correct returns; live RECONNECTING, unresolved scene
acceptance. See docs/phase_reports/BLUR_EVIDENCE_REPAIR.md. No all-condition guarantee.

2026-09-27 — D-044 — Counting-input repair: user confirmed head-left sideways
feed; add explicit pre-perception quarter-turn rotation (default 0, live trial 90).
Separate Re-ID sampling from comparison cooldown; retain quality/novelty/match
thresholds. Show actionable pending reasons on boxes. 151 tests pass; real GPU
static two-person smoke creates S1/S2 and shuts down cleanly. Phone unavailable
for corrected acceptance; no perfect-angle/live identity claim. Read
docs/phase_reports/COUNTING_INPUT_REPAIR.md. No database cleanup or later phase.

2026-09-27 — D-043 — User accepted explicit New Mission / Reset Demo, NOT
reset-on-refresh. Dashboard now has a guarded POST control; all other operations
remain read-only. Same-origin/token/current-mission validation, one pending command,
perception-thread rotation and non-destructive SQLite preservation. Last 20 reset
summaries exportable in session; old databases remain on disk. LIVE/LOCAL VIDEO
only. See docs/phase_reports/RESET_DEMO.md. This changes lifecycle, not ML accuracy.

2026-09-27 — D-042 — D-041 co-visible novelty policy REJECTED: user brings a
different person sequentially and cannot enroll. Replaced with three fresh,
non-overlapping consistent novelty batches over >=1.5 seconds, fixed-anchor
consistency, quality reset and unchanged match/novelty thresholds. Storage audit
found two live databases structurally consistent; no destructive cleanup needed.
Real-model 276-frame replay: two enrollments/two correct returns, no false merge
or split in that pair, persisted/resumed totals correct. Live acceptance remains
OPEN; phone refused connection. Full report: SEQUENTIAL_IDENTITY_DATABASE_FIX.md.

2026-09-27 — D-041 — PROJECT-DECISION: live user failure supersedes offline
identity acceptance as evidence of field readiness. Add 1% edge crop rejection,
managed co-visible novelty guard and same-frame enrollment conflict check. Weak
match is not novelty proof. Trade-off explicitly explained: sequential different
people can remain unresolved, not falsely merged. 137 tests pass; actual-model
one-pair replay recovers A, leaves B pending, zero splits/merges. Old data retained;
new config signature prevents incompatible resume. Corrected live test unavailable
because phone HTTP refused connection. No claim that all identity errors are fixed.
See docs/phase_reports/IDENTITY_FALSE_SPLIT_FIX.md.

| ID | Decision | Status | Reason |
|---|---|---|---|
| D-001 | Build a phone-to-laptop ML proof of concept before the drone. | Accepted | isolates perception risk from flight risk and gives a fast demonstrable milestone |
| D-002 | Use an Android phone as a moving camera over local Wi-Fi/hotspot. | Accepted for demo | cheaply simulates camera motion and viewpoint changes |
| D-003 | Use latest-frame-only threaded capture. | Accepted | prevents stale frames and growing latency |
| D-004 | Start with a lightweight pretrained YOLO person model. | Accepted | validates the full pipeline before dataset/fine-tuning work |
| D-005 | Use BoT-SORT for temporary tracking in the demo. | Accepted | provides short-term track continuity under moving-camera conditions |
| D-006 | Separate tracker IDs from persistent survivor IDs. | Architectural invariant | tracker IDs can change after occlusion or re-entry |
| D-007 | Require multi-frame temporal confirmation. | Accepted | prevents brief false detections from creating survivors |
| D-008 | Use selective appearance Re-ID with cosine similarity. | Accepted for demo | supports re-entry matching without paying the cost every frame |
| D-009 | Store multiple representative embeddings per survivor. | Accepted | one crop may not represent all angles/lighting |
| D-010 | Add spatial/depth/SLAM evidence in the final drone. | Accepted | appearance alone cannot safely solve duplicate identity |
| D-011 | Use a Pixhawk-class flight controller for flight-critical control. | Accepted direction | separates deterministic flight control from AI workload |
| D-012 | Use Raspberry Pi 5-class companion compute for ROS 2, SLAM, fusion, and mission logic. | Accepted direction | provides general-purpose onboard integration compute |
| D-013 | Use OAK-D Lite-class RGB/stereo depth sensing. | Accepted direction | combines RGB, depth, IMU, and edge vision capability |
| D-014 | Target a professional operator dashboard after backend ML is stable. | Accepted | avoids polishing an unstable pipeline while ensuring demo clarity |
| D-015 | Keep recording off in the primary demo until performance is verified. | Accepted | reduces I/O load and latency risk |
| D-016 | Use INR 2,00,000 as the team's planned hardware ceiling. | Accepted project target | controls scope; not yet verified as an official rule |

## 2. Working assumptions

| ID | Assumption | Validation needed |
|---|---|---|
| A-001 | The phone can provide a stable 720p/30 FPS local stream. | test actual phone app, codec, network, and end-to-end latency |
| A-002 | RTX 3050 4 GB can run Nano-class YOLO, BoT-SORT, and selective Re-ID smoothly. | benchmark full pipeline; expected 20–30 FPS class is not guaranteed |
| A-003 | A person-pretrained model is sufficient for the first demo. | test standing, sitting, lying, occluded people and any dummy representation |
| A-004 | Appearance Re-ID will be useful in a controlled room demo. | measure false merge/split rates with similar clothing and angle changes |
| A-005 | OAK-D Lite depth range and image quality are adequate for the arena. | test in actual light, distance, texture, and motion conditions |
| A-006 | Raspberry Pi 5 can sustain the selected SLAM and mission workload. | benchmark with actual sensors, runtime, cooling, and power |
| A-007 | The final architecture can fit under INR 2 lakh excluding existing laptop/phone. | obtain vendor quotes and full weight/power BOM |
| A-008 | A local communication link is available in the venue. | verify rulebook and venue radio/network conditions |

## 3. Open technical decisions

| ID | Question | Decision method |
|---|---|---|
| O-001 | Which exact YOLO version and model size? | Laptop baseline resolved: YOLO11n (D-026); final-drone export/model still open |
| O-002 | Which exact Re-ID encoder? | compare licence, CPU/GPU cost, controlled re-entry accuracy, and export support |
| O-003 | What are the Re-ID thresholds and pending-state duration? | calibrate on project-specific positive and negative pairs |
| O-004 | Which ROS 2 distribution? | match Pi OS/Ubuntu support window and package compatibility |
| O-005 | Which SLAM/VIO stack? | compare OAK support, loop closure, CPU load, map output, and indoor robustness |
| O-006 | Where will final YOLO inference run? | benchmark OAK accelerator vs Pi runtime on the final model |
| O-007 | What airframe/propulsion combination? | size from payload mass, thrust margin, guards, corridor width, and flight time |
| O-008 | Which Pixhawk model/autopilot? | choose current supported hardware after I/O, firmware, cost, and rulebook review |
| O-009 | What grid-cell definition and localisation tolerance are required? | verify official rulebook/mission briefing |
| O-010 | How are survivors represented in finals? | verify real person/dummy specification and allowed training information |
| O-011 | What data may be stored during/after the mission? | verify privacy, rules, and demo consent requirements |

## 4. Explicit non-decisions

- Laptop Phase 2 model/thresholds selected in D-026; final-drone model and later identity thresholds remain open.
- A specific ROS 2 SLAM package has not been selected.
- INR 2 lakh is not claimed to be an official NIDAR budget cap.
- The indexed rulebook copy is not treated as the final authority.
- No claim is made that appearance Re-ID alone can uniquely identify survivors in the final arena.
- No ready-to-fly aircraft or final propulsion system has been selected.

## 5. How to update this file

Add a dated note whenever a decision changes:

```text
YYYY-MM-DD — ID — old status -> new status — evidence and reason — affected files
```

Initial context package created: 2026-09-26.

2026-09-27 — D-017 — Phase 0 environment selected — repository-root `.venv`,
Python 3.11, CUDA-enabled torch 2.11.0/torchvision 0.26.0 pair, graphical OpenCV,
NumPy and Ultralytics. Existing global Python remains separate. Final validation
is recorded in `docs/phase_reports/PHASE_00_ENVIRONMENT_SETUP.md`.

2026-09-27 — D-018 — Phase handoff formalised — user requires structured learning
reports and confirmation before the next major phase. Current scope is Phase 0
only; tracker and Re-ID dependencies deferred to their relevant phases. Affected:
AGENTS.md, ML_DEMO_PLAN.md, phase report.

2026-09-27 — D-017 validation — PASS: Python 3.11.15, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, actual RTX 3050 CUDA computation, OpenCV synthetic
video/GUI and package imports. Full package snapshot saved. Phase 0 complete;
no phone stream or person inference implemented. See phase report for limitations.

2026-09-27 — D-019 — User requested COCO installation — prepare official COCO 2017
validation images and annotations, with a derived person-only evaluation set.
Full training images deferred because the current goal is pretrained baseline
evaluation. Crowd-person images excluded only from the derived list; original
data retained. This is an explicit dataset-preparation task, not Phase 1/2 approval.
See docs/COCO_DATASET_SETUP.md and generated datasets/coco2017/manifest.json.

2026-09-27 — D-020 — User authorised Phase 1 and lag fixes. Selected IP Webcam
for local Android MJPEG streaming. Implemented latest-frame capture, stale-frame
rejection, reconnect, preview and tests. Default preview refresh ceiling is
60 Hz independently of the 30 FPS camera request; it does not create extra
camera frames or set the phone's FPS. Cached background and fresh-frame redraw
reduce laptop rendering work. Actual wireless improvement remains unverified
after phone server became unavailable. No Android battery/network settings
were changed and no Phase 2 inference was added. Affected: demo source/tests,
AGENTS.md, PROJECT_CONTEXT.md, ML_DEMO_PLAN.md and Phase 1 report.

2026-09-27 — D-021 — Actual-phone lag retest — discovered 1920x1080 input;
changed only IP Webcam video_size to supported 1280x720 through local API.
JPEG quality remained 49; Android system/battery settings unchanged. Continuous
MJPEG still visibly delayed per user, despite higher decode throughput. Added
optional snapshot transport using the app's documented /shot.jpg latest-frame
endpoint. User confirmed noticeably reduced visible delay; 251 frames consumed
in 30 seconds (~8.4 FPS), clean stop, no reconnect. Retain both modes; use snapshot
for current low-delay testing, not as a claim of 20–30 FPS acceptance. No ML
integration started. Files: snapshot_capture.py, config.py, phone_stream.py,
network tests, README and phase report.

2026-09-27 — D-022 — User requested smoother low-delay video. Added configurable
FFmpeg decoder thread count (default 1; 0 restores automatic) to reduce potential
frame-thread buffering in live input. Official OpenCV confirms this is an
open-only FFmpeg property; latency benefit is a hypothesis until visually
validated. First 720p live run lasted about 23.7 seconds before clean exit,
consumed 502 frames, and sampled display FPS ~19–26.5. No extra Android changes.
Snapshot remains the user-confirmed lower-delay fallback. Phase 1 still active.

2026-09-27 — D-022 rejected — user reported increased visible delay with one
FFmpeg decoder thread. Default restored to 0 (automatic); retain CLI option for
explicit comparisons only. Do not describe single-thread mode as a verified fix.

2026-09-27 — D-023 — Added optional direct MJPEG transport after user requested
further reduction of delay. Separate bounded multipart receiver and latest-JPEG
slot bypass FFmpeg's stream/decode queues; OpenCV decodes only latest JPEG.
40-second live run consumed 821 frames (~20.5 FPS including interruption),
mostly 20–22 displayed FPS, one recovered interruption. Visual latency
confirmation pending; zero-latency guarantee explicitly rejected. Snapshot
fallback retained; no new dependencies or phone setting changes in this step.

2026-09-27 — D-023 latency acceptance FAILED — user confirmed seconds of delay
remain in direct MJPEG after restart/retest. Restart returned phone to 1080p;
720p was reapplied and verified, but visible delay remained. Latest status:
2.4 GHz link, battery 2%, no charging flag. USB-tethered comparison proposed to
separate network-link delay from phone/source behaviour; user must connect
cable and enable tethering. Wired operation is diagnostic, not a replacement
for the moving-wireless-camera goal. Exact root cause remains open.

2026-09-27 — D-024 — User explicitly rejected USB; wireless-only troubleshooting.
Applied supported 960x540 / JPEG quality 35 / app motion detection OFF through
IP Webcam local controls (previous 1280x720 / 49 / ON). This is a trial profile,
not a permanent reduction of the 720p target. Preview now states local frame
age is NOT total delay and camera-to-screen delay is unmeasured. User confirms
5 GHz hotspot option exists; asked user to switch band after test and restart
camera, then provide address. No Android network settings changed remotely.
14 unit tests passed after label edit. Latest battery check: 1%, charging AC.

2026-09-27 — D-024 wireless comparison — laptop verified 5 GHz / 802.11ac,
channel 149, negotiated link rate 433.3 Mbps (not measured throughput).
Phone restart reset 1080p/quality49/motion-on; reapplied 960x540/35/motion-off.
Direct MJPEG live test 12:14:51–12:15:14 (~22.3 seconds before clean exit)
consumed 580 frames (~26 FPS including startup), one connection. Displayed
rolling rate initially 31–32, dipped to 14–20, recovered to 30–32. User confirmed
"Smooth hai aur delay bhi kam hai". Adopt this as the current working wireless
profile; no zero-latency or sustained-30 guarantee. 720p restoration, longer
movement/reconnect tests and measured end-to-end latency remain pending.

2026-09-27 — D-025 — Phase 1 functional acceptance COMPLETE after final tests.
720p/5GHz/quality35/motion-off with direct MJPEG; actual 120-second headless
test: 4,751 frames, all sampled states ONLINE, one connection, clean shutdown.
Final launcher preview: 677 frames/~31.5s; user confirmed movement test pass
and automatic server-restart recovery. Recovery is user-observed, not separately
timed in the one-connection launcher log; automated recovery tests also passed.
25 tests passed overall and pip check passed. Added Start-PhoneDemo.ps1 with
profile verification and best-effort setup rollback; no new dependencies.
Numeric total latency, sustained minimum FPS and multi-hour endurance remain
unmeasured, not guaranteed. Phase 2 still requires explicit confirmation.

2026-09-27 — D-026 — PROJECT-DECISION: User explicitly authorized Phase 2 after
accepting Phase 1. Integrated official YOLO11n COCO detection weights using the
existing pinned Ultralytics 8.4.163 stack. Default fixed-square letterboxed 640,
confidence .35, NMS IoU .45, class 0 only, CUDA FP16 (CPU FP32 fallback).
Balanced seed-42 200-image COCO evaluation: 640 precision .9086, recall .5652,
median/p95 full model-call 28.29/33.09 ms; 512 .8927/.5284, 28.98/36.04 ms.
Not official mAP, not full COCO evaluation, not survivor accuracy. The larger
input found more labelled people on this sample without a latency penalty.
No custom training or downstream tracking/re-identification/counting introduced.
Local weight SHA-256 and source recorded in the Phase 2 report. AGPL/Enterprise
licence options documented; no licensing compliance determination made.

2026-09-27 — D-027 — PROJECT-DECISION / measured bug fix: New live AI run exposed
intermittent invalid JPEG decode results. A single invalid frame previously
triggered a full reconnect and two-second retry gap. Reader now skips bad images
within a fixed per-read timeout, preserving the connection and latest-only
semantics; continuous corruption/stall still times out. Upstream cause of malformed
images is not established; do not blame Wi-Fi without evidence. Added sanitized
diagnostics and a corrupt-image regression test. All 36 tests pass. Post-fix
graphical check: 613 processed frames/~28.3 seconds, all 27 sampled states ONLINE,
one connection and clean exit. User confirmed one-person boxes then replied
"Perfect, ok" to the fixed-preview movement/blank-screen check. Number of additional
people was not supplied; 2-4-person physical acceptance remains unverified.

D-027 follow-up: 60-second headless phone+GPU test processed 1,384 frames with
all 59 sampled states ONLINE, one connection and clean exit. A real malformed
JPEG was skipped without reconnecting. Mean sampled consumer rate 23.13 FPS,
max sampled local age 109 ms (not end-to-end latency). User requested a new
graphical run specifically to test 2-3 people; opened a 180-second test window.

2026-09-27 — D-028 — Phase 2 functional acceptance COMPLETE. User explicitly
reported: "Yes, it can detects 2-4 person's, now finish phase 2 fast and precisely".
This closes the remaining controlled-demo multi-person acceptance check; earlier
unverified notes above are historical. Final graphical test ended cleanly after
1,135 processed frames, with one connection. Existing 36-test suite passed;
COCO sample baseline, GPU performance and bad-frame recovery evidence retained.
No inference code changed during closeout. Reports, phase plan, README, working
instructions and context now agree. Phase 3 remains unstarted pending authorization.

2026-09-27 — D-029 — PROJECT-DECISION: user authorized and accepted Phase 3
BoT-SORT temporary tracking. Separate adapter consumes existing YOLO output;
tracking confidence .10, high/low/new .35/.10/.40, 30 processed-frame lost buffer,
match cost .80, confidence fusion ON, sparseOptFlow GMC downscale 2, Re-ID OFF.
lap==0.5.12 installed in project .venv and pinned in both dependency files.
No additional model downloads. JSON tracker settings and optional exclusive-create
JSONL trajectory metadata added; no camera recording. Empty updates age tracks;
camera outages, shape changes and >1s frame gaps clear motion/history. Epoch-qualified
temporary labels cannot be mistaken for the same local ID across a reset in one run.
Phase 2 detection-only and Phase 1 camera-only modes retained.

2026-09-27 — D-030 — Phase 3 functional acceptance COMPLETE. 50 final automated
tests passed in 28.288s, pip consistency passed. Four 80-frame synthetic association
scenarios (motion, crossing paths, brief occlusion, camera pan) had 100% matched
visible observations and zero ID switches. These use oracle boxes, not real-world
YOLO accuracy. Real-phone GUI: 367 processed frames/~25.4s, 24 sampled states all
ONLINE, one connection, zero resets, clean exit. Mean sampled display 14.67 FPS,
range 11.78-16.77; no sustained-20/30 claim. User answered "Yes, All working fine"
to walking/crossing/occlusion/pan acceptance and asked to finish without repeated
physical tests. Numerical live switch count not supplied; retained as unknown.
Trial GMC downscale 4 failed pan regression (27 switches, .646875 coverage) and
was rolled back; downscale 2 revalidated. At that closeout Phase 4 was not started; no persistent
survivor identity or Re-ID implemented. Full handoff: PHASE_03_BOTSORT_TRACKING.md.

2026-09-27 — D-031 — PROJECT-DECISION: user authorized Phase 4. Added independent
temporal verifier and JSON configuration, without new packages, model downloads,
training or architecture redesign. Default 3-of-5 processed fresh updates, score
>= .35 and clipped visible area ratio >= .0001. Current evidence must qualify;
weak/missing observations revoke confirmation. Missing updates age the window,
five consecutive missing updates expire it; outage/epoch/>1s gaps reset evidence.
History is bounded to a rolling window per recently observed temporary track.
Freshness is checked after inference before verification. Phase 4 reports verified
visible tracks and transition events only, not unique survivors or medical status.
Persistent counting remains Phase 6; appearance Re-ID remains Phase 5.

2026-09-27 — D-032 — Phase 4 live functional acceptance passed. User answered
"Haan, Phase 4 demo working" to green-label, count-removal and pan/re-entry check.
60-second configured GUI run: 779 consumed frames, 55 sampled states all ONLINE,
one connection, no tracker reset, clean shutdown, up to 3 verified-visible tracks.
Mean sampled display 13.70 FPS (10.93–15.24); maximum rolling verification p95
0.072 ms. First confirmation per temporary ID averaged 173.32 ms over 19 events;
these are track episodes, not 19 people, and elapsed time starts at first tracked
observation, not sensor exposure. CPU regression tests overlapped part of this run;
this is not an isolated performance benchmark. No numerical glass-to-glass delay,
real-world false-positive rate, or live ID-switch count established.
Final test evidence and full learning handoff are in PHASE_04_TEMPORAL_VERIFICATION.md.

2026-09-27 — D-033 — PROJECT-DECISION: Phase 5 authorized. Implemented selective
OSNet x0.25 MSMT17 appearance matching using author-hosted checksum-pinned weights
and pinned MIT network source (no full torchreid installation). Existing Pillow
12.3.0 explicitly pinned; no environment upgrades. Local RAM-only R references
separate from temporary E:T IDs and future S survivor records. Three quality-gated
views, normalized cosine, ambiguity margin .08, conservative novelty .55, max 2
crops per frame, 64-reference cap, 600s inactive expiry, frozen reference vectors.
Active/simultaneous reference conflicts remain uncertain. Stale results cannot
commit identities; outages clear bindings but retain session gallery.

2026-09-27 — D-034 — Calibration/validation evidence: 40 independent COCO images
with same-image distinct-person pairs selected for similar colour histograms;
first 20 calibrate match threshold to .8146608 (runtime rounded up .815), next 20
yield 0/20 negative threshold crossings and 20/20 augmented positive accepts.
This is a labelled-image proxy, not true cross-camera re-entry accuracy or local
similar-clothing certification. Actual CUDA batch-three median 59.01ms/p95 72.84ms.
87 automated tests passed in 29.382s, pip check clean. Live one-person test being
performed; user has only one participant. Two-person similar-clothing field check
unavailable, not passed. Phase 6 not implemented/authorized.

2026-09-27 — D-035 — User revised scope: phone unavailable; finish precisely
without camera and self-validate. Completed camera-free labelled replay using
actual OSNet/temporal/matcher with real COCO crops, oracle tracks and synthetic
view changes. 10 held-out person pairs, 1,000 frames, 20 return attempts: 15
correct recoveries, zero false-merge pairs, zero false splits, 4 ambiguous and
1 blur-rejected unresolved returns. All 15 enrolled references recovered; this
conditional result is not 100% overall accuracy. Retained conservative thresholds.
Actual full YOLO/BoT-SORT/verifier/OSNet GPU smoke: 24 frames, two verified persons,
two session references, 8 crops encoded, peak PyTorch allocated 56.30 MiB (not
whole-process/driver VRAM). CPU/CUDA normalized-vector cosine .999991 on a random
input smoke, not identity accuracy. Final 87 tests pass in 29.095s; pip check clean.
Live run ended cleanly, 1,743 frames, one connection, mean sampled display 9.71 FPS;
three references and one match are not ground-truth-validated successes.
Phase 5 implementation/offline engineering complete; field/live acceptance OPEN.
Do not claim robust similar-clothing recognition or advance Phase 6 automatically.

2026-09-27 — D-036 — PROJECT-DECISION: user accepted Phase 5, explicitly authorized
Phase 6 and retained the camera-free scope. Implemented mission-scoped S records,
SQLite atomic record/count/event/gallery persistence, explicit compatible resume,
three fresh identity confirmations and conservative pending states. --manage
retains the bounded 64-reference gallery for the mission rather than expiring it.
Embeddings now persist locally in managed mode; no raw camera images or cloud
upload. Standalone --reid remains session-only. No new dependency or training.

2026-09-27 — D-037 — Phase 6 acceptance and handoff COMPLETE for offline scope.
112 tests passed in 31.618s. Ten missions/1,000 replay frames/ten resumes: 15
accepted returns, 15 duplicate events, zero return-count inflations, five unresolved
returns (four ambiguity, one blur). Zero false-merge pairs/splits in this bounded
synthetic-view test, not a field guarantee. Full GPU 24-frame static-image smoke:
two current, two unique, zero duplicates. Camera-free graphical pair demo and
read-only JSON export passed. Report: docs/phase_reports/PHASE_06_SURVIVOR_MANAGER.md.
Live/field identity reliability remains OPEN. Phase 7 awaits authorization.

2026-09-27 — D-038 — PROJECT-DECISION: user authorized Phase 7. Added local-only
FastAPI 0.141.1/Uvicorn 0.54.0 dashboard with plain HTML/CSS/JS, no CDN/build server,
no browser write routes, fixed safe asset/API paths and Host/Origin checks.
Bounded mailbox and latest JPEG publishing (10 FPS cap) separate presentation from
identity authority. Camera/file/replay/archive are explicitly distinguished;
unknown current counts remain null. Added launchers, safe export and recent-event
restore without changing matching thresholds or database schema.

2026-09-27 — D-039 — Phase 7 offline/browser acceptance and handoff COMPLETE.
130 tests pass in 32.275s; pip check clean. Browser desktop/mobile/search/filter/
fullscreen/export/disconnect checks passed. Actual-model one-pair replay: 100
frames, two correct returns, two records, two duplicate events, one resume, no
count inflation. Full GPU static COCO video: 254 frames initial and 208 resumed,
both clean shutdown; two records preserved. Mean sampled initial consumer rate
9.09 FPS on a 10-FPS input, not a live performance guarantee. Live phone/field
identity accuracy remains OPEN. At this point Phase 8 had not started. Complete report and commands:
docs/phase_reports/PHASE_07_DASHBOARD.md.

2026-09-27 — D-040 — PROJECT-DECISION: user authorized Phase 8 performance work.
Removed the 10 FPS dashboard ceiling and browser's additional fixed 100 ms wait;
added configurable 30 FPS ceiling, unique-frame encode/decode suppression, single
annotation pass, two CPU worker threads and resource metrics. Preserve 640 YOLO
input, FP32 Re-ID, identity thresholds, freshness checks and SQLite durability.
Final 133 tests passed in 46.031s; pip check clean. GPU local video consumed 1288
frames and shut down cleanly; browser visual/search checks passed. Static source
is approximately 10 FPS, so this does not certify 20–30 FPS live performance.
Phone address refused HTTP and failed TCP checks. Wireless movement acceptance
remains OPEN; do not label Phase 8 fully live-accepted or start Phase 9 yet.
See docs/phase_reports/PHASE_08_PERFORMANCE.md for evidence and teaching.

2026-09-27 — D-041 — PROJECT-DECISION: before Phase 9, user authorized acquisition
of a posture-diverse public fallen-person dataset. Downloaded Roboflow Fallen Person
v2 YOLO11 export (source claims CC BY 4.0): 129,018,934 bytes, SHA-256
2055abd525e8903047682cb8c55dd69bfb87376b87aadcefa4a2c86a2d29e517. All 2,876
images and labels validate; 3,297 boxes cover fallen, lying, sitting and standing.
Created a non-destructive one-class `person_candidate` view because RGB posture is
not proof of survivor/dead/injured status. Supplied splits share coarse source groups,
so their metrics are diagnostic only. Unchanged COCO YOLO11n baseline on 300 images:
precision .3639, recall .6144, mAP50 .3729, mAP50-95 .1852, 34.67 ms/image.
Acquisition/preparation is complete; fine-tuning and live-model promotion are not.
Phase 9 remains unstarted and Phase 8 wireless acceptance remains open. Report:
docs/phase_reports/FALLEN_PERSON_DATASET_ACQUISITION.md.

2026-10-06 — D-053 — PROJECT-DECISION: integrate the complete converted
VisDrone2019-DET source into a detector research path without silently replacing the
accepted demo checkpoint. Added exact-count/mapping validation, atomic unified-data
preparation, interrupt-safe training resume, held-out evaluation/gate orchestration,
a PowerShell background launcher, explicit offline candidate selection and VS Code
tasks. Unified v4 has 11,351/1,971/2,956 train/valid/test images; all 8,629 converted
VisDrone images are present, all 16,278 image-label pairs match and 158,890 boxes pass
schema validation. `pedestrian`/`people` map to `person_candidate`; no medical or
liveness label is inferred. Dataset integration is COMPLETE. Candidate training,
evaluation and promotion remain separate states. Runtime default remains COCO
YOLO11n. Exact VisDrone2019 deployment reuse terms, project-owned OAK-D locked
evaluation and edge export remain gates. Report:
docs/phase_reports/FULL_VISDRONE_INTEGRATION.md.

2026-10-07 — D-054 — PROJECT-DECISION: improve rejected v8 through a measured
COCO recovery curriculum, not by weakening promotion thresholds. Root cause is
source imbalance: full-v8 train annotations contained 106,396 VisDrone boxes versus
4,730 COCO and 2,433 Fallen. Recovery v5 uses fourfold in-train COCO replay, all
Fallen train images and a deterministic 6,000-box aerial reminder; early backbone
layers are frozen and later layers use low-rate AdamW. The unchanged full 2,956-image
test and original 10 gates remain authoritative. Dataset/test-index/schema audits,
9 focused tests, 179 application tests and pip check pass. Training/evaluation are
running; default model remains unchanged. Report:
docs/phase_reports/DETECTOR_PRECISION_RECOVERY.md.

2026-10-07 — D-055 — PROJECT-DECISION: promote a two-model detector fusion after
validation-only calibration and one unchanged held-out test. COCO YOLO11n remains
primary; the v7 posture/aerial checkpoint is a supplemental specialist. Fusion
preserves the caller's primary confidence, removes boxes overlapping at IoU >= .50,
and admits specialist-only general >= .70, horizontal >= .25 or small-target >= .20
evidence. Held-out test: COCO precision/recall .7771/.7214, Fallen recall .9222,
lying .9318, sitting .9375, standing 1.0, VisDrone .1217, small-person .2371,
large-person .9220 and p95 40.90 ms in that run. The unchanged public-data gate
passed 10/10. Offline real-YOLO scenes and all motion variants preserved exact
counts 1/2/4/6/4/6. Full suite: 192 tests and 36 subtests passed. Phone/USB/offline
launchers use hybrid by default with `-BaselineDetector` rollback. This promotes
the laptop demo detector, not medical classification, universal detection, field
identity reliability or final OAK-D deployment. Report:
docs/phase_reports/HYBRID_PERSON_DETECTOR_PROMOTION.md.

2026-10-07 — D-056 — BUG-FIX: `max_batch=2` in selective OSNet inference was a
GPU workload bound, but stable per-frame track ordering could repeatedly select
the same first tracks and starve later people. Re-ID now selects usable unresolved
tracks by least-recently-sampled time, preserving batch two while fairly rotating
through a crowd. An eight-person integrated regression creates and persists exact
S1-S8 records, then maps eight new temporary return tracks back to those records;
unique remains 8 and duplicate events become 8. SQLite integrity is ok. The full
post-code suite passed 193 tests plus 36 subtests; pip check is clean. Configured
gallery capacity remains 64, not two. This is software scaling evidence, not field
Re-ID accuracy or medical survivor verification. Report:
docs/phase_reports/MULTI_PERSON_IDENTITY_SCALING.md.
