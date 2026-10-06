# NIDAR AirMouse — ML Architecture

## 1. Architecture goal

The perception system must answer four different questions without mixing them:

1. Is a person visible now?
2. Which current detections belong to the same short-lived track?
3. Is a new/reappearing track actually a person seen earlier?
4. Where is that survivor in the physical map?

## 2. Current demo data flow

```text
Phone stream (IP Webcam) OR Windows camera device (USB webcam)
  -> latest-frame buffer
  -> frame timestamp and health checks
  -> lightweight YOLO, person class only
  -> BoT-SORT
  -> temporal verifier
  -> selective Re-ID embedding
  -> survivor manager
  -> annotated frame + metrics + event log
  -> dashboard
```

Only the capture adapter changes between Wi-Fi and USB. Both inputs enter the
same one-slot latest-frame buffer, so detection, tracking, verification, Re-ID,
persistent IDs, thresholds and annotations are identical. USB mode does not
create a second detector and does not alter the final-drone architecture.

### Latest-frame buffer

The capture thread continuously reads from the phone but stores only the newest valid frame. This is like replacing the photograph on a notice board: the consumer always sees the newest picture instead of reading an ever-growing stack of old pictures.

### Detector

A lightweight pretrained YOLO model produces person bounding boxes and confidence values. Only the person class is required for the baseline. Start pretrained; fine-tune only after collecting baseline failure data.

2026-09-27 Phase 2 project decision: YOLO11n COCO weights, class 0 only,
confidence 0.35, NMS IoU 0.45, fixed square 640 letterboxed input, CUDA FP16.
512 and CPU/FP32 remain configurable alternatives. Capture runs independently;
the inference/display loop consumes the newest frame without an inference queue.
Annotations retain their source frame. Results older than the stale budget or
from a previous connection are hidden. Model faults stop the app safely.
No tracking or cumulative survivor count is implemented in Phase 2.

### Tracker

BoT-SORT associates boxes over nearby frames and creates temporary `track_id` values. A tracker ID is a short-term handle, not a permanent identity.

2026-09-27 Phase 3 implemented: separate BoT-SORT adapter, sparseOptFlow GMC at
downscale 2, Re-ID disabled, pinned lap 0.5.12. Config lives in
`nidar_survivor_demo/settings/botsort.json`. Tracking uses .10 detector confidence,
.35 high/.10 low/.40 new-track thresholds, 30 processed-frame lost buffer.
Empty observations still age tracks. Camera outage, frame-shape change or >1s
frame gap clears association; `E<epoch>:T<id>` labels prevent accidental ID aliasing
across resets within a run. These IDs do not persist across application restarts.
That Phase 3 increment introduced no survivor records/counting; Phase 4 is below.

### Temporal verifier

Implemented 2026-09-27 in `verification.py`, settings `settings/verification.json`.
The last five **processed fresh frames**, not capture frames or preview redraws,
form the rolling window per epoch-qualified temporary ID. Three good hits plus a
good current observation give CONFIRMED. Good means score >= .35 and clipped box
area/frame area >= .0001 (about 92 pixels at 720p); this is not a blur/pose or
medical-quality test. Missing frames append false and never render a lost track.
Low-quality or missing current evidence revokes confirmation; a return may confirm
again if sufficient recent evidence remains. Five consecutive missing updates
expire history. Tracker epoch change, >1s gap and camera/result loss clear it.
The application checks freshness after inference before adding evidence.
State and processing-time telemetry are exposed; there is no persistent counter,
database, Re-ID embedding, injury assessment or liveness inference.

Use a configurable rule such as “detected in at least 3 of the last 5 frames above a minimum confidence”. The numbers are initial tuning values, not final facts. States:

```text
DETECTED -> VERIFYING -> CONFIRMED
                    \-> REJECTED / EXPIRED
```

### Re-ID encoder

D-047 exception to D-046 below: an EMPTY gallery may bootstrap the first partial
reference after existing temporal/quality/consistency gates. Assignment-time
guard prevents a second partial enrollment. Later partial views are recognition
only. See FIRST_PARTIAL_ENROLLMENT.md; this does not certify all-pose identity.

Latest D-046 supersedes the fully frozen three-vector description below: immutable
initial anchors plus bounded strongly anchor-matched full-view additions (max12).
Edge-touching views may match existing references at stricter .90 threshold, never
enroll or update galleries. Other quality/conflict checks remain; variable-length
gallery persists in existing SQLite metadata with a new configuration signature.
See docs/phase_reports/IDENTITY_CORE_REPAIR.md. Live accuracy remains OPEN.

2026-09-27 Phase 5 implemented: author-hosted OSNet x0.25 MSMT17 checkpoint,
verified SHA-256, pinned MIT architecture, existing PyTorch CUDA FP32/CPU runtime.
`--reid` implies temporal verification. Quality-gated RGB 256x128 crops produce
normalized 512D embeddings only for new/reappearing/unresolved confirmed tracks.
Three spaced views are compared to a bounded RAM-only reference gallery R1/R2;
references are NOT persistent survivor records. Match floor .815 derives from
20-image COCO hard-colour-negative calibration, with a separate 20-image proxy
evaluation (0 negative pair threshold crossings). Augmented positives are not
true re-entry data. Novelty <.55 and margin >=.08 are conservative project choices.
Similar/ambiguous queries remain uncertain, including active-reference conflicts.
Gallery: up to 64 references, 3 frozen vectors each, 600s inactive TTL. No raw
camera crops/embeddings are stored to disk. Track reset retains references but
clears bindings; restart loses the entire gallery. This is a Phase 5 matcher;
Phase 6 now implements permanent survivor records, counting and duplicate decisions.
The RAM-only/TTL description above applies to standalone --reid. With --manage,
the gallery is retained for the mission and saved locally to SQLite (no raw
camera crops). The 64-reference cap remains; capacity/ambiguity stays uncertain.
Explicit resume validates schema, encoder checksum and matching/manager config,
restores frozen gallery vectors and S records, but never old tracker bindings.
Three consecutive fresh confirmed manager observations are required to create
an S record. Existing accepted R associations reuse the corresponding S ID.
Counts, events and gallery commit together before published counts advance.
Temporary E:T, appearance R and mission-scoped persistent S IDs remain separate.
See PHASE_06_SURVIVOR_MANAGER.md for limits, event semantics and offline evidence.
See `docs/phase_reports/PHASE_05_APPEARANCE_REID.md` for evidence/limitations.

For a new or reappearing track, crop several good person images and create appearance embeddings. Compare normalised embeddings with cosine similarity. Do not run full Re-ID for every active track in every frame.

### Survivor Manager

This is the identity authority. It maps many temporary tracks to one persistent survivor:

```text
Track 4  ----\
Track 19 -----+--> Survivor S1
Track 31 ----/
```

Suggested record:

```text
SurvivorRecord
  survivor_id
  status
  current_track_id | null
  first_seen_at
  last_seen_at
  observations[]
  representative_embeddings[]
  best_crops[] (optional and privacy-controlled)
  latest_camera_confidence
  estimated_map_position | null
  grid_cell | null
  position_uncertainty | null
  identity_confidence
```

## 3. Identity and duplicate-suppression logic

Latest correction: the co-visible policy below is rejected after genuine sequential
arrivals were blocked. Managed novelty now needs three non-overlapping good-view
batches over >=1.5s, consistent with a fixed candidate anchor and dissimilar to
all references. No requirement that old people remain visible. Quality/ambiguity
resets evidence. Match thresholds unchanged; strong returns bypass novelty delay.
See `docs/phase_reports/SEQUENTIAL_IDENTITY_DATABASE_FIX.md`.

2026-09-27 live repair supersedes novelty behavior described above for managed
missions: low similarity alone cannot enroll while any old reference is not
quality-qualified and confirmed on a separate visible track. Keep such candidates
pending. Reject image-edge crops before embedding and similar enrollment proposals
within the same frame. Strong matching thresholds and frozen vectors unchanged.
Sequential new-person recall is intentionally reduced; this is conservative
containment, not a fully calibrated Re-ID solution. See IDENTITY_FALSE_SPLIT_FIX.md.

### Implemented Phase 7 presentation boundary

Authorized extension D-043: explicit confirmed POST /api/mission/reset is the sole
write control, guarded by same-origin/token/current mission. Processing thread owns
rotation, old SQLite is retained, all identity state replaced. Refresh stays GET-only.
See RESET_DEMO.md. Historical read-only-only description follows.

The existing pipeline publishes committed manager results into a bounded,
thread-safe display mailbox. FastAPI/Uvicorn serves fixed read-only routes on
127.0.0.1:8765 (configurable port). The browser renders the latest JPEG (publisher
cap 10 FPS), count estimates, health, records and at most 100 recent events.
No browser operation changes mission identities or ML thresholds. Unknown current
occupancy is preserved on stale/outage/disconnect; old frames are not shown as live.
Modes explicitly distinguish live input, local video, oracle-track offline replay
and historical archive. Export omits embeddings and detailed observation arrays.
See docs/phase_reports/PHASE_07_DASHBOARD.md for API/security/validation details.

For every newly confirmed track:

1. Reject poor crops: too small, blurred, mostly occluded, or clipped at the frame edge.
2. Create several embeddings from different good frames.
3. Compare against existing survivor galleries.
4. If a strong match exists and ambiguity is low, attach the track to that survivor.
5. If the evidence is uncertain, keep the track in a pending state; do not immediately create a unique survivor.
6. If repeated evidence supports novelty, create the next survivor ID.
7. Record why the match/new decision was made.

The similarity threshold must be calibrated using project footage. It must not be copied blindly from an example.

### Final-drone fusion

Appearance alone is fragile when people wear similar clothing or are seen from different angles. The final score should combine independent evidence:

```text
identity score = appearance evidence
               + spatial/map consistency
               + time continuity
               + optional pose/colour/shape cues
```

Spatial gating should consider survivor/map uncertainty, not only a hard distance. Two visually similar people in clearly different rooms should remain different survivors. A person revisited near the same stable map coordinate should be treated as a likely duplicate even if appearance quality is poor.

## 4. Final drone data flow

```text
OAK-D RGB + stereo depth + IMU
  -> detections / image features / depth
  -> timestamp synchronisation
  -> Raspberry Pi ROS 2 nodes
       -> visual-inertial odometry / SLAM
       -> camera-to-map transform
       -> 3D survivor observation
       -> 2D/grid projection
       -> survivor identity fusion
       -> map marker and mission database
  -> GCS live map, feed and mission status
```

If YOLO is deployed on the OAK accelerator, the Pi receives compact detections plus depth rather than performing all vision work itself. If model compatibility or accuracy blocks this, inference can temporarily run on the Pi with an optimised runtime. Benchmark before deciding.

## 5. Dataset and fine-tuning plan

### Stage A — baseline without training

- Use a pretrained person model.
- Run recorded and live tests.
- Save only approved diagnostic examples.
- Build a failure taxonomy: missed lying person, false dummy, motion blur, dark room, partial body, etc.

### Stage B — dataset specification

- Define `person/survivor` labelling policy.
- Decide whether real people and dummies share one class or need separate handling.
- Include camera angle, height, distance, light, pose, occlusion, debris, and blur diversity.
- Add negative images containing furniture, mannequins not used as survivors, posters, and human-like shapes.
- Document consent, licence, source URL, capture date, and permitted use.

### Stage C — collection and labelling

- Prefer project-owned staged footage and properly licensed public datasets.
- Sample frames to avoid thousands of near-duplicates.
- Use consistent bounding-box rules.
- Perform a second-person label audit on a representative subset.

### Stage D — leakage-safe split

- Split by room, video, subject, or recording session.
- Never put adjacent frames of the same event into both train and validation/test sets.
- Keep a final hidden indoor-drone-style test set.

### Stage E — fine-tune and evaluate

- Compare against the unchanged pretrained baseline.
- Measure precision, recall, mAP, failure groups, model latency, and end-to-end FPS.
- Promote a model only if it improves mission-relevant behaviour without breaking the latency budget.

### Stage F — edge deployment

- Export to the selected OAK/Pi runtime.
- Validate numerical accuracy after export/quantisation.
- Re-measure performance on the actual final hardware.

## 6. Metrics

### Detection

- precision and recall;
- false positives per minute;
- miss rate by pose/light/occlusion;
- inference latency.

### Tracking and identity

- track continuity and ID switches;
- re-entry match accuracy;
- false merge: two people incorrectly become one survivor;
- false split: one person becomes multiple survivors;
- duplicates prevented;
- time to confirm a survivor.

### System

- capture FPS, inference FPS, dashboard FPS;
- end-to-end glass-to-glass latency;
- CPU, RAM, GPU/accelerator, temperature, and power use;
- recovery time after stream/module failure.

## 7. Main risks and controls

| Risk | Control |
|---|---|
| Similar clothing causes false merge | combine multiple embeddings with spatial evidence and ambiguity checks |
| Fast phone/drone motion causes blur | shorter exposure where possible, good light, blur-quality gate, tracker tuning |
| Wireless stream builds delay | latest-frame-only capture and bounded queues |
| Re-ID overloads the GPU | run only for new/reappearing tracks and cache galleries |
| SLAM drift moves survivor markers | store uncertainty, loop-closure updates, and observation history |
| Map frame and camera frame disagree | calibrated transforms and synchronised timestamps |
| One false frame creates a survivor | temporal verification and pending state |
| Dataset performs well only in one room | scene-separated splits and varied evaluation footage |

## 8. Decisions still open

- Final-drone YOLO model/export; laptop Phase 2 baseline is YOLO11n (`yolo11n.pt`).
- Final-drone Re-ID encoder/export; laptop baseline is OSNet x0.25 MSMT17 (MIT).
- Broad local-domain cosine calibration; Phase 5 proxy baseline and frozen gallery selected.
- Exact ROS 2 distribution and SLAM/VIO package.
- Whether primary inference runs on OAK-D or Raspberry Pi.
- Map representation and survivor-position uncertainty format.

## 9. Posture-domain dataset prepared (2026-09-27)

Roboflow Fallen Person v2 is locally validated and available for future fine-tuning.
Its original `fallen`, `lying`, `sitting`, `standing` boxes are mapped into one
detector class, `person_candidate`. This is intentionally a perception label, not a
medical-state label. The specialized detector will still feed the same BoT-SORT,
verification, Re-ID and Survivor Manager interfaces. Fine-tuning and model promotion
require comparison against COCO and leakage-safe local footage first.

## 10. NIDAR one-class detector dataset (2026-09-30)

The detector's final semantic is `person_candidate`: any visible human/body-like
evidence that should be tracked and verified. It covers posture and viewpoint but
does not label survivor health, injury, responsiveness, life or death.

`datasets/nidar_person_v1` combines:

- Fallen Person v2 for fallen, lying, sitting and standing bodies;
- COCO person positives and empty/person-free hard negatives;
- VisDrone2019-DET `pedestrian` and `people` for aerial, small, crowded and
  non-upright views.

Fallen coarse source groups are reassigned as indivisible groups, COCO IDs use
disjoint deterministic buckets, and VisDrone official splits are preserved. The
training candidate restarts from COCO YOLO11n rather than the over-specialised
fallen checkpoint. Promotion requires held-out metrics by source and target size,
COCO regression protection, latency testing and later OAK-D export validation.
Project-owned indoor drone/OAK-D footage remains a mandatory domain gap.

### Training outcome

Three YOLO11n candidates were measured rather than promoted on training loss alone.
The final v7 curriculum candidate substantially improved pose coverage (fallen
recall .949, lying .932) and remained fast (38.97 ms p95), but its COCO recall and
precision fell to .469 and .685. It also triggered false detections in 43 COCO
negative images. The automated public-data gate passed only 7/10 checks, therefore
the production/demo default remains the original COCO checkpoint. This is a safe
failure: research weights are preserved, but weak generalisation is not hidden.

Future detector work must use project-owned OAK-D scenes and a fresh locked test.
Detection continues to feed BoT-SORT, temporal verification and persistent identity;
depth/SLAM remains required for final spatial duplicate suppression.

## 11. Complete VisDrone research path (2026-10-06)

`datasets/nidar_person_v4_full_visdrone` removes the earlier experimental VisDrone
train/validation limits. It contains all 8,629 converted DET images while retaining
official boundaries and the one-class `person_candidate` interface. The complete
mixture contains 16,278 images and 158,890 boxes. `full_visdrone_pipeline.py` owns
exact source validation, atomic preparation, candidate training, held-out evaluation
and public regression gating. Interrupted runs can resume from `last.pt`.

This changes the research path, not the runtime architecture. The current demo model
remains COCO YOLO11n until the candidate passes public regression, exact licence
review, project-owned OAK-D locked evaluation and edge-export gates.

## 12. Promoted laptop hybrid detector (2026-10-07)

The public-data engineering gate now passes 10/10 for a validation-calibrated
hybrid: COCO YOLO11n remains the primary and the v7 posture/aerial model is a
specialist. Fusion de-duplicates overlapping boxes and accepts specialist-only
evidence under external JSON thresholds. The output remains one-class
`person_candidate`, so BoT-SORT, temporal verification, selective Re-ID and the
survivor manager require no architectural change.

This replaces the laptop launchers' single-model default, not the final drone
acceptance process. OAK-D footage, exported-runtime accuracy/latency and spatial
depth/SLAM evidence remain mandatory. Evidence and commands are in
`docs/phase_reports/HYBRID_PERSON_DETECTOR_PROMOTION.md`.

## 13. Crowded-scene Re-ID scheduling (2026-10-07)

Selective Re-ID retains `max_batch=2` to bound per-frame GPU work, but unresolved
tracks are now scheduled by least-recently-sampled time. Therefore batch size is a
latency/VRAM control, not an identity-count ceiling. Automated integrated evidence
allocates and persists S1-S8, then recovers all eight return tracks without count
inflation. Mission gallery capacity remains 64. See
`docs/phase_reports/MULTI_PERSON_IDENTITY_SCALING.md`.
