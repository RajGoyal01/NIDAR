# Validated Hybrid Person Detector Promotion

Date: 2026-10-07  
Status: **public-data engineering gate passed; integrated in demo launchers**  
Field/OAK-D acceptance: **OPEN**

## 1. Objective

The original COCO YOLO11n was reliable for ordinary upright people, but it missed
many lying bodies, small aerial people and unusual poses. A posture-specialised
model detected more of those cases but was not safe as a total replacement because
earlier candidates reduced normal-person precision or recall.

The selected solution is a **hybrid detector**:

1. the general COCO YOLO11n remains the primary detector;
2. the v7 posture/aerial candidate acts as a specialist;
3. overlapping boxes are merged;
4. a specialist-only box is added only when validation-selected confidence,
   horizontal-posture or small-target rules accept it.

This is like using a general doctor first and asking a specialist for a second
opinion only on cases the general doctor may have missed. Both outputs still mean
only `person_candidate`: visible human/body evidence. They do not predict death,
injury, consciousness or medical survivor status.

## 2. Why this approach was selected

- Replacing COCO with the custom model was rejected by previous gates because it
  regressed ordinary COCO detection and added false positives.
- Lowering one model's confidence globally was rejected because it increases weak
  duplicate boxes everywhere.
- Another blind epoch run was rejected because v8/v9 already showed that more
  epochs alone do not solve source imbalance.
- Validated hybrid fusion was selected because it preserves the known general
  model and adds domain evidence under explicit, testable rules.

No training set or held-out test labels were used to choose runtime thresholds.
The policy was selected on `valid`; the unchanged `test` split was evaluated once.

## 3. Runtime data flow

```text
phone / USB / replay frame
          |
          +--> general YOLO11n ----------+
          |                               |
          +--> posture/aerial YOLO11n ----+--> IoU de-duplication
                                                  |
                                                  +--> qualified person boxes
                                                          |
                                                          +--> BoT-SORT
                                                               -> temporal verification
                                                               -> selective Re-ID
                                                               -> survivor manager / display
```

`IoU` means intersection-over-union: a number describing how much two rectangles
overlap. If the general model and specialist cover the same person, fusion keeps
one box rather than counting two.

The detector has no hard limit of six. Ultralytics is configured with `max_det=100`.
Actual capacity still depends on resolution, occlusion, target size and GPU speed.

## 4. Files and responsibilities

- `nidar_survivor_demo/hybrid_detector.py`: validates fusion configuration, runs
  both models, removes overlapping duplicates and returns normal detector output.
- `nidar_survivor_demo/settings/hybrid_detector.json`: reviewed thresholds outside
  implementation code.
- `nidar_survivor_demo/calibrate_hybrid_detector.py`: validation-only selection and
  held-out evaluation by source, pose, size, false positives and latency.
- `models/nidar-person-posture-specialist-yolo11n.pt`: deployed v7 specialist;
  SHA-256 `5E62B5594FEDD06B9E679CBD069D08565FFCEA24802BB6A179D00513119A3B88`.
- `nidar_survivor_demo/config.py` and `main.py`: validated CLI inputs and runtime
  detector selection.
- `Start-PhoneDemo.ps1`, `Start-USBCameraDemo.ps1`, `Start-OfflineDemo.ps1`: use
  hybrid detection by default; `-BaselineDetector` is the rollback/comparison.
- `nidar_survivor_demo/tests/test_final_dataset.py`: fusion/config regressions.

## 5. Selected configuration

```json
{
  "primary_confidence": 0.2,
  "general_confidence": 0.7,
  "posture_confidence": 0.25,
  "horizontal_aspect": 1.0,
  "small_confidence": 0.2,
  "small_area_ratio": 0.006,
  "dedupe_iou": 0.5
}
```

The caller's primary threshold is preserved. The offline showcase uses 0.25,
while BoT-SORT deliberately uses 0.10 so it can recover weak boxes over time.
Specialist thresholds are not silently weakened.

## 6. Held-out test results

Dataset: unchanged `nidar_person_v4_full_visdrone/test`, 2,956 images.

| Metric | COCO baseline | Promoted hybrid |
|---|---:|---:|
| COCO recall | 71.86% | 72.14% |
| COCO precision | 81.92% | 77.71% |
| COCO negative images with a false box | 16 | 30 |
| Fallen-person recall | 60.89% | 92.22% |
| Lying recall | 40.91% | 93.18% |
| Sitting recall | diagnostic source value not used here | 93.75% |
| Standing recall | diagnostic source value not used here | 100.00% |
| VisDrone/aerial recall | 4.42% | 12.17% |
| Small-person recall | 6.78% | 23.71% |
| Large-person recall | 80.29% | 92.20% |
| Inference p95 | 60.99 ms | 40.90 ms in this test run |

All **10/10** unchanged promotion checks passed. The latency comparison was measured
in separate runs and is not a glass-to-glass camera-latency guarantee. Public data
is not a substitute for project-owned drone footage.

Evidence: `logs/hybrid-v1-calibration.json`, `logs/hybrid-v1-test.json`, and
`logs/hybrid-v1-gate.json`.

## 7. Offline demo regression and repaired error

The actual YOLO showcase detected exact scene counts `1, 2, 4, 6, 4, 6`.
Seven simulated motion positions per scene also preserved the same counts.

The first hybrid run exposed one borderline seventh box in a six-person scene.
The cause was that the wrapper changed the requested primary confidence from 0.25
to 0.20. The wrapper now honours the caller's threshold exactly, removing the weak
duplicate. A similar error can be recognised by inspecting raw overlapping boxes
and looking for a second box close to the confidence boundary.

## 8. Automated validation

- focused detector/config/launcher tests: 41 passed + 14 subtests;
- complete project suite: **192 passed + 36 subtests**;
- offline hybrid showcase: all six base counts and all 42 motion-position counts
  matched expected values;
- PowerShell phone, USB and offline launchers parsed successfully;
- strict hybrid promotion gate: **10/10 pass**.

One existing Starlette/httpx deprecation warning remains; it is unrelated to ML
results and did not fail tests.

Runtime used Python 3.13.7, PyTorch 2.11.0+cu128, Ultralytics 8.4.163,
OpenCV 4.13.0 and NumPy 2.2.6 on the RTX 3050 laptop GPU.

## 9. Exact Windows PowerShell commands

Phone camera:

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Manage
```

USB camera:

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\Start-USBCameraDemo.ps1 -ListCameras
.\Start-USBCameraDemo.ps1 -CameraIndex 0 -Mode Full
```

Phone-free professor demo:

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\Start-OfflineDemo.ps1
```

COCO-only comparison:

```powershell
.\Start-OfflineDemo.ps1 -BaselineDetector
```

## 10. What happens when the code runs

1. The launcher activates the local Python environment.
2. Both local YOLO files load on CUDA when available.
3. Each fresh frame goes to both detectors; no internet is required.
4. Fusion preserves requested primary detections and checks specialist-only boxes.
5. Boxes flow to the existing tracker, verifier, Re-ID and database.
6. Temporary tracker IDs remain separate from persistent survivor IDs.
7. A single frame still cannot immediately create a unique survivor record.

## 11. Acceptance and remaining limits

Accepted for the current laptop demonstration:

- gate-passing improvement for standing/sitting/lying, fallen, aerial and small
  visible-person evidence;
- demonstrated detection of at least six people with no coded two/six-person cap;
- compatibility with phone, USB and offline launchers;
- explicit baseline rollback.

Still open for the final drone:

- project-owned OAK-D footage across rooms, heights, blur, occlusion and lighting;
- exact false-positive rate per mission minute;
- OAK/Pi export and on-device latency/thermal tests;
- spatial duplicate suppression using depth plus SLAM;
- real multi-person re-entry ground truth.

No RGB model can guarantee detection in every blur/occlusion/lighting condition.
The measured public-data coverage improved substantially and passed the engineering
gate; that is not a claim that misses are impossible.

## 12. Professor/viva questions and answers

**Why use two models?** The primary protects ordinary-person performance. The
specialist adds evidence for poses and scales the primary often misses.

**Can one person become two boxes?** IoU de-duplication merges overlapping model
evidence. Temporal verification also prevents one weak frame from immediately
becoming a survivor record.

**Does lying mean dead?** No. It means only visible body evidence was detected.
Health or life status cannot be inferred safely from RGB posture.

**Is the model limited to six people?** No. `max_det=100`; six is the largest
checked professor-demo scene so far.

**What remains before drone deployment?** OAK-D-specific data/evaluation, edge
export, thermal/latency validation and depth/SLAM spatial identity fusion.

## 13. What you learned

- Training loss alone cannot decide whether a model is safe to deploy.
- Validation chooses thresholds; a locked test set measures generalisation.
- A hybrid can improve domain coverage without replacing a strong general model.
- Confidence, overlap and temporal evidence solve different problems.
- Detection counts bodies in a frame; Re-ID/database logic estimates whether a
  returning observation belongs to an existing person.
