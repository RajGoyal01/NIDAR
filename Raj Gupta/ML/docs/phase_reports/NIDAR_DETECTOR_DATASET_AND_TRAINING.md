# NIDAR Person-Candidate Dataset and Detector Training

Status: **TRAINING COMPLETE; CANDIDATE REJECTED, NOT PROMOTED (2026-09-30)**  
Decision: D-050  
Scope: detector data and candidate weights; not flight, SLAM, medical status or field certification.

## 1. Objective

Improve the detector so it can find a visible person or body from more NIDAR-like
views: standing, sitting, crouching, lying, partly hidden, small, aerial, side-view
and low-light-like conditions. The output class is `person_candidate`.

The model does **not** predict survivor health, injury, responsiveness, life or
death. A box means “human/body evidence worth tracking and operator review.”

## 2. Why one dataset was not enough

- COCO protects general standing-person performance and provides difficult empty
  scenes, but it under-represented lying/fallen and drone-top views.
- Fallen Person v2 adds posture diversity, but a specialist trained only on it
  catastrophically forgot normal people: COCO guard recall fell from 0.615 to 0.189.
- VisDrone adds real drone viewpoints, tiny targets, crowding and occlusion, but
  it is mostly outdoor and is not a substitute for the final indoor arena.

The selected approach is therefore a **mixture**, not a claim that one public
dataset perfectly matches NIDAR.

## 3. Dataset architecture

```text
Fallen Person v2 ─┐
COCO person/empty ├─> one class: person_candidate ─> YOLO11n fine-tuning
VisDrone people ──┘
```

Three reproducible mixtures were used. The final curriculum dataset deliberately
reduces dense VisDrone dominance while keeping the full regression test unchanged.

| Curriculum split | Images | Person boxes | Purpose |
|---|---:|---:|---|
| train | 5,130 | 11,874 | short label-balanced fine-tuning |
| validation | 1,523 | 4,602 | choose the best curriculum epoch |
| full regression test | 2,956 | 29,282 | same broad comparison used for baseline/v6/v7 |

The original v1 mixture remains available with 8,880/1,971/2,956 images. V2 limits
VisDrone training to 1,000 images; v3 curriculum limits it to 250 train and 100
validation images. Negative images teach the detector that furniture, vehicles and
clutter should not automatically become people.

## 4. Leakage protection

“Leakage” means the model sees almost the same event during training and testing,
making test scores look unrealistically high.

- Fallen images are grouped by their coarse source family (`img`, `split1`, etc.).
  An entire group goes to only one split.
- COCO image IDs use deterministic, disjoint SHA-256 buckets.
- VisDrone official train/validation/test-dev boundaries remain unchanged.

This is stronger than the old Roboflow supplied split, but it is still not a final
field test because public-source visual styles may overlap internally.

## 5. Source and licence ledger

| Source | Role | Licence/evidence status |
|---|---|---|
| Fallen Person v2 | fallen/lying/sitting/standing | included dataset card says CC BY 4.0 |
| COCO 2017 val | normal people and hard negatives | COCO annotations; image-specific upstream terms remain applicable |
| VisDrone2019-DET | aerial/small/occluded people | citation/source verified; reusable-data licence still REVIEW-REQUIRED |

VisDrone is therefore used only for a research candidate until the team verifies
its terms for the intended competition/deployment. The archive URLs, byte sizes and
SHA-256 hashes are in `datasets/visdrone_person/manifest.json`.

## 6. Important files

- `nidar_survivor_demo/prepare_visdrone_person.py` — downloads, hashes and converts
  only VisDrone pedestrian/person categories.
- `nidar_survivor_demo/prepare_nidar_person.py` — creates the unified one-class
  dataset with leakage controls and hard links.
- `nidar_survivor_demo/train_nidar_detector.py` — deterministic COCO-initialised
  YOLO11n candidate training.
- `nidar_survivor_demo/evaluate_nidar_detector.py` — held-out metrics by source,
  body size and fallen posture.
- `nidar_survivor_demo/gate_nidar_detector.py` — explicit, reproducible promotion
  checks; a failed check prevents automatic model replacement.
- `datasets/nidar_person_v1/manifest.json` — exact counts, grouping and known gaps.
- `datasets/nidar_person_v3_curriculum/manifest.json` — final curriculum counts.
- `runs/training/nidar-person-yolo11n-v7-curriculum/` — final candidate artefacts.
- `logs/nidar-person-yolo11n-v7-curriculum-test.json` — full measured report.
- `logs/nidar-person-yolo11n-v7-gate.json` — 7/10 rejection evidence.

## 7. Why YOLO11n and these settings

YOLO11n is the smallest selected YOLO11 detector. It keeps the model suitable for
the RTX laptop demonstration and gives a realistic future path to an OAK/Pi edge
runtime. Training uses 640-pixel input, FP16 automatic mixed precision and batch 16.
Rotation, translation, scale, horizontal/vertical flips and mosaic augmentation
simulate changing drone angle and framing. Augmentation creates variation; it does
not replace real final-camera data.

The run starts from `models/yolo11n.pt`, not the old fallen-only specialist. This
preserves useful COCO features and reduces catastrophic forgetting.

## 8. Exact Windows commands

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML

# Download/convert aerial data
.\.venv\Scripts\python.exe -m nidar_survivor_demo.prepare_visdrone_person

# Build the mixed leakage-aware dataset
.\.venv\Scripts\python.exe -m nidar_survivor_demo.prepare_nidar_person

# Train the first reviewed candidate
.\.venv\Scripts\python.exe -m nidar_survivor_demo.train_nidar_detector `
  --name nidar-person-yolo11n-v5 --epochs 20 --batch 16 --workers 2

# Build the label-balanced curriculum mixture
.\.venv\Scripts\python.exe -m nidar_survivor_demo.prepare_nidar_person `
  --output datasets\nidar_person_v3_curriculum `
  --visdrone-train-limit 250 --visdrone-valid-limit 100

# Short curriculum stage from the balanced v6 checkpoint
.\.venv\Scripts\python.exe -m nidar_survivor_demo.train_nidar_detector `
  --data datasets\nidar_person_v3_curriculum\data.yaml `
  --base-model runs\training\nidar-person-yolo11n-v6-balanced\weights\best-person-candidate.pt `
  --name nidar-person-yolo11n-v7-curriculum `
  --epochs 6 --batch 16 --workers 2

# Full regression evaluation after training
.\.venv\Scripts\python.exe -m nidar_survivor_demo.evaluate_nidar_detector `
  --model runs\training\nidar-person-yolo11n-v7-curriculum\weights\best-person-candidate.pt `
  --dataset datasets\nidar_person_v2 `
  --output logs\nidar-person-yolo11n-v7-curriculum-test.json

# Apply the promotion contract
.\.venv\Scripts\python.exe -m nidar_survivor_demo.gate_nidar_detector `
  --baseline logs\nidar-person-baseline-test.json `
  --candidate logs\nidar-person-yolo11n-v7-curriculum-test.json `
  --output logs\nidar-person-yolo11n-v7-gate.json
```

## 9. Promotion gates

The candidate must not replace `models/yolo11n.pt` merely because training ended.
It must demonstrate:

1. improved held-out fallen/lying and aerial-small-person recall;
2. no unacceptable regression on the fixed balanced COCO guard;
3. controlled false positives on empty/hard-negative images;
4. RTX 3050 p95 detector latency within the current real-time budget;
5. clean one-class schema (`0: person_candidate`);
6. full demo regression tests passing;
7. later ONNX/OpenVINO/OAK export numerical comparison;
8. later project-owned OAK-D indoor field test.

Until these gates pass, the current COCO checkpoint remains the safe default.

## 10. Current measured evidence

- Unified v1 file/label audit: 13,807 paired images/labels, invalid labels 0.
- Ultralytics dataset parser: one class, all paths valid.
- Batch-16 training remained within the RTX 3050 4 GB model allocation budget.
- V5 improved pose/aerial recall but over-specialised; it was not promoted.
- Balanced v6: fallen .940, lying .750, COCO recall .486; not promoted.
- Curriculum v7 checkpoint: 5,480,551 bytes, SHA-256
  `5e62b5594fedd06b9e679cbd069d08565ffcea24802bb6a179d005133119a3b88`.

| Full regression metric at confidence .20 | COCO baseline | v7 candidate |
|---|---:|---:|
| COCO precision | .819 | .685 |
| COCO recall | .719 | .469 |
| COCO negative images with false positives | 16 | 43 |
| Fallen-source precision | .723 | .943 |
| Fallen-source recall | .609 | .949 |
| Lying recall | .409 | .932 |
| Sitting recall | .938 | 1.000 |
| Standing recall | 1.000 | .929 |
| Aerial/VisDrone recall | .044 | .116 |
| Small-target recall | .068 | .227 |
| Large-target recall | .803 | .790 |
| p95 inference latency | 60.99 ms | 38.97 ms |

The explicit engineering gate passed 7/10. It failed general COCO recall, COCO
precision and negative-image false-positive limits, so the candidate was
**rejected and not copied** to `models/nidar-person-candidate-yolo11n.pt`.
The existing COCO model remains the runtime default.

Automated validation: 176 tests and 36 subtests passed in 117.88 seconds; one
third-party Starlette deprecation warning; `pip check` reported no broken
requirements. No phone or OAK-D hardware test was performed in this training task.

## 11. Known limitations and mandatory next data

The public mixture cannot honestly make the detector “perfect in every condition.”
The full regression set has now also influenced model-selection decisions, so a
new locked project-owned field set is required for the next unbiased acceptance.
Before final-drone acceptance, collect consented project-owned OAK-D footage with:

- exact camera height, tilt and lens;
- indoor rooms/corridors, debris and competition-like dummy;
- standing, sitting, crouching, lying, covered and partial people;
- slow/fast flight-like blur, low light and shadows;
- small/far targets and people touching frame edges;
- empty rooms, clothes piles, beds, posters, mannequins and furniture negatives;
- subject-, room- and recording-separated hidden test sessions.

Depth and SLAM/map position must still perform localisation and strengthen duplicate
suppression. Detector training alone cannot solve persistent unique counting.

## 12. Professor/viva questions

**Why not call the class `survivor`?**  An RGB image can show a person/body but
cannot prove medical survivor status. `person_candidate` is technically honest.

**Why merge pedestrian and non-upright people?**  Detection should maximise recall
for any visible person. Posture/status can be separate evidence; splitting classes
can make the detector miss uncertain poses.

**Why keep a held-out test set?**  It is like an exam the model never studies from.
Using it during tuning would make the final score unreliable.

**Does more data always improve accuracy?**  No. Wrong labels, duplicated frames,
domain imbalance or missing hard negatives can make performance worse. That is why
we use source-specific regression gates.

**Why was the new model not enabled after its lying recall improved?**  Detection
is multi-objective. It found lying people much better but missed too many ordinary
COCO people and increased false positives. A rescue detector must not silently
exchange one failure type for another.

**What remains for the real drone?**  OAK-D data, edge export, depth localisation,
SLAM coordinate fusion, hardware-in-loop testing and guarded field validation.

## 13. Complete VisDrone integration follow-up (2026-10-06)

The earlier v5/v6/v7 candidates did not all study the complete converted VisDrone
training set. This is now corrected through `nidar_person_v4_full_visdrone`: all
8,629 converted DET images are present across preserved official splits. The unified
dataset has 16,278 images, 158,890 boxes and zero invalid audited labels. A resumable
v8 train -> evaluate -> gate workflow is available and its runtime default promotion
is deliberately disabled. Full commands, status and teaching are in
`docs/phase_reports/FULL_VISDRONE_INTEGRATION.md`.
