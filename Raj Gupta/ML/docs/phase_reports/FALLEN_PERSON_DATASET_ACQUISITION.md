# Fallen-person dataset acquisition and preparation

**Date:** 2026-09-27  
**Status:** download, integrity validation, conversion and baseline audit complete  
**Phase gate:** prerequisite/domain-data work before Phase 9; Phase 9 has not started

## 1. Objective

Add a licensed, posture-diverse object-detection dataset for people who are fallen,
lying, sitting or standing. The existing COCO model is good at ordinary people but
can miss unusual body poses and rotated/sideways camera views.

This work does **not** claim that an RGB camera can decide whether a person is
alive, dead or injured. The safe label is `person_candidate`: a visible human body
that the later tracking, temporal verification and identity stages can process.

## 2. What was downloaded

Source project:
`https://universe.roboflow.com/fallen-people-data-set/fallen-person-uhif8/dataset/2`

The downloaded Roboflow YOLO11 archive is stored locally at:

```text
datasets/fallen_person/archives/fallen-person-roboflow-v2-yolov11.zip
```

Integrity evidence:

| Check | Result |
|---|---:|
| Archive bytes | 129,018,934 |
| SHA-256 | `2055abd525e8903047682cb8c55dd69bfb87376b87aadcefa4a2c86a2d29e517` |
| Images decoded | 2,876 / 2,876 |
| Image-label pairs | 2,876 / 2,876 |
| Corrupt images | 0 |
| Invalid YOLO labels | 0 |
| Total annotated bodies | 3,297 |

The source export states **CC BY 4.0**. It is a user-provided Roboflow dataset,
so its upstream image provenance must still be audited before public redistribution
or commercial use. The temporary signed storage URL was deliberately not stored.

## 3. Original data and safe derived view

The untouched extraction is at:

```text
datasets/fallen_person/roboflow_v2/
```

Original labels:

| Class | Boxes |
|---|---:|
| fallen | 1,830 |
| lying | 233 |
| sitting | 131 |
| standing | 1,103 |

The live NIDAR pipeline first needs one body detector, not a medical classifier.
Therefore all four pose labels are mapped to class `0 person_candidate` in:

```text
datasets/fallen_person/person_candidate/
```

This preserves the source files and avoids treating `fallen` as proof of injury or
death. The derived images are hard links where Windows permits it, so preparation
does not waste another 129 MB; copying is used as a safe fallback.

## 4. Data flow

```text
signed Roboflow ZIP
  -> SHA-256 and size check
  -> untouched four-class extraction
  -> decode every image + validate every box
  -> map fallen/lying/sitting/standing to person_candidate
  -> one-class YOLO training view + manifest
  -> baseline evaluation with unchanged COCO YOLO11n
```

The future runtime path will remain:

```text
phone frame -> detector -> BoT-SORT -> temporal verification
            -> Re-ID -> survivor manager -> dashboard
```

Better detection supplies boxes to the later stages. It does not by itself fix
identity matching or unique counting; those depend on Re-ID and the survivor
manager too.

## 5. Files and responsibilities

- `nidar_survivor_demo/prepare_fallen_person.py`: repeatable validator and safe
  four-class-to-one-class converter.
- `datasets/fallen_person/roboflow_v2/`: original extracted export, unchanged.
- `datasets/fallen_person/person_candidate/data.yaml`: Ultralytics dataset entry.
- `datasets/fallen_person/person_candidate/manifest.json`: checksums, counts,
  mappings, licence claim and leakage warning.
- `docs/metrics/FALLEN_PERSON_COCO_BASELINE.json`: unchanged-model baseline.
- `.gitignore`: already excludes datasets, weights, caches and generated runs.

## 6. Dataset split warning

The Roboflow export has 2,000 training, 576 validation and 300 test images. Coarse
filename groups such as `split1`, `split2`, `split3`, `split4`, `split5`, `split9`
occur in more than one supplied split. This suggests related frames may cross split
boundaries. That is called **data leakage**: a test can become easier because the
model has seen near-related scenes during training.

Consequently:

- supplied validation/test metrics are useful engineering diagnostics;
- they are not independent rescue-field accuracy;
- a later project-owned test set must be split by person, room and recording
  session, not by random frames.

## 7. Baseline before fine-tuning

The unchanged COCO-pretrained `models/yolo11n.pt` was evaluated on the 300-image
derived test split using the RTX 3050, 640 input and batch 8:

| Metric | Result |
|---|---:|
| Precision | 0.3639 |
| Recall | 0.6144 |
| mAP50 | 0.3729 |
| mAP50-95 | 0.1852 |
| Inference | 34.67 ms/image |

Recall means “of all labelled bodies, what fraction did the model find?” A recall
of 0.6144 confirms a meaningful posture-domain gap. It does not mean exactly 38.6%
of live people will be missed because this test split is not independent field data.

## 8. Exact Windows commands

Re-run the preparation and integrity audit:

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\.venv\Scripts\python.exe -m nidar_survivor_demo.prepare_fallen_person
```

Inspect the generated manifest:

```powershell
Get-Content .\datasets\fallen_person\person_candidate\manifest.json
```

Count the prepared pairs:

```powershell
(Get-ChildItem .\datasets\fallen_person\person_candidate\images -File -Recurse).Count
(Get-ChildItem .\datasets\fallen_person\person_candidate\labels -File -Recurse).Count
```

Expected result for both commands: `2876`.

## 9. What is and is not complete

Complete:

- authorized archive downloaded;
- archive hash and byte size recorded;
- all images decoded;
- all YOLO labels validated;
- original four-class dataset retained;
- safe one-class detector view generated;
- dataset and generated outputs excluded from Git;
- unchanged detector baseline measured.

Not performed in this acquisition task:

- model fine-tuning;
- promotion of a custom model into the live dashboard;
- leakage-safe local field evaluation;
- medical/death/injury classification;
- Phase 9 stress testing.

## 10. Final validation evidence

Automated validation completed on 2026-09-27:

- preparation script ran twice successfully, proving repeatability;
- SHA-256 recomputation matched the recorded archive fingerprint;
- all 2,876 images decoded and all 2,876 label files parsed;
- Python package compilation passed;
- full project regression suite: **168 tests passed**;
- `pip check`: **No broken requirements found**;
- actual RTX 3050 baseline evaluation completed on all 300 test images.

No phone/hardware test was required for downloading and validating the dataset.
No live custom-model test can be claimed because a model has not been fine-tuned or
promoted yet.

## 11. Next controlled step

Fine-tune a new YOLO11n candidate from the existing COCO weights, compare it with
the unchanged baseline on both fallen-person data and the retained COCO person
evaluation subset, then run phone tests for standing, sitting, lying, rotated,
blurred, occluded and hard-negative scenes. Promote the model only when recall
improves without unacceptable false positives or latency.

## 12. Viva questions and answers

**Why not keep four posture classes?**  
The current mission needs to detect a human body first. Four weak pose labels could
make the detector confuse posture with medical status. One `person_candidate` class
gives the downstream tracker one consistent input.

**Does `fallen` mean survivor or dead body?**  
No. A still RGB image cannot establish life status or injury. It only shows a
person-shaped/body candidate.

**Why retain COCO pretraining?**  
COCO gives broad general-person features. Fine-tuning can adapt those features to
unusual poses with much less data than training from zero.

**Why hash the archive?**  
A SHA-256 hash is like a digital fingerprint. It lets another run verify that it
uses the exact same bytes.

**Why can the supplied test accuracy be optimistic?**  
Related source groups appear across train and test. The model may recognise scene
details instead of demonstrating truly independent generalisation.

**Will this alone repair unique counting?**  
No. It can reduce missed body boxes. Unique counts additionally require stable
tracking, temporal evidence, Re-ID and correct survivor-manager decisions.
