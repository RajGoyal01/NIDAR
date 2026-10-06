# Full VisDrone Integration — Training Candidate Handoff

**Date:** 2026-10-06  
**Scope:** detector research path before Phase 9  
**Dataset integration:** COMPLETE  
**Candidate training/evaluation:** RUNNING / NOT YET ACCEPTED  
**Runtime model changed:** NO

## 1. Objective

Integrate every locally converted VisDrone2019-DET image into a reproducible NIDAR
training path. The goal is higher recall for small, distant and crowded people seen
from an elevated/drone-like viewpoint.

This does **not** turn an image into proof that a person is alive, dead or injured.
The detector class remains `person_candidate`: visible human/body evidence that the
rest of the pipeline can track and verify.

## 2. Beginner-friendly concepts

- **Pretrained model:** a model that already learned general visual features.
  Training starts from local COCO YOLO11n weights instead of learning from zero.
- **Fine-tuning:** continuing training on project-relevant examples. It is like
  teaching someone with general knowledge the special vocabulary of this project.
- **Annotation:** a box and class label describing where a person appears.
- **Hard negative:** an image with no person but confusing objects. It teaches the
  model when it should remain silent.
- **Held-out split:** images training cannot study. They act like an exam.
- **Promotion gate:** measurable conditions a candidate must pass before replacing
  the working model.

## 3. Dataset and class policy

The unified dataset is `datasets/nidar_person_v4_full_visdrone`.

| Split | Fallen | COCO | VisDrone | Total images | Boxes |
|---|---:|---:|---:|---:|---:|
| Train | 2,021 | 2,859 | 6,471 | 11,351 | 113,559 |
| Validation | 408 | 1,015 | 548 | 1,971 | 16,049 |
| Test | 447 | 899 | 1,610 | 2,956 | 29,282 |
| **Total** | **2,876** | **4,773** | **8,629** | **16,278** | **158,890** |

VisDrone source classes `pedestrian` and `people` map to class 0,
`person_candidate`. Vehicle classes do not become people. Their images remain useful
background/hard-negative evidence. Official train/val/test-dev boundaries are kept.

Independent audit result:

- 16,278 image files and 16,278 matching label files;
- 158,890 YOLO boxes;
- 3,454 deliberately empty label files (hard negatives);
- zero invalid class IDs, coordinates or box dimensions;
- all 8,629 converted VisDrone DET images included.

The exact VisDrone2019 reuse terms remain `REVIEW-REQUIRED` for intended final
deployment. The current path is research/training integration, not licence approval.

## 4. Architecture and runtime flow

```text
Fallen + COCO + complete VisDrone manifests
              |
              v
  validate source counts and class mapping
              |
              v
 staging dataset (hard links, one class, fixed splits)
              |
              v
 independent schema/count checks
              |
              v
 clean COCO YOLO11n -> 20-epoch candidate training
              |
              v
 fixed per-source held-out evaluation -> regression gate
              |
              +--> fail: preserve research candidate; default unchanged
              +--> pass: still wait for licence, OAK-D and edge gates
```

The detector output interface does not change. Its boxes still flow to BoT-SORT,
temporal verification, selective Re-ID, and the persistent Survivor Manager.
VisDrone improves detector evidence; it does not solve identity or duplicate counting.

## 5. Files and responsibilities

- `nidar_survivor_demo/full_visdrone_pipeline.py`: validates the complete source,
  builds the mixture atomically, launches training, evaluates and gates the candidate.
- `nidar_survivor_demo/train_nidar_detector.py`: deterministic YOLO fine-tuning and
  safe interrupted-run resume from `weights/last.pt`.
- `Start-FullVisDroneTraining.ps1`: foreground/background Windows launcher, log
  redirection, duplicate-process protection and resume switch.
- `Start-OfflineDemo.ps1`: accepts explicit `-Model` selection without overwriting
  the known-good default.
- `.vscode/tasks.json`: one-click prepare and background-training tasks.
- `nidar_survivor_demo/tests/test_final_dataset.py`: rejects partial VisDrone source
  or a limited VisDrone mixture.
- `datasets/nidar_person_v4_full_visdrone/manifest.json`: provenance, counts, split
  controls, source hash and deployment policy.

Generated datasets, weights, logs and runs are intentionally ignored by Git.

## 6. Dependencies and choices

Validated local environment:

- Python 3.13.7;
- PyTorch 2.11.0+cu128 with CUDA 12.8;
- Ultralytics 8.4.163;
- OpenCV 4.13.0;
- RTX 3050, 4 GB VRAM;
- base model `models/yolo11n.pt`, 5,613,764 bytes, SHA-256
  `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`.

YOLO11n matches the laptop real-time architecture and is a realistic small-model
starting point for later edge export. FP16 mixed precision reduces VRAM use. Batch
16 previously fit this GPU. The clean COCO base is used because previous custom v7
over-specialised and failed 3 of 10 gates.

## 7. Configuration

- epochs 20, input 640, batch 16, CUDA device 0, workers 2;
- seed 42 and deterministic mode;
- patience 12;
- rotation, translation, scale, flips and mosaic for viewpoint variation;
- cache disabled to control RAM/disk pressure.

Augmentation creates plausible variation, but cannot replace real OAK-D footage.

## 8. Exact Windows / VS Code commands

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML

# Prepare and validate only
.\Start-FullVisDroneTraining.ps1 -PrepareOnly

# Long train -> evaluate -> gate workflow; terminal remains usable
.\Start-FullVisDroneTraining.ps1 -Background

# Follow progress and errors
Get-Content .\logs\full-visdrone-training.stdout.log -Tail 30 -Wait
Get-Content .\logs\full-visdrone-training.stderr.log -Tail 30 -Wait

# Use only after a genuine interruption and when weights\last.pt exists
.\Start-FullVisDroneTraining.ps1 -Background -Resume
```

In VS Code: `Ctrl+Shift+P` -> **Tasks: Run Task** -> choose the complete VisDrone
prepare or background-training task. Do not train while presenting; both share GPU.

After training, explicitly test a candidate without changing the default:

```powershell
.\Start-OfflineDemo.ps1 `
  -Model .\runs\training\nidar-person-yolo11n-v8-full-visdrone\weights\best-person-candidate.pt
```

## 9. Validation evidence

Automated/current-machine checks:

- Python compilation and VS Code JSON parsing: PASS;
- full source-manifest limits/mapping tests: PASS;
- prepare-only pipeline: PASS;
- exact image/label pair and YOLO schema audit: PASS;
- `pip check`: PASS;
- full application regression result is recorded in the final status update.

Final pre-training validation: 179 application regression tests passed in 52.335
seconds; 7 dataset-contract tests passed in 0.41 seconds; `pip check` reported no
broken requirements.

### Background run status at handoff

The 20-epoch v8 workflow started at `2026-10-06T09:02:33Z`. Launcher PID 9220 is
running, the first epoch is actively consuming batches at about 3.0–3.3 iterations
per second, PyTorch reports about 3.01 GB GPU memory, and stderr is empty. Training
completion and gate acceptance are deliberately not claimed in this report yet.

Not performed here: project OAK-D footage, drone height/tilt/blur, competition-like
dummy/debris scenes, edge export, quantised comparison or field latency test.

## 10. Error prevention and debugging

**Partial data:** old experiments deliberately limited VisDrone. The validator now
requires exact source counts and zero limits, failing before GPU training. Inspect
the unified `manifest.json` to diagnose this.

**Interrupted training:** do not delete the run. Verify `weights/last.pt`, then use
`-Resume`. Resume checks dataset, name and major settings so a different experiment
cannot silently continue the run.

**Training finishes but demo still uses COCO:** intentional. Training loss is not
proof of generalisation. Read the generated test, gate and pipeline JSON files.

## 11. Acceptance criteria and limitations

Dataset integration is accepted because source counts, schema, splits and pairing
independently pass. Model integration is **not accepted yet** merely because training
started. It requires completed weights, held-out evaluation, all regression gates,
approved usage terms, project-owned locked OAK-D testing and edge export comparison.

The application runtime default remains `models/yolo11n.pt` until those gates pass.
No public dataset can honestly guarantee every angle, blur, occlusion or rescue scene.

## 12. What you learned

“Dataset integrated” means the source is validated, reproducibly mixed and connected
to training/evaluation. It does not mean a trained model is already trustworthy.
More aerial data can improve small-person recall while harming ordinary-person
precision; the gate reveals that trade-off.

## 13. Professor/viva questions

**Was all VisDrone data used?** Yes: 6,471 train, 548 validation and 1,610 test-dev
images. Only its human-related classes are positive.

**How many detector models are active at once?** One checkpoint is selected per run.
Stored baselines/candidates are not silently ensembled.

**Why not label the boxes `survivor`?** RGB can detect human/body evidence but cannot
prove life, injury or medical status.

**Why not replace the model after training?** Held-out evaluation must show it
generalises and did not create unacceptable false positives/regressions.

**Does VisDrone solve lying-person detection?** No. It adds aerial/small/crowded
evidence. Fallen Person adds posture diversity; OAK-D field data is still required.

**Does it solve unique counting?** No. YOLO produces frame-local boxes. Tracking,
verification, Re-ID and later depth/SLAM handle continuity and duplicate suppression.
