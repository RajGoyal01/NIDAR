# Detector Precision and Promotion-Gate Recovery

**Date:** 2026-10-07  
**Status:** implementation/data validation complete; 8-epoch candidate running  
**Runtime default changed:** no

## Objective

Recover ordinary-person precision/recall and reduce false positives after the full
VisDrone stage, without erasing the useful fallen, aerial and small-person features.
The fixed promotion thresholds and held-out test remain unchanged.

## Root cause

The full training split had 106,396 VisDrone boxes but only 4,730 COCO and 2,433
Fallen boxes. Dense aerial annotations dominated gradient updates. V8 consequently
improved aerial/fallen recall but its COCO recall/precision fell to .492/.673 and
71 COCO-negative images produced detections.

Changing the gate or merely raising confidence would hide rather than solve the
problem. Raising confidence improves precision but further reduces recall, while
the failed COCO-recall gate requires recall to rise from .492 to at least .65.

## Selected recovery design

```text
v8 full-VisDrone candidate
          +
COCO replay x4 + all Fallen + deterministic 6,000-box aerial reminder
          |
frozen early backbone + low learning rate + lighter augmentation
          |
same full 2,956-image test -> same 10 promotion checks
```

COCO repetition is sampling weight, not new evidence. Repeated files are hard links.
No sample crosses its original split, and the test image/label index is byte-size and
name identical to the v4 test index.

## Dataset evidence

| Split | Images | Boxes | Empty/hard-negative images |
|---|---:|---:|---:|
| Train | 13,826 | 27,364 | 5,505 |
| Validation | 1,523 | 4,551 | 494 |
| Test | 2,956 | 29,282 | 794 |

Training contains 11,436 COCO replay entries, all 2,021 Fallen images and 369
deterministically selected VisDrone reminders covering 6,011 boxes. Label audit
found zero invalid rows.

## Files

- `prepare_detector_recovery.py`: deterministic replay builder and leakage controls.
- `recover_nidar_detector.py`: prepare, train, full evaluation and unchanged gate.
- `train_nidar_detector.py`: adds the reviewed `recovery` training profile.
- `Start-DetectorRecovery.ps1`: background/resume PowerShell launcher.
- `.vscode/tasks.json`: one-click background recovery task.
- `test_final_dataset.py`: unchanged-test and deterministic-selection contracts.

## Training configuration

- predecessor: v8 full-VisDrone candidate;
- epochs 8, batch 16, 640 pixels, CUDA AMP;
- layers 0–9 frozen to preserve early general visual features;
- AdamW, initial learning rate 0.0005, final factor 0.10;
- lighter rotation/translation/scale/vertical flip and mosaic than full training;
- seed 42 and deterministic mode.

## Exact commands

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\Start-DetectorRecovery.ps1 -PrepareOnly
.\Start-DetectorRecovery.ps1 -Background
Get-Content .\logs\detector-recovery.stdout.log -Tail 30 -Wait
Get-Content .\logs\detector-recovery.stderr.log -Tail 30 -Wait
```

After an interruption, and only when the run has `weights\last.pt`:

```powershell
.\Start-DetectorRecovery.ps1 -Background -Resume
```

## Validation completed before training

- recovery source/manifest contracts: passed;
- 13,826/1,523/2,956 exact image-label pairs;
- test index unchanged from full v4;
- zero invalid YOLO labels;
- 9 focused tests passed;
- 179 application regression tests passed in 56.563 seconds;
- `pip check`: no broken requirements.

## Acceptance

The candidate must pass the original 10 checks. No automatic model replacement is
allowed. Even a 10/10 public-data pass remains a research gate; project-owned OAK-D
locked footage, licence approval and edge-export comparison remain mandatory.

## Viva questions

**Why repeat COCO?** Dataset sampling controls how often each domain contributes a
gradient. It counters the much denser VisDrone annotations without pretending the
repeats are new data.

**Why freeze early layers?** Early layers learn reusable edges/textures. Freezing
them reduces destructive forgetting while later layers adapt the detection decision.

**Why not lower gate thresholds?** That would improve the scorecard, not the model.
The same gate is retained so results remain comparable.

**Why keep aerial reminders?** COCO-only recovery could swing the model back and
erase small/aerial gains. The reminder subset provides bounded anti-forgetting data.
