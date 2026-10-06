# COCO dataset for the NIDAR demo

## Scope

User-requested dataset preparation after Phase 0, 2026-09-27. This does not advance
the phone-streaming or detection phases. We selected COCO 2017 validation (5,000
images) for baseline evaluation. No training split or model weights are installed.

## What each item means

- Images: examples on which we will later test person detection.
- Annotations: human-provided boxes telling us where people actually are.
- Weights: learned model parameters; these are different files, loaded in Phase 2.
- Evaluation: comparing predictions against annotations; no weights are updated.

The initial pretrained model does not need COCO downloaded to operate. This local
dataset is useful for repeatable evaluation and understanding label formats.

## Sources and contents

Official project: https://cocodataset.org/#download

Archives use COCO's storage bucket through valid HTTPS:

- https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
- https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip

The original images.cocodataset.org HTTPS hostname failed certificate hostname
validation on this machine. The alternate bucket URL passed with verification
enabled. Original annotation JSON and image license metadata are preserved.
COCO terms: https://cocodataset.org/#termsofuse. Image licenses are per-image;
do not assume one blanket license for every photograph.

Local structure under `datasets/coco2017/`:

```text
archives/                  downloaded archives, kept for reproducibility
annotations/               original instances_val2017.json
images/val2017/             all 5,000 validation images
labels/val2017/             derived person-only YOLO txt labels
person_val.txt             explicit eligible evaluation image list
person_val.yaml            local evaluation configuration (train: null)
manifest.json              sources, SHA256, counts, license metadata, disk usage
```

The whole datasets directory is excluded by .gitignore. The generated YAML stores
the resolved local dataset path; rerun preparation after relocating the project.
The script itself has no machine-specific path.

## How preparation works

`nidar_survivor_demo/prepare_coco.py` downloads the two archives concurrently,
checks download lengths, validates ZIP CRCs, and records SHA256 fingerprints.
Fingerprints are locally computed for future comparison, not verification against
independently published official hashes. Failed downloads retry up to three times.
Incomplete files have a .part suffix and resume using HTTP byte ranges; completed
archives are reused on reruns. Source length and returned byte range are checked.

It extracts the validation annotation file, decodes every image using Pillow,
checks dimensions against annotation metadata, and converts person boxes.
Existing derived files in this dedicated dataset folder are regenerated on rerun;
do not hand-edit them. No dependency installs are needed beyond the Phase 0 stack.

COCO `category_id=1` maps to YOLO `class_id=0`. A source box `[x, y, width, height]`
becomes `[centre_x / image_width, centre_y / image_height, width / image_width,
height / image_height]`, preceded by class 0. Values are checked to stay in [0,1].

Example YOLO line: `0 0.5 0.5 0.2 0.6` means a person box at the image centre,
20% of its width and 60% of its height. Empty label files mean no annotated
person in that eligible image; these examples help measure false detections.

## Crowd handling and limits

COCO has special crowd annotations that ordinary YOLO txt cannot represent as
ignore regions. Images containing person crowd annotations are excluded from the
derived evaluation list, while all source images/annotations remain available.
This prevents a skipped crowd from becoming a false negative-image label.

This derived person-only set is not the full official COCO benchmark. Use the
original JSON and official COCO evaluation rules for official metric comparisons.
When evaluating an 80-class pretrained model later, explicitly restrict predictions
to person and verify the class mapping; a one-class YAML alone does not filter
the model's predictions. No evaluation has been run in this dataset preparation.

Do not train on this validation set and then report its results as unseen-test
accuracy. Future training data and a separate project test set need independent
scenes. COCO does not establish NIDAR dummy/survivor reliability or phone latency.

## Run and inspect

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\nidar_survivor_demo\prepare_coco.py
Get-Content .\datasets\coco2017\manifest.json
```

The first command downloads data if absent, rebuilds labels and verifies images.
The manifest contains the measured results. Inspect source JPGs with your image
viewer and matching txt files to learn how boxes are represented. Do not invoke
training on person_val.yaml; it intentionally has no training split.

## Verified installation results — 2026-09-27

- All 5,000 original validation images decoded successfully.
- Original JSON contains 11,004 person annotations (including crowd annotations).
- Derived evaluation list: 4,773 unique image/label pairs.
- 2,466 eligible images contain person boxes; 2,307 have no annotated person.
- 7,846 person boxes converted to normalized YOLO format.
- 227 images containing person crowd annotations excluded from the derived list.
- Both archives passed ZIP CRC checks and local SHA256 fingerprints were recorded.
- Disk usage approximately 1.90 GB decimal, including retained archives.
- Network DNS failure interrupted images near 80%; HTTP range recovery resumed
  successfully at 635 MiB. Preparation finished with exit code 0.
- Independent config/list check confirmed train is null and all 4,773 unique
  listed images have matching labels. No training or inference was executed.

## Questions you should be able to answer

**Did downloading COCO train our model?** No. It only prepared examples and labels.

**Why validation first?** Our next detector is already pretrained. We first need
repeatable baseline tests; custom training comes after measuring its failures.

**Why retain no-person images?** They reveal false detections that person-only
positive images cannot fully measure.

**Why are some images excluded from the derived list?** YOLO txt lacks COCO crowd
ignore semantics. Original data stays intact for future official evaluation.

**How is this connected to our moving phone?** It supports offline detector checks.
Actual phone footage is still required to measure motion, latency and domain errors.
