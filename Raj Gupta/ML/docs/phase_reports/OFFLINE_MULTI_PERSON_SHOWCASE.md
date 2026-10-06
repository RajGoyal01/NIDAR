# Phone-free multi-person offline showcase

Date: 2026-10-01  
Decision: D-051 — PROJECT-DECISION  
Status: implementation and local engineering validation complete

## 1. Objective

Replace the repeated two-person photograph used in the old offline run with a
clear professor-facing demonstration that works without an Android phone or web
dashboard. The new demo must show the real detector processing several different
images, including four-person and six-person scenes.

This change demonstrates **current-frame detection capacity**. It does not claim
that every detected box is a medically verified survivor and does not create
persistent identities from unrelated stock photographs.

## 2. What caused the confusing result

The earlier command opened `runs/phase7-gpu/coco-static.avi`. That file is a
short video made from one COCO photograph containing two people. At end-of-file,
the capture worker reopened the same file. Therefore the operator kept seeing
the same two people and the same two valid persistent records.

There was no `max_people = 2` rule. The Survivor Manager creates IDs dynamically
with the next sequence (`S1`, `S2`, `S3`, and so on), bounded only by the reviewed
gallery capacity. The input simply never supplied a genuine third person.

How to recognise this problem later: compare `people_in_frame`, the source file,
and the scene content. If the file repeats the same photograph, refreshing or
restarting cannot introduce a new person.

## 3. Why detection and identity are separated

Think of detection as counting faces in one class photograph. Persistent identity
is like recognising the same student in another photograph. Six boxes in one
image prove that YOLO can report six visible people; they do not prove that the
system can recognise those people after they leave and return.

The new showcase therefore uses `P1...P6`:

- `P` means a box number in the current picture;
- the number resets for every new image;
- it is not a survivor identity;
- `S1...` remains reserved for the temporal Re-ID and Survivor Manager pipeline.

This prevents an impressive-looking but technically false identity claim.

## 4. Runtime flow

```text
Start-OfflineDemo.ps1
  -> repository virtual-environment Python
  -> load reviewed scene playlist
  -> load local models/yolo11n.pt
  -> CUDA/CPU YOLO person inference on every image
  -> validate actual count against the stored expected count
  -> continuously rerun inference on subtly moving replay frames
  -> OpenCV 1280x720 original camera-test layout
  -> automatic scene change or operator keyboard controls
```

There is no phone stream, FastAPI dashboard, SQLite mission database, tracking,
Re-ID, cloud upload or internet request in this flow. The interface reports
`REPLAY SOURCE` rather than falsely claiming that an absent camera is connected.

## 5. Files and responsibilities

| File | Responsibility |
|---|---|
| `Start-OfflineDemo.ps1` | Beginner-friendly Windows launcher and parameter validation. |
| `nidar_survivor_demo/offline_showcase.py` | Loads images, performs actual YOLO inference, validates counts and renders the slideshow. |
| `nidar_survivor_demo/settings/offline_showcase.json` | Small reviewable list of scene names, image paths and expected model counts. |
| `nidar_survivor_demo/tests/test_offline_showcase.py` | Ensures 1/2/4/6 coverage and verifies rendering does not mutate source images. |

The images already exist under `datasets/coco2017/images/val2017`. The demo does
not copy or modify them.

## 6. Libraries and configuration

- **Ultralytics YOLO11n:** existing lightweight person detector; selected because
  it already powers the accepted laptop baseline.
- **PyTorch CUDA:** runs inference on the RTX 3050 when `-Device 0` is used.
- **OpenCV:** decodes images, draws boxes and opens the native demonstration window.
- **NumPy:** creates the fixed presentation canvas.

Showcase settings:

```text
model: models/yolo11n.pt
image size: 640
confidence: 0.25
NMS IoU: 0.45
device: NVIDIA GPU (launcher default)
scene duration: 5 seconds
continuous inference target: 12 FPS
```

Confidence .25 was selected for this presentation after repeated-motion validation:
it preserves the reviewed 1/2/4/6 detections while suppressing borderline boxes
that flickered under replay motion. The full tracking pipeline remains configurable
at .20. Neither value is a universal deployment threshold; false positives and
false negatives must still be measured on project-owned indoor footage.

## 7. Exact Windows PowerShell commands

From the repository root:

```powershell
cd C:\path\to\NIDAR\Raj Gupta\ML
.\Start-OfflineDemo.ps1
```

Controls:

```text
N or right arrow = next image
P or left arrow  = previous image
R                = rerun real YOLO on the current image
Space            = pause/resume automatic cycling
Q or Esc         = close safely
```

Slower 8-second scenes and one complete cycle:

```powershell
.\Start-OfflineDemo.ps1 -SceneSeconds 8 -Cycles 1
```

Engineering validation without a window:

```powershell
.\Start-OfflineDemo.ps1 -Headless -Device 0
```

CPU fallback:

```powershell
.\Start-OfflineDemo.ps1 -Device cpu
```

## 8. Measured validation

### Automated validation

- 3 focused unit tests: PASS.
- Full project regression suite: 174 tests PASS in 52.644 seconds.
- Dependency consistency (`pip check`): PASS.
- Six-scene real-model CUDA validation: PASS.
- Seven moving-frame samples per scene: every count remained stable.
- Expected counts: `1, 2, 4, 6, 4, 6`.
- Actual counts: `1, 2, 4, 6, 4, 6`.
- `all_counts_match`: true.
- First real image inference: 138.69 ms (one-time shape/startup effect).
- Remaining measured image inference times: 26.54–31.68 ms, except the
  two-person image at 28.78 ms.

These timings are per-image detector calls, not camera-to-screen latency.

### Hardware/user validation

No phone is required. The GPU and local data were validated on the development
laptop. A professor-viewing/usability acceptance is not silently claimed; the
operator should run the exact command above and use N/R once before presentation.

## 9. Limitations

- COCO photographs are general public-image data, not final NIDAR indoor-drone
  footage.
- Exact counts are model outputs for the selected images, not ground-truth proof
  of perfect recall in every scene.
- This does not test lying-body domain performance, blur, darkness, depth,
  localisation, SLAM or autonomous flight.
- It does not test persistent identity or duplicate suppression. Use temporally
  related recordings/participants for S-ID acceptance.
- The possible competition requirement of up to six survivors remains
  RULEBOOK-VERIFY until the official current PDF is stored and checked.

## 10. Manual professor-demo checklist

1. Disconnect/ignore the phone; the demo does not need it.
2. Open PowerShell in the repository root.
3. Run `.\Start-OfflineDemo.ps1`.
4. Wait for all six `offline_scene_ready` records in PowerShell.
5. Show automatic `1 -> 2 -> 4 -> 6 -> 4 -> 6` scene progression.
6. Press `R` on a six-person image and explain that YOLO inference reruns.
7. Explain that green boxes and the count are current-frame perception.
8. Explain that persistent S identity needs tracking, repeated evidence and Re-ID.
9. Press Q to close cleanly.

## 11. What was learned

- Input diversity and model capacity are different things: a two-person source
  cannot demonstrate a six-person detector.
- Current occupancy and mission-unique identity are different metrics.
- A trustworthy demo labels oracle, detector and identity evidence separately.
- Offline fallback should use multiple reviewed scenes, not silently loop one
  convenient photograph.

## 12. Viva questions and answers

**Q: Why did the previous demo stop at S2?**  
A: Its source contained the same two people on every frame. Re-ID correctly kept
the mission unique count at two instead of inventing more people.

**Q: Does this prove the final drone detects six survivors?**  
A: No. It proves the current laptop YOLO produces six person boxes on two reviewed
offline images. Final proof needs NIDAR-like indoor drone footage and hardware tests.

**Q: Why are the new labels P1 rather than S1?**  
A: P is a current-image box. S is a persistent mission identity that requires
temporal and re-entry evidence. Using S across unrelated photos would be false.

**Q: Is the count hard-coded?**  
A: No. The expected value is only an acceptance check. The displayed value comes
from `len(result.people)` after actual YOLO inference; rerun with R to repeat it.

**Q: How would this connect to the complete system?**  
A: YOLO boxes feed BoT-SORT, temporal verification, selective Re-ID and the
Survivor Manager. In the final drone, depth and SLAM/map position strengthen
identity and localisation.
