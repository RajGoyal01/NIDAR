# Phase 2 — Person detection: implementation and learning report

Date: 2026-09-27. Scope: laptop/phone perception prototype, NOT an autonomous drone.

## 1. Status and scope

**Final status: Phase 2 functional acceptance COMPLETE (2026-09-27).**

Implemented: pretrained person detection, GPU inference, boxes/confidence, configurable
thresholds, current-frame person count, latency/FPS, latest-frame processing,
stale-result suppression, explicit offline model setup, benchmark and tests.
User confirmed a box appears for one person and answered "Perfect, ok" to the
post-fix movement/blank-screen check. The bad-frame reconnect bug is fixed.
User subsequently confirmed successful detection of 2-4 people in the final live
test and requested Phase 2 closeout. All Phase 2 functional acceptance checks are
closed. Phase 3 has not started and requires separate user authorization.

No custom training, tracker IDs, persistent survivor IDs, cumulative counts,
injury classification, Re-ID, database, web dashboard or flight commands were added.

## 2. How the complete runtime works

```text
Moto + IP Webcam -> 5 GHz wireless -> multipart JPEG receiver
    -> newest compressed image -> decoder -> newest BGR camera frame
    -> YOLO11n on RTX 3050 -> class-0 person boxes + scores
    -> annotate THAT frame -> OpenCV preview + JSON performance messages
```

Camera reception continues while AI runs. There is no backlog of AI jobs.
If the camera produces 45 frames/s and AI displays 23, we deliberately skip
intermediate frames rather than replay old movement later.

The model warms up before the camera timer starts. This pays initial loading/GPU
setup cost once. Startup is slower than an ordinary subsequent frame.

The preview checks camera health again after inference. Old results are hidden
if the camera disconnected, the connection changed or the frame expired.
An unavailable result is `null`, not a misleading zero-person observation.

## 3. Dataset versus pretrained model

COCO images and their box annotations are evaluation material. They do not become
an AI model merely by downloading them. `models/yolo11n.pt` contains weights:
numbers already learned by Ultralytics using training data. Phase 2 loads these
weights and performs inference (using learned knowledge on a new image).

COCO has `person`, not a certified `survivor`, `injured` or rescue-dummy class.
A green box means a person-like object was detected. It does not prove injury,
life status, need for rescue or unique identity. Photos/screens/reflections can
also fool a visual detector. Never use this prototype as a life-safety system.

## 4. Model provenance and dependency choices

- Model: YOLO11n detection, not pose/segmentation. Nano limits compute demand.
- Official source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
- Bytes: 5,613,764.
- Local SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- This is a locally measured integrity fingerprint, not an independent publisher checksum.
- Model docs: https://docs.ultralytics.com/models/yolo11/
- Prediction docs: https://docs.ultralytics.com/modes/predict/
- Licensing: https://www.ultralytics.com/license — AGPL-3.0 / Enterprise options;
  review applicable terms before distributing/deploying a closed-source product.
- Existing pinned stack retained: Python 3.11.15, torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, Ultralytics 8.4.163, OpenCV 4.13.0.92, NumPy 2.2.6.
- No new dependency installation was needed. No cloud inference/API key is used.

Only load trusted `.pt` weights: model files can contain executable serialized
Python objects. Ordinary demo startup requires an existing local file and does
not silently download a model. The explicit setup command downloads only the
fixed official asset, checks size and replaces the target only after completion.
Weights/datasets/runtime logs remain ignored by Git. Camera images are not recorded.

## 5. Files and important code blocks

| File | Responsibility |
|---|---|
| `nidar_survivor_demo/detector.py` | Validated settings, YOLO adapter, plain detection records, drawing |
| `nidar_survivor_demo/setup_model.py` | Explicit official model download and fingerprint |
| `nidar_survivor_demo/config.py` | CLI options and invalid-value rejection |
| `nidar_survivor_demo/main.py` | Connect capture, detection, freshness checks, preview and cleanup |
| `nidar_survivor_demo/preview.py` | Camera health and Phase 2 detection footer |
| `nidar_survivor_demo/benchmark_detection.py` | Reproducible balanced COCO sample comparison |
| `nidar_survivor_demo/tests/test_phase2.py` | Regression tests using fake inputs, no download |
| `Start-PhoneDemo.ps1` | Verify phone profile and enable detection with `-Detect` |
| `nidar_survivor_demo/camera/mjpeg_capture.py` | Wireless reader and sanitized failure diagnostics |

`DetectorConfig` groups settings and rejects NaN, invalid confidence, invalid sizes
and unsupported devices before inference. A frozen dataclass is a small record
whose fields are not meant to change after construction.

`PersonDetector.__init__` checks the file, chooses GPU/CPU, checks the class map and
warms the model. CUDA means NVIDIA GPU execution. FP16 uses half-precision
numbers; CPU uses FP32. `--fp32` remains available for comparison/debugging.

`predict()` accepts an 8-bit BGR image (OpenCV's blue-green-red channel order).
`classes=[0]` filters COCO output to people; it does not retrain the network.
`rect=False` uses fixed square letterboxing: padding, not stretching, preserves
aspect ratio while keeping the GPU input shape constant. Results use original
image coordinates. Copying results to CPU waits for GPU completion, so the
reported time includes preprocessing, inference, postprocessing and result transfer.

`annotate()` draws on a copy, preserving the capture-owned frame. The main loop
retains that same frame with its boxes; it never pastes old boxes on a newer image.

## 6. Configuration and commands

Run from `C:\path\to\NIDAR\Raj Gupta\ML`. Replace PHONE_IP with the current phone address.

```powershell
# One-time setup (already completed here)
.\.venv\Scripts\python.exe -m nidar_survivor_demo.setup_model

# Recommended wireless Phase 2 demo
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Detect

# Alternative inference size; capture remains 720p
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Detect -ImageSize 512

# Explicit settings; this command does not reconfigure the phone
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --transport mjpeg --detect --imgsz 640 --confidence 0.35 --iou 0.45 --device 0

# Camera-only baseline
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080

# Automated tests and repeatable sample benchmark
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m nidar_survivor_demo.benchmark_detection
```

`--confidence 0.35`: reject predictions below this score. Increasing it can reduce
false detections but miss real people. It is not a calibrated medical probability.

`--iou 0.45`: overlap threshold used by non-maximum suppression (NMS), which removes
competing boxes for the same object in a single frame. This is NOT cross-frame
duplicate-person suppression. IoU is overlap area divided by combined area.

Default `--device auto` prefers CUDA, otherwise CPU; `--device 0` requires CUDA.
`--seconds 30` ends after 30 seconds of runtime; Q/Esc closes a graphical preview.
`--headless` runs detection/metrics without the window. Credentials must not be
embedded in the direct MJPEG URL. Keep phone HTTP traffic on a trusted local network;
do not port-forward the camera. Phone address is runtime configuration, not source code.

## 7. Measured COCO baseline (not a survivor accuracy claim)

Fixed seed 42; 100 person-positive plus 100 negative images from prepared COCO2017
evaluation data, excluding crowd-person images. Same 200 images for both sizes.
Confidence .35, NMS IoU .45; confidence-ordered one-to-one matching at IoU >= .5.
This is a smoke test, NOT official COCO mAP and NOT the full 4,773-image evaluation.

| Input | TP / FP / FN | Precision | Recall | Median / p95 AI ms | Negative images with FP |
|---|---|---|---|---|---|
| 512 | 158 / 19 / 141 | 89.27% | 52.84% | 28.98 / 36.04 | 3 / 100 |
| 640 | 169 / 17 / 130 | 90.86% | 56.52% | 28.29 / 33.09 | 2 / 100 |

Precision asks: how many predicted boxes matched labelled people? Recall asks:
how many labelled people were found? Low recall is a real baseline limitation,
particularly relevant to small/occluded people. Do not present 90.86% as overall
survivor accuracy. Default 640 is a project decision based on this limited sample;
the tiny timing difference does not prove 640 is universally faster.

CUDA FP16 was used. Peak PyTorch allocated tensor memory was approximately
44.6 MiB (512) and 53.7 MiB (640); this excludes CUDA context, caching/reservation,
desktop apps and other GPU memory. It is NOT total GPU memory consumption.
Exact sampled filenames and metrics: `logs/phase2_benchmark.json`.

Initial variable-shape benchmark had large outliers (512 p95 ~2,616 ms).
After fixed-shape letterboxing, p95 fell to ~36 ms. This supports fixed shape for
the demo; it does not measure camera-to-screen delay. The installed library's
deprecated `half` argument was also replaced with supported `quantize=16/32`.

## 8. Live evidence and acceptance checklist

Initial real-phone 720p graphical run, ~85 seconds: 1,048 processed frames,
clean shutdown, six total connections with automatic recovery. Stable online
segments mostly showed 20-25 displayed FPS; outages reduce whole-run throughput.
Person observations included 0/1/2 boxes, but box counts alone do not validate
ground truth. At that initial stage the user confirmed only one-person detection.
After the fix, the user additionally confirmed 2-4-person detection in a separate
live test. Final test log: `logs/phase2_multi_person_test.txt`; 1,135 processed frames,
one connection, clean shutdown. Physical person-count acceptance is user-observed,
not inferred solely from sampled model output.
All 60 one-second samples were ONLINE. Mean sampled display rate was 18.76 FPS
(range 10.58-24.70), so a sustained 20/30 FPS minimum is not certified. Maximum
sampled box count was two; the user's observation supplies the 2-4-person acceptance
evidence, since one-second sampling does not capture every displayed frame.

- [x] Official local weights and CUDA inference work.
- [x] Configurable confidence/IoU, person-only output and box confidence.
- [x] 512/640 measured comparison.
- [x] Real phone produces person boxes (user-confirmed, one person).
- [x] Intermittent bad-frame reconnect fix; unit and live retest.
- [x] Fixed-preview movement/blank-screen check accepted by user ("Perfect, ok").
- [x] Two-to-four-person controlled-demo test accepted by user. No numerical
  live false-positive rate was measured; COCO sample errors remain documented above.

FPS counts output frequency. Local decoded-frame age starts AFTER reception and
decode; it excludes phone/network latency. Neither is glass-to-glass latency.
Numerical end-to-end latency remains unmeasured. Tracking metrics are not applicable
yet because there is no tracker.

## 9. Manual acceptance test

1. Start phone server; connect laptop over 5 GHz, use wall power if phone is low.
2. Run the recommended launcher; wait for model warmup and live video.
3. Show one full person, then two/three/four with consent; note missed/false boxes.
4. Pan moderately and move near/far. Do boxes stay attached to their source image?
5. Point at an empty area for 10 seconds. Check false detections; never expect
   the person-in-frame number to be a unique cumulative count.
6. Stop server for five seconds and restart. Offline video/results must disappear;
   live detection should recover without restarting the laptop app.
7. Press Q; check clean shutdown. Do not claim automatic recovery means the Wi-Fi
   connection cannot drop.

## 10. Debugging and what you learned

### Fixed intermittent blank screen

Detailed read diagnostics showed `invalid_jpeg`, rather than a receiver/network
exception. Previously a single undecodable JPEG returned `False` to PhoneStream,
which treated it as a dead stream and waited two seconds before reconnecting.
Now the adapter skips that image and waits for the newest next image on the same
connection. A fixed read deadline still forces recovery for sustained corruption,
a real stall or a disconnected server. No old-frame replay or hidden failure.
The origin of malformed JPEGs upstream is not proven.

A regression test supplies a bad JPEG then a valid one, checks successful recovery,
then verifies that absence of a fresh image still times out. All 36 tests passed
in 24.204 seconds; dependency consistency also passed. One test initially used
an unrealistically small 1 ms age with a 10 ms sleep; Windows clock granularity
made it intermittent. A larger intentional test delay now crosses the clock tick.

Post-fix graphical check: 613 processed frames over ~28.3 seconds, 27 logged
samples all ONLINE, one connection, clean exit. Mean sampled displayed FPS 21.82,
maximum sampled local frame age 78 ms. These are limited-run measurements, not
a guaranteed sustained frame rate or total camera latency.

Longer post-fix check: 60 seconds of actual phone + GPU inference, 1,384 frames
processed, all 59 logged samples ONLINE, one connection, clean timed shutdown.
Mean sampled consumer rate 23.13 FPS, maximum sampled local age 109 ms. A bad
JPEG actually recurred and was skipped without reconnecting, confirming the
fix on the real phone as well as the synthetic regression test.
Evidence: `logs/phase2_final_endurance.txt`. Headless consumer rate is not GUI FPS.

- No model: run setup; only use official trusted weights.
- Wrong Python/CUDA: use the project `.venv` command; don't install into global Python.
- Test import error: use `-t .` so relative imports retain their package context.
- Fresh-video warning: inspect state, receiver error, age and connection count.
  Do not remove the warning or freeze an old frame to make a failure look healthy.
- Detection flicker: inspect lighting, full-body visibility, blur and confidence;
  temporal verification is a later phase, not a reason to invent survivor certainty.
- Too many GPU delays: compare benchmark versus live run and keep input shape fixed.

## 11. Professor/viva questions

**Did you train a new model?** No. We integrated pretrained weights and measured a baseline.

**Why download COCO if the model is already trained?** For labelled evaluation and
repeatable failure analysis; inference itself does not need the dataset.

**Why Nano?** A lightweight baseline for a 4 GB laptop GPU and low-latency demo.

**Why drop frames?** To process what is happening now rather than accumulate delay.

**Is NMS the final duplicate-survivor system?** No. It removes overlapping boxes
within one frame; later tracking/Re-ID/persistent management address repeat visits.

**What does 28 ms measure?** Local model-call processing, not phone-to-screen latency.

**Why no database yet?** Phase 2 has no persistent identities. Storage comes with the
survivor manager; a database now would not solve detection reliability.

**What is next?** Only after Phase 2 acceptance and user authorization: BoT-SORT
temporary tracking. No Re-ID integration before the tracking baseline is stable.
