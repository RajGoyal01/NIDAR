# Phase 4 — Temporal verification: complete learning handoff

Date: 2026-09-27. Status: **controlled-demo functional acceptance COMPLETE**.
User accepted: "Haan, Phase 4 demo working" after the live green-label,
count-removal and phone-pan/re-entry check. Phase 5 is not started.
Everything here is a PROJECT-DECISION or measured demo evidence, not a competition rule.

## 1. What we built and why

A person box in one picture can be a mistake. Phase 4 asks for repeated evidence
before calling a temporary track visually confirmed. Think of checking attendance:
one uncertain glimpse is not enough, but three good observations can be.

Default rule: at least **3 good observations among the last 5 processed frames**,
including a good observation now. Yellow means VERIFYING; green means CONFIRMED.
The denominator shown on the box is the configured five-observation window, even
while the first few observations are still arriving. A track can confirm on its
third observation; it does not have to wait for all five slots to fill.

CONFIRMED means repeatedly observed person evidence. It does **not** establish
injury, entrapment, consciousness, survival or a need for rescue. A persistent
false person detection can still pass this temporal rule.

## 2. Complete runtime flow

Phone camera -> wireless 5 GHz MJPEG -> newest-frame capture -> YOLO11n person
boxes -> BoT-SORT temporary IDs -> freshness check -> temporal verifier -> coloured
boxes, current verified-visible count and JSON diagnostic events.

The phone simulates a moving drone camera. All ML runs on the laptop. There are
no drone-control commands, cloud uploads, new database tables or persistent
survivor identities in this phase.

We preserve camera-only, detection-only and tracking-only modes. `-Verify` implies
tracking and detection. Re-ID remains OFF.

## 3. Important concepts and algorithm, block by block

**Temporal** means "across time". **Window** means a small recent history, not the
entire session. **Threshold** means the minimum acceptable value.

1. Validate incoming sequence, timestamp, dimensions, coordinates, score and
   duplicate IDs. Reprocessing the same frame is rejected: it cannot earn more votes.
2. Clear evidence on a tracker epoch change or a gap over one second. An epoch
   is a new tracking session after reset; `E2:T1` is not the same as `E1:T1`.
3. Create a five-slot deque for each new temporary ID. A deque is a small list
   that automatically drops its oldest entry when full.
4. Append true for a good current box, otherwise false. IDs missing from the frame
   also receive false, so old evidence cannot survive indefinitely.
5. Sum the true values. Confirm only with at least three and a good current box.
   Otherwise remain VERIFYING or revoke an earlier confirmation.
6. Draw only currently visible tracks. An absent person is not drawn or counted,
   even if a few old positive observations remain in memory.
7. Remove history after five consecutive missing processed frames. Camera/result
   loss clears all verification history immediately when detected by the loop.

Examples (`1` = good, `0` = missing/weak):

| Recent evidence | Current result |
|---|---|
| 1 | VERIFYING |
| 1, 1, 1 | CONFIRMED |
| 1, 0, 1, 0, 1 | CONFIRMED |
| 1, 0, 0, 0, 1 | VERIFYING |
| 1, 1, 1, 0 | Not confirmed now; absent boxes are hidden |

Frames deliberately skipped by latest-frame capture are not artificial failures:
we do not know their detections. Only processed fresh observations fill the window.
Five frames therefore represent different durations at different processing FPS.

## 4. Files and responsibilities

| File | Responsibility |
|---|---|
| `nidar_survivor_demo/verification.py` | Validated settings, bounded history, transitions and annotation |
| `nidar_survivor_demo/settings/verification.json` | Editable thresholds outside implementation code |
| `nidar_survivor_demo/config.py` | `--verify` and `--verification-config` command-line options |
| `nidar_survivor_demo/main.py` | Connect verifier after tracking; check freshness, reset, display and log |
| `Start-PhoneDemo.ps1` | Apply verified phone profile and launch `-Verify` |
| `nidar_survivor_demo/tests/test_phase4.py` | 16 Phase 4 deterministic/integration tests |
| Context, architecture, plan, decisions, README, AGENTS | Keep scope and acceptance consistent |

`VerificationResult` is a data record containing visible track states, transition
events and computation time. `confirmed_count` is calculated from that frame's
confirmed visible tracks, never a cumulative counter. The future survivor manager
will decide persistent identities; this module cannot do that job.

## 5. Configuration and tool choices

| Setting | Default | Meaning |
|---|---:|---|
| window | 5 | Maximum recent processed observations per ID |
| required | 3 | Minimum good observations |
| min_confidence | .35 | Minimum detector score for a positive observation |
| min_visible_area_ratio | .0001 | Minimum clipped box area divided by frame area |
| max_gap_seconds | 1.0 | Clear history across a long processing/input gap |

At 1280x720 the area threshold is about 92 pixels of box area. It rejects tiny,
offscreen or inverted boxes, not blur, incorrect pose or every bad crop. Clipping
means counting only the part inside the image. Scores are model outputs, not
calibrated probabilities that a person is real or injured.

Tracking still receives low-confidence boxes down to .10 to recover movement.
Verification requires .35 to treat a recovered box as positive evidence. These
thresholds serve different purposes; raising the tracking input to .35 would
remove its low-score recovery capability.

No new dependency, training or model download was needed. Python's standard
`deque`, dataclasses and JSON handle a tiny deterministic state machine; NumPy
and OpenCV already supply frame dimensions and drawing. Existing versions remain:
Python 3.11.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.163,
OpenCV 4.13.0.92, NumPy 2.2.6 and lap 0.5.12. Existing YOLO11n CUDA FP16 stays in use.

Alternative: require three consecutive observations. That is simpler but rejects
every brief miss. The selected rolling 3-of-5 rule tolerates two misses while the
current-good requirement avoids displaying a weak/lost observation as confirmed.
We deliberately do not latch confirmation forever or add appearance Re-ID early.

## 6. Exact Windows commands

Run from `C:\path\to\NIDAR\Raj Gupta\ML`, replacing PHONE_IP with the current app address.

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Verify
# Optional timed run:
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Verify -Seconds 60
# Automated full regression:
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m pip check
# Optional numeric trajectory/evidence logging; choose a NEW file name:
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --transport mjpeg --verify --track-log logs/my-phase4-session.jsonl
```

The direct Python command does not apply phone settings: use the launcher first
for 720p/JPEG35/motion-off. Press Q/Esc to stop. `-Track` runs Phase 3 only,
`-Detect` runs Phase 2 only, and omitting those switches runs camera-only.

## 7. Tests and measured results

Final full suite: **66 tests passed in 28.130 seconds**. This includes 16 Phase 4
tests plus the earlier 50 tests. `pip check`: no broken requirements.

Phase 4 covers third-hit confirmation, 3/5 with misses, rejection of 2/5 and
single flashes, low-confidence revocation, tiny/offscreen/inverted boxes,
missing/expired evidence, independent IDs, reset/epoch/gap, duplicate/backwards
inputs, malformed settings/values, bounded history, non-mutating drawing,
end-to-end metadata integration and stale-inference rejection before confirmation.
The stale test uses a deliberately slow detector and a one-hit verifier to make
an accidental stale confirmation easy to catch. It produces none.

Live wireless phone + actual GPU + GUI run, configured for 60 seconds:

| Measurement | Observed |
|---|---:|
| Processed frames | 779 |
| Sampled health states | 55/55 ONLINE |
| Connections / tracker resets | 1 / 0 |
| Mean sampled displayed FPS | 13.70 |
| Sampled FPS range | 10.93–15.24 |
| Maximum rolling verification p95 | 0.072 ms |
| Maximum simultaneously verified tracks | 3 |
| First confirmation per temporary ID, mean | 173.32 ms over 19 events |
| Shutdown | clean |

The first-confirmation interval starts at first tracked observation, not sensor
exposure; it is not end-to-end camera latency. Reconfirmations are excluded from
that mean. Nineteen temporary-ID events do NOT mean nineteen unique people.
Some regression tests ran concurrently, so this is not an isolated FPS benchmark.
The live run preceded only a redundant-drawing removal and metadata configuration
addition; the final code passed the full suite. No extra physical test was required
for these internal changes. Peak VRAM, glass-to-glass latency and numerical live
ID switches were not measured in this phase.

Evidence: `logs/phase4_live_demo.txt`, `logs/phase4_tests_final.txt` (local ignored
diagnostics, not camera recordings). User explicitly accepted the live demo.
Automated isolated-flash rejection complements live observation; it is not a
measured field false-positive rate or a rescue-dummy accuracy benchmark.

## 8. How to demonstrate it yourself

1. Start IP Webcam, connect laptop over the 5 GHz hotspot, launch `-Verify`.
2. Place a person clearly in view. Expect yellow briefly, then green CONFIRMED.
   The transition is normally too quick to study visually; JSON events record it.
3. Add other people. Each ID earns its own evidence; one cannot confirm another.
4. Remove a person. Verified-visible count falls; boxes do not remain on empty space.
5. Pan away and back. A returning track is checked against recent evidence; an
   expired/new track starts again. Do not call a changed ID a new unique survivor.
6. Optional outage check: stop/restart phone server. Video automatically recovers;
   old confirmations cannot transfer to the new epoch. This branch was tested
   automatically in Phase 4; a fresh physical outage was not requested this run.
7. Run the automated suite to demonstrate a one-observation flash is never confirmed.

## 9. Error handling, privacy and debugging

- **No green label:** check score >= .35 and sufficient observations. A tiny or
  mostly offscreen box may fail the area check. Read hits/5 and transition logs.
- **Yellow/green changes:** current score/missing evidence can revoke confirmation.
  This is intentional conservative behaviour, not a persistent identity decision.
- **Never confirms after fast motion:** inspect tracker ID changes first. Evidence
  is per ID; Phase 4 cannot repair a Phase 3 association failure.
- **Old-frame risk:** inference may finish after a frame has become stale. We
  check latest stream health, connection and age before adding evidence. The slow
  detector regression identifies this class of bug without relying on phone luck.
- **Invalid JSON:** startup fails before opening the camera. Unknown keys, invalid
  thresholds or missing files are rejected with a configuration message.
- **Log file exists:** logging uses exclusive creation to avoid overwriting evidence.
  Pick a new file name. No credentials or camera images are added to these logs.
- **Delay:** this module costs far less than a millisecond in this run. Most elapsed
  work is still camera/YOLO/tracking/display; more votes increase confirmation delay,
  not capture buffering. The latest-frame policy remains intact.

Keep IP Webcam on a trusted local network, not port-forwarded to the internet.
Settings and URLs are supplied via configuration/CLI, not hard-coded into Python.
There is no authentication service or website deployment in this phase.

## 10. Professor/viva answers and boundaries

**Why not count every YOLO box?** One-frame noise and repeated views would cause
false/duplicate counts. Temporal verification filters brief noise; later identity
management handles duplicates across revisits.

**Does CONFIRMED mean survivor?** No. It means temporally verified person evidence;
injury, liveness and domain-specific survivor/dummy evaluation are separate problems.

**Why 3 of 5?** A configurable initial tradeoff between quick confirmation and brief
miss tolerance, not an official NIDAR rule or universally optimal threshold.

**Why isn't this unique counting?** Temporary IDs can change after occlusion or a
reset. Phase 5 appearance Re-ID and Phase 6 persistent management address that.

**What did I learn?** A detector finds boxes, a tracker associates motion, and a
verifier checks recent evidence. Small bounded state, freshness checks, explicit
configuration and repeatable tests keep those responsibilities separate.

Acceptance is complete for this controlled Phase 4 demo. Real-world survivor
certification, persistent counting, Re-ID, the professional dashboard and flight
integration remain outside this phase and must not be claimed as delivered.
