# Phase 1 — Phone stream and lag investigation

**Final functional status: Phase 1 COMPLETE, 2026-09-27.** The user subsequently
passed the 720p movement and automatic-recovery checks; read
`PHASE_01_HANDOFF.md` for the current launch instructions, acceptance matrix,
25 automated tests, 120-second capture result and remaining performance limits.
The dated experiments below are retained as history, not the current status.

## Latest result — 5 GHz working profile, user-confirmed improvement

On 2026-09-27 the laptop confirmed 5 GHz / 802.11ac, channel 149, negotiated
433.3 Mbps link speed and 99% signal. Link speed is not actual throughput.
The phone restart had restored 1920x1080, quality 49 and motion detection ON.
Reapplied the same lighter comparison profile: 960x540, quality 35, motion OFF.
Direct MJPEG was used; no code or dependencies were changed for this test.

The run started at 12:14:51.851 and ended cleanly at 12:15:14.170, before the
requested 45-second duration. It consumed 580 distinct decoded frames over
about 22.3 seconds (~26 displayed FPS including startup) with one connection.
Rolling display FPS was initially 31–32, dipped as low as 14, then recovered to
30–32. Render p95 settled around 4.5 ms. No reconnect was required.

The user confirmed: **"Smooth hai aur delay bhi kam hai"**. This validates a
qualitative improvement on this setup, not a measurement of end-to-end delay
in milliseconds. The earlier 2.4 GHz comparison used the same lighter profile
and mostly displayed 16–21 FPS. Scenes/phone load were not controlled, so this
is an observational comparison rather than a rigorous isolated benchmark.

Current working profile: **5 GHz + 960x540 + quality 35 + motion OFF + direct
MJPEG**. Restart can reset settings; recheck the phone and decoded actual_size.
Do not require USB. The main remaining work is longer movement/reconnect
testing, measuring glass-to-glass delay and testing whether 720p can be restored
without losing responsiveness. Phase 1 is not yet declared fully accepted.

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --transport mjpeg --display-fps 60
```

Older sections below retain the experiment history and are superseded by this
result for the current 5 GHz profile only.

## Wireless-only tuning (latest user constraint)

USB testing is no longer proposed: the user explicitly requires wireless.
Applied a temporary lighter phone profile through its local API: 960x540,
JPEG quality 35, app motion detection OFF. Previous values were 1280x720,
quality 49 and motion detection ON; those can be restored through the same
IP Webcam settings. Lower resolution/compression can remove useful distant-person
detail, so this is a transport comparison, not an accepted ML input specification.
The app's motion detector is not our future YOLO detector and is unnecessary here.

Preview wording now reads `Local frame age (NOT total delay)` and explicitly
states `Camera-to-screen delay: NOT MEASURED`. Fourteen unit tests passed after
this label edit. No new transport code or dependency was added in this step.

The 2.4 GHz lighter-profile test showed rolling display rates mostly 16–21 FPS,
render p95 around 4–5 ms after startup. This does not prove reduced visible delay.
The user confirms a 5 GHz hotspot option exists; switching band/reconnecting
and comparing the same profile are pending. The phone reports AC charging now.

To compare: finish preview, switch hotspot to 5 GHz on phone, reconnect laptop,
restart IP Webcam and provide its current address. Verify the band with
`netsh wlan show interfaces`, then repeat the same direct-MJPEG command and
phone movement test. Restart may reset camera resolution; verify actual_size.
No guarantee of zero delay or a particular latency improvement is made.

## Latest acceptance result: direct mode still delayed

The user confirmed seconds of visible delay after the direct-MJPEG 720p retest.
Therefore its latency acceptance FAILED; prior pending statements below are
historical. The restart initially restored 1080p; 720p was reapplied and verified
in decoded images. Preview at 720p was about 18–21 FPS, but that does not mean
low latency. All 23 automated tests passed; they test logic, not wireless latency.

Latest read-only checks: Wi-Fi 2.4 GHz, signal 99%, phone battery 2%, no charging
flag. Neither low battery nor Wi-Fi has been proven the root cause. Proposed
next step is USB tethering as a controlled wired comparison using the same app
and code. This requires the user to connect a data cable and enable tethering;
no USB connection or Android setting has been changed remotely. The final demo
still targets a moving wireless phone; wired testing is only diagnostic.

## Current state — direct MJPEG candidate, single-decoder experiment rejected

The user reported WORSE delay with the single-thread FFmpeg experiment below.
Its default was rolled back to 0 (automatic). Historical results below do not
mean that experiment solved latency. The user has requested near-immediate video;
exactly zero milliseconds is not physically attainable.

Added `camera/mjpeg_capture.py`, selected explicitly with `--transport mjpeg`.
The phone's actual response was inspected: multipart JPEG with Content-Length
per image. `MultipartJpegs.feed()` parses split headers/bodies, bounds headers
to 16 KiB and each encoded JPEG to 8 MiB, and returns only the last complete JPEG
from a received batch. A receiver thread continuously drains HTTP using read1;
a single compressed-image slot is replaced, not queued. The existing capture
worker decodes the newest slot with cv2.imdecode and publishes its usual latest
BGR frame. There are two bounded latest-image slots, not a growing video queue.

```text
Phone /video -> HTTP receiver -> latest compressed JPEG
    -> OpenCV JPEG decode -> latest decoded frame -> health checks -> preview
```

The receiver does not record or upload images. It reuses the standard-library
HTTP client, so no package installation was needed. Socket shutdown wakes reads
on release; the receiver owns connection close and is joined with a deadline.
Malformed streams, non-JPEG responses, read failure and timeout trigger the
existing reconnect path. This is for the trusted local IP Webcam; HTTP(S) URL
credentials, redirects and MJPEG without per-part Content-Length are unsupported.
Phone-side encoding or TCP buffers can still contribute delay.

Live test 11:52:02–11:52:42: 821 distinct decoded frames displayed over about
40 seconds (~20.5 FPS including interruption). Rolling display samples were
mostly 20–22 FPS, initially up to 28.3. One interruption occurred around second
32, with automatic recovery around second 34. Cause was not conclusively
identified. The process stopped cleanly. Visual latency confirmation is still
pending; console frame age cannot establish camera-to-screen milliseconds.

Tests added: byte-by-byte split parsing, burst/latest replacement with a partial
next image, malformed/oversized headers and images, real local MJPEG stale-frame
rejection, reconnect, and receiver thread termination. The four new tests passed.

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --transport mjpeg --display-fps 60
```

Move the phone, compare with the snapshot fallback, and explicitly report
whether seconds of visible delay remain. Phase 1 acceptance is not complete.

## Smoothness follow-up — single-thread decoder experiment

Added `--decoder-threads` (default 1, 0 restores automatic, maximum 32) for the
OpenCV/FFmpeg transport. OpenCV documents this as an open-only maximum decoder
thread count. We pass it when opening the stream, not afterward. This tests
whether reducing frame-thread buffering can retain continuous-video smoothness
without its earlier delay; it does not eliminate phone or TCP buffering.

The live 720p run began 11:48:46 and exited cleanly at 11:49:10 (about 23.7 s,
not its requested 40 s duration). It consumed 502 distinct decoded frames.
Sampled display rates were approximately 19–26.5 FPS; user confirmation of
visual delay for this particular mode is still pending. A decode-rate spike
to 97 FPS is not claimed as camera sensor FPS: transport/decoder bursts can
produce misleading short-window rates. Snapshot remains a fallback with the
previous user-confirmed lower visible delay. End-to-end milliseconds unmeasured.

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --decoder-threads 1 --display-fps 60
```

Reference: https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html
(CAP_PROP_N_THREADS). No new dependency installed; this property is supported
by the project's already-pinned OpenCV version. Validation rejects negative
thread counts and counts above 32.

## Latest follow-up — actual-phone delay reduced, smoothness still pending

The previous unavailable-server limitation below is historical. The user restarted
the server and the following sequential live tests were performed on 2026-09-27:

| Mode | Measured result over 30 seconds |
|---|---|
| Optimised preview, existing 1080p MJPEG | 449 distinct frames consumed; substantial FPS dips |
| Optimised preview, changed to 720p MJPEG | 576 distinct frames consumed; user still reported large delay |
| 720p latest-snapshot mode | 251 frames consumed (~8.4 FPS), no skipped decoded frames, clean shutdown |

The phone's local status API and actual decoded dimensions showed 1920x1080,
not the planned 1280x720. Using its advertised local settings API, video_size
was changed to supported 1280x720. JPEG quality stayed at 49. Motion detection,
exposure, battery and hotspot settings were not changed. The phone reported
6% battery; the user was asked to charge it. Restoring 1920x1080 is possible in
IP Webcam Video resolution settings, but adds pixels/work again.

720p reduced workload but did not solve visible delay in the continuous stream.
We therefore added `camera/snapshot_capture.py`: a capture-compatible adapter
for `/shot.jpg`, which the phone's own help describes as its latest frame.
It uses Python's standard HTTP client, one request at a time, no continuous
video decoder queue. OpenCV decodes each returned JPEG into the same frame
format used by the existing capture worker. The rest of the backend is unchanged.
No pictures are saved. Connections close on shutdown or reconnect failure.

The adapter checks HTTP status and JPEG content type, rejects bodies over 8 MiB,
and applies socket timeouts. It does not follow redirects or support URL-embedded
credentials. A socket timeout bounds an inactive operation, not a guaranteed
total deadline against a server that trickles bytes indefinitely. It is intended
for this trusted local camera, not arbitrary public image services.

The user explicitly reported **"Delay noticeably kam hua"** during the snapshot
preview. This is a qualitative actual-hardware result, not a measured millisecond
glass-to-glass delay. Snapshot FPS was roughly 6.6–9.6 in sampled rolling windows;
this is less smooth than the desired 20–30 FPS. Keep the current phase open.

Use this low-delay option (replace PHONE_IP with current phone address):

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/shot.jpg --transport snapshot --display-fps 60
```

Q/Esc closes the preview without stopping the phone's server. The default OpenCV
transport remains available for webcams, files, RTSP and `/video`. Additions to
tests cover real HTTP JPEG acquisition and rejecting a non-JPEG/error response.

Learning: throughput (FPS) and freshness (delay) are independent. A higher
decode FPS is not proof that pictures represent the present moment. Prefer a
fresh but less smooth image during this comparison; continue tuning the network
and phone before claiming Phase 1 completion.

Date: 2026-09-27. Status: implementation tested; actual-phone performance
acceptance pending. These are PROJECT-DECISION implementation notes, not NIDAR rules.

## 1. What we built and why

The moving Moto Android phone sends camera images to the Windows laptop.
IP Webcam supplies an MJPEG stream: a continuous sequence of JPEG pictures.
OpenCV decodes these pictures. A separate capture worker keeps only the latest
picture, while the main program draws a preview and camera-health metrics.
This prepares the input for a future detector; no AI detection runs in Phase 1.

```text
Phone camera -> IP Webcam -> local Wi-Fi/hotspot -> OpenCV capture worker
    -> one latest-frame slot -> freshness check -> preview + console metrics
```

Analogy: replace the picture on a noticeboard instead of piling pictures into a
queue. A slow reader sees the newest picture, not all the old ones first.
This bounds our application buffer; it cannot eliminate buffering inside the
phone, network or decoder.

## 2. Files and important blocks

| File under nidar_survivor_demo | Responsibility |
|---|---|
| config.py | Validate source, dimensions, timeouts and CLI arguments |
| camera/phone_stream.py | Own capture handle, background reads, reconnect, freshness |
| utils/fps.py | Count frames over a bounded rolling time window |
| preview.py | Fit image without stretching; display health and metrics |
| main.py | Run preview/event loop, log measurements, release resources |
| tests/test_phase1.py | Fake-camera unit tests for logic and cleanup |
| tests/test_network.py | Local synthetic HTTP stream and real GUI tests |

`PhoneStream.start()` creates a thread: a second execution path for camera reads.
The main thread can keep the window responsive while a read waits for data.
A lock protects shared values during short updates; blocking camera reads do
not hold the lock. An Event signals shutdown and interrupts reconnect waiting.

`snapshot()` returns the latest frame and health together. It hides a frame
older than the configured freshness limit. Its timestamp is taken AFTER decode;
it is not a timestamp from the camera sensor.

`render()` copies a prebuilt background and overlays the latest image/text.
It never draws on the capture frame itself, so future ML consumers can use it.

`main.run()` checks for new frame sequence numbers. It draws new frames when
the display ceiling allows, updates idle health four times per second and
always processes quit/close events. It does not count repeated screen redraws
as new camera frames. Quit, Ctrl+C and timed exit release the capture handle.

## 3. Dependencies and configuration

No new packages were needed: OpenCV 4.13.0.92 handles decode/display, NumPy 2.2.6
holds image arrays, and Python 3.11.15 supplies threads, HTTP test server and
unittest. Installed PyTorch/Ultralytics are not used by the streaming code.
Existing requirements and lock files remain unchanged.

| Option | Default | Meaning |
|---|---|---|
| --source | NIDAR_CAMERA_SOURCE or 0 | URL, webcam index, or existing video |
| --width / --height | 1280 / 720 | Local webcam request, not network camera control |
| --fps | 30 | Local webcam request / fallback file pacing |
| --display-fps | 60 | Preview ceiling; validated 1–240, independent of phone rate |
| --stale-after | 1 second | Hide expired decoded image |
| --reconnect-delay | 2 seconds | Delay between reconnect attempts |
| --open-timeout-ms | 3000 | Network open timeout |
| --read-timeout-ms | 2000 | Network read timeout |
| --seconds | 0 | Unlimited; positive value auto-stops |
| --headless | Off | Capture and metrics without a preview |

An environment variable is a value supplied outside code. The camera URL can
be provided through NIDAR_CAMERA_SOURCE; an explicit --source overrides it.
We do not hard-code phone IPs or write credentials to configuration files.
Camera dimensions and FPS requests may be ignored by hardware/backends.

## 4. Lag: diagnosis, fix and evidence

Initial actual-phone tests before optimisation:

- 15-second headless run: 427 distinct frames consumed, one connection, clean stop.
- 20-second GUI run: 265 distinct frames consumed, one connection, clean stop.
- GUI rolling display FPS was mostly 15–17, temporarily about 5; a sampled
  decoded-image age reached 703 ms. Input FPS also dipped, so preview work alone
  cannot explain the entire problem.
- The laptop network check showed 2.4 GHz Wi-Fi, 99% signal. Strong signal does
  not establish consistent available bandwidth or phone encoding performance.

Identified avoidable work: every preview call filled a 1280x720 RGB background
from a colour tuple, even when no new video frame arrived. The old event loop
also slept on the same nominal 30 FPS period as capture, potentially missing
frame arrivals. This is a laptop-side optimisation opportunity, not proof that
all wireless lag came from those operations.

Changes:

1. Build the background once and copy contiguous pixels on refresh.
2. Draw new frame sequences, not the same frame on every loop iteration.
3. Separate preview ceiling (60 Hz) from capture request (30 FPS). This does
   not manufacture 60 FPS video; it lets the UI check for incoming frames sooner.
4. Use high-resolution performance timing for preview scheduling.
5. Log actual image dimensions, skipped frame count and render p95 milliseconds.

Measured isolated render benchmark, same 720p synthetic frame, 100 calls:

| Before | After | Interpretation |
|---|---|---|
| 9.92 ms/call | 2.97 ms/call | About 70% less rendering time in this short benchmark |

This is NOT a 70% reduction in camera-to-screen delay. It excludes phone
encoding, transport, decode, window event handling and monitor scanout. A
separate pre-fix synthetic GUI timing test averaged about 33.7 ms per loop,
showing the UI could approach 30 refreshes/s without the real phone workload.

Post-change automated tests: **17 passed in 10.404 seconds**. They cover source
validation, independent display rate, isolated render buffers, latest-frame
replacement, stale rejection, reconnect/release, interruptible stop, blocked
driver detection, credential-safe app logs, all quit paths, HTTP failure,
real MJPEG timeout recovery and timed GUI shutdown. The synthetic GUI consumed
59 distinct frames in 3 seconds; its test server is not a calibrated 30 FPS source.

The phone server subsequently actively refused connections. No post-change
actual-phone FPS or glass-to-glass improvement is claimed. Android battery,
hotspot band and camera settings were not changed remotely.

## 5. Exact Windows tests

From the repository root, replace PHONE_IP with the address in IP Webcam:

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --seconds 30 --display-fps 60
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --seconds 30 --headless
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
```

Run GUI and headless sequentially, not together: multiple clients add load.
Q/Esc quits the preview. `$LASTEXITCODE` immediately after a command shows 0 for
a run that consumed frames and stopped cleanly; unavailable input returns 1.

Phone settings starting point: rear camera, landscape, 1280x720, 30 FPS if
supported, audio/cloud disabled. Keep app foreground, provide good lighting,
and avoid heat. If Wi-Fi transport remains limiting, compare lower JPEG quality
or resolution and a supported 5 GHz link, one change at a time. Laptop-side
--fps does not adjust IP Webcam settings. Do not blindly raise input FPS.

## 6. Acceptance checklist (still pending on actual phone)

- Repeat 30-second headless and GUI runs under comparable light and movement.
- Run a longer moving-camera test: rotate, pan, walk, near/far, then stop.
- Compare actual_size, capture_fps, display_fps, skipped frames and render_p95_ms.
- Aim for the documented 720p/30-input and 20–30-display targets; report dips.
- Measure glass-to-glass delay with a visible stopwatch/event and both views.
  Decoded-image age alone cannot verify total latency or network backlog.
- Briefly stop/start IP Webcam at the same address; confirm automatic recovery
  and no frozen old image treated as fresh. Address changes require a new URL.
- Verify Q/Esc/window close. Do not start Phase 2 until this gate is accepted.

## 7. Debugging and limits

If capture FPS falls in headless mode too, investigate phone/network/decode,
not just UI. If capture is steady but display falls and render time rises,
investigate UI work and machine load. Skipped frames are intentional under a
slow consumer; they prevent our application queue accumulating old pictures.

Ping success only proves reachability. A refused/timed-out port 8080 means the
camera service is unavailable or blocked; it is not evidence of a YOLO fault.
Check the running app and current address before changing laptop firewall rules.

Native webcam drivers can ignore timeouts. A stuck worker returns an explicit
unclean-shutdown status rather than pretending release succeeded. OpenCV
network timeout support is backend-dependent. A native MJPEG decoder warning
was seen in the initial phone test; no frames were recorded for inspection.

No database, login system, deployment or cloud upload is needed for Phase 1.
Our code saves no camera images/audio. Use only a trusted local network: HTTP
video is not encrypted, and no public port forwarding is needed. Application
errors avoid echoing source URLs; native library diagnostics vary by backend.

## 8. Viva questions

**Why not increase FPS to fix lag?** More frames can mean more bandwidth and
encoding work. Remove the bottleneck first, then measure again.

**FPS vs latency?** FPS counts pictures per second. Latency is how old a picture
is when displayed. High FPS can still show delayed pictures.

**Why a thread?** Camera reads may wait for the network; the UI must still respond.

**Why skip pictures?** For live perception, the newest observation matters more
than replaying every old observation. This is not a recording pipeline.

**Why is 60 display-fps not 60 camera FPS?** It limits preview refresh opportunities.
The phone determines how many genuinely new camera images exist.

**Why not use the GPU here?** No ML inference runs yet. The measured optimisation
removes CPU image-construction work; GPU inference is a later phase.

**Is the full lag fixed?** Laptop rendering work is reduced and regression tests
pass. Actual-phone performance and end-to-end delay still need revalidation.

## 9. References

- IP Webcam official listing: https://play.google.com/store/apps/details?id=com.pas.webcam
- OpenCV video properties/timeouts: https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html
- OpenCV performance timing utilities: https://docs.opencv.org/4.11.0/db/de0/group__core__utils.html
