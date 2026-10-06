# Phase 1 handoff — wireless phone camera

Date: 2026-09-27. Status: **Phase 1 functional acceptance COMPLETE**.
After the final test, the user confirmed movement responsiveness and automatic
video recovery after a phone-server restart. Phase 2 requires separate confirmation.
This is the concise current handoff; PHASE_01_PHONE_STREAM.md retains the full
experiment history, rejected approaches, troubleshooting and viva questions.

## 1. What is built

- Configurable phone URL, webcam index and recorded-file input.
- Wireless IP Webcam capture: standard OpenCV, latest-snapshot fallback, and
  direct MJPEG transport with a bounded latest-compressed-image slot.
- Separate capture work, latest decoded frame, stale-image rejection and retry.
- Camera health, actual resolution, capture/display FPS and honest local-age label.
- Quit, timed exit, Ctrl+C, handle release and timeout reporting.
- One-command Windows launcher that applies and verifies phone settings.
- Tests for logic, HTTP camera streams, malformed input, reconnect, GUI exit
  and launcher argument validation. No AI/model/database work was added.

## 2. Run the project

Keep the phone on the same **5 GHz** hotspot as the laptop, open IP Webcam and
start its server. In PowerShell from the repository root:

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080
```

Replace PHONE_IP with the phone's current local address. The launcher defaults
to 720p, JPEG quality 35, phone motion detection OFF and direct MJPEG.
It applies these settings on each invocation because app restart reset them
during testing. It does not change Android hotspot settings or start the app.

If 720p is not responsive in a new environment:

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 540p
```

Use `-Seconds 30` for auto-stop; Q/Esc/window close ends the preview. If local
PowerShell policy prevents running scripts, do not change machine-wide policy:
set the three settings in IP Webcam and run the equivalent Python command:

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.main --source http://PHONE_IP:8080/video --transport mjpeg --display-fps 60
```

Settings remain on the phone after normal exit. Previous settings are restored
on setup failure where the phone remains reachable; failed restoration gives a
warning. The launcher rejects non-HTTP URLs, embedded passwords, queries and
non-root paths; it requires the base camera address, not the /video endpoint.

## 3. Complete runtime flow

```text
Launcher -> validate base URL -> read available phone settings
  -> apply profile -> verify it -> start Python preview
Phone camera -> 5 GHz Wi-Fi -> HTTP multipart receiver
  -> latest compressed JPEG -> OpenCV decode -> latest decoded frame
  -> age/health check -> aspect-ratio-preserving preview + console metrics
```

An API is a way for software to send requests to another program. Here the
phone's local settings API receives POST requests to change resolution/quality.
The launcher first checks that the requested resolution exists. Verification
means reading the setting back, not just assuming a successful request worked.

A thread lets network receiving continue while another part draws the window.
Each latest-frame slot behaves like a noticeboard with one replaceable picture;
there is no application queue of old video pictures. This cannot guarantee
that the camera/operating system/network has no internal delay.

## 4. Important files

| File | What to learn from it |
|---|---|
| Start-PhoneDemo.ps1 | Validate input, configure external device, verify, rollback on failure |
| nidar_survivor_demo/config.py | Keep runtime settings outside processing logic |
| camera/mjpeg_capture.py | Parse partial HTTP image data, bounded buffers, thread lifecycle |
| camera/snapshot_capture.py | Alternative latest-photo requests with response checks |
| camera/phone_stream.py | One capture owner, health state, freshness and retry |
| preview.py | Draw a separate canvas without modifying the shared camera image |
| main.py | Coordinate input, display, metrics and cleanup |
| utils/fps.py | Calculate rolling rates without growing memory history |
| tests/ | Make failures repeatable without depending on a physical phone |

## 5. What the numbers do and do not mean

- Capture FPS: decoded image deliveries per second, not a guarantee of sensor FPS.
- Display FPS: distinct decoded frame sequences consumed by preview.
- Local frame age: time since laptop decode; NOT camera-to-screen latency.
- Render p95: 95% of recorded render calls took no longer than this value.
- Skipped frames: intentional newest-frame selection, not a survivor count.
- Negotiated Wi-Fi link speed: radio connection rate, NOT measured throughput.

The 5 GHz 540p comparison consumed 580 frames in ~22.3 seconds (~26 FPS), with
rolling dips to 14 FPS. The user confirmed smoother video and less delay.
The later 720p run consumed 1,659 frames over ~56 seconds (~29.6 FPS), one
connection and clean stop. It ended before its requested 90-second duration.
The user said they could not perform the requested movement check; therefore
these metrics do not prove moving-camera latency acceptance at 720p.

Actual-phone headless stability check: 120 seconds, all 120 sampled states
ONLINE, 1280x720 throughout, one connection, 4,751 distinct frames consumed,
clean shutdown. Rolling capture FPS min/median/max 16.0/38.19/59.45; maximum
sampled local frame age 125 ms. This verifies capture continuity, not display
smoothness or numerical glass-to-glass latency.

The user then requested the final manual test immediately. The new launcher
successfully applied/verified 720p, quality35 and motion-off and opened the
actual-phone preview. This final launcher run lasted ~31.5 seconds and consumed
677 frames (~21.5 FPS including startup), with variable rolling display rates.
The user confirmed "Haan, movement test pass" and "Haan, automatically wapas
aayi" for phone-server restart recovery without restarting the laptop app.
These are user-observed hardware acceptance results. The captured launcher log
itself showed one connection and does not independently time that reported
restart/recovery; automated stall/reconnect tests separately verify the logic.

| Acceptance item | Evidence/result |
|---|---|
| Wireless-only input | 5 GHz verified on laptop; no USB used |
| 720p capture | Actual decoded dimensions 1280x720, 120-second continuity test |
| Responsive movement | User-confirmed pan/rotate/near-far acceptance |
| No growing visible seconds of delay | User-confirmed final movement test |
| Latest-frame / stale rejection | Automated tests pass |
| Reconnect | Automated tests pass; phone-server recovery user-confirmed |
| Clean shutdown | GUI, timed, Ctrl+C tests and live runs pass |
| Repeatable setup | Actual phone launcher profile apply/read-back and preview pass |

This completes the Phase 1 functional camera pipeline. Numerical camera-to-screen
latency, sustained minimum FPS, multi-hour endurance and new network/lighting
conditions are not certified. These are ongoing performance/demo validation
work, not claimed as completed tests. No person/survivor detection exists yet.

## 6. Repeat the accepted manual test

1. Start the launcher on 5 GHz with 720p.
2. For 30–60 seconds, pan slowly, rotate, walk a short distance and move near/far.
3. Confirm movement remains responsive and delay does not grow to seconds.
4. Stop/start IP Webcam briefly without closing the laptop app. If IP changes,
   use the new URL. Confirm offline/stale status and live recovery.
5. Close the preview with Q/Esc. Confirm clean shutdown.

For numerical glass-to-glass latency, point the camera at a running stopwatch
and compare the live stopwatch with its camera preview. Account for display
refresh and measurement uncertainty; local frame age is not a substitute.
No exact zero-delay or sustained-30-FPS claim is made.

## 7. Repeat automated tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m pip check
```

Latest runs: 23 Python streaming tests passed, plus two launcher tests passed;
dependency consistency check passed. Direct-MJPEG test
includes stall, stale rejection, reconnect and receiver-thread termination.
The attempted remote phone idle endpoint returned 404: it did NOT establish
physical-phone pause/resume success. Automated recovery tests are separate evidence.

## 8. Scope, privacy and learning summary

No new dependencies: standard Python/PowerShell plus the already pinned OpenCV
and NumPy. No cloud upload, saved camera media, model inference, people counting,
authentication system or database. Use only a trusted local network; HTTP camera
video is unencrypted. No router port forwarding or USB is required.

You learned why FPS and latency differ, why fresh frames beat a growing queue,
how a local API configures the phone, why configuration must be verified after
restart, how automated and physical tests differ, and why a phase cannot be
declared fully passed from good-looking FPS alone.
