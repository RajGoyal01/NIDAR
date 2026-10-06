# Phase 8 — Performance optimization

Date: 2026-09-27. PROJECT-DECISION, not an official competition rule.

## Status and honest scope

Performance fixes are implemented. Automated regression and real-model local-video/browser checks passed. Wireless phone acceptance is **not yet verified**: the last supplied phone address refused the HTTP connection and a subsequent TCP check returned false. Phase 8 cannot be called fully live-accepted until the phone movement test passes. Phase 9 has not started.

## What caused the avoidable choppiness?

The Phase 7 backend encoded no more than ten JPEG images per second. The browser then waited another 100 milliseconds after each frame request, limiting it further. The processing loop also drew verification, appearance and survivor overlays before throwing those drawings away and drawing the dashboard overlay again. A 5 GHz connection cannot remove these software limits.

## What changed, and why

| File | Responsibility / change |
|---|---|
| `nidar_survivor_demo/dashboard.py` | Latest-image mailbox: configurable 30 FPS ceiling, encode each processed timestamp only once, measure JPEG encoding, return the image's sequence number atomically with its bytes. |
| `nidar_survivor_demo/web/app.js` | One frame request at a time; subtract request/decode time from the next wait; decode before showing; skip repeated sequence numbers; recheck freshness after decoding. |
| `nidar_survivor_demo/web/index.html` | Replace obsolete 10 FPS explanatory text. |
| `nidar_survivor_demo/main.py` | Draw one dashboard overlay instead of four; remove unconditional 5 ms headless sleep; bound CPU workers; log RAM, CPU, CUDA peak allocation and JPEG cost. |
| `nidar_survivor_demo/config.py` | Validated `--dashboard-fps 1..60` and `--cpu-threads 1..16`; defaults 30 and 2. Configuration is supplied through CLI, not machine-specific addresses in code. |
| `nidar_survivor_demo/benchmark_performance.py` | Repeatable real YOLO + tracker timing comparison, with warm-up samples excluded. |
| `nidar_survivor_demo/tests/test_phase8.py` | Regression tests for limits, fresh sequence numbers, stale-frame withholding and duplicate-encode avoidance. |

No identity thresholds, database transactions, model weights, Re-ID precision, GMC settings, security boundaries or stale thresholds were weakened. SQLite still commits accepted identity changes before publishing counts. FP16 YOLO and FP32 OSNet remain as before. No frame recording is enabled by these changes.

## Simple runtime explanation

Phone/file → latest-frame camera slot → YOLO person boxes → BoT-SORT temporary IDs → temporal confirmation → selective appearance matching → persistent survivor manager/SQLite → one annotated JPEG → browser.

A **mailbox** holds only the newest frame, like replacing an old newspaper with today's edition. A **sequence number** identifies a particular encoded image, so repeated HTTP requests do not pretend to be new displayed frames. A **ceiling** limits work; it does not manufacture 30 fresh images if inference or the camera supplies only ten.

The API still exposes read-only local routes. The frame endpoint adds `X-Frame-Sequence`; it does not expose embeddings, credentials or arbitrary filesystem paths. Only one browser request/decode is in flight, so slow rendering cannot build a request queue. After a failed request the loop retries; expired pictures are withheld rather than labelled live.

## Evidence

- Initial regression: 133 tests passed in 42.850 seconds, `logs/phase8-tests.txt`; final rerun: **133 passed in 46.031 seconds**, `logs/phase8-tests-final.txt`. Dependency check: no broken requirements.
- Static 720p image, 640 inference, two CPU workers: inference median **30.12 ms**, p95 **36.29 ms**; tracking median **9.77 ms**, p95 **10.98 ms**. `logs/phase8-thread-size-benchmark.json` contains all measurements. Thread-order effects exist; this is tuning evidence, not a universal speedup claim.
- 512 did not produce a consistent benefit across settings. Keep 640 rather than trading detection quality for an unproven speedup.
- Real GPU local-video run: **1288 frames, clean shutdown**, `logs/phase8-gpu.txt`, database `runs/phase8-gpu/mission.sqlite3`. Two identities were accepted; replay loops matched those references and avoided duplicate records. This is a static two-person fixture, not field Re-ID accuracy.
- Actual Chrome check: meaningful dashboard, image visible, S1/S2, search correctly filters to S1, no browser errors reported. Screenshot: `logs/phase8-browser.png`. Browser verification skill was used to inspect the rendered page, not merely the HTTP server.
- Browser unique-decoded frame rate sampled at **9.21 FPS** on the approximately 10 FPS recorded source. This is NOT proof of a 20–30 FPS live feed. GPU memory sampled at 1597 MiB with desktop/browser overhead included; this is not solely model allocation.
- Decoded-frame age excludes phone exposure, encoding and transport before local decoding. It is NOT camera-to-screen latency. Zero latency is not promised.

## Run and test yourself

From the repository directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Dashboard -DashboardPort 8766
```

Replace PHONE_IP with the current IP Webcam address; keep phone and laptop connected through the 5 GHz hotspot. Open http://127.0.0.1:8766. Move left/right, near/far and pan for one minute. Confirm movement does not develop seconds of growing delay. Stop/start the phone server and confirm automatic recovery without showing old pictures as live. Report perceived delay, actual processed FPS and whether a returning person retained their ID; those are separate checks.

For browser diagnostics, `window.nidarPerformance` in DevTools contains the cumulative unique decoded frame count and a two-second sample of unique decode FPS. This measures successfully decoded images assigned to the page, not physical monitor scanout. For bottlenecks inspect inference/tracking/manager p95, JPEG milliseconds, process RAM and CPU in console JSON logs. CPU percent may exceed 100 for a multi-core process; CUDA peak allocation excludes driver/desktop memory.

## What you learned / viva answers

- High capture FPS is not high displayed FPS: every downstream stage matters.
- 5 GHz improves the wireless link but cannot fix application waits or repeated work.
- Dropping outdated frames helps latency; processing all queued frames creates delay.
- Safe optimization removes redundant work first. Changing precision, image size or identity thresholds needs separate accuracy evidence.
- Measured offline correctness and live wireless acceptance are different. Only the latter can establish whether this particular phone setup feels smooth.
