# Phase 7 — Professional operator dashboard

Date: 2026-09-27. Status: implementation, camera-free engineering/browser acceptance and learning handoff COMPLETE. Phase 8 is not started. Live-phone performance and field identity accuracy remain unverified; this phase does not change those limits.

## 1. What we built

A responsive, dark operator interface with a large camera workspace, four count cards, module health, measured pipeline metrics, a searchable identity register, filtered decision history, fullscreen feed and downloadable JSON report. The visual hierarchy highlights the estimated unique count without hiding pending evidence.

The dashboard is a window into the existing Python system, not a second identity engine. It cannot create/delete identities, change thresholds, start flight, configure a phone or modify the database through the browser.

All choices below are PROJECT-DECISION, not official NIDAR rules. The user authorized Phase 7 after Phase 6 handoff and retained the camera-free working constraint.

## 2. Runtime flow and simple concepts

```text
Camera or local input -> existing YOLO/tracker/verifier/Re-ID
 -> Survivor Manager -> SQLite commit
 -> bounded DashboardState mailbox
 -> FastAPI routes served by Uvicorn on 127.0.0.1
 -> browser fetch -> counters, image, records and event trail
```

An **API** is a structured way for two programs to exchange information. Here the browser asks Python for current state. A **route** is the address for one kind of information, such as /api/state. Its small handler returns data; it does not decide whether someone is a new survivor.

The **backend** is the Python process holding verified results. The **frontend** is the HTML/CSS/JavaScript page presenting those results. HTML defines the page structure, CSS controls appearance/responsive layout, and JavaScript retrieves and renders updates.

A **mailbox** stores the newest state rather than a growing queue. A short **lock** prevents the perception thread and web thread from reading/writing half an update. Data is copied out so a reader cannot mutate the identity authority.

**Middleware** is a check applied around requests: this dashboard rejects unexpected Host/Origin/cross-site requests and write methods, and adds browser security headers. The browser never receives the camera URL, database path, raw embedding gallery or observation arrays from these routes.

## 3. Stack and why

- FastAPI 0.141.1: clear Python API routes alongside the existing ML application.
- Uvicorn 0.54.0: serves the application on a dedicated thread, with bounded concurrent requests and explicit shutdown.
- Plain HTML/CSS/JavaScript: sufficient for this focused local dashboard, with no frontend build server, CDN fonts, external scripts or internet requirement at runtime. React was unnecessary for the present read-only interface; it remains an option if later interaction complexity warrants it.
- Existing SQLite: remains the only persistent identity authority. No new database, duplicate schema or cloud service was introduced.
- OpenCV: existing image tools encode the latest annotated frame as JPEG. Dashboard publishing is capped at 10 frames/s and resized only when wider than 960 pixels; camera/inference rates remain separate.

New transitive pins: annotated-doc 0.0.5, annotated-types 0.8.0, click 8.5.0, pydantic 2.13.5, pydantic-core 2.46.5, starlette 1.7.0 and typing-inspection 0.4.4. Existing ML dependencies were not upgraded. Direct and complete requirement files are updated; pip check reports no broken requirements.

Implementation references: [FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/), [Uvicorn programmatic deployment](https://www.uvicorn.org/deployment/). Browser verification used the installed skill workflow with agent-browser 0.38.1 and the existing Chrome executable; this is a development tool, not an application dependency.

## 4. Operating modes

| Mode | Input and behavior |
|---|---|
| LIVE | Phone/device pipeline. Counts and module state come from actual current processing. |
| LOCAL VIDEO | Full YOLO/BoT-SORT/verifier/OSNet/manager pipeline on a file. Explicitly not a live phone. |
| OFFLINE REPLAY | Existing actual-OSNet acceptance replay with known boxes/tracks and synthetic view changes. Camera/YOLO/tracker labels say REPLAY/ORACLE rather than pretending live inference. |
| ARCHIVE | One read-only snapshot of a selected saved mission. No camera recording or live occupancy is implied. |

Archive mode does not continuously follow a database being modified elsewhere. To view live results, run the actual pipeline with --dashboard. To refresh an archive snapshot, restart its viewer.

Current confirmed includes unresolved identities. Unique people is an **estimate** of accepted mission records. Duplicate events counts accepted reassociations, not distinct people. Identity pending displays uncertainty instead of silently adding a person. On stale/unavailable perception, current counters become unknown while historical totals remain. On backend loss the browser hides video, clears current metrics and labels retained totals as last-known.

## 5. Important files and code blocks

| File | Responsibility |
|---|---|
| nidar_survivor_demo/dashboard.py | Thread-safe mailbox, safe record projection, JPEG publishing, routes, server lifecycle and archive CLI |
| nidar_survivor_demo/web/index.html | Semantic structure: counts, feed, diagnostics, register, events and controls |
| nidar_survivor_demo/web/style.css | Dark theme, restrained lime accents, mobile breakpoints, focus styles and reduced-motion support |
| nidar_survivor_demo/web/app.js | Non-overlapping polling, safe text rendering, reconnection, search/filter/fullscreen and blob cleanup |
| nidar_survivor_demo/main.py | Publishes committed manager results and fresh annotated frames; loads recent persisted events when starting/resuming |
| nidar_survivor_demo/config.py | --dashboard implies --manage; validates --dashboard-port |
| nidar_survivor_demo/demo_reid_offline.py | Optional web presentation of the existing real-model replay and final review hold |
| nidar_survivor_demo/tests/test_phase7.py | 18 additional tests covering contracts, integration, security and lifecycle |
| Start-Dashboard.ps1 | Camera-free replay or saved-mission launcher, with unique generated output names |
| Start-PhoneDemo.ps1 | -Dashboard starts the existing wireless pipeline with web output and headless OpenCV |

Important blocks to study:

1. **annotate_dashboard:** draws one readable label per track. An accepted label includes S ID and detector person-confidence, not an invented identity probability. Original source pixels remain unchanged.
2. **publish:** encodes at most one recent JPEG, copies whitelisted fields, retains at most 100 events and 64 record summaries. A mission change clears the previous mission's event window.
3. **snapshot/frame:** applies freshness checks when reading, so even a stalled publisher cannot keep appearing healthy. Frame timestamps refer to decoded input, not sensor exposure.
4. **create_app:** fixed routes only. The server does not mount the repository or accept arbitrary file paths from requests.
5. **DashboardServer:** reserves the loopback port before starting; an occupied port fails visibly. Shutdown joins the server thread and releases the socket.
6. **Browser polling:** state requests run about every 350 ms plus request time; frame requests about every 100 ms plus request time. Each loop awaits completion before sending another request. Timeouts prevent hanging requests from forming a backlog. Old image blob URLs are revoked.
7. **Safe rendering:** dynamic strings use textContent, not innerHTML, so event values are not interpreted as executable markup.

## 6. Routes and security boundary

| Route | Purpose |
|---|---|
| GET / | Dashboard document |
| GET /app.js, /style.css | Only the fixed local frontend assets |
| GET /api/state | Sanitized current display state and recent events |
| GET /api/frame | Latest fresh JPEG; 204 when no fresh frame exists |
| GET /api/report | Downloadable sanitized dashboard snapshot, not the entire forensic audit |

The existing mission_report CLI remains available for a richer database report. Dashboard export contains only its recent 100-event window and summarized records. The complete persisted event history remains in SQLite, subject to the Phase 6 configured capacity.

The application binds only to 127.0.0.1; no firewall changes, public hosting or 0.0.0.0 listener were added. Only local same-origin reads are allowed. Responses are no-store; the page uses a same-origin content security policy and cannot be embedded in other pages. Concurrent requests are limited to 32. Camera/database secrets are not accepted from browser forms.

This is **not multi-user authentication**. Other trusted or malicious processes running as the local user may access loopback services. Do not port-forward this application. Remote deployment would require explicit authentication, TLS, authorization, deployment hardening and a privacy review. Appearance embeddings remain private local database data as documented in Phase 6. A report still contains person metadata; handle it accordingly.

## 7. Exact Windows launch commands

Run PowerShell from C:\path\to\NIDAR\Raj Gupta\ML. Dependencies are already installed in this workspace.

Camera-free actual-model demonstration:

```powershell
.\Start-Dashboard.ps1 -Replay
```

Open http://127.0.0.1:8765 in the laptop browser. The replay needs the existing local COCO images/annotations, OSNet weights and logs/phase5_coco_proxy.json manifest. It uses one known pair, restores the mission mid-test, and holds the final result for one hour by default. Each launch creates fresh GUID-named files under runs/ and logs/. Use -HoldSeconds 120 for a shorter review. Ctrl+C during the post-test hold stops the review server; it does not erase the saved report.

Review saved Phase 6 evidence without camera or model inference:

```powershell
.\Start-Dashboard.ps1 -MissionDb runs/phase6-acceptance/pair-18150.sqlite3
```

When the phone returns (optional live validation, not already passed):

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Dashboard -MissionDb runs/new-live-dashboard.sqlite3
# Continue the same compatible mission after clean shutdown:
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Dashboard -MissionDb runs/new-live-dashboard.sqlite3 -ResumeMission
```

Replace PHONE_IP with the actual base address. Keep the established wireless 5 GHz profile. Use -DashboardPort 8766 on the phone launcher or -Port 8766 on Start-Dashboard if 8765 is occupied; open the matching URL. Existing camera-only/-Detect/-Track/-Verify/-ReID/-Manage modes still work.

Tests and dependency check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m pip check
```

The -t . option preserves package context required by relative imports in tests.

## 8. Manual acceptance walkthrough

1. Launch replay and verify OFFLINE REPLAY, not LIVE, at the top. Camera and tracker diagnostics must describe the actual test inputs.
2. Watch A and B enroll; new identity evidence initially stays pending. Final unique estimate becomes two.
3. On accepted returns, observe the same S IDs, two final duplicate events and unique estimate still two.
4. At completion, the feed says Replay complete and current occupancy becomes unknown; historical count stays two.
5. Search S2: only the S2 row should remain. Search a nonexistent ID: an explicit empty-search message appears.
6. Select Mission & health: identity events disappear from that view. Return to All events to restore the list.
7. Expand the camera workspace and exit fullscreen. Export report and inspect its mode, mission, counters and events; it must contain no embedding gallery.
8. Stop the backend while leaving the page open: it must say DISCONNECTED, hide video and label retained totals as last-known. Restart at the same port and the page reconnects. Refresh the page when frontend code itself has changed.
9. At a 390-pixel browser width, content should stack without page-wide horizontal overflow. The wide identity table scrolls inside its own container.

## 9. Measured results and evidence

| Check | Result | Evidence |
|---|---|---|
| Final regression | 130 tests passed in 32.275 seconds | logs/phase7_tests_final.txt |
| Dependency consistency | No broken requirements | pip check |
| New dashboard tests | 18: state/privacy/security/stale/archive/events/CLI/server/pipeline/annotation contracts | tests/test_phase7.py |
| Real-model browser replay | 100 frames; two correct returns; two accepted records; two duplicate events; one resume; zero count inflation | logs/phase7_dashboard_replay.json |
| Full GPU initial mission | 254 processed frames, clean shutdown, two persistent records | logs/phase7_full_gpu.txt |
| Full GPU resumed mission | 208 processed frames, clean shutdown, same two records, two accepted reassociations | logs/phase7_full_gpu_resume.txt |
| Desktop browser | Meaningful page, controls/records/events visible, no app console errors | logs/phase7-final-archive.png and phase7-replay-complete.png |
| Responsive/browser controls | S2 search, health filtering, fullscreen, JSON export verified; 390px page overflow=false | logs/phase7-mobile-filtered.png |
| Backend loss/recovery | Current unknown; frame hidden; historical totals preserved with warning; same-origin restart reconnects | logs/phase7-disconnected.png; full GPU browser run |

The GPU input was a generated local video repeating an existing COCO image with two people, not phone footage. Initial run mean sampled consumer rate 9.09 FPS (8.00–9.84) on a 10-FPS file. This is an integration measurement, not a promised live-phone frame rate. Startup inference caused one safely handled stale/reset episode in each GPU test; no accepted count inflation resulted. Display image publishing is deliberately capped at 10 FPS. Higher preview smoothness/performance tuning is Phase 8, not secretly claimed complete here.

Offline replay manager median 2.89 ms and p95 24.79 ms with dashboard active; test timings are not isolated hardware benchmarks. No numerical end-to-end phone latency or new real-world identity accuracy was established. The earlier 15/20 larger Phase 6 replay remains the relevant recognition limitation, not superseded by one successful pair.

## 10. Errors and diagnosis lessons

- Initial automated control commands used unquoted @ references, which PowerShell interpreted. Quoted selectors corrected the test command; application controls were not broken. Complex browser JavaScript was passed through stdin to avoid Windows quote loss.
- Two attempts to open a short-lived GPU test server hit connection refused around its startup/end window. The server process and full-pipeline logs were checked separately. Open only after the printed dashboard-ready URL, and keep it running during browser testing. Persistent archive/replay browser verification passed.
- Keeping an already-open page across a source-code change does not reload its JavaScript. Refresh after code updates. Backend reconnect alone correctly refreshes data, not the application bundle.
- Repeated overlay layers made small-image labels crowded. Dashboard-only annotation now draws one compact label; earlier phase rendering and identity algorithms remain unchanged.
- Unit tests emit a Starlette deprecation warning about its existing httpx test transport. This is not a runtime dashboard failure; no tests failed. A future test-tool update can migrate the transport deliberately rather than changing working ML packages.
- If the port is occupied, stop the known owner or choose another explicit port. Do not kill unrelated processes or silently switch URLs.
- If an archive is missing/corrupt, startup fails; it must not invent an empty successful mission. Keep the database and diagnose it using the Phase 6 guide.

## 11. Professor/viva answers

**Does the web page run YOLO?** No. Python runs perception; the browser reads results and a JPEG preview.

**Why a separate state mailbox?** Slow browser requests must not build up old camera frames or own the database writer. The mailbox keeps only the latest display data.

**Why not show zero when disconnected?** Zero means a valid observation of nobody. Disconnected means the system cannot currently observe occupancy.

**Why is unique count an estimate?** Appearance matching can still merge different people or split one person. The dashboard preserves that uncertainty instead of making a stronger claim.

**Can the dashboard overwrite IDs?** No browser write routes exist. Only the existing manager changes records after its evidence checks and database transaction.

**What does archive health mean?** It means historical evidence was loaded. It does not mean the camera or model is currently online.

**Is this an autonomous drone dashboard?** It is the laptop perception prototype's operator view. Map coordinates, mission navigation and flight controls are intentionally not connected.

## 12. Handoff and next phase

Implemented and checked: visual interface, fresh-frame display, truthful modes/counters, health, searchable persistent register, event filters, report export, unavailable/reconnect states, local security boundary, launchers, automated tests, real-model offline demonstration and this learning report. The browser-verification skill added actual rendered-page, interaction and screenshot checks to backend tests.

No independent user comprehension study or new live-phone acceptance was performed. Camera-free engineering acceptance is complete; these limitations are documented rather than described as passed. Phase 8 performance optimization requires separate user authorization.
