# Phase 6 — Persistent Survivor Manager and duplicate suppression

Date: 2026-09-27. Status: implementation, offline acceptance and documentation handoff COMPLETE. The user accepted Phase 5 and authorized Phase 6 with the phone unavailable. Live identity reliability remains OPEN; Phase 7 is not started. All design choices here are PROJECT-DECISION, not official competition rules.

## 1. What we built and why

The system now remembers accepted person identities across temporary track changes and application restarts. Think of the manager as an attendance register: seeing the same accepted person again updates their existing row instead of adding another person.

This is an appearance-based person-count estimate. A person detection does not prove injury, survival, medical condition or a unique real-world identity. Wrong Re-ID decisions can still cause a false split or false merge.

## 2. Complete runtime flow

```text
Phone / test frame -> latest-frame capture -> YOLO person boxes
 -> BoT-SORT temporary tracks -> temporal confirmation
 -> selective OSNet appearance comparison -> Survivor Manager
 -> atomic SQLite save -> S labels, counters and structured events
 -> read-only mission report
```

The offline acceptance replay supplies known boxes/tracks from labelled images. It exercises the real encoder, verifier, matcher and manager; it is not a live camera or full detector accuracy benchmark. A separate static-image GPU smoke exercises the complete detector/tracker pipeline.

| Identifier | Meaning | Lifetime |
|---|---|---|
| E1:T7 | Temporary tracker label, including reset epoch | Current application run |
| R2 | Appearance reference with three representative vectors | Session in --reid; mission in --manage |
| S2 | Accepted persistent person record | One saved mission |
| mission_id | Random UUID identifying the mission | Survives resume |
| run_id | Random UUID identifying one execution | New on each resume |

Saved track history includes run_id so E1:T7 after a restart cannot silently mean the old E1:T7. S1 in one mission is unrelated to S1 in another mission.

## 3. Important implementation blocks

1. **Initialization:** load configuration, reserve a new database path or explicitly resume an existing one, acquire a single-writer lock, and validate stored evidence. Existing mission files are never silently overwritten.
2. **Input validation:** reject repeated/non-chronological frames, malformed values and mismatched observations. Stale results cannot create records.
3. **Candidate confirmation:** only temporally CONFIRMED tracks with accepted appearance evidence qualify. A new reference needs three consecutive qualifying manager updates before receiving an S ID. This is additional to the earlier temporal and three-view appearance gates.
4. **Identity association:** accepted references already in the register reuse their S ID. Two simultaneous claims on the same reference remain pending rather than merging automatically.
5. **Transactional save:** prepare a working copy, save records/events/gallery/count together, then publish the new in-memory count. A failed write stops managed processing rather than presenting unsaved counts as successful.
6. **Loss and return:** leaving the frame marks a record NOT_VISIBLE without deleting it. An accepted return updates the same row and logs one duplicate-prevented association event, not one event per video frame.
7. **Shutdown/resume:** close connections and release the owned lock. Resume restores records and embeddings, not stale tracker bindings or current visibility.

## 4. Counter meanings

| Field | Meaning |
|---|---|
| current_persons | Currently temporally confirmed tracks, including unresolved identities |
| resolved_visible | Confirmed visible tracks assigned an S ID |
| pending_persons | Confirmed visible tracks without an accepted S assignment |
| unique_survivors | Number of accepted appearance-based records in this mission; an estimate |
| duplicates_prevented | Accepted reassociation episodes to existing records, not number of distinct people |

When perception is unavailable, current/resolved/pending values are null (unknown), not zero. Saved totals remain available. The archival report always reports current_persons as null because it is not a live occupancy sensor.

## 5. Database concepts in simple English

SQLite is a database stored in a local file. It comes with Python, so no database server, account, internet service or new package is required. PostgreSQL would add deployment and administration work that this single-laptop phase does not need.

A **schema** describes the tables and allowed structure. The three tables are metadata (mission/config/gallery), survivors (one accepted record per reference) and events (decision history). A **primary key** uniquely identifies a row. The unique survivor-ID constraint prevents duplicate S labels in the table.

A **transaction** is an all-or-nothing group of writes: record, count and audit must agree. WAL, or write-ahead logging, stores database changes in a journal; FULL synchronous mode requests durable synchronization. These settings do not replace backups or guarantee survival of faulty hardware. SQL parameters keep values separate from SQL commands.

The resume signature checks schema, encoder checksum, vector dimension and Re-ID/manager settings. It is a compatibility check, not a cryptographic authenticity guarantee or complete detector/temporal configuration provenance.

Records store first/last accepted observation times in UTC, recent track history, recent box/confidence evidence and appearance score. First seen here means creation of the accepted record, not camera exposure time. Monotonic timestamps are run-local diagnostics. Cosine similarity is not a percentage probability. Map position, grid and spatial uncertainty remain null because mapping is not implemented.

## 6. Files and responsibilities

| File under repository root | Responsibility |
|---|---|
| nidar_survivor_demo/survivors.py | Identity authority, validation, database transactions, counters, annotations |
| nidar_survivor_demo/settings/survivors.json | Adjustable manager limits outside application logic |
| nidar_survivor_demo/reid.py | Mission-retained gallery export/restore; standalone behavior preserved |
| nidar_survivor_demo/config.py | Validated --manage, --mission-db and --resume-mission options |
| nidar_survivor_demo/main.py | Connects perception to manager, preview, events and shutdown |
| nidar_survivor_demo/mission_report.py | Read-only consistent JSON export; omits embeddings |
| nidar_survivor_demo/demo_reid_offline.py | Repeatable camera-free replay and resume checks |
| tests/test_phase6.py | 25 additional automated tests |
| Start-PhoneDemo.ps1 | Optional phone launcher with -Manage and explicit resume |
| runs/ | Generated private mission databases, excluded from Git |
| logs/ | Generated measurements/reports, excluded from Git |

## 7. Configuration, dependencies and privacy

Defaults: new_identity_hits=3; observation_interval=1.0 seconds; latest 100 observations per record; latest 100 track-history entries; max_events=100000. Event history is not silently truncated: reaching its limit stops managed operation. Observations are sampled periodically and on important association changes, not saved on every frame.

Managed mode retains the gallery for the mission instead of applying the standalone 600-second inactivity expiry. Otherwise an expired known person could be counted again. The 64-reference capacity remains; uncertain/capacity-limited evidence must not become a fabricated identity. No recognition thresholds were loosened: match .815, novelty .55, ambiguity margin .08.

No new dependencies or training were needed. Existing environment: Python 3.11.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.2.6, OpenCV 4.13.0.92, Ultralytics 8.4.163, lap 0.5.12 and Pillow 12.3.0. Existing YOLO11n and OSNet x0.25 MSMT17 weights are reused. See the dependency lock and Phase 5 report for model provenance.

Privacy change: --manage saves appearance embeddings locally to support resume; standalone --reid remains session-only. Embeddings are sensitive derived person data, not anonymous merely because they are vectors. No raw camera photos are saved by this feature, no cloud upload occurs, and databases are not encrypted. Protect the machine/account, obtain participant consent and choose an appropriate retention policy. Database/WAL/SHM/lock patterns and default output folders are Git-ignored. Reports omit vectors but still contain person observation metadata; do not publish them casually.

No web API, authentication server, dashboard or deployment service was introduced in Phase 6.

## 8. Recorded acceptance evidence

| Check | Observed result | Evidence |
|---|---|---|
| Full regression | 112 tests passed in 31.618 s | logs/phase6_tests_final.txt |
| Ten offline missions | 1,000 frames, 20 return attempts, 15 correct returns, 5 unresolved | logs/phase6_offline_acceptance.json |
| Duplicate suppression | 15 duplicate events; zero return-count inflations in this replay | Same report |
| Persistence | Ten close/resume checks, accepted counts preserved | Same report and runs/phase6-acceptance/ |
| Identity errors in replay | Zero false-merge pairs and zero false splits; not a universal guarantee | Same report |
| Full GPU integration | 24 static-image frames; 2 current, 2 unique, 0 duplicate events | logs/phase6_full_gpu_smoke.txt |
| Camera-free graphical demo | One pair, two correct returns, two records, two duplicate events; clean exit | logs/phase6_visual_demo.json |
| Read-only export | Saved without embedding vectors | logs/phase6_example_mission_report.json |

The five unresolved returns comprise four appearance ambiguities and one blur rejection. All 15 enrolled references returned correctly; that conditional result is not 100% overall recognition accuracy (15/20 overall attempts resolved).

Offline manager median 2.36 ms, p95 14.94 ms; matcher-plus-manager loop median 2.66 ms, p95 83.36 ms. Regression work overlapped the run, so these are observed timings, not an isolated benchmark or camera FPS. Numerical glass-to-glass latency, live identity stability and real-world false-match rates remain unmeasured here.

Tests include stale/repeated frames, uncertain evidence, simultaneous-reference conflicts, weak-evidence flicker, reconnect semantics, second-writer rejection, no overwrite, bounded history, corrupt/incompatible/missing resume data, database failure, event-cap failure and report privacy.

## 9. Exact commands and manual checks

Run in PowerShell from C:\path\to\NIDAR\Raj Gupta\ML. Use fresh output names on every new replay; existing files intentionally cause errors instead of being overwritten. The installed COCO validation data, local models and logs/phase5_coco_proxy.json manifest are required.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m nidar_survivor_demo.demo_reid_offline --show --pairs 1 --manage-dir runs/my-phase6-demo-01 --output logs/my-phase6-demo-01.json
.\.venv\Scripts\python.exe -m nidar_survivor_demo.mission_report runs/my-phase6-demo-01/pair-18150.sqlite3 --output logs/my-phase6-report-01.json
```

The first pair in the current manifest is image 18150. Expect S1/S2 to return without increasing the final unique count above two. The replay includes an automatic database close/resume. The export should contain two records and two duplicate events, no embedding gallery, and current_persons=null. For the ten-pair acceptance replay, omit --pairs 1 and --show and choose fresh paths.

When the phone is available, optional live validation (not claimed as already passed):

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Manage -MissionDb runs/my-live-mission.sqlite3
# After clean exit, deliberately resume that same mission:
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Manage -MissionDb runs/my-live-mission.sqlite3 -ResumeMission
```

Replace PHONE_IP with the actual address; keep 5 GHz wireless. Ask consenting A and B to enter separately, leave, return, cross and change view. Record ground-truth identity, S labels, pending cases, incorrect merges/splits and response delay. After restart, accepted returns should reuse the saved IDs. Do not call ambiguous results successful matches. Q/Esc closes the preview. Existing -ReID/-Verify/-Track/-Detect modes remain available without managed counting.

## 10. Errors, recovery and what to learn

- **Windows test cleanup initially failed:** reader connections remained open. Python's SQLite connection context handles transactions but does not close the connection. Explicit connection closing fixed test cleanup. Recognize this class of issue when Windows says a temporary database is still in use after a test.
- **File exists:** new-mission/export exclusive creation protects previous evidence. Choose a fresh path or explicitly resume the same compatible mission; do not delete data to bypass the check.
- **Writer lock exists:** another writer may own the mission. Inspect the lock's PID and running process first. A hard crash can leave an orphan lock. Only after proving no writer remains should an operator remove that exact orphan lock; never delete the database/WAL to force startup.
- **Settings/model mismatch or corrupt evidence:** resume refuses to reinterpret old data. Preserve the database and use a new mission, or restore the original compatible environment after diagnosis.
- **Database failure/event capacity:** managed processing stops; unsaved count changes are not published. Inspect the error and storage condition. This is not permission to claim recovery or silently reset totals.
- **Backups:** prefer a cleanly closed mission before copying it. Copying only the main file during active WAL writes can omit data; do not discard WAL/SHM files to repair an active database.

## 11. Professor/viva answers

**Why not count tracker IDs?** The tracker may give the same person a new ID after disappearance. The manager maps accepted appearance references back to a stable mission record.

**What is an embedding?** A list of numbers describing appearance, produced by OSNet. Similar vectors suggest similar appearance; they do not prove identity.

**Why wait before creating S1?** Repeated evidence reduces the chance that a brief or unstable observation creates a permanent count.

**Why SQLite?** This phase has one local writer and bounded data. SQLite provides persistent transactions without a separate database service.

**Does duplicate count increase every frame?** No. It records accepted reassociation episodes, including return after lost visibility or restart.

**Does camera loss mean nobody is present?** No. Current occupancy becomes unknown; historical accepted totals remain.

**Can identical clothes fool it?** Yes. Conservative uncertainty helps but cannot guarantee identity. Final-drone depth/map evidence and broader labelled live tests remain necessary.

**Is Phase 6 complete?** Yes for the user-authorized implementation, camera-free acceptance and learning handoff. No claim is made that field recognition, injury detection, mapping, autonomous flight or the Phase 7 dashboard is complete.

## 12. Handoff checklist

- [x] Persistent identity authority, counters and duplicate-event semantics implemented.
- [x] Local database, compatibility checks, resume and read-only export implemented.
- [x] Regression, camera-free replay and full GPU smoke evidence retained.
- [x] File responsibilities, concepts, configuration, privacy, errors and commands documented.
- [x] Project status and dated decision log aligned with the accepted offline scope.
- [x] Earlier modes preserved; no Phase 7 implementation started.

Next phase is the professional dashboard, only after user authorization. It must consume these counters faithfully, display uncertainty and unavailable states, and label unique counts as estimates.
