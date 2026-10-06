# New Mission / Reset Demo

2026-09-27 — User-authorized mission control; no reset on refresh.

## Behavior

- Refresh/reload: same mission and accepted counts remain. Opening a second tab does not create a mission.
- New Mission / Reset Demo: opens a confirmation dialog. Cancel changes nothing.
- Confirm: saves/closes the old mission, creates a new SQLite mission, replaces tracker, verifier and appearance matcher/gallery, clears cached detection/annotations/metrics and starts counting from zero.
- Existing model weights and the camera connection are reused. This saves reload time; image processing starts again on newly received frames. A person still in view can quickly enroll again, so zero is a starting state, not a paused counter.
- All tabs observe the same new mission. Previous missions (this session) offers saved summary exports for the last 20 resets. Full SQLite records remain on disk even after that list rolls over or the app exits. Older files remain available through the existing archive launcher; no global history browser was added.
- ARCHIVE/OFFLINE REPLAY viewers cannot reset. LIVE and LOCAL VIDEO pipelines can. Local-video reset does not rewind the file; it resets perception/mission history at the current camera/file position.

## How it works — beginner explanation

The browser sends a **POST request** (an explicit change request) only after confirmation. A normal page refresh sends GET requests, which read data and do not reset it. The server validates the site's address, origin, random session token and current mission ID. The token helps reject requests from unrelated websites; the mission ID prevents an old tab from resetting a mission that has already changed.

The HTTP thread queues one command; it does not touch SQLite. The perception loop consumes the command between frames, on the same thread that owns the database. It prepares fresh components and a new database before closing the old one. The old close must commit successfully before handing over. Failure stops safely instead of pretending the reset succeeded. No database files are deleted or overwritten. A preparation/close failure may leave an empty new mission file for diagnosis, not lose old records.

After the new mission is published, browser polls show its counts and clear old frames. Image responses include their mission ID so a delayed image from the previous mission is not attached to the new mission's UI. Double submission while reset is pending is rejected; retries bearing the old mission ID are rejected after completion. Refresh can lose a pending dialog, but it does not duplicate the command.

## Files

| File | Responsibility |
|---|---|
| `mission_control.py` | Prepare new tracker/verifier/matcher/SQLite manager and safely close the old mission. |
| `main.py` | Consume reset between frames; clear previous results and skip the already-buffered capture frame. |
| `dashboard.py` | Guarded POST route, single-command mailbox, bounded saved report list and read-only archive exports. |
| `web/index.html`, `web/app.js`, `web/style.css` | Accessible confirmation dialog, cancel/confirm actions, loading/error feedback and archive links. |
| `tests/test_mission_reset.py` | Refresh preservation, security, duplicate/stale commands, archive behavior, rotation and disk-failure checks. |

No new dependencies or database schema migration. SQLite remains single-writer with atomic commits. Exported reports omit reset tokens and appearance vectors. The app remains loopback-only; this is not a public multi-user authenticated service.

## Test it yourself

1. Start a dashboard mission and wait for a person to be accepted. Note mission ID/count.
2. Refresh: mission ID and historical unique count must remain unchanged.
3. Click Reset Demo → Cancel: nothing changes.
4. Click Reset Demo → Save old mission and reset: mission ID changes, gallery/counts start fresh, then current subjects can enroll again.
5. Expand Previous missions (this session), download the previous summary and check its old count.
6. In another tab, refresh: it must show the new mission, not its own independent counter.

Launch example:

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Dashboard -DashboardPort 8766
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
```

## Verification evidence — 2026-09-27

- Full automated suite: **146 tests passed in 43.359 seconds** (`logs/reset-demo-tests.txt`).
- Real browser verification covered confirmation dialog, Cancel, refresh preservation, confirmed reset and previous-mission export. The browser-verification skill guided these interaction checks; the automation browser was closed afterwards.
- Actual GPU pipeline used a local padded COCO diagnostic video, not a live phone recording. The old mission `de54170d-0b8b-483d-b2af-3822b1372b64` retained two accepted records after refresh and in its saved export.
- Confirm created mission `802296ce-67c0-4fb8-9e92-c2c49bf5fc74`. The first logged state after reset had zero unique records; subsequent frames enrolled two records again. Thus the observed pipeline sequence was **2 → 0 → 2**, with a distinct database and mission ID.
- The timed pipeline exited cleanly. Evidence: `logs/reset-padded-mission.txt`; screenshots: `logs/reset-confirmation.png` and `logs/reset-after.png`.
- This verifies mission lifecycle and preservation, not general live-phone Re-ID accuracy or Phase 8 acceptance.

Reset is a demo-lifecycle feature, **not a correction to identity accuracy**. Do not reset just to hide false counts. Preserve and inspect those missions as test evidence.
