# Sequential person counting and database validation

2026-09-27 — PROJECT-DECISION. Supersedes the co-visible guard in D-041.

## Cause and correction

The previous repair required every earlier identity to be reliably visible before accepting another new identity. That prevented the user's real workflow: A leaves, then a different B enters. SQLite was saving the accepted decisions correctly; B's decision was blocked before persistence.

Removed that co-visible requirement entirely. In a managed mission with existing references, enrollment now needs three non-overlapping batches of good embeddings, at least 1.5 seconds since the first novel batch, low similarity to **every** existing reference and consistent appearance against a fixed candidate anchor. The first batch has three samples; every subsequent vote needs three new samples, not the same cached evidence. With the default retry interval, clear enrollment can take several seconds (allow 5–6 seconds). Poor quality, ambiguity, inconsistent views, missing track and outages reset evidence. Strong existing matches still recover immediately after their normal three-view check, without waiting for novelty timing.

The first identity still uses the existing three-view and manager-confirmation gates. Same-frame conflicting new proposals and image-edge crop rejection remain. No similarity threshold was loosened. No new models, dependencies, face recognition or frame recording were added.

**Limit:** repeated novelty evidence is a better decision rule, not proof of real-world identity. A consistently dissimilar view of the same human can still split; similar clothing can remain uncertain. The model still needs local labelled evaluation. This patch does not promise to resolve every person under every camera angle.

## Files and runtime flow

- `nidar_survivor_demo/reid.py`: pending candidate stores a time, batch count, new-sample count and fixed vector anchor. Validates new config fields; resets evidence on rejected quality or conflicting appearance.
- `settings/reid.json`: replaces `require_covisible_novelty` with `novelty_seconds: 1.5` and `novelty_batches: 3`.
- `tests/test_identity_regression.py`: brief changed views remain pending, stable sequential B can enroll, A returns as R1, quality failure resets votes, and a real SQLite mission saves/reloads S1/S2 without count inflation.
- `demo_reid_offline.py`: optional `--visible-frames 60` provides enough *simulated* observation time for the new rule; default remains 16. Do not compare this longer fixture directly with old 100-frame results as an accuracy improvement percentage.

Flow: camera → YOLO → tracker → temporal confirmation → appearance match OR sustained novelty → manager's three confirmations → SQLite atomic commit → dashboard. A pending person has not reached the commit step. The dashboard never increments totals independently.

## Database audit

Read-only SQLite `integrity_check` returned `ok` for both live missions:

| Mission | Saved unique records | Creation events | Saved duplicate total / events |
|---|---:|---:|---:|
| f22c3303… | 4 | 4 | 1 / 1 |
| f3ec7fe6… | 1 | 1 | 0 / 0 |

Distinct S-ID counts matched row counts. This validates storage consistency, not the truth of the identities. The user reported three actual participants; the first mission's four records are not retroactively relabelled. No data was deleted, merged or rewritten. Correcting historical identities requires identifying exactly which records refer to the same person.

Updated tests use a temporary database, accept A then B separately, close the writer, check SQLite integrity and two persisted rows, reopen with matching configuration, and verify returning A maps to S1 with two unique records and one duplicate event. Database atomicity/durability behavior is unchanged.

New settings change the compatibility signature. Old missions stay readable as archives but cannot be resumed under mismatching settings. Start a new mission for live validation; this avoids silently changing the meaning of old accepted records.

## Evidence and test commands

Final regression: **140 tests passed in 42.140 seconds**, including SQLite
commit/restart/return validation (`logs/sequential-identity-tests-final.txt`).
Dependency check: no broken requirements. Python compilation passed.

Actual OSNet, oracle-track synthetic-view replay: **276 frames, 2 correct returns, 0 unresolved returns, 0 false splits/merge pairs, 2 saved identities, 2 duplicate events, 1 resume, no return count inflation**. This is one held-out pair, not live field accuracy. Evidence: `logs/sequential-identity-replay.json`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
.\.venv\Scripts\python.exe -m nidar_survivor_demo.demo_reid_offline --pairs 1 --visible-frames 60 --manage-dir runs/NEW_TEST_FOLDER --output logs/NEW_TEST_RESULT.json
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Dashboard -DashboardPort 8766
```

Use unused output names and the current phone address. Show A clearly until S1, remove A, show B clearly for 5–6 seconds and expect S2, then return A and expect S1 with unique count staying two. Add C separately, then test each return. Report which person and resulting ID, not just whether the count stopped increasing. The corrected live test could not start because the phone HTTP server refused connection.

## What you learned

An undercount can be a decision-rule failure even when the database is healthy. Inspect pending reasons before changing SQL or deleting records. `confirming_new_person` means evidence is accumulating; `similarity_or_margin_ambiguous` means appearance remains inconclusive; `frame_edge_crop` requires a better-framed view. We now test both false-split prevention **and** genuine new-person admission, rather than treating a frozen counter as success.
