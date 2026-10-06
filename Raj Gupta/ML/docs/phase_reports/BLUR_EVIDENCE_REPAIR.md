# Brief-blur evidence retention — 2026-09-27

## What was wrong

The user's live run repeatedly reported clipped crops and low confidence; the
mission never accepted an identity. This is not proof of a database error.
The screenshot also showed zero temporally confirmed tracks. A candidate box
does not necessarily pass that separate confirmation gate.

Code inspection found a recoverable weakness: each blurred/low-confidence crop
erased all previously collected clear Re-ID samples. With a moving camera,
intermittent quality dips could keep restarting collection.

## Repair and flow

Camera -> detector -> tracker -> temporal verifier -> Re-ID quality checks ->
bounded clear-view memory -> appearance comparison -> SQLite -> dashboard.

`reid.py` now timestamps each good sample. On brief blur or low confidence, good
samples remain usable for at most 0.75 seconds. Bad frames never enter the
encoder and cannot trigger an identity decision. Each old sample expires
individually, so a recent frame cannot keep very old evidence alive. Novelty
votes still reset on poor quality. Clipping, overlap, track disappearance,
temporal revocation, outages and conflicting samples retain conservative behavior.

This helps only intermittent crop-quality failures on a continuously confirmed
track. It does not fix sustained detector misses, temporal-confirmation failure,
edge-clipped people, fully blurred input, or ambiguous different-person matches.
An already persisted unique count is not erased by blur.

## Files and concepts

- `reid.py`: sample timestamps, expiry and bounded short-dip memory.
- `settings/reid.json`: `sample_memory_seconds: 0.75` (validated >0 and <=1).
- `dashboard.py`: unconfirmed tracks say “Person candidate” with confidence and
  evidence hits instead of implying an accepted identity.
- `tests/test_sample_memory.py`: short blur recovery, long blur expiry, individual
  expiry, clipping/overlap rejection and invalid configuration tests.
- `tests/test_identity_regression.py`: bad quality still resets novelty, but a
  short blur preserves recent good samples under the new documented policy.
- `tests/test_phase7.py`: annotation fixture includes actual verification fields.

No dependency, training or schema change. No threshold relaxation or counting of
unknown identities. Historical databases are preserved. The config signature
changes; use a fresh mission rather than rewriting old signatures for resume.

“Bounded memory” means remembering only a few recent useful observations, not
inventing what the camera cannot currently see. This is not deblurring.

## Validation

Targeted regression suite: 17 tests passed. Real OSNet one-pair synthetic replay:
two accepted identities, two correct return associations, two duplicate events,
one resumed database, zero return-count inflation. This uses labelled COCO crops
and oracle tracks, not a certification of the user's live scene.

Final full suite: **156 tests passed in 41.899 seconds**.

Evidence: `logs/sample-memory-replay.json`, `logs/sample-memory-tests-final.txt`.
The initial full suite found an outdated annotation test fixture without `state`;
the fixture was corrected to match real `VerifiedTrack` data and the suite rerun.

## Run and test

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Rotation 90 -Dashboard -DashboardPort 8766
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
```

Rotation 90 is specific to the user's reported head-left camera mounting. Use 0
for already-upright camera input. A lying person alone is not a reason to rotate
the entire scene.

1. Show a clear person, briefly move the camera, then return to a clear view.
2. Check that clear samples spanning a short crop-quality dip can complete a batch.
3. Sustained blur must remain pending, never generate a fabricated unique record.
4. Test a genuinely different person and a same-person return separately.
5. Save the mission report and reason/track ID if a person remains pending.

Live input was RECONNECTING during this repair. Corrected code was restarted in
the local dashboard; phone-dependent acceptance remains OPEN. The user's overall
counting complaint is not declared solved solely from these engineering tests.

## Viva / limitations

**Can it count in any condition?** No. Severe blur or occlusion can remove the
information required to distinguish one person from another.

**Why not drop every quality check?** That can turn one person seen in different
poses into many identities, or merge two people. A larger number is not accuracy.

**What still needs work?** Labelled local multi-pose detection/re-entry evaluation
and calibration. Repeated fully clipped views need usable evidence or a separately
validated partial-body recognition design; this patch does not claim that capability.
