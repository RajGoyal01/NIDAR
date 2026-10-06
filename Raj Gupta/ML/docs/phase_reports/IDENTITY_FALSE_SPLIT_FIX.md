# Live identity false-split repair — 2026-09-27

## Outcome and limits

Two unsafe enrollment paths are corrected; robust live identity recognition is **not certified**. 137 automated tests passed in 42.328 seconds. A real-OSNet, 100-frame, two-person synthetic-view replay recovered A as S1 after resume, recorded one duplicate event, and produced no false merges/splits. B, shown separately while A was absent, remained unresolved. This is a deliberate conservative trade-off, not successful recognition of both people.

## Evidence and cause

The user reported duplicate people were not matched and unique count was inflated. The live mission f22c3303 ended with four appearance records and one accepted-return event. Numeric logs cannot establish the true number of participants. New-record similarities included .380, .474 and .471. S3/S4 observations included edge-touching crops; S4's first saved box reached x≈1280 in a 1280-wide image. A low score was previously sufficient evidence to enroll a new reference.

Two different questions were mixed: **does this crop resemble the reference?** and **is this definitely a different person?** A changed pose, close-up or partial body can fail the first question without proving the second. Also, a detector may already clip its box to image boundaries, making the old visible-fraction check incorrectly report a fully visible crop.

## Implementation and data flow

The unchanged flow is camera → detector → temporary tracker → temporal verifier → appearance matcher → transactional survivor manager → dashboard.

- `nidar_survivor_demo/reid.py`: reject crops within a configurable 1% image-edge margin before encoding. This is a quality heuristic, not a body-completeness detector; it can reject valid edge-positioned people.
- In managed missions, a novel-looking candidate cannot enroll while any existing appearance reference lacks a separate, quality-qualified, confirmed current track. It remains `UNCERTAIN / unseen_reference_cannot_exclude_duplicate`. No forced old-ID match occurs. Strong matches still recover old references under the unchanged .815 threshold and .08 ambiguity margin.
- Before creating a reference, recheck against references created earlier in that same frame. Similar simultaneous proposals stay pending instead of becoming two records.
- `settings/reid.json`: exposes the edge margin and `require_covisible_novelty` policy. Standalone non-managed appearance experiments retain their previous novelty rule. No new packages, training or model downloads.
- `tests/test_identity_regression.py`: four regressions cover changed-appearance return and subsequent good recovery, genuinely distinct co-visible enrollment, simultaneous similar proposals and clipped boxes.

The guard is intentionally conservative. Separate-room/sequential new arrivals can remain pending indefinitely; this is not a complete final-drone counting solution. Similar clothing, tracker swaps, prolonged false detector boxes and heavily rotated/partial views remain unresolved risks. It does not magically improve OSNet's embeddings or guarantee first-enrollment correctness.

## Persistence and old data

Old mission data is preserved, not silently rewritten. New config fields change the validated mission signature; old missions fail explicit resume compatibility checks. Use a **new mission** for the corrected demo. Old archives remain readable. There is no supported identity merge/migration in this patch; guessing which old S records should merge would fabricate ground truth.

## Test yourself

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t .
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Dashboard -DashboardPort 8766
```

1. Use an upright view, good light and a person away from the frame edges. Wait for S1.
2. Leave and return in comparable view. Expect S1 and an accepted duplicate event, not S2.
3. Repeat with a partial/edge view: expect pending rather than a new unique record.
4. Show two consenting participants together. A distinct second person may enroll as S2 after evidence; similar/conflicting appearance can remain pending. Then test returns.
5. Show person B alone after A leaves: the conservative policy may keep B pending. Do not call that successful full counting.

Phone HTTP connection was refused during the corrected test attempt, so live acceptance remains open. Evidence: `logs/identity-fix-tests-final.txt`, `logs/identity-fix-replay.json`; previous live evidence remains `logs/phase8-phone-test.txt`.

## Debugging / what you learned

Inspect `reason` in identity events. `frame_edge_crop` means move the subject inside the image; `similarity_or_margin_ambiguous` means insufficient appearance evidence; `unseen_reference_cannot_exclude_duplicate` means novelty is not established. None of these should increase the unique estimate. Count stabilization alone is not proof of matching success: also verify correct same-ID returns and true new-person enrollment. For general moving-camera accuracy, consented labelled local re-entry footage and separate positive/negative calibration remain necessary.
