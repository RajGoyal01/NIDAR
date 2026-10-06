# Lying-person count path — 2026-09-27

## Observed root cause

The attached dashboard showed an old unique record (1), current (0), pending (0),
while the camera panel displayed a lying person. Running the installed YOLO model
on the visible person crop from that screenshot, at diagnostic confidence .05,
produced person scores up to **.299**. The active BoT-SORT `new_track_thresh` was
.40, so that detection could not start a track. Even if tracked, the temporal
verifier required .35, and Re-ID required .50. The candidate was below all three
gates. This explains why a visible person box could fail to become a current or
unique count.

This crop diagnostic is one screenshot, not a pose accuracy benchmark.

## Changes

- BoT-SORT high/new thresholds changed from .35/.40 to **.20/.20**; low threshold
  remains .10, matching detector input confidence .10.
- Temporal verifier minimum confidence changed from .35 to **.20**. Its 3 of 5
  fresh-frame rule, visible-area check, outage reset and stale-result guard remain.
- Re-ID crop minimum confidence changed from .50 to **.20**. Three consistent
  samples, blur/size/overlap/edge rules, ambiguity margin and match threshold stay.
- Horizontal body crops (width >= existing upright minimum height; height >=
  existing upright minimum width) are rotated 90 degrees into portrait form for
  OSNet only. Source frames, display, detector/tracker coordinates stay unchanged.
- Dashboard `People in view` now counts tracked candidates before identity is
  accepted. Unique remains persisted accepted identities; pending remains
  unresolved tracks. This prevents the current-view card showing zero merely
  because identity processing is still underway.

## Runtime flow and files

Phone frame -> YOLO score>=.10 -> BoT-SORT can create track at .20 -> verifier
collects 3 good observations (score>=.20) -> Re-ID takes quality views, rotating
horizontal body crop for embedding -> SurvivorManager commits S record after its
confirmation hits -> dashboard updates current/unique/pending separately.

- `settings/botsort.json`: lower new/high-confidence gate for low-score poses.
- `settings/verification.json`: aligned temporal confidence gate.
- `settings/reid.json`: aligned crop score gate.
- `reid.py`: normalize a horizontal body crop for the upright encoder.
- `survivors.py`: current occupancy counts tracked candidates, even while identity
  remains unresolved.
- `web/index.html`: accurately labels current occupancy as People in view.

No weights, data, packages, schema or historical databases are modified. Re-ID
threshold remains .815 for full matches and .90 for partial matches. Unique counts
do not increment just because a box appears. The changed configuration signature
starts a new mission; prior missions remain available for review.

## Measured evidence and limits

- Screenshot crop produced maximum raw person confidence **.299** in installed
  YOLO11n at 640 input when diagnostic cutoff was .05; cutoff is not the deployed
  setting. Rotation experiments on that tiny screenshot crop ranged .258-.317.
- The previous tracker .40 / verifier .35 / Re-ID .50 settings blocked this signal.
- Changes align gates below that observed score; multi-frame and database identity
  safeguards remain. The lying pose is now eligible to enter the pipeline.
- Phone/dashboard server is not currently reachable, so corrected live count,
  false-positive rate and end-to-end delay are NOT measured/accepted.
- Lower score gates can admit more false detections. Temporal evidence and Re-ID
  reduce risk but do not certify that every visible body is a distinct survivor.
- Per-image clockwise crop normalization cannot fix extreme occlusion, tiny crops,
  severe blur, or ambiguous identity. Train/evaluate a domain model using diverse
  standing/sitting/lying, overhead/side, lighting and occlusion examples before
  making an all-angle accuracy claim.

## Launch and manual run

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Profile 720p -Dashboard -DashboardPort 8766
```

The latest screenshot's room orientation appears upright, so default camera
rotation 0 is used. Show one lying body; the People in view card should reflect a
tracked candidate before S identity resolves. Then show another separate body;
unique should increment only after distinct identity evidence is accepted. Repeat
standing, sitting, lying, partial edge, motion blur, people overlapping, and same
person returning. Record false positives and delay as well as successful counts.

## Learning / viva

**Why did the box appear while counts stayed zero?** A detector box is only a
candidate. Three later confidence gates prevented it from becoming a track, a
verified observation and an identity crop.

**Why change current count semantics?** Current occupancy answers “how many body
tracks are visible?” Unique answers “how many appearances have been accepted as
distinct people?” They need separate labels and can differ briefly.

**Does this guarantee every lying person?** No. It removes a measured confidence
barrier for this screenshot and normalizes horizontal crops. Scene/pose-specific
live and labelled-data performance still needs measurement.
