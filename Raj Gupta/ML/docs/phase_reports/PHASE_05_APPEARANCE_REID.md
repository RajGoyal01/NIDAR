# Phase 5 — Selective appearance Re-ID

Date: 2026-09-27. PROJECT-DECISION, not an official competition rule.
**Implementation and offline engineering validation COMPLETE** under the user's
revised request: phone unavailable, finish without camera. Live/field reliability
remains OPEN, not passed. Phase 6 is NOT implemented.

## 1. What it does

Phase 3 associates nearby motion and may give a returning person a different
temporary ID. Phase 5 compares their appearance to earlier good views. It keeps
session-local references `R1`, `R2`, etc., separate from `E1:T5` tracking labels.
A strong, unambiguous match can recover R1 after the temporary ID changes.

An R ID is an appearance hypothesis, not a name, medical status, permanent survivor
record or unique-survivor count. References disappear when the application exits
and expire after inactivity. Phase 6 will own persistent survivor management.

## 2. Data flow

Phone -> latest frame -> YOLO person boxes -> BoT-SORT temporary tracks -> Phase 4
temporal confirmation -> quality-filtered person crop -> OSNet embedding -> compare
against session gallery -> match/new reference/uncertain -> overlay and numeric log.

The existing source frame is used throughout. The application checks freshness
before verification and again after the Re-ID model runs, before identity commits.
Slow/stale results cannot create a new reference or match. Camera capture remains
latest-only; there is no queue of old crops waiting for GPU processing.

## 3. Model and provenance

Selected **OSNet x0.25 trained on MSMT17**, a compact person Re-ID network. This is
not the ImageNet-only checkpoint and not COCO retraining. Existing YOLO11n still
detects persons; OSNet describes the appearance inside their boxes.

Author's model source: https://huggingface.co/kaiyangzhou/osnet
Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
Architecture: https://github.com/KaiyangZhou/deep-person-reid

- Model repository revision: `a5c5cc037c24235cda3b21085b93ad77c9616224`.
- Original checkpoint: `osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`.
- Local file: `models/osnet_x0_25_msmt17.pth`, 9,336,983 bytes.
- Publisher LFS SHA-256 verified on download AND before load:
  `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.
- Architecture source commit: `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`.
- Author model card is labelled MIT; copied architecture has adjacent MIT notice.
  Dataset terms and future deployment/privacy review remain separate obligations.

The small network architecture is vendored (copied with licence and pinned source)
to avoid installing an entire old training framework into our working environment.
Download helpers and unused factories were removed; network layers are unchanged.
Loading uses `weights_only=True`, verifies the checksum and rejects missing or
unexpected backbone parameters. Only the training classifier is omitted because
inference uses the 512-number feature vector, not old training-person classes.

Runtime preprocessing: BGR -> RGB, PIL bilinear resize to 256 high x 128 wide,
scale to [0,1], ImageNet channel mean/std, OSNet inference, L2 normalization.
OSNet uses FP32 on CUDA or CPU fallback; YOLO retains its own FP16 setting.
No new package installation was needed. Existing Pillow 12.3.0 is now explicitly
pinned in requirements; torch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.2.6,
OpenCV 4.13.0.92, Ultralytics 8.4.163 and lap 0.5.12 remain unchanged.

## 4. Beginner-friendly concepts

**Embedding:** a list of 512 numbers describing visual patterns. It is not a
guaranteed fingerprint; two people with similar clothes can look similar to it.

**Normalization:** rescale the number list to length one, so comparisons focus on
its direction rather than magnitude. Invalid/zero vectors are rejected.

**Cosine similarity:** dot product of normalized vectors. Higher means more similar
appearance; 0.90 does NOT mean a 90% chance of the same person.

**Gallery:** a bounded collection of reference vectors. We store three good views
per session reference, not the original photographs.

**Ambiguity margin:** the best match must be sufficiently better than the second
best. Like choosing between two similar keys: when both fit almost equally well,
do not confidently pick one.

## 5. Matching logic, block by block

1. Only Phase 4 CONFIRMED observations can request an embedding.
2. Reject low scores, small/clipped crops, blurred images and substantial overlap
   with another tracked person. These are heuristics, not a complete occlusion or
   body-pose estimator; lying and partially visible people need later evaluation.
3. Gather three good views from different processed frames at least .15s apart.
4. Normalize embeddings and check their mutual consistency to avoid enrolling a
   mixed batch after a tracker identity switch.
5. For each query view, average its two best reference similarities. The weakest
   query-view score determines that reference's multi-view score.
6. Match only if score >= .815 AND best-minus-second-best >= .08. Compare against
   active references too, but forbid reuse of a reference already on another
   visible track. Simultaneous competing queries are both left uncertain.
7. A new reference is allowed only when all query-to-gallery similarities are below
   .55, or the gallery is empty. The middle range remains UNCERTAIN; it is not
   silently converted into a new identity after a timer.
8. Once associated, continuous tracks reuse their binding without per-frame OSNet.
   Missing/reappearing tracks and resets require fresh appearance matching. A
   new, not-yet-bound track losing temporal confirmation drops its collected views.
9. Galleries are frozen after enrollment to avoid reinforcing an incorrect match
   with more wrong samples. This conservative choice can reduce view-change recall.

No face recognition, injury prediction, real-name identity or cloud inference is used.

## 6. Configuration and limits

`nidar_survivor_demo/settings/reid.json` is the editable source of runtime choices.

| Setting | Default | Purpose |
|---|---:|---|
| match_threshold | .815 | Calibrated proxy similarity floor |
| novelty_threshold | .55 | Conservative separate-reference gate |
| ambiguity_margin | .08 | Avoid close first/second choices |
| samples | 3 | Views per enrollment/query |
| sample_interval | .15 seconds | Avoid repeated adjacent sampling |
| retry_interval | .5 seconds | Throttle unresolved queries |
| min_confidence | .5 | Stronger than tracker input threshold |
| min_width / min_height | 32 / 96 pixels | Reject very small crops |
| min_visible_fraction | .9 | Limit clipping at frame boundary |
| min_blur_variance | 35 | Laplacian variance on resized crop |
| max_overlap | .35 | Maximum other-person overlap / current box area |
| max_batch | 2 | Bounded embedding requests per frame |
| max_gallery | 64 | Hard session-reference limit |
| gallery_ttl | 600 seconds | Expire inactive references |

At capacity, unresolved new observations stay uncertain instead of evicting active
references or growing memory without limit. Labels are not reused during a run.
Outages clear track bindings but retain the session gallery for re-entry matching.
Reference expiry means a very late return may obtain a new R ID; these labels must
not be treated as globally unique people.

## 7. Files and responsibilities

| File | Responsibility |
|---|---|
| `setup_reid.py` | Explicit bounded download, publisher checksum verification |
| `vendor/osnet.py`, `vendor/LICENSE.osnet.txt` | Pinned network layers and retained licence |
| `reid_encoder.py` | Safe loading, preprocessing, GPU/CPU embeddings and normalization |
| `reid.py` | Crop gates, selective schedule, bounded gallery, ambiguity/conflict handling |
| `settings/reid.json` | Tunable thresholds and limits |
| `config.py`, `main.py` | CLI, freshness integration, overlay and metadata |
| `Start-PhoneDemo.ps1` | New `-ReID` launch mode, existing camera profile |
| `benchmark_reid.py` | Repeatable labelled-image calibration/evaluation proxy |
| `demo_reid_offline.py` | Camera-free labelled replay with actual OSNet and optional visual demo |
| `tests/test_phase5.py` | Matching, safety, quality and end-to-end regression tests |
| `.gitignore`, `requirements.txt` | Keep weights out of Git; explicitly pin Pillow |

## 8. Exact commands

From `C:\path\to\NIDAR\Raj Gupta\ML`:

```powershell
.\.venv\Scripts\python.exe -m nidar_survivor_demo.setup_reid
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -ReID
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo/tests -t . -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m nidar_survivor_demo.benchmark_reid --output logs/new-reid-proxy.json
# No phone needed: use existing local benchmark manifest and a NEW report filename.
.\.venv\Scripts\python.exe -m nidar_survivor_demo.demo_reid_offline --show --pairs 2 --output logs/new-offline-demo.json
# Full 10-pair headless replay:
.\.venv\Scripts\python.exe -m nidar_survivor_demo.demo_reid_offline --output logs/new-offline-regression.json
```

Use the current IP shown by IP Webcam; keep 5 GHz wireless. Q/Esc exits.
Fallbacks: `-Verify` = Phase 4, `-Track` = Phase 3, `-Detect` = Phase 2, no flag =
camera-only. Model construction is avoided in these fallback modes.

Offline replay expects `logs/phase5_coco_proxy.json` plus the already installed
COCO dataset. If recreating the environment, run benchmark_reid with that output
path first, or pass `--manifest logs/new-reid-proxy.json`. The replay never reads
the phone and does not save camera pixels. It pastes known person crops into
synthetic frames and labels stages A_enroll/A_return/B_enroll/B_return on screen.

Optional JSONL metadata: direct Python CLI `--reid --track-log logs/new-session.jsonl`.
It saves states, IDs and scores only—not crops or embeddings. Exclusive creation
prevents overwriting existing evidence. Direct CLI does not configure phone profile.

## 9. Automated/model evidence and calibration boundaries

Real author checkpoint loaded successfully on `cuda:0`, with strict backbone
compatibility checks and normalized output. No randomly initialized replacement.

COCO proxy experiment uses 40 separate images, each containing at least two
quality-eligible non-crowd person annotations. In each image the pair with the most
similar colour histograms is selected as a harder negative. Different annotations
in the same image provide a different-person proxy. They are NOT verified people
wearing identical clothing in our room.

First 20 qualifying image IDs calibrate; next 20 are held out. Threshold selection:
maximum calibration negative cosine + .03, giving .8146608, rounded up to .815.
Held-out result: **0/20 negative pairs exceeded the threshold; 20/20 augmented
same-crop positives passed**. Three-crop CUDA call median 59.01 ms, p95 72.84 ms.
The stored report contains every image/annotation ID and score for reproducibility.

Positive examples are horizontal-flip/brightness transformations of a single crop,
NOT independently captured re-entry views. This is a sanity/calibration proxy,
not a claim of cross-camera accuracy, zero field false merges or robust similar-
clothing recognition. Margin/novelty/quality defaults are conservative project
choices, not fully calibrated deployment guarantees. Local multi-person footage
and genuinely different viewpoints are still needed for broader calibration.

Evidence: `logs/phase5_coco_proxy.json`; source calibration configuration in that
file records the initial .78 default, while the derived threshold is .8146608.
The runtime configuration was explicitly changed to .815 before live testing.

## 10. Live acceptance

One consenting participant was available. The requested protocol: establish R1,
point to an empty wall for 5–8 seconds, return to the person, repeat three times.
Record the temporary ID and R reference separately. A changed temporary ID with
R1 MATCHED is evidence of appearance re-entry; uninterrupted tracker continuity is
not sufficient. Real two-person similar-clothing false-merge testing is unavailable
in this session and must not be reported as passed.

Final closeout regression: **87 tests passed in 29.095 seconds**, including 21 Phase 5 tests.
`pip check`: no broken requirements. Coverage includes multi-view enrollment,
new/same temporary-ID re-entry, distinct vectors, ambiguity, active/conflicting
matches, temporal gates, stale results, capacity/TTL, crop quality, invalid models
and embeddings, selective scheduling, failure cleanup and main-loop integration.

Initial live run mostly viewed no person and eventually rejected low-confidence
crops; it is not an acceptance pass. User requested a fresh test run, started as
`logs/phase5_live_reentry_test2.txt`. It produced R1 from E1:T6, R2 from E1:T8,
R3 from E1:T16, then a match E1:T19 -> R3 at cosine .821535. Because only one
participant was available, the multiple references may be false splits. Numeric
logs cannot retrospectively establish physical identities; raw crops were not
recorded. No claim of three successful re-entry trials is made from one match.
Run ended cleanly: 1,743 processed frames, one connection, 174 sampled health
states, mean sampled display 9.71 FPS, 32 Re-ID crops encoded. This is not a
20–30 FPS guarantee. User then said their phone was unavailable and requested
completion without camera; no further live test was attempted.

## 10A. Camera-free completion evidence

The offline replay uses the first ten held-out COCO pairs, separate matcher
sessions per pair, actual OSNet, actual temporal confirmation and actual matching
logic. Each pair goes through A enrollment, empty frames, A return under a new
temporary ID, B enrollment, empty frames, and B return under another temporary ID.
Known boxes/temporary tracks isolate Phase 5; synthetic movement, horizontal flip
and brightness change simulate views. They are not natural live viewpoint changes.

| Offline result | Measured |
|---|---:|
| Labelled different-person pairs | 10 |
| Processed frames | 1,000 |
| Return opportunities | 20 |
| Correct reference recoveries | 15 |
| Wrong-person merge pairs | 0 |
| False splits among assigned returns | 0 |
| Unresolved appearance ambiguity | 4 |
| Unresolved because crop blurred | 1 |
| OSNet crops encoded | 122 |
| Matcher-loop median / p95 | .019 / 48.49 ms |

The four ambiguous B examples never enrolled because they were neither a strong
match nor sufficiently novel relative to A; they remained uncertain on return.
The blurred B example was rejected by quality filtering. All **15 enrolled**
references recovered, but this is a conditional result, not 100% overall accuracy.
Overall resolution was 15/20 on these synthetic-view opportunities. We did NOT
weaken the novelty/quality thresholds merely to make this score look better.

Evidence: `logs/phase5_offline_replay.json` contains settings, input annotation IDs,
every stage's final observation and transition events. The offline GUI was also
run on one pair; it requires no phone and shows known A/B re-entry labels.

Separate **actual full GPU pipeline smoke** used a real COCO image, YOLO detections,
BoT-SORT, temporal verifier and OSNet together over 24 processed observations:
two persons became verified, R1/R2 references were created, eight crops were encoded,
and the process completed. This checks integration, not detector accuracy or motion
robustness. Peak PyTorch allocated memory was 56.30 MiB, excluding CUDA context,
driver, reserved allocations and other processes—not total GPU VRAM usage.
Evidence: `logs/phase5_full_gpu_smoke.txt`.

CPU fallback and CUDA produced unit-length vectors with cosine .999991 on the same
seeded random test input. This verifies numerical/runtime compatibility, not person
recognition accuracy. Re-ID-only peak PyTorch allocated memory was 13.73 MiB in
that smoke. No camera or extra training dataset was needed.

Acceptance distinction: the requested **offline implementation/testing work is
complete**. Robust live re-entry, real similar-clothing confusion and wide-angle/
pose changes remain unverified. Do not promote those unknowns to passed tests or
claim competition-ready perception. Phase 6 needs separate authorization and an
explicit review of these limitations.

## 11. Debugging and limitations

- LOW_QUALITY: inspect the reason in console observations. Improve light, distance,
  body visibility or separation; do not lower every quality gate just to force a label.
- UNCERTAIN: similarity, ambiguity or view consistency failed. Keep it uncertain;
  this is a safe output, not permission to claim the person is new.
- Wrong reference: collect labelled positive/negative views and evaluate both false
  merges and missed matches before changing thresholds. Lower thresholds may improve
  recall while merging different people.
- GPU delay: only new/reappearing or unresolved tracks use OSNet. Smaller batches
  bound spikes, but a frame with an embedding still takes longer. No zero-delay claim.
- Missing weights or checksum mismatch: startup refuses to load. Run setup_reid;
  an existing mismatched file is deliberately not overwritten automatically.
- Model fault: the application stops cleanly and releases the camera. It does not
  pretend identity succeeded or report a fake zero count.
- Active tracker ID switch: continuous bindings trust short-term tracking. Re-ID
  is not run every frame, so this layer cannot guarantee correction of every switch.
- Appearance changes, similar uniforms, overhead/lying views, shadows and large
  occlusions remain hard. Final drone depth/map position will strengthen identity.

No original camera pixels or embeddings are persisted, and no camera URL or secret
is hard-coded into the new Python code. Use a trusted local network, no public
port-forwarding. Only numeric diagnostics and session-local RAM are used.

## 12. Professor/viva answers

**Detection vs tracking vs Re-ID?** Detection locates a person; tracking associates
nearby motion; Re-ID compares appearance across disappearance/re-entry.

**Why OSNet and not another YOLO class?** A detector class answers "person?", not
"same person?". OSNet was trained for person appearance discrimination.

**Why multiple views?** One crop can be blurred or misleading. Multiple consistent
observations reduce dependence on one frame but cannot eliminate all errors.

**Why R rather than S?** R is a bounded session appearance reference. The Phase 6
survivor manager will own persistent S records, identity history and counting.

**Does a high cosine score prove identity?** No. It is appearance similarity, not
identity probability; ambiguity, distinct-person tests and later spatial evidence matter.

**What did I learn?** Safe model provenance, preprocessing, normalized embeddings,
conservative matching, bounded state, selective inference and honest evaluation.
