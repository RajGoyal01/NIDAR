# NIDAR Final-Camera Data Collection Checklist

This is a **project decision**, not an official competition requirement. Public
datasets make the detector stronger, but only final-camera data can validate the
actual NIDAR environment.

## Label rule

Use one detector class: `person_candidate`.

Draw one tight box around every visible real person or approved survivor dummy,
including standing, seated, crouched, lying, partially covered and partially
occluded bodies. Do not label posters, mirrors/reflections, clothes piles or empty
mannequins as people; keep them as hard negatives. Do not label a person as alive,
dead or injured from RGB imagery.

## Capture matrix

For each consenting subject/dummy, capture separate recording sessions across:

| Variable | Required variation |
|---|---|
| posture | standing, sitting, crouching, lying face-up/down/side, curled |
| camera view | front, back, side, top-down, 30–60° oblique |
| distance | near, medium, far/small target |
| visibility | full body, 75%, 50%, 25%, edge clipped |
| environment | corridor, room, doorway, bed/floor, debris/clutter |
| light | bright, ordinary indoor, shadow, safe low light |
| motion | static camera, slow pan, fast pan/blur, approach/retreat |
| people | one, two, four and up to six if the verified rules require it |
| clothing | varied colour/pattern; similar clothing pairs for identity tests |

Also record empty passes containing bedsheets, bags, chairs, posters, mirrors,
mannequins and human-shaped clutter. These are essential for false-positive control.

## Safe collection rules

- Get consent from every recognisable participant.
- Stage “fallen” poses safely; never ask anyone to perform a hazardous action.
- Do not use real injured/deceased-person imagery.
- Avoid storing names; assign random subject/session IDs.
- Keep raw footage local, access-controlled and out of Git.
- Record camera, lens, resolution, height, tilt, lighting and consent metadata.

## Leakage-safe split

Split by **subject + room + recording session**, never by random adjacent frames.
For example, all frames from `subject03_roomB_session2` must live in only one of
train, validation or hidden test. Sample frames sparsely so the dataset does not
contain thousands of nearly identical images.

Suggested project starting target (not an official rule):

- 6,000–10,000 labelled final-camera training frames;
- 1,000–2,000 validation frames from different sessions;
- 1,500+ hidden test frames from unseen subject-room sessions;
- at least 25–35% empty/hard-negative frames;
- at least 20% lying/crouching/partial-body positives;
- dedicated small/far and low-light test subsets.

The target should be changed based on error analysis, not treated as a magic number.

## Acceptance measurements

Report precision, recall and false positives per minute overall and separately for
posture, target size, light, occlusion and camera angle. Then run full-pipeline tests
for time-to-confirm, ID switches, false merges, false splits and duplicate count
inflation. Finally repeat after ONNX/OpenVINO/OAK export and under thermal load.
