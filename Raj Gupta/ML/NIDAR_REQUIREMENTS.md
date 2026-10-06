# NIDAR AirMouse — Requirements and Evidence Ledger

## 1. Purpose

This file prevents an important failure mode: confusing a team design choice with an official competition requirement.

## 2. Evidence levels

- **OFFICIAL-CONFIRMED** — directly supported by the official NIDAR website.
- **RULEBOOK-VERIFY** — visible in a search-indexed copy of the 2026–27 rulebook, but the current official PDF has not yet been stored in this repository and checked clause by clause.
- **PROJECT-DECISION** — chosen by the team.
- **ASSUMPTION / OPEN** — requires confirmation.

## 3. Officially confirmed high-level requirements

The official NIDAR website describes AirMouse as an autonomous GPS-denied indoor search, mapping, and survivor-localisation challenge. It says the system is intended to navigate unknown indoor spaces, map them, and locate survivors for disaster response.

| ID | Requirement | Evidence |
|---|---|---|
| OFF-01 | The solution is an autonomous indoor drone system. | OFFICIAL-CONFIRMED |
| OFF-02 | It must operate in a GPS-denied environment. | OFFICIAL-CONFIRMED |
| OFF-03 | It must navigate an unknown/confined indoor environment. | OFFICIAL-CONFIRMED |
| OFF-04 | It must generate/map the explored environment. | OFFICIAL-CONFIRMED |
| OFF-05 | It must detect/localise survivors or targets. | OFFICIAL-CONFIRMED |

Official high-level source, checked 2026-09-26: https://www.nidar.org.in/

## 4. Detailed items that require official-PDF verification

The following were visible in an indexed copy labelled “NIDAR 26–27 Rulebook Ver 2.1”. They are useful working constraints, but Codex must not call them fully verified until the official current-version PDF is added locally.

| ID | Working requirement | Evidence |
|---|---|---|
| RBV-01 | Use one autonomous drone for AirMouse. | RULEBOOK-VERIFY |
| RBV-02 | Enter and exit through the same designated access point. | RULEBOOK-VERIFY; indexed clauses 8.20–8.21 |
| RBV-03 | Arena is no larger than 15 m × 15 m. | RULEBOOK-VERIFY; indexed clause 8.22 |
| RBV-04 | Corridor clear width is at least 1 m. | RULEBOOK-VERIFY; indexed clause 8.22 |
| RBV-05 | Minimum vertical clearance is 8 ft. | RULEBOOK-VERIFY; indexed clause 8.22 |
| RBV-06 | Standard room size is 2 m × 2 m. | RULEBOOK-VERIFY; indexed clause 8.22 |
| RBV-07 | Take-off is from a designated 2 ft × 2 ft launch area. | RULEBOOK-VERIFY; indexed clause 8.23 |
| RBV-08 | Navigate corridors, rooms, turns, junctions, and obstacles autonomously. | RULEBOOK-VERIFY; indexed clause 8.24 |
| RBV-09 | GPS/GNSS navigation is prohibited during mission execution. | RULEBOOK-VERIFY; indexed clause 8.24 |
| RBV-10 | Detect and localise up to six survivors. | RULEBOOK-VERIFY; indexed clause 8.25 |
| RBV-11 | Tag each detected survivor at the corresponding map/grid location. | RULEBOOK-VERIFY |
| RBV-12 | Generate and continuously display a real-time 2D map. | RULEBOOK-VERIFY; indexed clause 8.26 / mission brief |
| RBV-13 | Display a live drone camera feed at the Mission Planner / Ground Control Station. | RULEBOOK-VERIFY; mission brief |
| RBV-14 | Mission performance considers time, safe flight, autonomy, mapping accuracy, and survivor-localisation accuracy. | RULEBOOK-VERIFY; mission brief |
| RBV-15 | Manual navigation or mission changes after launch can be penalised, except permitted safety actions. | RULEBOOK-VERIFY |
| RBV-16 | Required safety capabilities may include communication-loss recovery, low-battery failsafe, geofence protection, mission abort, and return/recovery behaviour. Exact AirMouse applicability needs clause verification. | RULEBOOK-VERIFY |

Indexed reference used for discovery, not final authority: https://www.scribd.com/document/1076429354/NIDAR-26-27-Rulebook-Ver-2-1

## 5. Team-selected engineering requirements

These are not claimed as official rules.

| ID | Requirement | Evidence |
|---|---|---|
| PRJ-01 | Use OAK-D Lite-class RGB/stereo-depth perception. | PROJECT-DECISION |
| PRJ-02 | Use Raspberry Pi 5 as the companion computer. | PROJECT-DECISION |
| PRJ-03 | Use a Pixhawk-class controller for flight-critical control. | PROJECT-DECISION |
| PRJ-04 | Use ROS 2 and a suitable visual-inertial SLAM stack on the companion side. | PROJECT-DECISION; exact stack OPEN |
| PRJ-05 | Use lightweight YOLO for survivor/person detection. | PROJECT-DECISION |
| PRJ-06 | Use BoT-SORT for temporary multi-object tracking in the laptop demo. | PROJECT-DECISION |
| PRJ-07 | Use appearance Re-ID plus cosine similarity for re-entry matching in the demo. | PROJECT-DECISION |
| PRJ-08 | Use depth/map position to strengthen duplicate suppression in the final drone. | PROJECT-DECISION |
| PRJ-09 | Use an Android phone as a moving-camera simulator during the ML proof of concept. | PROJECT-DECISION |
| PRJ-10 | Keep the planned final hardware bill below INR 2,00,000. | PROJECT TARGET, not verified as an official cap |

## 6. Acceptance requirements derived for engineering

These translate the mission into testable behaviours:

- No stale-frame queue during phone streaming.
- Camera reconnection does not require restarting the entire application.
- A brief false box does not create a survivor record.
- `track_id` and persistent `survivor_id` are stored separately.
- Every survivor marker has evidence: observation time, confidence, source track, and map/grid estimate where available.
- Repeated views of one person should not increase the unique count.
- Detection, tracking, Re-ID, mapping, and communication failures are visible to the operator.
- Safety-critical flight control continues to fail safely if companion-computer perception fails.

## 7. Mandatory verification work

Before design freeze or procurement:

1. Download the current official NIDAR 2026–27 rulebook from the official organiser portal.
2. Store it under a clearly named `docs/official/` location without altering it.
3. Record its version, publication date, checksum, and source URL.
4. Verify every `RBV-*` row against exact page and clause numbers.
5. Add any size, weight, radio, autonomy, safety, battery, propeller-guard, communication, software, team, or procurement constraints not captured here.
6. Recheck the official portal before each formal review because rules can change.

## 8. Explicitly unknown today

- Whether INR 2,00,000 is an official competition cap or only the team's budget target.
- Exact aircraft weight and dimension limits for AirMouse.
- Whether a particular autopilot, GCS, communication band, or map format is mandated.
- Exact survivor representation, clothing, pose, heating/lighting conditions, and dummy specification.
- Exact grid size, localisation tolerance, and scoring thresholds.
- Whether propeller guards, optical flow, range sensors, or specific failsafes are mandatory.

These must remain OPEN until verified.

