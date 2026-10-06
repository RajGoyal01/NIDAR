# NIDAR AirMouse — Codex Working Instructions

This file governs all work in this repository.

## 1. Mandatory reading before changing code

Before writing, modifying, debugging, generating, or deleting code, read these files completely in this order:

1. `PROJECT_CONTEXT.md`
2. `NIDAR_REQUIREMENTS.md`
3. `DECISIONS_AND_ASSUMPTIONS.md`
4. `ML_ARCHITECTURE.md`
5. `ML_DEMO_PLAN.md`
6. `HARDWARE_ARCHITECTURE.md`
7. `BOM_UNDER_2L.md`
8. `CHAT_HISTORY_SUMMARY.md`

After reading them, inspect the current repository and identify the active development phase. Do not redesign the system before understanding the documented decisions and open questions.

## 2. Source-of-truth and evidence rules

Every requirement or claim must be treated as one of these categories:

- **OFFICIAL-CONFIRMED**: supported by an official NIDAR source or a locally stored official rulebook.
- **RULEBOOK-VERIFY**: seen in a search-indexed or third-party copy of the rulebook, but not yet verified against a locally stored official current-version PDF.
- **PROJECT-DECISION**: selected by this team for its implementation; it is not an official competition rule.
- **ASSUMPTION**: useful for planning but still needs validation.
- **OPEN**: unresolved and must not silently become a requirement.

Never present a project decision, budget target, inference, or old rulebook statement as an official rule. If an official rulebook is later added, update `NIDAR_REQUIREMENTS.md` with exact version, page, clause, and date.

## 3. Current immediate goal

Build a laptop-based **survivor-perception and duplicate-suppression proof of concept** using a dynamically moving Android phone as the camera.

The phone simulates the future drone camera. The Windows laptop performs the current ML inference. This prototype is not the final autonomous NIDAR drone and must never be described as one.

Current known development hardware:

- Windows laptop
- NVIDIA GeForce RTX 3050 with 4 GB VRAM
- 24 GB system RAM
- Android phone used as a moving wireless camera

## 4. Phase-gated development

Proceed in this order:

1. Environment and GPU validation
2. Low-latency phone stream
3. YOLO person detection
4. BoT-SORT multi-object tracking
5. Temporal survivor verification
6. Appearance Re-ID
7. Persistent survivor manager and duplicate suppression
8. Professional dashboard
9. Performance optimisation
10. Stress testing
11. Professor/demo mode
12. Migration toward the real drone

Do not implement a later phase until the acceptance criteria of the previous phase pass. In particular, stabilise **phone stream -> detection -> tracking** before integrating Re-ID.

## 5. Architecture invariants

- Keep temporary tracker IDs separate from persistent survivor IDs.
- A single-frame detection must not immediately increase the unique-survivor count.
- Keep only the latest camera frame; do not allow stale frames to accumulate.
- Run Re-ID selectively for new or reappearing tracks, not for every person on every frame.
- During the laptop prototype, use appearance history for duplicate suppression.
- In the final drone, strengthen duplicate suppression with depth and SLAM/map position.
- Keep flight-critical control and safety on the Pixhawk-class flight controller.
- Use the Raspberry Pi-class companion computer for mission logic, ROS 2, mapping/localisation, sensor fusion, and system integration.
- Use the OAK-D Lite-class camera for RGB/depth perception and, where practical, accelerated vision inference.
- Keep the Ground Control Station responsible for operator visibility, live feed, 2D map, survivor markers, health, and mission reporting—not continuous manual navigation.

## 6. Teaching and communication requirements

The repository owner is learning while building. Before coding, briefly explain:

- what is being built;
- why it is needed;
- how it will work; and
- how it connects to the rest of the project.

When implementing a feature:

- explain important files and folders and their responsibilities;
- explain important blocks in simple, beginner-friendly English;
- define technical terms before relying on them;
- explain why each library or tool was chosen;
- explain request/data flow from input to output;
- explain validation, error handling, environment variables, security, storage, and deployment when they appear;
- prefer small analogies or examples where useful;
- do not silently make major architectural changes.

When fixing an error, explain:

1. what caused it;
2. how it was identified;
3. why the fix works; and
4. how to recognise the same class of problem later.

After each feature, provide a short summary of:

- what was built;
- the complete runtime flow;
- files involved;
- what was learned; and
- exact manual test steps.

## 7. Engineering quality rules

### Per-phase learning handoff (user requirement, 2026-09-27)

After each phase, create a permanent report in `docs/phase_reports/` covering
objective, beginner-friendly concepts, architecture/data flow, file responsibilities,
dependency versions and choices, important code blocks, configuration, exact Windows
commands, manual tests, measured results, errors and debugging, limitations,
acceptance criteria, and professor/viva questions with answers. Link the report in
the final response. Distinguish automated validation from user hardware tests.
Wait for the user's confirmation before moving to the next major phase.

Current authorization (2026-09-27): Phases 0-4 accepted, including user-accepted
live temporal verification. Preserve camera-only/detection-only/tracking-only modes.
Phase 5 implementation and camera-free engineering validation are complete under
the user's revised offline-only request (phone unavailable). Live/field identity
reliability remains OPEN; do not describe it as passed. The user subsequently
accepted Phase 5 and authorized Phase 6. Phase 6 implementation and camera-free
acceptance are COMPLETE (112 tests; 10 persisted/resumed replay missions).
Local SQLite S records and appearance-based estimated counts now exist under
--manage. Uncertain matches remain pending. Read PHASE_06_SURVIVOR_MANAGER.md.
User authorized Phase 7; its implementation, offline/browser engineering acceptance
and learning handoff are complete (130 tests). Local read-only FastAPI dashboard
preserves estimated/unknown counters and explicit replay/archive labels. Read
docs/phase_reports/PHASE_07_DASHBOARD.md. Phase 8 is not started or authorized.
Field identity reliability remains OPEN. Do not expose the local dashboard publicly.

- Use clean, modular, typed Python where practical.
- Keep configuration outside implementation code.
- Pin working dependency versions before demo freeze.
- Use structured logs and clear, recoverable errors.
- Do not hard-code secrets, camera credentials, addresses, or machine-specific paths.
- Keep generated recordings, model weights, datasets, logs, and secrets out of Git unless intentionally versioned.
- Add tests proportional to the phase: unit tests for logic, recorded-video regression tests for perception, and hardware-in-loop tests for flight integration.
- Record performance metrics instead of saying the demo is “fast”: capture FPS, inference latency, end-to-end latency, track stability, ID switches, duplicate errors, and memory/VRAM use.
- Prefer safe failure: loss of perception must not issue unsafe flight-control commands.

## 8. Change control

When a decision changes, update the relevant context file and add a dated entry to `DECISIONS_AND_ASSUMPTIONS.md`. When a rule is verified, update its evidence level and add the exact source. Keep the documentation aligned with the code.
